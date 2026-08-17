#!/usr/bin/env python3
"""Run one immutable frozen R3 batch through eager and TE-graph training arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SIDECAR_SCHEMA = "nemo_rl_r3_frozen_batch_v1"
WORLD_SIZE = 16
RTOL = 5e-2
ATOL = 5e-2
_SIDECAR_FIELDS = {
    "schema",
    "json_sha256",
    "token_input_sha256",
    "route_sha256",
    "content_sha256",
    "token_ids",
    "input_lengths",
    "routed_experts",
}


@dataclass(frozen=True)
class FrozenBatch:
    source_path: Path
    sidecar_path: Path
    source_sha256: str
    sidecar_content_sha256: str
    row_count: int
    batch: dict[str, np.ndarray]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink: {path}")
    return resolved


def _sidecar_scalar(payload: Mapping[str, np.ndarray], name: str) -> str:
    value = payload[name]
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"sidecar {name} must be a Unicode scalar")
    result = str(value.item())
    if name != "schema" and re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"sidecar {name} must be lowercase SHA256")
    return result


def load_runtime_attestation(path: Path) -> tuple[dict[str, Any], str]:
    """Load the launcher's canonical-verifier input and recheck the typed R3 row."""
    path = _regular_file(Path(path), label="runtime attestation")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("runtime attestation is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("runtime attestation must contain a JSON object")
    if (
        payload.get("runtime_feature_set")
        != "dropless_hybridep_nano16_r3_router_graph_v1"
    ):
        raise ValueError("runtime attestation has the wrong runtime feature")
    capabilities = payload.get("mcore_capabilities")
    if not isinstance(capabilities, dict) or (
        capabilities.get("router_replay_cuda_graph_input")
        != "r3_router_cuda_graph_input_v1"
    ):
        raise ValueError("runtime attestation lacks the exact MCore capability")
    return payload, _sha256(path)


def load_frozen_batch(source: Path) -> FrozenBatch:
    """Load one exact JSONL plus its mandatory safe, content-bound route sidecar."""
    source = Path(source)
    if re.fullmatch(r"train_data_step[0-9]+\.jsonl", source.name) is None:
        raise ValueError("frozen source must be exactly train_data_step*.jsonl")
    source = _regular_file(source, label="frozen JSONL")
    sidecar = source.with_suffix(".r3-parity.npz")
    if not sidecar.exists():
        raise FileNotFoundError(f"mandatory R3 parity sidecar is missing: {sidecar}")
    sidecar = _regular_file(sidecar, label="R3 parity sidecar")

    rows = []
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid frozen JSONL row {line_number}") from error
        if not isinstance(row, dict) or row.get("idx") != line_number - 1:
            raise ValueError("frozen JSONL idx values must be contiguous from zero")
        rows.append(row)
    if not rows:
        raise ValueError("frozen JSONL contains no rows")

    with np.load(sidecar, allow_pickle=False) as archive:
        if set(archive.files) != _SIDECAR_FIELDS:
            raise ValueError("R3 parity sidecar field schema is not exact")
        payload = {name: archive[name].copy() for name in archive.files}
    if any(array.dtype.kind == "O" for array in payload.values()):
        raise ValueError("R3 parity sidecar contains an unsafe object array")

    schema = _sidecar_scalar(payload, "schema")
    if schema != SIDECAR_SCHEMA:
        raise ValueError(f"unsupported R3 parity sidecar schema: {schema}")
    source_sha256 = _sha256(source)
    if _sidecar_scalar(payload, "json_sha256") != source_sha256:
        raise ValueError("R3 parity sidecar JSON SHA256 does not match source")

    token_ids = payload["token_ids"]
    input_lengths = payload["input_lengths"]
    routed_experts = payload["routed_experts"]
    if token_ids.ndim != 2 or token_ids.dtype.kind not in "iu":
        raise ValueError("sidecar token_ids must be a rank-2 integer array")
    if input_lengths.ndim != 1 or input_lengths.dtype.kind not in "iu":
        raise ValueError("sidecar input_lengths must be a rank-1 integer array")
    if routed_experts.ndim != 4 or routed_experts.dtype.kind not in "iu":
        raise ValueError("sidecar routed_experts must be a rank-4 integer array")
    if token_ids.shape[0] != len(rows) or input_lengths.shape != (len(rows),):
        raise ValueError("sidecar batch dimensions do not match frozen JSONL")
    if routed_experts.shape[:2] != token_ids.shape:
        raise ValueError("sidecar route dimensions do not match token_ids")
    if np.any(input_lengths < 1) or np.any(input_lengths > token_ids.shape[1]):
        raise ValueError("sidecar input_lengths are outside the token capacity")

    token_input_sha256 = _array_digest(
        {"input_lengths": input_lengths, "token_ids": token_ids}
    )
    route_sha256 = _array_digest(
        {"input_lengths": input_lengths, "routed_experts": routed_experts}
    )
    if _sidecar_scalar(payload, "token_input_sha256") != token_input_sha256:
        raise ValueError("R3 parity sidecar token/input digest does not match arrays")
    if _sidecar_scalar(payload, "route_sha256") != route_sha256:
        raise ValueError("R3 parity sidecar route digest does not match arrays")
    content_sha256 = hashlib.sha256(
        (
            f"{schema}\0{source_sha256}\0{token_input_sha256}\0{route_sha256}"
        ).encode()
    ).hexdigest()
    if _sidecar_scalar(payload, "content_sha256") != content_sha256:
        raise ValueError("R3 parity sidecar content digest does not match payload")

    required_json_fields = {
        "token_ids",
        "input_lengths",
        "token_loss_mask",
        "sample_loss_mask",
        "advantages",
        "generation_logprobs",
        "prev_logprobs",
        "rewards",
    }
    for row in rows:
        missing = required_json_fields - set(row)
        if missing:
            raise ValueError(f"frozen JSONL row is missing: {sorted(missing)}")
    json_tokens = np.asarray([row["token_ids"] for row in rows], dtype=np.int64)
    json_lengths = np.asarray([row["input_lengths"] for row in rows], dtype=np.int64)
    if not np.array_equal(token_ids.astype(np.int64), json_tokens):
        raise ValueError("sidecar token_ids do not exactly match frozen JSONL")
    if not np.array_equal(input_lengths.astype(np.int64), json_lengths):
        raise ValueError("sidecar input_lengths do not exactly match frozen JSONL")

    def rows_array(name: str, *, dtype: Any) -> np.ndarray:
        return np.ascontiguousarray(np.asarray([row[name] for row in rows], dtype=dtype))

    prev_logprobs = rows_array("prev_logprobs", dtype=np.float32)
    token_mask = rows_array("token_loss_mask", dtype=np.float32)
    advantages = rows_array("advantages", dtype=np.float32)
    generation_logprobs = rows_array("generation_logprobs", dtype=np.float32)
    for name, array in (
        ("token_loss_mask", token_mask),
        ("advantages", advantages),
        ("generation_logprobs", generation_logprobs),
        ("prev_logprobs", prev_logprobs),
    ):
        if array.shape != token_ids.shape:
            raise ValueError(f"frozen JSONL {name} shape does not match token_ids")
    sample_mask = rows_array("sample_loss_mask", dtype=np.float32)
    rewards = rows_array("rewards", dtype=np.float32)
    if sample_mask.shape != (len(rows),) or rewards.shape != (len(rows),):
        raise ValueError("frozen JSONL sample masks and rewards must be per-sample")
    numeric_arrays = (
        token_mask,
        advantages,
        generation_logprobs,
        prev_logprobs,
        sample_mask,
        rewards,
    )
    if any(not np.all(np.isfinite(array)) for array in numeric_arrays):
        raise ValueError("frozen JSONL numeric training fields must be finite")
    batch = {
        "input_ids": token_ids.astype(np.int64, copy=False),
        "input_lengths": input_lengths.astype(np.int64, copy=False),
        "routed_experts": routed_experts,
        "token_mask": token_mask,
        "sample_mask": sample_mask,
        "advantages": advantages,
        "generation_logprobs": generation_logprobs,
        "prev_logprobs": prev_logprobs,
        "rewards": rewards,
    }
    return FrozenBatch(
        source_path=source,
        sidecar_path=sidecar,
        source_sha256=source_sha256,
        sidecar_content_sha256=content_sha256,
        row_count=len(rows),
        batch=batch,
    )


def _compare_number(
    eager: Any,
    graph: Any,
    *,
    label: str,
    rank: int,
    rtol: float,
    atol: float,
) -> None:
    if not np.allclose(eager, graph, rtol=rtol, atol=atol, equal_nan=False):
        raise ValueError(f"{label} parity failed on rank {rank}: {eager!r} != {graph!r}")


def _compare_tensor_evidence(
    eager: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    label: str,
    rank: int,
    rtol: float,
    atol: float,
) -> None:
    for side, evidence in (("eager", eager), ("graph", graph)):
        if re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256", ""))) is None:
            raise ValueError(f"{label} lacks a valid {side} SHA256 on rank {rank}")
    for field in ("shape", "dtype", "numel", "sample_indices"):
        if field in eager or field in graph:
            if eager.get(field) != graph.get(field):
                raise ValueError(f"{label} {field} parity failed on rank {rank}")
    for field in ("l2_norm", "max_abs", "mean", "values"):
        if field not in eager or field not in graph:
            raise ValueError(f"{label} lacks numeric {field} evidence on rank {rank}")
        _compare_number(
            eager[field],
            graph[field],
            label=f"{label}.{field}",
            rank=rank,
            rtol=rtol,
            atol=atol,
        )


def validate_parity(
    eager_results: Sequence[Mapping[str, Any]],
    graph_results: Sequence[Mapping[str, Any]],
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> dict[str, Any]:
    """Validate exact content identity and bounded numeric parity on all 16 ranks."""
    if len(eager_results) != WORLD_SIZE or len(graph_results) != WORLD_SIZE:
        raise ValueError("parity requires exactly one result on all 16 ranks")
    eager = {int(result["rank"]): result for result in eager_results}
    graph = {int(result["rank"]): result for result in graph_results}
    expected_ranks = set(range(WORLD_SIZE))
    if set(eager) != expected_ranks or set(graph) != expected_ranks:
        raise ValueError("parity requires exactly one eager and graph result on all 16 ranks")

    compared_gradients = 0
    for rank in range(WORLD_SIZE):
        eager_rank = eager[rank]
        graph_rank = graph[rank]
        if eager_rank.get("arm") != "eager" or graph_rank.get("arm") != "graph":
            raise ValueError(f"parity arm labels are invalid on rank {rank}")
        for digest in (
            "token_digest",
            "route_digest",
            "mask_digest",
            "reward_digest",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(eager_rank.get(digest, ""))) is None:
                raise ValueError(f"{digest} is malformed on eager rank {rank}")
            if re.fullmatch(r"[0-9a-f]{64}", str(graph_rank.get(digest, ""))) is None:
                raise ValueError(f"{digest} is malformed on graph rank {rank}")
            if eager_rank.get(digest) != graph_rank.get(digest):
                raise ValueError(f"{digest} mismatch on rank {rank}")
        _compare_number(
            eager_rank["loss"],
            graph_rank["loss"],
            label="loss",
            rank=rank,
            rtol=rtol,
            atol=atol,
        )
        for field in (
            "selected_output",
            "selected_output_gradient",
            "selected_input_gradient",
        ):
            if field not in eager_rank or field not in graph_rank:
                raise ValueError(f"{field} evidence is missing on rank {rank}")
            _compare_tensor_evidence(
                eager_rank[field],
                graph_rank[field],
                label=field,
                rank=rank,
                rtol=rtol,
                atol=atol,
            )
        if not eager_rank["parameter_gradients"]:
            raise ValueError(f"parameter gradient evidence is empty on rank {rank}")
        for collection in ("parameter_gradients", "simulated_parameter_deltas"):
            eager_collection = eager_rank[collection]
            graph_collection = graph_rank[collection]
            if set(eager_collection) != set(graph_collection):
                raise ValueError(f"{collection} parameter names differ on rank {rank}")
            for name in eager_collection:
                _compare_tensor_evidence(
                    eager_collection[name],
                    graph_collection[name],
                    label=f"{collection}.{name}",
                    rank=rank,
                    rtol=rtol,
                    atol=atol,
                )
                if collection == "parameter_gradients":
                    compared_gradients += 1
        if set(eager_rank["metrics"]) != set(graph_rank["metrics"]):
            raise ValueError(f"metric names differ on rank {rank}")
        for name in eager_rank["metrics"]:
            _compare_number(
                eager_rank["metrics"][name],
                graph_rank["metrics"][name],
                label=f"metrics.{name}",
                rank=rank,
                rtol=rtol,
                atol=atol,
            )
        graph_metrics = graph_rank["graph_metrics"]
        for positive in ("requested_graph_calls", "graph_calls", "cache_hits"):
            if int(graph_metrics.get(positive, 0)) < 1:
                raise ValueError(f"graph {positive} is absent on rank {rank}")
        if int(graph_metrics.get("captures", -1)) != 1:
            raise ValueError(f"graph capture count is not exactly one on rank {rank}")
        for zero in ("recaptures", "fallback_count", "unsafe_route_events"):
            if int(graph_metrics.get(zero, -1)) != 0:
                raise ValueError(f"graph {zero} is nonzero on rank {rank}")

    return {
        "status": "passed",
        "world_size": WORLD_SIZE,
        "rtol": rtol,
        "atol": atol,
        "compared_parameter_gradients": compared_gradients,
    }


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create exactly one read-only JSON artifact without replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"parity artifact already exists: {path}") from error
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), 0o444)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _run_distributed(
    args: argparse.Namespace,
    frozen: FrozenBatch,
    runtime_attestation: Mapping[str, Any],
    runtime_attestation_sha256: str,
) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.algorithms.loss import ClippedPGLossFn
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
    from nemo_rl.models.policy.lm_policy import Policy
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    config = load_config(args.config)
    config = parse_hydra_overrides(config, args.overrides)
    master = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    if master.cluster["num_nodes"] != 4 or master.cluster["gpus_per_node"] != 4:
        raise ValueError("R3 parity requires the exact 4-node, 4-GPU cluster")
    megatron = master.policy["megatron_cfg"]
    modules = megatron.get("cuda_graph_modules")
    if modules != ["moe_router"]:
        raise ValueError("R3 parity requires exact cuda_graph_modules=[moe_router]")
    generation = master.policy.get("generation") or {}
    vllm_kwargs = generation.get("vllm_kwargs") or {}
    sequence_packing = master.policy.get("sequence_packing") or {}
    dynamic_batching = master.policy.get("dynamic_batching") or {}
    fp8 = megatron.get("fp8_cfg") or {}
    exact_policy_contract = (
        master.policy.get("model_name")
        == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
        and (master.policy.get("router_replay") or {}).get("enabled") is True
        and generation.get("backend") == "vllm"
        and vllm_kwargs.get("moe_backend") == "triton"
        and master.policy.get("precision") == "bfloat16"
        and megatron.get("cuda_graph_impl") == "transformer_engine"
        and megatron.get("moe_token_dispatcher_type") == "flex"
        and megatron.get("moe_flex_dispatcher_backend") == "hybridep"
        and megatron.get("thd_max_packed_sequences") == 16
        and megatron.get("tensor_model_parallel_size") == 2
        and megatron.get("pipeline_model_parallel_size") == 2
        and megatron.get("context_parallel_size") == 2
        and megatron.get("expert_model_parallel_size") == 8
        and megatron.get("sequence_parallel") is True
        and fp8.get("enabled") is not True
        and sequence_packing.get("enabled") is True
        and dynamic_batching.get("enabled") is False
        and os.environ.get("NRL_ROUTER_REPLAY_VALIDATE") == "1"
    )
    if not exact_policy_contract:
        raise ValueError("R3 parity runtime policy contract is not exact")
    if master.loss_fn.reference_policy_kl_penalty != 0:
        raise ValueError("R3 parity requires reference_policy_kl_penalty=0")
    if master.grpo.skip_reference_policy_logprobs_calculation is not True:
        raise ValueError("R3 parity requires reference policy logprobs to be skipped")

    init_ray()
    cluster = RayVirtualCluster(
        name="r3_router_graph_parity",
        bundle_ct_per_node_list=[4, 4, 4, 4],
        use_gpus=True,
        num_gpus_per_node=4,
        max_colocated_worker_groups=1,
    )
    tokenizer = get_tokenizer(master.policy["tokenizer"])
    policy = Policy(
        cluster=cluster,
        config=master.policy,
        tokenizer=tokenizer,
        init_optimizer=True,
        init_reference_model=False,
    )
    try:
        batch_arrays = dict(frozen.batch)
        # This synthetic field is valid only after the exact zero-KL contract
        # above has been resolved and checked from the runtime configuration.
        batch_arrays["reference_policy_logprobs"] = np.zeros_like(
            frozen.batch["prev_logprobs"]
        )
        batch = BatchedDataDict(
            {name: torch.from_numpy(value.copy()) for name, value in batch_arrays.items()}
        )
        shards = policy._shard_for_train(
            batch,
            batch_size=master.policy["train_global_batch_size"],
        )
        loss_fn = ClippedPGLossFn(
            master.loss_fn,
            use_fused_linear_logprobs=bool(
                master.policy["megatron_cfg"].get("use_fused_linear_logprobs", False)
            ),
        )

        def arm(name: str) -> list[dict[str, Any]]:
            futures = policy.worker_group.run_all_workers_sharded_data(
                "run_r3_router_graph_parity_arm",
                data=shards,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                output_is_replicated=[],
                common_kwargs={
                    "loss_fn": loss_fn,
                    "arm": name,
                    "simulated_learning_rate": args.simulated_learning_rate,
                },
            )
            return policy.worker_group.get_all_worker_results(futures)

        eager = arm("eager")
        graph = arm("graph")
        comparison = validate_parity(eager, graph, rtol=args.rtol, atol=args.atol)
        return {
            "schema": "nemo_rl_r3_router_graph_parity_v1",
            "comparison": comparison,
            "source": {
                "jsonl": str(frozen.source_path),
                "json_sha256": frozen.source_sha256,
                "sidecar": str(frozen.sidecar_path),
                "sidecar_content_sha256": frozen.sidecar_content_sha256,
                "rows": frozen.row_count,
            },
            "config": OmegaConf.to_container(config, resolve=True),
            "provenance": {
                "profile_sha256": args.profile_sha256,
                "runtime_attestation_path": str(
                    Path(args.runtime_attestation).resolve(strict=True)
                ),
                "runtime_attestation_sha256": runtime_attestation_sha256,
                "runtime_attestation": runtime_attestation,
            },
            "eager": eager,
            "graph": graph,
        }
    finally:
        policy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-batch", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--simulated-learning-rate", type=float, default=1e-4)
    args, args.overrides = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"[0-9a-f]{64}", args.profile_sha256) is None:
        raise ValueError("--profile-sha256 must be a full lowercase SHA256")
    frozen = load_frozen_batch(args.frozen_batch)
    if frozen.source_sha256 != args.expected_source_sha:
        raise ValueError("frozen source SHA does not match --expected-source-sha")
    runtime_attestation, runtime_attestation_sha256 = load_runtime_attestation(
        args.runtime_attestation
    )
    artifact = _run_distributed(
        args,
        frozen,
        runtime_attestation,
        runtime_attestation_sha256,
    )
    write_immutable_json(args.output, artifact)


if __name__ == "__main__":
    main()
