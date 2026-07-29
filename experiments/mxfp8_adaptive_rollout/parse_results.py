#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parse and validate the adaptive MXFP8 rollout A/B experiment."""

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence


METADATA_PREFIX = "MXFP8_AB_METADATA "
CONFIG_ENV_KEY = "VLLM_MXFP8_DENSE_CONFIG_FILE"
CSV_FIELDS = (
    "step",
    "arm",
    "repeat",
    "rollout_wall_time_s",
    "generation_time_s",
    "total_step_time_s",
    "output_tokens",
    "output_tokens_per_second_per_gpu",
    "vllm_commit",
    "nemo_rl_commit",
    "container_digest",
    "config_hash",
    "tensor_parallel_size",
    "seed",
)
REQUIRED_METADATA = (
    "arm",
    "repeat",
    "vllm_commit",
    "nemo_rl_commit",
    "container_digest",
    "config_hash",
    "tensor_parallel_size",
    "seed",
    "num_samples",
    "generation_num_gpus",
)
MATCHED_PROVENANCE_FIELDS = (
    "nemo_rl_commit",
    "vllm_commit",
    "container_digest",
    "checkpoint",
    "topology",
)
STEP_RE = re.compile(r"={3,}\s*Step\s+(?P<step>\d+)/\d+\s*={3,}")
FLOAT_PATTERNS = {
    "mean_generation_length": re.compile(
        r"Mean Generation Length:\s*(?P<value>\d+(?:\.\d+)?)"
    ),
    "total_step_time_s": re.compile(r"Total step time:\s*(?P<value>\d+(?:\.\d+)?)s"),
    "generation_time_s": re.compile(
        r"(?m)^\s*[•*-]\s+generation:\s*(?P<value>\d+(?:\.\d+)?)s"
    ),
    "rollout_wall_time_s": re.compile(
        r"timing/rollout/(?:run_rollouts|total):\s*"
        r"(?P<value>\d+(?:\.\d+)?)s"
    ),
}


class ResultRecord(NamedTuple):
    """One measured NeMo-RL step with immutable experiment provenance."""

    step: int
    arm: str
    repeat: int
    rollout_wall_time_s: float
    generation_time_s: float
    total_step_time_s: float
    output_tokens: int
    output_tokens_per_second_per_gpu: float
    vllm_commit: str
    nemo_rl_commit: str
    container_digest: str
    config_hash: str
    tensor_parallel_size: int
    seed: int


def _required_value(metadata: Mapping[str, object], key: str) -> object:
    if key not in metadata:
        raise ValueError(f"metadata is missing required field {key!r}")
    return metadata[key]


def _positive_int(metadata: Mapping[str, object], key: str) -> int:
    value = _required_value(metadata, key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"metadata field {key!r} must be a positive integer")
    return value


def _metadata_from_log(text: str) -> dict[str, object]:
    metadata_lines = [
        line.removeprefix(METADATA_PREFIX)
        for line in text.splitlines()
        if line.startswith(METADATA_PREFIX)
    ]
    if len(metadata_lines) != 1:
        raise ValueError(
            f"expected exactly one {METADATA_PREFIX.strip()} record, "
            f"found {len(metadata_lines)}"
        )
    try:
        metadata = json.loads(metadata_lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("MXFP8 A/B metadata is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("MXFP8 A/B metadata must be a JSON object")
    for key in REQUIRED_METADATA:
        _required_value(metadata, key)
    arm = metadata["arm"]
    if arm not in {"original", "adaptive"}:
        raise ValueError("metadata field 'arm' must be original or adaptive")
    _positive_int(metadata, "repeat")
    _positive_int(metadata, "tensor_parallel_size")
    _positive_int(metadata, "num_samples")
    _positive_int(metadata, "generation_num_gpus")
    seed = metadata["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("metadata field 'seed' must be an integer")
    for key in (
        "vllm_commit",
        "nemo_rl_commit",
        "container_digest",
        "config_hash",
    ):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f"metadata field {key!r} must be a non-empty string")
    return metadata


def _metric(block: str, name: str) -> float:
    match = FLOAT_PATTERNS[name].search(block)
    if match is None:
        raise ValueError(f"step block is missing required metric {name!r}")
    return float(match.group("value"))


def parse_log(text: str) -> list[ResultRecord]:
    """Parse all measured step blocks from one launcher log.

    Args:
        text: Complete run log containing one ``MXFP8_AB_METADATA`` line.

    Returns:
        Step records in log order.

    Raises:
        ValueError: If provenance or a required metric is missing or malformed.
    """
    metadata = _metadata_from_log(text)
    step_matches = list(STEP_RE.finditer(text))
    if not step_matches:
        raise ValueError("log contains no NeMo-RL step blocks")

    records: list[ResultRecord] = []
    num_samples = _positive_int(metadata, "num_samples")
    generation_num_gpus = _positive_int(metadata, "generation_num_gpus")
    for index, step_match in enumerate(step_matches):
        block_end = (
            step_matches[index + 1].start()
            if index + 1 < len(step_matches)
            else len(text)
        )
        block = text[step_match.end() : block_end]
        generation_time_s = _metric(block, "generation_time_s")
        rollout_match = FLOAT_PATTERNS["rollout_wall_time_s"].search(block)
        rollout_wall_time_s = (
            float(rollout_match.group("value"))
            if rollout_match is not None
            else generation_time_s
        )
        if rollout_wall_time_s <= 0:
            raise ValueError("rollout wall time must be positive")
        mean_generation_length = _metric(block, "mean_generation_length")
        output_tokens = round(mean_generation_length * num_samples)
        records.append(
            ResultRecord(
                step=int(step_match.group("step")),
                arm=str(metadata["arm"]),
                repeat=int(metadata["repeat"]),
                rollout_wall_time_s=rollout_wall_time_s,
                generation_time_s=generation_time_s,
                total_step_time_s=_metric(block, "total_step_time_s"),
                output_tokens=output_tokens,
                output_tokens_per_second_per_gpu=(
                    output_tokens / rollout_wall_time_s / generation_num_gpus
                ),
                vllm_commit=str(metadata["vllm_commit"]),
                nemo_rl_commit=str(metadata["nemo_rl_commit"]),
                container_digest=str(metadata["container_digest"]),
                config_hash=str(metadata["config_hash"]),
                tensor_parallel_size=int(metadata["tensor_parallel_size"]),
                seed=int(metadata["seed"]),
            )
        )
    return records


def _strip_adaptive_config_key(config: object) -> object:
    normalized = copy.deepcopy(config)
    if not isinstance(normalized, dict):
        raise ValueError("resolved_config must be a JSON object")
    policy = normalized.get("policy")
    if not isinstance(policy, dict):
        return normalized
    generation = policy.get("generation")
    if not isinstance(generation, dict):
        return normalized
    vllm_cfg = generation.get("vllm_cfg")
    if not isinstance(vllm_cfg, dict):
        return normalized
    env_vars = vllm_cfg.get("env_vars")
    if not isinstance(env_vars, dict):
        return normalized
    env_vars.pop(CONFIG_ENV_KEY, None)
    if not env_vars:
        vllm_cfg.pop("env_vars")
    return normalized


def validate_ab_pair(
    original: Mapping[str, object], adaptive: Mapping[str, object]
) -> None:
    """Reject an A/B pair with any difference outside the adaptive JSON key."""
    for field in MATCHED_PROVENANCE_FIELDS:
        original_value = _required_value(original, field)
        adaptive_value = _required_value(adaptive, field)
        if original_value != adaptive_value:
            raise ValueError(f"A/B pair mismatch in {field}")

    original_config = _strip_adaptive_config_key(
        _required_value(original, "resolved_config")
    )
    adaptive_config = _strip_adaptive_config_key(
        _required_value(adaptive, "resolved_config")
    )
    if original_config != adaptive_config:
        raise ValueError(
            f"A/B pair mismatch in resolved Hydra config outside {CONFIG_ENV_KEY}"
        )


def _record_sort_key(record: ResultRecord) -> tuple[int, int, int]:
    arm_order = 0 if record.arm == "original" else 1
    return record.repeat, arm_order, record.step


def _record_dict(record: ResultRecord) -> dict[str, object]:
    return dict(zip(ResultRecord._fields, record))


def _write_new_text(path: Path, text: str, *, newline: str | None = None) -> None:
    """Atomically create a text output, refusing to replace any existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline=newline) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def write_summaries(
    records: Sequence[ResultRecord], json_path: Path, csv_path: Path
) -> None:
    """Create deterministic JSON and CSV summaries without replacing outputs."""
    for path in (json_path, csv_path):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")

    ordered = sorted(records, key=_record_sort_key)
    dictionaries = [_record_dict(record) for record in ordered]
    _write_new_text(
        json_path,
        json.dumps(dictionaries, indent=2, sort_keys=True) + "\n",
    )

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=CSV_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(dictionaries)
    _write_new_text(csv_path, stream.getvalue(), newline="")


def not_applicable_result(reason: str) -> dict[str, object]:
    """Build the stable zero-hit Qwen result and named efficacy fallback."""
    if not reason.strip():
        raise ValueError("not-applicable reason must be non-empty")
    return {
        "fallback": {
            "model": "Nemotron 3 Ultra",
            "tensor_parallel_size": 4,
        },
        "reason": reason,
        "status": "not-applicable",
        "workload": "Qwen/Qwen3-30B-A3B",
    }


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON object from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_object(path: Path, value: Mapping[str, object]) -> None:
    _write_new_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_config(config_path: Path, overrides: Sequence[str]) -> dict[str, object]:
    # NeMo-RL/OmegaConf are runtime-only dependencies in the staged container.
    from nemo_rl.utils.config import (  # pylint: disable=import-outside-toplevel
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

    register_omegaconf_resolvers()
    config = load_config(config_path)
    if overrides:
        config = parse_hydra_overrides(config, list(overrides))
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("resolved Hydra config must be a mapping")
    return resolved


def _parse_command(args: argparse.Namespace) -> int:
    records: list[ResultRecord] = []
    for path in args.log:
        records.extend(parse_log(path.read_text(encoding="utf-8", errors="replace")))
    write_summaries(records, args.json_output, args.csv_output)
    return 0


def _validate_pair_command(args: argparse.Namespace) -> int:
    validate_ab_pair(_load_json_object(args.original), _load_json_object(args.adaptive))
    print("matched-ab-pair")
    return 0


def _emit_context_command(args: argparse.Namespace) -> int:
    metadata = _load_json_object(args.metadata)
    context = {key: _required_value(metadata, key) for key in REQUIRED_METADATA}
    print(METADATA_PREFIX + json.dumps(context, sort_keys=True, separators=(",", ":")))
    return 0


def _not_applicable_command(args: argparse.Namespace) -> int:
    _write_json_object(args.output, not_applicable_result(args.reason))
    return 0


def _resolve_config_command(args: argparse.Namespace) -> int:
    _write_json_object(args.output, _resolve_config(args.config, tuple(args.override)))
    print(_sha256(args.output))
    return 0


def _make_metadata_command(args: argparse.Namespace) -> int:
    resolved_config = _load_json_object(args.resolved_config)
    metadata: dict[str, object] = {
        "arm": args.arm,
        "checkpoint": args.checkpoint,
        "config_hash": args.config_hash,
        "container_digest": args.container_digest,
        "generation_num_gpus": args.num_nodes * args.gpus_per_node,
        "nemo_rl_commit": args.nemo_rl_commit,
        "num_samples": args.num_samples,
        "repeat": args.repeat,
        "resolved_config": resolved_config,
        "resolved_config_sha256": _sha256(args.resolved_config),
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "topology": {
            "gpus_per_node": args.gpus_per_node,
            "num_nodes": args.num_nodes,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "vllm_commit": args.vllm_commit,
    }
    _write_json_object(args.output, metadata)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse = subparsers.add_parser("parse")
    parse.add_argument("--log", type=Path, action="append", required=True)
    parse.add_argument("--json-output", type=Path, required=True)
    parse.add_argument("--csv-output", type=Path, required=True)
    parse.set_defaults(handler=_parse_command)

    validate_pair = subparsers.add_parser("validate-pair")
    validate_pair.add_argument("--original", type=Path, required=True)
    validate_pair.add_argument("--adaptive", type=Path, required=True)
    validate_pair.set_defaults(handler=_validate_pair_command)

    emit_context = subparsers.add_parser("emit-context")
    emit_context.add_argument("--metadata", type=Path, required=True)
    emit_context.set_defaults(handler=_emit_context_command)

    not_applicable = subparsers.add_parser("not-applicable")
    not_applicable.add_argument("--reason", required=True)
    not_applicable.add_argument("--output", type=Path, required=True)
    not_applicable.set_defaults(handler=_not_applicable_command)

    resolve_config = subparsers.add_parser("resolve-config")
    resolve_config.add_argument("--config", type=Path, required=True)
    resolve_config.add_argument("--override", action="append", default=[])
    resolve_config.add_argument("--output", type=Path, required=True)
    resolve_config.set_defaults(handler=_resolve_config_command)

    make_metadata = subparsers.add_parser("make-metadata")
    make_metadata.add_argument(
        "--arm",
        choices=("trace", "shmoo", "original", "adaptive"),
        required=True,
    )
    make_metadata.add_argument("--repeat", type=int, required=True)
    make_metadata.add_argument("--nemo-rl-commit", required=True)
    make_metadata.add_argument("--vllm-commit", required=True)
    make_metadata.add_argument("--container-digest", required=True)
    make_metadata.add_argument("--config-hash", required=True)
    make_metadata.add_argument("--checkpoint", required=True)
    make_metadata.add_argument("--tensor-parallel-size", type=int, required=True)
    make_metadata.add_argument("--seed", type=int, required=True)
    make_metadata.add_argument("--num-nodes", type=int, required=True)
    make_metadata.add_argument("--gpus-per-node", type=int, required=True)
    make_metadata.add_argument("--num-samples", type=int, required=True)
    make_metadata.add_argument("--resolved-config", type=Path, required=True)
    make_metadata.add_argument("--output", type=Path, required=True)
    make_metadata.set_defaults(handler=_make_metadata_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one parser, validation, or metadata command."""
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
