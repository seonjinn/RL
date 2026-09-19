# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real GB200 refit correctness gates, deliberately outside the unit GPU lane.

Collectable on CPU. Execution requires the registered 4x4 or 6x4 wrapper.
Only test inputs, raw-logprob reporting and the test worker are specialized;
the inherited performance recipe's parallelism and kernel settings are unchanged.
"""

import json
import os
import subprocess
import time
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    ROOT / "examples/configs/recipes/llm/vllm-destructive-refit-qwen3-30ba3b-4n4g.yaml"
)
CONTROL = (
    ROOT / "tests/test_suites/llm/grpo-qwen3.5-35ba3b-6n4g-async-1off-bf16-trtllm.sh"
)
PROMPTS = (
    "The capital of France is",
    "Calculate 17 plus 25. The answer is",
    "A prime number is",
    "Water freezes at",
)


def _write_record(name: str, value: Any) -> None:
    path = Path(os.environ["NRL_REFIT_SLEEP_RUN_DIR"]) / f"{name}.json"
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass
class Runtime:
    cluster: Any
    policy: Any
    generation: Any
    generation_config: Any
    policy_config: Any
    tokenizer: Any
    prompts: Any
    timeout_s: float

    def fresh(self, state: str, checkpoint: Path) -> Any:
        # Import GPU-facing controllers only during the GB200 execution path.
        from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
        from nemo_rl.models.policy.lm_policy import Policy
        from nemo_rl.weight_sync.ipc_weight_synchronizer import IPCWeightSynchronizer

        assert self.generation is None and self.policy is None
        fresh = VllmGeneration(
            self.cluster, deepcopy(self.generation_config), name_prefix=f"fresh_{state}"
        )
        policy = None
        try:
            assert fresh.finish_generation() is True
            policy = Policy(
                self.cluster,
                deepcopy(self.policy_config),
                self.tokenizer,
                name_prefix=f"fresh_policy_{state}",
                weights_path=str(checkpoint),
                init_reference_model=False,
            )
            sync = IPCWeightSynchronizer(policy, fresh, refit_timeout_s=self.timeout_s)
            fresh.weight_synchronizer = sync
            sync.init_communicator()
            assert not sync.can_discard_generation_weights
            assert not sync.generation_weights_discarded
            sync.sync_weights()
            assert not sync.is_stale
            return fresh.generate(self.prompts, greedy=True)
        finally:
            try:
                assert fresh.shutdown() is True
            finally:
                if policy is not None:
                    assert policy.shutdown() is True

    def shutdown_models(self) -> None:
        try:
            if self.generation is not None:
                assert self.generation.shutdown() is True
                self.generation = None
        finally:
            if self.policy is not None:
                assert self.policy.shutdown() is True
                self.policy = None


@pytest.fixture
def qwen3_runtime() -> Iterator[Runtime]:
    if os.environ.get("NRL_REFIT_SLEEP_RUN") != "1":
        pytest.skip("requires the dedicated 4x4 GB200 wrapper")

    # The driver and workers use the repository's normal isolated environments.
    import torch
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.utils import get_tokenizer, set_seed
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
        get_actor_python_env,
    )
    from nemo_rl.distributed.virtual_cluster import (
        RayVirtualCluster,
        init_ray,
        prepare_segment_topology,
    )
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
    from nemo_rl.models.policy.lm_policy import Policy
    from nemo_rl.utils.config import load_config
    from nemo_rl.weight_sync.ipc_weight_synchronizer import IPCWeightSynchronizer
    from tests.functional.refit_sleep_runtime import verify_gb200_nodes

    config = OmegaConf.to_container(load_config(RECIPE), resolve=True)
    assert config["cluster"]["num_nodes"] == 4
    assert config["cluster"]["gpus_per_node"] == 4
    assert config["cluster"]["segment_size"] == 4
    policy_config = config["policy"]
    # Normal GRPO setup supplies this scheduler horizon before Policy construction.
    policy_config["megatron_cfg"]["train_iters"] = config["grpo"]["max_num_steps"]
    assert policy_config["megatron_cfg"]["expert_model_parallel_size"] == 16
    generation_config = deepcopy(policy_config["generation"])
    assert generation_config["colocated"]["enabled"] is True
    assert generation_config["vllm_kwargs"]["moe_backend"] == "flashinfer_trtllm"
    assert generation_config["vllm_cfg"]["precision"] == "fp8"
    assert generation_config["vllm_cfg"]["is_mx"] is True
    assert generation_config["vllm_cfg"]["quantization_ignore_patterns"] == [
        "model.layers.*.self_attn.*",
        "model.layers.*.mlp.gate",
        "lm_head",
    ]
    set_seed(config["grpo"]["seed"])
    tokenizer = get_tokenizer(policy_config["tokenizer"])
    generation_config["model_name"] = policy_config["model_name"]
    generation_config["vllm_cfg"]["logprobs_mode"] = "raw_logprobs"
    generation_config = configure_generation_config(generation_config, tokenizer)
    _write_record("resolved-recipe", config)
    _write_record("generation-config", generation_config)
    init_ray()
    verify_gb200_nodes(expected_nodes=4)
    constraints, _, _ = prepare_segment_topology(4, 4)
    cluster = RayVirtualCluster(
        name="refit_sleep_gate",
        bundle_ct_per_node_list=[4] * 4,
        use_gpus=True,
        num_gpus_per_node=4,
        max_colocated_worker_groups=2,
        segment_size=4,
        node_resource_constraints=constraints,
    )
    extension = "tests.functional.refit_sleep_policy_worker.RefitSleepPolicyWorker"
    ACTOR_ENVIRONMENT_REGISTRY[extension] = get_actor_python_env(
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )
    timeout_s = float(os.environ["NRL_REFIT_SLEEP_DEADLINE_S"])
    generation = policy = runtime = None
    try:
        generation = VllmGeneration(cluster, generation_config, name_prefix="candidate")
        assert generation.finish_generation() is True
        policy = Policy(
            cluster,
            policy_config,
            tokenizer,
            worker_extension_cls_fqn=extension,
            init_reference_model=False,
        )
        sync = IPCWeightSynchronizer(policy, generation, refit_timeout_s=timeout_s)
        generation.weight_synchronizer = sync
        sync.init_communicator()
        assert not sync.can_discard_generation_weights
        assert not sync.generation_weights_discarded
        sync.sync_weights()  # A: initial preserving refit, never level 2.
        assert sync.can_discard_generation_weights, (
            "initial real A refit did not establish runtime coverage"
        )
        assert not sync.is_stale and not sync.generation_weights_discarded
        encoded = tokenizer(
            list(PROMPTS) * 4, padding=True, return_tensors="pt", padding_side="right"
        )
        prompts = BatchedDataDict(
            {
                "input_ids": encoded["input_ids"],
                "input_lengths": encoded["attention_mask"].sum(dim=1).to(torch.int32),
            }
        )
        _write_record(
            "fixed-inputs", {key: value.tolist() for key, value in prompts.items()}
        )
        runtime = Runtime(
            cluster,
            policy,
            generation,
            generation_config,
            policy_config,
            tokenizer,
            prompts,
            timeout_s,
        )
        yield runtime
    finally:
        try:
            if runtime is not None:
                runtime.shutdown_models()
            else:
                try:
                    if generation is not None:
                        assert generation.shutdown() is True
                finally:
                    if policy is not None:
                        policy.shutdown()
        finally:
            cluster.shutdown()


def _measure(runtime: Runtime, state: str) -> tuple[Any, dict[str, float]]:
    # Match the real Moonlight / HF parity helpers' sequence alignment.
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from tests.functional.refit_sleep_utils import observe, parity_metrics

    result = runtime.generation.generate(runtime.prompts, greedy=True)
    observation = observe(runtime.prompts["input_lengths"], result)
    _write_record(f"{state}-candidate", asdict(observation))
    assert runtime.generation.finish_generation() is True
    runtime.policy.prepare_for_lp_inference()
    try:
        policy_result = runtime.policy.get_logprobs(
            BatchedDataDict(
                {
                    "input_ids": result["output_ids"],
                    "input_lengths": result["unpadded_sequence_lengths"],
                }
            )
        )
        policy_observation = observe(
            runtime.prompts["input_lengths"],
            {
                **result,
                "logprobs": policy_result["logprobs"],
            },
        )
        _write_record(f"{state}-megatron", asdict(policy_observation))
        return observation, parity_metrics(observation, policy_observation)
    finally:
        runtime.policy.finish_inference()
        runtime.policy.offload_after_refit()


@pytest.mark.mcore
@pytest.mark.vllm
def test_qwen3_mxfp8_destructive_refit_abc(qwen3_runtime: Runtime) -> None:
    import ray  # GPU runtime is optional during collection.

    from nemo_rl.models.generation.interfaces import GenerationNextPhase
    from tests.functional.refit_sleep_utils import (
        Tolerances,
        assert_distinct,
        fresh_error,
        observe,
    )

    runtime = qwen3_runtime
    previous, metrics_a = _measure(runtime, "A")
    _write_record("A-metrics", metrics_a)
    observations = {"A": {**metrics_a, "fresh_logprob_max_abs": 0.0}}
    candidates = {"A": previous}
    checkpoints = {}
    for state, factor in (("B", 1.0625), ("C", 0.875)):
        assert runtime.generation.prepare_for_generation() is True
        assert (
            runtime.generation.finish_generation_for_next_phase(
                GenerationNextPhase.TRAIN_THEN_FULL_REFIT
            )
            is True
        )
        sync = runtime.generation.weight_synchronizer
        assert sync.generation_weights_discarded, (
            f"{state} did not select destructive sleep"
        )
        runtime.policy.prepare_for_training()
        changes = ray.get(
            runtime.policy.worker_group.run_all_workers_single_data(
                "apply_refit_test_update", factor=factor
            )
        )
        assert len(changes) == 16 and all(count > 0 for count in changes)
        runtime.policy.finish_training()
        sync.sync_weights()
        assert sync.can_discard_generation_weights
        assert not sync.is_stale and not sync.generation_weights_discarded
        candidate, metrics = _measure(runtime, state)
        assert_distinct(previous, candidate, logprob_atol=0)
        # Checkpoint exact B/C through the normal Megatron lifecycle. Keeping a
        # second live receiver would share the per-device IPC address and could
        # steal candidate batches, even if its generation engine were asleep.
        checkpoint = (
            Path(os.environ["NRL_REFIT_SLEEP_ARTIFACT_DIR"]) / f"checkpoint-{state}"
        )
        assert not checkpoint.exists(), "use a new artifact directory for every run"
        runtime.policy.save_checkpoint(str(checkpoint), is_final_checkpoint=True)
        runtime.policy.finalize_async_save()
        runtime.policy.offload_after_refit()
        checkpoints[state] = checkpoint
        candidates[state] = candidate
        observations[state] = metrics
        _write_record(f"{state}-metrics", metrics)
        _write_record(
            f"{state}-update", {"factor": factor, "changed_elements_per_rank": changes}
        )
        previous = candidate
    runtime.shutdown_models()
    for state in ("B", "C"):
        fresh = observe(
            runtime.prompts["input_lengths"], runtime.fresh(state, checkpoints[state])
        )
        _write_record(f"{state}-fresh", asdict(fresh))
        observations[state]["fresh_logprob_max_abs"] = fresh_error(
            candidates[state], fresh
        )
    _write_record("observations", observations)
    tolerance_path = os.environ.get("NRL_REFIT_SLEEP_TOLERANCES")
    if not tolerance_path:
        pytest.fail(
            "GB200 observations recorded; tolerance calibration is required, this is NOT an accepted correctness gate"
        )
    record = json.loads(Path(tolerance_path).read_text())
    limits = Tolerances.from_record(record)
    _write_record("tolerances", record)
    assert_distinct(
        candidates["A"], candidates["B"], logprob_atol=limits.fresh_logprob_atol
    )
    assert_distinct(
        candidates["B"], candidates["C"], logprob_atol=limits.fresh_logprob_atol
    )
    for metrics in observations.values():
        limits.check(metrics)


@pytest.mark.mcore
@pytest.mark.vllm
def test_qwen3_mxfp8_missing_manifest_after_discard(
    qwen3_runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ray  # GPU runtime is optional during collection.

    from nemo_rl.models.generation.interfaces import GenerationNextPhase
    from tests.functional.refit_sleep_utils import (
        failure_deadline,
        incomplete_receiver_manifest,
    )

    runtime = qwen3_runtime
    generation = runtime.generation
    sync = generation.weight_synchronizer
    assert generation.prepare_for_generation() is True
    # Obtain the sender's real metadata before starting the discarded-state clock.
    source = runtime.policy.prepare_refit_info(
        refit_payload_mode=generation.get_refit_payload_mode()
    )
    sender_refs: list[Any] = []
    receiver_refs: list[Any] = []
    wakes: list[list[str]] = []
    original_send = runtime.policy.stream_weights_via_ipc_zmq
    original_receive = generation.update_weights_via_ipc_zmq
    original_wake = generation.prepare_for_generation

    def send(**kwargs: Any) -> list[Any]:
        refs = original_send(**kwargs)
        sender_refs.extend(refs)
        return refs

    def receive() -> list[Any]:
        refs = original_receive()
        receiver_refs.extend(refs)
        return refs

    def wake(*, tags: list[str], **kwargs: Any) -> bool:
        wakes.append(tags)
        return original_wake(tags=tags, **kwargs)

    monkeypatch.setattr(runtime.policy, "stream_weights_via_ipc_zmq", send)
    monkeypatch.setattr(generation, "update_weights_via_ipc_zmq", receive)
    monkeypatch.setattr(generation, "prepare_for_generation", wake)
    started = time.monotonic()
    with failure_deadline(runtime.timeout_s):
        try:
            assert (
                generation.finish_generation_for_next_phase(
                    GenerationNextPhase.TRAIN_THEN_FULL_REFIT
                )
                is True
            )
            assert sync.generation_weights_discarded
            # No transport hook: all real batches drain normally, then the existing
            # COMPLETE validation rejects the unsent entry and ACKs the sender.
            generation.prepare_refit_info(incomplete_receiver_manifest(source))
            runtime.policy.prepare_for_lp_inference()
            with pytest.raises(
                (RuntimeError, ray.exceptions.RayTaskError),
                match="Weight transfer failed|missing keys",
            ) as failure:
                sync.sync_weights()
            assert len(sender_refs) == 16 and len(receiver_refs) == generation.dp_size
            remaining = runtime.timeout_s - (time.monotonic() - started)
            assert remaining > 0, "failure exceeded the configured deadline"
            ready, pending = ray.wait(
                sender_refs + receiver_refs,
                num_returns=len(sender_refs + receiver_refs),
                timeout=remaining,
            )
            assert not pending, "IPC failure stranded sender or receiver work"
            ray.get(
                sender_refs, timeout=remaining
            )  # All senders observed COMPLETE ACK.
            assert sync.is_stale and sync.generation_weights_discarded
            assert not sync.can_discard_generation_weights
            assert wakes == [["weights"]], "failed discarded-state refit woke KV cache"
        finally:
            runtime.shutdown_models()
        elapsed = time.monotonic() - started
        assert elapsed < runtime.timeout_s
        _write_record(
            "manifest-failure",
            {
                "error": str(failure.value),
                "elapsed_s_including_shutdown": elapsed,
                "deadline_s": runtime.timeout_s,
                "sender_count": len(sender_refs),
                "receiver_count": len(receiver_refs),
                "pending_refs": len(pending),
                "wakes": wakes,
                "discarded": True,
                "stale": True,
            },
        )


@pytest.mark.mcore
@pytest.mark.vllm
def test_qwen35_bf16_nccl_reshard_preserving_control() -> None:
    if os.environ.get("NRL_REFIT_SLEEP_RUN") != "1":
        pytest.skip("requires the dedicated 6x4 GB200 wrapper")
    # Reuse the existing non-colocated recipe, runtime and KL control bound.
    # This is a preserving/reset-only control, not a level-2 test.
    from nemo_rl.distributed.virtual_cluster import init_ray
    from tests.functional.refit_sleep_runtime import verify_gb200_nodes
    from tests.functional.refit_sleep_utils import assert_bf16_control_metrics

    init_ray()
    verify_gb200_nodes(expected_nodes=6)
    log_path = Path(os.environ["NRL_REFIT_SLEEP_RUN_DIR"]) / "bf16-control.log"
    with log_path.open("w") as log:
        subprocess.run(
            ["bash", str(CONTROL)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=13800,
        )
    log_text = log_path.read_text()
    assert "selected_mode=discard" not in log_text
    assert "fallback_reason=generation_not_colocated" in log_text
    metrics_path = Path(os.environ["NRL_TEST_RUN_DIR"]) / "metrics.json"
    assert_bf16_control_metrics(json.loads(metrics_path.read_text()))
