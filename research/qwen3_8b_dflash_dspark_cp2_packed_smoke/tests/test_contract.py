import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).parents[1]


def _module(name: str) -> ModuleType:
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("arm", "config_name", "draft_field", "draft_value"),
    (
        ("dflash", "grpo-qwen3-8b-1n8g-megatron-dflash.yaml", "gamma", "5"),
        ("dspark", "grpo-qwen3-8b-1n8g-megatron-dspark.yaml", "block_size", "7"),
    ),
)
def test_runtime_contract_is_cp2_packed_online_smoke(
    arm: str,
    config_name: str,
    draft_field: str,
    draft_value: str,
) -> None:
    contract = _module("contract.py")
    cell = contract.resolve_arm(arm)
    overrides = contract.runtime_overrides(
        cell,
        target_snapshot="/lustre/target/b968826d9c46dd6066d109eabc6255188de91218",
        drafter_snapshot=f"/lustre/{arm}/snapshot",
        scratch_root=f"/raid/scratch/123/{arm}",
        wandb_run_id=f"q8-cp2-{arm}-run",
        expected_head="a" * 40,
        wandb_project="sna-specdec-cp2-validation",
    )

    assert cell.config_path.name == config_name
    for required in (
        "data_plane.enabled=true",
        "grpo.max_num_steps=2",
        "grpo.num_prompts_per_step=2",
        "grpo.num_generations_per_prompt=4",
        "policy.train_global_batch_size=8",
        "policy.train_micro_batch_size=1",
        "policy.logprob_batch_size=1",
        "policy.megatron_cfg.tensor_model_parallel_size=2",
        "policy.megatron_cfg.pipeline_model_parallel_size=1",
        "policy.megatron_cfg.context_parallel_size=2",
        "policy.megatron_cfg.sequence_parallel=true",
        "policy.megatron_cfg.use_fused_linear_logprobs=false",
        "policy.sequence_packing.enabled=true",
        "policy.make_sequence_length_divisible_by=16",
        "policy.draft.enabled=true",
        "policy.draft.update_probe_enabled=true",
        f"policy.draft.{draft_field}={draft_value}",
        "policy.generation.vllm_cfg.tensor_parallel_size=1",
        "policy.generation.vllm_cfg.pipeline_parallel_size=1",
        "policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN",
        "policy.generation.vllm_kwargs.compilation_config.backend=eager",
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE",
        "checkpointing.enabled=false",
        "logger.wandb_enabled=true",
    ):
        assert required in overrides
    assert f"policy.draft.model_name=/lustre/{arm}/snapshot" in overrides
    assert (
        f"policy.generation.vllm_kwargs.speculative_config.model=/lustre/{arm}/snapshot"
        in overrides
    )
    if arm == "dspark":
        assert "policy.draft.model_revision=null" in overrides


def test_unknown_arm_is_rejected() -> None:
    contract = _module("contract.py")
    with pytest.raises(ValueError, match="Unsupported smoke arm"):
        contract.resolve_arm("baseline")


def test_runner_requires_online_update_refit_and_cuda_graph_evidence() -> None:
    runner = (ROOT / "run_oci_hsg.sbatch").read_text()

    for marker in (
        "#SBATCH --nodes=1",
        "#SBATCH --gres=gpu:4",
        'readonly scratch_root="/raid/scratch/${SLURM_JOB_ID}"',
        '[[ "${REMOTE_REPO}" == /home/* ]]',
        '[[ "${FINAL_DIR}" == /lustre/* ]]',
        'git -C "${REMOTE_REPO}" submodule status --recursive',
        "Draft Loss:",
        "draft_update_probe=complete",
        "draft_refit_manifest=draft_count=",
        "draft_post_update_refit=complete step=1",
        "Capturing CUDA graphs (PIECEWISE)",
        "Graph capturing finished",
        "Step 2",
    ):
        assert marker in runner
    assert "WANDB_API_KEY" in runner
    assert "CONTAINER_SHA256" in runner
