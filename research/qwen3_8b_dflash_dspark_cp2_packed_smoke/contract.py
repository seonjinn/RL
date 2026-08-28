#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import NamedTuple


_REPO_ROOT = Path(__file__).parents[2]
_RECIPE_ROOT = _REPO_ROOT / "examples/configs/recipes/llm"


class SmokeArm(NamedTuple):
    name: str
    config_path: Path
    draft_size_field: str
    draft_size: int


_ARMS = {
    "dflash": SmokeArm(
        "dflash",
        _RECIPE_ROOT / "grpo-qwen3-8b-1n8g-megatron-dflash.yaml",
        "gamma",
        5,
    ),
    "dspark": SmokeArm(
        "dspark",
        _RECIPE_ROOT / "grpo-qwen3-8b-1n8g-megatron-dspark.yaml",
        "block_size",
        7,
    ),
}


def resolve_arm(name: str) -> SmokeArm:
    try:
        return _ARMS[name]
    except KeyError as error:
        raise ValueError(f"Unsupported smoke arm: {name}") from error


def runtime_overrides(
    arm: SmokeArm,
    *,
    target_snapshot: str,
    drafter_snapshot: str,
    scratch_root: str,
    wandb_run_id: str,
    expected_head: str,
    wandb_project: str,
    context_parallel_size: int,
) -> tuple[str, ...]:
    if context_parallel_size not in (1, 2):
        raise ValueError("context_parallel_size must be 1 or 2 for this single-node smoke")
    overrides = [
        "data_plane.enabled=true",
        "grpo.max_num_steps=2",
        "grpo.num_prompts_per_step=2",
        "grpo.num_generations_per_prompt=4",
        "grpo.val_period=0",
        "grpo.val_at_start=false",
        "grpo.val_at_end=false",
        "checkpointing.enabled=false",
        f"checkpointing.checkpoint_dir={scratch_root}/checkpoints",
        f"policy.model_name={target_snapshot}",
        f"policy.tokenizer.name={target_snapshot}",
        f"policy.draft.model_name={drafter_snapshot}",
        "policy.train_global_batch_size=8",
        "policy.train_micro_batch_size=1",
        "policy.logprob_batch_size=1",
        "policy.megatron_cfg.tensor_model_parallel_size=2",
        "policy.megatron_cfg.pipeline_model_parallel_size=1",
        f"policy.megatron_cfg.context_parallel_size={context_parallel_size}",
        "policy.megatron_cfg.sequence_parallel=true",
        "policy.megatron_cfg.use_fused_linear_logprobs=false",
        "policy.sequence_packing.enabled=true",
        "policy.make_sequence_length_divisible_by=16",
        "policy.draft.enabled=true",
        "+policy.draft.update_probe_enabled=true",
        f"policy.draft.{arm.draft_size_field}={arm.draft_size}",
        f"policy.generation.vllm_kwargs.speculative_config.model={drafter_snapshot}",
        f"policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens={arm.draft_size}",
        "+policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN",
        "policy.generation.vllm_cfg.tensor_parallel_size=1",
        "policy.generation.vllm_cfg.pipeline_parallel_size=1",
        "policy.generation.vllm_kwargs.compilation_config.backend=eager",
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE",
        f"logger.log_dir={scratch_root}/logs",
        "logger.wandb_enabled=true",
        "logger.tensorboard_enabled=false",
        "++logger.wandb.entity=nvidia",
        f"++logger.wandb.project={wandb_project}",
        f"+logger.wandb.group=qwen3-8b-cp{context_parallel_size}-packed-online-smoke",
        f"logger.wandb.name=qwen3-8b-{arm.name}-cp{context_parallel_size}-packed-online",
        f"+logger.wandb.tags=[cp{context_parallel_size},packing,online,draft-update,refit,{arm.name}]",
        f"++logger.wandb.id={wandb_run_id}",
        "++logger.wandb.resume=never",
        f"++logger.wandb.config.smoke_arm={arm.name}",
        f"++logger.wandb.config.context_parallel_size={context_parallel_size}",
        "++logger.wandb.config.sequence_packing=true",
        f"++logger.wandb.config.harness_sha={expected_head}",
    ]
    if arm.name == "dspark":
        overrides.append("policy.draft.model_revision=null")
    return tuple(overrides)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--target-snapshot", required=True)
    parser.add_argument("--drafter-snapshot", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--context-parallel-size", required=True, type=int)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    arm = resolve_arm(args.arm)
    if args.print_config:
        print(arm.config_path)
        return
    print(
        *runtime_overrides(
            arm,
            target_snapshot=args.target_snapshot,
            drafter_snapshot=args.drafter_snapshot,
            scratch_root=args.scratch_root,
            wandb_run_id=args.wandb_run_id,
            expected_head=args.expected_head,
            wandb_project=args.wandb_project,
            context_parallel_size=args.context_parallel_size,
        ),
        sep="\n",
    )


if __name__ == "__main__":
    main()
