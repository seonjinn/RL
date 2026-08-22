#!/usr/bin/env python3

import argparse
from typing import NamedTuple


class BaselineArm(NamedTuple):
    name: str
    draft_enabled: bool


_ARMS = {
    "fixed": BaselineArm("fixed", False),
    "online": BaselineArm("online", True),
}


def resolve_arm(name: str) -> BaselineArm:
    try:
        return _ARMS[name]
    except KeyError as error:
        raise ValueError(f"Unsupported baseline arm: {name}") from error


def runtime_overrides(
    arm: BaselineArm,
    *,
    target_snapshot: str,
    drafter_snapshot: str,
    scratch_root: str,
    wandb_run_id: str,
    expected_head: str,
    wandb_project: str = "sna-nemo-rl-dflash-pack-cp2-baseline",
) -> tuple[str, ...]:
    draft_enabled = str(arm.draft_enabled).lower()
    return (
        "grpo.seed=42",
        "grpo.max_num_steps=30",
        "grpo.num_prompts_per_step=8",
        "grpo.num_generations_per_prompt=4",
        "grpo.val_period=0",
        "grpo.val_at_start=false",
        "grpo.val_at_end=false",
        "checkpointing.enabled=false",
        f"checkpointing.checkpoint_dir={scratch_root}/checkpoints",
        f"policy.model_name={target_snapshot}",
        f"policy.tokenizer.name={target_snapshot}",
        f"policy.draft.model_name={drafter_snapshot}",
        "policy.train_global_batch_size=32",
        "policy.train_micro_batch_size=1",
        "policy.logprob_batch_size=1",
        "policy.sequence_packing.enabled=true",
        "policy.megatron_cfg.tensor_model_parallel_size=2",
        "policy.megatron_cfg.pipeline_model_parallel_size=1",
        "policy.megatron_cfg.context_parallel_size=2",
        "policy.megatron_cfg.sequence_parallel=true",
        "policy.make_sequence_length_divisible_by=16",
        f"policy.draft.enabled={draft_enabled}",
        "policy.draft.gamma=5",
        "policy.draft.update_probe_enabled=false",
        "policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5",
        f"policy.generation.vllm_kwargs.speculative_config.model={drafter_snapshot}",
        "data.shuffle=true",
        "data.train.dataset_name=DAPOMath17K",
        "data.train.seed=42",
        f"logger.log_dir={scratch_root}/logs",
        "logger.wandb_enabled=true",
        "logger.tensorboard_enabled=false",
        "++logger.wandb.entity=nvidia",
        f"++logger.wandb.project={wandb_project}",
        "logger.wandb.group=qwen3-8b-dflash-pack-cp2-k5-baseline",
        f"logger.wandb.name=qwen3-8b-dflash-pack-cp2-k5-{arm.name}-r5",
        f"logger.wandb.tags=[baseline,cache-hash-bound,packing,tp2,cp2,dflash,k5,{arm.name}]",
        f"++logger.wandb.id={wandb_run_id}",
        "++logger.wandb.resume=never",
        f"++logger.wandb.config.ab_arm={arm.name}",
        f"++logger.wandb.config.draft_training_enabled={draft_enabled}",
        f"++logger.wandb.config.draft_refit_enabled={draft_enabled}",
        "++logger.wandb.config.cache_hash_bound=true",
        "++logger.wandb.config.frozen_jsonl=false",
        f"++logger.wandb.config.harness_sha={expected_head}",
        "++logger.wandb.config.product_source_sha=443e7243ae2a235b6dcd8f4918fea86e693630a9",
        "++logger.wandb.config.dataset_revision=65877096c24ffa7abc4e4fa5edb95cf3413a5674",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--target-snapshot", required=True)
    parser.add_argument("--drafter-snapshot", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--wandb-project", default="sna-nemo-rl-dflash-pack-cp2-baseline"
    )
    args = parser.parse_args()
    print(
        *runtime_overrides(
            resolve_arm(args.arm),
            target_snapshot=args.target_snapshot,
            drafter_snapshot=args.drafter_snapshot,
            scratch_root=args.scratch_root,
            wandb_run_id=args.wandb_run_id,
            expected_head=args.expected_head,
            wandb_project=args.wandb_project,
        ),
        sep="\n",
    )


if __name__ == "__main__":
    main()
