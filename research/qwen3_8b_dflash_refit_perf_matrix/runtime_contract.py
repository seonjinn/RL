#!/usr/bin/env python3

import argparse
from typing import NamedTuple


class MatrixCell(NamedTuple):
    name: str
    shape: str
    arm: str
    config_path: str
    gbs: int
    mbs: int
    logprob_mbs: int
    prompts: int
    generations: int
    first_arm: str


_CONFIGS = {
    "fixed": "research/qwen3_8b_dflash_fixed_dense_control/config.yaml",
    "online": "research/qwen3_8b_dflash_online_cp1/config.yaml",
}
_SHAPES = {
    "gbs32_mbs1": (32, 1, 8, "fixed"),
    "gbs64_mbs1": (64, 1, 16, "online"),
    "gbs64_mbs2": (64, 2, 16, "fixed"),
}


def matrix_cells() -> tuple[MatrixCell, ...]:
    return tuple(
        MatrixCell(
            name=f"{shape}_{arm}",
            shape=shape,
            arm=arm,
            config_path=_CONFIGS[arm],
            gbs=gbs,
            mbs=mbs,
            logprob_mbs=1,
            prompts=prompts,
            generations=4,
            first_arm=first_arm,
        )
        for shape, (gbs, mbs, prompts, first_arm) in _SHAPES.items()
        for arm in ("fixed", "online")
    )


def resolve_cell(name: str) -> MatrixCell:
    cells = {cell.name: cell for cell in matrix_cells()}
    try:
        return cells[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported matrix cell {name!r}; expected one of: "
            f"{', '.join(sorted(cells))}"
        ) from error


def runtime_overrides(
    cell: MatrixCell,
    *,
    target_snapshot: str,
    drafter_snapshot: str,
    scratch_root: str,
    wandb_run_id: str,
    expected_head: str,
    wandb_project: str = "sna-nemo-rl-online-drafter",
) -> tuple[str, ...]:
    draft_training = str(cell.arm == "online").lower()
    draft_refit = draft_training
    fixed_drafter = str(cell.arm == "fixed").lower()
    return (
        "grpo.max_num_steps=50",
        f"grpo.num_prompts_per_step={cell.prompts}",
        f"grpo.num_generations_per_prompt={cell.generations}",
        "grpo.val_period=1000000",
        "grpo.val_at_start=false",
        "grpo.val_at_end=false",
        "checkpointing.enabled=false",
        f"checkpointing.checkpoint_dir={scratch_root}/checkpoints",
        f"policy.model_name={target_snapshot}",
        f"policy.tokenizer.name={target_snapshot}",
        f"policy.draft.model_name={drafter_snapshot}",
        f"policy.train_global_batch_size={cell.gbs}",
        f"policy.train_micro_batch_size={cell.mbs}",
        f"policy.logprob_batch_size={cell.logprob_mbs}",
        "policy.draft.update_probe_enabled=false",
        "policy.sequence_packing.enabled=false",
        (f"policy.generation.vllm_kwargs.speculative_config.model={drafter_snapshot}"),
        f"logger.log_dir={scratch_root}/logs",
        "logger.wandb_enabled=true",
        "logger.tensorboard_enabled=false",
        "++logger.wandb.entity=nvidia",
        f"++logger.wandb.project={wandb_project}",
        "logger.wandb.group=qwen3-8b-dflash-refit-perf-matrix",
        f"logger.wandb.name=qwen3-8b-dflash-refit-{cell.name}",
        (
            "logger.wandb.tags=[dflash,qwen3-8b,k7,cudagraph,nonnsys,"
            f"refit-perf,{cell.shape},{cell.arm}]"
        ),
        f"++logger.wandb.id={wandb_run_id}",
        "++logger.wandb.resume=never",
        f"++logger.wandb.config.matrix_cell={cell.name}",
        f"++logger.wandb.config.matrix_shape={cell.shape}",
        f"++logger.wandb.config.ab_arm={cell.arm}",
        f"logger.wandb.config.draft_training_enabled={draft_training}",
        f"logger.wandb.config.draft_refit_enabled={draft_refit}",
        f"++logger.wandb.config.fixed_public_drafter={fixed_drafter}",
        f"++logger.wandb.config.gbs={cell.gbs}",
        f"++logger.wandb.config.mbs={cell.mbs}",
        f"++logger.wandb.config.logprob_mbs={cell.logprob_mbs}",
        f"++logger.wandb.config.harness_sha={expected_head}",
        "++logger.wandb.config.product_source_sha="
        "4d8a54538d694f81f65bf2b431c5b5ed6a3017ca",
        "++logger.wandb.config.performance_window=steps_5_through_49",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--target-snapshot", required=True)
    parser.add_argument("--drafter-snapshot", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--wandb-project", default="sna-nemo-rl-online-drafter")
    args = parser.parse_args()
    cell = resolve_cell(args.cell)
    print(
        *runtime_overrides(
            cell,
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
