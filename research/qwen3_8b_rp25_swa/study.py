"""New export study settings on the previously exercised Q8 cadence workload."""

from __future__ import annotations

import argparse
from dataclasses import replace

from research.qwen3_8b_draft_cadence_200step.matrix import (
    Arm,
    USER_ROOT,
    build_arms,
    render_hydra_overrides,
)


def build_new_arms() -> tuple[Arm, ...]:
    arms = []
    for old in build_arms():
        if old.cadence == "adaptive":
            continue
        snapshot = (
            None
            if old.drafter == "none"
            else (
                f"{USER_ROOT}/specdec_ptv23/ptv3_swa/"
                f"sd2p3rp-q8b-base-ptv3rp25-{old.drafter}-b8-16n/exported-checkpoint-44000"
            )
        )
        schedule = dict(old.schedule) if old.schedule is not None else None
        if old.cadence == "static":
            assert schedule is not None
            schedule["fixed_interval"] = 201
        arms.append(
            replace(
                old,
                name=old.name.replace("static", "frozen"),
                max_steps=200,
                required_checkpoint_steps=(50, 100, 150, 200),
                schedule=schedule,
                drafter_snapshot=snapshot,
                drafter_revision=None,
                wandb_group="q8-new-draft-rp25-44000-b8-k5-cadence-200step-20260914",
            )
        )
    return tuple(arms)


def overrides(
    arm: Arm, result_dir: str, *, canary: bool = False, resume_check: bool = False
) -> tuple[str, ...]:
    if canary and resume_check:
        raise ValueError("canary and resume_check are mutually exclusive")
    if canary or resume_check:
        if arm.cadence != "always":
            raise ValueError("first canary requires always-online cadence")
        arm = replace(
            arm,
            max_steps=4 if resume_check else 2,
            required_checkpoint_steps=(2, 4) if resume_check else (2,),
        )
    values = dict(
        item.lstrip("+").split("=", 1)
        for item in render_hydra_overrides(arm, result_dir=result_dir)
    )
    label = (
        "Baseline"
        if arm.drafter == "none"
        else (
            f"{'DFlash' if arm.drafter == 'dflash' else 'DSpark'}K5-"
            f"{'frozen' if arm.cadence == 'static' else arm.cadence}"
        )
    )
    suffix = "-resume-check" if resume_check else "-canary" if canary else ""
    values["logger.wandb.name"] = f"Qwen3-8B-{label}{suffix}"
    values["logger.wandb.group"] += suffix
    if arm.drafter != "none":
        values.update(
            {
                "policy.draft.model_revision": "null",
                "policy.generation.vllm_kwargs.speculative_config.revision": "null",
                "policy.draft.sliding_window": "2048",
                "policy.draft.update_probe_enabled": "true"
                if canary or resume_check
                else "false",
                "policy.draft.num_layers": "5",
                "policy.draft.target_hidden_state_layer_ids": "[1,9,17,25,33]",
                "policy.draft.mask_token_id": "151669",
                "policy.generation.vllm_kwargs.speculative_config.attention_backend": "FLASH_ATTN",
            }
        )
        values[
            "policy.draft.gamma"
            if arm.drafter == "dflash"
            else "policy.draft.block_size"
        ] = "7" if arm.drafter == "dflash" else "8"
    return tuple(f"++{key}={value}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", required=True, choices=[a.name for a in build_new_arms()]
    )
    parser.add_argument("--result-dir", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--canary", action="store_true")
    mode.add_argument("--resume-check", action="store_true")
    parser.add_argument("--recipe", action="store_true")
    args = parser.parse_args()
    arm = next(a for a in build_new_arms() if a.name == args.arm)
    if args.recipe:
        print(arm.config_path)
    else:
        print(
            "\n".join(
                overrides(
                    arm,
                    args.result_dir,
                    canary=args.canary,
                    resume_check=args.resume_check,
                )
            )
        )


if __name__ == "__main__":
    main()
