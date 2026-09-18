"""Opt-in FAP/Inductor diagnostics; never alter the completed cadence cohort."""

from __future__ import annotations

import argparse
import json

from research.qwen3_8b_rp25_swa.study import build_new_arms, overrides
from research.qwen3_8b_draft_cadence_200step.matrix import Arm


def graph_overrides(
    arm: Arm,
    result_dir: str,
    seqs: int | None,
    *,
    packed: bool = False,
) -> tuple[str, ...]:
    if seqs not in (None, 8, 32, 64):
        raise ValueError("graph study supports S8, S32, S64 or baseline default")
    if seqs is None and arm.drafter != "none":
        raise ValueError("default concurrency is reserved for the baseline control")
    values = dict(
        x[2:].split("=", 1) for x in overrides(arm, result_dir, long_context=True)
    )
    # Target verification uses K+1=6 queries; DSpark's anchor-as-first uses K=5.
    # Cover both draft layouts, including every tail request count in this sweep.
    request_cap = seqs or 128
    sizes = {
        n * q
        for n in range(1, request_cap + 1)
        for q in ((1,) if arm.drafter == "none" else (1, 5, 6))
    }
    sizes.update((256, 512, 1024, 2048, 4096, 8192, 16384))
    prefix = "policy.generation.vllm_kwargs."
    values.update(
        {
            prefix + "compilation_config.mode": "3",
            prefix + "compilation_config.backend": "inductor",
            prefix + "compilation_config.cudagraph_mode": "FULL_AND_PIECEWISE",
            prefix + "compilation_config.cudagraph_capture_sizes": json.dumps(
                sorted(sizes), separators=(",", ":")
            ),
            prefix + "compilation_config.max_cudagraph_capture_size": "16384",
            prefix + "max_num_batched_tokens": "16384",
            prefix + "cudagraph_metrics": "true",
            "policy.generation.vllm_cfg.env_vars.NRL_Q8_CG_AUDIT": "1",
            "logger.wandb.group": "q8-gbs512-32k-fap-coverage-20260916",
        }
    )
    if packed:
        values.update(
            {
                "policy.sequence_packing.enabled": "true",
                "policy.sequence_packing.train_mb_tokens": "32768",
                "policy.sequence_packing.logprob_mb_tokens": "32768",
                "logger.wandb.group": "q8-gbs512-32k-packed-fap-20260917",
            }
        )
    if seqs is not None:
        values[prefix + "max_num_seqs"] = str(seqs)
    label = "default" if seqs is None else str(seqs)
    suffix = f"-Packed-FAP-S{label}" if packed else f"-FAP-S{label}"
    values["logger.wandb.name"] = values["logger.wandb.name"].replace(
        "-gate", suffix
    )
    return tuple(f"++{key}={value}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--seqs", choices=("default", "8", "32", "64"), required=True)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--recipe", action="store_true")
    args = parser.parse_args()
    arm = next(a for a in build_new_arms() if a.name == args.arm)
    seqs = None if args.seqs == "default" else int(args.seqs)
    values = graph_overrides(arm, args.result_dir, seqs, packed=args.packed)
    print(arm.config_path if args.recipe else "\n".join(values))


if __name__ == "__main__":
    main()
