"""Opt-in FAP/Inductor diagnostics; never alter the completed cadence cohort."""

from __future__ import annotations

import argparse
import json

from research.qwen3_8b_rp25_swa.study import build_new_arms, overrides
from research.qwen3_8b_draft_cadence_200step.matrix import Arm


_GRAPH_PREFIX = "policy.generation.vllm_kwargs."


def _apply_graph_settings(
    values: dict[str, str],
    arm: Arm,
    seqs: int | None,
) -> None:
    request_cap = seqs or 128
    sizes = {
        n * q
        for n in range(1, request_cap + 1)
        for q in ((1,) if arm.drafter == "none" else (1, 5, 6))
    }
    sizes.update((256, 512, 1024, 2048, 4096, 8192, 16384))
    values.update(
        {
            _GRAPH_PREFIX + "compilation_config.mode": "3",
            _GRAPH_PREFIX + "compilation_config.backend": "inductor",
            _GRAPH_PREFIX + "compilation_config.cudagraph_mode": "FULL_AND_PIECEWISE",
            _GRAPH_PREFIX + "compilation_config.cudagraph_capture_sizes": json.dumps(
                sorted(sizes), separators=(",", ":")
            ),
            _GRAPH_PREFIX + "compilation_config.max_cudagraph_capture_size": "16384",
            _GRAPH_PREFIX + "max_num_batched_tokens": "16384",
            _GRAPH_PREFIX + "cudagraph_metrics": "true",
            "policy.generation.vllm_cfg.enforce_eager": "false",
            "policy.generation.vllm_cfg.env_vars.NRL_Q8_CG_AUDIT": "1",
        }
    )
    if seqs is None:
        values.pop(_GRAPH_PREFIX + "max_num_seqs", None)
    else:
        values[_GRAPH_PREFIX + "max_num_seqs"] = str(seqs)


def graph_overrides(
    arm: Arm,
    result_dir: str,
    seqs: int | None,
    *,
    packed: bool = False,
) -> tuple[str, ...]:
    if seqs not in (None, 8, 32, 64, 128):
        raise ValueError("graph study supports S8, S32, S64, S128 or baseline default")
    if seqs is None and arm.drafter != "none":
        raise ValueError("default concurrency is reserved for the baseline control")
    values = dict(
        x[2:].split("=", 1) for x in overrides(arm, result_dir, long_context=True)
    )
    _apply_graph_settings(values, arm, seqs)
    values["logger.wandb.group"] = "q8-gbs512-32k-fap-coverage-20260916"
    if packed:
        values.update(
            {
                "policy.sequence_packing.enabled": "true",
                "policy.sequence_packing.train_mb_tokens": "32768",
                "policy.sequence_packing.logprob_mb_tokens": "32768",
                "logger.wandb.group": "q8-gbs512-32k-packed-fap-20260917",
            }
        )
    label = "default" if seqs is None else str(seqs)
    suffix = f"-Packed-FAP-S{label}" if packed else f"-FAP-S{label}"
    values["logger.wandb.name"] = values["logger.wandb.name"].replace("-gate", suffix)
    return tuple(f"++{key}={value}" for key, value in values.items())


def online_graph_overrides(
    arm: Arm,
    result_dir: str,
    seqs: int | None,
) -> tuple[str, ...]:
    if arm.cadence not in ("baseline", "static", "always", "fixed-10"):
        raise ValueError("online graph study supports baseline/frozen/fixed-10/always")
    if seqs not in (None, 64, 128):
        raise ValueError("online graph study supports S64, S128 or baseline default")
    if seqs is None and arm.drafter != "none":
        raise ValueError("default concurrency is reserved for the baseline control")
    values = dict(
        item[2:].split("=", 1)
        for item in overrides(arm, result_dir, production_steps=20)
    )
    values.update(
        {
            "grpo.num_prompts_per_step": "64",
            "grpo.num_generations_per_prompt": "8",
            "policy.train_global_batch_size": "512",
            "policy.max_total_sequence_length": "32768",
            "policy.generation.max_new_tokens": "30720",
            "policy.generation.vllm_cfg.max_model_len": "32768",
            "policy.megatron_cfg.activation_checkpointing": "true",
            "policy.logprob_chunk_size": "256",
            "policy.megatron_cfg.defer_fp32_logits": "true",
            "policy.sequence_packing.enabled": "true",
            "policy.sequence_packing.train_mb_tokens": "32768",
            "policy.sequence_packing.logprob_mb_tokens": "32768",
            "logger.wandb.group": "q8-gbs512-32k-packed-online-20step-20260919",
        }
    )
    _apply_graph_settings(values, arm, seqs)
    label = "default" if seqs is None else str(seqs)
    values["logger.wandb.name"] = (
        f"Qwen3-8B-{arm.name}-GBS512-32K-Packed-FAP-S{label}-20step"
    )
    return tuple(f"++{key}={value}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument(
        "--seqs", choices=("default", "8", "32", "64", "128"), required=True
    )
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--recipe", action="store_true")
    args = parser.parse_args()
    arm = next(a for a in build_new_arms() if a.name == args.arm)
    seqs = None if args.seqs == "default" else int(args.seqs)
    values = (
        online_graph_overrides(arm, args.result_dir, seqs)
        if args.online
        else graph_overrides(arm, args.result_dir, seqs, packed=args.packed)
    )
    print(arm.config_path if args.recipe else "\n".join(values))


if __name__ == "__main__":
    main()
