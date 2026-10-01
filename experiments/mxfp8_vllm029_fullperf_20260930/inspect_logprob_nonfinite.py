#!/usr/bin/env python3
"""Summarize non-finite rollout and policy logprobs in GRPO JSONL dumps."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Counts:
    samples: int = 0
    tokens: int = 0
    valid_tokens: int = 0
    generation_nonfinite: int = 0
    generation_nonfinite_valid: int = 0
    policy_nonfinite: int = 0
    policy_nonfinite_valid: int = 0
    log_ratio_nonfinite: int = 0
    log_ratio_nonfinite_valid: int = 0
    k3_nonfinite: int = 0
    k3_nonfinite_valid: int = 0
    poisoned_samples: int = 0
    poisoned_valid_samples: int = 0


def _flatten_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, list):
        for item in value:
            yield from _flatten_numbers(item)
        return
    yield float(value)


def _k3(log_ratio: float) -> float:
    try:
        return math.exp(log_ratio) - 1.0 - log_ratio
    except OverflowError:
        return math.inf


def inspect(path: Path) -> Counts:
    counts = Counts()
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            generation = list(_flatten_numbers(record["generation_logprobs"]))[1:]
            policy = list(_flatten_numbers(record["prev_logprobs"]))[1:]
            token_mask = list(_flatten_numbers(record["token_loss_mask"]))[1:]
            sample_mask = list(_flatten_numbers(record["sample_loss_mask"]))
            if not (len(generation) == len(policy) == len(token_mask)):
                raise ValueError(
                    f"{path}:{line_number}: mismatched token arrays: "
                    f"generation={len(generation)}, policy={len(policy)}, "
                    f"mask={len(token_mask)}"
                )

            sample_enabled = bool(sample_mask and sample_mask[0])
            sample_poisoned = False
            valid_sample_poisoned = False
            for generation_lp, policy_lp, token_enabled in zip(
                generation, policy, token_mask, strict=True
            ):
                valid = sample_enabled and bool(token_enabled)
                generation_bad = not math.isfinite(generation_lp)
                policy_bad = not math.isfinite(policy_lp)
                log_ratio = policy_lp - generation_lp
                ratio_bad = not math.isfinite(log_ratio)
                k3_bad = not math.isfinite(_k3(log_ratio))

                counts.tokens += 1
                counts.valid_tokens += int(valid)
                counts.generation_nonfinite += int(generation_bad)
                counts.generation_nonfinite_valid += int(valid and generation_bad)
                counts.policy_nonfinite += int(policy_bad)
                counts.policy_nonfinite_valid += int(valid and policy_bad)
                counts.log_ratio_nonfinite += int(ratio_bad)
                counts.log_ratio_nonfinite_valid += int(valid and ratio_bad)
                counts.k3_nonfinite += int(k3_bad)
                counts.k3_nonfinite_valid += int(valid and k3_bad)
                sample_poisoned |= k3_bad
                valid_sample_poisoned |= valid and k3_bad

            counts.samples += 1
            counts.poisoned_samples += int(sample_poisoned)
            counts.poisoned_valid_samples += int(valid_sample_poisoned)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {str(path): asdict(inspect(path)) for path in args.paths}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
