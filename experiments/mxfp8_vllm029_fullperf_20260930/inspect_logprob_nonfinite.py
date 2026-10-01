#!/usr/bin/env python3
"""Summarize non-finite rollout and policy logprobs in GRPO JSONL dumps."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Counts:
    samples: int = 0
    tokens: int = 0
    valid_tokens: int = 0
    generation_nonfinite: int = 0
    generation_nonfinite_valid: int = 0
    generation_nan: int = 0
    generation_posinf: int = 0
    generation_neginf: int = 0
    policy_nonfinite: int = 0
    policy_nonfinite_valid: int = 0
    policy_nan: int = 0
    policy_posinf: int = 0
    policy_neginf: int = 0
    log_ratio_nonfinite: int = 0
    log_ratio_nonfinite_valid: int = 0
    k3_nonfinite: int = 0
    k3_nonfinite_valid: int = 0
    poisoned_samples: int = 0
    poisoned_valid_samples: int = 0
    input_length_max: int = 0
    input_length_max_samples: int = 0
    generation_nonfinite_at_max_length_samples: int = 0
    generation_nonfinite_examples: list[dict[str, Any]] = field(default_factory=list)


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
            input_lengths = list(_flatten_numbers(record.get("input_lengths", [])))
            input_length = int(input_lengths[0]) if input_lengths else 0
            if input_length > counts.input_length_max:
                counts.input_length_max = input_length
                counts.input_length_max_samples = 1
                counts.generation_nonfinite_at_max_length_samples = 0
            elif input_length == counts.input_length_max:
                counts.input_length_max_samples += 1
            if not (len(generation) == len(policy) == len(token_mask)):
                raise ValueError(
                    f"{path}:{line_number}: mismatched token arrays: "
                    f"generation={len(generation)}, policy={len(policy)}, "
                    f"mask={len(token_mask)}"
                )

            sample_enabled = bool(sample_mask and sample_mask[0])
            sample_poisoned = False
            valid_sample_poisoned = False
            generation_bad_positions: list[int] = []
            for token_index, (generation_lp, policy_lp, token_enabled) in enumerate(
                zip(generation, policy, token_mask, strict=True), start=1
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
                counts.generation_nan += int(math.isnan(generation_lp))
                counts.generation_posinf += int(generation_lp == math.inf)
                counts.generation_neginf += int(generation_lp == -math.inf)
                counts.policy_nonfinite += int(policy_bad)
                counts.policy_nonfinite_valid += int(valid and policy_bad)
                counts.policy_nan += int(math.isnan(policy_lp))
                counts.policy_posinf += int(policy_lp == math.inf)
                counts.policy_neginf += int(policy_lp == -math.inf)
                counts.log_ratio_nonfinite += int(ratio_bad)
                counts.log_ratio_nonfinite_valid += int(valid and ratio_bad)
                counts.k3_nonfinite += int(k3_bad)
                counts.k3_nonfinite_valid += int(valid and k3_bad)
                sample_poisoned |= k3_bad
                valid_sample_poisoned |= valid and k3_bad
                if valid and generation_bad:
                    generation_bad_positions.append(token_index)

            counts.samples += 1
            counts.poisoned_samples += int(sample_poisoned)
            counts.poisoned_valid_samples += int(valid_sample_poisoned)
            if generation_bad_positions and input_length == counts.input_length_max:
                counts.generation_nonfinite_at_max_length_samples += 1
            if generation_bad_positions and len(counts.generation_nonfinite_examples) < 16:
                rewards = list(_flatten_numbers(record.get("rewards", [])))
                token_ids = [
                    int(token_id)
                    for token_id in _flatten_numbers(record.get("token_ids", []))
                ]
                bad_token_ids = [
                    token_ids[position]
                    for position in generation_bad_positions
                    if position < len(token_ids)
                ]
                most_common = Counter(bad_token_ids).most_common(3)
                counts.generation_nonfinite_examples.append(
                    {
                        "idx": record.get("idx", line_number - 1),
                        "reward": rewards[0] if rewards else None,
                        "input_length": input_length or None,
                        "bad_valid_tokens": len(generation_bad_positions),
                        "first_bad_token": generation_bad_positions[0],
                        "last_bad_token": generation_bad_positions[-1],
                        "unique_bad_token_ids": len(set(bad_token_ids)),
                        "most_common_bad_token_ids": most_common,
                        "first_bad_token_ids": bad_token_ids[:16],
                        "last_bad_token_ids": bad_token_ids[-16:],
                    }
                )
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
