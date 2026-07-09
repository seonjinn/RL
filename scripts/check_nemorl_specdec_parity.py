#!/usr/bin/env python3
"""Check output-distribution and logprob parity for baseline versus SpecDec runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


@dataclass(frozen=True)
class Sample:
    prompt_id: str
    sample_id: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    reward: float | None

    @property
    def key(self) -> tuple[str, str]:
        return self.prompt_id, self.sample_id


def _read_json_records(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list")
        yield from payload
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[tuple[str, str]] = set()
    for row_number, row in enumerate(_read_json_records(path), start=1):
        token_ids = tuple(int(token) for token in row["token_ids"])
        token_logprobs = tuple(float(value) for value in row["token_logprobs"])
        if not token_ids:
            raise ValueError(f"{path}: row {row_number} has no generated tokens")
        if len(token_ids) != len(token_logprobs):
            raise ValueError(f"{path}: row {row_number} token/logprob length mismatch")
        if not all(math.isfinite(value) for value in token_logprobs):
            raise ValueError(f"{path}: row {row_number} has non-finite logprobs")

        reward_value = row.get("reward")
        reward = None if reward_value is None else float(reward_value)
        if reward is not None and not math.isfinite(reward):
            raise ValueError(f"{path}: row {row_number} has a non-finite reward")

        sample = Sample(
            prompt_id=str(row["prompt_id"]),
            sample_id=str(row["sample_id"]),
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            reward=reward,
        )
        if sample.key in seen:
            raise ValueError(f"{path}: duplicate sample key {sample.key}")
        seen.add(sample.key)
        samples.append(sample)

    if not samples:
        raise ValueError(f"{path} contains no samples")
    return samples


def _mean_logprob(samples: Iterable[Sample]) -> float:
    values = [value for sample in samples for value in sample.token_logprobs]
    return fmean(values)


def _mean_reward(samples: Iterable[Sample]) -> float | None:
    values = [sample.reward for sample in samples if sample.reward is not None]
    return fmean(values) if values else None


def _total_variation(left: Counter[int], right: Counter[int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    tokens = left.keys() | right.keys()
    return 0.5 * sum(
        abs(left[token] / left_total - right[token] / right_total) for token in tokens
    )


def compare_samples(
    baseline: list[Sample],
    specdec: list[Sample],
    *,
    mode: str,
    max_token_logprob_delta: float,
    max_mean_logprob_delta: float,
    max_first_token_tv: float,
    max_reward_delta: float,
) -> dict[str, Any]:
    baseline_by_key = {sample.key: sample for sample in baseline}
    specdec_by_key = {sample.key: sample for sample in specdec}
    if baseline_by_key.keys() != specdec_by_key.keys():
        missing = sorted(baseline_by_key.keys() - specdec_by_key.keys())[:5]
        extra = sorted(specdec_by_key.keys() - baseline_by_key.keys())[:5]
        raise ValueError(f"sample keys differ: missing={missing}, extra={extra}")

    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {
        "mode": mode,
        "samples": len(baseline),
        "prompts": len({sample.prompt_id for sample in baseline}),
    }

    if mode == "greedy":
        token_mismatches = 0
        max_logprob_delta = 0.0
        for key, baseline_sample in baseline_by_key.items():
            specdec_sample = specdec_by_key[key]
            if baseline_sample.token_ids != specdec_sample.token_ids:
                token_mismatches += 1
                continue
            max_logprob_delta = max(
                max_logprob_delta,
                max(
                    abs(left - right)
                    for left, right in zip(
                        baseline_sample.token_logprobs,
                        specdec_sample.token_logprobs,
                        strict=True,
                    )
                ),
            )
        metrics["token_mismatches"] = token_mismatches
        metrics["max_token_logprob_delta"] = max_logprob_delta
        checks["exact_tokens"] = token_mismatches == 0
        checks["token_logprobs"] = (
            token_mismatches == 0
            and max_logprob_delta <= max_token_logprob_delta
        )
    elif mode == "sampled":
        baseline_first_tokens: dict[str, Counter[int]] = defaultdict(Counter)
        specdec_first_tokens: dict[str, Counter[int]] = defaultdict(Counter)
        for sample in baseline:
            baseline_first_tokens[sample.prompt_id][sample.token_ids[0]] += 1
        for sample in specdec:
            specdec_first_tokens[sample.prompt_id][sample.token_ids[0]] += 1
        if baseline_first_tokens.keys() != specdec_first_tokens.keys():
            raise ValueError("baseline and SpecDec prompt sets differ")

        tv_by_prompt = {
            prompt_id: _total_variation(
                baseline_first_tokens[prompt_id], specdec_first_tokens[prompt_id]
            )
            for prompt_id in baseline_first_tokens
        }
        metrics["mean_first_token_tv"] = fmean(tv_by_prompt.values())
        metrics["max_first_token_tv"] = max(tv_by_prompt.values())
        checks["first_token_distribution"] = (
            metrics["max_first_token_tv"] <= max_first_token_tv
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")

    baseline_mean_logprob = _mean_logprob(baseline)
    specdec_mean_logprob = _mean_logprob(specdec)
    mean_logprob_delta = abs(baseline_mean_logprob - specdec_mean_logprob)
    metrics.update(
        {
            "baseline_mean_logprob": baseline_mean_logprob,
            "specdec_mean_logprob": specdec_mean_logprob,
            "mean_logprob_delta": mean_logprob_delta,
        }
    )
    checks["mean_logprob"] = mean_logprob_delta <= max_mean_logprob_delta

    baseline_reward = _mean_reward(baseline)
    specdec_reward = _mean_reward(specdec)
    if baseline_reward is not None or specdec_reward is not None:
        if baseline_reward is None or specdec_reward is None:
            raise ValueError("reward must be present in both baseline and SpecDec data")
        reward_delta = abs(baseline_reward - specdec_reward)
        metrics.update(
            {
                "baseline_mean_reward": baseline_reward,
                "specdec_mean_reward": specdec_reward,
                "mean_reward_delta": reward_delta,
            }
        )
        checks["mean_reward"] = reward_delta <= max_reward_delta

    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--specdec", type=Path, required=True)
    parser.add_argument("--mode", choices=("greedy", "sampled"), required=True)
    parser.add_argument("--max-token-logprob-delta", type=float, default=1e-4)
    parser.add_argument("--max-mean-logprob-delta", type=float, default=0.05)
    parser.add_argument("--max-first-token-tv", type=float, default=0.1)
    parser.add_argument("--max-reward-delta", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_samples(
        load_samples(args.baseline),
        load_samples(args.specdec),
        mode=args.mode,
        max_token_logprob_delta=args.max_token_logprob_delta,
        max_mean_logprob_delta=args.max_mean_logprob_delta,
        max_first_token_tv=args.max_first_token_tv,
        max_reward_delta=args.max_reward_delta,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
