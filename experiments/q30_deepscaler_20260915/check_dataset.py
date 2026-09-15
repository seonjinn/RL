"""Compute-node integration gate for the built-in DeepScaler loader and grader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.data.datasets.response_datasets.deepscaler import DeepScalerDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset = DeepScalerDataset().dataset
    if len(dataset) < 2048:
        raise RuntimeError(f"Unexpectedly small DeepScaler train split: {len(dataset)}")
    verify = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    cases = [(r"\frac{1}{2}", "0.5", 1.0), ("17", "17", 1.0), ("17", "18", 0.0)]
    scores = []
    for gold, prediction, expected in cases:
        score, _ = verify([f"\\boxed{{{gold}}}"], [f"\\boxed{{{prediction}}}"])
        if float(score) != expected:
            raise RuntimeError(f"Grader smoke failed: {score} != {expected}")
        scores.append(float(score))
    sample_scores = []
    for row in dataset.select(range(16)):
        messages = row["messages"]
        if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            raise RuntimeError("Unexpected DeepScaler message schema")
        if any(
            not isinstance(m["content"], str) or not m["content"].strip()
            for m in messages
        ):
            raise RuntimeError("Empty or non-text problem/answer")
        boxed = "\\boxed{" + messages[1]["content"] + "}"
        score, _ = verify([boxed], [boxed])
        sample_scores.append(float(score))
    if any(score != 1.0 for score in sample_scores):
        raise RuntimeError(
            f"Sample reference answers cannot be graded: {sample_scores}"
        )
    result = {
        "status": "passed",
        "dataset": "agentica-org/DeepScaleR-Preview-Dataset",
        "split": "train",
        "rows": len(dataset),
        "fingerprint": dataset._fingerprint,
        "verifier": "hf_math_verify",
        "synthetic_scores": scores,
        "reference_self_check_scores": sample_scores,
        "note": "Loader/grader gate only; no evidence of model output quality or long reasoning.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("DEEPSCALER_DATASET_GATE_PASSED " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
