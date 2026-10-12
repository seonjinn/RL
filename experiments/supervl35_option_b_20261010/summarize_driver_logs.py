"""Compare completed SingleController steps from SuperVL driver logs."""

import argparse
import re
from pathlib import Path
from statistics import mean

DEFAULT_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "nemo-rl/supervl35-optionb-20261010"
)
METRICS = (
    "total_step_time",
    "policy_training",
    "policy_and_reference_logprobs",
    "weight_sync",
    "exposed_generation",
    "valid_tokens_per_sec_per_gpu",
)
LABELS = (
    "E2E (s)",
    "Policy (s)",
    "Logprob (s)",
    "Weight sync (s)",
    "Gen wait (s)",
    "Valid tok/s/GPU",
)
TOTAL_RE = re.compile(r"Total step time:\s*([0-9]+(?:\.[0-9]+)?)s")
METRIC_RE = re.compile(r"\b([a-z][a-z0-9_/]+):\s*([0-9]+(?:\.[0-9]+)?)s\b")
STEP_RE = re.compile(r"\btrain step\s+(\d+)/(\d+)\b")


def parse_steps(path: Path) -> dict[int, dict[str, float]]:
    """Bind each printed timing block to its following completed step marker."""
    steps: dict[int, dict[str, float]] = {}
    pending: dict[str, float] | None = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            for part in line.split("\r"):
                if part.startswith("(") and "SingleControllerActor" not in part:
                    continue
                total = TOTAL_RE.search(part)
                if total:
                    pending = {"total_step_time": float(total.group(1))}
                    continue
                if pending is None:
                    continue
                metric = METRIC_RE.search(part)
                if metric and metric.group(1) in METRICS:
                    pending[metric.group(1)] = float(metric.group(2))
                step = STEP_RE.search(part)
                if step:
                    number = int(step.group(1))
                    if number in steps:
                        raise ValueError(f"Duplicate completed step {number} in {path}")
                    steps[number] = pending
                    pending = None
    return steps


def summarize(
    steps: dict[int, dict[str, float]], selected: list[int]
) -> dict[str, float | None]:
    """Do not report a partial metric as if it covered the full step window."""
    return {
        key: mean(steps[step][key] for step in selected)
        if all(key in steps[step] for step in selected)
        else None
        for key in METRICS
    }


def format_metric(value: float | None, baseline: float | None, throughput: bool) -> str:
    if value is None:
        return "n/a"
    if baseline is None or baseline <= 0 or value <= 0:
        return f"{value:.2f}"
    ratio = value / baseline if throughput else baseline / value
    return f"{value:.2f} ({ratio:.2f}x)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="ARM:JOB_ID",
        help="Repeat for each precision arm; include bf16-bf16 as the baseline.",
    )
    parser.add_argument("--first-step", type=int, default=2)
    parser.add_argument("--last-step", type=int, default=20)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use only the same completed steps present in every supplied run.",
    )
    args = parser.parse_args()
    if args.first_step < 1 or args.last_step < args.first_step:
        parser.error("Expected 1 <= first-step <= last-step")

    parsed: list[tuple[str, str, dict[int, dict[str, float]]]] = []
    for run in args.run:
        arm, separator, job_id = run.partition(":")
        if not separator or not arm or not job_id.isdigit():
            parser.error(f"Invalid --run {run!r}; expected ARM:JOB_ID")
        path = args.root / "runs" / job_id / arm / "driver.log"
        if not path.is_file():
            parser.error(f"Missing {path}; capture the running driver's log first")
        parsed.append((arm, job_id, parse_steps(path)))

    requested = set(range(args.first_step, args.last_step + 1))
    available = requested.intersection(*(set(steps) for _, _, steps in parsed))
    if not args.allow_partial and available != requested:
        missing = sorted(requested - available)
        parser.error(f"Incomplete common step window; missing {missing}")
    if not available:
        parser.error("No completed steps in the common step window")
    selected = sorted(available)
    baseline_rows = [steps for arm, _, steps in parsed if arm == "bf16-bf16"]
    if len(baseline_rows) != 1:
        parser.error("Supply exactly one bf16-bf16 baseline")
    baseline = summarize(baseline_rows[0], selected)

    print(
        f"Completed common steps: {selected[0]}-{selected[-1]} ({len(selected)} steps)"
    )
    print("| Arm | Job ID | " + " | ".join(LABELS) + " |")
    print("|---|---:|" + "|".join("---:" for _ in METRICS) + "|")
    for arm, job_id, steps in parsed:
        values = summarize(steps, selected)
        cells = [
            format_metric(
                values[key],
                baseline[key] if arm != "bf16-bf16" else None,
                key == "valid_tokens_per_sec_per_gpu",
            )
            for key in METRICS
        ]
        print(f"| {arm} | {job_id} | " + " | ".join(cells) + " |")
    print(
        "\nRatios are speedups versus BF16/BF16. Gen wait is exposed generation "
        "stall, not backend generation time; weight sync includes more than "
        "tensor transfer. For actual generation throughput, use TensorBoard/W&B."
    )


if __name__ == "__main__":
    main()
