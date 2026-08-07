"""Fail-closed collection for Qwen3-30B MXFP8 MoE tactic-audit evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import fmean, pstdev


FIRST_MEASURED_STEP = 3
LAST_MEASURED_STEP = 8
REQUIRED_PHASES = ("refit", "rollout", "logprob", "train")
# The launcher intentionally encodes the arm in run_kind; it is only a cache-arm
# identity and is not an execution-input difference.
CACHE_IDENTITY_FIELDS = frozenset({"cache_sha256", "run_kind"})

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_PATTERNS = {
    "total_step_seconds": re.compile(r"Total step time:\s*([0-9.eE+-]+)s"),
    "generation_seconds": re.compile(r"generation:\s*([0-9.eE+-]+)s"),
    "e2e_tokens_per_second_per_gpu": re.compile(
        r"E2E \(Tokens/sec/gpu\):\s*([0-9.eE+-]+)"
    ),
    "generated_tokens_per_second_per_gpu": re.compile(
        r"Generation Worker Group \(Tokens/sec/gpu\):\s*([0-9.eE+-]+)"
    ),
    "realized_generated_tokens": re.compile(
        r"(?:Realized )?(?:generated|generation) tokens:\s*([0-9]+)", re.IGNORECASE
    ),
    "reward": re.compile(r"Reward:\s*([0-9.eE+-]+)"),
    "kl": re.compile(r"KL:\s*([0-9.eE+-]+)"),
    "loss": re.compile(r"Loss:\s*([0-9.eE+-]+)"),
}


class EvidenceError(ValueError):
    """Raised when an input does not provide complete audit evidence."""


@dataclass(frozen=True)
class StepMetrics:
    """One complete NeMo-RL ``Training Results`` block."""

    step: int
    total_step_seconds: float
    generation_seconds: float
    e2e_tokens_per_second_per_gpu: float
    generated_tokens_per_second_per_gpu: float
    realized_generated_tokens: int
    reward: float
    kl: float
    loss: float


@dataclass(frozen=True)
class RunSummary:
    """Steady-state evidence for exactly the six measured validation steps."""

    steps: tuple[StepMetrics, ...]
    measured_steps: int
    generated_tokens_per_second_per_gpu: float
    total_step_seconds: float
    realized_generated_tokens: int
    all_metrics_finite: bool
    variation: float


def sha256_file(path: Path) -> str:
    """Return a content hash for one source artifact."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    """Load a JSON object, rejecting missing files and non-object payloads."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvidenceError(f"JSON evidence {path} must be an object")
    return value


def load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    """Load nonempty JSONL evidence rows."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvidenceError(f"cannot read JSONL evidence {path}: {error}") from error
    if not lines:
        raise EvidenceError(f"JSONL evidence {path} is empty")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"invalid JSONL evidence {path}:{line_number}") from error
        if not isinstance(row, dict) or not all(isinstance(key, str) for key in row):
            raise EvidenceError(f"JSONL evidence {path}:{line_number} must be an object")
        rows.append(row)
    return tuple(rows)


def _number(block: str, field_name: str, *, integer: bool = False) -> float | int:
    match = METRIC_PATTERNS[field_name].search(block)
    if match is None:
        raise EvidenceError(f"missing {field_name} in Training Results block")
    value = int(match.group(1)) if integer else float(match.group(1))
    if not math.isfinite(value):
        raise EvidenceError(f"nonfinite {field_name} in Training Results block")
    return value


def parse_training_results(log_text: str, *, source: str) -> tuple[StepMetrics, ...]:
    """Parse complete NeMo-RL Training Results blocks without skipping failures."""
    clean_text = ANSI_ESCAPE.sub("", log_text)
    blocks = clean_text.split("Training Results:")[1:]
    if len(blocks) < LAST_MEASURED_STEP:
        raise EvidenceError(f"{source} has fewer than {LAST_MEASURED_STEP} Training Results blocks")
    parsed: list[StepMetrics] = []
    for step, block in enumerate(blocks, start=1):
        if step < FIRST_MEASURED_STEP or step > LAST_MEASURED_STEP:
            continue
        total_step_seconds = float(_number(block, "total_step_seconds"))
        generation_seconds = float(_number(block, "generation_seconds"))
        e2e = float(_number(block, "e2e_tokens_per_second_per_gpu"))
        generated = float(_number(block, "generated_tokens_per_second_per_gpu"))
        tokens = int(_number(block, "realized_generated_tokens", integer=True))
        reward = float(_number(block, "reward"))
        kl = float(_number(block, "kl"))
        loss = float(_number(block, "loss"))
        if min(total_step_seconds, generation_seconds, e2e, generated) <= 0 or tokens <= 0:
            raise EvidenceError(f"nonpositive timing, throughput, or token count at step {step}")
        parsed.append(
            StepMetrics(
                step=step,
                total_step_seconds=total_step_seconds,
                generation_seconds=generation_seconds,
                e2e_tokens_per_second_per_gpu=e2e,
                generated_tokens_per_second_per_gpu=generated,
                realized_generated_tokens=tokens,
                reward=reward,
                kl=kl,
                loss=loss,
            )
        )
    return tuple(parsed)


def find_driver_log(run_root: Path) -> Path:
    logs = sorted(path for path in run_root.rglob("ray-driver.log") if path.is_file())
    if len(logs) != 1:
        raise EvidenceError(f"expected exactly one ray-driver.log under {run_root}, found {len(logs)}")
    return logs[0]


def _require_successful_phases(run_root: Path) -> None:
    phases = load_json_object(run_root / "phase_status.json")
    for phase in REQUIRED_PHASES:
        if phases.get(phase) != "success":
            raise EvidenceError(f"{run_root} phase {phase} is not successful")


def summarize_run(
    run_root: Path,
    *,
    first_step: int = FIRST_MEASURED_STEP,
    last_step: int = LAST_MEASURED_STEP,
) -> RunSummary:
    """Require complete six-step execution evidence and return steady-state means."""
    if (first_step, last_step) != (FIRST_MEASURED_STEP, LAST_MEASURED_STEP):
        raise EvidenceError("the audit requires exactly measured steps 3-8")
    _require_successful_phases(run_root)
    log = find_driver_log(run_root)
    steps = parse_training_results(log.read_text(errors="replace"), source=str(log))
    expected_steps = tuple(range(FIRST_MEASURED_STEP, LAST_MEASURED_STEP + 1))
    if tuple(step.step for step in steps) != expected_steps:
        raise EvidenceError("measured steps must be exactly 3-8")
    throughputs = [step.generated_tokens_per_second_per_gpu for step in steps]
    mean_throughput = fmean(throughputs)
    variation = pstdev(throughputs) / mean_throughput if mean_throughput else math.inf
    return RunSummary(
        steps=steps,
        measured_steps=len(steps),
        generated_tokens_per_second_per_gpu=mean_throughput,
        total_step_seconds=fmean(step.total_step_seconds for step in steps),
        realized_generated_tokens=sum(step.realized_generated_tokens for step in steps),
        all_metrics_finite=all(
            math.isfinite(value)
            for step in steps
            for value in (
                step.total_step_seconds,
                step.generation_seconds,
                step.e2e_tokens_per_second_per_gpu,
                step.generated_tokens_per_second_per_gpu,
                step.reward,
                step.kl,
                step.loss,
            )
        ),
        variation=variation,
    )


def compare_manifests(stock_path: Path, candidate_path: Path) -> tuple[str, ...]:
    """Return differing non-cache manifest fields, preserving exact provenance gates."""
    stock = load_json_object(stock_path)
    candidate = load_json_object(candidate_path)
    stock_invariant = {
        key: value for key, value in stock.items() if key not in CACHE_IDENTITY_FIELDS
    }
    candidate_invariant = {
        key: value
        for key, value in candidate.items()
        if key not in CACHE_IDENTITY_FIELDS
    }
    fields = sorted(set(stock_invariant) | set(candidate_invariant))
    return tuple(
        field
        for field in fields
        if stock_invariant.get(field) != candidate_invariant.get(field)
    )


def require_boolean_gates(payload: Mapping[str, object], keys: Sequence[str]) -> None:
    """Require every named correctness gate to be explicitly true."""
    missing_or_failed = [key for key in keys if payload.get(key) is not True]
    if missing_or_failed:
        raise EvidenceError("correctness gates failed or absent: " + ", ".join(missing_or_failed))
