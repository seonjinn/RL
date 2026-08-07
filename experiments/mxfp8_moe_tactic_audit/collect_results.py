"""Strict collection of producer-shaped Qwen3-30B audit evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import argparse
from statistics import fmean


FIRST_MEASURED_STEP = 3
LAST_MEASURED_STEP = 8
REQUIRED_PHASES = ("refit", "rollout", "logprob", "train")
COMPARABILITY_METADATA_FIELDS = (
    "batch",
    "topology",
    "run_kind",
    "generation_settings",
)
CACHE_IDENTITY_FIELDS = frozenset({"cache_sha256"})
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PATTERNS = {
    "loss": re.compile(r"(?:Policy )?Loss:\s*([0-9.eE+-]+)"),
    "kl": re.compile(r"Generation KL Error:\s*([0-9.eE+-]+)"),
    "reward": re.compile(r"Avg Reward:\s*([0-9.eE+-]+)"),
    "mean_generation_length": re.compile(r"Mean Generation Length:\s*([0-9.eE+-]+)"),
    "total_step_seconds": re.compile(r"Total step time:\s*([0-9.eE+-]+)s"),
    "generation_seconds": re.compile(r"generation:\s*([0-9.eE+-]+)s"),
    "e2e_tokens_per_second_per_gpu": re.compile(
        r"E2E \(Tokens/sec/gpu\):\s*([0-9.eE+-]+)"
    ),
    "generated_tokens_per_second_per_gpu": re.compile(
        r"Generation Worker Group \(Tokens/sec/gpu\):\s*([0-9.eE+-]+)"
    ),
}


class EvidenceError(ValueError):
    """Raised when audit evidence is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class StepMetrics:
    """One measured GRPO Training Results block plus exact token evidence."""

    step: int
    loss: float
    kl: float
    reward: float
    mean_generation_length: float
    total_step_seconds: float
    generation_seconds: float
    e2e_tokens_per_second_per_gpu: float
    generated_tokens_per_second_per_gpu: float
    realized_generated_tokens: int


@dataclass(frozen=True)
class RunSummary:
    """One complete measured run, not an aggregation across repetitions."""

    run_id: str
    arm: str
    metadata: Mapping[str, str]
    runtime_fingerprints: Mapping[str, str]
    steps: tuple[StepMetrics, ...]
    measured_steps: int
    generated_tokens_per_second_per_gpu: float
    total_step_seconds: float
    realized_generated_tokens: int
    all_metrics_finite: bool


def sha256_file(path: Path) -> str:
    """Hash one supplied evidence artifact."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    """Load an object-shaped JSON artifact."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise EvidenceError(f"JSON evidence {path} must be an object")
    return payload


def load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    """Load nonempty object-shaped JSONL evidence."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvidenceError(f"cannot read JSONL evidence {path}: {error}") from error
    if not lines:
        raise EvidenceError(f"JSONL evidence {path} is empty")
    rows: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"invalid JSONL evidence {path}:{number}") from error
        if not isinstance(row, dict) or not all(isinstance(key, str) for key in row):
            raise EvidenceError(f"JSONL evidence {path}:{number} must be an object")
        rows.append(row)
    return tuple(rows)


def comparison_artifact_bindings(
    stock_artifact: Path, candidate_artifact: Path
) -> dict[str, object]:
    """Bind a paired producer result to exact run manifests and logical IDs."""
    return comparison_run_bindings((stock_artifact,), (candidate_artifact,))


def comparison_run_bindings(
    stock_artifacts: tuple[Path, ...], candidate_artifacts: tuple[Path, ...]
) -> dict[str, object]:
    """Bind one or more paired producer results to exact run manifests."""
    if not stock_artifacts or len(stock_artifacts) != len(candidate_artifacts):
        raise EvidenceError("stock/candidate binding run counts differ")
    hashes: dict[str, dict[str, str]] = {"stock": {}, "candidate": {}}
    for expected_arm, artifacts in (
        ("stock", stock_artifacts),
        ("candidate", candidate_artifacts),
    ):
        for artifact in artifacts:
            run_id, manifest_sha = _run_artifact_binding(artifact, expected_arm)
            if run_id in hashes[expected_arm]:
                raise EvidenceError(f"duplicate {expected_arm} comparison run ID")
            hashes[expected_arm][run_id] = manifest_sha
    if set(hashes["stock"]) != set(hashes["candidate"]):
        raise EvidenceError("stock/candidate logical comparison IDs differ")
    return {
        "candidate_arm_id": "candidate",
        "candidate_run_manifests": dict(sorted(hashes["candidate"].items())),
        "comparison_run_ids": sorted(hashes["stock"]),
        "stock_arm_id": "stock",
        "stock_run_manifests": dict(sorted(hashes["stock"].items())),
    }


def _run_artifact_binding(artifact: Path, expected_arm: str) -> tuple[str, str]:
    """Return one logical run ID and exact manifest hash."""
    run_root = artifact if artifact.is_dir() else artifact.parent
    while (
        run_root != run_root.parent and not (run_root / "run_manifest.json").is_file()
    ):
        run_root = run_root.parent
    manifest = run_root / "run_manifest.json"
    evidence = load_json_object(run_root / "run_evidence.json")
    arm = evidence.get("arm")
    metadata = evidence.get("metadata")
    if arm != expected_arm or not isinstance(metadata, dict):
        raise EvidenceError(f"{expected_arm} producer artifact has invalid arm")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise EvidenceError(f"{expected_arm} producer artifact has no run ID")
    return run_id, sha256_file(manifest)


def find_driver_log(run_root: Path) -> Path:
    """Find the sole driver log in either direct or launcher log layout."""
    logs = sorted(path for path in run_root.rglob("ray-driver.log") if path.is_file())
    if len(logs) != 1:
        raise EvidenceError(
            f"expected exactly one ray-driver.log under {run_root}, found {len(logs)}"
        )
    return logs[0]


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def _metric(block: str, name: str) -> float:
    match = PATTERNS[name].search(block)
    if match is None:
        raise EvidenceError(f"missing GRPO {name} in Training Results block")
    return _finite_number(float(match.group(1)), name)


def _token_counts(
    run_root: Path,
) -> tuple[dict[int, int], str, str, dict[str, str], dict[str, str]]:
    evidence = load_json_object(run_root / "run_evidence.json")
    if evidence.get("exit_code") != 0:
        raise EvidenceError(f"{run_root} run exit code is not zero")
    phases = evidence.get("phases")
    if not isinstance(phases, dict):
        raise EvidenceError(f"{run_root} has no phase evidence")
    failed = [phase for phase in REQUIRED_PHASES if phases.get(phase) != "success"]
    if failed:
        raise EvidenceError(f"{run_root} unsuccessful phases: {', '.join(failed)}")
    raw_steps = evidence.get("steps")
    if not isinstance(raw_steps, list):
        raise EvidenceError(f"{run_root} has no per-step token evidence")
    tokens: dict[int, int] = {}
    for row in raw_steps:
        if not isinstance(row, dict):
            raise EvidenceError(f"{run_root} has malformed per-step token evidence")
        step, count = row.get("step"), row.get("realized_generated_tokens")
        if isinstance(step, bool) or not isinstance(step, int):
            raise EvidenceError(f"{run_root} token evidence has invalid step")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise EvidenceError(
                f"{run_root} token evidence has invalid realized_generated_tokens"
            )
        if step in tokens:
            raise EvidenceError(f"{run_root} token evidence duplicates step {step}")
        tokens[step] = count
    metadata = evidence.get("metadata")
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in metadata.items()
    ):
        raise EvidenceError(f"{run_root} has invalid run metadata")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise EvidenceError(f"{run_root} metadata has no run_id")
    arm = evidence.get("arm")
    if arm not in {"stock", "candidate"}:
        raise EvidenceError(f"{run_root} evidence has invalid arm")
    if not isinstance(arm, str):  # narrows JSON object values for static analysis
        raise EvidenceError(f"{run_root} evidence has invalid arm")
    fingerprints = evidence.get("runtime_fingerprints")
    if (
        not isinstance(fingerprints, dict)
        or not fingerprints
        or not all(
            isinstance(key, str) and isinstance(value, str) and value
            for key, value in fingerprints.items()
        )
    ):
        raise EvidenceError(f"{run_root} has invalid runtime fingerprints")
    return (
        tokens,
        run_id,
        arm,
        {str(key): str(value) for key, value in metadata.items()},
        {str(key): str(value) for key, value in fingerprints.items()},
    )


_PHASE_MARKER = re.compile(
    r"\[MXFP8_MOE_AUDIT\]\s+step=(\d+)\s+phase=(refit|rollout|logprob|train)\s+status=success"
)


def _generated_token_count(path: Path) -> int:
    """Sum GRPO's producer-written token loss masks without estimating tokens."""
    count = 0
    for row in load_jsonl(path):
        mask = row.get("token_loss_mask")
        if not isinstance(mask, list):
            raise EvidenceError(f"{path} has no token_loss_mask")
        for value in mask:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvidenceError(f"{path} token_loss_mask must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0 or numeric != int(numeric):
                raise EvidenceError(
                    f"{path} token_loss_mask must contain nonnegative integers"
                )
            count += int(numeric)
    if count <= 0:
        raise EvidenceError(f"{path} realized generated tokens must be positive")
    return count


def write_run_evidence(
    run_root: Path,
    *,
    arm: str,
    run_id: str,
    metadata: Mapping[str, str],
    runtime_fingerprints: Mapping[str, str],
) -> Path:
    """Write launcher evidence from GRPO outputs that completed successfully."""
    if arm not in {"stock", "candidate"}:
        raise EvidenceError("run evidence arm must be stock or candidate")
    if not run_id or metadata.get("run_id") != run_id:
        raise EvidenceError("run evidence metadata must bind the run ID")
    missing_metadata = set(COMPARABILITY_METADATA_FIELDS) - set(metadata)
    if missing_metadata:
        raise EvidenceError(
            "run evidence metadata is missing comparability fields: "
            + ", ".join(sorted(missing_metadata))
        )
    if not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in runtime_fingerprints.items()
    ):
        raise EvidenceError("runtime fingerprints must be a nonempty string mapping")
    log = find_driver_log(run_root)
    markers: dict[int, set[str]] = {}
    for match in _PHASE_MARKER.finditer(log.read_text(errors="replace")):
        markers.setdefault(int(match.group(1)), set()).add(match.group(2))
    steps: list[dict[str, int]] = []
    for step in range(FIRST_MEASURED_STEP, LAST_MEASURED_STEP + 1):
        missing = set(REQUIRED_PHASES) - markers.get(step, set())
        if missing:
            raise EvidenceError(
                f"{log} missing successful phase markers for step {step}: {', '.join(sorted(missing))}"
            )
        dumps = sorted(run_root.rglob(f"train_data_step{step}.jsonl"))
        if len(dumps) != 1:
            raise EvidenceError(
                f"expected exactly one producer train_data_step{step}.jsonl under {run_root}, found {len(dumps)}"
            )
        steps.append(
            {
                "step": step,
                "realized_generated_tokens": _generated_token_count(dumps[0]),
            }
        )
    payload = {
        "arm": arm,
        "exit_code": 0,
        "metadata": dict(metadata),
        "phases": {phase: "success" for phase in REQUIRED_PHASES},
        "runtime_fingerprints": dict(runtime_fingerprints),
        "steps": steps,
    }
    output = run_root / "run_evidence.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return output


def parse_training_results(
    log_text: str, *, tokens_by_step: Mapping[int, int], source: str
) -> tuple[StepMetrics, ...]:
    """Parse exactly steps 3-8 from actual GRPO Training Results labels."""
    blocks = ANSI_ESCAPE.sub("", log_text).split("Training Results:")[1:]
    if len(blocks) < LAST_MEASURED_STEP:
        raise EvidenceError(f"{source} has fewer than eight Training Results blocks")
    parsed: list[StepMetrics] = []
    for step, block in enumerate(blocks, start=1):
        if step not in range(FIRST_MEASURED_STEP, LAST_MEASURED_STEP + 1):
            continue
        values = {name: _metric(block, name) for name in PATTERNS}
        token_count = tokens_by_step.get(step)
        if token_count is None:
            raise EvidenceError(f"{source} has no realized token count for step {step}")
        if (
            min(
                values["total_step_seconds"],
                values["generation_seconds"],
                values["e2e_tokens_per_second_per_gpu"],
                values["generated_tokens_per_second_per_gpu"],
                values["mean_generation_length"],
            )
            <= 0
        ):
            raise EvidenceError(f"{source} has nonpositive metric at step {step}")
        parsed.append(
            StepMetrics(step=step, realized_generated_tokens=token_count, **values)
        )
    if tuple(item.step for item in parsed) != tuple(range(3, 9)):
        raise EvidenceError("measured steps must be exactly 3-8")
    return tuple(parsed)


def summarize_run(run_root: Path) -> RunSummary:
    """Collect one run without treating within-run variation as run-to-run variance."""
    tokens, run_id, arm, metadata, runtime_fingerprints = _token_counts(run_root)
    log = find_driver_log(run_root)
    steps = parse_training_results(
        log.read_text(errors="replace"), tokens_by_step=tokens, source=str(log)
    )
    finite = all(
        math.isfinite(value)
        for step in steps
        for value in (
            step.loss,
            step.kl,
            step.reward,
            step.mean_generation_length,
            step.total_step_seconds,
            step.generation_seconds,
            step.e2e_tokens_per_second_per_gpu,
            step.generated_tokens_per_second_per_gpu,
        )
    )
    return RunSummary(
        run_id=run_id,
        arm=arm,
        metadata=metadata,
        runtime_fingerprints=runtime_fingerprints,
        steps=steps,
        measured_steps=len(steps),
        generated_tokens_per_second_per_gpu=fmean(
            step.generated_tokens_per_second_per_gpu for step in steps
        ),
        total_step_seconds=fmean(step.total_step_seconds for step in steps),
        realized_generated_tokens=sum(step.realized_generated_tokens for step in steps),
        all_metrics_finite=finite,
    )


def compare_manifests(stock_path: Path, candidate_path: Path) -> tuple[str, ...]:
    """Compare all non-cache execution provenance exactly."""
    stock = {
        key: value
        for key, value in load_json_object(stock_path).items()
        if key not in CACHE_IDENTITY_FIELDS
    }
    candidate = {
        key: value
        for key, value in load_json_object(candidate_path).items()
        if key not in CACHE_IDENTITY_FIELDS
    }
    return tuple(
        sorted(
            key
            for key in set(stock) | set(candidate)
            if stock.get(key) != candidate.get(key)
        )
    )


def main() -> None:
    """Write strict evidence from an already-completed GRPO producer run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-run-evidence", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--runtime-fingerprints-json", required=True)
    args = parser.parse_args()
    if not args.write_run_evidence:
        parser.error("--write-run-evidence is required")
    try:
        metadata = json.loads(args.metadata_json)
        fingerprints = json.loads(args.runtime_fingerprints_json)
    except json.JSONDecodeError as error:
        parser.error(f"invalid JSON argument: {error}")
    if not isinstance(metadata, dict) or not isinstance(fingerprints, dict):
        parser.error("metadata and runtime fingerprints must be JSON objects")
    write_run_evidence(
        args.run_root,
        arm=args.arm,
        run_id=args.run_id,
        metadata=metadata,
        runtime_fingerprints=fingerprints,
    )


if __name__ == "__main__":
    main()
