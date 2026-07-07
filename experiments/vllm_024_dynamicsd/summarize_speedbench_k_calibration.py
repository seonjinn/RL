#!/usr/bin/env python3
"""Build and validate signed model/profile-specific SPEED-Bench K schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_FIELDS = (
    "model",
    "model_config_hash",
    "context_profile",
    "request_plan_hash",
    "runtime_image_sha256",
    "method",
    "dataset_config",
    "temperature",
    "top_p",
    "seed",
    "sampling_protocol",
)
SELECTION_POLICY = "smallest K within 2% of best median throughput"
FIT_POLICY = "monotone non-increasing K by active concurrency"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sign_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    signed = json.loads(json.dumps(payload))
    signed.pop("signature", None)
    signed["signature"] = {
        "algorithm": "sha256",
        "payload_sha256": hashlib.sha256(_canonical_bytes(signed)).hexdigest(),
    }
    return signed


def verify_artifact_signature(artifact: Mapping[str, Any]) -> None:
    signature = artifact.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "sha256":
        raise ValueError("calibration artifact signature is missing or unsupported")
    unsigned = dict(artifact)
    unsigned.pop("signature", None)
    expected = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if signature.get("payload_sha256") != expected:
        raise ValueError("calibration artifact signature mismatch")


def _validate_artifact_provenance(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != 1:
        raise ValueError("calibration artifact schema_version must be 1")
    if artifact.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("calibration artifact selection_policy is missing or invalid")
    if artifact.get("fit_policy") != FIT_POLICY:
        raise ValueError("calibration artifact fit_policy is missing or invalid")

    medians = artifact.get("median_output_tok_s_per_gpu")
    if not isinstance(medians, dict) or not medians:
        raise ValueError("calibration artifact median throughput provenance is missing")
    for by_k in medians.values():
        if not isinstance(by_k, dict) or not by_k:
            raise ValueError("calibration artifact median throughput cells are invalid")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
            for value in by_k.values()
        ):
            raise ValueError("calibration artifact median throughputs must be positive")

    selected = artifact.get("selected_k")
    selected_raw = artifact.get("selected_k_before_monotone_fit")
    if not isinstance(selected, dict) or not selected:
        raise ValueError("calibration artifact selected_k is missing")
    if not isinstance(selected_raw, dict) or set(selected_raw) != set(selected):
        raise ValueError("calibration artifact pre-fit selected K values are invalid")

    schedule = artifact.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("calibration artifact schedule provenance is missing")
    normalized_schedule: list[dict[str, int]] = []
    previous_end: int | None = None
    previous_k: int | None = None
    for index, row in enumerate(schedule):
        if not isinstance(row, dict):
            raise ValueError("calibration artifact schedule rows must be objects")
        start = row.get("start")
        end = row.get("end")
        static_k = row.get("k")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (start, end, static_k)
        ):
            raise ValueError("calibration artifact schedule values must be positive integers")
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(static_k, int)
        if end < start or (previous_end is not None and start != previous_end + 1):
            raise ValueError("calibration artifact schedule ranges must be contiguous")
        if previous_k is not None and static_k > previous_k:
            raise ValueError("calibration artifact schedule K must be monotone non-increasing")
        if selected.get(str(start)) != static_k:
            raise ValueError(
                f"calibration artifact selected_k mismatch at schedule row {index}"
            )
        normalized_schedule.append({"start": start, "end": end, "k": static_k})
        previous_end = end
        previous_k = static_k
    if set(selected) != {str(row["start"]) for row in normalized_schedule}:
        raise ValueError("calibration artifact selected_k does not match schedule ranges")
    if artifact.get("dynamic_schedule") != _schedule_string(normalized_schedule):
        raise ValueError("calibration artifact dynamic_schedule does not match schedule")

    source_results = artifact.get("source_results")
    if (
        not isinstance(source_results, list)
        or len(source_results) < 3
        or len(source_results) % 3 != 0
    ):
        raise ValueError("calibration artifact source_results must contain repeat triplets")
    for source in source_results:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not source["path"]
        ):
            raise ValueError("calibration artifact source_results paths are invalid")
        source_hash = source.get("sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError("calibration artifact source_results hashes are invalid")
        try:
            int(source_hash, 16)
        except ValueError as exc:
            raise ValueError(
                "calibration artifact source_results hashes are invalid"
            ) from exc


def _required_config_value(config: Mapping[str, Any], field: str) -> Any:
    value = config.get(field)
    if value is None or value == "" or value == "unknown":
        raise ValueError(f"calibration result missing {field}")
    return value


def _load_calibration_row(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    config = payload.get("config")
    summary = payload.get("summary")
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError(f"invalid calibration result structure: {path}")
    if config.get("mode") not in {"static", "mtp_static"}:
        return None
    row = {field: _required_config_value(config, field) for field in IDENTITY_FIELDS}
    for field in ("active_concurrency", "static_k", "calibration_repeat"):
        value = _required_config_value(config, field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"calibration result {field} must be a positive integer")
        row[field] = value
    throughput = summary.get("output_tok_s_per_gpu")
    if not isinstance(throughput, (int, float)) or isinstance(throughput, bool):
        raise ValueError("calibration result output_tok_s_per_gpu must be numeric")
    if throughput <= 0:
        raise ValueError("calibration result output_tok_s_per_gpu must be positive")
    row["output_tok_s_per_gpu"] = float(throughput)
    row["path"] = str(path)
    row["sha256"] = sha256_file(path)
    return row


def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in IDENTITY_FIELDS)


def _schedule_rows(selected_k: Mapping[int, int]) -> list[dict[str, int]]:
    concurrencies = sorted(selected_k)
    return [
        {
            "start": concurrency,
            "end": (
                concurrencies[index + 1] - 1
                if index + 1 < len(concurrencies)
                else concurrency
            ),
            "k": selected_k[concurrency],
        }
        for index, concurrency in enumerate(concurrencies)
    ]


def _schedule_string(rows: Sequence[Mapping[str, int]]) -> str:
    return ",".join(f"{row['start']}:{row['end']}:{row['k']}" for row in rows)


def build_calibration_artifact(paths: Iterable[Path]) -> dict[str, Any]:
    rows = [row for path in sorted(paths) if (row := _load_calibration_row(path))]
    if not rows:
        raise ValueError("no successful static calibration results found")
    identities = {_identity_key(row) for row in rows}
    if len(identities) != 1:
        raise ValueError("calibration results mix model/profile/runtime/sampling identities")

    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["active_concurrency"]), int(row["static_k"]))
        cells.setdefault(key, []).append(row)
    medians: dict[int, dict[int, float]] = {}
    for (concurrency, static_k), cell_rows in sorted(cells.items()):
        repeats = sorted(int(row["calibration_repeat"]) for row in cell_rows)
        if repeats != [1, 2, 3]:
            raise ValueError(
                "each concurrency/K cell requires three successful repeats "
                f"numbered 1,2,3; c={concurrency} k={static_k} got={repeats}"
            )
        medians.setdefault(concurrency, {})[static_k] = statistics.median(
            float(row["output_tok_s_per_gpu"]) for row in cell_rows
        )

    selected_raw: dict[int, int] = {}
    for concurrency, by_k in sorted(medians.items()):
        best = max(by_k.values())
        eligible = [static_k for static_k, value in by_k.items() if value >= best * 0.98]
        selected_raw[concurrency] = min(eligible)
    selected: dict[int, int] = {}
    previous: int | None = None
    for concurrency, static_k in sorted(selected_raw.items()):
        fitted = static_k if previous is None else min(previous, static_k)
        selected[concurrency] = fitted
        previous = fitted

    identity = rows[0]
    schedule = _schedule_rows(selected)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "model": identity["model"],
        "model_config_hash": identity["model_config_hash"],
        "context_profile": identity["context_profile"],
        "request_plan_hash": identity["request_plan_hash"],
        "runtime_image_sha256": identity["runtime_image_sha256"],
        "method": identity["method"],
        "dataset_config": identity["dataset_config"],
        "sampling": {
            "temperature": identity["temperature"],
            "top_p": identity["top_p"],
            "seed": identity["seed"],
            "sampling_protocol": identity["sampling_protocol"],
        },
        "selection_policy": SELECTION_POLICY,
        "fit_policy": FIT_POLICY,
        "repeats_per_cell": 3,
        "median_output_tok_s_per_gpu": {
            str(concurrency): {
                str(static_k): value for static_k, value in sorted(by_k.items())
            }
            for concurrency, by_k in sorted(medians.items())
        },
        "selected_k_before_monotone_fit": {
            str(key): value for key, value in selected_raw.items()
        },
        "selected_k": {str(key): value for key, value in selected.items()},
        "schedule": schedule,
        "dynamic_schedule": _schedule_string(schedule),
        "source_results": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in sorted(rows, key=lambda item: str(item["path"]))
        ],
    }
    return sign_artifact(payload)


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    verify_artifact_signature(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_calibration_artifact(
    path: Path,
    *,
    model: str,
    model_config_hash: str,
    context_profile: str,
    request_plan_hash: str,
    runtime_image_sha256: str,
    method: str,
    dataset_config: str,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("calibration artifact must be an object")
    verify_artifact_signature(artifact)
    if artifact.get("status") != "complete" or artifact.get("repeats_per_cell") != 3:
        raise ValueError("calibration artifact is not a successful three-repeat artifact")
    _validate_artifact_provenance(artifact)
    expected = {
        "model": model,
        "model_config_hash": model_config_hash,
        "context_profile": context_profile,
        "request_plan_hash": request_plan_hash,
        "runtime_image_sha256": runtime_image_sha256,
        "method": method,
        "dataset_config": dataset_config,
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise ValueError(
                f"calibration artifact {field} mismatch: {artifact.get(field)!r} != {value!r}"
            )
    sampling = artifact.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("calibration artifact sampling is missing")
    sampling_expected = {
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "sampling_protocol": "sync-rl-overlay-user",
    }
    for field, value in sampling_expected.items():
        if sampling.get(field) != value:
            raise ValueError(
                f"calibration artifact {field} mismatch: {sampling.get(field)!r} != {value!r}"
            )
    schedule = artifact.get("dynamic_schedule")
    if not isinstance(schedule, str) or not schedule:
        raise ValueError("calibration artifact dynamic_schedule is missing")
    return schedule


def _result_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/result.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("root", type=Path)
    summarize.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--model-config-hash", required=True)
    validate.add_argument("--context-profile", required=True)
    validate.add_argument("--request-plan-hash", required=True)
    validate.add_argument("--runtime-image-sha256", required=True)
    validate.add_argument("--method", required=True)
    validate.add_argument("--dataset-config", required=True)
    validate.add_argument("--temperature", type=float, required=True)
    validate.add_argument("--top-p", type=float, required=True)
    validate.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        artifact = build_calibration_artifact(_result_paths(args.root))
        write_artifact(args.output, artifact)
        print(args.output)
    else:
        schedule = validate_calibration_artifact(
            args.artifact,
            model=args.model,
            model_config_hash=args.model_config_hash,
            context_profile=args.context_profile,
            request_plan_hash=args.request_plan_hash,
            runtime_image_sha256=args.runtime_image_sha256,
            method=args.method,
            dataset_config=args.dataset_config,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        print(schedule)


if __name__ == "__main__":
    main()
