#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Mapping, NamedTuple, Sequence, TypedDict, cast


BACKENDS = ("flashinfer_cutlass", "flashinfer_cutedsl")
MODELS = ("qwen3-30b", "qwen3-235b", "nemotron3-super")
MODEL_NAMES = {
    "qwen3-30b": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b": "Qwen/Qwen3-235B-A22B",
    "nemotron3-super": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EXACT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
TOTAL_STEP_PATTERN = re.compile(r"Total step time:\s*([0-9.]+)s")
GENERATION_PATTERN = re.compile(r"generation:\s*([0-9.]+)s")
E2E_THROUGHPUT_PATTERN = re.compile(r"E2E \(Tokens/sec/gpu\):\s*([0-9.]+)")
GENERATION_THROUGHPUT_PATTERN = re.compile(
    r"Generation Worker Group \(Tokens/sec/gpu\):\s*([0-9.]+)"
)
MEAN_GENERATION_LENGTH_PATTERN = re.compile(r"Mean Generation Length:\s*([0-9.]+)")


class StepMetrics(NamedTuple):
    step: int
    mean_generation_length: float
    total_step_seconds: float
    generation_seconds: float
    e2e_tokens_per_sec_per_gpu: float
    generation_tokens_per_sec_per_gpu: float


class StepSummary(NamedTuple):
    first_step: int
    last_step: int
    num_steps: int
    mean_generation_length_mean: float
    total_step_seconds_mean: float
    generation_seconds_mean: float
    e2e_tokens_per_sec_per_gpu_mean: float
    generation_tokens_per_sec_per_gpu_mean: float


class CsvRow(TypedDict):
    model: str
    backend: str
    step: int
    mean_generation_length: float
    total_step_seconds: float
    generation_seconds: float
    e2e_tokens_per_sec_per_gpu: float
    generation_tokens_per_sec_per_gpu: float


class RunManifest(TypedDict):
    model: str
    nemo_rl_commit: str
    dependency_state_sha256: str
    vllm_commit: str
    vllm_source_sha256: str
    vllm_dependency_state_sha256: str
    vllm_tracked_files_clean: bool
    container: str
    recipe: str
    recipe_sha256: str
    cuda_graph: bool
    precision: str
    is_mx: bool
    quantization_ignored_layer_kws: list[str]
    moe_backend: str
    num_nodes: int
    gpus_per_node: int
    segment_size: int
    num_prompts_per_step: int
    num_generations_per_prompt: int
    train_global_batch_size: int
    max_total_sequence_length: int
    max_input_sequence_length: int
    max_new_tokens: int
    max_model_len: int
    generation_tensor_parallel_size: int
    max_steps: int
    gpu_memory_utilization: float
    logprob_batch_size: int
    logprob_chunk_size: int | None
    activation_checkpointing: bool
    defer_fp32_logits: bool
    sequence_packing: bool
    linear_backend: str


CSV_FIELDNAMES = tuple(CsvRow.__annotations__)
MANIFEST_FIELDS = tuple(RunManifest.__annotations__)
MANIFEST_INVARIANT_FIELDS = tuple(
    field for field in MANIFEST_FIELDS if field != "linear_backend"
)


def _metric(pattern: re.Pattern[str], block: str) -> float | None:
    match = pattern.search(block)
    return float(match.group(1)) if match else None


def parse_training_results(log_text: str, source: str = "log") -> list[StepMetrics]:
    clean_text = ANSI_ESCAPE.sub("", log_text)
    steps: list[StepMetrics] = []
    for step, block in enumerate(clean_text.split("Training Results:")[1:], start=1):
        mean_generation_length = _metric(MEAN_GENERATION_LENGTH_PATTERN, block)
        total_step_seconds = _metric(TOTAL_STEP_PATTERN, block)
        generation_seconds = _metric(GENERATION_PATTERN, block)
        e2e_throughput = _metric(E2E_THROUGHPUT_PATTERN, block)
        generation_throughput = _metric(GENERATION_THROUGHPUT_PATTERN, block)
        if (
            mean_generation_length is None
            or total_step_seconds is None
            or generation_seconds is None
            or e2e_throughput is None
            or generation_throughput is None
        ):
            missing_metrics = [
                metric_name
                for metric_name, metric_value in (
                    ("Mean Generation Length", mean_generation_length),
                    ("Total step time", total_step_seconds),
                    ("generation", generation_seconds),
                    ("E2E (Tokens/sec/gpu)", e2e_throughput),
                    (
                        "Generation Worker Group (Tokens/sec/gpu)",
                        generation_throughput,
                    ),
                )
                if metric_value is None
            ]
            raise ValueError(
                f"Incomplete Training Results block {step} for {source}: missing "
                f"{', '.join(missing_metrics)}"
            )
        steps.append(
            StepMetrics(
                step=step,
                mean_generation_length=mean_generation_length,
                total_step_seconds=total_step_seconds,
                generation_seconds=generation_seconds,
                e2e_tokens_per_sec_per_gpu=e2e_throughput,
                generation_tokens_per_sec_per_gpu=generation_throughput,
            )
        )
    return steps


def _find_driver_log(model: str, run_root: Path, backend: str) -> Path:
    matches = sorted((run_root / backend).glob("*-logs/ray-driver.log"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one driver log for {model}/{backend}, found {len(matches)}"
        )
    return matches[0]


def _load_run_manifest(model: str, run_root: Path, backend: str) -> RunManifest:
    manifest_path = run_root / backend / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing run manifest for {model}/{backend}")
    try:
        raw_manifest: object = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON run manifest for {model}/{backend}: {error.msg}"
        ) from error
    if not isinstance(raw_manifest, dict):
        raise ValueError(f"Run manifest for {model}/{backend} must be a JSON object")
    manifest = cast(dict[str, object], raw_manifest)
    missing_fields = [field for field in MANIFEST_FIELDS if field not in manifest]
    if missing_fields:
        raise ValueError(
            f"Incomplete run manifest for {model}/{backend}: missing "
            f"{', '.join(missing_fields)}"
        )
    unknown_fields = sorted(set(manifest) - set(MANIFEST_FIELDS))
    if unknown_fields:
        raise ValueError(
            f"Unknown run manifest fields for {model}/{backend}: "
            f"{', '.join(unknown_fields)}"
        )

    string_fields = (
        "model",
        "nemo_rl_commit",
        "dependency_state_sha256",
        "vllm_commit",
        "vllm_source_sha256",
        "vllm_dependency_state_sha256",
        "container",
        "recipe",
        "recipe_sha256",
        "precision",
        "moe_backend",
        "linear_backend",
    )
    for field in string_fields:
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ValueError(
                f"Invalid run manifest field for {model}/{backend}: {field}"
            )
    for field in (
        "vllm_tracked_files_clean",
        "cuda_graph",
        "is_mx",
        "activation_checkpointing",
        "defer_fp32_logits",
        "sequence_packing",
    ):
        if not isinstance(manifest[field], bool):
            raise ValueError(
                f"Invalid run manifest field for {model}/{backend}: {field}"
            )
    if manifest["vllm_tracked_files_clean"] is not True:
        raise ValueError(
            f"Custom vLLM clean attestation is false for {model}/{backend}"
        )
    quantization_scope = manifest["quantization_ignored_layer_kws"]
    if not isinstance(quantization_scope, list) or not all(
        isinstance(keyword, str) for keyword in quantization_scope
    ):
        raise ValueError(
            "Invalid run manifest field for "
            f"{model}/{backend}: quantization_ignored_layer_kws"
        )
    integer_fields = (
        "num_nodes",
        "gpus_per_node",
        "segment_size",
        "num_prompts_per_step",
        "num_generations_per_prompt",
        "train_global_batch_size",
        "max_total_sequence_length",
        "max_input_sequence_length",
        "max_new_tokens",
        "max_model_len",
        "generation_tensor_parallel_size",
        "max_steps",
        "logprob_batch_size",
    )
    for field in integer_fields:
        value = manifest[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Invalid run manifest field for {model}/{backend}: {field}"
            )
    logprob_chunk_size = manifest["logprob_chunk_size"]
    if logprob_chunk_size is not None and (
        isinstance(logprob_chunk_size, bool)
        or not isinstance(logprob_chunk_size, int)
        or logprob_chunk_size <= 0
    ):
        raise ValueError(
            f"Invalid run manifest field for {model}/{backend}: logprob_chunk_size"
        )
    gpu_memory_utilization = manifest["gpu_memory_utilization"]
    if (
        isinstance(gpu_memory_utilization, bool)
        or not isinstance(gpu_memory_utilization, (int, float))
        or not 0 < gpu_memory_utilization <= 1
    ):
        raise ValueError(
            f"Invalid run manifest field for {model}/{backend}: gpu_memory_utilization"
        )
    for commit_field in ("nemo_rl_commit", "vllm_commit"):
        commit = cast(str, manifest[commit_field])
        if EXACT_COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError(
                f"Invalid exact commit in run manifest for {model}/{backend}: "
                f"{commit_field}"
            )
    for sha256_field in (
        "dependency_state_sha256",
        "vllm_source_sha256",
        "vllm_dependency_state_sha256",
        "recipe_sha256",
    ):
        sha256 = cast(str, manifest[sha256_field])
        if SHA256_PATTERN.fullmatch(sha256) is None:
            raise ValueError(
                f"Invalid SHA256 in run manifest for {model}/{backend}: {sha256_field}"
            )
    if manifest["model"] != MODEL_NAMES[model]:
        raise ValueError(
            f"Declared model mismatch for {model}/{backend}: {manifest['model']}"
        )
    if manifest["linear_backend"] != backend:
        raise ValueError(
            f"Declared linear backend mismatch for {model}/{backend}: "
            f"{manifest['linear_backend']}"
        )
    return cast(RunManifest, manifest)


def validate_paired_manifests(
    model: str, cutlass: RunManifest, cutedsl: RunManifest
) -> None:
    cutlass_mapping = cast(Mapping[str, object], cutlass)
    cutedsl_mapping = cast(Mapping[str, object], cutedsl)
    for field in MANIFEST_INVARIANT_FIELDS:
        if cutlass_mapping[field] != cutedsl_mapping[field]:
            raise ValueError(f"Invariant manifest mismatch for {model}: {field}")


def _measured_steps(
    model: str,
    backend: str,
    steps: Sequence[StepMetrics],
    first_step: int,
    last_step: int,
) -> list[StepMetrics]:
    selected = [step for step in steps if first_step <= step.step <= last_step]
    expected_steps = list(range(first_step, last_step + 1))
    actual_steps = [step.step for step in selected]
    if actual_steps != expected_steps:
        raise ValueError(
            f"Expected complete measured steps for {model}/{backend}: "
            f"expected {expected_steps}, found {actual_steps}"
        )
    return selected


def summarize_steps(steps: Sequence[StepMetrics]) -> StepSummary:
    return StepSummary(
        first_step=steps[0].step,
        last_step=steps[-1].step,
        num_steps=len(steps),
        mean_generation_length_mean=fmean(
            step.mean_generation_length for step in steps
        ),
        total_step_seconds_mean=fmean(step.total_step_seconds for step in steps),
        generation_seconds_mean=fmean(step.generation_seconds for step in steps),
        e2e_tokens_per_sec_per_gpu_mean=fmean(
            step.e2e_tokens_per_sec_per_gpu for step in steps
        ),
        generation_tokens_per_sec_per_gpu_mean=fmean(
            step.generation_tokens_per_sec_per_gpu for step in steps
        ),
    )


def validate_paired_steps(
    model: str,
    cutlass_steps: Sequence[StepMetrics],
    cutedsl_steps: Sequence[StepMetrics],
) -> None:
    if len(cutlass_steps) != len(cutedsl_steps):
        raise ValueError(
            f"Paired measured-step count mismatch for {model}: "
            f"flashinfer_cutlass={len(cutlass_steps)}, "
            f"flashinfer_cutedsl={len(cutedsl_steps)}"
        )
    for cutlass_step, cutedsl_step in zip(cutlass_steps, cutedsl_steps, strict=True):
        if cutlass_step.step != cutedsl_step.step:
            raise ValueError(
                f"Paired measured-step mismatch for {model}: "
                f"flashinfer_cutlass={cutlass_step.step}, "
                f"flashinfer_cutedsl={cutedsl_step.step}"
            )
        if cutlass_step.mean_generation_length != cutedsl_step.mean_generation_length:
            raise ValueError(
                f"Paired mean generation length mismatch for {model} at step "
                f"{cutlass_step.step}: "
                f"flashinfer_cutlass={cutlass_step.mean_generation_length}, "
                f"flashinfer_cutedsl={cutedsl_step.mean_generation_length}"
            )


def _with_cutlass_normalization(
    summary: StepSummary, cutlass_summary: StepSummary
) -> dict[str, int | float]:
    metrics = summary._asdict()
    metrics.update(
        {
            "generation_tokens_per_sec_per_gpu_cutlass_normalized": (
                summary.generation_tokens_per_sec_per_gpu_mean
                / cutlass_summary.generation_tokens_per_sec_per_gpu_mean
            ),
            "e2e_tokens_per_sec_per_gpu_cutlass_normalized": (
                summary.e2e_tokens_per_sec_per_gpu_mean
                / cutlass_summary.e2e_tokens_per_sec_per_gpu_mean
            ),
            "generation_latency_speedup_vs_cutlass": (
                cutlass_summary.generation_seconds_mean
                / summary.generation_seconds_mean
            ),
            "e2e_latency_speedup_vs_cutlass": (
                cutlass_summary.total_step_seconds_mean
                / summary.total_step_seconds_mean
            ),
        }
    )
    return metrics


def validate_normalization_denominators(
    model: str,
    backend: str,
    summary: StepSummary,
    first_step: int,
    last_step: int,
) -> None:
    for metric_name, metric_value in (
        (
            "generation_tokens_per_sec_per_gpu_mean",
            summary.generation_tokens_per_sec_per_gpu_mean,
        ),
        ("e2e_tokens_per_sec_per_gpu_mean", summary.e2e_tokens_per_sec_per_gpu_mean),
        ("generation_seconds_mean", summary.generation_seconds_mean),
        ("total_step_seconds_mean", summary.total_step_seconds_mean),
    ):
        if metric_value <= 0:
            raise ValueError(
                f"Invalid normalization denominator for {model}/{backend}, "
                f"steps {first_step}-{last_step}: {metric_name} must be positive"
            )


def write_results(
    model_run_roots: Mapping[str, Path],
    output_dir: Path,
    first_step: int = 3,
    last_step: int = 8,
) -> None:
    if first_step > last_step:
        raise ValueError("first_step must be less than or equal to last_step")

    rows: list[CsvRow] = []
    summaries: dict[str, dict[str, StepSummary]] = {}
    manifests: dict[str, dict[str, RunManifest]] = {}
    for model, raw_run_root in model_run_roots.items():
        run_root = Path(raw_run_root)
        manifests[model] = {
            backend: _load_run_manifest(model, run_root, backend)
            for backend in BACKENDS
        }
        validate_paired_manifests(
            model,
            manifests[model]["flashinfer_cutlass"],
            manifests[model]["flashinfer_cutedsl"],
        )
        measured_steps: dict[str, list[StepMetrics]] = {}
        for backend in BACKENDS:
            log_path = _find_driver_log(model, run_root, backend)
            steps = parse_training_results(
                log_path.read_text(errors="replace"), source=f"{model}/{backend}"
            )
            rows.extend(
                CsvRow(model=model, backend=backend, **step._asdict()) for step in steps
            )
            measured_steps[backend] = _measured_steps(
                model, backend, steps, first_step, last_step
            )

        validate_paired_steps(
            model,
            measured_steps["flashinfer_cutlass"],
            measured_steps["flashinfer_cutedsl"],
        )
        summaries[model] = {
            backend: summarize_steps(measured_steps[backend]) for backend in BACKENDS
        }
        for backend, summary in summaries[model].items():
            validate_normalization_denominators(
                model, backend, summary, first_step, last_step
            )

    normalized_summaries: dict[str, dict[str, dict[str, int | float | RunManifest]]] = {
        model: {
            backend: {
                **_with_cutlass_normalization(
                    summary, model_summaries["flashinfer_cutlass"]
                ),
                "manifest": manifests[model][backend],
            }
            for backend, summary in model_summaries.items()
        }
        for model, model_summaries in summaries.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "step_metrics.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(normalized_summaries, indent=2, sort_keys=True) + "\n"
    )


def _parse_model_run(value: str) -> tuple[str, Path]:
    model, separator, run_root = value.partition("=")
    if not separator or not model or not run_root:
        raise argparse.ArgumentTypeError("Expected MODEL=RUN_ROOT")
    return model, Path(run_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-run",
        action="append",
        type=_parse_model_run,
        required=True,
        metavar="MODEL=RUN_ROOT",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-step", type=int, default=3)
    parser.add_argument("--last-step", type=int, default=8)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a known subset of the three model rows.",
    )
    return parser.parse_args()


def validate_model_set(
    model_run_roots: Mapping[str, Path], allow_partial: bool
) -> None:
    actual_models = set(model_run_roots)
    required_models = set(MODELS)
    unknown_models = sorted(actual_models - required_models)
    if unknown_models:
        raise ValueError(f"Unknown models: {', '.join(unknown_models)}")
    missing_models = sorted(required_models - actual_models)
    if missing_models and not allow_partial:
        raise ValueError(f"Missing required models: {', '.join(missing_models)}")


def main() -> None:
    args = parse_args()
    model_run_roots = dict(args.model_run)
    if len(model_run_roots) != len(args.model_run):
        raise ValueError("Each MODEL may be supplied only once")
    validate_model_set(model_run_roots, args.allow_partial)
    write_results(
        model_run_roots,
        args.output_dir,
        first_step=args.first_step,
        last_step=args.last_step,
    )


if __name__ == "__main__":
    main()
