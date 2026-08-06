from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "mxfp8_linear_backend_model_matrix"
    / "summarize_results.py"
)
BACKENDS = ("flashinfer_cutlass", "flashinfer_cutedsl")
MODELS = ("qwen3-30b", "qwen3-235b", "nemotron3-super")
MODEL_NAMES = {
    "qwen3-30b": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b": "Qwen/Qwen3-235B-A22B",
    "nemotron3-super": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}
NEMO_RL_COMMIT = "1" * 40
VLLM_COMMIT = "2" * 40
MANIFEST_FIELDS = {
    "model",
    "nemo_rl_commit",
    "dependency_state_sha256",
    "vllm_commit",
    "vllm_source_sha256",
    "vllm_tracked_files_clean",
    "container",
    "recipe",
    "recipe_sha256",
    "cuda_graph",
    "precision",
    "is_mx",
    "quantization_ignored_layer_kws",
    "moe_backend",
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
    "gpu_memory_utilization",
    "linear_backend",
}


def _load_summary_module():
    spec = importlib.util.spec_from_file_location("mxfp8_model_matrix", SUMMARY_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _training_results_block(
    generation_length: float,
    total_step_seconds: float,
    generation_seconds: float,
    e2e_throughput: float,
    generation_throughput: float,
) -> str:
    return f"""
Training Results:
  Mean Generation Length: {generation_length:.2f}
  Total step time: {total_step_seconds:.2f}s
  generation: {generation_seconds:.2f}s
    - E2E (Tokens/sec/gpu): {e2e_throughput:.2f}
    - Generation Worker Group (Tokens/sec/gpu): {generation_throughput:.2f}
"""


def _write_driver_log(
    run_root: Path,
    backend: str,
    model_offset: float,
    generation_lengths: list[float] | None = None,
    num_steps: int = 8,
) -> None:
    blocks = []
    for step in range(1, num_steps + 1):
        generation_length = (
            generation_lengths[step - 1]
            if generation_lengths is not None
            else 1_000.0 + model_offset + step
        )
        if backend == "flashinfer_cutlass":
            total_step_seconds = 200.0 + model_offset + step
            generation_seconds = 100.0 + model_offset + step
            e2e_throughput = 400.0 + model_offset + step
            generation_throughput = 800.0 + model_offset + step
        else:
            total_step_seconds = 160.0 + model_offset + step
            generation_seconds = 80.0 + model_offset + step
            e2e_throughput = 480.0 + model_offset + step
            generation_throughput = 960.0 + model_offset + step
        blocks.append(
            _training_results_block(
                generation_length,
                total_step_seconds,
                generation_seconds,
                e2e_throughput,
                generation_throughput,
            )
        )

    log_dir = run_root / backend / "123-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "ray-driver.log").write_text("".join(blocks))
    manifest = {
        "model": MODEL_NAMES[run_root.name],
        "nemo_rl_commit": NEMO_RL_COMMIT,
        "dependency_state_sha256": "3" * 64,
        "vllm_commit": VLLM_COMMIT,
        "vllm_source_sha256": "4" * 64,
        "vllm_tracked_files_clean": True,
        "container": "/containers/nemo-rl.sqsh",
        "recipe": f"recipes/{run_root.name}.yaml",
        "recipe_sha256": "5" * 64,
        "cuda_graph": True,
        "precision": "fp8",
        "is_mx": True,
        "quantization_ignored_layer_kws": ["lm_head", "mlp.gate"],
        "moe_backend": "flashinfer_trtllm",
        "num_nodes": 4,
        "gpus_per_node": 4,
        "segment_size": 4,
        "num_prompts_per_step": 64,
        "num_generations_per_prompt": 32,
        "train_global_batch_size": 2_048,
        "max_total_sequence_length": 4_096,
        "max_input_sequence_length": 4_096,
        "max_new_tokens": 4_096,
        "max_model_len": 4_096,
        "generation_tensor_parallel_size": 1,
        "max_steps": 8,
        "gpu_memory_utilization": 0.6,
        "linear_backend": backend,
    }
    (run_root / backend / "run_manifest.json").write_text(json.dumps(manifest) + "\n")


def _set_metric_to_zero(log_path: Path, metric_label: str) -> None:
    log_path.write_text(
        re.sub(
            rf"({re.escape(metric_label)}:\s*)[0-9.]+",
            r"\g<1>0.00",
            log_path.read_text(),
        )
    )


def _write_complete_matrix(tmp_path: Path) -> dict[str, Path]:
    run_roots = {model: tmp_path / model for model in MODELS}
    for model_offset, run_root in enumerate(run_roots.values(), start=1):
        for backend in BACKENDS:
            _write_driver_log(run_root, backend, float(model_offset * 10))
    return run_roots


def test_write_results_summarizes_paired_steps_and_normalizes_to_cutlass(
    tmp_path: Path,
) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    output_dir = tmp_path / "summary"

    summary.write_results(run_roots, output_dir)

    with (output_dir / "step_metrics.csv").open(newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert len(rows) == 48
    assert {row["model"] for row in rows} == set(MODELS)

    results = json.loads((output_dir / "summary.json").read_text())
    for model in MODELS:
        cutlass = results[model]["flashinfer_cutlass"]
        cutedsl = results[model]["flashinfer_cutedsl"]
        assert cutlass["first_step"] == 3
        assert cutlass["last_step"] == 8
        assert cutlass["num_steps"] == 6
        assert (
            cutlass["mean_generation_length_mean"]
            == cutedsl["mean_generation_length_mean"]
        )
        assert cutlass["generation_tokens_per_sec_per_gpu_cutlass_normalized"] == 1.0
        assert cutlass["e2e_tokens_per_sec_per_gpu_cutlass_normalized"] == 1.0
        assert cutlass["generation_latency_speedup_vs_cutlass"] == 1.0
        assert cutlass["e2e_latency_speedup_vs_cutlass"] == 1.0
        assert cutedsl["generation_tokens_per_sec_per_gpu_cutlass_normalized"] > 1.0
        assert cutedsl["e2e_tokens_per_sec_per_gpu_cutlass_normalized"] > 1.0
        assert cutedsl["generation_latency_speedup_vs_cutlass"] > 1.0
        assert cutedsl["e2e_latency_speedup_vs_cutlass"] > 1.0
        assert cutlass["manifest"]["nemo_rl_commit"] == NEMO_RL_COMMIT
        assert cutedsl["manifest"]["linear_backend"] == "flashinfer_cutedsl"
        assert set(cutlass["manifest"]) == MANIFEST_FIELDS


def test_write_results_rejects_mismatched_launcher_run_config(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    manifest_path = run_roots["qwen3-30b"] / "flashinfer_cutedsl" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["num_prompts_per_step"] = 32
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(
        ValueError,
        match="Invariant manifest mismatch for qwen3-30b: num_prompts_per_step",
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_write_results_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    manifest_path = run_roots["qwen3-235b"] / "flashinfer_cutlass" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unvalidated_override"] = "different"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(
        ValueError,
        match=(
            "Unknown run manifest fields for qwen3-235b/flashinfer_cutlass: "
            "unvalidated_override"
        ),
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_write_results_requires_clean_custom_vllm_attestation(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    manifest_path = (
        run_roots["nemotron3-super"] / "flashinfer_cutedsl" / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["vllm_tracked_files_clean"] = False
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(
        ValueError,
        match=(
            "Custom vLLM clean attestation is false for "
            "nemotron3-super/flashinfer_cutedsl"
        ),
    ):
        summary.write_results(run_roots, tmp_path / "summary")


@pytest.mark.parametrize(
    ("field", "different_value"),
    (
        ("nemo_rl_commit", "3" * 40),
        ("vllm_commit", "4" * 40),
        ("dependency_state_sha256", "6" * 64),
        ("vllm_source_sha256", "7" * 64),
        ("container", "/containers/different.sqsh"),
        ("recipe", "recipes/different.yaml"),
        ("recipe_sha256", "8" * 64),
        ("cuda_graph", False),
        ("precision", "bfloat16"),
        ("is_mx", False),
        ("quantization_ignored_layer_kws", ["lm_head"]),
        ("moe_backend", "different_moe_backend"),
        ("num_nodes", 8),
        ("gpus_per_node", 8),
        ("segment_size", 8),
        ("num_prompts_per_step", 32),
        ("num_generations_per_prompt", 16),
        ("train_global_batch_size", 512),
        ("max_total_sequence_length", 8_192),
        ("max_input_sequence_length", 2_048),
        ("max_new_tokens", 2_048),
        ("max_model_len", 8_192),
        ("generation_tensor_parallel_size", 4),
        ("max_steps", 2),
        ("gpu_memory_utilization", 0.7),
    ),
)
def test_write_results_rejects_mismatched_invariant_manifest(
    tmp_path: Path, field: str, different_value: object
) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    manifest_path = run_roots["qwen3-30b"] / "flashinfer_cutedsl" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = different_value
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(
        ValueError,
        match=f"Invariant manifest mismatch for qwen3-30b: {field}",
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_write_results_requires_machine_readable_manifest(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    manifest_path = (
        run_roots["nemotron3-super"] / "flashinfer_cutlass" / "run_manifest.json"
    )
    manifest_path.unlink()

    with pytest.raises(
        ValueError,
        match=("Missing run manifest for nemotron3-super/flashinfer_cutlass"),
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_write_results_requires_exact_commit_ids(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    manifest_path = run_roots["qwen3-30b"] / "flashinfer_cutlass" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["nemo_rl_commit"] = "not-an-exact-commit"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(
        ValueError,
        match=(
            "Invalid exact commit in run manifest for "
            "qwen3-30b/flashinfer_cutlass: nemo_rl_commit"
        ),
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_write_results_rejects_wrong_declared_backend(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    manifest_path = run_roots["qwen3-235b"] / "flashinfer_cutedsl" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["linear_backend"] = "flashinfer_cutlass"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(
        ValueError,
        match=(
            "Declared linear backend mismatch for qwen3-235b/flashinfer_cutedsl: "
            "flashinfer_cutlass"
        ),
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_cli_requires_exact_model_matrix_by_default(tmp_path: Path) -> None:
    run_roots = _write_complete_matrix(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--model-run",
            f"qwen3-30b={run_roots['qwen3-30b']}",
            "--output-dir",
            str(tmp_path / "summary"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing required models: nemotron3-super, qwen3-235b" in result.stderr


def test_cli_rejects_unknown_model_label(tmp_path: Path) -> None:
    run_roots = _write_complete_matrix(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--model-run",
            f"qwen30={run_roots['qwen3-30b']}",
            "--output-dir",
            str(tmp_path / "summary"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Unknown models: qwen30" in result.stderr


def test_cli_allows_known_partial_matrix_only_with_opt_in(tmp_path: Path) -> None:
    run_roots = _write_complete_matrix(tmp_path)
    output_dir = tmp_path / "summary"
    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--model-run",
            f"qwen3-30b={run_roots['qwen3-30b']}",
            "--allow-partial",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert set(json.loads((output_dir / "summary.json").read_text())) == {"qwen3-30b"}


def test_write_results_rejects_missing_backend_log(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    missing_root = run_roots["qwen3-235b"]
    (missing_root / "flashinfer_cutedsl" / "123-logs" / "ray-driver.log").unlink()

    with pytest.raises(
        ValueError,
        match=(
            "Expected exactly one driver log for qwen3-235b/flashinfer_cutedsl, found 0"
        ),
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_write_results_rejects_unpaired_generation_lengths(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    mismatched_lengths = [
        1_011.0,
        1_012.0,
        1_013.0,
        1_014.0,
        9_999.0,
        1_016.0,
        1_017.0,
        1_018.0,
    ]
    _write_driver_log(
        run_roots["qwen3-30b"],
        "flashinfer_cutedsl",
        10.0,
        generation_lengths=mismatched_lengths,
    )

    with pytest.raises(
        ValueError,
        match="Paired mean generation length mismatch for qwen3-30b at step 5",
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_write_results_requires_every_requested_measured_step(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    for backend in BACKENDS:
        _write_driver_log(run_roots["qwen3-30b"], backend, 10.0, num_steps=3)

    with pytest.raises(
        ValueError,
        match=(
            "Expected complete measured steps for qwen3-30b/flashinfer_cutlass: "
            r"expected \[3, 4, 5, 6, 7, 8\], found \[3\]"
        ),
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_write_results_rejects_short_backend_result_row(tmp_path: Path) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    _write_driver_log(run_roots["qwen3-235b"], "flashinfer_cutedsl", 20.0, num_steps=7)

    with pytest.raises(
        ValueError,
        match=(
            "Expected complete measured steps for qwen3-235b/flashinfer_cutedsl: "
            r"expected \[3, 4, 5, 6, 7, 8\], found \[3, 4, 5, 6, 7\]"
        ),
    ):
        summary.write_results(run_roots, tmp_path / "summary")


def test_write_results_rejects_incomplete_training_results_block(
    tmp_path: Path,
) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    log_path = (
        run_roots["qwen3-30b"] / "flashinfer_cutedsl" / "123-logs" / "ray-driver.log"
    )
    log_path.write_text(
        log_path.read_text().replace("    - E2E (Tokens/sec/gpu): 495.00\n", "", 1)
    )

    with pytest.raises(
        ValueError,
        match=(
            "Incomplete Training Results block 5 for qwen3-30b/flashinfer_cutedsl: "
            r"missing E2E \(Tokens/sec/gpu\)"
        ),
    ):
        summary.write_results(run_roots, tmp_path / "summary")


@pytest.mark.parametrize(
    ("metric_label", "summary_metric"),
    (
        ("Total step time", "total_step_seconds_mean"),
        ("generation", "generation_seconds_mean"),
        ("E2E (Tokens/sec/gpu)", "e2e_tokens_per_sec_per_gpu_mean"),
        (
            "Generation Worker Group (Tokens/sec/gpu)",
            "generation_tokens_per_sec_per_gpu_mean",
        ),
    ),
)
def test_write_results_rejects_zero_normalization_denominator(
    tmp_path: Path, metric_label: str, summary_metric: str
) -> None:
    summary = _load_summary_module()
    run_roots = _write_complete_matrix(tmp_path)
    _set_metric_to_zero(
        run_roots["qwen3-30b"] / "flashinfer_cutlass" / "123-logs" / "ray-driver.log",
        metric_label,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid normalization denominator for qwen3-30b/flashinfer_cutlass, "
            f"steps 3-8: {summary_metric} must be positive"
        ),
    ):
        summary.write_results(run_roots, tmp_path / "summary")
