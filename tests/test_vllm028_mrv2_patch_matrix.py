from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "experiments" / "vllm_028_nemotron_bf16_matrix"
SUBMIT_PATH = PACKAGE_ROOT / "submit_mrv2_patch_matrix.py"
MANIFEST_PATH = PACKAGE_ROOT / "mrv2_patch_manifest.json"
PATCHED_IMAGE = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
    "vllm-openai-v0.28.0-mrv2-dynamick-core-aarch64-ubuntu2404.sqsh"
)
EXPECTED_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 128, 512)
EXPECTED_DYNAMIC_SCHEDULE = "1:4:5,5:16:3,17:64:2,65:128:1,129:512:0"


def load_submit() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_submit_mrv2_patch_matrix", SUBMIT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_full_matrix_has_one_row_for_each_required_cell() -> None:
    submit = load_submit()

    rows = submit.build_matrix_rows(manifest_path=MANIFEST_PATH)

    assert len(rows) == 64
    assert {row["model_key"] for row in rows} == {"super", "ultra"}
    assert {(row["shape_key"], row["isl"], row["osl"]) for row in rows} == {
        ("isl1k_osl10k", 1000, 10000),
        ("isl10k_osl1k", 10000, 1000),
    }
    assert {row["method_key"] for row in rows} == {
        "baseline",
        "mtp_dynamic_max_k5",
    }
    assert tuple(sorted({row["batch_size"] for row in rows})) == EXPECTED_BATCH_SIZES
    assert {row["runner_key"] for row in rows} == {"mrv2"}
    assert {row["cudagraph_mode"] for row in rows} == {"FULL_AND_PIECEWISE"}
    assert all(row["dtype"] == "bfloat16" for row in rows)
    assert all(row["kv_cache_dtype"] == "fp8" for row in rows)


def test_edge_gate_selects_bs1_and_bs512_for_every_model_shape_method() -> None:
    submit = load_submit()
    rows = submit.build_matrix_rows(manifest_path=MANIFEST_PATH)

    selected = submit.select_matrix_rows(rows, gate="edge")

    assert len(selected) == 16
    assert {row["batch_size"] for row in selected} == {1, 512}
    assert len(submit.select_matrix_rows(rows, gate="full")) == 64
    remaining = submit.select_matrix_rows(rows, gate="remaining")
    assert len(remaining) == 48
    assert {row["batch_size"] for row in remaining} == {2, 4, 8, 16, 32, 128}


def test_result_directory_contains_patch_model_shape_method_and_batch() -> None:
    submit = load_submit()
    row = next(
        row
        for row in submit.build_matrix_rows(manifest_path=MANIFEST_PATH)
        if row["model_key"] == "ultra"
        and row["shape_key"] == "isl10k_osl1k"
        and row["method_key"] == "mtp_dynamic_max_k5"
        and row["batch_size"] == 512
    )

    result_dir = submit.result_dir_for_row(row)

    expected_patch_id = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()[:12]
    assert result_dir.parts[-5:] == (
        expected_patch_id,
        "ultra",
        "isl10k_osl1k",
        "mtp_dynamic_max_k5",
        "bs512",
    )


def test_render_super_baseline_pins_patched_mrv2_full_graph_and_one_cell() -> None:
    submit = load_submit()
    row = next(
        row
        for row in submit.build_matrix_rows(manifest_path=MANIFEST_PATH)
        if row["model_key"] == "super"
        and row["shape_key"] == "isl1k_osl10k"
        and row["method_key"] == "baseline"
        and row["batch_size"] == 1
    )
    result_dir = Path("/lustre/results/patch/super/isl1k_osl10k/baseline/bs1")

    script = submit.render_cell_sbatch(
        row,
        experiment_dir=PACKAGE_ROOT,
        result_dir=result_dir,
        manifest=load_manifest(),
    )

    assert f"readonly STABLE_CONTAINER_IMAGE={PATCHED_IMAGE}" in script
    assert "export VLLM_USE_V2_MODEL_RUNNER=1" in script
    assert "FULL_AND_PIECEWISE" in script
    assert "BF16 weights; FP8 KV cache" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --time=05:00:00" in script
    assert "--tensor-parallel-size 2" in script
    assert "--runner-key mrv2" in script
    assert "--method-key baseline" in script
    assert "--batch-size 1" in script
    assert "--speculative-config-json null" in script
    assert script.count("python3 /workspace/exp/benchmark.py") == 1
    assert '--output "${RESULT_RUN_DIR}/result.json"' in script
    assert "patchset manifest changed after rendering" in script
    assert "patched container patchset mismatch" in script
    assert "patched container base commit mismatch" in script
    assert "patched container artifact digest mismatch" in script
    assert '--vllm-base-commit "${EXPECTED_BASE_COMMIT}"' in script
    assert '--patched-vllm-head "${PATCHED_VLLM_HEAD}"' in script
    assert '--patchset-manifest-sha256 "${PATCHSET_MANIFEST_SHA256}"' in script
    assert '--container-digest "sha256:${CONTAINER_ARTIFACT_SHA256}"' in script
    assert 'readonly RESULT_RUN_DIR="' in script
    assert '/job-${SLURM_JOB_ID}"' in script
    assert 'tee "${BENCHMARK_LOG}"' in script
    assert "cuda_graph_evidence.json" in script
    assert "incomplete target CUDA Graph capture evidence" in script
    assert "vllm-openai-v0.28.0-aarch64-ubuntu2404.sqsh" not in script


def test_render_ultra_dynamic_pins_ray_topology_and_exact_schedule() -> None:
    submit = load_submit()
    row = next(
        row
        for row in submit.build_matrix_rows(manifest_path=MANIFEST_PATH)
        if row["model_key"] == "ultra"
        and row["shape_key"] == "isl10k_osl1k"
        and row["method_key"] == "mtp_dynamic_max_k5"
        and row["batch_size"] == 128
    )

    script = submit.render_cell_sbatch(
        row,
        experiment_dir=PACKAGE_ROOT,
        result_dir=Path(
            "/lustre/results/patch/ultra/isl10k_osl1k/mtp_dynamic_max_k5/bs128"
        ),
        manifest=load_manifest(),
    )

    assert "#SBATCH --nodes=2" in script
    assert "--tensor-parallel-size 8" in script
    assert "--enable-expert-parallel" in script
    assert "--distributed-executor-backend ray" in script
    assert "/workspace/exp/run_multinode_ray.sh" in script
    assert "assert ray.__version__ == '2.48.0'" in script
    assert EXPECTED_DYNAMIC_SCHEDULE in script
    assert (
        '--speculative-config-json \'{"method":"mtp","num_speculative_tokens":5,'
        '"num_speculative_tokens_per_batch_size":'
        "[[1,4,5],[5,16,3],[17,64,2],[65,128,1],[129,512,0]]}'" in script
    )


def test_render_rejects_manifest_not_equal_to_committed_manifest() -> None:
    submit = load_submit()
    row = submit.build_matrix_rows(manifest_path=MANIFEST_PATH)[0]
    manifest = load_manifest()
    manifest["base_commit"] = "0" * 40

    with pytest.raises(ValueError, match="committed manifest"):
        submit.render_cell_sbatch(
            row,
            experiment_dir=PACKAGE_ROOT,
            result_dir=Path("/lustre/results/cell"),
            manifest=manifest,
        )


@pytest.mark.parametrize(
    ("gate", "expected_count"),
    [("edge", 16), ("remaining", 48), ("full", 64)],
)
def test_cli_only_renders_expected_number_of_unique_sbatch_files(
    tmp_path: Path, gate: str, expected_count: int
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SUBMIT_PATH),
            "--gate",
            gate,
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rendered = sorted(tmp_path.glob("*.sbatch"))
    assert len(rendered) == expected_count
    assert len({path.name for path in rendered}) == expected_count
    assert len(completed.stdout.strip().splitlines()) == expected_count
    assert all("\nsbatch " not in path.read_text(encoding="utf-8") for path in rendered)


def test_render_refuses_stale_sbatch_files(tmp_path: Path) -> None:
    submit = load_submit()
    stale = tmp_path / "stale.sbatch"
    stale.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="stale sbatch"):
        submit.render_matrix(gate="edge", output_dir=tmp_path)


def test_invalid_gate_fails_closed() -> None:
    submit = load_submit()

    with pytest.raises(ValueError, match="gate"):
        submit.select_matrix_rows([], gate="submit")


def test_dynamic_bs512_k0_rejects_hidden_draft_work() -> None:
    spec = importlib.util.spec_from_file_location(
        "test_mrv2_patch_matrix_results", PACKAGE_ROOT / "results.py"
    )
    assert spec is not None and spec.loader is not None
    results = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(results)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "config": {
            "method_key": "mtp_dynamic_max_k5",
            "effective_k": None,
            "requested_batch_schedule_k": 0,
        },
        "summary": {
            "tokens_ok": True,
            "spec_decode_metrics": {
                "num_draft_tokens": 1,
                "num_accepted_tokens": 0,
                "acceptance_rate": 0.0,
            },
        },
    }

    with pytest.raises(ValueError, match="K=0"):
        results.validate_result_payload(payload)
