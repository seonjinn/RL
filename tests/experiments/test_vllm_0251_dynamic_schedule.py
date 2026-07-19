import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.vllm_0251_drafter_matrix.matrix import (
    build_runtime_command,
    load_dynamic_schedule,
    resolve_run,
)


G_TARGET_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
G_DRAFTER_REVISION = "a1403e07b73a66fc9ef561463631c31864616933"
G_PROFILE_SHA256 = "efcd9ad3f74ecb260ab7a580a062e56266b67196fd16d90b792c4176a25e5f69"
G_REPO_ROOT = Path(__file__).resolve().parents[2]


def _schedule_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "calibration_status": "seed",
        "model_key": "qwen32",
        "target_revision": G_TARGET_REVISION,
        "drafter_revision": G_DRAFTER_REVISION,
        "source_runtime_vllm": "0.24.0",
        "target_runtime_vllm": "0.25.1",
        "target_cuda_graph_mode": "FULL_AND_PIECEWISE",
        "profile_sha256": G_PROFILE_SHA256,
        "ranges": [[1, 127, 3], [128, 256, 1]],
    }


def _schedule_v2_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "calibration_status": "calibrated",
        "model_key": "qwen32",
        "target_revision": G_TARGET_REVISION,
        "drafter_revision": G_DRAFTER_REVISION,
        "source_runtime_vllm": "0.25.1",
        "target_runtime_vllm": "0.25.1",
        "target_cuda_graph_mode": "FULL_AND_PIECEWISE",
        "profile_sha256": G_PROFILE_SHA256,
        "max_num_speculative_tokens": 5,
        "selection_metric": "accepted_length_over_median_itl",
        "minimum_goodput_gain": 0.0,
        "ranges": [[1, 31, 5], [32, 127, 3], [128, 256, 1]],
    }


def _write_schedule(tmp_path: Path, **overrides: object) -> Path:
    payload = _schedule_payload()
    payload.update(overrides)
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_schedule_v2(tmp_path: Path, **overrides: object) -> Path:
    payload = _schedule_v2_payload()
    payload.update(overrides)
    path = tmp_path / "schedule-v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_seed_schedule_resolves_for_smoke_with_exact_dynamic_overrides(
    tmp_path: Path,
) -> None:
    schedule = load_dynamic_schedule(_write_schedule(tmp_path))
    run = resolve_run(
        "qwen32",
        "eagle3_thinking_dynamic_k123",
        "smoke2",
        "lyris",
        dynamic_schedule=schedule,
    )

    assert schedule.vllm_ranges() == ((1, 127, 3), (128, 256, 1))
    assert schedule.source_sha256
    assert (
        "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3"
    ) in run.hydra_overrides
    assert (
        "++policy.generation.vllm_kwargs.speculative_config."
        "num_speculative_tokens_per_batch_size=[[1,127,3],[128,256,1]]"
    ) in run.hydra_overrides


def test_calibrated_k5_schedule_resolves_without_silent_clamping(
    tmp_path: Path,
) -> None:
    schedule = load_dynamic_schedule(_write_schedule_v2(tmp_path))
    run = resolve_run(
        "qwen32",
        "eagle3_thinking_dynamic_k5",
        "smoke2",
        "lyris",
        dynamic_schedule=schedule,
    )

    assert schedule.max_num_speculative_tokens == 5
    assert schedule.vllm_ranges()[0] == (1, 31, 5)
    assert (
        "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5"
    ) in run.hydra_overrides
    assert (
        "++policy.generation.vllm_kwargs.speculative_config."
        "num_speculative_tokens_per_batch_size="
        "[[1,31,5],[32,127,3],[128,256,1]]"
    ) in run.hydra_overrides


def test_schedule_max_k_must_match_dynamic_variant(tmp_path: Path) -> None:
    schedule = load_dynamic_schedule(_write_schedule_v2(tmp_path))

    with pytest.raises(ValueError, match="maximum K"):
        resolve_run(
            "qwen32",
            "eagle3_thinking_dynamic_k123",
            "smoke2",
            "lyris",
            dynamic_schedule=schedule,
        )


def test_schema_v2_rejects_invalid_selection_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selection_metric"):
        load_dynamic_schedule(
            _write_schedule_v2(tmp_path, selection_metric="throughput")
        )
    with pytest.raises(ValueError, match="minimum_goodput_gain"):
        load_dynamic_schedule(_write_schedule_v2(tmp_path, minimum_goodput_gain=-0.1))


def test_dynamic_runtime_applies_only_the_run_scoped_cuda_graph_patch(
    tmp_path: Path,
) -> None:
    schedule = load_dynamic_schedule(_write_schedule(tmp_path))
    dynamic = resolve_run(
        "qwen32",
        "eagle3_thinking_dynamic_k123",
        "smoke2",
        "lyris",
        dynamic_schedule=schedule,
    )
    fixed = resolve_run("qwen32", "eagle3_thinking_k3", "smoke2", "lyris")

    dynamic_command = build_runtime_command(
        dynamic, tmp_path / "repo", tmp_path / "dynamic", "dynamic"
    )
    fixed_command = build_runtime_command(
        fixed, tmp_path / "repo", tmp_path / "fixed", "fixed"
    )
    final_command = build_runtime_command(
        replace(dynamic, phase=replace(dynamic.phase, key="final20")),
        tmp_path / "repo",
        tmp_path / "dynamic-final",
        "dynamic-final",
    )
    patch = (
        tmp_path / "repo" / "experiments/vllm_0251_eagle3_perfcfg/"
        "apply_vllm0251_dynamic_sd_cg_fix.py"
    )

    assert f"NRL_VENV_POST_SYNC_SCRIPT={patch}" in dynamic_command
    assert (
        "NRL_VENV_POST_SYNC_TARGET="
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
    ) in dynamic_command
    assert "NRL_VLLM_DYNAMIC_SD_SMOKE_TELEMETRY=1" in dynamic_command
    assert "NRL_VLLM_DYNAMIC_SD_SMOKE_TELEMETRY=1" in final_command
    assert not any(item.startswith("NRL_VENV_POST_SYNC_") for item in fixed_command)
    assert "NRL_VLLM_DYNAMIC_SD_SMOKE_TELEMETRY=1" not in fixed_command
    assert not any("cudagraph_capture_sizes" in item for item in dynamic_command)


def test_qwen235_dynamic_runtime_patches_the_async_generation_worker() -> None:
    schedule = load_dynamic_schedule(
        G_REPO_ROOT / "experiments/vllm_0251_drafter_matrix/calibration/"
        "qwen235_thinking_k5_vllm0251_schedule.json"
    )
    run = resolve_run(
        "qwen235",
        "eagle3_thinking_dynamic_k5_cg384",
        "smoke2",
        "lyris",
        dynamic_schedule=schedule,
        optimizer_offload_mode="coalesced-pinned",
    )

    command = build_runtime_command(
        run,
        G_REPO_ROOT,
        Path("/tmp/qwen235-dynamic"),
        "qwen235-dynamic",
    )

    assert (
        "NRL_VENV_POST_SYNC_TARGET="
        "nemo_rl.models.generation.vllm.vllm_worker_async."
        "VllmAsyncGenerationWorker"
    ) in command


@pytest.mark.parametrize("phase", ["smoke2", "smoke5"])
def test_seed_schedule_is_limited_to_smoke_phases(tmp_path: Path, phase: str) -> None:
    schedule = load_dynamic_schedule(_write_schedule(tmp_path))

    resolve_run(
        "qwen32",
        "eagle3_thinking_dynamic_k123",
        phase,
        "lyris",
        dynamic_schedule=schedule,
    )


def test_seed_schedule_is_rejected_for_final20(tmp_path: Path) -> None:
    schedule = load_dynamic_schedule(_write_schedule(tmp_path))

    with pytest.raises(ValueError, match="calibrated"):
        resolve_run(
            "qwen32",
            "eagle3_thinking_dynamic_k123",
            "final20",
            "lyris",
            dynamic_schedule=schedule,
        )


def test_final20_requires_a_matched_vllm0251_source_profile(tmp_path: Path) -> None:
    schedule = load_dynamic_schedule(
        _write_schedule(tmp_path, calibration_status="calibrated")
    )

    with pytest.raises(ValueError, match="source profile.*0.25.1"):
        resolve_run(
            "qwen32",
            "eagle3_thinking_dynamic_k123",
            "final20",
            "lyris",
            dynamic_schedule=schedule,
        )


def test_final20_requires_an_allowlisted_schedule_artifact(tmp_path: Path) -> None:
    schedule = load_dynamic_schedule(_write_schedule_v2(tmp_path))

    with pytest.raises(ValueError, match="approved calibration artifact"):
        resolve_run(
            "qwen32",
            "eagle3_thinking_dynamic_k5",
            "final20",
            "lyris",
            dynamic_schedule=schedule,
        )


def test_checked_in_calibrated_schedule_is_approved_for_final20() -> None:
    schedule = load_dynamic_schedule(
        G_REPO_ROOT / "experiments/vllm_0251_drafter_matrix/calibration/"
        "qwen32_thinking_k5_vllm0251_schedule.json"
    )

    run = resolve_run(
        "qwen32",
        "eagle3_thinking_dynamic_k5",
        "final20",
        "lyris",
        dynamic_schedule=schedule,
    )

    assert "grpo.max_num_steps=20" in run.hydra_overrides


def test_checked_in_qwen235_calibrated_schedule_is_approved_for_final20() -> None:
    schedule = load_dynamic_schedule(
        G_REPO_ROOT / "experiments/vllm_0251_drafter_matrix/calibration/"
        "qwen235_thinking_k5_vllm0251_schedule.json"
    )

    run = resolve_run(
        "qwen235",
        "eagle3_thinking_dynamic_k5_cg384",
        "final20",
        "lyris",
        dynamic_schedule=schedule,
        optimizer_offload_mode="coalesced-pinned",
    )

    assert "grpo.max_num_steps=20" in run.hydra_overrides
    assert "++policy.use_pinned_optimizer_offload=true" in run.hydra_overrides


def test_final20_requires_schema_v2_even_for_a_matched_v1_profile(
    tmp_path: Path,
) -> None:
    schedule = load_dynamic_schedule(
        _write_schedule(
            tmp_path,
            calibration_status="calibrated",
            source_runtime_vllm="0.25.1",
        )
    )

    with pytest.raises(ValueError, match="schema version 2"):
        resolve_run(
            "qwen32",
            "eagle3_thinking_dynamic_k123",
            "final20",
            "lyris",
            dynamic_schedule=schedule,
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"calibration_status": "unknown"}, "calibration_status"),
        ({"schema_version": True}, "schema_version"),
        ({"ranges": [[2, 256, 3]]}, "batch size 1"),
        ({"ranges": [[1, 127, 3], [129, 256, 1]]}, "contiguous"),
        ({"ranges": [[1, 128, 3], [128, 256, 1]]}, "contiguous"),
        ({"ranges": [[1, 256, 4]]}, "between 0 and 3"),
        ({"ranges": [[1, 127, 2], [128, 256, 1]]}, "maximum K"),
        ({"target_revision": "a" * 41}, "target_revision"),
        ({"profile_sha256": "a" * 40}, "profile_sha256"),
    ],
)
def test_invalid_schedule_shapes_fail_closed(
    tmp_path: Path, overrides: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        load_dynamic_schedule(_write_schedule(tmp_path, **overrides))


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"model_key": "qwen30"}, "model"),
        ({"target_revision": "0" * 40}, "target revision"),
        ({"drafter_revision": "0" * 40}, "drafter revision"),
        ({"target_runtime_vllm": "0.24.0"}, "vLLM"),
        ({"target_cuda_graph_mode": "PIECEWISE"}, "CUDA Graph"),
    ],
)
def test_schedule_identity_must_match_the_resolved_run(
    tmp_path: Path, overrides: dict[str, object], error: str
) -> None:
    schedule = load_dynamic_schedule(_write_schedule(tmp_path, **overrides))

    with pytest.raises(ValueError, match=error):
        resolve_run(
            "qwen32",
            "eagle3_thinking_dynamic_k123",
            "smoke2",
            "lyris",
            dynamic_schedule=schedule,
        )


def test_qwen32_recipe_rejects_schedule_without_profiled_batch_256(
    tmp_path: Path,
) -> None:
    schedule = load_dynamic_schedule(
        _write_schedule(tmp_path, ranges=[[1, 127, 3], [128, 255, 1]])
    )

    with pytest.raises(ValueError, match="profiled batch size 256"):
        resolve_run(
            "qwen32",
            "eagle3_thinking_dynamic_k123",
            "smoke2",
            "lyris",
            dynamic_schedule=schedule,
        )


def test_dynamic_and_fixed_variants_require_exact_schedule_handoff(
    tmp_path: Path,
) -> None:
    schedule = load_dynamic_schedule(_write_schedule(tmp_path))

    with pytest.raises(ValueError, match="requires.*schedule"):
        resolve_run(
            "qwen32",
            "eagle3_thinking_dynamic_k123",
            "smoke2",
            "lyris",
        )
    with pytest.raises(ValueError, match="does not accept.*schedule"):
        resolve_run(
            "qwen32",
            "eagle3_thinking_k3",
            "smoke2",
            "lyris",
            dynamic_schedule=schedule,
        )


def test_show_cli_requires_dynamic_schedule_and_rejects_it_for_fixed_k(
    tmp_path: Path,
) -> None:
    matrix_path = (
        Path(__file__).parents[2] / "experiments/vllm_0251_drafter_matrix/matrix.py"
    )
    common = (
        sys.executable,
        str(matrix_path),
        "show",
        "--model",
        "qwen32",
        "--phase",
        "smoke2",
        "--cluster",
        "lyris",
        "--experiment-root",
        str(tmp_path / "runs"),
    )
    missing = subprocess.run(
        (*common, "--variant", "eagle3_thinking_dynamic_k123"),
        check=False,
        capture_output=True,
        text=True,
    )
    fixed = subprocess.run(
        (
            *common,
            "--variant",
            "eagle3_thinking_k3",
            "--dynamic-schedule",
            str(_write_schedule(tmp_path)),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert missing.returncode != 0
    assert "requires" in missing.stderr
    assert fixed.returncode != 0
    assert "does not accept" in fixed.stderr


def test_show_cli_records_dynamic_schedule_provenance_and_fixed_k_isolation(
    tmp_path: Path,
) -> None:
    matrix_path = (
        Path(__file__).parents[2] / "experiments/vllm_0251_drafter_matrix/matrix.py"
    )
    schedule_path = _write_schedule(tmp_path)
    common = (
        sys.executable,
        str(matrix_path),
        "show",
        "--model",
        "qwen32",
        "--phase",
        "smoke2",
        "--cluster",
        "lyris",
        "--experiment-root",
        str(tmp_path / "runs"),
    )
    dynamic = subprocess.run(
        (
            *common,
            "--variant",
            "eagle3_thinking_dynamic_k123",
            "--dynamic-schedule",
            str(schedule_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    fixed = subprocess.run(
        (*common, "--variant", "eagle3_thinking_k3"),
        check=True,
        capture_output=True,
        text=True,
    )

    dynamic_payload = json.loads(dynamic.stdout)
    fixed_payload = json.loads(fixed.stdout)
    provenance = dynamic_payload["dynamic_schedule"]
    assert provenance["source_path"] == str(schedule_path.resolve())
    assert (
        provenance["source_sha256"]
        == load_dynamic_schedule(schedule_path).source_sha256
    )
    assert provenance["calibration_status"] == "seed"
    assert provenance["ranges"] == [[1, 127, 3], [128, 256, 1]]
    assert provenance["source_runtime_vllm"] == "0.24.0"
    assert provenance["target_runtime_vllm"] == "0.25.1"
    assert fixed_payload["dynamic_schedule"] is None
    assert not any(
        item.startswith("NRL_VENV_POST_SYNC_")
        for item in fixed_payload["runtime_command"]
    )
