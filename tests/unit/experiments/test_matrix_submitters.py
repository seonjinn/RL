from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
BASELINE = "scopes/00_baseline_no_cg.sh"
NANO_PERFORMANCE_SCOPES = (
    BASELINE,
    "scopes/17_attn.sh",
    "scopes/09_mlp.sh",
    "scopes/05_mamba.sh",
    "scopes/03_moe_router.sh",
    "scopes/31_attn_mlp_mamba_moe_router.sh",
)
SUPER_PERFORMANCE_SCOPES = (
    BASELINE,
    "scopes/17_attn.sh",
    "scopes/09_mlp.sh",
    "scopes/05_mamba.sh",
    "scopes/03_moe_router.sh",
    "scopes/04_moe_router_preprocess.sh",
    "scopes/32_attn_mlp_mamba_moe_router_preprocess.sh",
)
QWEN_PERFORMANCE_SCOPES = (
    BASELINE,
    "scopes/17_attn.sh",
    "scopes/03_moe_router.sh",
    "scopes/04_moe_router_preprocess.sh",
    "scopes/20_attn_moe_router_preprocess.sh",
)


def _write_launcher(path: Path, relative_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\t%s\\t%s\\t%s\\t%s\\n' "
        f'\'{relative_path}\' "${{STEPS}}" "${{RUN_GROUP:-}}" '
        '"${REPEAT_INDEX:-}" "${RUN_TAG}" >>"${CAPTURE_FILE}"\n'
    )
    path.chmod(0o755)


def _make_harness(
    tmp_path: Path, submitter: str, launchers: tuple[str, ...]
) -> tuple[Path, Path]:
    harness = tmp_path / "experiment"
    harness.mkdir()
    (harness / submitter).symlink_to(EXPERIMENT_DIR / submitter)
    capture_file = tmp_path / "captured.tsv"
    for relative_path in launchers:
        _write_launcher(harness / relative_path, relative_path)
    return harness, capture_file


def _run_submitter(
    harness: Path,
    submitter: str,
    capture_file: Path,
    *,
    model: str = "nano",
    arguments: tuple[str, ...] = (),
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (
        "ACCURACY_SCRIPT",
        "PERFORMANCE_SCRIPTS",
        "RUN_GROUP",
        "REPEAT_INDEX",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "CAPTURE_FILE": str(capture_file),
            "CLUSTER": "oci-hsg",
            "MODEL": model,
            "MODE": "nemorl",
            "RUN_TAG": "unit",
            "TEST_ONLY": "1",
        }
    )
    environment.update(extra_environment or {})
    return subprocess.run(
        ["/bin/bash", str(harness / submitter), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _captured_rows(capture_file: Path) -> list[tuple[str, str, str, str, str]]:
    if not capture_file.exists():
        return []
    rows = []
    for line in capture_file.read_text().splitlines():
        fields = line.split("\t")
        assert len(fields) == 5
        rows.append((fields[0], fields[1], fields[2], fields[3], fields[4]))
    return rows


@pytest.mark.parametrize(
    ("submitter", "selector_name"),
    (
        ("submit_performance_matrix.sh", "argument"),
        ("submit_accuracy_soak.sh", "environment"),
    ),
)
@pytest.mark.parametrize(
    "unsafe_path",
    (
        "scopes/../outside.sh",
        "scopes/nested/leaf.sh",
        "scopes//leaf.sh",
    ),
)
def test_selected_launcher_rejects_traversal_and_embedded_path_segments(
    tmp_path: Path, submitter: str, selector_name: str, unsafe_path: str
) -> None:
    harness, capture_file = _make_harness(tmp_path, submitter, (BASELINE,))
    _write_launcher(harness / unsafe_path, unsafe_path)
    if selector_name == "argument":
        arguments = (unsafe_path,)
        extra_environment = None
    else:
        arguments = ()
        extra_environment = {"ACCURACY_SCRIPT": unsafe_path}

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        arguments=arguments,
        extra_environment=extra_environment,
    )

    assert result.returncode == 2
    assert "single persistent" in result.stderr
    assert _captured_rows(capture_file) == []


@pytest.mark.parametrize(
    ("submitter", "arguments", "extra_environment"),
    (
        ("submit_smoke_matrix.sh", (), None),
        (
            "submit_performance_matrix.sh",
            ("scopes/17_attn.sh",),
            None,
        ),
        (
            "submit_accuracy_soak.sh",
            (),
            {"ACCURACY_SCRIPT": "scopes/17_attn.sh"},
        ),
    ),
)
def test_submitter_rejects_launcher_symlink_escape(
    tmp_path: Path,
    submitter: str,
    arguments: tuple[str, ...],
    extra_environment: dict[str, str] | None,
) -> None:
    harness, capture_file = _make_harness(tmp_path, submitter, (BASELINE,))
    outside = tmp_path / "outside.sh"
    _write_launcher(outside, "outside.sh")
    escaped_launcher = harness / "scopes" / "17_attn.sh"
    escaped_launcher.parent.mkdir(exist_ok=True)
    escaped_launcher.symlink_to(outside)

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        arguments=arguments,
        extra_environment=extra_environment,
    )

    assert result.returncode == 2
    assert "escapes" in result.stderr
    assert _captured_rows(capture_file) == []


@pytest.mark.parametrize(
    ("model", "expected_scopes"),
    (
        ("nano", NANO_PERFORMANCE_SCOPES),
        ("super", SUPER_PERFORMANCE_SCOPES),
        ("qwen3_30ba3b", QWEN_PERFORMANCE_SCOPES),
    ),
)
def test_performance_defaults_submit_three_matched_model_compatible_repeats(
    tmp_path: Path, model: str, expected_scopes: tuple[str, ...]
) -> None:
    submitter = "submit_performance_matrix.sh"
    all_launchers = tuple(
        dict.fromkeys(
            (*NANO_PERFORMANCE_SCOPES, *SUPER_PERFORMANCE_SCOPES, *expected_scopes)
        )
    )
    harness, capture_file = _make_harness(tmp_path, submitter, all_launchers)

    result = _run_submitter(harness, submitter, capture_file, model=model)

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    assert len(rows) == 3 * len(expected_scopes)
    assert {row[2] for row in rows} == {f"performance-{model}-nemorl-oci-hsg-unit"}
    for repeat_index in (1, 2, 3):
        repeat_rows = [row for row in rows if row[3] == str(repeat_index)]
        assert tuple(row[0] for row in repeat_rows) == expected_scopes
        assert {row[1] for row in repeat_rows} == {"20"}
        assert {row[4] for row in repeat_rows} == {f"unit-r{repeat_index}"}
        assert sum(row[0] == BASELINE for row in repeat_rows) == 1


def test_performance_custom_selection_deduplicates_baseline_per_repeat(
    tmp_path: Path,
) -> None:
    submitter = "submit_performance_matrix.sh"
    selected = (
        BASELINE,
        BASELINE,
        "scopes/17_attn.sh",
        "scopes/17_attn.sh",
    )
    harness, capture_file = _make_harness(
        tmp_path, submitter, (BASELINE, "scopes/17_attn.sh")
    )

    result = _run_submitter(harness, submitter, capture_file, arguments=selected)

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    for repeat_index in (1, 2, 3):
        repeat_scopes = tuple(row[0] for row in rows if row[3] == str(repeat_index))
        assert repeat_scopes == (BASELINE, "scopes/17_attn.sh")


@pytest.mark.parametrize(
    ("model", "best_combined"),
    (
        ("nano", "scopes/31_attn_mlp_mamba_moe_router.sh"),
        ("super", "scopes/32_attn_mlp_mamba_moe_router_preprocess.sh"),
        ("qwen3_30ba3b", "scopes/20_attn_moe_router_preprocess.sh"),
    ),
)
def test_accuracy_defaults_pair_baseline_and_best_combined_for_three_repeats(
    tmp_path: Path, model: str, best_combined: str
) -> None:
    submitter = "submit_accuracy_soak.sh"
    harness, capture_file = _make_harness(
        tmp_path, submitter, (BASELINE, best_combined)
    )

    result = _run_submitter(harness, submitter, capture_file, model=model)

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    assert len(rows) == 6
    assert {row[2] for row in rows} == {f"accuracy-{model}-nemorl-oci-hsg-unit"}
    for repeat_index in (1, 2, 3):
        repeat_rows = [row for row in rows if row[3] == str(repeat_index)]
        assert tuple(row[0] for row in repeat_rows) == (BASELINE, best_combined)
        assert {row[1] for row in repeat_rows} == {"100"}
        assert {row[4] for row in repeat_rows} == {f"unit-r{repeat_index}"}


def test_smoke_keeps_five_step_batch_rows_without_repeats(tmp_path: Path) -> None:
    submitter = "submit_smoke_matrix.sh"
    launchers = (BASELINE, "scopes/17_attn.sh")
    harness, capture_file = _make_harness(tmp_path, submitter, launchers)

    result = _run_submitter(harness, submitter, capture_file)

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    assert tuple(row[0] for row in rows) == launchers
    assert {row[1] for row in rows} == {"5"}
