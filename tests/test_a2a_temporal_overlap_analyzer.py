# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = (
    REPO_ROOT
    / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g/analyze_a2a_temporal_overlap.py"
)
EXPECTED_FIELDS = {
    "schema_version",
    "source_profile_sha256",
    "a2a_interval_count",
    "expert_gemm_interval_count",
    "overlap_duration_ns",
    "a2a_overlap_ratio",
    "gemm_overlap_ratio",
    "temporal_overlap_verified",
    "limitations",
}


def _create_profile(
    path: Path,
    *,
    names: dict[int, str] | None = None,
    intervals: list[tuple[int, int, int, int]] | None = None,
    extra_kernel_columns: str = "",
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            'CREATE TABLE "StringIds" (id INTEGER PRIMARY KEY, value TEXT NOT NULL)'
        )
        connection.execute(
            'CREATE TABLE "CUPTI_ACTIVITY_KIND_KERNEL" ('
            "start INTEGER NOT NULL, end INTEGER NOT NULL, "
            "deviceId INTEGER NOT NULL, shortName INTEGER, demangledName INTEGER"
            f"{extra_kernel_columns})"
        )
        connection.executemany(
            'INSERT INTO "StringIds" (id, value) VALUES (?, ?)',
            (names or {}).items(),
        )
        connection.executemany(
            'INSERT INTO "CUPTI_ACTIVITY_KIND_KERNEL" '
            "(start, end, deviceId, shortName) VALUES (?, ?, ?, ?)",
            intervals or [],
        )


def _run_analyzer(
    profile: Path,
    output: Path,
    *,
    nsys_bin: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(ANALYZER), str(profile), "--output", str(output)]
    if nsys_bin is not None:
        command.extend(["--nsys-bin", str(nsys_bin)])
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def _read_result(output: Path) -> dict[str, object]:
    return json.loads(output.read_text())


def test_analyzer_unions_intervals_and_measures_same_device_overlap(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "positive.sqlite"
    output = tmp_path / "a2a_temporal_overlap.json"
    _create_profile(
        profile,
        names={
            1: "ncclDevKernel_SendRecv_RING_SIMPLE",
            2: "BlockScaledMoEGroupedGemmGluBiasKernel_object_at_0x1",
            3: "unrelated_kernel",
        },
        intervals=[
            (0, 10, 0, 1),
            (8, 20, 0, 1),
            (5, 12, 0, 2),
            (15, 25, 0, 2),
            (0, 100, 0, 3),
        ],
    )

    result = _run_analyzer(profile, output)

    assert result.returncode == 0, result.stderr
    analysis = _read_result(output)
    assert set(analysis) == EXPECTED_FIELDS
    assert analysis["schema_version"] == 1
    assert (
        analysis["source_profile_sha256"]
        == hashlib.sha256(profile.read_bytes()).hexdigest()
    )
    assert analysis["a2a_interval_count"] == 2
    assert analysis["expert_gemm_interval_count"] == 2
    assert analysis["overlap_duration_ns"] == 12
    assert analysis["a2a_overlap_ratio"] == pytest.approx(12 / 20)
    assert analysis["gemm_overlap_ratio"] == pytest.approx(12 / 17)
    assert analysis["temporal_overlap_verified"] is True
    assert all(
        isinstance(item, str) and item
        for item in analysis["limitations"]  # type: ignore[union-attr]
    )


def test_analyzer_does_not_count_cross_device_concurrency_as_overlap(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "cross-device.sqlite"
    output = tmp_path / "a2a_temporal_overlap.json"
    _create_profile(
        profile,
        names={
            1: "ncclKernel_AllToAll",
            2: "nvjet_sm100_grouped_gemm_wgrad",
        },
        intervals=[(0, 10, 0, 1), (0, 10, 1, 2)],
    )

    result = _run_analyzer(profile, output)

    assert result.returncode == 0, result.stderr
    analysis = _read_result(output)
    assert analysis["a2a_interval_count"] == 1
    assert analysis["expert_gemm_interval_count"] == 1
    assert analysis["overlap_duration_ns"] == 0
    assert analysis["a2a_overlap_ratio"] == 0.0
    assert analysis["gemm_overlap_ratio"] == 0.0
    assert analysis["temporal_overlap_verified"] is False


def test_analyzer_rejects_ambiguous_kernel_tables(tmp_path: Path) -> None:
    profile = tmp_path / "ambiguous-kernel.sqlite"
    output = tmp_path / "a2a_temporal_overlap.json"
    _create_profile(profile)
    with sqlite3.connect(profile) as connection:
        connection.execute(
            'CREATE TABLE "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL" ('
            "start INTEGER, end INTEGER, deviceId INTEGER, shortName INTEGER)"
        )

    result = _run_analyzer(profile, output)

    assert result.returncode != 0
    assert "ambiguous kernel interval tables" in result.stderr
    assert not output.exists()


def test_analyzer_rejects_ambiguous_string_id_tables(tmp_path: Path) -> None:
    profile = tmp_path / "ambiguous-strings.sqlite"
    output = tmp_path / "a2a_temporal_overlap.json"
    _create_profile(profile)
    with sqlite3.connect(profile) as connection:
        connection.execute(
            'CREATE TABLE "StringValues" (id INTEGER PRIMARY KEY, value TEXT)'
        )

    result = _run_analyzer(profile, output)

    assert result.returncode != 0
    assert "ambiguous string ID tables" in result.stderr
    assert not output.exists()


def test_analyzer_rejects_duplicate_string_id_mapping(tmp_path: Path) -> None:
    profile = tmp_path / "duplicate-string-id.sqlite"
    output = tmp_path / "a2a_temporal_overlap.json"
    with sqlite3.connect(profile) as connection:
        connection.execute('CREATE TABLE "StringIds" (id INTEGER, value TEXT)')
        connection.executemany(
            'INSERT INTO "StringIds" (id, value) VALUES (?, ?)',
            [
                (1, "ncclDevKernel_SendRecv"),
                (1, "BlockScaledMoEGroupedGemmGluBiasKernel"),
            ],
        )
        connection.execute(
            'CREATE TABLE "CUPTI_ACTIVITY_KIND_KERNEL" ('
            "start INTEGER, end INTEGER, deviceId INTEGER, shortName INTEGER)"
        )
        connection.execute(
            'INSERT INTO "CUPTI_ACTIVITY_KIND_KERNEL" VALUES (0, 10, 0, 1)'
        )

    result = _run_analyzer(profile, output)

    assert result.returncode != 0
    assert "ambiguous string ID mapping" in result.stderr
    assert not output.exists()


def test_analyzer_rejects_ambiguous_direct_and_string_id_name_columns(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "ambiguous-columns.sqlite"
    output = tmp_path / "a2a_temporal_overlap.json"
    _create_profile(profile, extra_kernel_columns=", name TEXT")

    result = _run_analyzer(profile, output)

    assert result.returncode != 0
    assert "ambiguous kernel name columns" in result.stderr
    assert not output.exists()


def test_analyzer_rejects_nonpositive_kernel_intervals(tmp_path: Path) -> None:
    profile = tmp_path / "invalid-interval.sqlite"
    output = tmp_path / "a2a_temporal_overlap.json"
    _create_profile(
        profile,
        names={1: "ncclDevKernel_SendRecv"},
        intervals=[(10, 10, 0, 1)],
    )

    result = _run_analyzer(profile, output)

    assert result.returncode != 0
    assert "non-positive kernel interval" in result.stderr
    assert not output.exists()


def _write_fake_nsys(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, shutil, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "Path(os.environ['NSYS_ARGV_LOG']).write_text(json.dumps(args))\n"
        "output = Path(args[args.index('-o') + 1])\n"
        "shutil.copyfile(os.environ['NSYS_EXPORTED_SQLITE'], output)\n"
        "if os.environ.get('NSYS_MUTATE_SOURCE') == '1':\n"
        "    with Path(args[-1]).open('ab') as stream:\n"
        "        stream.write(b'changed')\n"
    )
    path.chmod(0o755)


def test_nsys_rep_export_uses_argv_without_shell_and_temp_output_in_result_root(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "shell-injection-marker"
    profile = tmp_path / f"profile;touch {marker.name}.nsys-rep"
    profile.write_bytes(b"nsys report")
    exported = tmp_path / "export-source.sqlite"
    _create_profile(
        exported,
        names={
            1: "ncclDevKernel_SendRecv",
            2: "BlockScaledMoEGroupedGemmDgluDbiasKernel_object_at_0x2",
        },
        intervals=[(0, 10, 0, 1), (5, 15, 0, 2)],
    )
    fake_nsys = tmp_path / "fake-nsys"
    argv_log = tmp_path / "nsys-argv.json"
    _write_fake_nsys(fake_nsys)
    output = tmp_path / "a2a_temporal_overlap.json"

    result = _run_analyzer(
        profile,
        output,
        nsys_bin=fake_nsys,
        env={
            "NSYS_ARGV_LOG": str(argv_log),
            "NSYS_EXPORTED_SQLITE": str(exported),
        },
    )

    assert result.returncode == 0, result.stderr
    argv = json.loads(argv_log.read_text())
    assert argv[:4] == ["export", "-t", "sqlite", "-o"]
    assert argv[-1] == str(profile)
    assert Path(argv[4]).suffix == ".sqlite"
    assert Path(argv[4]).parent.parent == tmp_path
    assert not marker.exists()
    analysis = _read_result(output)
    assert (
        analysis["source_profile_sha256"] == hashlib.sha256(b"nsys report").hexdigest()
    )


def test_nsys_rep_export_rejects_source_mutation_during_digest_window(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "mutable.nsys-rep"
    profile.write_bytes(b"nsys report")
    exported = tmp_path / "export-source.sqlite"
    _create_profile(exported)
    fake_nsys = tmp_path / "fake-nsys"
    argv_log = tmp_path / "nsys-argv.json"
    _write_fake_nsys(fake_nsys)
    output = tmp_path / "a2a_temporal_overlap.json"

    result = _run_analyzer(
        profile,
        output,
        nsys_bin=fake_nsys,
        env={
            "NSYS_ARGV_LOG": str(argv_log),
            "NSYS_EXPORTED_SQLITE": str(exported),
            "NSYS_MUTATE_SOURCE": "1",
        },
    )

    assert result.returncode != 0
    assert "source profile changed during analysis" in result.stderr
    assert not output.exists()


def test_analyzer_rejects_symlink_profile(tmp_path: Path) -> None:
    target = tmp_path / "profile.sqlite"
    _create_profile(target)
    profile = tmp_path / "profile-link.sqlite"
    profile.symlink_to(target)
    output = tmp_path / "a2a_temporal_overlap.json"

    result = _run_analyzer(profile, output)

    assert result.returncode != 0
    assert "must not be a symbolic link" in result.stderr
    assert not output.exists()
