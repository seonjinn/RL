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

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).parents[3]
PARSER_PATH = REPO_ROOT / "experiments" / "mxfp8_adaptive_rollout" / "parse_results.py"
NEMO_COMMIT = "8" * 40
VLLM_COMMIT = "b" * 40
CONTAINER_DIGEST = "sha256:" + "c" * 64
CONFIG_HASH = "d" * 64
QUALIFIED_CONFIG_NAME = "qwen3_30ba3b_tp1_v0202_qualified.json"


def _load_parser() -> ModuleType:
    assert PARSER_PATH.is_file(), f"missing experiment parser: {PARSER_PATH}"
    spec = importlib.util.spec_from_file_location(
        "mxfp8_adaptive_rollout_results", PARSER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_log(*, arm: str = "original", repeat: int = 1) -> str:
    metadata = {
        "arm": arm,
        "repeat": repeat,
        "vllm_commit": VLLM_COMMIT,
        "nemo_rl_commit": NEMO_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "config_hash": CONFIG_HASH if arm == "adaptive" else "none",
        "tensor_parallel_size": 1,
        "seed": 42,
        "num_samples": 8,
        "generation_num_gpus": 4,
        "warmup_steps": 1,
    }
    coverage = ""
    if arm == "adaptive":
        coverage = (
            "MXFP8_TACTIC_COVERAGE "
            + json.dumps(
                {
                    "fallback_record_count": 1,
                    "fallback_record_rate": 0.25,
                    "qualified_tactic_count": 2,
                    "qualified_tactics_hit": 2,
                    "runtime_record_count": 4,
                    "tactic_hit_record_count": 3,
                },
                sort_keys=True,
            )
            + "\n"
        )
    return f"""\
MXFP8_AB_METADATA {json.dumps(metadata, sort_keys=True)}
========================= Step 1/3 =========================
  • Mean Generation Length: 9.0000
  • Total step time: 12.00s
  • generation: 6.00s (50.0%)
========================= Step 2/3 =========================
  • Mean Generation Length: 12.5000
  • Total step time: 9.00s
  • generation: 4.00s (44.4%)
========================= Step 3/3 =========================
  • Mean Generation Length: 10.0000
  • Total step time: 8.00s
  • generation: 3.00s (37.5%)
MXFP8_RUN_WALL_TIME_S 21.125
{coverage}\
"""


def test_parse_log_extracts_step_metrics_and_provenance() -> None:
    parser = _load_parser()

    records = parser.parse_log(_literal_log())

    assert len(records) == 2
    first = records[0]
    assert first.step == 2
    assert first.arm == "original"
    assert first.repeat == 1
    assert first.run_wall_time_s == pytest.approx(21.125)
    assert first.generation_time_s == pytest.approx(4.0)
    assert first.total_step_time_s == pytest.approx(9.0)
    assert first.output_tokens == 100
    assert first.output_tokens_per_second_per_gpu == pytest.approx(100 / 4 / 4)
    assert first.vllm_commit == VLLM_COMMIT
    assert first.nemo_rl_commit == NEMO_COMMIT
    assert first.container_digest == CONTAINER_DIGEST
    assert first.config_hash == "none"
    assert first.tensor_parallel_size == 1
    assert first.seed == 42

    second = records[1]
    assert second.step == 3
    assert second.run_wall_time_s == pytest.approx(21.125)
    assert second.generation_time_s == pytest.approx(3.0)
    assert second.output_tokens == 80


def test_parse_log_rejects_missing_independent_run_wall_measurement() -> None:
    parser = _load_parser()
    log = _literal_log().replace("MXFP8_RUN_WALL_TIME_S 21.125\n", "")

    with pytest.raises(ValueError, match="run wall"):
        parser.parse_log(log)


def test_parse_log_rejects_adaptive_result_without_runtime_tactic_hits() -> None:
    parser = _load_parser()
    log = "\n".join(
        line
        for line in _literal_log(arm="adaptive").splitlines()
        if not line.startswith("MXFP8_TACTIC_COVERAGE ")
    )

    with pytest.raises(ValueError, match="tactic coverage"):
        parser.parse_log(log)


def test_parse_log_rejects_missing_required_provenance() -> None:
    parser = _load_parser()
    metadata = {
        "arm": "original",
        "repeat": 1,
        "vllm_commit": VLLM_COMMIT,
        "nemo_rl_commit": NEMO_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "tensor_parallel_size": 1,
        "seed": 42,
        "num_samples": 8,
        "generation_num_gpus": 4,
        "warmup_steps": 0,
    }
    log = f"""\
MXFP8_AB_METADATA {json.dumps(metadata, sort_keys=True)}
========================= Step 1/1 =========================
  • Mean Generation Length: 12.5000
  • Total step time: 9.00s
  • generation: 4.00s (44.4%)
"""

    with pytest.raises(ValueError, match="config_hash"):
        parser.parse_log(log)


def test_validate_ab_pair_allows_only_the_json_environment_key_to_differ() -> None:
    parser = _load_parser()
    common = {
        "nemo_rl_commit": NEMO_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "checkpoint": "Qwen/Qwen3-30B-A3B",
        "topology": {
            "num_nodes": 4,
            "gpus_per_node": 4,
            "tensor_parallel_size": 1,
        },
    }
    original = {
        **common,
        "config_hash": "none",
        "resolved_config": {
            "grpo": {"seed": 42},
            "policy": {
                "generation": {
                    "vllm_cfg": {
                        "precision": "fp8",
                        "is_mx": True,
                        "env_vars": {"KEEP_ME": "same"},
                    }
                }
            },
        },
    }
    adaptive = {
        **common,
        "config_hash": CONFIG_HASH,
        "resolved_config": {
            "grpo": {"seed": 42},
            "policy": {
                "generation": {
                    "vllm_cfg": {
                        "precision": "fp8",
                        "is_mx": True,
                        "env_vars": {
                            "KEEP_ME": "same",
                            "VLLM_MXFP8_DENSE_CONFIG_FILE": QUALIFIED_CONFIG_NAME,
                        },
                    }
                }
            },
        },
    }

    parser.validate_ab_pair(
        original,
        adaptive,
        expected_config_file=QUALIFIED_CONFIG_NAME,
        expected_config_sha256=CONFIG_HASH,
    )

    adaptive["resolved_config"]["grpo"]["seed"] = 43
    with pytest.raises(ValueError, match="resolved Hydra config"):
        parser.validate_ab_pair(
            original,
            adaptive,
            expected_config_file=QUALIFIED_CONFIG_NAME,
            expected_config_sha256=CONFIG_HASH,
        )


def test_validate_ab_pair_rejects_original_with_adaptive_key() -> None:
    parser = _load_parser()
    base_config = {
        "policy": {
            "generation": {
                "vllm_cfg": {
                    "env_vars": {"VLLM_MXFP8_DENSE_CONFIG_FILE": QUALIFIED_CONFIG_NAME}
                }
            }
        }
    }
    common = {
        "nemo_rl_commit": NEMO_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "checkpoint": "Qwen/Qwen3-30B-A3B",
        "topology": {"num_nodes": 4, "gpus_per_node": 4},
    }
    original = {**common, "config_hash": "none", "resolved_config": base_config}
    adaptive = {
        **common,
        "config_hash": CONFIG_HASH,
        "resolved_config": base_config,
    }

    with pytest.raises(ValueError, match="original.*absent"):
        parser.validate_ab_pair(
            original,
            adaptive,
            expected_config_file=QUALIFIED_CONFIG_NAME,
            expected_config_sha256=CONFIG_HASH,
        )


def test_validate_ab_pair_binds_adaptive_key_to_expected_file_and_hash() -> None:
    parser = _load_parser()
    common = {
        "nemo_rl_commit": NEMO_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "checkpoint": "Qwen/Qwen3-30B-A3B",
        "topology": {"num_nodes": 4, "gpus_per_node": 4},
    }
    original = {
        **common,
        "config_hash": "none",
        "resolved_config": {"policy": {"generation": {"vllm_cfg": {}}}},
    }
    adaptive = {
        **common,
        "config_hash": "e" * 64,
        "resolved_config": {
            "policy": {
                "generation": {
                    "vllm_cfg": {
                        "env_vars": {"VLLM_MXFP8_DENSE_CONFIG_FILE": "wrong.json"}
                    }
                }
            }
        },
    }

    with pytest.raises(ValueError, match="adaptive.*exact"):
        parser.validate_ab_pair(
            original,
            adaptive,
            expected_config_file=QUALIFIED_CONFIG_NAME,
            expected_config_sha256=CONFIG_HASH,
        )


def test_validate_ab_pair_rejects_provenance_mismatch() -> None:
    parser = _load_parser()
    original = {
        "nemo_rl_commit": NEMO_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "checkpoint": "Qwen/Qwen3-30B-A3B",
        "topology": {"num_nodes": 4, "gpus_per_node": 4},
        "config_hash": "none",
        "resolved_config": {"grpo": {"seed": 42}},
    }
    adaptive = {
        **original,
        "container_digest": "sha256:" + "e" * 64,
        "config_hash": CONFIG_HASH,
    }

    with pytest.raises(ValueError, match="container_digest"):
        parser.validate_ab_pair(
            original,
            adaptive,
            expected_config_file=QUALIFIED_CONFIG_NAME,
            expected_config_sha256=CONFIG_HASH,
        )


def test_write_summaries_is_stable_across_input_order(tmp_path: Path) -> None:
    parser = _load_parser()
    original = parser.parse_log(_literal_log(arm="original", repeat=1))[0]
    adaptive = parser.parse_log(_literal_log(arm="adaptive", repeat=1))[0]
    first_json_path = tmp_path / "first-summary.json"
    first_csv_path = tmp_path / "first-summary.csv"
    second_json_path = tmp_path / "second-summary.json"
    second_csv_path = tmp_path / "second-summary.csv"

    parser.write_summaries([adaptive, original], first_json_path, first_csv_path)
    first_json = first_json_path.read_bytes()
    first_csv = first_csv_path.read_bytes()
    parser.write_summaries([original, adaptive], second_json_path, second_csv_path)

    assert second_json_path.read_bytes() == first_json
    assert second_csv_path.read_bytes() == first_csv
    assert [row["arm"] for row in json.loads(first_json)] == [
        "original",
        "adaptive",
    ]
    assert first_json.endswith(b"\n")
    assert first_csv.startswith(b"step,arm,repeat,run_wall_time_s,generation_time_s,")
    assert first_csv.endswith(b"\n")


def test_write_summaries_refuses_to_overwrite_existing_output(
    tmp_path: Path,
) -> None:
    parser = _load_parser()
    record = parser.parse_log(_literal_log())[0]
    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"
    json_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        parser.write_summaries([record], json_path, csv_path)

    assert json_path.read_text(encoding="utf-8") == "existing\n"
    assert not csv_path.exists()


def test_not_applicable_result_names_zero_hit_and_ultra_tp4_fallback() -> None:
    parser = _load_parser()

    result = parser.not_applicable_result(
        "trace files contain zero eligible dense MXFP8 records"
    )

    assert result == {
        "fallback": {
            "model": "Nemotron 3 Ultra",
            "tensor_parallel_size": 4,
        },
        "reason": "trace files contain zero eligible dense MXFP8 records",
        "status": "not-applicable",
        "workload": "Qwen/Qwen3-30B-A3B",
    }


def _qualified_manifest(
    *entries: tuple[str, int, int, int, int],
) -> dict[str, object]:
    tactics: dict[str, list[dict[str, int]]] = {"8x4": [], "128x4": []}
    for layout, m, n, k, tactic in entries:
        tactics[layout].append({"m": m, "n": n, "k": k, "tactic": tactic})
    return {"tactics": tactics}


def test_validate_qualified_manifest_rejects_zero_promoted_tactics() -> None:
    parser = _load_parser()

    with pytest.raises(ValueError, match="zero promoted tactics"):
        parser.validate_qualified_manifest(_qualified_manifest())


def test_validate_runtime_tactic_coverage_rejects_all_fallback(
    tmp_path: Path,
) -> None:
    parser = _load_parser()
    manifest = _qualified_manifest(("8x4", 8, 2048, 8192, 17))
    trace_path = tmp_path / "adaptive_dispatch.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "event": "mxfp8_adaptive_dispatch",
                "config_sha256": CONFIG_HASH,
                "layout": "128x4",
                "m": 16,
                "n": 2048,
                "k": 8192,
                "tactic": -1,
                "tactic_source": "runner_default",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="zero runtime tactic-hit records"):
        parser.validate_runtime_tactic_coverage(
            manifest,
            (trace_path,),
            expected_config_sha256=CONFIG_HASH,
        )


def test_validate_runtime_tactic_coverage_requires_every_promoted_shape(
    tmp_path: Path,
) -> None:
    parser = _load_parser()
    manifest = _qualified_manifest(
        ("8x4", 8, 2048, 8192, 17),
        ("128x4", 16, 2048, 8192, 23),
    )
    trace_path = tmp_path / "adaptive_dispatch.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "event": "mxfp8_adaptive_dispatch",
                "config_sha256": CONFIG_HASH,
                "layout": "8x4",
                "m": 8,
                "n": 2048,
                "k": 8192,
                "tactic": 17,
                "tactic_source": "static_hint",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not hit at runtime"):
        parser.validate_runtime_tactic_coverage(
            manifest,
            (trace_path,),
            expected_config_sha256=CONFIG_HASH,
        )


def test_validate_runtime_tactic_coverage_reports_unseen_fallback_rate(
    tmp_path: Path,
) -> None:
    parser = _load_parser()
    manifest = _qualified_manifest(("8x4", 8, 2048, 8192, 17))
    trace_path = tmp_path / "adaptive_dispatch.jsonl"
    rows = [
        {
            "event": "mxfp8_adaptive_dispatch",
            "config_sha256": CONFIG_HASH,
            "layout": "8x4",
            "m": 8,
            "n": 2048,
            "k": 8192,
            "tactic": 17,
            "tactic_source": "static_hint",
        },
        {
            "event": "mxfp8_adaptive_dispatch",
            "config_sha256": CONFIG_HASH,
            "layout": "128x4",
            "m": 16,
            "n": 2048,
            "k": 8192,
            "tactic": -1,
            "tactic_source": "runner_default",
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert parser.validate_runtime_tactic_coverage(
        manifest,
        (trace_path,),
        expected_config_sha256=CONFIG_HASH,
    ) == {
        "fallback_record_count": 1,
        "fallback_record_rate": 0.5,
        "qualified_tactic_count": 1,
        "qualified_tactics_hit": 1,
        "runtime_record_count": 2,
        "tactic_hit_record_count": 1,
    }


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _run_spool_launcher(
    tmp_path: Path,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    spool_dir = tmp_path / "slurm-spool"
    spool_dir.mkdir()
    launcher = spool_dir / "slurm_script"
    shutil.copy2(
        REPO_ROOT / "experiments" / "mxfp8_adaptive_rollout" / "run_ab.sh",
        launcher,
    )

    container = tmp_path / "runtime.sqsh"
    container.touch()
    experiment_root = tmp_path / "experiment-output"
    profile = tmp_path / "profile.env"
    profile.write_text(
        "\n".join(
            (
                'export SLURM_ACCOUNT="test-account"',
                'export PARTITION="batch"',
                'export QOS="normal"',
                'export NUM_NODES="4"',
                'export GPUS_PER_NODE="4"',
                'export SLURM_SWITCHES="1@600"',
                'export WALLTIME="00:10:00"',
                f'export NEMO_RL_REPO_ROOT="{REPO_ROOT}"',
                f'export NEMO_RL_EXPERIMENT_ROOT="{experiment_root}"',
                f'export CONTAINER_IMAGE="{container}"',
                f'export CONTAINER_MOUNTS="{REPO_ROOT}:{REPO_ROOT},{tmp_path}:{tmp_path}"',
                f'export HF_HOME="{tmp_path / "hf"}"',
                f'export CACHE_ROOT="{tmp_path / "cache"}"',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "sbatch.calls"
    _write_executable(
        fake_bin / "git",
        f"""#!/bin/bash
if [[ "$*" == *"rev-parse HEAD"* ]]; then
  echo "{NEMO_COMMIT}"
fi
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
printf '%s\\n' "$*" >>"$SBATCH_CAPTURE"
echo "sbatch preflight accepted"
""",
    )
    environment = {
        **os.environ,
        "ACTION": "test-only",
        "NEMO_RL_REPO_ROOT": str(REPO_ROOT),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REPEATS": "3",
        "SBATCH_CAPTURE": str(capture),
        "SUITE_ID": "spool-harness",
    }
    result = subprocess.run(
        ["bash", str(launcher), "ab", str(profile)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    calls = capture.read_text(encoding="utf-8").splitlines() if capture.exists() else []
    return result, calls


def test_submitted_spool_copy_exports_canonical_shared_paths(
    tmp_path: Path,
) -> None:
    result, calls = _run_spool_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    checked_in_launcher = (
        REPO_ROOT / "experiments" / "mxfp8_adaptive_rollout" / "run_ab.sh"
    )
    assert calls
    assert all(f"NEMO_RL_REPO_ROOT={REPO_ROOT}" in call for call in calls)
    assert all("NEMO_RL_EXPERIMENT_ROOT=" in call for call in calls)
    assert all(call.endswith(str(checked_in_launcher)) for call in calls)


def test_ab_schedule_uses_same_job_warmup_and_three_alternating_repeats(
    tmp_path: Path,
) -> None:
    result, calls = _run_spool_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    assert len(calls) == 6
    job_names = [
        next(
            argument for argument in call.split() if argument.startswith("--job-name=")
        )
        for call in calls
    ]
    assert job_names == [
        "--job-name=mxfp8-original-1",
        "--job-name=mxfp8-adaptive-1",
        "--job-name=mxfp8-original-2",
        "--job-name=mxfp8-adaptive-2",
        "--job-name=mxfp8-original-3",
        "--job-name=mxfp8-adaptive-3",
    ]
    assert all("WARMUP=" not in call for call in calls)


@pytest.mark.parametrize("source_commit", [None, "", "abc123"])
def test_stage_script_requires_full_source_commit(
    tmp_path: Path, source_commit: str | None
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    enroot_marker = tmp_path / "enroot-invoked"
    _write_executable(
        fake_bin / "enroot",
        """#!/bin/bash
touch "$ENROOT_MARKER"
touch "$3"
""",
    )
    environment = {
        **os.environ,
        "CONTAINER_DIR": str(tmp_path / "containers"),
        "ENROOT_MARKER": str(enroot_marker),
        "OUTPUT_PREFIX": "test-image",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SLURM_JOB_ID": "123",
        "SOURCE_IMAGE": "registry.example.com/team/image:immutable",
    }
    if source_commit is not None:
        environment["SOURCE_COMMIT"] = source_commit

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "stage_enroot_image.sbatch")],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "SOURCE_COMMIT" in result.stderr
    assert not enroot_marker.exists()
