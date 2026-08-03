import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from experiments.mxfp8_adaptive_rollout_v0251 import summarize
from experiments.mxfp8_adaptive_rollout_v0251.generation_timing import (
    AsyncCallTimer,
    GenerationLengthAudit,
)
from experiments.mxfp8_adaptive_rollout_v0251.contract import (
    AdaptiveInputs,
    TraceInputs,
    build_arm_environment,
)
from experiments.mxfp8_adaptive_rollout_v0251.flashinfer_preflight import (
    prepare_symlink_parents,
)
from experiments.mxfp8_adaptive_rollout_v0251.runtime_overlay import (
    prepare_runtime_overlay,
)
from experiments.mxfp8_adaptive_rollout_v0251.summarize import summarize_log
from experiments.mxfp8_adaptive_rollout_v0251.shape_trace import (
    summarize_shape_trace,
)


def _write_table(path: Path) -> str:
    payload = {"schema_version": 1, "metadata": {}, "entries": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_baseline_uses_same_custom_source_without_adaptive_table(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    (runtime_root / "vllm").mkdir(parents=True)

    env = build_arm_environment("baseline", runtime_root=runtime_root)

    assert env["PYTHONPATH"] == str(runtime_root)
    assert env["VLLM_SUBPROCESS_PYTHONPATH"] == str(runtime_root)
    assert env["NEMORL_MXFP8_LINEAR_BACKEND"] == "flashinfer_cutedsl"
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE" not in env


def test_trtllm_default_cli_uses_adaptive_layout_without_offline_tactic(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    (runtime_root / "vllm").mkdir(parents=True)
    root = Path(__file__).parents[3]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.mxfp8_adaptive_rollout_v0251.contract",
            "--arm",
            "trtllm_default",
            "--runtime-root",
            str(runtime_root),
            "--layer-allowlist-b64",
            "MTI4MCw4MTkyCg==",
            "--switch-m",
            "512",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        env=os.environ | {"PYTHONPATH": str(root)},
        text=True,
    )

    assert result.returncode == 0, result.stderr
    env = dict(line.split("=", maxsplit=1) for line in result.stdout.splitlines())
    assert env["NEMORL_MXFP8_LINEAR_BACKEND"] == "flashinfer_trtllm"
    assert env["VLLM_MXFP8_DENSE_TRTLLM_LAYOUT"] == "adaptive"
    assert env["VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M"] == "512"
    assert env["VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64"] == "MTI4MCw4MTkyCg=="
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE" not in env
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256" not in env


def test_adaptive_requires_matching_table_hash(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    (runtime_root / "vllm").mkdir(parents=True)
    table = tmp_path / "tactics.json"
    digest = _write_table(table)
    inputs = AdaptiveInputs(
        tactic_file=table,
        tactic_sha256=digest,
        layer_allowlist_b64="MTI4MCw4MTkyCg==",
    )

    env = build_arm_environment("adaptive", runtime_root=runtime_root, adaptive=inputs)

    assert env["NEMORL_MXFP8_LINEAR_BACKEND"] == "flashinfer_trtllm"
    assert env["VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256"] == digest
    assert env["VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M"] == "256"

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_arm_environment(
            "adaptive",
            runtime_root=runtime_root,
            adaptive=AdaptiveInputs(
                tactic_file=table,
                tactic_sha256="0" * 64,
                layer_allowlist_b64="MTI4MCw4MTkyCg==",
            ),
        )


def test_trace_uses_trtllm_without_an_offline_tactic_table(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    (runtime_root / "vllm").mkdir(parents=True)
    trace_dir = tmp_path / "trace"

    env = build_arm_environment(
        "trace",
        runtime_root=runtime_root,
        trace=TraceInputs(trace_dir=trace_dir, trace_max=8192),
    )

    assert env["NEMORL_MXFP8_LINEAR_BACKEND"] == "flashinfer_trtllm"
    assert env["VLLM_MXFP8_DENSE_SHAPE_TRACE"] == "1"
    assert env["VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR"] == str(trace_dir.resolve())
    assert env["VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX"] == "8192"
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE" not in env
    assert "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64" not in env


def test_trace_requires_a_positive_limit(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    (runtime_root / "vllm").mkdir(parents=True)

    with pytest.raises(ValueError, match="trace_max must be positive"):
        build_arm_environment(
            "trace",
            runtime_root=runtime_root,
            trace=TraceInputs(trace_dir=tmp_path / "trace", trace_max=0),
        )


def test_summary_rejects_partial_log(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("NEMORL_CANARY arm=baseline event=start\n", encoding="utf-8")

    summary = summarize_log(log)

    assert summary == {
        "arm": "baseline",
        "complete": False,
        "elapsed_seconds": None,
        "generation_calls": None,
        "generation_seconds": None,
        "gpu_count": 8,
        "measurement_scope": "rollout_eval_wall",
        "model_load_seconds": None,
        "output_tokens": None,
        "tokens_per_second": None,
        "tokens_per_second_per_gpu": None,
    }


def test_summary_reads_completed_run(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "NEMORL_CANARY arm=adaptive event=start epoch=10.0",
                "NEMORL_CANARY event=model_ready epoch=14.5",
                "NEMORL_CANARY event=outputs tokens=8192",
                "NEMORL_CANARY arm=adaptive event=complete epoch=20.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_log(log, gpu_count=4)

    assert summary == {
        "arm": "adaptive",
        "complete": True,
        "elapsed_seconds": 10.0,
        "generation_calls": None,
        "generation_seconds": 5.5,
        "gpu_count": 4,
        "measurement_scope": "rollout_eval_wall",
        "model_load_seconds": 4.5,
        "output_tokens": 8192,
        "tokens_per_second": pytest.approx(1489.4545454545455),
        "tokens_per_second_per_gpu": pytest.approx(372.3636363636364),
    }


def test_generation_timer_accumulates_only_wrapped_async_calls() -> None:
    clock_values = iter((10.0, 12.5, 20.0, 21.5))
    timer = AsyncCallTimer(clock=lambda: next(clock_values))

    async def generate(value: str) -> str:
        return value.upper()

    timed_generate = timer.wrap(generate)

    assert asyncio.run(timed_generate("first")) == "FIRST"
    assert asyncio.run(timed_generate("second")) == "SECOND"
    assert timer.calls == 2
    assert timer.elapsed_seconds == pytest.approx(4.0)


def test_generation_length_audit_requires_exact_forced_output_length() -> None:
    audit = GenerationLengthAudit()
    audit.record([32768, 32768])
    audit.record([32768])

    audit.validate(expected_requests=3, expected_tokens_per_response=32768)

    assert audit.request_count == 3
    assert audit.total_tokens == 98304
    assert audit.min_tokens == 32768
    assert audit.max_tokens == 32768


def test_generation_length_audit_rejects_early_stop() -> None:
    audit = GenerationLengthAudit()
    audit.record([32768, 512])

    with pytest.raises(RuntimeError, match="forced output length mismatch"):
        audit.validate(expected_requests=2, expected_tokens_per_response=32768)


def test_summary_prefers_generation_call_timing_marker(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "NEMORL_CANARY arm=adaptive event=start epoch=0",
                "NEMORL_CANARY event=model_ready epoch=10",
                "NEMORL_CANARY event=generation seconds=4.0 calls=2",
                "NEMORL_CANARY event=outputs tokens=1000",
                "NEMORL_CANARY event=complete epoch=20",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_log(log, gpu_count=8)

    assert summary["generation_seconds"] == 4.0
    assert summary["generation_calls"] == 2
    assert summary["measurement_scope"] == "generation_calls"
    assert summary["tokens_per_second"] == 250.0
    assert summary["tokens_per_second_per_gpu"] == 31.25


def test_summary_prefers_engine_generated_token_count(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "NEMORL_CANARY arm=adaptive event=start epoch=0",
                "NEMORL_CANARY event=model_ready epoch=1",
                "NEMORL_CANARY event=generation seconds=4.0 calls=1",
                "NEMORL_CANARY event=generated_outputs requests=1 "
                "min_tokens=32768 max_tokens=32768 tokens=32768",
                "NEMORL_CANARY event=outputs tokens=512",
                "NEMORL_CANARY event=complete epoch=5",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_log(log, gpu_count=8)

    assert summary["output_tokens"] == 32768
    assert summary["tokens_per_second"] == 8192.0
    assert summary["tokens_per_second_per_gpu"] == 1024.0


def test_summary_rejects_non_positive_gpu_count(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="gpu_count must be positive"):
        summarize_log(log, gpu_count=0)


def test_summary_safely_represents_malformed_and_zero_duration_log(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "NEMORL_CANARY arm=baseline event=start epoch=not-a-number",
                "NEMORL_CANARY event=start epoch=10.0",
                "NEMORL_CANARY event=model_ready epoch=20.0",
                "NEMORL_CANARY event=outputs tokens=invalid",
                "NEMORL_CANARY event=outputs tokens=0",
                "NEMORL_CANARY event=complete epoch=20.0",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_log(log)

    assert summary["complete"] is True
    assert summary["elapsed_seconds"] == 10.0
    assert summary["model_load_seconds"] == 10.0
    assert summary["generation_seconds"] == 0.0
    assert summary["output_tokens"] == 0
    assert summary["tokens_per_second"] is None
    assert summary["tokens_per_second_per_gpu"] is None


def test_summary_rejects_reversed_timestamps(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "NEMORL_CANARY arm=adaptive event=start epoch=30.0",
                "NEMORL_CANARY event=model_ready epoch=20.0",
                "NEMORL_CANARY event=outputs tokens=100",
                "NEMORL_CANARY event=complete epoch=10.0",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_log(log)

    assert summary["complete"] is False
    assert summary["elapsed_seconds"] is None
    assert summary["model_load_seconds"] is None
    assert summary["generation_seconds"] is None
    assert summary["tokens_per_second"] is None


def test_summary_reports_adaptive_speedup_for_matched_pair(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.log"
    adaptive = tmp_path / "adaptive.log"
    baseline.write_text(
        "\n".join(
            [
                "NEMORL_CANARY arm=baseline event=start epoch=0",
                "NEMORL_CANARY event=model_ready epoch=10",
                "NEMORL_CANARY event=outputs tokens=1000",
                "NEMORL_CANARY event=complete epoch=20",
            ]
        ),
        encoding="utf-8",
    )
    adaptive.write_text(
        "\n".join(
            [
                "NEMORL_CANARY arm=adaptive event=start epoch=100",
                "NEMORL_CANARY event=model_ready epoch=110",
                "NEMORL_CANARY event=outputs tokens=1000",
                "NEMORL_CANARY event=complete epoch=115",
            ]
        ),
        encoding="utf-8",
    )

    report = summarize.summarize_logs([adaptive, baseline], gpu_count=8)

    assert [run["arm"] for run in report["runs"]] == ["adaptive", "baseline"]
    assert report["adaptive_vs_baseline_speedup"] == pytest.approx(2.0)
    assert "trtllm_default_vs_baseline_speedup" not in report
    assert "adaptive_vs_trtllm_default_speedup" not in report


def test_summary_omits_pair_speedup_when_generated_work_differs(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.log"
    adaptive = tmp_path / "adaptive.log"
    for path, arm, tokens in (
        (baseline, "baseline", 32768),
        (adaptive, "adaptive", 32767),
    ):
        path.write_text(
            "\n".join(
                [
                    f"NEMORL_CANARY arm={arm} event=start epoch=0",
                    "NEMORL_CANARY event=model_ready epoch=1",
                    f"NEMORL_CANARY event=generated_outputs tokens={tokens}",
                    "NEMORL_CANARY event=complete epoch=2",
                ]
            ),
            encoding="utf-8",
        )

    report = summarize.summarize_logs([baseline, adaptive])

    assert report["adaptive_vs_baseline_speedup"] is None


@pytest.mark.parametrize(
    (
        "baseline_tokens",
        "default_tokens",
        "adaptive_tokens",
        "adaptive_vs_baseline",
        "default_vs_baseline",
        "adaptive_vs_default",
    ),
    [
        (1000, 1000, 1000, 2.0, 1.25, 1.6),
        (0, 1000, 1000, None, None, 1.6),
        (900, 1000, 1000, None, None, 1.6),
        (1000, 900, 1000, 2.0, None, None),
    ],
)
def test_summary_reports_valid_three_arm_speedups(
    tmp_path: Path,
    baseline_tokens: int,
    default_tokens: int,
    adaptive_tokens: int,
    adaptive_vs_baseline: float | None,
    default_vs_baseline: float | None,
    adaptive_vs_default: float | None,
) -> None:
    baseline = tmp_path / "baseline.log"
    trtllm_default = tmp_path / "trtllm_default.log"
    adaptive = tmp_path / "adaptive.log"
    for path, arm, generation_seconds, output_tokens in (
        (baseline, "baseline", 10.0, baseline_tokens),
        (trtllm_default, "trtllm_default", 8.0, default_tokens),
        (adaptive, "adaptive", 5.0, adaptive_tokens),
    ):
        path.write_text(
            "\n".join(
                [
                    f"NEMORL_CANARY arm={arm} event=start epoch=0",
                    "NEMORL_CANARY event=model_ready epoch=1",
                    (
                        "NEMORL_CANARY event=generation "
                        f"seconds={generation_seconds} calls=1"
                    ),
                    f"NEMORL_CANARY event=outputs tokens={output_tokens}",
                    "NEMORL_CANARY event=complete epoch=20",
                ]
            ),
            encoding="utf-8",
        )

    report = summarize.summarize_logs([baseline, trtllm_default, adaptive])

    if adaptive_vs_baseline is None:
        assert report["adaptive_vs_baseline_speedup"] is None
    else:
        assert report["adaptive_vs_baseline_speedup"] == pytest.approx(
            adaptive_vs_baseline
        )
    if default_vs_baseline is None:
        assert report["trtllm_default_vs_baseline_speedup"] is None
    else:
        assert report["trtllm_default_vs_baseline_speedup"] == pytest.approx(
            default_vs_baseline
        )
    if adaptive_vs_default is None:
        assert report["adaptive_vs_trtllm_default_speedup"] is None
    else:
        assert report["adaptive_vs_trtllm_default_speedup"] == pytest.approx(
            adaptive_vs_default
        )


@pytest.mark.parametrize(
    "second_arm,baseline_tokens",
    [("trace", 1000), ("baseline", 0)],
)
def test_summary_omits_speedup_for_unmatched_or_zero_throughput_pair(
    tmp_path: Path,
    second_arm: str,
    baseline_tokens: int,
) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text(
        "\n".join(
            [
                "NEMORL_CANARY arm=adaptive event=start epoch=0",
                "NEMORL_CANARY event=model_ready epoch=1",
                "NEMORL_CANARY event=outputs tokens=1000",
                "NEMORL_CANARY event=complete epoch=2",
            ]
        ),
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(
            [
                f"NEMORL_CANARY arm={second_arm} event=start epoch=0",
                "NEMORL_CANARY event=model_ready epoch=1",
                f"NEMORL_CANARY event=outputs tokens={baseline_tokens}",
                "NEMORL_CANARY event=complete epoch=2",
            ]
        ),
        encoding="utf-8",
    )

    report = summarize.summarize_logs([first, second])

    assert report["adaptive_vs_baseline_speedup"] is None


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_run_ab_executes_three_arms_and_reports_three_way_speedups(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[3]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    run_arms = tmp_path / "run_arms.txt"
    summary_logs = tmp_path / "summary_logs.txt"
    result_root = tmp_path / "results"
    vllm_source = tmp_path / "vllm-source"
    vllm_source.mkdir()
    driver_venv = tmp_path / "driver-venv"
    driver_bin = driver_venv / "bin"
    driver_bin.mkdir(parents=True)

    _write_executable(
        fake_bin / "bash",
        """#!/bin/bash
if [[ "${1##*/}" == "run_arm.sh" ]]; then
  printf '%s\\n' "$2" >> "${RUN_ARM_LOG:?}"
  exit 0
fi
exec /bin/bash "$@"
""",
    )
    _write_executable(
        fake_bin / "git",
        """#!/bin/bash
if [[ "$*" == *"rev-parse HEAD"* ]]; then
  printf '%s\\n' "${EXPECTED_VLLM_COMMIT:?}"
fi
""",
    )
    _write_executable(
        fake_bin / "python3",
        """#!/bin/bash
if [[ "$1" == "-m" && "$2" == "experiments.mxfp8_adaptive_rollout_v0251.summarize" ]]; then
  printf '%s\\n' "$3" "$4" "$5" > "${SUMMARY_LOGS:?}"
  output=""
  for ((index = 1; index <= $#; index++)); do
    if [[ "${!index}" == "--output" ]]; then
      next=$((index + 1))
      output="${!next}"
      break
    fi
  done
  exec "${REAL_PYTHON:?}" -c 'import json; import sys; from pathlib import Path; output = Path(sys.argv[1]); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps({"runs": [{"arm": "baseline", "tokens_per_second": 100.0}, {"arm": "trtllm_default", "tokens_per_second": 150.0}, {"arm": "adaptive", "tokens_per_second": 200.0}], "adaptive_vs_baseline_speedup": None}) + "\\n")' "$output"
fi
exec "${REAL_PYTHON:?}" "$@"
""",
    )
    _write_executable(
        driver_bin / "python",
        """#!/bin/bash
for argument in "$@"; do
  if [[ "$argument" == *"runtime_overlay.py" ]]; then
    printf '%s\\n' "${FAKE_RUNTIME_ROOT:?}"
    break
  fi
done
""",
    )

    environment = os.environ | {
        "CANARY_RESULT_ROOT": str(result_root),
        "CUSTOM_VLLM_RUNTIME_BASE": str(tmp_path / "runtime-base"),
        "CUSTOM_VLLM_SOURCE": str(vllm_source),
        "EXPECTED_VLLM_COMMIT": "deadbeef",
        "FAKE_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "NEMO_RL_DRIVER_VENV_DIR": str(driver_venv),
        "NEMO_RL_REPO_ROOT": str(root),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "RUN_ARM_LOG": str(run_arms),
        "SUMMARY_LOGS": str(summary_logs),
    }

    result = subprocess.run(
        ["/bin/bash", str(root / "experiments/mxfp8_adaptive_rollout_v0251/run_ab.sh")],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert run_arms.read_text(encoding="utf-8").splitlines() == [
        "baseline",
        "trtllm_default",
        "adaptive",
    ]
    assert summary_logs.read_text(encoding="utf-8").splitlines() == [
        str(result_root / "baseline/run.log"),
        str(result_root / "trtllm_default/run.log"),
        str(result_root / "adaptive/run.log"),
    ]


def test_run_ab_pair_mode_skips_trtllm_default() -> None:
    launcher = (
        Path(__file__).parents[3]
        / "experiments/mxfp8_adaptive_rollout_v0251/run_ab.sh"
    ).read_text(encoding="utf-8")

    assert "run|pair|smoke" in launcher
    pair_branch = launcher.split('if [[ "$ACTION" == pair ]]', maxsplit=1)[1]
    pair_branch = pair_branch.split("else", maxsplit=1)[0]
    assert 'run_arm.sh" baseline' in pair_branch
    assert 'run_arm.sh" adaptive' in pair_branch
    assert "trtllm_default" not in pair_branch


def test_run_ab_baseline_mode_does_not_require_summary() -> None:
    launcher = (
        Path(__file__).parents[3]
        / "experiments/mxfp8_adaptive_rollout_v0251/run_ab.sh"
    ).read_text(encoding="utf-8")

    assert 'if [[ "$ACTION" != baseline ]]; then' in launcher
    assert 'cat "$RESULT_ROOT/summary.json"' in launcher


def test_arm_reuses_locked_driver_interpreter_for_ray_actors() -> None:
    root = Path(__file__).parents[3]
    launcher = (root / "experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh").read_text(
        encoding="utf-8"
    )

    assert "export NEMO_RL_PY_EXECUTABLES_SYSTEM=1" in launcher
    assert 'driver_python="${NEMO_RL_DRIVER_VENV_DIR:' in launcher
    assert '"$driver_python"' in launcher
    assert "uv run" not in launcher


def test_ab_uses_locked_driver_interpreter_without_resyncing_packages() -> None:
    root = Path(__file__).parents[3]
    launcher = (root / "experiments/mxfp8_adaptive_rollout_v0251/run_ab.sh").read_text(
        encoding="utf-8"
    )

    assert 'driver_python="${NEMO_RL_DRIVER_VENV_DIR:' in launcher
    assert 'runtime_python=("$driver_python")' in launcher
    assert "uv run" not in launcher


def test_flashinfer_preflight_precreates_shared_symlink_parents(
    tmp_path: Path,
) -> None:
    prepared = prepare_symlink_parents(tmp_path)

    assert prepared == (
        tmp_path / "flashinfer/trtllm/batched_gemm",
        tmp_path / "flashinfer/trtllm/gemm",
    )
    assert all(path.is_dir() for path in prepared)


def test_overlay_builder_only_clears_pythonpath_during_discovery() -> None:
    root = Path(__file__).parents[3]
    launcher = (root / "experiments/mxfp8_adaptive_rollout_v0251/run_ab.sh").read_text(
        encoding="utf-8"
    )

    builder = launcher.split("builder_python=(", maxsplit=1)[1].split(")", maxsplit=1)[
        0
    ]
    runtime = launcher.split("runtime_python=(", maxsplit=1)[1].split(")", maxsplit=1)[
        0
    ]
    assert "-u PYTHONPATH" in builder
    assert "-u PYTHONPATH" not in runtime


def test_canary_config_does_not_duplicate_worker_owned_vllm_arguments() -> None:
    root = Path(__file__).parents[3]
    config_path = (
        root / "experiments/mxfp8_adaptive_rollout_v0251/configs/eval_ultra_tp8.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    vllm_kwargs = config["generation"]["vllm_kwargs"]

    assert "enable_prefix_caching" not in vllm_kwargs


def test_qwen_trace_config_matches_performance_generation_scope() -> None:
    root = Path(__file__).parents[3]
    config_path = (
        root / "experiments/mxfp8_adaptive_rollout_v0251/configs/"
        "eval_qwen3_30ba3b_trace.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["eval"]["num_tests_per_prompt"] == 32
    assert config["generation"]["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert config["generation"]["num_prompts_per_step"] == 64
    assert config["generation"]["vllm_cfg"]["precision"] == "fp8"
    assert config["generation"]["vllm_cfg"]["is_mx"] is True
    assert config["generation"]["vllm_cfg"]["tensor_parallel_size"] == 1
    assert config["generation"]["vllm_cfg"]["expert_parallel_size"] == 1
    assert config["generation"]["vllm_cfg"]["enforce_eager"] is True
    assert config["generation"]["vllm_cfg"]["quantization_ignored_layer_kws"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert config["generation"]["colocated"]["resources"] == {
        "gpus_per_node": 4,
        "num_nodes": 2,
    }
    dataset_path = (
        root / "experiments/mxfp8_adaptive_rollout_v0251/data/qwen_trace_math.jsonl"
    )
    assert len(dataset_path.read_text(encoding="utf-8").splitlines()) == 64


def test_trace_launcher_uses_eager_discovery_then_summarizes_shapes() -> None:
    root = Path(__file__).parents[3]
    launcher = (root / "experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh").read_text(
        encoding="utf-8"
    )

    assert '--trace-dir "${SHAPE_TRACE_DIR:?set SHAPE_TRACE_DIR}"' in launcher
    assert "shape_trace" in launcher
    assert "shape_summary.json" in launcher


def test_qwen_submitter_uses_ptyche_without_ultra_tactic_artifacts() -> None:
    root = Path(__file__).parents[3]
    submitter = (
        root / "experiments/mxfp8_adaptive_rollout_v0251/submit_qwen30_ptyche.sh"
    ).read_text(encoding="utf-8")

    assert "--nodes=2" in submitter
    assert "--segment=2" in submitter
    assert "--time=05:00:00" in submitter
    assert "run_trace.sh" in submitter
    assert "TACTIC_FILE=" not in submitter
    assert "LAYER_ALLOWLIST_B64=" not in submitter


def test_shape_trace_summary_accepts_zero_eligible_dense_calls(
    tmp_path: Path,
) -> None:
    summary = summarize_shape_trace(tmp_path)

    assert summary == {
        "eligible": False,
        "record_count": 0,
        "unique_signature_count": 0,
        "signatures": [],
    }


def test_shape_trace_summary_deduplicates_exact_signatures(tmp_path: Path) -> None:
    record = {
        "event": "mxfp8_dense_shape",
        "family": "OtherDense",
        "hostname": "node-a",
        "k": 2048,
        "layout": "8x4",
        "m": 8,
        "n_logical": 6144,
        "n_physical": 6144,
        "pid": 10,
        "prefix": "model.layers.0.proj",
    }
    (tmp_path / "rank0.jsonl").write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )
    second = dict(record, m=16, pid=11)
    (tmp_path / "rank1.jsonl").write_text(json.dumps(second) + "\n", encoding="utf-8")

    summary = summarize_shape_trace(tmp_path)

    assert summary["eligible"] is True
    assert summary["record_count"] == 3
    assert summary["unique_signature_count"] == 2
    assert [entry["m"] for entry in summary["signatures"]] == [8, 16]


def test_runtime_overlay_preserves_wheel_extensions_and_overlays_source(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "installed" / "vllm"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("ORIGIN = 'wheel'\n", encoding="utf-8")
    extension = installed / "_C_stable_libtorch.abi3.so"
    extension.write_bytes(b"compiled-extension")
    (installed / "wheel_only.py").write_text("VALUE = 1\n", encoding="utf-8")

    source = tmp_path / "source" / "vllm"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("ORIGIN = 'source'\n", encoding="utf-8")
    (source / "patched.py").write_text("VALUE = 2\n", encoding="utf-8")

    runtime_root = prepare_runtime_overlay(
        installed_package=installed,
        source_package=source,
        destination_base=tmp_path / "runtime",
        source_revision="deadbeef",
    )

    runtime_package = runtime_root / "vllm"
    assert (runtime_package / "__init__.py").read_text(encoding="utf-8") == (
        "ORIGIN = 'source'\n"
    )
    assert (runtime_package / "patched.py").is_file()
    assert (runtime_package / "wheel_only.py").is_file()
    assert (runtime_package / extension.name).read_bytes() == b"compiled-extension"


def test_runtime_overlay_rejects_wheel_without_stable_extension(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "installed" / "vllm"
    source = tmp_path / "source" / "vllm"
    installed.mkdir(parents=True)
    source.mkdir(parents=True)

    with pytest.raises(ValueError, match="_C_stable_libtorch"):
        prepare_runtime_overlay(
            installed_package=installed,
            source_package=source,
            destination_base=tmp_path / "runtime",
            source_revision="deadbeef",
        )


def test_runtime_overlay_is_immutable_and_content_addressed(tmp_path: Path) -> None:
    installed = tmp_path / "installed" / "vllm"
    source = tmp_path / "source" / "vllm"
    installed.mkdir(parents=True)
    source.mkdir(parents=True)
    extension = installed / "_C_stable_libtorch.abi3.so"
    extension.write_bytes(b"extension-v1")

    first = prepare_runtime_overlay(
        installed_package=installed,
        source_package=source,
        destination_base=tmp_path / "runtime",
        source_revision="deadbeef",
    )
    reused = prepare_runtime_overlay(
        installed_package=installed,
        source_package=source,
        destination_base=tmp_path / "runtime",
        source_revision="deadbeef",
    )
    extension.write_bytes(b"extension-v2")
    second = prepare_runtime_overlay(
        installed_package=installed,
        source_package=source,
        destination_base=tmp_path / "runtime",
        source_revision="deadbeef",
    )

    assert reused == first
    assert second != first
    assert first.is_dir()
    assert second.is_dir()
