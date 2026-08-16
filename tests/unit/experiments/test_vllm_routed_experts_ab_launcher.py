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

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    ROOT
    / "experiments"
    / "cuda_graph"
    / "nemotron_thd_te_graph_20260731"
    / "diagnostics"
    / "submit_vllm_routed_experts_ab.sh"
)


def _render(backend: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "ACCOUNT": "coreai_dlalgo_llm",
        "ARTIFACT_DIR": "/lustre/test/results",
        "ATTESTED_NEMORL_SHA": "1" * 40,
        "BACKEND": backend,
        "CONTAINER": "/lustre/test/runtime.sqsh",
        "CONTAINER_SHA256": "2" * 64,
        "EXPECTED_BRIDGE_SHA": "3" * 40,
        "EXPECTED_MCORE_SHA": "4" * 40,
        "EXPECTED_NEMORL_SHA": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "EXPECTED_TE_SHA": "5" * 40,
        "EXPECTED_TE_VERSION_BASE_SHA": "5" * 40,
        "HF_HOME": "/lustre/test/hf",
        "MODEL_PATH": "/lustre/test/model",
        "PARTITION": "batch",
        "PROJECT_ROOT": str(ROOT),
        "RUNTIME_ATTESTATION": "/lustre/test/runtime.json",
        "RUNTIME_PREFLIGHT_JOB_ID": "12345",
        "TEST_ONLY": "1",
        "UV_EXECUTABLE": "/lustre/test/stage/uv/uv",
        "VLLM_PYTHON": "/lustre/test/vllm-environment/bin/python",
    }
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_triton_render_changes_only_the_moe_backend() -> None:
    result = _render("triton")

    assert result.returncode == 0, result.stderr
    assert "--nodes=1" in result.stdout
    assert "--gpus-per-node" not in result.stdout
    assert "--gres" not in result.stdout
    assert "--segment" not in result.stdout
    assert "--exclusive" not in result.stdout
    assert "RUNTIME_ATTESTATION_COMMAND:" in result.stdout
    assert "--runtime-feature-set dropless_hybridep_nano16" in result.stdout
    assert "--expected-container-sha256" in result.stdout
    assert "--num-prompts 1" in result.stdout
    assert "--max-tokens 2" in result.stdout
    assert "--llm-kwarg load_format=dummy" in result.stdout
    assert "--llm-kwarg seed=123" in result.stdout
    assert "--llm-kwarg moe_backend=triton" in result.stdout
    assert "/lustre/test/stage/vllm-environment/bin/python" in result.stdout
    assert "/lustre/test/vllm-environment/bin/python" not in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout


def test_auto_render_does_not_force_a_moe_backend() -> None:
    result = _render("auto")

    assert result.returncode == 0, result.stderr
    diagnostic_command = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("DIAGNOSTIC_COMMAND:")
    )
    assert "moe_backend=" not in diagnostic_command


def test_both_render_runs_auto_then_triton_in_one_allocation() -> None:
    result = _render("both")

    assert result.returncode == 0, result.stderr
    assert "DIAGNOSTIC_COMMAND[auto]:" in result.stdout
    assert "DIAGNOSTIC_COMMAND[triton]:" in result.stdout
    assert "CACHE_ROOT[auto]: /lustre/test/results/cache-auto-JOB_ID" in result.stdout
    assert (
        "CACHE_ROOT[triton]: /lustre/test/results/cache-triton-JOB_ID" in result.stdout
    )
    assert result.stdout.count("SBATCH:") == 1


def _run_fake_both_job(
    tmp_path: Path, *, publish_results: bool
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "srun-calls.txt"
    fake_srun = bin_dir / "srun"
    fake_srun.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [[ "$PUBLISH_RESULTS" == 1 ]]; then\n'
        "  while (( $# )); do\n"
        '    if [[ "$1" == --output ]]; then printf "{}\\n" > "$2"; break; fi\n'
        "    shift\n"
        "  done\n"
        "fi\n"
        '[[ " $* " == *" moe_backend=triton "* ]]\n'
    )
    fake_srun.chmod(0o755)
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == *verify_runtime_attestation.py ]]; then exit 0; fi\n'
        'exec /usr/bin/python3 "$@"\n'
    )
    fake_python.chmod(0o755)
    container = tmp_path / "runtime.sqsh"
    container.write_text("runtime")
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    uv_executable = tmp_path / "stage" / "uv" / "uv"
    uv_executable.parent.mkdir(parents=True)
    uv_executable.write_text("runtime")
    vllm_python = tmp_path / "stage" / "vllm-environment" / "bin" / "python"
    vllm_python.parent.mkdir(parents=True)
    vllm_python.symlink_to("/usr/bin/true")
    env = {
        **os.environ,
        "ACCOUNT": "coreai_dlalgo_llm",
        "ARTIFACT_DIR": str(tmp_path / "results"),
        "ATTESTED_NEMORL_SHA": "1" * 40,
        "BACKEND": "both",
        "CALL_LOG": str(call_log),
        "CONTAINER": str(container),
        "CONTAINER_SHA256": "2" * 64,
        "EXPECTED_BRIDGE_SHA": "3" * 40,
        "EXPECTED_MCORE_SHA": "4" * 40,
        "EXPECTED_NEMORL_SHA": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "EXPECTED_TE_SHA": "5" * 40,
        "EXPECTED_TE_VERSION_BASE_SHA": "5" * 40,
        "HF_HOME": str(hf_home),
        "MODEL_PATH": str(model_path),
        "PARTITION": "batch",
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PROJECT_ROOT": str(ROOT),
        "PUBLISH_RESULTS": "1" if publish_results else "0",
        "RUNTIME_ATTESTATION": str(tmp_path / "runtime.json"),
        "RUNTIME_PREFLIGHT_JOB_ID": "12345",
        "SLURM_JOB_ID": "123",
        "UV_EXECUTABLE": str(uv_executable),
    }

    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    ), call_log


def test_both_job_continues_to_triton_after_expected_auto_failure(
    tmp_path: Path,
) -> None:
    result, call_log = _run_fake_both_job(tmp_path, publish_results=True)

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text().splitlines()
    assert len(calls) == 2
    assert "moe_backend=triton" not in calls[0]
    assert "moe_backend=triton" in calls[1]


def test_both_job_fails_when_an_arm_does_not_publish_a_result(tmp_path: Path) -> None:
    result, call_log = _run_fake_both_job(tmp_path, publish_results=False)

    assert result.returncode != 0
    assert len(call_log.read_text().splitlines()) == 2
