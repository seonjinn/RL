from __future__ import annotations

import csv
import os
import subprocess
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "experiments/eagle3_online/submit_lyris_nemorl_v020_best_math_20260704.sh"
)
REMOTE_REPO = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701"
)
EXPECTED_REPO_HEAD = "1271b1530181a7378e40de40b4b46ad223e6596c"


def run_dry(
    *, check: bool = True, **overrides: str
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ["PATH"],
        "DRY_RUN": "true",
        **overrides,
    }
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def job_sections(output: str) -> dict[tuple[str, str, str], str]:
    sections: dict[tuple[str, str, str], str] = {}
    for section in output.split("[DRY-RUN] model=")[1:]:
        header = section.splitlines()[0].split()
        model = header[0]
        mode = header[1].removeprefix("mode=")
        method = header[2].removeprefix("method=")
        sections[(model, mode, method)] = section
    return sections


def write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip())
    path.chmod(0o755)


def prepare_fake_submission(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    remote_repo = tmp_path / "remote-repo"
    container = tmp_path / "nemo-rl.sqsh"
    hf_home = tmp_path / "hf-home"
    target_model = hf_home / "qwen235b"
    draft_model = hf_home / "qwen235b-eagle3"
    source_site = tmp_path / "arctic-site"
    run_root = tmp_path / "runs"
    manifest = tmp_path / "manifest.csv"

    required_files = (
        remote_repo / "examples/run_grpo.py",
        remote_repo / "ray.sub",
        remote_repo
        / "examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml",
        remote_repo
        / "examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off.yaml",
        target_model / "config.json",
        draft_model / "config.json",
    )
    for path in required_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    (source_site / "arctic_inference/suffix_decoding").mkdir(parents=True)
    container.write_text("container fixture\n")

    write_executable(
        fake_bin / "git",
        r"""
        #!/usr/bin/env bash
        set -euo pipefail
        worktree=""
        if [[ "${1:-}" == "-C" ]]; then
          worktree="$2"
          shift 2
        fi
        case "${1:-}:${2:-}" in
          status:--porcelain)
            if [[ "${worktree}" == "${FAKE_REMOTE_REPO}" ]]; then
              printf '%s' "${FAKE_REMOTE_STATUS:-}"
            else
              printf '%s' "${FAKE_LOCAL_STATUS:-}"
            fi
            ;;
          rev-parse:HEAD)
            printf '%s\n' "${FAKE_REMOTE_SHA}"
            ;;
          rev-parse:--abbrev-ref)
            printf '%s\n' 'origin/fake-branch'
            ;;
          rev-list:--count)
            printf '%s\n' "${FAKE_LOCAL_AHEAD:-0}"
            ;;
          *)
            printf 'unsupported fake git invocation: %q ' "$@" >&2
            printf '\n' >&2
            exit 91
            ;;
        esac
        """,
    )
    write_executable(
        fake_bin / "ssh",
        r"""
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\0' "$@" > "${FAKE_SSH_LOG}"
        while [[ "${1:-}" == -* ]]; do
          case "$1" in
            -o) shift 2 ;;
            *) shift ;;
          esac
        done
        host="$1"
        shift
        printf '%s\n' "${host}" > "${FAKE_SSH_HOST_LOG}"
        remote_command="$*"
        bash -c "${remote_command}"
        """,
    )
    write_executable(
        fake_bin / "sbatch",
        r"""
        #!/usr/bin/env bash
        set -euo pipefail
        count=0
        if [[ -s "${FAKE_SBATCH_COUNT}" ]]; then
          count="$(cat "${FAKE_SBATCH_COUNT}")"
        fi
        count=$((count + 1))
        printf '%s\n' "${count}" > "${FAKE_SBATCH_COUNT}"
        printf '%s\0' "$@" > "${FAKE_SBATCH_PREFIX}.${count}"
        last_arg="${!#}"
        if [[ "${last_arg}" != "${FAKE_REMOTE_REPO}/ray.sub" ]]; then
          printf 'sbatch: unable to open file %s\n' "${last_arg}" >&2
          exit 43
        fi
        if [[ "${1:-}" == "--test-only" ]]; then
          if [[ "${FAKE_TEST_ONLY_FAIL:-false}" == "true" ]]; then
            echo 'sbatch: test-only rejected' >&2
            exit 42
          fi
          echo 'sbatch: Job 101 to start at 2026-07-04T12:00:00'
        else
          echo 'Submitted batch job 202'
        fi
        """,
    )

    return {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DRY_RUN": "false",
        "TEST_ONLY": "false",
        "RUN_KIND": "final",
        "MODELS": "qwen235b",
        "MODES": "sync",
        "METHODS": "baseline",
        "REMOTE_HOST": "login-lyris",
        "REMOTE_REPO": str(remote_repo),
        "CONTAINER": str(container),
        "HF_HOME": str(hf_home),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "QWEN235B_MODEL": str(target_model),
        "QWEN235B_EAGLE3_MODEL": str(draft_model),
        "SOURCE_VLLM_SITE": str(source_site),
        "RUN_ROOT": str(run_root),
        "RUN_ID": "test-run",
        "WANDB_HOME": str(tmp_path / "wandb-home"),
        "WANDB_PROJECT": "test-project",
        "ACCOUNT": "test-account",
        "PARTITION": "gb200",
        "WALLTIME": "00:10:00",
        "OUT": str(manifest),
        "FAKE_REMOTE_REPO": str(remote_repo),
        "FAKE_REMOTE_SHA": EXPECTED_REPO_HEAD,
        "FAKE_REMOTE_STATUS": "",
        "FAKE_LOCAL_STATUS": "",
        "FAKE_LOCAL_AHEAD": "0",
        "FAKE_SSH_LOG": str(tmp_path / "ssh.args"),
        "FAKE_SSH_HOST_LOG": str(tmp_path / "ssh.host"),
        "FAKE_SBATCH_COUNT": str(tmp_path / "sbatch.count"),
        "FAKE_SBATCH_PREFIX": str(tmp_path / "sbatch.args"),
    }


def run_fake_submission(
    tmp_path: Path,
    *,
    check: bool = False,
    overrides: dict[str, str] | None = None,
    **env_overrides: str,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    env = {
        **prepare_fake_submission(tmp_path),
        **(overrides or {}),
        **env_overrides,
    }
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed, env


def read_nul_args(path: Path) -> list[str]:
    return [os.fsdecode(value) for value in path.read_bytes().split(b"\0") if value]


def sbatch_calls(env: dict[str, str]) -> list[list[str]]:
    count_path = Path(env["FAKE_SBATCH_COUNT"])
    if not count_path.exists():
        return []
    count = int(count_path.read_text())
    prefix = env["FAKE_SBATCH_PREFIX"]
    return [read_nul_args(Path(f"{prefix}.{index}")) for index in range(1, count + 1)]


def test_default_final_run_renders_all_supported_contracts() -> None:
    sections = job_sections(run_dry().stdout)

    assert set(sections) == {
        ("qwen30ba3b", "sync", "suffix"),
        ("qwen30ba3b", "async1off", "suffix"),
        ("qwen32", "sync", "suffix"),
        ("qwen32", "sync", "eagle3"),
        ("qwen32", "async1off", "suffix"),
        ("qwen32", "async1off", "eagle3"),
        ("qwen235b", "sync", "baseline"),
        ("qwen235b", "sync", "suffix"),
        ("qwen235b", "sync", "eagle3"),
        ("qwen235b", "async1off", "baseline"),
        ("qwen235b", "async1off", "suffix"),
        ("qwen235b", "async1off", "eagle3"),
    }

    shapes = {
        ("qwen30ba3b", "sync"): ("grpo-qwen3-30ba3b-4n4g.yaml", 4, 4),
        ("qwen30ba3b", "async1off"): (
            "grpo-qwen3-30ba3b-4n4g-async-1off.yaml",
            4,
            4,
        ),
        ("qwen32", "sync"): ("grpo-qwen3-32b-4n4g.yaml", 4, 4),
        ("qwen32", "async1off"): (
            "grpo-qwen3-32b-8n4g-async-1off.yaml",
            8,
            8,
        ),
        ("qwen235b", "sync"): ("grpo-qwen3-235b-32n4g.yaml", 32, 16),
        ("qwen235b", "async1off"): (
            "grpo-qwen3-235b-32n4g-async-1off.yaml",
            32,
            16,
        ),
    }
    for (model, mode, _method), section in sections.items():
        recipe, nodes, segment = shapes[(model, mode)]
        assert recipe in section
        assert f"--nodes={nodes}" in section
        assert f"--segment={segment}" in section
        assert "policy.generation.temperature=1.0" in section
        assert "policy.generation.top_p=1.0" in section
        assert "policy.generation.vllm_cfg.enforce_eager=false" in section
        assert "attention_backend=TRITON_ATTN" in section
        assert "kernel_config.moe_backend=triton" in section
        assert "export NEMO_RL_VENV_DIR=/opt/ray_venvs" in section
        assert "grpo.max_num_steps=20" in section
        assert "logger.wandb_enabled=true" in section
        assert "policy.generation.max_new_tokens" not in section
        assert "--gres" not in section


def test_qwen235b_contracts_use_required_topology_and_k_values() -> None:
    sections = job_sections(
        run_dry(
            MODELS="qwen235b",
            MODES="sync async-1off",
            METHODS="baseline,suffix,eagle3",
        ).stdout
    )

    assert len(sections) == 6
    for section in sections.values():
        assert "--nodes=32" in section
        assert "--segment=16" in section
        assert "--network=sharp" not in section
        assert "compilation_config.pass_config.fuse_allreduce_rms=false" in section
        assert "policy.generation.temperature=1.0" in section
        assert "policy.generation.top_p=1.0" in section
        assert "policy.generation.vllm_cfg.enforce_eager=false" in section
        assert "attention_backend=TRITON_ATTN" in section
        assert "kernel_config.moe_backend=triton" in section

    for mode in ("sync", "async1off"):
        baseline = sections[("qwen235b", mode, "baseline")]
        suffix = sections[("qwen235b", mode, "suffix")]
        eagle3 = sections[("qwen235b", mode, "eagle3")]
        assert "speculative_config" not in baseline
        assert "speculative_config.method=suffix" in suffix
        assert "speculative_config.num_speculative_tokens=32" in suffix
        assert "speculative_config.method=eagle3" in eagle3
        assert "speculative_config.num_speculative_tokens=3" in eagle3
        assert "models--nvidia--Qwen3-235B-A22B-Eagle3" in eagle3


def test_q32_async_keeps_four_worker_cluster_segment_size() -> None:
    sections = job_sections(
        run_dry(MODELS="qwen32", MODES="async-1off", METHODS="suffix,eagle3").stdout
    )

    assert set(sections) == {
        ("qwen32", "async1off", "suffix"),
        ("qwen32", "async1off", "eagle3"),
    }
    for section in sections.values():
        assert "cluster.segment_size=4" in section
        assert "--nodes=8" in section
        assert "--segment=8" in section


def test_run_kind_enforces_steps_and_smoke_model_scope() -> None:
    smoke = run_dry(
        RUN_KIND="smoke",
        MODELS="qwen235b",
        MODES="sync",
        METHODS="baseline",
    )
    wrong_final_steps = run_dry(check=False, RUN_KIND="final", MAX_STEPS="2")
    wrong_smoke_steps = run_dry(
        check=False,
        RUN_KIND="smoke",
        MAX_STEPS="20",
        MODELS="qwen235b",
    )
    wrong_smoke_model = run_dry(
        check=False,
        RUN_KIND="smoke",
        MODELS="qwen32",
        MAX_STEPS="2",
    )
    unknown_kind = run_dry(check=False, RUN_KIND="adhoc")

    assert "grpo.max_num_steps=2" in smoke.stdout
    assert wrong_final_steps.returncode == 2
    assert "RUN_KIND=final requires MAX_STEPS=20" in wrong_final_steps.stderr
    assert wrong_smoke_steps.returncode == 2
    assert "RUN_KIND=smoke requires MAX_STEPS=2" in wrong_smoke_steps.stderr
    assert wrong_smoke_model.returncode == 2
    assert "RUN_KIND=smoke permits only qwen235b" in wrong_smoke_model.stderr
    assert unknown_kind.returncode == 2
    assert "unsupported RUN_KIND: adhoc" in unknown_kind.stderr


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("REMOTE_HOST", "other-host", "REMOTE_HOST must be login-lyris"),
        ("PARTITION", "debug", "PARTITION must be gb200"),
    ),
)
def test_lyris_endpoint_and_partition_are_enforced(
    name: str, value: str, message: str
) -> None:
    completed = run_dry(check=False, **{name: value})

    assert completed.returncode == 2
    assert message in completed.stderr


def test_selected_eagle3_preflight_validates_only_selected_assets() -> None:
    output = run_dry(MODELS="qwen235b", MODES="sync", METHODS="eagle3").stdout

    assert set(job_sections(output)) == {("qwen235b", "sync", "eagle3")}
    assert EXPECTED_REPO_HEAD in output
    assert "status --porcelain --untracked-files=normal" in output
    assert "grpo-qwen3-235b-32n4g.yaml" in output
    assert "models--Qwen--Qwen3-235B-A22B" in output
    assert "models--nvidia--Qwen3-235B-A22B-Eagle3" in output
    assert "models--Qwen--Qwen3-30B-A3B" not in output
    assert "models--Qwen--Qwen3-32B" not in output
    assert "arctic-inference" not in output


def test_suffix_k32_uses_source_vllm_site_and_shared_hf_caches() -> None:
    section = job_sections(
        run_dry(
            MODELS="qwen235b",
            MODES="sync",
            METHODS="suffix",
            HF_HOME="/shared/hf",
            HF_DATASETS_CACHE="/shared/hf/datasets",
            SOURCE_VLLM_SITE="/shared/arctic-site",
        ).stdout
    )[("qwen235b", "sync", "suffix")]

    assert "SOURCE_VLLM_SITE=/shared/arctic-site" in section
    assert "export HF_HOME=/shared/hf" in section
    assert "export HF_DATASETS_CACHE=/shared/hf/datasets" in section
    assert "speculative_config.method=suffix" in section
    assert "speculative_config.num_speculative_tokens=32" in section


def test_q30_eagle3_is_not_supported() -> None:
    completed = run_dry(
        check=False,
        MODELS="qwen30ba3b",
        MODES="sync",
        METHODS="eagle3",
    )

    assert completed.returncode == 2
    assert "selection contains no supported model/mode/method combinations" in (
        completed.stderr
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"FAKE_LOCAL_STATUS": "?? untracked\n"}, "local worktree is not clean"),
        ({"FAKE_LOCAL_AHEAD": "1"}, "local HEAD is ahead of upstream"),
    ),
)
def test_local_preflight_stops_before_ssh(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    completed, env = run_fake_submission(tmp_path, overrides=overrides)

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not Path(env["FAKE_SSH_LOG"]).exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"FAKE_REMOTE_STATUS": " M examples/run_grpo.py\n"}, "not clean"),
        ({"FAKE_REMOTE_SHA": "deadbeef"}, "does not match pinned SHA"),
    ),
)
def test_remote_preflight_stops_before_sbatch(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    completed, env = run_fake_submission(tmp_path, overrides=overrides)

    assert completed.returncode != 0
    assert message in completed.stderr
    assert Path(env["FAKE_SSH_LOG"]).exists()
    assert sbatch_calls(env) == []


def test_failed_sbatch_test_only_prevents_actual_submission(tmp_path: Path) -> None:
    completed, env = run_fake_submission(tmp_path, FAKE_TEST_ONLY_FAIL="true")

    calls = sbatch_calls(env)
    assert completed.returncode != 0
    assert "test-only rejected" in completed.stderr
    assert len(calls) == 1
    assert calls[0][0] == "--test-only"


def test_sbatch_gate_reuses_identical_arguments_and_writes_manifest(
    tmp_path: Path,
) -> None:
    completed, env = run_fake_submission(tmp_path, check=True)

    calls = sbatch_calls(env)
    assert completed.returncode == 0
    assert len(calls) == 2
    assert calls[0][0] == "--test-only"
    assert calls[0][1:] == calls[1]
    assert "--gres" not in calls[1]
    assert "--nodes=32" in calls[1]
    assert "--segment=16" in calls[1]
    assert "--network=sharp" not in calls[1]

    with Path(env["OUT"]).open(newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    assert len(rows) == 1
    row = rows[0]
    assert row["actual_job_id"] == "202"
    assert row["test_only_job_id"] == "101"
    assert row["repo_sha"] == EXPECTED_REPO_HEAD
    assert len(row["container_sha256"]) == 64
    assert row["rendered_command"].startswith("set -euo pipefail\n")
    assert row["config"].endswith("grpo-qwen3-235b-32n4g.yaml")
    assert row["nodes"] == "32"
    assert row["gpus_per_node"] == "4"
    assert row["segment"] == "16"
    assert row["cluster_segment_size"] == ""
    assert row["network"] == ""
    assert row["wandb_enabled"] == "true"
    assert row["wandb_project"] == "test-project"
    assert row["wandb_name"] == "qwen235b_math_sync_baseline_test-run"


def test_user_values_are_shell_safe_and_preserved_in_manifest(tmp_path: Path) -> None:
    marker = tmp_path / "injected"
    hostile_account = f"acct$(touch${{IFS}}{marker})"
    wandb_project = "project with spaces ' quotes $dollars and ; semicolons"
    completed, env = run_fake_submission(
        tmp_path,
        check=True,
        ACCOUNT=hostile_account,
        WANDB_PROJECT=wandb_project,
        RUN_ID="run with spaces ' and $signs",
    )

    assert completed.returncode == 0
    assert not marker.exists()
    assert Path(env["FAKE_SSH_HOST_LOG"]).read_text().strip() == "login-lyris"
    with Path(env["OUT"]).open(newline="") as manifest_file:
        row = next(csv.DictReader(manifest_file))
    assert row["wandb_project"] == wandb_project
    assert r"run\ with\ spaces" in row["rendered_command"]


def test_launcher_is_declarative_pinned_and_shell_valid() -> None:
    source = LAUNCHER.read_text()

    assert f'REMOTE_REPO="${{REMOTE_REPO:-{REMOTE_REPO}}}"' in source
    assert EXPECTED_REPO_HEAD in source
    assert "printf -v remote_command" in source
    assert "%q" in source
    assert "eval " not in source
    assert "scontrol show job" not in source
    assert "sacct" not in source
    assert "ray-driver.log" not in source
    assert "--gres" not in source
    subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=ROOT, check=True)
