from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "vllm_024_upgrade"
    / "submit_tail_gated_specdec_step20.sh"
)
V1_VARIANTS = ("baseline_v1", "always_on_v1_k5", "stock_dynamic_v1")
V2_VARIANTS = (
    "baseline_v2",
    "always_on_v2_k5",
    "fastrl_threshold_v2_k5",
    "efficient_roofline_v2_k5",
)
GATED_V2_VARIANTS = ("fastrl_threshold_v2_k5", "efficient_roofline_v2_k5")
QWEN32_TARGET_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
QWEN32_DRAFT_REVISION = "dc84fe7ff1db31efa824776f49c141fc8195eb47"
CALIBRATION_TIMESTAMP = "2026-07-10T12:34:56Z"
CALIBRATION_CLUSTER = "lyris-gb200"
VLLM_COMMIT = "ee0da84a"
VLLM_VERSION = "0.24.0"
RUNTIME_VERSION = "nightly-20260707"


def _run_launcher(*args: str, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **environment},
        check=False,
        capture_output=True,
        text=True,
    )


def _dry_run(model: str, variant: str, **environment: str) -> str:
    launcher_environment = {
        "REPO_DIR": "/lustre/test/nemo-rl",
        "LYRIS_ROOT": "/lustre/test",
        "HF_HOME": "/lustre/test/hf_home",
        "CONTAINER": "/lustre/test/nemo-rl.sqsh",
        "EXPERIMENT_ROOT": "/lustre/test/tail-gate-runs",
        "RUN_TAG": "contract",
        "ATTEMPT_ID": "attempt-1",
        "QWEN30_ROOFLINE_CONFIG": "/lustre/test/calibrations/qwen30.json",
        "QWEN32_ROOFLINE_CONFIG": "/lustre/test/calibrations/qwen32.json",
        **environment,
    }
    result = _run_launcher(
        "dry-run",
        model,
        variant,
        **launcher_environment,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_dry_run_exposes_exactly_the_seven_planned_variants() -> None:
    result = _run_launcher(
        "dry-run",
        "all",
        "all",
        REPO_DIR="/lustre/test/nemo-rl",
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER="/lustre/test/nemo-rl.sqsh",
        EXPERIMENT_ROOT="/lustre/test/tail-gate-runs",
        RUN_TAG="contract",
        ATTEMPT_ID="attempt-1",
        QWEN30_ROOFLINE_CONFIG="/lustre/test/calibrations/qwen30.json",
        QWEN32_ROOFLINE_CONFIG="/lustre/test/calibrations/qwen32.json",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[DRY-RUN] job ") == 14
    for variant in (*V1_VARIANTS, *V2_VARIANTS):
        assert result.stdout.count(f"variant={variant}") == 2


def test_v1_variants_preserve_stock_runner_and_dynamic_contract() -> None:
    for variant in V1_VARIANTS:
        output = _dry_run("qwen30ba3b", variant)

        assert "VLLM_USE_V2_MODEL_RUNNER=0" in output
        assert "cudagraph_mode=PIECEWISE" in output
        assert "scheduler_cls=" not in output
        assert "sd_tail_gate_mode=" not in output
        assert "grpo-qwen3-30ba3b-4n4g.yaml" in output
        assert "--nodes=4" in output
        assert "--segment=4" in output

    dynamic_output = _dry_run("qwen30ba3b", "stock_dynamic_v1")
    assert (
        "num_speculative_tokens_per_batch_size=[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]"
        in dynamic_output.replace("\\", "")
    )


def test_v2_variants_preserve_runner_graph_and_gate_boundaries() -> None:
    for variant in V2_VARIANTS:
        output = _dry_run("qwen32b", variant)

        assert "VLLM_USE_V2_MODEL_RUNNER=1" in output
        assert "cudagraph_mode=FULL_AND_PIECEWISE" in output
        assert "grpo-qwen3-32b-4n4g.yaml" in output
        assert "--nodes=4" in output
        assert "--segment=4" in output

        if variant in GATED_V2_VARIANTS:
            assert (
                "scheduler_cls=nemo_rl.models.generation.vllm.tail_gate_scheduler.TailGatedScheduler"
                in output
            )
            assert "sd_tail_gate_mode=" in output
            assert "sd_tail_gate_threshold=32" in output
            assert "sd_tail_gate_consecutive_checks=10" in output
        else:
            assert "scheduler_cls=" not in output
            assert "sd_tail_gate_mode=" not in output

    roofline_output = _dry_run("qwen32b", "efficient_roofline_v2_k5")
    assert "sd_tail_gate_mode=roofline" in roofline_output
    assert "sd_tail_gate_margin=0.05" in roofline_output
    assert (
        "sd_tail_gate_config_path=/lustre/test/calibrations/qwen32.json"
        in roofline_output
    )


@pytest.mark.parametrize(
    ("setting", "value", "expected"),
    (
        ("TAIL_GATE_THRESHOLD", "17", "sd_tail_gate_threshold=17"),
        (
            "TAIL_GATE_CONSECUTIVE_CHECKS",
            "3",
            "sd_tail_gate_consecutive_checks=3",
        ),
    ),
)
def test_gate_settings_use_validated_environment_values(
    setting: str, value: str, expected: str
) -> None:
    for variant in GATED_V2_VARIANTS:
        output = _dry_run("qwen32b", variant, **{setting: value})

        assert expected in output


@pytest.mark.parametrize(
    ("setting", "value"),
    (
        ("TAIL_GATE_THRESHOLD", "0"),
        ("TAIL_GATE_THRESHOLD", "not-an-integer"),
        ("TAIL_GATE_CONSECUTIVE_CHECKS", "0"),
        ("TAIL_GATE_CONSECUTIVE_CHECKS", "not-an-integer"),
    ),
)
def test_gate_settings_reject_non_positive_integers(setting: str, value: str) -> None:
    result = _run_launcher(
        "dry-run",
        "qwen32b",
        "baseline_v2",
        REPO_DIR="/lustre/test/nemo-rl",
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER="/lustre/test/nemo-rl.sqsh",
        EXPERIMENT_ROOT="/lustre/test/tail-gate-runs",
        RUN_TAG="contract",
        ATTEMPT_ID="attempt-1",
        QWEN30_ROOFLINE_CONFIG="/lustre/test/calibrations/qwen30.json",
        QWEN32_ROOFLINE_CONFIG="/lustre/test/calibrations/qwen32.json",
        **{setting: value},
    )

    assert result.returncode == 2
    assert f"ERROR: {setting} must be a positive integer" in result.stderr


def test_roofline_dry_run_selects_a_separate_config_per_model() -> None:
    qwen30_output = _dry_run("qwen30ba3b", "efficient_roofline_v2_k5")
    qwen32_output = _dry_run("qwen32b", "efficient_roofline_v2_k5")

    assert (
        "sd_tail_gate_config_path=/lustre/test/calibrations/qwen30.json"
        in qwen30_output
    )
    assert "/lustre/test/calibrations/qwen32.json" not in qwen30_output
    assert (
        "sd_tail_gate_config_path=/lustre/test/calibrations/qwen32.json"
        in qwen32_output
    )
    assert "/lustre/test/calibrations/qwen30.json" not in qwen32_output


def test_matched_recipe_geometry_and_provenance_are_explicit() -> None:
    output = _dry_run("qwen32b", "always_on_v2_k5")

    for expected in (
        "grpo.max_num_steps=20",
        "checkpointing.enabled=false",
        "grpo.num_prompts_per_step=64",
        "grpo.num_generations_per_prompt=32",
        "policy.train_global_batch_size=512",
        "policy.max_total_sequence_length=4096",
        "policy.generation.max_new_tokens=4096",
        "policy.generation.vllm_cfg.max_model_len=4128",
        "max_num_batched_tokens=16384",
        "max_num_seqs=1024",
        "tensor_parallel_size=2",
        "draft_tensor_parallel_size=1",
        "moe_backend=triton",
        "WANDB_RUN_GROUP=contract",
        "logger.wandb.project=nemorl-vllm024-tail-gated-lyris",
        "logger.wandb.name=contract-attempt-1-qwen32b-always_on_v2_k5",
        "BASE_LOG_DIR=/lustre/test/tail-gate-runs/qwen32b/always_on_v2_k5",
    ):
        assert expected in output


def test_long_output_length_derives_engine_length_with_lookahead_headroom() -> None:
    output = _dry_run("qwen32b", "always_on_v2_k5", MAX_OSL="32768")

    assert "policy.generation.max_new_tokens=32768" in output
    assert "policy.generation._output_max_model_len=32768" in output
    assert "policy.generation.vllm_cfg.max_model_len=32800" in output
    assert "policy.generation.vllm_cfg.max_model_len=4128" not in output


@pytest.mark.parametrize(
    ("model", "variant"),
    [
        ("qwen30ba3b", "baseline_v1"),
        ("qwen32b", "baseline_v2"),
        ("qwen32b", "fastrl_threshold_v2_k5"),
    ],
)
def test_all_cohorts_enable_cuda_graph_and_vllm_metric_collection(
    model: str, variant: str
) -> None:
    output = _dry_run(model, variant)

    assert "policy.generation.vllm_cfg.enable_vllm_metrics_logger=true" in output
    assert "policy.generation.vllm_cfg.vllm_metrics_logger_interval=0.5" in output
    assert (
        "policy.generation.vllm_cfg.env_vars.NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS=true"
        in output
    )
    assert "env VLLM_USE_V2_MODEL_RUNNER=" in output
    assert (
        "NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS=true"
        in output.split("/opt/nemo_rl_venv/bin/python", maxsplit=1)[0]
    )


def test_test_only_uses_lyris_scheduler_without_gres(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\nprintf 'sbatch-args'\nprintf ' %q' \"$@\"\nprintf '\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()

    result = _run_launcher(
        "test-only",
        "qwen32b",
        "baseline_v2",
        REPO_DIR=str(REPO_ROOT),
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER=str(container),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    assert "sbatch-args --test-only" in result.stdout
    assert "--segment=4" in result.stdout
    assert "--gres" not in result.stdout


def _create_pushed_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=sna/tail-gate", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    for command in (
        [
            "git",
            "-C",
            str(repo),
            "config",
            "user.email",
            "launcher-test@example.invalid",
        ],
        ["git", "-C", str(repo), "config", "user.name", "Launcher Test"],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True)
    for path in (
        repo / "ray.sub",
        repo / "experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh",
        repo / "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        repo / "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    subprocess.run(
        ["git", "-C", str(repo), "add", "."], check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "test: seed launcher repo"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "push", "--set-upstream", "origin", "sna/tail-gate"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


def _write_roofline_config(
    path: Path,
    *,
    model: str,
    target_tp: int,
    draft_tp: int,
    container: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "hardware": {"gpu": "GB200", "tp": target_tp, "BW_eff": 1.0},
                "model": {
                    "name": model,
                    "W_t": 1.0,
                    "W_d": 1.0,
                    "C_dense": 1.0,
                    "C_attn": 1.0,
                    "kappa_theoretical": 1,
                },
                "calibration": {
                    "eta_d": 1.0,
                    "kappa_eff": 1.0,
                    "F_eff": 1.0,
                    "per_gamma": {"5": {"c_T": 1.0, "c_D": 1.0, "c_V": 1.0}},
                },
                "metadata": {
                    "model": model,
                    "target_tp": target_tp,
                    "draft_tp": draft_tp,
                    "container": str(container),
                    "container_sha256": hashlib.sha256(
                        container.read_bytes()
                    ).hexdigest(),
                    "target_checkpoint_revision": QWEN32_TARGET_REVISION,
                    "draft_checkpoint_revision": QWEN32_DRAFT_REVISION,
                    "calibration_timestamp": CALIBRATION_TIMESTAMP,
                    "cluster": CALIBRATION_CLUSTER,
                    "vllm_commit": VLLM_COMMIT,
                    "k_values": [1, 3, 5],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_submit_records_complete_manifest_and_does_not_push(tmp_path: Path) -> None:
    repo, commit = _create_pushed_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\nprintf '4242\\n'\n", encoding="utf-8")
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.write_text("container contract\n", encoding="utf-8")
    roofline = tmp_path / "roofline.json"
    _write_roofline_config(
        roofline,
        model="Qwen/Qwen3-32B",
        target_tp=2,
        draft_tp=1,
        container=container,
    )
    draft_model = tmp_path / "draft-model"
    draft_model.mkdir()
    experiment_root = tmp_path / "runs"

    result = _run_launcher(
        "submit",
        "qwen32b",
        "efficient_roofline_v2_k5",
        REPO_DIR=str(repo),
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER=str(container),
        QWEN32_DRAFT_MODEL=str(draft_model),
        QWEN32_ROOFLINE_CONFIG=str(roofline),
        EXPERIMENT_ROOT=str(experiment_root),
        RUN_TAG="submit-contract",
        ATTEMPT_ID="attempt-1",
        QWEN32_CALIBRATION_TIMESTAMP=CALIBRATION_TIMESTAMP,
        WANDB_API_KEY="test-only-key",
        RUNTIME_VERSION=RUNTIME_VERSION,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    manifest_path = experiment_root / "submissions.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(manifest_rows) == 1
    manifest_row = manifest_rows[0]
    assert {
        "cluster",
        "runtime",
        "runtime_version",
        "runtime_commit",
        "vllm_version",
        "vllm_commit",
        "target_tp",
        "draft_tp",
        "dp",
        "ep",
        "temperature",
        "top_p",
        "max_osl",
        "max_model_len",
        "max_sequence_length",
        "num_prompts",
        "num_generations",
        "train_gbs",
        "max_num_batched_tokens",
        "max_num_seqs",
        "sampling",
        "runner",
        "graph_mode",
        "gate_mode",
        "k",
        "threshold",
        "consecutive_checks",
        "roofline_config_sha256",
        "container",
        "container_sha256",
        "recipe",
        "job_id",
    }.issubset(manifest_row)
    expected_manifest_values = {
        "model": "qwen32b",
        "variant": "efficient_roofline_v2_k5",
        "gate_mode": "roofline",
        "k": "5",
        "threshold": "32",
        "consecutive_checks": "10",
        "cluster": "lyris-gb200",
        "runtime": "nemo-rl",
        "runtime_version": RUNTIME_VERSION,
        "runtime_commit": commit,
        "vllm_version": VLLM_VERSION,
        "vllm_commit": VLLM_COMMIT,
        "target_tp": "2",
        "draft_tp": "1",
        "dp": "8",
        "ep": "1",
        "temperature": "1.0",
        "top_p": "1.0",
        "max_osl": "4096",
        "max_model_len": "4128",
        "max_sequence_length": "4096",
        "num_prompts": "64",
        "num_generations": "32",
        "train_gbs": "512",
        "max_num_batched_tokens": "16384",
        "max_num_seqs": "1024",
        "runner": "v2",
        "graph_mode": "FULL_AND_PIECEWISE",
        "sampling": "standard",
        "job_id": "4242",
    }
    assert expected_manifest_values.items() <= manifest_row.items()
    assert (
        manifest_row["roofline_config_sha256"]
        == hashlib.sha256(roofline.read_bytes()).hexdigest()
    )
    assert "test-only-key" not in manifest_path.read_text(encoding="utf-8")
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_submit_records_environment_gate_settings_in_manifest(tmp_path: Path) -> None:
    repo, _commit = _create_pushed_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\nprintf '4242\\n'\n", encoding="utf-8")
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.write_text("container contract\n", encoding="utf-8")
    draft_model = tmp_path / "draft-model"
    draft_model.mkdir()
    experiment_root = tmp_path / "runs"

    result = _run_launcher(
        "submit",
        "qwen32b",
        "fastrl_threshold_v2_k5",
        REPO_DIR=str(repo),
        CONTAINER=str(container),
        QWEN32_DRAFT_MODEL=str(draft_model),
        EXPERIMENT_ROOT=str(experiment_root),
        WANDB_API_KEY="test-only-key",
        TAIL_GATE_THRESHOLD="19",
        TAIL_GATE_CONSECUTIVE_CHECKS="4",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    with (experiment_root / "submissions.tsv").open(
        encoding="utf-8", newline=""
    ) as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert row["threshold"] == "19"
    assert row["consecutive_checks"] == "4"
    assert "sd_tail_gate_threshold=19" in row["command"]
    assert "sd_tail_gate_consecutive_checks=4" in row["command"]


def test_submit_records_long_output_engine_length_as_separate_cohort(
    tmp_path: Path,
) -> None:
    repo, _commit = _create_pushed_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\nprintf '5252\\n'\n", encoding="utf-8")
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.write_text("container contract\n", encoding="utf-8")
    experiment_root = tmp_path / "runs"

    result = _run_launcher(
        "submit",
        "qwen32b",
        "baseline_v2",
        REPO_DIR=str(repo),
        CONTAINER=str(container),
        EXPERIMENT_ROOT=str(experiment_root),
        MAX_OSL="32768",
        WANDB_API_KEY="test-only-key",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    with (experiment_root / "submissions.tsv").open(
        encoding="utf-8", newline=""
    ) as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert row["max_osl"] == "32768"
    assert row["max_model_len"] == "32800"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model", "Qwen/Qwen3-30B-A3B"),
        ("target_tp", 1),
        ("draft_tp", 2),
        ("container", "/lustre/other/container.sqsh"),
        ("container_sha256", "0" * 64),
        ("target_checkpoint_revision", "3" * 40),
        ("draft_checkpoint_revision", "4" * 40),
        ("calibration_timestamp", "2026-07-11T12:34:56Z"),
        ("cluster", "oci-hsg-gb200"),
        ("vllm_commit", "different-vllm-commit"),
        ("k_values", [1, 3]),
        ("k_values", [1, 3, "5"]),
    ),
)
def test_submit_rejects_mismatched_roofline_metadata_before_sbatch(
    tmp_path: Path, field: str, value: object
) -> None:
    repo, _commit = _create_pushed_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$SBATCH_LOG\"\nprintf '4242\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.write_text("container contract\n", encoding="utf-8")
    draft_model = tmp_path / "draft-model"
    draft_model.mkdir()
    roofline = tmp_path / "roofline.json"
    _write_roofline_config(
        roofline,
        model="Qwen/Qwen3-32B",
        target_tp=2,
        draft_tp=1,
        container=container,
    )
    payload = json.loads(roofline.read_text(encoding="utf-8"))
    payload["metadata"][field] = value
    roofline.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _run_launcher(
        "submit",
        "qwen32b",
        "efficient_roofline_v2_k5",
        REPO_DIR=str(repo),
        CONTAINER=str(container),
        QWEN32_DRAFT_MODEL=str(draft_model),
        QWEN32_ROOFLINE_CONFIG=str(roofline),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        QWEN32_CALIBRATION_TIMESTAMP=CALIBRATION_TIMESTAMP,
        WANDB_API_KEY="test-only-key",
        SBATCH_LOG=str(sbatch_log),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 2
    assert f"roofline metadata mismatch: {field}" in result.stderr
    assert not sbatch_log.exists()


def test_submit_requires_exact_per_gamma_k5_before_sbatch(tmp_path: Path) -> None:
    repo, _commit = _create_pushed_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$SBATCH_LOG\"\nprintf '4242\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.write_text("container contract\n", encoding="utf-8")
    draft_model = tmp_path / "draft-model"
    draft_model.mkdir()
    roofline = tmp_path / "roofline.json"
    _write_roofline_config(
        roofline,
        model="Qwen/Qwen3-32B",
        target_tp=2,
        draft_tp=1,
        container=container,
    )
    payload = json.loads(roofline.read_text(encoding="utf-8"))
    payload["calibration"]["per_gamma"] = {"3": {"c_T": 1.0, "c_D": 1.0, "c_V": 1.0}}
    roofline.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _run_launcher(
        "submit",
        "qwen32b",
        "efficient_roofline_v2_k5",
        REPO_DIR=str(repo),
        CONTAINER=str(container),
        QWEN32_DRAFT_MODEL=str(draft_model),
        QWEN32_ROOFLINE_CONFIG=str(roofline),
        QWEN32_CALIBRATION_TIMESTAMP=CALIBRATION_TIMESTAMP,
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        WANDB_API_KEY="test-only-key",
        SBATCH_LOG=str(sbatch_log),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 2
    assert 'roofline config requires exact calibration.per_gamma["5"]' in result.stderr
    assert not sbatch_log.exists()


def test_submit_rejects_malformed_manifest_before_real_sbatch(tmp_path: Path) -> None:
    repo, _commit = _create_pushed_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$SBATCH_LOG\"\nprintf '4242\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    experiment_root = tmp_path / "runs"
    experiment_root.mkdir()
    (experiment_root / "submissions.tsv").write_text(
        "wrong\theader\n", encoding="utf-8"
    )

    result = _run_launcher(
        "submit",
        "qwen30ba3b",
        "baseline_v1",
        REPO_DIR=str(repo),
        CONTAINER=str(container),
        EXPERIMENT_ROOT=str(experiment_root),
        WANDB_API_KEY="test-only-key",
        SBATCH_LOG=str(sbatch_log),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 2
    assert "submissions manifest header mismatch" in result.stderr
    assert not sbatch_log.exists()


def test_submit_ignores_known_unit_artifacts_but_rejects_other_untracked_files(
    tmp_path: Path,
) -> None:
    repo, _commit = _create_pushed_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\nprintf '4242\\n'\n", encoding="utf-8")
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    (repo / "tests/unit/unit_results").mkdir(parents=True)
    (repo / "tests/unit/unit_results.json").write_text("{}\n", encoding="utf-8")
    (repo / "tests/unit/unit_results/result.json").write_text("{}\n", encoding="utf-8")

    result = _run_launcher(
        "submit",
        "qwen30ba3b",
        "baseline_v1",
        REPO_DIR=str(repo),
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER=str(container),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        WANDB_API_KEY="test-only-key",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )
    assert result.returncode == 0, result.stderr
    assert "4242\tqwen30ba3b\tbaseline_v1" in result.stdout

    (repo / "tests/unit/unit_results.json.bak").write_text(
        "not allowed\n", encoding="utf-8"
    )
    rejected = _run_launcher(
        "submit",
        "qwen30ba3b",
        "baseline_v1",
        REPO_DIR=str(repo),
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER=str(container),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        WANDB_API_KEY="test-only-key",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )
    assert rejected.returncode == 2
    assert "untracked" in rejected.stderr
    assert "unit_results.json.bak" in rejected.stderr


def test_submit_rejects_dirty_submodules(tmp_path: Path) -> None:
    repo, _commit = _create_pushed_repo(tmp_path)
    submodule_source = tmp_path / "submodule-source"
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(submodule_source)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(submodule_source),
            "config",
            "user.email",
            "test@example.invalid",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(submodule_source), "config", "user.name", "Test"], check=True
    )
    tracked = submodule_source / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(submodule_source), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(submodule_source), "commit", "-m", "seed"], check=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(repo),
            "submodule",
            "add",
            str(submodule_source),
            "third_party/component",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-am", "add submodule"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "push"], check=True, capture_output=True)
    (repo / "third_party/component/tracked.txt").write_text("dirty\n", encoding="utf-8")
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()

    result = _run_launcher(
        "submit",
        "qwen30ba3b",
        "baseline_v1",
        REPO_DIR=str(repo),
        CONTAINER=str(container),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        WANDB_API_KEY="test-only-key",
    )

    assert result.returncode == 2
    assert "clean tracked checkout" in result.stderr
