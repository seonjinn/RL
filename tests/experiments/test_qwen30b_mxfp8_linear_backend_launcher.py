from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT / "experiments" / "qwen30b_mxfp8_linear_backends" / "submit_ptyche.sh"
)
PREPARE_SCRIPT = LAUNCHER.with_name("prepare_custom_vllm_ptyche.sh")
BUILD_CUSTOM_VLLM_SCRIPT = REPO_ROOT / "tools" / "build-custom-vllm.sh"


def _dry_run(
    tmp_path: Path, backend: str, extra_env: dict[str, str] | None = None
) -> str:
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    custom_vllm = tmp_path / "vllm"
    custom_vllm.mkdir(exist_ok=True)
    (custom_vllm / ".git").mkdir(exist_ok=True)

    env = os.environ | {
        "ACTION": "dry-run",
        "BACKEND": backend,
        "CONTAINER": str(container),
        "CUSTOM_VLLM_ROOT": str(custom_vllm),
        "EXPERIMENT_ROOT": str(tmp_path / backend),
        "QOS": "interactive",
        "WANDB_MODE": "disabled",
        "WORK_ROOT": str(tmp_path),
    }
    if extra_env is not None:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_dry_run_changes_only_backend(tmp_path: Path) -> None:
    outputs = {
        backend: _dry_run(tmp_path, backend)
        for backend in (
            "flashinfer_cutedsl",
            "flashinfer_cutlass",
            "flashinfer_trtllm",
            "flashinfer_trtllm_adaptive",
        )
    }

    for backend, output in outputs.items():
        effective_backend = (
            "flashinfer_trtllm" if backend == "flashinfer_trtllm_adaptive" else backend
        )
        assert f"linear_backend={effective_backend}" in output
        assert "policy.train_global_batch_size=2048" in output
        assert "policy.generation.vllm_cfg.enforce_eager=false" in output
        assert "quantization_ignored_layer_kws=[lm_head,mlp.gate]" in output
        assert "moe_backend=flashinfer_trtllm" in output
        assert "cluster.num_nodes=4" in output
        assert "cluster.gpus_per_node=4" in output
        assert "cluster.segment_size=4" in output
        assert "grpo.max_num_steps=8" in output
        assert (
            "NRL_VENV_BOOTSTRAP_PACKAGES='--torch-backend cu130 "
            "torch==2.11.0 numpy setuptools setuptools-rust setuptools-scm'" in output
        )
        assert "SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1" in output
        assert "--qos=interactive" in output
        assert output.count("uv run --frozen --extra vllm") == 2
        assert f"uv venv {tmp_path}" in output
        assert "uv pip install --python" in output
        assert "setuptools_rust" in output

    adaptive_output = outputs["flashinfer_trtllm_adaptive"]
    assert "VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK=1" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_LAYOUT=adaptive" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M=256" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE=" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256=" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64=" in adaptive_output

    for backend in (
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
        "flashinfer_trtllm",
    ):
        assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE=" not in outputs[backend]


def test_rejects_unknown_backend(tmp_path: Path) -> None:
    env = os.environ | {
        "ACTION": "dry-run",
        "BACKEND": "auto",
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Unsupported BACKEND" in result.stderr


def test_long_context_overrides_are_forwarded(tmp_path: Path) -> None:
    output = _dry_run(
        tmp_path,
        "flashinfer_trtllm_adaptive",
        {
            "MAX_STEPS": "20",
            "MAX_TOTAL_SEQUENCE_LENGTH": "34816",
            "MAX_NEW_TOKENS": "32768",
            "MAX_INPUT_SEQUENCE_LENGTH": "2048",
            "NUM_PROMPTS_PER_STEP": "64",
            "NUM_GENERATIONS_PER_PROMPT": "4",
            "TRAIN_GLOBAL_BATCH_SIZE": "256",
            "ACTIVATION_CHECKPOINTING": "true",
            "SEQUENCE_PACKING": "false",
            "LOGPROB_BATCH_SIZE": "1",
            "LOGPROB_CHUNK_SIZE": "2048",
        },
    )

    assert "grpo.max_num_steps=20" in output
    assert "policy.max_total_sequence_length=34816" in output
    assert "policy.generation.max_new_tokens=32768" in output
    assert "policy.generation.vllm_cfg.max_model_len=34816" in output
    assert "data.max_input_seq_length=2048" in output
    assert "grpo.num_prompts_per_step=64" in output
    assert "grpo.num_generations_per_prompt=4" in output
    assert "policy.train_global_batch_size=256" in output
    assert "policy.megatron_cfg.activation_checkpointing=true" in output
    assert "policy.sequence_packing.enabled=false" in output
    assert "policy.logprob_batch_size=1" in output
    assert "policy.logprob_chunk_size=2048" in output


def test_custom_vllm_build_is_recoverable() -> None:
    prepare_text = PREPARE_SCRIPT.read_text()
    build_text = BUILD_CUSTOM_VLLM_SCRIPT.read_text()
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text()

    assert "3rdparty/vllm/nemo-rl.env" in prepare_text
    assert "vllm.incomplete" in prepare_text
    assert "git submodule update --init --recursive --depth 1" in prepare_text
    assert "3rdparty/vllm/.venv/bin/python -c 'import vllm'" in prepare_text
    assert "3rdparty/vllm/.venv/bin/python - <<'PY'" in prepare_text
    assert "uv run --frozen python - <<'PY'" not in prepare_text
    assert "3rdparty/vllm/.venv uv lock" in prepare_text
    assert "SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1" in prepare_text
    assert "setuptools_rust" in build_text
    assert "existing_vllm_valid=false" in prepare_text
    assert "Replacing custom vLLM commit" in prepare_text
    assert "3rdparty/vllm/.venv/bin/python -c 'import vllm'" in prepare_text
    assert 'SBATCH_ARGS+=(--qos="${QOS}")' in prepare_text
    assert "TORCH_REQUIREMENT=$(sed -nE" in build_text
    assert "VLLM_TORCH_BACKEND:-cu130" in build_text
    assert "torch==2.10.0" not in build_text
    assert 'vllm = ["setuptools", "setuptools-rust"]' in pyproject_text
