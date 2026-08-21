from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.qwen3_8b_dspark_online_cp1 import resume_contract
from research.qwen3_8b_dspark_online_cp1.validate_gate import validate_gate


EXPERIMENT = Path(__file__).parents[1]
UPDATE = (
    "draft_update_probe=complete grad_l2=0.25 "
    "checksum_sum_before=10 checksum_sum_after=10.125 "
    "checksum_l2_before=20 checksum_l2_after=20 delta=0.125\n"
)
REFIT = (
    "draft_refit_manifest=draft_count=17\n"
    "draft_refit_load=complete\n"
    "draft_refit_finalize=complete\n"
)


def _checkpoint(root: Path, step: int) -> None:
    step_dir = root / f"step_{step}"
    weights = step_dir / "policy" / "weights"
    weights.mkdir(parents=True)
    (weights / "latest_train_state.pt").write_bytes(b"state")
    (step_dir / "train_dataloader.pt").write_bytes(b"loader")
    (step_dir / "training_info.json").write_text(
        json.dumps(
            {"current_step": step, "total_steps": step, "consumed_samples": 8 * step}
        )
    )
    (step_dir / "config.yaml").write_text("grpo:\n  max_num_steps: 1000\n")


def test_gate_requires_two_updates_and_two_live_refits() -> None:
    metrics = {
        "train/draft_grad_norm": {"2": 0.25},
        "train/draft_loss": {"2": 1.5},
        "train/vllm/spec_acceptance_rate": {"2": 0.41},
    }
    validate_gate(metrics, 2 * (UPDATE + REFIT))

    with pytest.raises(RuntimeError, match="two proven"):
        validate_gate(metrics, UPDATE + REFIT)


@pytest.mark.parametrize("num_speculative_tokens", [5, 7])
def test_resume_manifest_binds_dspark_arm_and_one_wandb_id(
    tmp_path: Path,
    num_speculative_tokens: int,
) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, 2)
    manifest = tmp_path / "gate-manifest.json"
    identity = {
        "git_sha": "a" * 40,
        "checkpoint_root": root,
        "target_revision": "b968",
        "drafter_revision": "03326",
        "container_sha256": "6940",
        "num_speculative_tokens": num_speculative_tokens,
    }

    resume_contract.validate_checkpoint(root, expected_step=2)
    resume_contract.write_manifest(manifest, wandb_run_id="fresh123", **identity)
    payload = resume_contract.validate_manifest(manifest, **identity)

    assert payload["wandb_run_id"] == "fresh123"
    assert payload["speculator_type"] == "dspark"
    assert payload["num_speculative_tokens"] == num_speculative_tokens

    smoke_proof = tmp_path / "smoke-proof.json"
    resume_contract.write_smoke_proof(smoke_proof, wandb_run_id="smoke123", **identity)
    proof = resume_contract.validate_smoke_proof(smoke_proof, **identity)
    assert proof["status"] == "complete"
    assert proof["validated_steps"] == 2
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        resume_contract.validate_smoke_proof(
            smoke_proof,
            **{
                **identity,
                "num_speculative_tokens": 7 if num_speculative_tokens == 5 else 5,
            },
        )


def test_launcher_pins_runtime_storage_wandb_and_dependency_contract() -> None:
    runner = (EXPERIMENT / "run_segment_oci_hsg.sbatch").read_text()
    submit = (EXPERIMENT / "submit_chain.sh").read_text()

    assert "openai-2.25.0-py3-none-any.whl#sha256=" in runner
    assert 'm.version("vllm") == "0.25.1"' in runner
    assert "from openai.types.responses import NamespaceTool" in runner
    assert 'readonly scratch_root="/raid/scratch/' in runner
    assert 'readonly actor_venv_root="${scratch_root}/actor-venvs"' in runner
    assert "export UV_CACHE_DIR='${scratch_root}/cache/uv'" in runner
    assert "export UV_LINK_MODE=symlink" in runner
    assert "export NEMO_RL_VENV_DIR='${actor_venv_root}'" in runner
    assert (
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
        in runner
    )
    assert "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker" in runner
    assert "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor" in runner
    assert "uv sync --frozen --extra mcore --no-install-project" in runner
    assert (
        "'${actor_venv_root}/${mcore_actor}/bin/python' -c "
        "'import megatron.core'\n"
        "echo 'MCORE_ACTOR_PREFLIGHT=complete'"
    ) in runner
    assert 'print("MCORE_ACTOR_PREFLIGHT=complete")' not in runner
    assert runner.count("uv sync --frozen --extra vllm --no-install-project") == 2
    assert "--extra mcore --extra vllm" not in runner
    assert "--no-install-package deep-ep --no-install-package deep-gemm" in runner
    assert (
        "UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync" in runner
    )
    assert (
        'readonly scheduler_log="/raid/scratch/nrl-dspark-online-${SLURM_JOB_ID}.out"'
        in runner
    )
    assert '[[ "${REMOTE_REPO}" == /home/* ]]' in runner
    assert '[[ "${FINAL_DIR}" == /lustre/* ]]' in runner
    assert "nvidia/sna-nemo-rl-online-drafter" in runner
    assert "WANDB_API_KEY" in runner
    assert "set -x" not in runner
    assert "update_probe_enabled=true" in runner
    assert "update_probe_enabled=false" in runner
    assert "policy.draft.update_probe_enabled='${update_probe_enabled}'" in runner
    assert 'elif test -f "${scheduler_log}"; then' in runner
    assert 'tail -n 4000 "${scheduler_log}"' in runner
    assert "afterok:${previous}" in submit
    assert "--kill-on-invalid-dep=yes" in submit
    assert 'readonly arm="${2:?Set dspark-k5 or dspark-k7}"' in submit
    assert "dspark-k5) num_speculative_tokens=5" in submit
    assert "dspark-k7) num_speculative_tokens=7" in submit
    assert "ARM_NAME=${arm}" in submit
    assert "NUM_SPECULATIVE_TOKENS=${num_speculative_tokens}" in submit
    assert ': "${ARM_NAME:?Set dspark-k5 or dspark-k7}"' in runner
    assert ': "${NUM_SPECULATIVE_TOKENS:?Set K=5 or K=7}"' in runner
    assert "num_speculative_tokens='${NUM_SPECULATIVE_TOKENS}'" in runner
    assert "logger.wandb.group='qwen3-8b-dapomath17k-online-matrix-v1'" in runner
    assert "logger.wandb.name='qwen3-8b-${ARM_NAME}-online-seed42'" in runner
    assert "logger.wandb.config.k='${NUM_SPECULATIVE_TOKENS}'" in runner
    assert "run_max_steps=2" in runner
    assert "checkpointing_enabled=false" in runner
    assert "grpo.max_num_steps='${run_max_steps}'" in runner
    assert "checkpointing.enabled='${checkpointing_enabled}'" in runner
    assert 'job_id="$(submit smoke "" 04:00:00 2' in submit
    assert 'previous="$(submit start "${previous}" 04:00:00 350' in submit
    assert 'smoke_job="$(submit smoke "" 04:00:00 2' in submit
    assert 'submit_science_chain "${smoke_job}" "${run_id}"' in submit
    assert '"${previous}" 04:00:00 "${milestone}"' in submit
    assert "350 00:03:30:00" in submit
    assert "700 00:03:30:00" in submit
    assert "1000 00:03:30:00" in submit


def test_smoke_is_not_reused_as_a_science_checkpoint() -> None:
    runner = (EXPERIMENT / "run_segment_oci_hsg.sbatch").read_text()
    submit = (EXPERIMENT / "submit_chain.sh").read_text()

    assert 'if [[ "${STAGE_MODE}" == smoke ]]; then' in runner
    assert 'echo "WANDB_URL=' in runner
    assert "exit 0" in runner
    assert "--print-manifest-wandb-id" not in submit
    assert 'elif [[ "${mode}" == start ]]' in submit
    assert "--write-smoke-proof" in runner
    assert "--validate-smoke-proof" in submit
    assert 'readonly smoke_proof="${FINAL_DIR}/smoke-proof.json"' in runner
    assert "usage: $0 smoke ARM | $0 start ARM | $0 all ARM" in submit


def test_smoke_validator_runs_inside_the_pyxis_runtime() -> None:
    runner = (EXPERIMENT / "run_segment_oci_hsg.sbatch").read_text()

    container_body, outer_body = runner.split('"\n\ngrep -Fq "Capturing CUDA graphs', 1)
    assert (
        "if [[ '${STAGE_MODE}' == smoke ]]; then\n"
        "  /opt/nemo_rl_venv/bin/python '${experiment}/validate_gate.py'"
    ) in container_body
    assert "/opt/nemo_rl_venv/bin/python" not in outer_body
