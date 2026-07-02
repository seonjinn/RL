from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "experiments/eagle3_online/submit_lyris_qwen235b_sync_eagle3_k7_k9_k11_recipe_step20_20260701.sh"
)


def test_dry_run_preserves_qwen235b_performance_recipe() -> None:
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "DRY_RUN": "true"},
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout

    assert output.count("[DRY-RUN] variant=") == 4
    assert "[DRY-RUN] variant=baseline" in output
    for k in (7, 9, 11):
        assert f"[DRY-RUN] variant=eagle3_k{k}" in output
        assert f"speculative_config.num_speculative_tokens={k}" in output

    assert output.count("grpo-qwen3-235b-32n4g.yaml") == 4
    assert output.count("grpo.max_num_steps=20") == 4
    assert output.count("policy.generation.vllm_cfg.enforce_eager=false") == 4
    assert output.count("checkpointing.checkpoint_dir=") == 4
    for variant in ("baseline", "eagle3_k7", "eagle3_k9", "eagle3_k11"):
        assert f"training_checkpoints/qwen235b_sync_{variant}" in output
    assert output.count("--nodes=32") == 4
    assert output.count("--segment=16") == 4
    assert output.count("--job-name=coreai_dlalgo_llm-specdec.q235-") == 4
    assert "--gres" not in output
    assert "--dependency" not in output

    forbidden_overrides = (
        "max_num_batched_tokens",
        "max_num_seqs",
        "attention_backend",
        "checkpointing.enabled",
        "grpo.val_period",
        "moe_token_dispatcher_type",
        "moe_flex_dispatcher_backend",
        "policy.max_total_sequence_length",
    )
    for override in forbidden_overrides:
        assert override not in output


def test_dry_run_uses_offline_eagle3_only_for_speculative_variants() -> None:
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "DRY_RUN": "true"},
        check=True,
        capture_output=True,
        text=True,
    )
    sections = completed.stdout.split("[DRY-RUN] variant=")[1:]
    by_variant = {section.splitlines()[0]: section for section in sections}

    assert "speculative_config" not in by_variant["baseline"]
    for k in (7, 9, 11):
        section = by_variant[f"eagle3_k{k}"]
        assert "speculative_config.method=eagle3" in section
        assert "speculative_config.draft_tensor_parallel_size=1" in section
        assert "policy.draft.enabled=false" in section


def test_dry_run_isolates_compilation_caches_from_wandb_home() -> None:
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "DRY_RUN": "true"},
        check=True,
        capture_output=True,
        text=True,
    )
    sections = completed.stdout.split("[DRY-RUN] variant=")[1:]

    for section in sections:
        variant = section.splitlines()[0]
        node_cache = (
            "/tmp/sna/"
            "20260701_lyris_qwen235b_sync_eagle3_k7_k9_k11_recipe_"
            f"step20_cudagraph_{variant}"
        )
        expected_cache_env = {
            "NODE_LOCAL_CACHE_ROOT": node_cache,
            "XDG_CACHE_HOME": f"{node_cache}/xdg",
            "VLLM_CACHE_ROOT": f"{node_cache}/vllm",
            "FLASHINFER_WORKSPACE_BASE": f"{node_cache}/flashinfer_workspace",
            "TORCHINDUCTOR_CACHE_DIR": f"{node_cache}/torchinductor",
            "TRITON_CACHE_DIR": f"{node_cache}/triton",
            "CUDA_CACHE_PATH": f"{node_cache}/cuda",
            "TORCH_EXTENSIONS_DIR": f"{node_cache}/torch_extensions",
            "PYTHONPYCACHEPREFIX": f"{node_cache}/pycache",
        }
        for name, value in expected_cache_env.items():
            assert f"export {name}='{value}'" in section

        assert "/wandb_netrc_home/.cache/" not in section
