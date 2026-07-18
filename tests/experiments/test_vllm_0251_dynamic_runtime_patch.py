from pathlib import Path

from experiments.vllm_0251_eagle3_perfcfg.apply_vllm0251_dynamic_sd_cg_fix import (
    apply_ray_worker_environment_patch,
)


def test_dynamic_k0_returns_only_after_the_drafter_prefill_anchor() -> None:
    patcher = (
        Path(__file__).parents[2]
        / "experiments/vllm_0251_eagle3_perfcfg"
        / "apply_vllm0251_dynamic_sd_cg_fix.py"
    )
    source = patcher.read_text(encoding="utf-8")

    prefill_complete_anchor = source.index(
        '"        if self.num_speculative_steps == 1:\\n"'
    )
    dynamic_k0_return = source.index(
        '"        if selected_num_speculative_steps == 0:\\n"'
    )

    assert dynamic_k0_return > prefill_complete_anchor


def test_ray_workers_inherit_the_locked_profile_pythonpath(tmp_path: Path) -> None:
    ray_executor = tmp_path / "vllm/v1/executor/ray_executor_v2.py"
    ray_executor.parent.mkdir(parents=True)
    ray_executor.write_text(
        "        env_vars = runtime_env.setdefault(\"env_vars\", {})\n"
        "        env_vars.update(\n"
        "            {v: \"1\" for v in current_platform.ray_noset_device_env_vars}\n"
        "        )\n",
        encoding="utf-8",
    )

    assert apply_ray_worker_environment_patch(tmp_path) is True
    patched = ray_executor.read_text(encoding="utf-8")
    assert 'pythonpath := os.environ.get("PYTHONPATH")' in patched
    assert 'env_vars.setdefault("PYTHONPATH", pythonpath)' in patched
    assert apply_ray_worker_environment_patch(tmp_path) is False
