from pathlib import Path


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
