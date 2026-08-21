import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_refit_perf_matrix"


def _module() -> ModuleType:
    path = EXPERIMENT_DIR / "analyze_wandb.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_uses_sums_after_warmup_and_reports_optional_peak_memory() -> None:
    analysis = _module()
    rows = [
        {
            "_step": step,
            "timing/train/total_step_time": total,
            "timing/train/policy_training": policy,
            "timing/train/weight_sync": refit,
            "timing/train/generation": generation,
            "train/total_num_tokens": tokens,
            "train/vllm/spec_acceptance_rate": acceptance,
            "train/draft_loss": draft_loss,
            "train/peak_memory_allocated_mb": peak,
        }
        for step, total, policy, refit, generation, tokens, acceptance, draft_loss, peak in [
            (0, 1000.0, 100.0, 100.0, 100.0, 9999.0, 0.1, 9.0, 99.0),
            (4, 1000.0, 100.0, 100.0, 100.0, 9999.0, 0.1, 9.0, 99.0),
            (5, 10.0, 5.0, 1.0, 3.0, 100.0, 0.4, 2.0, 10.0),
            (6, 20.0, 6.0, 2.0, 5.0, 200.0, 0.6, 1.0, 12.0),
            (50, 1000.0, 100.0, 100.0, 100.0, 9999.0, 0.1, 9.0, 99.0),
        ]
    ]

    summary = analysis.summarize_history(
        rows,
        cell="gbs32_mbs1_online",
        gbs=32,
        arm="online",
        evidence={"draft_update_count": 2, "draft_refit_count": 2},
    )

    assert summary["steps"] == 2
    assert summary["e2e_seconds_per_sample"] == 0.46875
    assert summary["e2e_seconds_per_token"] == 0.1
    assert summary["policy_seconds_mean"] == 5.5
    assert summary["refit_seconds_mean"] == 1.5
    assert summary["e2e_seconds_mean"] == 15.0
    assert summary["generation_tokens_per_second"] == 37.5
    assert summary["acceptance_rate_mean"] == 0.5
    assert summary["peak_memory_allocated_mb"] == 12.0
    assert summary["draft_loss_mean"] == 1.5
    assert summary["update_refit_correct"] is True


def test_fixed_summary_accepts_absent_draft_and_peak_metrics() -> None:
    analysis = _module()
    rows = [
        {
            "_step": 5,
            "timing/train/total_step_time": 8.0,
            "timing/train/policy_training": 4.0,
            "timing/train/weight_sync": 1.0,
            "timing/train/generation": 2.0,
            "train/total_num_tokens": 80.0,
            "train/vllm/spec_acceptance_rate": 0.5,
        }
    ]

    summary = analysis.summarize_history(
        rows,
        cell="gbs32_mbs1_fixed",
        gbs=32,
        arm="fixed",
        evidence={"draft_update_count": 0, "draft_refit_count": 0},
    )

    assert summary["peak_memory_allocated_mb"] is None
    assert summary["draft_loss_mean"] is None
    assert summary["update_refit_correct"] is True


def test_pair_comparison_uses_fixed_as_the_denominator() -> None:
    analysis = _module()

    comparison = analysis.compare_pair(
        {"cell": "gbs64_mbs2_fixed", "e2e_seconds_per_token": 0.2},
        {"cell": "gbs64_mbs2_online", "e2e_seconds_per_token": 0.25},
    )

    assert comparison == {
        "shape": "gbs64_mbs2",
        "fixed_e2e_seconds_per_token": 0.2,
        "online_e2e_seconds_per_token": 0.25,
        "online_overhead_percent": 25.0,
    }
