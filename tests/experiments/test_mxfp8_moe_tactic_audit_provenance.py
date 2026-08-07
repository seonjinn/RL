import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "experiments/mxfp8_moe_tactic_audit/README.md"
PROVENANCE = ROOT / "experiments/mxfp8_moe_tactic_audit/provenance.json"


def test_readme_pins_runtime_and_privacy_contract() -> None:
    text = README.read_text()
    assert "a76062edee3a3ac23d47a93c7ce466f06a19111f" in text
    assert "FlashInfer 0.6.13" in text
    assert "Ptyche GB200" in text
    assert "prompts, token IDs, hidden values, or model outputs" in text
    assert "CUDA Graphs" in text
    assert "1,319-example GSM8K" in text


def test_provenance_json_defines_launcher_contract() -> None:
    contract = json.loads(PROVENANCE.read_text())

    assert contract["schema_version"] == 1
    assert contract["vllm_commit"] == "a76062edee3a3ac23d47a93c7ce466f06a19111f"
    assert contract["flashinfer_version"] == "0.6.13"
    assert contract["model"] == "Qwen3-30B-A3B"
    assert contract["target"] == {"cluster": "Ptyche", "gpu": "GB200"}
    assert contract["topology_recipe"] == {
        "cluster": "Ptyche",
        "gpu": "GB200",
        "nodes": 4,
        "recipe": "current NeMo-RL MXFP8 performance recipe",
    }
    assert contract["privacy_forbidden_payloads"] == [
        "prompts",
        "token IDs",
        "hidden values",
        "model outputs",
        "credentials",
        "Hugging Face/W&B tokens",
    ]
    assert contract["minimum_profile_coverage"] == 0.95
    assert contract["warmups"] == 3
    assert contract["timed_repetitions_minimum"] >= 10
    assert contract["minimum_weighted_median_improvement"] == 0.02
    assert contract["maximum_coefficient_variation"] == 0.03
    assert contract["maximum_high_weight_profile_regression"] == 0.01
    assert contract["execution_modes"] == {
        "trace_collection": "eager only",
        "shmoo_replay": "CUDA Graphs required",
        "vllm_validation": "CUDA Graphs required",
        "nemorl_performance": "CUDA Graphs required",
    }
    assert contract["profile_scope"] == {
        "tactic_pairs": "every legal FC1/FC2 tactic pair",
        "inputs": "cold-L2 inputs",
    }
    assert contract["cuda_graphs_required"] is True
    assert contract["exact_miss_fallback"] == "stock FlashInfer behavior; cache misses are not errors"
    assert contract["matched_gsm8k_examples"] == 1319
    assert contract["validation_gates"] == [
        "micro-correctness",
        "CUDA Graph replay",
        "deterministic vLLM generation",
        "matched GSM8K",
        "NeMo-RL finite metrics",
    ]
    assert contract["pipeline_entry_points"] == [
        "submit_trace_ptyche.sh",
        "select_profiles.py",
        "shmoo_moe_tactics.py",
        "qualify_cache.py",
        "submit_validation_ptyche.sh",
        "build_report.py",
    ]
