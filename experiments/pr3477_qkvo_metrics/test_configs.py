import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "pr3477_qkvo_metrics"


def load(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    default = config.get("defaults")
    if not isinstance(default, str):
        return config
    return merge(load(path.parent / default), config)


def merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def patterns(name: str) -> list[str]:
    config = load(EXPERIMENT / name)
    return config["policy"]["generation"]["vllm_cfg"][
        "quantization_ignore_patterns"
    ]


def test_qwen_qkvo_only_removes_attention_exclusions() -> None:
    baseline = patterns("qwen30_moe_only.yaml")
    qkvo = patterns("qwen30_qkvo.yaml")
    assert set(baseline) - set(qkvo) == {"model.layers.*.self_attn.*"}
    assert set(qkvo) == {"model.layers.*.mlp.gate", "lm_head"}


def test_nano_qkvo_only_removes_attention_exclusions() -> None:
    baseline = patterns("nano_moe_only.yaml")
    qkvo = patterns("nano_qkvo.yaml")
    assert set(baseline) - set(qkvo) == {
        "model.layers.*.mixer.qkv_proj",
        "model.layers.*.mixer.o_proj",
    }
    assert "model.layers.*.mixer.gate" in qkvo
    assert "model.layers.*.mixer.shared_experts.*" in qkvo
    assert "lm_head" in qkvo


def test_submitter_is_matched_and_uses_cuda_graphs() -> None:
    submitter = (EXPERIMENT / "submit_oci_hsg.sh").read_text(encoding="utf-8")
    assert "MAX_STEPS=${MAX_STEPS:-20}" in submitter
    assert "cluster.segment_size=2" in submitter
    assert "grpo.seed=42" in submitter
    assert "policy.generation.vllm_cfg.enforce_eager=false" in submitter
    assert "policy.generation.refit_transport=nccl_reshard" in submitter
    assert "audit_scope.py" in submitter
    assert "aggregate_steps=3-20" in submitter


def test_scope_audit_does_not_require_vllm_at_import_time() -> None:
    module_path = EXPERIMENT / "audit_scope.py"
    spec = importlib.util.spec_from_file_location("pr3477_scope_audit", module_path)
    assert spec is not None
    assert spec.loader is not None

    original_vllm = sys.modules.pop("vllm", None)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if original_vllm is not None:
            sys.modules["vllm"] = original_vllm

    patterns = ["model.layers.*.self_attn.*", "lm_head"]
    assert module.excluded(patterns, "model.layers.7.self_attn.qkv_proj")
    assert module.excluded(patterns, "lm_head")
    assert not module.excluded(patterns, "model.layers.7.mlp.experts")
