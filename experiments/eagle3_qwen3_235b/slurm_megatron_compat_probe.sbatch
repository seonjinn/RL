#!/bin/bash
#SBATCH --job-name=q235b-megatron-compat-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
SPECDEC_RL_DIR="${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$REPO_ROOT:$REPO_ROOT,$SPECDEC_RL_DIR:$SPECDEC_RL_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR="$REPO_ROOT/experiments/eagle3_qwen3_235b/megatron_bridge_qwen3moe"
MEGATRON_BRIDGE_PLUGIN_DIR="${MEGATRON_BRIDGE_PLUGIN_DIR:-}"
if [[ -z "$MEGATRON_BRIDGE_PLUGIN_DIR" && -d "$DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR" ]]; then
  MEGATRON_BRIDGE_PLUGIN_DIR="$DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR"
fi
MEGATRON_BRIDGE_QWEN3MOE_PLUGIN="${MEGATRON_BRIDGE_QWEN3MOE_PLUGIN:-1}"
MEGATRON_BRIDGE_SRC="${MEGATRON_BRIDGE_SRC:-}"
MEGATRON_LM_SRC="${MEGATRON_LM_SRC:-}"
JSON_OUT="${JSON_OUT:-$ARTIFACT_ROOT/reports/megatron_compat_probe.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$ARTIFACT_ROOT/reports/megatron_compat_probe.md}"

SRUN_CONTAINER_ARGS=()
if [[ -n "$CONTAINER" ]]; then
  SRUN_CONTAINER_ARGS+=(--container-image="$CONTAINER")
  [[ -n "$MOUNTS" ]] && SRUN_CONTAINER_ARGS+=(--container-mounts="$MOUNTS")
  SRUN_CONTAINER_ARGS+=(--container-workdir="$SPECDEC_RL_DIR")
fi

mkdir -p logs "$(dirname "$JSON_OUT")" "$(dirname "$MARKDOWN_OUT")"

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 bash -lc '
  set -euo pipefail
  cd "'"$SPECDEC_RL_DIR"'"
  _extra_pythonpath=""
  for _path in "'"$MEGATRON_BRIDGE_PLUGIN_DIR"'" "'"$MEGATRON_BRIDGE_SRC"'" "'"$MEGATRON_LM_SRC"'" "'"$SPECDEC_RL_DIR"'"; do
    if [ -n "$_path" ]; then
      _extra_pythonpath="${_extra_pythonpath}${_path}:"
    fi
  done
  MEGATRON_BRIDGE_QWEN3MOE_PLUGIN="'"$MEGATRON_BRIDGE_QWEN3MOE_PLUGIN"'" \
  PYTHONPATH="${_extra_pythonpath}${PYTHONPATH:-}" \
  "'"$PYTHON_BIN"'" - "'"$JSON_OUT"'" "'"$MARKDOWN_OUT"'" "'"$MEGATRON_BRIDGE_PLUGIN_DIR"'" "'"$MEGATRON_BRIDGE_SRC"'" "'"$MEGATRON_LM_SRC"'" <<'"'"'PY'"'"'
import importlib
import json
import platform
import sys
import traceback
from inspect import signature
from pathlib import Path

json_out = Path(sys.argv[1])
markdown_out = Path(sys.argv[2])
megatron_bridge_plugin_dir = sys.argv[3]
megatron_bridge_src = sys.argv[4]
megatron_lm_src = sys.argv[5]

payload = {
    "overall_status": "pass",
    "python": sys.version,
    "platform": platform.platform(),
    "megatron_bridge_plugin_dir": megatron_bridge_plugin_dir,
    "megatron_bridge_src": megatron_bridge_src,
    "megatron_lm_src": megatron_lm_src,
    "modules": {},
    "api": {},
    "errors": [],
}


def record_error(scope: str, exc: BaseException) -> None:
    payload["overall_status"] = "fail"
    payload["errors"].append(
        {
            "scope": scope,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    )


try:
    import torch

    payload["torch"] = {
        "version": getattr(torch, "__version__", None),
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "file": getattr(torch, "__file__", None),
    }
except Exception as exc:  # pragma: no cover - probe diagnostics
    record_error("torch_import", exc)

for module_name in [
    "megatron.bridge",
    "megatron.bridge.models",
    "qwen3_moe_bridge_plugin",
    "nemo_rl.models.policy.workers.megatron_policy_worker",
    "nemo_rl.models.megatron.setup",
]:
    try:
        module = importlib.import_module(module_name)
        payload["modules"][module_name] = {
            "status": "pass",
            "file": getattr(module, "__file__", None),
        }
    except Exception as exc:
        payload["modules"][module_name] = {"status": "fail"}
        record_error(f"import:{module_name}", exc)

try:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.causal_bridge import CausalLMBridge

    supported_models = CausalLMBridge.list_supported_models()
    payload["api"]["autobridge_file"] = getattr(
        importlib.import_module("megatron.bridge"), "__file__", None
    )
    payload["api"]["causal_bridge_file"] = getattr(
        importlib.import_module("megatron.bridge.models.causal_bridge"), "__file__", None
    )
    payload["api"]["supported_models_sample"] = supported_models[:50]
    payload["api"]["qwen3_moe_bridge_registered"] = (
        "Qwen3MoeForCausalLM" in supported_models
    )
    if "Qwen3MoeForCausalLM" not in supported_models:
        raise RuntimeError(
            "Qwen3MoeForCausalLM is not registered in AutoBridge supported models"
        )
except Exception as exc:
    record_error("autobridge:qwen3_moe_registration", exc)

try:
    from megatron.bridge import AutoBridge
    from transformers import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        max_position_embeddings=1024,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
    )
    config.architectures = ["Qwen3MoeForCausalLM"]
    bridge = AutoBridge.from_hf_config(config)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 4
    provider.expert_model_parallel_size = 16
    provider.pipeline_model_parallel_size = 4
    provider.expert_tensor_parallel_size = 1
    before_finalize_etp = getattr(provider, "expert_tensor_parallel_size", None)
    payload["api"]["qwen3_moe_provider_smoke"] = {
        "bridge": type(bridge).__name__,
        "provider": type(provider).__name__,
        "has_finalize": callable(getattr(provider, "finalize", None)),
        "expert_tensor_parallel_size_before_finalize": before_finalize_etp,
        "num_layers": getattr(provider, "num_layers", None),
        "num_moe_experts": getattr(provider, "num_moe_experts", None),
        "moe_router_topk": getattr(provider, "moe_router_topk", None),
        "moe_ffn_hidden_size": getattr(provider, "moe_ffn_hidden_size", None),
    }
    if not callable(getattr(provider, "finalize", None)):
        raise RuntimeError("Qwen3 MoE provider does not expose finalize()")
    finalized_provider = provider.finalize()
    after_finalize_etp = getattr(
        finalized_provider, "expert_tensor_parallel_size", None
    )
    payload["api"]["qwen3_moe_provider_smoke"][
        "expert_tensor_parallel_size_after_finalize"
    ] = after_finalize_etp
    if after_finalize_etp != 1:
        raise RuntimeError(
            "Qwen3 MoE provider finalize() did not preserve expert_tensor_parallel_size=1"
        )
except Exception as exc:
    record_error("autobridge:qwen3_moe_provider_smoke", exc)

try:
    import qwen3_moe_bridge_plugin

    bridge = qwen3_moe_bridge_plugin.Qwen3MoEBridge()
    registry = bridge.mapping_registry()
    entries = []
    seen = set()

    def collect_mapping_entries(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if hasattr(value, "megatron_param"):
            entries.append(
                {
                    "type": type(value).__name__,
                    "megatron_param": getattr(value, "megatron_param", None),
                    "hf_param": getattr(value, "hf_param", None),
                }
            )
            return
        if isinstance(value, dict):
            for item in value.values():
                collect_mapping_entries(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                collect_mapping_entries(item)
            return
        if hasattr(value, "__dict__"):
            for item in vars(value).values():
                if isinstance(item, (dict, list, tuple, set)) or hasattr(
                    item, "megatron_param"
                ):
                    collect_mapping_entries(item)

    collect_mapping_entries(registry)
    layernorm_entries = [
        entry
        for entry in entries
        if entry["megatron_param"]
        == "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight"
    ]
    payload["api"]["qwen3_moe_mapping_registry"] = {
        "auto_mapping_available": getattr(qwen3_moe_bridge_plugin, "AutoMapping", None)
        is not None,
        "entry_count": len(entries),
        "layernorm_entries": layernorm_entries,
        "mapping_types_sample": [entry["type"] for entry in entries[:20]],
    }
    if getattr(qwen3_moe_bridge_plugin, "AutoMapping", None) is None:
        raise RuntimeError("Qwen3 MoE plugin did not import AutoMapping")
    if not layernorm_entries:
        raise RuntimeError("Qwen3 MoE mapping registry did not expose layernorm mapping")
    if any(entry["type"] != "AutoMapping" for entry in layernorm_entries):
        raise RuntimeError(
            f"Qwen3 MoE layernorm mapping is not AutoMapping: {layernorm_entries}"
        )
    from megatron.bridge.models.param_mapping import TPAwareMapping

    grouped_linear_detection = {}
    for module_name, expected_parallelism in {
        "TEColumnParallelGroupedLinear": "replicated",
        "TERowParallelGroupedLinear": "replicated",
    }.items():
        module = type(module_name, (torch.nn.Module,), {})()
        mapping = TPAwareMapping(megatron_param="dummy.weight", hf_param="dummy")
        grouped_linear_detection[module_name] = mapping._detect_parallelism_type(module)
        if grouped_linear_detection[module_name] != expected_parallelism:
            raise RuntimeError(
                f"{module_name} detected as {grouped_linear_detection[module_name]}, expected {expected_parallelism}"
            )
    payload["api"]["tpaware_grouped_linear_detection"] = grouped_linear_detection
    module = type("TEColumnParallelGroupedLinear", (torch.nn.Module,), {})()
    module.weight0 = torch.nn.Parameter(torch.empty(4, 4))
    with qwen3_moe_bridge_plugin._temporary_weight_attr(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0", module
    ):
        payload["api"]["grouped_linear_temporary_weight_attr"] = {
            "has_weight": hasattr(module, "weight"),
            "weight_is_weight0": module.weight is module.weight0,
        }
        if module.weight is not module.weight0:
            raise RuntimeError("temporary grouped weight attr did not expose weight0")
    if hasattr(module, "weight"):
        raise RuntimeError("temporary grouped weight attr leaked after context")
except Exception as exc:
    record_error("qwen3_moe:mapping_registry", exc)

try:
    from nemo_rl.models.megatron import community_import

    save_helper = getattr(community_import, "_save_megatron_model", None)
    checkpoint_fallback = getattr(
        community_import, "_save_megatron_model_with_checkpointing", None
    )
    try:
        from megatron.bridge.training.model_load_save import save_megatron_model
        model_load_save_available = callable(save_megatron_model)
        model_load_save_signature = str(signature(save_megatron_model))
    except Exception as exc:
        model_load_save_available = False
        model_load_save_signature = f"{type(exc).__name__}: {exc}"
    from megatron.bridge.training.checkpointing import save_checkpoint
    from megatron.bridge.training.config import MockGPTDatasetConfig

    mock_dataset_config = MockGPTDatasetConfig(
        random_seed=0,
        sequence_length=1,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )

    payload["api"]["community_import_save_compat"] = {
        "helper_available": callable(save_helper),
        "checkpoint_fallback_available": callable(checkpoint_fallback),
        "model_load_save_available": model_load_save_available,
        "checkpointing_save_available": callable(save_checkpoint),
        "mock_dataset_config_constructible": mock_dataset_config is not None,
        "helper_signature": str(signature(save_helper)) if callable(save_helper) else None,
        "checkpoint_fallback_signature": str(signature(checkpoint_fallback))
        if callable(checkpoint_fallback)
        else None,
        "model_load_save_signature": model_load_save_signature,
        "checkpointing_save_signature": str(signature(save_checkpoint)) if callable(save_checkpoint) else None,
    }
    if not callable(save_helper):
        raise RuntimeError("community_import._save_megatron_model is not available")
    if not callable(checkpoint_fallback):
        raise RuntimeError("community_import checkpointing save fallback is not available")
    if not (model_load_save_available or callable(save_checkpoint)):
        raise RuntimeError("no native Megatron checkpoint save fallback is available")
except Exception as exc:
    record_error("community_import:save_compat", exc)

try:
    from megatron.bridge.models.model_provider import get_model

    payload["api"]["get_model_has_pg_collection"] = (
        "pg_collection" in signature(get_model).parameters
    )
except Exception as exc:
    record_error("signature:get_model", exc)

try:
    from megatron.bridge.training.setup import _update_model_config_funcs

    payload["api"]["update_model_config_funcs_has_pg_collection"] = (
        "pg_collection" in signature(_update_model_config_funcs).parameters
    )
except Exception as exc:
    record_error("signature:_update_model_config_funcs", exc)

try:
    from megatron.core.process_groups_config import ProcessGroupCollection

    payload["api"]["process_group_collection"] = {
        "status": "present",
        "repr": repr(ProcessGroupCollection),
    }
except Exception as exc:
    payload["api"]["process_group_collection"] = {
        "status": "missing",
        "type": type(exc).__name__,
        "message": str(exc),
    }

try:
    from torch.distributed.tensor._ops._tensor_ops import (
        propagate_single_input_strategy,
    )

    payload["api"]["torch_propagate_single_input_strategy"] = {
        "status": "present",
        "repr": repr(propagate_single_input_strategy),
    }
except Exception as exc:
    payload["api"]["torch_propagate_single_input_strategy"] = {
        "status": "missing",
        "type": type(exc).__name__,
        "message": str(exc),
    }

json_out.parent.mkdir(parents=True, exist_ok=True)
markdown_out.parent.mkdir(parents=True, exist_ok=True)
json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

overall = payload["overall_status"].upper()
lines = [
    "# Megatron Compatibility Probe",
    "",
    f"Overall: **{overall}**",
    "",
    "| item | value |",
    "| --- | --- |",
]
for module_name, result in payload["modules"].items():
    status = result.get("status")
    lines.append(f"| import `{module_name}` | `{status}` |")
for key, value in payload["api"].items():
    lines.append(f"| `{key}` | `{value}` |")
if payload["errors"]:
    lines.extend(["", "## Errors", ""])
    for error in payload["errors"]:
        scope = error["scope"]
        error_type = error["type"]
        message = error["message"]
        lines.append(f"- `{scope}`: {error_type}: {message}")
markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

if payload["overall_status"] != "pass":
    raise SystemExit(1)
PY
'
