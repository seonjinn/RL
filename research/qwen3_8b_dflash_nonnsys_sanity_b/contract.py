from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PRODUCT_HEAD = "79e80af96a13522e6049658663a8c40ab21e8314"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFTER_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_resolved_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    defaults = raw.pop("defaults", None)
    if defaults is None:
        return raw
    if not isinstance(defaults, str):
        raise ValueError(f"defaults must be a relative path: {path}")
    return _merge(load_resolved_config((path.parent / defaults).resolve()), raw)


def _nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing config key: {'.'.join(keys)}")
        value = value[key]
    return value


def validate_experiment_contract(experiment: Path) -> dict[str, Any]:
    online = load_resolved_config(experiment / "online_config.yaml")
    fixed = load_resolved_config(experiment / "fixed_config.yaml")
    manifest = yaml.safe_load((experiment / "manifest.yaml").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a mapping")

    expected = {
        ("grpo", "max_num_steps"): 50,
        ("grpo", "seed"): 42,
        ("grpo", "num_prompts_per_step"): 8,
        ("grpo", "num_generations_per_prompt"): 4,
        ("policy", "train_global_batch_size"): 32,
        ("policy", "sequence_packing", "enabled"): False,
        ("policy", "megatron_cfg", "sequence_parallel"): False,
        ("policy", "draft", "update_probe_enabled"): False,
        (
            "policy",
            "generation",
            "vllm_kwargs",
            "speculative_config",
            "method",
        ): "dflash",
        (
            "policy",
            "generation",
            "vllm_kwargs",
            "speculative_config",
            "num_speculative_tokens",
        ): 7,
        ("logger", "wandb_enabled"): True,
        ("logger", "wandb", "entity"): "nvidia",
        ("logger", "wandb", "project"): "sna-nemo-rl-online-drafter",
    }
    for arm, config in (("online", online), ("fixed", fixed)):
        for keys, value in expected.items():
            actual = _nested(config, *keys)
            if actual != value:
                raise ValueError(
                    f"{arm} {'.'.join(keys)} must be {value!r}, got {actual!r}"
                )

    shared_paths = (
        ("policy", "model_name"),
        ("policy", "draft", "model_name"),
        ("policy", "generation", "vllm_cfg"),
        ("policy", "generation", "vllm_kwargs", "speculative_config"),
        ("policy", "generation", "vllm_kwargs", "compilation_config"),
        ("cluster",),
    )
    for keys in shared_paths:
        if _nested(online, *keys) != _nested(fixed, *keys):
            raise ValueError(f"arm mismatch at {'.'.join(keys)}")
    if _nested(online, "policy", "draft", "enabled") is not True:
        raise ValueError("online draft training must be enabled")
    if _nested(fixed, "policy", "draft", "enabled") is not False:
        raise ValueError("fixed draft training must be disabled")

    for launcher_name in ("run_online_oci_hsg.sbatch", "run_fixed_oci_hsg.sbatch"):
        launcher = (experiment / launcher_name).read_text()
        if "nsys " in launcher or "nsys profile" in launcher:
            raise ValueError(f"Nsys must be disabled in {launcher_name}")
        for marker in (
            "grpo.max_num_steps=50",
            "grpo.val_period=1",
            "checkpointing.save_period=50",
            "policy.draft.update_probe_enabled=false",
        ):
            if marker not in launcher:
                raise ValueError(f"missing {marker} in {launcher_name}")

    return {
        "label": manifest["label"],
        "product_head": PRODUCT_HEAD,
        "shared": {
            "max_num_steps": 50,
            "probe_enabled": False,
            "nsys_enabled": False,
            "wandb_path": "nvidia/sna-nemo-rl-online-drafter",
            "target_revision": TARGET_REVISION,
            "drafter_revision": DRAFTER_REVISION,
        },
    }
