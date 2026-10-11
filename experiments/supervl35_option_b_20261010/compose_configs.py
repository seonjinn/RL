"""Compose the six performance arms from the inspected HSG precision profile."""

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--te-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_config(args.profile)
    source.token_capture.control_auth_token = "${oc.env:NRL_TOKEN_CAPTURE_AUTH}"
    source.policy.megatron_cfg.te_precision_config_file = (
        "${oc.env:NRL_EXPERIMENT_SOURCE}/experiments/"
        "supervl35_option_b_20261010/configs/te_precision.yaml"
    )
    common = {
        "grpo": {
            "max_num_steps": 20,
            "val_period": 0,
            "val_at_start": False,
            "val_at_end": False,
        },
        "checkpointing": {
            "enabled": False,
            "checkpoint_must_save_by": None,
        },
        "logger": {"wandb_enabled": False, "tensorboard_enabled": True},
    }
    arms = OmegaConf.load(Path(__file__).with_name("arm_overrides.yaml")).arms
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "te_precision.yaml").write_bytes(args.te_recipe.read_bytes())
    for name, delta in arms.items():
        config = OmegaConf.merge(source, common, delta)
        assert config.policy.router_replay.enabled
        assert not config.policy.megatron_cfg.fp32_lm_head
        assert not config.policy.generation.vllm_cfg.fp32_lm_head
        assert config.grpo.num_prompts_per_step == 128
        assert config.grpo.num_generations_per_prompt == 16
        if not config.policy.megatron_cfg.fp8_cfg.enabled:
            assert config.policy.megatron_cfg.te_precision_config_file is None
        if name == "bf16-bf16":
            assert not config.policy.generation.vllm_cfg.quantization_ignore_patterns
        OmegaConf.save(config, args.output / f"{name}.yaml", resolve=False)
    inputs = sorted(args.profile.parent.glob("*.yaml")) + [args.te_recipe]
    manifest = {
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
        },
        "steps": 20,
        "arms": list(arms.keys()),
        "scope": "Routed experts only; decoder layer 0 and layers 80-87 BF16",
    }
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Composed {len(arms)} arms in {args.output}")


if __name__ == "__main__":
    main()
