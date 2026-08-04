from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def require_valid_eval_weight_source(config: Mapping[str, Any]) -> None:
    """Reject dynamic MXFP8 evals that cannot perform a policy refit."""
    canary = config.get("canary", {})
    generation = config.get("generation", {})
    vllm_cfg = generation.get("vllm_cfg", {})
    if (
        canary.get("requires_policy_refit")
        and vllm_cfg.get("precision") == "fp8"
        and vllm_cfg.get("is_mx") is True
    ):
        raise RuntimeError(
            "This dynamic MXFP8 canary requires a policy-to-generation refit, "
            "but standalone evaluation loads checkpoint weights directly. Use "
            "the GRPO refit canary instead."
        )
