#!/usr/bin/env python3
"""Validate Qwen235B MathRL speculative-decoding launcher invariants.

This catches the two regressions that caused the latest MathRL PARD-2 failures:
sequence packing disabled while MCore context parallelism stayed above 1, and
online PARD-2 missing the draft/refit controls. It also protects the Eagle-3
MathRL case and RL sampling controls used by the generation-bound runs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


SCRIPT = Path("experiments/eagle3_online/submit_oci_hsg_qwen235b_mathrl_latest_main_20260613.sh")


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def pass_check(message: str) -> None:
    print(f"PASS: {message}")


def case_body(text: str, label: str) -> str:
    pattern = re.compile(
        rf"^\s*{re.escape(label)}\)\n(?P<body>.*?)(?=^\s*(?:[A-Za-z0-9_]+|\*)\))",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        fail(f"case {label!r} not found")
    return match.group("body")


def require(body: str, token: str, label: str) -> None:
    if token not in body:
        fail(f"{label}: missing {token!r}")
    pass_check(f"{label}: {token}")


def require_absent(body: str, token: str, label: str) -> None:
    if token in body:
        fail(f"{label}: forbidden {token!r}")
    pass_check(f"{label}: no {token}")


def main() -> None:
    text = SCRIPT.read_text()

    for label in ("static_pard2_k3", "online_pard2_k3"):
        body = case_body(text, label)
        require(body, "draft_format=pard2", label)
        require(body, "specdec_method=pard2", label)
        require(body, "sequence_packing_enabled=false", label)
        require(body, "context_parallel_size=1", label)
        require(body, "force_local_transformer_spec=true", label)

    static = case_body(text, "static_pard2_k3")
    require_absent(static, "policy_draft_enabled=true", "static_pard2_k3")
    require_absent(static, "pard_online_training=true", "static_pard2_k3")
    require(static, "actor_uv_lock_mode=unlocked", "static_pard2_k3")
    require(static, "serialize_actor_venv_creation=true", "static_pard2_k3")
    require(static, 'method_nemo_rl_venv_dir="${PARD2_SHARED_VENV_DIR}"', "static_pard2_k3")

    online = case_body(text, "online_pard2_k3")
    for token in (
        "policy_draft_enabled=true",
        "pard_online_training=true",
        "policy_draft_type=pard2",
        "policy_draft_loss=pard2",
        "policy_draft_cat_weighting=true",
        "policy_draft_training_mode=k_slot",
        "debug_draft_refit=true",
        "actor_uv_lock_mode=unlocked",
        "serialize_actor_venv_creation=true",
        'method_nemo_rl_venv_dir="${PARD2_SHARED_VENV_DIR}"',
    ):
        require(online, token, "online_pard2_k3")

    eagle3 = case_body(text, "eagle3_k3")
    for token in (
        "draft_format=eagle3",
        "specdec_method=eagle3",
        'draft_model="${EAGLE3_DRAFT_MODEL}"',
        'spec_tokens="${EAGLE3_SPEC_TOKENS}"',
        'draft_tp="${EAGLE3_DRAFT_TP}"',
        "parallel_drafting=false",
    ):
        require(eagle3, token, "eagle3_k3")

    for token in (
        "GENERATION_TEMPERATURE",
        "GENERATION_TOP_P",
        "GENERATION_TOP_K",
        "QWEN235B_LOCAL_SNAPSHOT",
        'TARGET_MODEL="${TARGET_MODEL:-${QWEN235B_LOCAL_SNAPSHOT}}"',
        'TOKENIZER_NAME="${TOKENIZER_NAME:-${QWEN235B_LOCAL_SNAPSHOT}}"',
        "EAGLE3_DRAFT_MODEL",
        "PARD2_SHARED_VENV_DIR",
        "NRL_SERIALIZE_ACTOR_VENV_CREATION",
        "test -e '${TARGET_MODEL}'",
        'test -e \'${EAGLE3_DRAFT_MODEL}\'',
        "policy.generation.temperature=${GENERATION_TEMPERATURE}",
        "policy.generation.top_p=${GENERATION_TOP_P}",
        "policy.generation.top_k=${GENERATION_TOP_K}",
        "policy.megatron_cfg.context_parallel_size=${context_parallel_size}",
        "policy.sequence_packing.enabled=${sequence_packing_enabled}",
        "POLICY_DRAFT_ENABLED='${policy_draft_enabled}'",
        "PARD_ONLINE_TRAINING='${pard_online_training}'",
        "NRL_DEBUG_DRAFT_REFIT='${debug_draft_refit}'",
        "NRL_SERIALIZE_ACTOR_VENV_CREATION='${serialize_actor_venv_creation}'",
        "NEMO_RL_VENV_DIR='${method_nemo_rl_venv_dir}'",
        "NRL_MEGATRON_CHECKPOINT_DIR='${REMOTE_REPO}/nrl_megatron_ckpts_qwen235b_mathrl_${RUN_ID}_${method_label}'",
        "NRL_REFIT_INFO_MERGE_V1",
        'str(sum(1 for key in merged if key.startswith("draft.")))',
        "ONLINE_EXTRA_OVERRIDES='${extra_overrides}'",
    ):
        require(text, token, "generated command")

    pass_check("Qwen235B MathRL speculative-decoding contract validated")


if __name__ == "__main__":
    main()
