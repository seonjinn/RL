#!/bin/bash
# Idempotently (re)apply the 4 vLLM-0.17 FP8KV fixes to the ruit repo + verify.
# Run ON cw-dfw-cs. The fixes are otherwise UNCOMMITTED working-tree edits (incl.
# one inside the Megatron-LM submodule), so a git pull/checkout/submodule-update
# wipes them -> rerun this to restore. See VLLM017_FP8KV_SETUP_HANDOFF.md.
#
#   bash apply_vllm017_fixes.sh            # apply + verify
#   bash apply_vllm017_fixes.sh --verify   # verify only
set -u
R=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/repos/nemo-rl-swe-bench-ruit
SU="$R/nemo_rl/models/megatron/setup.py"
SER="$R/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/dist_checkpointing/serialization.py"
MPW="$R/nemo_rl/models/policy/workers/megatron_policy_worker.py"
LAUNCH="$R/examples/swe_bench/run_grpo_swe2_qwen235b_fp8kv.sh"
HFCFG=/lustre/fsw/portfolios/coreai/users/sna/hf_home_vllm017/nemo_rl/Qwen/Qwen3-235B-A22B-Thinking-2507/iter_0000000/run_config.yaml
QSCALE_KEYS="async_tensor_model_parallel_allreduce|moe_extended_tp|moe_use_legacy_grouped_gemm|mtp_detach_heads|mtp_grad_scale_func"

if [ "${1:-}" != "--verify" ]; then
  # Fix 1: megatron-bridge config STRICT -> LENIENT
  sed -i 's/mode=InstantiationMode\.STRICT/mode=InstantiationMode.LENIENT/' "$SU"
  # Fix 2: DCP load validate_access_integrity default True -> False (load() def, line ~57)
  sed -i '57s/validate_access_integrity: bool = True/validate_access_integrity: bool = False/' "$SER"
  # Fix 4: seq_logprob_error_threshold incompatible with force_on_policy_ratio=True
  sed -i -E 's/^SEQ_LOGPROB_ERROR_THRESHOLD=.*/SEQ_LOGPROB_ERROR_THRESHOLD=null/' "$LAUNCH"
  # Patched converted-ckpt run_config.yaml: drop the 5 deprecated GPTModelProvider keys
  if [ -f "$HFCFG" ]; then
    sed -i -E "/^[[:space:]]*($QSCALE_KEYS)[[:space:]]*:/d" "$HFCFG"
  fi
  # Fix 3 (q_scale) is a multi-line code edit -> applied via the dedicated applier, not sed:
  #   experiments/eagle3_qwen3_235b/remote_patches/apply_ruit_fp8kv_qscale_fix.py
  if ! grep -q 'if k != "q_scale"' "$MPW"; then
    echo "[WARN] q_scale fix MISSING in $MPW -> run apply_ruit_fp8kv_qscale_fix.py"
  fi
fi

echo "=== VERIFY ==="
echo -n "1 LENIENT (setup.py):            "; grep -c "InstantiationMode.LENIENT" "$SU"
echo -n "2 validate_access_integrity F:   "; sed -n '57p' "$SER" | grep -c "= False"
echo -n "3 q_scale skip (mpw):            "; grep -c 'if k != "q_scale"' "$MPW"
echo -n "4 seq_logprob null (launch):     "; grep -cE '^SEQ_LOGPROB_ERROR_THRESHOLD=null' "$LAUNCH"
echo -n "  force_on_policy True (launch):  "; grep -cE '^FORCE_ON_POLICY_RATIO=True' "$LAUNCH"
echo -n "  ckpt run_config 5 keys removed: "; if grep -qE "^[[:space:]]*($QSCALE_KEYS)[[:space:]]*:" "$HFCFG" 2>/dev/null; then echo "0 (STILL PRESENT - BAD)"; else echo "1"; fi
echo "All counts should be 1. If any is 0, the fix is missing."
