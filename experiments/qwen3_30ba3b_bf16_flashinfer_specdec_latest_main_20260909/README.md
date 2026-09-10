# Latest-main BF16 FlashInfer TRTLLM + SpecDec smoke

This isolated gate verifies Qwen3-30B-A3B Math GRPO on the upstream NeMo-RL
`main` commit fetched on 2026-09-09. It preserves the official 4-node × 4-GPU
BF16 performance recipe and changes only the runtime settings needed to use the
FlashInfer TRTLLM MoE backend and frozen speculative drafters.

The three matched 3-step arms are a no-SpecDec baseline, DFlash K3, and DSpark
K3. Completing step 3 proves generation startup and two post-update target
refits. The drafter stays frozen (`policy.draft.enabled=false`), so this gate
does not claim that BF16 native reload refit supports online drafter updates.

The DSpark arm applies a source-verified vLLM 0.25.1 compatibility overlay for
FULL_AND_PIECEWISE CUDA Graph capture. Every result directory records the exact
NeMo-RL commit and recursive submodule revisions used by the job.

Run locally:

```bash
PYTHONPATH=. pytest -q \
  experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/tests/test_contract.py
```

Run on OCI-HSG after pulling the committed branch and initializing recursive
submodules:

```bash
bash experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/submit_matrix.sh --test-only
bash experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/submit_matrix.sh --submit
```
