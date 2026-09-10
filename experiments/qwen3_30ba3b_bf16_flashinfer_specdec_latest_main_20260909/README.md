# Latest-main BF16 FlashInfer TRTLLM + SpecDec smoke

This isolated gate verifies Qwen3-30B-A3B Math GRPO on the upstream NeMo-RL
`main` commit fetched on 2026-09-09. It preserves the official 4-node × 4-GPU
BF16 performance recipe and changes only the runtime settings needed to use the
FlashInfer TRTLLM MoE backend and frozen speculative drafters.

The matched short-context arms are a no-SpecDec baseline, DFlash K3, and
DSpark K3/K5/K7. Completing step 3 proves generation startup and two
post-update target refits; the same launcher accepts
`Q30_LATEST_MAIN_MAX_STEPS=20` for the measurement run. The drafter stays
frozen (`policy.draft.enabled=false`), so this gate does not claim that BF16
native reload refit supports online drafter updates.

The 32K gate keeps the official 4-node × 4-GPU recipe as its base and applies
the previously matched long-context workload shape: 16 prompts × 16
generations, GBS 256, packed sequences, CP2, activation checkpointing, and
32K generation/model/token limits. Its matrix is the no-SpecDec baseline plus
DSpark K3/K5/K7. Run 3 steps first because earlier Triton-backed 32K runs
reached step 1 but failed while waking vLLM after the first target refit.

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

### 32K CUDA Graph scope correction

The 32K gate uses FULL_AND_PIECEWISE and 16 concurrent requests. Capture
sizes are scheduled **target tokens**, not context lengths or request counts.
Each request schedules K+1 verification tokens. Capture all request counts
1–16: baseline 1–16, DSpark K3 4–64 (stride 4), K5 6–96 (stride 6), and
K7 8–128 (stride 8). This also avoids MRV1 rounding away the largest bucket.
The previous cap of 16 did not cover a full speculative decode batch.

This scope covers uniform target verification; it does not claim full graph
coverage for 32K prefill. Mixed/prefill batches above the capture cap may run
without a graph. Keep max_num_batched_tokens=32768 unchanged for this gate;
do not allocate a 32768-token graph merely because the context limit is 32K.
Verify resolved graph mode, final capture sizes, drafter graph coverage,
capture memory, and runtime fallbacks before attributing a speedup to graphs.
Successful capture alone is not proof of replay coverage or OOM safety.

Source: vLLM v0.25.1 `vllm/v1/cudagraph_dispatcher.py` and
`vllm/config/compilation.py::adjust_cudagraph_sizes_for_spec_decode`.

Run the 32K gate:

```bash
bash experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/submit_long_context_matrix.sh --test-only
bash experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/submit_long_context_matrix.sh --submit
```

After every arm completes step 3, extend the same matrix to 20 steps:

```bash
Q30_LATEST_MAIN_MAX_STEPS=20 \
  bash experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/submit_long_context_matrix.sh --submit
```
