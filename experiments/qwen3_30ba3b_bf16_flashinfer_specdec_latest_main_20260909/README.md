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
DFlash and DSpark at K3/K5/K7. Run 3 steps first because earlier Triton-backed
32K runs reached step 1 but failed while waking vLLM after the first target
refit.

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

Corrected runs use `CGScopeV2` in the run name and
`q30-latest-main-bf16-flashinfer-specdec-32k-cgscope-v2` as their W&B group,
so they cannot be confused with the earlier target-only capture runs. They use
the isolated remote worktree
`/home/sna/nemorl-bf16-flashinfer-specdec-cgscope-v2-20260910`; the source tree
used by already-running jobs is not modified.

The target model uses FULL_AND_PIECEWISE and 16 concurrent requests. The
DFlash/DSpark query drafter uses FULL_DECODE_ONLY because its manager does not
support piecewise graphs. Capture
sizes are scheduled **target tokens**, not context lengths or request counts.
Each request schedules K+1 target verification tokens. DFlash uses the same
K+1 query width. The anchor-as-first
DSpark drafter separately schedules K query tokens. The capture list starts
with every target K+1 shape for request counts 1–16, then adds only the DSpark
K shapes that would otherwise be padded. This is the minimal list that gives
exact target and drafter decode shapes: 20 sizes for K3, 19 for K5, and 18 for
K7. DFlash needs only the 16 corresponding K+1 sizes. The previous cap of 16
did not cover a full speculative decode batch, while a target-only K+1 list
padded some DSpark drafter batches. Capturing the full K/K+1 union would add
unnecessary PIECEWISE graphs and startup memory in this 32K gate.

This scope covers uniform target verification; it does not claim full graph
coverage for 32K prefill. Mixed/prefill batches above the capture cap may run
without a graph. Keep max_num_batched_tokens=32768 unchanged for this gate;
do not allocate a 32768-token graph merely because the context limit is 32K.
Verify resolved graph mode, final capture sizes, drafter graph coverage,
capture memory, and runtime fallbacks before attributing a speedup to graphs.
Successful capture alone is not proof of replay coverage or OOM safety.

Source: vLLM v0.25.1 MRV2 `vllm/v1/worker/gpu/cudagraph_utils.py` and
`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`.

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

### DFlash CUDA Graph diagnostic

The focused diagnostic keeps the matched 32K workload and step-44000 frozen
drafter unchanged. It compares DFlash K5 FAP against DFlash K5 eager, with
DSpark K5 FAP as the method reference. All arms use seed 42 and profile only
the vLLM generation worker during step 2 after one warm-up step. Nsight traces
include CUDA Graph node tracing and are synchronized into each durable Ray log
directory.

```bash
bash experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/submit_dflash_cudagraph_diagnostic.sh --test-only
bash experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/submit_dflash_cudagraph_diagnostic.sh --submit
```

Compare `cudaGraphLaunch` frequency and proposal/verification kernel time. If
DFlash FAP replays graphs and outperforms its eager arm but remains behind
DSpark, the method-specific draft runtime is the dominant cost. If FAP does
not replay or performs like eager, the CUDA Graph dispatch path needs further
correction.
