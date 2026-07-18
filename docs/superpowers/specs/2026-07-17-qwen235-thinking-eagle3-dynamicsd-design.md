# Qwen3-235B Thinking EAGLE3 DynamicSD Design

## Objective

Measure a Qwen3-235B-specific DynamicSD policy over Thinking EAGLE3 K0-K3 and
compare it with the corrected fixed-K3 CUDA Graph run. The schedule is derived
from a matched vLLM 0.25.1 offline profile rather than copied from Qwen3-32B.

## Controlled NeMo-RL Comparison

- Recipe: `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml`
- Target revision: `8efa61729e24bd65b1d152b5ab5409052aa80e65`
- Thinking drafter revision: `3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87`
- Runtime: vLLM 0.25.1, model runner v2, target TP8, draft TP1
- Sampling: temperature 1.0 and top-p 1.0
- CUDA Graph: `FULL_AND_PIECEWISE`, capture sizes
  `[1,2,4,8,16,32,64,128,192,256]`
- Checkpoint saving: disabled
- Final metric window: steps 2-20 inclusive

Historical fixed-K3 job `2416712` changed the capture-size list from the
performance-recipe default through 64 to `[1,2,4,8,16,32,64,128,256]` and
achieved 1.771x generation-time speedup and 1.376x E2E throughput speedup over
the matched baseline. It proves the coverage diagnosis, but it is not the final
DynamicSD comparator because the DynamicSD contract also captures 192 tokens
for K2 at 64 active requests. Fixed K3 must be rerun with the identical expanded
capture list before the final comparison.

## Offline Profile

The Qwen3-235B profile uses one TP8 vLLM engine across two Lyris GB200 nodes.
It measures K0, K1, K2, and K3 independently at active per-engine batch sizes
`1,4,8,16,32,48,64`, with twenty steady-state batches per point. The maximum
active batch is 64 because the recipe produces 512 trajectories across eight
TP8 generation replicas.

The profile uses OpenMathInstruct-2 prompts rendered with `cot.txt`, output
length 256, max model length 8192, engine capacity 128 requests, temperature
1.0, top-p 1.0, chunked prefill, prefix caching enabled, Triton MoE, and capture
sizes `[1,2,4,8,16,32,64,128,192,256]`. K0 loads the same drafter and performs
drafter prefill before selecting zero draft tokens so a later K0-to-K>0 switch
cannot consume stale draft KV. K3 separately records position-level acceptance
for the three draft positions.

The calibrator computes `accepted_length(K) / median_ITL(batch_size,K)`,
interpolates only between measured batch-size points, chooses K0 when no
speculative K improves goodput, and emits contiguous ranges through batch size
64. vLLM carries the final configured K above the last range, but the NeMo-RL
matrix rejects a Qwen3-235B schedule whose measured range does not end at the
recipe's active maximum of 64.

## Runtime And Promotion

The DynamicSD variant keeps global max K3, applies the existing source-guarded
variable-width vLLM patch, and retains the corrected capture list through 256.
The schedule identity must match the exact target, drafter, vLLM version, and
CUDA Graph mode.

Promotion order is:

1. complete all K0-K3 profile cells and acceptance data;
2. derive and review the immutable schedule;
3. run smoke2 and verify selected/requested/actual K telemetry plus graph
   coverage;
4. allowlist the reviewed schedule hash;
5. run final20 and compare steps 2-20 with baseline and fixed K3.

No change is pushed to NVIDIA-NeMo/RL. Development and submission use only the
existing private fork branch.
