# Qwen3-32B Thinking EAGLE3 K0-K5 DynamicSD Calibration Design

## Objective

Derive the DynamicSD batch-size schedule before running NeMo-RL final20. The
schedule is not hand-authored. It is generated from a matched vLLM 0.25.1
offline profile using the goodput method introduced in vLLM PR #32374.

## Controlled Profile

The profile matches the Qwen3-32B NeMo-RL performance recipe:

- target revision `9216db5781bf21249d130ec9da846c4624c16137`;
- Thinking EAGLE3 revision `a1403e07b73a66fc9ef561463631c31864616933`;
- vLLM 0.25.1, model-runner v2, target TP2, draft TP1;
- CUDA Graph enabled with `FULL_AND_PIECEWISE` and native capture sizing;
- maximum model/output cap 4096, with 256 generated tokens per profiling
  request as in the upstream offline-profiler methodology;
- temperature 1.0, top-p 1.0, prefix caching disabled, chunked prefill enabled;
- `max_num_batched_tokens=16384` and `max_num_seqs=256`;
- OpenMathInstruct-2 prompts rendered with `examples/prompts/cot.txt`;
- OpenMathInstruct-2 revision
  `469216e3f46f4dacf476b382e192485ea51a143e`;
- twenty steady-state batches per profiled point after warmup.

The profile grid is batch sizes `1, 4, 16, 32, 64, 128, 192, 256` and draft
lengths `K=0,1,2,3,4,5`. K0 is a true no-drafter baseline. Every K1-K5 row
uses the same target and Thinking drafter revision. Results from a different
runtime, checkpoint, CUDA Graph mode, sampling configuration, or prompt set
cannot populate this profile.

## Schedule Derivation

The derivation follows the original vLLM DynamicSD profiler:

1. Measure position-level acceptance probabilities `a[0]..a[4]` with K5.
2. Record median inter-token latency for every measured batch-size/K cell.
3. For unmeasured integer batch sizes, linearly interpolate ITL between the
   neighboring profiled batch sizes for the same K.
4. Compute accepted length `AL(K) = 1 + sum(a[:K])`.
5. Compute `goodput(BS,K) = AL(K) / ITL(BS,K)` and select the maximizing K.
6. Break exact ties toward the lower K. Select K0 when a speculative K does
   not exceed K0 goodput by the configured minimum-gain threshold.
7. Compress identical adjacent choices into inclusive vLLM ranges covering
   every batch size from 1 through 256.

Unlike the early upstream implementation, all K0-K5 points are measured, so
K interpolation is not used. Batch-size interpolation remains necessary to
create a dense runtime schedule from the bounded profile grid.

## Reproducibility Contract

The raw profile is an immutable JSON artifact containing the full grid,
position-level acceptance, warmup/repetition counts, exact model/runtime
identity, prompt identity, and controlled settings. The derived schedule
records the raw profile SHA-256 and its own deterministic schedule ranges.

The NeMo-RL matrix declares a DynamicSD maximum K5. The launcher rejects a
schedule whose declared maximum differs from the variant maximum, preventing
vLLM from silently clamping K4/K5 ranges to K3. Final20 additionally requires
the reviewed schedule artifact SHA-256 in the local allowlist.

## Validation Gates

1. Unit tests cover interpolation, accepted-length/goodput calculation,
   deterministic tie handling, K0 fallback, range compression, incomplete
   grids, invalid acceptance values, and identity mismatch.
2. Local Ruff, Pyright, focused pytest, and shell syntax checks pass.
3. Lyris submission passes scheduler test-only with `--segment=4`, no
   `--gres`, dependency, or singleton option.
4. The profile records all 48 grid cells and twenty measured batches per cell.
5. A replay benchmark compares the derived DynamicSD schedule with fixed K0,
   K1, K2, K3, K4, and K5 before final20 promotion.
6. Only after the replay gate passes is the calibrated schedule hash added to
   the final20 allowlist and used by NeMo-RL.

## Deliverables

- typed offline profile and schedule generator;
- focused unit tests;
- Lyris profile launcher and immutable raw artifacts;
- calibrated K0-K5 schedule with hash provenance;
- DynamicSD final20 job and step 2-20 report after calibration.
