# vLLM 0.28 Qwen3-30B-A3B Synchronous DynamicSD OSL1K Design

## Objective

Measure target-only, fixed-K, and DynamicSD generation performance for DFlash
and DSpark under the synchronous NeMo-RL Qwen3-30B-A3B rollout shape, with
`max_tokens=1024` and a global barrier after every rollout step.

## Runtime and assets

- Runtime: vLLM `0.28.0`, base commit
  `2cf0a6915ce544dc493a0990f2ea38d81601128a`.
- Primary container: the existing authenticated MRV2 Dynamic-K image
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm-openai-v0.28.0-mrv2-dynamick-core-aarch64-ubuntu2404.sqsh`.
- Target: `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-specdec/assets/q30-base-opb-drafters-s4166-eval-v1/q30-base`.
- DFlash: the sibling `dflash-s4166` checkpoint, physical block size 8.
- DSpark: the sibling `dspark-s4166` checkpoint, physical block size 8.
- The PTV2-final paths supplied for another filesystem are not present on
  Lyris. They are a follow-up cohort, not silently substituted into this run.

## Workload

- Match one NeMo-RL synchronous step: 64 prompts and 32 generations per
  prompt, for 2,048 rollouts globally.
- Match the 4n4g generation layout using 16 independent TP1/DP1 vLLM engines,
  four per node. Each engine receives 128 requests.
- `temperature=1.0`, `top_p=1.0`, per-request deterministic seeds, natural EOS,
  and `max_tokens=1024`.
- Use `FULL_AND_PIECEWISE`; record the effective graph dispatch and reject
  unreported eager fallback.
- Keep vLLM internal data parallelism at one. The sixteen engines are
  coordinated externally because DynamicSD is not safe with vLLM DP greater
  than one.

## Experiment stages

1. One-GPU canaries for baseline, DFlash, and DSpark prove model loading,
   output production, speculative counters, selected-K behavior, and CUDA
   Graph capture.
   A mandatory DynamicSD K0 diagnostic also traces drafter execution. Stock
   v0.28 MRV2 is expected to suppress target verification tokens while still
   executing the parallel drafter's physical Kmax block; this distinction is
   reported rather than hidden.
2. A one-engine calibration grid measures batch sizes
   `{1,2,4,8,16,32,64,96,128}` and K `{0,1,2,3,5,7}` independently for each
   drafter.
3. Calibration produces one monotone batch-to-K schedule per drafter. K may
   stay equal or decrease as batch size rises; K0 is allowed at high load.
4. The 16-engine barrier comparison runs target-only, each drafter's best
   fixed K, and each drafter's calibrated DynamicSD schedule with three
   repetitions.
5. DSpark confidence-based adaptive verification is evaluated as a separate
   v0.28 arm after its confidence-head compatibility canary. It is not mixed
   with DynamicSD in the first comparison.

## Metrics and validity gates

- Primary: global barrier seconds and output tokens/s/GPU.
- Comparisons: baseline-relative and best-fixed-relative speedups.
- Tail: request finish p50/p90/p99/max and time spent below active batch sizes
  32, 16, and 8.
- SpecDec: proposed and accepted tokens, acceptance rate, mean accepted length,
  selected-K histogram, actual mean draft width, and a K0 drafter-kernel/time
  trace proving whether physical draft work was skipped.
- Provenance: exact container digest, vLLM commit, target/drafter config hashes,
  prompt manifest hash, sampling settings, Slurm topology, and job IDs.
- Natural-stopping rows retain both wall-time and throughput metrics so output
  length variation cannot masquerade as speedup.

## Safety

- Preserve all v0.25.1 and v0.27.1 worktrees and containers.
- Source and scripts live under `/home`; caches and compilation outputs use
  `/raid/scratch`; checkpoints and durable results remain on `/lustre`.
- Render and test jobs locally, commit and push, run `sbatch --test-only`, then
  submit canaries. Expand only after the canary validator succeeds.
