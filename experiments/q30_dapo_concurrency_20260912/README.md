# DAPO GBS-2048 concurrency sweep

## Question and scope

Does limiting per-engine rollout concurrency improve speculative decoding at a
fixed policy global batch size? This is a controlled DAPOMath17K workload, not
the unchanged OpenMathInstruct-2 performance-40K recipe.

The first stage fixes K=5 and compares no-SpecDec baseline, frozen DFlash K5,
and frozen DSpark K5 at `max_num_seqs=16,32,64,128`: 12 configurations.
It does not change online-training cadence or enable DynamicSD.

## Matched settings

| Field | Value |
|---|---|
| Target | Qwen/Qwen3-30B-A3B Base, local HF checkpoint |
| Draft lineage | New Draft Base, PTV3-SWA exported checkpoint 44000 |
| Policy GBS / rollout samples | 2048 / 128 prompts × 16 generations |
| Dataset / reward | DAPOMath17K / dapo_math_verify |
| Input / response / total caps | 2048 / 47104 / 49152 tokens |
| Training TP / EP / CP / PP | 2 / 8 / 4 / 1 |
| Hardware | OCI-HSG GB200, 8 nodes × 4 GPUs |
| vLLM TP | 1, inherited from the Qwen 4n4g performance recipe |
| Generation token budget | max_num_batched_tokens=49152 |
| Precision / MoE backend | BF16 / flashinfer_trtllm |
| CUDA Graph mode | FULL_AND_PIECEWISE |
| FlashInfer autotune | disabled equally for baseline and SpecDec |
| Draft training | disabled (frozen) |
| W&B project / group | sna-specdec / q30-dapo-gbs2048-concurrency-20260912 |
| Runtime | nightly 20260909_7023221, vLLM 0.25.1 |

Only concurrency, its graph shapes, and identifying metadata vary within a
method. Graph buckets include powers-of-two request counts multiplied by the
target query width (K+1); DSpark additionally includes its draft width (K).
For K5 and 128 requests, terminal target/draft shapes are 768/640 tokens.
Coverage tests validate the intended shape envelope, not actual GPU graph replay.

With colocated generation on 32 GPUs and vLLM TP=1, equal sharding of 2048
samples yields 64 samples per engine. The 128-request setting may therefore
not reach 128 active requests; it also serves as a control for a nonbinding
scheduler limit and a larger graph envelope. Record realized engine load before
interpreting 64-versus-128 as a concurrency comparison. Do not change topology
or rollout sample count to force saturation within this cohort.

This worktree starts at `3499431e7fa794dc20763f882c8cd67a9620066a`.
The stable DAPO and SWE worktrees and existing submitted scripts remain unchanged.

## Launch and verification

1. Render and validate all 12 combinations locally; assert unchanged workload.
2. Commit and push; create an isolated `/home` worktree on OCI-HSG, pull,
   and initialize submodules recursively. Reuse the existing nightly image.
3. Run `--test-only`, then submit a one-step gate for each combination.
4. Submit each 20-step measurement with `afterok` on its own gate only.
   No dependencies are placed between different configurations.
5. Monitor for at least five minutes after jobs start. Pending jobs have not
   validated runtime or graph coverage. The batch walltime is four hours;
   checkpointing remains disabled to preserve the frozen performance workload.
   A timeout is incomplete, not a finished 20-step result.

```bash
bash experiments/q30_dapo_concurrency_20260912/submit.sh --render dspark_k5 32
bash experiments/q30_dapo_concurrency_20260912/submit.sh --submit dspark_k5 32
Q30_DAPO47K_MAX_STEPS=20 bash experiments/q30_dapo_concurrency_20260912/submit.sh --submit dspark_k5 32 GATE_JOB_ID
uv run --no-project python -m unittest discover -s experiments/q30_dapo_concurrency_20260912/tests -v
```

Submission uses the inherited W&B secret without writing it to artifacts.
Each run records the generated sbatch, source SHA, submodule SHAs, dry-run output,
and submission receipt under the durable experiment directory.

## Analysis criteria

- Compare steps 3–20 only for completed, equivalent windows.
- Report generation TPS/GPU, generation walltime, E2E step time and TPS/GPU,
  policy/logprob/refit timing, and actual generated token counts.
- Compare each SpecDec arm against the baseline with the same concurrency.
  Also compare against the fastest no-SpecDec baseline across concurrency values;
  do not manufacture a benefit by selecting a deliberately slower baseline.
- Inspect reward, entropy, generation KL error, output-length distribution,
  truncation, acceptance, and available output samples before ranking runs.
- Attribute CUDA Graph misses only from runtime dispatch/profiling evidence,
  not from capture-success messages or YAML alone. Profile if ordinary logs do
  not distinguish verification cost, graph fallback, and tail waiting.
- GBS is a training batch size. `max_num_seqs` is a per-engine upper bound,
  not a measurement of realized concurrent requests.

## Status

Local launcher tests passed; remote deployment and job receipts are recorded
in `SUBMISSIONS.md` once available. No speedup is claimed in this document.
