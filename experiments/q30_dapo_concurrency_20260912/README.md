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
| Input / response / total caps | 2048 / 38912 / 40960 tokens (r4) |
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

Revision r4 stays within the staged Base target's native 40,960 positions;
`rope_scaling` is null. It does not bypass vLLM's context validation or modify
the model. The scheduler token budget remains 49,152: that is an aggregate
per-iteration budget, not a single-sequence context length. Historical
`Q30_DAPO47K_*` launcher environment names remain for compatibility, but r4
names explicitly say `DAPO40K`. Do not combine r1–r3 failures with r4 results.

## Launch and verification

### Default-concurrency SpecDec KL control (September 15)

User approved two independent 20-step Qwen3-30B-A3B runs: frozen DFlash K5
and DSpark K5 with no `max_num_seqs` override. Compared with each completed
S64 arm, only that override and identifying run metadata are removed/changed.
The exact S64 CUDA Graph capture sizes remain unchanged to avoid introducing
a second configuration change. Default baseline behavior remains unchanged.
The runtime code, recipe, container, data, token budget and checkpoint lineage
are unchanged. `default` means engine defaults, not unlimited concurrency.

```bash
Q30_DAPO47K_ACCOUNT=nemotron_n3_post Q30_DAPO47K_MAX_STEPS=20 bash experiments/q30_dapo_concurrency_20260912/submit.sh --submit dflash_k5 default
Q30_DAPO47K_ACCOUNT=nemotron_n3_post Q30_DAPO47K_MAX_STEPS=20 bash experiments/q30_dapo_concurrency_20260912/submit.sh --submit dspark_k5 default
```

No gate dependencies are used for these already-approved measurements. Keep
the existing eight-hour `batch_long` budget: earlier 20-step workload timings
plus initialization can exceed the four-hour `batch` limit. Checkpointing stays
disabled, as in the matched frozen controls; preemption is an incomplete run.
Check scheduling with `--test-only` before submission.

Compare Steps 3–20 with DFlash S64 `8k92djcz`, DSpark S64 `1z42ngg1`, and the
no-SpecDec default baseline `6wl01vj0`. Report Policy KL mean, median, maximum
and spike steps, Generation KL, probability-ratio diagnostics, reward,
entropy, generated length, and generation/E2E timing. Large sampled reverse-KL
diagnostics do not quantify accuracy loss. Record the resolved engine limit
and realized load; equal 2048/32 sharding can make S64 nonbinding. Capture
buckets outside the retained envelope are not claimed covered. This is not
a separately optimized default-concurrency performance configuration.

### Default-concurrency Baseline control (September 13 PDT)

User approved a direct 20-step no-SpecDec control after both S16 SpecDec
measurements completed. `--submit baseline default` omits `max_num_seqs` and
the explicit S16 capture-size list, retaining FAP and allowing vLLM to derive
its default scheduler limit and graph sizes. All other workload overrides are
identical to Baseline S16. Record the resolved scheduler value and graph sizes
from this run; `default` does not assert a numeric limit or realized concurrency.
This is a DAPO control, not the unchanged upstream OpenMath performance recipe.

```bash
Q30_DAPO47K_MAX_STEPS=20 bash experiments/q30_dapo_concurrency_20260912/submit.sh --submit baseline default
```

Compare this control against both Baseline S16 and the completed DFlash/DSpark
S16 measurements. The latter is a deployment-configuration comparison, not
an isolated SpecDec-only ablation: concurrency and graph sizing differ too.

1. Render and validate all 12 combinations locally; assert unchanged workload.
2. Commit and push; create an isolated `/home` worktree on OCI-HSG, pull,
   and initialize submodules recursively. Reuse the existing nightly image.
3. Run `--test-only`, then submit only a Baseline S16 one-step pilot. After
   actual completion, submit the remaining one-step gates.
4. Submit each 20-step measurement with `afterok` on its own gate only.
   No dependencies are placed between different configurations.
5. Monitor for at least five minutes after jobs start. Pending jobs have not
   validated runtime or graph coverage. Gates use `batch` / four hours;
   20-step measurements use `batch_long` / eight hours because the successful
   baseline pilot took 1132.85 seconds per step. Checkpointing remains disabled
   to preserve the frozen performance workload; preempted runs need restarting.
   A timeout is incomplete, not a finished 20-step result.

```bash
bash experiments/q30_dapo_concurrency_20260912/submit.sh --render dspark_k5 32
bash experiments/q30_dapo_concurrency_20260912/submit.sh --submit dspark_k5 32
Q30_DAPO47K_MAX_STEPS=20 bash experiments/q30_dapo_concurrency_20260912/submit.sh --submit dspark_k5 32 GATE_JOB_ID
uv run --no-project --with hydra-core python -m unittest discover -s experiments/q30_dapo_concurrency_20260912/tests -v
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

Latest: r3 pilot 7108119 failed before generation because 49,152 exceeded
the staged target's 40,960-position context. The r4 correction aligns input,
response, training/packing and vLLM context limits without changing GBS,
topology, MoE backend or CUDA Graph buckets. See `RECOVERY.md` for evidence
and new submission receipts. No GPU success or speedup is yet established.

The initial cohort failed before model initialization: validation data was
disabled, but the inherited `grpo.val_period=10` still requested validation.
Seven launched gates hit the same assertion; remaining pending jobs were
cancelled on September 12 at approximately 09:31 UTC to avoid repeating it.
Their logs and receipts are retained in `SUBMISSIONS.md`.

Revision r2 explicitly disables periodic, initial, and final validation.
A regression test composes the real inherited recipe with all launcher
overrides using NeMo-RL's Hydra parser, resolves interpolation, and checks
that validation cannot be requested without validation data. All 12 settings
failed this test before the fix. Runtime verification and recovery receipts
are tracked in `RECOVERY.md`. No speedup is claimed before valid runs complete.
