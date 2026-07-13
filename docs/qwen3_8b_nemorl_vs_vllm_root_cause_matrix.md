# Qwen3-8B NeMo-RL vs vLLM Standalone Root-Cause Matrix

Updated: 2026-06-04 07:15 PDT

## Current Answer

Qwen3-8B SpecDec is not mechanically broken in NeMo-RL. The latest timing-split
20-step online aggregate shows that always-on K=1 improves the direct generation
path, while full GRPO E2E remains almost flat because generation is only a small
fraction of the step.

| Config | Jobs | Scope | Generate-path speedup | E2E speedup | Mean acceptance |
|---|---:|---|---:|---:|---:|
| K=1 always | 3149928/3149929 | 20-step timing split | 1.204x generation-worker TPS | 1.004x | ~61.2% log-derived |
| K=1 always | 3149928/3149929 | Step 5-20 steady window | 1.269x direct policy-generate TPS | 1.009x | ~61.2% log-derived |
| K=1 always | 3149454/3149455 | Step-5 isolation | 1.427x direct policy-generate TPS | N/A | 71.27% |

This means K=1 online drafter training now recovers a real generation-side
speedup. The full GRPO step is not decode-bound enough for a large E2E gain:
over Step 5-20, baseline generation is only about `4.37%` of total step time, so
the Amdahl projection is `1.009x`, matching the measured `1.009x` E2E speedup.
K=3 remains a weak Qwen3-8B path on this workload because later-position
acceptance is low and drafter/verification overhead dominates.

The remaining gap to the earlier synthetic vLLM standalone 2x-plus ceiling is
not a single NeMo-RL bug. It is a combination of:

1. synthetic standalone vs DAPOMath prompt/domain mismatch,
2. NeMo-RL E2E composition outside generation,
3. low K=3 later-position acceptance,
4. finish/sleep and full-loop overhead that sits outside target-token decode.

r97/r98/r99 tested the remaining engine-shape hypothesis directly. It improved
K=1 from `1.093x` to `1.120x`, but it did not close the gap to r22's
exact-engine `1.299x` because finish/sleep was still part of the reported
generation metric. Engine shape was a small factor; r106/r107 showed the larger
issue was metric composition, and r108/r109 confirmed the same read over a full
20-step run.

r104/r105 finished the first fixed post-update isolation, but still included
`finish_generation()` inside `timing/train/generation`. r106/r107 split that
finish/sleep path into `timing/train/generation_finish` and added a direct
`policy_generate_time_s` timer around `policy_generation.generate(...)`.

With that split, Qwen3-8B K=1 Step 5 reaches `1.417x` on the generation timer
and `1.427x` around the direct generate call, with `71.27%` acceptance and
acceptance length `1.713`. The earlier r105 `1.182x` was conservative because
vLLM sleep/finish was mixed into the generation metric.

r108/r109 then ran the same timing-split patch for the full 20 steps. K=1
always-on averaged `1.204x` generation-worker throughput and `1.004x` E2E
throughput. That confirms the root-cause split: the generate slice is positive,
while the full GRPO loop remains dominated by non-generation work.

When warmup is excluded, the generation gap narrows further: over Step 5-20,
K=1 reaches `1.256x` generation-worker throughput and `1.244x` generation
step-time speedup. The direct W&B `policy_generate_time_s` output-token speedup,
which is the closest NeMo metric to standalone vLLM output throughput, is
`1.269x` over Step 5-20. E2E remains only `1.009x`, matching the Amdahl
projection from the baseline generation share of about `4.37%`.

This means online drafter training and refit do recover a real K=1 generate-path
speedup in NeMo-RL. The remaining E2E gap is now primarily full-loop composition
and non-decode overhead, not a broken SpecDec path. r107 logs confirm that
online-trained draft weights reach the serving drafter (`[draft] Loading 14
trainer-owned draft weights into vLLM drafter.`).

## Evidence Matrix

| Area | Evidence | Root-cause read | Current status |
|---|---|---|---|
| FP8 / KV cache | Clean online and fixed/offline diagnostics use BF16 generation and KV cache `auto`. | FP8/KV q-scale was an earlier real issue, but it is not the current speedup gap. | Keep BF16/KV-auto as the comparison baseline. |
| Synthetic standalone ceiling | Qwen3-8B synthetic static standalone reaches K=2 bs32 `2.346x` with `96.3%` acceptance and K=3 bs32 about `2.145x`. | This is a best-case fixed-token synthetic ceiling, not a default-GRPO prediction. | Do not compare synthetic standalone directly to DAPOMath NeMo-RL E2E. |
| Real-prompt standalone | Qwen3-8B DAPOMath standalone bs32 regresses: K=1 `0.665x` / `61.11%`, K=2 `0.817x` / `46.91%`, K=3 `0.853x` / `36.62%`. | Public HF drafter quality is weaker on real DAPOMath prompts. This explains why synthetic standalone speedup does not transfer directly. | Strong evidence for prompt/domain mismatch. |
| Fixed/offline exact-engine diagnostic | r22 exact-engine/no-gate: K=1 `1.299x` with `64.61%` acceptance; K=3 `0.452x` with `40.66%` acceptance. | Native K=1 is the useful Qwen3-8B path. Fixed K=3 overhead is too high when later positions are weak. | Use native K=1, not K=3, for throughput-oriented Qwen3-8B runs. |
| Online finite-loss + gradfix | r94/r95/r96 completed 20 steps. K=1 `1.093x` generation, `1.029x` E2E, `61.20%` acceptance. K=3 `0.968x` generation, `0.995x` E2E, `37.46%` acceptance. | Online draft updates help K=1 generation, but E2E remains capped by low generation share. | Completed; documented in `docs/qwen3_8b_online_gradfix_r94_r96_20step.png`. |
| Engine-shaped online follow-up | r97/r98/r99 completed 20 steps with `max_num_seqs=32`, `max_num_batched_tokens=64000`, `gpu_memory_utilization=0.82`, metrics logger off, no chunked prefill, custom all-reduce disabled. K=1 `1.120x` generation, `1.060x` E2E, `61.75%` acceptance. K=3 `1.009x` generation, `1.041x` E2E, `37.40%` acceptance. | Engine shape gives K=1 only a small improvement and does not improve acceptance. K=3 remains too weak to recommend despite crossing 1.0x by a tiny margin. | Completed; documented in `docs/qwen3_8b_engine_shape_r97_r99_20step.png`. |
| Post-update timing-split diagnostic | r106 baseline `3149454` and r107 K=1 always `3149455` completed with `finish_generation()` split out. Baseline Step-5 generation `2.0613s`, `7948.49 tok/s/GPU`; K=1 Step-5 generation `1.4547s`, `11262.69 tok/s/GPU`; direct `policy_generate` speedup `1.427x`, acceptance `71.27%`, acceptance length `1.713`. | Online drafter weights reach vLLM and K=1 recovers real generate-path speedup. The earlier `1.182x` was finish-included and under-reported the decode-path gain. | Completed; root cause narrowed to E2E composition and non-decode overhead. |
| Timing-split 20-step aggregate | r108 baseline `3149928` and r109 K=1 always `3149929` completed 20/20 steps. K=1 averaged `12515.50` vs baseline `10393.49` generation-worker tok/s/GPU (`1.204x`), while E2E was `456.91` vs `454.93` tok/s/GPU (`1.004x`). Step 5-20 steady-state reaches `1.269x` direct policy-generate output TPS, `1.256x` generation-worker throughput, and `1.009x` E2E. | The full-loop aggregate confirms SpecDec helps generation but does not move default GRPO E2E much. Baseline generation share is only about `4.37%`, so Amdahl projection is `1.009x` for Step 5-20. | Completed; documented in `docs/qwen3_8b_r108_r109_policy_generate_window_speedups.png`. |
| Finish/sleep overhead | r106/r107 show `generation_finish` is `1.3353s` baseline and `1.5266s` for K=1, while direct `policy_generate_time_s` is `2.0219s` baseline and `1.4165s` K=1. | SpecDec speeds up decode, but it does not speed up vLLM sleep/reset; including finish in generation hides a substantial part of the gain. | Keep `generation_finish` separate for future reporting. |
| E2E composition | r94 baseline generation time share is 6.83% of total step time. Logprob-skip changed-objective diagnostic has higher generation share around 58% and larger E2E speedup. | Default GRPO work after rollout dominates E2E. Generation-only vLLM speedup cannot directly become E2E speedup. | This is algorithmic composition, not a single vLLM defect. |
| Gated/TLT mode | Prior gated jobs showed near-zero enabled ratio under high-concurrency rollout. | The request/token gate mostly disables speculation for current GRPO batch shapes. | Always-on is the meaningful performance path until gate thresholds are retuned. |
| Engine-shape mismatch | r97/r98/r99 added r22-style engine knobs and completed. K=1 moved from `1.093x` to `1.120x`, while acceptance stayed near `62%`. | Engine shape explained only a small slice. The later r106/r107 timer split shows finish/sleep was hiding more of the generate-path gain. | Completed. |

## Code-Level Evidence

| Code path | What it proves | Why it matters |
|---|---|---|
| `nemo_rl/algorithms/grpo.py`, generation timer | r106/r107 patch moved `policy_generation.finish_generation()` out of `with timer.time("generation")` and times it separately as `generation_finish`. | The latest `1.417x` generation speedup excludes vLLM sleep/reset; the older r104/r105 `1.182x` included finish and was conservative. |
| `nemo_rl/experience/rollouts.py`, policy generate timer | r106/r107 patch records `policy_generate_time_s` around the direct `policy_generation.generate(...)` call. | The direct generate path reaches `1.427x`, which is the closest NeMo-RL metric to standalone vLLM decode throughput. |
| `nemo_rl/algorithms/grpo.py`, `refit_policy_generation()` | Weight sync/refit is timed under prepare/weight-sync paths, not under `timing/train/generation`. | E2E includes refit, but generation throughput does not. The K=1 generation gap is not explained by refit overhead. |
| `nemo_rl/models/generation/vllm/vllm_generation.py`, `get_vllm_logger_metrics()` | SpecDecode counters are collected when `speculative_config` is present, even if the regular vLLM metrics logger is disabled. | r97/r98/r99 can turn off timeline metrics logger while still collecting acceptance counters. |
| `nemo_rl/models/generation/vllm/vllm_worker.py` | Generic `policy.generation.vllm_kwargs.*` are passed through to `vllm.LLM(...)`. | r97/r98/r99 can set `max_num_seqs`, `max_num_batched_tokens`, `enable_chunked_prefill`, and `disable_custom_all_reduce` without another source patch. |
| `nemo_rl/algorithms/grpo.py`, post-update stop patch | The online checkout now supports `NRL_STOP_AFTER_GENERATION_AFTER_STEP=N`, writes JSON after generation and finish timers are closed, and calls `policy_generation.get_step_metrics()` before returning. Scalar spec metrics are preserved instead of being skipped. | r106/r107 measure post-update generate speed, finish/sleep time, and acceptance after online draft training/refit has already happened. |
| `nemo_rl/models/generation/vllm/vllm_backend.py`, draft load log | vLLM now prints the count of trainer-owned `draft.*` weights loaded into the drafter. | This verifies whether online draft training is actually reaching the serving drafter during refit. |
| `nemo_rl/models/megatron/train.py` | Greedy `temperature=0.0` training no longer divides logits by zero after the finite-loss patch. | r94-r99 are valid online draft-training runs, unlike r86-r89 where draft loss was NaN. |
| `nemo_rl/models/policy/workers/megatron_policy_worker.py` | Grad offload fallback clears CUDA param grads instead of assigning CPU grads to CUDA params. | Multi-step online draft update/refit now completes. |

## Completed Engine-Shaped Follow-Up

Completed engine-shaped online follow-up:

| Job | Mode | Status |
|---:|---|---|
| 3146944 | r97 baseline, fixed512/greedy + enginebs32 | COMPLETED, W&B `yjl5cfd0` |
| 3146945 | r98 K=1 always, fixed512/greedy + enginebs32 | COMPLETED, W&B `0b06bzpm` |
| 3146946 | r99 K=3 always, fixed512/greedy + enginebs32 | COMPLETED, W&B `e4ohjsge` |

Final 20-step aggregate:

| Config | Gen throughput | Gen step time | E2E throughput | E2E step time | Mean acceptance |
|---|---:|---:|---:|---:|---:|
| r98 K=1 vs r97 baseline | 1.120x | 1.120x | 1.060x | 1.060x | 61.75% |
| r99 K=3 vs r97 baseline | 1.009x | 1.009x | 1.041x | 1.041x | 37.40% |

The final read is that engine shape is a small positive factor for K=1 but not
the main limiter. K=3 still does not have enough acceptance margin to be useful.

## Practical Recommendation

For Qwen3-8B, use always-on native K=1 as the only currently positive online
path. Treat K=3 as marginal/noisy rather than useful under the public HF drafter
on DAPOMath-style prompts. Report `generation_finish` separately from generation
throughput. For E2E speedup, optimize or overlap non-generation GRPO work;
SpecDec alone cannot overcome a small generation share in the completed online
setup.
