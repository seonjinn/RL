# Qwen3-8B NeMo-RL SpecDec Root-Cause Notes

Updated: 2026-06-04 06:48 PDT

## 2026-06-04 06:48 PDT r108/r109 Completed: Timing-Split 20-Step Aggregate

I ran the same timing-split patch for a full 20-step aggregate instead of
stopping at Step 5:

| Job | Mode | Status | W&B |
|---:|---|---|---|
| `3149928` | r108 baseline | COMPLETED, 00:35:29 | `jorrhdfl` |
| `3149929` | r109 K=1 always | COMPLETED, 00:36:51 | `1k74f3rk` |

20-step mean result:

| Metric | Baseline | K=1 always | Speedup / ratio |
|---|---:|---:|---:|
| Generation worker throughput | 10393.49 tok/s/GPU | 12515.50 tok/s/GPU | 1.204x |
| E2E throughput | 454.93 tok/s/GPU | 456.91 tok/s/GPU | 1.004x |
| Generation time | 2.0975s | 1.8180s | 1.154x |
| `generation_finish` time | 1.4505s | 1.5250s | 0.951x |
| Total step time | 47.916s | 47.739s | 1.004x |
| Policy training time | 19.696s | 19.821s | 0.994x |
| Policy/reference logprob time | 16.476s | 16.661s | 0.989x |

Warmup-excluded steady-state windows:

| Window | Direct policy-generate output speedup | Gen throughput speedup | Gen step-time speedup | E2E throughput speedup | Amdahl E2E projection |
|---|---:|---:|---:|---:|---:|
| Step 1-20 | 1.224x | 1.204x | 1.154x | 1.004x | 1.006x |
| Step 2-20 | 1.251x | 1.227x | 1.213x | 1.005x | 1.008x |
| Step 5-20 | 1.269x | 1.256x | 1.244x | 1.009x | 1.009x |

Interpretation: the 20-step aggregate confirms the root-cause picture. K=1
always-on improves the NeMo generation worker throughput by `1.204x`, but E2E is
almost flat at `1.004x` because the step is dominated by policy/reference
logprobs, policy training, prepare/refit, and vLLM finish/sleep. The
`generation_finish` path is slightly slower under K=1 and is not accelerated by
SpecDec. If cold-start steps are excluded, the generation slice improves to
`1.256x` over Step 5-20, and the W&B direct `policy_generate_time_s` output-token
speedup is `1.269x`. Baseline generation is only about `4.37%` of the step, so
Amdahl's law projects only `1.009x` E2E. The measured E2E throughput speedup
over Step 5-20 is also `1.009x`.

This is weaker than the isolated r106/r107 Step-5 `policy_generate` speedup
(`1.427x`) because the full 20-step run includes step-to-step variance, different
node/runtime conditions, and the NeMo summary metric uses total rollout tokens
over `timing/train/generation`. It is still directionally consistent: SpecDec
helps the generate slice, but default GRPO is not decode-bound enough for large
E2E gains.

Visible vLLM worker log lines in r109 show mean acceptance around `61.2%` and
mean acceptance length around `1.61`; those are supporting log-derived values
because repeated worker log lines are compressed. The exact throughput and time
numbers above come from the NeMo per-step summary tables.

Artifacts:

- `docs/qwen3_8b_r108_r109_timing_split_20step_metrics.csv`
- `docs/qwen3_8b_r108_r109_timing_split_20step_summary.csv`
- `docs/qwen3_8b_r108_r109_timing_split_20step_window_summary.csv`
- `docs/qwen3_8b_r108_r109_policy_generate_wandb_metrics.csv`
- `docs/qwen3_8b_r108_r109_policy_generate_window_summary.csv`
- `docs/qwen3_8b_r108_r109_timing_split_20step.png`
- `docs/qwen3_8b_r108_r109_timing_split_20step_window_speedups.png`
- `docs/qwen3_8b_r108_r109_policy_generate_window_speedups.png`

## 2026-06-04 05:58 PDT r106/r107 Completed: Generation Finish Split Confirms Real K=1 Speedup

r104/r105 still timed `policy_generation.finish_generation()` inside
`timing/train/generation`. That made the reported Step-5 generation speedup look
too low because vLLM sleep/reset work was mixed into the generate-path timer. I
patched the remote online checkout to:

- time `finish_generation()` separately as `timing/train/generation_finish`,
- add `policy_generate_time_s` inside `nemo_rl/experience/rollouts.py` around
  the actual `policy_generation.generate(...)` call,
- write both values into the stop-after-generation JSON.

Completed timing-split jobs:

| Job | Mode | Status | W&B |
|---:|---|---|---|
| `3149454` | r106 baseline | COMPLETED, 00:23:55 | `yvum529g` |
| `3149455` | r107 K=1 always | COMPLETED, 00:21:58 | `v4nuvfm3` |

Step-5 timing-split result:

| Mode | Generation time | Gen tok/s/GPU | Policy-generate time | Policy-generate tok/s/GPU | Finish time | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2.0613s | 7948.49 | 2.0219s | 8103.34 | 1.3353s | N/A |
| K=1 always | 1.4547s | 11262.69 | 1.4165s | 11566.89 | 1.5266s | 71.27% |
| K=1 speedup | 1.417x | 1.417x | 1.427x | 1.427x | 0.875x | 71.27% |

Interpretation: online drafter training is now confirmed to work on the actual
generate path. The key number is no longer r105's finish-included `1.182x`; with
`finish_generation()` split out, Qwen3-8B K=1 reaches `1.417x` on the generation
timer and `1.427x` around the direct `policy_generation.generate(...)` call.

The remaining gap is therefore mostly not a broken SpecDec path. It is a metric
composition and E2E-composition issue:

- vLLM sleep/finish is not accelerated by SpecDec and is slightly slower for K=1
  in this diagnostic (`1.5266s` vs `1.3353s`).
- full GRPO E2E still includes prepare/refit, reward processing, logprob
  inference, advantage computation, training, and offload work.
- synthetic vLLM standalone remains an optimistic ceiling for DAPOMath-style
  prompts; real-prompt acceptance and workload shape are different.

Artifact:

- `docs/qwen3_8b_r106_r107_timing_split_step5_summary.csv`

## 2026-06-04 05:18 PDT r104/r105 Completed: Fixed Step-5 Generation-Only JSON

r102/r103 proved that the stop-after-generation hook could write JSON, but the
hook was still inside `with timer.time("generation")`. That meant the current
generation interval had not been committed to `timer.get_timing_metrics()` yet.
I moved the hook outside the generation timer context, called
`policy_generation.get_step_metrics()` directly, and fixed scalar metric
aggregation so spec counters such as `vllm/spec_acceptance_rate` are not
dropped.

Completed fixed diagnostic jobs:

| Job | Mode | Status | Purpose |
|---:|---|---|---|
| `3148859` | r104 baseline | COMPLETED, 00:21:31, W&B `4n9pgwf5` | SpecDec disabled; stop after Step 5 generation. |
| `3148860` | r105 K=1 always | COMPLETED, 00:21:42, W&B `4ili8xky` | K=1 always-on; stop after Step 5 generation. |

Step-5 generation-only result:

| Mode | Generation time | Gen tokens/s/GPU | Acceptance | Acceptance length |
|---|---:|---:|---:|---:|
| Baseline | 3.4827s | 4704.36 | N/A | 1.000 |
| K=1 always | 2.9458s | 5561.73 | 71.21% | 1.712 |
| K=1 speedup | 1.182x | 1.182x | 71.21% | 1.712 |

Interpretation at the time: online draft training plus refit did recover K=1
speed by Step 5, and the r105 logs showed `[draft] Loading 14 trainer-owned draft
weights into vLLM drafter.` The later r106/r107 split showed that this `1.182x`
number was conservative because `finish_generation()` was still included in the
generation timer.

Artifact:

- `docs/qwen3_8b_r104_r105_postupdate_step5_genonly_summary.csv`

## 2026-06-04 04:00 PDT r100/r101 Completed: Post-Update Generation/Refit Isolation

I found a concrete diagnostic gap in the online checkout: launch scripts could
pass `NRL_STOP_AFTER_GENERATION`, but `nemo_rl/algorithms/grpo.py` in
`SpecDec-RL-origin-main-online-20260603` did not implement that env var. This
meant r97/r98/r99 could not directly isolate generation-only speed after online
training/refit.

I patched the remote online checkout with:

- `NRL_STOP_AFTER_GENERATION` support.
- `NRL_STOP_AFTER_GENERATION_AFTER_STEP=N`, which runs full online steps until
  step `N`, then logs rollout/timing metrics and exits immediately after that
  generation.
- A vLLM backend diagnostic log showing how many trainer-owned `draft.*` weights
  are loaded into the serving drafter during refit.

Validation passed remotely:

- `python3 -m py_compile nemo_rl/algorithms/grpo.py nemo_rl/models/generation/vllm/vllm_backend.py`
- `bash -n experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh experiments/eagle3_online/submit_qwen8_r100r101online_postupdate_genonly_step5.sh`

Completed jobs:

| Job | Mode | Status | Purpose |
|---:|---|---|---|
| `3147906` | r100 baseline | COMPLETED, 00:24:25, W&B `j5rjbkqy` | Same r97 engine-shaped online baseline, stopped after Step 5 generation. |
| `3147907` | r101 K=1 always | COMPLETED, 00:22:55, W&B `bmnaumpg` | Same r98 engine-shaped K=1 always-on online run, stopped after Step 5 generation. |

The Step 5 stop log fired in both jobs, but W&B history retained completed
Step 1-4 timing rows only. The reliable numeric comparison is therefore Step 1-4
full-loop history, with Step 2-4 used as the post-update subset.

| Step | K=1 gen speedup | K=1 E2E step-time speedup | K=1 acceptance |
|---:|---:|---:|---:|
| 1 | 0.662x | 0.959x | 42.61% |
| 2 | 1.227x | 0.954x | 74.51% |
| 3 | 0.989x | 0.981x | 55.59% |
| 4 | 0.997x | 0.991x | 49.60% |

Aggregates:

| Window | Gen time speedup | Gen TPS mean speedup | E2E step-time speedup | Mean acceptance |
|---|---:|---:|---:|---:|
| Step 1-4 | 0.910x | 0.979x | 0.971x | 55.58% |
| Step 2-4 | 1.062x | 1.068x | 0.975x | 59.90% |

r101 logs repeatedly printed:
`[draft] Loading 14 trainer-owned draft weights into vLLM drafter.`
That confirms the online-trained draft weights are reaching the serving vLLM
drafter during refit.

Interpretation: post-update K=1 does not recover standalone-like speedup. Step 4
is effectively flat on generation and the Step 2-4 post-update average is only
`1.062x` generation while E2E still slows down. The remaining gap to r22
exact-engine/no-gate K=1 `1.299x` is not explained by reward/logprob/training
time being counted inside generation. The root cause is now narrowed to the
online generation/refit state and unstable/insufficient drafter acceptance on
the DAPOMath workload.

Artifact:

- `docs/qwen3_8b_r100_r101_postupdate_history.csv`

## 2026-06-04 03:15 PDT r97/r98/r99 Completed: Engine Shape Helps K=1 Slightly, Does Not Close the Gap

The standalone-shaped engine follow-up completed successfully:
r97 baseline `3146944`, r98 K=1 always `3146945`, and r99 K=3 always
`3146946`. These runs kept the successful finite-loss + gradfix online path and
added the vLLM engine knobs used to isolate the remaining gap:
`gpu_memory_utilization=0.82`, metrics logger off, `max_num_seqs=32`,
`max_num_batched_tokens=64000`, `enable_chunked_prefill=false`, and
`disable_custom_all_reduce=true`.

| Config | Job | Gen throughput speedup | Gen step-time speedup | E2E throughput speedup | E2E step-time speedup | Mean acceptance | Mean acceptance length |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=1 always | `3146945` | 1.120x | 1.120x | 1.060x | 1.060x | 61.75% | 1.617 |
| K=3 always | `3146946` | 1.009x | 1.009x | 1.041x | 1.041x | 37.40% | 2.122 |

Interpretation: engine shape is a small positive factor for K=1, moving the
generation result from r95's `1.093x` to r98's `1.120x`. It does not explain the
remaining gap to r22 exact-engine/no-gate K=1 `1.299x`, and it does not improve
acceptance (`61.20% -> 61.75%`). K=3's acceptance is also unchanged
(`37.46% -> 37.40%`) and the generation speedup is only `1.009x`, so K=3 is
still not a useful Qwen3-8B online path.

Artifacts:

- `docs/qwen3_8b_engine_shape_r97_r99_20step_metrics.csv`
- `docs/qwen3_8b_engine_shape_r97_r99_20step_summary.csv`
- `docs/qwen3_8b_engine_shape_r97_r99_20step.png`
- `scripts/plot_qwen3_8b_engine_shape_r97_r99_20step.py`

## 2026-06-04 02:21 PDT r94/r95/r96 Completed: K=1 Recovers Generation, K=3 Still Loses

The longer 20-step finite-loss + gradfix follow-up completed successfully:
r94 baseline `3145919`, r95 K=1 always `3145920`, and r96 K=3 always
`3145921`. All runs used `policy.draft.enabled=true`, BF16 vLLM generation, KV
cache `auto`, fixed 512-token greedy decoding, 4 generation workers, generation
batch size 32, and `train_global_batch_size=128`.

| Config | Job | Gen throughput speedup | Gen step-time speedup | E2E throughput speedup | E2E step-time speedup | Mean acceptance | Mean acceptance length |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=1 always | `3145920` | 1.093x | 1.093x | 1.029x | 1.029x | 61.20% | 1.612 |
| K=3 always | `3145921` | 0.968x | 0.968x | 0.995x | 0.995x | 37.46% | 2.124 |

Interpretation: K=1 online drafter training is now demonstrably useful for the
generation slice under this fixed512 worker≈32 setup. It recovers a `1.093x`
generation speedup over the matched online baseline. The E2E gain is only
`1.029x` because baseline generation is still only `6.83%` of total step time.

K=3 does not work yet for this Qwen3-8B online workload. Mean acceptance is only
`37.46%`, and several steps fall in the `20-30%` range. The extra speculative
tokens add verification/drafter overhead faster than they reduce target decode
work, so aggregate generation throughput is a `0.968x` slowdown and E2E is
essentially flat/slightly negative.

Artifacts:

- `docs/qwen3_8b_online_gradfix_r94_r96_20step_metrics.csv`
- `docs/qwen3_8b_online_gradfix_r94_r96_20step_summary.csv`
- `docs/qwen3_8b_online_gradfix_r94_r96_20step.png`
- `scripts/plot_qwen3_8b_online_gradfix_20step.py`

## 2026-06-04 01:30 PDT r92/r93 Completed: Online Draft Update Helps Acceptance, Not Yet E2E

The clean finite-loss + gradfix rerun completed successfully. This validates
both code fixes: greedy training no longer produces NaN draft loss, and
`offload_before_refit` no longer fails after Step 1.

| Step | r92 baseline gen tok/s/GPU | r93 K=1 gen tok/s/GPU | Gen speedup | E2E speedup | r93 acceptance | r93 draft loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5247.03 | 3990.15 | 0.760x | 1.020x | 42.24% | 9.0968 |
| 2 | 6557.43 | 7281.81 | 1.110x | 0.989x | 70.42% | 7.0224 |
| 3 | 6556.19 | 6284.84 | 0.959x | 0.935x | 56.98% | 7.9329 |
| 4 | 6727.15 | 6386.33 | 0.949x | 0.964x | 48.91% | 8.2979 |
| 5 | 6256.15 | 7616.58 | 1.217x | 0.985x | 71.70% | 7.1756 |

Aggregate over all five steps: generation time speedup `0.964x`, E2E throughput
speedup `0.980x`, mean acceptance `58.05%`. If Step 1 is excluded to focus on
post-update behavior, Step 2-5 generation speedup is `1.049x` and E2E is
`0.968x`.

Interpretation: online drafter training is now functional and does improve
acceptance materially after the first update. The main remaining problem is that
the improved acceptance is not stable enough, and the full NeMo-RL GRPO step is
not decode-bound enough, to recover standalone-like performance. The gap is no
longer explained by a broken training path alone.

Artifacts:

- `docs/qwen3_8b_online_gradfix_r92_r93_metrics.csv`
- `docs/qwen3_8b_online_gradfix_r92_r93.png`
- `scripts/plot_qwen3_8b_online_gradfix.py`
- `experiments/eagle3_online/remote_patches_qwen8_online_finiteloss_gradfix.patch`

The 20-step follow-up described above has now completed. It shows that K=1 can
recover generation speedup after online updates, while K=3 still loses because
its acceptance remains too low.

## 2026-06-04 00:58 PDT r90/r91 Finite Loss Confirmed, Grad Offload Fix Submitted

r90/r91 validated the `temperature=0.0` training-loss fix, but they did not
complete the intended 5-step online-drafter measurement. Both failed after Step
1 when entering the refit/offload path for Step 2.

| Job | Mode | Status | Step-1 Gen tok/s/GPU | Step-1 Speedup | Step-1 Step Time | Step-1 Acceptance | Step-1 Draft Loss |
|---:|---|---|---:|---:|---:|---:|---:|
| `3144473` | r90 greedy paired baseline, patched finite-loss path | FAILED before Step 2 | 4920.99 | 1.000x | 58.17s | n/a | 9.0362 |
| `3144474` | r91 greedy K=1 always, patched finite-loss path | FAILED before Step 2 | 3910.24 | 0.794x | 57.25s | 42.24% | 9.0676 |

Interpretation: the `train.py` patch worked. Greedy training no longer reports
`Loss/Draft Loss/Generation KL Error = NaN`; r90/r91 both show finite draft loss
around `9.0`. However, Step-1 K=1 generation is still slower than the matched
baseline, and acceptance is still around `42%`, so this is not yet a performance
recovery.

The new blocker was in
`nemo_rl/models/policy/workers/megatron_policy_worker.py`. `offload_before_refit`
calls `move_model(..., device="cpu", move_params=False, move_grads=True)` to
drop grad buffers while keeping parameters on CUDA. The fallback buffer path
moved `param.grad` to CPU and assigned it back to a CUDA parameter, causing:
`RuntimeError: attempting to assign a gradient with device type 'cpu' to a
tensor with device type 'cuda'`.

Patch applied: when `device == "cpu"` and `move_params == False`, the fallback
path now clears `param.grad = None` rather than assigning a CPU grad tensor.
Remote `python3 -m py_compile` passed.

Clean rerun submitted:

| Job | Mode | Status |
|---:|---|---|
| `3144992` | r92 greedy baseline, finite-loss + gradfix, 5 steps | COMPLETED later, 00:24:01 |
| `3144993` | r93 greedy K=1 always, finite-loss + gradfix, 5 steps | COMPLETED later, 00:25:47 |

## 2026-06-04 00:36 PDT Greedy Training NaN Root Cause and r90/r91 Submitted

A second code-level issue was found after r88/r89 completed. It does not explain
the Step-1 generation acceptance, but it does explain why all greedy online
training diagnostics reported `draft_loss=nan` and `grad_norm=0`.

| Area | Finding |
|---|---|
| Bug | `nemo_rl/models/megatron/train.py::apply_temperature_scaling()` divided training logits by `sampling_params.temperature` whenever temperature was not `1.0`. The greedy diagnostics use `temperature=0.0`, so training logits became Inf/NaN. |
| Evidence | r86/r87/r88/r89 logs show `Loss: nan`, `Draft Loss: nan`, `Generation KL Error: nan`, and W&B `train/grad_norm=0`. In contrast, default-style r84/r85 used `temperature=1.0` and had finite draft losses: r84 `9.0157`, r85 `8.8250`. |
| Patch | Remote `train.py` now skips temperature scaling when `sampling_params is None` or `temperature <= 0.0`. `python3 -m py_compile nemo_rl/models/megatron/train.py` passed. |

Follow-up jobs were submitted to verify finite-loss greedy online training and
measure whether online draft updates affect later-step acceptance:

| Job | Mode | Status | Purpose |
|---:|---|---|---|
| `3144473` | r90 greedy paired baseline, patched finite-loss path, 5 steps | PENDING | Same as r88 but with patched `train.py` and fresh actor venv suffix |
| `3144474` | r91 greedy K=1 always, patched finite-loss path, 5 steps | PENDING | Same as r89 but with patched `train.py`, fresh actor venv suffix, and five rollout/train/refit cycles |

Expected read: if `draft_loss` becomes finite but Step 2-5 acceptance remains
near `42%`, then low public-drafter/domain acceptance is robust to the tested
online draft update. If acceptance improves after Step 1, online draft
adaptation becomes the next fix path and should be tuned with draft LR/loss
weight and longer runs.

## 2026-06-04 00:31 PDT Post-Patch Greedy r88/r89 Completed

The post-worker-patch matched rerun completed. This closes the last ambiguity
that r87 might have missed worker-side fixed-length `SamplingParams` propagation:
r88/r89 rebuilt actor venvs from the patched online checkout, and r89's vLLM
engine logged active `SpeculativeConfig(method='eagle3',
model='RedHatAI/Qwen3-8B-speculator.eagle3', num_spec_tokens=1)`.

| Job | Mode | Status | Gen tok/s/GPU | Gen speedup | E2E tok/s/GPU | E2E speedup | Gen time | Step time | Acceptance |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3143675` | r88 greedy paired baseline, online draft enabled, vLLM SpecDec off | COMPLETED | 5471.97 | 1.000x | 363.54 | 1.000x | 3.86s | 58.14s | n/a |
| `3143676` | r89 greedy K=1 always, online draft enabled, vLLM EAGLE3 on | COMPLETED | 4005.30 | 0.732x | 374.12 | 1.029x | 5.28s | 56.49s | 42.01% |

Interpretation: post-patch r89 is still a generation slowdown. Acceptance is
`42.01%`, essentially the same as r87's `41.82%`, and generation throughput is
`0.732x` of the matched baseline. Therefore missing fixed-length/stop-disable
`SamplingParams` propagation was not the root cause of the Qwen3-8B online K=1
slowdown.

The small E2E `1.029x` should not be reported as generation acceleration.
Generation itself is slower by `36.6%`, and baseline generation is only `6.64%`
of the step. The strongest current root-cause read remains public Qwen3-8B
drafter mismatch on the DAPOMath/online prompt distribution plus online vLLM
drafter/verification overhead.

## 2026-06-03 23:46 PDT Greedy Online r86/r87 Diagnostic Completed

The matched Qwen3-8B online-drafter greedy/fixed512 diagnostic completed. It
isolated whether r84's low acceptance was mainly from default-GRPO stochastic
sampling by keeping the r84/r85 online path (`policy.draft.enabled=true`, BF16
generation, KV `auto`, four generation workers, worker≈32, fixed 512-token
decode) and adding standalone-like greedy sampling: `temperature=0.0`,
`top_p=1.0`, `top_k=-1`, fixed 512-token output, and no stop strings/token IDs.

| Job | Mode | Status | Gen tok/s/GPU | Gen speedup | E2E tok/s/GPU | E2E speedup | Gen time | Step time | Acceptance |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3142745` | r86 greedy paired baseline, online draft enabled, vLLM SpecDec off | COMPLETED | 5364.60 | 1.000x | 357.13 | 1.000x | 3.94s | 59.18s | n/a |
| `3142746` | r87 greedy K=1 always, online draft enabled, vLLM EAGLE3 on | COMPLETED | 4059.60 | 0.757x | 380.83 | 1.066x | 5.21s | 55.50s | 41.82% |

Interpretation: greedy/fixed512 did not recover acceptance. r87 K=1 acceptance
is `41.82%`, very close to r84's `40.80%`, and generation throughput is still a
slowdown (`0.757x`; generation time +32.1%). Therefore the r84 negative result
is not primarily explained by stochastic/default-GRPO sampling. The stronger
root-cause read is Qwen3-8B public HF drafter mismatch on the DAPOMath/online
prompt distribution plus online vLLM drafter/verification overhead.

The r87 E2E number (`1.066x`) should not be reported as a generation speedup.
Baseline generation is only `6.66%` of the step, and one-step non-generation
timing varied enough to make total step time lower even though generation itself
was slower. Both r86 and r87 also reported `train/draft_loss=nan`, so these are
throughput/acceptance diagnostics rather than evidence of healthy online draft
loss training.

After the metric read, the remote online checkout was patched so future
standalone-like reruns consistently pass fixed-length controls into vLLM
`SamplingParams`: `NRL_VLLM_GENERATION_MIN_TOKENS`,
`NRL_VLLM_GENERATION_IGNORE_EOS`,
`NRL_VLLM_GENERATION_DISABLE_STOP_STRINGS`, and
`NRL_VLLM_GENERATION_DISABLE_STOP_TOKEN_IDS`. Local and remote `python3 -m
py_compile nemo_rl/models/generation/vllm/vllm_worker.py` both passed.

## 2026-06-03 23:10 PDT Online Drafter Matched r84/r85 Result

The first matched Qwen3-8B online-drafter smoke completed end-to-end. Both jobs
used `policy.draft.enabled=true`, BF16 generation, KV cache dtype `auto`, fixed
512-token decode (`min_tokens=512`, `ignore_eos=true`, stop strings/token IDs
disabled), `NUM_PROMPTS=4`, `NUM_GENERATIONS=32`, `TRAIN_GLOBAL_BATCH_SIZE=128`,
and four generation workers, so each worker saw about 32 responses. The only
intended difference is that r84 enabled vLLM EAGLE3 K=1 always-on SpecDec while
r85 disabled vLLM `speculative_config`.

| Job | Mode | Status | Gen tok/s/GPU | Gen speedup | E2E tok/s/GPU | E2E speedup | Gen time | Step time | Acceptance |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3142137` | r85 paired baseline, online draft enabled, vLLM SpecDec off | COMPLETED | 5302.53 | 1.000x | 375.78 | 1.000x | 3.99s | 56.25s | n/a |
| `3141902` | r84 K=1 always, online draft enabled, vLLM EAGLE3 on | COMPLETED | 3206.21 | 0.605x | 343.62 | 0.914x | 6.59s | 61.51s | 40.80% |

Interpretation: the online drafter path is runnable now, including generation,
reward processing, policy/reference logprobs, policy training, and draft loss.
However, this workload is a slowdown: generation time increased by 65.4%, and
E2E step time increased by 9.4%. The immediate cause is low K=1 acceptance
(40.80%) plus drafter/verification overhead exceeding the target-side work saved.
The E2E impact is smaller than the generation slowdown because baseline
generation is only about 7.1% of step time.

## 2026-06-03 Latest Qwen3-8B Diagnostic Result

This section supersedes the older running-status notes below for
`3132444/3132635/3132636`. All three changed-objective diagnostic jobs completed
and emitted both generation and E2E step summaries. The experiment is still not
a default GRPO result: it uses BF16 generation, KV cache `auto`,
`force_on_policy_ratio=true`, `reference_policy_kl_penalty=0`, and the
policy/reference logprob-skip diagnostic path.

| Job | Mode | Status | Mean gen tok/s/GPU | Gen speedup | Mean E2E tok/s/GPU | E2E speedup | Acceptance | Inferred generation share |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| `3132444` | baseline r5 | COMPLETED, 01:53:34 | 4260.76 | 1.000x | 2608.80 | 1.000x | n/a | n/a |
| `3132635` | K=1 always r6 | COMPLETED, 01:39:09 | 5498.34 | 1.290x | 3004.60 | 1.152x | 60.57% | 58.5% |
| `3132636` | K=3 always r6 | COMPLETED, 01:38:41 | 5721.98 | 1.343x | 3065.85 | 1.175x | 35.75% | 58.4% |

Step-level matched comparison:

| Step | Baseline gen / E2E tok/s/GPU | K=1 gen / E2E speedup | K=1 acceptance | K=3 gen / E2E speedup | K=3 acceptance |
|---|---:|---:|---:|---:|---:|
| 1 | 4319.01 / 2608.47 | 1.314x / 1.168x | 63.93% | 1.343x / 1.185x | 38.72% |
| 2 | 4168.85 / 2575.65 | 1.319x / 1.166x | 59.22% | 1.420x / 1.205x | 34.54% |
| 3 | 4362.34 / 2637.33 | 1.268x / 1.137x | 58.43% | 1.308x / 1.152x | 34.06% |
| 4 | 4192.84 / 2613.76 | 1.262x / 1.136x | 60.68% | 1.303x / 1.159x | 35.67% |
| Mean | 4260.76 / 2608.80 | 1.290x / 1.152x | 60.57% | 1.343x / 1.175x | 35.75% |

Updated chart:

- `docs/specdec_qwen3_8b_logprob_skip_diag_step1_step4.png`

Interpretation:

1. Qwen3-8B SpecDec is not mechanically broken in NeMo-RL. With the logprob
   bottleneck reduced, always-on K=1/K=3 improves rollout generation by
   1.290x/1.343x.
2. The E2E gain is smaller, 1.152x/1.175x, because generation accounts for only
   about 58% of the matched step time in this diagnostic. The remaining
   non-generation work caps E2E even when generation improves.
3. This still does not match the synthetic standalone K=3 ceiling. The main
   reasons are workload/domain mismatch, low later-position K=3 acceptance, and
   NeMo-RL E2E composition. For Qwen3-8B, the real-prompt DAPOMath standalone
   isolation also regressed at K=3 with 36.62% acceptance, which is close to the
   NeMo-RL K=3 acceptance level here.
4. The issue is not explained by FP8 generation or KV-cache quantization in this
   diagnostic: generation is BF16 and KV cache dtype is `auto`.
5. Online drafter training is now measured for one matched smoke. The r73-r83
   sequence progressively cleared several blockers: policy-side TransformerEngine
   RMSNorm, local transformer spec conversion mapping, ModelOpt EAGLE decoder
   norms, Megatron DDP overlap, model-parallel helper signature mismatch, Slurm
   GRES parsing, and Ray Python/version mismatch. r84/r85 then completed
   end-to-end, but the result is negative: K=1 always generation `0.605x`, E2E
   `0.914x`, and acceptance `40.80%` versus the matched online baseline.

## 2026-06-03 22:20 PDT Follow-up: Online Drafter and 235B Decode-Heavy Evidence

Qwen3-8B online K=1 always short512 progressed past the earlier model/code
blockers, but still has no completed NeMo-RL rollout metric:

| Job | Status | What it proves | Metric status |
|---:|---|---|---|
| `3139703` r80 | FAILED, 00:35:10 | Disabling Megatron DDP overlap avoided the r79 `Communication call has not been issued` assertion and reached optimizer/update-success reduction | vLLM internal acceptance around 57.2%; no final rollout metric |
| `3140266` r81 | FAILED, 00:02:03 | model-parallel helper compatibility patch was ready, but launcher GRES parsing failed before Ray driver | no model metric |
| `3140576` r82 | FAILED, 00:08:09 | GRES parsing was fixed; failure was Ray head/driver version mismatch | no model metric |
| `3140704` r83 | CANCELLED by request, 00:11:31 | Ray head and driver were aligned to Python 3.13.13 / Ray 2.54.0; Ray init, MasterConfig, data loaders, vLLM init, and CUDA Graph capture succeeded | cancelled before rollout/acceptance aggregate |

The new Qwen3-235B standalone decode-heavy sweep is also important for the
root-cause interpretation. It used target `Qwen/Qwen3-235B-A22B`, drafter
`nvidia/Qwen3-235B-A22B-Eagle3`, ISL/OSL `1000/10000`, TP=4, one node / four
GPUs, BF16/auto KV, CUDA Graph enabled, custom all-reduce disabled, and vLLM log
version `v0.17.0`.

| Batch size | K=1 acceptance | K=1 throughput speedup |
|---:|---:|---:|
| 1 | 92.10% | 0.629x |
| 2 | 85.17% | 0.608x |
| 4 | 95.94% | 0.661x |
| 8 | 94.34% | 0.676x |
| 16 | 98.39% | 0.732x |
| 32 | 93.59% | 0.823x |

Artifacts:

- `docs/qwen3_235b_vllm_decodeheavy10k_metrics.csv`
- `docs/qwen3_235b_vllm_decodeheavy10k_speedup_acceptance.png`

This strengthens the main conclusion: acceptance is necessary but insufficient.
Even with 85-98% acceptance, SpecDec can slow down when drafter, verification,
scheduler, cache, and parallelism overhead exceed the target-side work saved in
the tested load regime.

Code audit addendum:

| Code path | Finding |
|---|---|
| `vllm_worker.py::_build_sampling_params()` | SpecDec runs now omit vLLM request logprobs unless explicitly forced. The guard documents the vLLM V1 behavior that logprob requests disable SpecDec, so zero-acceptance logprob artifacts are not used as throughput evidence. |
| `grpo.py::_repair_specdec_generation_logprobs_if_safe()` and `_populate_policy_and_reference_logprobs()` | Behavior-logprob repair is limited to synchronous on-policy SpecDec, and policy/reference fprop skipping is available only through the explicit changed-objective diagnostic path. |
| `specdec_runtime_gate_patch.py` | The TLT-style gate computes active request pressure plus scheduled-token pressure, disables runner draft proposal, zeroes draft ids, and scrubs scheduler speculative-token metadata. This code path explains why gated results are baseline-like when enabled ratio is near zero. |
| `submit_qwen8_r83...raymatch_retry.sh` | r83 is the current online restart point: local transformer spec, recursive EAGLE torch RMSNorm, DDP no-overlap, Ray/Python version alignment, max sequence `2560`, and max new tokens `512`. The job was cancelled after vLLM CUDA Graph capture and before rollout metrics. |
| `submit_qwen8_r84online_short512_k1always_worker32_fixedlen_raymatch.sh` | Completed online metric smoke as job `3141902`. It keeps r83 fixes and adds standalone-like workload controls: K=1 always-on, fixed 512-token decode, BF16/KV-auto, and `NUM_PROMPTS=4`, `NUM_GENERATIONS=32`, `TRAIN_GLOBAL_BATCH_SIZE=128` so four generation workers see about 32 responses each. It emitted generation `3206.21` tok/s/GPU, E2E `343.62` tok/s/GPU, and acceptance `40.80%`. |
| `submit_qwen8_r85online_short512_baseline_worker32_fixedlen_raymatch.sh` | Completed paired online baseline as job `3142137`. It keeps the same worker≈32/fixed512 workload and `policy.draft.enabled=true`, but disables vLLM `speculative_config` so r84 can be compared against a matched online-drafter baseline. It emitted generation `5302.53` tok/s/GPU and E2E `375.78` tok/s/GPU. |
| `submit_qwen8_r86online_short512_greedy_baseline_worker32_fixedlen_raymatch.sh` | Completed paired greedy online baseline as job `3142745`. It keeps r85's matched baseline shape but adds standalone-like `temperature=0.0`, `top_p=1.0`, `top_k=-1`, fixed 512-token decode, and no stop strings/token IDs. It emitted generation `5364.60` tok/s/GPU, E2E `357.13` tok/s/GPU, generation time `3.94s`, and step time `59.18s`. |
| `submit_qwen8_r87online_short512_greedy_k1always_worker32_fixedlen_raymatch.sh` | Completed paired greedy online K=1 always smoke as job `3142746`. It keeps r84's K=1 online SpecDec path but adds the same greedy/fixed decode controls as r86 and allows throughput-only greedy logprob repair mismatch. It emitted generation `4059.60` tok/s/GPU (`0.757x`), E2E `380.83` tok/s/GPU (`1.066x`), generation time `5.21s`, step time `55.50s`, and acceptance `41.82%`. |

## 2026-06-03 19:59 PDT r78/r79 EAGLE Decoder Norm Update

Qwen3-8B online r78 `3137739` is final:

| Job | Mode | Status | Positive evidence | Failure |
|---:|---|---|---|---|
| `3137739` | online K=1 always, short512, local spec + HF reconvert + dense Qwen3Bridge mapping fix + draft torch RMSNorm | FAILED, 00:52:59, ExitCode 1:0 | The r77 conversion failure did not recur; vLLM generation emitted Avg Draft acceptance 63.0% on one worker and 57.5% repeated on the others; policy logprobs completed | Training policy failed in ModelOpt EAGLE decoder `TransformerLayer.input_layernorm`, which was still TransformerEngine RMSNorm |

Stack summary:

```text
nemo_rl/models/megatron/train.py
  -> draft_model(...)
  -> nemo_rl/models/megatron/draft/eagle.py
  -> modelopt/torch/speculative/plugins/megatron_eagle.py EagleModule.forward
  -> EagleTransformerBlock
  -> TransformerLayer._forward_attention
  -> self.input_layernorm(hidden_states)
  -> transformer_engine RMSNorm
  -> rmsnorm_fwd_cuda_kernel.cu:49 CUDA invalid argument
```

This narrows the remaining online-training blocker: replacing top-level
`EagleModule.enorm/hnorm` was not enough because ModelOpt builds the EAGLE
decoder from `megatron.core.post_training.modelopt.gpt.model_specs.Norm`, whose
`Norm` wrapper instantiates TransformerEngine RMSNorm.

r79 `3139283` is final: `FAILED`, elapsed `00:28:40`, ExitCode `1:0` on
`nvl72082-T08`. The new remote patch kept the same env guard but recursively
replaced TransformerEngine `LayerNorm` and `RMSNorm` child modules inside the
EAGLE module with torch RMSNorm, preserving weights and logging the replaced
module paths. Remote `py_compile` passed, and the r79 submitter passed local and
remote `bash -n`.

The r79 marker has now fired on policy ranks:

```text
NRL_FORCE_DRAFT_TORCH_RMSNORM=true: replaced EagleModule TE norms with torch RMSNorm:
enorm,decoder.layers.0.input_layernorm,decoder.layers.0.pre_mlp_layernorm,decoder.final_layernorm
```

Generation also completed with recovered vLLM internal SpecDec acceptance:
66.6% Avg Draft acceptance, Mean acceptance length 1.67, Accepted 102861 /
Drafted 154549 on one worker; the other workers repeated 57.6% Avg Draft
acceptance, Accepted 89294 / Drafted 155144. Policy logprobs completed, so the
r78 EAGLE decoder TransformerEngine RMSNorm CUDA invalid-argument did not recur.

The new r79 failure is later, during `Training policy` gradient finalization:

```text
MegatronPolicyWorker.train
  -> nemo_rl/models/megatron/train.py megatron_forward_backward
  -> megatron/core/pipeline_parallel/schedules.py finalize_model_grads_func
  -> megatron/core/distributed/distributed_data_parallel.py finish_grad_sync
  -> megatron/core/distributed/param_and_grad_buffer.py finish_grad_sync
  -> AssertionError: Communication call has not been issued for this bucket
     (0/3 params have grad available)
```

This points to Megatron DDP async overlap, not vLLM generation or RMSNorm. In
Megatron's `BucketGroup.finish_grad_sync()`, the assertion is only on the
`overlap_grad_reduce=True` path; the `False` path calls `start_grad_sync()`
synchronously. r80 `3139703` was submitted at 2026-06-03 20:34 PDT with the same
r79 settings plus:

```text
policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=false
policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
```

r80 was the causal test for whether the online draft training blocker was DDP
async overlap handling of unused/no-grad EAGLE draft parameter buckets. It did
avoid the r79 `Communication call has not been issued` assertion, so that
specific blocker was DDP-overlap related. The next blocker was a
model-parallel helper signature mismatch in optimizer/update-success reduction.
r81/r82 then exposed launcher/Ray setup issues, and r83 fixed Ray/vLLM
initialization through CUDA Graph capture before being cancelled prior to a
rollout metric.

## 2026-06-03 17:55 PDT r77/r78 Local-Spec Reconvert Update

Qwen3-8B online r77 `3137513` is final:

| Job | Mode | Status | Evidence | Failure |
|---:|---|---|---|---|
| `3137513` | online K=1 always, short512, local spec + HF reconvert + draft torch RMSNorm | FAILED, 00:17:41, ExitCode 1:0 | `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true: using local GPT decoder block spec during HF->Megatron import` printed on policy ranks | Dense `Qwen3Bridge` had no local layernorm mappings; conversion warned for `decoder.layers.*.input_layernorm.weight` and `decoder.layers.*.pre_mlp_layernorm.weight`, then failed with `AttributeError: 'NoneType' object has no attribute 'megatron_module'` |

Code-level root cause:

- Dense `qwen3_bridge.py` mapped the TE fused layernorm names:
  `decoder.layers.*.self_attention.linear_qkv.layer_norm_weight` and
  `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`.
- Local Megatron layer spec uses separate layernorm params:
  `decoder.layers.*.input_layernorm.weight` and
  `decoder.layers.*.pre_mlp_layernorm.weight`.
- `qwen3_moe_bridge.py` and `llama_bridge.py` already include the local mapping
  form, but dense `qwen3_bridge.py` did not.

Patch applied on the remote online worktree:

| Added Megatron param | HF param |
|---|---|
| `decoder.layers.*.input_layernorm.weight` | `model.layers.*.input_layernorm.weight` |
| `decoder.layers.*.pre_mlp_layernorm.weight` | `model.layers.*.post_attention_layernorm.weight` |

Remote `py_compile` passed. Follow-up r78 `3137739` used the same local-spec
reconvert / draft torch RMSNorm configuration and a fresh actor venv suffix. It
is now superseded by the 19:59 PDT update above: the mapping fix worked, but
training exposed the remaining EAGLE decoder TE RMSNorm blocker.

## 2026-06-03 16:25 PDT r73 Online Short512 Failure

Qwen3-8B online short512 K=1 always smoke `3136275` is now final:

| Job | Mode | Status | Config evidence | Generation evidence | Failure |
|---:|---|---|---|---|---|
| `3136275` | online K=1 always, short512, 1 step | FAILED, 00:23:16, ExitCode 1:0 | `policy.draft.enabled=true`, draft `RedHatAI/Qwen3-8B-speculator.eagle3`, BF16 generation, KV `auto`, `max_new_tokens=512`, `max_total_sequence_length=2560` | vLLM internal SpecDec metrics emitted before failure: one worker acceptance 57.6%, repeated workers 66.6% | policy-side Megatron forward failed in TransformerEngine RMSNorm CUDA invalid-argument |

Failure stack summary:

```text
MegatronPolicyWorker.get_logprobs
  -> megatron_forward_backward
  -> GPTModel decoder
  -> TransformerLayer self_attention
  -> self.linear_qkv(hidden_states)
  -> megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear
  -> transformer_engine.pytorch.module.layernorm_linear.apply_normalization
  -> rmsnorm_fwd_cuda_kernel.cu:49 CUDA Error: invalid argument
```

Code evidence from the online worktree:

```text
Megatron-Bridge qwen_provider.py:
transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
```

The current online environment has TransformerEngine available, so the Qwen
provider chooses the TE fused layer spec. The next smoke should force the Qwen
provider to use the local Megatron GPT decoder block spec, guarded by an env var,
to test whether disabling the fused TE layernorm-linear path lets online
drafter training pass policy logprob forward.

Applied remote guard patch and submitted r74:

| Job | Change vs r73 | Status |
|---:|---|---|
| `3136718` | same short512 K=1 always online smoke, plus `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true`; remote `setup.py` forces `get_gpt_decoder_block_spec(..., use_transformer_engine=False)` only under that env | FAILED after generation. Original TE RMSNorm failure did not recur; new failure is missing local fused-softmax extension `scaled_masked_softmax_cuda`. |
| `3137127` | same as r74, plus guard now also sets `model_cfg.masked_softmax_fusion=False` to avoid the missing local fused-softmax extension | FAILED, 00:23:38. `get_logprobs()` completed, but training failed in online EAGLE draft forward at `EagleModule.enorm(embeddings)` -> TransformerEngine RMSNorm CUDA invalid-argument. vLLM acceptance was 0.0%, likely because the run reused a TE-converted checkpoint while forcing local spec at runtime, producing target layernorm mapping misses. |
| `3137466` | r76: local spec is also passed into HF->Megatron import, checkpoint is reconverted into a separate local-spec cache, and `EagleModule.enorm/hnorm` are replaced with torch RMSNorm under `NRL_FORCE_DRAFT_TORCH_RMSNORM=true` | FAILED before Ray driver startup, 00:00:35. Slurm ray-head `srun` on `nvl72083-T05` failed with `Memory required by task is not available`; no model/code metric. |
| `3137513` | r77: same code/config as r76, same separate local-spec checkpoint dir, excluding `nvl72083-T05` | Submitted. This is the active test for local-spec checkpoint alignment plus draft torch RMSNorm. |

## 2026-06-03 13:06 PDT Latest Online and Diagnostic Status

The r67 Qwen3-8B online retry added failure/status evidence, but no new rollout
performance metric. The r68 gated scrub smoke later failed before rollout during
actor-venv build with `Disk quota exceeded`, so the scheduler-output scrub patch
is still unvalidated.

| Job | Mode | Status at 13:06 PDT | Notes |
|---:|---|---|---|
| `3133619` | K=1 gated V9 smoke, 1 step | failed, 00:17:53 | reached rollout generation, then failed in vLLM V1 `_bookkeeping_sync` with `assert sampled_token_ids.shape[-1] == 1`; V9 did not fully resolve the scheduler/runner bookkeeping mismatch |
| `3133620` | K=1 always online, 20 steps | failed, 00:21:12 | reached rollout generation, then failed in policy-side Megatron/TransformerEngine forward with `rmsnorm_fwd_cuda_kernel.cu:49 ... CUDA Error: invalid argument`; no rollout metric |
| `3133622` | K=2 always online, 20 steps | failed, 00:18:40 | reached rollout generation, then failed in policy-side Megatron/TransformerEngine forward with `rmsnorm_fwd_cuda_kernel.cu:49 ... CUDA Error: invalid argument`; no rollout metric |
| `3133624` | K=3 always online, 20 steps | failed, 00:24:26 | same policy-side Megatron/TransformerEngine RMSNorm CUDA invalid-argument; no rollout metric |
| `3133621/3133623/3133625` | K=1/2/3 gated V9, 20 steps | cancelled | cancelled because the smoke failed |
| `3133925` | K=1 gated r68 scrub smoke, 1 step | failed, 00:11:02 | actor venv build hit `Disk quota exceeded` before rollout; no driver metric |

The K=1 gated failure dump confirms that `scheduled_spec_decode_tokens` is still
present and many requests have `num_scheduled_tokens=2`, while vLLM
`_bookkeeping_sync` enters the non-spec assertion path. That is the immediate
debug target for gated online Qwen3-8B. The K=2 always failure is separate: it
is policy/Megatron-side TransformerEngine RMSNorm, not the vLLM generation
bookkeeping assertion. The same policy-side RMSNorm failure now reproduces for
K=1/K=2/K=3 always-on online runs.

The r68 patch targets the exact r67 inconsistency by scrubbing scheduler output
before `SchedulerOutput(...)`: when the active-batch gate disables speculation,
it clears `scheduled_spec_decode_tokens` and clamps per-request
`num_scheduled_tokens` back to 1.

The Qwen3-8B logprob-skip diagnostic now has Step 1-4 generation and E2E
metrics. Step 1-4 mean speedup is 1.290x generation / 1.152x E2E for K=1
always and 1.343x generation / 1.175x E2E for K=3 always.

## 2026-06-03 12:44 PDT Qwen3-8B Online V9 Retry and Step 1-3 Diagnostic

Qwen3-8B online V6 smoke `3133314` failed with the same vLLM V1 bookkeeping assertion as r61:

```text
assert sampled_token_ids.shape[-1] == 1
```

V6 only aligned scheduler lookahead gating with active scheduler pressure. It did not fully align the runner-side disabled path. V9 adds two changes to the runtime patch:

- runner-side batch pressure uses `max(len(scheduler_output.num_scheduled_tokens), len(self.input_batch.req_ids))`, so active batch size is included when deciding whether the gate disables SpecDec.
- when the gate disables SpecDec, the runner copies zero draft tokens via the normal draft-token buffer path instead of forcing an inconsistent non-spec bookkeeping path.

Patched files:

- Local staging: `.tmp_online_gate/specdec_runtime_gate_patch.py`
- Remote online worktree: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-origin-main-online-20260603/nemo_rl/models/generation/vllm/specdec_runtime_gate_patch.py`

Validation: local and remote `py_compile` passed; remote grep confirms `NRL_SPECDEC_BATCH_GATE_PATCH_V9`, zero-draft disabled path, and active-batch pressure. New Qwen3-8B online jobs at 12:44 PDT:

| Job | Mode | Status at 12:44 PDT | Notes |
|---:|---|---|---|
| `3133619` | K=1 gated V9 smoke, 1 step | running, ~10m | smoke for the V9 active-batch + zero-draft disabled path; no rollout metric yet |
| `3133621` | K=1 gated V9, 20 steps | pending dependency | waits for `afterok:3133619` |
| `3133623` | K=2 gated V9, 20 steps | pending dependency | waits for `afterok:3133619` |
| `3133625` | K=3 gated V9, 20 steps | pending dependency | waits for `afterok:3133619` |
| `3133620` | K=1 always online, 20 steps | running, ~10m | online drafter training with runtime gate disabled; no rollout metric yet |
| `3133622` | K=2 always online, 20 steps | running, ~10m | same for K=2; no rollout metric yet |
| `3133624` | K=3 always online, 20 steps | running, ~10m | same for K=3; no rollout metric yet |

Qwen3-8B logprob-skip r5/r6 diagnostic has now emitted Step 1-4 generation metrics. This is a changed-objective diagnostic, not a default GRPO result: `force_on_policy_ratio=true`, `reference_policy_kl_penalty=0`, `NRL_GRPO_SKIP_POLICY_LOGPROBS_IF_SAFE=true`, BF16 generation, and KV cache `auto`.

| Job | Mode | Step | Generated tok/s/GPU | Speedup vs diagnostic baseline | Acceptance | Per-position acceptance |
|---:|---|---:|---:|---:|---:|---|
| `3132444` | baseline r5 | 1 | 4319.01 | 1.000x | n/a | n/a |
| `3132444` | baseline r5 | 2 | 4168.85 | 1.000x | n/a | n/a |
| `3132444` | baseline r5 | 3 early | 4362.34 | 1.000x | n/a | n/a |
| `3132444` | baseline r5 | 4 derived | 4192.84 | 1.000x | n/a | n/a |
| `3132444` | baseline r5 | mean 1-4 | 4260.76 | 1.000x | n/a | n/a |
| `3132635` | K=1 always r6 | 1 | 5674.47 | 1.314x | 63.93% | 63.93% |
| `3132635` | K=1 always r6 | 2 | 5498.59 | 1.319x | 59.22% | 59.22% |
| `3132635` | K=1 always r6 | 3 early | 5530.84 | 1.268x | 58.43% | 58.43% |
| `3132635` | K=1 always r6 | 4 | 5289.47 | 1.262x | 60.68% | 60.68% |
| `3132635` | K=1 always r6 | mean 1-4 | 5498.34 | 1.290x | 60.57% | n/a |
| `3132636` | K=3 always r6 | 1 | 5801.55 | 1.343x | 38.72% | 59.07%, 36.32%, 20.76% |
| `3132636` | K=3 always r6 | 2 | 5918.88 | 1.420x | 34.54% | 54.31%, 31.82%, 17.50% |
| `3132636` | K=3 always r6 | 3 early | 5705.44 | 1.308x | 34.06% | 53.09%, 31.38%, 17.70% |
| `3132636` | K=3 always r6 | 4 | 5462.06 | 1.303x | 35.67% | 55.59%, 33.05%, 18.37% |
| `3132636` | K=3 always r6 | mean 1-4 | 5721.98 | 1.343x | 35.75% | n/a |

Interpretation: after reducing policy/reference logprob bottleneck, Qwen3-8B does show NeMo-RL generation-only speedup. K=3 remains more sensitive because later-position acceptance stays low; Step 4 per-position acceptance is `55.59%, 33.05%, 18.37%`.

## 2026-06-03 12:25 PDT Qwen3-8B Online V6 Retry

Applied V6 of the online runtime SpecDec gate patch to the remote online worktree:

- Remote file: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-origin-main-online-20260603/nemo_rl/models/generation/vllm/specdec_runtime_gate_patch.py`
- Local staging file: `.tmp_online_gate/specdec_runtime_gate_patch.py`

The V6 change fixes the mismatch that caused Qwen3-8B online r61 `3131454` to fail in vLLM V1 bookkeeping. In r61, scheduler lookahead gating only used `len(num_scheduled_tokens)+1`, so a large active running batch could still leave lookahead/specdecode metadata on the first few scheduled requests. The runner-side batch gate then disabled drafter proposal for the whole large batch. That left scheduler metadata saying "speculative decode" while runner bookkeeping followed the non-spec path and asserted `sampled_token_ids.shape[-1] == 1`.

V6 changes the scheduler gate decision to use:

```text
max(len(self.running), len(num_scheduled_tokens) + 1)
```

This makes the scheduler and runner agree for large running batches. Remote `py_compile` passed and remote grep confirms `NRL_SPECDEC_BATCH_GATE_PATCH_V6`, `NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V6`, and the `max(...)` lookahead checks.

Submitted Qwen3-8B online V6 retry with a fresh actor venv suffix `qwen8_r66_gatev6`. As of 12:25 PDT the smoke has failed and did not emit rollout performance metrics:

| Job | Mode | Status at 12:25 PDT | Notes |
|---:|---|---|---|
| `3133314` | K=1 gated smoke, 1 step | failed, 00:13:41 | reached rollout generation, then failed before `SpecDec early` metric with the same vLLM V1 `assert sampled_token_ids.shape[-1] == 1` bookkeeping assertion |
| `3133315` | K=1 always, 20 steps | cancelled dependency | cancelled after `3133314` failed |
| `3133316` | K=1 gated, 20 steps | cancelled dependency | cancelled after `3133314` failed |
| `3133317` | K=2 always, 20 steps | cancelled dependency | cancelled after `3133314` failed |
| `3133318` | K=2 gated, 20 steps | cancelled dependency | cancelled after `3133314` failed |
| `3133319` | K=3 always, 20 steps | cancelled dependency | cancelled after `3133314` failed |
| `3133320` | K=3 gated, 20 steps | cancelled dependency | cancelled after `3133314` failed |

Conclusion: V6 fixed one plausible scheduler/runner gate mismatch, but it is not sufficient. The online Qwen3-8B failure now points to a deeper vLLM V1 SpecDec bookkeeping path where sampled-token output shape still carries speculative width while the NeMo-RL/vLLM worker expects the non-spec shape.

## 2026-06-03 12:18 PDT New Diagnostic Result

This section is superseded by the 12:44 PDT Step 1-3 table above. At this point Qwen3-8B logprob-skip r5/r6 diagnostic had emitted Step 1-2 rollout generation metrics. This is a changed-objective diagnostic, not a default GRPO result: `force_on_policy_ratio=true`, `reference_policy_kl_penalty=0`, `NRL_GRPO_SKIP_POLICY_LOGPROBS_IF_SAFE=true`, BF16 generation, and KV cache `auto`.

| Job | Mode | Step | Generated tok/s/GPU | Speedup vs diagnostic baseline | Acceptance | Per-position acceptance |
|---:|---|---:|---:|---:|---:|---|
| `3132444` | baseline r5 | 1 | 4319.01 | 1.000x | n/a | n/a |
| `3132444` | baseline r5 | 2 | 4168.85 | 1.000x | n/a | n/a |
| `3132444` | baseline r5 | mean 1-2 | 4243.93 | 1.000x | n/a | n/a |
| `3132635` | K=1 always r6 | 1 | 5674.47 | 1.314x | 63.93% | 63.93% |
| `3132635` | K=1 always r6 | 2 | 5498.59 | 1.319x | 59.22% | 59.22% |
| `3132635` | K=1 always r6 | mean 1-2 | 5586.53 | 1.316x | 61.58% | n/a |
| `3132636` | K=3 always r6 | 1 | 5801.55 | 1.343x | 38.72% | 59.07%, 36.32%, 20.76% |
| `3132636` | K=3 always r6 | 2 | 5918.88 | 1.420x | 34.54% | 54.31%, 31.82%, 17.50% |
| `3132636` | K=3 always r6 | mean 1-2 | 5860.22 | 1.381x | 36.63% | n/a |

Interpretation: when the policy/reference logprob bottleneck is intentionally reduced, Qwen3-8B does show generation-only speedup in NeMo-RL. This supports the current root-cause split: standalone-vs-NeMo gap is not caused by BF16/FP8 or KV quantization; it is primarily a mix of gated runtime behavior, workload/domain acceptance, and non-generation GRPO work. These are Step 1-2 generation-only metrics, so the next check is Step 3+ stability and E2E step time.

Qwen3-32B online r64 prep `3132637` also reached rollout generation setup but failed before a `SpecDec early` metric. The failure is now isolated to policy-to-vLLM draft-weight streaming: vLLM worker `VllmInternalWorkerExtension.update_weights_via_ipc_zmq` looked up `state_dict_info['draft.fc.weight']` and raised `KeyError: 'draft.fc.weight'`; the policy worker then hit a ZMQ timeout. Dependent matrix jobs `3132638/3132639/3132641/3132642/3132643/3132644` were cancelled.

## 2026-06-03 11:47 PDT Retry Status

This section is the previous poll and is superseded by the 12:44 PDT diagnostic
metrics above for `3132444/3132635/3132636`. The useful progress at that time
was failure isolation, retry setup, and confirming that the Qwen3-8B r5/r6
diagnostic jobs reached rollout generation setup with the scheduler fix
applied. Two online-drafter failures were isolated: Qwen3-8B fails inside vLLM
V1 SpecDec bookkeeping during generation, and Qwen3-30B-A3B fails during MoE
policy-to-vLLM weight streaming.

| Job | Model / mode | Status at 11:47 PDT | Result status |
|---:|---|---|---|
| `3131454` | Qwen3-8B online r61 K=1 gated smoke | failed, 01:11:02 | not a performance result; config confirms `policy.draft.enabled=true`, BF16 generation, KV auto, and scheduler override. It reached rollout generation, then vLLM V1 failed in `gpu_model_runner.py::_bookkeeping_sync` with `assert sampled_token_ids.shape[-1] == 1` |
| `3130929` | Qwen3-30B-A3B online K=1 gated prep | failed, 01:30:31 | not a performance result; failed before rollout during policy-to-vLLM weight streaming because vLLM Qwen3-MoE fused-MoE loader rejected `shard_dim=0` for a 3D expert tensor, then policy worker hit ZMQ timeout |
| `3131904` | Qwen3-32B online r62 K=1 gated prep | failed, 00:08:41 | old `_ParamAndGradBuffer.offload_to_cpu` failure is gone; next failure was cached actor venv missing `tensordict` |
| `3132468` | Qwen3-32B online r63 K=1 gated prep | failed, 00:00:47 | Ray head failed at Slurm step creation with `Memory required by task is not available`; also revealed online submit wrapper did not pass actor-venv rebuild/suffix env vars into `COMMAND` |
| `3132637` | Qwen3-32B online r64 K=1 gated prep | failed, 00:32:17 | reached rollout generation setup, then failed before `SpecDec early` metric during policy-to-vLLM weight streaming with `KeyError: 'draft.fc.weight'`; policy worker then hit ZMQ timeout |
| `3132638-3132644` | Qwen3-32B online r64 matrix | cancelled | dependency cancelled after `3132637` failed |
| `3132444` | Qwen3-8B logprob-skip baseline r5 | running | Step 1-2 metrics emitted after this poll: 4319.01 / 4168.85 generated tok/s/GPU |
| `3132445-3132446` | Qwen3-8B logprob-skip K=1/K=3 always r5 | failed, 00:00:31 | Slurm Ray-head step memory failure, not a model/code result |
| `3132635` | Qwen3-8B logprob-skip K=1 always r6 | running | Step 1-2 metrics emitted after this poll: 5674.47 / 5498.59 generated tok/s/GPU, 63.93% / 59.22% acceptance |
| `3132636` | Qwen3-8B logprob-skip K=3 always r6 | running | Step 1-2 metrics emitted after this poll: 5801.55 / 5918.88 generated tok/s/GPU, 38.72% / 34.54% acceptance, Step 2 per-position 54.31% / 31.82% / 17.50% |

Patches applied and validated:

| File | Change | Validation |
|---|---|---|
| online `nemo_rl/distributed/worker_groups.py` | actor local venv cache names now include `UV_PYTHON` and optional `NRL_ACTOR_VENV_CACHE_SUFFIX` | local `py_compile`, remote `py_compile`, remote grep |
| online `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` | passes `NRL_FORCE_REBUILD_ACTOR_VENVS` and `NRL_ACTOR_VENV_CACHE_SUFFIX` into driver `COMMAND` | local/remote `bash -n`, remote grep |
| `experiments/eagle3_online/submit_qwen32_r64tensordict_lowcpu.sh` | Qwen3-32B r64 low-CPU retry with actor venv rebuild/suffix | local/remote `bash -n`, submitted |
| `experiments/eagle3_qwen3_8b/submit_qwen3_8b_logskip_diag_r6_spec_schedfix_lowcpu.sh` | Qwen3-8B K=1/K=3 logprob-skip retry with scheduler fix and lower CPU request | local/remote `bash -n`, submitted |

## 2026-06-03 11:12 PDT Status Refresh

No new online rollout performance metrics are available yet.

| Job | Model / mode | Status at 11:12 PDT | Result status |
|---:|---|---|---|
| `3130929` | Qwen3-30B-A3B online K=1 gated prep | running, ~1h05m | `MasterConfig` confirms BF16 generation, KV auto, and `policy.draft.enabled=true`; no rollout metric |
| `3131454` | Qwen3-8B online r61 K=1 gated smoke | running, ~35m | no rollout metric |
| `3131904` | Qwen3-32B online r62 K=1 gated prep | running, ~6m | r62 offload-buffer retry reached `MasterConfig`; BF16 generation, KV auto, and online draft are confirmed; no rollout metric |
| `3131906-3131911` | Qwen3-32B online K=1/2/3 always/gated matrix | pending | waiting on `afterok:3131904` |

The Qwen3-8B logprob-skip diagnostic did not produce usable performance
numbers. The r2 batch `3131108/3131109/3131110` failed before rollout, and r4
K=1/K=3 `3131649/3131650` also failed before rollout. The common failure is
Megatron's optimizer scheduler assertion:

```text
assert self.lr_warmup_steps < self.lr_decay_steps
```

`3131648` still appears as running in Slurm, but its driver log already shows
the same scheduler assertion and no rollout metric. Treat the logprob-skip
diagnostic as needing scheduler overrides before it can be used as evidence.

## 2026-06-03 11:01 PDT Qwen3-32B Online Offload Patch and Resubmission

The Qwen3-32B online prep failure in `3130930` was caused by a Megatron API
compatibility mismatch: NeMo-RL's `MegatronPolicyWorker.move_model` assumed
DDP param/grad buffers implement `offload_to_cpu` and `reload_from_cpu`, but the
current Megatron `_ParamAndGradBuffer` stores `param_data`, `grad_data`, and
bucket views directly without those helper methods.

Patch applied:

| File | Change | Validation |
|---|---|---|
| `nemo_rl/models/policy/workers/megatron_policy_worker.py` in online worktree | Use existing `offload_to_cpu` / `reload_from_cpu` when present; otherwise move `param_data` / `grad_data` tensors directly, remap bucket views and `param.data` / `param.main_grad`, and clear cached bucket-group shard views | local `py_compile`, remote `py_compile`, remote grep |

Remote patched file:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-origin-main-online-20260603/nemo_rl/models/policy/workers/megatron_policy_worker.py`

New Qwen3-32B online-drafter submission:

| Job | Mode | Status at 11:01 PDT | Notes |
|---:|---|---|---|
| `3131904` | K=1 gated prep/smoke | running | r62 offload-buffer compatibility retry; reuses existing r60 Megatron checkpoint |
| `3131906` | K=1 always | pending | `afterok:3131904` |
| `3131907` | K=1 gated | pending | `afterok:3131904` |
| `3131908` | K=2 always | pending | `afterok:3131904` |
| `3131909` | K=2 gated | pending | `afterok:3131904` |
| `3131910` | K=3 always | pending | `afterok:3131904` |
| `3131911` | K=3 gated | pending | `afterok:3131904` |

No new online rollout metrics are available yet. The important validation point
for `3131904` is whether it passes the previous `move_model(..., "cpu")` failure
site and reaches rollout.

## 2026-06-03 10:48 PDT Team-Share Refresh

Latest Slurm poll:

| Job | Model / mode | Status at 10:48 PDT | Evidence |
|---:|---|---|---|
| `3127152` | Qwen3-8B K=1 always fixed/offline | timeout | 4h walltime expired; final observed metric is 9 steps, mean generation `5547.07` tok/s/GPU, acceptance `60.84%` |
| `3127155` | Qwen3-8B K=1 gated fixed/offline | timeout | 8 observed steps, mean generation `4027.48`, acceptance `51.74%`, scheduler gate `0.02%` |
| `3127158` | Qwen3-8B K=3 always fixed/offline | timeout | 10 observed steps, mean generation `5915.66`, acceptance `36.04%` |
| `3127161` | Qwen3-8B K=3 gated fixed/offline | timeout | 8 observed steps, mean generation `4038.35`, acceptance `28.17%`, scheduler gate `0.02%` |
| `3130929` | Qwen3-30B-A3B online prep | running | `policy.draft.enabled=true`, BF16 generation, KV auto, K=1 gated; no rollout metric |
| `3130930` | Qwen3-32B online prep | failed | reached `MasterConfig` and worker init, then `MegatronPolicyWorker.move_model(..., "cpu")` failed because Megatron `_ParamAndGradBuffer` lacks `offload_to_cpu` |
| `3131454` | Qwen3-8B online r61 smoke | running | scheduler override replacement for r60; no rollout metric |
| `3131648` | Qwen3-8B logprob-skip baseline r4 | running | corrected reuse-venv launcher; no rollout metric |
| `3131649` | Qwen3-8B logprob-skip K=1 always r4 | running | corrected reuse-venv launcher; no rollout metric |
| `3131650` | Qwen3-8B logprob-skip K=3 always r4 | running | corrected reuse-venv launcher; no rollout metric |

The r3 reuse-venv diagnostic jobs `3131615/3131616/3131617` were canceled and
should not be used as evidence because they called stale root launchers. The
r4 replacement batch `3131648/3131649/3131650` uses the patched launchers under
`experiments/eagle3_qwen3_8b/`.

The Qwen3-32B online failure is not a performance result. It is an online
drafter initialization compatibility issue in the Megatron policy worker CPU
offload path. The next code fix should inspect and handle Megatron
`_ParamAndGradBuffer` objects that do not provide `offload_to_cpu` /
`reload_from_cpu`.

Team-share summary document:

- `docs/qwen3_specdec_team_update_2026_06_03.md`

## 2026-06-03 10:40 PDT Qwen3-32B/30B Commonality and Reuse-Venv Diagnostic

The Qwen3-32B and Qwen3-30B-A3B cases are the same at the online
draft-builder/root-cause level, but not numerically identical performance cases.
All three models use the common `policy.draft.enabled=true` online draft path,
so the `rope_scaling`/provider-attribute compatibility patch applies to 8B,
32B, and 30B-A3B. The fixed/offline results also show the same gate pattern:
gated SpecDec rarely enables lookahead, while always-on is the throughput-
relevant path. Acceptance and throughput still differ by model.

Current online status:

| Job | Model | Status at 10:40 PDT | Evidence |
|---:|---|---|---|
| `3131454` | Qwen3-8B r61 smoke | running | still in driver venv/setup; no rollout metric |
| `3130929` | Qwen3-30B-A3B prep | running | `MasterConfig` confirms `policy.draft.enabled=true`, BF16, KV auto, K=1; no rollout metric |
| `3130930` | Qwen3-32B prep | running | `MasterConfig` confirms `policy.draft.enabled=true`, BF16, KV auto, K=1; vLLM workers initialized `16/16`; no rollout metric |

The first Qwen3-8B logprob-skip diagnostic batch `3131108/3131109/3131110`
has the intended settings verified in Slurm stdout, but is still spending time
building fresh driver environments (`mamba-ssm`/`transformer-engine-torch`).
I left it running and submitted a reuse-venv duplicate batch:

| Job | Mode | Status at 10:40 PDT | Venv strategy |
|---:|---|---|---|
| `3131615` | baseline | running | reuse `.driver_venvs/qwen3_8b_baseline_py312`, `NRL_FORCE_REBUILD_VENVS=false` |
| `3131616` | K=1 always-on | running | reuse `.driver_venvs/qwen3-8b-k1-always-r30_20260603`, runtime gate disabled |
| `3131617` | K=3 always-on | running | reuse `.driver_venvs/qwen3-8b-k3-always-r30_20260603`, runtime gate disabled |

These r2/r3 diagnostic jobs are still changed-objective diagnostics, not
default GRPO results: BF16, KV auto, generation request logprobs omitted,
`force_on_policy_ratio=true`, `reference_policy_kl_penalty=0`, and
`NRL_GRPO_SKIP_POLICY_LOGPROBS_IF_SAFE=true`.

## 2026-06-03 10:25 PDT Logprob Bottleneck Patch

The fixed/offline GRPO loop now has a changed-objective diagnostic path that
removes avoidable post-rollout logprob work:

| Patch | Behavior | Default GRPO impact |
|---|---|---|
| `reference_policy_kl_penalty=0` skip | Skip `policy.get_reference_policy_logprobs(...)` and supply zero placeholders because the loss does not consume reference logprobs when KL penalty is zero | No change for default runs with KL penalty `0.01` |
| `NRL_GRPO_SKIP_POLICY_LOGPROBS_IF_SAFE=true` | If `force_on_policy_ratio=true`, KL penalty is zero, and importance-sampling paths are disabled, skip `policy.get_logprobs(...)` prev-logprob fprop and supply zero placeholders | No change unless the explicit diagnostic env is enabled |

Validation: local `py_compile` passed; remote fixed/offline `py_compile` passed.
The Qwen3-8B launchers now propagate
`NRL_GRPO_SKIP_POLICY_LOGPROBS_IF_SAFE` into the driver command; local and
remote `bash -n` passed.

Submitted Qwen3-8B changed-objective diagnostic jobs:

| Job | Mode | Status at 10:26 PDT | Settings |
|---:|---|---|---|
| `3131108` | baseline | running; driver log not created yet | BF16, KV auto, 64x32, 4 steps, generation logprobs omitted, `force_on_policy_ratio=true`, KL penalty `0`, policy/reference logprob skip |
| `3131109` | K=1 always-on | running; driver log not created yet | same plus Eagle-3 K=1, runtime gate disabled |
| `3131110` | K=3 always-on | running; driver log not created yet | same plus Eagle-3 K=3, runtime gate disabled |

This is not a default GRPO result. It is meant to answer whether NeMo-RL E2E
speedup moves closer to the generation/standalone ceiling after removing the
policy/reference logprob region from the step.

Online Qwen3-8B r60 smoke `3130928` has now printed `MasterConfig` with
`policy.draft.enabled=true`, BF16 generation, and KV cache `auto`. Its recipe is
not shape-identical to the fixed/offline 4096-token OpenMath matrix: it uses
`max_new_tokens=8192`, `max_model_len=8192`, DAPO math verification, and
`use_importance_sampling_correction=true`.

`3130928` then failed before rollout in `lm_policy` initialization:
Megatron's optimizer scheduler asserted `lr_warmup_steps < lr_decay_steps`.
The online 8B recipe had `lr_warmup_iters=10`, `lr_decay_iters=None`, and the
smoke used only `max_num_steps=2`, so the short smoke made the scheduler invalid.
I canceled the old failed-smoke dependents `3130931/3130934/3130937/3130940/
3130943/3130946` and resubmitted r61 with
`policy.megatron_cfg.scheduler.lr_warmup_iters=0` and
`policy.megatron_cfg.scheduler.lr_decay_iters=1000`:

| Job | Model | K | Gate | Status at 10:31 PDT |
|---:|---|---:|---|---|
| `3131454` | Qwen3-8B online | 1 | gated smoke | running; driver log pending |
| `3131455` | Qwen3-8B online | 1 | always | pending, afterok `3131454` |
| `3131456` | Qwen3-8B online | 1 | gated | pending, afterok `3131454` |
| `3131457` | Qwen3-8B online | 2 | always | pending, afterok `3131454` |
| `3131458` | Qwen3-8B online | 2 | gated | pending, afterok `3131454` |
| `3131459` | Qwen3-8B online | 3 | always | pending, afterok `3131454` |
| `3131460` | Qwen3-8B online | 3 | gated | pending, afterok `3131454` |

## 2026-06-03 10:15 PDT Online Drafter Status

The Qwen3-8B online drafter-training failure mode is not unique to 8B. It is
the common NeMo-RL draft-builder path used when `policy.draft.enabled=true`.
The latest failures are config/provider compatibility bugs in
`nemo_rl/models/megatron/draft/utils.py`, not KV-cache quantization and not an
FP8 generation issue.

| Model | Failed job | Failure | Current replacement |
|---|---:|---|---|
| Qwen3-8B | `3130576` | `rope_scaling` was `null`, but the builder called `.get()` on it | smoke `3130928`, then K=1/2/3 always/gated `3130931/3130934/3130937/3130940/3130943/3130946` |
| Qwen3-32B | `3130535` | `Qwen3ModelProvider` has no `rope_scaling_factor` attribute | prep `3130930`, then K=1/2/3 always/gated `3130933/3130935/3130938/3130942/3130945/3130948` |
| Qwen3-30B-A3B | `3130615` | `Qwen3MoEModelProvider` has no `rope_scaling_factor` attribute | prep `3130929`, then K=1/2/3 always/gated `3130932/3130936/3130939/3130941/3130944/3130947` |

The common patch now treats `rope_scaling=None` as disabled scaling and uses
`getattr(model_provider, "rope_scaling_factor", None)` and
`getattr(model_provider, "rope_scaling", False)` fallbacks. Remote
`python3 -m py_compile` passed after the patch. A second online/main patch now
matches the fixed/offline vLLM behavior for SpecDec generation logprobs:
SpecDec requests omit vLLM request logprobs by default unless explicitly forced
through `NRL_VLLM_SPECDEC_REQUEST_LOGPROBS=true`. Remote `py_compile` passed for
`vllm_worker.py`, `vllm_generation.py`, and `draft/utils.py`.

The r60 prep/smoke jobs `3130928`, `3130929`, and `3130930` are running as of
the 10:15 PDT poll, while their K=1/2/3 always/gated matrix jobs are waiting on
`afterok` dependencies. The Qwen3-30B-A3B and Qwen3-32B prep logs already show
`policy.draft.enabled=true`, BF16 generation, and `kv_cache_dtype=auto`. There
are still no online acceptance-rate, generation-speedup, or E2E-speedup metrics
to plot.

Qwen3-8B also had one infrastructure-only retry failure, `3130760`, before the
driver reached Python: the Ray cluster came up, but the single-node driver
`srun` failed with `Memory required by task is not available`. The r60 8B
script lowers `CPUS_PER_WORKER` to leave room for the driver step.

## Code-Level E2E Root Cause

The current NeMo-RL path is not equivalent to a vLLM standalone throughput
benchmark even when both use the same verifier model, drafter, K, and BF16
generation.

1. The online/main worktree previously built `SamplingParams(logprobs=0)` for
   generation, which asks vLLM to return per-token `generation_logprobs` for
   rollout samples. The fixed/offline worktree already had a SpecDec-specific
   omission patch (`NRL_VLLM_OMIT_GENERATION_LOGPROBS`) because vLLM V1 disables
   SpecDec when request logprobs are requested. The online/main worktree has now
   been patched to match that behavior. Many standalone throughput benchmarks
   also omit request logprobs, so this variable is now aligned for the online
   matrix.
2. `nemo_rl/algorithms/grpo.py` then computes `policy.get_logprobs(...)` and,
   unless explicitly skipped, `policy.get_reference_policy_logprobs(...)` after
   rollout. This policy/reference forward-pass work is outside the generation
   region, so generation tok/s speedup cannot directly become the same E2E
   speedup.
3. `nemo_rl/algorithms/loss/loss_functions.py` uses behavior logprobs in
   PPO/GRPO ratio, KL/error metrics, and importance-weight paths. Therefore the
   default training path must still have behavior logprobs available somewhere;
   the current fixed/offline path repairs them with a post-rollout policy
   forward pass instead of asking vLLM generation to return request logprobs.
   Removing the remaining policy/reference logprob work is only a diagnostic or
   changed-objective optimization unless paired with `force_on_policy_ratio`,
   disabled reference KL, or another explicit loss-level change.

The next useful performance diagnostic is a Qwen3-8B run that keeps SpecDec
always-on but removes as much post-rollout logprob work as the algorithm allows
(`force_on_policy_ratio=true`, reference KL penalty `0`, and reference logprob
skip). That would measure how close NeMo-RL can get when the non-generation
E2E bottleneck is reduced, but it should be reported as a changed-training-
objective diagnostic rather than the default GRPO result.

## 2026-06-03 10:10 PDT Fixed/Offline Follow-Up Status

The fixed/offline matrix has more complete rollout coverage now. The most
important pattern is unchanged: always-on SpecDec is the throughput path, while
the current TLT-style gated path enables speculation for only a tiny fraction of
scheduler decisions.

| Model | K | Gate | Job | Status | Observed steps | Mean gen tok/s/GPU | Mean acceptance | Latest gate enabled |
|---|---:|---|---:|---|---:|---:|---:|---|
| Qwen3-8B | 1 | always | `3127152` | running | 9 | `5547.06` | `60.84%` | n/a |
| Qwen3-8B | 1 | gated | `3127155` | running | 8 | `4027.48` | `51.73%` | scheduler `0.02%`, runner `0.63%` |
| Qwen3-8B | 3 | always | `3127158` | running | 10 | `5915.66` | `36.04%` | n/a |
| Qwen3-8B | 3 | gated | `3127161` | running | 8 | `4038.35` | `28.17%` | scheduler `0.02%`, runner `0.63%` |
| Qwen3-32B | 1 | always | `3128147` | completed | 20 | `2388.46` | `69.45%` | n/a |
| Qwen3-32B | 1 | gated | `3128148` | completed | 20 | `1643.20` | `45.58%` | scheduler `0.26%`, runner `4.10%` |
| Qwen3-32B | 3 | always | `3128428` | completed | 20 | `2405.82` | `45.28%` | n/a |
| Qwen3-32B | 3 | gated | `3128429` | completed | 20 | `1651.10` | `23.65%` | scheduler `0.22%`, runner `3.21%` |
| Qwen3-30B-A3B | 1 | always | `3128151` | completed | 20 | `5601.48` | `57.51%` | n/a |
| Qwen3-30B-A3B | 1 | gated | `3128152` | completed | 20 | `4372.35` | `48.00%` | scheduler `0.00%`, runner `0.02%` |
| Qwen3-30B-A3B | 3 | always | `3128153` | completed | 20 | `4896.03` | `31.80%` | n/a |
| Qwen3-30B-A3B | 3 | gated | `3128154` | completed | 20 | `4357.91` | `23.64%` | scheduler `0.00%`, runner `0.01%` |

The old Qwen3-8B-only online drafter smoke below is superseded by the r60
three-model prep/matrix listed above and is retained only for traceability:

| Job | Model | K | Gate | Status | Current progress |
|---:|---|---:|---|---|---|
| `3129296` | Qwen3-8B | 1 | gated | superseded | Ray workers connected `4/4`; driver started; `vllm_policy` workers initialized `4/4`; superseded before online rollout metrics were available |

The `3129296` effective config confirms `policy.draft.enabled=true`,
`policy.draft.model_name=RedHatAI/Qwen3-8B-speculator.eagle3`,
`policy.draft.loss_weight=1.0`, BF16 generation, KV cache `auto`, and
`policy.sequence_packing.enabled=false`. The previous smoke `3129244` failed at
`ModuleNotFoundError: megatron.core.inference.config`; a compatibility shim was
added under the online Megatron-LM path before submitting `3129296`.

Updated artifacts:

- `docs/specdec_followup_latest_metrics.json`
- `docs/specdec_followup_latest_generation_bars.png`
- `docs/specdec_followup_latest_acceptance_by_model.png`

## 2026-06-03 04:12 PDT Live Follow-Up Status

Qwen3-8B fixed-drafter follow-up jobs are still running and have emitted Step 1
through Step 3 early metrics for always-on jobs:

| Model | K | Gate | Job | Status | Step 1 gen tok/s/GPU / acc | Step 2 gen tok/s/GPU / acc | Step 3 gen tok/s/GPU / acc |
|---|---:|---|---:|---|---:|---:|---:|
| Qwen3-8B | 1 | always | `3127152` | running | `5648.64 / 63.93%` | `5551.14 / 59.33%` | `5432.50 / 58.06%` |
| Qwen3-8B | 1 | gated | `3127155` | running | `4091.79 / 54.20%` | `4021.44 / 49.68%` | n/a |
| Qwen3-8B | 3 | always | `3127158` | running | `5812.01 / 38.65%` | `5983.01 / 34.64%` | `5698.97 / 33.98%` |
| Qwen3-8B | 3 | gated | `3127161` | running | `4095.04 / 30.06%` | `4039.39 / 26.64%` | n/a |

The gated runs are again mostly disabled by the scheduler gate:

| Job | K | Step | Scheduler enabled ratio | Runner enabled ratio |
|---:|---:|---:|---:|---:|
| `3127155` | 1 | 1 | `0.04%` | `1.61%` |
| `3127155` | 1 | 2 | `0.03%` | `0.87%` |
| `3127161` | 3 | 1 | `0.03%` | `1.03%` |
| `3127161` | 3 | 2 | `0.02%` | `1.01%` |

Qwen3-32B and Qwen3-30B-A3B fixed-drafter retry jobs are now past the fs1 quota
failure point. Qwen3-32B K=1 has Step 1/2 metrics; Qwen3-32B K=3 failed in the
4 GPUs/node layout from node-memory OOM and was resubmitted as 8 nodes x 2 GPUs.

| Model | K | Gate | Job | Status | Step metrics / reason |
|---|---:|---|---:|---|---|
| Qwen3-32B | 1 | always | `3128147` | running | S1 `2327.40 / 70.13%`, S2 `2519.32 / 69.13%` |
| Qwen3-32B | 1 | gated | `3128148` | running | S1 `1584.49 / 47.13%`, S2 `1803.00 / 40.56%`; scheduler enabled `0.26-0.28%` |
| Qwen3-32B | 3 | always | `3128149` | failed | node-memory OOM before rollout metrics |
| Qwen3-32B | 3 | gated | `3128150` | failed | S1 `1588.65 / 24.73%`, then node-memory OOM before Step 2 completed |
| Qwen3-32B | 3 | always | `3128428` | running | 8 nodes x 2 GPUs retry of `3128149` |
| Qwen3-32B | 3 | gated | `3128429` | running | 8 nodes x 2 GPUs retry of `3128150` |

Qwen3-30B-A3B with the 500K drafter is running and has Step 1-3 metrics:

| Model | K | Gate | Job | Status | Step 1 gen tok/s/GPU / acc | Step 2 gen tok/s/GPU / acc | Step 3 gen tok/s/GPU / acc |
|---|---:|---|---:|---|---:|---:|---:|
| Qwen3-30B-A3B | 1 | always | `3128151` | running | `5745.88 / 57.91%` | `5311.72 / 57.01%` | `5987.16 / 59.15%` |
| Qwen3-30B-A3B | 1 | gated | `3128152` | running | `4425.95 / 64.88%` | `4209.63 / 46.08%` | `4621.79 / 52.53%` |
| Qwen3-30B-A3B | 3 | always | `3128153` | running | `4918.23 / 32.11%` | `4860.85 / 31.35%` | `5195.76 / 33.13%` |
| Qwen3-30B-A3B | 3 | gated | `3128154` | running | `4434.02 / 38.81%` | `4185.37 / 20.11%` | `4608.09 / 24.80%` |

Online drafter training remains in smoke/debug. The current `sj/nemo-rl-perf`
checkout does not expose `policy.draft` or `draft_loss`, so online testing uses
`SpecDec-RL-origin-main-online-20260603`. After checking PR
`NVIDIA-NeMo/RL#2658`, the sleep path is already aligned with the revert
intent (`sleep(level=1)`, no `discard_weights`). The online failures so far were
dependency/environment issues before rollout:

| Job(s) | Status | Reason |
|---|---|---|
| `3127214-3127225` | failed | launcher used Python 3.12 while `origin/main` requires Python 3.13 |
| `3127272-3127283` | failed | missing third-party workspace members |
| `3127358-3127369` | failed | `uv --locked` rejected lockfile freshness |
| `3127386`, `3127399-3127409` | failed | `causal-conv1d`/`mamba-ssm` build under Python 3.13/CUDA13 |
| `3127663` | failed | system Python path lacked `tensordict` |
| `3127911` | failed | fs1 quota before Ray log directory creation |
| `3127945` | cancelled | stale `--frozen` lock still selected mamba/causal |
| `3128048` | failed | `deep_ep` source build CUDA 12.9 vs PyTorch CUDA 13.0 mismatch |
| `3128181` | cancelled | Git-sourced TE path superseded by wheel-only TE path |
| `3128307` | cancelled | Git-sourced TE fast-build path superseded by wheel-only TE path |
| `3128357` | failed | TE wheel path was correct, but fsw driver venv quota was hit while copying `mlflow/semantic_kernel/autolog.py` |
| `3128421` | running | current Qwen3-8B online K=1 always smoke; TE wheel path, no `TransformerEngine.git`/`transformer-engine-torch`/`Building transformer-engine` log line, no failure yet |

New artifacts:

- `docs/specdec_followup_qwen3_8b_step1.png`
- `docs/specdec_followup_latest_generation_bars.png`
- `docs/specdec_followup_latest_acceptance_by_model.png`

## 2026-06-03 Follow-Up Matrix Submitted

The requested follow-up matrix has been submitted for `Qwen3-8B`,
`Qwen3-32B`, and `Qwen3-30B-A3B`, with `K=1`, `K=3`, always-on SpecDec, and
batch-size-gated SpecDec.

The table below records the initial fixed/offline submission. The Qwen3-32B and
Qwen3-30B-A3B entries in this initial batch later failed from fs1 quota and were
replaced by the retry jobs shown in the live snapshot above.

| Model | K | Gate | Job | Status |
|---|---:|---|---:|---|
| Qwen3-8B | 1 | always | `3127152` | running |
| Qwen3-32B | 1 | always | `3127153` | failed, replaced by `3128147` |
| Qwen3-30B-A3B | 1 | always | `3127154` | failed, replaced by `3128151` |
| Qwen3-8B | 1 | gated | `3127155` | running |
| Qwen3-32B | 1 | gated | `3127156` | failed, replaced by `3128148` |
| Qwen3-30B-A3B | 1 | gated | `3127157` | failed, replaced by `3128152` |
| Qwen3-8B | 3 | always | `3127158` | running |
| Qwen3-32B | 3 | always | `3127159` | failed, replaced by `3128149` |
| Qwen3-30B-A3B | 3 | always | `3127160` | failed, replaced by `3128153` |
| Qwen3-8B | 3 | gated | `3127161` | running |
| Qwen3-32B | 3 | gated | `3127162` | failed, replaced by `3128150` |
| Qwen3-30B-A3B | 3 | gated | `3127163` | failed, replaced by `3128154` |

Online drafter training jobs are submitted from
`SpecDec-RL-origin-main-online-20260603` because the current `sj/nemo-rl-perf`
checkout does not contain `policy.draft` or `draft_loss` support, while
`origin/main` at `ab079b81f` does. These jobs set
`policy.draft.enabled=true`, `policy.draft.model_name=<drafter>`,
`policy.draft.loss_weight=1.0`, and `policy.sequence_packing.enabled=false`.
Gated online jobs use vLLM `speculative_config.disable_by_batch_size=8`.
The first submitted online batch `3127214-3127225` was superseded after it
failed immediately because the launcher forced `UV_PYTHON=3.12.13` while
`origin/main` requires Python `>=3.13.13,<3.14`. The second batch
`3127272-3127283` was superseded because the detached `origin/main` worktree
had uninitialized third-party workspace members, causing uv to reject
`nemo_gym = { workspace = true }`. `3127358-3127369` was superseded because
`uv run --locked` rejected the restored workspace/lockfile freshness state.
The worktree now has the Gym, Automodel, and Megatron-Bridge workspace contents
restored, the launcher defaults to `UV_PYTHON=3.13.13`, and the online path
uses `UV_LOCK_MODE=--frozen`.

| Model | K | Gate | Job | Status |
|---|---:|---|---:|---|
| Qwen3-8B | 1 | always | `3127386` | failed, superseded |
| Qwen3-32B | 1 | always | `3127399` | failed, superseded |
| Qwen3-30B-A3B | 1 | always | `3127400` | failed, superseded |
| Qwen3-8B | 1 | gated | `3127401` | failed, superseded |
| Qwen3-32B | 1 | gated | `3127402` | failed, superseded |
| Qwen3-30B-A3B | 1 | gated | `3127403` | failed, superseded |
| Qwen3-8B | 3 | always | `3127404` | failed, superseded |
| Qwen3-32B | 3 | always | `3127405` | failed, superseded |
| Qwen3-30B-A3B | 3 | always | `3127406` | failed, superseded |
| Qwen3-8B | 3 | gated | `3127407` | failed, superseded |
| Qwen3-32B | 3 | gated | `3127408` | failed, superseded |
| Qwen3-30B-A3B | 3 | gated | `3127409` | failed, superseded |

PR `NVIDIA-NeMo/RL#2658` was also checked. It is not the online drafter
implementation; it reverts vLLM weight discard during sleep. Both the current
patched branch and the `origin/main` online worktree have the relevant
`sleep(level=1)` path and no `discard_weights` plumbing in
`finish_generation()`.

## 2026-06-03 Latest Result

The strongest new result is the vLLM standalone real-prompt isolation. After
patching the benchmark tokenizer fallback, the Qwen3-8B DAPOMath bs32 retry2
completed:

| Setting | Tok/s/GPU | Speedup vs baseline | Acceptance | Per-position acceptance |
|---|---:|---:|---:|---|
| Baseline | `6410.19` | `1.000x` | n/a | n/a |
| K=1 | `4261.02` | `0.665x` | `61.11%` | `61.11%` |
| K=2 | `5238.19` | `0.817x` | `46.91%` | `57.85%, 35.97%` |
| K=3 | `5468.75` | `0.853x` | `36.62%` | `56.07%, 34.56%, 19.25%` |

This changes the root-cause interpretation. The earlier Qwen3-8B standalone
`2.145x` K=3 and `2.346x` K=2 results were synthetic fixed-prompt ceilings, not
real DAPOMath prompt results. On the real prompt distribution, vLLM standalone
itself regresses for every K. Therefore NeMo-RL is not simply "losing a proven
real-prompt standalone speedup"; the public HF drafter is not strong enough on
the current DAPOMath workload at these settings.

The latest NeMo-RL exact-engine diagnostics are:

| Setting | Gen tok/s/GPU | Speedup vs exact baseline | Acceptance | Interpretation |
|---|---:|---:|---:|---|
| exact baseline `3126355` | `5652.52` | `1.000x` | n/a | short512/greedy/no-gate baseline |
| native K=1 `3126356` | `7345.13` | `1.299x` | `64.61%` | best current exact-engine diagnostic |
| native K=3 `3126357` | `2555.29` | `0.452x` | `40.66%` | K=3 overhead dominates |
| dynamic K3 engine, effective K=1 `3126685` | `4446.23` | `0.787x` | `64.60%` | acceptance recovers, but dynamic-cap/K=3-engine overhead is too high |

The current fix direction is therefore:

1. Use native always-on K=1 for Qwen3-8B throughput experiments, not K=3 with a
   dynamic cap pretending to be K=1.
2. Keep TLT-style runtime gating as a research/control experiment, but not as
   the throughput result path. It enables SpecDec for only a tiny long-tail
   fraction of the rollout and cannot reproduce standalone or always-on speed.
3. Treat K=2/K=3 as dependent on either a better in-domain/online-trained
   drafter or a lower-overhead dynamic-K implementation. With the public HF
   drafter and current NeMo-RL/vLLM path, later draft positions are rejected too
   often to pay for the extra work.

Follow-up validation submitted:

- `3127002`: native K=1 always-on BF16/KV-auto 20-step validation.
- It disables the runtime scheduler gate and uses `NUM_SPECULATIVE_TOKENS=1`.
- Purpose: verify that the current best NeMo-RL Qwen3-8B throughput path holds
  beyond the prior 4-step diagnostic window.
- Latest status at 02:05 PDT: running, Ray actors connected `4/4`, driver
  started, no rollout metric yet.

## Drafter Training Config Check

The Qwen3-8B and Qwen3-32B NeMo-RL jobs evaluated here did **not** train the
EAGLE-3 drafter online. The actual launch overrides only attach the public HF
draft checkpoint to vLLM:

- `++policy.generation.vllm_kwargs.speculative_config.method=eagle3`
- `++policy.generation.vllm_kwargs.speculative_config.model=RedHatAI/Qwen3-8B-speculator.eagle3`
- `++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=<K>`
- `++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1`

For Qwen3-32B the same pattern uses
`RedHatAI/Qwen3-32B-speculator.eagle3`. The effective logged config contains
`policy.generation.vllm_kwargs.speculative_config`, but not a `policy.draft`
training section. A targeted grep of the remote NeMo-RL branch did not find a
live `policy.draft.enabled`, `draft_loss`, or online-draft-training path under
`nemo_rl`, `examples`, or `tests`. Therefore the current acceptance results are
fixed-public-drafter rollout results, not online-adapted drafter results.

Code/config evidence:

- `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml` defines
  the Qwen3-32B GRPO performance recipe, but it has no `policy.draft` section
  and no drafter loss/training stanza. SpecDec is injected only through launch
  overrides.
- `nemo_rl/models/generation/__init__.py` treats speculative decoding as a
  generation config feature: when `speculative_config` is present, it switches
  vLLM `load_format` to `auto` so drafter/verifier weights are loaded.
- `nemo_rl/models/generation/vllm/vllm_worker.py` logs the effective
  `method`, `num_speculative_tokens`, `draft_tensor_parallel_size`, and gate
  env vars, then passes the config to vLLM engine creation. This is engine
  setup, not optimizer/training setup for the drafter.
- `nemo_rl/algorithms/grpo.py` only detects `speculative_config` to handle the
  behavior-logprob path for rollout accounting. It does not instantiate or
  optimize a drafter model.

This matters because the public drafter is not being updated as the policy or
rollout distribution changes. The observed DAPOMath real-prompt standalone
regression shows the mismatch exists even before considering GRPO policy drift.

## Current Status

The completed Qwen3-8B vLLM standalone benchmark is a short, deterministic generation-only ceiling:

- Target: `Qwen/Qwen3-8B`
- Drafter: `RedHatAI/Qwen3-8B-speculator.eagle3`
- `dtype=auto`, `kv_cache_dtype=auto`
- ISL/OSL: `1000/512`
- `temperature=0.0`
- `ignore_eos=True`
- fixed `min_tokens=max_tokens=512`
- K=3 speedup: up to `2.145x` at batch size 32
- Engine shape: `max_num_seqs=32`, because the standalone script sets `max_num_seqs=max(batch_sizes)` and this sweep used batch sizes up to `32`
- `gpu_memory_utilization=0.82`
- `max_num_batched_tokens=64000`

The current Qwen3-8B NeMo-RL r5 result is not the same workload:

- Recipe: `examples/configs/recipes/llm/grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron.yaml`
- Effective target/tokenizer from launcher override: `Qwen/Qwen3-8B`
- Dataset: `DAPOMath17K`
- `policy.max_total_sequence_length=8192`
- inherited `policy.generation.max_new_tokens=${policy.max_total_sequence_length}`
- inherited `policy.generation.temperature=1.0`
- real EOS stop tokens and per-sample stop strings are used
- Step 1 generated tokens:
  - baseline r5: `13,490,235` generated tokens / `2048` samples ~= `6587` generated tokens/sample
  - SpecDec r5: `13,458,677` generated tokens / `2048` samples ~= `6572` generated tokens/sample
- Step 1 early speedup:
  - baseline r5 generated tok/s/GPU: `3913.51`
  - SpecDec r5 generated tok/s/GPU: `4136.19`
  - speedup: `1.057x`
- Step 1 SpecDec acceptance:
  - aggregate acceptance: `21.96%`
  - per-position acceptance: `36.10%, 18.00%, 11.05%`
- Step 2 early generation:
  - baseline r5 generated tok/s/GPU: `3805.95`
  - SpecDec r5 generated tok/s/GPU: `3984.47`
  - speedup: `1.047x`
  - SpecDec acceptance: `29.32%`
  - per-position acceptance: `51.02%, 25.88%, 11.01%`
- Step 3 early generation:
  - baseline r5 generated tok/s/GPU: `3931.53`
  - SpecDec r5 generated tok/s/GPU: `4123.43`
  - speedup: `1.049x`
  - SpecDec acceptance: `23.43%`
  - per-position acceptance: `42.05%, 19.86%, 8.32%`
- E2E Step 1/2 speedups:
  - Step 1: `1.028x`
  - Step 2: `1.022x`
  - Step 3: `1.018x`

This means the current NeMo-RL r5 result is a long, stochastic, real-prompt rollout, while the standalone result is short, greedy, synthetic/static generation. The standalone `2x` result should not be expected to transfer directly under those conditions.

## E2E Dilution, Not Just Non-Generation Bottleneck

The current NeMo-RL result is not simply "generation is not the bottleneck." In the long rollout runs, generation is still the largest single timing bucket, but the generation speedup is too small to move total step time much.

Observed Qwen3-8B long-rollout timing:

- r5 baseline Step 1: generation `861.77s` / total `1560.43s` = `55.2%`
- r5 SpecDec Step 1: generation `813.47s` / total `1514.16s` = `53.7%`
- r5 baseline Step 2: generation `908.06s` / total `1619.49s` = `56.1%`
- r5 SpecDec Step 2: generation `861.75s` / total `1575.27s` = `54.7%`
- r5 baseline Step 3: generation `910.93s` / total `1651.24s` = `55.2%`
- r5 SpecDec Step 3: generation `872.84s` / total `1629.18s` = `53.6%`
- r6 BF16 baseline Step 1: generation `858.83s` / total `1549.58s` = `55.4%`
- r6 BF16 SpecDec Step 1: generation `826.97s` / total `1519.94s` = `54.4%`

The rest of the step is dominated by policy/reference logprob forward and policy training:

- policy/reference logprobs: about `20-21%`
- policy training: about `23-25%`

Therefore, a `~1.05x` generation speedup naturally becomes only about `~1.02x` E2E speedup. To recover standalone-like E2E, either the generation speedup must become much larger, or the optimization must also reduce the logprob/training portions. SpecDec only targets generation.

## Code-Level Evidence

Standalone vLLM benchmark:

- File: `/lustre/.../vllm-benchmark/standalone_vllm_specdec_breakdown.py`
- The benchmark builds:
  - `SamplingParams(min_tokens=args.osl, max_tokens=args.osl, ignore_eos=True, temperature=0.0, seed=0)`
- For the completed Qwen3-8B run, `args.osl=512`.

NeMo-RL vLLM worker:

- File: `/lustre/.../SpecDec-RL/nemo_rl/models/generation/vllm/vllm_worker.py`
- `_build_sampling_params()` sets:
  - `temperature = 0.0 if greedy else self.cfg["temperature"]`
  - `max_tokens = max_new_tokens if provided else self.cfg["max_new_tokens"]`
  - `stop_token_ids=self.cfg["stop_token_ids"]`
  - `stop=stop_strings`
- The generation path computes per-sample allowed tokens as:
  - `remaining_ctx_per_sample = clamp(max_model_len - input_lengths, min=0, max=max_new_tokens)`
  - then passes per-sample `max_new_tokens=allowed_new_tokens_per_sample[idx]`.

8B recipe/config:

- File: `/lustre/.../SpecDec-RL/examples/configs/recipes/llm/grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron.yaml`
- The file default is `policy.model_name: Qwen/Qwen3-8B-Base`, but the launchers override both `policy.model_name` and `policy.tokenizer.name` to `Qwen/Qwen3-8B`. The target model ID is therefore aligned with the standalone run.
- It sets `policy.max_total_sequence_length: 8192`.
- It inherits from `examples/configs/grpo_math_1B.yaml`, where:
  - `policy.generation.max_new_tokens: ${policy.max_total_sequence_length}`
  - `policy.generation.temperature: 1.0`
  - `policy.generation.top_p: 1.0`
  - `policy.generation.vllm_cfg.max_model_len: ${policy.max_total_sequence_length}`

## Fixes and Diagnostics Applied

1. KV/q_scale compatibility fix

   r3 failed with vLLM TRITON attention q_scale assertion under FP8 KV cache. r4 then failed during FP8 weight refit because vLLM expected public `Attention.q_scale`. The remote `nemo_rl/models/generation/fp8.py` was patched so the refit-safe KV-cache post-load hook applies for `kv_cache_dtype in auto/fp8/fp8_e4m3` and tolerates missing public `q_scale`/`prob_scale` attributes. r5 passed the r4 failure point and emitted Step 1 metrics.

2. BF16 generation override

   The Qwen3-8B launchers now accept `VLLM_PRECISION`, and r6 jobs run with `policy.generation.vllm_cfg.precision=bfloat16`, `kv_cache_dtype=auto`. This matches the completed standalone benchmark more closely than the original FP8-weight recipe path.

3. Apples-to-apples short greedy diagnostic

   The Qwen3-8B launchers now accept:

   - `POLICY_MAX_TOTAL_SEQUENCE_LENGTH`
   - `GENERATION_MAX_NEW_TOKENS`
   - `GENERATION_TEMPERATURE`
   - `GENERATION_TOP_P`
   - generic extra overrides

   r7 jobs were submitted to isolate workload mismatch:

   - baseline `3124999`
   - SpecDec K=3 `3125000`
   - `precision=bfloat16`
   - `kv_cache_dtype=auto`
   - `policy.max_total_sequence_length=2560`
   - `policy.generation.max_new_tokens=512`
   - `policy.generation.temperature=0.0`
   - `policy.generation.top_p=1.0`
   - `max_steps=1`
   - SpecDec request/token gate thresholds set to `0` so K=3 is always enabled, closer to standalone.

4. Greedy diagnostic logprob-repair guard

   NeMo-RL omits vLLM request logprobs for SpecDec because vLLM V1 disables speculative decoding when request logprobs are asked for. The sync GRPO path then repairs behavior logprobs from a fresh policy fprop only when the sampler is identity (`temperature=1`, `top_p=1`, `top_k=-1`). The r7 diagnostic intentionally uses greedy `temperature=0.0`, so it may fail after generation unless `NRL_ALLOW_SPECDEC_LOGPROB_REPAIR_WITH_SAMPLER_MISMATCH=true` is set for throughput-only testing. A corrected r8 SpecDec diagnostic (`3125058`) was submitted with that override.

5. Fixed-length decode diagnostic support

   The completed standalone vLLM benchmark uses `SamplingParams(min_tokens=512, max_tokens=512, ignore_eos=True, temperature=0.0)`. The original NeMo-RL vLLM worker only passed `max_tokens`, not `min_tokens` or `ignore_eos`, so the r7/r8 diagnostics were still not fully apples-to-apples. The remote NeMo-RL vLLM worker and launchers were patched to pass `NRL_VLLM_GENERATION_MIN_TOKENS` and `NRL_VLLM_GENERATION_IGNORE_EOS` into `SamplingParams`. Exact r9 diagnostics were submitted:

   - baseline `3125141`
   - SpecDec K=3 `3125142`
   - `precision=bfloat16`, `kv_cache_dtype=auto`
   - `policy.max_total_sequence_length=2560`
   - `policy.generation.max_new_tokens=512`
   - `NRL_VLLM_GENERATION_MIN_TOKENS=512`
   - `NRL_VLLM_GENERATION_IGNORE_EOS=true`
   - `policy.generation.temperature=0.0`
   - SpecDec gate thresholds `0`, so K=3 is always enabled

## Latest Poll, 2026-06-02 21:42 PDT

- r5 FP8-weight/kvauto/qscalefix is still running. The matched Step 2 early-generation comparison is:
  - baseline `3124055`: `3805.95` generated tok/s/GPU
  - SpecDec K=3 `3124056`: `3984.47` generated tok/s/GPU
  - speedup: `1.047x`
  - SpecDec acceptance: `29.32%`
  - per-position acceptance: `51.02%, 25.88%, 11.01%`
- r6 BF16/kvauto baseline `3124761` and SpecDec K=3 `3124762` both reached setup complete and are actively generating Step 1/20 responses. No aggregate generation throughput or acceptance line has emitted yet.
- r7/r8 short512 greedy diagnostics remain running with no rollout metric yet. r8 is the corrected SpecDec run with `NRL_ALLOW_SPECDEC_LOGPROB_REPAIR_WITH_SAMPLER_MISMATCH=true`.
- Qwen3-32B standalone vLLM K=1/K=3 is complete and plotted. Qwen3-32B NeMo-RL baseline/K=1/K=3 remain running with no rollout metric yet.

## Latest BF16 Result, 2026-06-02 21:50 PDT

r6 uses `precision=bfloat16` and `kv_cache_dtype=auto`, matching the standalone dtype/KV mode more closely than the original FP8-weight recipe path. Step 1 early generation:

- baseline `3124761`: `3918.69` generated tok/s/GPU
- SpecDec K=3 `3124762`: `4079.89` generated tok/s/GPU
- speedup: `1.041x`
- E2E speedup: `1.022x`
- SpecDec acceptance: `30.06%`
- per-position acceptance: `49.88%, 26.10%, 13.93%`

This does not materially improve over r5's long stochastic rollout speedup (`1.057x` Step 1, `1.047x` Step 2, `1.049x` Step 3; E2E around `1.02x`). Precision is therefore not the primary explanation for the missing standalone-like speedup. The stronger explanation remains workload/sampling/length mismatch, pending the exact r11 short512 greedy diagnostic.

## Latest Patch, 2026-06-02 21:49 PDT

The exact short-decode comparison needed one more code change: NeMo-RL had no path to force `min_tokens=512` and `ignore_eos=True`, while standalone uses both. I added env-driven support in:

- `nemo_rl/models/generation/vllm/vllm_worker.py`
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- `nemo_rl/models/generation/vllm/vllm_generation.py`
- Qwen3-8B and Qwen3-32B baseline/SpecDec launchers

Local and remote `py_compile` passed for the vLLM worker files, and local/remote `bash -n` passed for the patched launchers. The patched files were synced to the remote SpecDec-RL checkout before submitting r9.

## Scheduler Diagnostic Correction, 2026-06-02 21:56 PDT

r7 baseline `3124999` and r7 SpecDec `3125000` failed before rollout generation. This was not a vLLM or SpecDec failure. The short diagnostic set `grpo.max_num_steps=1`, while the 8B recipe sets `policy.megatron_cfg.scheduler.lr_warmup_iters=10`. Megatron policy worker initialization asserted:

- `lr_warmup_steps < lr_decay_steps`

r8 `3125058` and r9 `3125141`/`3125142` had the same `max_steps=1` issue, so I issued `scancel` after confirming the root cause. r9 did verify that the command line carried `NRL_VLLM_GENERATION_MIN_TOKENS=512` and `NRL_VLLM_GENERATION_IGNORE_EOS=true`, but it would not reach generation.

Corrected r10 diagnostics were submitted:

- baseline `3125192`
- SpecDec K=3 `3125193`
- `precision=bfloat16`, `kv_cache_dtype=auto`
- `policy.max_total_sequence_length=2560`
- `policy.generation.max_new_tokens=512`
- `NRL_VLLM_GENERATION_MIN_TOKENS=512`
- `NRL_VLLM_GENERATION_IGNORE_EOS=true`
- `policy.generation.temperature=0.0`
- SpecDec gate thresholds `0`, so K=3 is always enabled
- scheduler override: `policy.megatron_cfg.scheduler.lr_warmup_iters=0`, `policy.megatron_cfg.scheduler.lr_decay_iters=2`

This keeps the Step 1 generation workload aligned with standalone while avoiding a setup-time scheduler assertion.

As of 2026-06-02 22:00 PDT, both r10 jobs reached Ray workers `4/4` and entered driver venv/package setup. No rollout metric has emitted yet. The slurm `ray status` traceback was a readiness-poll transient; a subsequent `ray status` call succeeded and the driver command launched.

## Exact Diagnostic Correction, 2026-06-02 22:05 PDT

One more standalone mismatch remained in r10:

- baseline r10 still requested vLLM generation request logprobs (`logprobs=0`), while the standalone benchmark does not request logprobs;
- NeMo-RL still passed stop strings and `stop_token_ids` into `SamplingParams`, while the standalone benchmark uses fixed-length `min_tokens=max_tokens=512`, `ignore_eos=True`, and no stop strings.

I patched the vLLM generation path so exact diagnostics can disable these:

- `NRL_VLLM_GENERATION_DISABLE_STOP_STRINGS=true`
- `NRL_VLLM_GENERATION_DISABLE_STOP_TOKEN_IDS=true`

The envs are propagated through `vllm_generation.py` into Ray actors and are honored in `vllm_worker.py` and `vllm_worker_async.py`. The 8B/32B launchers now pass them via `GENERATION_DISABLE_STOP_STRINGS` and `GENERATION_DISABLE_STOP_TOKEN_IDS`. Local and remote `py_compile` passed for the vLLM worker files, and local/remote `bash -n` passed for the 8B/32B launchers.

r10 `3125192`/`3125193` was cancelled before rollout metrics. Corrected r11 diagnostics are running:

- baseline `3125264`
- SpecDec K=3 `3125265`
- `precision=bfloat16`, `kv_cache_dtype=auto`
- `policy.max_total_sequence_length=2560`
- `policy.generation.max_new_tokens=512`
- `NRL_VLLM_GENERATION_MIN_TOKENS=512`
- `NRL_VLLM_GENERATION_IGNORE_EOS=true`
- `NRL_VLLM_GENERATION_DISABLE_STOP_STRINGS=true`
- `NRL_VLLM_GENERATION_DISABLE_STOP_TOKEN_IDS=true`
- `NRL_VLLM_OMIT_GENERATION_LOGPROBS=true` for both baseline and SpecDec
- `policy.generation.temperature=0.0`
- scheduler override: `policy.megatron_cfg.scheduler.lr_warmup_iters=0`, `policy.megatron_cfg.scheduler.lr_decay_iters=2`
- SpecDec gate thresholds `0`, so K=3 is always enabled

This is the closest NeMo-RL GRPO diagnostic to the completed standalone vLLM `LLM.generate` setup. Remaining differences after r11 are mostly system-level: NeMo-RL still shards 2048 responses across generation workers and runs inside the GRPO loop, while standalone uses static batch sweeps up to batch size 32.

As of 2026-06-02 22:09 PDT, r11 baseline `3125264` and SpecDec K=3 `3125265` are running with the intended command-line settings. No rollout metric has emitted yet.

## Engine-Shape Diagnostic, 2026-06-02 22:16 PDT

## Dashboard and K=2 Update, 2026-06-02 22:34 PDT

The HTML dashboard was refreshed at:

- `docs/specdec_completed_eval_bar_graphs.html`

It now includes:

- completed vLLM standalone ceilings;
- current Qwen3-8B and Qwen3-32B NeMo-RL early speedup tables;
- `docs/specdec_current_nemorl_early_speedup.png`;
- `docs/specdec_step_acceptance_rates.png`, including aggregate step acceptance and K=3 per-position acceptance;
- active job matrix for gated, always-on, and short512/greedy diagnostics.

Qwen3-8B vLLM standalone K=2 also emitted a result:

- bs1: `451.68` output tok/s/GPU, `1.838x`
- bs2: `893.40`, `1.851x`
- bs4: `1806.26`, `2.111x`
- bs8: `3486.00`, `1.945x`
- bs16: `6717.31`, `2.086x`
- bs32: `12552.57`, `2.346x`
- acceptance: `96.3%`, per-position `96.6%, 96.0%`

This strengthens the standalone ceiling: for Qwen3-8B short fixed greedy decode, K=2 is currently the strongest standalone setting at bs32. The corresponding NeMo-RL K=2 gated and always-on jobs are now running:

- gated BF16 long rollout: `3125443`
- always-on BF16 long rollout: `3125444`

The always-on K=1/K=3 long-rollout diagnostics are also running:

- K=1 always-on: `3125397`
- K=3 always-on: `3125396`

## Short512 Diagnostic Result, 2026-06-02 22:40 PDT

r11 emitted the first short512/greedy/no-logprob/no-stop generation metric:

- baseline `3125264`: `20642.41` generated tok/s/GPU
- SpecDec K=3 `3125265`: `17860.50` generated tok/s/GPU
- generation speedup: `0.865x`
- acceptance: `40.79%`
- per-position acceptance: `61.16%, 38.78%, 22.18%`

Both r11 jobs then failed in policy training with NaN local grad norm, so r11 is not a valid E2E result. The generation metric is still useful because it emitted before reward/logprob/training.

This is important because it weakens the "only workload length/sampling mismatch" explanation. The r11 diagnostic matches the standalone benchmark much more closely than the long stochastic rollout: BF16, KV auto, fixed `max_new_tokens=512`, `min_tokens=512`, `ignore_eos=true`, greedy `temperature=0.0`, no stop strings/token IDs, and no vLLM request logprobs. Despite that, the NeMo-RL vLLM path is still slower with K=3.

The current root-cause picture is therefore:

1. Long stochastic GRPO rollout does dilute E2E speedup, but generation itself is only `~1.04-1.06x`, so E2E is naturally `~1.02x`.
2. The TLT-style gate is very restrictive in long rollout. Logs show SpecDec enables only when the vLLM scheduler active request count reaches `<=8`; most scheduler steps are disabled.
3. The short512 r11 diagnostic shows that even when the sampling/length mismatch is mostly removed, NeMo-RL does not reproduce standalone K=3. This points to remaining system-level differences: NeMo-RL's worker sharding and GRPO loop integration, vLLM engine shape/scheduler behavior, CUDA graph capture shape, or output post-processing/logprob tensor handling.
4. The r12 engine-shape diagnostic now also regressed badly: baseline `5732.16` generated tok/s/GPU vs K=3 SpecDec `2422.57`, or `0.423x`, despite `40.66%` acceptance and standalone-like `max_num_seqs=32`, `max_num_batched_tokens=64000`, and `gpu_memory_utilization=0.82`. This weakens engine-shape-only explanations and moves the next focus to NeMo-RL integration overhead or per-worker scheduling dynamics.

I also patched the 8B/32B launchers to pass `NRL_STOP_AFTER_GENERATION` into the driver command. Future generation-only diagnostics can set `NRL_STOP_AFTER_GENERATION=true` and exit after `generation_finish`, avoiding r11-style reward/logprob/training failures while preserving the generation throughput and acceptance measurements.

While r11 matches the standalone `SamplingParams` shape, it still does not match the standalone vLLM engine shape. The completed standalone Qwen3-8B benchmark creates the engine with:

- `max_num_seqs=32`
- `max_num_batched_tokens=64000`
- `gpu_memory_utilization=0.82`

The NeMo-RL recipe inherits `gpu_memory_utilization=0.6` from `grpo_math_1B.yaml` and has no explicit `max_num_seqs` or `max_num_batched_tokens`. Since NeMo-RL submits `2048` responses per rollout step, each generation worker can operate in a much larger scheduler regime than the standalone `bs<=32` benchmark. This is now a concrete remaining mismatch, not just a generic workload difference.

I added explicit launcher passthroughs for:

- `VLLM_GPU_MEMORY_UTILIZATION`
- `VLLM_ENABLE_METRICS_LOGGER`

and added an effective vLLM engine config log in `vllm_worker.py` so future diagnostics print `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `max_model_len`, metrics-logger state, and `disable_log_stats`.

New r12 diagnostics were submitted:

- baseline `3125310`
- SpecDec K=3 `3125311`

r12 keeps the r11 exact SamplingParams alignment and adds:

- `max_num_seqs=32`
- `max_num_batched_tokens=64000`
- `gpu_memory_utilization=0.82`
- `policy.generation.vllm_cfg.enable_vllm_metrics_logger=false`

This is currently the closest NeMo-RL GRPO diagnostic to the completed standalone vLLM engine and sampling setup.

r12 Step 1 result, emitted at the 2026-06-02 22:55 PDT poll:

- baseline `3125310`: `5732.16` generated tok/s/GPU
- SpecDec K=3 `3125311`: `2422.57` generated tok/s/GPU
- generation speedup: `0.423x`
- acceptance: `40.66%`
- per-position acceptance: `61.05%, 38.46%, 22.27%`
- E2E: pending

This is a stronger negative diagnostic than r11. It matched the short fixed greedy generation shape and the standalone-like vLLM engine shape more closely, yet NeMo-RL K=3 was substantially slower. The remaining gap is therefore unlikely to be explained by output length, greedy-vs-stochastic sampling, or `max_num_seqs` alone.

One important acceptance comparison is now available from the completed standalone K=2 run. The Qwen3-8B standalone bs32 K=2 synthetic/static prompt benchmark emitted `96.29%` acceptance with per-position acceptance `96.57%, 96.00%` and `2.346x` generation speedup. That is very different from r12's DAPOMath short512 K=3 acceptance of `40.66%`, with per-position acceptance `61.05%, 38.46%, 22.27%`. This makes prompt/domain acceptance mismatch a strong explanation for the missing speedup unless r19 exact-engine/no-gate recovers.

## r19 Exact-Engine Follow-Up, 2026-06-02 23:03 PDT

r12 still left three standalone-vs-NeMo engine differences:

- standalone passed `enable_chunked_prefill=False`;
- standalone passed `disable_custom_all_reduce=True`;
- standalone used `max_model_len=2536`;
- r12 still installed the NeMo-RL runtime scheduler gate patch, even with thresholds set to zero.

I patched the Qwen3-8B/32B launchers to pass through:

- `VLLM_MAX_MODEL_LEN`
- `VLLM_ENABLE_CHUNKED_PREFILL`
- `VLLM_DISABLE_CUSTOM_ALL_REDUCE`

and extended the vLLM worker effective-config log to print `enable_chunked_prefill` and `disable_custom_all_reduce`.

Submitted generation-only follow-up diagnostics:

- baseline `3125698`
- K=3 SpecDec `3125699`
- K=1 SpecDec `3125708`

These use BF16/KV-auto, fixed 512-token greedy decode, no stop strings/token IDs, no vLLM request logprobs, `NRL_STOP_AFTER_GENERATION=true`, `max_num_seqs=32`, `max_num_batched_tokens=64000`, `max_model_len=2536`, `gpu_memory_utilization=0.82`, `enable_chunked_prefill=false`, `disable_custom_all_reduce=true`, and `ENABLE_RUNTIME_SPECDEC_GATE_PATCH=false`.

Interpretation target:

- If r19 recovers speedup, the r12 regression was caused by the remaining engine/gate-patch differences.
- If r19 remains slow, the most likely root cause is drafter acceptance/domain mismatch or intrinsic EAGLE3 overhead in NeMo-RL's rollout request distribution rather than a simple engine-flag mismatch.

## Qwen3-32B Early NeMo-RL Signal, 2026-06-02 22:09 PDT

The Qwen3-32B standalone vLLM ceiling is complete and strong: K=1 is useful, and K=3 is roughly `2.17x-2.74x` across tested batch sizes.

The first NeMo-RL r1 Step 1 early metric is much weaker:

- baseline `3124830`: `1725.68` generated tok/s/GPU
- SpecDec K=1 `3124832`: `1587.87` generated tok/s/GPU
- K=1 speedup: `0.920x`
- K=1 E2E speedup: `0.937x`
- K=1 acceptance: `47.13%`
- SpecDec K=3 `3124831`: `1583.84` generated tok/s/GPU
- K=3 speedup: `0.918x`
- K=3 E2E speedup: `0.943x`
- K=3 acceptance: `24.73%`
- K=3 per-position acceptance: `42.87%, 21.05%, 10.08%`
- Step 2 baseline `3124830`: `1902.67` generated tok/s/GPU, `1061.69` E2E tok/s/GPU
- Step 2 SpecDec K=1 `3124832`: `1780.64` generated tok/s/GPU, generation speedup `0.936x`, E2E speedup `0.947x`, acceptance `42.50%`
- Step 2 SpecDec K=3 `3124831`: `1805.79` generated tok/s/GPU, generation speedup `0.949x`, E2E speedup `0.961x`, acceptance `22.18%`, per-position `39.91%, 18.20%, 8.24%`
- Step 3 baseline `3124830`: `1776.10` generated tok/s/GPU, `1026.21` E2E tok/s/GPU
- Step 3 SpecDec K=1 `3124832`: `1700.44` generated tok/s/GPU, generation speedup `0.957x`, E2E speedup `0.942x`, acceptance `48.71%`
- Step 3 SpecDec K=3 `3124831`: `1708.89` generated tok/s/GPU, generation speedup `0.962x`, E2E speedup `0.969x`, acceptance `24.65%`, per-position `43.27%, 20.54%, 9.94%`

This mirrors the 8B pattern so far: standalone vLLM shows a high generation-only ceiling, while the NeMo-RL rollout loop currently has lower acceptance and no matched speedup yet. For 32B, K=1 and K=3 are both slower than baseline across the first three NeMo-RL early matched metrics.

## Latest 8B Gate and K Evidence, 2026-06-02 23:10 PDT

The newest 8B long-rollout diagnostics materially change the root-cause ranking. The runtime scheduler gate is now a strong candidate, at least for the long stochastic rollout case:

- gated K=3 BF16 long rollout remains small:
  - Step 1: `1.041x` generation speedup, `30.06%` acceptance
  - Step 2: `1.059x` generation speedup, `27.38%` acceptance
  - Step 3: `1.048x` generation speedup, `26.12%` acceptance
  - Step 4: `1.019x` generation speedup, `23.68%` acceptance
- gated K=1 long rollout Step 1:
  - job `3125392`
  - generated tok/s/GPU: `4076.91`
  - speedup vs BF16 baseline Step 1 `3124761`: `1.040x`
  - acceptance: `54.20%`
- always-on K=1 long rollout Step 1:
  - job `3125397`
  - generated tok/s/GPU: `5644.76`
  - speedup vs BF16 baseline Step 1: `1.440x`
  - acceptance: `63.93%`
- always-on K=3 long rollout Step 1:
  - job `3125396`
  - generated tok/s/GPU: `5818.07`
  - speedup vs BF16 baseline Step 1: `1.485x`
  - acceptance: `38.65%`
  - per-position acceptance: `58.99%, 36.23%, 20.73%`

This suggests the TLT-style runtime gate was suppressing a large part of the long-rollout generation speedup. The evidence is especially strong because K=1 gated and K=1 always-on use the same draft depth and very similar acceptance regime, but the always-on run is much faster.

However, gate behavior is not the entire root cause:

- short512 K=1 engine diagnostic `3125393` reached `1.278x` generation speedup with `64.61%` acceptance before failing after generation/training NaN;
- short512 K=3 r11 reached only `0.865x` generation speedup with `40.79%` acceptance;
- short512 K=3 r12 engine-shaped diagnostic reached only `0.423x` generation speedup with `40.66%` acceptance.

That means K selection and workload fit are also important. For Qwen3-8B, K=1 currently looks much safer in the NeMo-RL short fixed-decode path, while K=3 is only compelling in the always-on long rollout so far.

K=2 remains unresolved. The 8B always-on K=2 job `3125444` failed before any rollout metric. Its log shows a vLLM V1 `dump_input` scheduler dump at generation start for `num_spec_tokens=2`, `max_seq_len=8192`, and `enable_chunked_prefill=True`; it should not be counted as a performance result.

## Cross-Model Gate Evidence, 2026-06-02 23:22 PDT

The 23:22 PDT poll adds E2E evidence and a matching 32B pattern.

For Qwen3-8B, always-on long rollout now improves both generation and E2E:

- K=1 always-on `3125397`:
  - generation speedup vs BF16 long baseline Step 1: `1.440x`
  - E2E speedup vs BF16 long baseline Step 1: `1.189x`
  - acceptance: `63.93%`
- K=3 always-on `3125396`:
  - generation speedup vs BF16 long baseline Step 1: `1.485x`
  - E2E speedup vs BF16 long baseline Step 1: `1.233x`
  - acceptance: `38.65%`
  - per-position acceptance: `58.99%, 36.23%, 20.73%`

The gated 8B K sweep stays much smaller:

- gated K=1 Step 1: `1.040x` generation speedup, `54.20%` acceptance
- gated K=2 Step 1: `1.035x` generation speedup, `42.05%` acceptance
- gated K=3 BF16 Step 1: `1.041x` generation speedup, `30.06%` acceptance

For Qwen3-32B, the same pattern appears:

- gated K=1 Step 1: `0.920x` generation, `0.937x` E2E, `47.13%` acceptance
- gated K=2 Step 1: `0.909x` generation, `0.936x` E2E, `32.75%` acceptance
- gated K=3 Step 1: `0.918x` generation, `0.943x` E2E, `24.73%` acceptance
- always-on K=1 Step 1: `1.337x` generation, `1.149x` E2E, `70.13%` acceptance
- always-on K=3 Step 1: `1.350x` generation, `1.163x` E2E, `45.82%` acceptance

The 32B always-on K=2 job `3125462` failed before aggregate rollout metrics. Its vLLM dump shows high KV-cache pressure near failure, with KV cache usage around `98.6%`, running requests in the low 60s, and waiting requests present. It should be rerun with memory/scheduler controls before drawing a K=2 always-on conclusion.

This cross-model evidence makes the TLT-style runtime scheduler gate the strongest current root-cause candidate for missing NeMo-RL long-rollout speedup. The gate appears to suppress SpecDec in exactly the regime where always-on produces useful throughput.

The code-level gate mechanism is now verified. The patched scheduler computes:

- `active_requests = max(num_requests, len(self.running))`
- `disabled = active_requests > request_threshold or num_tokens > token_threshold`

With the current gated setting (`request_threshold=8`, `token_threshold=4096`), NeMo-RL's 2048-response rollout keeps `active_requests` well above the request threshold for almost the whole decode. Observed gate metrics show this is not theoretical:

- Qwen3-8B gated runs: scheduler lookahead enabled only about `0.03-0.06%`; runner draft enabled about `1-2%`.
- Qwen3-32B gated runs: scheduler lookahead enabled only about `0.18-0.39%`; runner draft enabled about `3-6%`.

The dashboard now includes `docs/specdec_gate_enabled_ratio.png` to make this visible. This means the gated runs are mostly baseline decoding plus a small long-tail SpecDec tail. Always-on runs avoid this by disabling the runtime gate entirely.

The 23:31 PDT dashboard update adds model-specific step-acceptance charts and K=3 acceptance-to-speedup projection charts:

- `docs/specdec_qwen3_8b_step_acceptance_rates.png`
- `docs/specdec_qwen3_32b_step_acceptance_rates.png`
- `docs/specdec_qwen3_8b_acceptance_projection_speedup.png`
- `docs/specdec_qwen3_32b_acceptance_projection_speedup.png`

The projection is intentionally a what-if estimate, not a measured config. It scales generation lift linearly from the current always-on K=3 point and caps generation speedup at the measured vLLM standalone K=3 ceiling. E2E is estimated with an Amdahl-style generation fraction calibrated from the observed always-on generation/E2E pair.

Projection summary:

- Qwen3-8B K=3 current always-on Step 1-2 average: `36.61%` acceptance, `1.532x` generation speedup, `1.243x` E2E speedup. If acceptance reached `80%`, the projection hits the measured standalone K=3 ceiling of `2.145x` generation and about `1.430x` E2E.
- Qwen3-32B K=3 current always-on Step 1-4 average: `45.26%` acceptance, `1.372x` generation speedup, `1.173x` E2E speedup. At `90%` acceptance, the projection is about `1.740x` generation and `1.300x` E2E.

## Runtime-Env Diagnostic Fix

The Qwen3-8B exact-engine/no-gate diagnostic jobs `3125698`, `3125699`, and `3125708` did not produce metrics because they hit a Ray runtime-env mismatch, not a performance failure. The Ray default worker was launched from the Python 3.12 cluster environment, while the vLLM actor runtime env resolved to Python 3.13 and raised:

`TypeError: connect() got an unexpected keyword argument 'startup_token'`

I patched `nemo_rl/distributed/virtual_cluster.py` so `PY_EXECUTABLES.BASE/VLLM/AUTOMODEL/MCORE/NEMO_GYM` include `uv run --python ${UV_PYTHON}` when `UV_PYTHON` is set. The broken jobs were cancelled, and fresh exact-engine retries were submitted with rebuilt Python 3.12 envs:

- `3126144` Qwen3-8B exact-engine baseline
- `3126145` Qwen3-8B exact-engine K=1 no-gate
- `3126146` Qwen3-8B exact-engine K=3 no-gate

As of the 23:52 PDT poll these retries were still running and rebuilding fresh driver environments. The driver logs showed `Using CPython 3.12.13` and package installation completed; no `startup_token`/Python 3.13 mismatch reappeared. However, by the next poll the jobs were stuck in fresh native-extension build and had emitted no rollout metric, so I cancelled them and submitted replacement r21 jobs that reuse already-built py312 driver environments:

- `3126251` Qwen3-8B exact-engine baseline reuse
- `3126252` Qwen3-8B exact-engine K=1 no-gate reuse
- `3126253` Qwen3-8B exact-engine K=3 no-gate reuse

The r21 settings are otherwise the same exact-engine diagnostic settings: BF16, KV auto, fixed 512-token greedy decode, `ignore_eos=true`, no stop strings/token IDs, no generation logprobs, `NRL_STOP_AFTER_GENERATION=true`, `max_num_seqs=32`, `max_num_batched_tokens=64000`, `max_model_len=2536`, `gpu_memory_utilization=0.82`, `enable_chunked_prefill=false`, `disable_custom_all_reduce=true`, and runtime gate disabled for SpecDec.

## r21 Actor Cache Failure and r22 Retry, 2026-06-03 00:24 PDT

The r21 exact-engine/no-gate jobs did not produce throughput or acceptance metrics:

- `3126251` Qwen3-8B exact-engine baseline reuse
- `3126252` Qwen3-8B exact-engine K=1 no-gate reuse
- `3126253` Qwen3-8B exact-engine K=3 no-gate reuse

They reached Ray actor creation, but reused the stale actor-local vLLM worker cache:

`/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/lib/python3.13`

The Ray default worker for these r21 jobs was correctly launched from the Python 3.12 cluster env:

`/tmp/nemo_rl_ray_312625*_3.12.13_2.49.2/lib/python3.12`

That mixed Python/Ray ABI path caused the same startup failure:

`TypeError: connect() got an unexpected keyword argument 'startup_token'`

So r21 was cancelled. This is still a runtime-env/cache issue, not a performance result.

I patched the actor venv creation path:

- `nemo_rl/distributed/worker_groups.py` now adds `UV_PYTHON` and optional `NRL_ACTOR_VENV_CACHE_SUFFIX` to the actor-local venv cache name.
- `nemo_rl/utils/venvs.py` now honors `NRL_FORCE_REBUILD_ACTOR_VENVS=true`.
- Qwen3-8B and Qwen3-32B baseline/SpecDec launchers now pass `NRL_FORCE_REBUILD_ACTOR_VENVS` and `NRL_ACTOR_VENV_CACHE_SUFFIX` into the driver command.

Replacement r22 jobs were submitted with prebuilt py312 driver venvs, `NRL_FORCE_REBUILD_VENVS=false`, `NRL_FORCE_REBUILD_ACTOR_VENVS=true`, and `NRL_ACTOR_VENV_CACHE_SUFFIX=py312_r22`:

- `3126355` Qwen3-8B exact-engine baseline
- `3126356` Qwen3-8B exact-engine K=1 no-gate
- `3126357` Qwen3-8B exact-engine K=3 no-gate

All three r22 jobs were running at the `2026-06-03 00:23 PDT` poll. No r22 rollout metric had emitted yet.

At the `2026-06-03 00:40 PDT` poll, all three r22 jobs were still running:

- `3126355` baseline on `nvl72047-T08`
- `3126356` K=1 no-gate on `nvl72128-T18`
- `3126357` K=3 no-gate on `nvl72036-T14`

The important update is that the r21 actor-cache failure has not reappeared. The logs now show actor-local venvs under:

`/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker.py3.12.13.py312_r22`

The vLLM actor venv creation finished, and K=1/K=3 reached model checkpoint loading plus CUDA graph capture. The current work is MegatronPolicyWorker actor venv/native-extension setup, including `megatron-core`, `megatron-bridge`, `emerging-optimizers`, `nv-grouped-gemm`, and `transformer-engine-torch`. No r22 throughput or acceptance metric has emitted yet, so the plotted performance values are unchanged.

## r22 Exact-Engine Result, 2026-06-03 00:45 PDT

r22 produced the exact-engine/no-gate generation-only diagnostic metrics:

| Run | Job | Generated tok/s/GPU | Speedup vs r22 baseline | Acceptance | Per-position acceptance | Status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Baseline | `3126355` | `5652.52` | `1.000x` | n/a | n/a | stop-after-generation completed |
| K=1 no-gate | `3126356` | `7345.13` | `1.299x` | `64.61%` | `64.61%` | stop-after-generation completed |
| K=3 no-gate | `3126357` | `2555.29` | `0.452x` | `40.66%` | `61.05%, 38.46%, 22.27%` | stop-after-generation completed |

This confirms two things:

- The Python 3.13 actor-cache/runtime-env issue was a submission/cache problem and is fixed for this diagnostic path.
- Under the exact fixed 512-token greedy generation shape, K=1 helps, but K=3 still regresses badly despite the runtime gate being disabled.

The K=3 result is close to the earlier r12 engine-shaped diagnostic (`0.423x`, `40.66%` acceptance). That means the main remaining K=3 issue is not BF16 vs FP8 or the runtime gate. It is more likely K/workload fit plus NeMo-RL/vLLM integration overhead for multi-token draft verification under this rollout path. The per-position acceptance decays sharply after draft position 1, so K=3 pays extra draft/verify overhead while only a minority of positions 2-3 are useful.

## r23 Dynamic Draft-Cap Follow-Up, 2026-06-03 00:53 PDT

Submitted r23 job `3126527` to test the practical fix direction implied by r22:

- Engine config still requests EAGLE3 K=3.
- Runtime scheduler gate patch is enabled with very high request/token thresholds (`100000` requests, `100000000` tokens), so it should not disable SpecDec in this diagnostic.
- Dynamic draft-depth cap is enabled:
  - active requests `<=4`: select `3` draft tokens
  - active requests `<=16`: select `2` draft tokens
  - active requests `>16`: select `1` draft token
- The r22 exact shape is otherwise preserved: BF16, KV auto, fixed 512-token greedy decode, `ignore_eos=true`, no stop strings/token IDs, no generation logprobs, stop-after-generation, `max_num_seqs=32`, `max_num_batched_tokens=64000`, `max_model_len=2536`, `gpu_memory_utilization=0.82`, `enable_chunked_prefill=false`, and `disable_custom_all_reduce=true`.

This is a targeted fix experiment rather than another root-cause probe. If it recovers near the r22 K=1 speedup while reporting dynamic large-tier selection, the actionable Qwen3-8B recommendation is to avoid fixed K=3 in high-concurrency NeMo-RL rollouts and use dynamic depth or K=1 for this drafter/workload.

At the `2026-06-03 01:04 PDT` poll, r23 `3126527` was still running. The command and environment are present in `ray-head.log`, including `VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS=true`, small/medium/large dynamic token caps `3/2/1`, and the high request/token gate thresholds. No `Early Generation Worker Group` or `[SpecDec early metrics]` line had emitted yet.

At the `2026-06-03 01:12 PDT` poll, r23 had progressed past vLLM actor venv creation, vLLM checkpoint loading, and CUDA graph capture. It then failed before generation metrics in MegatronPolicyWorker optimizer scheduler construction:

`AssertionError: lr_warmup_steps < lr_decay_steps`

This is a submission override issue, not a dynamic draft-cap performance result. r23 used `grpo.max_num_steps=2` without overriding the recipe default `policy.megatron_cfg.scheduler.lr_warmup_iters=10`, so the optimizer scheduler assertion triggered during policy actor creation.

Submitted corrected r24 job `3126685` with the same dynamic draft-cap settings plus:

- `policy.megatron_cfg.scheduler.lr_warmup_iters=0`
- `policy.megatron_cfg.scheduler.lr_decay_iters=2`

No r24 throughput metric has emitted yet.

## Real-Prompt Standalone Isolation, 2026-06-03 01:13 PDT

The completed Qwen3-8B standalone ceiling used synthetic prompt token IDs. To isolate prompt/domain acceptance from NeMo-RL integration overhead, I generated a DAPOMath17K prompt JSONL from the same training source and shuffle seed used by NeMo-RL:

- Prompt source: `BytedTsinghua-SIA/DAPO-Math-17k`
- Shuffle seed: `42`
- Prompt file: `/lustre/.../vllm-benchmark/prompts/dapo_math17k_seed42_64.jsonl`
- JSONL format: `messages` with user prompt plus assistant ground truth; the standalone script stops at the assistant turn and applies the Qwen chat template.

Submitted Qwen3-8B vLLM standalone bs32 DAPOMath prompt jobs:

| Mode | Job | K | Purpose |
| --- | ---: | ---: | --- |
| baseline | `3126678` | n/a | real-prompt bs32 baseline |
| SpecDec | `3126679` | 1 | real-prompt K=1 acceptance/speedup |
| SpecDec | `3126680` | 2 | real-prompt K=2 acceptance/speedup |
| SpecDec | `3126681` | 3 | real-prompt K=3 acceptance/speedup |

Settings: OSL `512`, ISL cap `1000`, dtype `auto`, KV cache `auto`, `max_model_len=2536`, `max_num_batched_tokens=64000`, `gpu_memory_utilization=0.82`, `disable_custom_all_reduce=true`, profiler disabled. If DAPOMath standalone acceptance drops toward the NeMo-RL r22/r12 acceptance range, prompt/domain mismatch is a major explanation. If it stays high but NeMo-RL K=3 remains slow, the remaining cause is primarily NeMo-RL integration/scheduling overhead.

## Latest Long-Rollout Results, 2026-06-03 01:04 PDT

Qwen3-8B always-on continues to show real generation and E2E lift:

- K=1 always-on `3125397`:
  - Step 1: generation `1.440x`, E2E `1.189x`, acceptance `63.93%`
  - Step 2: generation `1.458x`, E2E `1.196x`, acceptance `59.38%`
  - Step 3: generation `1.383x`, E2E `1.163x`, acceptance `58.30%`
  - Step 4: generation `1.373x`, E2E `1.169x`, acceptance `60.70%`
- K=3 always-on `3125396`:
  - Step 1: generation `1.485x`, E2E `1.233x`, acceptance `38.65%`, per-position `58.99%, 36.23%, 20.73%`
  - Step 2: generation `1.578x`, E2E `1.253x`, acceptance `34.58%`, per-position `54.27%, 31.90%, 17.56%`
  - Step 3: generation `1.435x`, E2E `1.204x`, acceptance `34.02%`, per-position `53.05%, 31.33%, 17.68%`
  - Step 4: generation `1.433x`, E2E `1.211x`, acceptance `35.63%`, per-position `55.53%, 33.00%, 18.37%`

Qwen3-8B gated runs remain close to baseline despite nonzero acceptance:

- gated K=1 `3125392`: through Step 5, generation `1.013-1.055x`; E2E through Step 4 `1.010-1.019x`; acceptance `46.27-54.49%`
- gated K=2 `3125443`: through Step 5, generation `1.010-1.052x`; E2E through Step 4 `1.006-1.026x`; acceptance `34.12-42.05%`
- gated K=3 BF16 `3124762`: through Step 8, generation `1.000-1.059x`; E2E through Step 7 `1.004-1.025x`; acceptance `22.06-36.48%`

Qwen3-32B reinforces the same conclusion:

- gated K=1/K=2/K=3 remain below baseline through the latest matched steps.
- gated K=2 `3125460` is now observed through Step 15: generation `0.890-0.952x`; E2E through Step 14 `0.918-0.967x`; acceptance `27.37-36.98%`.
- always-on K=1 `3125461` completed four steps with generation `1.337-1.398x`, E2E `1.149-1.176x`, and acceptance about `68.76-70.13%`.
- always-on K=3 `3125463` completed four steps with generation `1.331-1.430x`, E2E `1.151-1.196x`, aggregate acceptance about `44.55-45.88%`, and per-position acceptance around `66%/42%/26%`.

The gate metrics were recomputed from all observed `[SpecDec early gate]` rows:

- Qwen3-8B gated K=1/K=2/K=3: scheduler enabled ratio `0.040%`/`0.038%`/`0.035%`; runner enabled ratio `1.44%`/`1.25%`/`1.32%`
- Qwen3-32B gated K=1/K=2/K=3: scheduler enabled ratio `0.270%`/`0.252%`/`0.232%`; runner enabled ratio `4.10%`/`3.88%`/`3.66%`

This is direct evidence that the TLT-style gate is not just mildly conservative. In the NeMo-RL GRPO rollout regime it disables scheduler lookahead for essentially the entire high-concurrency decode and only enables SpecDec in the long tail.

The dashboard and shareable PNGs have been updated:

- `docs/specdec_completed_eval_bar_graphs.html`
- `docs/specdec_gate_enabled_ratio.png`
- `docs/specdec_qwen3_8b_step_acceptance_rates.png`
- `docs/specdec_qwen3_32b_step_acceptance_rates.png`
- `docs/specdec_qwen3_8b_acceptance_projection_speedup.png`
- `docs/specdec_qwen3_32b_acceptance_projection_speedup.png`

## Working Root-Cause Hypothesis

The long-rollout gap is partly explained by workload mismatch, but r11/r12 and the new always-on jobs show that workload mismatch is not sufficient as the root cause. The strongest current hypothesis is now a combination of runtime gate behavior, K/workload fit, and NeMo-RL integration overhead that appears when SpecDec is used through the GRPO rollout worker path:

1. Output length mismatch

   Standalone Qwen3-8B generates exactly `512` tokens. NeMo-RL r5 Step 1 generated about `6.6K` tokens/sample.

2. Sampling mismatch

   Standalone is greedy (`temperature=0.0`). NeMo-RL rollout is stochastic (`temperature=1.0`). Stochastic sampling usually lowers draft acceptance because the verifier distribution is sampled rather than taking the deterministic top token path.

3. Prompt/domain mismatch

   Standalone uses fixed synthetic/static prompt token IDs. NeMo-RL uses DAPO Math prompts and long reasoning outputs.

4. Scheduler/load-regime mismatch

   Standalone tested batch sizes `1-32`. NeMo-RL r5 submits `2048` responses per step across four generation workers, so each vLLM worker handles a much larger queue and a long-tail completion regime.

5. Acceptance mismatch

   The Qwen3-8B NeMo-RL r5 K=3 Step 1 acceptance was only `21.96%`, with later draft positions especially weak. A K=3 speedup near standalone requires useful later-position acceptance.

6. Integration/scheduling-path overhead

   r12 still regressed at `0.423x` even with short fixed greedy generation, no request logprobs, disabled stop tokens/strings, and standalone-like engine limits. That points to the SpecDec path under NeMo-RL's rollout worker scheduling, request batching, CUDA graph capture shape, or drafter/verifier orchestration overhead.

## Next Evidence Needed

1. Code-level NeMo-RL/vLLM integration audit for the r12 path

   r12 is now far below standalone, so inspect NeMo-RL-specific overheads next:

   - generation logprob request/omission behavior,
   - SpecDec gate propagation,
   - per-worker request batching,
   - CUDA Graph capture sizes,
   - vLLM metrics/logger overhead,
   - scheduler batch-size and token-pressure behavior.

2. Full 20-step r5/r6 metrics

   Step 1 is useful but not final. Later steps can differ due to refit/training dynamics and updated policy distribution.

3. GuideLLM speculator acceptance evaluation

   The `vllm-project/speculators` GuideLLM example can be useful as a third axis: a drafter-quality and acceptance-length evaluation under vLLM serving. It should not replace standalone fixed-OSL throughput or NeMo-RL GRPO rollout measurements, but it can answer whether the drafter itself has healthy acceptance on the benchmark dataset.

### 15:35 PDT submitted follow-up

Qwen3-8B online drafter training smoke `3135673` reached rollout generation with
`policy.draft.enabled=true`, BF16 generation, KV cache `auto`, and the V2
scheduler scrub patch. The earlier vLLM V1 bookkeeping assertion and r69
scheduled-token broadcast failure did not reproduce. The run failed after
generation during `policy.get_logprobs()`:

- failure: TransformerEngine RMSNorm CUDA invalid argument
- call path: `grpo_train -> policy.get_logprobs -> MegatronPolicyWorker.get_logprobs -> megatron_forward_backward -> TELayerNormColumnParallelLinear`
- relevant config: `max_total_sequence_length=8192`, `max_new_tokens=8192`, `logprob_batch_size=1`, `train_micro_batch_size=1`, Megatron `TP=4`, BF16
- observed SpecDec metric before failure: K=1 gated emitted per-worker acceptance rows around `25.6%` and `42.9%`

This means the scheduler/bookkeeping fix moved the online 8B job past the
generation-side vLLM failure, but the online training path still needs a
policy/logprob stability fix for the long 8192-token rollout setting.

PR `NVIDIA-NeMo/RL#2658` was rechecked for this failure. Its relevant change is
the revert of vLLM sleep level-2 weight discard for colocated inference. The
current remote code already uses `sleep(level=1)` and has no `discard_weights`
plumbing in `finish_generation()`, so `3135673` is not the old level-2 weight
discard failure. It is now isolated to the Megatron/TransformerEngine policy
forward path after rollout.

## 2026-06-03 16:03 PDT Update

Qwen3-32B worker-batch matching experiments are now emitting early metrics. The
new condition uses `GBS=512` with 16 generation workers, so each generation
worker sees about `32` responses instead of the original `GBS=2048` condition's
about `128` responses per worker. This is the closest NeMo-RL comparison point
to the vLLM standalone `bs32` sweep.

| Mode | Job | Step scope | Gen speedup | E2E speedup | Acceptance | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| K=1 always | `3136001` | Step 1-2 mean | 1.42x | 1.21x | 70.0% | Step 3 early generation also emitted; Step 3 gen speedup is 1.42x |
| K=3 always | `3136002` | Step 1-2 mean | 1.66x | 1.32x | 46.4% | K=3 Step 3 is still pending |

Step-level detail:

| Mode | Step | Generated tok/s/GPU | E2E tokens/sec | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| baseline `3136000` | 1 | 1301.84 | 11570.09 | n/a |
| baseline `3136000` | 2 | 931.72 | 9604.41 | n/a |
| K=1 `3136001` | 1 | 1779.09 | 13526.61 | 69.29% |
| K=1 `3136001` | 2 | 1365.31 | 11972.18 | 70.73% |
| K=3 `3136002` | 1 | 2064.21 | 14647.18 | 45.46% |
| K=3 `3136002` | 2 | 1623.29 | 13228.23 | 47.29% |

Interpretation: reducing the per-worker response batch size makes the
Qwen3-32B always-on NeMo-RL signal closer to standalone. In the original
completed `GBS=2048` always-on run, K=3 was generation `1.36x`, E2E `1.18x`,
and acceptance `45.28%`. In this new worker32 early signal, K=3 is generation
`1.66x`, E2E `1.32x`, and acceptance `46.4%`. That means one root cause of the
standalone gap is the NeMo-RL rollout batching/workload shape. The remaining K=3
gap to vLLM standalone bs32 (`2.288x`, acceptance `67.1%`) is still explained by
lower NeMo-RL acceptance and E2E composition.

Later poll: K=3 `3136002` has progressed through Step 9 early generation. For
the matched baseline steps currently available, K=3 mean generation speedup is
`1.64x` over Step 1-8, mean E2E speedup is `1.31x` over Step 1-7, and mean
acceptance is `45.9%` over Step 1-8. Step 4/5/6/7/8 individually remain close
to that average, so the worker32 improvement is not only a first-step artifact.
Step 9 has K=3 metrics but its matched baseline is still pending.

New artifacts:

| Artifact | Content |
| --- | --- |
| `docs/qwen3_32b_nemorl_worker32_early.png` | Qwen3-32B worker32 Step 1-2 speedup/acceptance chart, 512 dpi |
| `docs/qwen3_32b_nemorl_worker32_early_metrics.csv` | Raw metric backing the chart |

Qwen3-8B online short512 r73 `3136275` is still running. The MasterConfig
confirms `policy.draft.enabled=true`, `draft.model_name=RedHatAI/Qwen3-8B-speculator.eagle3`,
`max_total_sequence_length=2560`, `max_new_tokens=512`, BF16 generation, and KV
auto. No rollout metric or RMSNorm failure has emitted yet.

## 2026-06-03 15:35 PDT Update

Submitted a shorter online-drafter smoke to separate long-sequence TE instability
from SpecDec integration:

| Job | Model | Mode | Key settings | Status |
| ---: | --- | --- | --- | --- |
| `3135990` | Qwen3-8B | K=1 always, online drafter | `max_total_sequence_length=2560`, `max_new_tokens=512`, vLLM `max_model_len=2560`, BF16/KV auto, `policy.draft.enabled=true` | failed before Ray startup; Slurm could not create the ray-head step because memory required by task was unavailable |
| `3136275` | Qwen3-8B | K=1 always, online drafter | same as `3135990`, but `CPUS_PER_WORKER=24` low-CPU retry | running |

Submitted Qwen3-32B worker-batch matching jobs to compare NeMo-RL more directly
against standalone `bs32`:

| Job | Model | Mode | Key settings | Status |
| ---: | --- | --- | --- | --- |
| `3136000` | Qwen3-32B | baseline | `8 nodes x 2 GPUs`, 16 generation workers, `GBS=512`, expected `32` responses/worker | running; Step 1-2 baseline metrics emitted, Step 3 early generation emitted |
| `3136001` | Qwen3-32B | K=1 always | same as baseline, runtime gate disabled | running; Step 1-2 mean gen/E2E speedup `1.42x`/`1.21x`, acceptance `70.0%`; Step 3 early generation emitted |
| `3136002` | Qwen3-32B | K=3 always | same as baseline, runtime gate disabled | running; Step 1-2 mean gen/E2E speedup `1.66x`/`1.32x`, acceptance `46.4%` |

The original Qwen3-32B completed always-on result used `GBS=2048`, which maps to
about `128` responses per generation worker with 16 workers. The new `GBS=512`
condition maps to about `32` responses per generation worker, making it closer
to the standalone `batch_size=32` comparison point.
