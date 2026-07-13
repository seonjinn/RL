# Qwen3 Speculative Decoding 평가 요약

Updated: 2026-06-04 02:21 PDT

## Executive Summary

현재까지의 결론은 아래와 같습니다.

1. 이번 이슈는 KV cache quantization이나 FP8 generation 때문으로 보이지 않습니다. NeMo-RL 실험은 generation precision을 BF16, KV cache dtype을 `auto`로 맞춰서 돌렸고, vLLM standalone 비교도 같은 축에서 해석하고 있습니다.
2. vLLM standalone에서는 Qwen3-32B public EAGLE3 drafter가 확실한 ceiling을 보입니다. K=3에서 2.288x speedup, acceptance 67.1%입니다.
3. NeMo-RL fixed/offline drafter에서는 always-on SpecDec가 generation throughput을 올립니다. 하지만 gated mode는 speculation이 거의 켜지지 않아 대부분 baseline decoding처럼 동작합니다.
4. Qwen3-8B도 logprob bottleneck을 줄인 changed-objective diagnostic에서는 speedup이 보입니다. Step 1-4 평균 기준 K=1 always는 generation 1.290x / E2E 1.152x, K=3 always는 generation 1.343x / E2E 1.175x입니다. 이 diagnostic에서 generation time share는 약 58%로 역산됩니다.
5. Qwen3-32B worker-batch matching 실험(`GBS=512`, 16 generation workers, 약 32 responses/worker)은 early signal이 좋아졌습니다. Step 1-2 평균 기준 K=1 always는 generation 1.42x / E2E 1.21x, K=3 always는 generation 1.66x / E2E 1.32x입니다.
6. Online drafter training(`policy.draft.enabled=true`)은 r84/r85 paired smoke에서 처음으로 end-to-end metric까지 완료됐습니다. 하지만 Qwen3-8B K=1 always는 matched online baseline 대비 generation `0.605x`, E2E `0.914x`, acceptance `40.80%`로 slowdown입니다.
7. Qwen3-8B greedy/fixed512 online follow-up r86/r87도 completed negative입니다. r87 K=1 acceptance는 `41.82%`로 r84와 거의 같고, generation throughput은 greedy baseline 대비 `0.757x`입니다. 따라서 r84 slowdown의 주원인은 stochastic/default-GRPO sampling이 아니라 public drafter/domain mismatch와 online drafter/verification overhead로 보는 것이 맞습니다.
8. post-SamplingParams-patch r88/r89도 completed negative입니다. actor venv rebuild로 patched worker를 실제 사용했지만 r89 K=1은 acceptance `42.01%`, generation `0.732x`, E2E `1.029x`였습니다. 따라서 worker-side fixed-length `SamplingParams` propagation 누락도 root cause가 아닙니다.
9. 추가 코드 분석에서 greedy online training의 `draft_loss=nan` 원인을 찾았습니다. `train.py::apply_temperature_scaling()`이 `temperature=0.0`에서도 training logits를 temperature로 나눠서 NaN을 만들었습니다. 이 때문에 r86-r89는 generation/acceptance throughput diagnostic으로는 유효하지만 online drafter 학습 결과로는 유효하지 않습니다. 패치 후 r90/r91 Step 1에서 finite draft loss를 확인했습니다.
10. r92/r93 clean rerun은 5/5 step 완료됐습니다. Online draft update는 acceptance를 Step 1 `42.24%`에서 Step 2 `70.42%`, Step 5 `71.70%`까지 올렸습니다. 하지만 5-step aggregate는 generation `0.964x`, E2E `0.980x`라 standalone-like speedup에는 아직 못 갔습니다.
11. r94/r95/r96 20-step clean rerun도 완료됐습니다. K=1 always는 generation `1.093x`, E2E `1.029x`, mean acceptance `61.20%`로 generation slice에서는 회복됐습니다. K=3 always는 generation `0.968x`, E2E `0.995x`, mean acceptance `37.46%`로 아직 손해입니다.
12. r97/r98/r99 engine-shaped rerun도 완료됐습니다. Standalone-shaped engine knobs를 넣어도 K=1은 generation/E2E `1.120x/1.060x`, mean acceptance `61.75%`라 r22 exact-engine `1.299x`까지는 못 갑니다. K=3는 generation/E2E `1.009x/1.041x`, mean acceptance `37.40%`로, acceptance 문제는 그대로입니다.
12. Qwen3-235B vLLM standalone decode-heavy OSL=10000 K=1 sweep은 acceptance가 85-98%인데도 throughput이 0.608-0.823x로 전부 slowdown입니다. 이것은 acceptance만으로 speedup이 보장되지 않고 drafter/verification/scheduler/cache/parallelism overhead가 saved target work보다 크면 손해가 난다는 보강 evidence입니다.

## 공유용 산출물

| Artifact | 내용 |
|---|---|
| `docs/qwen3_specdec_team_share_2026_06_03.md` | 이 문서, 팀 공유용 최신 요약 |
| `docs/qwen3_specdec_team_update_2026_06_03.md` | 상세 진행 로그와 전체 히스토리 |
| `docs/specdec_completed_eval_bar_graphs.html` | 완료/관측 결과 dashboard |
| `docs/specdec_followup_latest_generation_bars.png` | NeMo-RL fixed/offline generation throughput bar chart, 512 dpi |
| `docs/specdec_followup_latest_acceptance_by_model.png` | model별 acceptance chart, 512 dpi |
| `docs/specdec_qwen3_8b_logprob_skip_diag_step1_step4.png` | Qwen3-8B logprob-skip diagnostic Step 1-4 generation/E2E/acceptance chart, 512 dpi |
| `docs/qwen3_32b_nemorl_worker32_early.png` | Qwen3-32B NeMo-RL worker≈32 early speedup/acceptance chart, 512 dpi |
| `docs/qwen3_32b_nemorl_worker32_early_metrics.csv` | 위 worker≈32 chart의 원본 metric |
| `docs/qwen3_8b_nemorl_specdec_root_cause.md` | Qwen3-8B root-cause/debug note |
| `docs/qwen3_8b_nemorl_vs_vllm_root_cause_matrix.md` | Qwen3-8B NeMo-RL vs vLLM standalone root-cause matrix |
| `docs/qwen3_8b_online_gradfix_r92_r93.png` | Qwen3-8B online finite-loss + gradfix r92/r93 speedup and acceptance chart, 512 dpi |
| `docs/qwen3_8b_online_gradfix_r92_r93_metrics.csv` | r92/r93 step-level metrics used for the chart |
| `docs/qwen3_8b_online_gradfix_r94_r96_20step.png` | Qwen3-8B online finite-loss + gradfix r94/r95/r96 20-step speedup and acceptance chart, 512 dpi |
| `docs/qwen3_8b_online_gradfix_r94_r96_20step_summary.csv` | r94/r95/r96 20-step aggregate summary |
| `docs/qwen3_8b_online_gradfix_r94_r96_20step_metrics.csv` | r94/r95/r96 step-level source metrics |
| `docs/qwen3_8b_engine_shape_r97_r99_20step.png` | Qwen3-8B engine-shaped online finite-loss + gradfix r97/r98/r99 20-step chart, 512 dpi |
| `docs/qwen3_8b_engine_shape_r97_r99_20step_summary.csv` | r97/r98/r99 20-step aggregate summary |
| `docs/qwen3_8b_engine_shape_r97_r99_20step_metrics.csv` | r97/r98/r99 step-level source metrics |
| `docs/qwen3_235b_vllm_decodeheavy10k_speedup_acceptance.png` | Qwen3-235B OSL=10000 standalone high-acceptance slowdown chart |
| `experiments/eagle3_online/submit_qwen8_r84online_short512_k1always_worker32_fixedlen_raymatch.sh` | Submitted Qwen3-8B online drafter metric smoke, K=1 always, worker≈32, fixed 512 decode |
| `experiments/eagle3_online/submit_qwen8_r85online_short512_baseline_worker32_fixedlen_raymatch.sh` | Submitted paired Qwen3-8B online drafter baseline, same worker≈32/fixed512 shape with vLLM SpecDec disabled |
| `experiments/eagle3_online/submit_qwen8_r86online_short512_greedy_baseline_worker32_fixedlen_raymatch.sh` | Completed paired Qwen3-8B online drafter greedy baseline, worker≈32, fixed 512 decode |
| `experiments/eagle3_online/submit_qwen8_r87online_short512_greedy_k1always_worker32_fixedlen_raymatch.sh` | Completed paired Qwen3-8B online drafter greedy K=1 always diagnostic |
| `experiments/eagle3_online/submit_qwen8_r88online_short512_greedy_baseline_worker32_fixedlen_postpatch.sh` | Completed post-SamplingParams-patch greedy baseline with actor venv rebuild |
| `experiments/eagle3_online/submit_qwen8_r89online_short512_greedy_k1always_worker32_fixedlen_postpatch.sh` | Completed post-SamplingParams-patch greedy K=1 always diagnostic with actor venv rebuild |
| `experiments/eagle3_online/submit_qwen8_r92r93online_short512_greedy_finiteloss_gradfix_5step_postpatch.sh` | Completed clean Qwen3-8B finite-loss + gradfix 5-step paired rerun |
| `experiments/eagle3_online/submit_qwen8_r94r96online_short512_greedy_finiteloss_gradfix_20step.sh` | Completed Qwen3-8B finite-loss + gradfix 20-step baseline/K=1/K=3 follow-up |
| `experiments/eagle3_online/remote_patches_qwen8_online_finiteloss_gradfix.patch` | Remote source patch used by r92/r93: greedy finite-loss fix plus grad offload fix |

## 최신 상태

### Completed / Current

| Job | Model / Mode | Status at 17:55 PDT | Notes |
|---:|---|---|---|
| 3132444 | Qwen3-8B logprob-skip baseline r5 | COMPLETED, 01:53:34 | Step 1-4 generation and E2E metrics emitted. |
| 3132635 | Qwen3-8B logprob-skip K=1 always r6 | COMPLETED, 01:39:09 | Step 1-4 metric emitted. |
| 3132636 | Qwen3-8B logprob-skip K=3 always r6 | COMPLETED, 01:38:41 | Step 1-4 metric emitted. |
| 3133925 | Qwen3-8B online K=1 gated r68 scrub smoke | FAILED, 00:11:02 | Failed before rollout while building the policy actor venv: Megatron-Bridge editable build hit `Disk quota exceeded`. This is an infra/env failure, not a SpecDec performance result. |
| 3136275 | Qwen3-8B online K=1 always r73 short512 | FAILED, 00:23:16 | Config confirmed `policy.draft.enabled=true`, BF16 generation, KV auto, max_new_tokens 512. vLLM generated SpecDec acceptance logs, then policy-side TE RMSNorm failed during logprob forward. |
| 3136718 | Qwen3-8B online K=1 always r74 short512 local transformer spec | FAILED, 00:20:16 | Same as r73, plus `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true`. Local-spec marker was confirmed and the original TE RMSNorm failure did not recur. New failure: local Megatron fused softmax imports missing `scaled_masked_softmax_cuda`. |
| 3137127 | Qwen3-8B online K=1 always r75 short512 local transformer, no fused softmax | FAILED, 00:23:38 | `get_logprobs()` completed. Training then failed in online EAGLE draft forward at `EagleModule.enorm(embeddings)` -> TE RMSNorm CUDA invalid-argument. vLLM acceptance was 0.0%, likely from reusing a TE-converted checkpoint while forcing local spec at runtime. |
| 3137466 | Qwen3-8B online K=1 always r76 short512 local transformer, reconvert, draft torch RMSNorm | FAILED before Ray driver, 00:00:35 | Slurm ray-head `srun` failed on `nvl72083-T05`: `Memory required by task is not available`. No model/code metric. |
| 3137513 | Qwen3-8B online K=1 always r77 short512 local transformer, reconvert, draft torch RMSNorm | FAILED, 00:17:41 | Local-spec HF->Megatron import marker fired, but dense `Qwen3Bridge` lacked local layernorm mappings. Conversion warned about `decoder.layers.*.input_layernorm.weight` / `pre_mlp_layernorm.weight`, then failed with `AttributeError: 'NoneType' object has no attribute 'megatron_module'`. |
| 3137739 | Qwen3-8B online K=1 always r78 short512 local transformer, reconvert, draft torch RMSNorm, Qwen3 mapping fix | FAILED, 00:52:59 | Dense Qwen3Bridge mapping fix worked; vLLM generation acceptance recovered to 57.5-63.0% and policy logprobs completed. Training failed in ModelOpt EAGLE decoder `TransformerLayer.input_layernorm` -> TE RMSNorm CUDA invalid-argument. |
| 3139283 | Qwen3-8B online K=1 always r79 short512 recursive draft torch RMSNorm | FAILED, 00:28:40 | Recursive replacement marker fired for `enorm`, `decoder.layers.0.input_layernorm`, `decoder.layers.0.pre_mlp_layernorm`, and `decoder.final_layernorm`; generation completed with vLLM Avg Draft acceptance 66.6% / 57.6% repeated and policy logprobs completed. New failure: Megatron DDP `finish_grad_sync()` assertion, `0/3 params have grad available`. |
| 3139703 | Qwen3-8B online K=1 always r80 short512 no-overlap retry | FAILED, 00:35:10 | DDP no-overlap avoided the r79 `Communication call has not been issued` assertion and reached optimizer/update-success reduction, then failed because `logical_and_across_model_parallel_group(..., mp_group=...)` was not supported by the installed helper signature. |
| 3140266 | Qwen3-8B online K=1 always r81 helper-compat retry | FAILED, 00:02:03 | Launcher failure before model path: `ray.sub` parsed Slurm GRES `gpu:4(S:0-1)` as a non-numeric GPU count. |
| 3140576 | Qwen3-8B online K=1 always r82 GRES parser retry | FAILED, 00:08:09 | GRES parser fixed; failure was Ray head/driver mismatch, Ray 2.49.2/Python 3.12.13 vs Ray 2.54.0/Python 3.13.13. |
| 3140704 | Qwen3-8B online K=1 always r83 Ray-match retry | CANCELLED by request, 00:11:31 | Ray version mismatch fixed; Ray init, MasterConfig, data loaders, vLLM worker init, and CUDA Graph capture succeeded. Cancelled before rollout metric or online acceptance aggregate. |
| 3141902 | Qwen3-8B online K=1 always r84 worker≈32 fixed512 | COMPLETED | Generation `3206.21` tok/s/GPU, E2E `343.62` tok/s/GPU, acceptance `40.80%`; matched speedup vs r85 is generation `0.605x`, E2E `0.914x`. |
| 3142137 | Qwen3-8B online paired baseline r85 worker≈32 fixed512 | COMPLETED | Generation `5302.53` tok/s/GPU, E2E `375.78` tok/s/GPU. |
| 3142745 | Qwen3-8B online greedy paired baseline r86 worker≈32 fixed512 | COMPLETED, 00:21:29 | Generation `5364.60` tok/s/GPU, E2E `357.13` tok/s/GPU, generation time `3.94s`, step time `59.18s`. |
| 3142746 | Qwen3-8B online greedy K=1 always r87 worker≈32 fixed512 | COMPLETED, 00:19:38 | Generation `4059.60` tok/s/GPU (`0.757x`), E2E `380.83` tok/s/GPU (`1.066x`), acceptance `41.82%`, generation time `5.21s`, step time `55.50s`. |
| 3143675 | Qwen3-8B online greedy paired baseline r88 post-SamplingParams patch | COMPLETED, 00:27:13 | Generation `5471.97` tok/s/GPU, E2E `363.54` tok/s/GPU, generation time `3.86s`, step time `58.14s`. |
| 3143676 | Qwen3-8B online greedy K=1 always r89 post-SamplingParams patch | COMPLETED, 00:26:55 | Generation `4005.30` tok/s/GPU (`0.732x`), E2E `374.12` tok/s/GPU (`1.029x`), acceptance `42.01%`, generation time `5.28s`, step time `56.49s`. |
| 3144473 | Qwen3-8B online greedy baseline r90 finite-loss 5-step | FAILED before Step 2, 00:21:26 | Step 1 finite draft loss confirmed: generation `4920.99` tok/s/GPU, draft loss `9.0362`; failed in `offload_before_refit` CPU-grad/CUDA-param assignment. |
| 3144474 | Qwen3-8B online greedy K=1 always r91 finite-loss 5-step | FAILED before Step 2, 00:18:22 | Step 1 finite draft loss confirmed: generation `3910.24` tok/s/GPU (`0.794x` vs r90), acceptance `42.24%`, draft loss `9.0676`; same offload failure. |
| 3144992 | Qwen3-8B online greedy baseline r92 finite-loss + gradfix 5-step | COMPLETED, 00:24:01 | Step 1-5 baseline completed with finite draft loss; generation tok/s/GPU `5247.03, 6557.43, 6556.19, 6727.15, 6256.15`. |
| 3144993 | Qwen3-8B online greedy K=1 always r93 finite-loss + gradfix 5-step | COMPLETED, 00:25:47 | Completed after both patches. Acceptance `42.24%, 70.42%, 56.98%, 48.91%, 71.70%`; step gen speedup `0.760x, 1.110x, 0.959x, 0.949x, 1.217x`; aggregate gen/E2E `0.964x/0.980x`. |
| 3145919 | Qwen3-8B online greedy baseline r94 finite-loss + gradfix 20-step | COMPLETED | 20-step matched baseline. Baseline generation share is `6.83%` of total step time. |
| 3145920 | Qwen3-8B online greedy K=1 always r95 finite-loss + gradfix 20-step | COMPLETED | Generation `1.093x`, E2E `1.029x`, mean acceptance `61.20%`, mean acceptance length `1.612`. |
| 3145921 | Qwen3-8B online greedy K=3 always r96 finite-loss + gradfix 20-step | COMPLETED | Generation `0.968x`, E2E `0.995x`, mean acceptance `37.46%`, mean acceptance length `2.124`. |
| 3146944 | Qwen3-8B online greedy baseline r97 finite-loss + gradfix + enginebs32 20-step | COMPLETED, 00:37:49 | Standalone-shaped engine baseline. Baseline generation share is `7.00%` of total step time. |
| 3146945 | Qwen3-8B online greedy K=1 always r98 finite-loss + gradfix + enginebs32 20-step | COMPLETED, 00:39:29 | Generation `1.120x`, E2E `1.060x`, mean acceptance `61.75%`, mean acceptance length `1.617`. |
| 3146946 | Qwen3-8B online greedy K=3 always r99 finite-loss + gradfix + enginebs32 20-step | COMPLETED, 00:35:05 | Generation `1.009x`, E2E `1.041x`, mean acceptance `37.40%`, mean acceptance length `2.122`. |

### Newly Updated Failures

| Job | Model / Mode | Status | Failure point |
|---:|---|---|---|
| 3133619 | Qwen3-8B online K=1 gated V9 smoke | FAILED, 00:17:53 | Reached rollout generation, then failed in vLLM V1 `_bookkeeping_sync`: `assert sampled_token_ids.shape[-1] == 1`. V9 did not fix the scheduler/runner bookkeeping mismatch. |
| 3133620 | Qwen3-8B online K=1 always r67 | FAILED, 00:21:12 | Reached rollout generation, then failed in policy-side Megatron/TransformerEngine forward: `rmsnorm_fwd_cuda_kernel.cu:49 ... CUDA Error: invalid argument`. No rollout performance metric emitted. |
| 3133622 | Qwen3-8B online K=2 always r67 | FAILED, 00:18:40 | Reached rollout generation, then failed in policy-side Megatron/TransformerEngine forward: `rmsnorm_fwd_cuda_kernel.cu:49 ... CUDA Error: invalid argument`. No rollout performance metric emitted. |
| 3133624 | Qwen3-8B online K=3 always r67 | FAILED, 00:24:26 | Same policy-side Megatron/TransformerEngine RMSNorm CUDA invalid-argument. No rollout performance metric emitted. |
| 3133621/3133623/3133625 | Qwen3-8B online K=1/2/3 gated r67 | CANCELLED | Cancelled because the K=1 gated smoke `3133619` failed. |
| 3133925 | Qwen3-8B online K=1 gated r68 scrub smoke | FAILED, 00:11:02 | Actor venv build failed before rollout: `error: [Errno 122] Disk quota exceeded` under `3rdparty/Megatron-Bridge-workspace/megatron_bridge.egg-info/...`. No model metric emitted. |

## vLLM Standalone 결과

### Qwen3-32B + RedHatAI EAGLE3 Drafter

조건: `Qwen/Qwen3-32B`, `RedHatAI/Qwen3-32B-speculator.eagle3`, dtype/KV auto, ISL/OSL 1000/512, batch size 32.

| K | Baseline tok/s/GPU | SpecDec tok/s/GPU | Speedup | Acceptance |
|---:|---:|---:|---:|---:|
| 1 | 1905.67 | 3097.15 | 1.625x | 79.9% |
| 2 | 1905.67 | 3893.39 | 2.043x | 72.9% |
| 3 | 1905.67 | 4359.78 | 2.288x | 67.1% |

해석: standalone ceiling은 충분히 큽니다. 따라서 NeMo-RL에서 speedup이 낮은 것은 public drafter 자체가 무효라기보다는 NeMo-RL runtime/gating/workload/E2E composition 차이로 보는 것이 맞습니다.

### Qwen3-8B DAPOMath Real-Prompt Standalone

| K | Speedup | Acceptance |
|---:|---:|---:|
| 1 | 0.665x | 61.11% |
| 2 | 0.817x | 46.91% |
| 3 | 0.853x | 36.62% |

해석: Qwen3-8B public drafter는 synthetic fixed-prompt standalone에서 보였던 높은 ceiling이 DAPOMath real-prompt distribution에서는 재현되지 않았습니다. Qwen3-8B의 낮은 speedup에는 gate/E2E overhead뿐 아니라 prompt/domain mismatch도 포함됩니다.

### Qwen3-8B Online Drafter Matched r84/r85

조건: `Qwen/Qwen3-8B`, `RedHatAI/Qwen3-8B-speculator.eagle3`, online drafter training enabled, BF16 generation, KV auto, fixed 512-token decode, `NUM_PROMPTS=4`, `NUM_GENERATIONS=32`, `GBS=128`, 4 generation workers, worker당 약 32 responses.

| Job | Mode | Gen tok/s/GPU | Gen speedup | E2E tok/s/GPU | E2E speedup | Gen time | Step time | Acceptance |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `3142137` | paired baseline, vLLM SpecDec off, `policy.draft.enabled=true` | 5302.53 | 1.000x | 375.78 | 1.000x | 3.99s | 56.25s | n/a |
| `3141902` | K=1 always, vLLM EAGLE3 on | 3206.21 | 0.605x | 343.62 | 0.914x | 6.59s | 61.51s | 40.80% |

해석: online drafter training 자체는 이제 1-step smoke를 완료합니다. 다만 Qwen3-8B public drafter의 K=1 acceptance가 online/default GRPO workload에서 `40.80%`까지 내려가고, drafter/verification overhead가 saved target work보다 커서 generation이 baseline보다 느립니다. 또 baseline의 generation 비중이 `7.1%`뿐이라, generation slowdown `0.605x`가 E2E에서는 `0.914x`로 완화되어 보입니다.

### Qwen3-8B Greedy Online Follow-up r86/r87

목적: r84 slowdown이 default-GRPO stochastic sampling 때문에 acceptance가 낮아진 것인지, 아니면 online SpecDec integration overhead 자체가 큰 것인지 분리합니다. r84/r85와 같은 online-drafter path를 유지하되 `temperature=0.0`, `top_p=1.0`, `top_k=-1`, fixed 512 decode, no stop strings/token IDs로 standalone에 더 가깝게 맞췄습니다.

| Job | Mode | Status | Gen tok/s/GPU | Gen speedup | E2E tok/s/GPU | E2E speedup | Gen time | Step time | Acceptance |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3142745` | r86 greedy paired baseline, `policy.draft.enabled=true`, vLLM SpecDec off | COMPLETED | 5364.60 | 1.000x | 357.13 | 1.000x | 3.94s | 59.18s | n/a |
| `3142746` | r87 greedy K=1 always, online EAGLE3 on | COMPLETED | 4059.60 | 0.757x | 380.83 | 1.066x | 5.21s | 55.50s | 41.82% |

해석: greedy/fixed512로 맞춰도 acceptance가 `41.82%`라 r84의 `40.80%`와 거의 같습니다. 즉 r84의 낮은 acceptance는 stochastic/default-GRPO sampling 때문으로 보기 어렵고, Qwen3-8B public drafter의 DAPOMath/online prompt distribution mismatch와 online vLLM drafter/verification overhead가 더 직접적인 원인입니다. E2E `1.066x`는 generation이 빨라져서 나온 결과가 아닙니다. generation 자체는 `0.757x`로 느리고, baseline generation 비중이 `6.66%`뿐이라 one-step non-generation timing variance가 E2E를 뒤집어 보이게 한 것입니다.

### Qwen3-8B Post-SamplingParams Patch r88/r89

목적: r86/r87가 remote vLLM worker의 fixed-length `SamplingParams` env patch 전에 돌았기 때문에, `min_tokens`, `ignore_eos`, stop-disable 전파 누락 가능성을 제거하기 위해 actor venv rebuild를 켜고 동일한 greedy/fixed512 workload를 재실행했습니다.

| Job | Mode | Status | Gen tok/s/GPU | Gen speedup | E2E tok/s/GPU | E2E speedup | Gen time | Step time | Acceptance |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3143675` | r88 greedy paired baseline, patched worker, vLLM SpecDec off | COMPLETED | 5471.97 | 1.000x | 363.54 | 1.000x | 3.86s | 58.14s | n/a |
| `3143676` | r89 greedy K=1 always, patched worker, EAGLE3 on | COMPLETED | 4005.30 | 0.732x | 374.12 | 1.029x | 5.28s | 56.49s | 42.01% |

해석: post-patch에서도 acceptance는 `42.01%`로 r87의 `41.82%`와 거의 같고, generation은 baseline 대비 `0.732x`입니다. 따라서 Qwen3-8B online slowdown은 worker-side fixed-length `SamplingParams` propagation 문제가 아닙니다.

주의: r86-r89는 `temperature=0.0` greedy 설정 때문에 training loss path에서 NaN이 발생했습니다. 원인은 `train.py::apply_temperature_scaling()`이 `temperature=0.0`에서도 logits를 temperature로 나누는 코드였습니다. 이 패치와 `offload_before_refit` gradfix를 적용한 뒤 r92/r93이 5 step을 완료했습니다. Online draft update는 acceptance를 Step 2/5에서 `70.42%`/`71.70%`까지 올렸지만, 5-step aggregate speedup은 generation `0.964x`, E2E `0.980x`로 아직 부족합니다.

## NeMo-RL Fixed/Offline Drafter 결과

아래는 completed 또는 timeout 전까지 관측된 fixed/offline drafter matrix입니다. Qwen3-8B는 4시간 walltime으로 timeout되었고, 표에는 timeout 전 관측 step 평균을 넣었습니다. Qwen3-32B와 Qwen3-30B-A3B는 20 step 완료입니다.

| Model | K | Gate | Job | Status | Steps | Mean gen tok/s/GPU | Mean acceptance | Gate enabled |
|---|---:|---|---:|---|---:|---:|---:|---|
| Qwen3-8B | 1 | always | 3127152 | TIMEOUT | 9 | 5547.07 | 60.84% | n/a |
| Qwen3-8B | 1 | gated | 3127155 | TIMEOUT | 8 | 4027.48 | 51.74% | scheduler 0.02%, runner 0.63% |
| Qwen3-8B | 3 | always | 3127158 | TIMEOUT | 10 | 5915.66 | 36.04% | n/a |
| Qwen3-8B | 3 | gated | 3127161 | TIMEOUT | 8 | 4038.35 | 28.17% | scheduler 0.02%, runner 0.63% |
| Qwen3-32B | 1 | always | 3128147 | COMPLETED | 20 | 2388.46 | 69.45% | n/a |
| Qwen3-32B | 1 | gated | 3128148 | COMPLETED | 20 | 1643.20 | 45.58% | scheduler 0.26%, runner 4.10% |
| Qwen3-32B | 3 | always | 3128428 | COMPLETED | 20 | 2405.82 | 45.28% | n/a |
| Qwen3-32B | 3 | gated | 3128429 | COMPLETED | 20 | 1651.10 | 23.65% | scheduler 0.22%, runner 3.21% |
| Qwen3-30B-A3B | 1 | always | 3128151 | COMPLETED | 20 | 5601.48 | 57.51% | n/a |
| Qwen3-30B-A3B | 1 | gated | 3128152 | COMPLETED | 20 | 4372.35 | 48.00% | scheduler 0.00%, runner 0.02% |
| Qwen3-30B-A3B | 3 | always | 3128153 | COMPLETED | 20 | 4896.03 | 31.80% | n/a |
| Qwen3-30B-A3B | 3 | gated | 3128154 | COMPLETED | 20 | 4357.91 | 23.65% | scheduler 0.00%, runner 0.01% |

해석: 모든 모델에서 gated mode는 speculation을 거의 켜지 않습니다. Always-on path가 실제 acceleration 가능성을 보는 경로입니다.

## Qwen3-8B Logprob-Skip Diagnostic

이 실험은 default GRPO 결과가 아닙니다. 설정은 `force_on_policy_ratio=true`, `reference_policy_kl_penalty=0`, `NRL_GRPO_SKIP_POLICY_LOGPROBS_IF_SAFE=true`, BF16 generation, KV cache `auto`입니다. 목적은 policy/reference logprob forward bottleneck을 줄였을 때 NeMo-RL이 generation ceiling에 얼마나 가까워지는지 보는 것입니다.

요약하면, Qwen3-8B NeMo-RL에서도 generation speedup은 확인됐지만 E2E는 더 작게 올라갑니다. 동일 Step 1-4 로그에서 역산한 generation execution 비중은 약 58%입니다.

| Mode | Job | Mean gen tok/s/GPU | Gen speedup | Mean E2E tok/s/GPU | E2E speedup | Acceptance | Inferred generation share |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline r5 | 3132444 | 4260.76 | 1.000x | 2608.80 | 1.000x | n/a | n/a |
| K=1 always r6 | 3132635 | 5498.34 | 1.290x | 3004.60 | 1.152x | 60.57% | 58.5% |
| K=3 always r6 | 3132636 | 5721.98 | 1.343x | 3065.85 | 1.175x | 35.75% | 58.4% |

| Job | Mode | Step | Generated tok/s/GPU | Speedup vs matched baseline | Acceptance | Per-position acceptance |
|---:|---|---:|---:|---:|---:|---|
| 3132444 | baseline r5 | 1 | 4319.01 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | 2 | 4168.85 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | 3 | 4362.34 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | 4 derived | 4192.84 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | mean 1-4 | 4260.76 | 1.000x | n/a | n/a |
| 3132635 | K=1 always r6 | 1 | 5674.47 | 1.314x | 63.93% | 63.93% |
| 3132635 | K=1 always r6 | 2 | 5498.59 | 1.319x | 59.22% | 59.22% |
| 3132635 | K=1 always r6 | 3 | 5530.84 | 1.268x | 58.43% | 58.43% |
| 3132635 | K=1 always r6 | 4 | 5289.47 | 1.262x | 60.68% | 60.68% |
| 3132635 | K=1 always r6 | mean 1-4 | 5498.34 | 1.290x | 60.57% | n/a |
| 3132636 | K=3 always r6 | 1 | 5801.55 | 1.343x | 38.72% | 59.07%, 36.32%, 20.76% |
| 3132636 | K=3 always r6 | 2 | 5918.88 | 1.420x | 34.54% | 54.31%, 31.82%, 17.50% |
| 3132636 | K=3 always r6 | 3 | 5705.44 | 1.308x | 34.06% | 53.09%, 31.38%, 17.70% |
| 3132636 | K=3 always r6 | 4 | 5462.06 | 1.303x | 35.67% | 55.59%, 33.05%, 18.37% |
| 3132636 | K=3 always r6 | mean 1-4 | 5721.98 | 1.343x | 35.75% | n/a |

해석: Qwen3-8B도 NeMo-RL generation-only 기준으로는 speedup이 있습니다. 다만 K=3은 later-position acceptance가 낮습니다. Step 4 기준 2/3번째 draft position acceptance가 33.05% / 18.37%라서, default GRPO E2E에서 이 이득이 그대로 보장되지는 않습니다.

## Online Drafter Training 상태

모든 online drafter training 제출은 `policy.draft.enabled=true`, `policy.draft.loss_weight=1.0`, BF16 generation, KV cache `auto`를 기준으로 했습니다. 아직 성공한 rollout 성능 metric은 없습니다.

| Model | Job(s) | Status | Key observation |
|---|---|---|---|
| Qwen3-8B | 3133619 | FAILED | K=1 gated V9 smoke. vLLM V1 `_bookkeeping_sync`에서 `assert sampled_token_ids.shape[-1] == 1` 재발. |
| Qwen3-8B | 3133620 | FAILED | K=1 always r67. Policy-side TransformerEngine RMSNorm CUDA invalid-argument. |
| Qwen3-8B | 3133622 | FAILED | K=2 always r67. Policy-side TransformerEngine RMSNorm CUDA invalid-argument. |
| Qwen3-8B | 3133624 | FAILED | K=3 always r67. Policy-side TransformerEngine RMSNorm CUDA invalid-argument. |
| Qwen3-8B | 3133621/3133623/3133625 | CANCELLED | Gated K=1/2/3 20-step jobs. Smoke failure 때문에 cancelled. |
| Qwen3-8B | 3133925 | FAILED | K=1 gated r68 scrub smoke. Scheduler output scrub patch was submitted, but actor venv build failed from fs quota before rollout. Patch behavior is therefore still unvalidated. |
| Qwen3-8B | 3136275 | FAILED | K=1 always r73 short512. Reached generation and emitted vLLM internal acceptance 57.6-66.6%, then failed in policy-side TE RMSNorm during logprob forward. |
| Qwen3-8B | 3136718 | FAILED | K=1 always r74 short512. Local-spec marker confirmed; original TE RMSNorm avoided; failed on missing `scaled_masked_softmax_cuda`. |
| Qwen3-8B | 3137127 | FAILED | K=1 always r75 short512. Local spec plus `masked_softmax_fusion=False`; logprobs passed, then online EAGLE draft `enorm` failed in TE RMSNorm. |
| Qwen3-8B | 3137466 | FAILED before driver | K=1 always r76 short512. Slurm memory-step failure before ray-driver. |
| Qwen3-8B | 3137513 | FAILED | K=1 always r77 short512. Local-spec HF reconvert marker fired, but dense Qwen3Bridge local layernorm mappings were missing, causing conversion failure before rollout. |
| Qwen3-8B | 3137739 | FAILED | K=1 always r78 short512. Dense Qwen3Bridge mapping fix worked; vLLM acceptance recovered to 57.5-63.0% and logprobs completed, then training failed in ModelOpt EAGLE decoder TE RMSNorm. |
| Qwen3-8B | 3139283 | FAILED | K=1 always r79 short512. Recursive EAGLE TE norm replacement cleared the RMSNorm blocker; generation acceptance was 66.6% / 57.6% repeated and logprobs completed, then DDP grad sync failed on a no-grad bucket. |
| Qwen3-8B | 3139703 | FAILED | K=1 always r80 short512. DDP no-overlap passed the previous assertion and moved failure to helper signature mismatch. |
| Qwen3-8B | 3140266 | FAILED | K=1 always r81 short512. Slurm GRES parser failure before Ray driver. |
| Qwen3-8B | 3140576 | FAILED | K=1 always r82 short512. Ray Python/Ray version mismatch. |
| Qwen3-8B | 3140704 | CANCELLED | K=1 always r83 short512. Ray mismatch fixed and vLLM workers initialized through CUDA Graph capture; no rollout metric before cancellation. |
| Qwen3-32B | 3132637 | FAILED | Reached rollout setup, then policy-to-vLLM weight streaming failed with `KeyError: 'draft.fc.weight'`. Dependent matrix cancelled. |
| Qwen3-30B-A3B | 3130929 | FAILED | Failed before rollout in Qwen3-MoE weight streaming: vLLM fused-MoE loader rejected `shard_dim=0` for a 3D expert tensor. |

## Qwen3-32B Worker≈32 Early Signal

새 실험은 기존 `GBS=2048` 대신 `GBS=512`를 사용해서 16 generation workers 기준 worker당 response batch를 약 `128`에서 약 `32`로 낮춘 것입니다. Standalone `bs32`와 비교하기 위한 조건입니다. 아래 값은 아직 completed 20-step aggregate가 아니라 모든 job에서 공통으로 나온 Step 1-2 early metric입니다.

| Mode | Job | Step scope | Gen speedup | E2E speedup | Acceptance |
|---|---:|---|---:|---:|---:|
| K=1 always | 3136001 | Step 1-2 mean | 1.42x | 1.21x | 70.0% |
| K=3 always | 3136002 | Step 1-2 mean | 1.66x | 1.32x | 46.4% |

해석: per-worker batch를 standalone `bs32`에 가깝게 낮추자 K=3 generation speedup이 기존 completed `GBS=2048` always-on 결과의 1.36x에서 early 1.66x로 올라갔습니다. 다만 K=3 acceptance는 아직 46.4%로 standalone bs32의 67.1%보다 낮아서 standalone K=3 2.29x ceiling까지는 아직 차이가 있습니다.

추가 poll에서 K=3 `3136002`는 Step 9 early generation까지 진행했습니다. Baseline과 matched comparison이 가능한 Step 1-8 generation 평균은 1.64x, Step 1-7 E2E 평균은 1.31x, Step 1-8 acceptance 평균은 45.9%입니다. 즉 K=3 worker≈32 개선은 초기 한두 step만의 noise로 보이지 않습니다. Step 9는 K=3 metric만 있고 matched baseline은 아직 pending입니다.

## Root Cause Summary

| Area | Current read |
|---|---|
| Gated SpecDec | TLT-style gate is too restrictive for the current NeMo-RL workload. Scheduler enabled ratio is near zero for 8B/30B-A3B and only a few percent for 32B, so gated results mostly measure baseline decoding plus overhead. |
| E2E vs generation-only | vLLM standalone measures generation-only throughput. NeMo-RL E2E includes rollout, reward, policy logprob, reference logprob, and training update. Generation speedup does not directly convert to E2E speedup. |
| Acceptance rate | Acceptance is workload-dependent. Qwen3-32B standalone K=3 has 67.1% acceptance, but NeMo-RL fixed/offline K=3 is around 45.28% always and 23.65% gated. Qwen3-8B DAPOMath standalone K=3 is 36.62%. |
| Qwen3-8B online gated | The gated failure is still vLLM V1 SpecDec bookkeeping: scheduler output contains speculative-width scheduling while `_bookkeeping_sync` expects sampled token width 1. The r67 V9 zero-draft/active-batch patch did not fully resolve this. |
| Qwen3-8B online always | r84/r85 completed a matched online drafter training smoke. The path is no longer blocked by the earlier TE RMSNorm/DDP/Ray issues for this 1-step case, but K=1 always is slower than matched baseline: generation `0.605x`, E2E `0.914x`, acceptance `40.80%`. r86/r87 greedy/fixed512 also completed negative: generation `0.757x`, acceptance `41.82%`. |
| Qwen3-8B online paired baseline | r85 `3142137` is the matched baseline: same worker≈32/fixed512 workload and `policy.draft.enabled=true`, but vLLM `speculative_config` disabled. It emitted generation `5302.53` tok/s/GPU and E2E `375.78` tok/s/GPU. |
| Qwen3-8B r68 gated patch | Added scheduler-output scrub before `SchedulerOutput(...)`: when the active-batch gate disables speculation, clear `scheduled_spec_decode_tokens` and clamp `num_scheduled_tokens` to 1 per request. This targets the exact r67 dump where runner gate was disabled but scheduler output still had speculative width. Smoke job `3133925` failed before rollout due to disk quota, so the patch is not yet validated. |
| Online draft weight streaming | Qwen3-32B online shows policy side streams `draft.*` weights, but generation worker `state_dict_info` misses `draft.fc.weight`. Likely fix area is how policy worker refit metadata is aggregated and sent to vLLM workers. |
| Qwen3-30B-A3B MoE | Online policy-to-vLLM streaming hits a Qwen3-MoE loader dimensionality mismatch for 3D expert tensors. This is separate from the 8B vLLM bookkeeping issue. |

## Code Evidence

| Claim | Code evidence |
|---|---|
| SpecDec throughput runs must not request vLLM per-token generation logprobs. | `vllm_worker.py::_build_sampling_params()` detects `speculative_config`; unless explicitly forced, it sets `SamplingParams.logprobs=None`, and it raises if `NRL_VLLM_SPECDEC_REQUEST_LOGPROBS=1` is used without the explicit allow flag because vLLM V1 disables SpecDec for logprob requests. |
| Qwen3-8B logprob-skip numbers are diagnostic, not default GRPO. | `grpo.py::_populate_policy_and_reference_logprobs()` skips policy/reference logprob fprop only under the explicit safe diagnostic conditions: `NRL_GRPO_SKIP_POLICY_LOGPROBS_IF_SAFE=true`, `force_on_policy_ratio=true`, KL penalty `0`, and importance-sampling paths disabled. |
| Gated results are mostly baseline-like because the gate really disables lookahead/proposal at high load. | `specdec_runtime_gate_patch.py` computes active requests and scheduled tokens, disables drafter proposal, zeroes draft ids, and scrubs stale `scheduled_spec_decode_tokens` when thresholds are exceeded. The observed near-zero enabled ratios are therefore expected for the current large rollout batches. |
| Online r83 is the correct restart point if we resume online drafter training. | `submit_qwen8_r83...raymatch_retry.sh` includes BF16/KV-auto via the online submitter path, local transformer spec, recursive EAGLE torch RMSNorm, DDP no-overlap, Ray/Python version alignment, `max_total_sequence_length=2560`, and `max_new_tokens=512`; the job reached vLLM CUDA Graph capture before cancellation. |

## Next Actions

1. For Qwen3-8B online drafter, do not report K=1 always as a generation speedup path under the current worker≈32/fixed512 condition; both default-style r84 and greedy r87 are generation slowdowns.
2. If more Qwen3-8B online diagnosis is needed, focus on drafter quality/domain mismatch or online vLLM drafter/verification overhead rather than sampling mode; greedy fixed512 did not recover acceptance.
3. Clear/reduce the remote actor-venv/build quota pressure and resubmit the Qwen3-8B r68 gated scrub smoke. This is required before judging whether the scheduler-output scrub fixes the vLLM V1 bookkeeping assertion.
4. Fix Qwen3-32B online `draft.fc.weight` metadata mismatch. The likely code path is `LMPolicy.prepare_refit_info()` returning only one worker's metadata instead of a merged draft-aware metadata set.
5. Re-generate `docs/specdec_completed_eval_bar_graphs.html` if a new online-drafter performance metric is emitted. The r67-r83 online updates so far are failure/status updates, not chartable throughput results.

## Latest Submissions, 20:02 PDT

| Job | Purpose | Status |
|---:|---|---|
| `3135990` | Qwen3-8B online drafter K=1 always short512 smoke, to isolate long-sequence TE RMSNorm failure | FAILED before Ray startup due Slurm ray-head memory-step creation failure |
| `3136275` | Qwen3-8B online drafter K=1 always short512 low-CPU retry | FAILED after generation; vLLM internal acceptance 57.6-66.6%, then policy-side TE RMSNorm failure |
| `3136718` | Qwen3-8B online drafter K=1 always short512 local-transformer-spec smoke | FAILED after generation; original TE RMSNorm failure avoided, new missing `scaled_masked_softmax_cuda` failure |
| `3137127` | Qwen3-8B online drafter K=1 always short512 local-transformer no-fused-softmax smoke | FAILED; `get_logprobs()` passed, then online EAGLE draft `enorm` TE RMSNorm failed |
| `3137513` | Qwen3-8B online drafter local-spec HF reconvert + draft torch RMSNorm retry | FAILED; dense Qwen3Bridge lacked local layernorm mappings |
| `3137739` | Qwen3-8B online drafter dense Qwen3Bridge mapping fix retry | FAILED; conversion/generation/logprobs passed and acceptance recovered to 57.5-63.0%, then EAGLE decoder `input_layernorm` TE RMSNorm failed |
| `3139283` | Qwen3-8B online drafter recursive EAGLE TE norm replacement retry | FAILED; replacement marker fired, generation acceptance 66.6% / 57.6% repeated, logprobs completed, then DDP grad sync assertion on a `0/3` no-grad bucket |
| `3139703` | Qwen3-8B online drafter r80 no-overlap retry | FAILED; DDP no-overlap avoided the r79 grad-sync assertion, then failed at model-parallel helper signature mismatch |
| `3140266` | Qwen3-8B online drafter r81 helper-compat retry | FAILED before model path; Slurm GRES parser issue |
| `3140576` | Qwen3-8B online drafter r82 GRES parser retry | FAILED; Ray head/driver Python/Ray version mismatch |
| `3140704` | Qwen3-8B online drafter r83 Ray-match retry | CANCELLED by request; Ray init, MasterConfig, vLLM init, and CUDA Graph capture succeeded, but no rollout metric emitted |
| `3141902` | Qwen3-8B online drafter r84 K=1 always, worker≈32, fixed512 | COMPLETED; generation `3206.21` tok/s/GPU, E2E `343.62` tok/s/GPU, acceptance `40.80%` |
| `3142137` | Qwen3-8B online drafter r85 paired baseline, worker≈32, fixed512, vLLM SpecDec disabled | COMPLETED; generation `5302.53` tok/s/GPU, E2E `375.78` tok/s/GPU |
| `3142745` | Qwen3-8B online drafter r86 greedy paired baseline, worker≈32, fixed512, vLLM SpecDec disabled | COMPLETED; generation `5364.60` tok/s/GPU, E2E `357.13` tok/s/GPU |
| `3142746` | Qwen3-8B online drafter r87 greedy K=1 always, worker≈32, fixed512 | COMPLETED; generation `4059.60` tok/s/GPU (`0.757x`), E2E `380.83` tok/s/GPU (`1.066x`), acceptance `41.82%` |
| `3136000` | Qwen3-32B NeMo-RL baseline with `GBS=512`, about 32 responses/worker | RUNNING; Step 1-2 baseline metrics emitted, Step 3 early generation emitted |
| `3136001` | Qwen3-32B NeMo-RL K=1 always with `GBS=512`, gate disabled | RUNNING; Step 1-2 mean gen/E2E speedup `1.42x`/`1.21x`, acceptance `70.0%`; Step 3 early generation also emitted |
| `3136002` | Qwen3-32B NeMo-RL K=3 always with `GBS=512`, gate disabled | RUNNING; Step 1-2 mean gen/E2E speedup `1.66x`/`1.32x`, acceptance `46.4%` |

The Qwen3-32B `GBS=512` jobs are intended to compare against standalone `bs32`
more directly than the previous `GBS=2048` runs, which mapped to about 128
responses per generation worker.
