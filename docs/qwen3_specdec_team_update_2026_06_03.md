# Qwen3 SpecDec 평가 공유 메모

Updated: 2026-06-04 03:15 PDT

## 요약

현재까지의 결론은 아래와 같습니다.

1. 이번 이슈는 KV cache quantization이나 FP8 generation 문제가 아닙니다. NeMo-RL 쪽 generation은 BF16, KV cache는 `auto` 기준으로 맞춰서 돌렸고, vLLM standalone 비교도 같은 축으로 해석하고 있습니다.
2. vLLM standalone에서는 Qwen3-32B public EAGLE3 drafter가 강한 ceiling을 보입니다. 반면 NeMo-RL gated SpecDec는 scheduler gate가 거의 켜지지 않아서 대부분 baseline decoding처럼 동작합니다.
3. NeMo-RL always-on SpecDec는 generation 구간에서는 개선이 보이지만, E2E는 policy/reference logprob forward, reward/training 구간 때문에 standalone speedup만큼 올라가지 않습니다.
4. 새 Qwen3-8B logprob-skip diagnostic Step 1-4에서는 K=1 always가 generation 1.290x / E2E 1.152x, K=3 always가 generation 1.343x / E2E 1.175x입니다. 이 실험에서 generation execution 비중은 약 58%로 역산됩니다. 다만 이 실험은 `force_on_policy_ratio=true`, KL penalty 0, policy/reference logprob skip을 켠 changed-objective diagnostic이므로 default GRPO E2E 결과로 해석하면 안 됩니다.
5. Qwen3-8B online drafter training은 이제 r84/r85, r86/r87, post-patch r88/r89에서 end-to-end metric까지 완료됐습니다. 하지만 public HF drafter K=1은 현재 worker≈32/fixed512 online workload에서 generation speedup이 없습니다. r84 default-style은 generation `0.605x`, E2E `0.914x`, acceptance `40.80%`; r87 greedy/fixed512는 generation `0.757x`, E2E `1.066x`, acceptance `41.82%`; post-patch r89는 generation `0.732x`, E2E `1.029x`, acceptance `42.01%`입니다.
6. r88/r89는 actor venv rebuild를 켜서 patched `vllm_worker.py`를 실제 generation worker에 반영한 재실험입니다. 결과가 r87과 거의 같으므로 fixed-length `SamplingParams` propagation 누락은 Qwen3-8B online slowdown의 root cause가 아닙니다.
7. 다만 r86-r89의 greedy run은 online draft training 결과로는 invalid입니다. `temperature=0.0`에서 `train.py::apply_temperature_scaling()`이 training logits를 0으로 나누어 `Loss/Draft Loss/Generation KL Error`를 NaN으로 만들었습니다. 이 코드는 패치했고, r90/r91 Step 1에서 finite draft loss를 확인했습니다.
8. r92/r93 clean rerun은 5/5 step 완료됐습니다. Online draft update는 acceptance를 Step 1 `42.24%`에서 Step 2 `70.42%`, Step 5 `71.70%`까지 올렸습니다. 하지만 5-step aggregate는 generation `0.964x`, E2E `0.980x`라 standalone-like speedup에는 아직 못 갔습니다.
9. r94/r95/r96 20-step rerun도 완료됐습니다. K=1 always는 generation `1.093x`, E2E `1.029x`, mean acceptance `61.20%`로 generation slice에서는 회복됐습니다. K=3 always는 generation `0.968x`, E2E `0.995x`, mean acceptance `37.46%`로 아직 손해입니다.
10. r97/r98/r99 engine-shaped 20-step rerun도 완료됐습니다. `max_num_seqs=32`, `max_num_batched_tokens=64000`, metrics logger off, no chunked prefill 등 standalone-shaped vLLM engine knobs를 넣었지만 K=1은 generation/E2E `1.120x/1.060x`, mean acceptance `61.75%`입니다. r22 exact-engine K=1 `1.299x`까지는 못 갔고, K=3는 generation/E2E `1.009x/1.041x`, mean acceptance `37.40%`라 acceptance 문제는 그대로입니다.

## 2026-06-04 03:15 PDT 추가 결과

r97/r98/r99 engine-shaped finite-loss + gradfix 20-step paired run이 완료됐습니다.
세 run 모두 `policy.draft.enabled=true`, BF16 vLLM generation, KV `auto`, fixed
512-token greedy decode, 4 generation workers, generation batch size 32, GBS 128
조건이고, 추가로 아래 standalone-shaped engine knobs를 켰습니다:

- `policy.generation.vllm_cfg.gpu_memory_utilization=0.82`
- `policy.generation.vllm_cfg.enable_vllm_metrics_logger=false`
- `policy.generation.vllm_kwargs.max_num_seqs=32`
- `policy.generation.vllm_kwargs.max_num_batched_tokens=64000`
- `policy.generation.vllm_kwargs.enable_chunked_prefill=false`
- `policy.generation.vllm_kwargs.disable_custom_all_reduce=true`

| Config | Job | Gen throughput speedup | Gen step-time speedup | E2E throughput speedup | E2E step-time speedup | Mean acceptance |
|---|---:|---:|---:|---:|---:|---:|
| K=1 always | 3146945 | 1.120x | 1.120x | 1.060x | 1.060x | 61.75% |
| K=3 always | 3146946 | 1.009x | 1.009x | 1.041x | 1.041x | 37.40% |

해석: engine shape는 K=1 generation speedup을 r95의 `1.093x`에서 r98의
`1.120x`로 조금 올렸지만, r22 exact-engine/no-gate K=1 `1.299x`까지는 못
갑니다. K=1 acceptance도 `61.20% -> 61.75%`로 거의 동일합니다. 따라서 남은
gap의 대부분은 engine knob가 아니라 online path/integration overhead와
DAPOMath real-prompt drafter fit 문제로 보는 것이 맞습니다. K=3는 20-step
aggregate에서 `1.009x`로 간신히 1.0을 넘지만 acceptance가 `37.40%`라 안정적인
추천 경로는 아닙니다.

새 그래프/원본:

- `docs/qwen3_8b_engine_shape_r97_r99_20step.png`
- `docs/qwen3_8b_engine_shape_r97_r99_20step_summary.csv`
- `docs/qwen3_8b_engine_shape_r97_r99_20step_metrics.csv`
- `scripts/plot_qwen3_8b_engine_shape_r97_r99_20step.py`

아래의 오래된 online 실패 히스토리는 r84/r85와 r86/r87 완료 결과로 supersede됩니다. 실패 히스토리는 어떤 blocker를 제거했는지 설명하는 용도이고, 현재 Qwen3-8B online 성능 결론은 completed paired runs를 기준으로 봐야 합니다.

## 2026-06-04 02:21 PDT 추가 결과

r94/r95/r96 finite-loss + gradfix 20-step paired run이 완료됐습니다. 세 run 모두
`policy.draft.enabled=true`, BF16 vLLM generation, KV `auto`, fixed 512-token
greedy decode, 4 generation workers, generation batch size 32, GBS 128 조건입니다.

| Config | Job | Gen throughput speedup | Gen step-time speedup | E2E throughput speedup | E2E step-time speedup | Mean acceptance |
|---|---:|---:|---:|---:|---:|---:|
| K=1 always | 3145920 | 1.093x | 1.093x | 1.029x | 1.029x | 61.20% |
| K=3 always | 3145921 | 0.968x | 0.968x | 0.995x | 0.995x | 37.46% |

해석: K=1은 online drafter update가 충분히 들어간 뒤 generation 구간에서
실제로 speedup을 냅니다. 다만 baseline generation time share가 `6.83%`에
불과해서 E2E는 `1.029x`로 제한됩니다. K=3은 acceptance가 평균 `37.46%`라
verification/drafter overhead를 이기지 못했고, generation도 `0.968x`로
slowdown입니다.

새 그래프/원본:

- `docs/qwen3_8b_online_gradfix_r94_r96_20step.png`
- `docs/qwen3_8b_online_gradfix_r94_r96_20step_summary.csv`
- `docs/qwen3_8b_online_gradfix_r94_r96_20step_metrics.csv`

## 2026-06-04 01:30 PDT 추가 결과

r92/r93 finite-loss + gradfix paired run이 완료됐습니다.

| Step | r92 baseline gen tok/s/GPU | r93 K=1 gen tok/s/GPU | Gen speedup | E2E speedup | r93 acceptance | r93 draft loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5247.03 | 3990.15 | 0.760x | 1.020x | 42.24% | 9.0968 |
| 2 | 6557.43 | 7281.81 | 1.110x | 0.989x | 70.42% | 7.0224 |
| 3 | 6556.19 | 6284.84 | 0.959x | 0.935x | 56.98% | 7.9329 |
| 4 | 6727.15 | 6386.33 | 0.949x | 0.964x | 48.91% | 8.2979 |
| 5 | 6256.15 | 7616.58 | 1.217x | 0.985x | 71.70% | 7.1756 |

Aggregate: generation `0.964x`, E2E `0.980x`, mean acceptance `58.05%`.
Step 2-5만 보면 generation `1.049x`, E2E `0.968x`, mean acceptance `62.00%`입니다.

해석: online drafter training path는 이제 정상적으로 돌아갑니다. Acceptance도 Step 2/5에서 70%대로 올라갑니다. 하지만 throughput은 아직 안정적으로 좋아지지 않았습니다. 남은 gap은 broken training path가 아니라, post-update acceptance/throughput instability와 NeMo-RL full GRPO step의 낮은 generation 비중 때문으로 봐야 합니다.

새 그래프/원본:

- `docs/qwen3_8b_online_gradfix_r92_r93.png`
- `docs/qwen3_8b_online_gradfix_r92_r93_metrics.csv`

후속 20-step 안정화 실험도 이후 완료됐습니다.

| Job | Mode | Status |
|---:|---|---|
| 3145919 | r94 baseline, finite-loss + gradfix, 20 steps | COMPLETED |
| 3145920 | r95 K=1 always, finite-loss + gradfix, 20 steps | COMPLETED, generation/E2E `1.093x/1.029x` |
| 3145921 | r96 K=3 always, finite-loss + gradfix, 20 steps | COMPLETED, generation/E2E `0.968x/0.995x` |

## 2026-06-04 00:58 PDT 추가 결과

r90/r91은 `temperature=0.0` greedy training NaN fix가 실제로 작동하는지 확인하는 5-step 재실험이었습니다. Step 1 metric은 나왔지만, Step 2로 넘어가기 전 `offload_before_refit`에서 실패했습니다.

| Job | Mode | Status | Step-1 Gen tok/s/GPU | Step-1 Speedup | Acceptance | Draft Loss |
|---:|---|---|---:|---:|---:|---:|
| 3144473 | r90 greedy baseline, finite-loss patch | FAILED before Step 2 | 4920.99 | 1.000x | n/a | 9.0362 |
| 3144474 | r91 greedy K=1 always, finite-loss patch | FAILED before Step 2 | 3910.24 | 0.794x | 42.24% | 9.0676 |

해석: `draft_loss=nan` 문제는 해결됐습니다. 하지만 K=1 Step 1 generation은 여전히 baseline보다 느리고 acceptance도 약 `42%`라서 성능 회복 신호는 아직 없습니다. multi-step online draft update 효과는 offload bug 때문에 아직 측정되지 않았습니다.

새로 확인한 blocker는 `nemo_rl/models/policy/workers/megatron_policy_worker.py`의 grad offload fallback입니다. `move_params=false`로 parameter는 CUDA에 둔 채 grad만 CPU로 옮길 때 `param.grad`까지 CPU tensor로 재할당해서 PyTorch 오류가 났습니다. 이 경우 `param.grad=None`으로 clear하도록 패치했고, remote `py_compile`을 통과했습니다.

패치 후 clean rerun:

| Job | Mode | Status |
|---:|---|---|
| 3144992 | r92 greedy baseline, finite-loss + gradfix, 5 steps | COMPLETED later, 00:24:01 |
| 3144993 | r93 greedy K=1 always, finite-loss + gradfix, 5 steps | COMPLETED later, 00:25:47 |

## 2026-06-04 00:31 PDT 추가 결과

post-`SamplingParams` patch 검증용 Qwen3-8B online greedy/fixed512 matched pair가 완료됐습니다. actor venv rebuild를 켜서 patched source를 실제 worker가 사용하도록 했고, r89 vLLM engine 로그에서 EAGLE3 `SpeculativeConfig`가 활성화된 것도 확인했습니다.

| Job | Mode | Status | Gen tok/s/GPU | Gen speedup | E2E tok/s/GPU | E2E speedup | Acceptance |
|---:|---|---|---:|---:|---:|---:|---:|
| 3143675 | r88 greedy paired baseline, vLLM SpecDec off | COMPLETED, 00:27:13 | 5471.97 | 1.000x | 363.54 | 1.000x | n/a |
| 3143676 | r89 greedy K=1 always, EAGLE3 on | COMPLETED, 00:26:55 | 4005.30 | 0.732x | 374.12 | 1.029x | 42.01% |

해석: post-patch에서도 r89 acceptance는 `42.01%`로 r87의 `41.82%`와 거의 같고, generation throughput은 baseline 대비 `0.732x`로 더 느립니다. 따라서 worker-side `min_tokens` / `ignore_eos` / stop-disable 전파 누락이 아니라, public Qwen3-8B drafter와 DAPOMath/online prompt distribution mismatch 및 online drafter/verification overhead가 주된 원인으로 보입니다. E2E `1.029x`는 generation speedup이 아니라 one-step non-generation timing variance로 봐야 합니다.

## 2026-06-04 00:36 PDT 추가 코드 분석 및 제출

greedy online runs의 `draft_loss=nan` 원인을 확인했습니다.

| Item | 내용 |
|---|---|
| Root cause | `nemo_rl/models/megatron/train.py::apply_temperature_scaling()`이 `sampling_params.temperature != 1.0`이면 logits를 temperature로 나눴습니다. greedy run의 `temperature=0.0`에서 training logits가 Inf/NaN이 됩니다. |
| Evidence | r86/r87/r88/r89 로그는 `Loss: nan`, `Draft Loss: nan`, `Generation KL Error: nan`, W&B `train/grad_norm=0`입니다. 반면 temperature 1.0인 r84/r85는 draft loss가 finite입니다: `9.0157` / `8.8250`. |
| Patch | `sampling_params is None` 또는 `temperature <= 0.0`이면 scaling을 skip하도록 remote `train.py`를 수정했고, `python3 -m py_compile`을 통과했습니다. |
| New jobs | r90 baseline `3144473`, r91 K=1 always `3144474`; 둘 다 patched finite-loss path, worker≈32 fixed512 greedy, `MAX_STEPS=5`, fresh actor suffix + actor venv rebuild입니다. |

이 재실험의 목표는 두 가지입니다. 첫째, greedy 설정에서도 `draft_loss`가 finite로 돌아오는지 확인합니다. 둘째, Step 2-5 acceptance가 Step 1의 약 42%에서 개선되는지 봅니다.

## 2026-06-03 23:46 PDT 추가 결과

Qwen3-8B online-drafter greedy/fixed512 matched follow-up r86/r87가 완료됐습니다. 목적은 r84의 낮은 acceptance가 stochastic/default-GRPO sampling 때문인지 확인하는 것이었고, 결과는 “아니다”입니다.

| Job | Mode | Status | Gen tok/s/GPU | Gen speedup | E2E tok/s/GPU | E2E speedup | Acceptance |
|---:|---|---|---:|---:|---:|---:|---:|
| 3142745 | r86 greedy paired baseline, vLLM SpecDec off | COMPLETED, 00:21:29 | 5364.60 | 1.000x | 357.13 | 1.000x | n/a |
| 3142746 | r87 greedy K=1 always, EAGLE3 on | COMPLETED, 00:19:38 | 4059.60 | 0.757x | 380.83 | 1.066x | 41.82% |

해석: r87 acceptance `41.82%`는 r84의 `40.80%`와 거의 같고, generation 자체는 baseline보다 느립니다. 따라서 Qwen3-8B online K=1 slowdown의 주원인은 sampling mode가 아니라 public drafter/domain mismatch와 online drafter/verification overhead입니다. E2E `1.066x`는 generation speedup이 아니라 one-step non-generation timing variance로 봐야 합니다. baseline generation 비중은 `6.66%`뿐입니다.

추가로 원격 online checkout의 `nemo_rl/models/generation/vllm/vllm_worker.py`에 fixed-length SamplingParams env patch를 반영했습니다. `NRL_VLLM_GENERATION_MIN_TOKENS`, `NRL_VLLM_GENERATION_IGNORE_EOS`, `NRL_VLLM_GENERATION_DISABLE_STOP_STRINGS`, `NRL_VLLM_GENERATION_DISABLE_STOP_TOKEN_IDS`가 worker 내부에서 직접 적용됩니다. local/remote `python3 -m py_compile` 모두 통과했습니다.

## 2026-06-03 22:20 PDT 추가 결과

이 섹션이 현재 최신 상태입니다. 아래의 오래된 poll 섹션에서 `running`으로 남아 있는 job은 해당 poll 당시 상태이며, 최신 결론은 다음 표를 기준으로 봐야 합니다.

| Job | Mode | Status | Key result |
|---:|---|---|---|
| 3139703 | Qwen3-8B online K=1 always r80 short512 no-overlap retry | FAILED, 00:35:10 | DDP overlap을 끄면서 r79의 grad-sync assertion은 사라졌지만, `logical_and_across_model_parallel_group(..., mp_group=...)` helper signature mismatch에서 실패했습니다. |
| 3140266 | Qwen3-8B online K=1 always r81 helper-compat retry | FAILED, 00:02:03 | Slurm GRES `gpu:4(S:0-1)` parser issue로 Ray driver 전에 실패했습니다. |
| 3140576 | Qwen3-8B online K=1 always r82 GRES parser retry | FAILED, 00:08:09 | GRES parser는 고쳤지만 Ray head/driver Python/Ray version mismatch로 실패했습니다. |
| 3140704 | Qwen3-8B online K=1 always r83 Ray-match retry | CANCELLED by request, 00:11:31 | Ray version mismatch는 해결됐고 Ray init, MasterConfig, data loaders, vLLM worker init, CUDA Graph capture까지 성공했습니다. 다만 rollout metric/online acceptance aggregate 전에 취소됐습니다. |
| Qwen3-235B decode-heavy sweep | vLLM standalone, ISL=1000, OSL=10000, TP=4, K=1 | COMPLETED | Acceptance가 85-98%인데도 bs1-bs32 throughput speedup은 0.608-0.823x로 모두 slowdown입니다. High acceptance만으로는 speedup이 보장되지 않는다는 보강 evidence입니다. |

## 2026-06-03 17:55 PDT 추가 결과

Qwen3-8B online r77 `3137513`은 `FAILED`, elapsed 00:17:41, ExitCode 1:0입니다. 이 run은 r76과 같은 local-spec reconvert/draft torch RMSNorm 설정이며, `nvl72083-T05`를 제외하고 제출했습니다.

중요한 점은 r77에서 local-spec import patch가 실제로 실행됐다는 것입니다. 로그는 모든 policy rank에서 `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true: using local GPT decoder block spec during HF->Megatron import`를 출력했습니다. 하지만 dense `Qwen3Bridge`의 mapping table에는 TE fused layernorm mapping만 있고 local layernorm mapping이 없었습니다. 그 결과 `decoder.layers.*.input_layernorm.weight`와 `decoder.layers.*.pre_mlp_layernorm.weight`에 대해 `No mapping found`가 반복됐고, `load_weights_hf_to_megatron()`에서 `AttributeError: 'NoneType' object has no attribute 'megatron_module'`로 실패했습니다.

수정: remote dense `qwen3_bridge.py`에 이미 `qwen3_moe_bridge.py`와 `llama_bridge.py`에 존재하는 local layernorm mappings를 추가했습니다.

| Added Megatron param | HF param |
|---|---|
| `decoder.layers.*.input_layernorm.weight` | `model.layers.*.input_layernorm.weight` |
| `decoder.layers.*.pre_mlp_layernorm.weight` | `model.layers.*.post_attention_layernorm.weight` |

remote `py_compile`은 통과했습니다. 후속 r78 `3137739`는 conversion, generation, logprobs까지 통과했지만 online EAGLE decoder 내부 TE RMSNorm에서 실패했습니다. r79 `3139283`는 같은 r78 config에 fresh actor venv suffix를 사용하고, EAGLE module 내부 TransformerEngine LayerNorm/RMSNorm child modules를 torch RMSNorm으로 재귀 교체하는 patch를 추가했습니다. 이 patch는 성공적으로 검증됐고, 현재 남은 blocker는 Megatron DDP overlap grad-reduce의 no-grad bucket assertion입니다.

## 2026-06-03 17:25 PDT 추가 결과

Qwen3-8B online r75 `3137127`은 `FAILED`, elapsed 00:23:38, ExitCode 1:0입니다. r75는 r74의 local transformer spec에 더해 `model_cfg.masked_softmax_fusion=False`를 적용했습니다.

중요한 변화는 `get_logprobs()`가 통과했다는 점입니다. 즉 r74의 missing `scaled_masked_softmax_cuda` blocker는 제거됐습니다. 하지만 그 다음 `Training policy` 단계에서 online EAGLE draft model forward가 실패했습니다.

| Job | Mode | Status | What changed | Result |
|---:|---|---|---|---|
| 3137127 | Qwen3-8B online K=1 always r75 short512 | FAILED, 00:23:38 | local transformer spec + `masked_softmax_fusion=False` | `get_logprobs()` completed, then `nemo_rl/models/megatron/train.py` called `draft_model(...)`, ModelOpt `EagleModule` called `self.enorm(embeddings)`, and TransformerEngine RMSNorm failed at `rmsnorm_fwd_cuda_kernel.cu:49` |
| 3137466 | Qwen3-8B online K=1 always r76 short512 | FAILED before driver, 00:00:35 | local spec is passed into HF->Megatron import, checkpoint is reconverted in a separate local-spec cache, and EAGLE `enorm/hnorm` are replaced with torch RMSNorm under `NRL_FORCE_DRAFT_TORCH_RMSNORM=true` | Slurm ray-head `srun` failed on `nvl72083-T05` with `Memory required by task is not available`; no model metric |
| 3137513 | Qwen3-8B online K=1 always r77 short512 | FAILED, 00:17:41 | same code/config as r76, excluding `nvl72083-T05` | local-spec import marker fired; failed in dense Qwen3Bridge missing local layernorm mappings |
| 3137739 | Qwen3-8B online K=1 always r78 short512 | FAILED, 00:52:59 | r77 plus dense Qwen3Bridge local layernorm mapping fix | conversion/generation/logprobs completed; acceptance recovered to 57.5-63.0%; failed in ModelOpt EAGLE decoder `input_layernorm` TE RMSNorm |
| 3139283 | Qwen3-8B online K=1 always r79 short512 | FAILED, 00:28:40 | r78 plus recursive EAGLE TE norm replacement | replacement marker fired for EAGLE decoder norms; generation completed with 66.6% / 57.6% repeated Avg Draft acceptance and logprobs completed; training failed in DDP `finish_grad_sync()` with `0/3 params have grad available` |
| 3139703 | Qwen3-8B online K=1 always r80 short512 | FAILED, 00:35:10 | r79 plus `overlap_grad_reduce=false` and `overlap_param_gather=false` | DDP async-overlap blocker was avoided; next failure was model-parallel helper signature mismatch |
| 3140266 | Qwen3-8B online K=1 always r81 short512 | FAILED, 00:02:03 | helper-compat retry | Slurm GRES parser failure before Ray driver |
| 3140576 | Qwen3-8B online K=1 always r82 short512 | FAILED, 00:08:09 | GRES parser retry | Ray head/driver Python/Ray version mismatch |
| 3140704 | Qwen3-8B online K=1 always r83 short512 | CANCELLED, 00:11:31 | Ray-match retry | Ray/vLLM initialized through CUDA Graph capture; no rollout metric before cancellation |

r75 also showed vLLM internal SpecDec acceptance of 0.0%. This is not a drafter-quality conclusion. r78 recovered acceptance to 57.5-63.0% after local-spec HF reconvert and dense Qwen3Bridge mapping fix, so the r74/r75 0.0% signal was caused by checkpoint/spec mismatch rather than a broken public drafter.

## 2026-06-03 16:25 PDT 추가 결과

Qwen3-8B online short512 K=1 always smoke `3136275`가 `FAILED`, elapsed 00:23:16, ExitCode 1:0으로 종료됐습니다. 이 run은 `policy.draft.enabled=true`, `policy.draft.model_name=RedHatAI/Qwen3-8B-speculator.eagle3`, BF16 generation, KV `auto`, `max_new_tokens=512`, `max_total_sequence_length=2560`이 정상 반영됐습니다.

중요한 점은 failure가 generation 전에 난 것이 아니라는 점입니다. vLLM worker는 SpecDec internal metrics를 먼저 냈고, acceptance는 한 worker에서 57.6%, repeated worker 로그에서 66.6%였습니다. NeMo-RL aggregate rollout metric은 나오기 전에 post-generation policy logprob forward로 넘어가며 실패했습니다.

| Job | Mode | Status | Evidence | Root cause |
|---:|---|---|---|---|
| 3136275 | Qwen3-8B online K=1 always, short512, 1 step | FAILED, 00:23:16 | vLLM SpecDec internal acceptance 57.6-66.6%; no aggregate rollout metric | Megatron policy forward enters `TELayerNormColumnParallelLinear` and fails in TransformerEngine `rmsnorm_fwd_cuda_kernel.cu:49` with CUDA invalid argument |

해석: 같은 TE RMSNorm failure가 long 8192뿐 아니라 short512에서도 재현됐으므로, 단순히 max sequence length가 길어서 생긴 문제는 아닙니다. online worktree의 Megatron-Bridge Qwen provider는 `transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)`로 잡혀 있고, 현재 환경에는 TE가 있으므로 policy-side forward가 fused TE layernorm-linear path를 탑니다. 다음 smoke는 env-guarded patch로 Qwen provider를 local Megatron layer spec으로 강제해서, fused TE RMSNorm을 피하면 logprob forward가 통과하는지 검증해야 합니다.

r74 smoke `3136718`을 제출했습니다. 원격 `nemo_rl/models/megatron/setup.py`에 `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true`일 때만 `model_cfg.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=False)`로 바꾸는 guard patch를 넣었고, 기본 run은 그대로 둡니다. `3136718`은 r73과 같은 short512/K=1/always 조건이며, 첫 poll 상태는 `PENDING`입니다.

추가 결과: r74 `3136718`은 `FAILED`, 00:20:16입니다. Local-spec marker는 모든 MegatronPolicyWorker rank에서 확인됐고 기존 `rmsnorm_fwd_cuda_kernel` failure는 재현되지 않았습니다. 대신 local Megatron attention path의 `fused_softmax.py`가 `scaled_masked_softmax_cuda`를 import하다가 `ModuleNotFoundError`로 실패했습니다. 따라서 r75 `3137127`을 제출했습니다. r75는 r74와 동일하되 local-spec guard 안에서 `model_cfg.masked_softmax_fusion=False`도 설정합니다.

12:25 PDT 기준 Qwen3-8B online V6 smoke `3133314`는 `FAILED`, elapsed 00:13:41입니다. rollout generation에는 들어갔지만 `SpecDec early` metric 전에 다시 `assert sampled_token_ids.shape[-1] == 1`가 발생했습니다. 따라서 V6 scheduler/runner gate alignment만으로는 online Qwen3-8B의 vLLM V1 bookkeeping mismatch를 해결하지 못했습니다. 12:34 PDT에는 runner active-batch pressure와 zero-draft disabled path를 추가한 V9 runtime patch를 적용하고, gated retry `3133619/3133621/3133623/3133625` 및 always-on online `3133620/3133622/3133624`를 제출했습니다. 13:04 PDT 기준 K=1 gated V9 smoke `3133619`는 같은 vLLM V1 bookkeeping assertion으로 `FAILED`이고, K=1/K=2/K=3 always `3133620/3133622/3133624`는 policy-side TransformerEngine RMSNorm CUDA invalid-argument로 `FAILED`입니다. r67 dump에서 runner gate는 disabled였지만 scheduler output에는 `scheduled_spec_decode_tokens`와 `num_scheduled_tokens=2`가 남아 있었으므로 scheduler-output scrub patch를 추가하고 r68 smoke `3133925`를 제출했습니다. `3133925`는 rollout 전 actor venv build 중 `Disk quota exceeded`로 `FAILED`라서 scrub patch validation은 아직 못 했습니다.

## 주요 산출물

| Artifact | 내용 |
|---|---|
| `docs/qwen3_specdec_team_share_2026_06_03.md` | 팀 공유용 최신 요약 |
| `docs/specdec_completed_eval_bar_graphs.html` | 결과 dashboard, PNG chart 포함 |
| `docs/specdec_followup_latest_generation_bars.png` | fixed/offline drafter follow-up generation throughput bar chart, 512 dpi |
| `docs/specdec_followup_latest_acceptance_by_model.png` | model별 acceptance chart, 512 dpi |
| `docs/qwen3_8b_nemorl_specdec_root_cause.md` | code-level root-cause와 diagnostic 상태 |
| `docs/qwen3_8b_nemorl_vs_vllm_root_cause_matrix.md` | Qwen3-8B NeMo-RL vs vLLM standalone root-cause matrix |
| `docs/qwen3_235b_vllm_decodeheavy10k_speedup_acceptance.png` | Qwen3-235B OSL=10000 standalone speedup/acceptance chart |
| `docs/specdec_completed_qwen3_32b_vllm_standalone.png` | Qwen3-32B standalone speedup/acceptance |
| `docs/specdec_qwen3_8b_dapo_vllm_standalone.png` | Qwen3-8B DAPOMath real-prompt standalone isolation |
| `docs/specdec_qwen3_8b_logprob_skip_diag_step1_step4.png` | Qwen3-8B logprob-skip diagnostic Step 1-4 generation/E2E/acceptance, 512 dpi |

## 2026-06-03 13:35 PDT 추가 결과

Qwen3-8B logprob-skip diagnostic `3132444/3132635/3132636`은 모두 completed 상태이고, 같은 Step 1-4에서 E2E summary까지 확인됐습니다. 이 값은 default GRPO 결과가 아니라 changed-objective diagnostic입니다.

| Mode | Job | Status | Mean gen tok/s/GPU | Gen speedup | Mean E2E tok/s/GPU | E2E speedup | Acceptance | Inferred generation share |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline r5 | 3132444 | COMPLETED, 01:53:34 | 4260.76 | 1.000x | 2608.80 | 1.000x | n/a | n/a |
| K=1 always r6 | 3132635 | COMPLETED, 01:39:09 | 5498.34 | 1.290x | 3004.60 | 1.152x | 60.57% | 58.5% |
| K=3 always r6 | 3132636 | COMPLETED, 01:38:41 | 5721.98 | 1.343x | 3065.85 | 1.175x | 35.75% | 58.4% |

Interpretation: Qwen3-8B에서도 NeMo-RL rollout generation 자체는 빨라집니다. 하지만 generation이 전체 step time의 약 58%만 차지하므로 E2E는 1.15-1.18x 수준으로 눌립니다. 따라서 standalone 대비 gap의 원인은 SpecDec가 아예 안 붙은 것이 아니라, NeMo-RL E2E composition, gated mode가 거의 켜지지 않는 문제, 그리고 Qwen3-8B K=3 later-position acceptance/prompt distribution mismatch가 함께 만든 결과로 보는 게 맞습니다.

## 2026-06-03 13:06 PDT 추가 결과

이 섹션은 위 13:35 PDT E2E summary로 superseded되었습니다. Qwen3-8B logprob-skip diagnostic은 Step 1-4까지 matched generation speedup을 계산할 수 있게 됐고, 이후 baseline도 completed 상태로 확인됐습니다.

| Job | Mode | Current status | Step | Generated tok/s/GPU | Speedup vs matched baseline | Acceptance | Per-position acceptance |
|---:|---|---|---:|---:|---:|---:|---|
| 3132444 | baseline r5 | COMPLETED | 1 | 4319.01 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | COMPLETED | 2 | 4168.85 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | COMPLETED | 3 | 4362.34 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | COMPLETED | 4 derived | 4192.84 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | COMPLETED | mean 1-4 | 4260.76 | 1.000x | n/a | n/a |
| 3132635 | K=1 always r6 | COMPLETED | 1 | 5674.47 | 1.314x | 63.93% | 63.93% |
| 3132635 | K=1 always r6 | COMPLETED | 2 | 5498.59 | 1.319x | 59.22% | 59.22% |
| 3132635 | K=1 always r6 | COMPLETED | 3 | 5530.84 | 1.268x | 58.43% | 58.43% |
| 3132635 | K=1 always r6 | COMPLETED | 4 | 5289.47 | 1.262x | 60.68% | 60.68% |
| 3132635 | K=1 always r6 | COMPLETED | mean 1-4 | 5498.34 | 1.290x | 60.57% | n/a |
| 3132636 | K=3 always r6 | COMPLETED | 1 | 5801.55 | 1.343x | 38.72% | 59.07%, 36.32%, 20.76% |
| 3132636 | K=3 always r6 | COMPLETED | 2 | 5918.88 | 1.420x | 34.54% | 54.31%, 31.82%, 17.50% |
| 3132636 | K=3 always r6 | COMPLETED | 3 | 5705.44 | 1.308x | 34.06% | 53.09%, 31.38%, 17.70% |
| 3132636 | K=3 always r6 | COMPLETED | 4 | 5462.06 | 1.303x | 35.67% | 55.59%, 33.05%, 18.37% |
| 3132636 | K=3 always r6 | COMPLETED | mean 1-4 | 5721.98 | 1.343x | 35.75% | n/a |

Qwen3-8B online r68 gated scrub smoke `3133925`는 이후 `FAILED`, 00:11:02로 확인됐습니다. 실패 지점은 rollout 전 actor venv build의 `Disk quota exceeded`였으므로 scheduler-output scrub patch는 아직 검증되지 않았습니다.

## 2026-06-03 13:04 PDT 이전 상태

이 섹션은 위 13:06 PDT 섹션으로 superseded되었습니다. 당시에는 새 matched speedup metric이 아직 추가되지 않았고, 추가된 것은 Qwen3-8B online r67 실패 상태와 r68 gated smoke 제출이었습니다.

| Job | Mode | Status at 13:04 PDT | Notes |
|---:|---|---|---|
| 3133619 | K=1 gated V9 smoke, 1 step | FAILED, 00:17:53 | Reached rollout generation, then failed before `SpecDec early` metric in vLLM V1 `_bookkeeping_sync`: `assert sampled_token_ids.shape[-1] == 1`. V9 did not fully fix the scheduler/runner bookkeeping mismatch. |
| 3133620 | K=1 always online, 20 steps | FAILED, 00:21:12 | Reached rollout generation, then failed in policy-side Megatron/TransformerEngine forward: `rmsnorm_fwd_cuda_kernel.cu:49 ... CUDA Error: invalid argument`. No rollout metric. |
| 3133622 | K=2 always online, 20 steps | FAILED, 00:18:40 | Reached rollout generation, then failed in policy-side Megatron/TransformerEngine forward: `rmsnorm_fwd_cuda_kernel.cu:49 ... CUDA Error: invalid argument`. No rollout metric. |
| 3133624 | K=3 always online, 20 steps | FAILED, 00:24:26 | Same policy-side Megatron/TransformerEngine RMSNorm CUDA invalid-argument. No rollout metric. |
| 3133621/3133623/3133625 | K=1/2/3 gated V9, 20 steps | CANCELLED | Cancelled because smoke `3133619` failed. |
| 3133925 | K=1 gated r68 scrub smoke, 1 step | FAILED later, 00:11:02 | Fresh actor venv suffix with scheduler-output scrub patch, but actor venv build hit disk quota before rollout. |

당시 Qwen3-8B logprob-skip K=1/K=3 always r6만 Step 4 early-generation metrics를 냈고, baseline r5 Step 4가 아직 확인되지 않아 matched speedup을 계산하지 않았습니다. 이 내용은 위 13:06 PDT Step 1-4 표로 superseded되었습니다.

| Job | Mode | Step | Generated tok/s/GPU | Acceptance | Per-position acceptance |
|---:|---|---:|---:|---:|---|
| 3132635 | K=1 always r6 | 4 early | 5289.47 | 60.68% | 60.68% |
| 3132636 | K=3 always r6 | 4 early | 5462.06 | 35.67% | 55.59%, 33.05%, 18.37% |

Interpretation: online drafter training still has no usable performance result. The latest K=1 gated failure confirms the vLLM V1 SpecDec bookkeeping mismatch remains even after V9. The r67 failure dump shows the immediate inconsistency: runner-side gate logged disabled, but scheduler output still had speculative-width scheduling. r68 targets that exact mismatch. The K=1/K=2/K=3 always failures are separate policy/Megatron-side TransformerEngine errors and should be debugged independently from the generation-side assertion.

## 2026-06-03 12:44 PDT 추가 결과

Qwen3-8B logprob-skip r5/r6 diagnostic에서 Step 3 early-generation metric까지 나왔습니다. Step 3는 아직 step summary의 `Generation Worker Group` 라인이 아니라 `[SpecDec early generation]`의 `generated_tokens / elapsed_s / 4 GPU`로 계산했습니다. Step 1/2 summary 값과 같은 계산 방식입니다. 이 실험은 default GRPO가 아니라 post-rollout policy/reference logprob bottleneck을 제거했을 때 generation ceiling에 얼마나 가까워지는지 보는 changed-objective diagnostic입니다.

| Job | Mode | Status at 12:44 PDT | Step | Generated tok/s/GPU | Generation speedup vs matched baseline | Acceptance | Per-position acceptance |
|---:|---|---|---:|---:|---:|---:|---|
| 3132444 | baseline r5 | RUNNING | 1 | 4319.01 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | RUNNING | 2 | 4168.85 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | RUNNING | 3 early | 4362.34 | 1.000x | n/a | n/a |
| 3132444 | baseline r5 | RUNNING | mean 1-3 | 4283.40 | 1.000x | n/a | n/a |
| 3132635 | K=1 always r6 | RUNNING | 1 | 5674.47 | 1.314x | 63.93% | 63.93% |
| 3132635 | K=1 always r6 | RUNNING | 2 | 5498.59 | 1.319x | 59.22% | 59.22% |
| 3132635 | K=1 always r6 | RUNNING | 3 early | 5530.84 | 1.268x | 58.43% | 58.43% |
| 3132635 | K=1 always r6 | RUNNING | mean 1-3 | 5567.97 | 1.300x | 60.53% | n/a |
| 3132636 | K=3 always r6 | RUNNING | 1 | 5801.55 | 1.343x | 38.72% | 59.07%, 36.32%, 20.76% |
| 3132636 | K=3 always r6 | RUNNING | 2 | 5918.88 | 1.420x | 34.54% | 54.31%, 31.82%, 17.50% |
| 3132636 | K=3 always r6 | RUNNING | 3 early | 5705.44 | 1.308x | 34.06% | 53.09%, 31.38%, 17.70% |
| 3132636 | K=3 always r6 | RUNNING | mean 1-3 | 5808.62 | 1.356x | 35.77% | n/a |

Interpretation: logprob bottleneck을 줄인 changed-objective setting에서는 Qwen3-8B도 generation-only 기준으로 speedup이 확실히 보입니다. K=1은 Step 1-3 평균 1.300x이고, K=3은 Step 1-3 평균 1.356x입니다. 다만 K=3의 2/3번째 draft position acceptance가 Step 3에서도 31.38% / 17.70%로 낮아서, default GRPO E2E에서는 이 generation-only 이득이 그대로 보장되지 않습니다.

Qwen3-32B online r64 prep `3132637`은 12:01 PDT 확인 기준 `FAILED`, elapsed 00:32:17입니다. `▶ Generating responses for batch of size 2048...`까지 도달했지만 `SpecDec early` metric 전에 policy-to-vLLM weight streaming에서 실패했습니다. vLLM worker 쪽 `VllmInternalWorkerExtension.update_weights_via_ipc_zmq`가 `state_dict_info['draft.fc.weight']`를 찾다가 `KeyError: 'draft.fc.weight'`를 냈고, policy worker는 ZMQ timeout으로 종료됐습니다. Qwen3-32B matrix `3132638/3132639/3132641/3132642/3132643/3132644`는 prep 실패 때문에 모두 `CANCELLED`입니다.

Qwen3-8B online V6 retry:

| Job | Mode | Status at 12:25 PDT | Purpose |
|---:|---|---|---|
| 3133314 | K=1 gated smoke, 1 step | FAILED, 00:13:41 | Reached rollout generation, then failed before `SpecDec early` metric with the same vLLM V1 `sampled_token_ids.shape[-1] == 1` assertion. |
| 3133315 | K=1 always, 20 steps | CANCELLED dependency | Cancelled after `3133314` failed. |
| 3133316 | K=1 gated, 20 steps | CANCELLED dependency | Cancelled after `3133314` failed. |
| 3133317 | K=2 always, 20 steps | CANCELLED dependency | Cancelled after `3133314` failed. |
| 3133318 | K=2 gated, 20 steps | CANCELLED dependency | Cancelled after `3133314` failed. |
| 3133319 | K=3 always, 20 steps | CANCELLED dependency | Cancelled after `3133314` failed. |
| 3133320 | K=3 gated, 20 steps | CANCELLED dependency | Cancelled after `3133314` failed. |

Qwen3-8B online r67 retry/submissions:

| Job | Mode | Status at 12:44 PDT | Purpose |
|---:|---|---|---|
| 3133619 | K=1 gated V9 smoke, 1 step | RUNNING, ~10m | Verify V9 runner active-batch + zero-draft disabled path fixes the vLLM V1 bookkeeping assertion. No rollout metric yet. |
| 3133621 | K=1 gated V9, 20 steps | PENDING dependency | Runs after `afterok:3133619`. |
| 3133623 | K=2 gated V9, 20 steps | PENDING dependency | Runs after `afterok:3133619`. |
| 3133625 | K=3 gated V9, 20 steps | PENDING dependency | Runs after `afterok:3133619`. |
| 3133620 | K=1 always online, 20 steps | RUNNING, ~10m | Online drafter training with gate disabled; isolates whether online always-on can emit rollout metrics. No rollout metric yet. |
| 3133622 | K=2 always online, 20 steps | RUNNING, ~10m | Same as above for K=2. No rollout metric yet. |
| 3133624 | K=3 always online, 20 steps | RUNNING, ~10m | Same as above for K=3. No rollout metric yet. |

## 2026-06-03 11:47 PDT 추가 확인

이전 poll에서는 새로운 rollout 성능 metric이 아직 없었습니다. 위 12:41 PDT 섹션이 Qwen3-8B logprob-skip diagnostic과 online r67 status에 대해서는 최신 값을 supersede합니다. fixed/offline numeric table은 그대로 최신입니다.

| Area | 추가 확인 내용 |
|---|---|
| Qwen3-8B online drafter | `3131454` K=1 gated r61 smoke는 `FAILED`, 01:11:02입니다. `policy.draft.enabled=true`, BF16 generation, KV auto, scheduler override는 config에 반영됐고 vLLM worker startup까지 갔지만, rollout generation 중 `gpu_model_runner.py::_bookkeeping_sync`에서 `assert sampled_token_ids.shape[-1] == 1`가 발생했습니다. 성능 metric은 없습니다. |
| Qwen3-8B online V6 retry | Runtime patch V6를 remote online worktree에 적용했고 smoke `3133314`가 rollout generation까지 갔지만, 같은 vLLM V1 `sampled_token_ids.shape[-1] == 1` assertion으로 실패했습니다. Dependent matrix `3133315-3133320`은 dependency cancel입니다. |
| Qwen3-8B logprob-skip diagnostic | `3132444` baseline r5, `3132635` K=1 always r6, `3132636` K=3 always r6가 모두 Step 3 early-generation metric까지 냈습니다. Step 1-3 평균 generated tok/s/GPU는 baseline 4283.40, K=1 5567.97, K=3 5808.62입니다. |
| Qwen3-32B online drafter | `3132637` r64는 actor venv rebuild/suffix와 `tensordict` 문제를 통과했고 `▶ Generating responses for batch of size 2048...`까지 도달했지만, rollout metric 전에 policy-to-vLLM weight streaming에서 `KeyError: 'draft.fc.weight'`로 실패했습니다. |
| Qwen3-30B-A3B online drafter | `3130929`는 `FAILED`, 01:30:31입니다. `policy.draft.enabled=true`와 BF16/KV auto는 config에 들어갔지만 rollout 전에 vLLM Qwen3-MoE weight loader에서 `ValueError: shard_dim=0 is not a valid data dimension for a 3D tensor`가 났습니다. |
| Current running jobs | `3132444` Qwen3-8B diagnostic baseline, `3132635/3132636` Qwen3-8B diagnostic K=1/K=3, r67 V9 gated smoke `3133619`, and r67 always-on online `3133620/3133622/3133624`. Current pending online jobs: gated 20-step `3133621/3133623/3133625` waiting on `afterok:3133619`. No online-drafter run currently has rollout metrics. |

## vLLM Standalone 결과

### Qwen3-32B, RedHatAI public EAGLE3 drafter

조건: `Qwen/Qwen3-32B`, `RedHatAI/Qwen3-32B-speculator.eagle3`, dtype auto, KV auto, ISL/OSL 1000/512, batch size 32.

| K | Baseline tok/s/GPU | SpecDec tok/s/GPU | Speedup | Acceptance |
|---:|---:|---:|---:|---:|
| 1 | 1905.67 | 3097.15 | 1.625x | 79.9% |
| 2 | 1905.67 | 3893.39 | 2.043x | 72.9% |
| 3 | 1905.67 | 4359.78 | 2.288x | 67.1% |

Interpretation: standalone ceiling은 충분히 큽니다. 특히 K=3은 bs32에서 2.288x, bs1에서는 2.744x까지 나옵니다. 따라서 32B에서 NeMo-RL gated 결과가 낮은 것은 drafter가 전혀 안 되는 문제라기보다 NeMo-RL runtime/gate/workload 차이로 보는 것이 맞습니다.

### Qwen3-8B, DAPOMath real prompts

조건: real DAPOMath prompt 기반 standalone isolation.

| K | Speedup | Acceptance |
|---:|---:|---:|
| 1 | 0.665x | 61.11% |
| 2 | 0.817x | 46.91% |
| 3 | 0.853x | 36.62% |

Interpretation: Qwen3-8B public drafter는 synthetic fixed-prompt standalone에서 보였던 2x 수준 ceiling이 DAPOMath real-prompt distribution으로는 재현되지 않습니다. NeMo-RL에서 8B speedup이 제한적인 이유에는 gate/E2E overhead뿐 아니라 prompt/domain/workload mismatch도 포함됩니다.

## NeMo-RL 고정 Drafter Follow-Up Matrix

아래 표는 `docs/specdec_followup_latest_metrics.json` 기준입니다. Qwen3-8B 네 job은 4시간 walltime으로 timeout 되었고, 표의 값은 timeout 전까지 나온 최종 관측 step 평균입니다. Qwen3-32B와 Qwen3-30B-A3B는 20 step 완료입니다.

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

Interpretation: model이 달라도 공통 패턴은 같습니다. gated mode에서는 scheduler/runner가 speculation을 거의 켜지 않습니다. 항상 켜는 always-on이 실제 throughput을 볼 수 있는 경로입니다.

## NeMo-RL Speedup 관측치

### Qwen3-32B

기존 matched-baseline NeMo-RL 결과에서 gated K=1/K=2/K=3는 모두 20 step 완료했고 평균적으로 baseline보다 느렸습니다.

| Mode | K | Avg generation speedup | Avg E2E speedup | Avg acceptance |
|---|---:|---:|---:|---:|
| gated | 1 | 0.925x | 0.932x | 44.89% |
| gated | 2 | 0.920x | 0.945x | 31.86% |
| gated | 3 | 0.932x | 0.951x | 23.75% |

always-on Qwen3-32B는 K=1/K=3에서 generation 1.33-1.43x, E2E 1.15-1.20x 수준을 보였습니다. 즉 SpecDec 자체가 전혀 효과가 없는 것은 아니고, gated runtime path와 E2E overhead가 주된 차이를 만듭니다.

### Qwen3-8B

Qwen3-8B exact-engine/no-gate diagnostic은 K=1에서 generation 개선이 확인됐습니다.

| Run | K | Generated tok/s/GPU | Generation speedup | Acceptance |
|---|---:|---:|---:|---:|
| baseline 3126355 | n/a | 5652.52 | 1.000x | n/a |
| native K=1 3126356 | 1 | 7345.13 | 1.299x | 64.61% |
| native K=3 3126357 | 3 | 2555.29 | 0.452x | 40.66% |
| dynamic-cap K=3 engine, effective K=1 3126685 | effective 1 | 4446.23 | 0.787x | 64.60% |

Interpretation: 8B에서는 K=1 native path가 가장 의심 없이 좋은 경로입니다. K=3은 later-position acceptance가 낮아서 overhead가 커지고, K=3 engine을 dynamic cap으로 K=1처럼 쓰는 방식도 현재는 native K=1보다 느립니다.

## Online Drafter Training 상태

online drafter training은 모두 `policy.draft.enabled=true`, `policy.draft.loss_weight=1.0`, BF16 generation, KV auto를 기준으로 제출했습니다. 아직 rollout 성능 metric은 없습니다.

| Job | Model | K/Gate | Status at 12:44 PDT | Notes |
|---:|---|---|---|---|
| 3130929 | Qwen3-30B-A3B online prep | K=1 gated | FAILED, 01:30:31 | MasterConfig confirmed `policy.draft.enabled=true`, BF16, KV auto, max_model_len 4096. Failed before rollout during policy-to-vLLM weight streaming: vLLM Qwen3-MoE fused-MoE loader rejected `shard_dim=0` for a 3D expert tensor, then policy worker hit ZMQ timeout. |
| 3130932-3130947 | Qwen3-30B-A3B online matrix | K=1/2/3 always/gated | PENDING dependency | Waiting on `afterok:3130929`; expected not to run because prep failed. |
| 3130930 | Qwen3-32B online prep | K=1 gated | FAILED, 00:41:29 | Reached MasterConfig and worker init, then `MegatronPolicyWorker.move_model(..., "cpu")` called `offload_to_cpu` on Megatron `_ParamAndGradBuffer`, which does not expose that method. This failure is now patched. |
| 3131904 | Qwen3-32B online r62 prep | K=1 gated | FAILED, 00:08:41 | Passed the old offload failure, then failed before rollout because cached actor venv lacked `tensordict`. |
| 3132468 | Qwen3-32B online r63 prep | K=1 gated | FAILED, 00:00:47 | Added actor-venv rebuild/suffix intent, but online submit wrapper did not pass those env vars into the driver command and Ray head hit Slurm memory-step failure. |
| 3132637 | Qwen3-32B online r64 prep | K=1 gated | FAILED, 00:32:17 | Actor venv rebuild/suffix and `tensordict` issue are resolved. The run reached rollout generation setup, then failed before `SpecDec early` metric during policy-to-vLLM weight streaming. vLLM worker raised `KeyError: 'draft.fc.weight'` because the streamed online draft weight name was absent from generation worker `state_dict_info`; policy worker then hit ZMQ timeout. |
| 3132638-3132644 | Qwen3-32B online r64 matrix | K=1/2/3 always/gated | CANCELLED | Dependency cancelled after `3132637` failed. |
| 3131454 | Qwen3-8B r61 scheduler-override smoke | K=1 gated | FAILED, 01:11:02 | Replacement for r60 scheduler assertion. It passed config/worker startup and failed during rollout generation in vLLM V1 SpecDec bookkeeping: `assert sampled_token_ids.shape[-1] == 1`. No rollout metric. |
| 3131455-3131460 | Qwen3-8B online matrix | K=1/2/3 always/gated | PENDING dependency | Waiting on `afterok:3131454`; expected not to run because prep/smoke failed. |
| 3133314 | Qwen3-8B r66 V6 runtime-gate smoke | K=1 gated | FAILED, 00:13:41 | V6 patch used fresh actor venv suffix `qwen8_r66_gatev6` and reached rollout generation, but failed before `SpecDec early` metric with the same vLLM V1 `assert sampled_token_ids.shape[-1] == 1` bookkeeping assertion. |
| 3133315-3133320 | Qwen3-8B r66 V6 matrix | K=1/2/3 always/gated | CANCELLED | Dependency cancelled after `3133314` failed. |
| 3133619 | Qwen3-8B r67 V9 runtime-gate smoke | K=1 gated | RUNNING, ~10m at 12:44 PDT | V9 patch adds runner active-batch pressure and zero-draft disabled path. It has started; no rollout metric yet. |
| 3133620/3133622/3133624 | Qwen3-8B r67 online always-on | K=1/2/3 always | RUNNING, ~10m at 12:44 PDT | Online drafter training with runtime gate disabled. No rollout metric yet. |
| 3133621/3133623/3133625 | Qwen3-8B r67 V9 gated matrix | K=1/2/3 gated | PENDING dependency | Waiting on `afterok:3133619`. |

Important: `3130930` is not a performance result. It was an online-drafter initialization compatibility failure. The patch now keeps the old helper-method path when available and adds a fallback that moves `param_data` / `grad_data` directly, remaps bucket views and `param.data` / `param.main_grad`, and clears cached bucket-group shard views.

## Qwen3-8B Logprob-Skip Diagnostic 상태

이 diagnostic은 default GRPO가 아닙니다. `force_on_policy_ratio=true`, `reference_policy_kl_penalty=0`, `NRL_GRPO_SKIP_POLICY_LOGPROBS_IF_SAFE=true`로 policy/reference logprob bottleneck을 줄이면 NeMo-RL E2E가 standalone/generation ceiling에 더 가까워지는지 확인하는 changed-objective 실험입니다.

| Job | Mode | Status at 12:41 PDT | Notes |
|---:|---|---|---|
| 3131108 | baseline r2 | FAILED, 00:28:21 | No metric. Megatron optimizer scheduler asserted `lr_warmup_steps < lr_decay_steps` during `MegatronPolicyWorker` initialization. |
| 3131109 | K=1 always r2 | FAILED, 00:29:07 | Same scheduler assertion; no metric. |
| 3131110 | K=3 always r2 | FAILED, 00:32:50 | Same scheduler assertion; no metric. |
| 3131615-3131617 | r3 reuse-venv | CANCELED | Invalid run. It called stale root launchers, so do not use as evidence. |
| 3131648 | baseline r4 reuse-venv | FAILED, 00:27:59 | Same scheduler assertion; no metric. |
| 3131649 | K=1 always r4 reuse-venv | FAILED, 00:19:03 | Same scheduler assertion; no metric. |
| 3131650 | K=3 always r4 reuse-venv | FAILED, 00:20:43 | Same scheduler assertion; no metric. |
| 3132444 | baseline r5 scheduler-fix | RUNNING, ~78m | Step 1-3 metrics emitted: 4319.01 / 4168.85 / 4362.34 generated tok/s/GPU. Scheduler override is verified: `lr_warmup_iters=0`, `lr_decay_iters=1000`. |
| 3132445-3132446 | K=1/K=3 always r5 scheduler-fix | FAILED, 00:00:31 | Slurm Ray-head step failed with `Memory required by task is not available`; not a model/code result. |
| 3132635 | K=1 always r6 scheduler-fix low-CPU | RUNNING, ~71m | Step 1-3 metrics emitted: 5674.47 / 5498.59 / 5530.84 generated tok/s/GPU, mean 1.300x vs diagnostic baseline, acceptance 63.93% / 59.22% / 58.43%. |
| 3132636 | K=3 always r6 scheduler-fix low-CPU | RUNNING, ~71m | Step 1-3 metrics emitted: 5801.55 / 5918.88 / 5705.44 generated tok/s/GPU, mean 1.356x vs diagnostic baseline, acceptance 38.72% / 34.54% / 34.06%, Step 3 per-position 53.09% / 31.38% / 17.70%. |

## Root Cause 정리

### 1. Gated SpecDec가 거의 켜지지 않음

TLT-style gate는 request/token threshold를 기준으로 speculation을 제한합니다. 현재 workload에서는 scheduler enabled ratio가 매우 낮습니다.

| Model | K | Scheduler enabled | Runner enabled |
|---|---:|---:|---:|
| Qwen3-8B gated | 1 | 0.02% | 0.63% |
| Qwen3-8B gated | 3 | 0.02% | 0.63% |
| Qwen3-32B gated | 1 | 0.26% | 4.10% |
| Qwen3-32B gated | 3 | 0.22% | 3.21% |
| Qwen3-30B-A3B gated | 1 | 0.00% | 0.02% |
| Qwen3-30B-A3B gated | 3 | 0.00% | 0.01% |

따라서 gated 실험이 낮게 나오는 것은 자연스럽습니다. 항상 SpecDec를 켜는 always-on 실험이 실제 acceleration 가능성을 더 잘 보여줍니다.

### 2. NeMo-RL E2E는 generation-only benchmark가 아님

vLLM standalone은 generation-only throughput을 봅니다. NeMo-RL은 rollout 후에 reward, policy logprob, reference logprob, training update가 붙습니다. `nemo_rl/algorithms/grpo.py`에서 post-rollout policy/reference forward가 남아 있으므로 generation speedup이 그대로 E2E speedup으로 변환되지 않습니다.

### 3. Acceptance rate가 model/workload에 따라 크게 달라짐

Qwen3-32B standalone K=3 bs32 acceptance는 67.1%로 괜찮지만, NeMo-RL gated K=3 평균 acceptance는 23.75% 수준입니다. Qwen3-8B DAPOMath standalone도 K=3 acceptance가 36.62%로 낮습니다. 이 차이는 prompt distribution, sampling, sequence length, runtime batching 차이를 같이 의심해야 합니다.

## 다음 액션

1. Qwen3-32B r64 prep `3132637`의 `draft.fc.weight` streaming mismatch를 코드 레벨로 고칩니다. 현재 evidence는 Megatron side가 online draft weight를 stream하려고 하지만 vLLM generation worker의 `state_dict_info`에는 그 key가 없다는 것입니다.
2. Qwen3-8B r5/r6 logprob-skip diagnostic `3132444/3132635/3132636`의 Step 4 및 E2E step time을 확인합니다. Step 1-3 generation-only metric은 이미 나왔습니다.
3. Qwen3-8B online r67 `3133619/3133620/3133622/3133624`가 rollout metric까지 가는지 확인합니다. V9가 `3133314`와 같은 vLLM V1 sampled-token bookkeeping assertion을 해결하는지가 핵심입니다.
4. Qwen3-30B-A3B `3130929`의 MoE weight streaming failure를 코드 레벨로 분석합니다. 현재 증상은 vLLM Qwen3-MoE fused-MoE loader가 Megatron-streamed 3D expert tensor의 shard dimension mapping을 받아들이지 못하는 것입니다.
5. 새 rollout metric이 나오면 `docs/specdec_completed_eval_bar_graphs.html`과 PNG chart를 다시 재생성합니다.

## 16:03 PDT 추가 업데이트

Qwen3-32B worker-batch matching 실험이 Step 1-2 matched early metric을 냈습니다. 이 실험은 기존 `GBS=2048` 대신 `GBS=512`를 써서 16 generation workers 기준 worker당 response batch를 약 `128`에서 약 `32`로 낮춘 조건입니다.

| Mode | Job | Step scope | Generated tok/s/GPU | Gen speedup | E2E tokens/sec | E2E speedup | Acceptance | Per-position acceptance |
|---|---:|---|---:|---:|---:|---:|---:|---|
| baseline | 3136000 | Step 1 | 1301.84 | 1.000x | 11570.09 | 1.000x | n/a | n/a |
| baseline | 3136000 | Step 2 | 931.72 | 1.000x | 9604.41 | 1.000x | n/a | n/a |
| K=1 always | 3136001 | Step 1 | 1779.09 | 1.367x | 13526.61 | 1.169x | 69.29% | 69.29% |
| K=1 always | 3136001 | Step 2 | 1365.31 | 1.465x | 11972.18 | 1.247x | 70.73% | 70.73% |
| K=1 always | 3136001 | Step 1-2 mean | n/a | 1.42x | n/a | 1.21x | 70.0% | n/a |
| K=3 always | 3136002 | Step 1 | 2064.21 | 1.586x | 14647.18 | 1.266x | 45.46% | 66.34%, 42.82%, 27.20% |
| K=3 always | 3136002 | Step 2 | 1623.29 | 1.742x | 13228.23 | 1.377x | 47.29% | 67.97%, 44.73%, 29.16% |
| K=3 always | 3136002 | Step 1-2 mean | n/a | 1.66x | n/a | 1.32x | 46.4% | 67.16%, 43.77%, 28.18% |

Partial Step 3도 일부 나왔습니다. Baseline `3136000` Step 3 early generation은 `1356.22` generated tok/s/GPU이고, K=1 `3136001` Step 3 early generation은 `1922.32` generated tok/s/GPU, acceptance `70.01%`입니다. K=1 Step 3 generation speedup은 `1.42x`입니다. K=3 Step 3는 아직 metric이 없습니다.

Interpretation: worker당 response batch를 standalone `bs32`에 가깝게 낮추자 K=3 generation/E2E early signal이 기존 completed `GBS=2048` always-on aggregate보다 좋아졌습니다. 기존 Qwen3-32B always-on K=3 completed 결과는 generation `1.356x`, E2E `1.181x`, acceptance `45.28%`였고, 이번 worker32 Step 1-2 early 값은 generation `1.66x`, E2E `1.32x`, acceptance `46.4%`입니다. Acceptance가 standalone bs32 K=3의 `67.1%`에는 아직 못 미치므로, standalone K=3 `2.288x` ceiling까지 남은 gap은 acceptance/prompt distribution과 NeMo-RL E2E composition이 같이 설명합니다.

16:20 PDT 추가 poll: K=3 `3136002`는 Step 9 early generation까지 진행했습니다. Matched baseline comparison이 가능한 Step 1-8 generation 평균은 `1.64x`, Step 1-7 E2E 평균은 `1.31x`, Step 1-8 acceptance 평균은 `45.9%`입니다. Step 4/5/6/7/8도 각각 generation `1.657x`/`1.656x`/`1.631x`/`1.580x`/`1.651x`라서 Step 1-2의 개선이 유지됩니다. Step 9는 K=3 early generation `2027.98` generated tok/s/GPU, E2E tokens/sec `15966.43`, acceptance `45.52%`까지 나왔고 matched baseline Step 9은 아직 pending입니다.

새 chart와 raw metric을 추가했습니다.

| Artifact | 내용 |
|---|---|
| `docs/qwen3_32b_nemorl_worker32_early.png` | Qwen3-32B worker≈32 Step 1-2 speedup/acceptance, 512 dpi |
| `docs/qwen3_32b_nemorl_worker32_early_metrics.csv` | 위 chart의 raw metric |

Qwen3-8B online short512 r73 `3136275`는 이후 `FAILED`, 00:23:16으로 종료됐습니다. MasterConfig에는 `policy.draft.enabled=true`, `draft.model_name=RedHatAI/Qwen3-8B-speculator.eagle3`, `max_total_sequence_length=2560`, `max_new_tokens=512`, BF16 generation, KV auto가 정상 반영됐고, generation까지 도달해 vLLM internal acceptance 57.6-66.6% 로그를 냈습니다. 실패는 이후 policy-side Megatron forward의 TransformerEngine RMSNorm CUDA invalid-argument입니다. 이 결과는 위 16:25 PDT 섹션으로 supersede됩니다.

## 15:35 PDT 추가 업데이트

Qwen3-8B online drafter smoke `3135673`는 rollout generation까지 도달했습니다. 이전의 vLLM V1 bookkeeping assertion과 scheduler-token broadcast failure는 재현되지 않았습니다. 대신 generation 이후 `policy.get_logprobs()`에서 Megatron/TransformerEngine RMSNorm CUDA invalid argument가 발생했습니다. 설정은 `policy.draft.enabled=true`, BF16/KV auto, `max_total_sequence_length=8192`, `max_new_tokens=8192`, Megatron `TP=4`였습니다. 즉 online path는 generation-side failure를 넘겼고, 현재 blocker는 long-sequence policy/logprob forward 안정성입니다.

새로 제출한 확인 job:

| Job | Purpose | Settings | Status |
|---:|---|---|---|
| `3135990` | Qwen3-8B online drafter short512 smoke | K=1 always, `max_total_sequence_length=2560`, `max_new_tokens=512`, BF16/KV auto | FAILED before Ray startup; Slurm could not create ray-head step because memory required by task was unavailable |
| `3136275` | Qwen3-8B online drafter short512 low-CPU retry | same as `3135990`, but `CPUS_PER_WORKER=24` | FAILED after generation; policy-side TE RMSNorm failure |
| `3136718` | Qwen3-8B online drafter short512 local-transformer-spec smoke | same as `3136275`, plus `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true` | FAILED after generation; original TE RMSNorm avoided, then missing `scaled_masked_softmax_cuda` |
| `3137127` | Qwen3-8B online drafter short512 local-transformer no-fused-softmax smoke | same as r74, plus `model_cfg.masked_softmax_fusion=False` under the guard | PENDING |
| `3136000` | Qwen3-32B worker32 baseline | `8x2` GPUs, 16 workers, `GBS=512`, expected 32 responses/worker | RUNNING; Step 1-2 baseline metrics emitted, Step 3 early generation emitted |
| `3136001` | Qwen3-32B worker32 K=1 always | same as baseline, runtime gate disabled | RUNNING; Step 1-2 mean gen/E2E speedup `1.42x`/`1.21x`, acceptance `70.0%`; Step 3 early generation emitted |
| `3136002` | Qwen3-32B worker32 K=3 always | same as baseline, runtime gate disabled | RUNNING; Step 1-2 mean gen/E2E speedup `1.66x`/`1.32x`, acceptance `46.4%` |

Qwen3-32B 기존 completed always-on은 `GBS=2048`라서 16 generation workers 기준 worker당 약 128 responses였습니다. 새 `GBS=512` 실험은 worker당 약 32 responses로 standalone `bs32`와 더 직접 비교하기 위한 실험입니다.
