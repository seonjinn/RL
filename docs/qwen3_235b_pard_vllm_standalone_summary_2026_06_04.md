# Qwen3-235B PARD vLLM Standalone Summary

Date: 2026-06-04 PDT

## Setup

Target model: `Qwen/Qwen3-235B-A22B`

Drafter: `amd/PARD-Qwen3-0.6B`

vLLM container: `vllm-hsg-ultra-rl-v0.20.2-nemo-speed-pr24.sqsh`

Execution: 1 node, 4 GB200 GPUs, `TP=4`, `PP=1`, no draft TP override, `parallel_drafting=true`

Sampling: fixed-length greedy decode, `temperature=0.0`, `min_tokens=OSL`, `max_tokens=OSL`

Prompts: synthetic `prompt_token_ids`; no real prompt JSONL was used in these sweeps.

## Short Sweep: ISL=1000, OSL=512

`max_model_len=2536`, `max_num_batched_tokens=64000`

| Batch | K=5 speedup | K=5 acc | K=12 speedup | K=12 acc |
|---:|---:|---:|---:|---:|
| 1 | 1.33x | 95.5% | 2.90x | 90.7% |
| 2 | 1.39x | 95.5% | 2.68x | 90.7% |
| 4 | 1.44x | 95.5% | 2.92x | 90.7% |
| 8 | 1.47x | 95.5% | 3.11x | 92.7% |
| 16 | 1.50x | 95.5% | 2.96x | 92.9% |
| 32 | 1.71x | 95.5% | 3.29x | 93.0% |

Short synthetic result: PARD K=12 is strong. It reaches 2.68x-3.29x throughput speedup with roughly 90.7%-93.0% acceptance and mean acceptance length around 11.9-12.2 tokens.

## Long Sweep: ISL=10000, OSL=1000

`max_model_len=12024`, `max_num_batched_tokens=393216`

| Batch | K=5 speedup | K=5 acc | K=12 speedup | K=12 acc |
|---:|---:|---:|---:|---:|
| 1 | 0.40x | 24.6% | 0.34x | 7.2% |
| 2 | 0.35x | 16.8% | 0.49x | 13.4% |
| 4 | 0.40x | 21.0% | 0.49x | 13.4% |
| 8 | 0.43x | 23.8% | 0.48x | 13.4% |
| 16 | 0.34x | 13.9% | 0.47x | 13.4% |
| 32 | 0.39x | 13.7% | 0.55x | 13.4% |

Long synthetic result: PARD is consistently slower than baseline. The main observed cause is acceptance collapse, not just scheduler overhead. K=5 acceptance falls to 13.7%-24.6%; K=12 is around 7.2%-13.4%.

## Interpretation

PARD can reproduce the kind of large standalone speedup we expected only in the short synthetic decode regime. It does not currently solve the long-context Qwen3-235B case under `ISL=10000, OSL=1000` with synthetic prompt IDs.

The long run also has a capacity caveat: logs report roughly `193,343` KV tokens for baseline, but only around `116,274` KV tokens for PARD K=5 and `115,970` for PARD K=12. With `max_model_len=12024`, that is about `16.08x` max concurrency for baseline and about `9.6x` for PARD. Therefore bs16/bs32 are not equivalent to fully resident 16/32-request decode. However, the larger issue is still the low acceptance rate.

## Artifacts

CSV: `docs/qwen3_235b_vllm_pard_metrics.csv`

Short PNG: `docs/qwen3_235b_vllm_pard_isl1k_osl512_speedup_acceptance.png`

Long PNG: `docs/qwen3_235b_vllm_pard_isl10k_osl1k_speedup_acceptance.png`

Plot script: `scripts/plot_qwen3_235b_pard_results.py`

## Active Follow-Up: OpenMath Prompt Gate

The synthetic short sweep proves PARD can be fast for Qwen3-235B when acceptance
stays high. The synthetic long sweep proves that long context can collapse
acceptance. The next gate is therefore not another synthetic run; it is a real
prompt run with the same PARD implementation.

Submitted OpenMath real-prompt jobs:

| Run | Job | Status file |
|---|---:|---|
| Baseline | `3164321` | `latest_vllm_qwen235b_pard_openmath_isl1024_osl1024_baseline_jobs.txt` |
| PARD K=5 | `3164332` | `latest_vllm_qwen235b_pard_openmath_isl1024_osl1024_k5_jobs.txt` |
| PARD K=12 | `3164333` | `latest_vllm_qwen235b_pard_openmath_isl1024_osl1024_k12_jobs.txt` |

Shape: `ISL=1024`, `OSL=1024`, batch sizes `1 2 4 8 16 32`, `max_model_len=4096`,
`max_num_batched_tokens=131072`, OpenMath conversation JSONL prompts.

First attempt jobs `3163174`, `3163175`, and `3163176` failed before any batch
row. Root cause was a benchmark harness bug: real-prompt tokenization was not
normalized before passing `prompt_token_ids` into vLLM. Retry2 jobs `3163904`,
`3163905`, and `3163907` exposed the second half of the same bug:
`apply_chat_template(..., tokenize=True)` returned a dict-like value with
`input_ids` and `attention_mask`, and the harness tried to use those keys as
token ids. `standalone_vllm_specdec_breakdown.py` now extracts `input_ids`,
unwraps batched tokenization outputs, and casts token ids to `int`; the retry3
jobs above use that fix.

Decision rule:

- If K=12 keeps high acceptance and speedup on OpenMath, move PARD into a
  NeMo-RL generation-only validation before full GRPO.
- If K=12 slows down or acceptance collapses, do not spend 235B NeMo-RL nodes on
  PARD until either the prompt/domain mismatch is fixed or PARD-2 checkpoints
  become available.

## OpenMath Prompt Gate Result: ISL=1024, OSL=1024

The retry3 jobs completed. This run used real OpenMath conversation prompts,
not synthetic token ids.

| Batch | Baseline tok/s/GPU | K=5 tok/s/GPU | K=5 speedup | K=5 acc | K=12 tok/s/GPU | K=12 speedup | K=12 acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31.06 | 16.23 | 0.52x | 44.2% | 23.16 | 0.75x | 14.9% |
| 2 | 64.49 | 43.40 | 0.67x | 34.8% | 40.75 | 0.63x | 19.7% |
| 4 | 102.78 | 53.94 | 0.52x | 42.3% | 95.00 | 0.92x | 23.5% |
| 8 | 177.11 | 108.82 | 0.61x | 45.8% | 191.29 | 1.08x | 21.9% |
| 16 | 293.19 | 291.15 | 0.99x | 45.9% | 335.21 | 1.14x | 21.8% |
| 32 | 484.09 | 635.65 | 1.31x | 45.5% | 590.71 | 1.22x | 22.8% |

OpenMath interpretation:

- PARD does not reproduce the short synthetic K=12 result on real OpenMath
  prompts. Acceptance collapses from roughly 90%+ in the short synthetic sweep
  to 14.9%-23.5% for K=12.
- K=5 is the better practical setting in this run. It still slows down at
  batches 1-8, is roughly neutral at batch 16, and reaches 1.31x at batch 32.
- This supports the root-cause hypothesis that Qwen3-235B PARD success is
  prompt/domain and batch-regime dependent. The runtime path can produce
  standalone speedup, but real-prompt drafter quality is not yet good enough
  for universal speedup.
- For NeMo-RL, the next validation should be generation-only with K=5 first,
  and only under a shape where each generation worker gets a large enough
  effective batch. Full GRPO is not the next efficient test until this
  generation-only gate passes.

OpenMath artifact:
`docs/qwen3_235b_vllm_pard_openmath_isl1024_osl1024_speedup_acceptance.png`

## Drafter Training Direction For Math/OpenMath

The OpenMath gate strongly suggests a drafter-domain mismatch. Short synthetic
PARD K12 reaches about `93%` acceptance and `3.29x` speedup, while real
OpenMath drops to about `22.8%` acceptance for K12 and about `45.5%` for K5.
That is not enough evidence that PARD is algorithmically unsuitable for
Qwen3-235B; it is stronger evidence that the public `amd/PARD-Qwen3-0.6B`
drafter is not aligned to Qwen3-235B math-reasoning continuations.

The next drafter-training track should not be plain math SFT alone. It should
be target-aligned PARD/PARD-2-style distillation:

1. Build an external math prompt pool separate from the NeMo-RL training set:
   non-OpenMath public math sources such as OpenThoughts, NuminaMath,
   Math-DeepScaler, and DAPO-Math. Keep OpenMath as the main held-out gate
   unless we intentionally train an OpenMath-specific drafter and report a
   disjoint held-out split.
2. Generate teacher continuations with `Qwen/Qwen3-235B-A22B` using the same
   chat template and inference controls used in evaluation. Store prompt
   tokens, target continuations, target logits if affordable, and per-token
   metadata. Acceptance improves when the drafter learns what the target model
   actually emits, not just what a human solution corpus contains.
3. Start from `amd/PARD-Qwen3-0.6B` or a Qwen3-0.6B-family autoregressive
   checkpoint, then train a parallel-draft head/objective for K positions.
   Practical first target: train for `K=5` because OpenMath K5 was much more
   stable than K12. Train/evaluate K12 only after K5 acceptance improves.
4. Prefer PARD-2-style target alignment if implementation/checkpoints are
   available. PARD-2 explicitly shifts optimization toward inference-time
   consecutive acceptance length and adds target-model distillation, which is
   exactly the failure mode observed in the OpenMath K12 collapse.
5. Gate every trained drafter before NeMo-RL: run vLLM standalone on held-out
   OpenMath with `ISL=1024`, `OSL=1024`, batches `1,2,4,8,16,32`, and compare
   acceptance, mean accepted length, and speedup against the current public
   PARD baseline. Only move to NeMo-RL if held-out OpenMath acceptance improves
   materially and throughput speedup beats K5 public-PARD at batch 32.

Success criteria for the first training pass:

- K5 OpenMath acceptance should move from about `45.5%` toward at least
  `60%+`.
- K5 OpenMath speedup at batch 32 should exceed the current `1.31x`.
- K12 should not be prioritized until the K5-trained drafter proves target
  alignment; current K12 OpenMath acceptance around `22.8%` is too low.

Implementation status:

- The public AMD PARD trainer is now mapped into this repo's Qwen3-235B math
  track. It expects JSONL rows with `conversation: [{human, assistant}]`,
  trains from a base drafter such as `amd/PARD-Qwen3-0.6B`, and uses
  `general.para_num` as the parallel draft depth.
- For the first trainable gate, use `para_num=5` and Qwen3 PARD token id
  `151670`, which matches the public `amd/PARD-Qwen3-0.6B` config.
- PARD-2 is not yet a direct runnable path in this repo because AMD's public
  README says code and model checkpoints will be released soon. The practical
  immediate path is PARD K5 target-aligned distillation, then a PARD-2-style
  objective extension once implementation/checkpoint details are available.
- The first Qwen3-235B teacher-generation smoke created a 10K non-OpenMath
  prompt pool, then exposed two environment issues that are now patched in the
  submit wrapper: `/opt/venv/bin/python` did not exist in the container, and
  `huggingface-hub==1.16.1` conflicted with the container `transformers`
  requirement `huggingface-hub>=0.34.0,<1.0`.
- The active teacher-generation retry is job `3169844`, one GB200 node / 4
  GPUs, `Qwen/Qwen3-235B-A22B`, `temperature=0.0`, `top_p=1.0`,
  `max_tokens=1024`, `LIMIT=1000`. It reached vLLM readiness, completed
  Qwen3-235B TP=4 model loading (`118/118` shards, about `383.55s`,
  `109.55 GiB` model memory, `1,212,048` KV-cache tokens), and is generating
  teacher records. As of 2026-06-05 01:01 PDT it had written `64/1000` output
  records, so the 1K run is likely too slow for a fast smoke.
- To validate the downstream trainer without waiting for the full 1K teacher
  run, a partial-data PARD K5 training smoke was submitted as job `3170163`
  using `TRAIN_RECORD_LIMIT=20`, `para_num=5`, and PARD token id `151670`.
  That smoke exposed a wrapper bug: the official `pard.train` entrypoint calls
  `accelerate` through `os.system`, so an internal training failure returned a
  Slurm success code. It also showed that global requirement installation can
  corrupt the container's `numpy/accelerate` import state.
- The training wrapper was patched to use an isolated `--system-site-packages`
  venv, remove the trailing comma in `CUDA_VISIBLE_DEVICES`, and call
  `accelerate launch` directly so failures propagate as job failures. Retry
  job `3170284` used `TRAIN_RECORD_LIMIT=40`.
- Job `3170284` was cancelled after the `accelerate>=1.2.0` force install
  started pulling a new PyTorch/CUDA stack into the venv. The wrapper now
  force-installs only `numpy==1.26.4 --no-deps` and invokes global/system
  `accelerate` through the venv Python module path.
- Job `3170322` then failed quickly with an optimizer type error because the
  generated YAML wrote `learning_rate: 1e-05`, which entered the trainer as a
  string. The config writer now emits decimal YAML floats for LR and related
  float fields.
- The PARD math K5 train smoke job `3170398` completed successfully using the
  current partial teacher output (`TRAIN_RECORD_LIMIT=64`), `para_num=5`,
  Qwen3 PARD token id `151670`, one node / 4 GPUs. It ran one train step in
  about 25.6s, reported `train_loss=1.789607`, and produced
  `checkpoint-1/model.safetensors`. This validates the external math-data
  teacher-distillation path end to end; it is not a quality result yet because
  the train set is only 64 examples.
- Teacher generation is now the main data bottleneck. The original job
  `3169844` uses `GENERATION_CONCURRENCY=4`, `VLLM_MAX_NUM_SEQS=4`, offset 0,
  and had produced `78/1000` records as of 2026-06-05 01:06 PDT.
- A second teacher chunk, job `3170473`, was submitted with offset 1000 and a
  more aggressive serving shape: `GENERATION_CONCURRENCY=16`,
  `VLLM_MAX_NUM_SEQS=16`, `VLLM_MAX_NUM_BATCHED_TOKENS=32768`. This is a
  throughput experiment to compare against the conservative c4 run while also
  growing the external math teacher set.
- The c16 teacher job `3170473` reached generation and had produced `65`
  records as of 2026-06-05 01:27 PDT. The conservative c4 job `3169844` had
  produced `126` records at the same time.
- Two additional c16 teacher chunks were submitted to scale the teacher data:
  job `3170701` for offset 2000 and job `3170702` for offset 3000. Both use
  the same c16 serving shape as `3170473`.
- The PARD teacher-to-train converter and train wrapper now support multiple
  teacher JSONL inputs, so future training can combine offset chunks via
  `TEACHER_DATA='file_offset0.jsonl file_offset1000.jsonl ...'`.
- Multi-chunk K5 train job `3170705` was submitted with
  `TEACHER_DATA=offset0 + offset1000_c16`, `TRAIN_RECORD_LIMIT=180`, and
  `EXP_NAME=qwen235b_math_k5_teacher_chunks_partial180`. This validates that
  the multi-input converter/training path works on cluster data before waiting
  for full 1K chunks.
- Job `3170705` completed successfully. The converter wrote `212` train rows
  from the two teacher files, the trainer consumed first `180` rows, ran
  `2` update steps, reported `train_loss=1.633825`, and produced
  `checkpoint-2/model.safetensors` under
  `.../PARD-Qwen3-0.6B_qwen235b_math_k5_teacher_chunks_partial180/`.
- A vLLM standalone OpenMath compatibility gate for the 64-record smoke
  checkpoint first failed as job `3170585` because the submit wrapper passed
  `draft_tensor_parallel_size=1` while Qwen3-235B target `TP=4`. vLLM 0.17
  requires draft TP and target TP to match for `draft_model` speculative
  decoding.
- The PARD gate wrapper now defaults `DRAFT_TP=${TP}`. Retry job `3170614`
  uses `draft_tensor_parallel_size=4`, `K=5`, `parallel_drafting=true`,
  `ISL=1024`, `OSL=1024`, `batch_sizes=32`, and the local smoke checkpoint
  `.../PARD-Qwen3-0.6B_qwen235b_math_k5_teacher_partial64_venv3/checkpoint-1`.
  This is not intended as a quality result; it checks whether a locally trained
  PARD checkpoint can be loaded by vLLM and emit acceptance/throughput metrics.
- Job `3170614` completed successfully. It proves that a locally trained PARD
  checkpoint can be used by vLLM as a parallel drafter for Qwen3-235B when
  `draft_tensor_parallel_size=4`. Batch 32 OpenMath results:
  `524.99 tok/s/GPU`, acceptance `46.32%`, mean acceptance length `3.316`.
  Against the OpenMath baseline reference `484.09 tok/s/GPU`, this is about
  `1.084x`; against the public PARD K5 reference `635.65 tok/s/GPU`, it is
  about `0.826x`. This is expected for a 64-record smoke checkpoint and is a
  compatibility/integration result, not a final quality result.
- The `180`-record multi-chunk checkpoint gate, job `3170854`, completed with
  the same OpenMath batch-32 vLLM standalone shape. Results:
  `520.98 tok/s/GPU`, acceptance `46.34%`, mean acceptance length `3.317`.
  Against the OpenMath baseline reference `484.09 tok/s/GPU`, this is
  `1.076x`; against the public PARD K5 reference `635.65 tok/s/GPU`, it is
  `0.820x`. The important conclusion is negative but useful: 180 examples
  do not materially improve OpenMath acceptance over the 64-record smoke
  checkpoint or the public PARD K5 checkpoint.
- The next scale-up train is partial450. Early attempts exposed filesystem
  quota issues: creating new run directories under `${ARTIFACT_ROOT}/runs`
  failed, and reusing the partial180 venv while `INSTALL_PARD_REQUIREMENTS=true`
  retried a `numpy` reinstall and hit quota during rollback. The train wrapper
  now selects an existing venv whenever `USE_VENV=true`, even if requirement
  installation is disabled, and sets `PYTHONDONTWRITEBYTECODE=1` for the train
  launch. Retry job `3171014` was submitted with
  `RUN_DIR_ROOT=${ARTIFACT_ROOT}/train_runs_tmp`,
  `INSTALL_PARD_REQUIREMENTS=false`, and the partial180 venv reused directly.
- Retry job `3171014` completed successfully. It trained on `450` records for
  `7` steps, reported `train_loss=1.549805`, and produced
  `.../PARD-Qwen3-0.6B_qwen235b_math_k5_teacher_chunks_partial450_retry_venvreuse_nopip/checkpoint-7`.
  The OpenMath batch-32 vLLM gate for this checkpoint was submitted as job
  `3171063`.
- Gate job `3171063` completed, and the result regressed:
  `419.06 tok/s/GPU`, acceptance `44.75%`, mean acceptance length `3.237`.
  Against the OpenMath baseline reference this is only `0.866x`; against the
  public PARD K5 reference it is `0.659x`. This means the simple 450-record
  target-token distillation checkpoint is worse than both the 180-record
  checkpoint and the public `amd/PARD-Qwen3-0.6B` K5 drafter on held-out
  OpenMath.
- Do not blindly scale this exact objective from tiny teacher chunks. The
  observed pattern is: 64 records validates compatibility, 180 records leaves
  acceptance nearly unchanged, 450 records regresses held-out OpenMath. The next
  training attempt should either use a much larger and more diverse teacher set
  with a held-out validation gate, mix in generic/PARD source-domain data to
  avoid forgetting, or switch the objective toward PARD-2-style
  acceptance-length/CAT weighting rather than plain target-token CE.
- A PARD-2-style prefix-reward trainer has been added as
  `experiments/pard_qwen3_235b_math/pard_train_prefix_weighted.py`. It keeps
  the public PARD data path and default model loss, but masks labels by draft
  position so earlier positions are trained more often. For K5
  `prefix_reward`, the label keep probabilities are `[1.0, 0.8, 0.6, 0.4,
  0.2]`, matching the intuition that earlier mistakes truncate all later
  accepted-prefix reward.
- A manual per-token weighted CE implementation was tried first, but smoke jobs
  produced abnormally high loss around `22.95`; do not reuse that path. The
  trainer was changed back to the default model/SFT loss with label masking.
  Sanity checks on the same 128 records:
  official PARD trainer job `3171262` reported `train_loss=1.702830`;
  custom uniform-mask job `3171283` reported `train_loss=1.702978`; and
  custom prefix-reward-mask job `3171292` reported `train_loss=1.464686`.
  This validates that the custom path now matches the official path when
  masking is disabled.
- Scale-up prefix-reward jobs `3171324`, `3171341`, `3171381`, `3171402`,
  `3171406`, `3171425`, and a node-pinned retry `3171459` failed before
  trainer stdout with Slurm `RaisedSignal:53`. This was not a trainer loss
  failure. The same period also showed a teacher job failing only while writing
  a summary JSON with `Disk quota exceeded`, so the likely cause was
  fsw-artifact/cache quota pressure during container startup.
- The quota workaround is to move training artifacts, container cache,
  checkpoint output, HF cache, and logs to fs1 while still reading the existing
  fsw teacher JSONL chunks. This fixed the scale-up path: job `3171476`
  completed a K5 prefix-reward run on `1024` records from four teacher chunks
  in `16` steps with `train_loss=1.390633`. The checkpoint is
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_prefix_reward_1024_lr3e6_4chunk_fs1/checkpoint-16`.
- The OpenMath batch-32 vLLM gate for that prefix-reward checkpoint completed
  as job `3171517` with `ISL=1024`, `OSL=1024`, `TP=4`,
  `draft_tensor_parallel_size=4`, K5, and `parallel_drafting=true`. Result:
  `505.69 tok/s/GPU`, acceptance `45.45%`, mean acceptance length `3.272`,
  speedup `1.045x` vs the OpenMath baseline, and `0.796x` throughput ratio vs
  the public PARD K5 checkpoint. This does not improve the public PARD K5
  OpenMath result (`635.65 tok/s/GPU`, `45.51%` acceptance).
- Conclusion from the 1024-record prefix-reward run: the runnable PARD-2-style
  label-mask approximation is stable and avoids the plain-CE 450-record
  regression, but it is not yet a quality win. The next useful training change
  is not another small scale-up of the same label-mask objective; it should
  add true CAT/target-confidence weighting, mix public PARD/source-domain data
  to avoid forgetting, or collect target logits for a closer PARD-2 objective.
- A conservative forgetting-control sweep was started by linearly
  interpolating public `amd/PARD-Qwen3-0.6B` with the 1024-record prefix-reward
  checkpoint. Merge job `3171724` produced alpha `0.10`, `0.25`, and `0.50`
  checkpoints under
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/interpolated_public_pard_prefix_reward_1024/`.
  OpenMath bs32 gates completed as `3171764`, `3171765`, and `3171767`.
  Results:

| Checkpoint | Job | tok/s/GPU | Acceptance | Mean accepted length | Speedup vs baseline | Ratio vs public PARD K5 |
|---|---:|---:|---:|---:|---:|---:|
| merge alpha 0.10 | `3171764` | `528.28` | `46.55%` | `3.328` | `1.091x` | `0.831x` |
| merge alpha 0.25 | `3171765` | `448.89` | `45.78%` | `3.289` | `0.927x` | `0.706x` |
| merge alpha 0.50 | `3171767` | `497.07` | `46.28%` | `3.314` | `1.027x` | `0.782x` |

  The conservative alpha `0.10` merge is the best local checkpoint so far, and
  it raises acceptance slightly above public PARD K5 (`46.55%` vs `45.51%`).
  It still fails the throughput gate because it reaches only `528.28 tok/s/GPU`
  versus public PARD K5 `635.65 tok/s/GPU`. This means simple interpolation
  reduces forgetting compared with direct fine-tuning, but it does not recover
  the public checkpoint's runtime quality.
- A same-harness public PARD K5 recalibration was submitted as job `3171868`
  because the local checkpoints showed similar or slightly better acceptance
  than the historical public PARD row but much lower throughput. The
  recalibration completed with `484.48 tok/s/GPU`, acceptance `45.99%`, mean
  acceptance length `3.299`, and per-position acceptance
  `[75.33%, 55.43%, 41.46%, 32.21%, 25.52%]`.
  Against the OpenMath baseline reference `484.09 tok/s/GPU`, this is only
  `1.001x`; against the historical public PARD K5 row `635.65 tok/s/GPU`, it
  is `0.762x`. This means the historical public-PARD throughput reference is
  not reproducible under the current harness/node-era run, even though the
  acceptance rate is similar. The local alpha `0.10` checkpoint now compares
  as `528.28 / 484.48 = 1.090x` versus the current public-PARD recalibration,
  while still needing a repeated paired baseline/spec run before claiming a
  durable checkpoint improvement.
- Current teacher data counts as of 2026-06-05 03:31 PDT:
  offset0 c4 `427`, offset1000 c16 `1000`, offset2000 c16 `1000`, offset3000
  c16 `1000` records. Job `3170702` is Slurm-failed because of a final summary
  write quota error, but its 1000-row output and validation are usable.
- Local-checkpoint gate metrics are tracked in
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.csv`.

## Recent Method Shortlist

The immediate path remains PARD/PARD-2 because vLLM can already run PARD-style
`draft_model` speculative decoding with `parallel_drafting=true` for
Qwen3-235B. PARD's public paper frames the method as low-cost parallel draft
adaptation, using one draft forward pass to predict multiple future tokens:
https://arxiv.org/abs/2504.18583.

PARD-2 is the most directly relevant next objective. Its paper argues that
plain draft-token prediction is not aligned with the inference-time goal of
maximizing consecutive accepted length, and introduces Confidence-Adaptive
Token optimization: https://arxiv.org/abs/2605.08632. The local
`prefix_reward` label-mask trainer is only a runnable approximation of that
idea; a true PARD-2/CAT implementation should replace it once code details are
available or once we collect target logits/confidence labels.

Other 2025-2026 candidates are useful context but are not the shortest path for
this Qwen3-235B gate:

- DREAM-R uses RL-style drafter alignment for reasoning trajectories
  (SAPO/TBVM/FPSR): https://arxiv.org/abs/2605.28678. It supports the diagnosis
  that reasoning-domain drafter alignment matters, but it is not currently a
  drop-in vLLM/NeMo-RL text-generation drafter path.
- TreeFlash addresses the distribution mismatch of one-shot block/tree
  drafters by approximating autoregressive conditioning:
  https://arxiv.org/abs/2606.03819. This is relevant if PARD-2 still loses
  acceptance at larger K, but it would require a new drafter/runtime path.
- P-EAGLE parallelizes EAGLE-style drafting and scales long-sequence training:
  https://arxiv.org/abs/2602.01469. It is promising for long reasoning outputs,
  but Qwen3-235B has no ready public P-EAGLE drafter in the current experiment
  stack.

Training-track artifacts:

- `experiments/pard_qwen3_235b_math/README.md`
- `experiments/pard_qwen3_235b_math/prepare_math_prompt_pool.sh`
- `experiments/pard_qwen3_235b_math/generate_qwen235b_teacher_math_continuations.sh`
- `experiments/pard_qwen3_235b_math/submit_teacher_math_continuations.sh`
- `experiments/pard_qwen3_235b_math/convert_teacher_to_pard_jsonl.py`
- `experiments/pard_qwen3_235b_math/make_pard_train_config.py`
- `experiments/pard_qwen3_235b_math/pard_train_prefix_weighted.py`
- `experiments/pard_qwen3_235b_math/train_pard_math_k5.sh`
- `experiments/pard_qwen3_235b_math/submit_pard_math_k5_train.sh`
- `experiments/pard_qwen3_235b_math/submit_trained_pard_openmath_gate.sh`
- `scripts/poll_qwen235b_pard_math_teacher_status.sh`

## NeMo-RL Follow-Up: Worker32 Generation-Only Gate

Submitted Qwen3-235B PARD generation-only NeMo-RL jobs using the worker32
shape. These jobs stop after generation, so they test rollout generation time
before paying full GRPO training cost.

The first submission (`3165313`, `3165314`, `3165315`) failed before the
driver started the workload. Root cause was a launcher mismatch: this
`origin-main-online-20260603` checkout requires Python `>=3.13.13,<3.14`, but
the wrapper forced `UV_PYTHON=3.12.13`.

The second submission (`3166006`, `3166007`, `3166008`) fixed Python/Ray and
reached driver + generation-worker environment creation. It was cancelled
intentionally because config inspection showed `policy.generation.max_new_tokens`
was still `8192` with sampling `temperature=1.0`; that would not match the
standalone fixed `OSL=1024` comparison.

The third submission (`3166955`, `3166956`, `3166957`) had the right fixed
decode overrides, but baseline/K5 failed during driver venv creation with a
Lustre `Disk quota exceeded` error while installing `torch`. K12 was cancelled
before it could hit the same failure. The failed fixed1024 r3 driver venvs were
removed.

The active retry reuses the already-built py313 r2 driver venvs, keeps
driver/Ray on Python `3.13.13` and Ray `2.54.0`, and sets
`max_new_tokens=1024`, `min_tokens=1024`, `ignore_eos=true`, `temperature=0.0`,
`top_p=1.0`, and `top_k=-1`.

| Run | Active job | Status | SpecDec config |
|---|---:|---|---|
| Baseline | `3167916` | FAILED | disabled |
| Baseline auto-load retry | `3168242` | CANCELLED | disabled, `NRL_VLLM_FORCE_LOAD_FORMAT=auto` |
| PARD K=5 always-on | `3167917` | FAILED | `method=draft_model`, `model=amd/PARD-Qwen3-0.6B`, `num_speculative_tokens=5`, `parallel_drafting=true` |
| PARD K=12 always-on | `3167918` | CANCELLED | `method=draft_model`, `model=amd/PARD-Qwen3-0.6B`, `num_speculative_tokens=12`, `parallel_drafting=true` |
| Baseline Triton-MoE retry | `3168274` | RUNNING | disabled, `NRL_VLLM_FORCE_LOAD_FORMAT=auto`, `kernel_config.moe_backend=triton` |
| PARD K=5 Triton-MoE retry | `3168275` | RUNNING | `method=draft_model`, `num_speculative_tokens=5`, `parallel_drafting=true`, `kernel_config.moe_backend=triton` |
| PARD K=12 Triton-MoE retry | `3168276` | FAILED | launcher/Ray bring-up failed before driver; no generation result |

Latest observed status at 2026-06-04 21:56 PDT:

- The original baseline `3167916` failed at vLLM model load before generation.
  It used NeMo-RL's train-time default `load_format=dummy`, then vLLM's
  FlashInfer TRTLLM Unquantized MoE backend hit
  `AssertionError: K must be divisible by blockK` while converting Qwen3-235B
  MoE weights.
- K5 `3167917` also failed with the same assertion after loading real
  safetensors weights through the `load_format=auto` path. This confirms the
  issue is not only dummy weights; it is vLLM's selected MoE backend for this
  Qwen3-235B shape.
- K12 `3167918` began showing the same error class and was cancelled, as was
  baseline-auto `3168242`, to avoid spending 64 nodes on a known-failing
  backend path.
- The remote checkout now has a small env hook:
  `NRL_VLLM_FORCE_LOAD_FORMAT=auto` overrides the final
  `vllm_cfg.load_format`.
- A new retry set forces
  `++policy.generation.vllm_kwargs.kernel_config.moe_backend=triton`, keeps
  `load_format=auto`, and reuses the existing actor venv suffixes where
  possible. Baseline `3168274` and K5 `3168275` are running; K12 `3168276`
  failed during Ray bring-up before the driver workload started and should be
  resubmitted after baseline/K5 validate the Triton-MoE path.
- `sacct` for K12 `3168276` shows the batch job failed in about two minutes.
  The Ray head log exists, but no `ray-driver.log` was created; this is a
  launcher/Ray-worker bring-up failure, not a vLLM model-load or speculative
  decoding result.
- `MasterConfig` for `3168274` and `3168275` confirms
  `vllm_kwargs.kernel_config.moe_backend='triton'`; no generation metric has
  appeared yet.
- The K5 vLLM engine log for `3168275` confirms the runtime config all the way
  through engine initialization: vLLM `0.20.0`, `dtype=bfloat16`,
  `load_format=auto`, `tensor_parallel_size=16`, `max_seq_len=8192`,
  `max_num_seqs=32`, `max_num_batched_tokens=32768`,
  `parallel_drafting=true`, and `kernel_config.moe_backend='triton'`.
- The same K5 log also confirms the selected MoE backend is now
  `Using TRITON Unquantized MoE backend`, not FlashInfer TRTLLM. This is the
  key workaround for the previous `K must be divisible by blockK` assertion.
- Baseline `3168274` also reached the Triton MoE path and CUDA graph capture.
  It has no fatal errors and has not emitted generation metrics yet.
- K5 `3168275` has no fatal errors. The only observed error-level line is a
  transient Hugging Face repo file-list retrieval failure
  (`Connection reset by peer, retrying 1 of 2`); the job continued into CUDA
  graph capture afterward.
- The driver venv issue is past the previous Lustre quota failure point. The
  active retry reuses the existing py313 r2 driver venvs.
- `MasterConfig` has been emitted for the original three jobs, confirming fixed decode
  (`max_new_tokens=1024`, `temperature=0.0`, `top_p=1.0`, `top_k=-1`),
  generation `TP=16`, generation `PP=1`, generation `EP=1`,
  training `TP=2`, `PP=8`, `CP=2`, `EP=16`, and
  `generation_batch_size=32`.
- Generation metrics have not appeared yet.

Shape:

- NeMo-RL checkout: `SpecDec-RL-origin-main-online-20260603`
- vLLM dependency in that checkout: vLLM `0.20.0` wheels
- Driver/Ray Python: `3.13.13`
- Ray runtime: `2.54.0`
- Target: `Qwen/Qwen3-235B-A22B`
- Nodes/GPUs: `32` nodes x `4` GB200 GPUs
- Generation TP/PP/EP: `TP=16`, `PP=1`, `EP=1`
- Generation DP: `128 / 16 = 8` vLLM generation engines
- Samples: `num_prompts=8`, `num_generations_per_prompt=32`, total `256`
- Effective requests per generation engine: `256 / 8 = 32`
- Fixed decode controls: `max_new_tokens=1024`, `min_tokens=1024`, `ignore_eos=true`, stop strings/token ids disabled, `temperature=0.0`, `top_p=1.0`, `top_k=-1`
- Online draft training disabled: `policy.draft.enabled=false`

Submission artifacts:

- Wrapper: `experiments/eagle3_online/submit_qwen235b_pard_tritonmoe_gbs256_worker32_step1.sh`
- Polling script: `scripts/poll_qwen235b_pard_nemorl_status.sh`
- Status file: `latest_qwen235b_pard_nemorl_gbs256_worker32_jobs.txt`
- Retry status file: `latest_qwen235b_pard_nemorl_gbs256_worker32_jobs_retry2.txt`
- Fixed-1024 retry status file: `latest_qwen235b_pard_nemorl_gbs256_worker32_fixed1024_jobs.txt`
- Fixed-1024 r2-venv reuse status file: `latest_qwen235b_pard_nemorl_gbs256_worker32_fixed1024_reuse_r2_jobs.txt`
- Baseline auto-load retry status file: `latest_qwen235b_pard_nemorl_gbs256_worker32_baseline_auto_fixed1024_jobs.txt`
- Triton-MoE retry status file: `latest_qwen235b_pard_nemorl_gbs256_worker32_tritonmoe_fixed1024_jobs.txt`
- Triton-MoE tokenizer-fix retry status file: `latest_qwen235b_pard_nemorl_gbs256_worker32_tritonmoe_tokenizerfix_jobs.txt`
- Triton-MoE Megatron-tokenizer-fix retry status file:
  `latest_qwen235b_pard_nemorl_gbs256_worker32_tritonmoe_megatron_tokenizerfix_jobs.txt`

Lightweight NeMo-RL generation-backend follow-up:

- A direct `VllmGeneration`-only gate was added to avoid Megatron/GRPO setup
  while still exercising NeMo-RL's generation backend:
  `experiments/eagle3_qwen3_235b/run_nemo_vllm_generation_acceptance.py` and
  `experiments/pard_qwen3_235b_math/submit_pard_nemorl_vllmgeneration_gate.sh`.
  The script now supports batched prompts, `SPECDEC_METHOD=draft_model`,
  `parallel_drafting=true`, `draft_tensor_parallel_size=4`,
  `max_num_batched_tokens`, and `kernel_config.moe_backend`.
- Direct gate shape: one node / four GB200 GPUs, Qwen3-235B `TP=4`, OpenMath
  `prompt_limit=32`, `generation_batch_size=32`, `max_new_tokens=1024`,
  greedy decoding, public `amd/PARD-Qwen3-0.6B` K5 for the spec run.
- First direct gate attempt `3171924`/`3171925` reached Ray head startup but
  failed in the driver import path because `/opt/venv/bin/python` lacked
  `decord`, imported indirectly by `nemo_rl.distributed.batched_data_dict`.
- Second attempt `3171931`/`3171932` tried `uv run --python 3.13.13 --extra
  mcore`, but the container-side uv could not parse the checkout's newer
  `[tool.uv.extra-build-dependencies]` field and could not find Python
  `3.13.13`.
- Third attempt `3171952`/`3171953` tried `/opt/nemo_rl_venv/bin/python`, but
  that path does not exist in the container used by this Ray submit path.
- Fourth attempt `3171975`/`3171976` returned to `/opt/venv/bin/python` with
  a text-only `decord` stub, but failed before NeMo-RL import completed because
  `transformers` calls `importlib.util.find_spec("decord")` and the simple
  stub had `decord.__spec__ = None`.
- The local direct-gate script now sets a real `ModuleSpec("decord",
  loader=None)` on the text-only stub before importing NeMo-RL. This keeps the
  unused video dependency unavailable while satisfying the Transformers
  optional-dependency probe.
- Fifth attempt `3172012`/`3172013` was submitted with that fix at `2026-06-05
  05:04 PDT`. It uses the same shape: baseline and public PARD K5, one node /
  four GB200 GPUs, Qwen3-235B `TP=4`, `prompt_limit=32`,
  `generation_batch_size=32`, `max_new_tokens=1024`, `max_model_len=4096`,
  `max_num_batched_tokens=131072`, `temperature=0.0`, `top_p=1.0`,
  `top_k=-1`, Triton MoE, and vLLM `draft_model` PARD with
  `parallel_drafting=true` for the spec run. It passed the `decord` optional
  dependency probe but failed at the next unused logging dependency:
  `ModuleNotFoundError: No module named 'mlflow'` while importing
  `nemo_rl.utils.logger`.
- The local direct-gate script now also installs a text-only `mlflow` import
  stub before importing NeMo-RL. The stub intentionally raises if logger
  methods are actually used; it only avoids import-time failure in this
  generation-only gate.
- Sixth attempt `3172019`/`3172020` used the first `mlflow` stub, but the stub
  incorrectly handled dunder/module metadata. Torch import hit
  `AttributeError: '_UnavailableOptionalDependency' object has no attribute
  'endswith'` through Python `inspect`. The stub was fixed to set `__file__`
  and raise `AttributeError` for dunder attributes.
- Seventh attempt `3172032`/`3172033` passed the Torch import path but then hit
  `ModuleNotFoundError: No module named 'swanlab'` through the same
  `nemo_rl.algorithms.utils -> nemo_rl.utils.logger` import chain.
- The script now avoids that logger chain entirely by replacing
  `nemo_rl.algorithms.utils.get_tokenizer()` with a local
  `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` helper plus
  pad-token setup. This direct gate does not need algorithm-level logging
  utilities.
- Eighth attempt `3172078`/`3172079` got past driver tokenization and connected
  to the Ray cluster, but worker creation failed because the `decord` stub only
  existed in the driver process; Ray workers imported
  `nemo_rl.data.multimodal_utils` in separate processes and hit
  `ModuleNotFoundError: No module named 'decord'`.
- A real importable stub file was added at
  `experiments/eagle3_qwen3_235b/text_only_stubs/decord.py`, and the submit
  wrapper now prepends that directory to `PYTHONPATH` and copies it to the
  remote roadmap tree before submitting. This should make the text-only decord
  stub visible in both driver and Ray worker processes.
- Ninth attempt `3172102`/`3172103` passed Ray worker creation and reached
  `VllmAsyncGenerationWorker.__init__`, then failed with
  `ModuleNotFoundError: No module named 'vllm'`. Root cause: this PARD wrapper
  moved `ARTIFACT_ROOT` to fs1, so the generic submit script derived a missing
  source-built vLLM path under
  `.../qwen3_235b_pard_math_artifacts/python_site/...`.
- The wrapper now explicitly passes the known source-built vLLM site:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_10_2_cu129_torch28nv_source_py312`.
- Tenth attempt `3172131`/`3172132` imported vLLM from that 0.10.2 source site,
  but NeMo-RL's current vLLM worker patch expects the vLLM V1 layout and failed
  because `vllm/v1/executor/ray_executor.py` is absent in the 0.10.2 site.
- The direct-gate wrapper now uses the existing source-built vLLM 0.17 site
  instead:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_17_0_extract_py312`.
  That site contains `vllm/v1/executor/ray_executor.py` and has a
  `.vllm_bootstrap_spec` source-build marker
  `git+https://github.com/vllm-project/vllm.git@v0.17.0|source-build|torch28nv|repaired-tokenizers`.
- Attempts `3171924` through `3172132` are all driver environment, local stub,
  worker import, logger-import, missing vLLM-site, or vLLM-version-layout
  failures, not PARD acceptance or throughput results.
- Eleventh attempt `3172149`/`3172150` used the vLLM 0.17 source site and got
  through NeMo-RL worker creation, vLLM V1 patching, and Qwen3-235B model
  loading. Both baseline and PARD then failed during engine initialization /
  CUDA graph capture in the `FLASH_ATTN` path with
  `ModuleNotFoundError: No module named 'cutlass'` from
  `vllm.vllm_flash_attn.cute`.
- The wrapper now defaults the direct gate to
  `VLLM_ATTENTION_BACKEND=FLASHINFER` to avoid the `cutlass.cute` FlashAttention
  path. This is still an infrastructure/backend issue, not a PARD acceptance or
  throughput result.
- Twelfth attempt `3172206`/`3172207` still failed before metrics with the same
  `cutlass` import path. The important root cause is now narrower: the Slurm
  environment had `VLLM_ATTENTION_BACKEND=FLASHINFER`, but the direct-gate
  driver did not pass `attention_backend` into `vllm_kwargs`, so vLLM
  `AsyncEngineArgs` auto-selected `FLASH_ATTN` and imported
  `vllm.vllm_flash_attn.cute`. This is not a PARD result.
- The driver is patched so `generation_config()` now sets
  `vllm_kwargs["attention_backend"] = args.attention_backend`. The PARD wrapper
  also no longer hardcodes `VLLM_ENFORCE_EAGER=false`; it forwards
  `VLLM_ENFORCE_EAGER`, `VLLM_COMPILATION_LEVEL`, `VLLM_CUDAGRAPH_MODE`,
  `VLLM_CUDAGRAPH_CAPTURE_SIZES`, and `VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE` to the
  generic submit script.
- Backend-fix graph-on jobs `3172267`/`3172268` were submitted to test whether
  forcing `attention_backend=FLASHINFER` at the vLLM args level is sufficient.
- FlashInfer eager sanity-gate jobs `3172270`/`3172271` were submitted with
  `VLLM_ENFORCE_EAGER=true`, `VLLM_COMPILATION_LEVEL=0`, and
  `VLLM_CUDAGRAPH_MODE=NONE`. These are intended to prove that the NeMo-RL
  generation backend can execute baseline and public PARD K5 without CUDA
  graph capture. If they pass, their throughput is a sanity metric only; the
  comparable performance run still needs graph-on validation.
- Current direct-gate status files:
  `latest_qwen235b_pard_nemorl_vllmgeneration_gate_jobs.txt` for graph-on and
  `latest_qwen235b_pard_nemorl_vllmgeneration_gate_eager_jobs.txt` for the
  eager sanity gate.
- `3172267` failed quickly and proved the backend patch is active: vLLM now
  tries `AttentionBackendEnum.FLASHINFER`, but that backend is invalid in the
  current vLLM 0.17 site because the `flashinfer` Python package is missing.
  The remaining FlashInfer jobs `3172268`, `3172270`, and `3172271` were
  cancelled to avoid repeating a known-invalid backend.
- The next viable backend is `TRITON_ATTN`. Graph-on `TRITON_ATTN` jobs
  `3172289`/`3172290` are submitted for comparable timing. `TRITON_ATTN` eager
  sanity jobs `3172291`/`3172292` are also submitted to isolate any remaining
  CUDA graph issue from basic NeMo-RL/vLLM execution.
- Updated direct-gate status files:
  `latest_qwen235b_pard_nemorl_vllmgeneration_gate_triton_jobs.txt` and
  `latest_qwen235b_pard_nemorl_vllmgeneration_gate_triton_eager_jobs.txt`.
- `3172289`, `3172291`, and `3172292` reached Qwen3-235B model loading,
  selected `Using TRITON backend for Unquantized MoE`, computed KV capacity,
  and then failed in the direct-gate driver with
  `AssertionError: generate_async is restricted to handle only single samples,
  but received batch_size=32`. This is a driver batching bug, not a vLLM/PARD
  runtime result.
- The direct-gate driver now treats `generation_batch_size` as async
  concurrency: it launches up to 32 concurrent single-sample
  `policy.generate_async()` calls instead of passing a 32-row tensor into one
  worker call. This preserves bs32 scheduler pressure while satisfying the
  NeMo-RL worker API.
- Asyncfix `TRITON_ATTN` graph-on jobs `3172338`/`3172339` and asyncfix
  `TRITON_ATTN` eager jobs `3172341`/`3172342` were submitted. Status files:
  `latest_qwen235b_pard_nemorl_vllmgeneration_gate_triton_asyncfix_jobs.txt`
  and
  `latest_qwen235b_pard_nemorl_vllmgeneration_gate_triton_eager_asyncfix_jobs.txt`.
- Asyncfix result: all four jobs passed. This is the first clean Qwen3-235B
  direct NeMo-RL `VllmGeneration` result for public PARD K5 on OpenMath bs32.
  This is generation-backend only, not full GRPO E2E.

| Mode | Job | Drafter | Generation tok/s | Speedup vs matched baseline | Acceptance | Mean accepted length |
|---|---:|---|---:|---:|---:|---:|
| Graph-on | `3172338` | none | `1240.43` | `1.000x` | `0.00%` | `1.00` |
| Graph-on | `3172339` | `amd/PARD-Qwen3-0.6B` K5 | `1487.93` | `1.200x` | `46.81%` | `3.34` |
| Eager | `3172341` | none | `274.51` | `1.000x` | `0.00%` | `1.00` |
| Eager | `3172342` | `amd/PARD-Qwen3-0.6B` K5 | `548.32` | `1.997x` | `46.45%` | `3.32` |

Interpretation:

- The NeMo-RL generation backend can run Qwen3-235B public PARD K5 with
  `method=draft_model`, `parallel_drafting=true`, `draft_tp=4`, target `TP=4`,
  and OpenMath bs32-shaped async concurrency.
- Graph-on is the performance-relevant setting and shows a real but modest
  `1.20x` generation throughput speedup.
- Eager is a sanity setting and shows PARD is functionally useful when CUDA
  graph/compile acceleration is removed: `2.00x` generation throughput speedup.
  Do not compare eager throughput directly to graph-on throughput.
- Acceptance is about `46%-47%`, matching the earlier standalone OpenMath K5
  acceptance range. The remaining gap to synthetic PARD speedups is therefore
  still acceptance/domain plus runtime-overhead dependent, not a disabled
  SpecDec path.
- Metrics CSV:
  `docs/qwen3_235b_nemorl_direct_vllmgeneration_metrics_20260605.csv`.

Latest NeMo-RL retry outcome:

- The Triton-MoE retry jobs `3168274` and `3168275` got past the previous
  vLLM MoE backend failure, reached generation worker initialization, and
  selected the intended Triton MoE backend.
- They then failed before generation metrics were emitted because many
  Megatron policy workers tried to load the tokenizer from
  `Qwen/Qwen3-235B-A22B` and hit Hugging Face API `429 Too Many Requests`.
  This is a tokenizer/model-info lookup failure, not a speculative decoding
  runtime result.
- The first tokenizer-fix retry (`3168570`, `3168571`, `3168572`) proved that
  `policy.tokenizer.name` was set correctly, but baseline `3168570` still
  failed during `MegatronPolicyWorker` creation. Root cause:
  `nemo_rl/models/megatron/setup.py::finalize_megatron_setup()` hardcoded
  `TokenizerConfig(tokenizer_model=hf_model_name)`, so Megatron-Bridge still
  called `AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")`. The
  installed `transformers/tokenizers` path then called
  `huggingface_hub.model_info()` while patching tokenizer regex behavior and
  hit Hugging Face API `429 Too Many Requests`. This is an infrastructure
  tokenizer/model-info lookup failure, not a speculative decoding result.
- The active remote checkout is patched so both Megatron tokenizer construction
  sites read `NRL_MEGATRON_TOKENIZER_MODEL`, defaulting to the old
  `hf_model_name` behavior when the env var is absent. The generic submit
  wrapper exports `NRL_MEGATRON_TOKENIZER_MODEL=${TOKENIZER_NAME}`, and the
  Qwen3-235B wrapper pins it to the local HF snapshot:
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65`.
- The flawed `3168570`/`3168571`/`3168572` retry was cancelled and first
  superseded by fresh actor-venv-suffix `r5` jobs. That `r5` retry is also now
  superseded. The active retry is the `r4` actor-venv reuse set:

| Run | Active job | Status | SpecDec config |
|---|---:|---|---|
| Baseline | `3169284` | FAILED, no metrics | disabled |
| PARD K=5 always-on | `3169285` | FAILED, no metrics | `method=draft_model`, `num_speculative_tokens=5`, `parallel_drafting=true` |
| PARD K=12 always-on | `3169286` | FAILED, no metrics | `method=draft_model`, `num_speculative_tokens=12`, `parallel_drafting=true` |

- The `r4` actor-venv reuse retry was baseline `3169284`, K5 `3169285`, and
  K12 `3169286`. All three ultimately failed during generation before metrics.
- The `r5` retry (`3169177`, `3169178`, `3169179`) was superseded before any
  performance metric. Baseline `3169177` failed during generation actor venv
  creation because `uv sync` failed to fetch the direct `flash-attn` wheel from
  GitHub after three retries (`http2` refused stream). K5/K12 were cancelled to
  avoid spending nodes on the same fresh-venv path.
- The active retry reuses the previously built `r4` actor venv suffixes while
  keeping the new `NRL_MEGATRON_TOKENIZER_MODEL` code/env fix. This should
  avoid the fresh GitHub `flash-attn` fetch that killed the `r5` baseline.
- The `r5` baseline/K12 launch evidence confirmed the intended fixed-generation
  shape before the actor-venv fetch failure:
  `max_new_tokens=1024`, `temperature=0.0`, `top_p=1.0`, `top_k=-1`,
  generation `TP=16`, `max_num_seqs=32`, `max_num_batched_tokens=32768`,
  `kernel_config.moe_backend=triton`, local `policy.tokenizer.name`, and
  `NRL_MEGATRON_TOKENIZER_MODEL` pointing at the local HF snapshot.
- The `r5` K12 `MasterConfig` confirmed PARD was enabled as
  `method=draft_model`, `model=amd/PARD-Qwen3-0.6B`,
  `num_speculative_tokens=12`, and `parallel_drafting=True`.
- Baseline `3169284`, K5 `3169285`, and K12 `3169286` have all reached vLLM
  V1 engine initialization with vLLM `0.20.0`, bf16 target weights,
  `load_format=auto`, `tensor_parallel_size=16`, and
  `kernel_config.moe_backend=triton`.
- Baseline logs confirm `speculative_config=None`. K5 logs confirm
  `SpeculativeConfig(method='draft_model', model='amd/PARD-Qwen3-0.6B',
  num_spec_tokens=5)`. K12 logs confirm the same PARD drafter with
  `num_spec_tokens=12`.
- Baseline, K5, and K12 selected `Using TRITON Unquantized MoE backend`, which
  means the current retry is past the previous FlashInfer TRTLLM
  `K must be divisible by blockK` failure path.
- Direct log checks around 23:34 PDT showed baseline target shard loading at
  `100% Completed | 118/118`; K5 target and 1-shard PARD drafter loading also
  reached `100%`; K12 target and 1-shard PARD drafter loading reached `100%`.
- K5 and K12 have emitted `Model loading took 27.68 GiB memory` lines across
  workers. Baseline has emitted target shard completion but has not yet emitted
  enough post-load/KV-cache lines to classify it as ready for generation.
- Latest poll at 23:37 PDT still has zero generation metrics. Baseline and K12
  are now also building `transformer-engine-torch==2.14.1` in `_env_builder`
  processes, while K5 shows no Transformer Engine build lines at that poll.
  This is policy/Megatron-side runtime environment bring-up, not a speculative
  decoding performance result.
- Latest poll at 23:40 PDT still has zero generation metrics. Baseline has now
  reached vLLM KV/cache readiness: `GPU KV cache size` around
  `1,692,192`-`1,693,248` tokens, maximum concurrency around `206.57x`-`206.70x`
  for 8,192 tokens per request, and CUDA graph capture finished. Baseline also
  finished multiple Megatron policy worker venvs after Transformer Engine
  build. K5 and K12 are still in Transformer Engine build/venv completion, with
  some `Built transformer-engine-torch==2.14.1` lines already appearing.
- Latest poll at 23:42 PDT still has zero generation metrics, but all three
  jobs have now reached vLLM readiness:
  baseline KV cache around `1.692M`-`1.693M` tokens with graph capture in
  about 8 seconds; K5 KV cache around `1.266M`-`1.267M` tokens with graph
  capture in about 30-31 seconds; K12 KV cache around `1.270M` tokens with
  graph capture in about 30-31 seconds. Policy worker venv completion lines are
  present for baseline, K5, and K12. Hard error count remains zero.
- Direct tail checks after the 23:45 PDT poll show all three jobs reached
  `SETUP COMPLETE`, entered `Epoch 1/1` and `Step 1/1`, prepared a batch, and
  started `Generating responses for batch of size 256`. Generation is now
  running, but completion metrics have not been emitted yet.
- K5 has transient Hugging Face `429` lines from vLLM repo file-list lookup for
  `Qwen/Qwen3-235B-A22B`, but vLLM returned an empty list and continued into
  engine/model loading. This is not yet a fatal tokenizer failure like the
  earlier Megatron-Bridge path.
- Final poll at 23:57 PDT and `sacct` show all three jobs failed before any
  generation metric was emitted: baseline `3169284` failed after `00:46:57`,
  K5 `3169285` after `00:48:01`, and K12 `3169286` after `00:48:30`.
- The common failure mode is vLLM/Ray generation worker death during
  generation: `ActorDiedError`, worker connection error code 2 / end-of-file.
  K12 also shows `ProcessGroupNCCL::Watchdog::run()` stack traces.
- Because the same failure appeared for baseline, K5, and K12, this is not a
  PARD acceptance-rate or speedup result. It should be classified as a
  Qwen3-235B 32-node vLLM/Ray compiled-DAG stability or resource-pressure
  failure at the worker32/fixed-1024 shape.
- No JSON/CSV metric artifacts are present under the new job log directories.

## Additional Active Standalone Retry: High Batch EAGLE

The earlier high-batch standalone sweep for Qwen3-30B-A3B and Qwen3-235B
(`3156942`-`3156953`) failed before model initialization. Root cause was a
wrapper dependency issue: it installed `huggingface-hub<1.0` into the run-local
`pydeps` directory and placed that directory first on `PYTHONPATH`, which
conflicted with the vLLM 0.20.2 container's `transformers 5.8.0` import path.

The wrapper now defaults `HUGGINGFACE_HUB_PIP_SPEC` to empty, so the container's
own dependency set is used unless an override is explicitly requested.

Submitted dependency-fixed retry:

| Model | Jobs | Shape |
|---|---|---|
| Qwen3-30B-A3B-Thinking-2507 | baseline `3168368`, K1 `3168369`, K2 `3168370`, K3 `3168371` | vLLM standalone, `ISL=1000`, `OSL=512`, batches `64 128 256`, `TP=1` |
| Qwen3-235B-A22B | baseline `3168372`, K1 `3168373`, K2 `3168374`, K3 `3168375` | vLLM standalone, `ISL=1000`, `OSL=512`, batches `64 128 256`, `TP=4` |

Status file:
`latest_vllm_qwen30ba3b_qwen235b_bs64_128_256_k123_jobs_retry_depsfix.txt`

Latest high-batch status:

- Qwen3-30B-A3B retry jobs completed. K=3 gives the strongest high-batch
  standalone result: 2.44x at bs64, 1.91x at bs128, and 1.94x at bs256 with
  roughly 80% acceptance.
- Qwen3-235B original retry baseline `3168372` completed. Baseline throughput
  is 1279.06 / 2112.63 / 2736.26 output tok/s/GPU at bs64 / bs128 / bs256.
  K1 `3168373`, K2 `3168374`, and K3 `3168375` failed during vLLM engine
  initialization with `No available memory for the cache blocks`. This is a
  capacity/cudagraph/speculative-drafter issue, not the earlier dependency
  issue.
- The standalone harness now supports explicit `--max-num-seqs`,
  `--max-cudagraph-capture-size`, and `--moe-backend`. The Qwen3-235B-only
  retry completed with `moe_backend=triton`, `max_num_seqs=128`,
  `max_num_batched_tokens=196608`, `max_cudagraph_capture_size=256`, and
  `gpu_memory_utilization=0.96`.

Qwen3-30B-A3B high-batch result:

| Batch | K=1 speedup | K=1 acc | K=2 speedup | K=2 acc | K=3 speedup | K=3 acc |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.58x | 89.6% | 1.80x | 86.6% | 2.44x | 81.2% |
| 128 | 1.42x | 89.6% | 1.47x | 85.7% | 1.91x | 80.1% |
| 256 | 1.61x | 88.9% | 1.56x | 85.7% | 1.94x | 80.1% |

Qwen3-235B-A22B high-batch Triton/maxseq128 result:

| Batch | K=1 speedup | K=1 acc | K=2 speedup | K=2 acc | K=3 speedup | K=3 acc |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.42x | 98.1% | 1.37x | 96.6% | 2.20x | 92.4% |
| 128 | 1.59x | 97.7% | 0.44x | 96.6% | 0.61x | 90.1% |
| 256 | 1.46x | 97.7% | 0.46x | 96.6% | 0.51x | 90.1% |

Qwen3-235B interpretation:

- K=1 is the stable high-batch setting in this vLLM standalone shape.
- K=2 and K=3 have very high acceptance, but bs128/bs256 are slower than
  baseline. This points to runtime overhead or scheduler/cudagraph behavior,
  not drafter quality, as the dominant issue at larger batches.
- K=3 is only strong at bs64 in this run, reaching 2.20x.

High-batch artifacts:

- CSV: `docs/qwen3_30ba3b_235b_vllm_highbatch_k123_metrics.csv`
- PNG: `docs/qwen3_30ba3b_235b_vllm_highbatch_k123_speedup_acceptance.png`

Qwen3-235B Triton/maxseq128 retry:

| Run | Job |
|---|---:|
| Baseline | `3168445` |
| K1 | `3168446` |
| K2 | `3168448` |
| K3 | `3168449` |

Status file:
`latest_vllm_qwen235b_bs64_128_256_k123_tritonmoe_maxseq128_jobs.txt`

## Research Scan

| Method | Public implementation state | Qwen3-235B actionability | Notes |
|---|---|---|---|
| PARD | Available in vLLM parallel draft model support and AMD-AGI/PARD. Public Qwen3 drafter: `amd/PARD-Qwen3-0.6B`. vLLM docs use `speculative_config={"model": "amd/PARD-Qwen3-0.6B", "num_speculative_tokens": 12, "method": "draft_model", "parallel_drafting": true}`. | Already tested standalone and now running in NeMo-RL generation-only. | Strong on short synthetic. Real OpenMath prompts reduce acceptance; K5 is safer than K12 for real-prompt NeMo-RL validation. |
| PARD-2 | Paper is public. AMD-AGI/PARD README says PARD-2 code and model checkpoints will be released soon. | Not immediately runnable for Qwen3-235B without released checkpoint or reimplementation/training. | Highest-priority follow-up because it changes training toward acceptance-length optimization, directly addressing the acceptance-collapse failure mode. |
| P-EAGLE | vLLM has announced/merged parallel EAGLE support, and speculators has a training RFC. | Requires a Qwen3-235B-compatible P-EAGLE drafter or training/export before benchmark. | Relevant because it attacks the same sequential-drafting overhead that hurt plain EAGLE. Not a drop-in replacement today. |
| BanditSpec | Paper proposes training-free online selection of speculative decoding hyperparameters with UCB/EXP3-style policies. | Potentially actionable as a runtime policy over K/gate choices once baseline/K5/K12 telemetry is available. | This is directly relevant to our Qwen3-235B finding that K=1/K=3 speedups vary by batch/prompt; adaptive K could avoid bad K2/K3 regimes. |
| SPECTRE | 2026 SGLang serving framework that runs draft generation and target verification in parallel using remote drafters, scheduling, and prompt compression. | Not a vLLM/NeMo-RL drop-in, but very relevant for a standalone Qwen3-235B validation path if we can run SGLang. | Most directly relevant non-vLLM result found so far: paper reports Qwen3-235B-A22B with TP=8 and up to 2.28x speedup over autoregressive decoding. |
| SwiftSpec | System-level asynchronous/disaggregated speculative decoding with parallel tree generation and KV-cache management. | Not a drop-in vLLM/NeMo-RL method; would require engine-level implementation or upstream support. | Relevant root-cause direction because Qwen3-235B failures often look like draft/verification overhead on the critical path. |
| Speculative Speculative Decoding / Saguaro | Paper parallelizes speculation and verification by preparing draft branches for likely verification outcomes. | Not immediately runnable in current vLLM/NeMo-RL; requires implementation support. | Conceptually targets the same serial speculation-verification barrier seen in high-K runs. |
| DEL self-speculative decoding | Early-exit self-drafting with adaptive exit layer and speculation length. | Not directly actionable for Qwen3-235B MoE without target-model early-exit hooks and vLLM support. | Useful idea for adaptive K/exit selection, but less practical than PARD/BanditSpec for the current cluster path. |
| DREAM-R | 2026 speculative reasoning framework with RL-trained refined drafting and fully parallel speculative reasoning. | Not a drop-in text-only Qwen3-235B inference method. | Methodologically relevant: it trains drafts toward target-verified reasoning/acceptance rather than raw next-token accuracy, similar in spirit to why PARD-2 is attractive. |
| Mirror-SD | Parallel heterogeneous execution plus speculative streaming to reduce AR drafter latency. | Not immediately actionable without an implementation and heterogeneous scheduling path. | Reinforces that our bottleneck is not only acceptance; high acceptance can still lose if drafter/verification overhead stays serial. |
| D-PACE | Dynamic position-aware training loss for parallel speculative drafters, aimed at increasing expected accepted draft length. | Useful if we train or fine-tune a Qwen3-235B-specific parallel drafter; not an inference-only switch. | This is a plausible training-side answer to the long-context/OpenMath acceptance collapse seen with target-independent PARD. |
| PEARL/adaptive draft length | Public research/code exists for adaptive parallel speculative decoding. | Would need integration work plus a trained proposer for Qwen3-235B. | Useful follow-up if PARD real-prompt works but fixed K is unstable across batch/prompt regimes. |
| DFlash | vLLM speculators has DFlash/EAGLE-style workflows, but public examples are not a ready Qwen3-235B checkpoint. | Requires training/export for Qwen3-235B before benchmark. | Lower priority than PARD while no Qwen3-235B-ready public drafter is available. |
| Tree/MTP/EAGLE variants | Available conceptually, some libraries support tree speculative decoding. | We already tested EAGLE3 K=1/K=3/K=5; high acceptance did not always overcome overhead for 235B. | Avoid repeating plain EAGLE K sweeps unless the runtime path changes. |

## Decision Rules To Avoid Repeating Failures

- Do not spend full GRPO runs on Qwen3-235B PARD until the generation-only
  gate emits clean timing for baseline, K5, and K12. The currently running jobs
  are the correct gate because they stop after generation.
- Treat `3168274`/`3168275` and `3168570`/`3168571`/`3168572` as
  infrastructure failures, not performance data. They proved the Triton MoE
  workaround, then exposed the tokenizer/API-rate-limit path. The latter also
  exposed that Megatron-Bridge needed its own tokenizer snapshot override.
- Treat the short synthetic standalone PARD K12 result as a capability proof,
  not as a production prediction. Real OpenMath prompts showed much lower
  acceptance, so NeMo-RL comparison must use real rollout prompts and fixed
  generation controls.
- For Qwen3-235B EAGLE3, do not repeat K2/K3 high-batch sweeps unless the
  runtime path changes. The completed bs128/bs256 results had high acceptance
  but poor speedup, pointing to overhead/cudagraph/scheduler behavior rather
  than drafter quality.
- If the current PARD K5/K12 NeMo-RL run fails before metrics, classify the
  failure first: tokenizer/HF access, MoE backend, capacity/cudagraph, Ray
  bring-up, or genuine specdec runtime. Only the last category should drive
  algorithm conclusions.

## Next Experiment Priority

1. Re-run the NeMo-RL generation-only PARD gate at a smaller/stabler
   Qwen3-235B shape before drawing PARD conclusions. The worker32/fixed-1024
   retry failed baseline, K5, and K12 with the same vLLM/Ray actor-death mode.
2. If a smaller gate produces valid metrics and PARD K5 gives generation
   speedup, move only K5 into a short full-GRPO validation. K12 should move
   forward only if it clearly beats K5 on real rollout prompts.
3. If fixed K is unstable, implement a BanditSpec-style runtime policy over
   `K=0/K=1/K=5/K=12` or over always-on/gated modes, using observed
   generation throughput/acceptance as reward. This is lower-risk than another
   blind K sweep because it adapts away from bad high-overhead regimes.
4. If vLLM/NeMo-RL PARD remains bottlenecked by runtime overhead despite
   reasonable acceptance, validate SPECTRE separately in SGLang for
   Qwen3-235B-A22B. It is not a NeMo-RL integration yet, but it is the strongest
   recent Qwen3-235B-specific external evidence for large-model speculative
   speedup.
5. Track PARD-2 and D-PACE for training-side fixes. These are the most relevant
   paths if the root cause is real-prompt/long-context acceptance collapse
   rather than vLLM runtime overhead.

Sources:

- vLLM parallel draft model docs: https://docs.vllm.ai/en/stable/features/speculative_decoding/parallel_draft_model/
- AMD-AGI/PARD repository: https://github.com/AMD-AGI/PARD
- PARD paper: https://arxiv.org/abs/2504.18583
- PARD-2 paper: https://arxiv.org/abs/2605.08632
- vLLM P-EAGLE post: https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-03-13-p-eagle.md
- P-EAGLE paper: https://arxiv.org/abs/2602.01469
- vLLM speculators P-EAGLE training RFC: https://github.com/vllm-project/speculators/issues/292
- BanditSpec paper: https://arxiv.org/abs/2505.15141
- SPECTRE paper: https://arxiv.org/abs/2605.08151
- SwiftSpec paper: https://arxiv.org/abs/2506.11309
- Speculative Speculative Decoding paper: https://arxiv.org/abs/2603.03251
- DEL paper: https://arxiv.org/abs/2504.05598
- DREAM-R paper: https://arxiv.org/abs/2605.28678
- Mirror-SD page: https://machinelearning.apple.com/research/mirror
- D-PACE paper: https://arxiv.org/abs/2605.18810
- PEARL repository: https://github.com/smart-lty/ParallelSpeculativeDecoding
- vLLM speculators repository: https://github.com/vllm-project/speculators
