# Qwen3-235B PARD Full-GRPO Recovery Note

Date: 2026-06-06 PDT

## Current State

Remote access recovered after the earlier DNS/VPN issue. The latest-main
nightly worktree on `oci-hsg-cs-001-vscode-02` is reachable and NeMo-RL
full-GRPO controls are running with vLLM `0.20.0` inside the Ray
`VllmGenerationWorker`.

New small-model evidence is tracked in:

```text
docs/qwen32_qwen30_mainnightly_vllm020_fullgrpo5_status_20260606.md
docs/qwen32_qwen30_mainnightly_vllm020_fullgrpo5_metrics_20260606.csv
```

Key update: Qwen3-32B public PARD K5 job `3197802` completed Full-GRPO
`5/5` with Slurm `COMPLETED 0:0`. This proves the latest-main vLLM0.20
PARD path is functional in NeMo-RL full-GRPO. Original memory-envelope
Qwen3-32B/Qwen3-30B-A3B baseline and PARD runs can still hit Step 3 vLLM
CuMem OOM. The Qwen3-32B mem80/bt16k retry completed the matched
baseline/K3 comparison; the Qwen3-30B-A3B worker-batch-32 mem80/bt16k
baseline/K3 stability pair also completed 20/20 steps.

Latest poll update: Qwen3-32B mem80/bt16k retry baseline `3197980` and K3
`3197981` both completed `5/5` with Slurm `COMPLETED 0:0`. Full Step 1-5
comparison gives generation speedup `1.65x` and total step-time speedup
`1.20x`, close to the generation-fraction bound.

Qwen3-235B latest-main initial jobs `3197584`, `3197585`, `3197586` did not
produce performance metrics: baseline/K3 failed from a launcher path bug
(`./examples/run_grpo.py` from `experiments/eagle3_online`), and K5 failed at
worker startup with an srun memory allocation error. The launcher was fixed by
copying the current absolute-path `submit_nemorl_online_draft_specdec.sh`, and
pathfix/mem80/bt16k retries were submitted as `3198040` baseline, `3198041`
PARD K5, and `3198042` PARD K3.

Latest Qwen3-235B poll: origin-main local CAT/TPP-mask K5 job `3195285`
failed before rollout in `MegatronPolicyWorker.prepare_refit_info()` because
Megatron-Bridge MoE gate mapping parsed an empty expert suffix as `int`.
Latest-main baseline `3198040` failed before Step 1: the earlier
`VllmGenerationWorker.sleep()` / `prepare_refit_info()` actor-death path
resurfaced. Latest-main public PARD K5 `3198041` reached Step 1/speculative
generation but failed before a total step metric during
`policy.get_reference_policy_logprobs` with `MegatronPolicyWorker`
`ActorDiedError`/`SYSTEM_ERROR` and repeated CUDA copy segfault traces.
Latest-main public PARD K3 `3198042` failed before Step 1 during
`VllmGenerationWorker.prepare_refit_info()`; one worker step was reported by
Slurm as `OUT_OF_MEMORY`.

Local-HF-cache fallback jobs were submitted with
`gpu_memory_utilization=0.70` and `max_num_batched_tokens=8192`: baseline
`3198183` completed Step 1 generation but failed in
`policy.get_reference_policy_logprobs` with the same policy actor death /
CUDA copy segfault pattern; PARD K5 `3198184` failed during Ray-head startup
and is not performance evidence; PARD K3 `3198185` reached Step 1 generation
and reward/advantage. Its K3 SpecDec metrics emitted mean acceptance length
`3.08-3.34` and avg draft acceptance `69.4-77.9%`, but it then failed at
`Computing logprobs` / `policy.get_reference_policy_logprobs` with
`MegatronPolicyWorker` `SYSTEM_ERROR` and `cudaMemcpyAsync` /
`cuMemcpyDtoHAsync` segfault traces. This reproduces the baseline `3198183`
tail failure after proving the K3 generation path itself ran. To isolate the
reference-logprob tail, skip-reference diagnostic jobs were submitted:
baseline `3198324` and PARD K3 `3198325`. These set
`grpo.skip_reference_policy_logprobs_calculation=true` and
`loss_fn.reference_policy_kl_penalty=0.0`, so they are not default Full-GRPO
metrics. Both diagnostics failed before `ray-driver.log` was created because
Ray workers could not connect to the GCS/head and a background worker `srun`
died, so they do not yet prove anything about the logprob path or performance.
A clean K3-only skip-reference resubmit, `3198380`, was submitted at
2026-06-06 18:52 PDT. The driver confirmed
`grpo.skip_reference_policy_logprobs_calculation=true`,
`loss_fn.reference_policy_kl_penalty=0.0`, vLLM `0.20.0`, and local PARD K3
engine initialization. It has progressed past the previous Ray/GCS startup
failure: vLLM sleep/offload completed with about `127 GiB` freed per worker,
128 `lm_policy` workers initialized, and the vLLM generation backend was
selected. Step 1 reached generation for batch size `256`. The K3 acceptance
metrics emitted mean acceptance length `2.75-3.13` and avg draft acceptance
`58.4-71.1%`. It then moved through `Processing rewards` and `Computing
logprobs` far enough to enter `Training policy`. It then failed in
`prepare_packed_loss_input` / `_pack_input_ids` because the loss path tried to
copy `actual_len=600` tokens from an `input_ids` row with width `408`:

```text
RuntimeError: The expanded size of the tensor (600) must match the existing size (408)
```

This is the strongest evidence so far that the prior Qwen3-235B K3 failures
were not caused by PARD generation or drafter KV pressure. With reference
logprobs disabled, the same PARD K3 generation path reaches policy training.
The remaining failure is now a packed-loss sequence-length mismatch, not a
vLLM generation OOM and not the reference-logprob actor-death signature.

Root-cause note:

```text
docs/qwen3_235b_packed_loss_mismatch_root_cause_20260606.md
```

A follow-up diagnostic, `3198648`, was submitted at 2026-06-06 19:22 PDT with
the same PARD K3 skip-reference mem70/bt8k shape but
`policy.sequence_packing.fuse_loss=false`. It ran and failed during Step 1
policy training with a Megatron gradient check error:

```text
RuntimeError: Rank 126, node nvl72085-T08, device 0, iteration 1:
Unexpected result nan
message='found NaN in local grad norm for bucket #0 in backward pass before data-parallel communication collective'
```

It emitted K3 SpecDec metrics before the failure, with mean acceptance length
around `3.05` and avg draft acceptance around `68.3%`. This result is useful:
disabling fused loss avoided the earlier padded-length copy failure, so the
packed-loss mismatch is isolated to the fused sequence-packing loss metadata
path. The remaining Qwen3-235B skip-reference blocker is now training numeric
stability, not vLLM generation, PARD acceptance, or the fused-loss copy bug.

A default Full-GRPO retry, `3198436`, was also submitted at 2026-06-06
18:55 PDT. It keeps reference logprobs enabled but adds tighter logprob
chunking:

```text
policy.logprob_chunk_size=2048
policy.sequence_packing.logprob_mb_tokens=4096
policy.dynamic_batching.logprob_mb_tokens=4096
```

This tested whether the `3198183`/`3198185` post-generation policy/reference
logprob actor death could be avoided without changing the default GRPO
objective. It reached Step 1 generation for batch size `256` and emitted K3
acceptance metrics with mean acceptance length `3.19-3.38` and avg draft
acceptance `72.9-79.2%`. It still failed at `Computing logprobs` with
`MegatronPolicyWorker` `ActorDiedError` / `SYSTEM_ERROR`, so logprob chunking
at this level did not fix the default reference-logprob crash.

For Qwen3-30B-A3B, a matched 20-step stability pair was submitted under the
same latest-main/nightly vLLM0.20 runtime, using the conservative Qwen3-32B
memory envelope and worker-batch-32 shape:

| Model | Job | Mode | Status |
|---|---:|---|---|
| Qwen3-30B-A3B | `3198446` | baseline, GBS512, mem80/bt16k | `COMPLETED 0:0`, 20/20; Step 2-20 avg total `84.34s`, generation `15.77s`, E2E `141.17 tok/s/GPU` |
| Qwen3-30B-A3B | `3198447` | public PARD K3, GBS512, mem80/bt16k | `COMPLETED 0:0`, 20/20; Step 2-20 avg total `81.58s`, generation `10.82s`, E2E `146.21 tok/s/GPU`; generation speedup `1.46x`, E2E throughput `1.04x`, total step-time `1.03x`, avg acceptance `69.45%` |

These jobs prove that Qwen3-30B-A3B can pass the previous Step-3 vLLM CuMem
OOM point and complete 20 Full-GRPO steps under the worker-batch-32,
mem80/bt16k shape. The performance result is stable but modest end-to-end:
generation is `1.46x` faster, while total step-time improves only `1.03x`
because generation is about `18.7%` of the baseline step.

## Last Verified Evidence

| Model | Job | SpecDec | Status | Main Evidence |
|---|---:|---|---|---|
| Qwen3-32B | `3195498` | baseline | `COMPLETED 0:0` | Full-GRPO step completed, E2E `100.91 tok/s/GPU`, generation-worker `355.47 tok/s/GPU`. |
| Qwen3-32B | `3195499` | public PARD K5 | `COMPLETED 0:0` | Full-GRPO step completed, E2E `105.49 tok/s/GPU`, generation-worker `416.75 tok/s/GPU`, avg draft acceptance `49.4%`. |
| Qwen3-30B-A3B | `3195500` | baseline | failed | MoE `SequentialMLP deepcopy(config)` hit non-pickleable distributed `ProcessGroup`. |
| Qwen3-30B-A3B | `3195501` | local CAT/PARD-2-style K5 | failed | Same MoE `ProcessGroup` construction failure. |
| Qwen3-235B-A22B | `3195285` | local CAT/TPP-mask PARD K5 | `RUNNING` | 128/128 workers connected; driver is in actor-env build; no step metric yet. |

Qwen3-32B measured speedups:

| Metric | Baseline | PARD K5 | Speedup |
|---|---:|---:|---:|
| Total step time | `58.69s` | `56.14s` | `1.045x` |
| Generation time | `16.66s` | `14.21s` | `1.18x` |
| E2E throughput | `100.91 tok/s/GPU` | `105.49 tok/s/GPU` | `1.045x` |
| Generation-worker throughput | `355.47 tok/s/GPU` | `416.75 tok/s/GPU` | `1.17x` |

Current compact plot:

```text
docs/qwen3_235b_fullgrpo_validation_status_20260606.png
```

Interpretation: PARD is reducing generation cost in real NeMo-RL Full-GRPO.
E2E speedup is smaller because generation is only about `25-28%` of the
one-step Qwen3-32B run.

## Patch Applied Before DNS Loss

The Qwen3-30B-A3B failure came from Megatron MoE model construction:

```text
SequentialMLP -> deepcopy(config) -> TypeError:
cannot pickle 'torch._C._distributed_c10d.ProcessGroup' object
```

The active remote checkout was patched so `SequentialMLP` uses a shallow copy
when it only needs to override top-level `ffn_hidden_size`:

```text
3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/experts.py
```

Patch artifact:

```text
experiments/eagle3_online/remote_patch_files/megatron_moe_sequential_mlp_shallowcopy_processgroup.patch
```

The remote patched file passed:

```bash
python3 -m py_compile experts.py
```

## Post-Patch Validation Jobs

The following jobs were submitted after the patch:

| Model | Job | SpecDec | Purpose |
|---|---:|---|---|
| Qwen3-30B-A3B | `3195815` | baseline | Failed after the shallow-copy patch in `MegatronPolicyWorker.prepare_refit_info`; Megatron-Bridge MoE gate mapping parsed an empty expert suffix as `int`. |
| Qwen3-30B-A3B | `3195816` | local CAT/PARD-2-style K5 | Failed at the same `prepare_refit_info` MoE gate mapping path. |

Both use GBS `256`, no local transformer spec, no-overlap DDP,
`moe_grouped_gemm=false`, vLLM MoE backend `triton`, and a fresh `pgshallow`
cache/checkpoint suffix.

## Resume Commands

Useful current poll commands:

```bash
scripts/apply_remote_megatron_moe_pg_shallowcopy_patch.sh
scripts/poll_qwen235b_pard_fullgrpo_current_status.sh
scripts/poll_qwen32_qwen30_main_nightly_fullgrpo_status.sh
```

The first command is idempotent and re-applies the MoE patch if needed. The
poll commands check the Qwen3-235B and small-model latest-main controls.

```text
3195285,3195815,3195816
```

## Current Evidence Boundary

Do not claim Qwen3-235B Full-GRPO E2E speedup yet. Current proven evidence is:

- PARD works and speeds up Qwen3-235B in vLLM standalone and NeMo-RL generation
  gates.
- PARD works through a full NeMo-RL Full-GRPO step on dense Qwen3-32B.
- Qwen3-30B-A3B/Qwen3-235B-A22B MoE policy construction needed the
  `SequentialMLP` shallow-copy patch, and origin-main Qwen3-30B-A3B then hit a
  separate Megatron-Bridge MoE gate mapping/refit failure.

The missing evidence is a Qwen3-235B MoE Full-GRPO result that passes refit,
generation, policy/reference logprobs, and train/update with matched baseline
and SpecDec total step metrics. The latest default-path logprob chunking retry
`3198436` still fails in the reference-logprob tail. The skip-reference
diagnostic `3198648` gets past the earlier fused packed-loss length mismatch
when `policy.sequence_packing.fuse_loss=false`, but then fails in policy
backward with NaN local grad norm. The next Qwen3-235B fix should target
training numeric stability for the skip-reference path and reference-logprob
actor stability for the default Full-GRPO path, rather than vLLM generation or
PARD drafter quality.
