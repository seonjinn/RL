# Qwen3-235B Packed-Loss Length Mismatch Root Cause

Date: 2026-06-06 PDT

## Symptom

Qwen3-235B PARD K3 skip-reference diagnostic job `3198380` passed vLLM
generation, rewards, and policy logprobs far enough to enter `Training policy`.
It then failed in the fused sequence-packing loss path:

```text
RuntimeError: The expanded size of the tensor (600) must match the existing size (408)
```

The failing assignment is in
`nemo_rl/algorithms/loss/utils.py::_pack_input_ids`:

```python
actual_len = int((cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item())
seq[:actual_len] = input_ids[i, :actual_len]
```

## Root Cause

The fused loss path interprets `PackedSeqParams.cu_seqlens_q` as unpadded
actual sequence lengths. In the Megatron sequence packer, however,
`PackedSeqParams` is currently built with padded lengths for both fields:

```python
packed_seq_params = PackedSeqParams(
    cu_seqlens_q=cu_seqlens_padded,
    cu_seqlens_kv=cu_seqlens_padded,
    cu_seqlens_q_padded=cu_seqlens_padded,
    cu_seqlens_kv_padded=cu_seqlens_padded,
    ...
)
```

This is compatible with the forward pass metadata, but it is not compatible
with `_pack_input_ids`, which needs the true unpadded lengths to copy from the
unpacked `data["input_ids"]` tensor. In `3198380`, one packed sequence had a
padded length of `600` while the corresponding unpacked `input_ids` row had
width `408`, causing the copy to fail.

This is not a vLLM generation OOM, not a PARD drafter/KV-cache issue, and not
the default reference-logprob actor death. It is a fused sequence-packing loss
metadata mismatch exposed after the skip-reference diagnostic reached policy
training.

## Diagnostic Retry Result

Submitted job `3198648` to test the fastest non-invasive workaround:

```text
PARD K3
skip_reference_policy_logprobs_calculation=true
loss_fn.reference_policy_kl_penalty=0.0
gpu_memory_utilization=0.70
max_num_batched_tokens=8192
policy.sequence_packing.fuse_loss=false
```

`3198648` reached Step 1 generation and emitted K3 SpecDec metrics with mean
acceptance length around `3.05` and avg draft acceptance around `68.3%`. It did
not reproduce the padded-length copy error. Instead, it failed later in policy
training during gradient finalization:

```text
RuntimeError: Rank 126, node nvl72085-T08, device 0, iteration 1:
Unexpected result nan
message='found NaN in local grad norm for bucket #0 in backward pass before data-parallel communication collective'
```

This confirms the original length mismatch is isolated to the fused loss path.
The next durable fused-loss fix should pass both unpadded and padded sequence
lengths into `prepare_packed_loss_input`, instead of letting fused loss read
padded lengths as actual lengths.

The remaining Qwen3-235B skip-reference diagnostic blocker is now policy
training numeric stability after generation/logprob preparation, not vLLM
generation, drafter KV memory, reference-logprob actor death, or the fused-loss
copy bug.

## Follow-Up Temperature Diagnostic

`3198648` used fixed greedy decode with `policy.generation.temperature=0.0`.
The active Megatron training path still applies temperature scaling as
`logits.div_(sampling_params.temperature)` whenever the temperature is not
`1.0`, so temperature zero can create infinities/NaNs in the training/logprob
path. This is consistent with the observed failure:

```text
found NaN in local grad norm for bucket #0 in backward pass
```

Submitted follow-up job `3207001` with the same PARD K3, skip-reference,
`policy.sequence_packing.fuse_loss=false`, mem70/bt8k shape, but
`policy.generation.temperature=1.0`. If this run passes policy training, the
`3198648` NaN should be classified as a temperature-zero training-path issue
rather than a PARD or Qwen3-235B MoE issue.

## Patch Applied

On 2026-06-07 12:57 PDT, the latest-main/nightly worktree was patched to guard
temperature scaling in all three local training paths:

```text
nemo_rl/models/megatron/train.py
nemo_rl/models/automodel/train.py
nemo_rl/models/policy/workers/dtensor_policy_worker.py
```

The new guard skips scaling when sampling params are absent, `None`, or
`temperature <= 0.0`, and only divides logits for positive non-1.0
temperatures. `python3 -m py_compile` passed for all three files. Job
`3207001` is still pending in Slurm priority queue as of 2026-06-07 13:10:05
PDT, with Slurm predicting `StartTime=2026-06-07T13:26:20`. There is not yet a
new Qwen3-235B Full-GRPO metric or `ray-driver.log`.

## 2026-06-16 Durable cu_seqlens Fix

The OCI-HSG latest-main MathRL checkout was patched at:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-mathrl-20260613/nemo_rl/models/megatron/data.py
```

The patch separates actual and padded sequence boundaries:

- `PackedSeqParams.cu_seqlens_q` and `cu_seqlens_kv` now use unpadded actual
  lengths.
- `PackedSeqParams.cu_seqlens_q_padded` and `cu_seqlens_kv_padded` still use
  padded boundaries for Megatron/CP packing.
- The VLM/internal-packing helper now returns the actual `cu_seqlens` instead
  of `None`, while preserving `cu_seqlens_padded`.

Validation on OCI-HSG:

```text
python3 -m py_compile nemo_rl/models/megatron/data.py
```

passed after the patch. A backup of the pre-patch remote file was written as:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-mathrl-20260613/nemo_rl/models/megatron/data.py.pre_unpadded_cu_fix_20260616
```

New proof jobs were submitted with `max_steps=1`, OSL/min tokens `256`,
temperature `1.0`, top-p `1.0`, top-k `-1`, and the patched checkout:

| Job | Method | State at submit check | Purpose |
|---:|---|---|---|
| `3342356` | baseline | `PENDING (Priority)` | Verify fused sequence-packing loss no longer fails after generation. |
| `3342358` | PARD K3 | `PENDING (Resources)` | Verify PARD generation plus fused sequence-packing loss on the patched path. |

Tracker:

```text
latest_oci_hsg_qwen235b_mathrl_packedcu_fix_gate1_20260616_jobs.csv
```
