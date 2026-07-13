# OCI-HSG Qwen235B MathRL Packed-cu Fix Gate

Refreshed at `2026-06-16 09:19 PDT`.

## Patch

Patched remote file:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-mathrl-20260613/nemo_rl/models/megatron/data.py
```

Change:

- `PackedSeqParams.cu_seqlens_q` / `cu_seqlens_kv` now use actual unpadded sequence lengths.
- `PackedSeqParams.cu_seqlens_q_padded` / `cu_seqlens_kv_padded` still use padded sequence lengths.
- The internal-packing helper now returns actual `cu_seqlens` instead of `None`.

Validation:

```text
python3 -m py_compile nemo_rl/models/megatron/data.py
```

passed on OCI-HSG after the patch. The old remote file was backed up as
`data.py.pre_unpadded_cu_fix_20260616`.

## Proof Jobs

Tracker: `latest_oci_hsg_qwen235b_mathrl_packedcu_fix_gate1_20260616_jobs.csv`

| Job | Method | Steps | OSL/min tokens | Temperature/top-p/top-k | State | Reason |
|---:|---|---:|---:|---|---|---|
| `3342356` | baseline | 1 | 256/256 | 1.0 / 1.0 / -1 | `PENDING` | `Priority` |
| `3342358` | PARD K3 | 1 | 256/256 | 1.0 / 1.0 / -1 | `PENDING` | `Priority` |

These jobs use account `nemotron_n3_post` and the patched checkout above.

## Related Pending Step20 Jobs

The already-submitted qwen235B MathRL step20 jobs are still pending and should
also pick up the patched checkout when they start:

| Job | Method | State | Reason |
|---:|---|---|---|
| `3334220` | baseline | `PENDING` | `Priority` |
| `3333535` | PARD | `PENDING` | `Priority` |
| `3333537` | Eagle3 | `PENDING` | `Priority` |
| `3333717` | Suffix | `PENDING` | `Priority` |

## qwen3-8B PARD-1 Standalone

The OCI-HSG qwen3-8B standalone PARD-1 matrix is still running:

| Job | Domain | Temp | Methods | State |
|---:|---|---:|---|---|
| `3342234` | Math | 0.0 | baseline, PARD K3, PARD K5 | `RUNNING 28:05` |
| `3342235` | Math | 1.0 | baseline, PARD K3, PARD K5 | `RUNNING 28:05` |
| `3342236` | SWE | 0.0 | baseline, PARD K3, PARD K5 | `RUNNING 28:05` |
| `3342237` | SWE | 1.0 | baseline, PARD K3, PARD K5 | `RUNNING 28:05` |

Latest local refresh still had `metric_rows=0`; no final `breakdown.json` rows
were available yet.

Live driver logs are active. Math temp0 PARD K3/K5 and SWE temp1 PARD K3 are
emitting vLLM SpecDec telemetry, so the runs are not stalled; final speedups
still require `breakdown.json`.
