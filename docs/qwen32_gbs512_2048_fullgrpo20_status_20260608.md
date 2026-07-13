# Qwen3-32B GBS512/2048 Full-GRPO 20-Step OOM Check

Date: 2026-06-08

## Purpose

Use the smaller Qwen3-32B performance recipe to test whether the Step-3 vLLM
CuMem OOM still occurs at realistic rollout batch sizes before promoting the
same envelope to Qwen3-235B.

The key comparison is not `GBS=256`. The OOM check should use:

- `GBS=512`: `num_prompts=16`, `num_generations=32`.
- `GBS=2048`: `num_prompts=64`, `num_generations=32`.
- `max_steps=20`.
- fixed `max_new_tokens=256`.
- vLLM reservation envelope: `gpu_memory_utilization=0.80`,
  `max_num_batched_tokens=16384`, `max_num_seqs=32`.

This directly tests whether lowering the vLLM reservation from the failed
`gpu_memory_utilization=0.90` / `max_num_batched_tokens=32768` envelope removes
the Step-3 wake-up OOM.

## Submitted Qwen3-32B Jobs

| Job | Mode | GBS | Shape | Status |
| --- | --- | ---: | --- | --- |
| `3210222` | baseline | 512 | 4n4g, generation TP2, train TP2 | Completed 20/20. This is well past the old Step-3 CuMem OOM point. |
| `3210223` | public PARD K3 | 512 | 4n4g, generation TP2, draft TP2, train TP2 | Completed 20/20. This is well past the old Step-3 CuMem OOM point. |
| `3210224` | baseline | 2048 | 4n4g, generation TP2, train TP2 | Completed 20/20. This passes the old Step-3 CuMem OOM point. |
| `3210225` | public PARD K3 | 2048 | 4n4g, generation TP2, draft TP2, train TP2 | Completed 20/20. This passes the old Step-3 CuMem OOM point. |

Runtime is NeMo-RL latest-main worker venv with vLLM `0.20.0` and public
`amd/PARD-Qwen3-0.6B`.

Current early metrics:

| Window | Baseline | PARD K3 | Speedup |
| --- | ---: | ---: | ---: |
| GBS512 Step 2-20 total step time | `85.11s` | `73.59s` | `1.156x` |
| GBS512 Step 2-20 E2E throughput | `139.93 tok/s/GPU` | `161.72 tok/s/GPU` | `1.156x` |
| GBS512 Step 2-20 generation time | `32.02s` | `21.18s` | `1.512x` |
| GBS512 Step 2-20 generation throughput | `371.94 tok/s/GPU` | `562.69 tok/s/GPU` | `1.513x` |
| GBS2048 Step 2-20 total step time | `256.21s` | `208.78s` | `1.227x` |
| GBS2048 Step 2-20 E2E throughput | `183.22 tok/s/GPU` | `225.01 tok/s/GPU` | `1.228x` |
| GBS2048 Step 2-20 generation time | `117.15s` | `68.73s` | `1.704x` |
| GBS2048 Step 2-20 generation throughput | `400.83 tok/s/GPU` | `683.42 tok/s/GPU` | `1.705x` |

Machine-readable tracking:

```text
docs/qwen32_gbs512_2048_fullgrpo20_status_20260608.csv
```

## True Recipe Baseline

These OOM-check jobs deliberately modify the recipe envelope to fixed
`max_new_tokens=256`, `gpu_memory_utilization=0.80`, and
`max_num_batched_tokens=16384`. They are stability tests, not the final
performance baseline.

Separate true no-SpecDec baselines were submitted with the original performance
YAML shape and only `max_steps`/logger run-control overrides:

| Job | Model | Config | Status |
| --- | --- | --- | --- |
| `3210285` | Qwen3-30B-A3B | `grpo-qwen3-30ba3b-4n4g.yaml` | Failed before step metric during MoE weight streaming (`shard_dim=0` for 3D fused-MoE tensor), not an OOM |
| `3210286` | Qwen3-32B | `grpo-qwen3-32b-4n4g.yaml` | Running; Step 1-11 metrics emitted |

The matching original-recipe PARD K3 job `3210601` is currently negative on the
matched Step 2-4 window: total-step `0.830x`, generation-time `0.657x`, E2E
`0.832x`, and generation-throughput `0.660x` versus baseline. This confirms the
fixed256/mem80/bt16k result should not be generalized to the 4096-token
original performance YAML.

Track the original-recipe jobs in:

```text
docs/qwen32_qwen30_true_recipe_baseline_20260608.md
```

## Existing Qwen3-30B-A3B Evidence

Qwen3-30B-A3B already has the corresponding 20-step stability checks:

| GBS | Jobs | Result |
| ---: | --- | --- |
| 512 | baseline `3198446`, PARD K3 `3198447` | Both completed 20/20 with mem80/bt16k. Step 2-20 generation speedup `1.46x`, E2E `1.04x`, acceptance `69.45%`. |
| 2048 | baseline `3207492`, PARD K3 `3207978` | Both completed 20/20 with mem80/bt16k. Step 2-20 generation speedup `1.719x`, E2E `1.083x`, acceptance `69.10%`. |

The earlier Qwen3-30B-A3B GBS2048 failure was at the larger reservation
envelope: `gpu_memory_utilization=0.90`, `max_num_batched_tokens=32768`. The
passing mem80/bt16k retries show the OOM was from vLLM sleep/wake reservation
pressure, not from the public PARD drafter itself.

## Decision Rule

Qwen3-32B completed both `GBS512` and `GBS2048` for 20 steps under mem80/bt16k,
so this is now the preferred small-model stability gate. For 235B, keep
`GBS=256` for initial non-colocated TP4 validation, then raise GBS only after
the smaller-model envelope has been matched carefully.
