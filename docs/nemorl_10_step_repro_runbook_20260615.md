# NeMo-RL 10-Step Repro Runbook - 2026-06-15

This runbook records the currently usable short-run evidence and the proof
gates that are still waiting on the scheduler.

## Usable Examples

| Purpose | Jobs | What it proves | What it does not prove |
| --- | --- | --- | --- |
| 235B SWE-RL baseline short run | `3299487` | Qwen3-235B SWE-RL Full-GRPO can complete a short NeMo-RL run on OCI-HSG. | PARD-2 success or speculative speedup. |
| 235B SWE-RL non-PARD-2 speculative short run | `3299489`, `3299491` | PARD K5 and Eagle-3 K3 can complete short SWE-RL runs. | A performance win; both were slower than baseline. |
| Online PARD-2 mechanics | `3288181`, `3288182`, `3288183` | Static and online PARD-2 run end to end on Qwen3-8B; online refit changes acceptance. | 235B PARD-2 success or throughput win. |

## 235B SWE-RL Repro Handles

- Tracker: `latest_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1_20260614_jobs.csv`
- Run id: `20260614_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1`
- Remote repo: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613`
- Launcher: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613/test_assets/qwen-235B/run_grpo_qwen3_235b_swe_scale_gen.sh`
- Parsed metric summary: `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_stepge2_20260615.csv`
- Fetched logs root: `tmp/oci_hsg_swerl_fullgrpo_logs_live_extract/20260614_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1/`

Completed step>=2 metrics:

| Job | Method | Parsed steps | E2E tok/s/GPU | Gen-worker tok/s/GPU | Gen-worker vs baseline |
| ---: | --- | ---: | ---: | ---: | ---: |
| `3299487` | baseline | 8 | `103.75` | `213.06` | `1.0000x` |
| `3299489` | PARD K5 | 5 | `27.38` | `57.07` | `0.2679x` |
| `3299491` | Eagle-3 K3 | 7 | `50.08` | `103.47` | `0.4857x` |

Failed cells in the same matrix:

| Job | Method | Failure |
| ---: | --- | --- |
| `3299488` | suffix K32 | vLLM actor env missed `arctic_inference.suffix_decoding._C`. |
| `3299490` | PARD-2 K1 | Staged PARD-2 vLLM `_C.abi3.so` had a Torch/C10 ABI symbol mismatch. |

## Online PARD-2 Repro Handles

- Tracker: `latest_oci_hsg_qwen8_pard2_official_comparison_20260613_jobs.csv`
- Run id: `20260613_oci_hsg_qwen8_pard2_official_comparison_masterconfigfix_shortqos`
- Metrics: `docs/qwen8_pard2_official_comparison_metrics_20260613.md`
- Online impact note: `docs/qwen8_pard2_official_online_impact_20260613.md`

Completed Qwen3-8B metrics:

| Job | Method | Parsed steps | Acceptance | Gen-worker vs baseline | E2E vs baseline |
| ---: | --- | ---: | ---: | ---: | ---: |
| `3288181` | baseline | 9/9 | | `1.0000x` | `1.0000x` |
| `3288182` | static PARD-2 | 9/9 | `1.836` | `0.6071x` | `0.8291x` |
| `3288183` | online PARD-2 | 9/9 | `2.553` | `0.5887x` | `0.6705x` |

## Pending/Failed Proof Gates

As of `2026-06-15 08:00 PDT`, the failed Lyris r28/r29 gates have been
superseded by raymatch tmpcache r30. The r28 baseline `2126895` proved the
TransformerEngine source build path is fixed, then failed before any parsed
training step because the Ray head used `Ray 2.49.2`/`Python 3.12.13` while the
driver used `Ray 2.54.0`/`Python 3.13.13`. The r29 launcher fixed that mismatch
and reached 64/64 connected actors, but stalled at TransformerEngine source
build in the user persistent cache while the Lyris user inode quota was over
soft quota and close to hard limit. The r30 launcher keeps the Ray fix and moves
the heavy build/cache path to node-local `/tmp`. OCI-HSG PARD-2/MathRL gates
remain scheduler-limited.

| Job | Scope | Current state | Account | Priority |
| ---: | --- | --- | --- | ---: |
| `2129203` | Lyris 235B SWE-RL baseline step-1 retry | `RUNNING/None`, started `2026-06-15T07:57:46`; stdout/runtime logs present, TE build active with `/tmp` cache env and fresh `ninja`/`c++`/`cc1plus` work, no parsed step metrics yet | `coreai_dlalgo_llm` | `77298` |
| `2129271` | Lyris 235B SWE-RL PARD step-1 retry | `PENDING/Dependency`, afterok `2129203`; re-submitted with `ray.sub` shell trace disabled | `coreai_dlalgo_llm` | `77284` |
| `2129272` | Lyris 235B SWE-RL PARD-2 step-1 retry | `PENDING/Dependency`, afterok `2129203`; re-submitted with `ray.sub` shell trace disabled | `coreai_dlalgo_llm` | `77284` |
| `3308774` | 235B SWE-RL PARD-2 step-1 proof | `PENDING/Priority`; no stdout/runtime log yet | `nemotron_n3_post` | `133709` |
| `3315380` | 235B MathRL baseline 10-step | `PENDING/Priority`; no stdout/runtime log yet | `nemotron_n3_post` | `133657` |
| `3315381` | 235B MathRL PARD K3 10-step | `PENDING/Priority`; no stdout/runtime log yet | `nemotron_n3_post` | `133657` |
| `3315382` | 235B MathRL PARD K5 10-step | `PENDING/Priority`; no stdout/runtime log yet | `nemotron_n3_post` | `133657` |

The next useful action is to watch `2129203` until it emits parsed step metrics
before treating r30 as a working example. The OCI-HSG jobs should also continue
to be monitored until a ready gate emits parsed step metrics; their scheduler
start estimates are volatile and should be read from
`docs/nemorl_235b_active_gates_latest_20260615.md`.

## Excluded Historical Attempts

The Lyris integrated max-step-10 matrix is not a usable Math or SpecDec
training example. `docs/lyris_nemorl_integrated_specdec_maxsteps10_status_20260613.md`
is terminal with `FAILED=12` and `CANCELLED by 2001147693=6`, and
`docs/lyris_nemorl_integrated_specdec_maxsteps10_metrics_20260613.md` reports
`missing_log=18`. Keep it as negative evidence only.

The older 235B MathRL latest-main attempts are not usable examples either.
OCI-HSG jobs `3290316`-`3290318` and py3 guard jobs `3315267`-`3315269` were
cancelled before runtime. Lyris retry3 jobs `2113812`-`2113814` reached past
the earlier Python/decord/soundfile/tensordict blockers, but failed during
isolated policy worker creation on
`ModuleNotFoundError: No module named 'transformers.models.ernie4_5_vl_moe'`.
The active MathRL proof remains `3315380`/`3315381`/`3315382`.
