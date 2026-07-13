# SWE-RL Online Execution Success Audit - 2026-06-14

Checked from current Slurm/log state on Lyris and OCI-HSG.

## Verified Online Correctness

| Scope | Cluster | Job | Method | State | Proof |
| --- | --- | --- | --- | --- | --- |
| Qwen8 official PARD-2 online smoke | OCI-HSG | `3279229` | `pard2` | `COMPLETED 0:0` | `SETUP COMPLETE`, `Step 1/2`, `Step 2/2`, draft training/refit enabled on both steps, `target_proj.weight` loaded, no `Traceback`/`RuntimeError`/`ValueError`/`ERROR`. |
| Qwen8 official PARD-2 online 20-step | OCI-HSG | `3279589` | `pard2` | `COMPLETED 0:0` | `SETUP COMPLETE`, `Step 1/20`, `Step 20/20`, draft training/refit enabled on all 20 steps, `target_proj.weight` loaded, no `Traceback`/`RuntimeError`/`ValueError`/`ERROR`. |

## Qwen235B SWE-RL Gate

| Cluster | Job | Method | State | Proof Gap |
| --- | --- | --- | --- | --- |
| Lyris | `2124030` | baseline | `RUNNING` | Still in `transformer-engine` build; no `SETUP COMPLETE`, rollout, `Step 1`, or `global_step`. |
| Lyris | `2124031` | PARD | `RUNNING` | Still in `transformer-engine` build; no `SETUP COMPLETE`, rollout, `Step 1`, or `global_step`. |
| Lyris | `2124032` | PARD-2 | `RUNNING` | Still in `transformer-engine` build; no `SETUP COMPLETE`, rollout, `Step 1`, or `global_step`. |
| OCI-HSG | `3308774` | PARD-2 | `PENDING (Priority)` | Not started; no runtime log yet. |

Conclusion: official PARD-2 online drafter training is verified on the smaller Qwen8 correctness path, but Qwen235B SWE-RL/SWEBench has no successful step yet.
