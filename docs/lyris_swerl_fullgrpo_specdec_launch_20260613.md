# Lyris SWE-RL Full-GRPO SpecDec Launch - 2026-06-13

Status captured at `2026-06-13 12:22 CEST`.

## Submitted Lyris Jobs

Prewarm/reconvert job:

| job_id | role | state at submit check | dependency |
| --- | --- | --- | --- |
| 2114333 | baseline max_steps=1 Megatron reconvert/prewarm | RUNNING | none |

Full-GRPO matrix submitted behind `afterok:2114333`:

| job_id | method | max_steps | draft model | K | state at submit check |
| --- | --- | ---: | --- | ---: | --- |
| 2114342 | baseline | 10 |  | 0 | PENDING, Dependency |
| 2114343 | suffix | 10 |  | 32 | PENDING, Dependency |
| 2114344 | PARD | 10 | `amd/PARD-Qwen3-0.6B` | 5 | PENDING, Dependency |
| 2114345 | PARD-2 | 10 | `amd/PARD2-Qwen3-8B` | 1 | PENDING, Dependency |
| 2114346 | Eagle-3 | 10 | `nvidia/Qwen3-235B-A22B-Eagle3` | 3 | PENDING, Dependency |

`scontrol show job` confirmed all five matrix jobs had `Dependency=afterok:2114333(unfulfilled)`.

## Launch Inputs

- Tracker CSV: `latest_lyris_swerl_qwen235b_fullgrpo_specdec_after_prewarm_r1_20260613_jobs.csv`
- Run ID: `20260613_lyris_swerl_qwen235b_fullgrpo_specdec_after_prewarm_r1`
- Remote repo: `/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL-SWE_bench-20260613`
- Branch/head: `ruit/SWE_bench`, `4cc7d70c89e83876b06805bf439e866b0fcbe708`
- Rui launcher: `/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL-SWE_bench-20260613/test_assets/qwen-235B/run_grpo_qwen3_235b_swe_scale_gen.sh`
- Account/partition: `coreai_dlalgo_llm`, `gb200`
- Container: `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/ruit-swe_bench-6dc8fabea-aarch64-060426-mcore-apptainer.squashfs`
- Target model: `/lustre/fsw/coreai_dlalgo_llm/users/sna/model_snapshots/Qwen3-235B-A22B-Thinking-2507`
- Train data: `/lustre/fsw/coreai_dlalgo_llm/users/sna/datasets/swe/blends/balanced_language.jsonl`
- Val data: `/lustre/fsw/coreai_dlalgo_llm/users/sna/datasets/swe/swe_public_datasets_val_swebench.jsonl`
- Persistent cache: `/lustre/fsw/coreai_dlalgo_llm/users/sna/.cache/qwen3_235b_thinking_swe_scale`

Pre-submit checks passed for the remote repo branch/head, launcher syntax, container path, model path, and both SWE dataset paths.

## Current Caveat

After Lyris submission was confirmed, local SSH/DNS started resolving both `login-lyris` and `oci-hsg-cs-001-vscode-02` to `172.30.3.254`, and TCP 22 timed out. This blocks live log/status refresh from the local machine, but it happened after Lyris Slurm submission and dependency verification completed.
