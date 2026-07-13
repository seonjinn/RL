# Lyris SWE-RL Full-GRPO Staged-SIF Smoke r1 Launch

Launched: `2026-06-14T08:44-07:00` (`2026-06-14T15:44Z`).

Purpose: verify that Qwen3-235B SWE-RL can pass the previous W&B and missing-SIF blockers on Lyris by running a small step-1 baseline/PARD/PARD-2 smoke against only staged SWE-rebench images.

Dataset snapshot:

- Filter root: `/lustre/fsw/coreai_dlalgo_llm/users/sna/datasets/swe/filtered/staged_sif_20260614_1543`
- Train JSONL: `/lustre/fsw/coreai_dlalgo_llm/users/sna/datasets/swe/filtered/staged_sif_20260614_1543/train.jsonl`
- Validation JSONL: same as train for this smoke; the staged validation subset is empty.
- Train rows: `295`
- Staged SIFs: `295`
- Remaining required SIFs from the original failing rollout: `1,394`
- Image root: `/lustre/fsw/coreai_dlalgo_llm/users/sna/images/swerebench`

Submitted jobs:

| job_id | method | steps | K | state at launch check |
| ---: | --- | ---: | ---: | --- |
| `2120443` | baseline | 1 | 0 | `PENDING (Priority)` |
| `2120444` | PARD | 1 | 5 | `PENDING (Priority)` |
| `2120445` | PARD-2 | 1 | 1 | `PENDING (Priority)` |

Tracker: `latest_lyris_swerl_qwen235b_fullgrpo_stagedsif_smoke_r1_20260614_jobs.csv`

Status artifact: `docs/lyris_swerl_fullgrpo_stagedsif_smoke_r1_status_20260614.md`

Launcher notes:

- `experiments/eagle3_online/submit_lyris_swerl_qwen235b_fullgrpo_specdec_matrix_20260613.sh` now keeps Lyris on the active SSH ControlMaster by default while retaining `-S none` as the OCI-HSG default.
- The dry-run passed for baseline/PARD/PARD-2 step-1 before submit.
- Actual submit used a transient W&B API key environment and did not persist the key in repo files.
