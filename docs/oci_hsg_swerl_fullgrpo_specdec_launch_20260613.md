# OCI-HSG SWE-RL Full-GRPO SpecDec Launch - 2026-06-13

Status: submitted, pending on OCI-HSG priority. The original suffix rows were canceled before allocation and replaced with py312 arctic retry jobs.

Launcher:

- Local: `experiments/eagle3_online/submit_lyris_swerl_qwen235b_fullgrpo_specdec_matrix_20260613.sh`
- Tracker: `latest_oci_hsg_swerl_qwen235b_fullgrpo_specdec_20260613_jobs.csv`
- Remote repo: `/lustre/fsw/portfolios/nemotron/users/ruit/evolution_rl`
- Rui launcher: `/lustre/fsw/portfolios/nemotron/users/ruit/evolution_rl/test_assets/qwen-235B/run_grpo_qwen3_235b_swe_scale_gen.sh`

Preflight checks passed:

- SSH host: `oci-hsg-cs-001-vscode-02`
- SLURM account: `nemotron_n3_post`
- SLURM partition: `batch`
- Partition time limit: `04:00:00`
- Container: `/lustre/fsw/portfolios/nemotron/users/ruit/enroot-images/ruit-swe_bench-6dc8fabea-aarch64-060426-mcore-apptainer.squashfs`
- Suffix site: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/.container_cache/arctic-inference-0.1.1`
- PARD-2 vLLM site: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat`
- Repo assets: `ray.sub` and `test_assets/qwen-235B/grpo_qwen3_235b_async_swe.yaml`

Post-submit recheck at `2026-06-13T06:00+02:00`:

- OCI-HSG non-interactive SSH still works for `oci-hsg-cs-001-vscode-02`.
- Remote user is `sna`; cluster reports `oci-hsg-cs-001`.
- `nemotron_n3_post` is present in the SLURM association list for the user.
- `batch` is the active/default partition and reports a `04:00:00` limit.
- Remote repo, Rui launcher, `ray.sub`, SWE config, container, suffix site, suffix native `_C*.so`, and PARD-2 vLLM site all still exist.
- HF cache, temporary patched launcher directory, and log root are writable.
- Rui launcher still passes `bash -n` and still contains the expected `SBATCH_ACCOUNT`, `SBATCH_PARTITION`, and extra-Hydra-overrides patch points.
- No duplicate submission was made during this recheck; the existing 10 jobs are still the active matrix.

Launcher fixes made before submission:

- OCI-HSG profile defaults to `nemotron_n3_post`, `batch`, and `04:00:00`.
- The submitted job uses a temporary patched copy of Rui's launcher in the writable user area.
- The temp launcher patches Rui's hardcoded account/partition lines.
- `SOURCE_VLLM_SITE` is added to `PYTHONPATH` for suffix/PARD-2 source-site usage.
- The OCI suffix source site now points to the Python 3.12 arctic build and preflight requires the native `_C*.so` extension.
- `REPO_ROOT` points back to Rui's repo, while checkpoints/logs go to the writable user area.
- Rui's `latest_235b_scale_gen_job_id.txt` is redirected under `BASE_LOG_DIR`.

Submitted jobs:

| job_id | method | max_steps | draft model | K | status |
| --- | --- | ---: | --- | ---: | --- |
| 3286445 | baseline | 10 |  | 0 | PENDING (Priority) |
| 3286458 | suffix | 10 |  | 32 | CANCELLED before allocation; replaced by `3288611` |
| 3286519 | pard | 10 | `amd/PARD-Qwen3-0.6B` | 5 | PENDING (Priority) |
| 3286521 | pard2 | 10 | `amd/PARD2-Qwen3-8B` | 1 | PENDING (Priority) |
| 3286522 | eagle3 | 10 | `nvidia/Qwen3-235B-A22B-Eagle3` | 3 | PENDING (Priority) |
| 3286523 | baseline | 20 |  | 0 | PENDING (Priority) |
| 3286524 | suffix | 20 |  | 32 | CANCELLED before allocation; replaced by `3288612` |
| 3286525 | pard | 20 | `amd/PARD-Qwen3-0.6B` | 5 | PENDING (Priority) |
| 3286526 | pard2 | 20 | `amd/PARD2-Qwen3-8B` | 1 | PENDING (Priority) |
| 3286527 | eagle3 | 20 | `nvidia/Qwen3-235B-A22B-Eagle3` | 3 | PENDING (Priority) |

Suffix retry jobs:

| job_id | method | max_steps | K | status |
| --- | --- | ---: | ---: | --- |
| 3288611 | suffix | 10 | 32 | PENDING (Priority) |
| 3288612 | suffix | 20 | 32 | PENDING (Priority) |
