# Qwen3-235B Math RL latest-main launch - 2026-06-13

## Summary

Launched Qwen3-235B Math RL Full-GRPO smoke jobs from clean latest-main NeMo-RL worktrees. OCI-HSG is queued; Lyris is also launched as a faster-scheduling mirror.

Comparison rule: speedups must be computed only against a baseline with the same KV-cache dtype. Standalone OSL32K `fp8kv` runs use an `fp8` baseline; the Math RL retry3 jobs below use `VLLM_KV_CACHE_DTYPE=auto` for baseline and PARD jobs.

## OCI-HSG

- Remote host: `oci-hsg-cs-001-vscode-02`
- Remote worktree: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-mathrl-20260613`
- NeMo-RL `main` HEAD: `231462c16f306ec5429d1841b353720a511064ed`
- Config: `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml`
- Account / partition: `coreai_dlalgo_llm` / `batch`
- Container: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/.containers/nightly/nemo_rl_nightly.sqsh`
- HF cache: `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home`
- Target model: `Qwen/Qwen3-235B-A22B`
- PARD draft model: `amd/PARD-Qwen3-0.6B`

## Preflight

Validated before submission:

- Top-level worktree HEAD matches `main` SHA `231462c16f306ec5429d1841b353720a511064ed`.
- Required files exist: `examples/run_grpo.py`, `ray.sub`, and Qwen3-235B performance config.
- Submodules initialized at pinned commits:
  - Automodel: `92635e74f4fb16784268b9a9fd7b7d6a83fff6c5`
  - Gym: `f82b601a9f5951793226cbe2d77336b677c6311e`
  - Megatron-Bridge: `823b951952e90092a5adc6864dda9631a748442c`
  - Megatron-LM under Megatron-Bridge: `6204b925f3da8b998524c6bb47a9ca779d95ce2e`
- Container, HF cache, Qwen3-235B tokenizer snapshot, and PARD draft cache exist.
- Launcher syntax and dry-run passed for baseline, PARD K=3, and PARD K=5.

## Jobs

| job_id | method | steps | decode | draft | K | status at launch |
|---:|---|---:|---:|---|---:|---|
| 3290316 | baseline | 10 | 256 fixed | none | 0 | PENDING `(Priority)` |
| 3290317 | PARD | 10 | 256 fixed | `amd/PARD-Qwen3-0.6B` | 3 | PENDING `(Priority)` |
| 3290318 | PARD | 10 | 256 fixed | `amd/PARD-Qwen3-0.6B` | 5 | PENDING `(Priority)` |

OCI-HSG job CSV:

`latest_oci_hsg_qwen235b_mathrl_latest_main_20260613_jobs.csv`

OCI-HSG log root:

`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260613_qwen235b_mathrl_latest_main_fixed256`

## Lyris Mirror

Created and used a separate clean latest-main Lyris worktree instead of the dirty `sj/hybridep-support` checkout.

- Remote host: `login-lyris`
- Remote worktree: `/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-main-mathrl-20260613`
- NeMo-RL `main` HEAD: `231462c16f306ec5429d1841b353720a511064ed`
- Account / partition: `coreai_dlalgo_llm` / `gb200`
- Container: `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo-rl-nightly-ultra.sqsh`
- HF cache: `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home`
- Lyris SLURM shape: `32` nodes x `4` GPUs per node, no `--gres`, no `--segment`
- Original Lyris Python/Ray path: `uv run --python /opt/nemo_rl_venv/bin/python3`, `RAY_VERSION=2.54.0`
- Retry Lyris Python/Ray path: direct `/opt/nemo_rl_venv/bin/python`, `RAY_USE_EXISTING_ENV=true`, `USE_SYSTEM_ENV=true`

Validated before Lyris submission:

- Top-level worktree HEAD matches `main` SHA `231462c16f306ec5429d1841b353720a511064ed`.
- Submodules initialized at the same pinned commits listed above.
- Container, HF cache, Qwen3-235B tokenizer snapshot, and PARD draft cache exist.
- Lyris dry-run printed the expected `sbatch` line without OCI `--gres` or `--segment`.

| job_id | method | steps | decode | draft | K | status after launch |
|---:|---|---:|---:|---|---:|---|
| 2113550 | baseline | 10 | 256 fixed | none | 0 | FAILED |
| 2113551 | PARD | 10 | 256 fixed | `amd/PARD-Qwen3-0.6B` | 3 | FAILED |
| 2113552 | PARD | 10 | 256 fixed | `amd/PARD-Qwen3-0.6B` | 5 | CANCELLED |

Original Lyris attempt outcome:

- `2113550` and `2113551` started Ray, then failed in the driver before GRPO steps.
- `ray-driver.log` root cause: `/opt/nemo_rl_venv/bin/python3` is Python `3.12.12`, while latest `main` declares `requires-python >=3.13.13,<3.14`.
- `2113552` was cancelled before reaching the same driver failure.

Lyris-only runtime workaround:

- OCI-HSG remains the clean latest-main reference.
- Lyris worktree was patched to relax Python metadata from `>=3.13.13` to `>=3.12.12` in `pyproject.toml`, `uv.lock`, and `research/template_project/pyproject.toml`.
- A one-node container import probe passed with direct `/opt/nemo_rl_venv/bin/python` and `PYTHONPATH` pointed at the patched Lyris worktree.
- A separate Python `3.13.9` uv-managed path was tested, but full `uv run --extra mcore` attempted to rebuild DeepEP and failed on CUDA mismatch: detected CUDA `12.9` vs PyTorch CUDA `13.0`.

Lyris retry jobs use the container system environment directly:

| job_id | method | steps | decode | draft | K | status after retry submit |
|---:|---|---:|---:|---|---:|---|
| 2113662 | baseline | 10 | 256 fixed | none | 0 | FAILED |
| 2113663 | PARD | 10 | 256 fixed | `amd/PARD-Qwen3-0.6B` | 3 | FAILED |
| 2113664 | PARD | 10 | 256 fixed | `amd/PARD-Qwen3-0.6B` | 5 | CANCELLED |

Retry1 outcome:

- The Python-version issue was bypassed, but driver import failed on `ModuleNotFoundError: No module named 'decord'`.
- `decord` and `decord2` are absent from the Lyris container, while latest `main` imports `decord` at module import time through `nemo_rl/data/multimodal_utils.py`.
- Lyris worktree was patched so `decord` is an optional import and is required only when audio/video loading actually uses the decord backend.
- A one-node import probe then passed: `from nemo_rl.algorithms.grpo import MasterConfig`.

Lyris retry2 jobs use the Python 3.12.12 system environment plus the optional-decord patch:

| job_id | method | steps | decode | draft | K | status after retry2 submit |
|---:|---|---:|---:|---|---:|---|
| 2113744 | baseline | 10 | 256 fixed | none | 0 | FAILED |
| 2113745 | PARD | 10 | 256 fixed | `amd/PARD-Qwen3-0.6B` | 3 | FAILED |
| 2113746 | PARD | 10 | 256 fixed | `amd/PARD-Qwen3-0.6B` | 5 | CANCELLED |

Retry2 outcome:

- The optional `decord` import issue was bypassed, but driver import then failed on `ModuleNotFoundError: No module named 'soundfile'`.
- Root causes were eager imports of text-irrelevant multimedia datasets:
  - `eval_datasets/__init__.py` imported `MMAUDataset`, which imports `soundfile`.
  - `response_datasets/__init__.py` imported `AudioMCQDataset` and `AVQADataset`, which import `soundfile`.
- Lyris worktree was patched so MMAU, AudioMCQ, and AVQA classes are imported lazily only when those datasets are selected.
- The next import blocker was `ModuleNotFoundError: No module named 'tensordict'` from latest-main `nemo_rl.data_plane.codec`.
- Staged a Lyris-only py312 package shim at `/lustre/fsw/coreai_dlalgo_llm/users/sna/py312_site/tensordict_0.12.4` containing `tensordict==0.12.4` and `pyvers==0.2.2`; a one-node container import probe passed with `IMPORT_OK`.

Lyris retry3 jobs use direct system Python, lazy multimedia imports, and the py312 `tensordict` shim:

| job_id | method | steps | decode | KV cache | draft | K | latest status |
|---:|---|---:|---:|---|---|---:|---|
| 2113812 | baseline | 10 | 256 fixed | auto | none | 0 | FAILED |
| 2113813 | PARD | 10 | 256 fixed | auto | `amd/PARD-Qwen3-0.6B` | 3 | FAILED |
| 2113814 | PARD | 10 | 256 fixed | auto | `amd/PARD-Qwen3-0.6B` | 5 | FAILED |

Retry3 observation:

- Ray heads and driver logs started for `2113812` and `2113813`.
- Both jobs passed the earlier Python/decord/soundfile/tensordict import blockers.
- Both jobs loaded `OpenMathInstruct-2`, initialized the Ray policy cluster on 32 nodes, and reached vLLM policy worker initialization.
- Baseline `2113812` initialized all 128 vLLM policy workers; PARD K=3 `2113813` reached the same setup path with `speculative_config.method=draft_model`, `num_speculative_tokens=3`, and `policy.generation.vllm_cfg.kv_cache_dtype=auto`.
- Both drivers showed repeated warning `Could not apply llama_eagle3 lm_head ownership patch: expected code snippet not found ...`; this has not yet caused a job failure.

Retry3 terminal refresh on `2026-06-15 06:01 PDT`:

- `2113812`, `2113813`, and `2113814` all failed before GRPO step metrics.
- The failure moved past the earlier Python/decord/soundfile/tensordict
  blockers and reached isolated policy worker creation.
- The terminal driver error was
  `ModuleNotFoundError: No module named 'transformers.models.ernie4_5_vl_moe'`.
- OCI-HSG `3290316`, `3290317`, and `3290318` were later checked with `sacct`
  and found cancelled before runtime. The current active OCI-HSG MathRL proof is
  the N3Post/system-Python set `3315380`, `3315381`, and `3315382`.

Lyris job CSV:

`latest_lyris_qwen235b_mathrl_latest_main_20260613_jobs.csv`

Lyris retry job CSV:

`latest_lyris_qwen235b_mathrl_latest_main_py312system_retry1_20260613_jobs.csv`

Lyris retry2 job CSV:

`latest_lyris_qwen235b_mathrl_latest_main_py312system_decordlazy_retry2_20260613_jobs.csv`

Lyris retry3 job CSV:

`latest_lyris_qwen235b_mathrl_latest_main_py312system_lazydeps_retry3_20260613_jobs.csv`

Lyris log root:

`/lustre/fsw/coreai_dlalgo_llm/users/sna/mathrl_latest_main_logs/20260613_qwen235b_mathrl_latest_main_fixed256_lyris`

Lyris retry log root:

`/lustre/fsw/coreai_dlalgo_llm/users/sna/mathrl_latest_main_logs/20260613_qwen235b_mathrl_latest_main_fixed256_lyris_py312system_retry1`

Lyris retry2 log root:

`/lustre/fsw/coreai_dlalgo_llm/users/sna/mathrl_latest_main_logs/20260613_qwen235b_mathrl_latest_main_fixed256_lyris_py312system_decordlazy_retry2`

Lyris retry3 log root:

`/lustre/fsw/coreai_dlalgo_llm/users/sna/mathrl_latest_main_logs/20260613_qwen235b_mathrl_latest_main_fixed256_lyris_py312system_lazydeps_retry3`

## Reproduction

Local launcher:

`experiments/eagle3_online/submit_oci_hsg_qwen235b_mathrl_latest_main_20260613.sh`

Dry-run command used:

```bash
SUBMIT=false METHODS="baseline pard_k3 pard_k5" MAX_STEPS=10 \
  OUT=/Users/sna/Nemo-RL_Qwen3_Roadmap/tmp/oci_hsg_qwen235b_mathrl_latest_main_all_dryrun_20260613.csv \
  bash experiments/eagle3_online/submit_oci_hsg_qwen235b_mathrl_latest_main_20260613.sh
```

Submit command used:

```bash
SUBMIT=true METHODS="baseline pard_k3 pard_k5" MAX_STEPS=10 \
  OUT=/Users/sna/Nemo-RL_Qwen3_Roadmap/latest_oci_hsg_qwen235b_mathrl_latest_main_20260613_jobs.csv \
  bash experiments/eagle3_online/submit_oci_hsg_qwen235b_mathrl_latest_main_20260613.sh
```

Status command:

```bash
ssh oci-hsg-cs-001-vscode-02 \
  "squeue -j 3290316,3290317,3290318 -o '%i|%T|%R|%P|%j|%D|%M|%L'"
```

Lyris submit command used:

```bash
REMOTE_HOST=login-lyris \
REMOTE_REPO=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-main-mathrl-20260613 \
CONTAINER=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo-rl-nightly-ultra.sqsh \
HF_HOME=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home \
PARTITION=gb200 WALLTIME=05:00:00 \
RUN_ID=20260613_qwen235b_mathrl_latest_main_fixed256_lyris \
LOG_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/mathrl_latest_main_logs/20260613_qwen235b_mathrl_latest_main_fixed256_lyris \
TOKENIZER_NAME=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65 \
GRES_FLAG='' SEGMENT='' \
UV_PYTHON=/opt/nemo_rl_venv/bin/python3 \
RAY_VERSION=2.54.0 RAY_PYTHON_VERSION='' \
RAY_PYTHON_SPEC=/opt/nemo_rl_venv/bin/python3 \
RAY_USE_EXISTING_ENV=false USE_SYSTEM_ENV=false \
PYTHON_RUNNER_OVERRIDE='' NEMO_RL_PY_EXECUTABLES_SYSTEM='' \
SUBMIT=true METHODS="baseline pard_k3 pard_k5" MAX_STEPS=10 \
OUT=/Users/sna/Nemo-RL_Qwen3_Roadmap/latest_lyris_qwen235b_mathrl_latest_main_20260613_jobs.csv \
bash experiments/eagle3_online/submit_oci_hsg_qwen235b_mathrl_latest_main_20260613.sh
```

Lyris status command:

```bash
ssh login-lyris \
  "squeue -j 2113550,2113551,2113552 -o '%i|%T|%R|%P|%j|%D|%M|%L'"
```
