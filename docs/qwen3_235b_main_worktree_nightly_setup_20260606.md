# Qwen3-235B NeMo-RL Main Worktree / Nightly Setup

Date: 2026-06-06 PDT

## Remote Access

SSH to `oci-hsg-cs-001-vscode-02` is working again as of 2026-06-06 15:08 PDT.

## New Main-Based Worktree

| Field | Value |
|---|---|
| Remote host | `oci-hsg-cs-001-vscode-02` |
| Base repo | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL` |
| New worktree | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606` |
| Branch | `sna/qwen235b-main-vllm020-20260606` |
| Base ref | `refs/remotes/origin/main` |
| Commit | `37526dfac fix(grpo): penalize invalid tool call and malformed thinking (#2656)` |

The short ref name `origin/main` is ambiguous in the remote repo because a
local branch named `refs/heads/origin/main` also exists. Use
`refs/remotes/origin/main` explicitly when creating or resetting main-based
worktrees.

## Container Setup

The user-provided nightly installer is:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/docker_nightly_install_sbatch.sh
```

It was submitted from:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/.containers/nightly
```

Current Slurm job:

| Job | Purpose | Status at setup time |
|---:|---|---|
| `3196542` | Download `nvcr.io/nvidian/nemo-rl:nightly` into the new worktree container dir | Completed; produced `nemo_rl_nightly_20260606.sqsh` and `nemo_rl_nightly.sqsh` symlink |
| `3196672` | First import/version smoke inside the new nightly image | Failed with launcher issue: `python3: command not found` inside container |
| `3196734` | Second import/version smoke using `/opt/venv/bin/python` | Failed; the nightly image uses `/opt/nemo_rl_venv/bin/python`, not `/opt/venv/bin/python` |
| `3196760` | Corrected import/version smoke using `/opt/nemo_rl_venv/bin/python` | Cancelled after confirming the next launcher issue |
| `3196990` | Import smoke with `--container-remap-root` only | Failed; home mount hid image `/root/.local/...` Python tree |
| `3197009` | Import smoke with `--container-remap-root --no-container-mount-home` | Completed; NeMo-RL import passed, base venv has no `vllm` |
| `3197094`/`3197095`/`3197096` | Qwen3-235B generation smoke with `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` | Failed; VllmGenerationWorker used base venv and raised `ModuleNotFoundError: vllm` |
| `3197117`/`3197118`/`3197119` | Qwen3-235B generation smoke with worker venv enabled | Failed/cancelled after model load because the smoke runner mishandled dict-like tokenizer output |
| `3197236`/`3197248`/`3197259` | Qwen3-235B generation smoke after tokenizer fix | Completed; baseline, public PARD K3, public PARD K5 all passed |

Expected output after completion:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/.containers/nightly/nemo_rl_nightly_20260606.sqsh
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/.containers/nightly/nemo_rl_nightly.sqsh
```

The older shared nightly symlink under `/lustre/fsw/.../users/sna/nemo-rl`
currently points at `nemo_rl_nightly_20260502.sqsh`, so the new download is
preferred for main/vLLM 0.20 compatibility checks.

## Submodule State

The worktree is intentionally left with submodules uninitialized:

```text
-92635e74f4fb16784268b9a9fd7b7d6a83fff6c5 3rdparty/Automodel-workspace/Automodel
-50af84a5e2a7142c7d496dd9ea76b1e9d64202bd 3rdparty/Gym-workspace/Gym
-4bb633034aab2eaa2f7d7c2a771ef2bc80337be5 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
```

A first shallow submodule init attempt stalled while cloning Gym. It was
stopped and deinitialized so the main worktree stays clean. Retry submodules
only when the first main/nightly smoke needs them.

## Resolved Runtime Notes

The nightly image uses separate Python environments:

| Env | Contents |
|---|---|
| `/opt/nemo_rl_venv` | NeMo-RL driver env; `nemo_rl==0.6.0+37526dfac`, `torch==2.11.0+cu130`, `ray==2.55.1`; no top-level `vllm` |
| `/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker` | vLLM actor env; contains `vllm==0.20.0` |

Required settings for the main/nightly smoke:

```text
CONTAINER_REMAP_ROOT=true
NEMO_RL_PY_EXECUTABLES_SYSTEM=0
NEMO_RL_VENV_DIR=/opt/ray_venvs
RAY_PYTHON_VERSION=3.13.13
RAY_VERSION=2.55.1
```

The `ray.sub` in the main worktree was patched to append
`--container-remap-root` only when `CONTAINER_REMAP_ROOT=true`. Without
`--no-container-mount-home`, the image's `/root/.local/share/uv/python/...`
tree is hidden by the mounted home directory. Without disabling
`NEMO_RL_PY_EXECUTABLES_SYSTEM`, VllmGenerationWorker runs from the base
driver env and cannot import `vllm`.

## Completed Smoke

The initial generation smoke is intentionally small:

```text
model=Qwen/Qwen3-235B-A22B
draft=amd/PARD-Qwen3-0.6B
shape=1n4g tp4 draft_tp4 prompt_limit8 batch8 max_new_tokens128
source_vllm_site=container
python=/opt/nemo_rl_venv/bin/python
worker_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
```

Results:

| Job | Case | K | Status | Gen tokens | Gen elapsed sec | Gen throughput tok/s | Acceptance rate | Acceptance length |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| `3197236` | baseline | 0 | pass | 1024 | 16.847 | 60.783 | 0.000 | 1.000 |
| `3197248` | public PARD | 3 | pass | 1024 | 12.251 | 83.583 | 0.616 | 2.847 |
| `3197259` | public PARD | 5 | pass | 1024 | 11.225 | 91.228 | 0.504 | 3.522 |

Generation throughput speedup in this small compatibility smoke:

| Case | Speedup vs baseline |
|---|---:|
| public PARD K3 | 1.375x |
| public PARD K5 | 1.501x |

This is a compatibility and version-skew control gate, not a full GRPO
throughput claim. The next step is to expand the same main/nightly setup to the
matched OpenMath benchmark shape and then to Full-GRPO.

## Expanded OpenMath Generation Control

The next latest-main/nightly VllmGeneration control used a larger OpenMath
shape:

```text
model=Qwen/Qwen3-235B-A22B
draft=amd/PARD-Qwen3-0.6B
shape=1n4g tp4 draft_tp4 prompt_limit64 batch32 max_new_tokens256
source_vllm_site=container
worker_vllm=0.20.0
```

Results:

| Job | Case | K | Status | Gen tokens | Gen elapsed sec | Gen throughput tok/s | Speedup | Acceptance rate | Acceptance length |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| `3197507` | baseline | 0 | pass | 16384 | 61.506 | 266.382 | 1.000x | 0.000 | 1.000 |
| `3197508` | public PARD | 3 | pass | 16384 | 41.069 | 398.938 | 1.498x | 0.548 | 2.645 |
| `3197509` | public PARD | 5 | pass | 16384 | 38.662 | 423.778 | 1.591x | 0.422 | 3.108 |

This confirms that the NeMo-RL latest-main/nightly path using vLLM `0.20.0`
does show Qwen3-235B public PARD generation-path speedup on the OpenMath
bs32/o256 control.

## Latest-Main Full-GRPO Queue

After the expanded generation control passed, the same main/nightly setup was
used to submit no-stop Full-GRPO smoke jobs:

```text
shape=32n4g generation_tp4 train_tp2_pp8_cp2_ep16 gbs256
max_steps=5
max_new_tokens=256
fixed_decode=true
draft=amd/PARD-Qwen3-0.6B
driver_python=/opt/nemo_rl_venv/bin/python
worker_vllm=0.20.0 under /opt/ray_venvs
```

| Job | Case | K | Status at submission | Notes |
|---:|---|---:|---|---|
| `3197584` | baseline | 0 | pending | latest-main/nightly baseline for E2E comparison |
| `3197585` | public PARD | 5 | pending | public PARD K5 Full-GRPO step5 |
| `3197586` | public PARD | 3 | pending | public PARD K3 Full-GRPO step5, reuses baseline comparison |

Poll at 2026-06-06 17:00 PDT reported all three step5 jobs pending for
priority with scheduled start time `2026-06-06T19:09:53` PDT.

Dependent step20 Full-GRPO jobs were also submitted so longer E2E averages run
only if the corresponding step5 job succeeds:

| Job | Case | K | Dependency | Status | Notes |
|---:|---|---:|---|---|---|
| `3197620` | baseline | 0 | `afterok:3197584` | pending dependency | latest-main/nightly baseline step20 |
| `3197621` | public PARD | 5 | `afterok:3197585` | pending dependency | public PARD K5 Full-GRPO step20 |
| `3197622` | public PARD | 3 | `afterok:3197586` | pending dependency | public PARD K3 Full-GRPO step20 |

The key unresolved evidence gate remains Qwen3-235B no-stop Full-GRPO E2E
throughput and step-time. Generation-only speedup is now proven in the
latest-main/vLLM0.20 runtime, but E2E speedup must wait for these jobs to run
and emit Full-GRPO metrics.
