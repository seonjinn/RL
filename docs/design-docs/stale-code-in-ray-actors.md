# Code Changes Can Be Silently Ignored at Runtime

## The issue

A developer edits a Python file in their NeMo-RL checkout, submits a job, and
the job runs **old code** from the container without any error or warning.

There is no signal that this happened. The job appears to run normally. The
developer's change simply does not exist in the running process. This applies
to bug fixes, feature work, and even `print()` statements added for debugging.

## Why it happens

NeMo-RL runs each Ray actor class inside its own virtual environment. These
venvs are pre-built at container image build time and stored at
`/opt/ray_venvs/<actor_class>/`. When the container starts, the runtime checks
whether the venv already exists and, if so, uses it directly without
reinstalling or syncing source files (`NRL_VENVS_TRUST_EXISTING=1`).

The result: the actor's Python process imports from the venv's
`site-packages/`, which contains a frozen copy of the code as it existed when
the container image was built — not the developer's current checkout.

The developer's checkout IS mounted into the container (via
`--container-mounts`), but the actor processes do not import from it.

## Where this happens

The following actor classes each have their own venv and will ignore changes
to files they import:

| Actor | Venv location | Files that will be ignored if changed |
|---|---|---|
| VllmAsyncGenerationWorker | `/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/` | Everything under `nemo_rl/models/generation/vllm/` |
| MegatronPolicyWorker | `/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/` | Everything under `nemo_rl/models/policy/`, `nemo_rl/utils/` |
| NemoGym | `/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/` | Everything under `nemo_rl/environments/` |
| AsyncTrajectoryCollector | `/opt/ray_venvs/nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector/` | Everything under `nemo_rl/algorithms/` |
| ReplayBuffer | `/opt/ray_venvs/nemo_rl.algorithms.async_utils.ReplayBuffer/` | Everything under `nemo_rl/algorithms/` |

Additionally, the NeMo Gym third-party servers have their own venvs at
`/opt/gym_venvs/` with similar behavior.

### What is NOT affected

The **driver process** (the main script launched by `uv run`) uses PYTHONPATH
pointing to the live checkout. Changes to files only imported by the driver
(e.g., top-level training loop orchestration, config parsing) take effect
immediately.

## How to reproduce

1. Edit any file inside `nemo_rl/models/generation/vllm/` (e.g., add a
   `print("HELLO")` at module top level).
2. Submit a job with `NRL_VENVS_TRUST_EXISTING=1` (the default for all
   launcher scripts).
3. Check the logs. "HELLO" does not appear.

## Solution: `.pth` injection into actor venvs

The development workflow for NeMo-RL is to mount your source tree into the
container and iterate from there. The fix is to make mounting actually work
end-to-end by ensuring the mounted code takes precedence over the venv's
frozen copy.

### Mechanism

When reusing an existing venv (`NRL_VENVS_TRUST_EXISTING=1`), write a `.pth`
file into the venv's `site-packages/` directory that points to the mounted
source tree:

```python
# In create_local_venv(), after the trust_existing fast path:
def _maybe_inject_dev_overlay(venv_path: str) -> None:
    """If a live source tree is mounted, make it take precedence over the venv."""
    source_root = os.environ.get("NRL_DEV_OVERLAY")
    if not source_root:
        return
    site_packages = glob.glob(f"{venv_path}/lib/python*/site-packages")
    if not site_packages:
        return
    pth_file = os.path.join(site_packages[0], "_nrl_dev_overlay.pth")
    with open(pth_file, "w") as f:
        f.write(source_root + "\n")
    logger.info(f"Dev overlay: {source_root} → {pth_file}")
```

### How it works

The `.pth` file mechanism is built into CPython's `site` module. At
interpreter startup, Python reads every `.pth` file in `site-packages/` and
prepends the listed paths to `sys.path` — before any package imports occur.

This means:
- `import nemo_rl.models.generation.vllm.vllm_worker_async` resolves to the
  mounted checkout (your code), not the venv's installed copy.
- Third-party dependencies (vllm, torch, ray, etc.) still come from the venv.
- No network access, no reinstall, no rebuild.

### Usage

Set `NRL_DEV_OVERLAY` to your checkout root in the launcher:

```bash
export NRL_DEV_OVERLAY="/lustre/fs1/.../nemo-rl-super-vllm0.20"
```

Or launcher scripts can set it automatically when a mount is detected:

```bash
if [[ -d "${NEMORL}/nemo_rl" ]]; then
  export NRL_DEV_OVERLAY="${NEMORL}"
fi
```

### What this covers

- All Ray actor venvs under `/opt/ray_venvs/` — the injection happens in
  `create_local_venv()` which is the single entry point for all actor venvs.
- The Gym venvs under `/opt/gym_venvs/` can use the same mechanism (a `.pth`
  file pointing to the Gym source tree).

### What this does NOT cover

- Third-party dependencies installed in the venv (e.g., a different version of
  vllm or torch). Those still come from the container. This is intentional —
  the mounted source tree should not override compiled wheels.

### Gating

The injection is gated behind `NRL_DEV_OVERLAY` being set. When unset (the
default in CI and production), behavior is unchanged — venvs are used as-is.

## Additional fix: default `RAY_LOG_SYNC_FREQUENCY` in ray.sub

Separately, even when code changes ARE picked up, `print()` and logging output
from actor processes is not included in `ray-driver.log`. It is written to
`/tmp/ray/session_latest/logs/worker-*.out` inside the container, which is
ephemeral and not synced to the job log directory by default.

Fix: default `RAY_LOG_SYNC_FREQUENCY=30` in `ray.sub` so that worker logs are
always synced to the job log directory every 30 seconds. This ensures actor
output is available for post-mortem debugging without requiring the developer
to know about this setting.

```bash
# In ray.sub:
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-30}
```

## Summary

| Problem | Fix | Gating |
|---|---|---|
| Actor venvs shadow mounted code | `.pth` injection in `create_local_venv()` | `NRL_DEV_OVERLAY` env var |
| Actor stdout not in job logs | Default `RAY_LOG_SYNC_FREQUENCY=30` | Always on (opt-out by setting to empty) |
