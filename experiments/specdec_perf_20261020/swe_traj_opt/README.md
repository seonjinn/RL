# SWE trajectory phase optimization

Lossless system changes that cut the per-trajectory time of the Q30 SWE V2 rollout. Each item removes waste in one trajectory phase. No item changes the prompts, the sampled tokens, the tool results or the reward.

## Why

On the DSpark K8 ready-first + eval450 + drop4 arm, a trajectory takes 126.6 s. The model calls take 31.5 s. The startup phases (run_infer imports, connect, init) take 50.5 s and the final eval takes 32.2 s. The startup phases now cost more than the model.

The OpenHands setup tree mounts read-only in the agent container, and its venv has 443 .pyc files for 26,515 .py files. Each rollout thus compiles every module it imports, once in run_infer and once in the action server. On the login node the action server import takes 27-70 s without .pyc files and 9 s with them.

## Ladder

Each arm adds one item to the arm before it. All arms run the same NeMo-RL commit, the same Gym fork commit and the same nv-OpenHands fork commit. Only knobs differ, and each knob defaults to off.

| Step | Item | Phase | Knob |
|---|---|---|---|
| L0 | control | - | none |
| L1 | O1: compile the setup tree .pyc files at Gym setup | pre, connect | `precompile_setup_bytecode` (Gym) |
| L2 | O8: place each new rollout on the least-loaded vLLM endpoint (Gym#4989) | model | `endpoint_placement: least_loaded` (Gym) |
| L3 | O2 + O3: run_infer without `poetry run`, poll /alive every 0.1 s | pre, connect | `OH_DIRECT_VENV_PYTHON`, `OH_RUNTIME_ALIVE_POLL_S` (nv-OpenHands) |
| L4 | O4 + O5: reset the repo in one batched command, delete refs in one batch | init | `OH_FAST_DEEP_RESET` (nv-OpenHands) |
| L5 | O7: poll for the agent output every 0.2 s in the eval container | final eval | `eval_predictions_poll_interval_s` (Gym) |

Gym `openhands_env` sets the nv-OpenHands knobs. The agent command exports each entry before run_infer.sh.

Individual speedup = step time of the arm before / step time of the arm. Cumulative speedup = L0 step time / step time of the arm. The phase breakdown comes from `swe_agent_timeline.py`.

Excluded because they can change the rollout: FP8 KV cache, KV cache kept across refit, parallel test runs.

O6 (fast teardown) is not in the ladder. The exit phase takes about 4.2 s. About 2.1 s of it runs in the instance after the controller stops. A forced kill can leave tmux sessions and lose output writes for a gain of 2 s or less. The fork commit adds the `deep_reset_time`, `complete_runtime_time` and `close_runtime_time` metrics, so each arm shows the exit split.

## Run

Base: DSpark K8 ready-first + eval450 + drop4 (`sc_swe_rf_eval450_drop4_regen_frozen_dspark_k8.yaml`), 9 nodes, 16 vLLM replicas, 20 steps, no checkpoints. One clone per arm. All arms of one window run at the same time.

| Step | Config |
|---|---|
| L0 | `swe_traj_l0_control.yaml` |
| L1 | `swe_traj_l1_pyc.yaml` |
| L2 | `swe_traj_l2_least_loaded.yaml` |
| L3 | `swe_traj_l3_direct_python_alive_poll.yaml` |
| L4 | `swe_traj_l4_fast_deep_reset.yaml` |
| L5 | `swe_traj_l5_eval_poll.yaml` |

Each step config adds one `traj_opt_*.yaml` fragment to the step before it. L0 pins `seonjinn/nv-OpenHands` commit 1fe1c113c (`traj_opt_oh_fork.yaml`). That commit is 54d2a95 plus the opt-in gates and the exit metrics.

The branch points the Gym submodule at `seonjinn/Gym` branch `sna/swe-traj-opt-1024`. Run `git submodule sync --recursive` before `git submodule update` in an old clone.

## SpecDec on the optimized baseline

The five items cut the non-model time of a trajectory. The drafter cuts only the model time. Thus the SpecDec gain can grow on the optimized arm. A 2x2 run measures it in one window: L0 and L5, each with and without the DSpark K8 drafter.

| Arm | Drafter | Items | Config |
|---|---|---|---|
| b0 | none | none | `swe_traj_l0_baseline.yaml` |
| d0 | DSpark K8 | none | `swe_traj_l0_control.yaml` |
| b5 | none | L5 | `swe_traj_l5_baseline.yaml` |
| d5 | DSpark K8 | L5 | `swe_traj_l5_eval_poll.yaml` |

SpecDec gain = step time of the b arm / step time of the d arm at the same level. The b0 and d0 arms need clones with no .pyc files in the OpenHands tree.

## Admission depth

The 2x2 timeline shows that the ready-first gate sets the step time. The gate admits one batch of 4 prompts per trainer step, with one batch of lookahead. At step start only 2.4 to 2.8 groups run, and the vLLM engines have about 2 requests each. The exposed generation time is approximately the span of the last group minus its age at step start.

`rf_staleness2.yaml` sets `max_staleness_versions: 2`. The sampler then needs 12 buffer slots, so the in-flight and buffer caps go from 8 to 12. These three values change together as one axis. The cost is one more version of off-policy lag.

| Arm | Drafter | Config |
|---|---|---|
| b0s2 | none | `swe_traj_l0_baseline_staleness2.yaml` |
| d0s2 | DSpark K8 | `swe_traj_l0_control_staleness2.yaml` |

Compare each arm with b0 or d0 of the 2x2.

## Notes

- O1 writes .pyc files into the setup tree of the clone. A later control arm in that clone is not a control. Keep a clone with an uncompiled tree for L0.
- O2 needs `VIRTUAL_ENV` in the agent shell. Gym exports it. Without it, run_infer.sh logs a message and uses `poetry run`.
- O4 + O5 is strict. It classifies all refs before its first write. On any error it runs the default command sequence, which also cleans up a partial run. A local test gives the same repo state as the default sequence for each mutation case. nv-OpenHands 47b9861ac tried a best-effort in-place reset and 8b77317eb reverted it.
- Gym#4989 also adds `uvicorn_reuse_port`. The ladder does not turn it on.