# SWE trajectory phase optimization

Lossless system changes that cut the per-trajectory time of the Q30 SWE V2 rollout. Each item removes waste in one trajectory phase. No item changes the prompts, the sampled tokens, the tool results or the reward.

## Why

On the DSpark K8 ready-first + eval450 + drop4 arm, a trajectory takes 126.6 s. The model calls take 31.5 s. The startup phases (run_infer imports, connect, init) take 50.5 s and the final eval takes 32.2 s. The startup phases now cost more than the model.

The OpenHands setup tree mounts read-only in the agent container, and its venv has 443 .pyc files for 26,515 .py files. Each rollout thus compiles every module it imports, once in run_infer and once in the action server. On the login node the action server import takes 27-70 s without .pyc files and 9 s with them.

## Ladder

Each arm adds one item to the arm before it. All arms run the same NeMo-RL commit, the same Gym fork commit and the same nv-OpenHands commit. Only knobs differ, and each knob defaults to off.

| Step | Item | Phase | Knob |
|---|---|---|---|
| L0 | control | - | none |
| L1 | O1: compile the setup tree .pyc files at Gym setup | pre, connect | `precompile_setup_bytecode` (Gym) |
| L2 | O8: place each new rollout on the least-loaded vLLM endpoint (Gym#4989) | model | `endpoint_placement: least_loaded` (Gym) |
| L3 | O2 + O3: run_infer without `poetry run`, poll /alive every 0.1 s | pre, connect | nv-OpenHands env gate |
| L4 | O4 + O5: batch the init commands, delete refs in one batch first | init | nv-OpenHands env gate |
| L5 | O6 + O7: fast teardown, poll the eval result every 0.2 s | exit, final eval | nv-OpenHands env gate, Gym knob |

Individual speedup = step time of the arm before / step time of the arm. Cumulative speedup = L0 step time / step time of the arm. The phase breakdown comes from `swe_agent_timeline.py`.

Excluded because they can change the rollout: FP8 KV cache, KV cache kept across refit, parallel test runs.

## Run

Base: DSpark K8 ready-first + eval450 + drop4 (`sc_swe_rf_eval450_drop4_regen_frozen_dspark_k8.yaml`), 9 nodes, 16 vLLM replicas, 20 steps, no checkpoints. One clone per arm. All arms of one window run at the same time.

| Step | Config |
|---|---|
| L0 | `sc_swe_rf_eval450_drop4_regen_frozen_dspark_k8.yaml` |
| L1 | `sc_swe_rf_eval450_drop4_regen_frozen_dspark_k8_pyc.yaml` |
| L2 | `sc_swe_rf_eval450_drop4_regen_frozen_dspark_k8_pyc_least_loaded.yaml` |

The branch points the Gym submodule at `seonjinn/Gym` branch `sna/swe-traj-opt-1024`. Run `git submodule sync --recursive` before `git submodule update` in an old clone.

## Notes

- O1 writes .pyc files into the setup tree of the clone. A later control arm in that clone is not a control. Keep a clone with an uncompiled tree for L0.
- Gym#4989 also adds `uvicorn_reuse_port`. The ladder does not turn it on.