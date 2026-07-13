# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r11_r13_r14_r15_latest_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2121275 | baseline | 1 | RUNNING | 18 | [36m(AsyncTrajectoryCollector pid=1461859, ip=10.66.4.108)[0m Traceback (most recent call last): / [36m(AsyncTrajectoryCollector pid=1461859, ip=10.66.4.108)[0m   File "/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL-SWE_bench-20260613/nemo_rl/algorithms/async_utils/trajectory_collector.py", line 443, in _run_prompt_group_worker / [36m(AsyncTrajectoryCollector pid=1461859, ip=10.66.4.108)[0m     nemo_gym_rollout_result = run_async_nemo_gym_rollout( / [36m(AsyncTrajectoryCollector pid=1461859, ip=10.66.4.108)[0m         policy_generation=self.policy_generation, / [36m(AsyncTrajectoryCo |
| 2121276 | pard2 | 1 | FAILED | 17 | echo "[SETUP] apptainer install attempt $attempt failed, retrying..." / sleep 10 / done / if [ $RET -ne 0 ]; then / echo "[SETUP] WARNING: apptainer installation failed after $RETRIES attempts" / fi |
| 2121299 | pard | 1 | FAILED | 18 | Traceback (most recent call last): / File "/opt/nemo_rl_venv/lib/python3.13/site-packages/hydra/_internal/config_loader_impl.py", line 390, in _apply_overrides_to_config / OmegaConf.update(cfg, key, value, merge=True) / ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ / File "/opt/nemo_rl_venv/lib/python3.13/site-packages/omegaconf/omegaconf.py", line 741, in update / root.__setattr__(last_key, value) |
| 2121322 | pard2 | 1 | COMPLETING | 18 | [2026-06-14T15:11:58.659] error: *** STEP 2121322.24 ON lyris0163 CANCELLED AT 2026-06-14T15:11:58 DUE to SIGNAL Terminated *** |
| 2121346 | pard | 1 | PENDING | 0 | no local log/error excerpt available |
| 2121347 | pard2 | 1 | PENDING | 0 | no local log/error excerpt available |
