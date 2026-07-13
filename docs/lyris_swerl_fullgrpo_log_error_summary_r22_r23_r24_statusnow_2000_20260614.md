# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r22_r23_r24_statusnow_2000_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2123212 | pard2 | 1 | RUNNING | 18 | [36m(AsyncTrajectoryCollector pid=377005, ip=10.66.3.130)[0m Traceback (most recent call last): / [36m(AsyncTrajectoryCollector pid=377005, ip=10.66.3.130)[0m   File "/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL-SWE_bench-20260613/nemo_rl/algorithms/async_utils/trajectory_collector.py", line 443, in _run_prompt_group_worker / [36m(AsyncTrajectoryCollector pid=377005, ip=10.66.3.130)[0m     nemo_gym_rollout_result = run_async_nemo_gym_rollout( / [36m(AsyncTrajectoryCollector pid=377005, ip=10.66.3.130)[0m         policy_generation=self.policy_generation, / [36m(AsyncTrajectoryCollec |
| 2123407 | pard | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
| 2123638 | baseline | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
