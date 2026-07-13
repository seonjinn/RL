# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_active_r5_r6_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2121065 | baseline | 1 | RUNNING | 18 | [36m(MegatronPolicyWorker pid=3374520, ip=10.66.3.133)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=3374520, ip=10.66.3.133)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=3374520, ip=10.66.3.133)[0m /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning. / [36m(MegatronPolicyWorke |
| 2121066 | pard | 1 | RUNNING | 18 | [36m(MegatronPolicyWorker pid=1678562, ip=10.66.3.174)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=1678562, ip=10.66.3.174)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=2479049, ip=10.66.3.173)[0m /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning. / [36m(MegatronPolicyWorke |
| 2121067 | pard2 | 1 | CANCELLED by 2001147693 | 18 | [2026-06-14T11:23:15.694] error: *** STEP 2121067.24 ON lyris0074 CANCELLED AT 2026-06-14T11:23:15 DUE to SIGNAL Terminated *** |
| 2121079 | pard2 | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
