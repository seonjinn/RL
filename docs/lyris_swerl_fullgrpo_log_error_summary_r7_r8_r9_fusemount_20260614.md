# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r7_r8_r9_fusemount_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2121104 | baseline | 1 | COMPLETED | 18 | Traceback (most recent call last): / File "/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL-SWE_bench-20260613/nemo_rl/algorithms/grpo.py", line 3016, in async_grpo_train / train_results = policy.train( / train_data, / loss_fn, / timer=timer, |
| 2121199 | pard | 1 | RUNNING | 18 | [36m(MegatronPolicyWorker pid=3518817, ip=10.66.4.9)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=3518817, ip=10.66.4.9)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=2215931)[0m /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning. / [36m(MegatronPolicyWorker pid=2215931)[0m   |
| 2121201 | pard2 | 1 | RUNNING | 18 | [36m(MegatronPolicyWorker pid=3250050, ip=10.66.3.202)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=3250050, ip=10.66.3.202)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=397643, ip=10.66.3.196)[0m /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning. / [36m(MegatronPolicyWorker |
| 2121249 | baseline | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
