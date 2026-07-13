# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r22_r23_statusnow_1943_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2123210 | baseline | 1 | RUNNING | 18 | [36m(VllmAsyncGenerationWorker pid=479066, ip=10.66.4.118)[0m (EngineCore_DP0 pid=479536) [36m(RayWorkerWrapper pid=487713)[0m WARNING 06-14 19:01:04 [allreduce_rms_fusion.py:779] Failed to initialize FlashInfer All Reduce workspace: 'utf-8' codec can't decode byte 0xa5 in position 4: invalid start byte. AllReduce fusion pass will be disabled. / [36m(VllmAsyncGenerationWorker pid=479066, ip=10.66.4.118)[0m (EngineCore_DP0 pid=479536) [36m(RayWorkerWrapper pid=487710)[0m INFO 06-14 19:00:49 [parallel_state.py:1715] rank 1 in world size 4 is assigned as DP rank 0, PP rank 0, PCP rank 0,  |
| 2123212 | pard2 | 1 | RUNNING | 18 | [36m(MegatronPolicyWorker pid=3519518, ip=10.66.3.139)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=3519518, ip=10.66.3.139)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=3304760, ip=10.66.3.140)[0m /opt/ray_venvs_swerl_r22_pard2_wandb_venvsync_trainbs32/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute  |
| 2123407 | pard | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
