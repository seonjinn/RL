# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r21_r22_statusnow_1905_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2122466 | pard | 1 | RUNNING | 18 | [36m(VllmAsyncGenerationWorker pid=222967, ip=10.66.3.172)[0m (EngineCore_DP0 pid=223561) [36m(RayWorkerWrapper pid=231743)[0m WARNING 06-14 18:45:36 [allreduce_rms_fusion.py:779] Failed to initialize FlashInfer All Reduce workspace: 'utf-8' codec can't decode byte 0xf2 in position 1: invalid continuation byte. AllReduce fusion pass will be disabled. / [36m(VllmAsyncGenerationWorker pid=222967, ip=10.66.3.172)[0m (EngineCore_DP0 pid=223561) [36m(RayWorkerWrapper pid=231741)[0m INFO 06-14 18:45:22 [gpu_model_runner.py:4305] Loading drafter model...[32m [repeated 3x across cluster][0m  |
| 2123210 | baseline | 1 | RUNNING | 18 | [36m(VllmAsyncGenerationWorker pid=479066, ip=10.66.4.118)[0m (EngineCore_DP0 pid=479536) [36m(RayWorkerWrapper pid=487713)[0m WARNING 06-14 19:01:04 [allreduce_rms_fusion.py:779] Failed to initialize FlashInfer All Reduce workspace: 'utf-8' codec can't decode byte 0xa5 in position 4: invalid start byte. AllReduce fusion pass will be disabled. / [36m(VllmAsyncGenerationWorker pid=479066, ip=10.66.4.118)[0m (EngineCore_DP0 pid=479536) [36m(RayWorkerWrapper pid=487710)[0m INFO 06-14 19:00:49 [parallel_state.py:1715] rank 1 in world size 4 is assigned as DP rank 0, PP rank 0, PCP rank 0,  |
| 2123212 | pard2 | 1 | RUNNING | 18 | Loading safetensors checkpoint shards: 100% Completed / 1/1 [00:00<00:00,  6.62it/s][32m [repeated 2x across cluster][0m |
