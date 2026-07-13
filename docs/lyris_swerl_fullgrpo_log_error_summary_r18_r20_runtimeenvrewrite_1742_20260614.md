# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r18_r20_runtimeenvrewrite_1742_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2121604 | baseline | 1 | FAILED | 18 | [36m(VllmAsyncGenerationWorker pid=4144087, ip=10.66.3.207)[0m (EngineCore_DP0 pid=4144629) [36m(RayWorkerWrapper pid=4152810)[0m WARNING 06-14 17:23:34 [allreduce_rms_fusion.py:779] Failed to initialize FlashInfer All Reduce workspace: 'utf-8' codec can't decode byte 0xb5 in position 0: invalid start byte. AllReduce fusion pass will be disabled. / [36m(VllmAsyncGenerationWorker pid=4144087, ip=10.66.3.207)[0m (EngineCore_DP0 pid=4144629) [36m(RayWorkerWrapper pid=4152812)[0m INFO 06-14 17:23:19 [parallel_state.py:1715] rank 2 in world size 4 is assigned as DP rank 0, PP rank 0, PCP ra |
| 2121605 | pard | 1 | RUNNING | 18 | [36m(VllmAsyncGenerationWorker pid=1496625, ip=10.66.4.109)[0m (EngineCore_DP0 pid=1497330) [36m(RayWorkerWrapper pid=1505510)[0m WARNING 06-14 17:24:04 [allreduce_rms_fusion.py:779] Failed to initialize FlashInfer All Reduce workspace: 'utf-8' codec can't decode byte 0xa5 in position 4: invalid start byte. AllReduce fusion pass will be disabled. / [36m(VllmAsyncGenerationWorker pid=1496625, ip=10.66.4.109)[0m (EngineCore_DP0 pid=1497330) [36m(RayWorkerWrapper pid=1505386)[0m INFO 06-14 17:23:49 [gpu_model_runner.py:4305] Loading drafter model...[32m [repeated 3x across cluster][0m / |
| 2122398 | baseline | 1 | PENDING | 0 | no local log/error excerpt available |
| 2122399 | pard2 | 1 | PENDING | 0 | no local log/error excerpt available |
