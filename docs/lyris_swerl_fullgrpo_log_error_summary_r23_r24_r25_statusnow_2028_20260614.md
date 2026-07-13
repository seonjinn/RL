# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r23_r24_r25_statusnow_2028_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2123407 | pard | 1 | RUNNING | 2 | [36m(VllmAsyncGenerationWorker pid=1774805, ip=10.66.3.184)[0m (EngineCore_DP0 pid=1775131) [36m(RayWorkerWrapper pid=1783304)[0m WARNING 06-14 20:18:10 [allreduce_rms_fusion.py:779] Failed to initialize FlashInfer All Reduce workspace: 'utf-8' codec can't decode byte 0xf2 in position 1: invalid continuation byte. AllReduce fusion pass will be disabled. / [36m(VllmAsyncGenerationWorker pid=1774805, ip=10.66.3.184)[0m (EngineCore_DP0 pid=1775131) [36m(RayWorkerWrapper pid=1783305)[0m INFO 06-14 20:17:56 [gpu_model_runner.py:4305] Loading drafter model...[32m [repeated 3x across cluster |
| 2123638 | baseline | 1 | RUNNING | 2 | [36m(VllmAsyncGenerationWorker pid=550674, ip=10.66.4.113)[0m (EngineCore_DP0 pid=551218)   warnings.warn([32m [repeated 5x across cluster][0m |
| 2123875 | pard2 | 1 | RUNNING | 2 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
