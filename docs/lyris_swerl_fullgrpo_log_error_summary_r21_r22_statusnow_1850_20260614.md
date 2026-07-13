# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r21_r22_statusnow_1850_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2122466 | pard | 1 | RUNNING | 18 | [36m(VllmAsyncGenerationWorker pid=222967, ip=10.66.3.172)[0m (EngineCore_DP0 pid=223561) [36m(RayWorkerWrapper pid=231743)[0m WARNING 06-14 18:45:36 [allreduce_rms_fusion.py:779] Failed to initialize FlashInfer All Reduce workspace: 'utf-8' codec can't decode byte 0xf2 in position 1: invalid continuation byte. AllReduce fusion pass will be disabled. / [36m(VllmAsyncGenerationWorker pid=222967, ip=10.66.3.172)[0m (EngineCore_DP0 pid=223561) [36m(RayWorkerWrapper pid=231741)[0m INFO 06-14 18:45:22 [gpu_model_runner.py:4305] Loading drafter model...[32m [repeated 3x across cluster][0m  |
| 2123210 | baseline | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
| 2123212 | pard2 | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
