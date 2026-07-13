# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r18_r19_latestgoal_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2121604 | baseline | 1 | RUNNING | 18 | [36m(_env_builder pid=3387074, ip=10.66.3.195)[0m * torch-c-dlpack-ext (torch_c_dlpack_ext-0.1.5-cp313-cp313-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl)[32m [repeated 2x across cluster][0m |
| 2121605 | pard | 1 | RUNNING | 18 | [36m(_env_builder pid=1107210, ip=10.66.4.120)[0m Finished creating venv /opt/ray_venvs_swerl_r18_openhandspath_trainbs32/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker[32m [repeated 12x across cluster][0m |
| 2121609 | pard2 | 1 | RUNNING | 18 | [36m(_env_builder pid=463230, ip=10.66.4.101)[0m  + nemo-rl==0.6.0+4cc7d70c8 (from file:///lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL-SWE_bench-20260613) |
