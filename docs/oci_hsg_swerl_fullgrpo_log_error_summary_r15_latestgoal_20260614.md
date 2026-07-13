# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/oci_hsg_swerl_fullgrpo_log_fetch_manifest_r15_latestgoal_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 3304067 | pard2 | 1 | FAILED | 18 | [36m(VllmAsyncGenerationWorker pid=2530271, ip=10.109.16.158)[0m Exception raised in creation task: The actor died because of an error raised in its creation task, [36mray::vllm_policy-grp7-0:VllmAsyncGenerationWorker.__init__()[39m (pid=2530271, ip=10.109.16.158, actor_id=6b2a45134edc72fe142cbfae01000000, repr=VllmAsyncGenerationWorker) / [36m(VllmAsyncGenerationWorker pid=2530271, ip=10.109.16.158)[0m   File "/root/.local/share/uv/python/cpython-3.13.13-linux-aarch64-gnu/lib/python3.13/concurrent/futures/_base.py", line 456, in result / [36m(VllmAsyncGenerationWorker pid=2530271, ip=1 |
