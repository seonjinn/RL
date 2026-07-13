# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/oci_hsg_swerl_fullgrpo_pard2_openairender_n3post_r6_log_fetch_manifest_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 3299946 | pard2 | 10 | FAILED | 18 | [36m(MegatronPolicyWorker pid=2164464, ip=10.109.17.201)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=2164464, ip=10.109.17.201)[0m make: python3-config: No such file or directory / [36m(MegatronPolicyWorker pid=273917, ip=10.109.26.66)[0m /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning. / [36m(MegatronPolicyW |
| 3300101 | pard2 | 10 | RUNNING | 18 | 2026-06-14 02:13:24,128	INFO worker.py:2004 -- Connected to Ray cluster. View the dashboard at [1m[32mhttp://10.109.21.215:8265 [39m[22m |
