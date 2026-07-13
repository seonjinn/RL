# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_latest_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2120534 | baseline | 1 | FAILED | 18 | × Failed to download and build `transformer-engine @ / │ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26` / ├─▶ Failed to acquire lock on the distribution cache / ├─▶ Could not acquire lock / ╰─▶ Timeout (900s) when waiting for lock on / `/lustre/fsw/coreai_dlalgo_llm/users/sna/.cache/qwen235b_swerl_staged_smoke/uv_cache/sdists-v9/git/a20fac049004db0a/366798ef8a0a00d8` |
| 2120535 | pard | 1 | FAILED | 18 | × Failed to download and build `transformer-engine @ / │ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26` / ├─▶ Failed to acquire lock on the distribution cache / ├─▶ Could not acquire lock / ╰─▶ Timeout (900s) when waiting for lock on / `/lustre/fsw/coreai_dlalgo_llm/users/sna/.cache/qwen235b_swerl_staged_smoke/uv_cache/sdists-v9/git/a20fac049004db0a/366798ef8a0a00d8` |
| 2120536 | pard2 | 1 | FAILED | 18 | [36m(pid=1760993)[0m Exception ignored in atexit callback <function shutdown at 0xfffff4322160>: / [36m(pid=1760993)[0m Traceback (most recent call last): / [36m(pid=1760993)[0m   File "/opt/nemo_rl_venv/lib/python3.13/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper / [36m(pid=1760993)[0m     return func(*args, **kwargs) / [36m(pid=1760993)[0m   File "/opt/nemo_rl_venv/lib/python3.13/site-packages/ray/_private/worker.py", line 1144, in wrapper / [36m(pid=1760993)[0m     return func(*args, **kwargs) |
| 2120957 | baseline | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
| 2120958 | pard | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
