# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_active_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2120957 | baseline | 1 | RUNNING | 18 | [36m(pid=2843979)[0m Exception ignored in atexit callback <function shutdown at 0xfffff43220c0>: / [36m(pid=2843979)[0m Traceback (most recent call last): / [36m(pid=2843979)[0m   File "/opt/nemo_rl_venv/lib/python3.13/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper / [36m(pid=2843979)[0m     return func(*args, **kwargs) / [36m(pid=2843979)[0m   File "/opt/nemo_rl_venv/lib/python3.13/site-packages/ray/_private/worker.py", line 1144, in wrapper / [36m(pid=2843979)[0m     return func(*args, **kwargs) |
| 2120958 | pard | 1 | RUNNING | 18 | [36m(pid=3759123)[0m Exception ignored in atexit callback <function shutdown at 0xfffff43220c0>: / [36m(pid=3759123)[0m Traceback (most recent call last): / [36m(pid=3759123)[0m   File "/opt/nemo_rl_venv/lib/python3.13/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper / [36m(pid=3759123)[0m     return func(*args, **kwargs) / [36m(pid=3759123)[0m   File "/opt/nemo_rl_venv/lib/python3.13/site-packages/ray/_private/worker.py", line 1144, in wrapper / [36m(pid=3759123)[0m     return func(*args, **kwargs) |
