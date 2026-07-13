# Qwen3-235B No-Stop Full-GRPO PARD Summary

Last remote refresh: 2026-06-12 05:46:26 PDT. Rows remain `missing_log` until the remote `ray-driver.log` exists.

Scheduler snapshot:

| Label | Job | State | Reason | Dependency | Planned start | Ray driver log |
| --- | ---: | --- | --- | --- | --- | --- |
| baseline | 3186342 |  |  |  |  | exists |
| local_cat_tpp_k5 | 3186343 |  |  |  |  | exists |
| public_pard_k5 | 3186344 |  |  |  |  | missing |


| Label | Job | Status | Steps | Gen TPS speedup | Gen time speedup | E2E TPS speedup | E2E step speedup | Acceptance | Latest error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3186342 | parsed | 0 |  |  |  |  |  | ModuleNotFoundError: No module named 'rpds.rpds' |
| local_pard2_cat_tpp_k5 | 3186343 | parsed | 0 |  |  |  |  |  | ModuleNotFoundError: No module named 'rpds.rpds' |
| public_pard_k5 | 3186344 | missing_log |  |  |  |  |  |  |  |
