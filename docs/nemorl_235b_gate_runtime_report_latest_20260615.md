# NeMo-RL 235B Gate Runtime Report

Inspect time: `2026-06-15 09:11:33 PDT`
Summary: CANCELLED=2, FAILED=4, RUNNING=1; stdout_present=0; runtime_log_present=5
Action: Inspect present stdout/runtime logs and parse step metrics before updating success claims.

| Job | State | Reason | Start | Nodes | Account | Priority | Runtime logs | Stdout | Scope |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 2129203 | FAILED | sacct_exit=1:0 | 2026-06-15T07:57:46 |  | coreai_dlalgo_llm |  | 19 |  | Lyris SWE-RL Raymatch/PARD Proof Gates |
| 2129271 | CANCELLED | sacct_exit=0:0 | None |  | coreai_dlalgo_llm |  | 0 |  | Lyris SWE-RL Raymatch/PARD Proof Gates |
| 2129272 | CANCELLED | sacct_exit=0:0 | None |  | coreai_dlalgo_llm |  | 0 |  | Lyris SWE-RL Raymatch/PARD Proof Gates |
| 3308774 | RUNNING | None | 2026-06-15T08:46:16 | 16 | nemotron_n3_post | 134988 | 19 |  | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates |
| 3315380 | FAILED | sacct_exit=1:0 | 2026-06-15T08:46:16 |  | nemotron_n3_post |  | 20 |  | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates |
| 3315381 | FAILED | sacct_exit=1:0 | 2026-06-15T08:47:05 |  | nemotron_n3_post |  | 20 |  | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates |
| 3315382 | FAILED | sacct_exit=1:0 | 2026-06-15T08:47:05 |  | nemotron_n3_post |  | 20 |  | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates |

## First Stdout Paths

| Job | Stdout |
| ---: | --- |
| 2129203 | `` |
| 2129271 | `` |
| 2129272 | `` |
| 3308774 | `` |
| 3315380 | `` |
| 3315381 | `` |
| 3315382 | `` |

## Last Snapshot Changes

Previous snapshot: `2026-06-15 09:10:06 PDT`
Latest snapshot: `2026-06-15 09:11:28 PDT`
Latest summary: CANCELLED=2, FAILED=4, RUNNING=1; runtime_log_present=5

| Job | Field | Previous | Latest |
| --- | --- | --- | --- |
| 3308774 | `priority` | `134987` | `134988` |
| 3315382 | `state` | `COMPLETING` | `FAILED` |
| 3315382 | `reason` | `NonZeroExitCode` | `sacct_exit=1:0` |
| 3315382 | `priority` | `134935` | `` |
