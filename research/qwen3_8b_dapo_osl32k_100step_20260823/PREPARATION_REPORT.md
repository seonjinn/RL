# Preparation Report

Status: local harness implementation complete and pinned to product commit `a28df91a94b623f5108a2992ccac887cc8cbdaab`, which composes the cadence weight-synchronizer fix with checkpointed segment stop. The Linux composition gate and cluster submission remain pending.

## Fixed experiment design

| Field | Value |
|---|---|
| Arms | baseline K0, DFlash K5 always, DSpark K5 always |
| Dataset | DAPO-Math-17k revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`, first 64 rows |
| Target | Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Context | input 2,048; output 32,768; model length 40,960 |
| Training | 100 global steps, 4 maximum epochs, seed 42, shuffle disabled |
| Segments | absolute endpoints 25, 50, 75, 100 |
| Checkpoints | every 25 steps with optimizer and dataloader state |
| Topology | one node, four GPUs, TP2/PP1/CP1/DP2 |
| W&B | one stable run ID per arm; `must` resume for segments 2–4 |
| Dependencies | `afterok` |

## Proven locally

- The three configs preserve one 100-step optimizer horizon while varying only `grpo.segment_stop_step` per job.
- Segment receipts are keyed by arm, endpoint, harness SHA, product SHA, and config SHA.
- Resume rejects checkpoint-tree, ledger, dataloader, and optimizer drift.
- Intermediate segments reject terminal cadence artifacts; step 100 requires both terminal runtime artifacts.
- CUDA Graph, segment boundary steps, draft wake/refit, output-length, and fatal-log gates are fail closed.
- Test-only and actual chain submission records are identity-bound and exactly once.

## Remaining gates

1. Compose all three configs in the pinned Linux product environment.
2. Run scheduler `--test-only` for all three arms and preserve receipts.
3. Submit the three four-job chains and monitor the first five minutes.
4. Validate all receipts and publish baseline-relative generation and end-to-end metrics for steps 3–100.

No 100-step job has been submitted from this harness.
