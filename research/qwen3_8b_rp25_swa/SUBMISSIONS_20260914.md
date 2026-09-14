# New Draft online canary submissions

Actual submission receipts, September14 2026:

| Job | Method | Scope |
|---|---|---|
| 7152082 | DFlash B8, servingK5, window2048 | Two-step always-online, checkpoint at2 |
| 7152083 | DSpark B8, servingK5, window2048 | Two-step always-online, checkpoint at2 |

Account `nemotron_sw_post`, partition `batch`, one exclusive four-GPU GB200 node
per method, two-hour time limit, no inter-job dependencies. Both test-only
scheduling checks passed before submission; synthetic IDs7152078/7152079 are
not actual run IDs. Git push and remote ff-only pull preceded sbatch.

Source `f7dd041da05c8fc9fd85e4e7970b778e8e483b70`, bundle SHA256
`da831e8f14a0bb5e1183d5000ae4959163dd766d46c30cdfe775c20730e9f101`.
The previously pinned nightly container remains unchanged. Source and actor
environments are staged node-locally, including one target/drafter copy per node.
W&B credential availability was checked without printing its value.

Durable root:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/online-f7dd041da/`

Each `dflash/` or `dspark/` directory holds scheduler logs, resolved overrides,
source/asset identities, training log, cadence artifacts and checkpoints.
Planned W&B IDs (existence is not established until Python initializes W&B):

- `nvidia/sna-specdec/q8rp44-dflash-canary-7152082`
- `nvidia/sna-specdec/q8rp44-dspark-canary-7152083`

No200step production arm has been submitted. Before doing so, require real
train→refit→next-rollout evidence, generation-side attention semantics, and
checkpoint/optimizer/cadence resume verification. These canaries do not measure
production speedup and do not validate CP>1 or multi-node execution.
