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

Last live check: 22:09:41 UTC, both jobs RUNNING for4m14s; immutable source
checkout and input staging completed and dependency installation was progressing.
The following SSH checks failed at DNS resolution of the Park hostname. The
five-minute running-monitor requirement was therefore not fully verified; do
not infer that the cluster jobs stopped, passed, or failed from that local
connectivity error. Actual GRPO training metrics are not yet confirmed.

## Resolved status and launcher recovery

At 23:47 UTC, scheduler and logs confirmed both canaries FAILED before GRPO:
7152082 after41m52s, 7152083 after41m34s. Actor preflights passed, then the
hardcoded `/opt/nemo_rl_venv/bin/python` was not executable (ENOENT). No training,
refit, reward, or throughput result was produced. The image metadata identifies
the same pinned August18 nightly, not a newly selected container. Whether its
venv was absent or its interpreter link was masked remains unverified.

Recovery uses the already-created, frozen-lock Megatron environment as the
driver interpreter through `uv run --no-project --no-sync --python`. A real
`examples.run_grpo` import must pass before constructing the remaining runtime.
The renderer writes files synchronously so subprocess failure cannot be hidden
by `mapfile` process substitution. Missing/dangling interpreter tests and both
method rendering tests pass locally; this does not yet prove GPU execution.
The image, model, serving K, attention window and study workload are unchanged.

Recovery submissions (September15 UTC / September14 Pacific): DFlash7154339,
DSpark7154340. Source8f3875f0dab2e708aafe1240cc452cf84f9d809b; bundle SHA256
f8a441658f5e44d57ed8188acfb763192eefc062b0bd43fafa0fcd32775eebde.
Both passed test-only and were independently submitted to nemotron_sw_post/batch.
Durable root is the same experiment prefix with `driver-8f3875f0d/{dflash,dspark}`.
The first scheduler check showed both PENDING, not training success. W&B IDs
remain planned until driver initialization; no200step production job submitted.

Subsequent live check: both jobs RUNNING beyond6minutes, with pinned-source
checkout and dependency builds progressing. The five-minute startup observation
is satisfied, not the full runtime gate. New logs show the image Python is a
symlink to /root/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu/bin/python3.13;
the selected driver no longer relies on that image-internal link. Its underlying
target/mount failure still requires separate diagnosis; no new GRPO step yet.

## Root cause confirmed by same-image A/B probe

The read-only probe in allocation7154339 reproduced ENOENT with default home
mounting: the symlink existed, but its /root interpreter target did not. With
only `--no-container-mount-home` added, the exact target was present and ran
Python3.13.14. Host-home mounting was masking the image's interpreter, not a
missing interpreter in the original nightly. The failure is unrelated to drafter
size, GPU OOM, or model weights. Probe step7154339.1 exited127 as expected.

Future launcher invocations disable home auto-mount and keep UV-managed Python
installs in node-local scratch. Running canaries7154339/7154340 retain source
8f3875f0d and its independent validated-driver fix; they are not retrospectively
relabelled as using the later mount patch. Preserve their ongoing environment builds.
