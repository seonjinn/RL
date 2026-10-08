# PR #3294 refit ablation on current main

This experiment compares refit options on the same merged source, nightly image,
Qwen3-30B-A3B MXFP8 rollout recipe, and 4-node/16-GB200 Lyris allocation.
Each arm runs 20 synchronous GRPO steps. Report arithmetic means over steps
3-20; exclude startup and report the first two steps separately. Checkpoints
are disabled. All arms use a fixed 4 GiB IPC staging size.

| Arm | Trainer-side prequant | Persistent IPC | Slim offload | Loader-route cache |
| --- | --- | --- | --- | --- |
| `control` | off | off | off | off |
| `prequant` | on | off | off | off |
| `persistent` | on | on | off | off |
| `slim` | on | off | on | off |
| `cache` | on | off | off | on |
| `full` | on | on | on | on |

Compare `prequant` to `control`, then each single-feature arm to `prequant`.
Compare `full` to `prequant` only after checking the individual effects and
run-to-run variability. Capture E2E step time, tokens/s/GPU, generation,
policy training, total refit, transfer/update, and peak allocated GPU memory.
Check loss, reward, and KL signals across arms. Cache-on results require a
separate repeated-refit weight-parity check: matching training metrics alone
cannot rule out a partially replayed loader route.

The August 2026 vLLM 0.25.1 ablation is historical context, not a current
result. In that study, persistent IPC and slim offload were bundled, so their
individual effects were not measured. The loader cache's incremental E2E gain
was 0.14%, within run noise. Do not claim a benefit for either option from
that table alone.

Record the experiment commit, main parent, container path and SHA256, gitlink
SHAs, job IDs, terminal states, W&B links, and exact log paths with the results.
