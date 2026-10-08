# PR #3294 refit ablation on current main

This experiment compares refit options on the same merged source and nightly
image, using the existing synchronous MXFP8 rollout performance recipes on
Lyris. Qwen3-30B-A3B uses 4 nodes/16 GB200 GPUs, Qwen3-235B uses 16 nodes/64
GPUs, and Nemotron 3 Super uses 32 nodes/128 GPUs. Each arm runs 20 steps.
Report arithmetic means over steps 3-20; exclude startup and report the first
two steps separately. Checkpoints are disabled. IPC staging is fixed at 4 GiB
for the Qwen recipes and 0.5 GiB for Super; the smaller Super buffer avoids
replacing its recipe's low-memory ratio with a 4 GiB reservation.

| Arm | Trainer-side prequant | Persistent IPC | Slim offload | Loader-route cache |
| --- | --- | --- | --- | --- |
| `control` | off | off | off | off |
| `prequant` | on | off | off | off |
| `persistent` | recipe setting | on | off | off |
| `slim` | recipe setting | off | on | off |
| `cache` | recipe setting | off | off | on |
| `full` | on | on | on | on |

The Qwen recipes enable prequant by default; the Super recipe does not. Compare
each single-feature Qwen arm to `prequant`, and each Super arm to `control`.
Compare `full` only after checking the individual effects and run-to-run
variability. Capture E2E step time, tokens/s/GPU, generation,
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
