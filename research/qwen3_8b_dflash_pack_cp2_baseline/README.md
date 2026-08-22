# Qwen3-8B packed TP2×CP2 DFlash K5 baseline

This cache-hash-bound baseline compares a fixed public DFlash K5 drafter with
an always-online DFlash K5 drafter for 30 GRPO steps. It does not use frozen
JSONL data and makes no adaptive or fixed-interval cadence claim.

Three matched replicates each run both arms sequentially on the same exclusive
four-GPU GB200 node. First-arm order is fixed, online, fixed. The source is the
exact validated PR11 commit `443e7243ae2a235b6dcd8f4918fea86e693630a9`;
the only descendants are this experiment harness. The packed training recipe
is the PR11 E2E topology: TP2, CP2, target sequence parallelism enabled,
sequence packing enabled, and divisibility 16. R5 parity keeps DAPOMath17K,
shuffle enabled, GRPO seed 42, GBS32, eight prompts and four generations.

Before either arm, the job hashes every file in the exact DAPOMath17K cache,
validates the exact source Parquet blob, and writes `dataset-manifest.json`.
It re-hashes before each arm and fails if either manifest differs. Container,
source, snapshot, config, W&B ID, job-ID, and dataset parity all fail closed.

Run `submit_matrix.sh --test-only` once for all three forecasts, then run it
once without arguments for exactly three actual jobs. The submitter monitors
all job IDs with one filtered `sacct` query per 60-second pass for five passes.
