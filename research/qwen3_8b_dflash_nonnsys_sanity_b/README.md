# Qwen3-8B DFlash K7 non-Nsys sanity B

This is a 50-step secondary sanity A/B, not a science result. It compares the
true-online DFlash drafter arm with the fixed public DFlash dense-control arm on
the exact optimized product base `79e80af96a13522e6049658663a8c40ab21e8314`.

Both arms use the same target and drafter revisions, seed 42, 1 node × 4 GPUs,
GBS 32, 8 prompts per step, 4 generations per prompt, sequence packing off,
sequence parallel off, DFlash K7, and piecewise CUDA graphs. Update probes and
Nsys are disabled. W&B uses `nvidia/sna-nemo-rl-online-drafter` with a fresh ID
per arm.

`manifest.yaml` records the arm-specific validator and checkpoint-control
differences retained from the already-green launchers. `submit.sh` runs both
SLURM `--test-only` checks before submitting either independent actual job.
