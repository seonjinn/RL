# Qwen3-8B DSpark CP1 1000-Step Resume Plan

**Goal:** Resume the fixed/online K5/K7 CP1 matrix from each arm's validated step-400 checkpoint and run fail-closed to step 1000 without changing NeMo RL product code.

**Architecture:** Add one harness-only OCI submission entrypoint. It binds each arm to its original source SHA, Lustre checkpoint root, and W&B run ID; validates or creates the immutable resume manifest; runs every `sbatch --test-only` check before any actual submission; then submits four independent 4-GPU jobs. Existing arm-specific runners retain topology, runtime, checkpoint, and post-run validation contracts.

**Measured budget:** Steps 350-400 averaged 25.9-27.5 seconds, with p95 28.1-30.0 seconds. Six hundred remaining steps require approximately 4.3-5.0 hours. A 7-hour allocation with a 6-hour-30-minute checkpoint deadline leaves at least 90 minutes above the observed mean compute duration for environment setup, validation, checkpoint I/O, and variance.

## Tasks

1. Add RED contract tests for exact arm identities, step-400 resume-only behavior, a 1000-step milestone, 7-hour walltime, 6.5-hour deadline, all-test-only-before-submit ordering, and immutable W&B resume.
2. Implement the single matrix submission entrypoint using the existing online/fixed runners and resume contracts.
3. Run harness/config tests and shell syntax checks locally.
4. Commit with DCO, push the isolated branch, and create a clean remote harness checkout.
5. Revalidate FairShare and all four Lustre checkpoints, run all scheduling probes, submit all four jobs, and monitor for at least five minutes.
6. After completion, verify step-1000 checkpoints and W&B continuity, then report exact job/source/config/container provenance.
