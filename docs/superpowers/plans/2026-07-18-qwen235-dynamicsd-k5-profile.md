# Qwen3-235B DynamicSD K5 Profile Implementation Plan

**Goal:** Extend the matched Qwen3-235B DynamicSD profile and NeMo-RL launch contract from K0-K3 to K0-K5 without changing the target, drafter, sampling, topology, or CUDA Graph mode.

**Architecture:** Keep each fixed K in an independent two-node Lyris allocation, aggregate only a complete K0-K5 grid, and expose a separate Qwen3-235B DynamicSD K5 variant whose CUDA Graph capture list covers every profiled verification endpoint through 384 tokens. Preserve the fail-closed batch-size contract at 64 so unprofiled batch sizes cannot enter DynamicSD silently.

**Tech Stack:** Python 3.13, pytest, vLLM 0.25.1 MRv2, NeMo-RL, SLURM/Lyris GB200.

---

## Task 1: Lock the K0-K5 profile contract with failing tests

**Files:**
- Modify: `tests/experiments/test_vllm_0251_qwen235_dynamic.py`

1. Change the Qwen3-235B fixture and assertions to require K0-K5, six profile jobs, acceptance telemetry for five draft positions, and capture endpoints 320 and 384.
2. Add an assertion that omitting the K5 endpoint capture fails profile validation.
3. Run the focused test and confirm it fails against the current K0-K3 implementation.

## Task 2: Extend the profile launcher minimally

**Files:**
- Modify: `experiments/vllm_0251_drafter_matrix/profile_dynamic_sd.py`

1. Change Qwen3-235B `k_values` to K0-K5.
2. Extend `cudagraph_capture_sizes` through 320 and 384.
3. Run the focused profile tests and confirm they pass.

## Task 3: Add the matched Qwen3-235B DynamicSD K5 variant

**Files:**
- Modify: `tests/experiments/test_vllm_0251_qwen235_dynamic.py`
- Modify: `experiments/vllm_0251_drafter_matrix/matrix.py`

1. Add a failing test for a Qwen3-235B-only DynamicSD K5 variant with the exact Thinking drafter and capture list through 384.
2. Add the minimal variant definition and preserve the profiled batch-size endpoint of 64.
3. Verify that schedules ending above or below 64 still fail closed.

## Task 4: Verify, commit, and publish privately

**Files:**
- Verify all modified files.

1. Run focused pytest for Qwen3-235B DynamicSD, profile launcher, schedule validation, and matrix contracts.
2. Run Ruff formatting/checks on the changed Python files and the repository's configured type checker for those files when available.
3. Review the diff for accidental target, drafter, sampling, topology, or recipe changes.
4. Commit with signoff and push only to `github-seonjinn:seonjinn/RL.git`.

## Task 5: Submit and monitor the Lyris profile

**Files:**
- Remote immutable output root under `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/vllm0251_dynamic_profile/`.

1. Pull the private branch and run `git submodule update --init --recursive` on Lyris.
2. Run launcher `show` and SLURM `--test-only`; verify two nodes, `--segment=2`, no GRES, five-hour limit, pinned revisions, and capture endpoint 384.
3. Submit six independent K0-K5 jobs to `gb200` using account `coreai_dlalgo_llm`.
4. Monitor for at least five minutes and inspect any early failure logs.
5. Do not promote a DynamicSD final20 run until the profile is complete and the separate `prepare_for_generation` stall is diagnosed.
