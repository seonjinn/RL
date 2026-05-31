# Qwen3-235B-A22B-Thinking SWE Async GRPO — Full Run

Reproduction of Bin Hu's NeMo-RL Qwen3-235B SWE async GRPO training in sna's scratch.

## Goal

Establish the speed-of-light reference for Qwen3-235B-A22B-Thinking-2507 async GRPO SWE training on CW 16-node × 8 H100, measuring (1) per-step wall time, (2) phase breakdown, (3) checkpoint cost (cold vs warm), and (4) nsys profile of a representative step.

## Method

- Branch: super-v3 tip (HEAD bc4bd20cf)
- Config: `grpo_qwen3_235b_swe.yaml` (PPS=32, GPP=8, GBS=256, LR=1e-6, max_num_steps=1e6)
- Parallelism: TP=4, EP=8, CP=1, PP=8 (first=11, last=11), vLLM_TP=8
- Cluster: 16 actor nodes + 8 generation nodes (async non-colocated), CW partition `batch` 4h
- Submission: `bash run_grpo_qwen3_235b_swe.sh` (MAX_NUM_STEPS unset → full run cap by yaml/time)
- Checkpoint: save_period=5, keep_top_k=2, dp_reshardable optimizer state

## Key questions

1. Steady-state per-step time without checkpointing?
2. Cold vs warm checkpoint cost?
3. Where does the binding bottleneck live (logprobs/training/gen/comms)?
4. Async overlap effectiveness (vLLM idle time vs total)?
5. Headline numbers: tokens/sec/GPU, MFU, generation throughput

## Runs

- Job 11769694: smoke test (max_num_steps=1), 674.98s/step including 279s ckpt — see `../qwen3-235b-swe-smoke/`
- Job 11772327: full run (max_num_steps=1e6), 4h SBATCH, launched 2026-05-14 ~17:57 UTC

## Pipeline

1. Build: pre-download HF snapshot (max_workers=4), env.sh sourced for tokens
2. Launch: `run_grpo_qwen3_235b_swe.sh` → ray.sub → enroot container with Megatron-Bridge
3. Collect: WandB logger + Ray driver log at `${REPO_ROOT}/<JOBID>-logs/ray-driver.log`
4. Report: this directory after run terminates or hits time limit

See `report/README.md` for results once steps complete.
