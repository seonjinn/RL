# PR 3669 BF16 TRTLLM refit A/B

This experiment compares the exact PR base `432050b6106b00eab37a8adc149e30ce0380d1c5`
and PR head `cb9e2ae1d26523ac1131b16c5f73009c3db52ee3` on GB200 with
vLLM 0.29. The only source difference is PR 3669. Both arms use the same
model-specific Async-1off recipe, BF16 training and rollout,
`flashinfer_trtllm`, and `policy.generation.refit_transport=nccl_reshard`.
The launcher writes a source/image/config/command manifest into each run directory.

Run with `MODEL=qwen30|qwen35|super3|qwen235 ARM=base|pr STEPS=20
PAIR_ID=<pair> bash launch_ptyche.sh` on ptyche. On Lyris, use
`launch_lyris.sh` with the exact base/PR checkouts specified there.
Each launcher runs `sbatch --test-only` before submission.

| Model | 2-step base / PR jobs | W&B base / PR | Step 2 E2E base / PR | Step 2 refit bubble base / PR | Step 2 gen KL base / PR |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3-30B-A3B | 3034002 / 3034004 | [base](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/3ud2d4m8) / [PR](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/jzmddlqf) | 195.20 / 195.21 s | 4.17 / 3.72 s | 0.0019 / 0.0019 |
| Qwen3.5-35B-A3B | 3033961 / 3033968 | [base](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/bnfkqu3z) / [PR](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/tq9qws2i) | 240.99 / 237.48 s | 3.67 / 3.22 s | 0.0009 / 0.0010 |
| Nemotron3 Super | 3034006 / 3034008 | [base](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/zvsdeko0) / [PR](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/am19ws3y) | 19.15 / 20.62 s | 4.69 / 4.41 s | 0.0026 / 0.0026 |

All six 2-step smoke jobs finished with Slurm exit code 0 and completed a
generation refit. These are single steady-step observations, not performance
claims. In this nightly image, `nccl.m2n.reshard` is unavailable, so both
arms fall back to `xferdtensor_python`; the comparison is matched but does
not measure the optimized NCCL M-to-N transport.

20-step A/B jobs submitted on ptyche:

| Model | Base | PR |
| --- | ---: | ---: |
| Qwen3-30B-A3B | 3034206 | 3034209 |
| Qwen3.5-35B-A3B | 3034186 | 3034188 |
| Nemotron3 Super | 3034251 | 3034253 |

Qwen3-235B-A22B 2-step A/B jobs 3340552 / 3340554 are queued on Lyris
using the repository's 32-node Async-1off performance recipe. The original
16-node Sync recipe is colocated and cannot use `nccl_reshard`; the jobs
using that invalid pairing were canceled before training.

W&B project: https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab

Logs and manifests:

- ptyche: `/lustre/fsw/coreai_dlalgo_llm/users/sna/results/pr3669-bf16-ab-20261013-ptyche/<model>-<arm>-<steps>step-<pair>/`
- Lyris: `/lustre/fsw/coreai_dlalgo_llm/users/sna/results/pr3669-bf16-ab-20261013/<model>-<arm>-<steps>step-<pair>/`

Use the `<jobid>-logs/ray-driver.log` file under each run directory for
per-step timing and KL values. Exclude the first step and failed setup jobs
from any aggregate performance comparison.
