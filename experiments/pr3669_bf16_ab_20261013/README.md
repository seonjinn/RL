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

All six 20-step A/B jobs on ptyche completed with Slurm exit code 0 and
20 timing/KL records each. Means below exclude step 1. Time reduction is
`(base - PR) / base`; it is a measured difference, not a statistical
significance claim.

| Model | 20-step base / PR jobs | W&B base / PR | E2E base / PR | Refit bubble base / PR | Mean gen KL base / PR |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3-30B-A3B | 3034206 / 3034209 | [base](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/kpo9ywdy) / [PR](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/fsyukt8g) | 207.86 / 205.58 s (1.1% lower) | 4.18 / 3.68 s (12.1% lower) | 0.00189 / 0.00188 |
| Qwen3.5-35B-A3B | 3034186 / 3034188 | [base](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/4nbqjoqw) / [PR](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/hmh07ati) | 213.20 / 210.66 s (1.2% lower) | 3.62 / 3.29 s (9.0% lower) | 0.00103 / 0.00099 |
| Nemotron3 Super | 3034251 / 3034253 | [base](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/om8eaitk) / [PR](https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab/runs/vfay1o0b) | 46.89 / 45.60 s (2.7% lower) | 4.88 / 4.55 s (6.8% lower) | 0.00294 / 0.00309 |

Qwen3-235B-A22B 20-step A/B jobs 3341499 / 3341501 are queued on Lyris
using the repository's 32-node Async-1off performance recipe. The queued
2-step jobs were canceled so they do not delay the requested 20-step run.
The original 16-node Sync recipe is colocated and cannot use `nccl_reshard`.

W&B project: https://wandb.ai/nvidia/nemo-rl-pr3669-bf16-ab

Logs and manifests:

- ptyche: `/lustre/fsw/coreai_dlalgo_llm/users/sna/results/pr3669-bf16-ab-20261013-ptyche/<model>-<arm>-<steps>step-<pair>/`
- Lyris: `/lustre/fsw/coreai_dlalgo_llm/users/sna/results/pr3669-bf16-ab-20261013/<model>-<arm>-<steps>step-<pair>/`

Use the `<jobid>-logs/ray-driver.log` file under each run directory for
per-step timing and KL values. Exclude the first step and failed setup jobs
from any aggregate performance comparison.
