# GPU gate receipt

## Verification before submission

- Focused refit suite: Ptyche job `2845677`, 277 passed, 9 skipped, 70
  warnings in 1,081.80 seconds. The skips are pre-existing hardware/test
  constraints documented by their test markers.
- Async deep-refit lifecycle: Ptyche job `2845786`, 6 passed, 117 deselected,
  17 warnings in 23.08 seconds.
- Local static validation: Ruff check, Ruff format check, Python compileall,
  and `git diff --check` passed for all touched implementation and test files.

## Original three-step GPU gate

| Arm | Job | State at receipt update | Source SHA |
|---|---:|---|---|
| Baseline, legacy level-1 | `2845797` | Completed | `26366cd97657c09181a988de33b736bcf311e638` |
| DFlash K7 B8, deep refit | `2845799` | Completed; invalid drafter restore | `26366cd97657c09181a988de33b736bcf311e638` |

Artifact directories:

- `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q235-specdec-deep-refit-20260917/Qwen3-235B-Baseline-Legacy-3step-20260917T235620Z`
- `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q235-specdec-deep-refit-20260917/Qwen3-235B-DFlashK7-B8-DeepRefit-3step-20260917T235620Z`

The original deep-refit arm exposed the regression: average speculative-token
acceptance fell to approximately 1%, generation throughput fell to 196.2
tokens/s/GPU, and E2E throughput fell to 130.0 tokens/s/GPU. The matched
baseline reached 302.4 generation tokens/s/GPU and 169.2 E2E tokens/s/GPU.

## Patched matched three-step GPU gate

Source SHA: `dcb6a52aed1297a805ba7f1efdba8869931fa250`

| Arm | Job | W&B | State |
|---|---:|---|---|
| Baseline, legacy level-1 | `2849421` | [`7rk3bc7m`](https://wandb.ai/nvidia/sna-specdec/runs/7rk3bc7m) | Completed |
| DFlash K7 B8, patched deep refit | `2849419` | [`4ob5z8nk`](https://wandb.ai/nvidia/sna-specdec/runs/4ob5z8nk) | Completed |

Both runs use synchronous GRPO, 64 GPUs, generation TP=8, generation GBS=512,
maximum sequence length 8192, BF16 generation, and steps 1-3 inclusive.

| Metric, mean over steps 1-3 | Baseline | Patched DFlash K7 | Baseline-relative |
|---|---:|---:|---:|
| Mean tokens/sample | 5,348.95 | 5,332.42 | 0.997x |
| Generation throughput/GPU | 302.65 | 462.81 | 1.529x (+52.9%) |
| Generation time | 142.67 s | 93.37 s | 1.528x; 34.6% reduction |
| E2E throughput/GPU | 167.18 | 206.71 | 1.236x (+23.6%) |
| E2E step time | 268.39 s | 223.83 s | 1.199x; 16.6% reduction |
| Policy training time | 69.58 s | 70.30 s | 0.990x |
| Policy/reference logprob time | 41.57 s | 42.91 s | 0.969x |
| Refit time | 11.98 s | 14.87 s | +2.90 s |

Patched DFlash acceptance remained stable across repeated wake/refit cycles:

| Step | Acceptance rate | Mean accepted length |
|---:|---:|---:|
| 1 | 26.78% | 2.874 |
| 2 | 28.19% | 2.973 |
| 3 | 27.77% | 2.943 |
| Mean | 27.58% | 2.930 |

Quality-side three-step means remained matched within normal sampling variance:

| Metric | Baseline | Patched DFlash K7 |
|---|---:|---:|
| Reward | 0.6940 | 0.7012 |
| Approximate entropy | 0.5141 | 0.5127 |
| Generation KL error | 0.006119 | 0.006129 |
| Policy KL error | 0.012949 | 0.008514 |

The patch passes the GPU acceptance and performance gate. The next validation
stage is a longer run to confirm that the restored runtime state remains stable
across many repeated refit cycles.
