# Qwen3-235B-A22B Base Math GRPO

This experiment directly inherits the official NeMo-RL GB200 performance recipe,
`grpo-qwen3-235b-32n4g.yaml`, whose target is `Qwen/Qwen3-235B-A22B`.
The benchmark overrides only the 20-step run length and DSpark-specific settings.

The DSpark checkpoint is the step-25391, block-size-8 drafter trained against
the matching Qwen3-235B-A22B Base target. Its architecture, weight size,
hidden/vocabulary shape, and block size are checked before submission.

## CUDA Graph A/B matrix

| Pair | Default-small arm | Expanded-through-2048 arm | Method | K |
| --- | --- | --- | --- | ---: |
| baseline | `baseline` | `baseline_cg2048` | target-only | 0 |
| DSpark K3 | `dspark_k3` | `dspark_k3_cg2048` | DSpark block 8 | 3 |
| DSpark K5 | `dspark_k5` | `dspark_k5_cg2048` | DSpark block 8 | 5 |
| DSpark K7 | `dspark_k7` | `dspark_k7_cg2048` | DSpark block 8 | 7 |

The four `_cg2048` arms are paired graph-coverage variants. They preserve the
source, container, workload, target and drafter checkpoints, method, K, and
runtime settings of their corresponding default-small arms. Run IDs contain the
full arm name, so the A and B artifacts and W&B runs cannot collide.

The expanded baseline uses the existing vLLM default small capture buckets
through 512 plus target-only anchors at 1024 and 2048. Each expanded DSpark arm
unions its existing K-aware small buckets with exact verifier anchors
`(K + 1) * C` for `C = 64, 128, 256, 512`, capped at 2048. Thus the current C64
verification shapes are explicitly covered at 256 for K3, 384 for K5, and 512
for K7. The composition validator rejects drift from these exact sorted lists.

The drafters are serving-only in this experiment: `policy.draft` is omitted,
so no online drafter update, drafter optimizer step, or drafter refit is charged
to these measurements. DSpark adds only its checkpoint, K, draft TP,
`FLASH_ATTN`, disabled FlashInfer autotuning, and K-aware CUDA Graph capture
sizes. Target, data, batching, sequence length, parallelism, validation,
checkpointing, and the global CUDA Graph mode remain owned by the official
performance recipe.

Only the baseline and DSpark K3/K5/K7 pairs are launcher-allowlisted. The stale
`dflash_k3.yaml` and `dflash_k5.yaml` files remain archival inputs and cannot be
rendered, tested, or submitted by this launcher.

The clean Q235 product checkout is pinned to
`f6f8605da02675af4361cfc9fd4d5f4d23279ff1`. It contains the vLLM
collective-RPC selection normalization required by async policy-to-generation
refit, while the stable Q30 product checkout remains unchanged.
The Slurm request uses `--segment=16`, matching the inherited official recipe's
`cluster.segment_size: 16`, so a 32-node allocation forms two complete NVLink
domains.
Megatron source, vLLM overlays, venvs, and caches are staged under node-local
`/raid/scratch`; durable configs, logs, and receipts remain on Lustre. DSpark
uses the source-verified vLLM #48167 runtime patch plus the group-causality
follow-up that is first validated by the Q30B canary.

The default measurement is 20 GRPO steps. `Q235_MAX_STEPS=1` or `3` remains
available for correctness canaries. The launcher accepts only 1, 3, or 20.

## Submission protocol

Run from the immutable remote harness checkout. A passing `sbatch --test-only`
receipt is mandatory before an actual submission of the same arm, source SHA,
config SHA, harness SHA, and step count.

```bash
bash experiments/qwen235b_step25391_math_grpo_20260826/submit_qwen235b_math_grpo.sh \
  --emit-manifest dspark_k3_cg2048

Q235_MAX_STEPS=20 \
bash experiments/qwen235b_step25391_math_grpo_20260826/submit_qwen235b_math_grpo.sh \
  --test-only dspark_k3_cg2048

Q235_MAX_STEPS=20 \
bash experiments/qwen235b_step25391_math_grpo_20260826/submit_qwen235b_math_grpo.sh \
  --submit dspark_k3_cg2048
```

W&B project: `sna-specdec`.
