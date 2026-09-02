# Qwen3-30B-A3B 20-step SpecDec matrix

This harness compares the prior Qwen3-30B-A3B Math GRPO arms with matched
expanded CUDA Graph coverage. Expanded arms reuse the reference arm's config,
target, drafter checkpoint, K, training cadence, source SHA, container, and
scheduler request. Only the CUDA Graph capture list and unique W&B/run name
differ.

## Matched A/B arms

| Reference | Expanded arm | Method | K | Drafter cadence |
| --- | --- | --- | ---: | --- |
| `baseline` | `baseline-cg2048` | target only | 0 | none |
| `eagle3-k3` | `eagle3-k3-cg2048` | Eagle-3 | 3 | static |
| `dflash-k3` | `dflash-k3-cg2048` | DFlash | 3 | always online |
| `dflash-k5` | `dflash-k5-cg2048` | DFlash | 5 | always online |
| `dflash-k7` | `dflash-k7-cg2048` | DFlash | 7 | always online |
| `dspark-k3` | `dspark-k3-cg2048` | DSpark | 3 | always online |
| `dspark-k5` | `dspark-k5-cg2048` | DSpark | 5 | always online |
| `dspark-k7` | `dspark-k7-cg2048` | DSpark | 7 | always online |

The historical `dflash` and `dspark` aliases are intentionally not duplicated:
the explicit K5 arms above are the canonical checkpoint cohort for this A/B.

## Capture coverage

The expanded list preserves the union of the existing small K-specific buckets
and the default small-batch ladder through 512, then adds anchors through 2048.
It includes the exact verification shapes for 128 concurrent requests:

- K3: `128 * (3 + 1) = 512`
- K5: `128 * (5 + 1) = 768`
- K7: `128 * (7 + 1) = 1024`

DFlash K5 also captures 2046, the MRV1 floor-aligned value below the 2048 cap
for `K + 1 = 6`. DFlash K3 and K7 use the aligned 2048 cap. The harness does
not add 4096 because this experiment is intended to isolate useful coverage
without doubling the requested graph ceiling and its memory/startup exposure.

The runtime workload remains unchanged, including its existing
`max_num_seqs=8` launcher override. The higher-concurrency anchors are an A/B
of graph coverage only, not a request-batch-size change.

## Local contract checks

```bash
python3 -m pytest -q \
  experiments/qwen3_30ba3b_dflash_dspark_20step_20260822/tests/test_contract.py
bash -n \
  experiments/qwen3_30ba3b_dflash_dspark_20step_20260822/submit_qwen3_30ba3b_20step.sh
```

Render an expanded arm without submitting:

```bash
Q30_20STEP_RENDER_ROOT="$(mktemp -d)" \
bash experiments/qwen3_30ba3b_dflash_dspark_20step_20260822/submit_qwen3_30ba3b_20step.sh \
  --render-sbatch dflash-k5-cg2048
```

Actual submission still requires the existing per-arm preflight and
`sbatch --test-only` receipt. This change does not submit jobs.

If the pinned product worktree contains generated, untracked files, set
`Q30_20STEP_SOURCE_ROOT` to a clean equivalent worktree at the same immutable
source SHA. The harness rewrites only the copied config's `defaults` path,
checks the replacement worktree and all recursive submodules are clean, and
binds the selected source root into the test-only receipt. The checked-in
configs and reference lineage remain unchanged.
