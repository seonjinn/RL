# Qwen3-30B-A3B 20-step SpecDec matrix

This harness compares Qwen3-30B-A3B Math GRPO arms with K-specific, dense CUDA
Graph coverage in `FULL_AND_PIECEWISE` mode. The active profiles preserve the
target, drafter checkpoint, K, training cadence, source SHA, container, and
scheduler request while covering every decode shape up to the real runtime
limit.

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

The `*-cg2048` arms are retained only to reproduce the superseded coverage
experiment. Those runs showed host-memory OOMs and are not the recommended
profiles.

## Capture coverage

The active runtime fixes `max_num_seqs=8`, so the decode-complete upper bounds
are:

- Baseline: `8`
- K3: `8 * (3 + 1) = 32`
- K5: `8 * (5 + 1) = 48`
- K7: `8 * (7 + 1) = 64`

The active lists include `1, 2, 4, 8` and then every multiple of four through
the appropriate upper bound. This is denser than the original profile without
capturing unreachable decode shapes. The superseded 2048 profile combined
global PIECEWISE capture with 64 or 65 graph sizes; it raised startup and host
memory pressure without increasing decode coverage for this workload.

## Local contract checks

```bash
python3 -m pytest -q \
  experiments/qwen3_30ba3b_dflash_dspark_20step_20260822/tests/test_contract.py
bash -n \
  experiments/qwen3_30ba3b_dflash_dspark_20step_20260822/submit_qwen3_30ba3b_20step.sh
```

Render an active dense-capture arm without submitting:

```bash
Q30_20STEP_RENDER_ROOT="$(mktemp -d)" \
bash experiments/qwen3_30ba3b_dflash_dspark_20step_20260822/submit_qwen3_30ba3b_20step.sh \
  --render-sbatch dflash-k5
```

Actual submission still requires the existing per-arm preflight and
`sbatch --test-only` receipt. This change does not submit jobs.

If the pinned product worktree contains generated, untracked files, set
`Q30_20STEP_SOURCE_ROOT` to a clean equivalent worktree at the same immutable
source SHA. The harness rewrites only the copied config's `defaults` path,
checks the replacement worktree and all recursive submodules are clean, and
binds the selected source root into the test-only receipt. The checked-in
configs and reference lineage remain unchanged.
