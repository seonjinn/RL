# DAPO concurrency submission receipts

Submitted 2026-09-12 07:33 UTC to OCI-HSG, account `coreai_dlalgo_llm`,
partition `batch`. Every job uses eight 4-GPU GB200 nodes and a four-hour
walltime. The runtime source is `d9614c235` (launcher implementation
`c478e6117`); no NeMo-RL runtime code or native performance recipe was changed.

## Job matrix

| max_num_seqs | Method | 1-step gate | 20-step measurement |
|---:|---|---:|---:|
| 16 | Baseline | 7098261 | 7098263 |
| 16 | DFlash K5 | 7098265 | 7098267 |
| 16 | DSpark K5 | 7098269 | 7098271 |
| 32 | Baseline | 7098273 | 7098275 |
| 32 | DFlash K5 | 7098277 | 7098279 |
| 32 | DSpark K5 | 7098281 | 7098283 |
| 64 | Baseline | 7098285 | 7098287 |
| 64 | DFlash K5 | 7098290 | 7098292 |
| 64 | DSpark K5 | 7098294 | 7098296 |
| 128 | Baseline | 7098298 | 7098301 |
| 128 | DFlash K5 | 7098303 | 7098305 |
| 128 | DSpark K5 | 7098307 | 7098309 |

Each measurement has `afterok:<its gate ID>` and
`--kill-on-invalid-dep=yes`. There is no dependency between different
configurations. A gate submission receipt is not proof that it passed.

Snapshot at **2026-09-12 07:33:51 UTC**: all 12 gates are PENDING (Priority);
all 12 measurements are PENDING (Dependency). No GPU execution, completed
training step, graph replay, or speedup has been validated for this cohort.
Running-job startup monitoring still needs to occur once allocations start.

## Reproducibility and evidence

- Local worktree: `/Users/sna/Nemo-RL_Qwen3_Roadmap/.worktrees/q30-dapo-concurrency-20260912`
- Remote worktree: `/home/sna/nemorl-q30-dapo-concurrency-20260912`
- Branch: `codex/q30-dapo-concurrency-20260912`
- Durable results root: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-dapo-concurrency-20260912`
- Per-run directories use `Qwen3-30BA3B-DAPO-{Baseline,DFlashK5,DSparkK5}-S{16,32,64,128}-{1,20}step-<UTC timestamp>`.
- Each directory contains `job.sbatch`, `test-only.txt`, and `submission.txt`.
  Runtime adds `source_sha.txt`, `submodules.txt`, and logs.
- W&B project: [sna-specdec](https://wandb.ai/nvidia/sna-specdec).
  Group: `q30-dapo-gbs2048-concurrency-20260912`.
  Actual run URLs must be collected after W&B initialization, not guessed.

Remote recursive submodules verified:
- Automodel: `1814c6c93a66b9d59d254960ef6a99a64249b671`
- Gym: `fd5e84d6b1c485c80e7ae61553bbd485611c03b4`
- Megatron Bridge: `5ed97996cc2b422904d18179375b6d7366915097`
- Megatron-LM: `1e7598cbfae888cdd3d741a351aae588d56f66c0`

Validation before submission:
- New launcher suite: 5 tests passed across all 12 render combinations.
- Parent DAPO suite: 6 tests passed.
- All 12 rendered 20-step sbatch scripts passed `bash -n`.
- All non-study workload overrides matched across the 12 rendered commands.
- Ruff and git whitespace checks passed.
- Every submitted job passed `sbatch --test-only`.
- Source was committed/pushed, then pulled on the cluster before submission.

## Interpretation guardrails

Policy GBS and rollout count remain 2048 (128 prompts × 16 generations).
With 32 colocated TP1 generation engines, an equally divided batch has 64
samples per engine. S128 may be a nonbinding upper limit, not 128 realized
concurrent requests. Inspect actual scheduler load before interpreting it.

Use each matching-concurrency no-SpecDec baseline, plus the fastest no-SpecDec
baseline across the sweep. Report actual output tokens, quality and incomplete
runs alongside performance. A configured graph envelope is not graph replay
evidence. See [README](README.md) for the complete analysis criteria.
