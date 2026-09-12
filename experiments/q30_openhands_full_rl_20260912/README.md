# Qwen3-30B-A3B Thinking: full OpenHands SWE2 RL gate

Status: first attempt **7097165** reached RUNNING but hit Ray startup's
107-byte Unix socket path limit before training. Cancelled the retrying job;
fixed the launcher with a short job-scoped `RAY_TMPDIR` under node-local scratch.
The regression test reproduces the 111-byte generated path and passes after
the fix. Three training steps are not yet verified.

## Submission receipt

- Submitted: 2026-09-12 06:38 UTC (2026-09-11 23:38 PDT).
- Account/partition: `nemotron_n4_post` / `batch`.
- Resources: 4 nodes, 4 GPUs/node, 4-hour limit; no job dependency.
- Submitted source: `a82e4a3113df8ea976bfd3c258e1c70fb12b62d2`.
- Gym: `fd5e84d6b1c485c80e7ae61553bbd485611c03b4`.
- Bridge: `5ed97996cc2b422904d18179375b6d7366915097`.
- Megatron-LM: `1e7598cbfae888cdd3d741a351aae588d56f66c0`.
- Artifact directory:
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-openhands-full-rl-20260912/baseline-3step-20260912T063857Z`.
- W&B run URL: unavailable until initialization; no run ID inferred.
- Local validation: four launcher/staging tests passed, shell syntax and
  merged/resolved configuration passed; remote patch dry-run passed.

The `sbatch --test-only` output mentioned probe ID 7097164. The actual submitted
job is **7097165**. Its initial state is Priority, not an application failure.

## Scope

No-SpecDec baseline, BF16 vLLM with FlashInfer TRTLLM MoE, FAP,
128K context and the native SWE2 OpenHands/Thinking template. This is full
async GRPO (policy optimizer and refit enabled), not trajectory collection.
W&B project `sna-specdec`, group `sna-swe2-full-rl-gate`.

The small gate uses 4 GB200 nodes with 4 GPUs each: 2 training nodes
(TP2/CP4/EP4/PP1/ETP1) and 2 generation nodes (TP2). PPS1 × GPP8 = GBS8;
3 training steps, packing enabled, no drafter, no checkpoint saves.
This is not the native 16n8g/GBS64 performance recipe. A successful gate
does not establish representative SWE performance or baseline speedups.

## Validation sequence

1. Local shell/config checks; commit and push isolated source.
2. Remote pinned recursive submodules, source cleanliness, image/assets check.
3. `submit.sh --test-only`, then `submit.sh --submit` on `nemotron_n4_post/batch`.
4. Per-node imports, Apptainer presence, Gym server-venv dry run.
5. Real OpenHands trajectories, policy updates and repeated refit for 3 steps.

GPU success requires all three optimizer steps and weight-update cycles,
finite rewards/loss/logprob metrics, successful sandbox responses, and no
silent skip-training/trajectory-collection path. Zero reward or identical
group rewards may produce no useful policy gradient; inspect before claiming
learning. The dry run is environment preparation, not an RL success signal.

The initial probe intentionally does not reuse the older August Gym archive:
its Gym revision/runtime patches differ from this source. Cold environment
setup errors must be diagnosed before scaling to 20 steps.

The pinned Gym OpenHands setup downloads an amd64 jq executable unconditionally.
An OCI-HSG-only patch selects arm64 in the staged node-local copy; the shared
Gym submodule is unchanged. A regression test checks both the download target
and source-checkout preservation.

## Timing interpretation

Async `exposed_generation` is replay-buffer wait on the training critical
path, not total rollout time. For the blog, report that distinction and
separate model-call, tool/sandbox, and trajectory durations. Do not sum
concurrent trajectory durations and label that value a batch wall time.
