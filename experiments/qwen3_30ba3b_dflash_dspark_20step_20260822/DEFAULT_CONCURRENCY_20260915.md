# Historical Q30 baseline: default-concurrency control

Submitted job **7163439**, `nemotron_n3_post / batch`, four nodes × four GB200,
four-hour limit, 20 steps. Test-only planning ID 7163438 is not the actual job.

Question: does removing the explicit S8 limit improve the no-SpecDec baseline
used in the historical short-context DFlash/DSpark K7 comparison?

## Matched setup

- Baseline only; no SpecDec and no online drafter training.
- OpenMathInstruct-2; GBS512 = 16 prompts × 32 generations; seed42, no shuffle.
- Maximum output1024, total/model context8192.
- Training TP2 / EP8 / CP1 / PP1, packed sequences; vLLM TP1.
- BF16, MoE triton, PIECEWISE CUDA Graphs, eager compiler backend.
- Unchanged capture list `[1,2,4,8,12,16,24,32,40,48]`.
- **Only concurrency override removed**, in both the actual driver and the
  configuration verifier. Omitted does not mean unlimited; inspect the resolved
  vLLM scheduler default at engine initialization.
- No checkpointing, same as historical baseline.

Original W&B baseline:
`q30ba3b-20step-baseline-k0-346bf3ea6ece41308116d897245c6dd1`.
New run ID, once W&B initializes:
`q30ba3b-20step-baseline-k0-sdefault-1a742042cbc24dc59a56c4f1f768af14`.
Project: `nvidia/sna-specdec`.

Compare Steps3–20 (18 points) for generation TPS/GPU, generation time,
E2E time/TPS, reward, entropy, generation/policy KL, actual output lengths.
Do not mix with DAPO40K/GBS2048 or the Q8 long-context gates. Historical K7
arms retain S8: comparing them against this baseline is a differently tuned
deployment comparison, not a matched-S estimate of the method alone.

## Reproducibility and submission fix

Harness commit: `831029d44` (14 local tests pass).
Pinned product commit: `d0c4f1110cca28c75b7a1d98ed2d5f197e7d01dc`.
Container: original `nemo_rl_nightly_20260818_20260818_6296116.sqsh`.

The original `/home/sna/nemorl-pr11-q30-baseline-green` retained the correct
commit, but its Bridge/MCore subtree was not clean. The fail-closed preflight
prevented submission. No old files were removed or reverted. A clean local
clone was created at `/home/sna/nemorl-q30-sdefault-product-20260915`, with every
recursive submodule restored at its original gitlink. All clean-source guards
then passed, followed by test-only and actual submission.

Submodules: Automodel `24b47e856263d313b942f0ed666c63fff83306b4`,
Gym `c3bac96314a59f28b896f597eb9845d175bb0252`,
Bridge `8c46dc4259080c510b7455f43e836fdff222c5d3`,
MCore `14346b65a2d0790e451919858f7771078105c5f0`.

Harness path: `/home/sna/nemorl-q30-sdefault-harness-20260915`.
Result root:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_dflash_dspark_20step_20260822/concurrency-default-20260915/artifacts/q30ba3b-20step-baseline-k0-sdefault-1a742042cbc24dc59a56c4f1f768af14/`.

No completed result is claimed. GPU initialization/capture/step1/step2 remain
required checks after the scheduler starts the job.
