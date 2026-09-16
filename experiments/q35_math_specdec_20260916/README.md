# Qwen3.5-35B-A3B: graph-enabled Math SpecDec gate

## Approved intent

Compare no-SpecDec baseline, frozen DFlash K5 and frozen DSpark K5 at
default concurrency and S64. Start with 3-step functional gates, then run
20 steps only after the relevant gate passes. `enforce_eager=false` and
`moe_backend=flashinfer_trtllm` are required in every arm; no silent eager
or Triton fallback. FAP is requested, but runtime graph replay and actual
MoE backend selection must be audited before claiming coverage or speedup.

Use the shipped Megatron Qwen3.5 recipe as the parallelism/workload starting point,
but replace its Base target with the user-confirmed post-trained target:
`examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml`.
Keep GBS512 (32 prompts × 16 generations), maximum total length4096,
training TP2/CP2/EP16 and generation TP4. Repack 16 GPUs from 2n8g to 4n4g
for OCI-HSG. Precision is BF16; checkpointing and validation are disabled
for this short timing cohort. This is not an untouched performance recipe.
Thinking is explicitly enabled. The 4K context gate checks functionality only;
truncated reasoning cannot support accuracy or long-context speedup claims.
Inspect truncation before deciding whether to promote this workload to20 steps
or instead select a separate long-context cohort.

## Verified assets and confirmed lineage

The user confirmed `Qwen/Qwen3.5-35B-A3B`, NOT `-Base`, as the drafter target.
The public target is pinned to revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
It was not found in the checked local model/cache locations. Stage it first at:

```
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/models/Qwen3.5-35B-A3B-59d61f3ce65a6d9863b86d2e96597125219dc754
```

Replay B8 drafter directories exist beneath `specdec_ptv23/ptv3_swa`:

- `sd2p3rp-q35-a3b-ptv3rp25-dflash-b8-16n/exported-checkpoint-44000`
- `sd2p3rp-q35-a3b-ptv3rp25-dspark-b8-16n/exported-checkpoint-44000`

Both exports have `config.json` and weights; vocabulary248320, hidden2048,
target layer count40. Export configs do not record a target identifier, so
lineage is grounded in the user's confirmation, not inferred from dimensions.
The existing complete Base snapshot must NOT be used. No Base-target GPU job
was submitted before this correction.

## Graphs and concurrency

Four TP4 engines share GBS512, so equal request sharding gives128 per engine.
The explicit graph shape envelope covers128 requests for both default/S64:
width1, plus target verification width6 and DSpark draft width5 as applicable.
Power-of-two request buckets bound nominal shape padding to2×. The exact
default `max_num_seqs` is left to vLLM and must be read from runtime output.
Holding buckets fixed separates the scheduler-cap comparison from graph-list
changes. Envelope arithmetic does not prove hybrid-model graph compatibility.

## Runtime and submission

- Source: `/home/sna/nemorl-q35-math-specdec-20260916`, isolated branch
  `codex/q35-math-specdec-20260916`.
- Container: existing pinned `nemo_rl_nightly_20260909_7023221.sqsh`.
- Target/draft inputs copied once per node to `/raid/scratch/sna/q35-JOBID`.
- MCore and optional source-verified DSpark FAP overlay are node-local.
- Durable results: `.../users/sna/experiments/q35-math-specdec-20260916`.
- W&B: `nvidia/sna-specdec`, group `q35-math-bf16-fap-20260916`.
- No job dependencies or automatic promotion/resubmission.

After commit/push, remote pull and recursive submodule initialization:

```bash
python3 experiments/q35_math_specdec_20260916/launch.py --test-only baseline default --account coreai_dlalgo_nemorl
python3 experiments/q35_math_specdec_20260916/launch.py --submit baseline default --account coreai_dlalgo_nemorl
```

Every submit performs its own scheduler test-only check. Inspect current
FairShare and estimated starts before choosing the account. The 3-step
gate uses batch/4h; unmeasured 20-step budget is conservatively batch_long/8h.
Tighten only after measuring the gate. W&B key must be exported, never logged.

## Success criteria and analysis

Require model initialization, generation, finite logprobs/loss, policy
optimizer updates and repeated target refit through step3. For SpecDec also
check drafter loading, nonzero proposal/acceptance counters and graph runtime
evidence. On failure diagnose; do not change eager/backend requirements to
make the gate pass.

For completed cohorts use exact steps3–20, matched no-SpecDec baselines and
separate speed from quality: generation/E2E TPS and time; reward, generated
length, entropy, generation/policy KL with median/max and spike steps.
Changing scheduler concurrency does not guarantee accuracy invariance.

## Local verification

Existing parent launcher tests passed10/10 before changes. New launcher
tests failed because it did not exist (RED), then passed6/6 after adding it.
Resolved-config checks additionally verify inherited workload/parallelism.
No successful GPU run is implied by these tests. After the target correction,
8/8 tests, Ruff, shell syntax and git diff checks passed. The target-regression
test was observed failing against the Base path before the correction.

## Actual staging submission: 2026-09-16 10:30 UTC

- Job **7189332** stages the post-trained target only; it is NOT a Math run.
- Account/partition: `coreai_dlalgo_nemorl / batch`, one four-GPU node,1hour.
- Source: `40946fa7d`; recursive Bridge/MCore/Automodel/Gym revisions verified.
- Observed user FairShare: coreai_dlalgo_nemorl0.856994,
  nemotron_n3_post0.819207, coreai_dlalgo_llm0.710856.
- Test-only predicted starts:13:02,13:04,15:44 respectively on September16
  (scheduler timestamps, not reservations). Probe IDs7189326–7189328 are
  not submitted jobs.
- Receipt: `.../experiments/q35-math-specdec-20260916/stage-submission-20260916.txt`.
- The six Math configs are prepared but no Math GPU job was submitted.
  Validate stage receipt and files before running the launcher. No background
  auto-submitter or promotion daemon is installed.

## Separate Qwen3-30B-A3B SWE RL readiness

Remote readback on September16 confirms recovery job7097422 failed (1:0),
elapsed2:29:47. W&B: https://wandb.ai/nvidia/sna-specdec/runs/vfvzealt.
The final driver traceback reports four batch-worker failures exceeding
max_generation_failures=3, because NeMo-Gym returned no generation data
(`response.output=[]`). ClientOSError retries are also logged. These facts
do not establish whether the initiating cause is OpenHands runtime, server
connectivity, response conversion, or context handling; increasing retry or
context limits without tracing the original failure is not a fix.

Continue from the separate `q30-openhands-full-rl-20260912` worktree's full
async GRPO gate: native Thinking/tool-call template, BF16 vLLM TP2,
enforce_eager=false, flashinfer_trtllm, FAP,4n4g (two training/two rollout
nodes), GBS8 and3optimizer steps. First validate one real OpenHands trajectory
with generation and reward; then require repeated optimizer/refit cycles.
Do not reuse rollout-only speedups as full SWE RL speedups. The native SWE2
recipe expects an SWE1/pivot model, whereas the gate uses public Thinking-2507;
distinguish pipeline validation from reproduction of published training.
No SWE job or fix was submitted in this turn.
