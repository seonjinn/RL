# Qwen3.5-35B-A3B Base: graph-enabled Math SpecDec gate

## Approved intent

Compare no-SpecDec baseline, frozen DFlash K5 and frozen DSpark K5 at
default concurrency and S64. Start with 3-step functional gates, then run
20 steps only after the relevant gate passes. `enforce_eager=false` and
`moe_backend=flashinfer_trtllm` are required in every arm; no silent eager
or Triton fallback. FAP is requested, but runtime graph replay and actual
MoE backend selection must be audited before claiming coverage or speedup.

Use the shipped Megatron Qwen3.5 Base recipe, not the Qwen3 DAPO adaptation:
`examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml`.
Keep GBS512 (32 prompts × 16 generations), maximum total length4096,
training TP2/CP2/EP16 and generation TP4. Repack 16 GPUs from 2n8g to 4n4g
for OCI-HSG. Precision is BF16; checkpointing and validation are disabled
for this short timing cohort. This is not an untouched performance recipe.

## Verified assets and pending lineage

The Base target snapshot is cached completely (14 shards, 71,903,877,960
bytes) at:

```
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--Qwen--Qwen3.5-35B-A3B-Base/snapshots/0f0813072d2358973511097385626f21fcb6d422
```

Replay B8 drafter directories exist beneath `specdec_ptv23/ptv3_swa`:

- `sd2p3rp-q35-a3b-ptv3rp25-dflash-b8-16n/exported-checkpoint-44000`
- `sd2p3rp-q35-a3b-ptv3rp25-dspark-b8-16n/exported-checkpoint-44000`

Both exports have `config.json` and weights; vocabulary248320, hidden2048,
target layer count40 match the cached Base configuration. These dimensions
do NOT establish which target checkpoint trained the drafter. Export configs
do not record a target identifier. User confirmation of Base versus the
post-trained target is pending. SpecDec submission therefore requires an
explicit `--confirm-base-target-lineage` receipt; do not pass it speculatively.
Baseline gates can proceed independently.

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
No successful GPU run or job submission is implied by these tests.
