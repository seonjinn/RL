# Qwen3-235B DSpark and DFlash Four-Arm Experiment Design

**Goal:** Find a Qwen3-235B speculative drafter that approaches or exceeds 2x SWE generation throughput over the 87 tok/s non-speculative baseline, while separating architecture, block length, and continued-training data effects.

## Scope

Run four training candidates on AWS-DFW:

1. DSpark block 8 from scratch.
2. DSpark block 16 from scratch.
3. DFlash v2 continued training on fresh public SWE data.
4. DFlash v2 continued training on a token-matched public plus hard on-policy SWE mix.

The experiment compares deployment efficiency rather than raw historical training wall time. DFlash v2 was trained before the selected #823 and #832 training-performance patches, so its old wall time is not a clean training-speed control. Model quality, acceptance, GPU-hours consumed by the new arms, and held-out SWE rollout throughput remain comparable.

## Success Criteria

The primary metric is completion tokens divided by model-call time on a held-out SWE rollout harness.

- Non-speculative reference: 87 tok/s.
- Existing DFlash v2 reference: 133.21 tok/s at K=5.
- Primary target: at least 174 tok/s, or 2.0x the non-speculative reference.
- Minimum promotion gate: statistically credible improvement over DFlash v2 without a valid-rollout-rate or reward regression.

Validation EAL and position accuracy are screening metrics. They cannot replace the held-out rollout gate because validation distributions differ and have previously ranked checkpoints incorrectly.

## Software Baseline

Use the isolated AWS-DFW Speculators worktree:

```text
/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/speculators-dspark-v3-fixes-20260723
branch: sna/dspark-v3-fixes-20260723
HEAD: 9b1a6200f2204663e8f4f1542d3a1f52f4d53d97
```

The branch contains the merged DSpark foundation and patch-equivalent cherry-picks of:

- #805: correct slot-0 DSpark acceptance and confidence metrics.
- #848: detach confidence-head inputs from the drafter backbone.
- #823: sort anchors for contiguous flex-attention blocks.
- #832: compute verifier targets only at anchored positions.

The exact source commit, launcher commit, container identifier, dataset-manifest hashes, environment metadata, and SLURM job IDs must be recorded for every arm.

## Experiment Matrix

| Arm | Initialization | Block | Data | Main question |
|---|---|---:|---|---|
| DS8 | Random DSpark drafter; frozen Qwen3-235B verifier | 8 | Frozen DFlash v2 prompt manifest | Does DSpark outperform DFlash v2 on the same prompt distribution? |
| DS16 | Random DSpark drafter; frozen Qwen3-235B verifier | 16 | Same manifest as DS8 | Does longer native training improve late-position acceptance enough to pay for added draft work? |
| DF-PUBLIC | DFlash v2 weights, fresh optimizer and scheduler | 8 | Fresh filtered public SWE set | Does new SWE diversity improve the existing drafter? |
| DF-HARD | DFlash v2 weights, fresh optimizer and scheduler | 8 | Token-matched public plus hard on-policy SWE set | Does deployment-matched hard data beat public-only continuation? |

DS8 and DS16 use the exact same prompt manifest, train/validation split, manifest order seed, target-generation configuration, sequence cap, and epoch count. Online target regeneration means hidden states are not expected to be byte-identical, so reproducibility is defined by inputs, configuration, source state, and seeds.

## Shared Qwen3-235B Configuration

```text
verifier: Qwen3-235B-A22B-Thinking-2507
draft layers: 5
draft hidden size: 4096
draft vocabulary: 32,000
hidden activation: silu
sliding window: 2,048
target layer taps: 1, 23, 46, 68, 91
sequence length: 16,384
maximum anchors: 1,024
loss: kl_div
target model: frozen
checkpoint frequency: 0.25 epoch plus SIGTERM save
```

The Qwen3-235B target regenerates responses online. Original dataset responses do not define the target distribution; prompt, scaffold, tool history, context length, and target-generation settings do.
Use the verified DFlash v2 per-device batch, global batch, gradient accumulation, and distributed settings unchanged. If DS16 cannot fit 1,024 anchors, record the failed smoke and create a separately identified DS16-M512 retry; do not silently change the arm.

## DSpark Configuration

Shared DSpark settings:

```text
speculator type: dspark
sample from anchor: true
Markov head: vanilla
Markov rank: 256
confidence head: enabled
confidence head with Markov features: enabled
confidence-head alpha: 1.0
optimizer: Muon plus AdamW split
learning rate: 6e-4
schedule: cosine
epochs: 2
```

Block-specific settings:

| Arm | Block size | Decay gamma | Reason |
|---|---:|---:|---|
| DS8 | 8 | 4 | Matches the validated DFlash v2 scale and current DSpark default. |
| DS16 | 16 | 8 | Preserves an approximately comparable relative decay profile across the longer block. |

With gamma 4, DS16 position 15 would receive only `exp(-15/4)`, approximately 2.4% of the position-0 weight. Gamma 8 raises it to approximately 15.3%, so training a longer block does not nominally enable positions that receive negligible optimization pressure.

DS16 is expected to use more compute than DS8. Compare both final checkpoints at equal data exposure and intermediate checkpoints at matched GPU-hours. Report step time, GPU-hours, peak memory, number of target tokens, number of supervised draft positions, and checkpoint quality.

## Dataset Policy

### DSpark architecture controls

Use the frozen DFlash v2 manifest of 850,220 prompts:

- 818,540 generic prompts, including approximately 190,000 Open-PerfectBlend prompts.
- 30,000 filtered public SWE trajectories.
- 1,680 upsampled target-generated training-pool trajectories.

Do not increase Open-PerfectBlend for DS8 or DS16. It already represents approximately 22.3% of the complete corpus, and generic sources represent 96.3%. Increasing it would confound the architecture comparison and move the distribution away from the OpenHands SWE deployment path.

### DFlash continuation controls

Both continuation arms use a new save path and load DFlash v2 through `--from-pretrained`. They must not resume the completed v2 optimizer or cosine scheduler.

Start with a fresh learning rate of 1e-4. A short smoke must show finite loss and gradients with no step-to-step loss explosion before the full run. If it fails that gate, create a separately identified 6e-5 retry. Each continuation arm contains 100,000 example passes, approximately 5.9% of v2's 1.70 million passes, and is truncated at a matched cumulative target-token count.

DF-PUBLIC uses filtered NVIDIA Open-SWE-Traces prompts with these preferences:

- OpenHands scaffold only for the primary arm.
- 50% Python trajectories and 50% other supported languages by example count before token matching.
- Long-context trajectories sufficient to exercise the 8K-to-16K deployment range.
- Both resolved and unresolved trajectories, because target regeneration defines the training response.

DF-HARD uses the same total target-token mass and base source pool. It uses 80,000 examples from the DF-PUBLIC manifest and replaces 20,000 examples with:

- examples where DFlash v2 has low acceptance at positions 3 through 8;
- contexts longer than 8K where acceptance drops;
- target-generated rollouts from a broad, non-evaluation SWE training-instance pool.

Do not repeat the original seven-instance pool with another large fixed upsampling factor. Expand the disjoint training-instance pool and cap each instance at four selected trajectories before token matching to reduce memorization.

## Data Integrity Gates

Every manifest must pass before GPU submission:

1. Exact duplicate removal by normalized conversation hash.
2. Evaluation-instance ID exclusion.
3. Repository and issue fingerprint exclusion for all held-out astropy tasks.
4. Distinctive problem-statement n-gram and normalized-text similarity scan.
5. OpenHands system-prompt handling that preserves the required scaffold without using it as a false contamination match.
6. Source, repository, language, scaffold, resolved status, and token-length distribution report.
7. Manifest SHA256 plus immutable train and validation split hashes.
8. Cross-arm assertion that DS8 and DS16 manifest hashes are identical.
9. Token-mass assertion that DF-PUBLIC and DF-HARD differ by at most 1%.

A failed contamination or hash gate blocks submission. The launcher must not provide a bypass flag for production runs.

## Training Lifecycle

Each arm uses a unique experiment directory containing:

- `README.md` with objective and status.
- immutable config and source metadata;
- dataset manifest and integrity report;
- SLURM launcher;
- job-ID ledger;
- stdout and stderr logs;
- checkpoint inventory;
- per-epoch metrics;
- failure and retry journal.

Use two-node online training: one four-GPU Qwen3-235B target-server node and one four-GPU drafter-training node. Submit using reproducible containers and explicit source paths. Because the interactive queue has a four-hour limit, use checkpoint-aware chained jobs followed by a `batch_long` finisher only when needed.

Before every submission:

1. Pull the committed source and launcher state.
2. Run the scheduler test-only preflight.
3. Confirm account, partition, GPU count, container, paths, quota, and free inodes.
4. Confirm the previous job is not still writing the same save path.
5. Record the exact command and resulting job ID.

Monitor every newly running job for at least five minutes. A training smoke must reach data loading, target generation, forward pass, backward pass, optimizer step, metric emission, and checkpoint save before a full chain is eligible.

## Evaluation Design

Use the same held-out SWE evaluation set, target checkpoint, runtime image, generation configuration, concurrency, maximum sequence settings, and CUDA-graph coverage for every candidate.

K sweeps:

- DS8: K=3, 5, 7, and 8.
- DS16: K=5, 8, 12, and 16.
- DF-PUBLIC and DF-HARD: K=3, 5, and 7.

The model configuration must support the requested K. Any K that triggers eager fallback or misses the explicit `(K+1)`-aware CUDA-graph capture set is invalid rather than a performance result.

Collect:

- completion tok/s and model-call time;
- draft, verify, and non-model rollout time;
- mean accepted length and acceptance by position;
- confidence calibration and cumulative confidence bias for DSpark;
- request latency distribution;
- peak GPU memory and graph-capture status;
- valid-rollout rate, reward, timeouts, and failure class.

Use at least 20 valid held-out trajectories per promoted configuration. Run a warm-cache matched crossover rather than comparing isolated three-trajectory smoke jobs. Preserve the existing three-prompt harness only as an operational smoke.

## Decision Rules

1. Reject any checkpoint with contamination, checkpoint-load, CUDA-graph, correctness, or valid-rollout regression.
2. Reject a K setting that is slower than the same checkpoint at a shorter K under matched conditions.
3. Promote a candidate over DFlash v2 only when repeated held-out throughput improves and reward and valid-rollout rate remain non-inferior.
4. Declare the 2x objective met only at 174 tok/s or higher under the matched held-out protocol.
5. If DS16 improves late-position acceptance but not throughput, retain DS8 and attribute the result to additional draft or verification cost.
6. If both DSpark arms underperform DFlash v2 on the frozen v2 manifest, do not immediately change data and claim an architecture result. First check training convergence, Markov contribution, confidence calibration, and serving correctness.
7. If DF-HARD beats DF-PUBLIC, use its data policy for a later continuation of the winning architecture.

## Serving Gate

Training success does not imply that a DSpark checkpoint is deployable. Before rollout evaluation, build an immutable vLLM wheel containing the merged DSpark config-load and auxiliary-feature-width fixes identified in the upstream audit. Validate:

1. Speculators-format config loading.
2. `sample_from_anchor` propagation.
3. auxiliary `fc` input width.
4. one eager request.
5. one compiled request.
6. matched CUDA-graph capture at each requested K.

Do not edit an installed wheel or site-packages in place.

## Failure Handling

- Save checkpoint and training state on SIGTERM.
- Use a new experiment path for every arm and retry generation.
- Never overwrite a completed checkpoint.
- Classify infrastructure, data, target-server, trainer, OOM, convergence, and serving failures separately.
- A retry may change only the failing layer unless the experiment receives a new arm identifier.
- Record unsuccessful hypotheses, commands, job IDs, error excerpts, and the evidence that justified each fix.

## Reporting

Update `docs/dflash_drafter_training.html` throughout the experiment with:

1. frozen design and source provenance;
2. exact dataset composition and integrity results;
3. submitted and completed job ledger;
4. training curves and checkpoint decisions;
5. failed attempts and fixes;
6. matched K-sweep results;
7. final comparison against non-spec, DFlash v1, and DFlash v2;
8. whether the 2x target was met and why.

Do not publish projected throughput as a measured result.
