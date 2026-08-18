# SpecForge Qwen3 Drafter Matrix for Math and NeMo-RL SWE-RL

**Date:** 2026-08-17

**Status:** Approved for implementation planning

## Goal

Train sixteen independent SpecForge checkpoints for
`Qwen/Qwen3-235B-A22B-Thinking-2507` and
`Qwen/Qwen3-30B-A3B-Thinking-2507`, then deploy the resulting DFlash and
DSpark drafters through vLLM as NeMo-RL external drafters.

Pin the targets to revisions
`6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5` and
`144afc2f379b542fdd4e85a1fcd5e1f79112d95d`, respectively.

The optimization target is not standalone decoding alone. A checkpoint is useful
only if it improves Math or NeMo-RL SWE-RL rollout generation while preserving
rollout validity and task quality.

## Scope

This design has two deliberately separated stages:

1. Train the approved sixteen-checkpoint general-purpose matrix on
   target-regenerated Open-PerfectBlend and the public Red Hat EAGLE-3 data mix.
2. After the matrix has trustworthy Math and SWE baselines, continue selected
   checkpoints on an agentic SWE curriculum while replaying Math data.

Stage 2 is a later experiment and must use new checkpoint names. It must not
alter the sixteen baseline arms or be reported as a reproduction of the DSpark
paper.

## Evidence and Corrections

The local source paper is
`/Users/sna/Downloads/dspark (1).pdf`, SHA-256
`522036b0cc16ad4678bd7c278dd0a0ab4da31170af7b97c2041067cc09a8289a`.

The paper does not report additional OpenCode training. Section 4.1 reports one
training corpus: target-regenerated Open-PerfectBlend, approximately 1.3 million
usable samples after filtering. Its reported composition is 17.6% chat, 39.4%
Math, 38.9% code, and 4.1% instruction-following. All drafters were trained on
the same data for ten epochs, and the published Qwen3 experiments used
non-thinking generation.

The requested targets use Thinking-2507. Therefore this work is an adaptation,
not an exact paper reproduction. Responses must be regenerated independently by
each selected Thinking-2507 target with the exact production chat template.

The paper's most relevant architectural conclusions are:

- DSpark uses a five-layer parallel backbone, lightweight sequential Markov
  head, and confidence head.
- The objective combines cross entropy, total-variation distribution matching,
  and confidence loss with default coefficients 0.1, 0.9, and 1.0.
- DSpark's relative benefit over DFlash grows at longer proposal lengths because
  it reduces suffix acceptance decay.
- Confidence scheduling can reduce rejected suffix work, but it is a serving
  optimization distinct from raw fixed-length checkpoint quality.

### Public-drafter dataset evidence

The current public evidence supports two recurring prompt families, but it does
not establish a universal best dataset through a controlled dataset ablation:

| Source | Method and target relevance | Prompt source | Strength of evidence |
| --- | --- | --- | --- |
| DFlash paper | DFlash on Qwen3-4B, Qwen3-8B, and Qwen3-Coder-30B-A3B | approximately 800K prompts from Nemotron-Post-Training-Dataset-v2 plus CodeAlpaca, with target-generated responses | Strong method result for the full mixture; non-thinking and no component dataset ablation |
| DSpark paper | DSpark, DFlash, and EAGLE-3 on Qwen3 targets | target-regenerated Open-PerfectBlend, approximately 1.3 million usable samples | Strong controlled method comparison on a shared corpus; not a dataset ablation |
| SpecForge Qwen3-8B DFlash reproduction | DFlash with Thinking enabled | 175K target-regenerated Open-PerfectBlend rows | Direct SpecForge precedent; authors label it an untuned demo and attribute part of the remaining gap to data quality and coverage |
| Red Hat Qwen3 Thinking EAGLE-3 cards | Same Q235 and Q30 target families | Magpie plus UltraChat with reasoning enabled | Direct public model-family precedent with reported acceptance lengths; not compared against Open-PerfectBlend under a matched budget |
| Red Hat Qwen3-30B Instruct DFlash card | DFlash on the same Q30 architecture family | Magpie plus UltraChat, regenerated responses | Direct DFlash precedent with multi-domain acceptance results; target/card details require independent verification before reuse |
| Red Hat GLM-5.2 DSpark card | Public DSpark deployment | target-regenerated Open-PerfectBlend | Direct DSpark deployment precedent with multi-domain acceptance results; different target family |
| NVIDIA Q235 Thinking EAGLE-3 and current DFlash cards | Exact Q235 Thinking EAGLE-3 target or current DFlash targets | Nemotron-Post-Training-Dataset-v2 prompts, target-regenerated responses | Broad Math, code, STEM, multilingual, and chat adoption evidence; no matched comparison against the two approved families |

Consequently, the sixteen-arm baseline retains Open-PerfectBlend and the Red Hat
Magpie-plus-UltraChat family. This is the only design that both honors the
approved matrix and measures the two strongest directly relevant public recipes
under matched targets and compute. `nvidia/Nemotron-Post-Training-Dataset-v2` is
registered as a named follow-up candidate, not silently blended into either
baseline. It may replace the public-mix arm only if a pre-training audit finds a
license, availability, or contamination blocker, or it may become a separately
named continuation experiment after the sixteen baselines finish.

The DFlash-paper mixture is also a separately named reproduction candidate.
Its CodeAlpaca component is CC BY-NC 4.0, so it is not included in the baseline
without an explicit license/use review. The paper's Qwen3 results were
non-thinking and therefore do not override the same-family Thinking precedent
from the Red Hat public checkpoints.

Public adoption is evidence that a prompt family is viable, not proof that it
is optimal for these targets. Dataset effectiveness will be concluded only from
the matched Open-PerfectBlend versus public-mix arms with identical target,
algorithm, block size, optimizer-update budget, token budget, and evaluation
settings.

### Existing workspace evidence

Completed local runs provide stronger workload-specific priors, but they use
different trainers, corpora, or serving stacks and therefore are not substitutes
for the new matched matrix:

- Q235 Thinking DFlash trained from scratch on an 850K SWE-oriented mix reached
  matched K3 speedups of 1.747x on Math and 1.408x on SWE. This supports a later
  deployment-matched SWE continuation after clean baseline training.
- Q235 Thinking DSpark on a general 600K corpus produced smaller positive K3
  speedups, while the later 850K SWE continuation reduced acceptance and
  speedup. Every completed DSpark K5 arm was slower than baseline. DSpark must
  therefore earn continuation through its own small held-out pilot rather than
  inheriting the DFlash data decision.
- Q30 Open-PerfectBlend DFlash reached 1.325x on Math but only 1.063x on the
  matched SWE aggregate at K3; K5 was slower than baseline. General training
  alone is unlikely to satisfy the SWE-RL objective.
- A Q30 EAGLE-3 500K mixed/OpenMath checkpoint produced approximately 2.0x on
  Math/OpenMath and 1.784x on SWE in completed standalone K3 runs, the strongest
  balanced broad-domain local precedent.
- The corrected public Q235 Thinking EAGLE-3 run showed 65.71% acceptance yet
  only 0.986x generation throughput and 0.975x end-to-end throughput. High
  acceptance is necessary evidence of draft quality, not sufficient evidence
  of NeMo-RL acceleration.

These observations make method-specific continuation gates mandatory. In
particular, the DFlash SWE mixture is a candidate recipe, whereas DSpark first
requires a small controlled mixture sweep that preserves the general corpus and
tests whether SWE data improves actual held-out rollout metrics.

## Approaches Considered

### Selected: SpecForge online disaggregated training

A pinned SGLang target server captures target features while SpecForge consumers
train the drafter. This avoids materializing full hidden-state corpora, preserves
the upstream producer-consumer contract, and is the only practical full-corpus
path for the 235B target.

### Rejected for full runs: offline hidden-state materialization

Offline features are useful for small deterministic smokes, but storing several
target layers for hundreds of thousands of long Thinking samples would require
impractical Lustre capacity. B8/B16 reuse does not justify the quota and transfer
risk.

### Retained only as a control: Speculators training

Existing Speculators checkpoints and evaluations remain comparison evidence.
They are not the new training backend because DSpark loss-mask,
`sample_from_anchor`, confidence-gradient, and checkpoint-conversion issues are
part of the motivation for this independent SpecForge path.

## Training Matrix

Every row below expands into DFlash B8, DFlash B16, DSpark B8, and DSpark B16.

| Target | Data family | Independent checkpoints |
| --- | --- | ---: |
| Qwen3-235B-A22B-Thinking-2507 | Open-PerfectBlend | 4 |
| Qwen3-235B-A22B-Thinking-2507 | Red Hat EAGLE-3 public mix | 4 |
| Qwen3-30B-A3B-Thinking-2507 | Open-PerfectBlend | 4 |
| Qwen3-30B-A3B-Thinking-2507 | Red Hat EAGLE-3 public mix | 4 |

The complete Cartesian product is:

```text
2 targets x 2 data families x 2 algorithms x 2 block sizes = 16 checkpoints
```

All arms start from independently initialized drafter weights. They use unique
run IDs, output directories, manifests, training ledgers, and checkpoint hashes.
No B16 arm warm-starts from B8 because that would confound block-size effects.

## Data Contracts

### Open-PerfectBlend

Pin `mlabonne/open-perfectblend` to revision
`af60f3c18201652a83a93f46fcfee1b646ba3df7`. The source contains 1,420,909
rows before filtering. Filtering must reject rows without a non-empty assistant
turn, malformed conversations, duplicate normalized prompts, overlength prompts,
and evaluation contamination.

Only the prompt-side conversation is authoritative. Each target regenerates its
own assistant response with Thinking enabled. Existing OCI artifacts may be
reused only when their manifest proves the same source revision, target snapshot,
chat template, thinking mode, sampling configuration, and complete source-ID
coverage.

### Red Hat EAGLE-3 public mix

Use the datasets named by the corresponding public Red Hat model cards:

- `Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered` revision
  `1a982eea9ece373700dd8dfd04a4de08c2578c24` for the 235B family;
- `Magpie-Align/Magpie-Pro-300K-Filtered` revision
  `523df96eb7474e97bca6f378b3baa372a4735fcc` for the 30B family;
- `HuggingFaceH4/ultrachat_200k` revision
  `8049631c405ae6576f93f445c6b8166f76f5505a`, split `train_sft`, for both.

Record each exact dataset revision and license. Magpie's Llama 3.1 terms and
UltraChat's MIT license must be preserved in provenance. Responses are regenerated
by the matching target, not copied across Q30 and Q235.

### Split and contamination gates

Use immutable train, validation, Math-evaluation, and SWE-evaluation partitions.
The build fails closed on:

- duplicate normalized conversation hashes across partitions;
- SWE-bench, SWE-bench Verified, or selected NeMo-RL evaluation instance IDs;
- repository, issue, patch, problem-statement, and distinctive n-gram matches to
  held-out SWE tasks;
- GSM8K, MATH500, AIME, or selected Math evaluation-answer leakage;
- missing source IDs, malformed tool messages, empty supervised spans, or
  target/template mismatches.

Every dataset and generated-response artifact receives a manifest SHA-256,
ordered source-ID hash, schema version, tokenizer snapshot, chat-template hash,
target snapshot, sampling parameters, row count, token count, and rejection
report.

## Model and Loss Configuration

Generate each draft configuration from the pinned verifier config. Do not copy
the older Hayate reference's incompatible intermediate size, context limit, or
RoPE theta.

Shared settings:

- five dense Qwen3 draft layers;
- verifier-derived hidden size, attention geometry, vocabulary, normalization,
  positional encoding, and token IDs;
- target feature taps `[1, 23, 46, 68, 91]` for Q235 and
  `[1, 12, 23, 34, 45]` for Q30, subject to an exact capture-shape smoke;
- frozen target, shared frozen embedding, and shared frozen LM head;
- BF16 drafter training;
- 1,024 sampled anchors for production unless a separately named memory retry
  is required;
- deterministic data order and recorded seeds;
- checkpointing at fixed optimizer-update intervals plus SIGTERM recovery.

DFlash settings:

- `strategy: dflash`;
- requested checkpoint `block_size` 8 or 16;
- `sample_from_anchor: false`;
- five-layer parallel backbone;
- CE coefficient 0.1 and TV coefficient 0.9.

DSpark settings:

- `strategy: dspark`;
- requested checkpoint `block_size` 8 or 16;
- `sample_from_anchor: true`;
- Markov rank 256 with vanilla Markov head;
- confidence head enabled with Markov features;
- CE coefficient 0.1, TV coefficient 0.9, and confidence coefficient 1.0.

Use initial `loss_decay_gamma` values 4 for B8 and 8 for B16. A one-sample
overfit and 128-row smoke must validate the actual SpecForge objective, masks,
position weights, and confidence labels before production. Any changed decay,
anchor count, or loss coefficient creates a new retry arm rather than mutating
an approved arm.

### Block-size semantics

Training block size, proposal length, requested vLLM K, and physical KV-cache
block size are separate fields.

- DFlash B8/B16 normally exposes 7/15 speculative mask positions because the
  anchor is not sampled.
- DSpark B8/B16 samples the anchor position and exposes 8/16 query positions.
- vLLM FlashAttention and Triton use physical KV-cache block size 16 for both
  training labels.

Reports must state all four values and must not compare identically named B8 or
B16 arms as if they proposed the same number of tokens.

## Cluster and Reproducibility Design

OCI-HSG is the primary cluster because the target snapshots and existing
Open-PerfectBlend assets are already visible there. Lyris is the fallback when
its FairShare and scheduler estimate are materially better and all data,
container, and authentication gates pass.

Before submission:

1. Compare configured accounts and FairShare.
2. Run `sbatch --test-only` for every scheduler class.
3. Commit and push local source.
4. Pull the exact commit into a clean remote checkout.
5. Clone SpecForge at the selected immutable commit. The 2026-08-17 candidate
   is `bdeb7d8aa77b616874ac46303fc2546739b61119`; the implementation plan must
   verify its dependency contract before freezing it.
6. Record the SGLang, PyTorch, CUDA, Transformer Engine, tokenizer, container,
   compiler, driver, and architecture versions.

Use separate target-server and trainer nodes for production Q235 runs. Q30 may
use a smaller topology only after a matched throughput and memory smoke proves
it does not change the feature contract. Every newly running job is monitored
for at least five minutes.

## Training Lifecycle

### Gate 1: static and CPU validation

- validate all sixteen typed configs;
- validate manifest identity and disjointness;
- validate target-layer capture shapes;
- validate assistant/loss masks and nonzero supervision;
- validate output paths are empty and unique;
- run a one-sample overfit for each algorithm and target family.

### Gate 2: sixteen 128-row GPU smokes

Every smoke must reach target-server readiness, feature publication, forward
loss, backward, optimizer step, metric emission, checkpoint save, export, and
checkpoint reload. DSpark additionally requires Markov and confidence gradients
with no unintended confidence-to-backbone gradient path.

### Gate 3: staged full training

Promote only passing smokes. Save and evaluate checkpoints at approximately
1, 2, 5, and 10 epochs. Ten epochs matches the paper's convergence protocol,
but checkpoint selection is based on held-out Math and SWE results rather than
assuming the final epoch is best. A plateau does not authorize silently
shortening another arm's exposure.

Use checkpoint-aware SLURM continuations. A retry changes only the failing layer,
uses a new run generation, and preserves all failed logs and partial checkpoints.

## Export and vLLM Integration

Export each SpecForge checkpoint to a Hugging Face-compatible drafter directory.
DFlash uses the standard HF export path. DSpark requires an explicit format gate
that verifies:

- `DSparkDraftModel` architecture;
- algorithm, verifier, block size, proposal method, and speculative-token fields;
- target-layer IDs and feature projection width;
- Markov and confidence-head parameters;
- top-level and nested `sample_from_anchor=true` consistency;
- exact tensor-key and tensor-shape inventory.

Use the isolated patched vLLM 0.25.1 DSpark stack, not NeMo-RL latest-main's
default vLLM 0.20.0 environment. Validate draft TP1 and target TP4 first for
Q235, with FlashAttention for the draft and the verified Qwen3 MoE backend for
the target.

Each artifact must pass:

1. config-only load;
2. weight and shape load;
3. one eager deterministic request;
4. one compiled request;
5. FULL-decode CUDA Graph capture without eager downgrade;
6. requested-K sweep bounded by the checkpoint's real proposal contract;
7. speculative/non-speculative distribution and greedy-output correctness;
8. acceptance, latency, memory, and failure-metric serialization.

DFlash/DSpark PIECEWISE graph fallback is not a valid performance result.

## NeMo-RL External Drafter Path

NeMo-RL passes the exported path through
`policy.generation.vllm_kwargs.speculative_config.model` with explicit `method`,
`num_speculative_tokens`, and `draft_tensor_parallel_size`. The run must also
record `policy.draft.model_name` for provenance.

The NeMo-RL environment must use the same immutable vLLM wheel that passed the
standalone gate. Site-package edits and unrecorded `PYTHONPATH` overlays are
forbidden.

Run three levels of SWE validation:

1. static vLLM prompt smoke with representative long SWE contexts;
2. NeMo-RL generation-only SWE rollout with the real chat template, tool
   messages, stop conditions, and concurrency;
3. matched NeMo-RL SWE-RL steps with the same target weights, prompts,
   generation parameters, batch shape, environment image, and reward path.

## Evaluation and Promotion

Only Math and SWE determine promotion.

### Math suite

- GSM8K for short reasoning;
- MATH500 for broad mathematical reasoning;
- AIME25 or the project's current uncontaminated AIME partition for long
  reasoning;
- the existing NeMo-RL Math rollout set used by current baseline reports.

### SWE suite

- SWE-bench Verified or the exact uncontaminated NeMo-RL held-out subset;
- long-context repository/issue prompts matching the SWE-RL rollout format;
- generation-only and full environment-executed rollouts;
- separate reporting by repository, context length, tool count, outcome, and
  failure class.

### Public drafter comparability suite

Run the public Hugging Face-style benchmark as a secondary reporting suite. It
does not replace the Math and SWE promotion gates. Pin
`RedHatAI/speculator_benchmarks` to revision
`2ae86affa2cb97a972b7fc681dd51c04fbff083e` and load each JSONL file
independently because the repository contains heterogeneous schemas and the
combined Hugging Face dataset viewer currently fails schema casting.

Evaluate all files present in the pinned repository:

| Domain | Exact file |
| --- | --- |
| Coding | `HumanEval.jsonl` |
| Math reasoning | `math_reasoning.jsonl` |
| Question answering | `qa.jsonl` |
| General questions / MT-Bench-style prompts | `question.jsonl` |
| Retrieval-augmented generation | `rag.jsonl` |
| Summarization | `summarization.jsonl` |
| Tool calling | `tool_call.jsonl` |
| Translation | `translation.jsonl` |
| Writing | `writing.jsonl` |

For each dataset and for the macro average across datasets, report both:

- `mean_accept_len_bonus_inclusive = 1 + accepted_draft_tokens / draft_rounds`;
- `mean_accept_len_draft_only = accepted_draft_tokens / draft_rounds`.

The public Red Hat headline metric is bonus-inclusive: the added one is the
verifier's guaranteed token. Never label that number as accepted draft tokens.
Also report marginal per-position acceptance and conditional per-position
acceptance separately; they are not interchangeable.

Produce two matched sampling tables for every successful checkpoint:

1. deterministic NeMo-RL comparability: temperature 0, top-p 1, top-k disabled;
2. public-card comparability: temperature 0.6, top-p 0.95, top-k 20.

Each table records target and drafter revision, algorithm, training data family,
block size, `num_speculative_tokens`, sampling policy, prompt count, output-token
limit, vLLM and evaluator revisions, TP/DP shape, concurrency, CUDA Graph mode,
and repetitions. B8 and B16 results are never compared under an ambiguous `K`:
the table always names both the physical checkpoint block size and the actual
draft-only `num_speculative_tokens` used by the server.

The primary compact report contains per-domain rows plus an unweighted domain
macro average. A second micro average weighted by completed draft rounds is
reported separately so large or long subsets cannot silently dominate the
headline multi-domain number. Acceptance quality and latency/throughput are
separate tables; throughput comparisons require a matched non-speculative
baseline with identical sampling, batch/concurrency, input/output lengths, and
runtime configuration.

Collect both bonus-inclusive and draft-only mean accepted length, conditional
acceptance by position, draft acceptance rate, confidence calibration, proposed
and verified token counts, draft/verify/model-call/generation/environment time,
output tokens per second, peak memory, CUDA Graph status, valid-rollout rate,
reward, timeout rate, and end-to-end RL step time.

A baseline checkpoint is promoted only when it:

- is distribution-correct and graph-valid;
- improves a repeated matched Math or SWE generation metric over non-spec;
- does not reduce Math reward or valid-rollout rate;
- does not reduce SWE valid-rollout rate or reward;
- produces an end-to-end NeMo-RL benefit after environment time is included.

Standalone tokens per second alone cannot promote a checkpoint.

## Later SWE-Domain Continuation

The paper's Open-PerfectBlend already contains 38.9% code data, but code
generation is not equivalent to SWE-RL. The later curriculum must prioritize
repository-conditioned issue-solving and tool trajectories over isolated code
completion.

Candidate data families must be audited from their primary repositories and
dataset cards before selection. The desired hierarchy is:

1. target-generated trajectories from disjoint NeMo-RL SWE training instances;
2. public repository/issue/patch trajectories with preserved tool and file
   context;
3. high-quality code instruction or code-repair data for diversity;
4. base-corpus and Math replay for retention.

The primary public candidates are:

- `nebius/SWE-rebench-openhands-trajectories`: 67,074 OpenHands trajectories
  across 1,823 repositories, including issue, tool, patch, test, and resolution
  metadata under CC-BY-4.0. Its card reports SWE-bench Verified exclusion and
  repository-level decontamination for its Verified partition.
- `nvidia/Open-SWE-Traces`: 207,489 OpenHands and SWE-agent trajectories across
  multiple models and languages under CC-BY-4.0. It is useful for tool-style
  diversity but requires an independent SWE-bench and SWE-bench Verified
  contamination scan.
- SWE-Gym: 2,438 executable real repository tasks with successful public agent
  interactions under MIT. It is small and Python-heavy, so it is a high-quality
  anchor rather than the majority source.
- SWE-smith: approximately 50,000 executable synthetic repair tasks. Use it for
  repository and failure diversity under its MIT dataset/tooling terms, not as
  evaluation evidence, and recheck current SWE-bench repository overlap.
- `nvidia/OpenCodeInstruct` and OpenCoder SFT data: code instruction and
  unit-test data without repository/tool trajectories. OpenCodeInstruct is
  CC-BY-4.0; record the exact OpenCoder split and terms. Use either only as a
  small code diversity component. Use OpenCodeInstruct for the first candidate;
  OpenCoder is a separately named fallback if its selected split provides a
  cleaner contamination or license result.
- The exact parent-corpus Math slice and target-regenerated OpenR1-Math
  examples: Math replay candidates kept disjoint from the held-out Math suite.

The first continuation mixture is token-mass based rather than row-count based:

| Component | Initial token share |
| --- | ---: |
| Nebius OpenHands SWE-rebench trajectories | 45% |
| NVIDIA Open-SWE-Traces | 20% |
| SWE-Gym executable tasks | 5% |
| Validated SWE-smith executable tasks | 5% |
| NVIDIA OpenCodeInstruct code SFT | 10% |
| Exact parent-corpus Math replay | 10% |
| Target-regenerated OpenR1-Math replay | 5% |

Cap each repository and issue so a small set of long trajectories cannot
dominate token mass. If Math retention falls below its gate, raise Math replay
to 20%-25% and reduce code SFT before reducing agentic SWE data.

Public assistant turns were generated by heterogeneous teachers such as
Qwen3-Coder, Qwen3.5, Minimax, or other agents. They are not valid drafter
targets for the selected Qwen verifiers. Reuse their task selection, repository
state, tool schema, and provenance, but generate the supervised trajectories
with the exact Q30 or Q235 target, production chat template, tool parser, and
Thinking mode. Agent trajectories must be regenerated by executing the target
inside the environment because tool observations depend on preceding actions;
rewriting assistant text while retaining incompatible observations is invalid.

Separate resolved trajectories for supervised drafter training from failed or
partial trajectories intended for later RL, preference, or verifier work.
Deserialize tool-call arguments into the production JSON representation before
tokenization and preserve tool-call IDs and tool-response relationships.

Create continuation candidates from the best baseline checkpoints with new
optimizer state and low learning rate. Tune the mixture using held-out
acceptance by position while preserving the immutable first-run composition and
giving every retry a new arm identifier.

A SWE continuation checkpoint is promoted only if it achieves all of:

- at least 10% relative improvement in draft-only mean accepted length on the
  held-out NeMo-RL SWE set or a statistically credible equivalent improvement;
- improved SWE generation/model-call time under matched concurrency;
- Math mean accepted length at least 98% of its parent checkpoint;
- no material Math reward, SWE reward, or valid-rollout regression;
- positive NeMo-RL SWE-RL end-to-end step-time benefit.

If Math retention fails, increase exact parent-corpus and Math replay before
changing model architecture. If SWE acceptance improves but end-to-end time does
not, profile draft, verify, environment, and tool overhead before adding more
data.

## Failure Handling

- Never overwrite a completed checkpoint or accepted dataset artifact.
- Treat target-server, capture, data, trainer, OOM, NaN, convergence, export,
  vLLM-load, graph, NeMo-RL, and environment failures as separate classes.
- Preserve the exact failed command, job ID, source and container SHAs, bounded
  error excerpt, root cause, fix, and retry generation.
- A missing metric, timeout, unreachable cluster, or stale scheduler state is
  unknown, never success.
- Checkpoint and save training state on SIGTERM when the framework permits it.

## Testing and Reporting

Local tests cover typed matrix expansion, deterministic naming, manifest hashes,
contamination filters, target/template identity, block/proposal semantics,
launcher rendering, fail-closed scheduler gates, checkpoint tensor contracts,
vLLM config translation, and NeMo-RL override generation.

Cluster tests cover one-sample overfit, sixteen smokes, resume, export/reload,
eager/compiled correctness, graph capture, Math evaluation, generation-only SWE,
and end-to-end NeMo-RL SWE-RL.

Maintain one machine-readable ledger and one user-facing report containing all
sixteen arms, data and source provenance, job transitions, checkpoint inventory,
training curves, position-wise acceptance, Math and SWE results, unsuccessful
attempts, and the exact evidence behind every promotion decision.

## Completion Criteria

Stage 1 is complete when all sixteen arms have a terminal, evidence-backed state;
every successful arm has an immutable exported checkpoint and Math/SWE report;
and at least one DFlash and one DSpark arm per target have passed the standalone
vLLM and NeMo-RL external-drafter gates, or have an explicit technical blocker
with reproduced evidence.

Stage 2 is complete only when a separately named SWE continuation improves
held-out NeMo-RL SWE-RL accepted length and end-to-end time while satisfying the
Math retention and task-quality gates.
