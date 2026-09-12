# NVIDIA Technical Blog Post Template

**Working title:** Accelerating Reinforcement Learning Rollouts with DFlash and DSpark in NVIDIA NeMo RL

**JIRA:** TECHBLOG-5698 — TODO: paste the internal JIRA link

**Status:** Internal working draft; not publication-ready

**Last updated:** September 12, 2026

This package separates the public-facing article from the internal evidence and approval record. Text marked **TODO**, **Approval required**, or **Evidence pending** must be resolved before publication. Internal W&B links and experiment identifiers belong only in the review copy.

## Checklist

| Requirement | Information | Status / next action |
|---|---|---|
| JIRA + link | TECHBLOG-5698 | Add the internal JIRA URL. |
| Reviewers | Product: TODO; Campaign: TODO; NeMo RL engineering: TODO; serving/vLLM: TODO; drafter training and checkpoint owner: TODO | Record names only after each reviewer accepts. Alert every contributing team. |
| Claim backup | Claim ledger and evidence matrix appear in the internal appendix below. | Recompute publication numbers from canonical artifacts and send the approved backup bundle to `TechBlog-Editors@exchange.nvidia.com`. |
| Title (H1) | Accelerating Reinforcement Learning Rollouts with DFlash and DSpark in NVIDIA NeMo RL | Confirm product naming and headline with editorial. |
| Featured image | Proposed concept: an RL rollout loop with EAGLE-3, DFlash, and DSpark proposing candidate blocks before one target verifier. | Create a 16:9 asset, at least 1480 × 830 px and under 2 MB. Product and brand review required. |
| Captions | Caption register below covers every planned figure and table. | Replace planned captions with final captions after figures and tables are frozen. |
| Alt text | Alt-text register below covers every planned visual. | Validate against final artwork. |
| CTA | Follow the NeMo RL recipes, download a target-compatible drafter, and reproduce the matched Math or SWE rollout benchmark. | Replace placeholders with public release links. |
| Authors | Seonjin Na — TODO: confirm listing, 3–5 sentence bio, 500 × 500 px headshot, and email. Up to four additional authors: TODO. | Do not add names until authorship and order are approved. |
| Category | Developer Tools & Technologies | Editorial confirmation required. |
| Secondary categories | Agentic AI / Generative AI; AI / Deep Learning | Editorial confirmation required. |
| Industry | Software and Technology / Cross-industry | Select the approved TechBlog taxonomy value. |
| Products | NVIDIA NeMo RL; NVIDIA Model Optimizer; NVIDIA GPUs; vLLM where appropriate | Confirm whether NeMo Framework or NeMo AutoModel should be named. |
| Tags | Agentic AI / Generative AI; Developer Tools & Techniques; Reinforcement Learning; Speculative Decoding; NVIDIA NeMo RL; Performance Optimization; Intermediate Technical; Tutorial | Confirm each tag exists or is used by at least five posts. |
| Social copy | Draft first-person takeaways below. | Dev Social review required. |
| Social asset | Reuse or crop the approved featured image, with readable labels at social size. | Provide final image/video in the required format. |

### Reviewer and approval record

| Area | Suggested reviewer role | Approval | Notes |
|---|---|---|---|
| Product | NeMo RL product owner | TODO | Confirm release version, feature status, and product names. |
| Campaign | Developer marketing campaign owner | TODO | Confirm campaign alignment and publication timing. |
| Engineering | DFlash/DSpark online-training owners | TODO | Confirm architecture, configuration, and support matrix. |
| Serving | vLLM / speculative-decoding owner | TODO | Confirm `K` semantics, CUDA Graph guidance, and runtime claims. |
| Training | Drafter recipe and checkpoint owner | TODO | Confirm data wording, licenses, target compatibility, and model cards. |
| Performance | Benchmark and claims owner | TODO | Sign off on matched configurations, statistics, and claim language. |

### Featured image brief

Create a clean 16:9 hero image titled “Faster RL rollouts with modern speculative drafters.” On the left, show an evolving policy inside an RL loop. In the center, show three compact proposal paths: EAGLE-3 as a short left-to-right chain, DFlash as one parallel token block, and DSpark as a parallel block with a light dependency path. On the right, merge all proposals into a large target verifier and an accepted-token stream. Avoid performance numbers, internal model names, or unreleased version labels in the artwork.

### Caption and alt-text register

| Asset | Caption | Alt text / accessible summary |
|---|---|---|
| Figure 1 | Speculative rollout and online drafter update flow in NeMo RL. A lightweight drafter proposes candidate tokens, the target policy verifies them, and an optional update path refreshes the drafter as the policy changes. | Flow diagram showing prompts entering a drafter, candidate tokens entering a target verifier, accepted tokens returning to an RL rollout, and policy updates optionally triggering drafter training and refit. |
| Figure 2 | Target-only baseline step-time breakdown and the Amdahl-law opportunity from accelerating rollout generation. | Two panels show the measured share of generation, reward, log probability, policy training, refit or synchronization, and other time, followed by the maximum end-to-end speedup if generation becomes 1.5x, 2x, or infinitely fast. |
| Figure 3 | Drafter freshness and update events across RL steps for frozen, always-online, fixed-interval, and adaptive policies. | Time-series panels show acceptance changing as the policy trains, with vertical markers at actual drafter update and refit steps. |
| Figure 4 | NeMo RL data and weight flow for online drafter training. | Packed or padded tokens feed the policy and drafter objectives; synchronized policy and drafter updates flow through refit to rollout workers. |
| Figure 5 | Matched rollout-generation throughput for target-only, DFlash, and DSpark configurations. | Bar chart reporting absolute tokens per second per GPU and speedup against target-only generation, with the step window and hardware in the subtitle. |
| Figure 6 | End-to-end GRPO step-time breakdown. | Stacked bars separate generation, reward, log probability, policy training, drafter training, refit, and other measured time for each configuration. |
| Figure 7 | Online-update cadence trades drafter freshness against training and refit overhead. | Scatter plot places end-to-end throughput on one axis and acceptance on the other for frozen, always-online, fixed-interval, and adaptive policies. |
| Figure 8 | Long-context speculative-decoding benefit by generated-length bin. | Bars compare speedup for responses below 4K, 4–8K, 8–16K, and 16–32K generated tokens, with sample counts and truncation rates. |
| Table 1 | Architectural and runtime differences among EAGLE-3, DFlash, and DSpark. | Accessible summary: the methods share target verification but differ in draft dependency, proposal cost, and scheduling controls. |
| Table 2 | Public drafter training recipes and target-compatibility matrix. | Accessible summary: each row identifies the exact target, drafter, approved data description, training configuration, checkpoint, and model card. |
| Table 3 | Reproducible benchmark configuration for Math GRPO and SWE rollout-only evaluation. | Accessible summary: target, drafter, software revision, sampling, sequence lengths, parallelism, CUDA Graph mode, hardware, and repeat count are fixed per comparison. |
| Table 4 | Generation and end-to-end performance with target-only normalized to 1.00x. | Accessible summary: absolute throughput, latency, acceptance, and speedup are reported separately for rollout generation and the full RL step. |
| Table 5 | Quality and stability checks for each performance result. | Accessible summary: reward, task accuracy, KL, entropy, loss, failure count, truncation, and output-integrity checks prevent invalid speedups from becoming claims. |

### CTA

Follow the NeMo RL speculative-decoding recipes, select a drafter trained for the exact target checkpoint, and reproduce the matched Math GRPO or SWE rollout-only benchmark. The final post will link to release-tagged recipes, public model cards, and a configuration checklist for draft length, CUDA Graph coverage, online-update cadence, sequence packing, context parallelism, and multi-node refit.

### Social copy

1. We show how DFlash and DSpark can reduce the sequential cost of RL rollout generation while preserving target-model verification semantics.
2. We compare frozen, always-online, fixed-interval, and adaptive drafter updates to find when fresher proposals repay their training and refit overhead.
3. We separate rollout-only acceleration from end-to-end GRPO speedup and pair every performance result with reward, KL, entropy, and output-integrity checks.
4. We provide reproducible NeMo RL recipes and target-compatible draft checkpoints for developers evaluating Math and SWE workloads.

### Social asset brief

Use the approved featured image with a reduced label set: “Draft,” “Verify,” “Accept,” and “Update.” Preserve 16:9 composition and ensure the token paths remain legible on mobile. Do not place unapproved speedups or release numbers in the social asset.

### Motivational result to produce first

Run a target-only, no-SpecDec baseline with the same public performance recipe that will anchor the final comparison. Instrument mutually exclusive wall-time buckets for rollout generation, reward and environment work, policy and reference log probability, policy training, refit or synchronization, and remaining measured orchestration. Report the mean share of total step time after warmup, step-to-step variability, the observed generated-token distribution, and three independent repeats.

For a measured generation fraction `p`, show the Amdahl-law upper bound when rollout generation alone is accelerated by `s`:

`maximum end-to-end speedup = 1 / ((1 - p) + p / s)`

This figure should include Math GRPO and a genuinely long-output cohort. SWE rollout-only is useful evidence for generation behavior but cannot establish the share of a full RL step because the harness intentionally excludes training. Do not stack nested timers or compute “other” by subtracting overlapping ranges.

## Draft article

# Accelerating Reinforcement Learning Rollouts with DFlash and DSpark in NVIDIA NeMo RL

Large language model post-training increasingly relies on long reasoning traces and multi-turn agent trajectories. In these workloads, reinforcement learning (RL) systems can spend a substantial part of every step generating responses rather than updating parameters. Speculative decoding can reduce this cost: a lightweight drafter proposes future tokens, and the target policy verifies those candidates in parallel.

RL makes this optimization unusually dynamic. The target policy changes during training, so a drafter aligned to an earlier checkpoint can become stale. Updating the drafter can recover acceptance, but drafter training, synchronization, and refit also take time. The useful systems question is therefore not “Does acceptance increase?” It is “Does saved rollout time exceed the update cost while task quality remains stable?”

This post explains how a NeMo RL workflow can extend from EAGLE-3 to DFlash and DSpark, train target-compatible draft checkpoints, keep them aligned with an evolving policy, and measure both generation and end-to-end RL performance on Math and SWE workloads.

### Why rollout generation becomes the bottleneck

A GRPO step combines rollout generation, reward computation, policy and reference log probabilities, optimization, and weight refit. Math reasoning can emit thousands of tokens. Software-engineering agents can produce variable-length conversations with tool calls and a long completion tail. Because synchronous training waits for the rollout cohort, a few slow trajectories can extend the whole step.

Speculative decoding changes the decode loop:

1. A smaller drafter proposes a block of candidate tokens.
2. The target policy verifies the proposed positions together.
3. The serving runtime accepts the valid prefix and discards proposals after the first rejection.
4. Generation resumes from the last accepted token.

Target verification provides the correctness boundary. The drafter changes proposal efficiency, not the target distribution under the serving runtime’s normal verification semantics. Realized speedup depends on the proposal cost, accepted length, verification cost, active batch size, output length, and whether the relevant CUDA Graph shapes were captured.

The first result should quantify the opportunity before presenting any drafter. For the matched target-only baseline, measure the fraction `p` of end-to-end step time spent in rollout generation. Then show the Amdahl-law ceiling `1 / ((1 - p) + p / s)` for 1.5x and 2x generation acceleration. This prevents a large generation-only speedup from being mistaken for an equally large training speedup and explains why longer-output cohorts can have more headroom.

### Three proposal structures, one verifier

EAGLE-3, DFlash, and DSpark share the outer draft-and-verify loop but construct candidates differently.

| Method | Draft dependency | System opportunity | Qualification question |
|---|---|---|---|
| EAGLE-3 | Deeper draft candidates are constructed autoregressively from target features. | Mature high-acceptance path and an established online-training lifecycle. | How quickly does proposal latency grow with depth? |
| DFlash | A lightweight block-diffusion drafter predicts multiple positions in parallel. | Shorter proposal critical path as the draft block grows. | Does later-position quality remain high enough for the target and workload? |
| DSpark | A parallel backbone adds lightweight dependencies inside the proposed block. | Balance parallel proposal latency with stronger intra-block consistency. | Do added dependencies and scheduling controls repay their runtime cost? |

Calling EAGLE-3 “sequential” and DFlash or DSpark “parallel” is useful only when the boundary is explicit. The distinction concerns how deeper draft candidates are produced. Target verification still evaluates candidate positions in parallel.

The configured speculative-token value `K` also requires care. A recipe label does not prove that two runtimes propose or verify the same number of positions. The benchmark must record proposed draft tokens, verified positions, any bonus target token, and the resulting CUDA Graph shape before comparing equal-looking `K` values.

### Figure 1. Speculative rollout and online update flow

The final artwork will show a prompt entering a target-compatible drafter, a candidate block entering the target verifier, and accepted tokens returning to the RL trajectory. A second loop will show the evolving policy producing an optional drafter update and synchronized refit. The figure must distinguish work performed every rollout from work performed only at the selected update cadence.

### Why RL needs a drafter-update policy

A frozen drafter is trained against one target checkpoint. RL changes that target. As token probabilities and target features drift, the fixed drafter can produce fewer useful candidates. Online training can restore alignment, but updating every step may cost more than the extra accepted tokens save.

NeMo RL therefore needs a shared lifecycle with method-specific objectives:

1. The policy forward pass exposes the target signals required by the selected drafter.
2. The policy updates every training step, while the drafter updates only when its cadence is eligible.
3. Distributed workers make the same update decision at a synchronized boundary.
4. Refit installs a compatible target-and-drafter pair on every rollout worker.
5. The next rollout records accepted length, per-position acceptance, proposal time, and verification time.

The invariant is simple: no rollout worker should serve a partially refreshed target-drafter pair.

| Policy | Trigger | Benefit | Cost or risk |
|---|---|---|---|
| Frozen | Never | No online-training or draft-refit overhead. | Acceptance can decay as the policy changes. |
| Always-online | Every eligible policy step | Maximum drafter freshness. | Pays update and refit cost every step. |
| Fixed interval | Every `N` eligible steps | Amortizes update overhead. | May update too soon or wait too long. |
| Adaptive | A rolling benefit or acceptance signal with hysteresis and cooldown | Spends work when the drafter is likely stale. | A noisy gate can update too aggressively. |

An adaptive controller should not react to one batch. It needs a minimum sample count, a rolling statistic, separate trigger and recovery thresholds, a cooldown, and a cost model. The useful condition is:

`expected future rollout time saved > drafter training + synchronization + refit time`

### Training and exporting target-compatible drafters

The drafter must match the exact target family and checkpoint behavior. A drafter trained for a thinking target should not be presented as interchangeable with a base target only because the architecture and tokenizer appear similar.

The public recipe should specify:

- Exact target and tokenizer revisions.
- Approved dataset description, license, sampling proportions, and sequence-length distribution.
- DFlash block geometry or DSpark dependency configuration.
- Target feature taps and the drafter objective.
- Optimizer, learning-rate schedule, precision, batch size, steps, and hardware topology.
- Export format, weight names, checksums, supported runtime revision, and model card.
- Held-out draft loss, acceptance by token position, mean accepted length, and target-verification tests.

Public checkpoint URLs remain TODO until release approval. Internal filesystem paths and checkpoint nicknames must not enter the article.

### Integrating online training into NeMo RL

Fixed speculative decoding requires a target-compatible checkpoint and a speculative-token budget. Online training adds token-layout-aware objective construction, distributed parameter updates, and repeated target-plus-drafter refit.

Sequence packing changes example boundaries and loss normalization. Context parallelism shards long sequences across ranks. Multi-node execution adds collective ordering and recovery requirements. These configurations should appear in the public support matrix only after each method completes the exact end-to-end path, including repeated online updates and refits.

A publication recipe can use placeholders until the final model cards are approved:

```yaml
policy:
  draft:
    enabled: true
    update_policy: fixed_interval
    update_interval: 10
  generation:
    vllm_kwargs:
      speculative_config:
        method: dspark
        model: YOUR_TARGET_COMPATIBLE_DRAFTER
        num_speculative_tokens: 5
```

The final snippet must be copied from the release-tagged schema. Field names in this working example are illustrative, not a compatibility promise.

### Measure generation and the entire RL step

Rollout-only and end-to-end results answer different questions.

A rollout-only benchmark isolates target-only versus speculative generation. It should report generated tokens per second per GPU, generation time, mean accepted length, per-position acceptance, active batch size, output-length distribution, failure count, and repeat variability.

An end-to-end GRPO benchmark should additionally report reward time, policy and reference log-probability time, policy training, drafter training, refit, and other explicitly instrumented time. It should pair performance with reward, task accuracy, KL, entropy, loss, truncation, and output-integrity checks.

SWE rollout-only results demonstrate generation behavior on agent-shaped trajectories. They do not demonstrate end-to-end SWE RL speedup or patch correctness. Similarly, a 32K maximum output length is not a 32K workload unless the observed generation-length distribution contains enough long responses. Long-context results should be binned into `<4K`, `4–8K`, `8–16K`, and `16–32K` generated tokens.

### Does it work?

The current evidence is promising but not yet a publication claim. In one historical internal Qwen3-30B-A3B Math cohort averaged over steps 3–20, target-only generation measured 687.9 tokens/s/GPU. DFlash K5 and K7 measured 1,111.4 and 1,120.2 tokens/s/GPU, or 1.616x and 1.628x. DSpark K5 and K7 measured 1,074.6 and 1,081.3 tokens/s/GPU, or 1.562x and 1.572x.

Those numbers demonstrate component-level potential. They do not yet prove release-level end-to-end speedup because the historical drafter lineage and configuration must be matched to the final recipe, and quality, repeatability, CUDA Graph coverage, and full step-time breakdown still require sign-off. The publication table should be regenerated from the final target-only baseline and matched draft runs over one fixed window.

| Workload | Configuration | Generation TPS/GPU | Baseline ratio | Publication status |
|---|---|---:|---:|---|
| Math, historical cohort | Target-only | 687.9 | 1.000x | Internal reference only |
| Math, historical cohort | DFlash K5 | 1,111.4 | 1.616x (+61.6%) | Reproduce in release cohort |
| Math, historical cohort | DFlash K7 | 1,120.2 | 1.628x (+62.8%) | Reproduce in release cohort |
| Math, historical cohort | DSpark K5 | 1,074.6 | 1.562x (+56.2%) | Reproduce in release cohort |
| Math, historical cohort | DSpark K7 | 1,081.3 | 1.572x (+57.2%) | Reproduce in release cohort |

The headline result must present rollout generation and end-to-end GRPO separately. If generation accelerates but policy training, refit, or synchronization absorbs the gain, the article must say so directly.

### What developers will receive

The publication should link to:

- Release-tagged NeMo RL configurations for EAGLE-3, DFlash, and DSpark.
- Frozen, always-online, fixed-interval, and adaptive update examples that have passed their stated support matrix.
- Reproducible Math GRPO and SWE rollout-only launchers.
- Public draft checkpoints with exact target compatibility and model cards.
- A troubleshooting guide covering `K`, CUDA Graph capture, sequence packing, context parallelism, multi-node refit, and long-tail rollout analysis.

Modern drafters can make rollout generation faster, but acceptance alone is not the finish line. The production decision should use matched end-to-end performance, stable task metrics, and an update policy whose cost is lower than the rollout time it saves.

### Learn more

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)
- [NeMo RL EAGLE-3 speculative-decoding guide](https://docs.nvidia.com/nemo/rl/nightly/guides/eagle3-speculative-decoding.html)
- [Commit-pinned vLLM Dynamic Speculative Decoding reference](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/docs/features/speculative_decoding/dynamic_speculative_decoding.md#L14)

## Internal review appendix — remove before publication

### Claim-backup ledger

| ID | Proposed claim | Required canonical evidence | Current state | Sign-off |
|---|---|---|---|---|
| C1 | NeMo RL supports the stated EAGLE-3 lifecycle. | Release-tagged NeMo RL docs and code permalinks. | Public nightly guide identified; release version wording pending. | Product + engineering |
| C2 | The next release supports DFlash and DSpark fixed and online training. | Merged PRs, release tag, user docs, support matrix, and repeated end-to-end tests. | Approval required; keep “proposed” until merged. | Product + engineering |
| C3 | Public DFlash and DSpark checkpoints are available for named Qwen3 targets. | Approved Hugging Face model cards, licenses, hashes, and target-compatibility tests. | Evidence pending. | Model owner + legal/product |
| C4 | Speculative decoding improves Math rollout generation by a stated ratio. | Matched target-only and draft runs; same software, hardware, prompts, sampling, topology, output window, and quality checks. | Historical internal cohort exists; release cohort pending. | Performance + product |
| C5 | Faster generation improves end-to-end GRPO. | Complete step-time breakdown with repeated runs and finite reward/KL/entropy/loss. | Evidence pending; do not infer from generation TPS. | Performance + RL owner |
| C6 | Online updates outperform a frozen drafter. | Matched frozen/always/interval/adaptive runs from the same initial drafter and target, with actual update events and overhead. | Study in progress; unequal or crashed windows cannot rank policies. | Training + performance |
| C7 | SWE rollout-only improves by a stated ratio. | Matched trajectory set, completion criteria, failure/timeout counts, latency distribution, makespan, and output checks. | Evidence pending. | SWE harness + performance |
| C8 | Long-context generation benefits up to 32K. | Length-binned results with counts, actual output lengths, truncation, active batch, and KV-cache settings. | Evidence pending. | Performance |
| C9 | Sequence packing, CP>1, and multi-node online refit are supported. | Method-specific repeated-update tests for each claimed topology. | Do not claim until completed. | Distributed/runtime owners |
| C10 | Rollout generation is a dominant part of the selected RL workload. | Target-only baseline with non-overlapping stage timers, warmup exclusion, generated-token distribution, repeat variability, and Amdahl calculation. | Evidence pending; rollout-only harnesses cannot prove full-step share. | Performance + RL owner |

### Internal evidence index

These links are for reviewer access and must be removed from the external draft.

- W&B report: [Qwen3-8B and Qwen3-30B-A3B Speculative Decoding Study](https://wandb.ai/nvidia/sna-specdec/reports/Qwen3-8B-and-Qwen3-30B-A3B-Speculative-Decoding-Study--VmlldzoxNzgwNjYyNQ/edit?draftId=VmlldzoxNzgwNjYyNQ==)
- Historical Qwen3-30B-A3B DFlash K5 run: [W&B run](https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dflash-k5-lyris14500-be73db5620ed42d0a94a140ee278c719?nw=nwuserseonjin)
- Historical Qwen3-30B-A3B DFlash K7 run: [W&B run](https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dflash-k7-lyris14500-c630b91c18434e3bb9af2c6a61a2d107?nw=nwuserseonjin)
- Historical Qwen3-30B-A3B DSpark K5 run: [W&B run](https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dspark-k5-lyris14500-dc7cc1e70d0c4cab92f9f40bb57c9d07?nw=nwuserseonjin)
- Historical Qwen3-30B-A3B DSpark K7 run: [W&B run](https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dspark-k7-lyris14500-4ece4886d4e545aab96ed09b561301eb?nw=nwuserseonjin)

### Final publication gate

- Resolve every TODO and record named approvals.
- Freeze the exact release tag, model cards, dataset wording, and support matrix.
- Recompute all numbers from canonical exports with a single declared averaging window.
- Report hardware, topology, software revisions, samples, generated tokens, repeat count, and variability.
- Verify reward, accuracy, KL, entropy, loss, failures, truncation, and output integrity.
- Separate generation-only, rollout-only, and end-to-end claims.
- Replace internal evidence links with public artifacts or approved claim-backup references.
- Remove this appendix, internal links, draft-status language, and unapproved roadmap wording.
- Complete editorial, legal, brand, accessibility, product, and partner review.
