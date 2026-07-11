# NeMo-RL SpecDec CUDA Graph Ablation Design

## Objective

Determine whether CUDA graph execution changes the relative performance of
Qwen3-32B speculative decoding in the NeMo-RL performance recipe. Repeat the
current CUDA-graph-enabled cohorts with CUDA graphs disabled while holding all
other workload, model, sampling, and topology settings constant.

## Experiment Matrix

Run all rows for 20 GRPO steps. Compare only rows with the same model runner.

| Runner | Variant | CUDA graph on reference | CUDA graph off repeat |
|---|---|---|---|
| V2 | baseline | `2342623` | required |
| V2 | always-on Eagle-3 K5 | `2342632` | required |
| V2 | tail-gated Eagle-3 K0/5 | `2343185` | required |
| V1 | baseline | `2343202` | required |
| V1 | always-on Eagle-3 K5 | `2343210` | required |
| V1 | native DynamicSD | `2343216` | required |

The tail gate uses threshold 64 and three consecutive qualifying scheduler
observations. Native DynamicSD uses
`[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]`.

## Matched Configuration

Every CUDA-graph-off row preserves the corresponding CUDA-graph-on row's:

- upstream performance recipe;
- Qwen3-32B target and immutable checkpoint revision;
- Eagle-3 drafter and immutable checkpoint revision;
- 4 nodes with 4 GB200 GPUs per node and `segment=4`;
- target TP2, draft TP1, DP8, and generation EP1;
- 64 prompts, 32 generations per prompt, and train GBS 512;
- maximum output length 4096 and engine length 4128;
- temperature 1.0 and top-p 1.0;
- `draft_sample_method=probabilistic` and standard rejection sampling;
- `max_num_batched_tokens=16384`, `max_num_seqs=1024`, and Triton MoE;
- checkpoint saving disabled; and
- W&B timing, throughput, acceptance, and gate telemetry.

The only intended execution difference is CUDA graph use.

## Launcher Contract

Add an explicit launcher setting with two values:

- `CUDA_GRAPH_MODE=on` is the default and preserves current behavior exactly.
- `CUDA_GRAPH_MODE=off` sets `policy.generation.vllm_cfg.enforce_eager=true`,
  omits CUDA graph compilation mode, capture maximum, and capture-size
  overrides, and does not request CUDA graph dispatch metrics.

The off mode must not emit contradictory settings such as
`enforce_eager=true` together with `FULL_AND_PIECEWISE` or explicit capture
sizes.

## Provenance

Extend the submission manifest with explicit fields:

- `cuda_graph_enabled`;
- `enforce_eager`;
- `graph_mode` (`FULL_AND_PIECEWISE`, `PIECEWISE`, or `NONE`);
- CUDA graph request/token coverage and capture sizes, using
  `not_applicable` for graph-off rows.

Use a new experiment root and W&B project for graph-off runs. Submit from a
separate remote checkout pinned to the committed launcher revision. Do not
update the remote checkout used by running graph-on jobs.

## Comparison Rules

Produce two distinct comparisons:

1. Within-mode SpecDec speedup: each SpecDec row versus the baseline with the
   same runner and CUDA graph mode.
2. CUDA graph effect: each graph-off row versus the same variant with graph on.

Never use a V1 baseline for V2, or a graph-on baseline for a graph-off SpecDec
speedup. The CUDA graph effect comparison must match model, runner, recipe,
topology, checkpoint revisions, rollout geometry, sequence lengths, sampling,
and gate policy.

Report Steps 2-20 when complete. Until then, label every average with its exact
included step set.

## Metrics

For each row collect:

- generation time and generation throughput per GPU;
- E2E step time and E2E throughput per GPU;
- policy training and policy/reference-logprob time and throughput;
- acceptance rate and mean accepted length for SpecDec rows;
- active K distribution or tail-gate activation/advance-only ratios;
- reward, mean response length, KL, and logprob parity indicators; and
- CUDA graph dispatch coverage only for graph-on rows.

## Validation

Launcher tests must prove:

1. graph-on dry runs remain byte-for-byte equivalent in their graph settings;
2. graph-off dry runs set `enforce_eager=true` and omit every CUDA graph
   compilation and capture override;
3. graph-off manifests use `NONE` and `not_applicable` consistently;
4. graph-on and graph-off rows cannot be accidentally baseline-matched;
5. the explicit mode rejects unknown values; and
6. submission still uses Lyris `--segment=4` without `--gres`.

Run the focused launcher and summarizer suites before committing. On Lyris,
run scheduler test-only checks before submission and monitor every job for at
least five minutes for configuration errors, OOM, NCCL failures, and missing
W&B telemetry.

## Success Criteria

The experiment is complete when all six graph-off rows finish 20 steps and can
be compared with complete graph-on references. The result must distinguish:

- a CUDA graph interaction, where a variant's relative speed changes
  materially between graph modes; from
- an algorithmic/integration overhead, where the variant remains slower with
  graphs disabled.

No accuracy or distribution conclusion is accepted when reward, response
length, KL, or logprob behavior differs materially between matched rows.
