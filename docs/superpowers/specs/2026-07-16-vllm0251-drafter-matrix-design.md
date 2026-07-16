# vLLM 0.25.1 Drafter Matrix Design

## Objective

Measure the best applicable speculative-decoding proposers exposed by official
vLLM 0.25.1 in NeMo-RL performance recipes for Qwen3-30B-A3B, Qwen3-32B, and
Qwen3-235B-A22B. Results must isolate the proposer effect against an exact
baseline and report step 2-20 averages.

## Controlled Recipes

| Model | Recipe | Topology | Max OSL |
|---|---|---:|---:|
| Qwen3-30B-A3B | `grpo-qwen3-30ba3b-4n4g.yaml` | 4 nodes x 4 GPUs | 4096 |
| Qwen3-32B | `grpo-qwen3-32b-4n4g.yaml` | 4 nodes x 4 GPUs | 4096 |
| Qwen3-235B-A22B | `grpo-qwen3-235b-16n4g.yaml` | 16 nodes x 4 GPUs | 8192 |

The recipes remain authoritative for model, tokenizer, batching, placement,
parallelism, MoE backend, dataset, and sampling. The experiment may override
only step count, output paths, logging, checkpoint saving, CUDA Graph mode, and
speculative-decoding configuration. All recipes use temperature 1.0, top-p
1.0, and 32 generations per prompt.

## Applicable Matrix

| Method | Runner | Checkpoint | Candidate settings |
|---|---|---|---|
| Baseline | MRv2 | none | no speculative config |
| EAGLE3 | MRv2 | exact target-specific head | K=1,3,5; promote best |
| Dynamic EAGLE3 | MRv2 | best EAGLE3 head | offline-profiled batch/K schedule |
| DFlash | MRv2 | exact target-specific head | K=3,5; only where checkpoint exists |
| Draft model | MRv1 | AMD PARD Qwen3-0.6B | sequential K=1,5 |
| PARD | MRv1 | AMD PARD Qwen3-0.6B | parallel K=5,16 |
| Suffix | MRv1 | none | maximum tree depth 32 |
| N-gram | MRv1 | none | min=max=5 |
| N-gram GPU | MRv1 | none | min=max=5 |

Officially accepted methods that are not applicable are recorded rather than
silently omitted: native MTP requires target-embedded heads absent from these
three Qwen checkpoints; DSpark and Medusa require exact target-specific
checkpoints; `mlp_speculator` has a vLLM 0.25.1 MRv1 runtime gap;
`extract_hidden_states` and `custom_class` are not acceleration proposers.
PARD-2 and DFlare require non-upstream patches and therefore belong in a
separate experiment.

## Runtime Rules

- Pin official vLLM 0.25.1 and preserve the already validated NeMo-RL upgrade.
- Set `policy.generation.vllm_cfg.enforce_eager=false` for every variant.
- Use native CUDA Graph sizing. Do not apply compact capture-size overrides.
- Use `FULL_AND_PIECEWISE`; record the resolved runner and graph mode from logs.
- Use MRv2 only for EAGLE3 and DFlash. Select MRv1 for draft/PARD, suffix, and
  n-gram because vLLM 0.25.1 rejects them in MRv2.
- Set draft tensor parallelism to one for model-based drafters.
- Disable checkpoint saving and enable W&B logging.
- Derive SLURM nodes, GPUs per node, and `--segment` from the selected recipe.
- Submit without dependencies or singleton constraints.
- Never override `max_num_batched_tokens` or CUDA Graph capture sizes in the
  controlled matrix.

## Execution Gates

1. Resolve and validate the recipe, model/drafter compatibility, snapshot
   existence, runner, and generated Hydra overrides locally.
2. Run scheduler `--test-only` before each submission family.
3. Run a two-step load/config smoke test.
4. Promote clean variants to a five-step performance smoke test.
5. Promote useful or diagnostically necessary variants to 20 steps.
6. Monitor submitted jobs for at least five minutes and classify early exits.

The 20-step comparison excludes step 1. A baseline is reusable only when model,
recipe, vLLM version, container, CUDA Graph mode, sampling, and cluster match.

## Metrics And Reporting

Each final row records E2E step time and throughput, generation time and
throughput, policy-training time, logprob time, generation ratio, acceptance
rate, mean accepted length, resolved runner, resolved CUDA Graph mode and
coverage when available, job ID, source log, and W&B URL. Speedups are ratios
against the exact matched baseline. Failed and unsupported rows retain an
explicit reason.

## Safety And Reproducibility

Every run directory contains the Git commit, submodule SHAs, container path,
recipe path, target and drafter snapshots, generated command, scheduler
arguments, and SpecDec settings. Submission requires a clean pushed branch.
Only the user's fork is writable; NVIDIA upstream remains read-only.
