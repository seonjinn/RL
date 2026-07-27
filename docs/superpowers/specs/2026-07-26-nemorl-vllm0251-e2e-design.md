# NeMo-RL vLLM 0.25.1 E2E Benchmark Design

## Goal

Build a reproducible Lyris experiment that proves NeMo-RL full GRPO runs with
vLLM 0.25.1 in the generation workers and compares matched Qwen3-30B-A3B Math
E2E performance for non-speculative decoding, Eagle3, DFlash, and DSpark.

The first deliverable is an execution package and validation gate. A
performance result is final only after a run completes rollout collection,
policy-to-generation refit, policy training, and the configured GRPO steps.

## Scope

The primary benchmark contract is:

- cluster: Lyris GB200;
- target model: `Qwen/Qwen3-30B-A3B`;
- workload: the existing Qwen3-30B-A3B Math GRPO performance recipe;
- RL mode: synchronous GRPO;
- sampling: `temperature=1.0`, `top_p=1.0`, and `top_k=-1`;
- smoke duration: one completed GRPO step;
- measurement duration: ten completed GRPO steps;
- runtime: exactly `vllm==0.25.1`;
- methods: baseline, Eagle3 K3, DFlash, and DSpark;
- target and policy precisions, node shape, batching, prompt set, maximum
  generation length, and training configuration: identical across all methods.

This experiment does not treat rollout-only, standalone engine, setup-only, or
partially completed jobs as E2E measurements.

## Chosen Architecture

Create a focused package under `experiments/nemorl_vllm0251_e2e/` that wraps
the existing NeMo-RL performance launcher instead of modifying or copying the
historical vLLM 0.20 launcher.

The package owns four responsibilities:

1. a declarative method and benchmark contract;
2. local and remote runtime preflight;
3. gated Lyris submission;
4. normalized result collection and baseline matching.

The wrapper passes only the overrides needed for vLLM 0.25.1, SpecDec, CUDA
Graph coverage, checkpoint selection, and experiment identity. The underlying
NeMo-RL recipe remains the source of truth for the training topology.

## Files and Responsibilities

`experiments/nemorl_vllm0251_e2e/matrix.yaml`

- Defines the immutable benchmark fields and the four method rows.
- Uses K3 for Eagle3.
- Accepts DFlash and DSpark K values from explicit environment variables.
- Records both the smoke and measurement step counts.

`experiments/nemorl_vllm0251_e2e/preflight.py`

- Validates the matrix schema and selected method names.
- Validates the remote repository, commit, container, target model, draft
  checkpoint files, and writable result roots.
- Executes the container's driver Python and the selected generation-worker
  Python to obtain `vllm.__version__`, `vllm.__file__`, Python executable, Ray
  version, and platform architecture.
- Requires both driver and generation-worker probes to resolve
  `vllm==0.25.1`.
- Rejects `/opt/ray_venvs` or another hidden vLLM 0.20 environment winning
  import precedence.
- Produces a JSON provenance record consumed by submission and reporting.

`experiments/nemorl_vllm0251_e2e/submit_lyris.sh`

- Supports `PREFLIGHT_ONLY=true`, `STAGE=smoke`, and `STAGE=measurement`.
- Uses the vLLM 0.25.1-compatible NeMo-RL worktree and
  `nemo_rl_nightly_20260715.sqsh`.
- Requires an explicit `NEMO_RL_VENV_DIR` containing the lock-matched
  vLLM 0.25.1 actor environments.
- Allocates an independent actor-venv/cache suffix for every submitted job.
- Renders exact Hydra overrides for the selected method.
- Writes a job manifest before submission and appends the returned SLURM job
  ID without rewriting existing manifests.
- Submits the measurement stage only when all selected smoke methods have a
  passing gate record.

`experiments/nemorl_vllm0251_e2e/collect_results.py`

- Parses NeMo-RL driver logs using the existing full-GRPO metrics parser.
- Classifies each run as `PASS`, `PARTIAL`, `FAILED`, or
  `BLOCKED_MISSING_CHECKPOINT`.
- Emits step-level CSV, method summary CSV, and a machine-readable JSON report.
- Computes speedups only against the matched baseline from the same stage and
  benchmark contract.

`tests/test_nemorl_vllm0251_e2e.py`

- Covers schema validation, method rendering, version-probe rejection,
  checkpoint gating, capture-size derivation, completion classification, and
  matched-baseline speedup calculation without requiring GPUs or SSH.

## Runtime and Version Contract

The initial remote runtime is:

- container:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh`;
- actor environments:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/nrl_venvs_dynsd025`;
- NeMo-RL worktree: the `nemogym-dynsd` worktree derived from
  `RL-vllm0251-eagle3-fullcg-final-20260715` at `f868c977` with the required
  NeMo-RL integration cherry-picks.

Submission is rejected when the runtime probe cannot prove all of the
following:

- driver-visible vLLM version is `0.25.1`;
- generation-worker-visible vLLM version is `0.25.1`;
- generation workers use the requested actor environment;
- remote worktree commit and dirty state are recorded;
- container and draft checkpoint identities are recorded;
- Python and Ray versions are compatible with the selected worktree.

A container filename or `uv.lock` entry is not accepted as version proof.

## Method Contract

| Method | Runtime method | Initial K | Draft checkpoint |
|---|---|---:|---|
| baseline | none | 0 | none |
| Eagle3 | `eagle3` | 3 | `QWEN30_EAGLE3_DRAFT_MODEL` |
| DFlash | `dflash` | explicit | `QWEN30_DFLASH_DRAFT_MODEL` |
| DSpark | `dspark` | explicit | `QWEN30_DSPARK_DRAFT_MODEL` |

`QWEN30_DFLASH_SPEC_TOKENS` and `QWEN30_DSPARK_SPEC_TOKENS` are required when
their corresponding methods are selected. The launcher does not invent a K
from checkpoint block size.

The DFlash and DSpark checkpoint paths must contain `config.json` and model
weights. The preflight records SHA256 hashes for configuration files and the
size and modification time of weight files. A selected method with no usable
checkpoint is classified as `BLOCKED_MISSING_CHECKPOINT` before `sbatch`.

DSpark remains present in the matrix before its new checkpoint is available;
baseline, Eagle3, and DFlash can run independently by selecting those methods
explicitly.

## CUDA Graph Contract

All methods use `enforce_eager=false`. SpecDec rows use vLLM FULL CUDA Graph
mode with explicit capture sizes that cover the verification batch.

For fixed K and configured `max_num_seqs`, the required maximum verification
size is:

```text
max_num_seqs * (K + 1)
```

The package derives a dense list of `(K + 1)` multiples through that maximum
and includes the baseline-compatible decode capture sizes. It does not use
`max_cudagraph_capture_size`, which triggers the known vLLM 0.25.1
`pydantic_core.ArgsKwargs` pickle failure.

The smoke gate fails on CUDA Graph initialization failure, explicit eager
fallback, or missing speculative metrics for a SpecDec row.

## Execution Gates

1. Run local schema and dry-run tests.
2. Run remote import/version preflight without allocating a training job.
3. Submit a one-step baseline smoke.
4. Submit one-step SpecDec smokes only for methods with valid checkpoints.
5. Require every submitted smoke to show:
   - `SETUP COMPLETE`;
   - generation worker vLLM 0.25.1 provenance;
   - rollout collection completion;
   - policy-to-generation refit completion;
   - policy training completion;
   - one complete GRPO step;
   - no fatal Ray, NCCL, CUDA Graph, or environment error.
6. Submit the ten-step measurement matrix only for methods whose smoke gate
   passed.
7. Monitor every submitted job for at least five minutes and save the observed
   SLURM state and first fatal marker, if any.

The measurement launcher defaults to dry-run and requires `SUBMIT=true`.

## Metrics and Comparison Rules

For each method, aggregate completed measurement steps after excluding step 1
as warmup when at least three steps complete. Report:

- `generation_worker_tokens_per_sec_per_gpu_mean`;
- generation throughput speedup;
- `generation_time_s_mean`;
- generation-time speedup;
- `e2e_tokens_per_sec_per_gpu_mean`;
- E2E throughput speedup;
- `total_step_time_s_mean`;
- E2E step-time speedup;
- acceptance rate and mean accepted length for SpecDec methods;
- completed step span, generated token count, reward, and latest error;
- job ID, worktree commit, container identity, Python/Ray/vLLM provenance,
  checkpoint identity, K, and CUDA Graph capture sizes.

Speedups are emitted only when model, workload, sampling, RL mode, step window,
node/GPU topology, precision, batching, maximum output length, worktree,
container, and vLLM runtime match. Otherwise the speedup field is empty and
the mismatch reason is recorded.

## Failure and Partial-Run Handling

- A job that initializes engines but completes no GRPO step is `PARTIAL`.
- A job that completes rollout but fails refit or policy training is
  `PARTIAL`, not E2E.
- A terminal SLURM failure before setup is `FAILED`.
- A selected missing or incompatible draft checkpoint is
  `BLOCKED_MISSING_CHECKPOINT`.
- A vLLM version or import-path mismatch fails preflight and submits no jobs.
- Metrics from `PARTIAL` and `FAILED` rows remain visible but never contribute
  to speedup calculations.

## Acceptance Criteria

The execution package is ready when:

- local tests and Bash syntax checks pass;
- preflight proves generation-worker vLLM 0.25.1 rather than relying on a
  container label;
- dry-run output is complete and contains no secrets;
- baseline and each available SpecDec method can be submitted independently;
- a missing future DSpark checkpoint is represented as an explicit blocked
  state;
- result collection distinguishes rollout-only evidence from complete E2E
  GRPO evidence;
- a matched ten-step baseline and method pair can produce generation and E2E
  speedups with full provenance.
