# Qwen3-30B-A3B CuTeDSL OCI-HSG 1n4g gate

This experiment is the Linux/GB200 validation and performance path for the CuTeDSL fused grouped-MLP policy-training slice. The reduced two-node gate has completed three synchronous GRPO updates on Pre-Tyche; official-workload performance evidence is still pending.

## Official Qwen3-30B-A3B 4n4g performance path

`submit_nemo2606_4n4g_performance.sh` preserves the official `grpo-qwen3-30ba3b-4n4g.yaml` workload while enabling policy-training MXFP8 and the CuTeDSL prerequisites. It keeps the official BF16 rollout, 64 prompts x 32 generations, GBS2048, MBS1, logprob batch size 2, 4096-token limit, sequence packing, TP1/PP1/CP1/ETP1/EP16, and 4-node x 4-GPU segment. Router precision is fixed to FP32 in both arms because job 2362710 proved that the inherited FP64 router reaches Transformer Engine `Fp8Padding` as an unsupported Double tensor. The matched OFF arm changes only `policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP` from string `"1"` to `"0"`.

The performance contract is three paired replicas with alternating order (`ON/OFF`, `OFF/ON`, `ON/OFF`), five warmup updates, and twenty measured updates per arm. Exactly the first replicate also runs separate two-update Nsight ON/OFF diagnostic arms after its timing arms; profile samples are never included in accepted timing statistics. This designated profile replicate is required by the fail-closed aggregate collector.

Validate scheduler placement without consuming GPUs:

```bash
CUTEDSL_CLUSTER_PROFILE=pre_tyche \
  ./experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_4n4g_performance.sh --test-only
```

Run the fail-closed three-update CuTeDSL-ON functional gate first:

```bash
CUTEDSL_CLUSTER_PROFILE=pre_tyche \
NEMO2606_FUNCTIONAL_GATE=1 \
  ./experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_4n4g_performance.sh
```

After the functional gate passes, submit the default three-replica CuTeDSL OFF/ON timing matrix:

```bash
CUTEDSL_CLUSTER_PROFILE=pre_tyche \
  ./experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_4n4g_performance.sh
```

The manifest records the resolved workload as well as topology and fixed-feature evidence. The primary CuTeDSL endpoint is policy-training tokens/s/GPU; E2E, generation, logprob, refit, and component step times are secondary endpoints. No speedup is reported until all three paired replicas pass exact ON/OFF config-diff validation and the aggregate collector accepts them.

## NeMo 26.06 two-node factorial harness

`submit_nemo2606_2n4g_factorial.sh` adds the portable performance path. It submits the repository `ray.sub` launcher, whose head process starts the Ray GCS, whose second-node task joins as a Ray worker, and whose driver starts only after all eight `worker_units` are visible. The matrix payload switches to existing-Ray mode, so it executes directly in the head container and never starts a nested one-task local Ray cluster. The locked driver environment and result artifacts use the shared repository filesystem, while per-node Ray actor environments use a run-scoped node-local `/tmp` root to avoid concurrent rebuilds of one shared venv.

The fixed workload is Qwen3-30B-A3B on two GB200 nodes with four GPUs per node, TP1/PP1/CP1/ETP1/EP8, GBS16, and MBS1. World-size-derived DP8 therefore receives two local microbatches, the minimum useful combined-1F1B A2A-overlap shape. Each timing arm runs five warmup plus twenty measured updates. The `g0a0` and `g0a1` contexts run alternating CuTeDSL OFF/ON pairs for at least three replicas. The dependency-constrained `g1a0` and `g1a1` contexts run CuTeDSL ON only because device-initiated CuTeDSL is a full-iteration CUDA Graph prerequisite; those cells do not fabricate an invalid OFF result.

Use the same command on Pre-Tyche, Lyris, or AWS-DFW by selecting a cluster profile:

```bash
CUTEDSL_CLUSTER_PROFILE=pre_tyche \
  ./experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh --test-only
```

The runnable default contexts are `g0a0,g0a1`, submitted replica-major with a rotating context order. This measures CuTeDSL within `g0a0` and the A2A bundle incrementally as `g0a0/ON -> g0a1/ON` once the schedule-plan adapter and Bridge/MCore config propagation pass source preflight. Full-CG contexts are opt-in because the current branch lacks the full-iteration implementation, while the available implementation still rejects colocated generation/refit and policy/reference logprob operations. When that lifecycle support exists, `g1a0` and `g1a1` run CuTeDSL ON only; their CuTeDSL-OFF arm is structural `N/A`, since device-initiated CuTeDSL is a full-CG prerequisite. The harness never records a silent no-op or invalid full-CG/OFF result.

Each context writes a separate submission JSONL. For `g0a0` and `g0a1`, run `collect_cutedsl_ab_replicates.py` once per context JSONL against the common `results/` root. The existing collector accepts only a context-local CuTeDSL OFF/ON pair and retains E2E, generation, generation-finalize, logprob, policy-training, refit, and normalized-throughput series. It does not yet accept the ON-only `g1a0`/`g1a1` contract or calculate cross-context full-CG and A2A effects; those aggregates require the planned dependency-constrained collector with independent two-sample bootstrap before any full-CG/A2A speedup claim. The manifest records this limitation explicitly. Profile jobs are separate from accepted timing samples and fail closed unless kernel-presence attribution passes for every available arm. Enabled full-CG contexts additionally require policy-worker `cudaGraphLaunch` presence, and enabled A2A contexts require NCCL A2A kernel presence. These signatures do not by themselves prove full-iteration replay or temporal communication/GEMM overlap, so `feature_attribution.json` keeps those verification fields false until a separate Nsight timeline analysis passes. Raw artifact listings are capped at 200 paths and Nsight reports at 16 per arm.

## Fixed request and runtime

| Item | Value |
|---|---|
| Account | `nemotron_n3_post` |
| Partition | `batch` |
| Allocation | 1 node, `--gres=gpu:4` |
| Expected GPU | 4x GB200 |
| Image | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260711_4677250.sqsh` |
| Image SHA256 | `af1d2ca2a7b169aa13be4b129a0fad8e206c63576d4941b00ae312bd65d0f3e1` |
| Recipe | `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl.yaml` |
| GRPO steps | 3 |
| Profiler | Megatron policy worker, step range `2:3` |
| MXFP8 rollout/refit parity | `max(train/gen_kl_error) < 0.05`; `max(train/token_mult_prob_error) < 2.0` |

The wrapper does not accept a container override. The pinned nightly was staged by job `4677250`. Its baked environment is older than this branch, so the wrapper creates a locked Python 3.13.13 MCore environment in node-local storage before validation. If that compatibility build fails, classify it separately from policy-training failures. Do not overwrite the dated image.

## Preflight and scheduling check

Use a pushed branch on the OCI-HSG checkout. Before asking SLURM to schedule it:

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
git status --short --branch
git submodule status --recursive
git rev-parse '@{upstream}'
git rev-parse HEAD
sbatch --test-only experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh
```

`sbatch --test-only` parses the fixed resource request without executing the wrapper or consuming GPUs. Resolve account, partition, GRES, or feasibility errors before submitting.

The wrapper requires a configured upstream, requires `HEAD == @{upstream}`, rejects tracked changes and non-ignored untracked files, and requires every recursive submodule to be initialized and clean. Push the parent and Bridge branches before this preflight.

The runtime wrapper rejects tracked parent or submodule changes. It then hashes the image, creates a run-local Linux uv environment, and performs these gates in order:

1. Install the locked MCore, test, and dev groups with `uv sync --locked`, then run `uv lock --check`.
2. Record source/upstream, recursive submodule, image, runtime, topology, and resolved-config provenance.
3. Run the official Bridge parameter-mapping/checkpointing suites and the three focused NeMo-RL suites.
4. Run `pyrefly check`, parent pre-commit over the changed parent files, and Bridge pre-commit over the changed Bridge files.
5. Run the required Cutlass DSL `python -c` import.
6. Execute a Transformer Engine `Linear` forward and backward with finite output/input/parameter-gradient checks on each of the four visible GPUs.
7. Run three GRPO steps, persist Ray logs under the result directory, and capture the step-2 Megatron policy worker with Nsight Systems.
8. Export TensorBoard scalars to `metrics.json` and enforce explicit successful optimizer updates, finite loss/gradients, MXFP8 rollout/refit parity, non-empty post-warmup timing/throughput, and `.mem_gb` peak-memory metrics.

These validations are requirements of the job; they are not claimed as passing until the OCI job executes successfully.

## Submit and monitor

```bash
JOB_ID=$(sbatch --parsable experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh)
RUN_DIR="experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results/${JOB_ID}"
echo "${JOB_ID} ${RUN_DIR}"
squeue -j "${JOB_ID}" -o '%.18i %.2t %.10M %.6D %R'
tail -F "${RUN_DIR}/slurm.out"
```

On requeue, the result directory is `$JOB_ID-r$SLURM_RESTART_COUNT`. Discover every attempt safely with:

```bash
find experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results \
  -maxdepth 1 -type d -name "${JOB_ID}*" -print | sort
```

Poll the queue and tail the run log frequently for at least five minutes after the job starts. Model download and kernel compilation can be slow; cancel only for a confirmed unrecoverable error.

## Run artifacts

All generated output stays under `results/$RUN_ID/` (`$JOB_ID` initially or the requeue suffix documented above), which is ignored by Git.

| Artifact | Contents |
|---|---|
| `metadata.json` | Git/submodule SHAs, image SHA256, runtime versions, GPU topology, SLURM allocation, and embedded effective config |
| `effective_config.yaml` | Fully resolved recipe plus wrapper overrides |
| `slurm.out` | Complete wrapper, test, smoke, and GRPO output |
| `uv_lock_check.log` | In-container `uv lock --check` output |
| `focused_tests.log` | Official Bridge and focused NeMo-RL test output |
| `pyrefly.log` | Parent `pyrefly check` output |
| `parent_precommit.log` | Parent pre-commit output over changed parent files |
| `bridge_precommit.log` | Bridge pre-commit output over changed Bridge files |
| `cutlass_import.log` | Cutlass DSL import path |
| `gpu_smoke.log` | Per-device Transformer Engine forward/backward smoke output |
| `grpo.log` | GRPO driver output |
| `ray_tmp/` and `ray_artifacts.txt` | Persistent Ray logs/temp artifacts and their manifest, retained even when GRPO fails |
| `metrics.json` | TensorBoard scalars from the run |
| `metrics_summary.json` | Explicit successful updates, finite checks, parity thresholds/results, timing, throughput, `.mem_gb` peaks, and Nsight file list |
| `kernel_evidence.txt` | `nsys stats` CUDA kernel summary used to identify the fused CuTeDSL kernel |
| `nsight/*.nsys-rep` | Full step-2 policy-worker capture; keep on Lustre and do not commit |
| `status.json` | Wrapper exit code and completion time |

## Classification and acceptance

Classify missing Cutlass DSL, TE, cuDNN frontend, CUDA, binary symbols, container imports/mounts, or insufficient runtime versions as image/runtime failures. Classify config validation, model shape, refit tensor mismatch, loss divergence, and exceptions in the changed Python path as code failures. Use the systematic-debugging workflow for code failures; do not change the image to mask them.

The gate passes only when the wrapper completes and review confirms all of the following:

- At least two `train/optimizer_update_successful == 1` entries were logged; loss-row count is not used as an optimizer-update proxy.
- Megatron-to-HF export and rollout refit completed without shape or layout errors.
- The repository's existing GB200 MXFP8 rollout/refit parity metrics pass: `max(train/gen_kl_error) < 0.05` and `max(train/token_mult_prob_error) < 2.0`. These exact thresholds come from `tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh`; they are proxy metrics for rollout/refit parity, not a claim of elementwise logit equality.
- Loss and gradient norms are finite.
- `kernel_evidence.txt` or the retained Nsight capture identifies the CuTeDSL fused grouped-MLP kernel, not only the generic op-fuser path.
- Non-empty post-warmup policy timing and policy tokens/s series are present, and peak GPU memory is recorded from Ray logger keys ending in `.mem_gb`.

If functional checks pass but kernel evidence is absent, report the result as functionally passing with performance activation unverified. Do not claim a performance win from this three-update smoke; use at least five measured updates or a repeated microbenchmark for a comparison.

## Result record

Status: **Not run**

| Field | Baseline | CuTeDSL |
|---|---|---|
| Job ID | Pending | Pending |
| NeMo-RL SHA | Pending | Pending |
| Megatron-Bridge SHA | Pending | Pending |
| Image SHA256 | Pending | Pending |
| Feature flags | CuTeDSL off / contiguous GLU | MXFP8, grouped GEMM, TE op fuser, GLU interleave 32, `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` |
| Correctness | Pending | Pending |
| Kernel evidence | Pending | Pending |
| Median post-warmup policy step | Pending | Pending |
| Policy tokens/s | Pending | Pending |
| Peak memory | Pending | Pending |
| Direct log path | Pending | Pending |
| Failures/resolutions | None recorded | None recorded |

After the run, replace the pending cells with exact job IDs, SHAs, Lustre log/profile paths, correctness results, kernel evidence, and measurements. Commit only the README and small text/JSON summaries; never commit checkpoints, full Nsight captures, model caches, or large logs.
