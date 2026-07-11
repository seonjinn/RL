# NeMo-RL vLLM 0.24 Upgrade Validation

This experiment validates the vLLM 0.24 upgrade with unchanged upstream
NeMo-RL performance recipes on AWS-DFW and Pre-Tyche. Cluster-specific
overrides only set the run length, disable checkpoint writes, preserve CUDA
Graph execution, and separate logs and W&B runs.

| Label | Upstream recipe | Nodes | GPUs/node | Segment | Max sequence |
|---|---|---:|---:|---:|---:|
| qwen30ba3b | `grpo-qwen3-30ba3b-4n4g.yaml` | 4 | 4 | 4 | 4,096 |
| qwen32b | `grpo-qwen3-32b-4n4g.yaml` | 4 | 4 | 4 | 4,096 |
| qwen235b | `grpo-qwen3-235b-16n4g.yaml` | 16 | 4 | 16 | 8,192 |

Run scheduler validation before submission:

```bash
experiments/vllm_024_upgrade/submit_performance_step10.sh test-only all
experiments/vllm_024_upgrade/submit_performance_step10.sh submit all
```

The launcher reads the W&B key from `WANDB_API_KEY` or from the file named by
`WANDB_API_KEY_FILE`. It can also read the private `.netrc` created by the
cluster-setup skill when `WANDB_NETRC_HOME` is set. It never stores the key in
the repository or job logs. Set `USE_GRES=true` on OCI-HSG and AWS-DFW; keep it
false on Pre-Tyche.

## Eagle-3 DynamicSD on AWS-DFW

The DynamicSD cohort compares target-only decoding, fixed Eagle-3 K5/K7/K9, and
Eagle-3 with scheduler-batch-size-dependent K for Qwen3-30B-A3B, Qwen3-32B, and
Qwen3-235B-A22B.
All three variants use the same upstream performance recipe, vLLM 0.24,
`temperature=1.0`, `top_p=1.0`, and PIECEWISE CUDA Graphs. Keep this cohort
separate from historical eager-mode or full-CUDA-graph measurements.

Render all fifteen commands locally:

```bash
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  dry-run all all
```

On AWS-DFW, validate scheduling and run the two-step gate before the full
matrix:

```bash
MAX_STEPS=2 RUN_TAG=vllm024-dynamicsd-smoke-20260707 \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  test-only all all

MAX_STEPS=2 RUN_TAG=vllm024-dynamicsd-smoke-20260707 \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit all all
```

Each smoke must reach Step 2 policy training, expose positive draft and
acceptance counters for both Eagle-3 variants, and exit successfully. Then
submit the 20-step matrix:

```bash
MAX_STEPS=20 RUN_TAG=vllm024-dynamicsd-step20-20260707 \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit all all
```

Collect the completed W&B histories with:

```bash
uv run --no-sync python \
  experiments/vllm_024_upgrade/summarize_eagle3_dynamicsd.py \
  --manifest experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/submissions.tsv \
  --entity nvidia \
  --project nemorl-vllm024-dynamicsd-aws-dfw \
  --output-dir experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/summary
```

Final means use Steps 2-20. Every SpecDec row is matched only to the baseline
with the same model, recipe, graph mode, sampling, topology, container, and
commit. The summary reports generation and E2E time/throughput speedups,
acceptance rate, mean accepted length, job IDs, W&B links, and reward/response
length/KL health gates.

## Qwen3-30B Drafter Distribution Matrix on Lyris

The `qwen30-drafter` long-output profile separates a fixed-target drafter
comparison from matched target/drafter pairs. It uses the 40K performance
recipe topology, a 16,384-token output limit, CUDA Graphs, temperature 1.0,
top-p 1.0, probabilistic draft sampling, and checkpointing disabled.
The baseline, fixed K5, and native DynamicSD arms all use PIECEWISE graph mode
and the same capture-size set. vLLM 0.24 requires PIECEWISE for native
DynamicSD, so this avoids comparing it against a stronger fixed-K graph mode.
The 40K recipe uses TP4 and train micro-batch size 1, so this profile also sets
logprob batch size 1 to satisfy NeMo-RL's TP batch-variant accuracy guard.

The fixed-target cohort keeps `Qwen/Qwen3-30B-A3B` and compares the Base and
Instruct-2507 Red Hat Eagle-3 weights. The Base and Thinking-2507 repositories
have identical model and config blobs, so the Base row also represents the
Thinking-2507 alias without submitting duplicate jobs. The matched-pair
cohort runs Base/Base, Instruct-2507/Instruct-2507, and
Thinking-2507/Thinking-2507. Baselines are emitted once per unique target.
Every identity has a separate experiment root, manifest, cache namespace, and
W&B run group.

Render the eleven-job matrix without submitting:

```bash
MATRIX_SELECTION=qwen30-drafter MAX_STEPS=20 \
experiments/vllm_024_upgrade/submit_long_output_matrix.sh dry-run
```

Run a three-step compatibility gate before the final twenty-step cohort:

```bash
MATRIX_SELECTION=qwen30-drafter MAX_STEPS=3 \
RUN_TAG=vllm024-q30-drafter-osl16k-smoke-20260711 \
experiments/vllm_024_upgrade/submit_long_output_matrix.sh test-only

MATRIX_SELECTION=qwen30-drafter MAX_STEPS=3 \
RUN_TAG=vllm024-q30-drafter-osl16k-smoke-20260711 \
experiments/vllm_024_upgrade/submit_long_output_matrix.sh submit
```

The DynamicSD arm uses vLLM 0.24's native
`num_speculative_tokens_per_batch_size`; it does not patch the scheduler.
Because Qwen3-30B-A3B is MoE, the final schedule must come from GB200 offline
profiling rather than assuming K decreases monotonically with batch size.
For each measured batch-size bucket, select the K that maximizes acceptance
length divided by inter-token latency, then pass the resulting lookup table as
`DYNAMIC_SCHEDULE`. Keep the provisional schedule clearly identified until
that calibration is complete. The 20-step GRPO comparison is operational E2E
evidence because policy weights and sampled trajectories can diverge after
refits; use a frozen-target prompt replay when isolating the scheduler policy.
The launcher rejects a 20-step `submit` unless `DYNAMIC_SCHEDULE` is set
explicitly; short smoke runs may use the provisional default to validate
compatibility and collect scheduler-occupancy evidence.

After the final runs complete, collect all four identity manifests with:

```bash
python experiments/vllm_024_upgrade/summarize_qwen30_drafter_matrix.py \
  --manifest .../16k/base__base/submissions.tsv \
  --manifest .../16k/base__instruct2507/submissions.tsv \
  --manifest .../16k/instruct2507__instruct2507/submissions.tsv \
  --manifest .../16k/thinking2507__thinking2507/submissions.tsv \
  --project nemorl-vllm024-q30-drafter-lyris \
  --output-dir .../summary
```

The collector matches baselines by target revision and matches DynamicSD to
Fixed K5 by target plus drafter revision. It writes deterministic CSV and JSON
with Steps 2-20 timing, throughput, acceptance, W&B provenance, and
reward/response-length/KL health gates.

## Tail-Gated Eagle-3 Matrix on Lyris

The tail-gated matrix keeps the Qwen3-30B-A3B and Qwen3-32B four-node upstream
performance recipes matched. It exposes only three Model Runner V1 arms
(`baseline_v1`, `always_on_v1_k5`, `stock_dynamic_v1`) and four Model Runner V2
arms (`baseline_v2`, `always_on_v2_k5`, `fastrl_threshold_v2_k5`,
`efficient_roofline_v2_k5`). V1 uses PIECEWISE graphs; V2 uses
FULL_AND_PIECEWISE graphs. Do not compare performance across these runner
cohorts.

Render or validate the complete matrix before any submit:

```bash
experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh dry-run all all
experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh test-only all all
```

`all` is fail-closed for Qwen3-30B-A3B V2: it schedules that model's three V1
arms and all seven Qwen3-32B arms, but skips the four Qwen3-30B-A3B V2 arms.
After commit-scoped V2 support evidence exists, select `qwen30ba3b` and one V2
variant explicitly; neither an all-model nor an all-variant request includes
those jobs.

The launcher pins 64 prompts by 32 generations, train GBS 512, 4,096-token
sequence and output limits, a 4,128-token engine limit, and a 16,384-token
scheduler budget. It records runner, graph, gate, calibration, W&B, container,
commit, recipe, and Slurm provenance in the Lustre `submissions.tsv` manifest.
`submit` runs Slurm validation before `sbatch`, requires a clean and already
pushed `sna/` branch, and never pushes a remote itself. It fetches the current
branch from `fork` and requires local `HEAD` to equal
`refs/remotes/fork/<current-branch>` exactly, rejecting ahead, behind,
diverged, fetch-failed, or stale-tracking states.

Set `QWEN30_ROOFLINE_CONFIG` and `QWEN32_ROOFLINE_CONFIG` to separate fitted
JSON files. Before submitting a roofline arm, the launcher verifies the JSON's
model, target TP, draft TP, container path/SHA256, exact target and draft
checkpoint revisions, caller-supplied calibration timestamp, cluster, vLLM
commit, and integer K5 membership. Set the matching per-model
`*_CALIBRATION_TIMESTAMP` and, for Qwen3-30B-A3B, the exact
`QWEN30_TARGET_CHECKPOINT_REVISION`. The fixed-K5 arm additionally requires an
exact positive `calibration.per_gamma["5"]`; nearest-K fallback is rejected. The
launcher validates or creates the exact manifest header before asking Slurm to
create a job.

Create a roofline input from measured component latencies before running the
roofline arm. The CSV must carry a single model/TP/cluster/container identity,
the EfficientRollout constants, and `B,S,K,T_T,T_D,T_V` columns:

```bash
uv run --no-sync python experiments/vllm_024_upgrade/calibrate_tail_gate.py \
  --input /lustre/.../tail_gate_measurements.csv \
  --output-dir /lustre/.../tail_gate_calibrations
```

The CSV also requires immutable `target_checkpoint_revision` and
`draft_checkpoint_revision` commit hashes plus a caller-supplied, timezone-aware
`calibration_timestamp`. The converter preserves that timestamp, writes
deterministic output, fits additive timing residuals independently for every K,
and emits an EfficientRollout-compatible JSON plus SHA256 sidecar for each
calibration identity.

Calibration fails if any fitted per-K residual overhead is non-positive. Such
measurements are inconsistent with the configured roofline constants and must
be corrected instead of clamped into an apparently valid configuration.

## Mini Sync-GRPO Tail-Gate Smoke on Pre-Tyche

Run the short Qwen3-32B validation matrix before the full tail-gated cohort:

```bash
experiments/vllm_024_upgrade/submit_tail_gated_specdec_mini_sync_grpo.sh dry-run
experiments/vllm_024_upgrade/submit_tail_gated_specdec_mini_sync_grpo.sh test-only
experiments/vllm_024_upgrade/submit_tail_gated_specdec_mini_sync_grpo.sh submit
```

The wrapper renders exactly `baseline_v2`, `always_on_v2_k5`, and
`fastrl_threshold_v2_k5` on the Qwen3-32B 4n4g recipe. It defaults to two
steps, 16 prompts by four generations, train GBS 64, 1,024-token output and
sequence limits, a 1,056-token vLLM model limit, four nodes with
`--segment=4`, no GPU GRES, and the staged `nightly-20260705` image. The V2
arms use `FULL_AND_PIECEWISE` CUDA graphs with checkpointing disabled; only
the threshold arm configures the FastRL scheduler at local threshold 4 with
ten consecutive checks. The 64 global rollouts are distributed over DP 8,
giving each local scheduler eight initial requests; the wrapper rejects a
threshold greater than or equal to that local capacity. It sets explicit
Pre-Tyche W&B project/entity defaults and one shared run tag and attempt ID so
all three submitted jobs append to the same manifest. Eagle arms use standard
rejection sampling with explicit
probabilistic draft sampling; baseline arms do not receive a speculative draft
sampling override.

Caller-supplied environment values override these smoke defaults. The main
launcher validates `TAIL_GATE_THRESHOLD` and `TAIL_GATE_CONSECUTIVE_CHECKS` as
positive integers, validates `DRAFT_SAMPLE_METHOD` as `greedy` or
`probabilistic`, and records their resolved values in both the generated
command and submission manifest. The validator and activation chart must read
the local threshold from each manifest row rather than assume a global rollout
count or a fixed threshold.

The manifest also records `run_dir`, the resolved outer Slurm log, the initial
`<job-id>-logs/ray-driver.log` path, the recursively synchronized Ray log
directory, and the serialized `sbatch` launcher command. `run_dir` is the log
attempt root: validation discovers both `<job-id>-logs` and numeric requeue
attempts named `<job-id>-<restart-count>-logs`, scans every attempt in numeric
order, and requires driver and Ray evidence plus
`.ray_logs_final_sync_complete` in the final attempt. `ray.sub` writes that
marker only after the head and every worker acknowledge a synchronous final
log copy. The outer Slurm log alone is not sufficient.

Mini manifests use the strict extended schema, including
`draft_sample_method`, all log paths, `launcher_command`, and the exact
`env ... uv run examples/run_grpo.py ...` command. The production collector's
base schema remains compatible with historical manifests that predate those
mini-only fields. Missing historical draft-method provenance is rendered as
`not_applicable` for baselines and `legacy_unspecified` for SpecDec; it is never
silently mixed with newly recorded greedy or probabilistic cohorts. New mini
submissions must use the current extended header and remain fail-closed.

## Baseline/SpecDec Token and Logprob Parity on Lyris

The parity gate runs Qwen3-32B target-only and Eagle-3 K5 through the NeMo-RL
`VllmGeneration` adapter with vLLM 0.24, standard rejection sampling, PIECEWISE
CUDA Graphs, target TP2, draft TP1, chunked prefill enabled, prefix caching
disabled, and a 16,384-token scheduler budget. Greedy jobs use one sample per
prompt. Sampled jobs use 64 independent samples per prompt at temperature 1.0
and top-p 1.0. Both variants use NeMo-RL's deterministic topology-derived
worker seed; the runner does not inject a second vLLM seed keyword.

Render or validate the four-job matrix before submission:

```bash
experiments/vllm_024_upgrade/submit_generation_parity.sh dry-run all all
experiments/vllm_024_upgrade/submit_generation_parity.sh test-only all all
```

Submit after the launcher commit is present on a remote branch:

```bash
experiments/vllm_024_upgrade/submit_generation_parity.sh submit all all
```

Each batch appends and flushes generated token IDs and chosen-token logprobs to
`samples.jsonl`, so a timeout still leaves valid completed rows. `metadata.json`
records the resolved generation contract, commit, SLURM job ID, throughput, and
SpecDec counters. Reward parity remains a separate matched GRPO smoke gate; the
standalone parity producer intentionally does not synthesize an RL reward.
