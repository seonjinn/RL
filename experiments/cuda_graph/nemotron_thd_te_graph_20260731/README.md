# Nemotron packed-THD Transformer Engine CUDA Graph study

This directory is the persistent launcher, preflight, collection, and static
report surface for the 2026-07-31 Nemotron CUDA Graph study. It covers
Nemotron 3 Nano, Super, and Ultra plus the Qwen3-30B-A3B and
Qwen3-235B-A22B comparison selectors.
Every training launch disables checkpoints, uses exactly three successful
optimizer warmups, writes to W&B project `sna-cg-study`, and uses only the
current `cuda_graph_modules` and `thd_max_packed_sequences` configuration
fields.

No file here contains credentials. Profile files with real cluster paths and
source revisions remain local under `profiles/*.env`.

## Implementation explainer

`results/cudagraph_implementation_explainer.html` explains the NeMo-RL,
Megatron-Core, and Transformer Engine changes with diagrams, measured
performance and graph coverage, current problems, and an interactive quiz.
Its editorial content lives in `explainer_context.json`; performance,
telemetry, and correctness values are derived from the canonical persistent-bank
CSV files. Regenerate it after either code context or evidence changes:

```bash
python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_explainer.py
```

The explainer and `results/report.html` link to one another. Commit the context,
measured CSV changes, and regenerated HTML together so the explanation stays
reviewable.

## Current status

The completed Nano evidence below used outer NeMo-RL revision
`e30b4bb810356934893d5b4e2b807b5518f17b94`, with Bridge revision
`69c29747e85328d7a5ba39f8cbea844d60314b11`, MCore revision
`5d320e339003f5c2820b1ca0a163e1ca44dfb31e`, and Transformer Engine revision
`04a76c84423d9a4eb2f2010ef6692e347326cc00`
(`2.19.0.dev0+04a76c84`). The paired OCI-HSG run used the 2026-08-01 nightly
container with SHA256
`f863be73380afea5c545614612bcec9a38c9f59be54e88d9431fda4acba717aa`.

The current, not-yet-submitted Qwen campaign has merged upstream main through
`55296257c96d49cd95c7d77613cb0f36bd4a4dc7` and includes the review
remediation implementation through `75ddbef3d`. It preserves Bridge revision
`69c29747e85328d7a5ba39f8cbea844d60314b11` and MCore revision
`5d320e339003f5c2820b1ca0a163e1ca44dfb31e`. A new source snapshot and runtime
attestation must establish the exact final outer commit and container identity
before any Qwen GPU allocation; the historical Nano attestation is not reused.

The first paired 20-step Nano result is complete:

- no-CG baseline job `5784682` and TE partial-CG job `5784680` both completed
  with exit code `0:0`;
- the graph scope was `attn,mamba,moe_router`, with all-to-all dispatch,
  sequence packing, fused attention, and exactly three successful optimizer
  warmups;
- steps 11 through 19 reported 3,224 graph calls out of 3,224 eligible calls,
  zero fallbacks, and 99.89% aggregate padding utilization;
- E2E throughput regressed 4.81%, while policy-training throughput regressed
  50.73% relative to the baseline; and
- correctness is not cleared: step 18 reported
  `token_mult_prob_error=320.149` and `policy_kl_error=319.112`, caused by one
  rollout/eager-logprob mismatch. The configured sequence logprob error
  threshold was null, so the sequence was not masked.

The confirmed performance issue is lifecycle-level: entering the reference
model weight-swap context resets every cached TE graph bank. The measured
window therefore captured one bank per optimizer step and recorded zero cache
hits. The step-18 numerical outlier was present between vLLM rollout logprobs
and eager Megatron logprobs before that step's training replay. A fixed-rollout
eager-versus-graph output and gradient parity gate is required to determine
whether it is a downstream effect of earlier graphed updates or an independent
vLLM/Megatron mismatch.

The current HTML report and machine-readable summary are
`results/report.html` and `results/paired_20step_summary.json`. These files are
ignored by default; publish them only with an explicit reviewed force-add.

Nano HybridEP `moe_preprocess` rows remain fail-closed, while Super may run its
validated preprocess rows. Qwen uses only the A/B/C/E campaign matrix below;
Router Replay combined with `moe_router` or `moe_preprocess` remains
fail-closed. Qwen dense `mlp` and `mamba` rows are model-incompatible. Ultra remains
dependency-blocked until its external model, data, judge, and launch-profile
paths are supplied.

## Required gate order

Run these gates before submitting a model scope. The relative commands below
assume the current directory is
`experiments/cuda_graph/nemotron_thd_te_graph_20260731`; every quoted
`__REQUIRED_*__` value must be replaced before execution.

1. Stage a digest-pinned nightly container. The staging job publishes an
   immutable squashfs, SHA256 metadata, and a current symlink atomically.

   ```bash
   SOURCE_IMAGE=nvcr.io/nvidia/nemo-rl:nightly \
   SOURCE_DIGEST='sha256:__REQUIRED_REGISTRY_DIGEST__' \
   SOURCE_COMMIT='__REQUIRED_FULL_NEMORL_COMMIT__' \
   OUTPUT_PREFIX=nemo_rl_nightly_20260731 \
   CONTAINER_DIR=/absolute/shared/containers \
   scripts/stage_enroot_image.sbatch
   ```

2. Commit the outer repository and both nested gitlinks, then create one clean,
   detached source snapshot. The snapshot records the exact three commits and
   `uv.lock` SHA256 and is the only checkout used by queued jobs.

   ```bash
   SOURCE_ROOT=/absolute/clean/nemo-rl \
   SNAPSHOT_STORE=/absolute/shared/source-snapshots \
   EXPECTED_NEMORL_SHA='__REQUIRED_FULL_NEMORL_COMMIT__' \
   EXPECTED_BRIDGE_SHA='__REQUIRED_FULL_BRIDGE_COMMIT__' \
   EXPECTED_MCORE_SHA='__REQUIRED_FULL_MCORE_COMMIT__' \
   scripts/create_source_snapshot.sh
   ```

3. Build and attest one content-addressed runtime in two phases. The CPU-only
   `stage` phase copies the exact source snapshot into
   `ARTIFACT_DIR/staged-runtimes/<stage-key>`, builds the locked editable
   environment, runs the pinned test suite, removes build caches, and makes the
   complete tree read-only before atomically publishing its marker. Its uv is
   `staged-runtimes/<stage-key>/uv/uv`. The later `attest` phase is the first
   GPU allocation: it requires the stage job to be `COMPLETED|0:0`, verifies
   the marker and read-only tree, requires exactly four visible devices, and
   writes the immutable runtime JSON. It imports PyTorch, Transformer Engine,
   Megatron Core, Megatron Bridge, Mamba SSM, causal-conv1d, and CuPy, and
   validates the MCore TE grouped-linear symbols in a machine-readable success
   or failure artifact. Both phases bind the image, source
   commits, lock digest, TE source/version-base commits, feature set, package
   exclusions, Python version, uv version, and CUDA architectures.

   ```bash
   CONTAINER=/absolute/shared/containers/nemo_rl_nightly.sqsh \
   CONTAINER_SHA256='__REQUIRED_64_LOWERCASE_HEX__' \
   ARTIFACT_DIR=/absolute/shared/artifacts/container-runtime \
   PROJECT_ROOT=/absolute/shared/source-snapshots/__REQUIRED_SNAPSHOT__ \
   EXPECTED_NEMORL_SHA='__REQUIRED_FULL_NEMORL_COMMIT__' \
   EXPECTED_BRIDGE_SHA='__REQUIRED_FULL_BRIDGE_COMMIT__' \
   EXPECTED_MCORE_SHA='__REQUIRED_FULL_MCORE_COMMIT__' \
   EXPECTED_TE_SHA='__REQUIRED_FULL_TE_COMMIT__' \
   EXPECTED_TE_VERSION_BASE_SHA='__REQUIRED_FULL_TE_VERSION_BASE_COMMIT__' \
   RUNTIME_FEATURE_SET=dropless_hybridep_nano16 \
   RUNTIME_EXCLUDED_PACKAGES=fast-hadamard-transform \
   RUNTIME_STAGE_CAPABILITY=mcore-test-v1 \
   RUNTIME_PHASE=stage \
   scripts/validate_oci_container_runtime.sub
   ```

   After the stage job and marker pass, run the same command with
   `RUNTIME_PHASE=attest` and `RUNTIME_STAGE_JOB_ID=<completed-stage-job-id>`.
   Use `SBATCH_GRES=none` on ptyche; the wrapper still verifies four visible
   devices without adding unsupported GPU TRES options.

4. Optionally bootstrap a second fresh Bridge checkout in the immutable
   container, verify the exact nested MCore commit, relock only
   fast-hadamard-transform with Python 3.12, and run the Nano, Super, and Ultra
   recipe tests.

   ```bash
   BRIDGE_REPOSITORY=git@github.com:organization/Megatron-Bridge.git \
   EXPECTED_BRIDGE_SHA='__REQUIRED_FULL_BRIDGE_COMMIT__' \
   EXPECTED_MCORE_SHA='__REQUIRED_FULL_MCORE_COMMIT__' \
   CONTAINER=/absolute/shared/containers/nemo_rl_nightly.sqsh \
   CONTAINER_SHA256='__REQUIRED_64_LOWERCASE_HEX__' \
   ARTIFACT_DIR=/absolute/shared/artifacts/bridge-bootstrap \
   scripts/validate_oci_bridge_bootstrap.sub
   ```

   The submit wrappers export absolute persistent helper paths. This is
   required because Slurm executes a copied wrapper from its spool directory.

5. Populate a local profile from `profiles/*.env.example`. Set
   `RUNTIME_PREFLIGHT_JOB_ID` to the successful preflight job and
   `RUNTIME_ATTESTATION` to its exact non-symlink JSON artifact. Set
   `UV_EXECUTABLE` to the artifact's exact immutable `uv_executable` value;
   the launcher forwards it back to the full attestation verifier instead of
   reconstructing a historical per-job uv path. The verifier also requires
   the JSON's producer job ID to equal `RUNTIME_PREFLIGHT_JOB_ID`. The completed
   immutable attestation is verified directly by every dependency-free leaf,
   which validates exact source, lock, image identity, device count, package
   set, TE commit, Python version, managed interpreter path, interpreter
   SHA256, uv version, uv path, and uv SHA256 before starting Ray. Each NeMo-RL leaf
   derives `UV_PYTHON_INSTALL_DIR` from the immutable attestation directory,
   requires that path to be container-mounted, forces uv-managed Python with
   downloads disabled, and gives the NeMo-RL driver a fresh per-job
   `UV_PROJECT_ENVIRONMENT`. Typed MCore standalone leaves instead execute the
   attested staged Python directly, so GPU nodes never fetch or rebuild locked
   dependencies. The wrapper passes the explicit UV allowlist and
   an attested `CONTAINER_PATH_PREFIX` through Pyxis `--container-env`, then
   prepends that one directory inside each Ray head, worker, and standalone
   MCore container. It deliberately does not import the host `PATH`: preserving
   the image's CUDA, MPI, UCX, and EFA paths avoids replacing runtime libraries
   while still selecting the attested uv ahead of the image's older uv. Leaf
   jobs verify the uv SHA256 before any uv execution and do not rehash the
   image.

   The selected profile must be an absolute direct child of `profiles/`, a
   regular non-symlink UTF-8 file, and contain only literal allowlisted
   `NAME=value` assignments. Replace the example `CONTAINER` path as well as
   every `__REQUIRED_*__` value; a dated example image path is not an attested
   production image.

`validate_te_runtime.py` remains an offline provenance utility. Production
leaf jobs use `verify_runtime_attestation.py`, which requires exact equality
rather than accepting any TE descendant.

All production jobs use the `batch` partition and no singleton dependency.

## Scope matrix

The 33 numbered files in `scopes/` are one no-CG baseline plus the 32-product
matrix formed by:

- every subset of `attn`, `mlp`, and `mamba`; and
- `none`, `moe`, `moe_router`, or `moe_router,moe_preprocess`.

The nine files in `variants/` vary shared-expert overlap and `moe_act`
recomputation without pretending those settings are graph scope names.
`scope_matrix.py` classifies each row before any scheduler call as `runnable`,
`model-incompatible`, `capacity-blocked`, `dependency-blocked`, or
`submitted`. Full-MoE and whole-layer capture remain capacity-blocked until a
fixed drop-and-pad expert capacity is explicitly verified. Ultra remains
dependency-blocked until its external model, data, judge, and launch-profile
paths are supplied. MCore mode remains dependency-blocked until `MCORE_DRIVER`
names a genuine standalone training driver.

Inspect the matrix without importing the project environment:

```bash
python3 scope_matrix.py list
python3 scope_matrix.py classify \
  --model nano --scope attn,moe_router --mode nemorl
```

## Launching

Use a persistent leaf rather than invoking `run_scope.sh` directly. A dry run
does not create directories or contact Slurm:

```bash
CLUSTER=oci-hsg MODEL=nano MODE=nemorl STEPS=20 TEST_ONLY=1 \
  scopes/19_attn_moe_router.sh
```

`TEST_ONLY=1` renders the command without creating a run directory or
contacting Slurm. `SBATCH_TEST_ONLY=1` invokes `sbatch --test-only` to query
scheduler acceptance but creates no job and publishes no metadata. Both
controls accept only `0|1` and are mutually exclusive.

For a real launch, create `profiles/oci-hsg.env` from the example, replace
the example container and every `__REQUIRED_*__` value, remove `TEST_ONLY`,
and keep the source trees at the declared commits. Logs are written below
`exp_logs/nemotron_thd_te_graph_20260731/<run-name>/`.

The reusable phase launchers pin the supported run lengths:

```bash
CLUSTER=oci-hsg MODEL=nano submit_smoke_matrix.sh
CLUSTER=oci-hsg MODEL=nano submit_performance_matrix.sh
CLUSTER=oci-hsg MODEL=nano submit_accuracy_soak.sh
```

- smoke: 5 optimizer steps across all persistent scope leaves;
- performance: 20 steps across baseline and attribution scopes; and
- accuracy: 100 steps, paired baseline and the model's best currently runnable
  combined scope, with three matched repeats.

These generic performance and accuracy launchers accept only Nano, Super, and
Ultra. Qwen selectors fail locally before any launch output and must use
`submit_qwen_router_validation.sh`, which enforces arm identity and campaign
evidence.

## Local export and collection

`export_tensorboard.py` accepts one or more local event paths and atomically
exports a complete 5-, 20-, or 100-step JSONL file. It rejects missing or
non-finite required tags without replacing a previous good artifact. Duplicate
events at one step use the newest TensorBoard wall time. The canonical metric
name is `train/token_mult_prob_error`. New exports require Router Replay
identity from strict `run-metadata.env` or an explicit `--router-replay`; when
both are supplied, every overlapping identity and provenance field must match
before metrics are read.

```bash
uv run export_tensorboard.py \
  --event /shared/run/events \
  --run-metadata /shared/run/run-metadata.env \
  --router-replay off --status passed \
  --provenance /shared/run/provenance.json \
  --output results/raw/nano-attn-router.jsonl
```

The metadata parser reads one regular, non-symlink UTF-8 file and never
sources, evaluates, expands, or executes its contents. An explicit-only export
must supply all 12 identity fields: model, dispatcher, scope, mode, cluster,
profile, phase, steps, repeat, run group, job ID, and Router Replay state.

`--provenance` is JSON, not `run-metadata.env`. Populate it from the immutable
runtime attestation and run metadata with the exact keys `nemo_rl_commit`,
`bridge_commit`, `mcore_commit`, `te_commit`, `te_version`, and
`container_sha256`. The source commit plus container digest identify the pinned
`.python-version` and uv implementation; the leaf runtime gate separately
requires the attested managed interpreter and executable SHA256 before Ray
starts.

TensorBoard logging is enabled by the Nano, Super, Ultra, and
Qwen3-30B-A3B selectors. It is intentionally disabled for Qwen3-235B-A22B
until an exact-runtime compatibility smoke proves it safe. W&B remains enabled
for every selector under project `sna-cg-study`, so Qwen3-235B uses the W&B
fallback. The selected policy is recorded as `tensorboard_enabled` in run
metadata; backend files live under the run log directory's `tensorboard/` and
`wandb/` subdirectories.

```bash
uv run export_wandb.py \
  --wandb-run entity/sna-cg-study/run_id \
  --optimizer-step-key _step \
  --run-metadata /shared/run/run-metadata.env \
  --router-replay off --status passed \
  --provenance /shared/run/provenance.json \
  --output results/raw/qwen235-arm-a.jsonl
```

The exporter scans unfiltered W&B history so sparse metric rows are retained.
`--optimizer-step-key` is mandatory and repeatable; `_step` above is valid only
after confirming that the selected run used NeMo-RL's optimizer step as the
W&B step. The exporter never assumes that mapping, reads `summary`, or fills
missing values with zero. Baselines may omit graph telemetry; every graph arm
must report every graph and correctness metric for every planned optimizer
step. `cuda_graph/cache_miss_count` is measured at the graph-bank lookup: both
`warming` and `captured` outcomes are misses, while only `captured` increments
capture count. The runtime enforces `capture_count <= cache_miss_count` and
`eviction_count <= capture_count` rather than deriving misses from captures.

`analyze_cuda_graph_calls.py` consumes local Nsight Systems profiles. Its
`nsys_cuda_graph_launch_share_of_cuda_api_calls_pct` field uses all CUDA
runtime and driver API calls as the denominator; it must not be interpreted as
NeMo-RL eligible-call coverage.

```bash
uv run --no-project analyze_cuda_graph_calls.py \
  --label nano-attn=/shared/run/nsys \
  --output-json results/nsys_cuda_graph_calls.json
```

Collect every `results/raw/*.json*` artifact and render the static report:

```bash
uv run --no-project collect_results.py
uv run --no-project render_report.py
```

The normalized CSV and JSON retain failure-only rows and include model,
dispatcher, scope, mode, cluster/profile, phase, steps, job ID, four timing and
throughput families, graph/eligible calls, runtime coverage, three token
counts, capacity/padding utilization, reward/loss/error metrics, router and
expert parity, gradient norm, NaN/Inf status, source commits, TE provenance,
and container digest. Report medians exclude steps 1 through 5, keeping the
three warmups and capture transition outside the steady-state window.

The repository-wide `results/` ignore rule intentionally keeps live outputs
out of ordinary commits. Use an explicit reviewed `git add -f` only when a
task requires publishing a particular result ledger or report.

## Qwen MoE router-replay validation campaign

The Qwen campaign is a four-arm, paired comparison that deliberately separates
router-replay effects from CUDA Graph effects. It supports
`qwen3_30ba3b` (the 4-node by 4-GPU performance recipe) and
`qwen3_235b` (the 16-node by 4-GPU performance recipe). Both selectors use
packed THD, three successful CUDA Graph warmup steps, disabled checkpoints,
and W&B project `sna-cg-study`.

| Arm | Router Replay (R3) | CUDA Graph scope | Purpose |
| --- | --- | --- | --- |
| A | off | none | eager baseline |
| B | off | `moe_router` | isolates router graph without replay |
| C | on | none | isolates replay without router graph |
| E | on | `attn` | replay-safe graph comparison; router remains eager |

Default selection is intentionally asymmetric: Qwen30 smoke selects A/B/C/E,
Qwen235 smoke selects A/B, and performance for either model selects A/B. C/E
performance must be requested as positional arm arguments; Qwen235 C/E are
dependency-blocked by the disabled R3 preflight gate described below.

The omitted D arm, `R3=on` plus `moe_router` (or `moe_preprocess`) CUDA
Graph, is intentionally fail-closed. Route IDs are installed after capture
and are not graph replay inputs, so reusing the router graph could consume
stale routes. `run_scope.sh` rejects that combination before a scheduler call;
do not bypass this guard or claim it as a correctness experiment.

### Immutable campaign evidence

`validate_campaign_gate.py` accepts only absolute, regular, non-symlink JSON
artifacts plus an explicitly supplied 64-character lowercase SHA256. Promotion
artifacts contain this exact provenance object:

| Field | Required value |
| --- | --- |
| `nemo_rl_commit` | full commit matching the selected profile |
| `bridge_commit` | full commit matching the selected profile |
| `mcore_commit` | full commit matching the selected profile |
| `container_sha256` | staged image SHA256 matching the selected profile |
| `runtime_attestation_sha256` | SHA256 of the profile's exact runtime-attestation file |

Qwen3-235B C/E are currently dependency-blocked. The earlier hand-authored R3
gate schema asserted a job ID and diagnostic settings without cryptographically
binding the raw output, exact argv, process exit status, and successful Slurm
attempt. `validate_campaign_gate.py` now rejects every R3 gate, including a
formerly valid-looking content-addressed envelope, before any leaf or scheduler
contact. Re-enable C/E only after a committed Slurm producer atomically emits
one content-addressed record containing those identities and the validator
checks that record against the exact source/container/runtime profile.

Every 20-step performance request requires a smoke-promotion gate. Its exact
top-level fields are `gate_type="smoke_promotion"`, `status="passed"`, the
requested `model`, `phase="smoke"`, `steps=5`, `provenance`, and a non-empty
`arms` object. Each requested arm entry has exactly these fields:

| Field | A | B | C | E |
| --- | --- | --- | --- | --- |
| `job_id` | positive integer | positive integer | positive integer | positive integer |
| `status` | `passed` | `passed` | `passed` | `passed` |
| `completed_steps` | 5 | 5 | 5 | 5 |
| `metrics_finite` | true | true | true | true |
| `correctness_passed` | true | true | true | true |
| `undeclared_fallbacks` | 0 | 0 | 0 | 0 |
| `router_replay` | `off` | `off` | `on` | `on` |
| `graph_coverage_status` | `not_applicable` | `passed` | `not_applicable` | `passed` |
| `r3_trace_status` | `not_applicable` | `not_applicable` | `passed` | `passed` |

Create promotion evidence only after exporting all five planned optimizer
steps and reviewing finiteness, correctness, fallbacks, graph coverage, and R3
validation records. A/B promotion does not authorize C/E. Qwen3-235B C/E
cannot be promoted while R3 validation is disabled. Missing, stale,
wrong-profile, digest-mismatched, failed, or incomplete promotion gates are
rejected before leaf execution or scheduler contact.

Start with the five-step smoke on Qwen3-30B-A3B. `TEST_ONLY=1` renders all
commands without creating directories or contacting Slurm. A real launch
requires a refreshed, attested `profiles/oci-hsg.env`; the example profile is
not production-ready.

```bash
CLUSTER=oci-hsg MODEL=qwen3_30ba3b PHASE=smoke TEST_ONLY=1 \
  submit_qwen_router_validation.sh

CLUSTER=oci-hsg MODEL=qwen3_30ba3b PHASE=smoke RUN_TAG=qwen30-smoke \
  submit_qwen_router_validation.sh
```

Qwen3-235B smoke defaults to A/B. C/E remain rejected regardless of a supplied
R3 gate until the content-bound Slurm producer described above exists.

After a model's own five-step guard and paired correctness checks pass, create
that model's promotion artifact and use the 20-step performance phase. A
Qwen30 gate cannot promote Qwen235; model and runtime provenance must match
exactly.

```bash
QWEN30_PROMOTION=/shared/gates/qwen30-smoke-promotion.json
QWEN30_PROMOTION_SHA256=$(sha256sum "${QWEN30_PROMOTION}" | awk '{print $1}')
SMOKE_PROMOTION_FILE="${QWEN30_PROMOTION}" \
SMOKE_PROMOTION_SHA256="${QWEN30_PROMOTION_SHA256}" \
CLUSTER=oci-hsg MODEL=qwen3_30ba3b PHASE=performance RUN_TAG=qwen30-perf \
  submit_qwen_router_validation.sh

QWEN235_PROMOTION=/shared/gates/qwen235-smoke-promotion.json
QWEN235_PROMOTION_SHA256=$(sha256sum "${QWEN235_PROMOTION}" | awk '{print $1}')
SMOKE_PROMOTION_FILE="${QWEN235_PROMOTION}" \
SMOKE_PROMOTION_SHA256="${QWEN235_PROMOTION_SHA256}" \
CLUSTER=oci-hsg MODEL=qwen3_235b PHASE=performance RUN_TAG=qwen235-perf \
  submit_qwen_router_validation.sh
```

Qwen3-235B C/E performance remains blocked for the same reason. Do not create
promotion evidence for an arm that cannot pass its source-bound smoke gate.

Each R3-on command exports `NRL_ROUTER_REPLAY_VALIDATE=1`, `NRL_R3_TRACE=1`,
`NRL_R3_TRACE_STEPS=5`, and `NRL_R3_TRACE_VERIFY_FORWARD=1`. Before comparing
performance, inspect the driver/Ray logs for the trace and validation result
on every first-five-step R3 run, and require router/expert parity, finite
losses and gradients, plus no token-multiplicative-probability or policy-KL
outlier. An arm that lacks those records is not a correctness-passing result.

R3 jobs bind the rendered driver and `tools/check_r3_trace.py` bytes by SHA256,
then execute the checker with `--require-forward-verify` and
`--require-cp-identity`. Each Slurm attempt publishes
`r3-validation-job-<job-id>-restart-<count>/r3-validation.json` below its run
log directory. The authoritative record contains exact driver/checker paths,
digests and commands, trace directory, Slurm identity, raw and normalized exit
codes, and one of `pending`, `passed`, `failed`, or
`not_run_driver_failed`. Only `passed` is promotable; driver or checker failure
propagates to the Slurm job.

Every accepted real submission also atomically publishes `run-metadata.json`
and strict `run-metadata.env`. The JSON is authoritative for exact command and
scheduler argv values; the env form base64-encodes arbitrary strings and is
consumed only by the non-executing exporter parser. Scheduler test-only mode
publishes neither file.

Collect completed local artifacts and regenerate the static report only after
all paired arms have finished:

```bash
uv run --no-project collect_results.py
uv run --no-project render_report.py
```

Compare E2E, generation, policy-training, and logprob step time and
tokens/sec/GPU only within the same model, phase, dispatcher, cluster/runtime
profile, repeat, and R3 state. The report also retains graph/eligible-call
coverage, graph fallback information, reward, `gen_kl_error`,
`token_mult_prob_error`, router/expert parity, and gradient health for the
correctness decision.

## OCI readiness before submission

OCI-HSG access has been verified, and both Qwen HF snapshots plus the nightly
container image are available there. No campaign job has been submitted from
this branch. Before the first non-`TEST_ONLY` leaf, create the clean remote
campaign checkout and run a fresh runtime attestation against the exact
source, nested gitlinks, lockfile, container digest, and four-GPU topology.
The resulting successful attestation path and preflight job ID are required in
the OCI profile; all campaign leaves then depend on that preflight.
