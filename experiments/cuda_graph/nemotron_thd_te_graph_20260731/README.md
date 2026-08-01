# Nemotron packed-THD Transformer Engine CUDA Graph study

This directory is the persistent launcher, preflight, collection, and static
report surface for the 2026-07-31 Nemotron CUDA Graph study. It covers
Nemotron 3 Nano, Super, and Ultra plus the Qwen3-30B-A3B comparison selector.
Every training launch disables checkpoints, uses exactly three successful
optimizer warmups, writes to W&B project `sna-cg-study`, and uses only the
current `cuda_graph_modules` and `thd_max_packed_sequences` configuration
fields.

No file here contains credentials. Profile files with real cluster paths and
source revisions remain local under `profiles/*.env`.

## Current status

The MCore branch is pinned at `e835b64c55a5c3fc23da573d8c3e5e9c2e706694`
and the Bridge branch at `51481de3d8b0bd5139f9ac9c8dbc4e7d442e0712`.
The outer project pins official Transformer Engine main commit
`bffde8f4a0a4eea9036dc753e28269247e5de69d` (2.19 development metadata).
The local experiment suite passes; Linux/GB200 runtime, correctness, and
performance results are still pending and must not be inferred from local
tests.

The current nightly image provides Python 3.13.11 while this source snapshot
requires the exact version in `.python-version` (currently 3.13.13). The OCI
preflight therefore stages uv-managed 3.13.13 once under
`ARTIFACT_DIR/uv-python-installations`, validates the resolved base executable,
and disables further Python downloads before resolving the locked training
environment. This is a runtime compatibility gate, not a CUDA Graph result.

Nano HybridEP `moe_preprocess` rows are fail-closed, while Super and the Qwen
comparison selector may run their validated preprocess rows. Qwen dense `mlp`
and `mamba` rows are model-incompatible. Ultra remains dependency-blocked
until its external launcher inputs and adapter are validated.

## Required gate order

Run these gates before submitting a model scope.

1. Stage a digest-pinned nightly container. The staging job publishes an
   immutable squashfs, SHA256 metadata, and a current symlink atomically.

   ```bash
   SOURCE_IMAGE=nvcr.io/nvidia/nemo-rl:nightly \
   SOURCE_DIGEST=sha256:<registry-digest> \
   SOURCE_COMMIT=<full-nemo-rl-commit> \
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
   EXPECTED_NEMORL_SHA=<full-nemo-rl-commit> \
   EXPECTED_BRIDGE_SHA=<full-bridge-commit> \
   EXPECTED_MCORE_SHA=<full-mcore-commit> \
   scripts/create_source_snapshot.sh
   ```

3. Verify the staged image and exact snapshot on one OCI node with four GPUs.
   The job builds the same editable `uv run --locked --extra mcore` environment
   used by policy workers and requires exactly four visible devices. It imports
   PyTorch, Transformer Engine, Megatron Core, Megatron Bridge, Mamba SSM,
   causal-conv1d, CuPy, and grouped GEMM. It hashes the 67 GB image once and
   records source, lock, image identity, the exact TE VCS commit, the exact
   Python patch version, and the managed base-interpreter SHA256 in a
   machine-readable success or failure artifact. Python downloads are enabled
   only for the initial managed-interpreter staging command and are set to
   `never` before `uv run --locked`.

   ```bash
   CONTAINER=/absolute/shared/containers/nemo_rl_nightly.sqsh \
   CONTAINER_SHA256=<64-lowercase-hex> \
   ARTIFACT_DIR=/absolute/shared/artifacts/container-runtime \
   PROJECT_ROOT=/absolute/shared/source-snapshots/<snapshot> \
   EXPECTED_NEMORL_SHA=<full-nemo-rl-commit> \
   EXPECTED_BRIDGE_SHA=<full-bridge-commit> \
   EXPECTED_MCORE_SHA=<full-mcore-commit> \
   EXPECTED_TE_SHA=bffde8f4a0a4eea9036dc753e28269247e5de69d \
   scripts/validate_oci_container_runtime.sub
   ```

4. Optionally bootstrap a second fresh Bridge checkout in the immutable
   container, verify the exact nested MCore commit, relock only
   fast-hadamard-transform with Python 3.12, and run the Nano, Super, and Ultra
   recipe tests.

   ```bash
   BRIDGE_REPOSITORY=git@github.com:organization/Megatron-Bridge.git \
   EXPECTED_BRIDGE_SHA=<full-bridge-commit> \
   EXPECTED_MCORE_SHA=<full-mcore-commit> \
   CONTAINER=/absolute/shared/containers/nemo_rl_nightly.sqsh \
   CONTAINER_SHA256=<64-lowercase-hex> \
   ARTIFACT_DIR=/absolute/shared/artifacts/bridge-bootstrap \
   scripts/validate_oci_bridge_bootstrap.sub
   ```

   The submit wrappers export absolute persistent helper paths. This is
   required because Slurm executes a copied wrapper from its spool directory.

5. Populate a local profile from `profiles/*.env.example`. Set
   `RUNTIME_PREFLIGHT_JOB_ID` to the successful preflight job and
   `RUNTIME_ATTESTATION` to its exact non-symlink JSON artifact. Every leaf is
   submitted with `afterok:<preflight-job>` and validates exact source, lock,
   image identity, device count, package set, TE commit, Python version, managed
   interpreter path, and interpreter SHA256 before starting Ray. The leaf
   derives `UV_PYTHON_INSTALL_DIR` from the immutable attestation directory,
   requires that path to be container-mounted, forces uv-managed Python with
   downloads disabled, and gives the NeMo-RL driver a fresh per-job
   `UV_PROJECT_ENVIRONMENT`. Leaf jobs do not rehash the image.

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

For a real launch, create `profiles/oci-hsg.env` from the example, replace
every `__REQUIRED_*__` value, remove `TEST_ONLY`, and keep the source trees at
the declared commits. Logs are written below
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

## Local export and collection

`export_tensorboard.py` accepts one or more local event paths and atomically
exports a complete 5-, 20-, or 100-step JSONL file. It rejects missing or
non-finite required tags without replacing a previous good artifact. Duplicate
events at one step use the newest TensorBoard wall time. The canonical metric
name is `train/token_mult_prob_error`.

```bash
uv run export_tensorboard.py \
  --event /shared/run/events \
  --model nano --dispatcher hybridep --scope attn,moe_router \
  --mode nemorl --cluster oci-hsg --profile oci-hsg-runtime-attested \
  --phase performance --steps 20 --job-id 123456 \
  --repeat 1 --run-group nano-performance-20260731 --status passed \
  --provenance /shared/run/provenance.json \
  --output results/raw/nano-attn-router.jsonl
```

`--provenance` is JSON, not `run-metadata.env`. Populate it from the immutable
runtime attestation and run metadata with the exact keys `nemo_rl_commit`,
`bridge_commit`, `mcore_commit`, `te_commit`, `te_version`, and
`container_sha256`. The source commit plus container digest identify the pinned
`.python-version` and uv implementation; the leaf runtime gate separately
requires the attested managed interpreter and executable SHA256 before Ray
starts.

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
