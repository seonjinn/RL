# Mamba/MoE Transformer Engine CUDA Graph study

This directory is the persistent launcher and reporting surface for the
2026-07-29 packed Mamba/MoE training study.

## Matrix semantics

`scopes/` contains exactly 33 launchers:

- `00_baseline_no_cg.sh` is the sole no-CG baseline and sets
  `cuda_graph_impl=none`.
- The other 32 scripts are the Cartesian product of optional `attn`, `mlp`,
  and `mamba` modules with one mutually exclusive MoE selection: none, `moe`,
  `moe_router`, or `moe_router,moe_preprocess`.
- `01_whole_layer.sh` uses Transformer Engine with an empty module list. The
  empty TE scope means whole-layer capture; it is not the no-CG baseline.

`variants/` contains the eight persistent configurations formed by both values
of shared-expert overlap and selective `moe_act` recompute under only the
`moe` and `moe_router,moe_preprocess` graph scopes. It also contains the
correctness-first combined
`attn,mamba,moe_router,moe_preprocess` row with shared-expert overlap disabled.
`moe_act` and `shared_expert` are configuration knobs and never graph-scope
entries.

`pairs/` preserves the two drop-and-pad MoE reproducer launchers, but they are
intentional negative tests. Sequence packing produces rank-local packed token
extents, while drop-and-pad MoE derives expert capacity from that local extent.
An expert-parallel group can therefore enter different all-to-all and Mamba
collectives on the second update. This occurs with and without CUDA Graphs, so
the launchers now fail before printing or invoking `sbatch`. Use the standard
dropless scopes, especially `moe_router,moe_preprocess`, for partial capture.

Every launcher pins three successful warmup updates, two cached PP schedule
banks, at most 16 packed sequences, checkpoint writes disabled, and W&B
project `sna-cg-study`. Runtime names add model, cluster, phase, and a UTC tag
to the launcher-specific prefix.

## Models and profiles

`MODEL` accepts:

| Model | Immutable base recipe | Scope preflight |
|---|---|---|
| `nano-hybrid` | `examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml` | All 32 TE rows |
| `qwen3-30b-a3b` | `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` | Mamba rows fail before Slurm |
| `qwen3-235b-a22b` | `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml` | Enabled because the recipe exists; Mamba rows fail before Slurm |

The Nano recipe above is intentionally the packed hybrid recipe. The
similarly named generation recipe disables sequence packing and is not valid
for this study.

Cluster profiles retain explicit `__REQUIRED_*__` placeholders until their
immutable model snapshots, staged nightly sqsh path, and container SHA256 are
verified. The Ptyche Nano fields are resolved for this study. A real launch
refuses to call `sbatch` while any required field is unresolved. `TEST_ONLY=1`
prints every unresolved field and the complete training and Slurm commands,
then exits without contacting the scheduler.

## Native Transformer Engine preflight

Every Ptyche baseline and CUDA-Graph run uses the same validated native
Transformer Engine wheel runtime. The runtime is pinned to commit
`4a18653fc7274b10e33cd786b91be6261c523dc0`, version
`2.15.0+4a18653f`, and wheel SHA256
`029fdbcb3fc0aa17b1a4f7398f56040204307d4bc839d318feda1677c98fff5e`.
Its `site-packages` is first on `PYTHONPATH`; the legacy TE archive is excluded.
Before Ray starts, the launcher validates the runtime provenance, install
prefix, image digest, package files, version, Python package, PyTorch extension,
and core shared-library import paths. Any mismatch stops the launch. No
performance job builds or installs Transformer Engine.

## Persistent performance runtime contract

Ptyche performance launchers use the validated native TE runtime followed by
the immutable flash-attn, ml-dtypes, ONNX, ONNX IR, and ONNXScript archives,
then the pinned NeMo-RL checkout, Bridge, and Megatron-LM sources. The driver
is the external MCore environment keyed by lock blob
`96543608420ac6746cfd18d1fcd8ee1bd3c91caf` and the nightly image digest; the
bare image `/opt/nemo_rl_venv` is not used.

MCore policy actors receive the same driver interpreter through the explicit
`NEMO_RL_REQUIRE_SYSTEM_MCORE=1` contract. The registry fails before actor
creation unless both the lexical interpreter path and `sys.prefix` match the
pinned external venv. This bypasses NeMo-RL's normal per-node `uv` MCore venv
creation, so it cannot install or build another Transformer Engine artifact.
Other actor tiers retain normal forced venv rebuilding; every Ray runtime env
inherits the native-TE-first `PYTHONPATH`.

`TEST_ONLY=1` prints without contacting Slurm. For a scheduler-only validation
of the exact command, set `SBATCH_TEST_ONLY=1`; it adds `--test-only` while
preserving `sbatch --parsable` and prints the machine-readable Slurm result.
Before either real or scheduler-only submission, the launcher verifies the
working lock blob, six-entry runtime order, readable pinned container and
source paths, locked interpreter launcher, and native TE provenance. Pure
`TEST_ONLY=1` intentionally skips these host-dependent checks.

## GPU integration gate

`scripts/validate_nemorl_integration.sub` is the earlier source-integration
gate. It mounts the reviewed Transformer Engine FP64 overlay read-only and
validates it in the same Python process as each preserved pytest suite. It
accepts only the staged nightly
image `nemo_rl_nightly_20260729_2472184.sqsh`, recorded as SHA256
`cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91`;
container overrides are intentionally not accepted for this gate. It hashes the
25 GB image with `sha256sum --` before `srun` and fails closed on a mismatch.

## Persistent native Transformer Engine build

`scripts/build_te_pr2898_backport.sub` is the separate native-build gate for
TransformerEngine commit `4a18653fc7274b10e33cd786b91be6261c523dc0` from
`${ROOT}/src/TransformerEngine-fp64-thd-cudagraph-20260730`. It validates the
source HEAD, clean worktree, recursive submodule state, and the same pinned
nightly image SHA256 before entering the container. The container builds one
offline PyTorch wheel with `/opt/nemo_rl_venv/bin/python setup.py bdist_wheel`,
`NVTE_CUDA_ARCHS=100`, and at most 16 build jobs.

The launcher publishes only a fresh commit-named directory under
`${ROOT}/artifacts/transformer-engine/`, containing exactly one wheel under
`wheel/`, its SHA256 sidecar, and provenance JSON. An atomic output lock admits
one builder, and publication uses a no-nesting atomic rename; an existing
commit directory is never replaced. CMake, setuptools, and bdist intermediates
stay under a disposable build staging directory and are never published. A
separate publication staging directory contains only the wheel, checksum
sidecar, and provenance. Both are removed on failure, and the disposable build
staging is also removed before atomic publication. It is deliberately
outside every performance launcher, so no performance job builds or installs
native Transformer Engine code.

```bash
TE_BUILD_JOBS=16 \
  sbatch experiments/cuda_graph/mamba_moe_te_graph_20260729/scripts/build_te_pr2898_backport.sub
```

Ptyche job `2475656` compiled the original `ba256c5b` backport through target
50/78, then failed in `fused_moe_aux_loss.cu`: PR2898 assumed a newer
compile-time reducer signature and accumulator conversion helper that are not
present in the pinned nightly TE base. Commit `c16cb9a1` adapts only the new
graph-safe aux-loss path to the legacy private helpers while preserving the
public ABI and `Coeff_buf[0]/[1]` semantics. The failed job exited `1:0` after
`00:21:10`; its build and publication staging directories and output lock were
removed, and no immutable wheel directory was published.

Ptyche job `2475704` then compiled and linked the native common library through
target 78/78, including `fused_moe_aux_loss.cu.o`, but failed while compiling
the PyTorch extension. PR2898's pybind refactor referenced two NVFP4 helpers
that exist on its newer upstream base but not on the pinned e707 base. Commit
`4a18653f` omits only those unavailable, graph-unrelated bindings; they were
not part of the legacy Python API. Its regression suite verifies that the
graph-safe router binding remains present while the post-pin NVFP4 references
do not. The failed job exited `1:0` after `00:21:15`; cleanup again left no
artifact, staging directory, or lock.

The gate runs from the clean runner checkout
`${ROOT}/src/RL-pr5672-mamba-moe-graph-cache-runner-20260730`, which must be
freshly cloned and initialized at the submitted `EXPECTED_SHA`; it never uses
the developer's dirty checkout. It checks both the submitted commit's
`uv.lock` blob and the worktree lockfile against
`96543608420ac6746cfd18d1fcd8ee1bd3c91caf`, while retaining the exact
pyproject/submodule diff guards.

Before either suite, the gate locks a persistent, external venv keyed by the
lock blob, the immutable image SHA prefix, and `py313-aarch64`, then runs
`uv sync --frozen --extra mcore --group test --no-install-package transformer-engine --no-install-project
--no-install-local --python /opt/nemo_rl_venv/bin/python
--no-python-downloads`. This selects pure-Python locked dependencies without
installing repository, local sources, or Transformer Engine itself. It then
fails closed if a reused venv contains Transformer Engine module, dist-info, or
native-extension artifacts. The explicit locked `test` group supplies pytest and
its test-only dependencies; the gate never uses an ad-hoc `pip` or `uv tool`
installation. The immutable archive prefix remains first in `PYTHONPATH`, so
the mounted Transformer Engine archive is still what the validator and pytest
import. The gate also runs the packed-sequence, Transformer Engine graph-bank,
FP64 MoE-router boundary, packed-Mamba parity, and `5→3→5` graph-bank schedule
tests from the pinned Megatron-LM checkout. The first three MCore selections
remain single-rank to avoid changing unrelated file-level test topology. The
two world-size-dependent parity and `5→3→5` nodes instead run through the same
venv interpreter and overlay validator via `torch.distributed.run` with two
ranks. Torchrun's per-rank stdout is preserved under
`${ROOT}/logs/nemorl-integration-distributed-${SLURM_JOB_ID}`; each rank must
report exactly `2 passed` with no skip, or the gate fails. This gate must pass
before submitting the 20-step matrix.

## Native Transformer Engine wheel validation

`scripts/validate_te_pr2898_wheel.sub` is the one-GPU GB200 install and runtime
gate for the immutable wheel produced by the native build. It accepts only
TransformerEngine commit `4a18653fc7274b10e33cd786b91be6261c523dc0`, the
commit-named artifact directory, and the staged nightly image with SHA256
`cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91`.
Before allocating the container it requires exactly one wheel, its exact
SHA256 sidecar, the build provenance schema and values, and the artifact
whitelist of `provenance.json` plus `wheel/`.

The gate installs the wheel offline and without dependencies into disposable
staging. It disables uv config discovery so NeMo-RL's Transformer Engine
source override cannot replace the explicit immutable wheel. It validates that
the Python package, PyTorch native extension, and core Transformer Engine
shared library all resolve from that staged prefix. Non-TE runtime dependencies
come from the same pinned flash-attn, ml-dtypes, ONNX, ONNX IR, and ONNXScript
uv archives used by the integration gate; the container's legacy Transformer
Engine archive is deliberately excluded.
It then runs the static PR2898 compatibility suite and the fused MoE aux-loss
CUDA graph capture/replay test from the pinned source checkout. Only a fully
passing install is atomically published under
`${ROOT}/runtimes/transformer-engine/` with both the TE commit and full wheel
SHA256 in its immutable directory name. A per-prefix lock prevents concurrent
publication; existing prefixes are never replaced, and failures remove both
install and publication staging. This validation is separate from performance
jobs and does not rebuild NeMo-RL environments.

```bash
sbatch --test-only \
  experiments/cuda_graph/mamba_moe_te_graph_20260729/scripts/validate_te_pr2898_wheel.sub
sbatch \
  experiments/cuda_graph/mamba_moe_te_graph_20260729/scripts/validate_te_pr2898_wheel.sub
```

Ptyche build job `2475736` completed `0:0` and published
`transformer_engine-2.15.0+4a18653f-cp313-cp313-linux_aarch64.whl` with SHA256
`029fdbcb3fc0aa17b1a4f7398f56040204307d4bc839d318feda1677c98fff5e`.
GPU validation job `2475881` completed `0:0` on GB200: all TE Python,
PyTorch-extension, and core-library paths resolved below the staged wheel
prefix, and the five static compatibility checks plus the graph-safe MoE
aux-loss CUDA Graph capture/replay test reported `6 passed` in 3.58 seconds.
The validated runtime was atomically published under the commit-and-wheel-hash
immutable prefix, with no staging directories or locks left behind.

## Exact 20-step comparison

`submit_20step_native_te_comparison.sh` submits six independent, dropless Nano
performance jobs: baseline, `attn`, `mamba`, `moe_router`, correctness-first
`moe_router,moe_preprocess`, and the combined correctness-first scope. Every
job uses 20 steps, sequence packing, three warmup steps, checkpoint writes
disabled, and the same native TE runtime. There are no Slurm dependencies or
singleton constraints.

```bash
CLUSTER=ptyche \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_20step_native_te_comparison.sh
```

For the fixed packed-replay validation, use the smaller matched matrix:

```bash
CLUSTER=ptyche \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_20step_packed_hybrid_fix.sh
```

It submits only baseline, `moe_router,moe_preprocess`, and
`attn,mamba,moe_router,moe_preprocess`.

## Local preflight

Run one launcher:

```bash
TEST_ONLY=1 CLUSTER=ptyche \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/scopes/17_attn.sh
```

The standalone-MoE drop-and-pad reproducers are expected to exit `2` before
the scheduler:

```bash
TEST_ONLY=1 CLUSTER=ptyche \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/pairs/00_drop_pad_baseline_no_cg.sh
TEST_ONLY=1 CLUSTER=ptyche \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/pairs/01_drop_pad_moe.sh
```

Preflight every persistent smoke launcher:

```bash
TEST_ONLY=1 CLUSTER=ptyche \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_all_smokes.sh
```

Select reusable performance rows explicitly:

```bash
TEST_ONLY=1 CLUSTER=ptyche \
PERFORMANCE_SCRIPTS="scopes/00_baseline_no_cg.sh scopes/01_whole_layer.sh" \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_performance.sh
```

Selecting the drop-and-pad pair through the batch helper is also expected to
exit `2` without printing an `SBATCH:` command:

```bash
TEST_ONLY=1 CLUSTER=ptyche \
PERFORMANCE_SCRIPTS="pairs/00_drop_pad_baseline_no_cg.sh pairs/01_drop_pad_moe.sh" \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_performance.sh
```

Qwen Mamba preflight is expected to fail:

```bash
TEST_ONLY=1 CLUSTER=ptyche MODEL=qwen3-30b-a3b \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/scopes/05_mamba.sh
```

## Local result pipeline

For a completed TensorBoard run, first export the local event file or directory
to the collector's JSONL contract. The exporter never contacts a network. It
requires exactly one finite sample at each optimizer step 1--20 for every
canonical timing, throughput, and correctness metric; duplicate scalar steps
use the latest TensorBoard wall-time event. It fails without replacing an
existing output when any required metric is incomplete.

```bash
python3 experiments/cuda_graph/mamba_moe_te_graph_20260729/export_tensorboard.py \
  --event /path/to/tensorboard-run \
  --scope moe-router-preprocess \
  --job-id 2474000 \
  --status performance:passed \
  --output experiments/cuda_graph/results/mamba_moe_te_graph_20260729_events.jsonl
```

Then normalize the event JSONL into the CSV consumed by the static renderer:

```bash
python3 experiments/cuda_graph/mamba_moe_te_graph_20260729/collect_results.py \
  --input experiments/cuda_graph/results/mamba_moe_te_graph_20260729_events.jsonl \
  --output experiments/cuda_graph/results/mamba_moe_te_graph_20260729_results.csv
```

The collector can also read a local JSON/JSONL W&B export; it never calls the network:

```bash
python3 experiments/cuda_graph/mamba_moe_te_graph_20260729/collect_results.py \
  --input /path/to/local-export.jsonl \
  --output experiments/cuda_graph/results/mamba_moe_te_graph_20260729_results.csv
```

The checked-in submission ledger can be normalized before completed-run
metrics are available:

```bash
python3 experiments/cuda_graph/mamba_moe_te_graph_20260729/collect_results.py \
  --input experiments/cuda_graph/results/mamba_moe_te_graph_20260729_submissions.json \
  --output experiments/cuda_graph/results/mamba_moe_te_graph_20260729_results.csv
```

The submission ledger preserves failure text, Slurm exit code, elapsed wall
time, and completed-step count for the Failures table. Those fields are
provisional job telemetry, not CUDA Graph telemetry: performance JSONL rows
may legitimately leave them blank. Conversely, absent `eviction_count` or
`fallback_count` is never converted to zero; a steady-state comparison remains
invalid until both graph-telemetry counters are actually collected.

The required W&B mappings are:

| Result | W&B metric |
|---|---|
| E2E throughput | `performance/tokens_per_sec_per_gpu` |
| Generation throughput | `performance/generation_tokens_per_sec_per_gpu` |
| Policy throughput | `performance/policy_training_tokens_per_sec_per_gpu` |
| Logprob throughput | `performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu` |
| E2E time | `timing/train/total_step_time` |
| Generation time | `timing/train/generation` |
| Policy time | `timing/train/policy_training` |
| Logprob time | `timing/train/policy_and_reference_logprobs` |
| Reward / accuracy | `train/reward` (falls back to `train/accuracy`) |
| Generation KL error | `train/gen_kl_error` |
| Token multiplication probability error | `train/token_mult_prob_error` |
| Policy KL error | `train/policy_kl_error` |
| JS divergence error | `train/js_divergence_error` |
| Sampling importance ratio | `train/sampling_importance_ratio` |
| Masked sequences by logprob error | `train/num_masked_seqs_by_logprob_error` |
| Policy loss | `train/loss` |
| Gradient norm | `train/grad_norm` |

All of those correctness values are required to be finite at every step 1--20;
the exporter does not publish a partial JSONL. Older TensorBoard events may use
`train/num_mask_sample_filtered` for the masked-sequence value. The exporter
accepts that source tag only as a backward-compatible alias and writes the
canonical `train/num_masked_seqs_by_logprob_error` key.

### Correctness schema migration

`generation_kl_error` now means only `train/gen_kl_error`.
`token_mult_prob_error` is a separate CSV/report field and has its own
`*_delta` comparison. The policy-KL, JS-divergence, sampling-importance-ratio,
and masked-sequence fields are likewise explicit CSV fields, validity inputs,
and correctness deltas. CSV or JSONL files produced before this migration
mislabelled `train/token_mult_prob_error` as generation KL and omit the new
fields; do not compare them with new results. Re-export those runs from their
raw TensorBoard events, re-run the collector, then render the report.

For the checked raw events, the concrete re-export commands are:

```bash
python3 experiments/cuda_graph/mamba_moe_te_graph_20260729/export_tensorboard.py \
  --event experiments/cuda_graph/results/raw/2475435/events.out.tfevents.1785417835.ptyche0056.ptyche.clusters.nvidia.com.1871169.0 \
  --scope baseline-no-cg --job-id 2475435 --status performance:completed \
  --output /tmp/mamba-moe-2475435.jsonl
python3 experiments/cuda_graph/mamba_moe_te_graph_20260729/export_tensorboard.py \
  --event experiments/cuda_graph/results/raw/2475438/events.out.tfevents.1785418205.ptyche0258.ptyche.clusters.nvidia.com.1993316.0 \
  --scope moe-router --job-id 2475438 --status performance:completed \
  --output /tmp/mamba-moe-2475438.jsonl
```

Concatenate those two JSONL outputs (or use a single local event export), then
run the collector and renderer commands above to replace the obsolete CSV and
report.

Finally render the normalized CSV into the persistent static report:

```bash
python3 experiments/cuda_graph/mamba_moe_te_graph_20260729/render_report.py \
  --te-version 2.15.0+42b84005 \
  --te-source-commit e707aa46869dc2aec08dfea25402e97a61d49fef \
  --te-overlay-sha256 39f7b26b8cf127e3ca104c0375c97ce4e6d047178f9d00836b92469b1c2e544b
```

The report always keeps Correctness, Smoke, Performance, Accuracy, Failures,
and Provenance separate. Missing experiment rows remain visibly pending.

## Verified status ledger

| Task | Evidence | Status |
|---|---|---|
| MCore Task 1 | Slurm 2471224 | 66 passed |
| MCore Task 2 | Slurm 2471343 | 29 + 3 passed |
| MCore Task 3 | Slurm 2471570 | 38 + 3 passed |
| MCore Task 4 | Slurm 2471681 | 43 + 23 passed |
| MCore Task 5 | Slurm 2471988 | Completed exit 0 on 4xGB200. Every rank reported 2 passed / 108 deselected: packed Mamba parity 74.33s, MoE 5→3→5 6.96s, total 82.78s. Earlier jobs 2471820 and 2471877 exposed test-config/telemetry assertions rather than production graph failures; focused MoE job 2471888 passed on all four ranks. Its `routing_map.sum` token-count oracle is valid only for this EP1/TP1 test, not EP>1 post-communication counts. Final MCore head: `100047b517ea91526dc465448fcb3b37b2598388`. |
| NeMo-RL Task 6 | Host suite | 37 host tests plus Pyrefly passed |
| NeMo-RL Task 7 | Slurm 2472646 | 138 passed integration tests with exit 0 on the pinned nightly container. |
| NeMo-RL Task 8 | Slurm 2473134, 2473144–2473170 | Baseline plus 27 curated Nano hybrid smoke rows submitted on Ptyche without singleton dependencies. The submission ledger preserves the exact launcher-to-job mapping. |
| NeMo-RL Task 9 | Slurm 2475736, 2475881 | Native TE `4a18653f` wheel build completed `0:0`; wheel SHA256 `029fdbcb…fff5e`. GB200 validation completed `0:0`, resolved TE Python/native/core from the immutable wheel prefix, and reported 6 passed including graph-safe MoE aux-loss CUDA Graph capture/replay. |
| Packed hybrid replay fix | Slurm 2477954 | MCore commit `78f8f404` passed all five focused packed-sequence tests: hybrid Identity-attention replay, three logical post-MLP extents, and offload cleanup. |
| Fixed-scope Nano smoke | Slurm 2477897, 2477898 | Dropless `moe_router,moe_preprocess` and combined `attn,mamba,moe_router,moe_preprocess` both completed two GRPO updates with exit `0:0`; no illegal-memory, watchdog, or NCCL-timeout signature appeared. |
| Drop-and-pad fail-fast | Slurm 2478037 | The container integration gate completed `0:0`: 222 NeMo-RL tests, 78 MCore tests, and both distributed MCore targets on each rank passed. The no-CG and `[moe]` drop-and-pad deadlocks are now rejected during setup and by every persistent launcher path. |
