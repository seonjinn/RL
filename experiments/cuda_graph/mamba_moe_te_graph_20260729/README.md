# Mamba/MoE Transformer Engine CUDA Graph study

This directory is the persistent launcher and reporting surface for the
2026-07-29 packed Mamba/MoE training study. Task 8 creates and validates these
artifacts; it does not submit a Slurm job.

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
`moe` and `moe_router,moe_preprocess` graph scopes. `moe_act` and
`shared_expert` are configuration knobs and never graph-scope entries.

`pairs/` contains an isolated drop-and-pad MoE comparison. Both launchers set
`moe_expert_capacity_factor=1.0` and
`moe_pad_expert_input_to_capacity=true`; `00_drop_pad_baseline_no_cg.sh` uses
no CUDA Graph and `01_drop_pad_moe.sh` captures only `[moe]`. This pair is
separate from the standard dropless matrix because capacity-based routing can
drop tokens. The two pair members share all other model, runtime, resource,
and training settings. They are fail-closed to `MODEL=nano-hybrid`: the Qwen
Flex/HybridEP recipes are not compatible with this standalone-MoE graph path.

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

## Transformer Engine FP64 overlay preflight

Every Ptyche baseline and CUDA-Graph run mounts a commit-addressed Transformer
Engine `utils.py` artifact read-only over the image's exact uv-archive path.
The source artifact lives under `.overlay/<commit>/utils.py` in the shared
Ptyche Transformer Engine checkout with filesystem mode `0444`. The overlay is
pinned to commit `e707aa46869dc2aec08dfea25402e97a61d49fef`, version
`2.15.0+42b84005`, and SHA256
`39f7b26b8cf127e3ca104c0375c97ce4e6d047178f9d00836b92469b1c2e544b`.
Before Ray starts, the head validates the installed package version, mounted
source digest and mode, FP64 `<f8` registry entry, and CUDA FP64 weak-reference
dtype, shape, and data-pointer identity. Any mismatch stops the launch before
Ray; workers share the same immutable image and read-only mount, so they do not
repeat package installation.

## Persistent performance runtime contract

Ptyche performance launchers use the immutable archive prefix proven by the
integration gate: Transformer Engine, flash-attn, ml-dtypes, ONNX, ONNX IR,
and ONNXScript, followed by the pinned NeMo-RL checkout, Bridge, and
Megatron-LM sources. The driver is the external MCore environment keyed by lock
blob `96543608420ac6746cfd18d1fcd8ee1bd3c91caf` and the nightly image digest;
the bare image `/opt/nemo_rl_venv` is not used.

MCore policy actors receive the same driver interpreter through the explicit
`NEMO_RL_REQUIRE_SYSTEM_MCORE=1` contract. The registry fails before actor
creation unless both the lexical interpreter path and `sys.prefix` match the
pinned external venv. This bypasses NeMo-RL's normal per-node `uv` MCore venv
creation, so it cannot install or build another Transformer Engine artifact.
Other actor tiers retain normal forced venv rebuilding; every Ray runtime env
inherits the archive-first `PYTHONPATH` and the read-only overlay mount.

`TEST_ONLY=1` prints without contacting Slurm. For a scheduler-only validation
of the exact command, set `SBATCH_TEST_ONLY=1`; it adds `--test-only` while
preserving `sbatch --parsable` and prints the machine-readable Slurm result.
Before either real or scheduler-only submission, the launcher verifies the
working lock blob, six-archive order, readable pinned container and source
paths, locked interpreter launcher, and the overlay source's `0444` mode and
SHA256. Pure `TEST_ONLY=1` intentionally skips these host-dependent checks.

## GPU integration gate

`scripts/validate_nemorl_integration.sub` mounts the same reviewed
Transformer Engine overlay read-only and validates it in the same Python
process as each preserved pytest suite. It accepts only the staged nightly
image `nemo_rl_nightly_20260729_2472184.sqsh`, recorded as SHA256
`cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91`;
container overrides are intentionally not accepted for this gate. It hashes the
25 GB image with `sha256sum --` before `srun` and fails closed on a mismatch.

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

## Local preflight

Run one launcher:

```bash
TEST_ONLY=1 CLUSTER=ptyche \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/scopes/17_attn.sh
```

Preflight the standalone-MoE drop-and-pad pair:

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

The two approved drop-and-pad pair launchers can also be selected explicitly:

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
  --scope drop-pad-moe \
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
