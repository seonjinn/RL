# Qwen 4n4g adaptive MXFP8 rollout A/B

This experiment qualifies and measures the custom vLLM 0.20.2 adaptive dense
MXFP8 path in the exact NeMo-RL Qwen3-30B-A3B 4n4g MXFP8 rollout workload.
Both performance arms inherit
`grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml`. This is not a BF16-versus-MXFP8
comparison.

## Immutable contract

The required runtime is:

- NeMo-RL: the committed and pushed experiment checkout.
- vLLM repository: `https://github.com/seonjinn/vllm.git`.
- vLLM commit: `217ece36ee503ee8ccfbfaa0a5331765b21d2160`.
- vLLM public version: `0.20.2`.
- FlashInfer: `0.6.8.post1`.
- Hardware: four OCI-HSG GB200 nodes, four GPUs per node.
- Model: `Qwen/Qwen3-30B-A3B`, vLLM TP1.
- Both arms use the same custom vLLM overlay and direct
  `flashinfer_trtllm` adaptive path with `switch_m=256`; this is explicitly
  **not** a comparison against stock vLLM.
- Baseline (`original`) uses the package-relative bootstrap manifest
  `qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json`. Its `8x4` and
  `128x4` tactic tables are empty, so every recorded dispatch must take the
  TRTLLM runner default (`tactic=-1`, `tactic_source=runner_default`).
- Shmoo-qualified (`adaptive`) differs only by selecting
  `qwen3_30ba3b_tp1_v0202_qualified.json`, the manifest containing the 106
  qualified tactics for this handoff.

The launcher compares both fully resolved Hydra configs after removing exactly
the manifest-selection key. It requires the bootstrap filename and SHA256 for
`original`, the qualified filename and SHA256 for `adaptive`, and rejects
differences in source commits, container digest, checkpoint, topology, seed, or
any other resolved setting.

Set the canonical checkout before invoking any launcher command. Slurm runs a
spooled copy of submitted scripts, so the launcher never derives repository
paths from its own filename:

```bash
export NEMO_RL_REPO_ROOT=$(pwd -P)
source "$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout/cluster/oci-hsg.env"
```

Both `NEMO_RL_REPO_ROOT` and `NEMO_RL_EXPERIMENT_ROOT` must be absolute paths
on shared storage mounted at the same location in the container. The launcher
validates and explicitly exports both paths with every `sbatch` call.

## Stage the container

Build and push an immutable ARM64 image containing the exact custom vLLM
source, loader, and required package-relative JSON. The bootstrap JSON must be
pre-bundled in that immutable image; the launcher resolves and verifies the
same package path on every node instead of creating a node-local bootstrap.
Import it on OCI-HSG with the standard two-step scheduling protocol:

```bash
export NEMO_RL_REPO_ROOT=$(pwd -P)
source "$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout/cluster/oci-hsg.env"

sbatch --test-only \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --export=ALL,SOURCE_IMAGE=<registry/repository:immutable-tag>,OUTPUT_PREFIX=nemo_rl_mxfp8_adaptive,CONTAINER_DIR="${CONTAINER_ROOT}",SOURCE_COMMIT="$(git rev-parse HEAD)" \
  scripts/stage_enroot_image.sbatch

sbatch --parsable \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --export=ALL,SOURCE_IMAGE=<registry/repository:immutable-tag>,OUTPUT_PREFIX=nemo_rl_mxfp8_adaptive,CONTAINER_DIR="${CONTAINER_ROOT}",SOURCE_COMMIT="$(git rev-parse HEAD)" \
  scripts/stage_enroot_image.sbatch
```

Record the immutable `.sqsh` path from the job, its adjacent metadata, and its
SHA256. Do not launch through the updated convenience symlink.

The profile defaults to account `coreai_dlalgo_nemorl`. An explicitly
authorized account can be selected without editing the profile. For example,
the currently faster scheduling preflight is:

```bash
SLURM_ACCOUNT=nemotron_sw_pre \
CONTAINER_IMAGE=<immutable-sqsh> \
ACTION=test-only \
bash "$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout/run_ab.sh" trace
```

## One-node/four-GPU smoke

Run the checked-in smoke before the 4n4g trace. It validates CUDA visibility,
NeMo-RL, Transformer Engine, Megatron Core, the exact custom vLLM source and
commit, vLLM/FlashInfer versions, loader import, and JSON model/TP/hash.

```bash
export NEMO_RL_REPO_ROOT=$(pwd -P)
source "$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout/cluster/oci-hsg.env"
export CONTAINER_IMAGE=<immutable-sqsh>
export EXPECTED_CONTAINER_SHA256=<raw-sha256>
export EXPECTED_NEMO_RL_COMMIT=$(git rev-parse HEAD)
export EXPECTED_CONFIG_NAME=qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json
export EXPECTED_CONFIG_SHA256=<raw-json-sha256>

sbatch --test-only \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gres=gpu:4 \
  --switches="${SLURM_SWITCHES}" \
  --wrap="srun --no-container-mount-home --container-image=${CONTAINER_IMAGE} --container-mounts=${CONTAINER_MOUNTS} bash ${NEMO_RL_REPO_ROOT}/experiments/mxfp8_adaptive_rollout/smoke_container.sh"
```

The smoke follows NeMo-RL's intentionally split actor environments. It checks
the custom vLLM loader with `python-VllmGenerationWorker` and checks Megatron
Core plus Transformer Engine with `python-MegatronPolicyWorker`, matching this
recipe's unset `quant_cfg`. It does not attempt to install the conflicting
`vllm` and `mcore` extras into one environment. Override `VLLM_PYTHON_BIN` or
`MCORE_PYTHON_BIN` only when the immutable image uses different frozen-actor
wrapper names. Keep `--no-container-mount-home`, as `ray.sub` does, so the host
home does not hide the image's `/root` Python and uv cache used by those
wrappers. The OCI-HSG profile also overlays the pinned checkout onto
`/opt/nemo-rl`, the editable source path baked into the actor environments, so
the image cannot silently import its older NeMo-RL snapshot.

Repeat without `--test-only`, monitor for at least five minutes, and record the
smoke job ID and output in the local report.

## Trace and qualify

Task 3 owns the trace recipe and functional gate referenced by this experiment:

```text
examples/configs/recipes/llm/performance/
grpo-qwen3-30ba3b-4n4g-mxfp8-adaptive-trace.yaml
tests/functional/grpo_qwen3_30ba3b_mxfp8_adaptive_rollout_gb200.sh
```

Run every stage first with `ACTION=test-only`, then with `ACTION=submit`:

```bash
export CONTAINER_IMAGE=<immutable-sqsh>
export BOOTSTRAP_CONFIG_SHA256=<raw-bootstrap-json-sha256>

ACTION=test-only bash "$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout/run_ab.sh" trace
ACTION=submit bash "$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout/run_ab.sh" trace

ACTION=test-only bash "$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout/run_ab.sh" shmoo
ACTION=submit bash "$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout/run_ab.sh" shmoo
```

Each invocation gets a unique UTC-and-PID suite ID by default. A custom
`SUITE_ID` must be filesystem-safe and new. Suite directories, run directories,
raw logs, metadata, qualified artifacts, and parser outputs are create-only;
reruns refuse an existing destination. Keep failed outputs for diagnosis and
choose a new suite or experiment root rather than deleting or overwriting them.

Trace writes exact physical shapes and runs the custom vLLM `inventory`
stage. Shmoo uses the same 4n4g allocation and requires at least three repeats
before `promote` and byte-reproducing `validate --check`. Qualification fails
if both promoted tactic tables are empty.

If Qwen emits no eligible dense MXFP8 calls, the launcher writes the stable
`not-applicable.json`, skips both performance arms, and names Nemotron 3 Ultra
TP4 as the efficacy fallback. It never produces an empty optimized table.

## Alternating performance suite

After the qualified JSON is installed under the package config directory in a
new immutable container, invoke the launcher from the repository root. First
preflight each suite, then submit the same suite:

```bash
ACTION=test-only bash experiments/mxfp8_adaptive_rollout/run_ab.sh smoke-ab
ACTION=submit bash experiments/mxfp8_adaptive_rollout/run_ab.sh smoke-ab
ACTION=test-only bash experiments/mxfp8_adaptive_rollout/run_ab.sh ab
ACTION=submit bash experiments/mxfp8_adaptive_rollout/run_ab.sh ab
```

`smoke-ab` schedules one sequential matched `original`/`adaptive` pair and
sets exactly one measured step per arm. Production `ab` requires exactly three
sequential matched repeats: `original-r1`, `adaptive-r1`, `original-r2`,
`adaptive-r2`, `original-r3`, `adaptive-r3`. Every arm keeps its own cold
one-step in-job warmup, which the parser discards. Production then records the
launcher default of 20 measured steps per arm; `REPEATS` values other than
three are rejected for `ab`.

The W&B project is exactly `sna_mxfp8_kernel_test`. The baseline arm name is
`mxfp8-qwen-baseline-no-shmoo-trtllm-rN` and the shmoo-qualified arm name is
`mxfp8-qwen-shmoo-qualified-rN`, where `N` is the repeat number. Both use the
same model, seed, topology, container, custom vLLM overlay, and runtime trace;
the manifest is the intended sole A/B difference.

Both arms enable the same runtime dispatch tracing and use identical
container-visible paths. Fresh run-local shared directories are bind-mounted at
those paths so neither resolved config nor prior raw output is reused.
The required runtime evidence is the dispatch trace plus each arm's resolved
configuration, metadata, environment, and run log. The baseline's
`default_tactic_coverage.json` must show only TRTLLM runner defaults at
`tactic=-1`. The adaptive arm's `tactic_coverage.json` must report at least one
runtime tactic hit, all 106 qualified tactics hit, and its distinct-dispatch
fallback rate only for unqualified shapes. Fallback on a qualified shape,
zero tactic hits, or all-default/fallback adaptive execution is rejected.

The launcher writes these artifacts under the ignored shared
`$NEMO_RL_EXPERIMENT_ROOT/runs/<suite-id>/measured-{original,adaptive}-rN/`.
It also writes the suite submission manifest under
`$NEMO_RL_EXPERIMENT_ROOT/submissions/`; preserve that file with the Slurm job
IDs and raw logs as the handoff evidence.

Parse only measured logs:

```bash
python3 "$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout/parse_results.py" parse \
  --log <measured-original-run.log> \
  --log <measured-adaptive-run.log> \
  --json-output experiments/mxfp8_adaptive_rollout/report/results.json \
  --csv-output experiments/mxfp8_adaptive_rollout/report/results.csv
```

The stable summaries in `experiments/mxfp8_adaptive_rollout/report/results.json`
and `experiments/mxfp8_adaptive_rollout/report/results.csv` include independently
logged whole-run wall time, generation time, total step time, generated tokens,
generated-token throughput per GPU, runtime tactic hits, distinct-record
fallback rate, step, arm, repeat, source commits, container digest, config hash,
TP, and seed. Whole-run wall time is measured by monotonic launcher boundaries
and is repeated on measured step rows; it is not presented as per-step rollout
latency. Accept the shmoo arm only with correctness, higher median
output-token throughput, lower median generation time, and no total-step or
independently measured run-wall regression. Both output paths must be new; the
parser atomically creates them and refuses replacement.
