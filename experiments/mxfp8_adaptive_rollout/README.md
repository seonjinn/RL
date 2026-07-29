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
- vLLM commit: `bc5881924556fcf830f8158815d5a62cef0fbcba`.
- vLLM public version: `0.20.2`.
- FlashInfer: `0.6.8.post1`.
- Hardware: four OCI-HSG GB200 nodes, four GPUs per node.
- Model: `Qwen/Qwen3-30B-A3B`, vLLM TP1.
- Original arm: `VLLM_MXFP8_DENSE_CONFIG_FILE` is absent.
- Adaptive arm: only that key is added, with package-relative value
  `qwen3_30ba3b_tp1_v0202_qualified.json`.

The launcher compares both fully resolved Hydra configs after removing exactly
that nested key. It rejects differences in source commits, container digest,
checkpoint, topology, seed, or any other resolved setting.

## Stage the container

Build and push an immutable ARM64 image containing the exact custom vLLM
source, loader, and required package-relative JSON. The bootstrap JSON must be
pre-bundled in that immutable image; the launcher resolves and verifies the
same package path on every node instead of creating a node-local bootstrap.
Import it on OCI-HSG with the standard two-step scheduling protocol:

```bash
source experiments/mxfp8_adaptive_rollout/cluster/oci-hsg.env

sbatch --test-only \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --export=ALL,SOURCE_IMAGE=<registry/repository:immutable-tag>,OUTPUT_PREFIX=nemo_rl_mxfp8_adaptive,CONTAINER_DIR="${CONTAINER_ROOT}",SOURCE_COMMIT="nemo-$(git rev-parse HEAD)_vllm-bc5881924556fcf830f8158815d5a62cef0fbcba" \
  scripts/stage_enroot_image.sbatch

sbatch --parsable \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --export=ALL,SOURCE_IMAGE=<registry/repository:immutable-tag>,OUTPUT_PREFIX=nemo_rl_mxfp8_adaptive,CONTAINER_DIR="${CONTAINER_ROOT}",SOURCE_COMMIT="nemo-$(git rev-parse HEAD)_vllm-bc5881924556fcf830f8158815d5a62cef0fbcba" \
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
bash experiments/mxfp8_adaptive_rollout/run_ab.sh trace
```

## One-node/four-GPU smoke

Run the checked-in smoke before the 4n4g trace. It validates CUDA visibility,
NeMo-RL, Transformer Engine, Megatron Core, the exact custom vLLM source and
commit, vLLM/FlashInfer versions, loader import, and JSON model/TP/hash.

```bash
source experiments/mxfp8_adaptive_rollout/cluster/oci-hsg.env
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
  --wrap="srun --container-image=${CONTAINER_IMAGE} --container-mounts=/lustre:/lustre,/scratch:/scratch bash ${REPO_ROOT}/experiments/mxfp8_adaptive_rollout/smoke_container.sh"
```

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

ACTION=test-only bash experiments/mxfp8_adaptive_rollout/run_ab.sh trace
ACTION=submit bash experiments/mxfp8_adaptive_rollout/run_ab.sh trace

ACTION=test-only bash experiments/mxfp8_adaptive_rollout/run_ab.sh shmoo
ACTION=submit bash experiments/mxfp8_adaptive_rollout/run_ab.sh shmoo
```

Each invocation gets a unique UTC-and-PID suite ID by default. A custom
`SUITE_ID` must be filesystem-safe and new. Suite directories, run directories,
raw logs, metadata, qualified artifacts, and parser outputs are create-only;
reruns refuse an existing destination. Keep failed outputs for diagnosis and
choose a new suite or experiment root rather than deleting or overwriting them.

Trace writes exact physical shapes and runs the custom vLLM `inventory`
stage. Shmoo uses the same 4n4g allocation and requires at least three repeats
before `promote` and byte-reproducing `validate --check`.

If Qwen emits no eligible dense MXFP8 calls, the launcher writes the stable
`not-applicable.json`, skips both performance arms, and names Nemotron 3 Ultra
TP4 as the efficacy fallback. It never produces an empty optimized table.

## Alternating performance suite

After the qualified JSON is installed under the package config directory in a
new immutable container, smoke that JSON and run:

```bash
export CONTAINER_IMAGE=<qualified-immutable-sqsh>
export QUALIFIED_CONFIG_SHA256=<raw-qualified-json-sha256>
export REPEATS=3

ACTION=test-only bash experiments/mxfp8_adaptive_rollout/run_ab.sh ab
ACTION=submit bash experiments/mxfp8_adaptive_rollout/run_ab.sh ab
```

The schedule is matched original/adaptive warmups followed by
`original, adaptive` for each measured repeat. `REPEATS` values below three
are rejected. Every run writes `resolved_config.json`, `metadata.json`,
`runtime.env`, and `run.log` before or during the launch under the ignored
shared experiment root.

Parse only measured logs:

```bash
python3 experiments/mxfp8_adaptive_rollout/parse_results.py parse \
  --log <measured-original-run.log> \
  --log <measured-adaptive-run.log> \
  --json-output experiments/mxfp8_adaptive_rollout/report/results.json \
  --csv-output experiments/mxfp8_adaptive_rollout/report/results.csv
```

The stable summaries include rollout wall time, generation time, total step
time, generated tokens, generated-token throughput per GPU, step, arm, repeat,
source commits, container digest, config hash, TP, and seed. Both output paths
must be new; the parser atomically creates them and refuses replacement.
