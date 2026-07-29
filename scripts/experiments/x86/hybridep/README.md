# x86 HybridEP wheel build

`build_deepep_wheel.sbatch` builds the exact DeepEP `hybrid-ep` source in a
one-GPU allocation and publishes an immutable wheel only after extension
imports pass.

Use `GPU_ARCH=9.0` for H100 and `GPU_ARCH=10.0` for B200. Multi-node HybridEP
is always enabled. The default transport is the DOCA/RDMA path; set
`HYBRID_EP_TRANSPORT=nixl` together with `NIXL_HOME` and `UCX_HOME` only when
the container provides those dependencies.

For CW-DFW H100, source the checked-in hardware profile before building or
submitting. It pins the Hopper SM90 build, the eight-GPU NVLink domain,
non-MNNVL transport topology, the combine-chunk workaround used by the
Megatron-Bridge performance harness, and the CW account/partition defaults:

```bash
source scripts/experiments/x86/hybridep/clusters/cw-dfw-h100.env
```

The model recipe still owns model parallelism and dispatcher selection. The
Qwen3-30B-A3B HybridEP overlay inherits the baseline 4n8g performance recipe
and changes only the flex/HybridEP dispatcher settings. This keeps H100 and
B200 A/B comparisons matched.

From a clean, pushed NeMo-RL checkout on the target cluster:

```bash
export DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685
export CONTAINER=/lustre/absolute/path/nemo_rl_nightly.sqsh
export OUTPUT_DIR=/lustre/absolute/path/deepep-wheels
export GPU_ARCH=9.0  # supplied by cw-dfw-h100.env on CW

sbatch --test-only \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --export=ALL \
  scripts/experiments/x86/hybridep/build_deepep_wheel.sbatch

sbatch --parsable \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --output="${OUTPUT_DIR}/build-%j.out" \
  --export=ALL \
  scripts/experiments/x86/hybridep/build_deepep_wheel.sbatch
```

By default, source checkout, compilation, and import validation run below the
allocation's node-local `SLURM_TMPDIR`; only the final wheel, checksum, and
metadata are written to `OUTPUT_DIR`. Set `BUILD_ROOT` only when a retained
shared build tree is explicitly needed.

The job creates:

- `<commit>-sm<arch>-<job-id>/<wheel>.whl`;
- an adjacent SHA256 file;
- `metadata.env` with source, architecture, transport, container, and job
  provenance.

The build directory is removed only after success. A failed build directory is
retained under `BUILD_ROOT` (node-local by default) for bounded diagnosis.

## Prepare a version-matched Ray driver environment

When the nightly image's bundled Ray differs from the repository lock, prepare
one shared environment before launching the A/B jobs. The GRPO launcher can
then use the same Ray executable for the head, workers, and driver:

```bash
export CONTAINER=/lustre/absolute/path/nemo_rl_nightly.sqsh
export DRIVER_VENV=/lustre/absolute/path/hybridep-x86/driver-venv
export UV_CACHE_DIR=/lustre/absolute/path/hybridep-x86/uv-cache

scripts/experiments/x86/hybridep/submit_driver_venv.sh
```

After the preparation job completes, export both variables to the same path
for every paired run:

```bash
export DRIVER_VENV=/lustre/absolute/path/hybridep-x86/driver-venv
export RAY_VENV="${DRIVER_VENV}"
```

`submit_grpo.sh` rejects a different `RAY_VENV`, so the cluster cannot start
with one Ray release while the driver imports another.

For a matched all-to-all/HybridEP pair, prepare this environment once and
reuse it for both submissions. The x86 profiles set
`NRL_FORCE_REBUILD_VENVS=false`, which replaces per-actor `venvs.py` cache
isolation with the prepared shared driver/Ray environment and shared Lustre UV
cache. Do not allow concurrent actor source rebuilds for either arm; rebuild
only through the single preparation job before the pair is submitted.
