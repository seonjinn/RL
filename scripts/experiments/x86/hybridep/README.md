# x86 HybridEP wheel build

`build_deepep_wheel.sbatch` builds the exact DeepEP `hybrid-ep` source in a
one-GPU allocation and publishes an immutable wheel only after extension
imports pass.

Use `GPU_ARCH=9.0` for H100 and `GPU_ARCH=10.0` for B200. Multi-node HybridEP
is always enabled. The default transport is the DOCA/RDMA path; set
`HYBRID_EP_TRANSPORT=nixl` together with `NIXL_HOME` and `UCX_HOME` only when
the container provides those dependencies.

From a clean, pushed NeMo-RL checkout on the target cluster:

```bash
export DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685
export CONTAINER=/lustre/absolute/path/nemo_rl_nightly.sqsh
export OUTPUT_DIR=/lustre/absolute/path/deepep-wheels
export BUILD_ROOT=/lustre/absolute/path/build-cache
export GPU_ARCH=9.0

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

The job creates:

- `<commit>-sm<arch>-<job-id>/<wheel>.whl`;
- an adjacent SHA256 file;
- `metadata.env` with source, architecture, transport, container, and job
  provenance.

The build directory is removed only after success. A failed build directory is
retained under `BUILD_ROOT` for bounded diagnosis.
