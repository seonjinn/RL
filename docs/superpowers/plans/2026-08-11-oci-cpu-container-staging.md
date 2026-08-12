# OCI CPU Container Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permit reproducible NeMo-RL nightly staging on OCI-HSG CPU data-mover nodes without changing production GPU job placement.

**Architecture:** Keep `batch` as the staging default and add a strict allowlist for `cpu` and `cpu_datamover`. The existing wrapper remains responsible for validation, scheduler rendering, atomic image publication, and provenance metadata; experiment launchers continue to require `batch`.

**Tech Stack:** Bash, SLURM, enroot, pytest, OCI-HSG `cpu_datamover`, immutable squashfs metadata and SHA256 validation.

## Global Constraints

- Accepted staging partitions are exactly `batch`, `cpu`, and `cpu_datamover`.
- The wrapper must not request GPUs, exclusive nodes, or memory. CPU staging
  requests 32 CPUs and four hours; the default `batch` path adds no CPU flag.
- Production experiment launchers remain restricted to `batch`.
- The image digest is `sha256:09509475e2efdef6f6bc32726f16b2cfbf238e7128246dbf27cb17d4472c401d`.
- The image source commit is `0e687e6d07623d780a4174310e92382ce738a8a2`.
- The expected squashfs SHA256 is `67f63772db4e11bdae16d646706aec0ec49a5fd2f7c400ee62875ab869cf49b1`.
- CUDA Graph warmup remains exactly three successful optimizer steps and checkpointing remains disabled.

---

### Task 1: Add CPU staging partition support

**Files:**
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py:770-813`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/stage_enroot_image.sbatch:35-96`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md:95-105`

**Interfaces:**
- Consumes: the existing `PARTITION` environment variable and wrapper-side `sbatch_command` array.
- Produces: a validated `PARTITION` value in `{batch,cpu,cpu_datamover}` that is forwarded as `--partition=<value>` without GPU flags.

- [x] **Step 1: Write the failing launcher tests**

Add these focused cases beside the existing staging tests:

```python
def test_stage_enroot_image_allows_cpu_datamover_without_gpu(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidian/nemo-rl:nightly",
        SOURCE_DIGEST="sha256:" + "a" * 64,
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_oci",
        CONTAINER_DIR=str(tmp_path / "containers"),
        PARTITION="cpu_datamover",
    )

    assert result.returncode == 0, result.stderr
    assert "--partition=cpu_datamover" in result.stdout
    assert "--gpus-per-node" not in result.stdout
    assert "--gres=" not in result.stdout


def test_stage_enroot_image_rejects_unapproved_partition(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidian/nemo-rl:nightly",
        SOURCE_DIGEST="sha256:" + "a" * 64,
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_oci",
        CONTAINER_DIR=str(tmp_path / "containers"),
        PARTITION="interactive",
    )

    assert result.returncode == 2
    assert "PARTITION must be one of: batch, cpu, cpu_datamover" in result.stderr
    assert "SBATCH:" not in result.stdout
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. pytest -q --confcutdir=tests/unit/experiments \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py::test_stage_enroot_image_allows_cpu_datamover_without_gpu \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py::test_stage_enroot_image_rejects_unapproved_partition
```

Expected: the CPU data-mover case fails with the current batch-only error, and the rejection-message case fails because the old message names only `batch`.

- [x] **Step 3: Implement the strict partition allowlist**

Replace the batch-only condition with:

```bash
case "${PARTITION}" in
  batch | cpu | cpu_datamover) ;;
  *)
    echo "PARTITION must be one of: batch, cpu, cpu_datamover" >&2
    exit 2
    ;;
esac
```

Do not add any resource flags to `sbatch_command`.

- [x] **Step 4: Document the staging-only CPU partition**

Add `PARTITION=cpu_datamover` to the OCI staging example and state that this setting applies only to image staging; production scope jobs still require `batch`.

- [x] **Step 5: Run focused and regression verification**

Run:

```bash
PYTHONPATH=. pytest -q --confcutdir=tests/unit/experiments \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
PYTHONPATH=. pytest -q --confcutdir=tests/unit/experiments \
  tests/unit/experiments/test_container_harness_hardening.py
ruff check tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
bash -n experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/stage_enroot_image.sbatch
git diff --check
```

Expected: 146 launcher tests and 63 hardening tests pass, lint and shell syntax pass, and no whitespace errors are reported.

- [x] **Step 6: Commit and push the implementation**

```bash
git add \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/stage_enroot_image.sbatch \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  docs/superpowers/plans/2026-08-11-oci-cpu-container-staging.md
git commit -s -m "fix: stage OCI containers without GPU allocation"
git push seonjinn experiment/thd-cg-hybrid-nemotron-main-20260806
```

### Task 2: Stage and attest the exact OCI runtime

**Files:**
- Reuse: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/create_source_snapshot.sh`
- Reuse: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/stage_enroot_image.sbatch`
- Reuse: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/validate_oci_container_runtime.sub`
- Local-only update: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/oci-hsg.env`

**Interfaces:**
- Consumes: the pushed outer commit, Bridge `91d904d7acc607af1efd962305581f990a30e213`, MCore `97cc30cb9d528fe838b893f0eae4fc9e3b184e70`, and the pinned nightly digest.
- Produces: an immutable OCI `.sqsh`, provenance metadata, SHA256, read-only staged runtime, GPU attestation JSON, and a populated local OCI profile.

- [ ] **Step 1: Refresh OCI source and create a new immutable snapshot**

On `oci-hsg-cs-001-vscode-02`, run `git pull --ff-only`, verify recursive submodule SHAs, and invoke `create_source_snapshot.sh` with `EXPECTED_NEMORL_SHA=$(git rev-parse HEAD)`, the exact Bridge SHA, and the exact MCore SHA.

- [ ] **Step 2: Test scheduler placement and submit staging**

Immediately before submission, inspect FairShare and run:

```bash
PARTITION=cpu_datamover \
SBATCH_TEST_ONLY=1 \
SOURCE_IMAGE=nvcr.io/nvidian/nemo-rl:nightly \
SOURCE_DIGEST=sha256:09509475e2efdef6f6bc32726f16b2cfbf238e7128246dbf27cb17d4472c401d \
SOURCE_COMMIT=0e687e6d07623d780a4174310e92382ce738a8a2 \
OUTPUT_PREFIX=nemo_rl_nightly_thd_cg_0e687e6d_20260809 \
CONTAINER_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-cg/containers \
ACCOUNT=coreai_dlalgo_nemorl \
experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/stage_enroot_image.sbatch
```

Repeat without `SBATCH_TEST_ONLY=1` only after the scheduler accepts the request.

- [ ] **Step 3: Monitor and verify immutable publication**

Monitor the job for at least five minutes and through terminal state. Verify metadata records the exact digest and source commit, then run `sha256sum` and require the expected squashfs SHA256. Do not use the stable symlink as the profile's container path.

- [ ] **Step 4: Build and attest the exact runtime**

Submit the existing `RUNTIME_PHASE=stage` job on the CPU partition selected by its wrapper, wait for `COMPLETED|0:0`, then submit `RUNTIME_PHASE=attest` on `batch` with four GPUs. Require exact outer, Bridge, MCore, TE, container, Python, uv, HybridEP, Mamba, causal-conv1d, CuPy, and grouped-linear evidence in the attestation JSON.

- [ ] **Step 5: Populate the OCI profile and render the smoke job**

Set the immutable container path and SHA256, exact runtime attestation path and producer job ID, immutable uv path, exact source SHAs, four GPUs per node, warmup three, and checkpointing disabled. Render the `moe_router_overlap_param_gather_false` five-step job with `TEST_ONLY=1` and verify six nodes by four GPUs, packed THD, HybridEP, `overlap_param_gather=false`, and no dependency or singleton.

- [ ] **Step 6: Submit and monitor the five-step diagnostic**

After `git pull --ff-only`, FairShare inspection, and `sbatch --test-only`, submit the leaf job. Monitor at least five minutes and through terminal state, then collect graph capture/replay/fallback coverage, NCCL/illegal-memory errors, reward, KL metrics, token probability error, and gradient norms.
