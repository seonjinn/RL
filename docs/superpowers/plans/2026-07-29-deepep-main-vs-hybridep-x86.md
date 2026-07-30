# DeepEP Main Versus HybridEP on x86 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run reproducible all-to-all, standard DeepEP main, and HybridEP
comparisons on CW-DFW H100 and GCP-NRT B200 using each performance recipe's
native topology, then publish performance, numerical, storage, and padding
results.

**Architecture:** Keep f725 as the repository's canonical HybridEP dependency
and inject immutable branch-specific DeepEP wheels through a per-run overlay.
Add one Megatron-LM cross-version import compatibility patch, generalize the
x86 build and launch tooling by explicit variant, and give all three
Qwen3-30B arms the same NCCL 2.30.4 runtime. Resolve recipe topology before
submission, use it as the
training-config source of truth, and reject scheduler mismatches instead of
injecting `cluster.*` overrides. Store results in a validated Pages schema and
generate the concise HTML report from that schema.

**Tech Stack:** Bash, SLURM/pyxis, enroot, uv, PyTorch 2.11, NCCL 2.30.4,
DeepEP, Megatron-Core, Megatron-Bridge, NeMo-RL, pytest, JSON, HTML.

## Global Constraints

- Standard DeepEP is
  `main@dd758caf451848bd150e1046af3d0a73e5fff38d`.
- HybridEP is
  `hybrid-ep@f725d29699f5bda9ba789456bb9579af69844685`.
- All three Qwen3-30B benchmark arms use NCCL 2.30.4 from the same staged
  nightly or the same
  immutable `nvidia-nccl-cu13==2.30.4` overlay.
- Qwen3-30B-A3B uses the resolved performance topology of 4 nodes and 8 GPUs
  per node for all-to-all, DeepEP, and HybridEP.
- The launcher never overrides `cluster.num_nodes`, `cluster.gpus_per_node`,
  or `cluster.segment_size`; dispatcher fields are the only model-execution
  settings that differ across the three arms.
- SLURM `--segment=4` is scheduler placement metadata. It must not change the
  recipe's resolved `cluster.segment_size: null`.
- Keep the repository's canonical DeepEP dependency pin at f725.
- Do not modify `nemo_rl/utils/venvs.py`.
- All cluster artifacts live under `/lustre`; do not create environments,
  caches, checkpoints, wheels, containers, or logs under `/home`.
- Add a failing test and observe the expected failure before every production
  behavior change.
- Every source commit is signed off, pushed to a personal fork, and pulled
  with recursive submodules before job submission.
- Every job uses `sbatch --test-only` and is monitored for at least five
  minutes.

---

### Task 1: Make Megatron-Core accept both DeepEP export layouts

**Files:**

- Create:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_fused_a2a_deepep_imports.py`
- Modify:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py`

**Interfaces:**

- Consumes: dd758 exports `Buffer` and `EventOverlap` at package top level and
  `EventHandle` through `deep_ep.utils`; f725 has the same three usable
  locations.
- Produces: importing `fused_a2a` sets `HAVE_DEEP_EP=True` for both exact
  layouts without changing the HybridEP import path.

- [ ] **Step 1: Create and push an isolated Megatron-LM branch**

```bash
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
git status --short
git switch -c sna/deepep-main-dd758-compat-20260729
git remote set-url origin git@github.com:seonjinn/Megatron-LM.git
```

- [ ] **Step 2: Write a failing import-layout test**

The test must load `fused_a2a.py` with real module imports while replacing
only `deep_ep` and `deep_ep.utils` in `sys.modules`. Parameterize two layouts:
dd758 with `EventOverlap` only at top level, and f725 with top-level
`EventOverlap` plus `EventHandle` in `deep_ep.utils`. Assert
`module.HAVE_DEEP_EP is True`.

```python
@pytest.mark.parametrize("layout", ("dd758_main", "f725_hybrid_ep"))
def test_fused_a2a_accepts_supported_deepep_exports(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    deep_ep = ModuleType("deep_ep")
    deep_ep.Buffer = FakeBuffer
    deep_ep.EventOverlap = FakeEventOverlap
    utils = ModuleType("deep_ep.utils")
    utils.EventHandle = FakeEventHandle
    if layout == "f725_hybrid_ep":
        utils.EventOverlap = FakeEventOverlap
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
    monkeypatch.setitem(sys.modules, "deep_ep.utils", utils)

    module = load_fused_a2a_under_unique_name(layout)

    assert module.HAVE_DEEP_EP is True
```

- [ ] **Step 3: Run the new test and observe the dd758 failure**

Run inside the approved Megatron-LM container:

```bash
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/transformer/moe/test_fused_a2a_deepep_imports.py
```

Expected result: the dd758 case fails because the current code imports
`EventOverlap` from `deep_ep.utils` and sets `HAVE_DEEP_EP=False`.

- [ ] **Step 4: Apply the minimal cross-version import**

```python
try:
    from deep_ep import Buffer, EventOverlap
    from deep_ep.utils import EventHandle

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False
```

- [ ] **Step 5: Format and verify the focused MCore change**

```bash
uv run isort \
  megatron/core/transformer/moe/fused_a2a.py \
  tests/unit_tests/transformer/moe/test_fused_a2a_deepep_imports.py
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/transformer/moe/test_fused_a2a_deepep_imports.py
git diff --check
```

Expected result: both parameterized cases pass and the diff is clean.

- [ ] **Step 6: Commit and push Megatron-LM**

```bash
git add \
  megatron/core/transformer/moe/fused_a2a.py \
  tests/unit_tests/transformer/moe/test_fused_a2a_deepep_imports.py
git commit -s -m "fix(moe): support DeepEP main event exports"
git push -u origin sna/deepep-main-dd758-compat-20260729
```

Record the exact Megatron-LM commit for Task 4.

### Task 2: Stage and validate the exact NCCL runtime

**Files:**

- Copy from the container skill:
  `scripts/experiments/x86/hybridep/stage_enroot_image.sbatch`
- Create:
  `scripts/experiments/x86/hybridep/stage_nccl_wheel.sbatch`
- Modify:
  `tests/unit/tools/test_hybridep_x86_contract.py`

**Interfaces:**

- Consumes: mutable NeMo-RL image
  `nvcr.io/nvidian/nemo-rl:nightly`.
- Produces: immutable nightly `.sqsh` plus, when necessary, an immutable
  `nvidia-nccl-cu13==2.30.4` wheel, SHA256, and metadata on each cluster.

- [ ] **Step 1: Add failing source-contract tests**

Add tests that require the image staging script to produce immutable output
and metadata, and require the NCCL staging script to pin:

```bash
NCCL_PACKAGE=nvidia-nccl-cu13
NCCL_VERSION=2.30.4
```

The test must also require `pip download --no-deps`, SHA256 generation,
refusal to overwrite an existing artifact directory, and `/lustre` path
validation.

- [ ] **Step 2: Run the contract test and observe missing-script failures**

```bash
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_x86_contract.py \
  -k 'stage_enroot or stage_nccl'
```

Expected result: fail because both staged runtime contracts are absent.

- [ ] **Step 3: Copy the immutable image staging script**

```bash
cp \
  /Users/sna/.claude/plugins/marketplaces/e2etrain-container-image-skill/plugins/e2etrain/skills/stage-training-containers/scripts/stage_enroot_image.sbatch \
  scripts/experiments/x86/hybridep/stage_enroot_image.sbatch
```

- [ ] **Step 4: Implement the NCCL wheel staging job**

The job runs inside the selected container, downloads one exact wheel into a
job-local directory, validates its filename and package metadata, calculates
SHA256, writes `metadata.env`, and atomically publishes:

```text
${OUTPUT_DIR}/nvidia-nccl-cu13-2.30.4-${SLURM_JOB_ID}/
```

The metadata fields are:

```text
package=nvidia-nccl-cu13
version=2.30.4
wheel=${OUTPUT_DIR}/nvidia-nccl-cu13-2.30.4-${SLURM_JOB_ID}/nvidia_nccl_cu13-2.30.4-py3-none-manylinux_2_27_x86_64.whl
wheel_sha256=${wheel_sha256}
container=${CONTAINER}
container_sha256=${container_sha256}
slurm_job_id=${SLURM_JOB_ID}
staged_at=${staged_at}
```

- [ ] **Step 5: Verify the staging scripts**

```bash
bash -n scripts/experiments/x86/hybridep/stage_enroot_image.sbatch
bash -n scripts/experiments/x86/hybridep/stage_nccl_wheel.sbatch
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_x86_contract.py \
  -k 'stage_enroot or stage_nccl'
```

Expected result: all focused tests pass.

### Task 3: Generalize DeepEP wheel build and runtime overlay

**Files:**

- Modify:
  `scripts/experiments/x86/hybridep/build_deepep_wheel.sbatch`
- Modify:
  `scripts/experiments/oci-hsg/hybridep/render_deepep_setup_command.sh`
- Modify:
  `tests/unit/tools/test_hybridep_x86_contract.py`

**Interfaces:**

- Consumes: `DEEPEP_VARIANT=deepep|hybridep`, exact DeepEP SHA, immutable NCCL
  wheel and checksum, selected GPU architecture, and immutable container.
- Produces: immutable arm-specific wheel metadata and a setup command that
  installs both wheels into `/tmp/nemo-rl-deepep-*` and validates the correct
  API from the Ray runtime Python.

- [ ] **Step 1: Add failing builder tests for both variants**

Require this mapping:

```bash
case "${DEEPEP_VARIANT}" in
  deepep) DEEPEP_BRANCH=main ;;
  hybridep) DEEPEP_BRANCH=hybrid-ep ;;
esac
```

The standard DeepEP probe must import `deep_ep._C`, `Buffer`,
`ElasticBuffer`, `EventOverlap`, and `EventHandle`. The HybridEP probe must
import `deep_ep_cpp`, `hybrid_ep_cpp`, `Buffer`, and `HybridEPBuffer`.
Require the builder to reject an invalid variant and branch/SHA mismatch.

- [ ] **Step 2: Add failing renderer tests for a matched NCCL overlay**

Require the setup command to:

```bash
UV_NO_CONFIG=1 uv pip install --python "${runtime_python}" \
  --target "${overlay}" --reinstall --no-deps --no-index \
  "${nccl_wheel}" "${deepep_wheel}"
```

Require SHA256 checks for both wheels and require the probe to print:

```text
DEEPEP_RUNTIME_VARIANT
DEEPEP_RUNTIME_VERSION
DEEPEP_RUNTIME_PATHS
DEEPEP_RUNTIME_NCCL
```

- [ ] **Step 3: Run and observe the expected HybridEP-only failures**

```bash
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_x86_contract.py \
  -k 'wheel_build or setup_probe'
```

- [ ] **Step 4: Implement variant-specific checkout, build, and probes**

Keep HybridEP-only variables unset in the `deepep` build. For dd758, install
the staged NCCL wheel into a build overlay and export:

```bash
export PYTHONPATH="${build_overlay}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${build_overlay}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="${build_overlay}/nvidia/nccl/include:/usr/local/cuda/include/cccl"
```

Publish `deepep_variant`, `deepep_branch`, `deepep_commit`,
`deepep_wheel_sha256`, `nccl_version`, `nccl_wheel_sha256`, GPU architecture,
container SHA256, and build job ID.

- [ ] **Step 5: Verify shell syntax and focused contracts**

```bash
bash -n scripts/experiments/x86/hybridep/build_deepep_wheel.sbatch
bash -n scripts/experiments/oci-hsg/hybridep/render_deepep_setup_command.sh
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_x86_contract.py \
  -k 'wheel_build or setup_probe'
```

### Task 4: Add the standard DeepEP launcher and model profiles

**Files:**

- Modify:
  `scripts/experiments/oci-hsg/hybridep/submit_grpo.sh`
- Create:
  `scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86-deepep.env`
- Create:
  `scripts/experiments/oci-hsg/hybridep/models/qwen3-235b-16n8g-x86-deepep.env`
- Create:
  `scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n8g-sync-x86-deepep.env`
- Modify:
  `scripts/experiments/oci-hsg/hybridep/README.md`
- Modify:
  `scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86.env`
- Modify:
  `scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86-hybridep.env`
- Modify:
  `scripts/experiments/x86/hybridep/README.md`
- Modify:
  `tests/unit/tools/test_hybridep_submit_grpo.py`
- Modify:
  `tests/unit/tools/test_hybridep_x86_contract.py`

**Interfaces:**

- Consumes: branch-specific DeepEP and NCCL artifact metadata plus a matched
  all-to-all recipe.
- Produces: `DISPATCHER_MODE=deepep` with `flex`, backend `deepep`, 20 SMs,
  exact runtime provenance, and the same NCCL 2.30.4 overlay as HybridEP.

- [ ] **Step 1: Add a failing launcher behavior test**

```python
def test_deepep_dispatcher_applies_standard_backend(tmp_path: Path) -> None:
    args = _run_launcher(
        tmp_path,
        dispatcher_mode="deepep",
        extra_env=standard_deepep_artifact_env(tmp_path),
    )

    assert "policy.megatron_cfg.moe_token_dispatcher_type=flex" in args
    assert "++policy.megatron_cfg.moe_flex_dispatcher_backend=deepep" in args
    assert "++policy.megatron_cfg.moe_deepep_num_sms=20" in args
    assert not any("moe_hybridep_num_sms" in arg for arg in args)
```

Add separate failures for invalid variant, branch/variant mismatch, missing
NCCL wheel, wrong NCCL version, wrong checksum, and metadata mismatch.

- [ ] **Step 2: Run and observe the unsupported-mode failure**

```bash
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  -k deepep
```

- [ ] **Step 3: Implement `DISPATCHER_MODE=deepep`**

Extend mode validation to `recipe | deepep | hybridep`. Add:

```bash
if [[ "${DISPATCHER_MODE}" == "deepep" ]]; then
  driver_args+=(
    policy.megatron_cfg.moe_token_dispatcher_type=flex
    ++policy.megatron_cfg.moe_flex_dispatcher_backend=deepep
    ++policy.megatron_cfg.moe_deepep_num_sms=20
  )
fi
```

Validate artifact metadata before submission. Export the overlay's NCCL
library directory before Ray starts:

```bash
LD_LIBRARY_PATH="${DEEPEP_OVERLAY}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH
```

Record variant, branch, both wheel paths and hashes, NCCL version, resolved
config hash, and runtime probe fields in `submission.env`.

- [ ] **Step 4: Add reusable DeepEP model profiles**

Each new profile uses its existing `-alltoall.yaml` recipe, sets
`DISPATCHER_MODE=deepep`, and sets:

```bash
DEEPEP_VARIANT=deepep
DEFAULT_DEEPEP_COMMIT=dd758caf451848bd150e1046af3d0a73e5fff38d
REQUIRE_DEEPEP_WHEEL=true
REQUIRE_NCCL_WHEEL=true
```

Keep node, GPU, segment, step, and time values identical to its all-to-all and
HybridEP peer.

- [ ] **Step 5: Verify launcher, profiles, and unchanged venv code**

```bash
bash -n scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  tests/unit/tools/test_hybridep_x86_contract.py
git diff 4c14b04266a0b3ed8ec6121fae387d77d869bf1d \
  -- nemo_rl/utils/venvs.py
```

Expected result: tests pass and `venvs.py` has no diff.

### Task 5: Upgrade the structured report and HTML renderer

**Files in `/Users/sna/nemo-rl_release_perf_investigator`:**

- Modify:
  `public/scripts/hybridep_x86/collect_workload_metrics.py`
- Modify:
  `public/scripts/hybridep_x86/render_workload_report.py`
- Modify:
  `public/hybridep-x86-20260728/workload-metrics.json`
- Modify:
  `public/hybridep-x86-20260728/run-status.json`
- Modify:
  `public/hybridep-x86-20260728/index.html`
- Modify:
  `tests/test_hybridep_x86_workload_metrics.py`

**Interfaces:**

- Consumes: per-run logs, `submission.env`, storage measurements, packing
  diagnostics, and model/hardware metadata.
- Produces: schema version 3 with `alltoall`, `deepep`, and `hybridep` arms and
  a concise generated HTML page.

- [ ] **Step 1: Add failing schema and merge-preservation tests**

Require every executed arm to retain:

```python
REQUIRED_PROVENANCE = {
    "nemo_rl_commit",
    "bridge_commit",
    "megatron_lm_commit",
    "deepep_branch",
    "deepep_commit",
    "deepep_wheel_sha256",
    "container_sha256",
    "resolved_config_sha256",
    "dispatcher_backend",
    "runtime_nccl_version",
    "job_id",
    "state",
    "log_path",
}
```

Recollection must merge parsed metrics into an arm without deleting
provenance, storage, quality, or padding objects.

- [ ] **Step 2: Add failing quality and storage parser tests**

Extend synthetic logs with reward, generation KL, entropy, gradient norm,
validation accuracy, response length, and a non-finite line. Assert per-step
values, counts, means, deltas, and explicit non-finite detection.

Require normalized storage:

```json
{
  "bytes": 1030750208,
  "path": "/lustre/example/run",
  "scope": "run_root",
  "method": "du --bytes --summarize",
  "measured_at": "2026-07-29T12:00:00-0700"
}
```

- [ ] **Step 3: Run the report tests and observe schema-version failures**

```bash
cd /Users/sna/nemo-rl_release_perf_investigator
python3 -m pytest -q tests/test_hybridep_x86_workload_metrics.py
```

- [ ] **Step 4: Implement schema version 3 and lossless merging**

Migrate legacy `baseline` entries to `alltoall`. Allow not-yet-run arms with
an explicit `state="not_run"` while requiring full provenance for submitted
or completed arms. Generate `run-status.json` from the same object rather than
maintaining manual status text.

- [ ] **Step 5: Render concise charts and evidence**

Render per model and hardware:

- Policy, LogProb, and end-to-end throughput;
- corresponding time;
- reward, KL, and validation evidence;
- per-run, wheel, shared-cache, and checkpoint-excluded storage;
- HybridEP weighted padding, median, p95, and maximum;
- exact branch, commit, wheel, container, recipe, job, and measurement window.

- [ ] **Step 6: Verify the report**

```bash
python3 -m pytest -q tests/test_hybridep_x86_workload_metrics.py
python3 public/scripts/hybridep_x86/render_workload_report.py \
  --input public/hybridep-x86-20260728/workload-metrics.json \
  --output public/hybridep-x86-20260728/index.html
python3 -m json.tool \
  public/hybridep-x86-20260728/workload-metrics.json >/dev/null
python3 -m html.parser \
  public/hybridep-x86-20260728/index.html >/dev/null
git diff --check
```

### Task 6: Lock launcher topology to the resolved performance recipe

**Files:**

- Create:
  `scripts/experiments/oci-hsg/hybridep/resolve_recipe_topology.py`
- Modify:
  `scripts/experiments/oci-hsg/hybridep/submit_grpo.sh`
- Modify:
  `scripts/experiments/oci-hsg/hybridep/README.md`
- Test:
  `tests/unit/tools/test_hybridep_submit_grpo.py`
- Test:
  `tests/unit/tools/test_hybridep_x86_contract.py`

**Interfaces:**

- Consumes: a performance recipe path accepted by
  `nemo_rl.utils.config.load_config`, plus scheduler values
  `NUM_ACTOR_NODES`, `GPUS_PER_NODE`, and `SEGMENT_SIZE`.
- Produces:
  `resolve_recipe_topology.resolve_topology(config_path: Path) -> RecipeTopology`
  with positive `num_nodes`, positive `gpus_per_node`, optional
  `config_segment_size`, and a deterministic resolved-config SHA256.
- Produces: a launcher preflight that rejects scheduler node/GPU mismatches
  and constructs `examples/run_grpo.py` arguments without any `cluster.*`
  override.

- [ ] **Step 1: Add failing topology resolver and launcher tests**

Add a `hybridep_artifact_env(shared_root: Path) -> dict[str, str]` test helper
that creates an f725 wheel, an NCCL 2.30.4 wheel, and these two metadata files:

```python
def hybridep_artifact_env(shared_root: Path) -> dict[str, str]:
    env = _x86_shared_env(shared_root)
    deepep_wheel = Path(env["DEEPEP_WHEEL"])
    deepep_wheel.write_bytes(b"hybridep-wheel")
    nccl_dir = shared_root / "nccl-artifact"
    nccl_dir.mkdir()
    nccl_wheel = nccl_dir / "nvidia_nccl_cu13-2.30.4.whl"
    _write_nccl_wheel(nccl_wheel)
    deepep_sha = subprocess.check_output(
        ["sha256sum", str(deepep_wheel)], text=True
    ).split()[0]
    nccl_sha = subprocess.check_output(
        ["sha256sum", str(nccl_wheel)], text=True
    ).split()[0]
    _write_metadata(
        deepep_wheel.parent / "metadata.env",
        {
            "deepep_variant": "hybridep",
            "deepep_branch": "hybrid-ep",
            "deepep_commit": DEEPEP_COMMIT,
            "deepep_wheel_sha256": deepep_sha,
            "nccl_version": NCCL_VERSION,
            "nccl_wheel_sha256": nccl_sha,
            "wheel": str(deepep_wheel),
            "wheel_sha256": deepep_sha,
        },
    )
    _write_metadata(
        nccl_dir / "metadata.env",
        {
            "package": "nvidia-nccl-cu13",
            "version": NCCL_VERSION,
            "wheel": str(nccl_wheel),
            "wheel_sha256": nccl_sha,
        },
    )
    return {
        **env,
        "DEEPEP_VARIANT": "hybridep",
        "NCCL_WHEEL": str(nccl_wheel),
    }
```

Add resolver CLI coverage:

```python
@pytest.mark.parametrize(
    "recipe_name",
    (
        "grpo-qwen3-30ba3b-4n8g-alltoall.yaml",
        "grpo-qwen3-30ba3b-4n8g.yaml",
    ),
)
def test_qwen30_performance_recipe_resolves_native_topology(
    recipe_name: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RECIPE_TOPOLOGY_RESOLVER),
            str(PERFORMANCE_RECIPE_DIR / recipe_name),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    nodes, gpus, config_segment, resolved_sha = result.stdout.strip().split("\t")

    assert (nodes, gpus, config_segment) == ("4", "8", "null")
    assert len(resolved_sha) == 64
```

Add launcher behavior coverage:

```python
def test_launcher_preserves_recipe_topology(
    tmp_path: Path,
    lustre_tmp_path: Path,
) -> None:
    actor_venv_dir = _x86_actor_venv_dir(lustre_tmp_path)
    _write_prefetched_actor_pythons(actor_venv_dir)
    args = _run_launcher(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            **hybridep_artifact_env(lustre_tmp_path),
            "NEMO_RL_VENV_DIR": str(actor_venv_dir),
        },
    )

    assert not any(arg.startswith("cluster.") for arg in args)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"NUM_ACTOR_NODES": "2"}, "scheduler nodes 2 != recipe nodes 4"),
        ({"GPUS_PER_NODE": "4"}, "scheduler GPUs per node 4 != recipe GPUs per node 8"),
    ),
)
def test_qwen30_launcher_rejects_scheduler_topology_mismatch(
    tmp_path: Path,
    lustre_tmp_path: Path,
    override: dict[str, str],
    message: str,
) -> None:
    actor_venv_dir = _x86_actor_venv_dir(lustre_tmp_path)
    _write_prefetched_actor_pythons(actor_venv_dir)
    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            **hybridep_artifact_env(lustre_tmp_path),
            "NEMO_RL_VENV_DIR": str(actor_venv_dir),
            **override,
        },
    )

    assert result.returncode == 2
    assert message in result.stderr
```

- [ ] **Step 2: Run the focused tests and observe the expected failures**

```bash
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  tests/unit/tools/test_hybridep_x86_contract.py
```

Expected: the resolver import is missing, launcher arguments still contain
three `cluster.*` overrides, and `NUM_ACTOR_NODES=2` is accepted.

- [ ] **Step 3: Implement the typed recipe topology resolver**

Use the repository's real inheritance and interpolation path:

```python
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@dataclass(frozen=True)
class RecipeTopology:
    num_nodes: int
    gpus_per_node: int
    config_segment_size: int | None
    resolved_config_sha256: str


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def resolve_topology(config_path: Path) -> RecipeTopology:
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(config_path), resolve=True)
    if not isinstance(resolved, dict) or not isinstance(resolved.get("cluster"), dict):
        raise ValueError("resolved recipe must contain a cluster mapping")
    cluster = cast(dict[str, Any], resolved["cluster"])
    segment = cluster.get("segment_size")
    if segment is not None:
        segment = _positive_int(segment, "cluster.segment_size")
    encoded = json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return RecipeTopology(
        num_nodes=_positive_int(cluster.get("num_nodes"), "cluster.num_nodes"),
        gpus_per_node=_positive_int(
            cluster.get("gpus_per_node"), "cluster.gpus_per_node"
        ),
        config_segment_size=segment,
        resolved_config_sha256=sha256(encoded).hexdigest(),
    )
```

The CLI prints exactly four tab-separated fields: nodes, GPUs per node,
`null` or the config segment size, and resolved-config SHA256. It exits
nonzero with a concise message for an invalid recipe.

- [ ] **Step 4: Enforce preflight equality and remove training overrides**

In `submit_grpo.sh`, call the resolver with the prepared driver Python when
`DRIVER_VENV` is set, otherwise with the current Python. Compare
`NUM_ACTOR_NODES` and `GPUS_PER_NODE` to the resolved values before FairShare
inspection. Remove these entries from `driver_args`:

```bash
"cluster.num_nodes=${NUM_ACTOR_NODES}"
"cluster.gpus_per_node=${GPUS_PER_NODE}"
"cluster.segment_size=${SEGMENT_SIZE}"
```

Keep `--nodes`, `--gres`, and `--segment` in `sbatch_args`. Record both
`config_segment_size` and `scheduler_segment_size` in `submission.env`, along
with `resolved_config_sha256`.

- [ ] **Step 5: Match the Qwen3-30B runtime and replace two-node instructions**

Add `REQUIRE_NCCL_WHEEL=true` and `DEEPEP_VARIANT=hybridep` to the all-to-all
and HybridEP Qwen3-30B x86 profiles. Extend the profile contract test so the
all-to-all, HybridEP, and DeepEP-main arms all require the same NCCL 2.30.4
artifact.
The selected DeepEP wheel still differs by dispatcher arm, but its runtime
NCCL wheel path and checksum must match across the triplet.

Delete the `NUM_ACTOR_NODES=2` and `SEGMENT_SIZE=2` Qwen3-30B commands from
the README. Document a matched three-arm 4n8g gate that passes only:

```bash
MAX_STEPS=3
WANDB_ENABLED=False
NEMO_RL_HYBRIDEP_LOG_PACKING=0
```

The commands must not set `NUM_ACTOR_NODES`, `GPUS_PER_NODE`, or
`SEGMENT_SIZE`.

- [ ] **Step 6: Run focused and static verification**

```bash
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  tests/unit/tools/test_hybridep_x86_contract.py \
  tests/unit/tools/test_hybridep_default_8g_recipes.py
bash -n scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python \
  scripts/experiments/oci-hsg/hybridep/resolve_recipe_topology.py \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g-alltoall.yaml
git diff --check
```

Expected resolver output starts with `4<TAB>8<TAB>null<TAB>` and all tests
pass.

- [ ] **Step 7: Commit the topology lock**

```bash
git add \
  scripts/experiments/oci-hsg/hybridep/resolve_recipe_topology.py \
  scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/README.md \
  scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86.env \
  scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86-hybridep.env \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  tests/unit/tools/test_hybridep_x86_contract.py
git commit -s -m "fix: preserve performance recipe topology"
```

### Task 7: Update gitlinks, validate the source tree, and push

**Files:**

- Modify:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`
- Modify:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
- All NeMo-RL files from Tasks 2-4
- Add the implementation plan and amended spec commits as ancestors.

**Interfaces:**

- Consumes: the pushed Megatron-LM compatibility commit.
- Produces: clean pushed Megatron-Bridge and NeMo-RL commits with recursive
  gitlinks and focused verification evidence.

- [ ] **Step 1: Commit and push the Megatron-Bridge gitlink**

```bash
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git switch -c sna/deepep-main-dd758-compat-20260729
git add 3rdparty/Megatron-LM
git commit -s -m "build: update MCore for DeepEP main"
git push -u origin sna/deepep-main-dd758-compat-20260729
```

- [ ] **Step 2: Run complete focused NeMo-RL verification**

```bash
cd /Users/sna/Nemo-RL-HybridEP/.worktrees/hybridep-x86-b200-h100-20260728
bash -n scripts/experiments/x86/hybridep/stage_enroot_image.sbatch
bash -n scripts/experiments/x86/hybridep/stage_nccl_wheel.sbatch
bash -n scripts/experiments/x86/hybridep/build_deepep_wheel.sbatch
bash -n scripts/experiments/oci-hsg/hybridep/render_deepep_setup_command.sh
bash -n scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  tests/unit/tools/test_hybridep_x86_contract.py \
  tests/unit/tools/test_hybridep_default_8g_recipes.py
uv lock --check
git diff --check
git status --short
```

- [ ] **Step 3: Commit and push scoped NeMo-RL changes**

```bash
git add \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
  scripts/experiments/x86/hybridep \
  scripts/experiments/oci-hsg/hybridep \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  tests/unit/tools/test_hybridep_x86_contract.py
git commit -s -m "feat: compare DeepEP main with HybridEP on x86"
git push fork sna/hybridep-x86-b200-h100-20260728
```

Record the exact NeMo-RL, Bridge, and MCore SHAs.

### Task 8: Stage artifacts and pass allocated-GPU smoke gates

**Files:**

- Remote artifacts only under:
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/deepep-main-vs-hybridep-20260729`
- Remote artifacts only under:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/deepep-main-vs-hybridep-20260729`

**Interfaces:**

- Consumes: clean pushed source plus staging/build scripts.
- Produces: immutable H100 SM90 and B200 SM100 wheels for both variants,
  matched NCCL runtime, container provenance, and passing one-GPU import
  smokes.

- [ ] **Step 1: Pull and verify clean recursive checkouts**

On both clusters:

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
git status --short
git rev-parse HEAD
git submodule status --recursive
```

- [ ] **Step 2: Inspect FairShare and schedule tests**

Use `ssh gcp-nrt` for B200 and the established clean-shell CW connection for
H100. Record `sshare -l`, selected account, partition, container path, and
current jobs. Run `sbatch --test-only` before each image, NCCL, wheel, and
smoke submission.

- [ ] **Step 3: Stage the nightly and inspect its runtime**

Use:

```text
SOURCE_IMAGE=nvcr.io/nvidian/nemo-rl:nightly
OUTPUT_PREFIX=nemo_rl_nightly
SOURCE_COMMIT=$(git rev-parse HEAD)
```

After staging, run a one-GPU job that prints PyTorch, CUDA, NCCL, NVSHMEM,
DeepEP absence/presence, GPU name, compute capability, and container SHA256.

- [ ] **Step 4: Stage NCCL 2.30.4 when the nightly is older**

Submit `stage_nccl_wheel.sbatch`, monitor five minutes, then verify its
metadata and checksum. Use the same wheel path and checksum for DeepEP and
HybridEP on that hardware platform.

- [ ] **Step 5: Build both DeepEP variants per architecture**

For H100:

```text
GPU_ARCH=9.0
DEEPEP_VARIANT=deepep
DEEPEP_COMMIT=dd758caf451848bd150e1046af3d0a73e5fff38d
```

and:

```text
GPU_ARCH=9.0
DEEPEP_VARIANT=hybridep
DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685
```

Repeat with `GPU_ARCH=10.0` on B200. Verify wheel checksums, metadata, exact
source SHAs, and import logs.

- [ ] **Step 6: Pass one-GPU runtime smokes**

For dd758 assert:

```python
from deep_ep import Buffer, ElasticBuffer, EventOverlap
from deep_ep.utils import EventHandle
from megatron.core.transformer.moe import fused_a2a

assert fused_a2a.HAVE_DEEP_EP
assert not fused_a2a.HAVE_HYBRIDEP
assert torch.cuda.nccl.version() == (2, 30, 4)
```

For f725 assert `HAVE_DEEP_EP`, `HAVE_HYBRIDEP`, `HybridEPBuffer`, and the
same NCCL version. Preserve job IDs and bounded logs.

### Task 9: Run matched workloads, measure overhead, and publish

**Files:**

- Update the report files from Task 5.
- Preserve scripts and config snapshots under:
  `/Users/sna/nemo-rl_release_perf_investigator/public/scripts/hybridep_x86/`
  and
  `/Users/sna/nemo-rl_release_perf_investigator/public/configs/hybridep_x86/`.

**Interfaces:**

- Consumes: passing smoke artifacts and reusable model profiles.
- Produces: terminal job evidence, matched performance and quality metrics,
  storage/padding measurements, and the deployed Pages report.

- [ ] **Step 1: Submit the Qwen3-30B 4n8g three-arm gate**

Submit all-to-all, DeepEP, and HybridEP on H100 and B200 with:

```text
MAX_STEPS=3
NEMO_RL_HYBRIDEP_LOG_PACKING=0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Do not set `NUM_ACTOR_NODES`, `GPUS_PER_NODE`, or `SEGMENT_SIZE` in the
command environment. Require every `submission.env` to report recipe and
scheduler topology as 4 nodes and 8 GPUs per node, recipe segment null, and
the same scheduler segment. Use the same source, container, NCCL 2.30.4
runtime, model snapshot, topology, and run timestamp per hardware triplet.
Monitor at least five minutes and scan bounded logs.

- [ ] **Step 2: Gate and submit larger models**

After all three Qwen3-30B arms complete three steps with finite metrics, submit
matched Qwen3-235B and Nemotron3 Super three-step triplets at their existing
16n8g and 32n8g topologies. Verify zero missing checkpoint shards before
submission.

- [ ] **Step 3: Submit clean 20-step performance triplets**

For every three-step triplet that passes, submit matched 20-step arms with
padding diagnostics disabled. Use common completed steps 5-20 for reported
performance.

- [ ] **Step 4: Submit separate HybridEP padding diagnostics**

Use the same workload and f725 wheel with:

```text
NEMO_RL_HYBRIDEP_LOG_PACKING=1
NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS=4096
NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS=0
NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE=1
```

Do not use diagnostic timings in the clean performance comparison.

- [ ] **Step 5: Measure storage after terminal completion**

For each run, wheel, shared runtime, cache, and checkpoint scope, record:

```bash
du --bytes --summarize "${MEASURED_PATH}"
date '+%Y-%m-%dT%H:%M:%S%z'
```

Keep checkpoint bytes separate and excluded from dispatcher overhead.

- [ ] **Step 6: Extract and validate metrics**

Collect Policy, LogProb, and end-to-end time/TPS; reward, KL, entropy,
gradient norm, validation accuracy, response length, and non-finite events.
Compute ratio-of-sums throughput and all three matched deltas: DeepEP versus
all-to-all, HybridEP versus all-to-all, and HybridEP versus DeepEP. For
packing, compute weighted overhead, median, p95, and maximum.

- [ ] **Step 7: Regenerate, verify, commit, and push Pages**

```bash
cd /Users/sna/nemo-rl_release_perf_investigator
python3 -m pytest -q tests/test_hybridep_x86_workload_metrics.py
python3 public/scripts/hybridep_x86/render_workload_report.py \
  --input public/hybridep-x86-20260728/workload-metrics.json \
  --output public/hybridep-x86-20260728/index.html
python3 -m json.tool \
  public/hybridep-x86-20260728/workload-metrics.json >/dev/null
git diff --check
git add \
  public/hybridep-x86-20260728 \
  public/scripts/hybridep_x86 \
  public/configs/hybridep_x86 \
  tests/test_hybridep_x86_workload_metrics.py
git commit -s -m "report: compare DeepEP main and HybridEP on x86"
git push origin main
```

Verify that the Pages pipeline is created and that the deployed page shows
the exact measurement window, provenance, validity, storage scope, and
quality limitations.
