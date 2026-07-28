# NeMo-RL Main Adaptive MXFP8 Rollout Integration Plan

**Goal:** Make latest NeMo-RL main reproducibly consume the custom vLLM
0.20.2 branch, forward the single JSON configuration path through every Ray
layer, and measure original-versus-adaptive rollout performance with identical
workloads.

**Architecture:** Keep the custom vLLM source as the sole `vllm` dependency in
the locked NeMo-RL environment. Put the tactic JSON inside the custom vLLM
package and pass only `VLLM_MXFP8_DENSE_CONFIG_FILE` through
`policy.generation.vllm_cfg.env_vars`. Extend the existing vLLM source patch so
the configured key reaches vLLM's internal tensor-parallel Ray workers. Use the
exact latest-main Qwen3-30B-A3B 4n4g MXFP8 rollout as the primary integration
and performance workload. Trace and qualify Qwen TP1 shapes first; if the MoE
and ignored-projection execution path produces zero eligible dense MXFP8 calls,
record the workload as not applicable and use Nemotron 3 Ultra TP4 as the
secondary kernel-efficacy workload.

**Fixed inputs:**

- NeMo-RL base: `80555d3a0595ce3cf76f2ca1b2bf123339064556`
- NeMo-RL branch: `sna/mxfp8-adaptive-vllm-nemorl-main`
- vLLM fork: `https://github.com/puririshi98/vllm.git`
- vLLM branch: `sna/mxfp8-adaptive-v0.20.2-nemorl`
- vLLM base: `5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1`
- vLLM build version: `VLLM_VERSION_OVERRIDE=0.20.2`
- FlashInfer: `0.6.8.post1`
- Target hardware: GB200, compute capability 10.0
- Primary model topology: `Qwen/Qwen3-30B-A3B`, rollout tensor parallel size 1
- Primary cluster topology: four nodes, four GB200 GPUs per node
- Secondary model topology: Nemotron 3 Ultra, tensor parallel size 4

## Global Constraints

- The original and adaptive arms use the same NeMo-RL commit, custom vLLM
  commit, container, checkpoint, prompt set, seeds, TP/EP topology, and batch
  settings.
- The only adaptive-arm runtime difference is
  `VLLM_MXFP8_DENSE_CONFIG_FILE`.
- The file reference is package-relative whenever possible so no host-path
  mount is required.
- Inline tactic environment variables remain a legacy fallback and are not
  used in the performance experiment.
- Qwen TP1 starts from a model-specific trace bootstrap manifest with empty
  tactic tables, default tactic `-1`, and `pad_to_128=false`.
- No Qwen tactic is copied from the standalone Ultra TP4 seed.
- The custom vLLM offline tool owns trace inventory, GPU shmoo, qualification,
  and deterministic JSON generation. The rollout runtime never writes JSON.
- A trace with zero eligible dense MXFP8 calls fails promotion and is reported
  as `not-applicable`; it is never turned into an empty optimized table.
- GPU jobs follow the repository SLURM protocol: scheduling test, commit and
  push, `git pull` on the cluster, submit, then monitor for five minutes.
- Every experiment artifact is stored below
  `experiments/mxfp8_adaptive_rollout/`.
- New Python functions have complete type hints.
- Production changes follow red-green tests and all commits are signed off.

---

### Task 1: Make custom vLLM dependency configuration deterministic

**Files:**

- Create: `tools/configure_custom_vllm.py`
- Create: `tests/unit/tools/test_configure_custom_vllm.py`
- Modify: `tools/build-custom-vllm.sh`
- Modify: `docker/Dockerfile`
- Modify: `docs/guides/use-custom-vllm.md`

**Interface:**

```python
def configure_custom_vllm_pyproject(
    source: str,
    *,
    source_path: str = "3rdparty/vllm",
) -> str:
    ...
```

The function parses TOML, removes every normalized `vllm` requirement from
`project.optional-dependencies.vllm`, appends exactly one bare `vllm`
requirement, sets the editable local source, adds `vllm` to
`no-build-isolation-package`, and returns stable TOML.

- [ ] Add tests for bare, pinned, direct-URL, marked, and duplicated vLLM
  requirements. Run them and observe failure because the helper is absent.
- [ ] Implement the pure helper with `tomlkit` and
  `packaging.requirements.Requirement`. Make a second call byte-idempotent.
- [ ] Replace the heredoc mutation in `build-custom-vllm.sh` with the helper.
- [ ] Require non-empty Git URL, Git ref, and ABI-matching precompiled wheel
  arguments. Remove the stale 0.16 wheel defaults and torch/xformers rewrites.
- [ ] Preserve an unset project environment safely with
  `${UV_PROJECT_ENVIRONMENT:-}`.
- [ ] Export and record `VLLM_USE_PRECOMPILED=1`,
  `VLLM_PRECOMPILED_WHEEL_LOCATION`, and
  `VLLM_VERSION_OVERRIDE=0.20.2`.
- [ ] Quote all three custom vLLM Docker build arguments so an omitted optional
  value cannot shift positional arguments.
- [ ] Update the guide with the fixed fork/ref/wheel contract and verification:

```bash
uv lock --check
uv run --locked --extra vllm python -c \
  'import flashinfer, vllm; print(vllm.__version__, flashinfer.__version__, vllm.__file__)'
```

- [ ] Run:

```bash
uv run --group test pytest tests/unit/tools/test_configure_custom_vllm.py -q
uv run --group lint ruff check \
  tools/configure_custom_vllm.py \
  tests/unit/tools/test_configure_custom_vllm.py
bash -n tools/build-custom-vllm.sh
git diff --check
```

- [ ] Commit with signoff:

```bash
git commit -s -m "build: make custom vLLM source reproducible"
```

### Task 2: Forward configured environment keys to internal vLLM Ray workers

**Files:**

- Modify: `nemo_rl/models/generation/vllm/patches.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`
- Create: `tests/unit/models/generation/test_vllm_patches.py`
- Modify: `tests/unit/models/generation/test_vllm_generation.py`

**Interfaces:**

```python
def _merge_additional_env_vars_assignment(
    content: str,
    required_names: Collection[str],
) -> str:
    ...

def _configured_vllm_env_var_names(config: VllmConfig) -> list[str]:
    ...
```

- [ ] Add pure tests for sorted deterministic union, repeated application,
  sequential actors with disjoint key sets, malformed environment names, and
  a missing assignment anchor.
- [ ] Add worker tests proving keys from `vllm_cfg.env_vars` are merged with
  subclass-provided `extra_env_vars` before `_apply_vllm_patches`.
- [ ] Observe the new tests fail on the current first-writer-wins patch.
- [ ] Parse the existing `ADDITIONAL_ENV_VARS = {...}` assignment under the
  current file lock, union defaults and requested keys, and rewrite it in
  sorted order. Raise a descriptive error if the single expected assignment
  cannot be found or parsed.
- [ ] Validate environment variable names against
  `[A-Za-z_][A-Za-z0-9_]*`.
- [ ] In `_init_config`, derive names from `config["vllm_cfg"]["env_vars"]`,
  merge them with `extra_env_vars`, store the merged list, and pass it to
  `_apply_vllm_patches`.
- [ ] Run:

```bash
uv run --group test pytest \
  tests/unit/models/generation/test_vllm_patches.py \
  tests/unit/models/generation/test_vllm_generation.py -q
uv run --group lint ruff check \
  nemo_rl/models/generation/vllm/patches.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  tests/unit/models/generation/test_vllm_patches.py \
  tests/unit/models/generation/test_vllm_generation.py
git diff --check
```

- [ ] Commit with signoff:

```bash
git commit -s -m "fix(vllm): forward configured env to TP workers"
```

### Task 3: Add the Qwen3-30B-A3B 4n4g adaptive MXFP8 trace gate

**Files:**

- Create: `tests/functional/grpo_qwen3_30ba3b_mxfp8_adaptive_rollout_gb200.sh`
- Create:
  `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-adaptive-trace.yaml`

- [ ] Derive the workload from
  `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` and
  its `-mxfp8-rollout.yaml` sibling. Preserve four nodes, four GPUs per node,
  vLLM TP1, prompts, generations, maximum sequence length, seeds, and the
  sibling's MXFP8 precision and ignored q/k/v/o projections.
- [ ] Resolve the trace bootstrap JSON by its package-relative name:

```text
qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json
```

- [ ] The bootstrap manifest has exact model `Qwen/Qwen3-30B-A3B`, TP1, empty
  8x4/128x4 tactic tables, default `-1`, `pad_to_128=false`, and qualification
  scope `rollout_trace_bootstrap`.
- [ ] Pass it with a single Hydra override:

```text
++policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_CONFIG_FILE=<name>
```

- [ ] Assert the log includes vLLM version `0.20.2`, the exact custom source
  path or commit, FlashInfer `0.6.8.post1`, the config SHA256, and internal
  worker receipt of the config key.
- [ ] Enable dense shape tracing and assert every trace record carries the
  bootstrap config SHA256.
- [ ] Keep `enforce_eager=true` for this trace gate; CUDA Graph coverage is a
  later A/B gate.
- [ ] Run the custom vLLM offline `inventory` stage. At least one shape advances
  to shmoo; zero shapes produces the explicit `not-applicable` record and
  selects the Ultra TP4 fallback without creating a promoted Qwen manifest.
- [ ] Run `bash -n`, ShellCheck if available, and `git diff --check`.
- [ ] Commit with signoff:

```bash
git commit -s -m "test(vllm): add Qwen adaptive MXFP8 trace gate"
```

### Task 4: Create the reproducible Qwen 4n4g experiment

**Files:**

- Create: `experiments/mxfp8_adaptive_rollout/README.md`
- Create: `experiments/mxfp8_adaptive_rollout/PLAN.md`
- Create:
  `experiments/mxfp8_adaptive_rollout/configs/grpo_qwen3_30ba3b_4n4g.yaml`
- Create: `experiments/mxfp8_adaptive_rollout/run_ab.sh`
- Create: `experiments/mxfp8_adaptive_rollout/parse_results.py`
- Create: `tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py`

**Workload source:** Inherit the current-main model, dataset, prompt/batch,
generation, trainer, and four-node topology settings from:

```text
examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
```

Do not compare the linked BF16 recipe directly against MXFP8. Both A/B arms
use the sibling's MXFP8 rollout configuration; only the JSON config key differs.

- [ ] Add parser tests using small literal logs. Required fields are rollout
  wall time, generation time, tokens, throughput, step, arm, repeat, vLLM
  commit, NeMo-RL commit, container digest, config hash, TP, and seed.
- [ ] Implement the parser and stable JSON/CSV summaries.
- [ ] Create an experiment launcher with `trace`, `shmoo`, `original`, and
  `adaptive` modes. It writes a resolved config and metadata before launch.
- [ ] `trace` runs one short Qwen TP1 4n4g rollout with the bootstrap manifest
  and emits exact dense physical shapes.
- [ ] `shmoo` runs the custom vLLM offline pipeline in the same container and
  GPU topology, requires at least three repeats, and generates the qualified
  Qwen TP1 manifest only when eligible shapes exist.
- [ ] `original` unsets only `VLLM_MXFP8_DENSE_CONFIG_FILE`.
- [ ] `adaptive` sets only the package-relative JSON config name.
- [ ] If the trace is not applicable, both performance modes are skipped and a
  stable not-applicable result names the zero-hit reason and the Ultra TP4
  fallback plan.
- [ ] Use matched warmups followed by at least three alternating measured
  repeats. Keep prompt samples, seeds, and Ray placement identical.
- [ ] Reject an A/B pair if source commits, container digest, checkpoint,
  topology, or resolved Hydra config outside the one environment key differ.
- [ ] Run parser tests, Ruff, `bash -n`, and `git diff --check`.
- [ ] Commit with signoff:

```bash
git commit -s -m "bench: add Qwen 4n4g MXFP8 rollout A/B"
```

### Task 5: Build, lock, and perform local provenance checks

- [ ] Point the NeMo-RL worktree at the final local custom vLLM branch and run
  the deterministic configuration helper.
- [ ] Generate and review the `pyproject.toml` and `uv.lock` diff. Confirm
  exactly one locked vLLM package source points to `3rdparty/vllm`.
- [ ] Verify the checked-out `3rdparty/vllm` commit matches the recorded custom
  branch head.
- [ ] Run the focused unit suites from Tasks 1, 2, and 4.
- [ ] Run repository formatting/type checks required for modified files.
- [ ] Build the container with explicit URL, immutable ref, wheel URL, and
  version override. Record the resulting image digest.
- [ ] In the image, run:

```bash
python -c \
  'import flashinfer, vllm; print(vllm.__version__, flashinfer.__version__, vllm.__file__)'
python -c \
  'from vllm.model_executor.kernels.linear.mxfp8.tactic_config import load_mxfp8_dense_runtime_config; print("loader-ok")'
```

- [ ] Commit the lock and provenance updates with signoff.

### Task 6: Execute validation gates on GB200

This task starts only after both branches are clean, committed, pushed, and
referenced by immutable commit SHA.

- [ ] Record the OCI-HSG scheduling preflight for account
  `coreai_dlalgo_nemorl`: reachable login, four-GPU nodes, user FairShare,
  queue estimate, and an accepted four-node `srun --test-only` request.
- [ ] Use OCI-HSG for the primary run. Its vLLM 0.20.2 adaptive containers and
  user Lustre roots are already readable. Use Pre-Tyche for same-hardware
  reproduction after the local Kerberos credential is refreshed.
- [ ] Pull the exact NeMo-RL commit on the selected GB200 cluster.
- [ ] Submit the Qwen 4n4g MXFP8 trace gate.
- [ ] Monitor the job for five minutes and stop on import, source, worker-env,
  or MXFP8 correctness failures.
- [ ] Extract the exact Qwen TP1 dense rollout shape inventory.
- [ ] If the inventory is non-empty, shmoo each shape at least three times in
  the same container and topology. Promote only correctness-passing tactics
  with at least 1.02 median speedup versus runner default `-1`.
- [ ] Deterministically regenerate the Qwen JSON with rollout-qualified
  provenance, validate it through the production loader, commit, push, rebuild,
  and record the new config/image hashes.
- [ ] Run one Qwen CUDA Graph smoke for both original and adaptive arms, then
  run the alternating A/B repeats.
- [ ] If the Qwen inventory is empty, record not-applicable and execute the
  same trace/shmoo/CUDA-Graph/A/B sequence on Nemotron 3 Ultra TP4 instead.
- [ ] Accept a performance result only if:

```text
all correctness checks pass
no worker uses runner fallback for a supposedly qualified shape
median rollout throughput improves
median rollout latency decreases
p95 rollout latency does not regress by more than 2%
end-to-end step time does not regress
```

- [ ] Write results and raw artifact links under
  `experiments/mxfp8_adaptive_rollout/report/`.
- [ ] Run final verification and request whole-branch review before reporting
  the performance conclusion.
