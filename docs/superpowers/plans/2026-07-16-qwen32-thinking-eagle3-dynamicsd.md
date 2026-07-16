# Qwen3-32B Thinking EAGLE3 DynamicSD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fixed K2 and calibrated DynamicSD support to the Qwen3-32B Thinking EAGLE3 matrix, then submit matched Lyris smoke jobs without weakening the existing performance-recipe or CUDA Graph controls.

**Architecture:** Extend the typed experiment matrix rather than adding a second Qwen3-32B launcher. A validated JSON schedule artifact supplies DynamicSD ranges and provenance; the runtime applies the existing vLLM 0.25.1 CUDA Graph fix only for DynamicSD. Historical vLLM 0.24 profile data may seed smoke runs, while final20 rejects any artifact not marked as matched vLLM 0.25.1 calibration.

**Tech Stack:** Python 3.12+, dataclasses, argparse, Hydra overrides, pytest, Ruff, Pyright, Bash, SLURM, Ray, vLLM 0.25.1, W&B.

## Global Constraints

- Use `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml` unchanged except for max steps, output locations, checkpoint disabling, CUDA Graph mode, and SpecDec settings.
- Use target revision `9216db5781bf21249d130ec9da846c4624c16137` and Thinking drafter revision `a1403e07b73a66fc9ef561463631c31864616933`.
- Use vLLM 0.25.1, temperature 1.0, top-p 1.0, max OSL 4096, `enforce_eager=false`, `FULL_AND_PIECEWISE`, native capture sizing, and `VLLM_USE_V2_MODEL_RUNNER=1`.
- Use four Lyris GB200 nodes, four GPUs per node, `--segment=4`, no `--gres`, no singleton, and no scheduler dependency.
- Keep checkpoint saving disabled and W&B project `nemo-rl-vllm0251-drafter-matrix`.
- Fixed comparisons and final DynamicSD claims use completed steps 2-20 only.
- DynamicSD may select K0, K1, K2, or K3; fixed K2 and fixed K3 remain independent controls.
- A historical-profile seed may run only smoke2 or smoke5. DynamicSD final20 requires a matched vLLM 0.25.1 calibration artifact.
- Push only to `git@github-seonjinn:seonjinn/RL.git`; do not push to NVIDIA-NeMo/RL.

---

### Task 1: Add Fixed Thinking K2

**Files:**
- Modify: `tests/experiments/test_vllm_0251_drafter_matrix.py`
- Modify: `experiments/vllm_0251_drafter_matrix/matrix.py`
- Modify: `experiments/vllm_0251_drafter_matrix/README.md`

**Interfaces:**
- Consumes: existing `VariantSpec`, `resolve_run()`, and immutable Thinking checkpoint table.
- Produces: variant key `eagle3_thinking_k2` with `num_speculative_tokens=2`, MRv2, draft TP1, and Qwen3-32B/Qwen3-235B compatibility.

- [ ] **Step 1: Add the failing K2 matrix tests**

Add `eagle3_thinking_k2` to the official-override, Thinking-checkpoint, Qwen30-rejection, and Qwen235-native-capture parameter sets. Assert this exact override:

```python
assert (
    "++policy.generation.vllm_kwargs.speculative_config."
    "num_speculative_tokens=2"
) in run.hydra_overrides
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m pytest tests/experiments/test_vllm_0251_drafter_matrix.py -q
```

Expected: failure because `eagle3_thinking_k2` is not present in `G_VARIANTS`.

- [ ] **Step 3: Add the minimal K2 variant**

Add this entry between Thinking K1 and K3:

```python
VariantSpec(
    key="eagle3_thinking_k2",
    method="eagle3",
    runner="mrv2",
    num_speculative_tokens=2,
    compatible_models=frozenset(("qwen32", "qwen235")),
    checkpoints=G_EAGLE3_THINKING_CHECKPOINTS,
    uses_draft_model=True,
),
```

Update the experiment matrix documentation from K1/K3/K5 to K1/K2/K3/K5.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command. Expected: all focused matrix tests pass.

- [ ] **Step 5: Commit the fixed-K change**

```bash
git add experiments/vllm_0251_drafter_matrix/matrix.py \
  experiments/vllm_0251_drafter_matrix/README.md \
  tests/experiments/test_vllm_0251_drafter_matrix.py
git commit -s -m "feat: add qwen32 thinking eagle3 k2"
```

### Task 2: Validate DynamicSD Schedule Artifacts

**Files:**
- Modify: `tests/experiments/test_vllm_0251_drafter_matrix.py`
- Modify: `experiments/vllm_0251_drafter_matrix/matrix.py`
- Create: `experiments/vllm_0251_drafter_matrix/calibration/qwen32_thinking_k123_seed.json`

**Interfaces:**
- Consumes: a JSON object with schema version, calibration status, exact model/runtime identity, profile provenance, and contiguous ranges.
- Produces: immutable `DynamicSchedule` records and `load_dynamic_schedule(path, run)` validation.

- [ ] **Step 1: Add failing schedule-validation tests**

Define fixtures with this exact schema:

```json
{
  "schema_version": 1,
  "calibration_status": "seed",
  "model_key": "qwen32",
  "target_revision": "9216db5781bf21249d130ec9da846c4624c16137",
  "drafter_revision": "a1403e07b73a66fc9ef561463631c31864616933",
  "source_runtime_vllm": "0.24.0",
  "target_runtime_vllm": "0.25.1",
  "target_cuda_graph_mode": "FULL_AND_PIECEWISE",
  "profile_sha256": "efcd9ad3f74ecb260ab7a580a062e56266b67196fd16d90b792c4176a25e5f69",
  "ranges": [[1, 127, 3], [128, 256, 1]]
}
```

Tests must accept this artifact for smoke2/smoke5 and reject it for final20.
They must reject wrong target/drafter revisions, unknown status, non-contiguous
or overlapping ranges, a first batch other than one, K outside 0-3, and a
maximum K different from the dynamic variant's fixed maximum K3.

- [ ] **Step 2: Run the schedule tests and verify RED**

Run:

```bash
python3 -m pytest tests/experiments/test_vllm_0251_drafter_matrix.py \
  -k 'dynamic or schedule' -q
```

Expected: failure because `DynamicSchedule` and `load_dynamic_schedule()` do not exist.

- [ ] **Step 3: Implement immutable schedule parsing**

Add frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class DynamicRange:
    start_batch: int
    end_batch: int
    k: int


@dataclass(frozen=True, slots=True)
class DynamicSchedule:
    source_path: Path
    source_sha256: str
    calibration_status: str
    ranges: tuple[DynamicRange, ...]

    def vllm_ranges(self) -> tuple[tuple[int, int, int], ...]:
        return tuple((item.start_batch, item.end_batch, item.k) for item in self.ranges)
```

Implement `load_dynamic_schedule(path: Path) -> DynamicSchedule` using
`json.loads()`, `hashlib.sha256()`, direct required-key access, and the
validation rules from Step 1. Validate its model, checkpoint, runtime, phase,
and maximum K when `resolve_run()` binds it to a run. Do not supply hidden
defaults.

- [ ] **Step 4: Add the checked-in smoke seed**

Create the exact JSON artifact from Step 1. Its status remains `seed` because
the source profile used standalone vLLM 0.24.0. The file itself is hashed and
recorded in run provenance; its `profile_sha256` points to the historical raw
profile CSV.

- [ ] **Step 5: Run the schedule tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 6: Commit schedule validation**

```bash
git add experiments/vllm_0251_drafter_matrix/matrix.py \
  experiments/vllm_0251_drafter_matrix/calibration/qwen32_thinking_k123_seed.json \
  tests/experiments/test_vllm_0251_drafter_matrix.py
git commit -s -m "feat: validate qwen32 dynamicsd schedules"
```

### Task 3: Add the DynamicSD Runtime Path

**Files:**
- Modify: `tests/experiments/test_vllm_0251_drafter_matrix.py`
- Modify: `experiments/vllm_0251_drafter_matrix/matrix.py`
- Modify: `experiments/vllm_0251_drafter_matrix/README.md`

**Interfaces:**
- Consumes: `DynamicSchedule` from Task 2 and the existing opt-in patch script `experiments/vllm_0251_eagle3_perfcfg/apply_vllm0251_dynamic_sd_cg_fix.py`.
- Produces: `eagle3_thinking_dynamic_k123`, `--dynamic-schedule PATH`, exact Hydra range overrides, and run-scoped post-sync patch environment.

- [ ] **Step 1: Add failing DynamicSD command tests**

Assert that resolving the dynamic variant with the seed emits:

```text
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens_per_batch_size=[[1,127,3],[128,256,1]]
```

Assert `build_runtime_command()` includes both:

```text
NRL_VENV_POST_SYNC_SCRIPT=<repo>/experiments/vllm_0251_eagle3_perfcfg/apply_vllm0251_dynamic_sd_cg_fix.py
NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
```

Fixed K1/K2/K3 commands must contain neither variable. CLI tests must reject a
dynamic variant without `--dynamic-schedule` and reject a fixed variant with it.

- [ ] **Step 2: Run DynamicSD command tests and verify RED**

Run the Step 2 command from Task 2. Expected: failures for the missing variant, CLI option, Hydra override, and post-sync environment.

- [ ] **Step 3: Implement the dynamic variant and CLI handoff**

Add `dynamic_schedule_required: bool = False` to `VariantSpec`, add
`dynamic_schedule: DynamicSchedule | None` to `ResolvedRun`, and define
`eagle3_thinking_dynamic_k123` for Qwen3-32B only with maximum K3. Extend
`resolve_run()` with a keyword-only schedule argument and serialize ranges
using compact JSON separators so Hydra receives one deterministic value.

Add `--dynamic-schedule PATH` to show/test-only/submit. Load and validate the
artifact before producing the resolved run. Include artifact path, SHA-256,
status, and ranges in both JSON and text provenance.

- [ ] **Step 4: Apply the CUDA Graph patch only to DynamicSD**

Extend `build_runtime_command()` to append the two post-sync variables from
Step 1 only when `run.dynamic_schedule is not None`. Preserve native capture
sizing and reject any DynamicSD path that emits `cudagraph_capture_sizes` or
changes `FULL_AND_PIECEWISE`.

- [ ] **Step 5: Run DynamicSD and regression tests**

Run:

```bash
python3 -m pytest tests/experiments/test_vllm_0251_drafter_matrix.py \
  tests/test_vllm0251_dynamic_sd_patch.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the runtime path**

```bash
git add experiments/vllm_0251_drafter_matrix/matrix.py \
  experiments/vllm_0251_drafter_matrix/README.md \
  tests/experiments/test_vllm_0251_drafter_matrix.py
git commit -s -m "feat: add qwen32 thinking DynamicSD runs"
```

### Task 4: Verify, Push, And Submit Smoke Jobs

**Files:**
- Modify: `experiments/vllm_0251_drafter_matrix/REPORT.md`

**Interfaces:**
- Consumes: pushed private-fork branch, exact Lyris checkout, staged target/drafter snapshots, nightly image, and W&B credential from the submission environment.
- Produces: fixed K2, fixed K3, and DynamicSD smoke job IDs with immutable provenance and direct W&B links after startup.

- [ ] **Step 1: Run local verification**

```bash
python3 -m pytest tests/experiments/test_vllm_0251_drafter_matrix.py \
  tests/experiments/test_vllm_0251_drafter_results.py \
  tests/test_vllm0251_dynamic_sd_patch.py -q
python3 -m ruff check experiments/vllm_0251_drafter_matrix \
  tests/experiments/test_vllm_0251_drafter_matrix.py
python3 -m pyright experiments/vllm_0251_drafter_matrix/matrix.py
bash -n experiments/vllm_0251_drafter_matrix/submit_matrix.sh
git diff --check
```

Expected: zero test failures, Ruff errors, Pyright errors, shell syntax errors, or whitespace errors.

- [ ] **Step 2: Commit the plan/report state and push**

Update `REPORT.md` with the seed schedule caveat and planned smoke rows, then:

```bash
git add docs/superpowers/plans/2026-07-16-qwen32-thinking-eagle3-dynamicsd.md \
  experiments/vllm_0251_drafter_matrix/REPORT.md
git commit -s -m "docs: plan qwen32 DynamicSD smoke wave"
git push fork sna/nemorl-vllm0251-drafter-matrix-20260716
```

- [ ] **Step 3: Refresh the Lyris checkout**

```bash
ssh login-lyris 'cd /lustre/fsw/coreai_dlalgo_llm/users/sna/RL-vllm0251-drafter-matrix-20260716 && git pull --ff-only fork sna/nemorl-vllm0251-drafter-matrix-20260716 && git submodule update --init --recursive'
```

Expected: checkout reaches the pushed HEAD and recursive submodules have no
leading `-` or `+` in `git submodule status --recursive`.

- [ ] **Step 4: Run exact scheduler preflights**

From the Lyris checkout, run `show` and `test-only` for:

```bash
bash experiments/vllm_0251_drafter_matrix/submit_matrix.sh test-only \
  --model qwen32 --variant eagle3_thinking_k2 --phase smoke2 --cluster lyris
bash experiments/vllm_0251_drafter_matrix/submit_matrix.sh test-only \
  --model qwen32 --variant eagle3_thinking_k3 --phase smoke2 --cluster lyris
bash experiments/vllm_0251_drafter_matrix/submit_matrix.sh test-only \
  --model qwen32 --variant eagle3_thinking_dynamic_k123 --phase smoke2 \
  --cluster lyris \
  --dynamic-schedule experiments/vllm_0251_drafter_matrix/calibration/qwen32_thinking_k123_seed.json
```

Expected: every exact four-node shape is accepted by `sbatch --test-only`.

- [ ] **Step 5: Submit all three smoke2 jobs**

Repeat Step 4 with `submit` instead of `test-only`. Each submit action runs its
own identical scheduler preflight before `sbatch --parsable`. Record job IDs,
run directories, and W&B links in `REPORT.md`.

- [ ] **Step 6: Monitor for at least five minutes**

Check `squeue`, `sacct`, SLURM output, and `ray-driver.log`. A smoke passes only
after step 2 completes with E2E/generation time and throughput, acceptance,
mean accepted length, and no eager fallback, CUDA Graph downgrade, traceback,
NCCL watchdog, OOM, or port collision. DynamicSD additionally must show the
accepted schedule and positive draft counters. vLLM 0.25.1 does not export a
selected-K histogram, so do not infer K0/K1/K2/K3 fractions from accepted-token
position counters.

- [ ] **Step 7: Promote only validated fixed candidates**

Submit fixed K2 and K3 final20 after their smoke metrics are compared with the
already running matched baseline and K1. Do not submit DynamicSD final20 until
a replacement artifact has `calibration_status=calibrated`,
`target_runtime_vllm=0.25.1`,
and matched runtime/profile metadata.
