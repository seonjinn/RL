# NeMo-RL Eagle-3 DynamicSD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a matched 20-step NeMo-RL GRPO comparison of target-only decoding, fixed-K Eagle-3, and Eagle-3 DynamicSD for Qwen3-30B-A3B and Qwen3-32B on AWS-DFW.

**Architecture:** Keep DynamicSD experiment policy outside shared NeMo-RL defaults. A focused launcher passes vLLM 0.24 `speculative_config` and PIECEWISE CUDA Graph overrides into the unchanged upstream performance recipes, while existing NeMo-RL acceptance metrics flow to W&B. A separate collector fetches W&B histories, validates complete Steps 2-20, and computes matched baseline and fixed-K speedups.

**Tech Stack:** Bash, Python 3.13, pytest, OmegaConf/Hydra overrides, NeMo-RL GRPO, Ray, vLLM 0.24, SLURM/Pyxis, W&B API.

## Global Constraints

- Use `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` and `grpo-qwen3-32b-4n4g.yaml` without changing their model, dataset, rollout shape, topology, sampling, or sequence limits.
- Set `grpo.max_num_steps=20`, `checkpointing.enabled=false`, `temperature=1.0`, and `top_p=1.0`.
- Set `policy.generation.vllm_cfg.enforce_eager=false` and `policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE` for all three variants.
- Use fixed Eagle-3 `K=5` and DynamicSD `[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]`.
- Compare steady-state Steps 2-20 only; Step 1 is warmup.
- Submit from a clean, pushed commit after `sbatch --test-only`, and monitor each submitted smoke for at least five minutes and through policy training.
- Do not fall back to eager mode or alter recipe-owned batch/sequence settings after a failure.
- Qwen3-235B and async 1-off are follow-up cohorts and are not part of this implementation.

---

## File Map

- Create `experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh`: render, validate, test, and submit the six AWS-DFW jobs.
- Create `experiments/vllm_024_upgrade/summarize_eagle3_dynamicsd.py`: fetch W&B histories and emit validated CSV/JSON summaries.
- Modify `tests/test_vllm_024_launch_scripts.py`: verify launcher contracts and variant-specific overrides.
- Create `tests/test_vllm_024_dynamicsd_summary.py`: unit-test Step 2-20 validation, speedups, and health gates.
- Modify `tests/unit/models/generation/test_vllm_generation.py`: pin nested DynamicSD config preservation through generation configuration.
- Modify `experiments/vllm_024_upgrade/README.md`: document launch, collection, output paths, and comparison rules.

### Task 1: Pin the vLLM DynamicSD Configuration Contract

**Files:**
- Modify: `tests/unit/models/generation/test_vllm_generation.py`

**Interfaces:**
- Consumes: `configure_generation_config(config, tokenizer, is_eval=False, has_refit_draft_weights=False)`.
- Produces: a regression test proving that `num_speculative_tokens_per_batch_size` remains a list of integer triplets and reaches `vllm_kwargs` unchanged.

- [ ] **Step 1: Add the failing contract test**

Append this test next to the existing speculative-decoding configuration tests:

```python
def test_configure_generation_config_preserves_dynamic_eagle3_schedule():
    vllm_config = deepcopy(basic_vllm_test_config)
    schedule = [
        [1, 16, 5],
        [17, 32, 4],
        [33, 64, 3],
        [65, 128, 1],
        [129, 512, 0],
    ]
    vllm_config["vllm_kwargs"] = {
        "compilation_config": {"cudagraph_mode": "PIECEWISE"},
        "speculative_config": {
            "method": "eagle3",
            "model": "/tmp/draft-model",
            "num_speculative_tokens": 5,
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens_per_batch_size": schedule,
        },
    }
    tokenizer = MagicMock(pad_token_id=0, eos_token_id=1)

    with pytest.warns(UserWarning, match="Speculative decoding is enabled"):
        configured = configure_generation_config(
            vllm_config,
            tokenizer,
            is_eval=False,
            has_refit_draft_weights=False,
        )

    assert configured["vllm_kwargs"]["speculative_config"] == {
        "method": "eagle3",
        "model": "/tmp/draft-model",
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens_per_batch_size": schedule,
    }
    assert configured["vllm_kwargs"]["compilation_config"] == {
        "cudagraph_mode": "PIECEWISE"
    }
```

- [ ] **Step 2: Run the focused test and record the result**

Run:

```bash
uv run --no-sync pytest \
  tests/unit/models/generation/test_vllm_generation.py::test_configure_generation_config_preserves_dynamic_eagle3_schedule \
  -q
```

Expected: PASS. If it fails because configuration mutates the nested list, make the smallest correction in `nemo_rl/models/generation/__init__.py` and add that file to this task; do not introduce a DynamicSD default.

- [ ] **Step 3: Commit the contract test**

```bash
git add tests/unit/models/generation/test_vllm_generation.py
git commit -s -m "test: preserve DynamicSD generation config"
```

### Task 2: Build the Matched AWS-DFW Launcher

**Files:**
- Create: `experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh`
- Modify: `tests/test_vllm_024_launch_scripts.py`

**Interfaces:**
- Consumes: AWS-DFW account/container/HF paths, the two upstream performance recipes, and `ray.sub`.
- Produces: `dry-run`, `test-only`, and `submit` modes plus `${EXPERIMENT_ROOT}/submissions.tsv` with `timestamp`, `model`, `variant`, `job_id`, `nodes`, `segment`, `commit`, `wandb_run_id`, `wandb_url`, `recipe`, and `draft_model`.

- [ ] **Step 1: Add failing launcher tests**

Add these helpers and tests to `tests/test_vllm_024_launch_scripts.py`:

```python
DYNAMICSD_LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "vllm_024_upgrade"
    / "submit_eagle3_dynamicsd_step20.sh"
)


def _dry_run_dynamicsd(model: str, variant: str) -> str:
    return _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        model,
        variant,
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="contract-test",
    )


def test_dynamicsd_launcher_preserves_matched_runtime_contract() -> None:
    output = _dry_run_dynamicsd("qwen32b", "dynamic")

    assert "grpo.max_num_steps=20" in output
    assert "checkpointing.enabled=false" in output
    assert "policy.generation.vllm_cfg.enforce_eager=false" in output
    assert "compilation_config.cudagraph_mode=PIECEWISE" in output
    assert "cluster.segment_size=4" in output
    assert "--nodes=4" in output
    assert "--segment=4" in output
    assert "--gres=gpu:4" in output


def test_dynamicsd_launcher_renders_fixed_eagle3() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "eagle3_k5")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "num_speculative_tokens_per_batch_size" not in output
    assert "Qwen3-30B-A3B-Thinking-2507-speculator.eagle3" in output


def test_dynamicsd_launcher_renders_dynamic_schedule() -> None:
    output = _dry_run_dynamicsd("qwen32b", "dynamic")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert (
        "speculative_config.num_speculative_tokens_per_batch_size="
        "\\[\\[1\\,16\\,5\\]\\,\\[17\\,32\\,4\\]\\,"
        "\\[33\\,64\\,3\\]\\,\\[65\\,128\\,1\\]\\,"
        "\\[129\\,512\\,0\\]\\]"
    ) in output


def test_dynamicsd_launcher_keeps_baseline_free_of_specdec() -> None:
    output = _dry_run_dynamicsd("qwen32b", "baseline")

    assert "compilation_config.cudagraph_mode=PIECEWISE" in output
    assert "speculative_config" not in output
```

- [ ] **Step 2: Verify the tests fail because the launcher is absent**

Run:

```bash
uv run --no-sync pytest tests/test_vllm_024_launch_scripts.py -q
```

Expected: FAIL with `No such file or directory` for `submit_eagle3_dynamicsd_step20.sh`.

- [ ] **Step 3: Implement the launcher**

Create an executable Bash launcher following `submit_performance_step10.sh`. Use this exact public interface and configuration data:

```bash
MODE="${1:-test-only}"
MODEL_SELECTION="${2:-all}"
VARIANT_SELECTION="${3:-all}"
MAX_STEPS="${MAX_STEPS:-20}"
STATIC_K="${STATIC_K:-5}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]}"
ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
PARTITION="${PARTITION:-batch_long}"
USE_GRES="${USE_GRES:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-nemorl-vllm024-dynamicsd-aws-dfw}"
```

Resolve models with:

```bash
case "${model}" in
  qwen30ba3b)
    recipe="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
    draft_model="${QWEN30_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
    nodes=4
    model_port_offset=0
    ;;
  qwen32b)
    recipe="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
    draft_model="${QWEN32_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
    nodes=4
    model_port_offset=1000
    ;;
esac
```

Build every command as a Bash array before `%q` rendering. All variants receive:

```bash
overrides=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "checkpointing.enabled=false"
  "checkpointing.checkpoint_dir=${run_dir}/checkpoints"
  "policy.generation.vllm_cfg.enforce_eager=false"
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE"
  "cluster.segment_size=${nodes}"
  "logger.wandb_enabled=true"
  "logger.wandb.project=${WANDB_PROJECT}"
  "logger.wandb.name=${wandb_name}"
  "logger.log_dir=${run_dir}/nemo_logs"
)
```

Add these arrays only for the two SpecDec variants:

```bash
specdec_overrides=(
  "++policy.generation.vllm_kwargs.speculative_config.method=eagle3"
  "++policy.generation.vllm_kwargs.speculative_config.model=${draft_model}"
  "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${STATIC_K}"
  "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1"
)
if [[ "${variant}" == "dynamic" ]]; then
  specdec_overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens_per_batch_size=${DYNAMIC_SCHEDULE}"
  )
fi
```

Set deterministic W&B and rendezvous variables in the command environment:

```bash
wandb_run_id="${RUN_TAG}-${model}-${variant}"
variant_port_offset=0
[[ "${variant}" == "eagle3_k5" ]] && variant_port_offset=200
[[ "${variant}" == "dynamic" ]] && variant_port_offset=400
vllm_port=$((VLLM_PORT_BASE + model_port_offset + variant_port_offset))

command_env=(
  "WANDB_RUN_ID=${wandb_run_id}"
  "WANDB_RUN_GROUP=${RUN_TAG}"
  "WANDB_RESUME=allow"
  "VLLM_PORT=${vllm_port}"
  "PYTHONPATH=${REPO_DIR}"
  "TRITON_CACHE_DIR=${triton_cache_dir}"
  "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
)
```

Validate the container for non-dry-run modes and each drafter directory for SpecDec variants. Keep `--nodes=4`, `--segment=4`, `--exclusive`, `--gres=gpu:4`, and run-specific compiler/venv directories. Append the manifest only after successful `sbatch --parsable` output.

- [ ] **Step 4: Run launcher tests and shell validation**

Run:

```bash
bash -n experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh
uv run --no-sync pytest tests/test_vllm_024_launch_scripts.py -q
```

Expected: PASS for shell parsing and all launcher tests.

- [ ] **Step 5: Commit the launcher**

```bash
git add \
  experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  tests/test_vllm_024_launch_scripts.py
git commit -s -m "feat: launch NeMo-RL DynamicSD matrix"
```

### Task 3: Add a Reproducible W&B Result Collector

**Files:**
- Create: `experiments/vllm_024_upgrade/summarize_eagle3_dynamicsd.py`
- Create: `tests/test_vllm_024_dynamicsd_summary.py`

**Interfaces:**
- Consumes: `submissions.tsv` and W&B histories for the recorded run IDs.
- Produces: `summary.csv` and `summary.json`, including completion status, Steps 2-20 means, baseline-relative ratios, fixed-K-relative DynamicSD ratios, health flags, job IDs, and W&B URLs.

- [ ] **Step 1: Write pure-function tests for completed and incomplete histories**

Create tests around these public interfaces:

```python
from experiments.vllm_024_upgrade.summarize_eagle3_dynamicsd import (
    RunSummary,
    build_comparison_rows,
    summarize_history,
)


def _history(scale: float = 1.0) -> list[dict[str, float]]:
    return [
        {
            "_step": step,
            "timing/train/generation": 100.0 / scale,
            "timing/train/total_step_time": 200.0 / scale,
            "performance/generation_tokens_per_sec_per_gpu": 50.0 * scale,
            "performance/tokens_per_sec_per_gpu": 25.0 * scale,
            "train/vllm/spec_acceptance_rate": 0.5 if scale > 1 else 0.0,
            "train/vllm/spec_acceptance_length": 2.5 if scale > 1 else 1.0,
            "train/reward": 0.4,
            "train/mean_gen_tokens_per_sample": 1024.0,
            "train/gen_kl_error": 0.01,
        }
        for step in range(1, 21)
    ]


def test_summarize_history_uses_steps_2_through_20() -> None:
    summary = summarize_history("qwen32b", "baseline", _history())
    assert summary.complete
    assert summary.measured_steps == list(range(2, 21))
    assert summary.generation_time_s == 100.0
    assert summary.e2e_step_time_s == 200.0


def test_summarize_history_rejects_missing_step_20() -> None:
    summary = summarize_history("qwen32b", "dynamic", _history(1.2)[:-1])
    assert not summary.complete
    assert summary.reason == "missing_steps:20"


def test_build_comparison_rows_matches_model_baseline() -> None:
    summaries = [
        summarize_history("qwen32b", "baseline", _history()),
        summarize_history("qwen32b", "eagle3_k5", _history(1.25)),
        summarize_history("qwen32b", "dynamic", _history(1.5)),
    ]
    rows = {row.variant: row for row in build_comparison_rows(summaries)}
    assert rows["dynamic"].generation_throughput_speedup_vs_baseline == 1.5
    assert rows["dynamic"].e2e_step_time_speedup_vs_baseline == 1.5
    assert rows["dynamic"].generation_throughput_speedup_vs_fixed == 1.2
```

- [ ] **Step 2: Verify the collector tests fail before implementation**

Run:

```bash
uv run --no-sync pytest tests/test_vllm_024_dynamicsd_summary.py -q
```

Expected: FAIL because the collector module does not exist.

- [ ] **Step 3: Implement typed history aggregation**

Implement these dataclasses and exact metric map:

```python
from dataclasses import asdict, dataclass

METRIC_KEYS = {
    "generation_time_s": "timing/train/generation",
    "e2e_step_time_s": "timing/train/total_step_time",
    "generation_throughput": "performance/generation_tokens_per_sec_per_gpu",
    "e2e_throughput": "performance/tokens_per_sec_per_gpu",
    "acceptance_rate": "train/vllm/spec_acceptance_rate",
    "mean_acceptance_length": "train/vllm/spec_acceptance_length",
    "reward": "train/reward",
    "mean_response_length": "train/mean_gen_tokens_per_sample",
    "approx_kl": "train/gen_kl_error",
}
EXPECTED_STEPS = set(range(2, 21))


@dataclass(frozen=True)
class RunSummary:
    model: str
    variant: str
    complete: bool
    reason: str
    measured_steps: list[int]
    generation_time_s: float | None
    e2e_step_time_s: float | None
    generation_throughput: float | None
    e2e_throughput: float | None
    acceptance_rate: float | None
    mean_acceptance_length: float | None
    reward: float | None
    mean_response_length: float | None
    approx_kl: float | None
```

`summarize_history()` must filter `_step` to 2-20, reject missing steps or non-finite required metrics, and use `statistics.fmean`. Baseline acceptance fields may be zero; SpecDec variants require positive draft/acceptance evidence.

`build_comparison_rows()` must group by model, reject incomplete baselines, compute the four baseline ratios, compute DynamicSD-to-fixed ratios, and mark `health_gate_passed=False` when reward, mean response length, or approximate KL differs by more than 10% from baseline. Treat a baseline value of zero as non-comparable rather than dividing by zero.

- [ ] **Step 4: Implement manifest and W&B CLI integration**

The CLI must accept:

```text
--manifest PATH
--entity nvidia
--project nemorl-vllm024-dynamicsd-aws-dfw
--output-dir PATH
```

Read manifest rows with `csv.DictReader(delimiter="\t")`, fetch each run with
`wandb.Api().run(f"{entity}/{project}/{wandb_run_id}")`, and call
`scan_history(keys=["_step", *METRIC_KEYS.values()])`. Add `job_id`,
`wandb_run_id`, and `wandb_url` to the serialized comparison row. Write JSON
atomically through a temporary file and write CSV with an explicit field list.
Return nonzero if any submitted row is missing or incomplete.

- [ ] **Step 5: Run collector tests**

Run:

```bash
uv run --no-sync pytest tests/test_vllm_024_dynamicsd_summary.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the collector**

```bash
git add \
  experiments/vllm_024_upgrade/summarize_eagle3_dynamicsd.py \
  tests/test_vllm_024_dynamicsd_summary.py
git commit -s -m "feat: summarize NeMo-RL DynamicSD runs"
```

### Task 4: Document and Verify the Local Experiment

**Files:**
- Modify: `experiments/vllm_024_upgrade/README.md`

**Interfaces:**
- Consumes: launcher and collector from Tasks 2-3.
- Produces: reproducible operator commands and the exact validity rules used by the report.

- [ ] **Step 1: Add the DynamicSD workflow to the README**

Document these exact commands:

```bash
# Local rendering
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  dry-run all all

# Scheduler validation on AWS-DFW
MAX_STEPS=2 \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  test-only all all

# Two-step gate
MAX_STEPS=2 RUN_TAG=vllm024-dynamicsd-smoke-$(date +%Y%m%d) \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit all all

# Full run after every smoke reaches policy training and exits 0
MAX_STEPS=20 RUN_TAG=vllm024-dynamicsd-step20-$(date +%Y%m%d) \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit all all

# Final collection
uv run --no-sync python \
  experiments/vllm_024_upgrade/summarize_eagle3_dynamicsd.py \
  --manifest experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/submissions.tsv \
  --entity nvidia \
  --project nemorl-vllm024-dynamicsd-aws-dfw \
  --output-dir experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/summary
```

State that PIECEWISE rows are a separate cohort from historical full-graph and eager rows, and that Step 1 is excluded.

- [ ] **Step 2: Run the complete local verification set**

Run:

```bash
bash -n experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh
uv run --no-sync pytest \
  tests/test_vllm_024_launch_scripts.py \
  tests/test_vllm_024_dynamicsd_summary.py \
  tests/unit/models/generation/test_vllm_generation.py \
  -q
git diff --check
```

Expected: all tests pass and `git diff --check` emits no output.

- [ ] **Step 3: Commit documentation**

```bash
git add experiments/vllm_024_upgrade/README.md
git commit -s -m "docs: document NeMo-RL DynamicSD runs"
```

### Task 5: Push, Validate, and Submit AWS-DFW Smokes

**Files:**
- Generated remotely: `experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-smoke-20260707/submissions.tsv`
- Generated remotely: one SLURM log tree per model and variant

**Interfaces:**
- Consumes: clean pushed branch `sna/nemorl-vllm024-upgrade` and AWS cluster config `experiments/vllm_024_upgrade/cluster-aws-dfw.yaml`.
- Produces: six completed two-step smoke jobs with W&B links and active SpecDec metrics.

- [ ] **Step 1: Push the complete implementation**

```bash
git status --short
git push fork sna/nemorl-vllm024-upgrade
```

Expected: clean status and remote branch updated.

- [ ] **Step 2: Fast-forward the AWS-DFW worktree**

```bash
ssh aws-dfw-cs-001-login-01.nvidia.com '
  cd /lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna/RL-vllm024-upgrade-20260707 &&
  git fetch fork sna/nemorl-vllm024-upgrade &&
  git checkout sna/nemorl-vllm024-upgrade &&
  git pull --ff-only fork sna/nemorl-vllm024-upgrade
'
```

Expected: remote `HEAD` equals local `git rev-parse HEAD`.

- [ ] **Step 3: Verify container and drafter assets**

On AWS-DFW, verify the NeMo-RL nightly image and both exact snapshot directories with `test -s`/`test -d`. Stop before submission if an asset is missing; stage the missing checkpoint as a separate SLURM job.

- [ ] **Step 4: Run scheduler validation**

```bash
ssh aws-dfw-cs-001-login-01.nvidia.com '
  cd /lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna/RL-vllm024-upgrade-20260707 &&
  MAX_STEPS=2 RUN_TAG=vllm024-dynamicsd-smoke-20260707 \
  experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
    test-only all all
'
```

Expected: six successful `sbatch --test-only` responses.

- [ ] **Step 5: Submit all smoke variants**

```bash
ssh aws-dfw-cs-001-login-01.nvidia.com '
  cd /lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna/RL-vllm024-upgrade-20260707 &&
  MAX_STEPS=2 RUN_TAG=vllm024-dynamicsd-smoke-20260707 \
  experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
    submit all all
'
```

Capture all six job IDs from
`experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-smoke-20260707/submissions.tsv`.

- [ ] **Step 6: Monitor five minutes and through Step 2**

Poll `squeue` and tail at most 100 lines per driver log. Fail the gate on `Traceback`, `EngineCore failed`, `EADDRINUSE`, OOM, missing draft counters, or early exit. A successful smoke must show Step 2 policy training and `COMPLETED|0:0` in `sacct`.

### Task 6: Submit and Collect the 20-Step Matrix

**Files:**
- Generated remotely: `experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/`
- Pulled locally: `experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/summary/summary.csv`
- Pulled locally: `experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/summary/summary.json`

**Interfaces:**
- Consumes: six passing smoke gates.
- Produces: six final 20-step runs and matched performance/health summaries.

- [ ] **Step 1: Submit the full matrix**

```bash
MAX_STEPS=20 RUN_TAG=vllm024-dynamicsd-step20-20260707 \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit all all
```

- [ ] **Step 2: Monitor startup for five minutes**

Verify all jobs leave PENDING or report the scheduler reason; after RUNNING, check model load, PIECEWISE graph capture, rollout generation, and policy training. Do not claim progress solely from RUNNING state.

- [ ] **Step 3: Collect only after terminal states**

Run the collector against the full-run manifest. Incomplete or failed jobs remain explicit rows and cause a nonzero collector exit.

- [ ] **Step 4: Validate the final table**

Require all six rows to report Steps 2-20, matching recipe and runtime provenance, positive SpecDec counters for Eagle-3 variants, and finite timing/throughput values. Report generation and E2E time/throughput ratios against baseline and DynamicSD ratios against fixed K.

- [ ] **Step 5: Pull only final result artifacts**

Use `rsync` for `summary.csv`, `summary.json`, `submissions.tsv`, and concise terminal-state logs. Do not copy venvs, Ray session trees, model caches, or checkpoints.

- [ ] **Step 6: Commit reproducible result metadata**

Commit the manifest and summary artifacts after removing secrets and verifying that all W&B links resolve:

```bash
git add \
  experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/submissions.tsv \
  experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/summary/summary.csv \
  experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/summary/summary.json
git commit -s -m "report: add NeMo-RL DynamicSD results"
git push fork sna/nemorl-vllm024-upgrade
```
