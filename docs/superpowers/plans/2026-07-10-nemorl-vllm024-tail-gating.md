# NeMo-RL vLLM 0.24 Tail-Gated Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add matched stock DynamicSD, FastRL-inspired threshold gating, and EfficientRollout roofline gating experiments for external Eagle-3 drafters in NeMo-RL with vLLM 0.24.

**Architecture:** Keep policy in a typed NeMo-RL `TailGateController` and a custom vLLM `Scheduler` subclass selected through the official `scheduler_cls` extension point. Use NeMo-RL's version-pinned wheel-source patch mechanism only to add tail-gate fields to `SpeculativeConfig` and `SchedulerOutput`, make Model Runner V2 consume runtime K, preserve external-drafter KV state with an advance-only K0 path, and relay cumulative telemetry through the existing worker metric RPC. Keep Model Runner V1 stock DynamicSD unmodified as a separate cohort.

**Tech Stack:** Python 3.13, NeMo-RL, vLLM 0.24.0 (`ee0da84a`), PyTorch, pytest, Hydra/OmegaConf, SLURM, W&B.

## Global Constraints

- Work on branch `sna/nemorl-vllm024-tail-gating` from commit `9c71ad64` in the existing isolated worktree.
- Run `git submodule update --init --recursive` after switching branches.
- Do not modify GRPO, rewards, target sampling, rejection sampling, policy training, or model checkpoints.
- Target sampling remains `temperature=1.0`, `top_p=1.0`, and rejection sampling remains `standard`.
- Neural drafter sampling remains `probabilistic`; suffix has no draft-sampling arm.
- External Eagle-3 has independent KV state. K0 must either advance that state or rebuild it before reactivation; it must never silently skip and later reuse stale state.
- V1 and V2 are separate cohorts. Never compute speedup across runner versions.
- Every V1 arm explicitly sets `VLLM_USE_V2_MODEL_RUNNER=0` and uses `PIECEWISE` CUDA graphs.
- Every V2 arm explicitly sets `VLLM_USE_V2_MODEL_RUNNER=1`, `enforce_eager=false`, and `FULL_AND_PIECEWISE` CUDA graphs.
- Qwen3-32B is the required V2 cohort. Qwen3-30B-A3B is V1 by default; run a V2 support smoke before creating any Qwen3-30B-A3B V2 comparison. If vLLM rejects that architecture, run its threshold and roofline gates in V1 and do not force a V2 production row.
- Baseline and candidate rows must match model, recipe, cluster, container, vLLM commit, TP/DP/EP, sampling, OSL, batch geometry, graph mode, and step window.
- Initial runtime support is Eagle/Eagle-3 fixed K. Reject DFlash, PARD parallel drafting, suffix decoding, and variable K in the binary-gate path.
- Roofline configuration loading and all unsupported combinations fail closed before model loading.
- Use the upstream performance recipes unchanged except for run length, checkpoint disabling, W&B, vLLM 0.24 runtime, graph mode, runner selection, and SpecDec/gate settings.
- `checkpointing.enabled=false`; no experimental checkpoints are written.
- Preserve pre-existing untracked `tests/unit/unit_results.json` and `tests/unit/unit_results/`.

---

### Task 1: Pure Tail-Gate Controller and Roofline Predictor

**Files:**
- Create: `nemo_rl/models/generation/vllm/tail_gate.py`
- Create: `nemo_rl/models/generation/vllm/sd_toggle/__init__.py`
- Create: `nemo_rl/models/generation/vllm/sd_toggle/config.py`
- Create: `nemo_rl/models/generation/vllm/sd_toggle/roofline.py`
- Create: `nemo_rl/models/generation/vllm/sd_toggle/predict.py`
- Create: `nemo_rl/models/generation/vllm/sd_toggle/NOTICE`
- Create: `tests/unit/models/generation/test_vllm_tail_gate.py`
- Create: `tests/unit/models/generation/test_vllm_sd_toggle.py`

**Interfaces:**
- Consumes: a JSON-compatible tail-gate dictionary and EfficientRollout-compatible roofline JSON.
- Produces: `TailGateConfig`, `TailGateObservation`, `TailGateDecision`, `TailGateTelemetry`, and `TailGateController.observe()` / `finish_rollout()`.

- [ ] **Step 1: Add failing roofline tests**

Cover exact upstream behavior:

```python
def test_roofline_enables_only_above_margin(tmp_path):
    config = make_calibration(tmp_path)
    speedup = predict_speedup(B=4, S=4096, gamma=5, L_accept=3.0, config=config)
    assert should_enable_sd(config, 4, 4096, 5, 3.0, margin=speedup - 1.0)
    assert not should_enable_sd(
        config, 4, 4096, 5, 3.0, margin=speedup - 1.0 + 1e-6
    )

def test_roofline_rejects_non_finite_prediction(tmp_path):
    config = make_calibration(tmp_path, BW_eff=0.0)
    with pytest.raises(ValueError, match="finite"):
        should_enable_sd(config, 4, 4096, 5, 3.0, margin=0.05)
```

- [ ] **Step 2: Run the roofline tests and verify RED**

Run:

```bash
uv run --no-sync pytest tests/unit/models/generation/test_vllm_sd_toggle.py -q
```

Expected: collection fails because `nemo_rl.models.generation.vllm.sd_toggle` does not exist.

- [ ] **Step 3: Port the dependency-light EfficientRollout model**

Port `config.py`, `roofline.py`, and `predict.py` from local reference commit
`0ff0bc5bc7eb323a96391d556f4485137dd7f0f1`. Retain Apache-2.0 attribution in
`NOTICE`. Keep SciPy fitting and plotting out of the runtime package. Export:

```python
from .config import SDToggleConfig, load_config
from .predict import predict_decision, should_enable_sd
from .roofline import predict_speedup
```

`should_enable_sd` must reject non-finite or non-positive denominator results
instead of treating them as profitable.

- [ ] **Step 4: Run roofline tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Add failing controller tests**

Cover these state transitions:

```python
def test_threshold_gate_requires_ramp_and_consecutive_checks():
    gate = TailGateController(
        TailGateConfig(mode="threshold", threshold=32, consecutive_checks=3, gamma=5)
    )
    assert not gate.observe(TailGateObservation(8, 2048, True)).enabled
    assert not gate.observe(TailGateObservation(64, 2048, True)).enabled
    assert not gate.observe(TailGateObservation(32, 4096, True)).enabled
    assert not gate.observe(TailGateObservation(31, 4097, True)).enabled
    decision = gate.observe(TailGateObservation(30, 4098, True))
    assert decision.enabled
    assert decision.just_activated
    assert gate.observe(TailGateObservation(64, 4099, True)).enabled

def test_gate_reset_keeps_previous_rollout_acceptance():
    gate = TailGateController(make_roofline_gate())
    gate.finish_rollout(accepted_tokens=40, num_drafts=20, validation=False)
    assert gate.expected_accept_length == 3.0
    assert not gate.enabled

def test_validation_rollout_does_not_replace_acceptance():
    gate = TailGateController(make_roofline_gate(expected_accept_length=2.5))
    gate.finish_rollout(accepted_tokens=100, num_drafts=20, validation=True)
    assert gate.expected_accept_length == 2.5
```

- [ ] **Step 6: Implement the pure controller**

Use frozen dataclasses for config and observations. The controller states are
`RAMPING_OFF`, `ARMED_OFF`, and `ON_LATCHED`. `observe()` must:

```python
def observe(self, observation: TailGateObservation) -> TailGateDecision:
    self._tick += 1
    if self.config.mode == "off":
        return self._decision(enabled=True, reason="controller_off")
    if self._enabled:
        return self._decision(enabled=True, reason="latched")
    if not observation.is_decode or observation.active_requests == 0:
        return self._decision(enabled=False, reason="not_decode")
    if not self._seen_ramp:
        if observation.active_requests > self.config.ramp_threshold:
            self._seen_ramp = True
        return self._decision(enabled=False, reason="ramp_guard")
    predicate = self._threshold_predicate(observation)
    if predicate:
        self._qualifying_checks += 1
    else:
        self._qualifying_checks = 0
    if self._qualifying_checks >= self.config.consecutive_checks:
        self._enabled = True
        return self._decision(enabled=True, just_activated=True, reason="activated")
    return self._decision(enabled=False, reason="waiting")
```

Roofline mode uses `predict_decision()` and the previous training rollout's
`expected_accept_length`. `finish_rollout()` computes
`1 + accepted_tokens / num_drafts`, preserves the previous value for zero-cycle
or validation rollouts, then resets only rollout-local latch state.

- [ ] **Step 7: Run controller tests and commit**

Run:

```bash
uv run --no-sync pytest \
  tests/unit/models/generation/test_vllm_tail_gate.py \
  tests/unit/models/generation/test_vllm_sd_toggle.py -q
```

Expected: all tests pass.

Commit:

```bash
git add nemo_rl/models/generation/vllm/tail_gate.py \
  nemo_rl/models/generation/vllm/sd_toggle \
  tests/unit/models/generation/test_vllm_tail_gate.py \
  tests/unit/models/generation/test_vllm_sd_toggle.py
git commit -s -m "feat(vllm): add tail gate controller"
```

### Task 2: Tail-Gated Scheduler and NeMo-RL Validation

**Files:**
- Create: `nemo_rl/models/generation/vllm/tail_gate_scheduler.py`
- Modify: `nemo_rl/models/generation/__init__.py`
- Create: `tests/unit/models/generation/test_vllm_tail_gate_scheduler.py`
- Modify: `tests/unit/models/generation/test_vllm_generation.py`

**Interfaces:**
- Consumes: Task 1 controller and vLLM `VllmConfig`.
- Produces: importable scheduler class path `nemo_rl.models.generation.vllm.tail_gate_scheduler.TailGatedScheduler`.

- [ ] **Step 1: Add failing configuration-validation tests**

Test that tail gating requires Eagle/Eagle-3, a positive static K, a supported
mode, a positive threshold/check count, no stock DynamicSD schedule, no internal
vLLM DP, and a roofline config path for roofline mode. Include:

```python
with pytest.raises(ValueError, match="external Eagle"):
    _validate_speculative_config(config_with(method="suffix", sd_tail_gate_mode="threshold"))

with pytest.raises(ValueError, match="mutually exclusive"):
    _validate_speculative_config(config_with(
        sd_tail_gate_mode="threshold",
        num_speculative_tokens_per_batch_size=[[1, 32, 5]],
    ))
```

- [ ] **Step 2: Implement validation and verify tests pass**

Add validation without changing sampling parameters. The validated dictionary
continues unchanged into vLLM.

- [ ] **Step 3: Add failing scheduler tests with a stub base scheduler**

Test:

- runtime K is zero before activation and fixed gamma after activation;
- pending drafts are allowed to drain on the first K0 step;
- telemetry contains state, tick, active batch, mean sequence length, predicted
  speedup sum/count, expected acceptance length, and just-activated flag;
- `update_from_output()` accumulates accepted tokens using generated length minus
  the bonus token;
- empty running/waiting queues reset the rollout latch but retain training MAL.

- [ ] **Step 4: Implement `TailGatedScheduler`**

Subclass official vLLM `Scheduler`. Construct `TailGateConfig` from the patched
`SpeculativeConfig`. In `schedule()`:

```python
output = super().schedule()
observation = self._build_observation(output)
decision = self._tail_gate.observe(observation)
output.num_spec_tokens_to_schedule = self.num_spec_tokens if decision.enabled else 0
self._write_tail_gate_output(output, observation, decision)
return output
```

Use `len(self.running)` for the upstream-compatible ramp guard and record a
separate decode-active count for diagnostics. Do not use
`len(output.num_scheduled_tokens)` as the profitability input.

Override `update_from_output()` only to accumulate acceptance before delegating
and to reset after the final request. Preserve all superclass return values and
exception behavior.

- [ ] **Step 5: Run scheduler and validation tests and commit**

Run:

```bash
uv run --no-sync pytest \
  tests/unit/models/generation/test_vllm_tail_gate_scheduler.py \
  tests/unit/models/generation/test_vllm_generation.py -q
```

Expected: all tests pass.

Commit the four files with:

```bash
git commit -s -m "feat(vllm): add tail-gated scheduler"
```

### Task 3: Version-Pinned vLLM 0.24 Runtime Patch

**Files:**
- Modify: `nemo_rl/models/generation/vllm/patches.py`
- Modify: `tests/unit/models/generation/test_vllm_patches.py`

**Interfaces:**
- Consumes: `TailGatedScheduler` and `SchedulerOutput` telemetry contract.
- Produces: a fail-fast source patch for official vLLM 0.24 commit layout.

- [ ] **Step 1: Add failing source-contract tests**

Build temporary vLLM source fixtures for:

- `config/speculative.py`;
- `v1/core/sched/output.py`;
- `v1/worker/gpu/model_runner.py`;
- `v1/worker/gpu/spec_decode/autoregressive/speculator.py`; and
- `v1/worker/gpu_model_runner.py`.

Assert the patch adds tail-gate config fields and output telemetry fields,
reuses vLLM 0.24's existing `SchedulerOutput.num_spec_tokens_to_schedule`,
passes runtime K through V2 `ExecuteModelState`, implements advance-only K0,
clears the fixed-width request rows before publishing a zero-width proposal,
and records telemetry in V1 and V2. Reapplying the patch must be idempotent.
Every old/new anchor across all five files must be validated before the first
write, so a changed upstream snippet raises `RuntimeError` without leaving a
partially patched installation.

- [ ] **Step 2: Patch config and scheduler output**

Add exact fields:

```python
sd_tail_gate_mode: str = "off"
sd_tail_gate_threshold: int | None = None
sd_tail_gate_consecutive_checks: int = 10
sd_tail_gate_margin: float = 0.05
sd_tail_gate_config_path: str | None = None
sd_tail_gate_off_mode: str = "advance_only"
```

Add cumulative/instantaneous telemetry to `SchedulerOutput` with scalar defaults
so stock schedulers remain compatible. Do not add or rename
`num_spec_tokens_to_schedule`; it is already an official vLLM 0.24 field.

- [ ] **Step 3: Patch Model Runner V2 runtime K**

In `execute_model()`, validate
`scheduler_output.num_spec_tokens_to_schedule` before the target forward and
store it in an extended `ExecuteModelState`. At the proposal call, read runtime
K from that state. `AutoRegressiveSpeculator.propose()` accepts an optional
runtime K constrained to `0` or the configured static maximum during the binary
phase.

For K0 external Eagle/Eagle-3:

1. execute the first drafter state-advance pass;
2. do not execute the remaining `K-1` serial draft-decode iterations;
3. return a zero-width draft tensor;
4. clear the corresponding rows of the fixed-width request-state buffer; and
5. publish the zero-width tensor to `DraftTokensHandler` instead of the fixed
   backing tensor, so no stale IDs remain consumable.

Reject unsupported speculators and intermediate K values before executing a
model forward. Preserve official fixed-K behavior when gate mode is `off`.

- [ ] **Step 4: Patch V1/V2 telemetry recording**

Store cumulative scalar counters on each model runner under
`_nrl_tail_gate_metrics`. Counter names start with
`vllm:spec_decode_tail_gate_` so the existing collector includes them.

- [ ] **Step 5: Run source-contract tests and commit**

Run:

```bash
uv run --no-sync pytest tests/unit/models/generation/test_vllm_patches.py -q
```

Expected: all tests pass, including idempotency and changed-source failures.

Commit:

```bash
git commit -s -m "feat(vllm): patch runtime tail gating"
```

### Task 4: Tail-Gate Metrics and W&B Derivations

**Files:**
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `nemo_rl/models/generation/vllm/utils.py`
- Modify: `tests/unit/models/generation/test_vllm_generation.py`
- Modify: `tests/unit/models/generation/test_vllm_utils.py`

**Interfaces:**
- Consumes: model-runner `_nrl_tail_gate_metrics` counters.
- Produces: step metrics under `vllm/tail_gate_*` that GRPO already sends to W&B.

- [ ] **Step 1: Add failing aggregation tests**

Use cumulative snapshots to verify:

```python
assert metrics["vllm/tail_gate_enabled_step_ratio"] == pytest.approx(0.25)
assert metrics["vllm/tail_gate_activation_batch"] == pytest.approx(16.0)
assert metrics["vllm/tail_gate_activation_seq_len"] == pytest.approx(8192.0)
assert metrics["vllm/tail_gate_predicted_speedup"] == pytest.approx(1.12)
assert metrics["vllm/tail_gate_advance_only_step_ratio"] == pytest.approx(0.75)
```

Counters with zero denominator produce `0.0`, not NaN.

- [ ] **Step 2: Expose worker counters**

Extend `get_cudagraph_dispatch_metrics()` to merge
`model_runner._nrl_tail_gate_metrics` without changing existing CUDA graph
counters. Reject list/scalar conflicts.

- [ ] **Step 3: Compute derived metrics**

`compute_spec_decode_metrics()` converts cumulative counter deltas into enabled
ratio, activation means, predicted-speedup mean, K histogram, and advance-only
ratio. Do not delta instantaneous gauges; all raw tail-gate inputs are sums and
counts.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run --no-sync pytest \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_utils.py -q
```

Expected: all tests pass.

Commit:

```bash
git commit -s -m "bench(vllm): report tail gate metrics"
```

### Task 5: Matched V1/V2 Launcher and Calibration Assets

**Files:**
- Create: `experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh`
- Create: `experiments/vllm_024_upgrade/calibrate_tail_gate.py`
- Create: `tests/test_vllm_024_tail_gate_launch.py`
- Create: `tests/test_vllm_024_tail_gate_calibration.py`
- Modify: `experiments/vllm_024_upgrade/README.md`

**Interfaces:**
- Consumes: scheduler class path and config keys from Tasks 2-3.
- Produces: dry-run/test-only/submit commands and provenance manifests.

- [ ] **Step 1: Add failing launcher tests**

Assert exact variant contracts:

- V1 baseline/fixed/dynamic use `VLLM_USE_V2_MODEL_RUNNER=0` and `PIECEWISE`;
- V2 baseline/always-on/threshold/roofline use
  `VLLM_USE_V2_MODEL_RUNNER=1` and `FULL_AND_PIECEWISE`;
- only V2 gated arms set `scheduler_cls` and tail-gate fields;
- Qwen3-32B and Qwen3-30B-A3B use their upstream performance recipes, four
  nodes, and `--segment=4`;
- preserve `64x32` rollouts, train GBS `512`, max sequence/output `4096`,
  `max_num_batched_tokens=16384`, `max_num_seqs=1024`, and engine length `4128`;
- Qwen3-30B-A3B uses target/draft TP `1/1`; Qwen3-32B uses target/draft TP `2/1`;
- max steps defaults to 20, checkpointing is disabled, Triton MoE remains set,
  W&B project/run/group are explicit, and logs/manifests use Lustre;
- every submission records runner, graph mode, gate mode, K, threshold,
  consecutive checks, roofline config hash, commit, container, recipe, and job ID.

- [ ] **Step 2: Implement the launcher**

Support variants:

```text
baseline_v1 always_on_v1_k5 stock_dynamic_v1
baseline_v2 always_on_v2_k5 fastrl_threshold_v2_k5 efficient_roofline_v2_k5
```

Support `dry-run`, `test-only`, and `submit`. Refuse dirty/unpushed code for
submit, but ignore the known untracked unit-result artifacts. Run scheduler
`--test-only` before `sbatch`, and use no Lyris `--gres` option.

- [ ] **Step 3: Add calibration parser tests**

Given measured rows `(B,S,K,T_T,T_D,T_V)`, verify deterministic JSON output,
complete provenance, positive fitted parameters, and rejection of missing or
mixed model/TP/cluster rows.

- [ ] **Step 4: Implement calibration conversion**

Reuse EfficientRollout's fitted schema. The first implementation consumes
measured CSV; it does not run GPUs locally. Emit one JSON per
`(model,target_tp,draft_tp,cluster,container,K-set)` and a SHA256 sidecar.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
bash -n experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh
uv run --no-sync pytest \
  tests/test_vllm_024_tail_gate_launch.py \
  tests/test_vllm_024_tail_gate_calibration.py -q
```

Expected: all tests pass.

Commit:

```bash
git commit -s -m "bench(vllm): launch tail-gated rollouts"
```

### Task 6: Result Collector and HTML Report Integration

**Files:**
- Create: `experiments/vllm_024_upgrade/summarize_tail_gated_specdec.py`
- Create: `tests/test_vllm_024_tail_gate_summary.py`
- Modify: `public/reports/lyris_nemorl_perfcfg_specdec_live_status_latest.html`

**Interfaces:**
- Consumes: launcher manifest and W&B histories.
- Produces: validated CSV/JSON plus runner-separated HTML tables and charts.

- [ ] **Step 1: Add failing summary tests**

Test Step 2-20 means, matched-baseline keys, V1/V2 separation, partial-run
labels, W&B links, gate activation metrics, CUDA graph health, reward/logprob
health, and rejection of cross-runner or cross-graph speedups.

- [ ] **Step 2: Implement collector**

Emit one row per job with:

```text
model,runner,variant,gate_mode,K,steps,job_id,wandb_url,
e2e_time,generation_time,e2e_tps_gpu,generation_tps_gpu,
policy_time,logprob_time,acceptance_rate,mean_accept_len,
gate_enabled_ratio,activation_batch,activation_seq_len,predicted_speedup,
target_graph_ratio,draft_prefill_graph_ratio,draft_decode_graph_ratio,
reward,response_length,approx_kl,policy_loss,status,source
```

Compute speedups only within an exact matched key.

- [ ] **Step 3: Integrate compact report sections**

Add separate `Model Runner V1` and `Model Runner V2` tables, one compact chart
per model, centered two-line legends, and a one-sentence finding generated only
from final rows. Keep partial rows visibly marked and never overwrite historical
cohorts.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run --no-sync pytest tests/test_vllm_024_tail_gate_summary.py -q
```

Expected: all tests pass.

Commit:

```bash
git commit -s -m "report: add tail-gated specdec results"
```

### Task 7: Local Integration Verification and GB200 Smoke Matrix

**Files:**
- Generated: `experiments/vllm_024_upgrade/runs/<run-tag>/submissions.tsv`
- Generated: `experiments/vllm_024_upgrade/runs/<run-tag>/logs/`

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: verified two-step smoke jobs and calibration measurements.

- [ ] **Step 1: Run the focused local suite**

Run:

```bash
uv run --no-sync pytest \
  tests/unit/models/generation/test_vllm_tail_gate.py \
  tests/unit/models/generation/test_vllm_sd_toggle.py \
  tests/unit/models/generation/test_vllm_tail_gate_scheduler.py \
  tests/unit/models/generation/test_vllm_patches.py \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_utils.py \
  tests/test_vllm_024_tail_gate_launch.py \
  tests/test_vllm_024_tail_gate_calibration.py \
  tests/test_vllm_024_tail_gate_summary.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Push and update the selected GB200 cluster checkout**

Push the branch, pull it once on Lyris or AWS-DFW, and run recursive submodule
initialization. Use the existing staged nightly container and Lustre checkout.

- [ ] **Step 3: Run launcher test-only for Qwen3-32B**

Render all seven variants with `MAX_STEPS=2`, validate scheduling, then submit.
Monitor for at least five minutes and through the first policy-training step.

- [ ] **Step 4: Validate smoke gates**

Require:

- correct V1/V2 runner log;
- correct CUDA graph mode and no capture fallback;
- target and draft weights loaded;
- K0 advance-only counters before activation;
- activation after ramp and consecutive checks;
- positive drafts/accepts after activation;
- no stale-drafter, invalid-token, NaN, OOM, NCCL, or q-cache error;
- W&B metrics and provenance present; and
- clean completion through policy training.

- [ ] **Step 5: Collect calibration data**

Measure B `1,2,4,8,16,32,64,128`, S `2048,4096,8192,16384,32768`, and K
`1,3,5` using the exact V2 graph/runtime topology. Fit and validate the roofline
JSON before accepting the roofline smoke.

Require target-latency MAPE below `10%` and toggle sign accuracy at least `90%`.

### Task 8: FastRL Rebuild Mode and Adaptive-K V2

**Files:**
- Modify: `nemo_rl/models/generation/vllm/tail_gate.py`
- Modify: `nemo_rl/models/generation/vllm/tail_gate_scheduler.py`
- Modify: `nemo_rl/models/generation/vllm/patches.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `nemo_rl/models/generation/vllm/utils.py`
- Modify: `experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh`
- Modify: `tests/unit/models/generation/test_vllm_tail_gate.py`
- Modify: `tests/unit/models/generation/test_vllm_tail_gate_scheduler.py`
- Modify: `tests/unit/models/generation/test_vllm_patches.py`
- Modify: `tests/unit/models/generation/test_vllm_generation.py`
- Modify: `tests/unit/models/generation/test_vllm_utils.py`
- Modify: `tests/test_vllm_024_tail_gate_launch.py`
- Modify: `tests/test_vllm_024_tail_gate_summary.py`

**Interfaces:**
- Consumes: validated binary V2 gate and K1/K3/K5 calibration.
- Produces: `fastrl_rebuild_v2_k5`, predefined adaptive K, frozen bucketed
epsilon-greedy K, and EfficientRollout acceptance-aware K-ladder variants.

- [ ] **Step 1: Add failing rebuild correctness tests**

Test that OFF mode performs no drafter decode, activation enters a rebuilding
state, every active request is rebuilt before any drafts are returned, and
failed/partial rebuild blocks activation.

- [ ] **Step 2: Implement rebuild-on-enable as an isolated mode**

Do not alter advance-only behavior. Rebuild mode must use official vLLM request
state and proposer APIs; no direct KV tensor mutation. Record rebuild time and
request/token counts.

- [ ] **Step 3: Add failing variable-K tests**

Cover K `1/3/5`, rejection of uncaptured K, fixed maximum buffer sizing,
per-K graph dispatch, one-step K transition delay, and no stale draft tokens.

- [ ] **Step 4: Implement V2 variable K**

Keep buffers and lookahead sized to K5. Execute only runtime K iterations and
return only runtime K draft columns. Capture or validate graph coverage for each
K. Reject DBO and unsupported proposer types.

- [ ] **Step 5: Add strategy selectors**

Implement:

- offline predefined fastest K per batch bucket;
- bucketed epsilon-greedy training with reward `accepted_target_tokens / decode_ms`,
  frozen to epsilon `0` for final evaluation; and
- acceptance hysteresis ladder with up threshold `0.94`, down threshold `0.85`,
  and two consecutive training rollouts, excluding validation.

- [ ] **Step 6: Run local tests, smokes, and commit**

Run all focused tests from Task 7, then two-step smokes for each new arm. Commit
only after graph, correctness, and W&B gates pass.

### Task 9: Matched 20-Step and 32K Experiments

**Files:**
- Generated: final manifests, CSV/JSON summaries, and report data.
- Modify: `public/reports/lyris_nemorl_perfcfg_specdec_live_status_latest.html`

- [ ] **Step 1: Submit matched Qwen3-32B 20-step jobs**

Submit only variants whose two-step smoke passed. Monitor startup for five
minutes and preserve all logs.

- [ ] **Step 2: Submit matched Qwen3-30B-A3B 20-step jobs**

Repeat the same runner-separated matrix and gates.

- [ ] **Step 3: Submit the 32K-output long-tail cohort**

Keep prompt count, generations per prompt, training batch geometry, and all
recipe-owned settings unchanged. Change only `max_new_tokens` and required
context headroom consistently across matched arms.

- [ ] **Step 4: Publish final and partial results**

Compute Steps 2-20 means, speedups within matched cohorts, W&B links, gate
activation behavior, CUDA graph coverage, generation ratio, and accuracy-health
metrics. Partial timeout rows remain labeled partial.

- [ ] **Step 5: Final verification and review**

Run the focused local suite again, verify report links/files, request a broad
code review, and fix all Critical or Important findings before recommending a
gate policy.
