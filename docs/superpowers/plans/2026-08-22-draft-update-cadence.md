# Draft Update Cadence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in fixed and adaptive drafter update schedules while preserving target-policy refit on every policy step, exact resume behavior, and the existing `always` behavior.

**Architecture:** One controller-owned pure state machine consumes a count-weighted acceptance observation and emits one immutable decision per policy step. Megatron workers use that decision to gate hidden capture, draft forward/backward, and draft optimizer mutation, while weight synchronization always transfers target-policy weights and independently selects whether to transfer draft weights. Sync GRPO and the supported single-controller path share the same scheduler and observation helpers.

**Tech Stack:** Python 3.12, Pydantic v2, PyTorch, Megatron-Core, Ray, TransferQueue/DataPlane, vLLM, pytest, Ruff, Pyrefly.

**Spec:** `docs/superpowers/specs/2026-08-22-online-drafter-efficiency-and-cadence-design.md`

## Global Constraints

- Project 1 must have terminal GREEN packed E2E and performance gates before this project begins.
- `always` is the default and preserves the current step-one and per-step draft update/refit behavior.
- Target-policy weights are synchronized to generation after every successful policy step. Cadence controls only draft learning and draft payload application.
- A skipped sparse update performs no hidden capture, draft provider/loss/backward, draft parameter/moment/per-parameter-step mutation, or draft refit.
- The shared Megatron LR schedule remains indexed by global consumed samples. Draft parameter groups and optimizer state pause on skips, but their displayed scheduled LR may advance.
- Acceptance is `sum(accepted_tokens) / sum(draft_tokens)` across every rollout batch selected for the policy step. Per-batch rates are never averaged.
- Strict Step+1 acceptance/serving-version evidence, exclusive per-update receipts, and terminal science receipts are opt-in experiment instrumentation and run only when `cadence_runtime.enabled=true`. With the default `false`, legacy `always` and nonadaptive single-controller paths neither read extra science fields nor change their transport behavior. Experiment-mode startup fails before actor/worker creation if the controller cannot provide the canonical selected-rollout counts and serving-version provenance.
- `applied_draft_version` is the successful draft-refit decision ID and is distinct from target-policy `weight_version`, which advances every policy step.
- Workers never inspect acceptance metrics and never make rank-local cadence decisions.
- Scheduler checkpoints use state version 1 and exact resolved-config equality. A fresh fixed/adaptive run creates new state; only a legacy resumed checkpoint without scheduler state is restricted to `always` mode.
- Adaptive single-controller mode requires on-policy, zero-staleness selected rollouts produced by the current `applied_draft_version`.
- Decision history in scheduler state is JSON-safe and bounded to the newest 64 entries. A separate append-only `decision-ledger.jsonl` records every closed decision, and checkpoint receipts bind an immutable prefix so resumed analysis can merge the prefix with a new suffix without expecting the bounded in-memory history to contain all 1000 steps. Attempted, successful, skipped, and forced update/refit counters are checkpointed.
- Every successful serving-draft refit durably snapshots the exact applied draft tensor payload, its SHA256, and its decision ID. Resume startup syncs current target bytes separately, restores the serving draft from that snapshot, verifies the apply receipt, and only then republishes the matching saved `applied_draft_version`; it never transfers newer trainable draft bytes under an older serving version.
- IPC and collective transports may advertise component selection only after target-only coverage tests pass. HTTP, checkpoint-engine, Megatron, NCCL-reshard, and `VllmRemoteSparseWeightSynchronizer` fail fixed/adaptive startup as unsupported.
- Every implementation commit uses `git commit -S -s`; `git verify-commit HEAD` must pass before push.

## File Structure

- `nemo_rl/models/policy/draft_config.py`: user-facing discriminated schedule schemas.
- `nemo_rl/algorithms/draft_update_schedule.py`: pure scheduler, immutable decision, counters, bounded history, and state serialization.
- `nemo_rl/algorithms/draft_update_observation.py`: count validation and count-weighted acceptance reconstruction.
- `nemo_rl/models/megatron/draft/optimizer.py`: unconditional draft-only optimizer-group construction, including `draft.optimizer=null`, and suspension around `optimizer.step()` only.
- `nemo_rl/models/megatron/train.py`: existing hidden-capture gate consumed by worker calls; no cadence policy belongs here.
- `nemo_rl/models/policy/tq_policy.py` and `nemo_rl/models/policy/workers/megatron_policy_worker.py`: immutable decision fanout and worker compute gates.
- `nemo_rl/weight_sync/interfaces.py`, transport implementations, and `nemo_rl/weight_sync/factory.py`: component-selection capability and startup validation.
- `nemo_rl/algorithms/grpo.py` and `nemo_rl/algorithms/grpo_sync.py`: checkpoint schema, synchronous controller ownership, and target-every-step refit.
- `nemo_rl/experience/payload.py`, `nemo_rl/experience/rollout_manager.py`, and `nemo_rl/algorithms/async_utils/replay_buffer.py`: separate target/draft provenance and acceptance counts.
- `nemo_rl/algorithms/single_controller.py` and `nemo_rl/algorithms/single_controller_utils/config.py`: adaptive selection constraints and shared scheduler integration.
- `docs/superpowers/plans/2026-08-22-draft-update-cadence-experiments.md`: pilot and long-validation harness, submission, and statistical decision plan.

---

### Task 1: Define the discriminated schedule configuration

**Files:**
- Modify: `nemo_rl/models/policy/draft_config.py`
- Modify: `tests/unit/models/policy/test_draft_config.py`

**Interfaces:**
- Consumes: existing `DFlashDraftConfig` and `DSparkDraftConfig` constructors.
- Produces: `DraftUpdateScheduleConfig`, `AlwaysDraftUpdateScheduleConfig`, `FixedDraftUpdateScheduleConfig`, and `AdaptiveDraftUpdateScheduleConfig`; both block-draft configs expose `update_schedule: DraftUpdateScheduleConfig`.

- [ ] **Step 1: Write RED validation and default tests.**

```python
import math

import pytest
from pydantic import ValidationError

from nemo_rl.models.policy.draft_config import (
    AdaptiveDraftUpdateScheduleConfig,
    AlwaysDraftUpdateScheduleConfig,
    DFlashDraftConfig,
    FixedDraftUpdateScheduleConfig,
)


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "adaptive", "min_interval": 0},
        {"mode": "adaptive", "min_interval": 20, "max_interval": 10},
        {"mode": "adaptive", "ewma_alpha": 0.0},
        {"mode": "adaptive", "ewma_alpha": 1.1},
        {"mode": "adaptive", "degradation_threshold": math.inf},
        {
            "mode": "adaptive",
            "recovery_threshold": 0.02,
            "degradation_threshold": 0.02,
        },
    ],
)
def test_adaptive_schedule_rejects_invalid_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AdaptiveDraftUpdateScheduleConfig.model_validate(values)


def test_schedule_members_forbid_unrelated_fields() -> None:
    with pytest.raises(ValidationError):
        AlwaysDraftUpdateScheduleConfig.model_validate(
            {"mode": "always", "fixed_interval": 10}
        )
    with pytest.raises(ValidationError):
        FixedDraftUpdateScheduleConfig.model_validate(
            {"mode": "fixed", "action": "adaptive", "fixed_interval": 10}
        )


def test_dflash_omitted_schedule_resolves_to_always_member_only() -> None:
    config = DFlashDraftConfig(
        enabled=True,
        gamma=5,
        anchors_per_sample=4,
        mask_token_id=151665,
        target_hidden_state_layer_ids=[1, 17, 33],
    )
    assert config.update_schedule.model_dump(mode="json") == {"mode": "always"}
```

- [ ] **Step 2: Run the RED tests and confirm the missing imports.**

Run: `uv run --group test pytest -q tests/unit/models/policy/test_draft_config.py -k 'schedule'`

Expected: FAIL during collection with `ImportError: cannot import name 'AdaptiveDraftUpdateScheduleConfig'`.

- [ ] **Step 3: Add the schedule models and nest the default under DFlash and DSpark.**

```python
import math


class AlwaysDraftUpdateScheduleConfig(BaseModel, extra="forbid"):
    mode: Literal["always"] = "always"


class FixedDraftUpdateScheduleConfig(BaseModel, extra="forbid"):
    mode: Literal["fixed"] = "fixed"
    action: Literal["sparse_update", "refit_only"]
    fixed_interval: Annotated[int, Field(gt=0)]


class AdaptiveDraftUpdateScheduleConfig(BaseModel, extra="forbid"):
    mode: Literal["adaptive"] = "adaptive"
    action: Literal["sparse_update"] = "sparse_update"
    min_interval: Annotated[int, Field(gt=0)] = 10
    max_interval: Annotated[int, Field(gt=0)] = 100
    ewma_alpha: float = 0.1
    degradation_threshold: float = 0.02
    recovery_threshold: float = 0.01
    min_observations: Annotated[int, Field(gt=0)] = 20
    max_burst_updates: Annotated[int, Field(gt=0)] = 10

    @model_validator(mode="after")
    def validate_adaptive_schedule(self) -> Self:
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be at least min_interval")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        thresholds = (self.recovery_threshold, self.degradation_threshold)
        if not all(math.isfinite(value) for value in thresholds):
            raise ValueError("adaptive thresholds must be finite")
        if not 0.0 <= self.recovery_threshold < self.degradation_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= recovery < degradation <= 1"
            )
        return self


DraftUpdateScheduleConfig: TypeAlias = Annotated[
    AlwaysDraftUpdateScheduleConfig
    | FixedDraftUpdateScheduleConfig
    | AdaptiveDraftUpdateScheduleConfig,
    Field(discriminator="mode"),
]
```

Add this field to `DFlashDraftConfig` and `DSparkDraftConfig`, and do not add it to `Eagle3DraftConfig`:

```python
update_schedule: DraftUpdateScheduleConfig = Field(
    default_factory=AlwaysDraftUpdateScheduleConfig
)
```

- [ ] **Step 4: Run the GREEN configuration and static checks.**

Run: `uv run --group test pytest -q tests/unit/models/policy/test_draft_config.py && uv run ruff check nemo_rl/models/policy/draft_config.py tests/unit/models/policy/test_draft_config.py && uv run ruff format --check nemo_rl/models/policy/draft_config.py tests/unit/models/policy/test_draft_config.py`

Expected: all tests PASS; Ruff reports `All checks passed!` and no formatting diff.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/models/policy/draft_config.py tests/unit/models/policy/test_draft_config.py
git commit -S -s -m "feat(draft): define update schedule configuration"
git verify-commit HEAD
```

Expected: commit succeeds and `git verify-commit HEAD` exits 0.

### Task 2: Implement the deterministic scheduler and bounded state

**Files:**
- Create: `nemo_rl/algorithms/draft_update_schedule.py`
- Create: `tests/unit/algorithms/test_draft_update_schedule.py`

**Interfaces:**
- Consumes: `DraftUpdateScheduleConfig`, `global_step: int`, `acceptance: float | None`, and restored `Mapping[str, object] | None`.
- Produces: `DraftUpdateScheduler.create(config: DraftUpdateScheduleConfig, *, origin_step: int, restored: Mapping[str, object] | None = None) -> DraftUpdateScheduler`, `decide(*, global_step: int, acceptance: float | None) -> DraftUpdateDecision`, `record_outcome(decision: DraftUpdateDecision, *, update_attempted: bool, update_successful: bool, draft_refit_attempted: bool, draft_refit_successful: bool) -> None`, `state_dict() -> dict[str, object]`, `metrics(decision: DraftUpdateDecision) -> dict[str, float]`, and `DraftDecisionLedger.append_closed(...)`/`seal_prefix()` for the durable full decision sequence. `state_dict` and `metrics` are methods on `DraftUpdateScheduler`, not module-level functions.

- [ ] **Step 1: Write RED decision, forced-counter, history, and resume tests.**

```python
from dataclasses import replace

import pytest

from nemo_rl.algorithms import (
    draft_cadence_runtime as runtime_module,
    grpo_sync,
    single_controller,
)
from nemo_rl.algorithms.draft_update_schedule import (
    DraftDecisionLedger,
    DraftUpdateScheduler,
    FileDraftStepTransactionStore,
)
from nemo_rl.models.policy.draft_config import (
    AdaptiveDraftUpdateScheduleConfig,
    AlwaysDraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)
def _finish(scheduler: DraftUpdateScheduler, step: int, acceptance: float | None) -> tuple[bool, bool]:
    decision = scheduler.decide(global_step=step, acceptance=acceptance)
    scheduler.record_outcome(
        decision,
        update_attempted=decision.update_requested,
        update_successful=decision.update_requested,
        draft_refit_attempted=decision.draft_refit_requested,
        draft_refit_successful=decision.draft_refit_requested,
    )
    return decision.update_requested, decision.draft_refit_requested


def test_fixed_sparse_starts_after_interval_and_restores_exactly() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=7)
    assert _finish(scheduler, 8, None) == (False, False)
    assert _finish(scheduler, 9, None) == (True, True)
    restored = DraftUpdateScheduler.create(
        config, origin_step=7, restored=scheduler.state_dict()
    )
    assert restored.decide(global_step=10, acceptance=None) == scheduler.decide(
        global_step=10, acceptance=None
    )


@pytest.mark.parametrize("interval", [1, 10, 40, 100])
def test_fixed_sparse_fires_at_exact_interval(interval: int) -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=interval
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=11)
    for step in range(12, 11 + interval):
        assert _finish(scheduler, step, None) == (False, False)
    assert _finish(scheduler, 11 + interval, None) == (True, True)


def test_fixed_refit_only_updates_each_step_and_refits_periodically() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="refit_only", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    assert _finish(scheduler, 1, None) == (True, False)
    assert _finish(scheduler, 2, None) == (True, True)


def test_every_online_step_metric_contains_post_outcome_applied_version() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="refit_only", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    first = scheduler.decide(global_step=1, acceptance=None)
    scheduler.record_outcome(
        first,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=False,
        draft_refit_successful=False,
    )
    assert scheduler.metrics(first)["draft_schedule/applied_draft_version"] == 0.0
    second = scheduler.decide(global_step=2, acceptance=None)
    scheduler.record_outcome(
        second,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    assert scheduler.metrics(second)["draft_schedule/applied_draft_version"] == 0.0
    third = scheduler.decide(global_step=3, acceptance=None)
    assert scheduler.metrics(third)["draft_schedule/applied_draft_version"] == 2.0


def test_schedule_metrics_are_prefixed_once_by_the_train_logger() -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    metrics = scheduler.metrics(decision)
    assert all(key.startswith("draft_schedule/") for key in metrics)
    assert not any(key.startswith("train/") for key in metrics)
    logged = {f"train/{key}": value for key, value in metrics.items()}
    assert "train/draft_schedule/applied_draft_version" in logged
    assert not any(key.startswith("train/train/") for key in logged)


def test_duplicate_decide_and_stale_outcome_fail() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    decision = scheduler.decide(global_step=1, acceptance=None)
    with pytest.raises(RuntimeError, match="outstanding"):
        scheduler.decide(global_step=1, acceptance=None)
    stale = replace(decision, decision_id=decision.decision_id + 1)
    with pytest.raises(RuntimeError, match="stale or mismatched"):
        scheduler.record_outcome(
            stale,
            update_attempted=False,
            update_successful=False,
            draft_refit_attempted=False,
            draft_refit_successful=False,
        )


def test_steps_must_be_exactly_monotonic() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=7)
    with pytest.raises(ValueError, match="expected global_step=8"):
        scheduler.decide(global_step=9, acceptance=None)


def test_failed_requested_update_counts_attempt_but_not_success() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=1
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    decision = scheduler.decide(global_step=1, acceptance=None)
    with pytest.raises(RuntimeError, match="requested draft update failed"):
        scheduler.record_outcome(
            decision,
            update_attempted=True,
            update_successful=False,
            draft_refit_attempted=False,
            draft_refit_successful=False,
        )
    state = scheduler.state_dict()["state"]
    assert state["attempted_updates"] == 1
    assert state["successful_updates"] == 0
    assert state["failed_updates"] == 1
    assert state["attempted_refits"] == 0
    assert state["successful_refits"] == 0
    assert state["failed_refits"] == 0
    assert state["skipped_refits"] == 1
    assert state["last_update_step"] is None
    assert state["last_decided_step"] == 1


def test_restore_rejects_nonmonotonic_state_and_counters() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    _finish(scheduler, 1, None)
    saved = scheduler.state_dict()
    saved["state"]["successful_updates"] = 2
    with pytest.raises(ValueError, match="successful_updates.*attempted_updates"):
        DraftUpdateScheduler.create(config, origin_step=0, restored=saved)


def test_adaptive_forces_once_then_waits_for_evidence() -> None:
    config = AdaptiveDraftUpdateScheduleConfig(
        min_interval=2,
        max_interval=3,
        min_observations=1,
        ewma_alpha=1.0,
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    assert _finish(scheduler, 1, 0.8) == (False, False)
    assert _finish(scheduler, 2, None) == (False, False)
    assert _finish(scheduler, 3, None) == (True, True)
    assert _finish(scheduler, 4, None) == (False, False)
    state = scheduler.state_dict()["state"]
    assert state["forced_updates"] == 1
    assert state["forced_refits"] == 1


def test_history_is_bounded_and_cap_is_evaluated_after_last_refit() -> None:
    config = AdaptiveDraftUpdateScheduleConfig(
        min_interval=1,
        max_interval=100,
        min_observations=1,
        max_burst_updates=2,
        ewma_alpha=1.0,
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    _finish(scheduler, 1, 0.9)
    _finish(scheduler, 2, 0.5)
    _finish(scheduler, 3, 0.5)
    with pytest.raises(RuntimeError, match="max_burst_updates=2"):
        scheduler.decide(global_step=4, acceptance=0.5)
    assert len(scheduler.state_dict()["state"]["decision_history"]) <= 64
```

- [ ] **Step 2: Run the RED scheduler tests and confirm the module is absent.**

Run: `uv run --group test pytest -q tests/unit/algorithms/test_draft_update_schedule.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'nemo_rl.algorithms.draft_update_schedule'`.

- [ ] **Step 3: Add the concrete state and public scheduler implementation.**

```python
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Mapping, NamedTuple, Protocol

from nemo_rl.models.policy.draft_config import DraftUpdateScheduleConfig

DraftUpdatePhase = Literal[
    "monitoring", "training_burst", "awaiting_post_refit_observation"
]
DraftUpdateReason = Literal[
    "always", "fixed_interval", "adaptive_degradation", "adaptive_burst",
    "max_interval", "none"
]


class DecisionHistoryEntry(NamedTuple):
    global_step: int
    decision_id: int
    update_requested: bool
    draft_refit_requested: bool
    reason: str
    forced: bool


@dataclass(frozen=True, slots=True)
class DraftUpdateDecision:
    global_step: int
    decision_id: int
    update_requested: bool
    draft_refit_requested: bool
    reason: DraftUpdateReason
    observed_acceptance: float | None
    forced: bool = False
    applied_draft_version: int = 0


@dataclass(slots=True)
class DraftUpdateScheduleState:
    version: int
    schedule_origin_step: int
    last_update_step: int | None
    last_applied_refit_step: int | None
    applied_draft_version: int
    acceptance_ewma: float | None
    reference_acceptance_ewma: float | None
    valid_observations: int
    phase: DraftUpdatePhase
    burst_updates: int
    next_decision_id: int
    last_decided_step: int
    attempted_updates: int
    successful_updates: int
    failed_updates: int
    skipped_updates: int
    attempted_refits: int
    successful_refits: int
    failed_refits: int
    skipped_refits: int
    forced_updates: int
    forced_refits: int
    decision_history: tuple[DecisionHistoryEntry, ...]


def _finite_acceptance(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and 0.0 <= numeric <= 1.0 else None


class DraftUpdateScheduler:
    def __init__(
        self,
        config: DraftUpdateScheduleConfig,
        state: DraftUpdateScheduleState,
    ) -> None:
        self.config = config
        self.state = state
        self._pending: DraftUpdateDecision | None = None

    @classmethod
    def create(
        cls,
        config: DraftUpdateScheduleConfig,
        *,
        origin_step: int,
        restored: Mapping[str, object] | None = None,
    ) -> DraftUpdateScheduler:
        if type(origin_step) is not int or origin_step < 0:
            raise ValueError("origin_step must be a nonnegative integer")
        if restored is None:
            state = DraftUpdateScheduleState(
                version=1,
                schedule_origin_step=origin_step,
                last_update_step=None,
                last_applied_refit_step=None,
                applied_draft_version=0,
                acceptance_ewma=None,
                reference_acceptance_ewma=None,
                valid_observations=0,
                phase="monitoring",
                burst_updates=0,
                next_decision_id=1,
                last_decided_step=origin_step,
                attempted_updates=0,
                successful_updates=0,
                failed_updates=0,
                skipped_updates=0,
                attempted_refits=0,
                successful_refits=0,
                failed_refits=0,
                skipped_refits=0,
                forced_updates=0,
                forced_refits=0,
                decision_history=(),
            )
            return cls(config, state)
        if type(restored.get("state_version")) is not int or restored["state_version"] != 1:
            raise ValueError("unsupported draft update schedule state version")
        if restored.get("config") != config.model_dump(mode="json"):
            raise ValueError("resolved draft update schedule does not match checkpoint")
        restored_state = restored.get("state")
        if not isinstance(restored_state, Mapping):
            raise ValueError("draft update schedule state must be a mapping")
        validate_scheduler_state_invariants(config, restored_state)
        raw_state = dict(restored_state)
        raw_history = raw_state.pop("decision_history", None)
        if not isinstance(raw_history, list):
            raise ValueError("draft update decision history must be a list")
        history_entries: list[DecisionHistoryEntry] = []
        for entry in raw_history:
            if not isinstance(entry, Mapping):
                raise ValueError("draft update decision history entry must be a mapping")
            history_entries.append(
                DecisionHistoryEntry(
                    global_step=entry["global_step"],
                    decision_id=entry["decision_id"],
                    update_requested=entry["update_requested"],
                    draft_refit_requested=entry["draft_refit_requested"],
                    reason=str(entry["reason"]),
                    forced=entry["forced"],
                )
            )
        state = DraftUpdateScheduleState(
            **raw_state,
            decision_history=tuple(history_entries),
        )
        if state.schedule_origin_step != origin_step:
            raise ValueError("restored schedule origin does not match checkpoint step")
        return cls(config, state)

    def _consume_observation(self, acceptance: float | None) -> float | None:
        observation = _finite_acceptance(acceptance)
        if observation is None or self.config.mode != "adaptive":
            return observation
        previous = self.state.acceptance_ewma
        alpha = self.config.ewma_alpha
        self.state.acceptance_ewma = (
            observation if previous is None else alpha * observation + (1.0 - alpha) * previous
        )
        self.state.valid_observations += 1
        if (
            self.state.reference_acceptance_ewma is None
            and self.state.valid_observations >= self.config.min_observations
        ):
            self.state.reference_acceptance_ewma = self.state.acceptance_ewma
        elif (
            self.state.phase == "monitoring"
            and self.state.reference_acceptance_ewma is not None
            and self.state.acceptance_ewma > self.state.reference_acceptance_ewma
        ):
            self.state.reference_acceptance_ewma = self.state.acceptance_ewma
        return observation

    def decide(
        self,
        *,
        global_step: int,
        acceptance: float | None,
    ) -> DraftUpdateDecision:
        if self._pending is not None:
            raise RuntimeError("record the outstanding draft update decision first")
        expected_step = self.state.last_decided_step + 1
        if global_step != expected_step:
            raise ValueError(
                f"expected global_step={expected_step}, got global_step={global_step}"
            )
        observation = self._consume_observation(acceptance)
        update = False
        refit = False
        forced = False
        reason: DraftUpdateReason = "none"
        update_age = global_step - (
            self.state.last_update_step
            if self.state.last_update_step is not None
            else self.state.schedule_origin_step
        )
        refit_age = global_step - (
            self.state.last_applied_refit_step
            if self.state.last_applied_refit_step is not None
            else self.state.schedule_origin_step
        )
        if self.config.mode == "always":
            update, refit, reason = True, True, "always"
        elif self.config.mode == "fixed":
            if self.config.action == "sparse_update":
                update = update_age >= self.config.fixed_interval
                refit = update
            else:
                update = True
                refit = refit_age >= self.config.fixed_interval
            reason = "fixed_interval" if update or refit else "none"
        elif self.state.phase == "awaiting_post_refit_observation":
            if observation is not None:
                assert self.state.acceptance_ewma is not None
                assert self.state.reference_acceptance_ewma is not None
                gap = self.state.reference_acceptance_ewma - self.state.acceptance_ewma
                if gap <= self.config.recovery_threshold:
                    self.state.phase = "monitoring"
                    self.state.burst_updates = 0
                elif self.state.burst_updates >= self.config.max_burst_updates:
                    raise RuntimeError(
                        f"max_burst_updates={self.config.max_burst_updates} exhausted; "
                        f"reference={self.state.reference_acceptance_ewma}; "
                        f"current={self.state.acceptance_ewma}; "
                        f"history={self.state.decision_history}"
                    )
                else:
                    update, refit, reason = True, True, "adaptive_burst"
                    self.state.phase = "training_burst"
        elif update_age >= self.config.max_interval:
            update, refit, forced, reason = True, True, True, "max_interval"
        elif (
            update_age >= self.config.min_interval
            and self.state.reference_acceptance_ewma is not None
            and self.state.acceptance_ewma is not None
            and self.state.reference_acceptance_ewma - self.state.acceptance_ewma
            >= self.config.degradation_threshold
        ):
            update, refit, reason = True, True, "adaptive_degradation"
            self.state.phase = "training_burst"
        decision = DraftUpdateDecision(
            global_step=global_step,
            decision_id=self.state.next_decision_id,
            update_requested=update,
            draft_refit_requested=refit,
            reason=reason,
            observed_acceptance=observation,
            forced=forced,
            applied_draft_version=self.state.applied_draft_version,
        )
        self.state.next_decision_id += 1
        self.state.last_decided_step = global_step
        self._pending = decision
        return decision

    def record_outcome(
        self,
        decision: DraftUpdateDecision,
        *,
        update_attempted: bool,
        update_successful: bool,
        draft_refit_attempted: bool,
        draft_refit_successful: bool,
    ) -> None:
        if self._pending != decision:
            raise RuntimeError("stale or mismatched draft update decision outcome")
        if update_attempted != decision.update_requested:
            raise RuntimeError("draft update attempt does not match decision")
        if draft_refit_attempted and not decision.draft_refit_requested:
            raise RuntimeError("out-of-band draft refit attempt")
        if update_successful and not update_attempted:
            raise RuntimeError("draft update cannot succeed without an attempt")
        if draft_refit_successful and not draft_refit_attempted:
            raise RuntimeError("draft refit cannot succeed without an attempt")
        if update_attempted:
            self.state.attempted_updates += 1
            if update_successful:
                self.state.successful_updates += 1
                self.state.last_update_step = decision.global_step
            else:
                self.state.failed_updates += 1
        else:
            self.state.skipped_updates += 1
        if draft_refit_attempted:
            self.state.attempted_refits += 1
            if draft_refit_successful:
                self.state.successful_refits += 1
                self.state.last_applied_refit_step = decision.global_step
                self.state.applied_draft_version = decision.decision_id
            else:
                self.state.failed_refits += 1
        else:
            self.state.skipped_refits += 1
        if decision.forced:
            self.state.forced_updates += int(
                decision.update_requested and update_successful
            )
            self.state.forced_refits += int(
                decision.draft_refit_requested and draft_refit_successful
            )
        if (
            self.config.mode == "adaptive"
            and decision.draft_refit_requested
            and draft_refit_successful
        ):
            self.state.burst_updates += 1
            self.state.phase = "awaiting_post_refit_observation"
        history = deque(self.state.decision_history, maxlen=64)
        history.append(
            DecisionHistoryEntry(
                decision.global_step,
                decision.decision_id,
                decision.update_requested,
                decision.draft_refit_requested,
                decision.reason,
                decision.forced,
            )
        )
        self.state.decision_history = tuple(history)
        self._pending = None
        if decision.update_requested and not update_successful:
            raise RuntimeError("requested draft update failed")
        if decision.draft_refit_requested and not draft_refit_successful:
            raise RuntimeError("requested draft refit failed")
    def state_dict(self) -> dict[str, object]:
        state = asdict(self.state)
        state["decision_history"] = [
            entry._asdict() for entry in self.state.decision_history
        ]
        return {
            "state_version": 1,
            "config": self.config.model_dump(mode="json"),
            "state": state,
        }

    def metrics(self, decision: DraftUpdateDecision) -> dict[str, float]:
        update_origin = (
            self.state.last_update_step
            if self.state.last_update_step is not None
            else self.state.schedule_origin_step
        )
        refit_origin = (
            self.state.last_applied_refit_step
            if self.state.last_applied_refit_step is not None
            else self.state.schedule_origin_step
        )
        return {
            "draft_schedule/applied_draft_version": float(
                decision.applied_draft_version
            ),
            "draft_schedule/update_requested": float(
                decision.update_requested
            ),
            "draft_schedule/refit_requested": float(
                decision.draft_refit_requested
            ),
            "draft_schedule/steps_since_update": float(
                decision.global_step - update_origin
            ),
            "draft_schedule/steps_since_refit": float(
                decision.global_step - refit_origin
            ),
            "draft_schedule/acceptance_ewma": (
                float("nan")
                if self.state.acceptance_ewma is None
                else self.state.acceptance_ewma
            ),
            "draft_schedule/reference_acceptance_ewma": (
                float("nan")
                if self.state.reference_acceptance_ewma is None
                else self.state.reference_acceptance_ewma
            ),
        }
```

Add an append-only ledger alongside the bounded checkpoint history. The controller calls `append_closed` immediately after `record_outcome` closes successfully and before a checkpoint may be acknowledged. A failed requested update/refit is appended with its explicit failed outcome before the controller aborts. The ledger never mutates an existing byte: every resume binds zero or more sealed prefix segments and writes a new suffix segment.

```python
import hashlib
import json
import sys
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class DecisionLedgerReceipt:
    path: str
    size_bytes: int
    sha256: str
    first_decision_id: int | None
    last_decision_id: int
    entry_count: int


def decision_outcome_payload(
    decision: DraftUpdateDecision,
    *,
    update_attempted: bool,
    update_successful: bool,
    draft_refit_attempted: bool,
    draft_refit_successful: bool,
) -> dict[str, bool]:
    return {
        "update_attempted": update_attempted,
        "update_successful": update_successful,
        "update_skipped": not update_attempted,
        "draft_refit_attempted": draft_refit_attempted,
        "draft_refit_successful": draft_refit_successful,
        "draft_refit_skipped": not draft_refit_attempted,
        "forced_update": decision.forced and update_successful,
        "forced_refit": decision.forced and draft_refit_successful,
    }


def replace_bytes_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.recovery.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class DraftDecisionLedger:
    def __init__(
        self,
        path: Path,
        *,
        sealed_prefixes: tuple[DecisionLedgerReceipt, ...] = (),
    ) -> None:
        self.path = path.resolve()
        self.sealed_prefixes = sealed_prefixes
        expected_next = 1
        for prefix in sealed_prefixes:
            validate_decision_ledger_receipt(prefix)
            if prefix.first_decision_id != expected_next:
                raise ValueError("decision-ledger prefix is not contiguous")
            expected_next = prefix.last_decision_id + 1
        self.next_decision_id = expected_next
        self.sealed_prefix_high_water = expected_next - 1
        self._entry_count = 0
        self._first_decision_id: int | None = None
        self._sealed_receipt: DecisionLedgerReceipt | None = None
        if self.path.exists():
            raise FileExistsError("decision-ledger suffix path already exists")

    def append_closed(
        self,
        decision: DraftUpdateDecision,
        outcome: Mapping[str, bool],
    ) -> None:
        if self._sealed_receipt is not None:
            raise RuntimeError("cannot append to a sealed decision-ledger segment")
        if decision.decision_id != self.next_decision_id:
            raise ValueError("decision-ledger append is not contiguous")
        required_outcome = {
            "update_attempted", "update_successful", "update_skipped",
            "draft_refit_attempted", "draft_refit_successful",
            "draft_refit_skipped", "forced_update", "forced_refit",
        }
        if set(outcome) != required_outcome or any(
            type(outcome[key]) is not bool for key in required_outcome
        ):
            raise ValueError("decision-ledger outcome schema mismatch")
        if (
            outcome["update_attempted"] != decision.update_requested
            or outcome["update_skipped"] == outcome["update_attempted"]
            or outcome["update_successful"] and not outcome["update_attempted"]
            or outcome["draft_refit_attempted"]
            and not decision.draft_refit_requested
            or outcome["draft_refit_skipped"]
            == outcome["draft_refit_attempted"]
            or outcome["draft_refit_successful"]
            and not outcome["draft_refit_attempted"]
            or outcome["forced_update"]
            != (decision.forced and outcome["update_successful"])
            or outcome["forced_refit"]
            != (decision.forced and outcome["draft_refit_successful"])
        ):
            raise ValueError("decision-ledger outcome disagrees with decision")
        payload = {
            **asdict(decision),
            "outcome": dict(outcome),
        }
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._first_decision_id = (
            decision.decision_id
            if self._first_decision_id is None
            else self._first_decision_id
        )
        self._entry_count += 1
        self.next_decision_id += 1

    def append_closed_once(
        self,
        decision: DraftUpdateDecision,
        outcome: Mapping[str, bool],
    ) -> None:
        if decision.decision_id == self.next_decision_id:
            self.append_closed(decision, outcome)
            return
        if decision.decision_id > self.next_decision_id:
            raise ValueError("decision-ledger idempotent append has a gap")
        expected = {**asdict(decision), "outcome": dict(outcome)}
        matches = [
            json.loads(line)
            for line in self.path.read_text().splitlines()
            if json.loads(line)["decision_id"] == decision.decision_id
        ]
        if matches != [expected]:
            raise ValueError("decision-ledger replay differs from closed entry")

    def truncate_to(self, ledger_high_water: int) -> None:
        if self._sealed_receipt is not None:
            raise RuntimeError("truncate only the unsealed post-checkpoint suffix")
        entries = [
            json.loads(line) for line in self.path.read_text().splitlines()
        ] if self.path.exists() else []
        retained = [
            entry for entry in entries
            if int(entry["decision_id"]) <= ledger_high_water
        ]
        if [int(entry["decision_id"]) for entry in retained] != list(
            range(self.sealed_prefix_high_water + 1, ledger_high_water + 1)
        ):
            raise ValueError("checkpoint-bound ledger prefix is absent or gapped")
        encoded = b"".join(
            (json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for entry in retained
        )
        replace_bytes_fsync(self.path, encoded)
        self.next_decision_id = ledger_high_water + 1
        self._entry_count = len(retained)

    def seal_prefix(self) -> DecisionLedgerReceipt:
        if self._sealed_receipt is not None:
            return self._sealed_receipt
        if self._entry_count == 0 or self._first_decision_id is None:
            raise RuntimeError("cannot seal an empty decision-ledger segment")
        raw = self.path.read_bytes()
        receipt = DecisionLedgerReceipt(
            path=str(self.path),
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            first_decision_id=self._first_decision_id,
            last_decision_id=self.next_decision_id - 1,
            entry_count=self._entry_count,
        )
        validate_decision_ledger_receipt(receipt)
        self._sealed_receipt = receipt
        return receipt

    def open_suffix(self, path: Path) -> "DraftDecisionLedger":
        receipt = self.seal_prefix()
        return DraftDecisionLedger(
            path,
            sealed_prefixes=(*self.sealed_prefixes, receipt),
        )
```

`validate_decision_ledger_receipt` rereads the exact path, verifies byte size/SHA256, parses every JSONL row, rejects duplicate/nonintegral IDs, and requires `first_decision_id..last_decision_id` with `entry_count` exact. Every controller computes one `decision_outcome_payload`, passes its four attempted/success booleans to `record_outcome`, and appends that same payload even on the terminal failure path before raising. Sealing makes that path permanently append-ineligible; immediately after every periodic checkpoint the controller installs the new exclusive suffix returned by `checkpoint_closed`, and a resumed process likewise binds the copied checkpoint prefix before opening its new suffix. Tests cover a 1000-decision run whose scheduler history contains only IDs 937..1000, Step-100-to-Step-101 continuation, a Step-400 sealed prefix plus a new 401..1000 suffix, exact ledger/counter/forced reconciliation, append-after-seal rejection, a corrupted prefix, and a duplicate/gapped suffix. The merged ledger—not `decision_history`—is the only source for full-run reason/update/refit equality analysis.

- [ ] **Step 4: Run the GREEN scheduler matrix and static checks.**

Run: `uv run --group test pytest -q tests/unit/algorithms/test_draft_update_schedule.py && uv run ruff check nemo_rl/algorithms/draft_update_schedule.py tests/unit/algorithms/test_draft_update_schedule.py && uv run pyrefly check nemo_rl/algorithms/draft_update_schedule.py`

Expected: tests PASS, Ruff reports `All checks passed!`, and Pyrefly reports no errors.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/algorithms/draft_update_schedule.py tests/unit/algorithms/test_draft_update_schedule.py
git commit -S -s -m "feat(draft): add deterministic update scheduler"
git verify-commit HEAD
```

Expected: commit signature verifies and the commit contains only the two scheduler files.

### Task 3: Persist and validate scheduler state

**Files:**
- Modify: `nemo_rl/algorithms/draft_update_schedule.py`
- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `nemo_rl/algorithms/single_controller.py`
- Create: `nemo_rl/algorithms/draft_cadence_runtime.py`
- Create: `tests/unit/algorithms/test_draft_schedule_checkpoint.py`
- Create: `tests/unit/algorithms/test_draft_cadence_runtime.py`
- Modify: `tests/unit/single_controller/test_sc_checkpointing.py`

**Interfaces:**
- Consumes: `DraftUpdateScheduler.state_dict() -> dict[str, object]`, resolved `DraftUpdateScheduleConfig.model_dump(mode="json")`, `cadence_runtime.{enabled,result_dir,required_checkpoint_steps}`, and the controller's explicit `resuming_from_checkpoint: bool` lifecycle fact.
- Produces: `GRPOSaveState.draft_update_schedule: dict[str, object] | None`, `GRPOSaveState.applied_draft_snapshot: dict[str, object] | None`, `GRPOSaveState.draft_decision_ledger_prefixes: list[dict[str, object]]`, `FileDraftStepTransactionStore` with intent/resolution/atomic-bundle/commit recovery, centralized `validate_scheduler_state_invariants(config, state) -> None`, `scheduler_decision_high_water(schedule) -> int`, `load_checkpoint_bundle(checkpoint_path: Path) -> Mapping[str, object]`, `open_resume_decision_ledger(checkpoint_path: Path, result_root: Path) -> ResumeLedgerOpenResult`, `reconcile_ledger_quarantine(...)`, `recover_draft_step_transactions(..., config: DraftUpdateScheduleConfig | None) -> DraftUpdateScheduler | None`, checkpointed `CadenceTerminalEvidence`, `record_terminal_post_refit_observation(...) -> CadenceTerminalEvidence`, `CadenceRuntimeWriter.successful_update_closed(...) -> CadenceTerminalEvidence`, `build_terminal_schedule_payload(...) -> dict[str, object]`, `CadenceRuntimeWriter.checkpoint_closed(...) -> DraftDecisionLedger`, controller `close_successful_training(*, runtime_writer, current_step, final_checkpoint_path, terminal_evidence) -> Path`, `restore_draft_update_scheduler(config: DraftUpdateScheduleConfig, saved: Mapping[str, object] | None, *, origin_step: int, resuming_from_checkpoint: bool) -> DraftUpdateScheduler`, `restore_serving_draft_after_startup_sync(...) -> AppliedDraftSnapshot`, and concrete immutable `checkpoint-runtime.json`, `schedule-runtime.json`, exact `cadence-checkpoint-receipt.json` per checkpoint, applied-draft snapshots, and decision-ledger segments under the configured result directory. A fresh run and a legacy resume are never inferred from the same `saved is None` value.

- [ ] **Step 1: Write RED legacy, mismatch, and uninterrupted/resumed tests.**

```python
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.draft_update_schedule import (
    AppliedDraftSnapshot,
    DraftUpdateScheduler,
    FileDraftStepTransactionStore,
    validate_scheduler_state_invariants,
)
from nemo_rl.algorithms.draft_cadence_runtime import (
    CadenceRuntimeConfig,
    CadenceRuntimeWriter,
    CadenceTerminalEvidence,
    build_terminal_schedule_payload,
    load_checkpoint_bundle,
    scheduler_decision_high_water,
)
from nemo_rl.algorithms.grpo import (
    restore_serving_draft_after_startup_sync,
    restore_draft_update_scheduler,
)
from nemo_rl.models.policy.draft_config import (
    AdaptiveDraftUpdateScheduleConfig,
    AlwaysDraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)


def test_fresh_fixed_run_without_saved_state_is_allowed() -> None:
    fixed = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=10
    )
    scheduler = restore_draft_update_scheduler(
        fixed,
        None,
        origin_step=0,
        resuming_from_checkpoint=False,
    )
    assert scheduler.state.schedule_origin_step == 0


def test_legacy_checkpoint_is_allowed_only_for_always() -> None:
    always = AlwaysDraftUpdateScheduleConfig()
    assert restore_draft_update_scheduler(
        always,
        None,
        origin_step=4,
        resuming_from_checkpoint=True,
    ).state.schedule_origin_step == 4
    fixed = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=10
    )
    with pytest.raises(ValueError, match="legacy checkpoint.*always"):
        restore_draft_update_scheduler(
            fixed,
            None,
            origin_step=4,
            resuming_from_checkpoint=True,
        )


def test_restore_rejects_resolved_config_mismatch() -> None:
    original = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=10
    )
    changed = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=40
    )
    saved = DraftUpdateScheduler.create(original, origin_step=0).state_dict()
    with pytest.raises(ValueError, match="resolved draft update schedule"):
        restore_draft_update_scheduler(
            changed,
            saved,
            origin_step=0,
            resuming_from_checkpoint=True,
        )


def test_create_restored_rejects_outer_version_and_config_mismatch() -> None:
    config, saved = valid_saved_state()
    bad_version = copy.deepcopy(saved)
    bad_version["state_version"] = 2
    with pytest.raises(ValueError, match="outer|state version"):
        DraftUpdateScheduler.create(config, origin_step=0, restored=bad_version)
    bad_config = copy.deepcopy(saved)
    bad_config["config"] = {"mode": "always", "unexpected": True}
    with pytest.raises(ValueError, match="resolved draft update schedule"):
        DraftUpdateScheduler.create(config, origin_step=0, restored=bad_config)


def test_resume_restores_exact_applied_draft_snapshot_before_publication(
    tmp_path,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="refit_only", fixed_interval=10
        ),
        origin_step=0,
    )
    for step in range(1, 18):
        decision = scheduler.decide(global_step=step, acceptance=None)
        scheduler.record_outcome(
            decision,
            update_attempted=True,
            update_successful=True,
            draft_refit_attempted=decision.draft_refit_requested,
            draft_refit_successful=decision.draft_refit_requested,
        )
    applied_bytes = b"draft-bytes-applied-at-step-10"
    snapshot_path = tmp_path / "applied-draft-v10.safetensors"
    snapshot_path.write_bytes(applied_bytes)
    snapshot = AppliedDraftSnapshot(
        version=10,
        path=str(snapshot_path),
        size_bytes=len(applied_bytes),
        sha256=hashlib.sha256(applied_bytes).hexdigest(),
    )
    rollout_manager = MagicMock()
    synchronizer = MagicMock()
    synchronizer.sync_applied_draft_snapshot.return_value = {
        "successful": True,
        "version": 10,
        "sha256": snapshot.sha256,
    }
    lifecycle = []
    rollout_manager.set_applied_draft_version.side_effect = (
        lambda _version: lifecycle.append("published")
    )
    rollout_manager.enable_reservations.side_effect = (
        lambda: lifecycle.append("reservations")
    )

    save_state = MagicMock(applied_draft_snapshot=None)

    def flush_save_state(state) -> dict[str, object]:
        lifecycle.append("durable")
        installed = AppliedDraftSnapshot(**state.applied_draft_snapshot)
        return {
            "successful": True,
            "version": installed.version,
            "sha256": installed.sha256,
        }

    def install_snapshot(installed: AppliedDraftSnapshot) -> dict[str, object]:
        assert installed == snapshot
        return durably_install_startup_snapshot(
            save_state, installed, flush_save_state=flush_save_state
        )

    restore_serving_draft_after_startup_sync(
        scheduler.config,
        scheduler,
        rollout_manager,
        synchronizer,
        snapshot=snapshot,
        snapshot_path=None,
        resuming_from_checkpoint=True,
        install_snapshot=install_snapshot,
    )
    synchronizer.sync_target_from_current_checkpoint.assert_called_once_with()
    synchronizer.sync_applied_draft_snapshot.assert_called_once_with(snapshot)
    synchronizer.sync_current_trainable_draft.assert_not_called()
    rollout_manager.set_applied_draft_version.assert_called_once_with(10)
    rollout_manager.enable_reservations.assert_called_once_with()
    assert lifecycle == ["durable", "published", "reservations"]
    assert save_state.applied_draft_snapshot == asdict(snapshot)


def test_resume_rejects_snapshot_version_or_bytes_mismatch(tmp_path) -> None:
    config, saved = valid_saved_state()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0, restored=saved)
    path = tmp_path / "draft.safetensors"
    path.write_bytes(b"wrong")
    snapshot = AppliedDraftSnapshot(
        version=0,
        path=str(path),
        size_bytes=5,
        sha256=hashlib.sha256(b"right").hexdigest(),
    )
    with pytest.raises(ValueError, match="snapshot.*version|digest"):
        restore_serving_draft_after_startup_sync(
            scheduler.config,
            scheduler,
            MagicMock(),
            MagicMock(),
            snapshot=snapshot,
            snapshot_path=None,
            resuming_from_checkpoint=True,
            install_snapshot=MagicMock(),
        )


def test_resumed_refit_only_without_applied_snapshot_fails_before_sync() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="refit_only", fixed_interval=10
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    synchronizer = MagicMock()
    with pytest.raises(ValueError, match="resumed.*snapshot"):
        restore_serving_draft_after_startup_sync(
            config,
            scheduler,
            MagicMock(),
            synchronizer,
            snapshot=None,
            snapshot_path=None,
            resuming_from_checkpoint=True,
            install_snapshot=MagicMock(),
        )
    synchronizer.sync_target_from_current_checkpoint.assert_not_called()
    synchronizer.sync_current_trainable_draft.assert_not_called()


def test_startup_apply_must_succeed_before_persistence_or_reservations(
    tmp_path,
) -> None:
    config = AlwaysDraftUpdateScheduleConfig()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    path = tmp_path / "applied-draft-v0.safetensors"
    raw = b"initial"
    path.write_bytes(raw)
    synchronizer = MagicMock()
    synchronizer.sync_current_trainable_draft.return_value = {
        "successful": False,
        "version": 0,
        "snapshot_path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    rollout_manager = MagicMock()
    installer = MagicMock()
    with pytest.raises(RuntimeError, match="initial serving-draft snapshot"):
        restore_serving_draft_after_startup_sync(
            config,
            scheduler,
            rollout_manager,
            synchronizer,
            snapshot=None,
            snapshot_path=path,
            resuming_from_checkpoint=False,
            install_snapshot=installer,
        )
    installer.assert_not_called()
    rollout_manager.set_applied_draft_version.assert_not_called()
    rollout_manager.enable_reservations.assert_not_called()


def test_pre_first_refit_resume_restores_immutable_version_zero_snapshot(
    tmp_path,
) -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="refit_only", fixed_interval=10
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    version_zero_path = tmp_path / "applied-draft-v0.safetensors"
    version_zero_bytes = b"immutable-initial-serving-draft"
    version_zero_path.write_bytes(version_zero_bytes)
    digest = hashlib.sha256(version_zero_bytes).hexdigest()
    first_sync = MagicMock()
    first_sync.sync_current_trainable_draft.return_value = {
        "successful": True,
        "version": 0,
        "snapshot_path": str(version_zero_path.resolve()),
        "sha256": digest,
    }
    snapshot = restore_serving_draft_after_startup_sync(
        config,
        scheduler,
        MagicMock(),
        first_sync,
        snapshot=None,
        snapshot_path=version_zero_path,
        resuming_from_checkpoint=False,
        install_snapshot=lambda installed: {
            "successful": True,
            "version": installed.version,
            "sha256": installed.sha256,
        },
    )
    for step in range(1, 6):
        decision = scheduler.decide(global_step=step, acceptance=None)
        scheduler.record_outcome(
            decision,
            update_attempted=True,
            update_successful=True,
            draft_refit_attempted=False,
            draft_refit_successful=False,
        )
    restored = DraftUpdateScheduler.create(
        config, origin_step=0, restored=scheduler.state_dict()
    )
    resume_sync = MagicMock()
    resume_sync.sync_applied_draft_snapshot.return_value = {
        "successful": True, "version": 0, "sha256": digest,
    }
    restore_serving_draft_after_startup_sync(
        config,
        restored,
        MagicMock(),
        resume_sync,
        snapshot=snapshot,
        snapshot_path=None,
        resuming_from_checkpoint=True,
        install_snapshot=lambda installed: {
            "successful": True,
            "version": installed.version,
            "sha256": installed.sha256,
        },
    )
    resume_sync.sync_applied_draft_snapshot.assert_called_once_with(snapshot)
    resume_sync.sync_current_trainable_draft.assert_not_called()


@pytest.mark.parametrize(
    "crash_after",
    ["intent", "resolution", "outcome", "ledger", "bundle", "commit_marker"],
)
def test_draft_step_transaction_recovers_matching_scheduler_snapshot_and_ledger(
    tmp_path, crash_after
) -> None:
    store = FileDraftStepTransactionStore(tmp_path, crash_after=crash_after)
    run = transaction_test_run(store, requested_refit=True)
    with pytest.raises(InjectedCrash):
        run.execute_step(1)
    restored = run.restart_and_recover()
    bundle = load_checkpoint_bundle(
        store.checkpoint_path(run.last_full_checkpoint_id)
    )
    assert bundle["checkpoint_id"] == run.last_full_checkpoint_id
    assert set(bundle["components"]) == {"model", "optimizer", "dataloader_rng"}
    assert bundle["draft_update_schedule"]["state"]["applied_draft_version"] == 0
    assert bundle["applied_draft_snapshot"]["version"] == 0
    assert bundle["ledger_high_water"] == 0
    assert restored.ledger.entries_after(0) == []
    assert restored.scheduler.state.next_decision_id == 1
    assert store.pending_after_checkpoint(run.last_full_checkpoint_id) == []


@pytest.mark.parametrize("durable_apply_receipt", [None, "valid"])
def test_crash_after_intent_resolves_from_durable_transfer_receipt_then_truncates(
    tmp_path, durable_apply_receipt
) -> None:
    store = FileDraftStepTransactionStore(tmp_path)
    run = transaction_test_run(store, requested_refit=True)
    transaction = run.open_step_intent(1)
    if durable_apply_receipt == "valid":
        run.persist_durable_apply_receipt(
            transaction,
            successful=True,
            snapshot_version=1,
        )
    run.drop_process_without_resolution()
    restored = run.restart_and_recover()
    resolution = store.recovery_resolution_for_decision(1)
    assert resolution.outcome["draft_refit_successful"] is (
        durable_apply_receipt == "valid"
    )
    assert restored.scheduler.state.next_decision_id == 1
    assert restored.ledger.entries_after(0) == []


@pytest.mark.parametrize("draft_requested", [False, True])
def test_every_transfer_exception_closes_and_persists_exactly_one_outcome(
    tmp_path, draft_requested
) -> None:
    store = FileDraftStepTransactionStore(tmp_path)
    run = transaction_test_run(
        store,
        requested_refit=draft_requested,
        transfer_error=RuntimeError("target transfer failed"),
    )
    with pytest.raises(RuntimeError, match="target transfer failed"):
        run.execute_step(1)
    entry = run.ledger.entry(1)
    assert run.ledger.entries_for_decision(1) == 1
    assert entry["draft_refit_attempted"] is draft_requested
    assert entry["draft_refit_successful"] is False
    assert entry["draft_refit_skipped"] is (not draft_requested)
    restored = run.restart_and_recover()
    assert restored.ledger.entries_after(0) == []
    assert restored.scheduler.state.next_decision_id == 1


def test_cadence_advances_on_resume_only_after_full_training_checkpoint(tmp_path) -> None:
    store = FileDraftStepTransactionStore(tmp_path)
    run = transaction_test_run(store, requested_refit=True)
    run.execute_step(1)
    with pytest.raises(RuntimeError, match="optimizer checkpoint failed"):
        run.checkpoint(
            "step_1", model=True, optimizer=False, dataloader=True
        )
    assert run.restart_and_recover().scheduler.state.next_decision_id == 1
    run = transaction_test_run(store, requested_refit=True)
    run.execute_step(1)
    run.checkpoint("step_1", model=True, optimizer=True, dataloader=True)
    restored = run.restart_and_recover()
    assert restored.scheduler.state.next_decision_id == 2
    assert restored.scheduler.state.applied_draft_version == 1
    assert restored.ledger.entries_for_decision(1) == 1


@pytest.mark.parametrize("component", ["model", "optimizer", "dataloader_rng"])
def test_checkpoint_bundle_rehashes_every_training_component(
    tmp_path, component
) -> None:
    store = FileDraftStepTransactionStore(tmp_path)
    run = transaction_test_run(store, requested_refit=True)
    run.execute_step(1)
    run.checkpoint("step_1", model=True, optimizer=True, dataloader=True)
    checkpoint = store.checkpoint_path("step_1")
    receipt = json.loads(
        (checkpoint / "cadence-checkpoint-receipt.json").read_text()
    )
    artifact = checkpoint / receipt["components"][component]["relative_path"]
    target = (
        artifact
        if artifact.is_file()
        else next(path for path in artifact.rglob("*") if path.is_file())
    )
    target.write_bytes(target.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match=f"{component} checkpoint digest"):
        load_checkpoint_bundle(checkpoint)


def test_checkpoint_bundle_rehashes_ledger_scheduler_and_tree(tmp_path) -> None:
    store = FileDraftStepTransactionStore(tmp_path)
    run = transaction_test_run(store, requested_refit=True)
    run.execute_step(1)
    run.checkpoint("step_1", model=True, optimizer=True, dataloader=True)
    checkpoint = store.checkpoint_path("step_1")
    receipt_path = checkpoint / "cadence-checkpoint-receipt.json"
    original = receipt_path.read_bytes()
    receipt = json.loads(original)
    ledger = checkpoint / receipt["decision_ledger"]["relative_path"]
    ledger.write_bytes(ledger.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="decision-ledger receipt"):
        load_checkpoint_bundle(checkpoint)
    ledger.write_bytes(ledger.read_bytes()[:-3])
    (checkpoint / "unexpected.bin").write_bytes(b"changes-tree")
    with pytest.raises(ValueError, match="tree digest"):
        load_checkpoint_bundle(checkpoint)
    (checkpoint / "unexpected.bin").unlink()
    receipt = json.loads(original)
    receipt["draft_update_schedule"]["state"]["next_decision_id"] = 3
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="scheduler/ledger high-water"):
        load_checkpoint_bundle(checkpoint)


def test_checkpoint_high_water_is_derived_from_real_scheduler_cursor() -> None:
    config, saved = valid_saved_state()
    assert "decisions" not in saved["state"]
    assert scheduler_decision_high_water(saved) == (
        saved["state"]["next_decision_id"] - 1
    )


def test_disabled_fixed_control_checkpoint_has_explicit_empty_ledger(
    tmp_path,
) -> None:
    run = disabled_control_test_run(tmp_path)
    checkpoint = run.checkpoint("step_100")
    bundle = load_checkpoint_bundle(checkpoint)
    assert bundle["draft_update_schedule"]["mode"] == "disabled"
    assert scheduler_decision_high_water(bundle["draft_update_schedule"]) == 0
    assert bundle["decision_ledger"] == {
        "relative_path": "draft-decision-ledger.jsonl",
        "size_bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "first_decision_id": None,
        "last_decision_id": 0,
        "entry_count": 0,
    }
    assert (checkpoint / "draft-decision-ledger.jsonl").read_bytes() == b""
    restored = run.restart_from_checkpoint("step_100")
    assert restored.scheduler is None
    assert restored.ledger.next_decision_id == 1
    assert restored.transaction_store.pending_intents() == ()
    assert restored.transaction_store.resolutions_since("step_100") == ()


def test_step_100_checkpoint_installs_suffix_and_step_101_continues(tmp_path) -> None:
    run = transaction_test_run(
        FileDraftStepTransactionStore(tmp_path), requested_refit=True
    )
    for step in range(1, 101):
        run.execute_step(step)
    sealed = run.ledger
    writable = run.checkpoint(
        "step_100", model=True, optimizer=True, dataloader=True
    )
    assert run.ledger is writable
    assert run.ledger is not sealed
    assert run.ledger.next_decision_id == 101
    with pytest.raises(RuntimeError, match="sealed"):
        sealed.append_closed(run.decision_for_step(101), run.success_outcome(101))
    run.execute_step(101)
    assert run.ledger.entry(101)["decision_id"] == 101
    assert run.scheduler.state.next_decision_id == 102


def test_resume_from_step_100_opens_suffix_at_101(tmp_path) -> None:
    run = transaction_test_run(
        FileDraftStepTransactionStore(tmp_path), requested_refit=True
    )
    for step in range(1, 101):
        run.execute_step(step)
    run.checkpoint("step_100", model=True, optimizer=True, dataloader=True)
    resumed = run.restart_from_checkpoint("step_100")
    assert resumed.scheduler.state.next_decision_id == 101
    assert resumed.ledger.next_decision_id == 101
    resumed.execute_step(101)
    assert resumed.ledger.entry(101)["decision_id"] == 101
    assert resumed.scheduler.state.next_decision_id == 102


def test_resume_quarantines_written_post_checkpoint_suffix_before_replaying_101(
    tmp_path,
) -> None:
    run = transaction_test_run(
        FileDraftStepTransactionStore(tmp_path), requested_refit=True
    )
    for step in range(1, 101):
        run.execute_step(step)
    run.checkpoint("step_100", model=True, optimizer=True, dataloader=True)
    stale_suffix = run.ledger.path
    run.execute_step(101)
    stale_bytes = stale_suffix.read_bytes()
    assert json.loads(stale_bytes)["decision_id"] == 101
    run.drop_process_without_checkpoint()

    resumed = run.restart_from_checkpoint("step_100")
    assert not stale_suffix.exists()
    quarantine = json.loads(resumed.ledger_quarantine_receipt.read_text())
    assert quarantine["checkpoint_id"] == "step_100"
    assert quarantine["artifacts"] == [{
        "original_path": str(stale_suffix.resolve()),
        "quarantine_path": quarantine["artifacts"][0]["quarantine_path"],
        "size_bytes": len(stale_bytes),
        "sha256": hashlib.sha256(stale_bytes).hexdigest(),
    }]
    assert Path(quarantine["artifacts"][0]["quarantine_path"]).read_bytes() == stale_bytes
    assert resumed.ledger.path != stale_suffix
    assert "resume-step_100-" in resumed.ledger.path.name
    assert resumed.ledger.next_decision_id == 101
    resumed.execute_step(101)
    assert resumed.ledger.entry(101)["decision_id"] == 101
    assert resumed.scheduler.state.next_decision_id == 102


@pytest.mark.parametrize("crash_phase", ["after_intent", "before_receipt"])
def test_incomplete_quarantine_transaction_reconciles_after_crash(
    monkeypatch, tmp_path, crash_phase
) -> None:
    run = transaction_test_run(
        FileDraftStepTransactionStore(tmp_path), requested_refit=True
    )
    for step in range(1, 101):
        run.execute_step(step)
    run.checkpoint("step_100", model=True, optimizer=True, dataloader=True)
    run.execute_step(101)
    run.drop_process_without_checkpoint()
    real_move = runtime_module._move_ledger_to_quarantine
    real_write = runtime_module.write_json_exclusive_atomic

    if crash_phase == "after_intent":
        monkeypatch.setattr(
            runtime_module,
            "_move_ledger_to_quarantine",
            lambda *_args: (_ for _ in ()).throw(InjectedCrash("after intent")),
        )
    else:
        def crash_before_receipt(path: Path, payload: object) -> None:
            if path.name == "ledger-quarantine-receipt.json":
                raise InjectedCrash("before receipt")
            real_write(path, payload)
        monkeypatch.setattr(
            runtime_module, "write_json_exclusive_atomic", crash_before_receipt
        )
    with pytest.raises(InjectedCrash):
        run.restart_from_checkpoint("step_100")
    monkeypatch.setattr(
        runtime_module, "_move_ledger_to_quarantine", real_move
    )
    monkeypatch.setattr(
        runtime_module, "write_json_exclusive_atomic", real_write
    )
    resumed = run.restart_from_checkpoint("step_100")
    receipt = json.loads(resumed.ledger_quarantine_receipt.read_text())
    assert receipt["state"] == "resolved"
    assert receipt["checkpoint_id"] == "step_100"
    resumed.execute_step(101)
    assert resumed.ledger.entry(101)["decision_id"] == 101


def test_successful_update_receipt_is_exclusive_and_installed_before_return(
    tmp_path: Path,
) -> None:
    writer = CadenceRuntimeWriter(CadenceRuntimeConfig(
        enabled=True, result_dir=str(tmp_path)
    ))
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    evidence = CadenceTerminalEvidence(
        update_receipts_by_decision={}, observations_by_refit_step={}
    )
    save_state = SimpleNamespace(draft_terminal_evidence=None)
    worker_receipt = {
        "successful": True,
        "decision_id": 1,
        "global_step": 1,
        "draft_model_sha256": "a" * 64,
        "draft_optimizer_sha256": "b" * 64,
    }
    updated = writer.successful_update_closed(
        decision=decision,
        worker_receipt=worker_receipt,
        evidence=evidence,
        save_state=save_state,
    )
    binding = updated.update_receipts_by_decision[1]
    raw = Path(binding["path"]).read_bytes()
    assert binding["size_bytes"] == len(raw)
    assert binding["sha256"] == hashlib.sha256(raw).hexdigest()
    assert save_state.draft_terminal_evidence == updated.state_dict()
    with pytest.raises(FileExistsError):
        writer.successful_update_closed(
            decision=decision,
            worker_receipt=worker_receipt,
            evidence=CadenceTerminalEvidence({}, {}),
            save_state=save_state,
        )


def test_resume_can_replay_uncheckpointed_decision_without_receipt_collision(
    tmp_path: Path,
) -> None:
    config = CadenceRuntimeConfig(enabled=True, result_dir=str(tmp_path))
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    worker_receipt = {
        "successful": True,
        "decision_id": 1,
        "global_step": 1,
        "draft_model_sha256": "a" * 64,
        "draft_optimizer_sha256": "b" * 64,
    }
    first_evidence = CadenceTerminalEvidence({}, {})
    first = CadenceRuntimeWriter(config).successful_update_closed(
        decision=decision,
        worker_receipt=worker_receipt,
        evidence=first_evidence,
        save_state=SimpleNamespace(draft_terminal_evidence=None),
    )
    stale_path = Path(first.update_receipts_by_decision[1]["path"])

    replay_evidence = CadenceTerminalEvidence({}, {})
    replay = CadenceRuntimeWriter(config).successful_update_closed(
        decision=decision,
        worker_receipt=worker_receipt,
        evidence=replay_evidence,
        save_state=SimpleNamespace(draft_terminal_evidence=None),
    )
    replay_path = Path(replay.update_receipts_by_decision[1]["path"])
    assert stale_path.is_file() and replay_path.is_file()
    assert stale_path != replay_path


def test_terminal_payload_maps_decision_id_to_nonzero_origin_step(tmp_path) -> None:
    run = terminal_builder_test_run(
        tmp_path, controller_kind="sync", mode="always", origin_step=7
    )
    run.execute_step(8, acceptance=0.70)
    run.checkpoint("step_8", model=True, optimizer=True, dataloader=True)
    schedule = build_terminal_schedule_payload(
        load_checkpoint_bundle(run.checkpoint_path("step_8")),
        run.terminal_evidence,
    )
    assert schedule["decision_ids"] == [1]
    assert schedule["global_steps"] == [8]
    assert schedule["updated_steps"] == [8]
    assert schedule["update_receipts"][0]["decision_id"] == 1
    assert schedule["refit_versions"] == [{
        "refit_step": 8,
        "applied_draft_version": 1,
    }]


def test_resumed_terminal_payload_reports_only_post_boundary_observations(
    tmp_path,
) -> None:
    run = terminal_builder_test_run(
        tmp_path, controller_kind="sync", mode="always", origin_step=0
    )
    run.execute_step(1, acceptance=0.70)
    run.execute_step(2, acceptance=0.69)
    run.checkpoint("step_2", model=True, optimizer=True, dataloader=True)
    resumed = run.restart_from_checkpoint("step_2")
    resumed.execute_step(3, acceptance=0.68)
    resumed.checkpoint("step_3", model=True, optimizer=True, dataloader=True)
    assert set(resumed.terminal_evidence.observations_by_refit_step) == {1, 2}
    schedule = build_terminal_schedule_payload(
        load_checkpoint_bundle(resumed.checkpoint_path("step_3")),
        resumed.terminal_evidence,
    )
    assert schedule["post_event_observations"] == [{
        "refit_step": 2,
        "observation_step": 3,
        "applied_draft_version": 2,
        "acceptance_rate": 0.68,
    }]


def valid_saved_state() -> tuple[AlwaysDraftUpdateScheduleConfig, dict[str, object]]:
    config = AlwaysDraftUpdateScheduleConfig()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    decision = scheduler.decide(global_step=1, acceptance=None)
    scheduler.record_outcome(
        decision,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    return config, scheduler.state_dict()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "inner state version"),
        ("attempted_updates", -1, "nonnegative"),
        ("successful_updates", 2, "successful updates"),
        ("failed_updates", 1, "attempted updates.*partition"),
        ("skipped_updates", 1, "update partition"),
        ("forced_updates", 2, "forced updates"),
        ("last_update_step", 2, "last update step"),
        ("last_applied_refit_step", 2, "last applied refit step"),
        ("applied_draft_version", 99, "applied draft version"),
        ("next_decision_id", 3, "decision cursor"),
        ("decision_history", [], "history.*nonempty"),
        ("phase", "broken", "phase"),
        ("acceptance_ewma", float("inf"), "EWMA"),
        ("valid_observations", -1, "nonnegative"),
        ("burst_updates", -1, "nonnegative"),
    ],
)
def test_restore_rejects_each_corrupt_scheduler_invariant(
    field: str, value: object, message: str
) -> None:
    config, saved = valid_saved_state()
    corrupt = copy.deepcopy(saved)
    corrupt["state"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_scheduler_state_invariants(config, corrupt["state"])


def test_restore_rejects_invalid_history_reason_and_phase_fields() -> None:
    config, saved = valid_saved_state()
    bad_reason = copy.deepcopy(saved)
    bad_reason["state"]["decision_history"][0]["reason"] = "unknown"
    with pytest.raises(ValueError, match="history reason"):
        validate_scheduler_state_invariants(config, bad_reason["state"])
    bad_phase = copy.deepcopy(saved)
    bad_phase["state"]["phase"] = "training_burst"
    with pytest.raises(ValueError, match="non-adaptive.*monitoring"):
        validate_scheduler_state_invariants(config, bad_phase["state"])


def test_adaptive_restore_rejects_phase_inconsistent_observation_fields() -> None:
    config = AdaptiveDraftUpdateScheduleConfig()
    saved = DraftUpdateScheduler.create(config, origin_step=0).state_dict()
    saved["state"]["valid_observations"] = 1
    with pytest.raises(ValueError, match="valid observations require acceptance EWMA"):
        validate_scheduler_state_invariants(config, saved["state"])
    saved = DraftUpdateScheduler.create(config, origin_step=0).state_dict()
    saved["state"]["phase"] = "awaiting_post_refit_observation"
    with pytest.raises(ValueError, match="awaiting phase requires an applied refit"):
        validate_scheduler_state_invariants(config, saved["state"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last_decided_step", 1.0),
        ("next_decision_id", True),
        ("applied_draft_version", 1.0),
    ],
)
def test_restore_rejects_nonintegral_scheduler_steps_and_versions(
    field: str, value: object
) -> None:
    config, saved = valid_saved_state()
    saved["state"][field] = value
    with pytest.raises(ValueError, match="integer"):
        validate_scheduler_state_invariants(config, saved["state"])


def test_restore_derives_applied_version_from_last_refit_step() -> None:
    config, saved = valid_saved_state()
    saved["state"]["applied_draft_version"] = 0
    with pytest.raises(ValueError, match="last applied refit"):
        validate_scheduler_state_invariants(config, saved["state"])
```

`transaction_test_run` is a concrete test fixture in
`tests/unit/algorithms/test_draft_schedule_checkpoint.py`, not pseudocode. It
creates a real version-0 full checkpoint and `DraftDecisionLedger`, and calls
the production transaction/recovery functions. `open_step_intent` calls the
production `begin` and returns without writing a resolution;
`persist_durable_apply_receipt` uses the production exclusive receipt writer
with the exact transaction ID, decision ID, snapshot version/path/SHA, and
`successful=True`; `drop_process_without_resolution` closes all file handles
and discards the controller/scheduler objects. `restart_and_recover` constructs
fresh objects from the last full checkpoint and calls
`recover_draft_step_transactions`. `InjectedCrash` is the store's test-only
failpoint exception. Thus the crash-after-intent test does not fake an
impossible in-process continuation after the failpoint. Its `checkpoint`
method calls the production `CadenceRuntimeWriter.checkpoint_closed`, installs
the returned writable `DraftDecisionLedger`, and returns that suffix. Its
`restart_from_checkpoint` calls `open_resume_decision_ledger`, which validates
the checkpoint, quarantines stale live suffixes, constructs the
checkpoint-bound prefix plus a UUID-qualified new exclusive suffix, and only
then calls transaction recovery. `disabled_control_test_run` uses the same real checkpoint producer
with `draft.enabled=false`, the exact neutral `mode="disabled"` schedule state,
no scheduler object, no transactions, and no decision appends.
`terminal_builder_test_run` is a concrete local fixture backed by the real
`CadenceRuntimeWriter`, scheduler, ledger, checkpoint producer, digest-bound
decision-ID-keyed update receipts, and science-observation recorder. Its
`restart_from_checkpoint` restores the full checkpointed terminal evidence
before adding post-boundary observations. It always enables cadence runtime;
the paired default-path tests prove no writer or evidence is constructed when
disabled. Cross-contract controller coverage belongs to experiments Task 2,
which runs after this product plan's Task 10.

- [ ] **Step 2: Run the RED checkpoint tests and confirm the restore helper is missing.**

Run: `uv run --group test pytest -q tests/unit/algorithms/test_draft_schedule_checkpoint.py tests/unit/single_controller/test_sc_checkpointing.py -k 'draft_update_schedule'`

Expected: FAIL during collection with `ImportError: cannot import name 'restore_draft_update_scheduler'`.

- [ ] **Step 3: Add the checkpoint field and exact restore helper.**

```python
@dataclass
class GRPOSaveState:
    consumed_samples: int
    current_step: int
    current_epoch: int
    total_steps: int
    total_valid_tokens: int
    val_reward: float
    sampler_name: Optional[str] = None
    draft_update_schedule: dict[str, object] | None = None
    applied_draft_snapshot: dict[str, object] | None = None
    draft_terminal_evidence: dict[str, object] | None = None
    draft_decision_ledger_prefixes: list[dict[str, object]] = field(
        default_factory=list
    )


def validate_scheduler_state_invariants(
    config: DraftUpdateScheduleConfig,
    state: Mapping[str, object],
) -> None:
    if type(state.get("version")) is not int or state["version"] != 1:
        raise ValueError("unsupported inner state version")
    integer_fields = (
        "schedule_origin_step", "applied_draft_version", "valid_observations",
        "burst_updates", "next_decision_id", "last_decided_step",
        "attempted_updates", "successful_updates", "skipped_updates",
        "failed_updates", "attempted_refits", "successful_refits",
        "failed_refits", "skipped_refits",
        "forced_updates", "forced_refits",
    )
    if any(type(state.get(field)) is not int for field in integer_fields):
        raise ValueError("scheduler counters, steps, and versions must be integers")
    values = {field: state[field] for field in integer_fields}
    if any(value < 0 for value in values.values()):
        raise ValueError("scheduler state counters and steps must be nonnegative")
    decisions = values["last_decided_step"] - values["schedule_origin_step"]
    if decisions < 0:
        raise ValueError("last decided step precedes schedule origin")
    if values["next_decision_id"] != decisions + 1:
        raise ValueError("decision cursor must equal decided steps plus one")
    if values["successful_updates"] > values["attempted_updates"]:
        raise ValueError("successful updates exceed attempted updates")
    if values["successful_refits"] > values["attempted_refits"]:
        raise ValueError("successful refits exceed attempted refits")
    if values["attempted_updates"] != (
        values["successful_updates"] + values["failed_updates"]
    ):
        raise ValueError("attempted updates do not partition into success and failure")
    if values["attempted_refits"] != (
        values["successful_refits"] + values["failed_refits"]
    ):
        raise ValueError("attempted refits do not partition into success and failure")
    if values["attempted_updates"] + values["skipped_updates"] != decisions:
        raise ValueError("update partition does not equal decided steps")
    if values["attempted_refits"] + values["skipped_refits"] != decisions:
        raise ValueError("refit partition does not equal decided steps")
    if values["forced_updates"] > values["successful_updates"]:
        raise ValueError("forced updates exceed successful updates")
    if values["forced_refits"] > values["successful_refits"]:
        raise ValueError("forced refits exceed successful refits")
    for field, successes in (
        ("last_update_step", values["successful_updates"]),
        ("last_applied_refit_step", values["successful_refits"]),
    ):
        step = state.get(field)
        if step is not None and type(step) is not int:
            raise ValueError(f"{field.replace('_', ' ')} must be an integer")
        if successes == 0 and step is not None:
            raise ValueError(f"{field.replace('_', ' ')} exists without success")
        if successes > 0 and (
            step is None
            or not values["schedule_origin_step"] < int(step) <= values["last_decided_step"]
        ):
            raise ValueError(f"invalid {field.replace('_', ' ')}")
    last_update = state.get("last_update_step")
    last_refit = state.get("last_applied_refit_step")
    if last_refit is not None and last_update is not None and int(last_refit) > int(last_update):
        raise ValueError("last applied refit step exceeds last update step")
    expected_applied_version = (
        0
        if last_refit is None
        else last_refit - values["schedule_origin_step"]
    )
    if values["applied_draft_version"] != expected_applied_version:
        raise ValueError(
            "applied draft version must equal the last applied refit decision"
        )
    history = state.get("decision_history")
    if not isinstance(history, list) or len(history) > 64:
        raise ValueError("decision history must be a list of at most 64 entries")
    if decisions and not history:
        raise ValueError("decision history must be nonempty when steps exist")
    valid_reasons = {
        "always", "fixed_interval", "adaptive_degradation",
        "adaptive_burst", "max_interval", "none",
    }
    reasons_for_mode = {
        "always": {"always"},
        "fixed": {"fixed_interval", "none"},
        "adaptive": {
            "adaptive_degradation", "adaptive_burst", "max_interval", "none"
        },
    }[config.mode]
    previous_step = previous_id = None
    for entry in history:
        if not isinstance(entry, Mapping) or entry.get("reason") not in valid_reasons:
            raise ValueError("invalid decision history reason")
        if entry["reason"] not in reasons_for_mode:
            raise ValueError("decision history reason is invalid for schedule mode")
        if (
            type(entry.get("global_step")) is not int
            or type(entry.get("decision_id")) is not int
            or type(entry.get("update_requested")) is not bool
            or type(entry.get("draft_refit_requested")) is not bool
            or type(entry.get("forced")) is not bool
        ):
            raise ValueError("decision history steps, versions, and flags have invalid types")
        step, decision_id = entry["global_step"], entry["decision_id"]
        if (
            step <= values["schedule_origin_step"]
            or step > values["last_decided_step"]
            or decision_id != step - values["schedule_origin_step"]
        ):
            raise ValueError("decision history entry disagrees with exact decision cursor")
        if config.mode in {"always", "adaptive"} and (
            (entry["reason"] == "none")
            == (entry["update_requested"] or entry["draft_refit_requested"])
        ):
            raise ValueError("decision history reason disagrees with requested work")
        if previous_step is not None and (step <= previous_step or decision_id <= previous_id):
            raise ValueError("decision history is not strictly monotonic")
        previous_step, previous_id = step, decision_id
    if history and (
        previous_step != values["last_decided_step"]
        or values["next_decision_id"] != previous_id + 1
    ):
        raise ValueError("decision history tail disagrees with scheduler cursor")
    phase = state.get("phase")
    phases = {"monitoring", "training_burst", "awaiting_post_refit_observation"}
    if phase not in phases:
        raise ValueError("invalid scheduler phase")
    for field in ("acceptance_ewma", "reference_acceptance_ewma"):
        value = state.get(field)
        if value is not None and (
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("scheduler EWMA must be finite and within [0,1]")
    if config.mode != "adaptive" and (
        phase != "monitoring"
        or state.get("acceptance_ewma") is not None
        or state.get("reference_acceptance_ewma") is not None
        or values["valid_observations"] != 0
        or values["burst_updates"] != 0
    ):
        raise ValueError("non-adaptive scheduler must remain in monitoring phase")
    if phase == "awaiting_post_refit_observation" and state.get("last_applied_refit_step") is None:
        raise ValueError("awaiting phase requires an applied refit")
    if phase == "training_burst" and values["burst_updates"] == 0:
        raise ValueError("training_burst phase requires a positive burst count")
    if config.mode == "adaptive":
        if values["valid_observations"] == 0 and (
            state.get("acceptance_ewma") is not None
            or state.get("reference_acceptance_ewma") is not None
        ):
            raise ValueError("zero observations require empty adaptive EWMAs")
        if values["valid_observations"] > 0 and state.get("acceptance_ewma") is None:
            raise ValueError("valid observations require acceptance EWMA")
        if (
            state.get("reference_acceptance_ewma") is not None
            and state.get("acceptance_ewma") is None
        ):
            raise ValueError("reference EWMA requires acceptance EWMA")


def restore_draft_update_scheduler(
    config: DraftUpdateScheduleConfig,
    saved: Mapping[str, object] | None,
    *,
    origin_step: int,
    resuming_from_checkpoint: bool,
) -> DraftUpdateScheduler:
    if saved is None:
        if resuming_from_checkpoint and config.mode != "always":
            raise ValueError("legacy checkpoint without cadence state may resume only in always mode")
        return DraftUpdateScheduler.create(config, origin_step=origin_step)
    if type(saved.get("state_version")) is not int or saved["state_version"] != 1:
        raise ValueError("unsupported draft update schedule state version")
    if saved.get("config") != config.model_dump(mode="json"):
        raise ValueError("resolved draft update schedule does not match checkpoint")
    saved_state = saved.get("state")
    if not isinstance(saved_state, Mapping):
        raise ValueError("draft update schedule state must be a mapping")
    validate_scheduler_state_invariants(config, saved_state)
    restored_origin = saved_state["schedule_origin_step"]
    return DraftUpdateScheduler.create(
        config,
        origin_step=restored_origin,
        restored=saved,
    )


@dataclass(frozen=True, slots=True)
class AppliedDraftSnapshot:
    version: int
    path: str
    size_bytes: int
    sha256: str


def close_applied_draft_snapshot(
    decision: DraftUpdateDecision,
    apply_receipt: Mapping[str, object],
    *,
    snapshot_path: Path,
) -> AppliedDraftSnapshot:
    raw = snapshot_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (
        decision.draft_refit_requested is not True
        or apply_receipt.get("successful") is not True
        or apply_receipt.get("version") != decision.decision_id
        or apply_receipt.get("snapshot_path") != str(snapshot_path.resolve())
        or apply_receipt.get("sha256") != digest
    ):
        raise RuntimeError("applied draft snapshot receipt mismatch")
    return AppliedDraftSnapshot(
        version=decision.decision_id,
        path=str(snapshot_path.resolve()),
        size_bytes=len(raw),
        sha256=digest,
    )


def close_initial_draft_snapshot(
    apply_receipt: Mapping[str, object], snapshot_path: Path
) -> AppliedDraftSnapshot:
    raw = snapshot_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (
        apply_receipt.get("successful") is not True
        or apply_receipt.get("version") != 0
        or apply_receipt.get("snapshot_path") != str(snapshot_path.resolve())
        or apply_receipt.get("sha256") != digest
    ):
        raise RuntimeError("initial serving-draft snapshot receipt mismatch")
    return AppliedDraftSnapshot(
        version=0,
        path=str(snapshot_path.resolve()),
        size_bytes=len(raw),
        sha256=digest,
    )


def validate_applied_draft_snapshot(
    scheduler: DraftUpdateScheduler,
    snapshot: AppliedDraftSnapshot,
) -> None:
    path = Path(snapshot.path)
    raw = path.read_bytes()
    if (
        type(snapshot.version) is not int
        or snapshot.version != scheduler.state.applied_draft_version
        or type(snapshot.size_bytes) is not int
        or snapshot.size_bytes != len(raw)
        or hashlib.sha256(raw).hexdigest() != snapshot.sha256
    ):
        raise ValueError("applied draft snapshot version or digest mismatch")


def durably_install_startup_snapshot(
    grpo_save_state: GRPOSaveState,
    snapshot: AppliedDraftSnapshot,
    *,
    flush_save_state: Callable[[GRPOSaveState], Mapping[str, object]],
) -> Mapping[str, object]:
    grpo_save_state.applied_draft_snapshot = asdict(snapshot)
    receipt = flush_save_state(grpo_save_state)
    if (
        receipt.get("successful") is not True
        or receipt.get("version") != snapshot.version
        or receipt.get("sha256") != snapshot.sha256
    ):
        raise RuntimeError("serving-draft snapshot persistence receipt mismatch")
    return receipt


@dataclass(frozen=True, slots=True)
class DraftStepTransaction:
    transaction_id: str
    decision: DraftUpdateDecision
    base_checkpoint_id: str
    pre_scheduler_sha256: str
    expected_snapshot_path: str | None


@dataclass(frozen=True, slots=True)
class ResolvedDraftStepTransaction:
    transaction: DraftStepTransaction
    decision: DraftUpdateDecision
    outcome: Mapping[str, bool]
    applied_snapshot: AppliedDraftSnapshot | None


class DraftStepTransactionStore(Protocol):
    def begin(self, decision: DraftUpdateDecision) -> DraftStepTransaction: ...
    def resolve(
        self,
        transaction: DraftStepTransaction,
        *,
        decision: DraftUpdateDecision,
        outcome: Mapping[str, bool],
        applied_snapshot: AppliedDraftSnapshot | None,
    ) -> None: ...
    def commit_bundle_atomic(
        self,
        transaction: DraftStepTransaction,
        *,
        save_state: GRPOSaveState,
        ledger_high_water: int,
    ) -> Mapping[str, object]: ...
    def mark_committed(
        self, transaction: DraftStepTransaction, receipt: Mapping[str, object]
    ) -> None: ...
    def pending_intents(self) -> tuple[DraftStepTransaction, ...]: ...
    def resolutions_since(
        self, checkpoint_id: str
    ) -> tuple[ResolvedDraftStepTransaction, ...]: ...
    def resolution_for(
        self, transaction_id: str
    ) -> ResolvedDraftStepTransaction | None: ...
    def lookup_durable_apply_receipt(
        self, transaction: DraftStepTransaction
    ) -> Mapping[str, object] | None: ...
    def resolve_intent_for_recovery(
        self,
        transaction: DraftStepTransaction,
        *,
        apply_receipt: Mapping[str, object] | None,
    ) -> ResolvedDraftStepTransaction: ...
    def validate_checkpoint_contains(
        self, checkpoint_id: str, resolved: ResolvedDraftStepTransaction
    ) -> None: ...
    def discard_after_checkpoint(
        self, *, checkpoint_id: str, ledger_high_water: int
    ) -> None: ...


def close_draft_step_transaction(
    transaction: DraftStepTransaction,
    *,
    decision: DraftUpdateDecision,
    outcome: Mapping[str, bool],
    applied_snapshot: AppliedDraftSnapshot | None,
    scheduler: DraftUpdateScheduler,
    decision_ledger: DraftDecisionLedger,
    grpo_save_state: GRPOSaveState,
    transaction_store: DraftStepTransactionStore,
) -> RuntimeError | None:
    # Resolution is immutable and durable before any in-memory state changes.
    transaction_store.resolve(
        transaction,
        decision=decision,
        outcome=outcome,
        applied_snapshot=applied_snapshot,
    )
    outcome_error: RuntimeError | None = None
    try:
        scheduler.record_outcome(
            decision,
            update_attempted=outcome["update_attempted"],
            update_successful=outcome["update_successful"],
            draft_refit_attempted=outcome["draft_refit_attempted"],
            draft_refit_successful=outcome["draft_refit_successful"],
        )
    except RuntimeError as error:
        # record_outcome closes counters before signaling requested-work failure.
        outcome_error = error
    decision_ledger.append_closed_once(decision, outcome)
    grpo_save_state.draft_update_schedule = scheduler.state_dict()
    if applied_snapshot is not None:
        grpo_save_state.applied_draft_snapshot = asdict(applied_snapshot)
    bundle_receipt = transaction_store.commit_bundle_atomic(
        transaction,
        save_state=grpo_save_state,
        ledger_high_water=decision.decision_id,
    )
    if (
        bundle_receipt.get("successful") is not True
        or bundle_receipt.get("provisional") is not True
        or bundle_receipt.get("base_checkpoint_id")
        != transaction.base_checkpoint_id
        or bundle_receipt.get("decision_id") != decision.decision_id
        or bundle_receipt.get("scheduler_decision_id") != decision.decision_id
        or bundle_receipt.get("snapshot_version")
        != scheduler.state.applied_draft_version
        or bundle_receipt.get("ledger_high_water") != decision.decision_id
    ):
        raise RuntimeError("draft-step atomic bundle receipt mismatch")
    transaction_store.mark_committed(transaction, bundle_receipt)
    return outcome_error


def _checkpoint_member(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("checkpoint member path must be a nonempty string")
    member = (root / relative).resolve()
    if root not in member.parents:
        raise ValueError("checkpoint member escapes checkpoint root")
    return member


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        for member in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(member.relative_to(path)).encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(member.read_bytes()).digest())
        return digest.hexdigest()
    raise ValueError(f"checkpoint artifact is absent: {path}")


def scheduler_decision_high_water(schedule: Mapping[str, object]) -> int:
    state = schedule.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint schedule state is absent")
    next_decision_id = state.get("next_decision_id")
    if type(next_decision_id) is not int or next_decision_id < 1:
        raise ValueError("checkpoint schedule cursor is invalid")
    high_water = next_decision_id - 1
    if "decisions" in state and state.get("decisions") != high_water:
        raise ValueError("legacy decisions field disagrees with schedule cursor")
    return high_water


def load_checkpoint_bundle(checkpoint_path: Path) -> Mapping[str, object]:
    root = checkpoint_path.resolve()
    receipt_path = root / "cadence-checkpoint-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    required_keys = {
        "schema_version", "successful", "checkpoint_id", "checkpoint_path",
        "completed_policy_steps", "current_step", "checkpoint_tree_sha256",
        "components", "scheduler_state_sha256", "draft_update_schedule",
        "applied_draft_snapshot", "decision_ledger",
        "decision_ledger_prefixes", "ledger_high_water", "resumed_from",
        "cadence_terminal_evidence",
    }
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != required_keys
        or receipt.get("schema_version") != 1
        or receipt.get("checkpoint_id") != root.name
        or receipt.get("checkpoint_path") != str(root)
        or receipt.get("successful") is not True
        or type(receipt.get("current_step")) is not int
        or receipt.get("current_step") <= 0
        or receipt.get("completed_policy_steps") != receipt.get("current_step")
        or receipt.get("checkpoint_id")
        != f"step_{receipt.get('current_step')}"
    ):
        raise ValueError("invalid cadence checkpoint identity")
    components = receipt.get("components")
    if not isinstance(components, Mapping) or set(components) != {
        "model", "optimizer", "dataloader_rng"
    }:
        raise ValueError("cadence checkpoint component schema mismatch")
    for name, binding in components.items():
        if not isinstance(binding, Mapping) or set(binding) != {
            "relative_path", "sha256"
        }:
            raise ValueError(f"invalid {name} checkpoint binding")
        member = _checkpoint_member(root, binding.get("relative_path"))
        if binding.get("sha256") != _sha256_path(member):
            raise ValueError(f"{name} checkpoint digest mismatch")
    ledger = receipt.get("decision_ledger")
    if not isinstance(ledger, Mapping) or set(ledger) != {
        "relative_path", "size_bytes", "sha256", "first_decision_id",
        "last_decision_id", "entry_count",
    }:
        raise ValueError("missing checkpoint decision-ledger binding")
    ledger_path = _checkpoint_member(root, ledger.get("relative_path"))
    raw_ledger = ledger_path.read_bytes()
    rows = [json.loads(line) for line in raw_ledger.splitlines()]
    decision_ids = [row.get("decision_id") for row in rows]
    if (
        ledger.get("size_bytes") != len(raw_ledger)
        or ledger.get("sha256") != hashlib.sha256(raw_ledger).hexdigest()
        or ledger.get("entry_count") != len(rows)
        or decision_ids != list(range(1, len(rows) + 1))
        or ledger.get("first_decision_id") != (1 if rows else None)
        or ledger.get("last_decision_id") != (len(rows) if rows else 0)
        or receipt.get("decision_ledger_prefixes") != [ledger]
    ):
        raise ValueError("checkpoint decision-ledger receipt mismatch")
    schedule = receipt.get("draft_update_schedule")
    if (
        not isinstance(schedule, Mapping)
        or receipt.get("scheduler_state_sha256") != canonical_sha256(schedule)
        or not isinstance(schedule.get("state"), Mapping)
        or scheduler_decision_high_water(schedule) != len(rows)
        or receipt.get("ledger_high_water") != len(rows)
    ):
        raise ValueError("checkpoint scheduler/ledger high-water mismatch")
    disabled = schedule.get("mode") == "disabled"
    disabled_zero_fields = (
        "decisions", "attempted_updates", "successful_updates",
        "failed_updates", "skipped_updates", "attempted_refits",
        "successful_refits", "failed_refits", "skipped_refits",
        "forced_updates", "forced_refits",
    )
    if disabled and (
        schedule != disabled_draft_schedule_payload()
        or rows
        or ledger.get("size_bytes") != 0
        or ledger.get("sha256") != hashlib.sha256(b"").hexdigest()
        or ledger.get("first_decision_id") is not None
        or ledger.get("last_decision_id") != 0
        or ledger.get("entry_count") != 0
        or schedule["state"].get("next_decision_id") != 1
        or any(schedule["state"].get(field) != 0 for field in disabled_zero_fields)
        or schedule["state"].get("decision_history", []) != []
    ):
        raise ValueError("disabled draft checkpoint ledger must be exactly empty")
    if not disabled and not rows:
        raise ValueError("enabled draft checkpoint ledger cannot be empty")
    if receipt.get("checkpoint_tree_sha256") != sha256_tree(
        root, exclude={"cadence-checkpoint-receipt.json"}
    ):
        raise ValueError("checkpoint tree digest mismatch")
    terminal_evidence_state = receipt.get("cadence_terminal_evidence")
    if not isinstance(terminal_evidence_state, Mapping):
        raise ValueError("checkpoint cadence terminal evidence is absent")
    CadenceTerminalEvidence.from_state(terminal_evidence_state)
    return receipt


def recover_draft_step_transactions(
    *,
    config: DraftUpdateScheduleConfig | None,
    checkpoint_path: Path,
    transaction_store: DraftStepTransactionStore,
    decision_ledger: DraftDecisionLedger,
    grpo_save_state: GRPOSaveState,
) -> DraftUpdateScheduler | None:
    checkpoint_bundle = load_checkpoint_bundle(checkpoint_path)
    checkpoint_id = str(checkpoint_bundle["checkpoint_id"])
    checkpoint_high_water = int(checkpoint_bundle["ledger_high_water"])
    resolutions = transaction_store.resolutions_since(checkpoint_id)
    pending_intents = transaction_store.pending_intents()
    schedule_payload = checkpoint_bundle["draft_update_schedule"]
    disabled = (
        isinstance(schedule_payload, Mapping)
        and schedule_payload.get("mode") == "disabled"
    )
    if disabled and (
        config is not None
        or checkpoint_high_water != 0
        or resolutions
        or pending_intents
    ):
        raise ValueError("disabled draft resume must have no scheduler transactions")
    if not disabled and config is None:
        raise ValueError("enabled draft resume requires schedule config")
    resolved_by_transaction = {
        item.transaction.transaction_id: item for item in resolutions
    }
    if len(resolved_by_transaction) != len(resolutions):
        raise ValueError("duplicate draft-step transaction resolution")
    for intent in pending_intents:
        resolved = resolved_by_transaction.get(intent.transaction_id)
        if resolved is None:
            resolved = transaction_store.resolution_for(intent.transaction_id)
        if resolved is None:
            apply_receipt = transaction_store.lookup_durable_apply_receipt(intent)
            resolved = transaction_store.resolve_intent_for_recovery(
                intent,
                apply_receipt=apply_receipt,
            )
            resolved_by_transaction[intent.transaction_id] = resolved
        if (
            resolved.transaction != intent
            or resolved.decision != intent.decision
        ):
            raise ValueError("draft-step resolution differs from durable intent")
        # Resolution is deterministic evidence, not permission to advance beyond
        # the last full model/optimizer/dataloader checkpoint.
        if resolved.decision.decision_id <= checkpoint_high_water:
            transaction_store.validate_checkpoint_contains(
                checkpoint_id, resolved
            )
    for resolved in resolved_by_transaction.values():
        if resolved.decision != resolved.transaction.decision:
            raise ValueError("draft-step resolution decision mismatch")
        if resolved.decision.decision_id <= checkpoint_high_water:
            transaction_store.validate_checkpoint_contains(
                checkpoint_id, resolved
            )
    decision_ledger.truncate_to(checkpoint_high_water)
    transaction_store.discard_after_checkpoint(
        checkpoint_id=checkpoint_id,
        ledger_high_water=checkpoint_high_water,
    )
    grpo_save_state.draft_update_schedule = checkpoint_bundle[
        "draft_update_schedule"
    ]
    grpo_save_state.applied_draft_snapshot = checkpoint_bundle[
        "applied_draft_snapshot"
    ]
    grpo_save_state.draft_terminal_evidence = dict(
        checkpoint_bundle["cadence_terminal_evidence"]
    )
    grpo_save_state.draft_decision_ledger_prefixes = (
        []
        if disabled
        else list(checkpoint_bundle["decision_ledger_prefixes"])
    )
    if disabled:
        return None
    assert config is not None
    return restore_draft_update_scheduler(
        config,
        grpo_save_state.draft_update_schedule,
        origin_step=int(checkpoint_bundle["completed_policy_steps"]),
        resuming_from_checkpoint=True,
    )


def restore_serving_draft_after_startup_sync(
    config: DraftUpdateScheduleConfig,
    scheduler: DraftUpdateScheduler,
    rollout_manager: RolloutManager,
    synchronizer: WeightSynchronizer,
    *,
    snapshot: AppliedDraftSnapshot | None,
    snapshot_path: Path | None,
    resuming_from_checkpoint: bool,
    install_snapshot: Callable[
        [AppliedDraftSnapshot], Mapping[str, object]
    ],
) -> AppliedDraftSnapshot:
    if (
        resuming_from_checkpoint
        and snapshot is None
        and config.mode != "always"
    ):
        raise ValueError(
            "resumed non-always cadence requires an applied draft snapshot"
        )
    synchronizer.sync_target_from_current_checkpoint()
    if snapshot is None:
        if scheduler.state.applied_draft_version != 0:
            raise ValueError("resumed applied draft version requires a snapshot")
        if snapshot_path is None:
            raise ValueError("initial serving draft requires an immutable snapshot path")
        receipt = synchronizer.sync_current_trainable_draft(
            snapshot_path=snapshot_path
        )
        applied_snapshot = close_initial_draft_snapshot(receipt, snapshot_path)
        expected_version = 0
    else:
        validate_applied_draft_snapshot(scheduler, snapshot)
        receipt = synchronizer.sync_applied_draft_snapshot(snapshot)
        applied_snapshot = snapshot
        expected_version = snapshot.version
    if (
        receipt.get("successful") is not True
        or receipt.get("version") != expected_version
        or (
            snapshot is not None
            and receipt.get("sha256") != snapshot.sha256
        )
    ):
        raise RuntimeError("startup serving-draft apply receipt mismatch")
    persistence_receipt = install_snapshot(applied_snapshot)
    if (
        persistence_receipt.get("successful") is not True
        or persistence_receipt.get("version") != expected_version
        or persistence_receipt.get("sha256") != applied_snapshot.sha256
    ):
        raise RuntimeError("serving-draft snapshot was not durably installed")
    rollout_manager.set_applied_draft_version(expected_version)
    rollout_manager.enable_reservations()
    return applied_snapshot
```

Place `validate_scheduler_state_invariants`, `AppliedDraftSnapshot`, the transaction store, and snapshot/ledger validators in `draft_update_schedule.py`, and place the authoritative checkpoint producer/loader in `draft_cadence_runtime.py`. Set ledger prefixes to `[]` on a fresh run, but do not leave `applied_draft_snapshot=None` after serving startup. Every step writes an exclusive intent before workers/transfer and an immutable deterministic resolution afterward. Per-step cadence bundles are provisional and bound to the last full training checkpoint ID; they support live-process accounting but are never a resume authority. Only after model, optimizer, scheduler-independent dataloader/RNG state, and the complete sealed ledger prefix are durable does the checkpoint hook write `cadence-checkpoint-receipt.json`, containing the exact checkpoint path/ID, recomputed tree and component digests, scheduler state/hash, applied snapshot binding, and ledger receipt/high-water. Recovery accepts only `load_checkpoint_bundle(checkpoint_path)` output after that loader independently recomputes every binding. It does not accept a caller-supplied mapping. Recovery enumerates the complete post-checkpoint resolution set as well as every unresolved intent. For an unresolved intent it accepts only a durable apply receipt bound to the exact transaction/decision and validated snapshot digest; such a receipt deterministically produces the successful resolution, while a missing receipt deterministically produces the attempted-but-failed resolution with no applied snapshot. Recovery audits any entry at or below the checkpoint high-water against the checkpoint-local state, then quarantines all post-checkpoint intents/resolutions/snapshots and atomically truncates the unsealed ledger suffix to the checkpoint high-water. It never rolls cadence or serving draft past model/optimizer/dataloader state. Tests inject crashes after intent, apply receipt, resolution, outcome, ledger, provisional bundle, and checkpoint binding. Pass `resuming_from_checkpoint=False` only on a truly fresh launch and `True` whenever checkpoint metadata was loaded.

Startup is component-separated in both controller paths: restore current target checkpoint bytes with `sync_target_from_current_checkpoint`, then restore serving draft bytes from `applied_draft_snapshot` with `sync_applied_draft_snapshot`. Only a truly fresh run or a legacy `always` resume whose new schedule origin has version 0 may use `sync_current_trainable_draft`. Require the returned draft apply receipt to contain the exact snapshot SHA/version before publishing that version and enabling reservations. A fixed `refit_only` checkpoint taken after Step 17 with its last applied refit at Step 10 therefore restores Step-10 serving bytes and publishes version 10; it never sends Step-17 trainable draft bytes and labels them version 10.

`FileDraftStepTransactionStore.begin` durably binds the complete immutable decision and `base_checkpoint_id`. `commit_bundle_atomic` writes only a provisional live-process bundle and repeats that base ID. `checkpoint_closed` runs after the NeMo checkpoint writer exposes the exact model, optimizer, and dataloader/RNG paths. It seals and copies the complete ledger prefix into the checkpoint, recomputes every component digest and the whole checkpoint-tree digest, verifies scheduler decisions equal the ledger high-water, then writes exactly `checkpoints/step_<n>/cadence-checkpoint-receipt.json`. On restart `load_checkpoint_bundle` reloads that exact filename and recomputes all path, tree, component, scheduler, and ledger bindings before returning it as authority; the newest provisional bundle is never a resume authority. Post-checkpoint artifacts are quarantined under a recovery receipt and excluded from serving publication; ledger truncation is an atomic replace plus directory fsync.

Add this concrete runtime producer and call it from the shared checkpoint/terminal hooks used by synchronous GRPO and single-controller. The product never relies on the experiment launcher to invent runtime receipts after training.

```python
import uuid
from math import isfinite


def write_json_exclusive_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class CadenceRuntimeConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    result_dir: str | None = None
    required_checkpoint_steps: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.enabled and not self.result_dir:
            raise ValueError("cadence runtime result_dir is required")
        if any(type(step) is not int or step <= 0 for step in self.required_checkpoint_steps):
            raise ValueError("required checkpoint steps must be positive integers")
        return self


def disabled_draft_schedule_payload() -> dict[str, object]:
    return {
        "mode": "disabled",
        "state": {
            "decisions": 0,
            "next_decision_id": 1,
            "attempted_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "skipped_updates": 0,
            "attempted_refits": 0,
            "successful_refits": 0,
            "failed_refits": 0,
            "skipped_refits": 0,
            "forced_updates": 0,
            "forced_refits": 0,
            "decision_history": [],
        },
        "events": [],
        "not_applicable_metrics": [
            "draft_loss", "draft_grad_norm", "applied_draft_version"
        ],
    }


def seal_checkpoint_ledger(
    decision_ledger: DraftDecisionLedger,
    destination: Path,
    *,
    allow_empty: bool,
) -> dict[str, object]:
    if decision_ledger.next_decision_id == 1:
        if decision_ledger.sealed_prefixes or not allow_empty:
            raise RuntimeError("empty checkpoint ledger is valid only for disabled draft")
        segments: tuple[DecisionLedgerReceipt, ...] = ()
    else:
        current = decision_ledger.seal_prefix()
        segments = (*decision_ledger.sealed_prefixes, current)
    raw = b""
    expected_first = 1
    for segment in segments:
        validate_decision_ledger_receipt(segment)
        if segment.first_decision_id != expected_first:
            raise ValueError("checkpoint ledger segments are not contiguous")
        raw += Path(segment.path).read_bytes()
        expected_first = segment.last_decision_id + 1
    rows = [json.loads(line) for line in raw.splitlines()]
    if [row.get("decision_id") for row in rows] != list(
        range(1, len(rows) + 1)
    ):
        raise ValueError("checkpoint ledger prefix is not exactly 1..N")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "relative_path": str(destination.name),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "first_decision_id": 1 if rows else None,
        "last_decision_id": len(rows),
        "entry_count": len(rows),
    }


@dataclass(frozen=True, slots=True)
class ResumeLedgerOpenResult:
    ledger: DraftDecisionLedger
    quarantine_receipt_path: Path


@dataclass(slots=True)
class CadenceTerminalEvidence:
    update_receipts_by_decision: dict[int, Mapping[str, object]]
    observations_by_refit_step: dict[int, Mapping[str, object]]

    def state_dict(self) -> dict[str, object]:
        return {
            "update_receipts_by_decision": {
                str(key): dict(value)
                for key, value in self.update_receipts_by_decision.items()
            },
            "observations_by_refit_step": {
                str(key): dict(value)
                for key, value in self.observations_by_refit_step.items()
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "CadenceTerminalEvidence":
        updates = state.get("update_receipts_by_decision")
        observations = state.get("observations_by_refit_step")
        if not isinstance(updates, Mapping) or not isinstance(observations, Mapping):
            raise ValueError("invalid checkpointed cadence terminal evidence")
        if any(not isinstance(value, Mapping) for value in updates.values()) or any(
            not isinstance(value, Mapping) for value in observations.values()
        ):
            raise ValueError("invalid checkpointed cadence evidence entry")
        return cls(
            update_receipts_by_decision={
                int(key): value for key, value in updates.items()
            },
            observations_by_refit_step={
                int(key): value for key, value in observations.items()
            },
        )


def record_terminal_post_refit_observation(
    evidence: CadenceTerminalEvidence,
    *,
    decision: DraftUpdateDecision,
    last_applied_refit_step: int | None,
    acceptance_rate: float | None,
) -> CadenceTerminalEvidence:
    if (
        last_applied_refit_step is None
        or last_applied_refit_step != decision.global_step - 1
    ):
        return evidence
    if (
        type(acceptance_rate) not in (int, float)
        or not isfinite(float(acceptance_rate))
        or not 0.0 <= float(acceptance_rate) <= 1.0
        or decision.applied_draft_version <= 0
    ):
        raise ValueError("immediate post-refit science observation is invalid")
    observation = {
        "refit_step": last_applied_refit_step,
        "observation_step": decision.global_step,
        "applied_draft_version": decision.applied_draft_version,
        "acceptance_rate": float(acceptance_rate),
    }
    previous = evidence.observations_by_refit_step.setdefault(
        last_applied_refit_step, observation
    )
    if previous != observation:
        raise ValueError("conflicting post-refit science observation")
    return evidence


def _move_ledger_to_quarantine(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def reconcile_ledger_quarantine(
    recovery_dir: Path,
    result_root: Path,
) -> Mapping[str, object]:
    intent_path = recovery_dir / "ledger-quarantine-intent.json"
    intent = json.loads(intent_path.read_text())
    root = result_root.resolve()
    if (
        intent.get("schema_version") != 1
        or intent.get("state") != "intent"
        or not isinstance(intent.get("checkpoint_id"), str)
        or not isinstance(intent.get("recovery_id"), str)
        or not isinstance(intent.get("artifacts"), list)
    ):
        raise ValueError("invalid ledger quarantine intent")
    for artifact in intent["artifacts"]:
        if not isinstance(artifact, Mapping):
            raise ValueError("invalid ledger quarantine artifact")
        source = Path(str(artifact.get("original_path"))).resolve()
        destination = Path(str(artifact.get("quarantine_path"))).resolve()
        if source.parent != root or destination.parent != recovery_dir.resolve():
            raise ValueError("ledger quarantine path escapes transaction roots")
        source_exists = source.is_file()
        destination_exists = destination.is_file()
        if source_exists == destination_exists:
            raise RuntimeError("ledger quarantine has ambiguous source/destination state")
        if source_exists:
            raw = source.read_bytes()
            if (
                artifact.get("size_bytes") != len(raw)
                or artifact.get("sha256") != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError("ledger quarantine source digest mismatch")
            _move_ledger_to_quarantine(source, destination)
        raw = destination.read_bytes()
        if (
            artifact.get("size_bytes") != len(raw)
            or artifact.get("sha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("ledger quarantine destination digest mismatch")
    for directory in (root, recovery_dir.parent, recovery_dir):
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    receipt_path = recovery_dir / "ledger-quarantine-receipt.json"
    receipt = {
        **intent,
        "state": "resolved",
        "receipt_path": str(receipt_path.resolve()),
    }
    write_json_exclusive_atomic(receipt_path, receipt)
    return receipt


def open_resume_decision_ledger(
    checkpoint_path: Path,
    result_root: Path,
) -> ResumeLedgerOpenResult:
    bundle = load_checkpoint_bundle(checkpoint_path)
    root = result_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root not in checkpoint_path.resolve().parents:
        raise ValueError("resume checkpoint is outside cadence result root")
    recovery_parent = root / "recovery"
    recovery_parent.mkdir(parents=True, exist_ok=True)
    incomplete = sorted(
        directory for directory in recovery_parent.glob("resume-*")
        if (directory / "ledger-quarantine-intent.json").is_file()
        and not (directory / "ledger-quarantine-receipt.json").exists()
    )
    if len(incomplete) > 1:
        raise RuntimeError("multiple incomplete ledger quarantine transactions")
    if incomplete:
        quarantine_receipt = reconcile_ledger_quarantine(incomplete[0], root)
    else:
        recovery_id = str(uuid.uuid4())
        recovery_dir = recovery_parent / (
            f"resume-{bundle['checkpoint_id']}-{recovery_id}"
        )
        recovery_dir.mkdir(exist_ok=False)
        candidates = sorted({
            *root.glob("draft-decision-ledger-after-step_*.jsonl"),
            *root.glob("draft-decision-ledger-resume-step_*-*.jsonl"),
        })
        artifacts: list[dict[str, object]] = []
        for index, source in enumerate(candidates):
            resolved = source.resolve()
            if resolved.parent != root or not resolved.is_file():
                raise ValueError("post-checkpoint ledger suffix escapes result root")
            raw = resolved.read_bytes()
            destination = recovery_dir / f"{index:04d}-{resolved.name}"
            artifacts.append({
                "original_path": str(resolved),
                "quarantine_path": str(destination.resolve()),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        new_suffix = root / (
            f"draft-decision-ledger-resume-{bundle['checkpoint_id']}-{recovery_id}.jsonl"
        )
        write_json_exclusive_atomic(
            recovery_dir / "ledger-quarantine-intent.json",
            {
                "schema_version": 1,
                "state": "intent",
                "checkpoint_id": bundle["checkpoint_id"],
                "recovery_id": recovery_id,
                "artifacts": artifacts,
                "new_suffix_path": str(new_suffix),
            },
        )
        quarantine_receipt = reconcile_ledger_quarantine(recovery_dir, root)
    if quarantine_receipt.get("checkpoint_id") != bundle["checkpoint_id"]:
        raise ValueError("ledger quarantine checkpoint mismatch")
    ledger_binding = bundle["decision_ledger"]
    if not isinstance(ledger_binding, Mapping):
        raise ValueError("checkpoint ledger binding is absent")
    high_water = int(bundle["ledger_high_water"])
    prefixes: tuple[DecisionLedgerReceipt, ...]
    if high_water == 0:
        prefixes = ()
    else:
        prefix = DecisionLedgerReceipt(
            path=str(
                checkpoint_path.resolve()
                / str(ledger_binding["relative_path"])
            ),
            size_bytes=int(ledger_binding["size_bytes"]),
            sha256=str(ledger_binding["sha256"]),
            first_decision_id=int(ledger_binding["first_decision_id"]),
            last_decision_id=int(ledger_binding["last_decision_id"]),
            entry_count=int(ledger_binding["entry_count"]),
        )
        validate_decision_ledger_receipt(prefix)
        prefixes = (prefix,)
    suffix = Path(str(quarantine_receipt["new_suffix_path"])).resolve()
    if suffix.parent != root or suffix.exists():
        raise FileExistsError("resume ledger suffix identity collision")
    ledger = DraftDecisionLedger(suffix, sealed_prefixes=prefixes)
    return ResumeLedgerOpenResult(
        ledger=ledger,
        quarantine_receipt_path=Path(str(quarantine_receipt["receipt_path"])),
    )


def _validated_update_receipt(
    decision_id: int,
    global_step: int,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    path = Path(str(receipt.get("path"))).resolve()
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("terminal update receipt payload is not an object")
    digests = (
        payload.get("draft_model_sha256"),
        payload.get("draft_optimizer_sha256"),
    )
    if (
        receipt.get("successful") is not True
        or receipt.get("decision_id") != decision_id
        or receipt.get("global_step") != global_step
        or receipt.get("size_bytes") != len(raw)
        or receipt.get("sha256") != hashlib.sha256(raw).hexdigest()
        or payload.get("schema_version") != 1
        or payload.get("successful") is not True
        or payload.get("decision_id") != decision_id
        or payload.get("global_step") != global_step
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        )
    ):
        raise ValueError("terminal update receipt is not digest bound")
    return dict(receipt)


def build_terminal_schedule_payload(
    checkpoint: Mapping[str, object],
    evidence: CadenceTerminalEvidence,
) -> dict[str, object]:
    if checkpoint.get("cadence_terminal_evidence") != evidence.state_dict():
        raise ValueError("terminal evidence differs from final checkpoint")
    current_step = int(checkpoint["current_step"])
    saved = checkpoint["draft_update_schedule"]
    if not isinstance(saved, Mapping):
        raise ValueError("terminal checkpoint lacks schedule state")
    if saved.get("mode") == "disabled":
        if evidence.update_receipts_by_decision or evidence.observations_by_refit_step:
            raise ValueError("disabled draft cannot have terminal events")
        zero_fields = {
            key: 0 for key in (
                "attempted_updates", "successful_updates", "failed_updates",
                "skipped_updates", "attempted_refits", "successful_refits",
                "failed_refits", "skipped_refits", "forced_updates",
                "forced_refits",
            )
        }
        return {
            "mode": "disabled",
            "current_step": current_step,
            **zero_fields,
            "policy_refit_count": current_step,
            "decision_ids": [], "global_steps": [], "updated_steps": [],
            "refit_steps": [], "forced_update_steps": [],
            "forced_refit_steps": [], "update_receipts": [],
            "post_event_observations": [], "pending_post_event_steps": [],
            "refit_versions": [], "decision_reasons": [],
            "decision_ledger_segments": [],
            "not_applicable_metrics": [
                "applied_draft_version", "draft_grad_norm", "draft_loss"
            ],
        }
    ledger = checkpoint["decision_ledger"]
    if not isinstance(ledger, Mapping):
        raise ValueError("terminal checkpoint lacks decision ledger")
    ledger_path = (
        Path(str(checkpoint["checkpoint_path"]))
        / str(ledger["relative_path"])
    ).resolve()
    rows = [json.loads(line) for line in ledger_path.read_bytes().splitlines()]
    decision_count = scheduler_decision_high_water(saved)
    if (
        len(rows) != decision_count
        or [int(row["decision_id"]) for row in rows]
        != list(range(1, decision_count + 1))
        or not rows
        or int(rows[-1]["global_step"]) != current_step
    ):
        raise ValueError("terminal ledger does not cover the scheduler cursor")
    outcomes = [row["outcome"] for row in rows]
    if any(
        outcome["update_attempted"] and not outcome["update_successful"]
        or outcome["draft_refit_attempted"]
        and not outcome["draft_refit_successful"]
        for outcome in outcomes
    ):
        raise ValueError("successful terminal payload cannot contain failed work")
    update_rows = [row for row in rows if row["update_requested"]]
    refit_rows = [row for row in rows if row["draft_refit_requested"]]
    updated_steps = [int(row["global_step"]) for row in update_rows]
    refit_steps = [int(row["global_step"]) for row in refit_rows]
    updated_decision_ids = [int(row["decision_id"]) for row in update_rows]
    refit_version_by_step = {
        int(row["global_step"]): int(row["decision_id"]) for row in refit_rows
    }
    if set(evidence.update_receipts_by_decision) != set(updated_decision_ids):
        raise ValueError("terminal update-receipt cardinality mismatch")
    update_receipts = [
        _validated_update_receipt(
            int(row["decision_id"]),
            int(row["global_step"]),
            evidence.update_receipts_by_decision[int(row["decision_id"])],
        )
        for row in update_rows
    ]
    resumed_from = checkpoint.get("resumed_from")
    resume_after_step = (
        None
        if resumed_from is None
        else int(Path(str(resumed_from)).name.removeprefix("step_"))
    )
    all_observable_refits = [step for step in refit_steps if step < current_step]
    observable_refits = [
        step for step in all_observable_refits
        if resume_after_step is None or step >= resume_after_step
    ]
    pending_refits = [step for step in refit_steps if step == current_step]
    if set(evidence.observations_by_refit_step) != set(all_observable_refits):
        raise ValueError("terminal post-refit observation cardinality mismatch")
    observations: list[dict[str, object]] = []
    for step in observable_refits:
        observation = evidence.observations_by_refit_step[step]
        acceptance = observation.get("acceptance_rate")
        if (
            observation.get("refit_step") != step
            or observation.get("observation_step") != step + 1
            or observation.get("applied_draft_version")
            != refit_version_by_step[step]
            or type(acceptance) not in (int, float)
            or not isfinite(float(acceptance))
            or not 0.0 <= float(acceptance) <= 1.0
        ):
            raise ValueError("terminal post-refit observation mismatch")
        observations.append(dict(observation))
    def count(name: str) -> int:
        return sum(bool(outcome[name]) for outcome in outcomes)
    config_payload = saved.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("terminal schedule config is absent")
    return {
        "mode": config_payload["mode"],
        "current_step": current_step,
        "attempted_updates": count("update_attempted"),
        "successful_updates": count("update_successful"),
        "failed_updates": count("update_attempted") - count("update_successful"),
        "skipped_updates": count("update_skipped"),
        "attempted_refits": count("draft_refit_attempted"),
        "successful_refits": count("draft_refit_successful"),
        "failed_refits": count("draft_refit_attempted") - count("draft_refit_successful"),
        "skipped_refits": count("draft_refit_skipped"),
        "forced_updates": count("forced_update"),
        "forced_refits": count("forced_refit"),
        "policy_refit_count": current_step,
        "decision_ids": [int(row["decision_id"]) for row in rows],
        "global_steps": [int(row["global_step"]) for row in rows],
        "updated_steps": updated_steps,
        "refit_steps": refit_steps,
        "forced_update_steps": [
            int(row["global_step"]) for row in rows
            if row["outcome"]["forced_update"]
        ],
        "forced_refit_steps": [
            int(row["global_step"]) for row in rows
            if row["outcome"]["forced_refit"]
        ],
        "update_receipts": update_receipts,
        "post_event_observations": observations,
        "pending_post_event_steps": pending_refits,
        "refit_versions": [
            {
                "refit_step": int(row["global_step"]),
                "applied_draft_version": int(row["decision_id"]),
            }
            for row in refit_rows
        ],
        "decision_reasons": [str(row["reason"]) for row in rows],
        "decision_ledger_segments": [{
            "path": str(ledger_path),
            "size_bytes": ledger["size_bytes"],
            "sha256": ledger["sha256"],
            "first_decision_id": ledger["first_decision_id"],
            "last_decision_id": ledger["last_decision_id"],
            "entry_count": ledger["entry_count"],
        }],
    }


class CadenceRuntimeWriter:
    def __init__(self, config: CadenceRuntimeConfig) -> None:
        if not config.enabled or config.result_dir is None:
            raise ValueError("cadence runtime writer requires enabled config")
        self.root = Path(config.result_dir).resolve()
        self.required_steps = frozenset(config.required_checkpoint_steps)
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipt_session_id = str(uuid.uuid4())
        self.update_receipt_root = self.root / "update-receipts"
        self.update_receipt_root.mkdir(parents=True, exist_ok=True)
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def successful_update_closed(
        self,
        *,
        decision: DraftUpdateDecision,
        worker_receipt: Mapping[str, object],
        evidence: CadenceTerminalEvidence,
        save_state: GRPOSaveState,
    ) -> CadenceTerminalEvidence:
        if not decision.update_requested:
            raise ValueError("cannot receipt a skipped draft update")
        required = {
            "successful": True,
            "decision_id": decision.decision_id,
            "global_step": decision.global_step,
        }
        if any(worker_receipt.get(key) != value for key, value in required.items()):
            raise ValueError("worker update receipt disagrees with decision")
        for key in ("draft_model_sha256", "draft_optimizer_sha256"):
            value = worker_receipt.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"worker update receipt lacks {key}")
        if decision.decision_id in evidence.update_receipts_by_decision:
            raise RuntimeError("duplicate successful-update evidence")
        saved_evidence = getattr(save_state, "draft_terminal_evidence", None)
        if saved_evidence not in (None, evidence.state_dict()):
            raise RuntimeError("checkpointed terminal evidence diverged before update")
        receipt_path = (
            self.update_receipt_root
            / f"{self.receipt_session_id}-decision_{decision.decision_id}.json"
        )
        payload = {
            "schema_version": 1,
            **required,
            "draft_model_sha256": worker_receipt["draft_model_sha256"],
            "draft_optimizer_sha256": worker_receipt["draft_optimizer_sha256"],
        }
        write_json_exclusive_atomic(receipt_path, payload)
        raw = receipt_path.read_bytes()
        binding = {
            "successful": True,
            "decision_id": decision.decision_id,
            "global_step": decision.global_step,
            "path": str(receipt_path.resolve()),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        evidence.update_receipts_by_decision[decision.decision_id] = binding
        save_state.draft_terminal_evidence = evidence.state_dict()
        return evidence

    def checkpoint_closed(
        self,
        *,
        current_step: int,
        checkpoint_path: Path,
        save_state: GRPOSaveState,
        component_paths: Mapping[str, Path],
        decision_ledger: DraftDecisionLedger,
        terminal_evidence: CadenceTerminalEvidence,
    ) -> DraftDecisionLedger:
        expected = (self.root / "checkpoints" / f"step_{current_step}").resolve()
        if checkpoint_path.resolve() != expected or not checkpoint_path.is_dir():
            raise ValueError("checkpoint path is outside cadence result identity")
        checkpoint_id = f"step_{current_step}"
        if set(component_paths) != {"model", "optimizer", "dataloader_rng"}:
            raise RuntimeError("cadence cannot close a partial training checkpoint")
        components: dict[str, dict[str, str]] = {}
        for name, path in component_paths.items():
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(expected)
            except ValueError as error:
                raise ValueError("checkpoint component escapes checkpoint") from error
            components[name] = {
                "relative_path": str(relative),
                "sha256": _sha256_path(resolved),
            }
        schedule = save_state.draft_update_schedule
        if not isinstance(schedule, Mapping) or not isinstance(
            schedule.get("state"), Mapping
        ):
            raise ValueError("checkpoint requires scheduler state")
        disabled = schedule.get("mode") == "disabled"
        if disabled and schedule != disabled_draft_schedule_payload():
            raise ValueError("disabled draft schedule payload is not neutral")
        ledger = seal_checkpoint_ledger(
            decision_ledger,
            expected / "draft-decision-ledger.jsonl",
            allow_empty=disabled,
        )
        ledger_high_water = int(ledger["last_decision_id"])
        if scheduler_decision_high_water(schedule) != ledger_high_water:
            raise ValueError("terminal scheduler decisions differ from ledger")
        if disabled and (
            ledger != {
                "relative_path": "draft-decision-ledger.jsonl",
                "size_bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
                "first_decision_id": None,
                "last_decision_id": 0,
                "entry_count": 0,
            }
            or schedule["state"].get("next_decision_id") != 1
        ):
            raise ValueError("disabled draft must have an explicit empty ledger")
        if not disabled and ledger_high_water == 0:
            raise ValueError("enabled draft checkpoint cannot have an empty ledger")
        payload = {
            "schema_version": 1,
            "successful": True,
            "checkpoint_id": checkpoint_id,
            "completed_policy_steps": current_step,
            "current_step": current_step,
            "checkpoint_path": str(expected),
            "checkpoint_tree_sha256": sha256_tree(
                expected,
                exclude={"cadence-checkpoint-receipt.json"},
            ),
            "components": components,
            "scheduler_state_sha256": canonical_sha256(schedule),
            "draft_update_schedule": schedule,
            "applied_draft_snapshot": save_state.applied_draft_snapshot,
            "cadence_terminal_evidence": terminal_evidence.state_dict(),
            "decision_ledger": ledger,
            "decision_ledger_prefixes": [ledger],
            "ledger_high_water": ledger_high_water,
            "resumed_from": loaded_checkpoint_path_or_none(),
        }
        save_state.draft_terminal_evidence = terminal_evidence.state_dict()
        write_json_exclusive_atomic(
            expected / "cadence-checkpoint-receipt.json", payload
        )
        if current_step in self.required_steps:
            write_json_exclusive_atomic(
                self.root / f"checkpoint-runtime-step_{current_step}.json",
                payload,
            )
        if disabled:
            save_state.draft_decision_ledger_prefixes = []
            return DraftDecisionLedger(
                self.root / f"draft-decision-ledger-after-step_{current_step}.jsonl"
            )
        checkpoint_prefix = DecisionLedgerReceipt(
            path=str(expected / str(ledger["relative_path"])),
            size_bytes=int(ledger["size_bytes"]),
            sha256=str(ledger["sha256"]),
            first_decision_id=int(ledger["first_decision_id"]),
            last_decision_id=int(ledger["last_decision_id"]),
            entry_count=int(ledger["entry_count"]),
        )
        save_state.draft_decision_ledger_prefixes = [asdict(checkpoint_prefix)]
        return DraftDecisionLedger(
            self.root / f"draft-decision-ledger-after-step_{current_step}.jsonl",
            sealed_prefixes=(checkpoint_prefix,),
        )

    def terminal_closed(
        self,
        *,
        current_step: int,
        final_checkpoint_path: Path,
        terminal_evidence: CadenceTerminalEvidence,
    ) -> None:
        missing = [
            step for step in sorted(self.required_steps)
            if not (self.root / f"checkpoint-runtime-step_{step}.json").is_file()
        ]
        if missing:
            raise RuntimeError(f"missing required cadence checkpoints: {missing}")
        checkpoint = load_checkpoint_bundle(final_checkpoint_path)
        if checkpoint.get("current_step") != current_step:
            raise ValueError("final checkpoint step disagrees with terminal step")
        schedule_payload = build_terminal_schedule_payload(
            checkpoint, terminal_evidence
        )
        write_json_exclusive_atomic(
            self.root / "checkpoint-runtime.json", checkpoint
        )
        write_json_exclusive_atomic(
            self.root / "schedule-runtime.json",
            schedule_payload,
        )
```

Add `cadence_runtime: CadenceRuntimeConfig = Field(default_factory=CadenceRuntimeConfig)` to `MasterConfig`; its default is disabled. The runtime writer, terminal evidence, and science preflight exist only when enabled. `checkpoint_closed` runs after every successful periodic checkpoint and only after the model, optimizer, scheduler state, applied-draft snapshot, dataloader state, and ledger prefix are durable. Both controllers install its returned writable suffix before the next decision. Resume intent-first quarantines stale suffixes, validates the checkpoint-bound prefix, and opens a UUID-qualified suffix before transaction recovery. In experiment mode, a successful requested update first calls `successful_update_closed`, which exclusively persists model/optimizer digests in a writer-session-qualified path and installs the binding in checkpointable evidence before transfer, checkpoint, or publication. A replay from an older full checkpoint opens a new receipt session, so an uncheckpointed receipt from the crashed process cannot collide with the replayed decision; checkpointed evidence continues to bind its original immutable files. Both controllers then record strict Step+1 acceptance/serving-version observations and call `terminal_closed`. Fixed and `always` collect this evidence only in experiment mode and still pass `acceptance=None` to scheduling; adaptive alone consumes acceptance. With runtime disabled no writer/evidence/science access is added to legacy nonadaptive paths. `build_terminal_schedule_payload` derives counters, decision/global-step mappings, reasons, event steps, ledger hashes, decision-ID refit versions, pending events, and the post-resume observation subset from the final checkpoint. The real controller-to-writer-to-experiment-validator matrix lives in experiments Task 2 after product Task 10. Disabled-draft controls retain their explicit empty ledger and neutral schedule receipt. Unit tests cover periodic checkpoint continuation/resume, quarantine crashes, receipt exclusivity and ordering, exact snapshot/ledger binding, nonzero-origin mapping, resumed-observation selection, and missing producer artifacts.

- [ ] **Step 4: Run the GREEN checkpoint and resume tests.**

Run: `uv run --group test pytest -q tests/unit/algorithms/test_draft_schedule_checkpoint.py tests/unit/algorithms/test_draft_cadence_runtime.py tests/unit/algorithms/test_grpo.py tests/unit/single_controller/test_sc_checkpointing.py -k 'draft_update_schedule or draft_cadence_runtime or legacy_checkpoint or corrupt_scheduler or applied_draft_snapshot' && uv run ruff check nemo_rl/algorithms/draft_update_schedule.py nemo_rl/algorithms/draft_cadence_runtime.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/grpo_sync.py nemo_rl/algorithms/single_controller.py tests/unit/algorithms/test_draft_schedule_checkpoint.py tests/unit/algorithms/test_draft_cadence_runtime.py`

Expected: selected tests PASS and Ruff reports no errors.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/algorithms/draft_update_schedule.py nemo_rl/algorithms/draft_cadence_runtime.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/grpo_sync.py nemo_rl/algorithms/single_controller.py tests/unit/algorithms/test_draft_schedule_checkpoint.py tests/unit/algorithms/test_draft_cadence_runtime.py tests/unit/single_controller/test_sc_checkpointing.py
git commit -S -s -m "feat(draft): checkpoint update schedule state"
git verify-commit HEAD
```

Expected: commit signature verifies and the checkpoint schema is the only persisted format change.

### Task 4: Broadcast one immutable decision through both training APIs

**Files:**
- Modify: `nemo_rl/data_plane/worker_mixin.py`
- Modify: `nemo_rl/models/policy/interfaces.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/tq_policy.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `tests/unit/models/policy/test_split_api_wrappers.py`
- Modify: `tests/unit/models/megatron/test_draft_step_state.py`
- Create: `tests/unit/models/megatron/test_draft_cadence_entrypoints.py`
- Create: `tests/unit/distributed/test_draft_cadence_consensus.py`

**Interfaces:**
- Consumes: Task 2 `DraftUpdateDecision | None`.
- Produces: `train(data, loss_fn, eval_mode=False, gbs=None, mbs=None, check_dim_skip_keys=None, *, draft_update_decision=None) -> dict[str, Any]` for the monolithic CP1 path and `begin_train_step(loss_fn, gbs=None, mbs=None, *, draft_update_decision=None) -> None` for the split packed CP>1 path across interfaces, LMPolicy, TQ, presharded mixin, and worker. Draft-enabled workers validate the same decision before any mutation; split workers then store it in their open-step state.

- [ ] **Step 1: Write RED fanout, missing-decision, and non-draft compatibility tests.**

```python
import inspect
from unittest.mock import MagicMock, patch

from nemo_rl.algorithms.draft_update_schedule import DraftUpdateDecision
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorker,
)


def tq_policy_for_test():
    policy = object.__new__(TQPolicy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.flops_tracker = None
    policy.worker_group = MagicMock()
    policy.worker_group.run_all_workers_single_data.return_value = []
    return policy


@patch("nemo_rl.models.policy.tq_policy.ray.get")
def test_tq_begin_fans_out_identical_draft_decision(mock_get) -> None:
    tq_policy = tq_policy_for_test()
    decision = DraftUpdateDecision(3, 7, False, False, "none", None)
    tq_policy.begin_train_step(loss_fn=object(), draft_update_decision=decision)
    call = tq_policy.worker_group.run_all_workers_single_data.call_args
    assert call.kwargs["draft_update_decision"] == decision


def test_consensus_precedes_gradient_mutation_in_both_entrypoints() -> None:
    for entrypoint in (MegatronPolicyWorker.train, MegatronPolicyWorker.begin_train_step):
        source = inspect.getsource(entrypoint)
        assert source.index("validate_draft_update_decision_consensus(") < source.index(
            "optimizer.zero_grad("
        )

```

Add this executable two-rank test; it is not an inspection or mocked-collective test:

```python
import os

import pytest
import torch
import torch.distributed as dist

from nemo_rl.algorithms.draft_update_schedule import DraftUpdateDecision
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    validate_draft_update_decision_consensus,
)


def _init_torchrun_group() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://")


def test_full_decision_mismatch_raises_on_every_rank() -> None:
    _init_torchrun_group()
    rank = int(os.environ["RANK"])
    decision = DraftUpdateDecision(
        global_step=3,
        decision_id=7 + rank,
        update_requested=True,
        draft_refit_requested=True,
        reason="always",
        observed_acceptance=0.8,
    )
    with pytest.raises(RuntimeError, match="decision mismatch across ranks"):
        validate_draft_update_decision_consensus(
            decision,
            draft_enabled=True,
            group=dist.group.WORLD,
            device=torch.device("cpu"),
        )


def test_missing_required_decision_raises_on_every_rank_after_collectives() -> None:
    _init_torchrun_group()
    rank = int(os.environ["RANK"])
    decision = None if rank == 0 else DraftUpdateDecision(
        global_step=3,
        decision_id=7,
        update_requested=True,
        draft_refit_requested=True,
        reason="always",
        observed_acceptance=None,
    )
    with pytest.raises(RuntimeError, match="required.*missing on at least one rank"):
        validate_draft_update_decision_consensus(
            decision,
            draft_enabled=True,
            group=dist.group.WORLD,
            device=torch.device("cpu"),
        )
```

- [ ] **Step 2: Run the RED API tests and confirm the new keyword is rejected.**

Run: `uv run --group test pytest -q tests/unit/models/policy/test_split_api_wrappers.py tests/unit/models/megatron/test_draft_step_state.py tests/unit/models/megatron/test_draft_cadence_entrypoints.py -k 'draft_update_decision or cp1_and_cp2'`

Expected: FAIL with `TypeError: begin_train_step() got an unexpected keyword argument 'draft_update_decision'`.

- [ ] **Step 3: Thread and validate the immutable decision before mutation.**

```python
def begin_train_step(
    self,
    loss_fn: LossFunction,
    gbs: Optional[int] = None,
    mbs: Optional[int] = None,
    *,
    draft_update_decision: DraftUpdateDecision | None = None,
) -> None:
    draft_enabled = bool(self.cfg.get("draft") and self.cfg["draft"].enabled)
    draft_update_decision = validate_draft_update_decision_consensus(
        draft_update_decision,
        draft_enabled=draft_enabled,
    )
    existing = getattr(self, "_train_step_state", None)
    if existing is not None:
        raise RuntimeError("a train step is already open")
    state = self._split_step_state_init(loss_fn=loss_fn, gbs=gbs, mbs=mbs)
    state["draft_update_decision"] = draft_update_decision
```

Add the same keyword-only argument and validation to `MegatronPolicyWorker.train`, `PolicyInterface.train`, `LMPolicy.train`, `TQPolicy.train_from_meta`, and the relevant worker-mixin forwarding method. In `_train_policy_from_meta`, pass the controller decision exactly once to `policy.train_from_meta` when `_should_use_split_draft_training(...)` is false and exactly once to `policy.begin_train_step` when it is true. Perform existing model/gradient mutations only after validation. Include the decision in abort cleanup and finish metrics.

Use one world-wide min/max consensus before zeroing gradients. This group includes every DP replica as well as TP/PP/CP ranks; a model-parallel-only group is insufficient. Encode every decision field, including the exact float64 observation bits and a stable reason enum, so the comparison is a full-decision comparison rather than a four-field prefix:

```python
def validate_draft_update_decision_consensus(
    decision: DraftUpdateDecision | None,
    *,
    draft_enabled: bool,
    group: torch.distributed.ProcessGroup | None = None,
    device: torch.device | None = None,
) -> DraftUpdateDecision | None:
    group = group or torch.distributed.group.WORLD
    reason_code = {
        "always": 1,
        "fixed_interval": 2,
        "adaptive_degradation": 3,
        "adaptive_burst": 4,
        "max_interval": 5,
        "none": 6,
    }.get(decision.reason, -1) if decision is not None else 0
    observation = (
        0.0
        if decision is None or decision.observed_acceptance is None
        else decision.observed_acceptance
    )
    observation_bits = torch.tensor(
        observation, dtype=torch.float64
    ).view(torch.int64).item()
    signature = torch.tensor(
        [
            int(draft_enabled),
            int(decision is not None),
            0 if decision is None else decision.global_step,
            0 if decision is None else decision.decision_id,
            0 if decision is None else int(decision.update_requested),
            0 if decision is None else int(decision.draft_refit_requested),
            reason_code,
            0 if decision is None else int(decision.forced),
            0 if decision is None else decision.applied_draft_version,
            int(decision is not None and decision.observed_acceptance is not None),
            observation_bits,
        ],
        dtype=torch.int64,
        device=device or torch.device("cuda", torch.cuda.current_device()),
    )
    minimum = signature.clone()
    maximum = signature.clone()
    torch.distributed.all_reduce(
        minimum,
        op=torch.distributed.ReduceOp.MIN,
        group=group,
    )
    torch.distributed.all_reduce(
        maximum,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )
    if minimum[0].item() != maximum[0].item():
        raise RuntimeError("draft-enabled mode mismatch across ranks")
    if maximum[0].item() == 1 and minimum[1].item() == 0:
        raise RuntimeError(
            "draft_update_decision is required but missing on at least one rank"
        )
    if minimum[6].item() == -1 or maximum[6].item() == -1:
        raise RuntimeError("unsupported draft decision reason across ranks")
    if not torch.equal(minimum, maximum):
        raise RuntimeError("draft update decision mismatch across ranks")
    return decision
```

Both worker entrypoints call only `validate_draft_update_decision_consensus` before mutation; there is no local required-decision helper that can raise before peers enter the collectives. The tensor carries a draft-enabled bit, a decision-presence bit, and neutral placeholders for a missing decision. Every world rank completes both MIN and MAX collectives in the same order. Only afterward do all ranks reject draft-enabled disagreement, any missing required decision, an invalid reason, or a full-field mismatch.

- [ ] **Step 4: Run the GREEN fanout and split-state tests.**

Run: `uv run --group test pytest -q tests/unit/models/policy/test_split_api_wrappers.py tests/unit/models/megatron/test_draft_step_state.py tests/unit/models/megatron/test_draft_cadence_entrypoints.py tests/unit/algorithms/test_grpo.py -k 'draft_update_decision or train_policy_from_meta or cp1_and_cp2' && uv run torchrun --standalone --nproc-per-node=2 -m pytest -q tests/unit/distributed/test_draft_cadence_consensus.py -k 'full_decision or missing_required' && uv run ruff check nemo_rl/data_plane/worker_mixin.py nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/tq_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/algorithms/grpo_sync.py tests/unit/distributed/test_draft_cadence_consensus.py`

Expected: selected tests PASS and Ruff reports no errors.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/data_plane/worker_mixin.py nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/tq_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/algorithms/grpo_sync.py tests/unit/models/policy/test_split_api_wrappers.py tests/unit/models/megatron/test_draft_step_state.py tests/unit/models/megatron/test_draft_cadence_entrypoints.py tests/unit/distributed/test_draft_cadence_consensus.py
git commit -S -s -m "feat(draft): broadcast controller update decisions"
git verify-commit HEAD
```

Expected: signature verification exits 0.

### Task 5: Make sparse skips compute-free and optimizer-safe

**Files:**
- Modify: `nemo_rl/models/megatron/draft/optimizer.py`
- Modify: `nemo_rl/data_plane/worker_mixin.py`
- Modify: `nemo_rl/models/policy/interfaces.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/tq_policy.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `nemo_rl/algorithms/single_controller.py`
- Create: `tests/unit/models/megatron/test_draft_optimizer_suspension.py`
- Modify: `tests/unit/models/megatron/test_draft_hidden_capture.py`
- Modify: `tests/unit/models/megatron/test_dflash_training_provider.py`
- Modify: `tests/unit/models/megatron/test_dspark_training_provider.py`
- Modify: `tests/unit/models/policy/test_split_api_wrappers.py`
- Modify: `tests/unit/distributed/test_draft_cadence_consensus.py`

**Interfaces:**
- Consumes: the monolithic-call or open-step `DraftUpdateDecision.update_requested`, explicit keyword-only `capture_draft_update_receipt: bool = False`, existing `param.grad_norm_group == "draft"` tags, and `DraftOptimizerConfigOwner.optimizer`, which may be `None`. Controllers pass `capture_draft_update_receipt=cadence_runtime.enabled`; no worker infers it from schedule mode.
- Produces: `build_draft_optimizer_override_provider(...)` always emits a draft-only group selector whenever draft training is enabled, using an empty inherited `ParamGroupOverride()` when `draft.optimizer=null`; `suspend_draft_optimizer_groups(optimizer: Any) -> Iterator[None]` then removes whole draft-only groups around `optimizer.step()`. Both monolithic `train` and split `begin_train_step`/`finish_train_step` APIs preserve the explicit capture flag end-to-end, pass the actual gate `enable_hidden_capture=run_draft` into `megatron_forward_backward`, omit draft model/provider on skips, and return the world-consensused boolean `draft_update_successful` before the controller may transfer weights or publish either version. Only when capture is enabled and a requested update succeeds, `canonical_draft_state_shards(...)` and `build_consensused_draft_update_receipt(...)` return one canonical `draft_update_receipt={successful,decision_id,global_step,draft_model_sha256,draft_optimizer_sha256}` through the elected owner and policy wrapper. Capture-disabled, skipped, and failed paths perform no receipt hashing or receipt-specific collective and never fabricate the key.

- [ ] **Step 1: Write RED optimizer-state and hidden-capture dispatch tests.**

```python
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from megatron.core.optimizer_param_scheduler import ParamGroupOverride

from nemo_rl.algorithms.draft_update_schedule import DraftUpdateDecision
from nemo_rl.models.megatron.draft.optimizer import (
    build_draft_optimizer_override_provider,
    suspend_draft_optimizer_groups,
)
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorker,
    draft_execution_inputs,
)


def fake_override_context():
    context = MagicMock()
    context.optimizer_config.min_lr = None
    return context


def test_skip_preserves_draft_parameter_moments_and_step() -> None:
    policy = torch.nn.Parameter(torch.tensor([1.0]))
    draft = torch.nn.Parameter(torch.tensor([1.0]))
    draft.grad_norm_group = "draft"
    optimizer = torch.optim.AdamW(
        [{"params": [policy]}, {"params": [draft]}], lr=0.1, weight_decay=0.1
    )
    policy.grad = torch.ones_like(policy)
    draft.grad = torch.ones_like(draft)
    optimizer.step()
    before_param = draft.detach().clone()
    before_state = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in optimizer.state[draft].items()
    }
    policy.grad = torch.ones_like(policy)
    draft.grad = torch.ones_like(draft)
    with suspend_draft_optimizer_groups(optimizer):
        optimizer.step()
    assert torch.equal(draft, before_param)
    for key, value in before_state.items():
        current = optimizer.state[draft][key]
        assert torch.equal(current, value) if isinstance(value, torch.Tensor) else current == value


def test_enabled_draft_with_null_optimizer_still_builds_draft_only_group() -> None:
    config = SimpleNamespace(enabled=True, optimizer=None)
    provider = build_draft_optimizer_override_provider(config)
    overrides = provider.build_config_overrides(fake_override_context())
    draft_keys = [key for key in overrides if key.predicate.name == "draft_parameter"]
    assert len(draft_keys) == 1
    assert overrides[draft_keys[0]] == ParamGroupOverride()


def test_mixed_policy_and_draft_group_fails_before_step() -> None:
    policy = torch.nn.Parameter(torch.tensor([1.0]))
    draft = torch.nn.Parameter(torch.tensor([1.0]))
    draft.grad_norm_group = "draft"
    optimizer = torch.optim.SGD([{"params": [policy, draft]}], lr=0.1)
    with pytest.raises(RuntimeError, match="mixes policy and draft"):
        with suspend_draft_optimizer_groups(optimizer):
            optimizer.step()


def test_skip_disables_actual_megatron_capture_inputs() -> None:
    draft_model = object()
    draft_provider = object()
    decision = DraftUpdateDecision(2, 2, False, False, "none", None)
    inputs = draft_execution_inputs(decision, draft_model, draft_provider)
    assert inputs == {
        "run_draft": False,
        "enable_hidden_capture": False,
        "draft_model": None,
        "draft_provider": None,
    }


def test_both_worker_entrypoints_use_shared_draft_execution_inputs() -> None:
    for entrypoint in (MegatronPolicyWorker.train, MegatronPolicyWorker.train_microbatch):
        assert "draft_execution_inputs" in entrypoint.__code__.co_names


def test_both_worker_entrypoints_expose_disabled_receipt_capture_default() -> None:
    for entrypoint in (
        MegatronPolicyWorker.train,
        MegatronPolicyWorker.begin_train_step,
    ):
        parameter = inspect.signature(entrypoint).parameters[
            "capture_draft_update_receipt"
        ]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is False


def test_monolithic_preserves_real_normalization_counts() -> None:
    real_seqs = torch.tensor(17)
    real_toks = torch.tensor(941)
    seqs, toks = normalization_counts_for_path(
        split_deferred=False,
        global_valid_seqs=real_seqs,
        global_valid_toks=real_toks,
        placeholder_n=torch.tensor(-1),
    )
    assert seqs is real_seqs
    assert toks is real_toks


def test_only_split_deferred_path_uses_normalization_placeholder() -> None:
    placeholder = torch.tensor(-1)
    seqs, toks = normalization_counts_for_path(
        split_deferred=True,
        global_valid_seqs=torch.tensor(17),
        global_valid_toks=torch.tensor(941),
        placeholder_n=placeholder,
    )
    assert seqs is placeholder
    assert toks is placeholder
```

Extend `tests/unit/distributed/test_draft_cadence_consensus.py` with the outcome import and the real second-DP failure case:

```python
from unittest.mock import MagicMock

from nemo_rl.models.policy.workers.megatron_policy_worker import (
    CanonicalDraftShard,
    build_consensused_draft_update_receipt,
    maybe_capture_draft_update_receipt,
    select_owner_draft_update_receipt,
    validate_draft_update_outcome_consensus,
)


def test_second_dp_owner_failure_returns_false_on_every_rank() -> None:
    _init_torchrun_group()
    rank = int(os.environ["RANK"])
    assert validate_draft_update_outcome_consensus(
        run_draft=True,
        local_owner=True,
        local_update_successful=rank == 0,
        group=dist.group.WORLD,
        device=torch.device("cpu"),
    ) is False


def test_requested_update_without_owner_raises_on_every_rank() -> None:
    _init_torchrun_group()
    with pytest.raises(RuntimeError, match="no draft owner exists"):
        validate_draft_update_outcome_consensus(
            run_draft=True,
            local_owner=False,
            local_update_successful=False,
            group=dist.group.WORLD,
            device=torch.device("cpu"),
        )


def test_enabled_receipt_digest_is_identical_and_has_one_owner() -> None:
    _init_torchrun_group()
    rank = int(os.environ["RANK"])
    consensus = build_consensused_draft_update_receipt(
        decision=DraftUpdateDecision(
            global_step=3,
            decision_id=7,
            update_requested=True,
            draft_refit_requested=True,
            reason="always",
            observed_acceptance=None,
        ),
        local_shards=(
            CanonicalDraftShard.for_test(
                component="model", logical_key="draft.weight",
                global_offset=(rank * 2,), global_shape=(4,),
                local_tensor=torch.tensor([rank + 1, rank + 2], dtype=torch.int32),
            ),
            CanonicalDraftShard.for_test(
                component="optimizer", logical_key="draft.weight/exp_avg",
                global_offset=(rank * 2,), global_shape=(4,),
                local_tensor=torch.tensor([rank + 3, rank + 4], dtype=torch.float32),
            ),
        ),
        local_owner=True,
        group=dist.group.WORLD,
        device=torch.device("cpu"),
    )
    assert consensus.owner_rank == 0
    assert len(consensus.receipt["draft_model_sha256"]) == 64
    assert len(consensus.receipt["draft_optimizer_sha256"]) == 64
    published = consensus.receipt_for_rank(rank)
    assert (published is not None) == (rank == 0)
    local_result = {
        "world_rank": rank,
        "draft_update_receipt_owner_rank": consensus.owner_rank,
    }
    if published is not None:
        local_result["draft_update_receipt"] = published
    gathered: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_result)
    assert all(isinstance(item, dict) for item in gathered)
    selected = select_owner_draft_update_receipt(
        tuple(item for item in gathered if isinstance(item, dict)),
        capture_draft_update_receipt=True,
        receipt_required=True,
    )
    assert selected == consensus.receipt


def test_disabled_receipt_capture_does_no_hashing_or_collective(monkeypatch) -> None:
    shard_factory = MagicMock()
    digest = MagicMock(side_effect=AssertionError("digest called"))
    gather = MagicMock(side_effect=AssertionError("collective called"))
    reduce = MagicMock(side_effect=AssertionError("collective called"))
    monkeypatch.setattr(
        "nemo_rl.models.policy.workers.megatron_policy_worker._digest_shard",
        digest,
    )
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    assert maybe_capture_draft_update_receipt(
        capture_draft_update_receipt=False,
        decision=DraftUpdateDecision(
            global_step=3,
            decision_id=7,
            update_requested=True,
            draft_refit_requested=True,
            reason="always",
            observed_acceptance=None,
        ),
        draft_update_successful=True,
        shard_factory=shard_factory,
        local_owner=True,
        group=dist.group.WORLD,
        device=torch.device("cpu"),
    ) is None
    shard_factory.assert_not_called()
    digest.assert_not_called()
    gather.assert_not_called()
    reduce.assert_not_called()
```

- [ ] **Step 2: Run the RED compute/optimizer tests and confirm draft state mutates.**

Run: `uv run --extra mcore --group test pytest -q --mcore-only tests/unit/models/megatron/test_draft_optimizer_suspension.py tests/unit/models/megatron/test_draft_hidden_capture.py tests/unit/models/policy/test_split_api_wrappers.py -k 'skip or null_optimizer or mixed_policy or receipt_capture' && uv run torchrun --standalone --nproc-per-node=2 -m pytest -q tests/unit/distributed/test_draft_cadence_consensus.py -k 'receipt_digest or disabled_receipt'`

Expected: FAIL because optimizer suspension and the explicit receipt-capture API/digest builder are missing.

- [ ] **Step 3: Add optimizer suspension and use the real capture gate.**

```python
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Iterator, Literal


def normalization_counts_for_path(
    *,
    split_deferred: bool,
    global_valid_seqs: Any,
    global_valid_toks: Any,
    placeholder_n: Any,
) -> tuple[Any, Any]:
    if split_deferred:
        return placeholder_n, placeholder_n
    return global_valid_seqs, global_valid_toks


def draft_execution_inputs(
    decision: DraftUpdateDecision,
    draft_model: Any,
    draft_provider: Any,
) -> dict[str, Any]:
    run_draft = decision.update_requested
    return {
        "run_draft": run_draft,
        "enable_hidden_capture": run_draft,
        "draft_model": draft_model if run_draft else None,
        "draft_provider": draft_provider if run_draft else None,
    }


@contextmanager
def suspend_draft_optimizer_groups(optimizer: Any) -> Iterator[None]:
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    saved: list[tuple[Any, list[dict[str, Any]]]] = []
    try:
        for current in optimizers:
            base = getattr(current, "optimizer", current)
            original = list(base.param_groups)
            kept: list[dict[str, Any]] = []
            for group in original:
                flags = {
                    getattr(parameter, "grad_norm_group", None) == "draft"
                    for parameter in group["params"]
                }
                if len(flags) != 1:
                    raise RuntimeError("optimizer parameter group mixes policy and draft parameters")
                if False in flags:
                    kept.append(group)
            saved.append((base, original))
            base.param_groups[:] = kept
        yield
    finally:
        for base, original in reversed(saved):
            base.param_groups[:] = original
```

Change `DraftOptimizerConfigOverrideProvider.draft_optimizer` to `DraftOptimizerConfig | None`. In `build_config_overrides`, always install the `ParamKey(predicate=ParamPredicate(name="draft_parameter", fn=_is_draft_parameter))`; use `ParamGroupOverride()` when the optional config is `None`, otherwise populate its LR/weight-decay overrides. Change `build_draft_optimizer_override_provider` to return `None` only when draft is absent or disabled, never merely because `draft_config.optimizer is None`. The MCore test constructs real groups for both configurations and asserts every group has either all draft tags or no draft tags.

In both entrypoints derive `run_draft = bool(decision and decision.update_requested)` and construct `LossPostProcessor` with `draft_model=None` and `draft_provider=None` on a skip. Normalization remains path-specific: the CP1 monolithic global-batch path passes its already-computed real `global_valid_seqs` and `global_valid_toks` unchanged. Only the CP>1 split deferred `_train_microbatch_body` path, which cannot know final global counts until `finish_train_step`, passes `placeholder_n` for both fields. Never overwrite monolithic counts with the split placeholder. Call:

```python
normalization_seqs, normalization_toks = normalization_counts_for_path(
    split_deferred=split_deferred,
    global_valid_seqs=global_valid_seqs,
    global_valid_toks=global_valid_toks,
    placeholder_n=placeholder_n,
)
losses_reduced = megatron_forward_backward(
    model=self.model,
    data_iterator=data_iterator,
    num_microbatches=num_microbatches,
    seq_length=padded_seq_length,
    mbs=micro_batch_size,
    post_processing_fn=loss_post_processor,
    forward_only=False,
    defer_fp32_logits=self.defer_fp32_logits,
    global_valid_seqs=normalization_seqs,
    global_valid_toks=normalization_toks,
    sampling_params=self.sampling_params,
    straggler_timer=self.mcore_state.straggler_timer,
    draft_model=self.draft_model if run_draft else None,
    draft_provider=getattr(self, "draft_provider", None) if run_draft else None,
    draft_optimizer_step=int(self.scheduler.num_steps),
    enable_hidden_capture=run_draft,
    use_fused_linear_logprobs=self.cfg["megatron_cfg"].get(
        "use_fused_linear_logprobs", False
    ),
    use_router_replay=use_router_replay,
    router_replay_train=True,
)
```

Wrap only `self.optimizer.step()` with `suspend_draft_optimizer_groups` when `run_draft` is false. Restore groups before `self.scheduler.step(increment=state["gbs"])` so LR advances on the global policy-step schedule.

Immediately after the optimizer's existing success consensus, derive and world-consensus the requested draft outcome in both entrypoints:

```python
def validate_draft_update_outcome_consensus(
    *,
    run_draft: bool,
    local_owner: bool,
    local_update_successful: bool,
    group: torch.distributed.ProcessGroup | None = None,
    device: torch.device | None = None,
) -> bool:
    group = group or torch.distributed.group.WORLD
    tensor_device = device or torch.device("cuda", torch.cuda.current_device())
    owner_present = torch.tensor(
        int(run_draft and local_owner), dtype=torch.int32, device=tensor_device
    )
    owner_failed = torch.tensor(
        int(run_draft and local_owner and not local_update_successful),
        dtype=torch.int32,
        device=tensor_device,
    )
    torch.distributed.all_reduce(
        owner_present,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )
    torch.distributed.all_reduce(
        owner_failed,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )
    if run_draft and owner_present.item() == 0:
        raise RuntimeError("draft update requested but no draft owner exists")
    return bool(run_draft and owner_failed.item() == 0)


@dataclass(frozen=True, slots=True)
class CanonicalDraftShard:
    component: Literal["model", "optimizer"]
    logical_key: str
    global_offset: tuple[int, ...]
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    leaf_sha256: str
    replica_id: tuple[int, ...]

    @classmethod
    def for_test(
        cls,
        *,
        component: Literal["model", "optimizer"],
        logical_key: str,
        global_offset: tuple[int, ...],
        global_shape: tuple[int, ...],
        local_tensor: torch.Tensor,
    ) -> "CanonicalDraftShard":
        local_shape, dtype, size_bytes, leaf_sha256 = _digest_shard(local_tensor)
        return cls(
            component=component,
            logical_key=logical_key,
            global_offset=global_offset,
            global_shape=global_shape,
            local_shape=local_shape,
            dtype=dtype,
            size_bytes=size_bytes,
            leaf_sha256=leaf_sha256,
            replica_id=(0,),
        )


def _digest_shard(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], str, int, str]:
    if sys.byteorder != "little":
        raise RuntimeError("canonical draft receipt requires little-endian workers")
    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    return (
        tuple(contiguous.shape),
        str(contiguous.dtype),
        len(raw),
        hashlib.sha256(raw).hexdigest(),
    )


def validate_canonical_draft_shard_cover(
    gathered: Sequence[object],
) -> list[dict[str, object]]:
    ranks: set[int] = set()
    records: list[dict[str, object]] = []
    for item in gathered:
        if (
            not isinstance(item, Mapping)
            or type(item.get("rank")) is not int
            or type(item.get("owner")) is not bool
            or not isinstance(item.get("shards"), list)
            or item["rank"] in ranks
        ):
            raise RuntimeError("invalid gathered draft shard envelope")
        ranks.add(item["rank"])
        for raw in item["shards"]:
            if not isinstance(raw, Mapping):
                raise RuntimeError("invalid canonical draft shard")
            record = dict(raw)
            component = record.get("component")
            offset = record.get("global_offset")
            global_shape = record.get("global_shape")
            local_shape = record.get("local_shape")
            digest = record.get("leaf_sha256")
            if (
                component not in {"model", "optimizer"}
                or not isinstance(record.get("logical_key"), str)
                or not isinstance(offset, (list, tuple))
                or not isinstance(global_shape, (list, tuple))
                or not isinstance(local_shape, (list, tuple))
                or not len(offset) == len(global_shape) == len(local_shape)
                or any(type(value) is not int for value in (*offset, *global_shape, *local_shape))
                or any(value < 0 for value in offset)
                or any(value <= 0 for value in (*global_shape, *local_shape))
                or any(
                    start + length > whole
                    for start, length, whole in zip(
                        offset, local_shape, global_shape, strict=True
                    )
                )
                or not isinstance(record.get("dtype"), str)
                or type(record.get("size_bytes")) is not int
                or record["size_bytes"] <= 0
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or tuple(record.get("replica_id", ())) != (0,)
            ):
                raise RuntimeError("invalid canonical draft shard")
            records.append(record)
    if set(range(len(gathered))) != ranks:
        raise RuntimeError("draft shard envelopes do not cover WORLD ranks")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for record in records:
        key = (
            record["component"], record["logical_key"],
            tuple(record["global_shape"]), record["dtype"],
        )
        groups.setdefault(key, []).append(record)
    if {key[0] for key in groups} != {"model", "optimizer"}:
        raise RuntimeError("draft receipt requires model and optimizer state")
    for key, shards in groups.items():
        seen: set[tuple[object, ...]] = set()
        volume = 0
        for index, left in enumerate(shards):
            identity = (
                tuple(left["global_offset"]), tuple(left["local_shape"])
            )
            if identity in seen:
                raise RuntimeError("duplicate canonical draft shard")
            seen.add(identity)
            volume += prod(left["local_shape"])
            for right in shards[index + 1:]:
                overlaps = all(
                    max(a, b) < min(a + alen, b + blen)
                    for a, alen, b, blen in zip(
                        left["global_offset"], left["local_shape"],
                        right["global_offset"], right["local_shape"], strict=True,
                    )
                )
                if overlaps:
                    raise RuntimeError("overlapping canonical draft shards")
        if volume != prod(key[2]):
            raise RuntimeError("gapped canonical draft shard coverage")
    return records


@dataclass(frozen=True, slots=True)
class DraftUpdateReceiptConsensus:
    owner_rank: int
    receipt: Mapping[str, object]

    def receipt_for_rank(self, rank: int) -> Mapping[str, object] | None:
        return self.receipt if rank == self.owner_rank else None


def _canonical_component_digest(
    component: str,
    shards: Sequence[Mapping[str, object]],
) -> str:
    records = sorted(
        (
            json.dumps(
                shard, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            for shard in shards
            if shard["component"] == component
        )
    )
    if not records:
        raise RuntimeError(f"draft {component} has no canonical shards")
    return hashlib.sha256(
        b"nemo-rl-draft-state-v1\0" + component.encode() + b"\0"
        + b"\n".join(records)
    ).hexdigest()


def build_consensused_draft_update_receipt(
    *,
    decision: DraftUpdateDecision,
    local_shards: Sequence[CanonicalDraftShard],
    local_owner: bool,
    group: torch.distributed.ProcessGroup,
    device: torch.device,
) -> DraftUpdateReceiptConsensus:
    rank = torch.distributed.get_rank(group)
    gathered: list[object] = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(
        gathered,
        {
            "rank": rank,
            "owner": local_owner,
            "shards": [asdict(shard) for shard in local_shards],
        },
        group=group,
    )
    entries = validate_canonical_draft_shard_cover(gathered)
    owner_ranks = sorted(
        int(item["rank"]) for item in gathered
        if isinstance(item, Mapping) and item.get("owner") is True
    )
    if not owner_ranks:
        raise RuntimeError("draft receipt capture has no shard owner")
    model_digest = _canonical_component_digest("model", entries)
    optimizer_digest = _canonical_component_digest("optimizer", entries)
    signature = torch.tensor(
        list(bytes.fromhex(model_digest + optimizer_digest)),
        dtype=torch.uint8,
        device=device,
    )
    minimum, maximum = signature.clone(), signature.clone()
    torch.distributed.all_reduce(
        minimum, op=torch.distributed.ReduceOp.MIN, group=group
    )
    torch.distributed.all_reduce(
        maximum, op=torch.distributed.ReduceOp.MAX, group=group
    )
    if not torch.equal(minimum, maximum):
        raise RuntimeError("draft update digest mismatch across ranks")
    return DraftUpdateReceiptConsensus(
        owner_rank=owner_ranks[0],
        receipt={
            "successful": True,
            "decision_id": decision.decision_id,
            "global_step": decision.global_step,
            "draft_model_sha256": model_digest,
            "draft_optimizer_sha256": optimizer_digest,
        },
    )


def maybe_capture_draft_update_receipt(
    *,
    capture_draft_update_receipt: bool,
    decision: DraftUpdateDecision,
    draft_update_successful: bool,
    shard_factory: Callable[[], Sequence[CanonicalDraftShard]],
    local_owner: bool,
    group: torch.distributed.ProcessGroup,
    device: torch.device,
) -> DraftUpdateReceiptConsensus | None:
    if (
        not capture_draft_update_receipt
        or not decision.update_requested
        or not draft_update_successful
    ):
        return None
    return build_consensused_draft_update_receipt(
        decision=decision,
        local_shards=shard_factory(),
        local_owner=local_owner,
        group=group,
        device=device,
    )


def select_owner_draft_update_receipt(
    worker_results: Sequence[Mapping[str, object]],
    *,
    capture_draft_update_receipt: bool,
    receipt_required: bool,
) -> Mapping[str, object] | None:
    receipt_rows = [
        result for result in worker_results
        if "draft_update_receipt" in result
    ]
    if not capture_draft_update_receipt:
        if receipt_rows:
            raise RuntimeError("disabled cadence runtime produced a draft receipt")
        return None
    if not receipt_required:
        if receipt_rows:
            raise RuntimeError("skipped or failed draft update produced a receipt")
        return None
    owner_ranks = {
        int(result["draft_update_receipt_owner_rank"])
        for result in worker_results
    }
    if len(owner_ranks) != 1 or len(receipt_rows) != 1:
        raise RuntimeError("draft update receipt must have exactly one owner")
    owner_rank = next(iter(owner_ranks))
    if receipt_rows[0].get("world_rank") != owner_rank:
        raise RuntimeError("draft update receipt came from a nonowner rank")
    receipt = receipt_rows[0]["draft_update_receipt"]
    if not isinstance(receipt, Mapping):
        raise RuntimeError("draft update receipt is not a mapping")
    return dict(receipt)


local_owner = bool(draft_step_state.active)
local_update_successful = bool(
    update_successful
    and draft_grad_norm is not None
    and math.isfinite(float(draft_grad_norm))
)
draft_update_successful = validate_draft_update_outcome_consensus(
    run_draft=run_draft,
    local_owner=local_owner,
    local_update_successful=local_update_successful,
)
metrics["draft_update_successful"] = draft_update_successful
```

`canonical_draft_state_shards` is called only by the lazy `shard_factory` above.
It consumes MCore's model and distributed-optimizer sharded state dictionaries,
keeps only canonical `replica_id == 0` records, and names every record by
component, fully-qualified parameter/state key, global offset/shape, local shape,
dtype, byte count, and SHA256 of contiguous CPU C-order tensor bytes. Optimizer
records include moments, per-parameter step, and rank-zero canonical JSON records
for draft-group scalar hyperparameters; model and optimizer use separate domain-
separated root digests. `validate_canonical_draft_shard_cover` validates the
MCore global offset/shape metadata and rejects a duplicate logical slice,
overlap, gap, unexpected replica, missing model component, or missing optimizer
component before sorting records. All WORLD ranks, including empty PP/TP/CP
nonowners, enter the single object gather and digest MIN/MAX collectives in the
same order. They derive the same roots and elect the minimum WORLD rank owning a
canonical draft shard; only that rank exposes `receipt_for_rank`, and the policy
wrapper requires exactly that one owner result before returning one top-level
receipt to the controller.

Thread keyword-only `capture_draft_update_receipt: bool = False` through
`PolicyInterface`, `LMPolicy`, `TQPolicy`, the worker mixin, monolithic `train`,
and split `begin_train_step`. Split begin stores it in the open-step state and
`finish_train_step` consumes it only after optimizer success and outcome
consensus. On a successful captured update, both worker finalizers report
`world_rank` and the common elected `draft_update_receipt_owner_rank`; only the
elected worker adds its receipt.
Both the monolithic `train` wrapper and split `finish_train_step` wrapper call
`select_owner_draft_update_receipt` and add its result to their top-level return
only when non-`None`. Sync and single-controller callers pass the already-resolved
`cadence_runtime.enabled`; default/legacy callers omit it. The monolithic and
split wrapper tests execute one successful requested update with capture enabled
and require the same owner-selected receipt schema, then repeat with capture
disabled and assert the return has no `draft_update_receipt`. The disabled test
also patches the shard factory, leaf hasher, object gather, and digest collectives
to prove none is touched. Existing outcome-consensus collectives are independent
of receipt capture and are not counted as receipt overhead.

The owner-presence MAX and owner-failure MAX always execute in the same order on every world rank. Only structural absence of every owner raises after both collectives. Any owner failure returns `False` on every rank, allowing the controller to record `update_attempted=True`, `update_successful=False`, `draft_refit_attempted=False`, and only then terminate before transfer/version publication. Absence on a non-owner PP rank is neutral. The executable two-rank tests model two DP owners, prove a second-owner failure returns `False` everywhere without deadlock, and prove a structurally missing owner raises everywhere. Skipped decisions also return `False`, so controllers interpret the value only when `decision.update_requested` is true.

- [ ] **Step 4: Run the GREEN compute, provider, and optimizer checks.**

Run: `uv run --extra mcore --group test pytest -q --mcore-only tests/unit/models/megatron/test_draft_optimizer_suspension.py tests/unit/models/megatron/test_draft_hidden_capture.py tests/unit/models/megatron/test_dflash_training_provider.py tests/unit/models/megatron/test_dspark_training_provider.py tests/unit/models/policy/test_split_api_wrappers.py -k 'skip or suspension or hidden_capture or null_optimizer or monolithic or split or receipt_capture' && uv run torchrun --standalone --nproc-per-node=2 -m pytest -q tests/unit/distributed/test_draft_cadence_consensus.py -k 'second_dp or requested_update_without_owner or receipt_digest or disabled_receipt' && uv run ruff check nemo_rl/models/megatron/draft/optimizer.py nemo_rl/data_plane/worker_mixin.py nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/tq_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/algorithms/grpo_sync.py nemo_rl/algorithms/single_controller.py tests/unit/models/megatron/test_draft_optimizer_suspension.py tests/unit/models/policy/test_split_api_wrappers.py tests/unit/distributed/test_draft_cadence_consensus.py`

Expected: selected tests PASS; the skip test proves unchanged draft bytes/moments/per-parameter step and an advanced scheduled LR, enabled receipt roots agree on every rank with one owner, and capture-disabled monolithic/split paths perform no receipt hashing or receipt-specific collective.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/models/megatron/draft/optimizer.py nemo_rl/data_plane/worker_mixin.py nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/tq_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/algorithms/grpo_sync.py nemo_rl/algorithms/single_controller.py tests/unit/models/megatron/test_draft_optimizer_suspension.py tests/unit/models/megatron/test_draft_hidden_capture.py tests/unit/models/megatron/test_dflash_training_provider.py tests/unit/models/megatron/test_dspark_training_provider.py tests/unit/models/policy/test_split_api_wrappers.py tests/unit/distributed/test_draft_cadence_consensus.py
git commit -S -s -m "feat(draft): skip drafter compute and optimizer state safely"
git verify-commit HEAD
```

Expected: signature verification exits 0 and `nemo_rl/models/megatron/train.py` remains unchanged because its existing `enable_hidden_capture` interface is the gate.

### Task 6: Add component-selective weight synchronization

**Files:**
- Modify: `nemo_rl/weight_sync/interfaces.py`
- Modify: `nemo_rl/weight_sync/factory.py`
- Modify: `nemo_rl/weight_sync/ipc_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/collective_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/http_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/megatron_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/vllm_remote_sparse_weight_synchronizer.py`
- Modify: `nemo_rl/models/policy/interfaces.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/models/generation/interfaces.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `nemo_rl/models/generation/vllm/speculator_runtime.py`
- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `tests/unit/weight_sync/test_weight_synchronizer.py`
- Modify: `tests/unit/weight_sync/test_factory.py`
- Modify: `tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py`
- Modify: `tests/unit/models/generation/test_vllm_speculator_runtime.py`

**Interfaces:**
- Consumes: `WeightSyncSelection(target=True, draft=decision.draft_refit_requested)` plus the resolved schedule mode, generation backend, colocation flag, refit transport, remote-sparse flag, `cadence_runtime.enabled`, and controller-declared selected-rollout science capabilities.
- Produces: pure `preflight_component_selection(*, schedule_mode: str, generation_backend: str, colocated: bool, refit_transport: str | None, remote_sparse: bool) -> None`, `preflight_cadence_science(*, enabled: bool, capabilities: CadenceScienceCapabilities) -> None`, `WeightSynchronizer.supports_component_selection: bool`, `sync_weights(*, selection: WeightSyncSelection = WeightSyncSelection(), timer: Optional[Timer] = None, kv_scales: Optional[dict[str, float]] = None) -> Mapping[str, object]`, and defensive `require_component_selection(synchronizer: WeightSynchronizer, schedule_mode: str) -> None`; target is always selected. The return mapping carries `successful: bool`, string snapshot provenance, numeric timing values, and an optional nested `draft_apply_receipt`; it is never typed as a float-only dictionary.

- [ ] **Step 1: Write RED capability, target-only transfer, and remote-sparse rejection tests.**

```python
import inspect
from unittest.mock import patch

import pytest

from nemo_rl.algorithms import grpo as grpo_module
from nemo_rl.algorithms.single_controller_utils import setup as sc_setup
from nemo_rl.weight_sync.ipc_weight_synchronizer import IPCWeightSynchronizer
from nemo_rl.weight_sync.vllm_remote_sparse_weight_synchronizer import (
    VllmRemoteSparseWeightSynchronizer,
)
from nemo_rl.weight_sync.interfaces import (
    WeightSyncSelection,
    preflight_component_selection,
    require_component_selection,
)
from nemo_rl.algorithms.draft_cadence_runtime import (
    CadenceScienceCapabilities,
    preflight_cadence_science,
)


def test_selection_rejects_target_false() -> None:
    with pytest.raises(ValueError, match="target policy"):
        WeightSyncSelection(target=False, draft=True)


@patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
def test_target_only_sync_omits_draft_payload(mock_ray) -> None:
    mock_ray.get.return_value = [True]
    policy = _mock_policy()
    generation = _mock_generation()
    synchronizer = IPCWeightSynchronizer(policy, generation)
    synchronizer.sync_weights(selection=WeightSyncSelection(draft=False))
    selection = policy.stream_weights_via_ipc_zmq.call_args.kwargs["selection"]
    assert selection == WeightSyncSelection(target=True, draft=False)
    generation.finalize_draft_update.assert_not_called()


def test_remote_sparse_fixed_cadence_fails_at_startup() -> None:
    remote_sparse_synchronizer = object.__new__(VllmRemoteSparseWeightSynchronizer)
    with pytest.raises(ValueError, match="component-selective.*unsupported"):
        require_component_selection(remote_sparse_synchronizer, "fixed")


@pytest.mark.parametrize(
    ("generation_backend", "colocated", "refit_transport", "remote_sparse"),
    [
        ("sglang", True, None, False),
        ("megatron", True, None, False),
        ("vllm", False, "checkpoint_engine", False),
        ("vllm", False, "nccl_reshard", False),
        ("vllm", True, None, True),
    ],
)
def test_unsupported_transport_fails_before_worker_construction(
    generation_backend,
    colocated,
    refit_transport,
    remote_sparse,
) -> None:
    with pytest.raises(ValueError, match="component-selective.*unsupported"):
        preflight_component_selection(
            schedule_mode="fixed",
            generation_backend=generation_backend,
            colocated=colocated,
            refit_transport=refit_transport,
            remote_sparse=remote_sparse,
        )


def test_single_controller_calls_preflight_before_actor_creation() -> None:
    source = inspect.getsource(sc_setup.setup_single_controller)
    assert source.index("preflight_component_selection(") < source.index(
        "create_policy_cluster("
    )
    assert source.index("preflight_cadence_science(") < source.index(
        "create_policy_cluster("
    )


def test_sync_calls_science_preflight_before_cluster_construction() -> None:
    source = inspect.getsource(grpo_module.setup)
    assert source.index("preflight_cadence_science(") < source.index(
        "RayVirtualCluster("
    )


def test_default_runtime_does_not_require_science_capabilities() -> None:
    preflight_cadence_science(
        enabled=False,
        capabilities=CadenceScienceCapabilities(False, False, False),
    )


def test_legacy_always_keeps_preexisting_transport_eligibility() -> None:
    preflight_component_selection(
        schedule_mode="always",
        generation_backend="sglang",
        colocated=False,
        refit_transport="checkpoint_engine",
        remote_sparse=True,
    )


@pytest.mark.parametrize(
    "capabilities",
    [
        CadenceScienceCapabilities(False, True, True),
        CadenceScienceCapabilities(True, False, True),
        CadenceScienceCapabilities(True, True, False),
    ],
)
def test_experiment_runtime_fails_preflight_on_missing_science(capabilities) -> None:
    with pytest.raises(ValueError, match="cadence runtime science.*unavailable"):
        preflight_cadence_science(enabled=True, capabilities=capabilities)
```

- [ ] **Step 2: Run the RED weight-sync tests and confirm the selection API is missing.**

Run: `uv run --group test pytest -q tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/models/generation/test_vllm_speculator_runtime.py -k 'selection or cadence'`

Expected: FAIL during collection with `ImportError: cannot import name 'WeightSyncSelection'`.

- [ ] **Step 3: Add the capability contract and validate it in the factory/startup path.**

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightSyncSelection:
    target: bool = True
    draft: bool = True

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("target policy must synchronize on every policy step")


@dataclass(frozen=True, slots=True)
class CadenceScienceCapabilities:
    selected_acceptance_counts: bool
    selected_serving_version: bool
    canonical_metric_logging: bool


def preflight_cadence_science(
    *, enabled: bool, capabilities: CadenceScienceCapabilities
) -> None:
    if not enabled:
        return
    if not all((
        capabilities.selected_acceptance_counts,
        capabilities.selected_serving_version,
        capabilities.canonical_metric_logging,
    )):
        raise ValueError(
            "cadence runtime science is unavailable: selected acceptance counts, "
            "serving-version provenance, and canonical logging are required"
        )


def require_component_selection(
    synchronizer: WeightSynchronizer,
    schedule_mode: str,
) -> None:
    if schedule_mode != "always" and not synchronizer.supports_component_selection:
        raise ValueError(
            f"component-selective draft refit is unsupported by "
            f"{type(synchronizer).__name__}; use update_schedule.mode=always"
        )


def preflight_component_selection(
    *,
    schedule_mode: str,
    generation_backend: str,
    colocated: bool,
    refit_transport: str | None,
    remote_sparse: bool,
) -> None:
    if schedule_mode == "always":
        return
    supported = (
        generation_backend == "vllm"
        and not remote_sparse
        and refit_transport not in {"checkpoint_engine", "nccl_reshard"}
        and (colocated or refit_transport is None)
    )
    if not supported:
        raise ValueError(
            "component-selective draft refit is unsupported by the resolved "
            f"transport: backend={generation_backend!r}, colocated={colocated}, "
            f"refit_transport={refit_transport!r}, remote_sparse={remote_sparse}"
        )
```

Call both pure preflights from `setup_single_controller` and synchronous
`grpo.setup` immediately after resolved-config validation and before
`create_policy_cluster`, `create_generation_cluster`, Ray actor creation, or
communicator construction. Science capabilities are derived from registered
canonical metric/tag producers, not user booleans. With runtime instrumentation
disabled the science preflight returns immediately and does not narrow legacy
`always` transports. Component-selection validation remains independently driven
by schedule semantics. Then extend `create_weight_synchronizer` with
`draft_update_schedule_mode: str = "always"`, route every constructed synchronizer
through the instance validator, and use the same instance validator in
`nemo_rl/algorithms/grpo.py` after its separate remote-sparse construction:

```python
def build_checked_ipc(
    policy: Any,
    generation: Any,
    refit_buffer_size_gb: float | int | None,
    draft_update_schedule_mode: str,
) -> WeightSynchronizer:
    def checked(synchronizer: WeightSynchronizer) -> WeightSynchronizer:
        require_component_selection(synchronizer, draft_update_schedule_mode)
        return synchronizer

    return checked(
        IPCWeightSynchronizer(
            policy=policy,
            generation=generation,
            refit_buffer_size_gb=refit_buffer_size_gb,
        )
    )
```

For the remote-sparse branch in `nemo_rl/algorithms/grpo.py`, validate the constructed object before `init_communicator()`:

```python
assert policy_generation.weight_synchronizer is not None
require_component_selection(
    policy_generation.weight_synchronizer,
    master_config.policy["draft"].update_schedule.mode,
)
policy_generation.weight_synchronizer.init_communicator()
```

Add the abstract property and keyword to `WeightSynchronizer`:

```python
@property
@abstractmethod
def supports_component_selection(self) -> bool:
    raise NotImplementedError

@abstractmethod
def sync_weights(
    self,
    *,
    selection: WeightSyncSelection = WeightSyncSelection(),
    timer: Optional[Timer] = None,
    kv_scales: Optional[dict[str, float]] = None,
) -> Mapping[str, object]:
    raise NotImplementedError
```

IPC and collective return `True` only after their iterators, manifests, transfer byte counts, vLLM apply coverage, and draft finalizer obey `selection.draft`. HTTP, checkpoint-engine, Megatron, NCCL-reshard, and remote-sparse return `False` and raise if called with `draft=False`. Keep `require_component_selection` immediately after `create_weight_synchronizer` and before `init_communicator` as a defense against factory/preflight drift; the earlier pure check is the one that guarantees rejection before worker creation.

Refactor `refit_policy_generation` to accept `selection: WeightSyncSelection`; every post-policy call uses `target=True`, so `POLICY_GENERATION_STALE` is cleared every step even when draft transfer is skipped.

- [ ] **Step 4: Run the GREEN transport/factory and generation coverage tests.**

Run: `uv run --group test pytest -q tests/unit/weight_sync/test_factory.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/models/generation/test_vllm_speculator_runtime.py tests/unit/models/generation/test_vllm_backend.py tests/unit/single_controller/test_single_controller_setup.py -k 'selection or target_only or cadence or before_worker' && uv run ruff check nemo_rl/weight_sync nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/generation/vllm/speculator_runtime.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/single_controller_utils/setup.py`

Expected: tests PASS; target-only transfers report zero draft bytes; remote-sparse fixed/adaptive startup fails before communicator initialization.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/weight_sync nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/generation/vllm/speculator_runtime.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/single_controller_utils/setup.py tests/unit/weight_sync/test_factory.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/models/generation/test_vllm_speculator_runtime.py tests/unit/single_controller/test_single_controller_setup.py
git commit -S -s -m "feat(draft): select draft payload during refit"
git verify-commit HEAD
```

Expected: signature verification exits 0.

### Task 7: Reconstruct count-weighted acceptance and schedule synchronous GRPO

**Files:**
- Create: `nemo_rl/algorithms/draft_update_observation.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `tests/unit/algorithms/test_grpo.py`
- Create: `tests/unit/algorithms/test_draft_update_observation.py`
- Create: `tests/unit/algorithms/test_grpo_sync_draft_schedule.py`

**Interfaces:**
- Consumes: canonical `vllm/spec_num_accepted_tokens` and `vllm/spec_num_draft_tokens` from every generation/dynamic-sampling batch and, only with experiment instrumentation, `draft_schedule/applied_draft_version` from those same selected batches; Tasks 2, 4, and 6 APIs.
- Produces: `stamp_selected_rollout_science(metrics, *, enabled, applied_draft_version) -> Mapping[str, object]`, `acceptance_from_rollout_metric_batches(batches) -> float | None`, `rollout_science_from_metric_batches(batches, require_version) -> tuple[float | None, int | None]`, `acceptance_observation_for_schedule(config, acceptance) -> float | None`, `prepare_sync_draft_decision(..., cadence_runtime_enabled: bool) -> PreparedDraftDecision`, transaction-bound `apply_scheduled_refit(...) -> WeightSyncSelection`, one scheduler decision/outcome/ledger row per policy step, target sync every step, optional draft payload sync, and numeric schedule metrics. Adaptive always reconstructs acceptance for scheduling. Fixed/`always` reconstruct it only when experiment instrumentation is enabled, and still pass `None` into the state machine. Experiment mode additionally rejects missing, nonintegral, mixed, or stale selected serving-version tags before opening the step transaction.

- [ ] **Step 1: Write RED count-weighting and controller-order tests.**

```python
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.draft_update_schedule import (
    DraftDecisionLedger,
    DraftUpdateScheduler,
    FileDraftStepTransactionStore,
)
from nemo_rl.algorithms.draft_cadence_runtime import (
    CadenceRuntimeConfig,
    CadenceRuntimeWriter,
    CadenceTerminalEvidence,
)
from nemo_rl.algorithms.draft_update_observation import (
    VERSION_KEY,
    acceptance_observation_for_schedule,
    acceptance_from_rollout_metric_batches,
    prepare_sync_draft_decision,
    rollout_science_from_metric_batches,
    stamp_selected_rollout_science,
)
from nemo_rl.algorithms.grpo_sync import apply_scheduled_refit
from nemo_rl.models.policy.draft_config import AdaptiveDraftUpdateScheduleConfig
from nemo_rl.models.policy.draft_config import FixedDraftUpdateScheduleConfig
from nemo_rl.models.policy.draft_config import AlwaysDraftUpdateScheduleConfig
from nemo_rl.weight_sync.interfaces import WeightSyncSelection


class SyncHarness:
    def __init__(self) -> None:
        config = FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=2
        )
        self.scheduler = DraftUpdateScheduler.create(config, origin_step=0)
        self.sync_selections = []
        self.training_decisions = []
        self.worker_result = {"draft_update_successful": True}
        self.target_weight_version = 0
        self.applied_draft_version = 0
        self.root = Path(tempfile.mkdtemp())
        self.transaction_store = FileDraftStepTransactionStore(
            self.root / "transactions"
        )
        self.decision_ledger = DraftDecisionLedger(self.root / "ledger.jsonl")
        self.grpo_save_state = SimpleNamespace(
            draft_update_schedule=None,
            applied_draft_snapshot={
                "version": 0, "path": "initial", "size_bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
        )

    def run_one_step(self, fixed_sparse_interval: int = 2) -> None:
        assert fixed_sparse_interval == self.scheduler.config.fixed_interval
        step = self.scheduler.state.last_decided_step + 1
        decision = self.scheduler.decide(global_step=step, acceptance=None)
        transaction = self.transaction_store.begin(decision)
        self.training_decisions.append(decision)

        def sync_weights(*, selection):
            self.sync_selections.append(selection)
            receipt = {"successful": True}
            if selection.draft:
                path = self.root / f"draft-v{decision.decision_id}.bin"
                raw = f"draft-{decision.decision_id}".encode()
                path.write_bytes(raw)
                receipt["draft_apply_receipt"] = {
                    "successful": True,
                    "version": decision.decision_id,
                    "snapshot_path": str(path.resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            return receipt

        apply_scheduled_refit(
            decision,
            self.worker_result,
            self.scheduler,
            transaction=transaction,
            decision_ledger=self.decision_ledger,
            grpo_save_state=self.grpo_save_state,
            transaction_store=self.transaction_store,
            runtime_writer=None,
            terminal_evidence=None,
            sync_weights=sync_weights,
            publish_target_version=lambda: setattr(
                self, "target_weight_version", self.target_weight_version + 1
            ),
            publish_draft_version=lambda version: setattr(
                self, "applied_draft_version", version
            ),
        )

    def run_two_steps(self, fixed_sparse_interval: int) -> None:
        self.run_one_step(fixed_sparse_interval)
        self.run_one_step(fixed_sparse_interval)


def test_acceptance_sums_counts_instead_of_averaging_rates() -> None:
    batches = [
        {"vllm/spec_num_accepted_tokens": 9.0, "vllm/spec_num_draft_tokens": 10.0},
        {"vllm/spec_num_accepted_tokens": 1.0, "vllm/spec_num_draft_tokens": 90.0},
    ]
    assert acceptance_from_rollout_metric_batches(batches) == pytest.approx(0.1)


def test_sync_science_stamp_is_opt_in_and_binds_reserved_version() -> None:
    metrics = {
        "vllm/spec_num_accepted_tokens": 9.0,
        "vllm/spec_num_draft_tokens": 10.0,
    }
    assert stamp_selected_rollout_science(
        metrics, enabled=False, applied_draft_version=4
    ) is metrics
    stamped = stamp_selected_rollout_science(
        metrics, enabled=True, applied_draft_version=4
    )
    assert VERSION_KEY not in metrics
    assert stamped[VERSION_KEY] == 4


@pytest.mark.parametrize(
    "batches",
    [
        [],
        [{"vllm/spec_num_accepted_tokens": 1.0}],
        [{"vllm/spec_num_accepted_tokens": -1.0, "vllm/spec_num_draft_tokens": 2.0}],
        [{"vllm/spec_num_accepted_tokens": 0.0, "vllm/spec_num_draft_tokens": 0.0}],
    ],
)
def test_invalid_acceptance_counts_return_none(batches) -> None:
    assert acceptance_from_rollout_metric_batches(batches) is None


@pytest.mark.parametrize(
    "config",
    [
        AlwaysDraftUpdateScheduleConfig(),
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
    ],
)
def test_default_nonadaptive_sync_does_not_read_science_metrics(config) -> None:
    unreadable_batches = MagicMock()
    prepared = prepare_sync_draft_decision(
        DraftUpdateScheduler.create(config, origin_step=0),
        unreadable_batches,
        cadence_runtime_enabled=False,
        evidence=None,
        global_step=1,
    )
    assert prepared.decision.observed_acceptance is None
    assert prepared.terminal_evidence is None
    unreadable_batches.__iter__.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        AlwaysDraftUpdateScheduleConfig(),
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
    ],
)
def test_experiment_sync_nonadaptive_collects_science_without_scheduler_input(
    config,
) -> None:
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    evidence = CadenceTerminalEvidence({}, {})
    first = prepare_sync_draft_decision(
        scheduler,
        [{"vllm/spec_num_accepted_tokens": 8.0,
          "vllm/spec_num_draft_tokens": 10.0,
          "draft_schedule/applied_draft_version": 0}],
        cadence_runtime_enabled=True,
        evidence=evidence,
        global_step=1,
    )
    scheduler.record_outcome(
        first.decision,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    assert first.terminal_evidence is not None
    second = prepare_sync_draft_decision(
        scheduler,
        [{"vllm/spec_num_accepted_tokens": 6.0,
          "vllm/spec_num_draft_tokens": 10.0,
          "draft_schedule/applied_draft_version": 1}],
        cadence_runtime_enabled=True,
        evidence=first.terminal_evidence,
        global_step=2,
    )
    assert second.decision.observed_acceptance is None
    assert second.terminal_evidence is not None
    assert second.terminal_evidence.observations_by_refit_step[1] == {
        "refit_step": 1,
        "observation_step": 2,
        "applied_draft_version": 1,
        "acceptance_rate": pytest.approx(0.6),
    }


@pytest.mark.parametrize(
    "batches",
    [
        [{"vllm/spec_num_accepted_tokens": 6.0,
          "vllm/spec_num_draft_tokens": 10.0}],
        [{"vllm/spec_num_accepted_tokens": 6.0,
          "vllm/spec_num_draft_tokens": 10.0,
          "draft_schedule/applied_draft_version": 0.5}],
        [{"vllm/spec_num_accepted_tokens": 3.0,
          "vllm/spec_num_draft_tokens": 5.0,
          "draft_schedule/applied_draft_version": 0},
         {"vllm/spec_num_accepted_tokens": 3.0,
          "vllm/spec_num_draft_tokens": 5.0,
          "draft_schedule/applied_draft_version": 1}],
    ],
)
def test_experiment_sync_rejects_absent_nonintegral_or_mixed_versions(
    batches,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    with pytest.raises(ValueError, match="selected serving version"):
        prepare_sync_draft_decision(
            scheduler,
            batches,
            cadence_runtime_enabled=True,
            evidence=CadenceTerminalEvidence({}, {}),
            global_step=1,
        )
    assert scheduler.state.next_decision_id == 1


def test_experiment_sync_rejects_stale_selected_version_before_decision() -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    with pytest.raises(RuntimeError, match="stale selected rollout"):
        prepare_sync_draft_decision(
            scheduler,
            [{"vllm/spec_num_accepted_tokens": 6.0,
              "vllm/spec_num_draft_tokens": 10.0,
              "draft_schedule/applied_draft_version": 7}],
            cadence_runtime_enabled=True,
            evidence=CadenceTerminalEvidence({}, {}),
            global_step=1,
        )
    assert scheduler.state.next_decision_id == 1


def test_sync_adaptive_feeds_same_science_observation_to_scheduler() -> None:
    scheduler = DraftUpdateScheduler.create(
        AdaptiveDraftUpdateScheduleConfig(
            min_interval=1, max_interval=10, min_observations=1
        ),
        origin_step=0,
    )
    prepared = prepare_sync_draft_decision(
        scheduler,
        [{"vllm/spec_num_accepted_tokens": 6.0,
          "vllm/spec_num_draft_tokens": 10.0}],
        cadence_runtime_enabled=False,
        evidence=None,
        global_step=1,
    )
    assert prepared.decision.observed_acceptance == pytest.approx(0.6)
    assert prepared.terminal_evidence is None


def test_sync_controller_refits_target_every_step() -> None:
    harness = SyncHarness()
    harness.run_two_steps(fixed_sparse_interval=2)
    assert harness.sync_selections == [
        WeightSyncSelection(target=True, draft=False),
        WeightSyncSelection(target=True, draft=True),
    ]
    assert harness.training_decisions[0].update_requested is False
    assert harness.training_decisions[1].update_requested is True


def test_failed_draft_update_stops_before_transfer_or_version_publish(
) -> None:
    harness = SyncHarness()
    harness.scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
        origin_step=0,
    )
    harness.worker_result["draft_update_successful"] = False
    with pytest.raises(RuntimeError, match="draft update failed across workers"):
        harness.run_one_step(fixed_sparse_interval=1)
    assert harness.sync_selections == []
    assert harness.target_weight_version == 0
    assert harness.applied_draft_version == 0
    assert harness.scheduler.state.attempted_updates == 1
    assert harness.scheduler.state.successful_updates == 0
    assert harness.scheduler.state.failed_updates == 1
    assert harness.scheduler.state.attempted_refits == 0
    assert harness.scheduler.state.failed_refits == 0
    assert harness.scheduler.state.skipped_refits == 1


def test_sync_update_receipt_is_durable_before_transfer_and_publication(
    tmp_path: Path,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    store = FileDraftStepTransactionStore(tmp_path / "transactions")
    transaction = store.begin(decision)
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    evidence = CadenceTerminalEvidence({}, {})
    save_state = SimpleNamespace(
        draft_terminal_evidence=None,
        draft_update_schedule=None,
        applied_draft_snapshot={"version": 0},
    )
    writer = CadenceRuntimeWriter(CadenceRuntimeConfig(
        enabled=True, result_dir=str(tmp_path / "runtime")
    ))
    snapshot_path = tmp_path / "draft-v1.bin"
    snapshot_path.write_bytes(b"draft-v1")
    events = []

    def sync_weights(*, selection):
        assert 1 in evidence.update_receipts_by_decision
        assert save_state.draft_terminal_evidence == evidence.state_dict()
        events.append("transfer")
        return {
            "successful": True,
            "draft_apply_receipt": {
                "successful": True,
                "version": 1,
                "snapshot_path": str(snapshot_path),
                "sha256": hashlib.sha256(b"draft-v1").hexdigest(),
            },
        }

    apply_scheduled_refit(
        decision,
        {
            "draft_update_successful": True,
            "draft_update_receipt": {
                "successful": True,
                "decision_id": 1,
                "global_step": 1,
                "draft_model_sha256": "a" * 64,
                "draft_optimizer_sha256": "b" * 64,
            },
        },
        scheduler,
        transaction=transaction,
        decision_ledger=ledger,
        grpo_save_state=save_state,
        transaction_store=store,
        runtime_writer=writer,
        terminal_evidence=evidence,
        sync_weights=sync_weights,
        publish_target_version=lambda: events.append("publish-target"),
        publish_draft_version=lambda _version: events.append("publish-draft"),
    )
    assert events == ["transfer", "publish-target", "publish-draft"]
```

- [ ] **Step 2: Run the RED observation/controller tests and confirm the helper is absent.**

Run: `uv run --group test pytest -q tests/unit/algorithms/test_draft_update_observation.py tests/unit/algorithms/test_grpo_sync_draft_schedule.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'nemo_rl.algorithms.draft_update_observation'`.

- [ ] **Step 3: Add count validation and the single synchronous decision flow.**

```python
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

ACCEPTED_KEY = "vllm/spec_num_accepted_tokens"
DRAFT_KEY = "vllm/spec_num_draft_tokens"
VERSION_KEY = "draft_schedule/applied_draft_version"


def stamp_selected_rollout_science(
    metrics: Mapping[str, object],
    *,
    enabled: bool,
    applied_draft_version: int,
) -> Mapping[str, object]:
    if not enabled:
        return metrics
    if type(applied_draft_version) is not int or applied_draft_version < 0:
        raise ValueError("selected serving version must be a nonnegative integer")
    return {**metrics, VERSION_KEY: applied_draft_version}


def acceptance_from_rollout_metric_batches(
    batches: Iterable[Mapping[str, object]],
) -> float | None:
    accepted_total = 0.0
    draft_total = 0.0
    seen = False
    for metrics in batches:
        if ACCEPTED_KEY not in metrics or DRAFT_KEY not in metrics:
            return None
        accepted = float(metrics[ACCEPTED_KEY])
        draft = float(metrics[DRAFT_KEY])
        if not math.isfinite(accepted) or not math.isfinite(draft):
            return None
        if accepted < 0.0 or draft < 0.0 or accepted > draft:
            return None
        accepted_total += accepted
        draft_total += draft
        seen = True
    if not seen or draft_total <= 0.0:
        return None
    return accepted_total / draft_total


def rollout_science_from_metric_batches(
    batches: Iterable[Mapping[str, object]],
    *,
    require_version: bool,
) -> tuple[float | None, int | None]:
    materialized = tuple(batches)
    acceptance = acceptance_from_rollout_metric_batches(materialized)
    if not require_version:
        return acceptance, None
    versions: set[int] = set()
    for metrics in materialized:
        value = metrics.get(VERSION_KEY)
        if type(value) is not int or value < 0:
            raise ValueError("selected serving version is absent or nonintegral")
        versions.add(value)
    if len(versions) != 1:
        raise ValueError("selected serving version is mixed across rollout batches")
    return acceptance, next(iter(versions))


def acceptance_observation_for_schedule(
    config: DraftUpdateScheduleConfig,
    acceptance: float | None,
) -> float | None:
    return acceptance if config.mode == "adaptive" else None


@dataclass(frozen=True, slots=True)
class PreparedDraftDecision:
    decision: DraftUpdateDecision
    terminal_evidence: CadenceTerminalEvidence | None


def prepare_sync_draft_decision(
    scheduler: DraftUpdateScheduler,
    rollout_metric_batches: Iterable[Mapping[str, object]],
    *,
    cadence_runtime_enabled: bool,
    evidence: CadenceTerminalEvidence | None,
    global_step: int,
) -> PreparedDraftDecision:
    needs_acceptance = (
        scheduler.config.mode == "adaptive" or cadence_runtime_enabled
    )
    if cadence_runtime_enabled != (evidence is not None):
        raise ValueError("cadence runtime evidence enablement mismatch")
    if needs_acceptance:
        science_acceptance, selected_version = rollout_science_from_metric_batches(
            rollout_metric_batches,
            require_version=cadence_runtime_enabled,
        )
    else:
        science_acceptance, selected_version = None, None
    if (
        cadence_runtime_enabled
        and selected_version != scheduler.state.applied_draft_version
    ):
        raise RuntimeError(
            "stale selected rollout: "
            f"selected={selected_version}, "
            f"current={scheduler.state.applied_draft_version}"
        )
    prior_refit_step = scheduler.state.last_applied_refit_step
    decision = scheduler.decide(
        global_step=global_step,
        acceptance=acceptance_observation_for_schedule(
            scheduler.config, science_acceptance
        ),
    )
    if cadence_runtime_enabled:
        assert evidence is not None
        evidence = record_terminal_post_refit_observation(
            evidence, decision=decision,
            last_applied_refit_step=prior_refit_step,
            acceptance_rate=science_acceptance,
        )
    return PreparedDraftDecision(
        decision=decision, terminal_evidence=evidence
    )
```

Add the shared sync finalizer used by the synchronous controller and its tests:

```python
def apply_scheduled_refit(
    decision: DraftUpdateDecision,
    train_results: Mapping[str, object],
    scheduler: DraftUpdateScheduler,
    *,
    transaction: DraftStepTransaction,
    decision_ledger: DraftDecisionLedger,
    grpo_save_state: GRPOSaveState,
    transaction_store: DraftStepTransactionStore,
    runtime_writer: CadenceRuntimeWriter | None,
    terminal_evidence: CadenceTerminalEvidence | None,
    sync_weights: Callable[..., Mapping[str, object]],
    publish_target_version: Callable[[], None],
    publish_draft_version: Callable[[int], None],
) -> WeightSyncSelection:
    update_ok = (
        not decision.update_requested
        or train_results.get("draft_update_successful") is True
    )
    if not update_ok:
        outcome = decision_outcome_payload(
            decision,
            update_attempted=decision.update_requested,
            update_successful=False,
            draft_refit_attempted=False,
            draft_refit_successful=False,
        )
        close_draft_step_transaction(
            transaction, decision=decision, outcome=outcome,
            applied_snapshot=None, scheduler=scheduler,
            decision_ledger=decision_ledger, grpo_save_state=grpo_save_state,
            transaction_store=transaction_store,
        )
        raise RuntimeError("draft update failed across workers before weight transfer")
    selection = WeightSyncSelection(
        target=True,
        draft=decision.draft_refit_requested,
    )
    try:
        if runtime_writer is not None:
            if terminal_evidence is None:
                raise RuntimeError("cadence runtime writer requires terminal evidence")
            if decision.update_requested:
                update_receipt = train_results.get("draft_update_receipt")
                if not isinstance(update_receipt, Mapping):
                    raise RuntimeError("successful draft update lacks worker receipt")
                runtime_writer.successful_update_closed(
                    decision=decision,
                    worker_receipt=update_receipt,
                    evidence=terminal_evidence,
                    save_state=grpo_save_state,
                )
        elif terminal_evidence is not None:
            raise RuntimeError("terminal evidence is enabled without runtime writer")
        sync_receipt = sync_weights(selection=selection)
        if sync_receipt.get("successful") is not True:
            raise RuntimeError("target weight transfer receipt failed")
        applied_snapshot = None
        if selection.draft:
            raw_receipt = sync_receipt.get("draft_apply_receipt")
            if not isinstance(raw_receipt, Mapping):
                raise RuntimeError("draft apply receipt is absent")
            applied_snapshot = close_applied_draft_snapshot(
                decision,
                raw_receipt,
                snapshot_path=Path(str(raw_receipt["snapshot_path"])),
            )
    except BaseException as transfer_error:
        outcome = decision_outcome_payload(
            decision,
            update_attempted=decision.update_requested,
            update_successful=decision.update_requested,
            draft_refit_attempted=decision.draft_refit_requested,
            draft_refit_successful=False,
        )
        close_draft_step_transaction(
            transaction, decision=decision, outcome=outcome,
            applied_snapshot=None, scheduler=scheduler,
            decision_ledger=decision_ledger, grpo_save_state=grpo_save_state,
            transaction_store=transaction_store,
        )
        raise transfer_error
    outcome = decision_outcome_payload(
        decision,
        update_attempted=decision.update_requested,
        update_successful=decision.update_requested,
        draft_refit_attempted=decision.draft_refit_requested,
        draft_refit_successful=decision.draft_refit_requested,
    )
    close_error = close_draft_step_transaction(
        transaction, decision=decision, outcome=outcome,
        applied_snapshot=applied_snapshot, scheduler=scheduler,
        decision_ledger=decision_ledger, grpo_save_state=grpo_save_state,
        transaction_store=transaction_store,
    )
    if close_error is not None:
        raise close_error
    publish_target_version()
    if selection.draft:
        publish_draft_version(decision.decision_id)
    return selection
```

In `grpo_train_sync`, capture `scheduler.state.applied_draft_version` immediately
before each synchronous generation call and pass its returned metric mapping
through `stamp_selected_rollout_science`; the disabled fast path returns the
original mapping by identity. Pass those exact contributing mappings to
`prepare_sync_draft_decision`. The capability registry treats this call site,
the selected token-count producer, and canonical logger registration as the
three sync science capabilities; it cannot be enabled by config alone. Adaptive
mode always reconstructs the value it
needs for scheduling. With `cadence_runtime.enabled=true`, every mode also records
an immediate Step+1 acceptance/selected-serving-version observation when the
previous step refit the drafter. With the default `false`, fixed and legacy
`always` do not iterate or inspect those metrics. Only adaptive passes acceptance
to `scheduler.decide`; `always` and fixed decisions receive `None`. Install the
returned evidence only in experiment mode, then
durably create `transaction = transaction_store.begin(decision)` before entering
the CP1 monolithic or CP>1 split worker path. The same transaction is resolved on
worker failure, target-only transfer failure, draft transfer failure, or success.
Before marking target weights stale, starting transfer, incrementing
`weight_version`, or publishing `applied_draft_version`, require the worker's
world/DP-wide receipt:

```python
draft_update_ok = (
    not decision.update_requested
    or train_results.get("draft_update_successful") is True
)
if not draft_update_ok:
    outcome = decision_outcome_payload(
        decision,
        update_attempted=True,
        update_successful=False,
        draft_refit_attempted=False,
        draft_refit_successful=False,
    )
    close_draft_step_transaction(
        transaction, decision=decision, outcome=outcome,
        applied_snapshot=None, scheduler=scheduler,
        decision_ledger=decision_ledger, grpo_save_state=grpo_save_state,
        transaction_store=transaction_store,
    )
    raise RuntimeError("draft update failed across workers before weight transfer")
```

Only after that guard does every successful policy step set target stale and call:

```python
selection = WeightSyncSelection(target=True, draft=decision.draft_refit_requested)
applied_snapshot = None
try:
    refit_metrics = refit_policy_generation(
        policy,
        policy_generation,
        colocated_inference,
        timer=timer,
        kv_scales=kv_scales_cache if sync_kv_scales else None,
        selection=selection,
    )
    if refit_metrics.get("successful") is not True:
        raise RuntimeError("target transfer receipt failed")
    if selection.draft:
        draft_apply_receipt = refit_metrics.get("draft_apply_receipt")
        if not isinstance(draft_apply_receipt, Mapping):
            raise RuntimeError("draft apply receipt is absent")
        applied_snapshot = close_applied_draft_snapshot(
            decision,
            draft_apply_receipt,
            snapshot_path=Path(str(draft_apply_receipt["snapshot_path"])),
        )
except BaseException as transfer_error:
    outcome = decision_outcome_payload(
        decision,
        update_attempted=decision.update_requested,
        update_successful=decision.update_requested,
        draft_refit_attempted=decision.draft_refit_requested,
        draft_refit_successful=False,
    )
    close_draft_step_transaction(
        transaction, decision=decision, outcome=outcome,
        applied_snapshot=None, scheduler=scheduler,
        decision_ledger=decision_ledger, grpo_save_state=grpo_save_state,
        transaction_store=transaction_store,
    )
    raise transfer_error
outcome = decision_outcome_payload(
    decision,
    update_attempted=decision.update_requested,
    update_successful=decision.update_requested,
    draft_refit_attempted=decision.draft_refit_requested,
    draft_refit_successful=decision.draft_refit_requested,
)
close_error = close_draft_step_transaction(
    transaction, decision=decision, outcome=outcome,
    applied_snapshot=applied_snapshot, scheduler=scheduler,
    decision_ledger=decision_ledger, grpo_save_state=grpo_save_state,
    transaction_store=transaction_store,
)
if close_error is not None:
    raise close_error
publish_target_version()
if applied_snapshot is not None:
    publish_draft_version(applied_snapshot.version)
schedule_metrics = scheduler.metrics(decision)
schedule_metrics.update(
    {
        "draft_schedule/update_successful": float(
            train_results["draft_update_successful"]
        ),
        "draft_schedule/refit_successful": float(
            decision.draft_refit_requested and selection.draft
        ),
        "timing/draft_policy_total": float(
            train_results["timing/train/draft_policy_total"]
        ),
        "timing/draft_refit_total": float(
            refit_metrics.get("timing/train/draft_refit_total", 0.0)
        ),
    }
)
step_metrics.update(schedule_metrics)
```

Startup synchronization is a separate lifecycle event and does not call `record_outcome`: it restores current target bytes, then applies the checkpoint's exact serving-draft snapshot through `restore_serving_draft_after_startup_sync`. It is not a full refit, and a resumed non-`always` run may never substitute current trainable draft bytes for that snapshot. A validation refit against the serving engine uses `WeightSyncSelection(target=True, draft=False)`; any path requesting `draft=True` without the current scheduled decision raises `RuntimeError("out-of-band serving draft refit")` before transfer. An isolated validation engine may perform its own full load because it cannot mutate serving provenance.

Every worker or transfer exit routes through `close_draft_step_transaction` exactly once. This includes target-only transfer exceptions on steps whose draft action was skipped: they close `draft_refit_attempted=False`, `draft_refit_successful=False`, and `draft_refit_skipped=True`, append the ledger unconditionally, atomically persist the scheduler/snapshot/ledger bundle, and only then re-raise the original transfer exception. Requested draft failures close attempted-but-failed refit accounting the same way. No target or draft version is published before the atomic bundle receipt succeeds. The immutable decision captures the selected serving-draft version at the start of the policy step; merge `scheduler.metrics(decision)` into every enabled-draft canonical row so the logger emits `train/draft_schedule/applied_draft_version` even on no-update rows. Timing keys are emitted as unprefixed `timing/draft_policy_total` and `timing/draft_refit_total`; the train logger supplies its single namespace prefix.

- [ ] **Step 4: Run the GREEN sync scheduler and observation checks.**

Run: `uv run --group test pytest -q tests/unit/algorithms/test_draft_update_observation.py tests/unit/algorithms/test_grpo_sync_draft_schedule.py tests/unit/algorithms/test_grpo.py -k 'draft_schedule or train_policy_from_meta' && uv run ruff check nemo_rl/algorithms/draft_update_observation.py nemo_rl/algorithms/grpo_sync.py tests/unit/algorithms/test_draft_update_observation.py tests/unit/algorithms/test_grpo_sync_draft_schedule.py`

Expected: tests PASS; the multi-batch case returns `0.1`, and target sync occurs on both the skipped and updated policy steps.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/algorithms/draft_update_observation.py nemo_rl/algorithms/grpo_sync.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_draft_update_observation.py tests/unit/algorithms/test_grpo_sync_draft_schedule.py
git commit -S -s -m "feat(grpo): schedule online draft updates"
git verify-commit HEAD
```

Expected: signature verification exits 0.

### Task 8: Preserve separate target/draft provenance in DataPlane metadata

**Files:**
- Modify: `nemo_rl/experience/payload.py`
- Modify: `nemo_rl/experience/rollout_manager.py`
- Modify: `nemo_rl/algorithms/async_utils/replay_buffer.py`
- Modify: `nemo_rl/algorithms/draft_update_observation.py`
- Modify: `tests/unit/experience/test_payload.py`
- Modify: `tests/unit/single_controller/test_rollout_pump.py`
- Modify: `tests/unit/single_controller/test_tq_replay_buffer.py`
- Modify: `tests/unit/algorithms/test_draft_update_observation.py`

**Interfaces:**
- Consumes: each selected `PromptGroupRecord.rollout_metrics` and the `weight_version: int` / `applied_draft_version: int` pair captured atomically at reservation time. It must not consume a controller-, engine-, or logger-level aggregate that includes rollouts not selected for this policy step.
- Produces: `SPEC_ACCEPTED_TAG`, `SPEC_DRAFT_TAG`, `APPLIED_DRAFT_VERSION_TAG`, `CADENCE_GROUP_ID_TAG`, and `CADENCE_GROUP_ROWS_TAG` on `KVBatchMeta.tags`; `acceptance_from_selected_meta(meta) -> tuple[float | None, int]` validates complete groups, deduplicates repeated per-row group counts, and sums only the groups actually selected for training.

- [ ] **Step 1: Write RED tag/provenance and selected-meta aggregation tests.**

```python
from types import SimpleNamespace

import torch

from nemo_rl.experience.rollout_manager import RolloutManager


def train_batch(rows: int) -> dict[str, torch.Tensor]:
    return {
        "input_lengths": torch.ones(rows, dtype=torch.long),
        "input_ids": torch.arange(rows, dtype=torch.long).reshape(rows, 1),
    }


def meta_with_group_tags(
    groups: list[tuple[str, int, float, float, int]],
):
    tags = []
    for group_id, rows, accepted, draft, version in groups:
        tags.extend(
            cadence_tags(
                rows=rows,
                weight_version=9,
                applied_draft_version=version,
                group_id=group_id,
                accepted_tokens=accepted,
                draft_tokens=draft,
            )
        )
    return SimpleNamespace(tags=tags)


def rollout_manager_for_version_test() -> RolloutManager:
    manager = object.__new__(RolloutManager)
    manager._weight_version = 0
    manager._applied_draft_version = 0
    return manager


def test_payload_keeps_target_and_applied_draft_versions_distinct() -> None:
    _ids, _fields, tags = pack_payload(
        train_batch(2),
        weight_version=9,
        applied_draft_version=3,
        group_id="g",
        rollout_metrics={
            "vllm/spec_num_accepted_tokens": 7.0,
            "vllm/spec_num_draft_tokens": 10.0,
        },
    )
    assert [tag["weight_version"] for tag in tags] == [9, 9]
    assert [tag[APPLIED_DRAFT_VERSION_TAG] for tag in tags] == [3, 3]
    assert [tag[SPEC_ACCEPTED_TAG] for tag in tags] == [7.0, 7.0]
    assert [tag[SPEC_DRAFT_TAG] for tag in tags] == [10.0, 10.0]
    assert len({tag[CADENCE_GROUP_ID_TAG] for tag in tags}) == 1
    assert [tag[CADENCE_GROUP_ROWS_TAG] for tag in tags] == [2, 2]


def test_target_version_advances_while_draft_version_is_stable() -> None:
    rollout_manager = rollout_manager_for_version_test()
    rollout_manager.set_weight_version(8)
    rollout_manager.set_applied_draft_version(2)
    first = rollout_manager.reserve_versions()
    rollout_manager.set_weight_version(9)
    second = rollout_manager.reserve_versions()
    assert first == (8, 2)
    assert second == (9, 2)


def test_selected_meta_uses_only_selected_prompt_group_counts() -> None:
    selected = meta_with_group_tags(
        [
            ("selected-a", 2, 9.0, 10.0, 4),
            ("selected-b", 2, 1.0, 90.0, 4),
        ]
    )
    unrelated_engine_aggregate = {
        "vllm/spec_num_accepted_tokens": 999.0,
        "vllm/spec_num_draft_tokens": 1000.0,
    }
    acceptance, version = acceptance_from_selected_meta(selected)
    assert acceptance == pytest.approx(0.1)
    assert version == 4
    assert unrelated_engine_aggregate not in selected.tags


def test_selected_meta_rejects_partial_prompt_group() -> None:
    selected = meta_with_group_tags([("group-a", 2, 7.0, 10.0, 3)])
    selected.tags.pop()
    with pytest.raises(RuntimeError, match="partial cadence group group-a"):
        acceptance_from_selected_meta(selected)
```

- [ ] **Step 2: Run the RED metadata tests and confirm the new tags/keyword are missing.**

Run: `uv run --group test pytest -q tests/unit/experience/test_payload.py tests/unit/single_controller/test_rollout_pump.py tests/unit/single_controller/test_tq_replay_buffer.py tests/unit/algorithms/test_draft_update_observation.py -k 'applied_draft or acceptance_tags'`

Expected: FAIL with `NameError: name 'APPLIED_DRAFT_VERSION_TAG' is not defined` or an unexpected `applied_draft_version` keyword.

- [ ] **Step 3: Stamp deduplicable per-group counts and capture both versions at reservation.**

```python
SPEC_ACCEPTED_TAG = "draft_schedule/spec_accepted_tokens"
SPEC_DRAFT_TAG = "draft_schedule/spec_draft_tokens"
APPLIED_DRAFT_VERSION_TAG = "draft_schedule/applied_draft_version"
CADENCE_GROUP_ID_TAG = "draft_schedule/group_id"
CADENCE_GROUP_ROWS_TAG = "draft_schedule/group_rows"


def cadence_tags(
    *,
    rows: int,
    weight_version: int,
    applied_draft_version: int,
    group_id: str,
    accepted_tokens: float,
    draft_tokens: float,
) -> list[dict[str, float | int | str]]:
    return [
        {
            "weight_version": weight_version,
            APPLIED_DRAFT_VERSION_TAG: applied_draft_version,
            CADENCE_GROUP_ID_TAG: group_id,
            CADENCE_GROUP_ROWS_TAG: rows,
            SPEC_ACCEPTED_TAG: accepted_tokens,
            SPEC_DRAFT_TAG: draft_tokens,
        }
        for row in range(rows)
    ]
```

Add `_applied_draft_version: int = 0`, `set_applied_draft_version(version: int) -> None`, and `reserve_versions() -> tuple[int, int]` to the rollout manager. Capture `(start_weight_version, start_applied_draft_version)` before `TQReplayBuffer.reserve`; store both on the reserved slot and pass both through `commit`. Reject a rollout if either live version differs at commit. In `TQReplayBuffer.commit`, read accepted and draft counts directly from that call's `PromptGroupRecord.rollout_metrics`, validate finite `0 <= accepted <= draft`, and pass them to `pack_payload`; do not read `policy_generation.get_metrics()`, controller logger aggregates, or a run-global vLLM counter.

Implement `acceptance_from_selected_meta` by grouping selected tags by `CADENCE_GROUP_ID_TAG`. Require exactly `CADENCE_GROUP_ROWS_TAG` rows for each group, identical counts/version on every row in the group, and one applied-draft version across all groups. Add one accepted/draft pair per unique group and return `sum(accepted) / sum(draft)` plus that version. Missing or zero total draft count yields `(None, version)`; malformed or partial groups fail before `begin_train_step`.

- [ ] **Step 4: Run the GREEN payload, rollout, replay, and aggregation tests.**

Run: `uv run --group test pytest -q tests/unit/experience/test_payload.py tests/unit/single_controller/test_rollout_pump.py tests/unit/single_controller/test_tq_replay_buffer.py tests/unit/algorithms/test_draft_update_observation.py -k 'applied_draft or acceptance_tags or weight_version' && uv run ruff check nemo_rl/experience/payload.py nemo_rl/experience/rollout_manager.py nemo_rl/algorithms/async_utils/replay_buffer.py nemo_rl/algorithms/draft_update_observation.py`

Expected: selected tests PASS; `weight_version` can advance independently; concatenated count tags reproduce the count-weighted rate.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/experience/payload.py nemo_rl/experience/rollout_manager.py nemo_rl/algorithms/async_utils/replay_buffer.py nemo_rl/algorithms/draft_update_observation.py tests/unit/experience/test_payload.py tests/unit/single_controller/test_rollout_pump.py tests/unit/single_controller/test_tq_replay_buffer.py tests/unit/algorithms/test_draft_update_observation.py
git commit -S -s -m "feat(draft): preserve rollout acceptance provenance"
git verify-commit HEAD
```

Expected: signature verification exits 0.

### Task 9: Integrate the shared scheduler into single-controller training

**Files:**
- Modify: `nemo_rl/algorithms/single_controller.py`
- Modify: `nemo_rl/algorithms/single_controller_utils/config.py`
- Modify: `nemo_rl/algorithms/single_controller_utils/setup.py`
- Create: `tests/unit/single_controller/test_draft_schedule.py`
- Modify: `tests/unit/single_controller/test_train_pump.py`
- Modify: `tests/unit/single_controller/test_sc_checkpointing.py`
- Modify: `tests/unit/single_controller/test_single_controller_setup.py`

**Interfaces:**
- Consumes: shared scheduler, selected `KVBatchMeta`, `WeightSyncSelection`, and separate `weight_version`/`applied_draft_version` tags.
- Produces: `prepare_single_controller_draft_decision(scheduler, selected_meta, *, cadence_runtime_enabled, evidence, global_step, begin_train_step) -> PreparedDraftDecision`, one decision before either training entrypoint, target sync on every completed policy step, draft sync only when requested, and exact checkpoint/resume provenance. Adaptive always reads selected count/version tags for scheduling correctness. Fixed/`always` read and terminally receipt them only in experiment mode; the default nonadaptive path leaves metadata untouched.

- [ ] **Step 1: Write RED adaptive validation, decision-order, and provenance tests.**

```python
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.draft_cadence_runtime import CadenceTerminalEvidence
from nemo_rl.algorithms.draft_update_schedule import DraftUpdateScheduler
from nemo_rl.algorithms.grpo_sync import apply_scheduled_refit
from nemo_rl.algorithms.single_controller import (
    SingleController,
    prepare_single_controller_draft_decision,
)
from nemo_rl.algorithms.single_controller_utils.config import validate_config
from nemo_rl.experience.payload import cadence_tags
from nemo_rl.models.policy.draft_config import (
    AdaptiveDraftUpdateScheduleConfig,
    AlwaysDraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)


def adaptive_config_for_test():
    return SimpleNamespace(
        policy={
            "draft": SimpleNamespace(
                update_schedule=AdaptiveDraftUpdateScheduleConfig()
            )
        },
        async_rl=SimpleNamespace(
            sampler=SimpleNamespace(
                name="in_order",
                max_lookahead_versions=0,
            )
        ),
    )


def adaptive_meta_for_test(versions: list[int]):
    tags = []
    for index, version in enumerate(versions):
        tags.extend(
            cadence_tags(
                rows=1,
                weight_version=9,
                applied_draft_version=version,
                group_id=f"group-{index}",
                accepted_tokens=7.0,
                draft_tokens=10.0,
            )
        )
    return SimpleNamespace(tags=tags)


def adaptive_scheduler_for_test(version: int = 4):
    scheduler = DraftUpdateScheduler.create(
        AdaptiveDraftUpdateScheduleConfig(
            min_interval=1,
            max_interval=10,
            min_observations=1,
        ),
        origin_step=0,
    )
    scheduler.state.applied_draft_version = version
    return scheduler


def test_adaptive_requires_in_order_zero_staleness() -> None:
    async_config = adaptive_config_for_test()
    async_config.async_rl.sampler.name = "weight_fifo"
    async_config.async_rl.sampler.max_lookahead_versions = 1
    async_config.policy["draft"].update_schedule = AdaptiveDraftUpdateScheduleConfig()
    with pytest.raises(ValueError, match="adaptive draft cadence requires.*in_order.*zero"):
        validate_config(async_config)


def test_train_pump_decides_before_opening_step() -> None:
    events = []
    scheduler = adaptive_scheduler_for_test()
    prepared = prepare_single_controller_draft_decision(
        scheduler,
        adaptive_meta_for_test([4]),
        cadence_runtime_enabled=False,
        evidence=None,
        global_step=1,
        begin_train_step=lambda decision: events.append(
            ("policy.begin_train_step", decision.decision_id)
        ),
        on_decide=lambda decision: events.append(
            ("scheduler.decide", decision.decision_id)
        ),
    )
    assert events == [
        ("scheduler.decide", prepared.decision.decision_id),
        ("policy.begin_train_step", prepared.decision.decision_id),
    ]


def test_adaptive_rejects_mixed_applied_draft_versions() -> None:
    begin_train_step = MagicMock()
    with pytest.raises(RuntimeError, match="mixed applied_draft_version"):
        prepare_single_controller_draft_decision(
            adaptive_scheduler_for_test(),
            adaptive_meta_for_test([4, 5]),
            cadence_runtime_enabled=False,
            evidence=None,
            global_step=1,
            begin_train_step=begin_train_step,
        )
    begin_train_step.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        AlwaysDraftUpdateScheduleConfig(),
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
    ],
)
def test_experiment_nonadaptive_sc_collects_science_but_not_schedule_input(
    config,
) -> None:
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    first = prepare_single_controller_draft_decision(
        scheduler,
        adaptive_meta_for_test([0]),
        cadence_runtime_enabled=True,
        evidence=CadenceTerminalEvidence({}, {}),
        global_step=1,
        begin_train_step=lambda _decision: None,
    )
    scheduler.record_outcome(
        first.decision,
        update_attempted=first.decision.update_requested,
        update_successful=first.decision.update_requested,
        draft_refit_attempted=first.decision.draft_refit_requested,
        draft_refit_successful=first.decision.draft_refit_requested,
    )
    assert first.terminal_evidence is not None
    second = prepare_single_controller_draft_decision(
        scheduler,
        adaptive_meta_for_test([1]),
        cadence_runtime_enabled=True,
        evidence=first.terminal_evidence,
        global_step=2,
        begin_train_step=lambda _decision: None,
    )
    assert second.decision.observed_acceptance is None
    assert second.terminal_evidence is not None
    assert second.terminal_evidence.observations_by_refit_step[1] == {
        "refit_step": 1,
        "observation_step": 2,
        "applied_draft_version": 1,
        "acceptance_rate": pytest.approx(0.7),
    }


@pytest.mark.parametrize(
    "config",
    [
        AlwaysDraftUpdateScheduleConfig(),
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=10
        ),
    ],
)
def test_default_nonadaptive_sc_does_not_touch_science_tags(config) -> None:
    selected_meta = MagicMock()
    prepared = prepare_single_controller_draft_decision(
        DraftUpdateScheduler.create(config, origin_step=0),
        selected_meta,
        cadence_runtime_enabled=False,
        evidence=None,
        global_step=1,
        begin_train_step=lambda _decision: None,
    )
    assert prepared.decision.observed_acceptance is None
    assert prepared.terminal_evidence is None
    assert selected_meta.mock_calls == []


def test_single_controller_failed_update_never_syncs_or_publishes() -> None:
    scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
        origin_step=0,
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    sync_weights = MagicMock()
    publish_target = MagicMock()
    publish_draft = MagicMock()
    transaction = MagicMock(decision_id=1)
    ledger = MagicMock()
    save_state = SimpleNamespace(
        draft_update_schedule=None,
        applied_draft_snapshot={"version": 0},
    )
    store = MagicMock()
    store.commit_bundle_atomic.return_value = {
        "successful": True,
        "provisional": True,
        "base_checkpoint_id": transaction.base_checkpoint_id,
        "decision_id": 1,
        "scheduler_decision_id": 1,
        "snapshot_version": 0,
        "ledger_high_water": 1,
    }
    with pytest.raises(RuntimeError, match="draft update failed across workers"):
        apply_scheduled_refit(
            decision,
            {"draft_update_successful": False},
            scheduler,
            transaction=transaction,
            decision_ledger=ledger,
            grpo_save_state=save_state,
            transaction_store=store,
            runtime_writer=None,
            terminal_evidence=None,
            sync_weights=sync_weights,
            publish_target_version=publish_target,
            publish_draft_version=publish_draft,
        )
    sync_weights.assert_not_called()
    publish_target.assert_not_called()
    publish_draft.assert_not_called()
    ledger.append_closed_once.assert_called_once()
    assert scheduler.state.attempted_updates == 1
    assert scheduler.state.failed_updates == 1
    assert scheduler.state.attempted_refits == 0
    assert scheduler.state.failed_refits == 0
    assert scheduler.state.skipped_refits == 1


@pytest.mark.asyncio
async def test_single_controller_returns_closed_snapshot_without_publication(tmp_path) -> None:
    scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="refit_only", fixed_interval=1
        ),
        origin_step=0,
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    snapshot_path = tmp_path / "applied-draft-v1.safetensors"
    raw = b"applied-draft-v1"
    snapshot_path.write_bytes(raw)
    events = []
    synchronizer = MagicMock()
    synchronizer.sync_weights.return_value = {
        "successful": True,
        "draft_apply_receipt": {
            "successful": True,
            "version": 1,
            "snapshot_path": str(snapshot_path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    rollout_manager = MagicMock()
    rollout_manager.set_applied_draft_version.side_effect = (
        lambda _version: events.append("published")
    )
    controller = SimpleNamespace(
        _weight_synchronizer=synchronizer,
        _kv_scales_cache=None,
        _grpo_save_state=SimpleNamespace(applied_draft_snapshot=None),
        _flush_cadence_save_state=lambda state: (
            events.append("durable")
            or {
                "successful": True,
                "version": state.applied_draft_snapshot["version"],
                "sha256": state.applied_draft_snapshot["sha256"],
            }
        ),
        _rollout_manager=rollout_manager,
        _weight_version=4,
    )
    snapshot = await SingleController._apply_single_controller_refit(
        controller, decision
    )
    assert snapshot is not None and snapshot.version == 1
    assert events == []
    rollout_manager.set_applied_draft_version.assert_not_called()
    rollout_manager.set_weight_version.assert_not_called()


@pytest.mark.asyncio
async def test_single_controller_failed_sync_receipt_publishes_nothing() -> None:
    scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="refit_only", fixed_interval=1
        ),
        origin_step=0,
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    rollout_manager = MagicMock()
    controller = SimpleNamespace(
        _weight_synchronizer=MagicMock(),
        _kv_scales_cache=None,
        _grpo_save_state=MagicMock(),
        _flush_cadence_save_state=MagicMock(),
        _rollout_manager=rollout_manager,
        _weight_version=4,
    )
    controller._weight_synchronizer.sync_weights.return_value = {
        "successful": False
    }
    with pytest.raises(RuntimeError, match="sync receipt failed"):
        await SingleController._apply_single_controller_refit(controller, decision)
    rollout_manager.set_applied_draft_version.assert_not_called()
    rollout_manager.set_weight_version.assert_not_called()
```

- [ ] **Step 2: Run the RED single-controller tests and confirm no cadence integration exists.**

Run: `uv run --group test pytest -q tests/unit/single_controller/test_draft_schedule.py tests/unit/single_controller/test_train_pump.py tests/unit/single_controller/test_sc_checkpointing.py tests/unit/single_controller/test_single_controller_setup.py -k 'draft_schedule or applied_draft'`

Expected: FAIL because `tests/unit/single_controller/test_draft_schedule.py` imports scheduler integration helpers that do not exist.

- [ ] **Step 3: Validate config and integrate one metadata-first decision per step.**

```python
def validate_adaptive_draft_cadence(config: MasterConfig) -> None:
    draft = config.policy.get("draft")
    if draft is None or draft.update_schedule.mode != "adaptive":
        return
    sampler = config.async_rl.sampler
    lookahead = getattr(sampler, "max_lookahead_versions", None)
    if sampler.name != "in_order" or lookahead != 0:
        raise ValueError(
            "adaptive draft cadence requires async_rl.sampler.name='in_order' "
            "and async_rl.sampler.max_lookahead_versions=0"
        )


def prepare_single_controller_draft_decision(
    scheduler: DraftUpdateScheduler,
    selected_meta: KVBatchMeta,
    *,
    cadence_runtime_enabled: bool,
    evidence: CadenceTerminalEvidence | None,
    global_step: int,
    begin_train_step: Callable[[DraftUpdateDecision], None],
    on_decide: Callable[[DraftUpdateDecision], None] | None = None,
) -> PreparedDraftDecision:
    needs_science = scheduler.config.mode == "adaptive" or cadence_runtime_enabled
    science_acceptance = None
    if needs_science:
        science_acceptance, selected_version = acceptance_from_selected_meta(
            selected_meta
        )
        if selected_version != scheduler.state.applied_draft_version:
            raise RuntimeError(
                "mixed applied_draft_version or stale selected rollout: "
                f"selected={selected_version}, "
                f"current={scheduler.state.applied_draft_version}"
            )
    if cadence_runtime_enabled != (evidence is not None):
        raise ValueError("cadence runtime evidence enablement mismatch")
    prior_refit_step = scheduler.state.last_applied_refit_step
    decision = scheduler.decide(
        global_step=global_step,
        acceptance=(
            science_acceptance
            if scheduler.config.mode == "adaptive"
            else None
        ),
    )
    if cadence_runtime_enabled:
        assert evidence is not None
        evidence = record_terminal_post_refit_observation(
            evidence, decision=decision,
            last_applied_refit_step=prior_refit_step,
            acceptance_rate=science_acceptance,
        )
    if on_decide is not None:
        on_decide(decision)
    begin_train_step(decision)
    return PreparedDraftDecision(decision, evidence)
```

The train pump selects a complete logical step as metadata first. Adaptive mode
always calls `acceptance_from_selected_meta`, rejects mixed or stale selected
serving versions, and passes acceptance into scheduling. With
`cadence_runtime.enabled=true`, fixed and `always` also read the tags and record
count-weighted Step+1 acceptance/version science, but still pass `None` to the
scheduler. With the default `false`, nonadaptive mode does not touch those tags.
The controller installs returned evidence only in experiment mode, then durably
opens `transaction =
self._draft_step_transaction_store.begin(decision)` before either monolithic CP1
or split CP>1 training, and passes that transaction through worker failure and
transfer completion. After the worker result returns, apply the same
`draft_update_ok` guard from Task 7 before transfer or publication. On policy
success always call:

```python
async def _apply_single_controller_refit(
    self,
    decision: DraftUpdateDecision,
) -> AppliedDraftSnapshot | None:
    selection = WeightSyncSelection(
        target=True,
        draft=decision.draft_refit_requested,
    )
    sync_receipt = await asyncio.to_thread(
        self._weight_synchronizer.sync_weights,
        selection=selection,
        kv_scales=self._kv_scales_cache,
    )
    if not isinstance(sync_receipt, Mapping) or sync_receipt.get("successful") is not True:
        raise RuntimeError("single-controller target sync receipt failed")
    applied_snapshot = None
    if selection.draft:
        draft_apply_receipt = sync_receipt.get("draft_apply_receipt")
        if not isinstance(draft_apply_receipt, Mapping):
            raise RuntimeError("single-controller draft apply receipt is absent")
        snapshot_path = Path(str(draft_apply_receipt["snapshot_path"]))
        applied_snapshot = close_applied_draft_snapshot(
            decision, draft_apply_receipt, snapshot_path=snapshot_path
        )
    return applied_snapshot
```

Then close the outcome/ledger and emit the decision-captured selected serving version on that same row:

```python
try:
    if self._cadence_runtime_writer is not None:
        if self._cadence_terminal_evidence is None:
            raise RuntimeError("cadence runtime writer requires terminal evidence")
        if decision.update_requested:
            update_receipt = train_results.get("draft_update_receipt")
            if not isinstance(update_receipt, Mapping):
                raise RuntimeError("successful draft update lacks worker receipt")
            self._cadence_runtime_writer.successful_update_closed(
                decision=decision,
                worker_receipt=update_receipt,
                evidence=self._cadence_terminal_evidence,
                save_state=self._grpo_save_state,
            )
    elif self._cadence_terminal_evidence is not None:
        raise RuntimeError("terminal evidence is enabled without runtime writer")
    applied_snapshot = await self._apply_single_controller_refit(decision)
except BaseException as transfer_error:
    outcome = decision_outcome_payload(
        decision,
        update_attempted=decision.update_requested,
        update_successful=decision.update_requested,
        draft_refit_attempted=decision.draft_refit_requested,
        draft_refit_successful=False,
    )
    close_draft_step_transaction(
        transaction, decision=decision, outcome=outcome,
        applied_snapshot=None, scheduler=scheduler,
        decision_ledger=decision_ledger, grpo_save_state=self._grpo_save_state,
        transaction_store=self._draft_step_transaction_store,
    )
    raise transfer_error
outcome = decision_outcome_payload(
    decision,
    update_attempted=decision.update_requested,
    update_successful=(
        decision.update_requested
        and train_results["draft_update_successful"] is True
    ),
    draft_refit_attempted=decision.draft_refit_requested,
    draft_refit_successful=(applied_snapshot is not None),
)
close_error = close_draft_step_transaction(
    transaction, decision=decision, outcome=outcome,
    applied_snapshot=applied_snapshot, scheduler=scheduler,
    decision_ledger=decision_ledger, grpo_save_state=self._grpo_save_state,
    transaction_store=self._draft_step_transaction_store,
)
if close_error is not None:
    raise close_error
self._weight_version += 1
self._rollout_manager.set_weight_version(self._weight_version)
if applied_snapshot is not None:
    self._rollout_manager.set_applied_draft_version(applied_snapshot.version)
step_metrics.update(scheduler.metrics(decision))
```

`WeightSynchronizer.sync_weights(...)` therefore returns `Mapping[str, object]`: `successful=True` for target completion and, when `selection.draft`, a nested `draft_apply_receipt` containing `successful=True`, decision version, immutable snapshot path, and SHA256. In experiment mode, a successful requested update must first produce the worker model/optimizer digest receipt; `successful_update_closed` exclusively persists it and installs the updated evidence in `GRPOSaveState` before `_apply_single_controller_refit`, any checkpoint, or either version publication. The sync controller follows the identical order in `apply_scheduled_refit`. Worker failure and every target/draft transfer exception resolve the already-open transaction, close scheduler accounting, append exactly one ledger row, atomically persist the combined bundle, and then abort with the original error. Success publishes target/draft versions only after that same bundle commit. At setup/resume, recover transaction resolutions first, then use Task 3's component-separated startup sequence and enable reservations only after the recovered snapshot is durably installed. Never perform a mixed full refit from current trainable draft bytes on saved cadence state.

- [ ] **Step 4: Run the GREEN single-controller integration suite.**

Run: `uv run --group test pytest -q tests/unit/single_controller/test_draft_schedule.py tests/unit/single_controller/test_train_pump.py tests/unit/single_controller/test_sc_checkpointing.py tests/unit/single_controller/test_single_controller_setup.py tests/unit/single_controller/test_single_controller.py -k 'draft_schedule or applied_draft or sync_weights' && uv run ruff check nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/single_controller_utils/config.py nemo_rl/algorithms/single_controller_utils/setup.py tests/unit/single_controller/test_draft_schedule.py`

Expected: selected tests PASS; target version increments on every completed step while applied draft version increments only after draft apply.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/single_controller_utils/config.py nemo_rl/algorithms/single_controller_utils/setup.py tests/unit/single_controller/test_draft_schedule.py tests/unit/single_controller/test_train_pump.py tests/unit/single_controller/test_sc_checkpointing.py tests/unit/single_controller/test_single_controller_setup.py
git commit -S -s -m "feat(single-controller): schedule draft updates"
git verify-commit HEAD
```

Expected: signature verification exits 0.

### Task 10: Add recipes, documentation, and exact functional gates

**Files:**
- Modify: `docs/guides/dflash-dspark-speculative-decoding.md`
- Create: `examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dflash-cadence.yaml`
- Create: `examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dspark-cadence.yaml`
- Create: `tests/unit/models/policy/test_draft_schedule_recipes.py`
- Create: `tests/unit/algorithms/test_draft_schedule_resume_integration.py`
- Create: `tests/test_suites/llm/grpo-qwen3-8b-1n8g-megatron-draft-cadence.sh`
- Create: `tests/test_suites/llm/validate_draft_cadence_receipts.py`

**Interfaces:**
- Consumes: Tasks 1-9 public APIs and existing DFlash/DSpark recipe topology.
- Produces: Hydra-resolvable opt-in recipes and an immutable 20-arm matrix: DFlash/DSpark x packed TP2 CP1-monolithic/CP2-split x `always`/fixed-sparse/fixed-refit-only/adaptive/restore. Every behavior has enough logical steps to execute and inspect its scheduled event.

- [ ] **Step 1: Write RED recipe-resolution and resume-marker tests.**

```python
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.draft_update_schedule import DraftUpdateScheduler
from nemo_rl.models.policy.draft_config import FixedDraftUpdateScheduleConfig


RECIPE_ROOT = Path("examples/configs/recipes/llm")


def load_recipe(name: str):
    return OmegaConf.load(RECIPE_ROOT / name)


def run_resume_integration(
    tmp_path: Path,
    *,
    draft_type: str,
    context_parallel_size: int,
    interval: int,
    total_steps: int,
    split_after_step: int,
) -> dict[str, object]:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=interval
    )

    def advance(scheduler, start: int, end: int):
        decisions = []
        for step in range(start, end + 1):
            decision = scheduler.decide(global_step=step, acceptance=None)
            scheduler.record_outcome(
                decision,
                update_attempted=decision.update_requested,
                update_successful=decision.update_requested,
                draft_refit_attempted=decision.draft_refit_requested,
                draft_refit_successful=decision.draft_refit_requested,
            )
            decisions.append(decision)
        return decisions

    continuous_scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    continuous = advance(continuous_scheduler, 1, total_steps)
    resumed_scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    before_resume = advance(resumed_scheduler, 1, split_after_step)
    checkpoint = tmp_path / f"{draft_type}-cp{context_parallel_size}.json"
    checkpoint.write_text(json.dumps(resumed_scheduler.state_dict(), sort_keys=True))
    restored = DraftUpdateScheduler.create(
        config,
        origin_step=0,
        restored=json.loads(checkpoint.read_text()),
    )
    after_resume = advance(restored, split_after_step + 1, total_steps)
    resumed = before_resume + after_resume
    return {
        "policy_refit_steps": list(range(1, total_steps + 1)),
        "draft_update_steps": [d.global_step for d in resumed if d.update_requested],
        "draft_refit_steps": [d.global_step for d in resumed if d.draft_refit_requested],
        "continuous_decisions": [d.decision_id for d in continuous],
        "resumed_decisions": [d.decision_id for d in resumed],
    }


@pytest.mark.parametrize(
    ("recipe", "mode"),
    [
        ("grpo-qwen3-8b-1n8g-megatron-dflash-cadence.yaml", "fixed"),
        ("grpo-qwen3-8b-1n8g-megatron-dspark-cadence.yaml", "fixed"),
    ],
)
def test_cadence_recipe_resolves_without_changing_default_recipe(recipe, mode) -> None:
    resolved = load_recipe(recipe)
    assert resolved.policy["draft"].update_schedule.mode == mode
    assert resolved.policy["generation"]["colocated"]["enabled"] is True


def test_resume_driver_requires_target_refit_each_step(tmp_path: Path) -> None:
    receipt = run_resume_integration(
        tmp_path,
        draft_type="dflash",
        context_parallel_size=2,
        interval=2,
        total_steps=4,
        split_after_step=2,
    )
    assert receipt["policy_refit_steps"] == [1, 2, 3, 4]
    assert receipt["draft_update_steps"] == [2, 4]
    assert receipt["draft_refit_steps"] == [2, 4]
    assert receipt["resumed_decisions"] == receipt["continuous_decisions"]
```

- [ ] **Step 2: Run the RED recipe tests and confirm the recipe files are absent.**

Run: `uv run --group test pytest -q tests/unit/models/policy/test_draft_schedule_recipes.py tests/unit/algorithms/test_draft_schedule_resume_integration.py`

Expected: FAIL with `FileNotFoundError` for the cadence recipe path.

- [ ] **Step 3: Add opt-in recipes, exact driver arms, and user documentation.**

```yaml
policy:
  draft:
    update_schedule:
      mode: fixed
      action: sparse_update
      fixed_interval: 10
```

Keep the existing default DFlash and DSpark recipes unchanged. The functional driver accepts validated `DRAFT_TYPE`, `CONTEXT_PARALLEL_SIZE`, `CASE_NAME`, and `RESULT_ROOT`; a `case` statement maps those values to literal Hydra arrays without `eval`. Its complete matrix is:

```bash
draft_types=(dflash dspark)
context_parallel_sizes=(1 2)
case_names=(always fixed_sparse fixed_refit_only adaptive restore)

case "$DRAFT_TYPE" in
  dflash)
    CONFIG_PATH="examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dflash-cadence.yaml"
    ;;
  dspark)
    CONFIG_PATH="examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dspark-cadence.yaml"
    ;;
  *) echo "unknown DRAFT_TYPE=$DRAFT_TYPE" >&2; exit 2 ;;
esac
case "$CONTEXT_PARALLEL_SIZE" in
  1|2) ;;
  *) echo "CONTEXT_PARALLEL_SIZE must be 1 or 2" >&2; exit 2 ;;
esac

case "$CASE_NAME" in
  always)
    max_steps=2
    schedule_overrides=("policy.draft.update_schedule.mode=always")
    ;;
  fixed_sparse)
    max_steps=3
    schedule_overrides=(
      "policy.draft.update_schedule.mode=fixed"
      "policy.draft.update_schedule.action=sparse_update"
      "policy.draft.update_schedule.fixed_interval=2"
    )
    ;;
  fixed_refit_only)
    max_steps=3
    schedule_overrides=(
      "policy.draft.update_schedule.mode=fixed"
      "policy.draft.update_schedule.action=refit_only"
      "policy.draft.update_schedule.fixed_interval=2"
    )
    ;;
  adaptive)
    max_steps=4
    schedule_overrides=(
      "policy.draft.update_schedule.mode=adaptive"
      "policy.draft.update_schedule.min_interval=1"
      "policy.draft.update_schedule.max_interval=2"
      "policy.draft.update_schedule.min_observations=1"
      "policy.draft.update_schedule.ewma_alpha=1.0"
      "policy.draft.update_schedule.degradation_threshold=1.0"
      "policy.draft.update_schedule.recovery_threshold=0.999999"
    )
    ;;
  restore)
    max_steps=4
    schedule_overrides=(
      "policy.draft.update_schedule.mode=fixed"
      "policy.draft.update_schedule.action=sparse_update"
      "policy.draft.update_schedule.fixed_interval=2"
      "checkpointing.enabled=true"
      "checkpointing.save_period=2"
    )
    ;;
  *) echo "unknown CASE_NAME=$CASE_NAME" >&2; exit 2 ;;
esac
```

The `restore` case runs Step 1-2, exits after the checkpoint receipt is durable, then launches a second process from that checkpoint for Step 3-4; it also runs an uninterrupted four-step oracle with the same seed/data and compares serialized decisions. The deterministic unit integration test covers adaptive degradation and hysteresis recovery with observations `[0.8, 0.7, 0.8]`; the four-step GPU arm covers the real metric path plus a forced Step-2 maintenance update and a post-refit observation. `CONFIG_PATH` and `schedule_overrides` are defined exactly once by the literal `DRAFT_TYPE` and `CASE_NAME` cases above. Run from the verified product root and pass that already-built array without shell evaluation or a second override-construction path:

```bash
readonly REPO_ROOT="${NEMO_RL_SOURCE_ROOT:?set NEMO_RL_SOURCE_ROOT}"
readonly EXPECTED_PRODUCT_SHA="${PRODUCT_SHA:?set PRODUCT_SHA}"
cd "$REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_PRODUCT_SHA" ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]
uv run examples/run_grpo.py \
  --config "$CONFIG_PATH" \
  data_plane.enabled=true \
  cluster.gpus_per_node=4 \
  policy.megatron_cfg.tensor_model_parallel_size=2 \
  policy.megatron_cfg.context_parallel_size="$CONTEXT_PARALLEL_SIZE" \
  policy.megatron_cfg.sequence_parallel=true \
  policy.sequence_packing.enabled=true \
  grpo.max_num_steps="$max_steps" \
  checkpointing.enabled=true \
  "${schedule_overrides[@]}"
```

CP1 must assert `_should_use_split_draft_training(...) is False` and a recorded `worker.train` call; CP2 must assert it is `True` and record `begin_train_step`, `train_microbatches_from_meta`, and `finish_train_step`. Every arm emits `RESULT_ROOT/$PRODUCT_SHA/$CONTAINER_DIGEST/$DRAFT_TYPE/cp$CONTEXT_PARALLEL_SIZE/$CASE_NAME/receipt.json`, where slash and colon characters in the digest are normalized to underscores. The receipt includes exact product/harness/submodule SHAs, container digest, resolved-config SHA256, seed/data order, node/job ID, requested/completed steps, entrypoint, policy-refit steps, attempted/successful draft-update and refit counters, forced counters, decision IDs, applied draft versions, parameter/optimizer hashes, finite losses, checkpoint path, and terminal exit state. Creation uses an exclusive directory operation and fails if the identity already exists.

- [ ] **Step 4: Run the GREEN CPU/static gates and validate the GPU driver syntax.**

Run: `uv run --group test pytest -q tests/unit/models/policy/test_draft_schedule_recipes.py tests/unit/algorithms/test_draft_schedule_resume_integration.py && uv run ruff check nemo_rl tests/unit/models/policy/test_draft_schedule_recipes.py tests/unit/algorithms/test_draft_schedule_resume_integration.py tests/test_suites/llm/validate_draft_cadence_receipts.py && uv run pyrefly check && bash -n tests/test_suites/llm/grpo-qwen3-8b-1n8g-megatron-draft-cadence.sh`

Expected: tests PASS, Ruff and Pyrefly report no errors, and `bash -n` exits 0.

- [ ] **Step 5: Preflight and submit the immutable 20-arm functional matrix.**

Run `/fairshare oci-hsg` and record the selected account in `$RESULT_ROOT/submission.json`. The implementation and driver must be committed, signed, pushed, and recursively clean before the first preflight; every job receives and records that exact immutable SHA. Then run:

```bash
git pull --ff-only
git add docs/guides/dflash-dspark-speculative-decoding.md examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dflash-cadence.yaml examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dspark-cadence.yaml tests/unit/models/policy/test_draft_schedule_recipes.py tests/unit/algorithms/test_draft_schedule_resume_integration.py tests/test_suites/llm/grpo-qwen3-8b-1n8g-megatron-draft-cadence.sh tests/test_suites/llm/validate_draft_cadence_receipts.py
git commit -S -s -m "test(draft): validate scheduled online training"
git verify-commit HEAD
git push
PRODUCT_SHA="$(git rev-parse HEAD)"
UPSTREAM_SHA="$(git rev-parse '@{u}')"
test "$PRODUCT_SHA" = "$UPSTREAM_SHA"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git submodule status --recursive
export PRODUCT_SHA
export NEMO_RL_SOURCE_ROOT="$(pwd -P)"
functional_job_ids=()
for draft_type in dflash dspark; do
  for cp_size in 1 2; do
    for case_name in always fixed_sparse fixed_refit_only adaptive restore; do
      export DRAFT_TYPE="$draft_type"
      export CONTEXT_PARALLEL_SIZE="$cp_size"
      export CASE_NAME="$case_name"
      sbatch --test-only --export="NONE,PRODUCT_SHA=$PRODUCT_SHA,NEMO_RL_SOURCE_ROOT=$NEMO_RL_SOURCE_ROOT,DRAFT_TYPE=$DRAFT_TYPE,CONTEXT_PARALLEL_SIZE=$CONTEXT_PARALLEL_SIZE,CASE_NAME=$CASE_NAME,RESULT_ROOT=$RESULT_ROOT" tests/test_suites/llm/grpo-qwen3-8b-1n8g-megatron-draft-cadence.sh
      job_id="$(sbatch --parsable --export="NONE,PRODUCT_SHA=$PRODUCT_SHA,NEMO_RL_SOURCE_ROOT=$NEMO_RL_SOURCE_ROOT,DRAFT_TYPE=$DRAFT_TYPE,CONTEXT_PARALLEL_SIZE=$CONTEXT_PARALLEL_SIZE,CASE_NAME=$CASE_NAME,RESULT_ROOT=$RESULT_ROOT" tests/test_suites/llm/grpo-qwen3-8b-1n8g-megatron-draft-cadence.sh)"
      [[ "$job_id" =~ ^[0-9]+$ ]]
      functional_job_ids+=("$job_id")
    done
  done
done
job_ids_csv="$(IFS=,; echo "${functional_job_ids[*]}")"
monitor_started_at="$(date +%s)"
for check in 0 1 2 3 4 5; do
  queue_rows="$(squeue --jobs "$job_ids_csv" --noheader --format='%A|%T|%j|%R')"
  accounting_rows="$(sacct -j "$job_ids_csv" --noheader --parsable2 --format=JobIDRaw,State,JobName)"
  if grep -Eq '\|(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)(\||$)' <<<"$accounting_rows"; then
    echo "functional cadence job failed during startup" >&2
    exit 1
  fi
  for job_id in "${functional_job_ids[@]}"; do
    grep -Eq "^${job_id}(\\.|\\|)" <<<"$queue_rows"$'\n'"$accounting_rows" || {
      echo "job $job_id disappeared from scheduler visibility" >&2
      exit 1
    }
  done
  if find "$RESULT_ROOT" -name 'slurm-*.out' -type f -print0 | \
    xargs -0 -r grep -Eiq 'Traceback|ModuleNotFoundError|CUDA error|NCCL error|RuntimeError:'; then
    echo "functional cadence startup error found in logs" >&2
    exit 1
  fi
  if [[ "$check" -lt 5 ]]; then sleep 60; fi
done
(( $(date +%s) - monitor_started_at >= 300 ))
```

Expected: fresh highest-eligible FairShare selection and the exact pushed `PRODUCT_SHA` are recorded; all 20 `sbatch --test-only` calls exit 0; all submissions get collision-free job/result identities; six fail-closed queue/accounting/log checks span at least 300 seconds and show no disappearance, terminal failure, or startup-error signature.

- [ ] **Step 6: Wait for terminal state and validate every receipt.**

Run: `uv run python tests/test_suites/llm/validate_draft_cadence_receipts.py --root "$RESULT_ROOT" --product-sha "$(git rev-parse HEAD)" --require-terminal --require-matrix`

Expected: exits 0 only after all 20 arms are terminal GREEN; DFlash and DSpark each cover CP1 monolithic and CP2 split paths; all five behaviors have exact attempted/successful/skipped/forced accounting; the restore sequences match uninterrupted oracles; losses are finite; target refit occurs on every completed policy step.

- [ ] **Step 7: Commit only the terminal validation receipts/report.**

```bash
git add docs/superpowers/plans/2026-08-22-draft-update-cadence.md
git commit -S -s -m "docs(draft): record cadence functional validation"
git verify-commit HEAD
```

Expected: signature verification exits 0.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-22-draft-update-cadence.md`. Execute Task 1 through Task 10 in order. Use `docs/superpowers/plans/2026-08-22-draft-update-cadence-experiments.md` only after Task 10 and the exact packed functional gate are terminal GREEN.

Do not run or post `nemo-rl-pr-review` or a self-review in this execution session. After verification, hand the separate Claude Code reviewer one concise record containing: exact product SHA and recursive submodule SHAs; container digest; resolved DFlash/DSpark configs; every CPU/static/MCore/packed E2E command with terminal result; unsupported transport/topology list; and remaining correctness, performance, and rollout-provenance risks.
