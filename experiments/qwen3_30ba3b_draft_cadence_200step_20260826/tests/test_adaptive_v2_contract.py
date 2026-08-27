"""Contracts for the Qwen3-30B-A3B Adaptive-v2 input configs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EXPERIMENT = "qwen3_30ba3b_draft_cadence_200step_20260826"
ADAPTIVE_V2_SCHEDULE = {
    "mode": "adaptive",
    "action": "sparse_update",
    "min_interval": 10,
    "max_interval": 40,
    "ewma_alpha": 0.2,
    "degradation_threshold": 0.03,
    "recovery_threshold": 0.01,
    "min_observations": 10,
    "max_burst_updates": 2,
}


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def adaptive_schedule_config_class() -> type:
    """Load the standalone schedule schema without optional model dependencies."""
    schema_path = (
        Path(__file__).resolve().parents[3] / "nemo_rl/models/policy/draft_config.py"
    )
    spec = importlib.util.spec_from_file_location("draft_config", schema_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdaptiveDraftUpdateScheduleConfig


def config_for(variant: str) -> dict[str, object]:
    return json.loads((experiment_root() / "configs" / f"{variant}.yaml").read_text())


@pytest.mark.parametrize("drafter", ("dflash", "dspark"))
def test_adaptive_v2_changes_only_the_fixed10_update_schedule(drafter: str) -> None:
    """Catch adaptive inputs that change any matched fixed-10 workload field."""
    fixed10 = config_for(f"{drafter}-fixed10")
    adaptive_v2 = config_for(f"{drafter}-adaptive-v2")

    fixed10["policy"]["draft"]["update_schedule"] = None
    adaptive_v2["policy"]["draft"]["update_schedule"] = None

    assert adaptive_v2 == fixed10


@pytest.mark.parametrize("drafter", ("dflash", "dspark"))
def test_adaptive_v2_schedule_is_exact_and_pydantic_valid(drafter: str) -> None:
    """Catch a schedule literal that is not the selected validated Adaptive-v2 policy."""
    schedule = config_for(f"{drafter}-adaptive-v2")["policy"]["draft"][
        "update_schedule"
    ]

    assert schedule == ADAPTIVE_V2_SCHEDULE
    pytest.importorskip("pydantic")
    schedule_config = adaptive_schedule_config_class().model_validate(schedule)
    assert schedule_config.model_dump() == ADAPTIVE_V2_SCHEDULE
