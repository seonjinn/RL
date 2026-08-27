"""Contracts for the Qwen3-30B-A3B Adaptive-v2 input configs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TypeAlias, cast

import pytest
from pydantic import BaseModel

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
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


def adaptive_schedule_config_class() -> type[BaseModel]:
    """Load the standalone schedule schema without optional model dependencies."""
    schema_path = (
        Path(__file__).resolve().parents[3] / "nemo_rl/models/policy/draft_config.py"
    )
    spec = importlib.util.spec_from_file_location("draft_config", schema_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(type[BaseModel], module.AdaptiveDraftUpdateScheduleConfig)


def config_for(variant: str) -> JsonObject:
    raw_config = json.loads(
        (experiment_root() / "configs" / f"{variant}.yaml").read_text()
    )
    assert isinstance(raw_config, dict)
    assert all(isinstance(key, str) for key in raw_config)
    return cast(JsonObject, raw_config)


def draft_config(config: JsonObject) -> JsonObject:
    policy = config["policy"]
    assert isinstance(policy, dict)
    draft = policy["draft"]
    assert isinstance(draft, dict)
    return cast(JsonObject, draft)


@pytest.mark.parametrize("drafter", ("dflash", "dspark"))
def test_adaptive_v2_changes_only_the_fixed10_update_schedule(drafter: str) -> None:
    """Catch adaptive inputs that change any matched fixed-10 workload field."""
    fixed10 = config_for(f"{drafter}-fixed10")
    adaptive_v2 = config_for(f"{drafter}-adaptive-v2")

    draft_config(fixed10)["update_schedule"] = None
    draft_config(adaptive_v2)["update_schedule"] = None

    assert adaptive_v2 == fixed10


@pytest.mark.parametrize("drafter", ("dflash", "dspark"))
def test_adaptive_v2_schedule_is_exact_and_pydantic_valid(drafter: str) -> None:
    """Catch a schedule literal that is not the selected validated Adaptive-v2 policy."""
    schedule = draft_config(config_for(f"{drafter}-adaptive-v2"))["update_schedule"]
    assert isinstance(schedule, dict)

    assert schedule == ADAPTIVE_V2_SCHEDULE
    schedule_config = adaptive_schedule_config_class().model_validate(schedule)
    assert schedule_config.model_dump() == ADAPTIVE_V2_SCHEDULE
