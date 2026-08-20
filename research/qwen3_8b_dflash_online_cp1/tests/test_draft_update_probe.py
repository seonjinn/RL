import importlib.util
from pathlib import Path

import torch
import pytest

from research.qwen3_8b_dflash_online_cp1.validate_gate import (
    _validate_draft_update,
)


_PATH = Path(__file__).parents[3] / "nemo_rl/models/megatron/draft/diagnostics.py"
_SPEC = importlib.util.spec_from_file_location("draft_diagnostics", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
DraftUpdateProbe = _MODULE.DraftUpdateProbe
DraftUpdateResult = _MODULE.DraftUpdateResult
finalize_draft_update_probe = _MODULE.finalize_draft_update_probe
require_draft_update = _MODULE.require_draft_update
start_draft_update_probe = _MODULE.start_draft_update_probe


def _load_draft_config_module():
    path = Path(__file__).parents[3] / "nemo_rl/models/policy/draft_config.py"
    spec = importlib.util.spec_from_file_location("draft_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dflash_probe_is_an_explicit_opt_in() -> None:
    config_type = _load_draft_config_module().DFlashDraftConfig
    common = {
        "gamma": 5,
        "anchors_per_sample": 2,
        "mask_token_id": 151669,
        "target_hidden_state_layer_ids": [1, 9, 17, 25, 33],
    }

    assert config_type(**common).update_probe_enabled is False
    assert config_type(**common, update_probe_enabled=True).update_probe_enabled is True


def test_probe_reports_nonzero_gradient_and_parameter_change() -> None:
    module = torch.nn.Linear(3, 2, bias=False)
    module.weight.grad = torch.ones_like(module.weight)

    probe = start_draft_update_probe(module)
    assert isinstance(probe, DraftUpdateProbe)
    assert probe.grad_l2 > 0

    with torch.no_grad():
        module.weight.add_(0.25)

    result = finalize_draft_update_probe(module, probe)
    assert result.grad_l2 > 0
    assert result.checksum_delta > 0
    assert result.before != result.after


def test_probe_rejects_missing_or_unchanged_draft_updates() -> None:
    module = torch.nn.Linear(3, 2, bias=False)
    module.weight.grad = torch.zeros_like(module.weight)

    probe = start_draft_update_probe(module)
    result = finalize_draft_update_probe(module, probe)

    assert result.grad_l2 == 0
    assert result.checksum_delta == 0
    with pytest.raises(RuntimeError, match="nonzero gradient"):
        require_draft_update(result)


def test_probe_marker_preserves_small_checksum_changes() -> None:
    formatter = getattr(_MODULE, "format_draft_update_probe", None)
    assert formatter is not None
    result = DraftUpdateResult(
        before=(1.0001, 8_708_180.0),
        after=(1.0002, 8_708_180.0),
        grad_l2=0.75,
        checksum_delta=0.0001,
    )

    marker = formatter(result)

    _validate_draft_update(f"{marker}\n{marker}\n")
