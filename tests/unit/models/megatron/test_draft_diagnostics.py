import importlib.util
import math
from pathlib import Path
import sys

import torch
from torch import nn


DIAGNOSTICS_PATH = (
    Path(__file__).parents[4] / "nemo_rl/models/megatron/draft/diagnostics.py"
)


def _diagnostics_module():
    spec = importlib.util.spec_from_file_location("draft_diagnostics", DIAGNOSTICS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scalar_sync_count(profiler: torch.profiler.profile) -> int:
    return sum(
        event.count
        for event in profiler.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )


def test_update_probe_consolidates_scalar_device_syncs() -> None:
    diagnostics = _diagnostics_module()
    module = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        probe = diagnostics.start_draft_update_probe(module)
        with torch.no_grad():
            next(module.parameters()).add_(0.25)
        result = diagnostics.finalize_draft_update_probe(module, probe)

    expected_grad_l2 = (
        sum(
            parameter.grad.double().square().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        ** 0.5
    )
    assert math.isclose(result.grad_l2, expected_grad_l2, rel_tol=1e-6)
    assert result.checksum_delta > 0
    assert _scalar_sync_count(profiler) <= 2


def test_update_probe_supports_parameters_on_multiple_devices() -> None:
    if not torch.cuda.is_available():
        return

    diagnostics = _diagnostics_module()
    module = nn.Module()
    module.cpu_parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    module.cuda_parameter = nn.Parameter(torch.tensor([3.0, 4.0], device="cuda"))
    module.cpu_parameter.main_grad = torch.ones_like(
        module.cpu_parameter,
        device="cuda",
    )
    module.cuda_parameter.grad = torch.ones_like(module.cuda_parameter)

    probe = diagnostics.start_draft_update_probe(module)
    with torch.no_grad():
        module.cpu_parameter.add_(1.0)
        module.cuda_parameter.add_(1.0)
    result = diagnostics.finalize_draft_update_probe(module, probe)

    assert math.isclose(probe.before[0], 10.0, rel_tol=1e-6)
    assert math.isclose(probe.before[1], 30.0, rel_tol=1e-6)
    assert math.isclose(probe.grad_l2, 2.0)
    assert math.isclose(result.after[0], 14.0, rel_tol=1e-6)
    assert math.isclose(result.after[1], 54.0, rel_tol=1e-6)
    assert math.isclose(result.checksum_delta, 28.0, rel_tol=1e-6)
