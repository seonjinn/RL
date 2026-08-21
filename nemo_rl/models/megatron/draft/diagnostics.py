from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DraftUpdateProbe:
    before: tuple[float, float]
    grad_l2: float


@dataclass(frozen=True)
class DraftUpdateResult:
    before: tuple[float, float]
    after: tuple[float, float]
    grad_l2: float
    checksum_delta: float


def format_draft_update_probe(result: DraftUpdateResult) -> str:
    return (
        f"draft_update_probe=complete grad_l2={result.grad_l2:.17g} "
        f"checksum_sum_before={result.before[0]:.17g} "
        f"checksum_sum_after={result.after[0]:.17g} "
        f"checksum_l2_before={result.before[1]:.17g} "
        f"checksum_l2_after={result.after[1]:.17g} "
        f"delta={result.checksum_delta:.17g}"
    )


@torch.no_grad()
def _module_statistics(
    module: nn.Module,
    *,
    include_gradients: bool,
) -> tuple[float, float, float]:
    parameters = list(module.parameters())
    if not parameters:
        return 0.0, 0.0, 0.0

    statistics_by_device: dict[
        torch.device,
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
    ] = {}
    for parameter in parameters:
        detached = parameter.detach()
        value_sums, parameter_l2_squares, _ = statistics_by_device.setdefault(
            detached.device,
            ([], [], []),
        )
        value_sums.append(detached.sum(dtype=torch.float64))
        parameter_l2_squares.append(
            torch.linalg.vector_norm(detached).double().square()
        )
        if not include_gradients:
            continue
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is None:
            continue
        _, _, gradient_l2_squares = statistics_by_device.setdefault(
            gradient.device,
            ([], [], []),
        )
        gradient_l2_squares.append(
            torch.linalg.vector_norm(gradient.detach()).double().square()
        )

    statistics = [0.0, 0.0, 0.0]
    for device, (
        value_sums,
        parameter_l2_squares,
        gradient_l2_squares,
    ) in statistics_by_device.items():
        zero = torch.zeros((), dtype=torch.float64, device=device)
        device_statistics = torch.stack(
            (
                torch.stack(value_sums).sum() if value_sums else zero,
                (
                    torch.stack(parameter_l2_squares).sum()
                    if parameter_l2_squares
                    else zero
                ),
                (
                    torch.stack(gradient_l2_squares).sum()
                    if gradient_l2_squares
                    else zero
                ),
            )
        )
        for index, value in enumerate(device_statistics.cpu().tolist()):
            statistics[index] += value
    return statistics[0], statistics[1], statistics[2] ** 0.5


def start_draft_update_probe(module: nn.Module) -> DraftUpdateProbe:
    value_sum, l2_sum, grad_l2 = _module_statistics(
        module,
        include_gradients=True,
    )
    return DraftUpdateProbe(
        before=(value_sum, l2_sum),
        grad_l2=grad_l2,
    )


def finalize_draft_update_probe(
    module: nn.Module, probe: DraftUpdateProbe
) -> DraftUpdateResult:
    value_sum, l2_sum, _ = _module_statistics(
        module,
        include_gradients=False,
    )
    after = (value_sum, l2_sum)
    delta = abs(after[0] - probe.before[0]) + abs(after[1] - probe.before[1])
    return DraftUpdateResult(
        before=probe.before,
        after=after,
        grad_l2=probe.grad_l2,
        checksum_delta=delta,
    )


def require_draft_update(result: DraftUpdateResult) -> None:
    if result.grad_l2 <= 0:
        raise RuntimeError("draft update probe requires a nonzero gradient")
    if result.checksum_delta <= 0 or result.before == result.after:
        raise RuntimeError("draft update probe requires a parameter change")
