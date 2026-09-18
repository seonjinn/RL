"""Opt-in BF16 snapshot using refreshed, already allocated CPU backing."""

from collections.abc import Mapping, Sequence

import torch


def snapshot_policy_state(
    state: Mapping[str, object],
    buffers: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, object], int]:
    sources: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}
    destinations: set[int] = set()
    for source, backup in buffers:
        if (
            type(source) is not torch.Tensor
            or type(backup) is not torch.Tensor
            or source.device.type != "cuda"
            or backup.device.type != "cpu"
            or source.dtype != torch.bfloat16
            or backup.dtype != source.dtype
            or not backup.is_pinned()
            or not source.is_contiguous()
            or not backup.is_contiguous()
            or source.shape != backup.shape
            or source.untyped_storage().nbytes() < (source.storage_offset() + source.numel()) * source.element_size()
        ):
            raise ValueError("Snapshot reuse requires resident BF16 buffers and matching pinned CPU storage")
        identity = (source.device, source.untyped_storage().data_ptr())
        destination = backup.untyped_storage().data_ptr()
        if identity in sources or destination in destinations:
            raise ValueError("Snapshot buffers must have distinct backing storage")
        sources[identity] = (source, backup)
        destinations.add(destination)
    if not sources:
        raise ValueError("No reusable BF16 buffers provided")

    devices = {source.device for source, _ in buffers}
    for device in devices:
        torch.cuda.synchronize(device)
    with torch.no_grad():
        for source, backup in buffers:
            backup.copy_(source, non_blocking=False)
        saved: dict[str, object] = {}
        reused_bytes = 0
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                saved[name] = value
                continue
            pair = sources.get((value.device, value.untyped_storage().data_ptr()))
            if pair is not None and type(value) is torch.Tensor and value.dtype == torch.bfloat16 and value.numel():
                source, backup = pair
                offset = value.storage_offset() - source.storage_offset()
                span = 1 + sum((size - 1) * stride for size, stride in zip(value.shape, value.stride()))
                if offset < 0 or any(stride < 0 for stride in value.stride()) or offset + span > source.numel():
                    raise ValueError("State view exceeds reusable parameter buffer")
                saved[name] = backup.as_strided(value.shape, value.stride(), backup.storage_offset() + offset)
                reused_bytes += value.numel() * value.element_size()
            else:
                saved[name] = value.detach().to("cpu", non_blocking=False, copy=True)
    return saved, reused_bytes
