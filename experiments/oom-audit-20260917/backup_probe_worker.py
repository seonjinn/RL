"""Isolated CPU backup A/B; never selected by a production recipe."""

import json
import os
import time

import torch
import vllm.device_allocator.cumem as cumem
from vllm.v1.worker.gpu_worker import Worker

from backup_mode import pageable_backups
from cumem_probe_worker import ProbeWorker, log_cpu_backups


class BackupProbeWorker(ProbeWorker):
    def parameter_samples(self) -> torch.Tensor:
        values = []
        for parameter in self.model_runner.model.parameters():
            if parameter.numel() and parameter.is_cuda:
                values.append(parameter[(0,) * parameter.ndim].detach())
                values.append(
                    parameter[tuple(size - 1 for size in parameter.shape)].detach()
                )
        if not values:
            raise RuntimeError("No CUDA parameters sampled")
        return torch.stack(values).cpu().view(torch.uint8)

    def sleep(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("Backup probe supports level 1 only")
        mode = os.environ["AUDIT_BACKUP_MODE"]
        if mode not in ("pinned", "pageable"):
            raise ValueError(f"Invalid backup mode: {mode}")
        self.backup_reference_samples = self.parameter_samples()
        started = time.monotonic()
        with pageable_backups(cumem, mode == "pageable"):
            Worker.sleep(self, level)
        elapsed = time.monotonic() - started
        print(
            "BACKUP_TIMING "
            + json.dumps({"phase": "sleep", "mode": mode, "seconds": elapsed}),
            flush=True,
        )
        log_cpu_backups("after_sleep")

    def wake_up(self, tags: list[str] | None = None) -> None:
        if tags is not None:
            raise ValueError("Backup probe requires full wake")
        started = time.monotonic()
        Worker.wake_up(self, tags)
        # Pageable H2D cudaMemcpy can return before device DMA finishes.
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        restored = self.parameter_samples()
        if not torch.equal(self.backup_reference_samples, restored):
            raise RuntimeError("Restored parameter samples differ")
        print(
            "BACKUP_TIMING "
            + json.dumps(
                {
                    "phase": "wake",
                    "mode": os.environ["AUDIT_BACKUP_MODE"],
                    "seconds": elapsed,
                    "sample_check": "pass",
                    "sample_bytes": restored.numel(),
                }
            ),
            flush=True,
        )
        log_cpu_backups("after_wake")
