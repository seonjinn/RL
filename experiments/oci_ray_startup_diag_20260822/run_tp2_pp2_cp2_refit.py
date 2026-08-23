from __future__ import annotations

import os
import runpy
from pathlib import Path

import torch
import torch.distributed as dist

SOURCE = Path("/home/sna/pr3757-reconcile-df9daf62")
EXPECTED_SHA = "df9daf62fe4625609b3a71abd7179007cd6970f9"


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        namespace = runpy.run_path(str(SOURCE / "tests/unit/models/megatron/test_draft_refit.py"))
        for test_name in (
            "_run_tp2_pp2_cp2_worker_draft_refit",
            "_run_tp2_pp2_cp2_worker_draft_failure_consensus",
            "_run_cp_lane_manifest_mismatch",
        ):
            namespace[test_name](rank, world_size)
            dist.barrier()
        if rank == 0:
            print(
                "PR3757_TP2_PP2_CP2_PASS "
                f"sha={EXPECTED_SHA} tests=3 topology=TP2xPP2xCP2",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
