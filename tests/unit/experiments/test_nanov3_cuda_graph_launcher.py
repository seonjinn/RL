"""Contract tests for the persisted NanoV3 CUDA Graph experiment launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[3]
LAUNCHER = PROJECT_ROOT / "experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh"


def _run_launcher(scope_case: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"DRY_RUN": "1", "SCOPE_CASE": scope_case}
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_nanov3_launcher_resolves_packed_attn_router_scope() -> None:
    result = _run_launcher("attn-moe-router")

    assert result.returncode == 0, result.stderr
    assert "cuda_graph_scope=[attn,moe_router]" in result.stdout
    assert "cuda_graph_packed_seq=true" in result.stdout
    assert "cuda_graph_warmup_steps=3" in result.stdout


def test_nanov3_launcher_rejects_moe_act_as_a_graph_scope() -> None:
    result = _run_launcher("moe-act")

    assert result.returncode == 2
    assert "moe_act is an activation-recompute module, not a CUDA Graph scope" in result.stderr
