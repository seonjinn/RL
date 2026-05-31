"""Auto-register Qwen3 MoE support with the container Megatron-Bridge.

Python imports ``sitecustomize`` at interpreter startup when this directory is
on ``PYTHONPATH``.  The rollout launcher prepends this directory so the
registration happens before NeMo-RL imports ``AutoBridge``.
"""

from __future__ import annotations

import os
import sys
import traceback


def _enabled(value: str | None) -> bool:
    return value is None or value.lower() not in {"0", "false", "no", "off"}


if _enabled(os.environ.get("MEGATRON_BRIDGE_QWEN3MOE_PLUGIN")):
    try:
        import qwen3_moe_bridge_plugin  # noqa: F401
    except Exception as exc:  # pragma: no cover - startup diagnostic path
        if _enabled(os.environ.get("MEGATRON_BRIDGE_QWEN3MOE_PLUGIN_STRICT")):
            raise
        sys.stderr.write(
            "[qwen3moe-bridge-plugin] registration skipped: "
            f"{type(exc).__name__}: {exc}\n"
        )
        if _enabled(os.environ.get("MEGATRON_BRIDGE_QWEN3MOE_PLUGIN_TRACEBACK")):
            traceback.print_exc()
