from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from frozen_drafter_sleep_policy import (  # noqa: E402
    DRAFT_WEIGHT_TAG,
    draft_weight_tag,
    frozen_drafter_sleep_enabled,
    sleep_offload_tags,
    wake_tags,
)


class FakeAllocator:
    def __init__(self) -> None:
        self.current_tag = "weights"


class FrozenDrafterSleepPolicyTest(unittest.TestCase):
    def test_enabled_level_one_sleep_discards_refittable_target_only(self) -> None:
        self.assertEqual(sleep_offload_tags(level=1, enabled=True), (DRAFT_WEIGHT_TAG,))
        self.assertEqual(sleep_offload_tags(level=2, enabled=True), ())

    def test_disabled_policy_preserves_stock_vllm_sleep_contract(self) -> None:
        self.assertEqual(sleep_offload_tags(level=1, enabled=False), ("weights",))
        self.assertEqual(sleep_offload_tags(level=2, enabled=False), ())

    def test_target_weight_wake_also_restores_frozen_drafter(self) -> None:
        requested = ["weights"]

        self.assertEqual(
            wake_tags(requested, enabled=True),
            ["weights", DRAFT_WEIGHT_TAG],
        )
        self.assertEqual(requested, ["weights"])
        self.assertEqual(wake_tags(["kv_cache"], enabled=True), ["kv_cache"])
        self.assertIsNone(wake_tags(None, enabled=True))

    def test_draft_allocations_are_tagged_and_tag_is_restored_on_error(self) -> None:
        allocator = FakeAllocator()

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with draft_weight_tag(allocator, enabled=True):
                self.assertEqual(allocator.current_tag, DRAFT_WEIGHT_TAG)
                raise RuntimeError("stop")

        self.assertEqual(allocator.current_tag, "weights")

    def test_environment_gate_is_explicit(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(frozen_drafter_sleep_enabled())
        with patch.dict(
            os.environ,
            {"NRL_FROZEN_DRAFTER_DISCARD_REFIT_TARGET": "1"},
            clear=True,
        ):
            self.assertTrue(frozen_drafter_sleep_enabled())


if __name__ == "__main__":
    unittest.main()
