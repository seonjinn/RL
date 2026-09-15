# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the checkpoint entry block without importing GPU/Ray dependencies."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest


def checkpoint_entry() -> object:
    source = Path(__file__).resolve().parents[2] / "nemo_rl/algorithms/grpo_sync.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if "should_save_by_timeout" not in ast.unparse(node.test):
            continue
        for index, statement in enumerate(node.body):
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "grpo_save_state"
                for target in statement.targets
            ):
                entry = ast.Module(body=node.body[:index], type_ignores=[])
                return compile(entry, str(source), "exec")
    raise AssertionError("Checkpoint preparation block not found")


class CheckpointTransitionTest(unittest.TestCase):
    def run_entry(self, *, colocated: bool, awake: bool, sleep_ok: bool) -> list[str]:
        events: list[str] = []

        def finish_generation() -> bool:
            nonlocal awake
            events.append("sleep")
            if sleep_ok:
                awake = False
            return sleep_ok

        def prepare_for_training() -> None:
            if colocated and awake:
                self.fail("Optimizer onload overlaps resident generation memory")
            events.append("optimizer_onload")

        exec(checkpoint_entry(), {
            "colocated_inference": colocated,
            "policy_generation": SimpleNamespace(finish_generation=finish_generation),
            "policy": SimpleNamespace(prepare_for_training=prepare_for_training),
            "timer": SimpleNamespace(time=lambda name: nullcontext()),
        })
        return events

    def test_checkpoint_after_refit_releases_generation_before_optimizer(self) -> None:
        self.assertEqual(
            self.run_entry(colocated=True, awake=True, sleep_ok=True),
            ["sleep", "optimizer_onload"],
        )

    def test_already_sleeping_generation_is_safe(self) -> None:
        self.assertEqual(
            self.run_entry(colocated=True, awake=False, sleep_ok=True),
            ["sleep", "optimizer_onload"],
        )

    def test_failed_release_stops_before_optimizer_onload(self) -> None:
        with self.assertRaises(RuntimeError):
            self.run_entry(colocated=True, awake=True, sleep_ok=False)

    def test_noncolocated_checkpoint_does_not_sleep_generation(self) -> None:
        self.assertEqual(
            self.run_entry(colocated=False, awake=True, sleep_ok=False),
            ["optimizer_onload"],
        )


if __name__ == "__main__":
    unittest.main()
