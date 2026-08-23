# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Dependency-free contract test for segmented sync-GRPO lifecycle changes."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def _extract_function(
    source_path: Path, *, function_name: str, class_name: str | None = None
) -> Callable[..., Any]:
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    body: list[ast.stmt] = tree.body
    if class_name is not None:
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = class_node.body
    function_node = next(
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    function_node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[function_name]


def _assert_grpo_config_contract(grpo_path: Path) -> None:
    validate = _extract_function(
        grpo_path,
        class_name="GRPOConfig",
        function_name="validate_segment_stop_step",
    )
    for segment_stop_step in (25, 50, 75, 100):
        config = SimpleNamespace(max_num_steps=100, segment_stop_step=segment_stop_step)
        assert validate(config) is config
    for segment_stop_step in (0, 101):
        config = SimpleNamespace(max_num_steps=100, segment_stop_step=segment_stop_step)
        try:
            validate(config)
        except ValueError as error:
            assert "segment_stop_step" in str(error)
        else:
            raise AssertionError(
                f"unsafe segment_stop_step={segment_stop_step} was accepted"
            )


def _master_config(
    *,
    cadence_enabled: bool = True,
    checkpoint_enabled: bool = True,
    save_optimizer: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_plane={"enabled": True},
        policy={"draft": SimpleNamespace(enabled=True)},
        cadence_runtime=SimpleNamespace(
            enabled=cadence_enabled,
            required_checkpoint_steps=(25, 50, 75, 100),
        ),
        checkpointing={
            "enabled": checkpoint_enabled,
            "save_optimizer": save_optimizer,
        },
        grpo=SimpleNamespace(segment_stop_step=25),
    )


def _assert_master_config_contract(grpo_path: Path) -> None:
    validate = _extract_function(
        grpo_path,
        class_name="MasterConfig",
        function_name="validate_segmented_draft_lifecycle",
    )
    valid = _master_config()
    assert validate(valid) is valid
    for config, message in (
        (_master_config(cadence_enabled=False), "cadence_runtime.enabled"),
        (_master_config(checkpoint_enabled=False), "checkpointing.enabled"),
        (_master_config(save_optimizer=False), "save_optimizer"),
    ):
        try:
            validate(config)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"unsafe lifecycle missing {message} was accepted")


def _assert_segment_terminal_contract(grpo_sync_path: Path) -> None:
    is_segment_stop = _extract_function(
        grpo_sync_path, function_name="_is_segment_stop_step"
    )
    completed_step_exit_action = _extract_function(
        grpo_sync_path, function_name="_completed_step_exit_action"
    )
    completed_step_exit_action.__globals__["_is_segment_stop_step"] = is_segment_stop
    for segment_stop_step in (25, 50, 75, 100):
        assert is_segment_stop(
            next_step=segment_stop_step, segment_stop_step=segment_stop_step
        )
        assert completed_step_exit_action(
            completed_steps=segment_stop_step,
            max_num_steps=100,
            segment_stop_step=segment_stop_step,
        ) == ("terminal" if segment_stop_step == 100 else "segment")
    assert (
        completed_step_exit_action(
            completed_steps=25,
            max_num_steps=100,
            segment_stop_step=None,
        )
        is None
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    _assert_grpo_config_contract(source_root / "nemo_rl/algorithms/grpo.py")
    _assert_master_config_contract(source_root / "nemo_rl/algorithms/grpo.py")
    _assert_segment_terminal_contract(source_root / "nemo_rl/algorithms/grpo_sync.py")
    print("segment_lifecycle_contract=PASS")


if __name__ == "__main__":
    main()
