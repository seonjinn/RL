"""Regression contracts for resume-time draft identity diagnostics."""

from __future__ import annotations

import ast
from pathlib import Path

import nemo_rl.weight_sync.interfaces as interfaces


ROOT = Path(__file__).resolve().parents[2]


def test_draft_state_root_mismatches_names_and_truncates_changed_roots() -> None:
    helper = getattr(interfaces, "draft_state_root_mismatches", None)
    assert helper is not None

    mismatches = helper(
        {
            "draft_model_sha256": "a" * 64,
            "draft_optimizer_sha256": "b" * 64,
        },
        {
            "draft_model_sha256": "c" * 64,
            "draft_optimizer_sha256": "d" * 64,
        },
    )

    assert mismatches == (
        "draft_model_sha256 expected=aaaaaaaaaaaa actual=cccccccccccc",
        "draft_optimizer_sha256 expected=bbbbbbbbbbbb actual=dddddddddddd",
    )


def test_resume_identity_action_only_refreshes_optimizer_only_drift() -> None:
    classify = getattr(interfaces, "classify_draft_state_resume", None)
    assert classify is not None
    expected = {
        "draft_model_sha256": "a" * 64,
        "draft_optimizer_sha256": "b" * 64,
    }

    assert classify(expected, expected) == "reuse_identity"
    assert (
        classify(
            expected,
            {
                "draft_model_sha256": "a" * 64,
                "draft_optimizer_sha256": "c" * 64,
            },
        )
        == "refresh_optimizer_identity"
    )
    assert (
        classify(
            expected,
            {
                "draft_model_sha256": "c" * 64,
                "draft_optimizer_sha256": "b" * 64,
            },
        )
        == "reject"
    )
    assert classify(expected, {"draft_model_sha256": "a" * 64}) == "reject"


def _attribute_path(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_path(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_lines(function: ast.FunctionDef, dotted_name: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _attribute_path(node.func) == dotted_name
    ]


def test_draft_receipt_is_captured_after_scheduler_and_parameter_materialization() -> (
    None
):
    worker = ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
    module = ast.parse(worker.read_text())
    functions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"train", "_finish_train_step_body"}
    }
    assert functions.keys() == {"train", "_finish_train_step_body"}

    for name, function in functions.items():
        scheduler_steps = _call_lines(function, "self.scheduler.step")
        parameter_materializations = _call_lines(
            function, "self._materialize_updated_parameters_for_draft_receipt"
        )
        receipt_captures = _call_lines(
            function, "self._maybe_capture_draft_update_receipt"
        )
        assert len(scheduler_steps) == 1, name
        assert len(parameter_materializations) == 1, name
        assert len(receipt_captures) == 1, name
        assert (
            scheduler_steps[0] < parameter_materializations[0] < receipt_captures[0]
        ), name
