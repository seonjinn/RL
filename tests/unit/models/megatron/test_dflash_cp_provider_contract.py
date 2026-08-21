from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _function(
    tree: ast.AST,
    *,
    class_name: str | None,
    function_name: str,
) -> ast.FunctionDef:
    nodes = tree.body if isinstance(tree, ast.Module) else []
    if class_name is not None:
        cls = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = cls.body
    return next(
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_provider_threads_cp_local_layout_and_groups() -> None:
    training = ast.parse(
        (_REPO_ROOT / "nemo_rl/models/megatron/draft/training.py").read_text()
    )
    train = ast.parse((_REPO_ROOT / "nemo_rl/models/megatron/train.py").read_text())

    protocol_forward = _function(
        training,
        class_name="DraftTrainingProvider",
        function_name="forward",
    )
    protocol_parameters = {
        argument.arg for argument in protocol_forward.args.kwonlyargs
    }
    assert {
        "input_ids_cp_local",
        "sequence_layout",
        "context_parallel_group",
        "tensor_parallel_group",
    } <= protocol_parameters

    forward_driver = _function(
        train,
        class_name=None,
        function_name="forward_with_post_processing_fn",
    )
    provider_calls = [
        node
        for node in ast.walk(forward_driver)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "forward"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "draft_provider"
    ]
    assert len(provider_calls) == 1
    keyword_values = {
        keyword.arg: keyword.value for keyword in provider_calls[0].keywords
    }
    assert ast.unparse(keyword_values["input_ids_cp_local"]) == "input_ids_cp_sharded"
    assert (
        ast.unparse(keyword_values["sequence_layout"])
        == "processed_mb.draft_sequence_layout"
    )
    assert ast.unparse(keyword_values["context_parallel_group"]) == (
        "get_context_parallel_group()"
    )
    assert ast.unparse(keyword_values["tensor_parallel_group"]) == (
        "get_tensor_model_parallel_group()"
    )


def test_dflash_allows_target_sp_but_keeps_draft_sp_disabled() -> None:
    training = ast.parse(
        (_REPO_ROOT / "nemo_rl/models/megatron/draft/training.py").read_text()
    )
    build_model = _function(
        training,
        class_name="DFlashSpeculator",
        function_name="build_model",
    )
    source = ast.unparse(build_model)

    assert "if model_provider.sequence_parallel" not in source
    sequence_parallel_keywords = [
        keyword
        for node in ast.walk(build_model)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "sequence_parallel"
    ]
    assert len(sequence_parallel_keywords) == 1
    assert isinstance(sequence_parallel_keywords[0].value, ast.Constant)
    assert sequence_parallel_keywords[0].value.value is False


def test_dflash_body_receives_cp_layout_and_group() -> None:
    training = ast.parse(
        (_REPO_ROOT / "nemo_rl/models/megatron/draft/training.py").read_text()
    )
    forward = _function(
        training,
        class_name="DFlashSpeculator",
        function_name="forward",
    )
    calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "draft_model"
    ]
    assert len(calls) == 1
    keyword_values = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.unparse(keyword_values["sequence_layout"]) == "sequence_layout"
    assert (
        ast.unparse(keyword_values["context_parallel_group"])
        == "context_parallel_group"
    )


def test_draft_loss_wraps_packed_and_unpacked_policy_losses() -> None:
    train = ast.parse((_REPO_ROOT / "nemo_rl/models/megatron/train.py").read_text())
    loss_call = _function(
        train,
        class_name="LossPostProcessor",
        function_name="__call__",
    )
    packing_branch = next(
        node
        for node in loss_call.body
        if isinstance(node, ast.If) and "pack_sequences" in ast.unparse(node.test)
    )
    draft_wrappers = [
        node
        for node in ast.walk(loss_call)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DraftLossWrapper"
    ]

    assert len(draft_wrappers) == 1
    assert draft_wrappers[0].lineno > packing_branch.end_lineno
