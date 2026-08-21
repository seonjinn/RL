from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from nemo_rl.models.megatron.draft import hidden_capture
from nemo_rl.models.megatron.draft.hidden_capture import HiddenStateCapture


class _Layer(nn.Module):
    def __init__(self, layer_number: int) -> None:
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_layers=3)
        self.embedding = nn.Identity()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_Layer(1), _Layer(2), _Layer(3)])


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> HiddenStateCapture:
    monkeypatch.setattr(hidden_capture, "unwrap_model", lambda model: model)
    monkeypatch.setattr(
        hidden_capture.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        hidden_capture.parallel_state,
        "get_pipeline_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        hidden_capture.parallel_state,
        "is_pipeline_first_stage",
        lambda: True,
    )
    monkeypatch.setattr(
        hidden_capture.parallel_state,
        "is_pipeline_last_stage",
        lambda: True,
    )
    return HiddenStateCapture(_Model(), aux_layer_indices=(0, 1, 2))


def _run_capture(
    capture: HiddenStateCapture, batch_size: int
) -> tuple[Tensor, list[Tensor]]:
    embeds = torch.arange(2 * batch_size * 4, dtype=torch.float32).reshape(
        2, batch_size, 4
    )
    hidden_chunks = [embeds + offset for offset in (100.0, 200.0, 300.0)]

    capture.register_hooks()
    capture.model.embedding(embeds)
    for layer, hidden_states in zip(capture.model.decoder.layers, hidden_chunks):
        layer(hidden_states)

    return embeds, hidden_chunks


@pytest.mark.parametrize("batch_size", [1, 2])
def test_pp1_capture_fuses_outputs_into_one_backing_allocation(
    capture: HiddenStateCapture,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    clone_calls = 0
    cat_calls = 0
    original_clone = torch.Tensor.clone

    def count_clone(tensor: Tensor, *args: object, **kwargs: object) -> Tensor:
        nonlocal clone_calls
        clone_calls += 1
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", count_clone)
    embeds, hidden_chunks = _run_capture(capture, batch_size)
    expected_hidden_states = torch.cat(hidden_chunks, dim=-1)
    original_cat = torch.cat

    def count_cat(*args: object, **kwargs: object) -> Tensor:
        nonlocal cat_calls
        cat_calls += 1
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(hidden_capture.torch, "cat", count_cat)

    captured = capture.get_captured_states()

    torch.testing.assert_close(captured.inputs_embeds, embeds)
    torch.testing.assert_close(captured.hidden_states, expected_hidden_states)
    assert captured.inputs_embeds is not None
    assert captured.hidden_states is not None
    assert (
        captured.inputs_embeds.untyped_storage().data_ptr()
        == captured.hidden_states.untyped_storage().data_ptr()
    )
    assert capture.get_captured_states() is captured
    assert clone_calls == 0
    assert cat_calls == 1


def test_capture_rejects_source_tensor_modified_after_hook(
    capture: HiddenStateCapture,
) -> None:
    embeds, _ = _run_capture(capture, batch_size=1)

    embeds.add_(1)

    with pytest.raises(RuntimeError, match="modified in place"):
        capture.get_captured_states()
