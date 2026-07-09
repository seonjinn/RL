# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Any

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl


class _GenerationCaptured(RuntimeError):
    pass


class _CaptureLLM:
    def __init__(self) -> None:
        self.sampling_params: Any = None

    def generate(self, _prompts: Any, sampling_params: Any, **_kwargs: Any) -> None:
        self.sampling_params = sampling_params
        raise _GenerationCaptured


def _make_worker() -> Any:
    worker: Any = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker._refit_failure_reason = None
    worker.cfg = {
        "_pad_token_id": 0,
        "ignore_eos": False,
        "max_new_tokens": 4,
        "stop_strings": ["GLOBAL"],
        "stop_token_ids": [9],
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
        "vllm_cfg": {"async_engine": False, "use_tqdm": False},
    }
    worker.SamplingParams = lambda **kwargs: kwargs
    worker.llm = _CaptureLLM()
    return worker


def test_sync_generate_preserves_per_sample_stop_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    worker = _make_worker()
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 0], [2, 3]], dtype=torch.long),
            "input_lengths": torch.tensor([1, 2], dtype=torch.long),
            "stop_strings": [["ALPHA"], ["BETA"]],
        }
    )

    with pytest.raises(_GenerationCaptured):
        worker.generate(data)

    sampling_params = worker.llm.sampling_params
    assert isinstance(sampling_params, list)
    assert len(sampling_params) == 2
    assert set(sampling_params[0]["stop"]) == {"GLOBAL", "ALPHA"}
    assert set(sampling_params[1]["stop"]) == {"GLOBAL", "BETA"}


def test_sync_generate_text_preserves_per_prompt_stop_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    worker = _make_worker()
    data = BatchedDataDict(
        {
            "prompts": ["first", "second"],
            "stop_strings": [["ALPHA"], ["BETA"]],
        }
    )

    with pytest.raises(_GenerationCaptured):
        worker.generate_text(data)

    sampling_params = worker.llm.sampling_params
    assert isinstance(sampling_params, list)
    assert len(sampling_params) == 2
    assert set(sampling_params[0]["stop"]) == {"GLOBAL", "ALPHA"}
    assert set(sampling_params[1]["stop"]) == {"GLOBAL", "BETA"}


def test_stop_string_merge_is_stable_and_deduplicated() -> None:
    worker = _make_worker()
    worker.cfg["stop_strings"] = ["GLOBAL", "ALPHA"]

    assert worker._merge_stop_strings([["ALPHA", "BETA"]]) == [
        "GLOBAL",
        "ALPHA",
        "BETA",
    ]


def test_runtime_stop_strings_require_initialized_tokenizer() -> None:
    worker = _make_worker()
    worker.cfg["stop_strings"] = None
    worker.cfg["vllm_cfg"]["skip_tokenizer_init"] = True

    with pytest.raises(ValueError, match="skip_tokenizer_init=false"):
        worker._merge_stop_strings([["ALPHA"]])
