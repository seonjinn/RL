from __future__ import annotations

import json
from collections import UserDict
from io import StringIO

import pytest

from experiments.vllm_024_upgrade.run_generation_parity import (
    GenerationSettings,
    PromptRecord,
    _apply_cleanup_outcome,
    _tokenize_prompt,
    build_generation_config,
    cleanup_runtime,
    expand_prompt_records,
    extract_batch_samples,
    extract_generated_sample,
    load_prompt_records,
    run_generation_batches,
)


def test_tokenize_prompt_accepts_mapping_style_batch_encoding() -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return UserDict({"input_ids": [101, 102]})

    assert _tokenize_prompt(Tokenizer(), "hello") == [101, 102]


def test_load_and_expand_prompt_records_preserves_stable_sample_ids(tmp_path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(
            (
                json.dumps({"id": "math-0", "prompt": "Compute 2 + 2."}),
                json.dumps({"id": "swe-0", "prompt": "Fix the parser."}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = load_prompt_records(prompt_path, limit=2)
    expanded = expand_prompt_records(prompts, samples_per_prompt=3)

    assert prompts == [
        PromptRecord(prompt_id="math-0", text="Compute 2 + 2."),
        PromptRecord(prompt_id="swe-0", text="Fix the parser."),
    ]
    assert [(item.prompt.prompt_id, item.sample_id) for item in expanded] == [
        ("math-0", "0000"),
        ("math-0", "0001"),
        ("math-0", "0002"),
        ("swe-0", "0000"),
        ("swe-0", "0001"),
        ("swe-0", "0002"),
    ]


def test_extract_generated_sample_slices_only_generated_tokens_and_logprobs() -> None:
    sample = extract_generated_sample(
        prompt_id="math-0",
        sample_id="0000",
        output_ids=[101, 102, 201, 202, 0],
        token_logprobs=[0.0, 0.0, -0.25, -0.5, 0.0],
        input_length=2,
        generation_length=2,
        truncated=True,
    )

    assert sample == {
        "prompt_id": "math-0",
        "sample_id": "0000",
        "token_ids": [201, 202],
        "token_logprobs": [-0.25, -0.5],
        "truncated": True,
    }


def test_extract_batch_samples_preserves_request_identity() -> None:
    requests = expand_prompt_records(
        [PromptRecord(prompt_id="math-0", text="Compute 2 + 2.")],
        samples_per_prompt=2,
    )

    samples = extract_batch_samples(
        requests=requests,
        input_lengths=[2, 1],
        generation_lengths=[2, 2],
        output_ids=[[101, 102, 201, 202], [111, 211, 212, 0]],
        token_logprobs=[[0.0, 0.0, -0.2, -0.3], [0.0, -0.4, -0.5, 0.0]],
        truncated=[False, True],
    )

    assert samples == [
        {
            "prompt_id": "math-0",
            "sample_id": "0000",
            "token_ids": [201, 202],
            "token_logprobs": [-0.2, -0.3],
            "truncated": False,
        },
        {
            "prompt_id": "math-0",
            "sample_id": "0001",
            "token_ids": [211, 212],
            "token_logprobs": [-0.4, -0.5],
            "truncated": True,
        },
    ]


@pytest.mark.parametrize(
    ("token_logprobs", "match"),
    [
        ([0.0, 0.0, -0.25], "shorter than"),
        ([0.0, 0.0, float("nan"), -0.5], "non-finite"),
    ],
)
def test_extract_generated_sample_rejects_invalid_behavior_data(
    token_logprobs: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        extract_generated_sample(
            prompt_id="math-0",
            sample_id="0000",
            output_ids=[101, 102, 201, 202],
            token_logprobs=token_logprobs,
            input_length=2,
            generation_length=2,
            truncated=False,
        )


def test_build_generation_config_keeps_cuda_graph_and_exact_rejection() -> None:
    settings = GenerationSettings(
        model="/models/qwen32",
        tokenizer="/models/qwen32",
        draft_model="/models/qwen32-eagle3",
        method="eagle3",
        num_speculative_tokens=5,
        target_tp=2,
        draft_tp=1,
        max_model_len=4096,
        max_new_tokens=512,
        temperature=1.0,
        top_p=1.0,
    )

    config = build_generation_config(settings)

    assert config["vllm_cfg"]["enforce_eager"] is False
    assert config["vllm_cfg"]["async_engine"] is False
    assert config["vllm_kwargs"]["compilation_config"]["cudagraph_mode"] == (
        "PIECEWISE"
    )
    assert "seed" not in config["vllm_kwargs"]
    assert config["vllm_kwargs"]["enable_chunked_prefill"] is True
    assert config["vllm_cfg"]["enable_prefix_caching"] is False
    assert "enable_prefix_caching" not in config["vllm_kwargs"]
    assert config["vllm_kwargs"]["max_num_batched_tokens"] == 16384
    worker_explicit_llm_args = {
        "model",
        "served_model_name",
        "load_format",
        "skip_tokenizer_init",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "enable_expert_parallel",
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "dtype",
        "seed",
        "enforce_eager",
        "max_model_len",
        "trust_remote_code",
        "worker_extension_cls",
        "enable_sleep_mode",
        "disable_log_stats",
        "logprobs_mode",
    }
    assert worker_explicit_llm_args.isdisjoint(config["vllm_kwargs"])
    assert config["vllm_kwargs"]["speculative_config"] == {
        "method": "eagle3",
        "model": "/models/qwen32-eagle3",
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
        "rejection_sample_method": "standard",
        "draft_sample_method": "probabilistic",
    }


def test_build_generation_config_leaves_baseline_free_of_specdec() -> None:
    settings = GenerationSettings(
        model="/models/qwen32",
        tokenizer="/models/qwen32",
        draft_model=None,
        method="eagle3",
        num_speculative_tokens=5,
        target_tp=2,
        draft_tp=1,
        max_model_len=4096,
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
    )

    config = build_generation_config(settings)

    assert "speculative_config" not in config["vllm_kwargs"]


def test_run_generation_batches_streams_usable_jsonl_rows() -> None:
    requests = expand_prompt_records(
        [PromptRecord(prompt_id="math-0", text="Compute 2 + 2.")],
        samples_per_prompt=2,
    )

    class FakePolicy:
        def __init__(self) -> None:
            self.greedy_values: list[bool] = []

        def generate(self, _batch, *, greedy: bool):
            self.greedy_values.append(greedy)
            return {
                "generation_lengths": [1, 1],
                "output_ids": [[101, 201], [101, 202]],
                "logprobs": [[0.0, -0.2], [0.0, -0.3]],
                "truncated": [False, True],
            }

    output = StringIO()
    policy = FakePolicy()

    summary = run_generation_batches(
        policy,
        requests,
        batch_size=2,
        greedy=True,
        build_batch=lambda _requests: {"input_lengths": [1, 1]},
        output_file=output,
    )

    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert policy.greedy_values == [True]
    assert [row["sample_id"] for row in rows] == ["0000", "0001"]
    assert [row["truncated"] for row in rows] == [False, True]
    assert summary["completed_samples"] == 2
    assert summary["generated_tokens"] == 2


def test_cleanup_runtime_attempts_every_resource_without_masking_failure() -> None:
    calls: list[str] = []

    class FailedPolicy:
        def shutdown(self) -> bool:
            calls.append("policy")
            return False

    class FailedCluster:
        def shutdown(self) -> None:
            calls.append("cluster")
            raise RuntimeError("cluster cleanup failed")

    class FailedRay:
        @staticmethod
        def shutdown() -> None:
            calls.append("ray")
            raise RuntimeError("ray cleanup failed")

    errors = cleanup_runtime(FailedPolicy(), FailedCluster(), FailedRay())

    assert calls == ["policy", "cluster", "ray"]
    assert errors == [
        "VllmGeneration shutdown returned false",
        "cluster shutdown: RuntimeError: cluster cleanup failed",
        "ray shutdown: RuntimeError: ray cleanup failed",
    ]


def test_cleanup_failure_invalidates_an_otherwise_passed_run() -> None:
    metadata = {"status": "passed"}

    exit_code = _apply_cleanup_outcome(
        metadata,
        ["VllmGeneration shutdown returned false"],
    )

    assert exit_code == 1
    assert metadata == {
        "status": "failed",
        "failure_stage": "cleanup",
        "cleanup_errors": ["VllmGeneration shutdown returned false"],
    }
