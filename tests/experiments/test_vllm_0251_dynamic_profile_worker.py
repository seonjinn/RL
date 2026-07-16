import json
from pathlib import Path

import pytest

from experiments.vllm_0251_drafter_matrix.dynamic_profile_worker import (
    assemble_profile,
    build_benchmark_command,
    build_server_command,
    render_math_prompt,
)


G_TARGET = Path("/lustre/hf/target/snapshots/" + "1" * 40)
G_DRAFTER = Path("/lustre/hf/drafter/snapshots/" + "2" * 40)


class _Tokenizer:
    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "add_special_tokens": False,
        }
        return f"CHAT:{messages}"


def test_render_math_prompt_matches_recipe_processor_contract() -> None:
    rendered = render_math_prompt("2+2?", "Solve carefully: {}", _Tokenizer())

    assert rendered == "CHAT:[{'role': 'user', 'content': 'Solve carefully: 2+2?'}]"


def test_server_command_matches_performance_recipe_and_k5() -> None:
    command = build_server_command(5, G_TARGET, G_DRAFTER, 8100)
    joined = " ".join(command)

    assert "CUDA_VISIBLE_DEVICES=0,1" in command
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in command
    assert "--tensor-parallel-size 2" in joined
    assert "--max-model-len 4096" in joined
    assert "--max-num-seqs 256" in joined
    assert "--max-num-batched-tokens 16384" in joined
    assert "--enable-chunked-prefill" in command
    assert "--no-enable-prefix-caching" in command
    assert '"cudagraph_mode":"FULL_AND_PIECEWISE"' in joined
    assert '"num_speculative_tokens":5' in joined
    assert '"draft_tensor_parallel_size":1' in joined
    assert str(G_DRAFTER) in joined


def test_k0_server_is_a_true_no_drafter_baseline() -> None:
    command = build_server_command(0, G_TARGET, None, 8100)

    assert "--speculative-config" not in command


def test_benchmark_command_uses_twenty_batches_and_math_sampling(
    tmp_path: Path,
) -> None:
    command = build_benchmark_command(
        batch_size=64,
        prompt_file=tmp_path / "prompts.jsonl",
        tokenizer_snapshot=G_TARGET,
        result_dir=tmp_path / "k-3" / "bs-64",
        port=8100,
    )
    joined = " ".join(command)

    assert "--num-prompts 1280" in joined
    assert "--max-concurrency 64" in joined
    assert "--output-len 256" in joined
    assert "--temperature 1.0" in joined
    assert "--top-p 1.0" in joined
    assert "--dataset-name custom" in joined
    assert f"--tokenizer {G_TARGET}" in joined
    assert "--skip-chat-template" in command


def _write_result(root: Path, k: int, batch_size: int) -> None:
    path = root / f"k-{k}" / f"bs-{batch_size}" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "completed": batch_size * 20,
                "failed": 0,
                "median_itl_ms": 10.0 + k + batch_size / 1000,
                "vllm_version": "0.25.1",
            }
        ),
        encoding="utf-8",
    )


def test_assemble_profile_requires_and_preserves_the_complete_grid(
    tmp_path: Path,
) -> None:
    batch_sizes = (1, 4)
    for k in range(6):
        for batch_size in batch_sizes:
            _write_result(tmp_path, k, batch_size)
    (tmp_path / "acceptance.json").write_text(
        json.dumps(
            {
                "num_drafts": 100,
                "acceptance_rate_per_pos": [0.9, 0.8, 0.7, 0.6, 0.5],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "prompts.meta.json").write_text(
        json.dumps(
            {
                "dataset_name": "OpenMathInstruct-2",
                "prompt_template_sha256": "3" * 64,
                "prompt_file_sha256": "4" * 64,
                "num_prompts": 5120,
            }
        ),
        encoding="utf-8",
    )

    payload = assemble_profile(
        tmp_path,
        target_revision="1" * 40,
        drafter_revision="2" * 40,
        batch_sizes=batch_sizes,
    )

    assert payload["calibration_status"] == "complete"
    assert payload["k_values"] == list(range(6))
    assert len(payload["rows"]) == 12
    assert payload["acceptance_rate_per_pos"] == [0.9, 0.8, 0.7, 0.6, 0.5]


def test_assemble_profile_fails_when_a_cell_is_missing(tmp_path: Path) -> None:
    for k in range(6):
        _write_result(tmp_path, k, 1)
    (tmp_path / "k-5" / "bs-1" / "result.json").unlink()
    (tmp_path / "acceptance.json").write_text(
        json.dumps(
            {
                "num_drafts": 100,
                "acceptance_rate_per_pos": [0.9, 0.8, 0.7, 0.6, 0.5],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "prompts.meta.json").write_text(
        json.dumps(
            {
                "dataset_name": "OpenMathInstruct-2",
                "prompt_template_sha256": "3" * 64,
                "prompt_file_sha256": "4" * 64,
                "num_prompts": 5120,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing profile result"):
        assemble_profile(
            tmp_path,
            target_revision="1" * 40,
            drafter_revision="2" * 40,
            batch_sizes=(1,),
        )
