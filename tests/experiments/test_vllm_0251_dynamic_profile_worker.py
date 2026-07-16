import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from experiments.vllm_0251_drafter_matrix import dynamic_profile_worker
from experiments.vllm_0251_drafter_matrix.dynamic_profile_worker import (
    G_BATCH_SIZES,
    G_DATASET_REVISION,
    G_PYTHON_BIN,
    G_VLLM_BIN,
    assemble_profile,
    build_benchmark_command,
    build_server_command,
    main,
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


def test_worker_uses_the_active_locked_vllm_environment() -> None:
    assert Path(G_VLLM_BIN) == Path(G_PYTHON_BIN).with_name("vllm")


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


@pytest.mark.parametrize(
    "batch_sizes",
    ((), (35, 34), (34, 34), (0,), (257,)),
)
def test_validate_batch_sizes_rejects_noncanonical_subsets(
    batch_sizes: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="sorted unique integers in 1..256"):
        dynamic_profile_worker.validate_batch_sizes(batch_sizes)


def test_run_k_forwards_an_explicit_batch_size_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_run_fixed_k(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(dynamic_profile_worker, "run_fixed_k", fake_run_fixed_k)

    main(
        (
            "run-k",
            "--root",
            "/profile",
            "--k",
            "3",
            "--target-snapshot",
            "/hf/target",
            "--drafter-snapshot",
            "/hf/drafter",
            "--prompt-template",
            "/repo/cot.txt",
            "--batch-sizes",
            "34",
            "35",
        )
    )

    assert calls[0][1]["batch_sizes"] == (34, 35)


def test_run_k_preserves_the_default_batch_size_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_run_fixed_k(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(dynamic_profile_worker, "run_fixed_k", fake_run_fixed_k)

    main(
        (
            "run-k",
            "--root",
            "/profile",
            "--k",
            "5",
            "--target-snapshot",
            "/hf/target",
            "--drafter-snapshot",
            "/hf/drafter",
            "--prompt-template",
            "/repo/cot.txt",
        )
    )

    assert calls[0][1]["batch_sizes"] == G_BATCH_SIZES


def test_explicit_k5_subset_runs_only_the_requested_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_calls: list[tuple[str, ...]] = []

    class _Process:
        pass

    def fake_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        benchmark_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        dynamic_profile_worker,
        "_ensure_prompt_file",
        lambda *args, **kwargs: tmp_path / "prompts.jsonl",
    )
    monkeypatch.setattr(
        dynamic_profile_worker, "_runtime_vllm_version", lambda: "0.25.1"
    )
    monkeypatch.setattr(
        dynamic_profile_worker.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )
    monkeypatch.setattr(dynamic_profile_worker, "_wait_for_server", lambda *args: None)
    monkeypatch.setattr(
        dynamic_profile_worker, "_validate_benchmark_result", lambda *args: None
    )
    monkeypatch.setattr(
        dynamic_profile_worker, "_terminate_process_group", lambda *args: None
    )
    monkeypatch.setattr(dynamic_profile_worker.subprocess, "run", fake_run)

    dynamic_profile_worker.run_fixed_k(
        tmp_path,
        5,
        G_TARGET,
        G_DRAFTER,
        tmp_path / "cot.txt",
        8100,
        batch_sizes=(34,),
    )

    assert len(benchmark_calls) == 1
    assert "bench" in benchmark_calls[0]
    assert benchmark_calls[0][benchmark_calls[0].index("--max-concurrency") + 1] == "34"
    assert benchmark_calls[0][benchmark_calls[0].index("--result-dir") + 1] == str(
        tmp_path / "k-5" / "bs-34"
    )
    assert (tmp_path / "k-5" / "server-bs-34.log").is_file()
    assert not (tmp_path / "k-5" / "server.log").exists()


def test_benchmark_validation_preserves_full_result_arrays(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    payload = {
        "completed": 680,
        "failed": 0,
        "median_itl_ms": 4.5,
        "itl": [1.0, 2.0, 3.0],
        "request_latencies": [10.0, 11.0],
        "input_lens": [128, 256],
        "output_lens": [256, 256],
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    dynamic_profile_worker._validate_benchmark_result(
        result_path, batch_size=34, version="0.25.1"
    )

    validated = json.loads(result_path.read_text(encoding="utf-8"))
    assert validated == {**payload, "vllm_version": "0.25.1"}


def test_default_k5_grid_still_collects_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess_calls: list[tuple[str, ...]] = []

    class _Process:
        pass

    def fake_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        subprocess_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        dynamic_profile_worker,
        "_ensure_prompt_file",
        lambda *args, **kwargs: tmp_path / "prompts.jsonl",
    )
    monkeypatch.setattr(
        dynamic_profile_worker, "_runtime_vllm_version", lambda: "0.25.1"
    )
    monkeypatch.setattr(
        dynamic_profile_worker.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )
    monkeypatch.setattr(dynamic_profile_worker, "_wait_for_server", lambda *args: None)
    monkeypatch.setattr(
        dynamic_profile_worker, "_validate_benchmark_result", lambda *args: None
    )
    monkeypatch.setattr(
        dynamic_profile_worker, "_terminate_process_group", lambda *args: None
    )
    monkeypatch.setattr(dynamic_profile_worker.subprocess, "run", fake_run)

    dynamic_profile_worker.run_fixed_k(
        tmp_path,
        5,
        G_TARGET,
        G_DRAFTER,
        tmp_path / "cot.txt",
        8100,
    )

    assert len(subprocess_calls) == len(G_BATCH_SIZES) + 1
    assert all("bench" in command for command in subprocess_calls[:-1])
    assert "acceptance" in subprocess_calls[-1]
    assert (tmp_path / "k-5" / "server.log").is_file()


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
                "dataset_revision": G_DATASET_REVISION,
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
    assert payload["dataset_revision"] == G_DATASET_REVISION
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
                "dataset_revision": G_DATASET_REVISION,
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
