from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.vllm_029_q30_dspark_adaptive.aggregate import aggregate_workers
from experiments.vllm_029_q30_dspark_adaptive.contract import (
    ExperimentContract,
    build_arms,
    worker_assignment,
)
from experiments.vllm_029_q30_dspark_adaptive.runtime import build_llm_kwargs
from experiments.vllm_029_q30_dspark_adaptive.render import (
    render_arm_sbatch,
    render_baseline_smoke_sbatch,
)
from experiments.vllm_029_q30_dspark_adaptive.prepare_dspark_overlay import (
    prepare_overlay,
)


def test_contract_matches_one_q30_gbs2048_rollout_step() -> None:
    contract = ExperimentContract()

    assert contract.vllm_version == "0.29.0"
    assert contract.account == "coreai_dlalgo_nemorl"
    assert contract.vllm_commit == "98dff2a81d747d1dba01a47f939f48c3526d4206"
    assert contract.prompt_count == 64
    assert contract.generations_per_prompt == 32
    assert contract.global_sample_count == 2_048
    assert contract.worker_count == 16
    assert contract.samples_per_worker == 128
    assert contract.max_output_tokens == 4_096
    assert contract.max_model_len == 8_192
    assert contract.max_num_seqs == 128
    assert contract.max_num_batched_tokens == 32_768
    assert contract.target_attention_backend == "FLASH_ATTN"
    assert contract.cuda_graph_mode == "FULL_AND_PIECEWISE"
    assert contract.max_cudagraph_capture_size == 1_024
    assert contract.temperature == 1.0
    assert contract.top_p == 1.0


def test_matrix_separates_fixed_k_from_dspark_adaptive_verification() -> None:
    contract = ExperimentContract()
    arms = {arm.key: arm for arm in build_arms(contract)}

    assert tuple(arms) == ("baseline", "dspark_k5", "dspark_k7", "dspark_adaptive_k7")
    assert arms["baseline"].speculative_config(contract.drafter_path) is None
    assert arms["dspark_k5"].speculative_config(contract.drafter_path) == {
        "method": "dspark",
        "model": contract.drafter_path,
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
        "attention_backend": "FLASH_ATTN",
        "draft_sample_method": "probabilistic",
        "enable_adaptive_verification": False,
    }
    assert arms["dspark_k7"].speculative_config(contract.drafter_path)[
        "num_speculative_tokens"
    ] == 7
    adaptive = arms["dspark_adaptive_k7"].speculative_config(contract.drafter_path)
    assert adaptive is not None
    assert adaptive["num_speculative_tokens"] == 7
    assert adaptive["enable_adaptive_verification"] is True
    assert "num_speculative_tokens_per_batch_size" not in adaptive


def test_worker_assignment_partitions_prompts_and_seeds_without_overlap() -> None:
    contract = ExperimentContract()
    assignments = [worker_assignment(contract, worker) for worker in range(16)]

    assert assignments[0].prompt_offset == 0
    assert assignments[-1].prompt_offset == 60
    assert assignments[0].seed == contract.base_seed
    assert assignments[-1].seed == contract.base_seed + 15 * 128
    assert {assignment.prompt_offset for assignment in assignments} == set(range(0, 64, 4))
    assert {assignment.seed for assignment in assignments} == {
        contract.base_seed + worker * 128 for worker in range(16)
    }

    with pytest.raises(ValueError, match="worker_index"):
        worker_assignment(contract, 16)


def test_runtime_kwargs_pin_fap_capacity_and_matched_backend() -> None:
    contract = ExperimentContract()
    arms = {arm.key: arm for arm in build_arms(contract)}

    baseline = build_llm_kwargs(
        contract,
        arms["baseline"],
        target_path="/raid/target",
        drafter_path="/raid/draft",
    )
    adaptive = build_llm_kwargs(
        contract,
        arms["dspark_adaptive_k7"],
        target_path="/raid/target",
        drafter_path="/raid/draft",
    )

    for kwargs in (baseline, adaptive):
        assert kwargs["attention_backend"] == "FLASH_ATTN"
        assert kwargs["max_num_seqs"] == 128
        assert kwargs["max_num_batched_tokens"] == 32_768
        assert kwargs["compilation_config"] == {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "max_cudagraph_capture_size": 1_024,
        }
        assert kwargs["kernel_config"] == {
            "moe_backend": "flashinfer_trtllm",
            "enable_flashinfer_autotune": False,
        }
    assert "speculative_config" not in baseline
    assert adaptive["speculative_config"]["enable_adaptive_verification"] is True


def _worker_payload(worker_index: int, *, output_tokens: int = 1_000) -> dict[str, object]:
    contract = ExperimentContract()
    assignment = worker_assignment(contract, worker_index)
    per_request = output_tokens // contract.samples_per_worker
    remainder = output_tokens % contract.samples_per_worker
    request_timing = [
        {
            "request_id": str(index),
            "output_tokens": per_request + (1 if index < remainder else 0),
        }
        for index in range(contract.samples_per_worker)
    ]
    return {
        "runtime_contract": {
            "worker_index": worker_index,
            "prompt_offset": assignment.prompt_offset,
            "seed": assignment.seed,
            "vllm_version": contract.vllm_version,
            "vllm_commit": contract.vllm_commit,
            "arm": "baseline",
        },
        "partial": False,
        "results": [
            {
                "step": 0,
                "num_sequences": contract.samples_per_worker,
                "wall_s": 10.0 + worker_index,
                "output_tok_s": output_tokens / (10.0 + worker_index),
                "output_lengths": {"count": 128, "total": output_tokens},
                "request_timing": request_timing,
                "spec_decode": {},
            }
        ],
    }


def test_aggregation_validates_2048_samples_and_two_token_accounts(tmp_path: Path) -> None:
    paths = []
    for worker_index in range(16):
        path = tmp_path / f"worker-{worker_index:02d}.json"
        path.write_text(json.dumps(_worker_payload(worker_index)))
        paths.append(path)

    summary = aggregate_workers(paths, expected_arm="baseline")

    assert summary["status"] == "complete"
    assert summary["expected_samples"] == 2_048
    assert summary["actual_samples"] == 2_048
    assert summary["expected_output_tokens"] == 16_000
    assert summary["actual_output_tokens"] == 16_000
    assert summary["tokens_ok"] is True
    assert summary["barrier_seconds"] == 25.0
    assert summary["output_tokens_per_second"] == 640.0


def test_aggregation_fails_closed_on_missing_worker_or_token_mismatch(tmp_path: Path) -> None:
    paths = []
    for worker_index in range(15):
        path = tmp_path / f"worker-{worker_index:02d}.json"
        path.write_text(json.dumps(_worker_payload(worker_index)))
        paths.append(path)

    with pytest.raises(ValueError, match="exactly 16"):
        aggregate_workers(paths, expected_arm="baseline")

    mismatch = _worker_payload(15)
    mismatch["results"][0]["request_timing"][0]["output_tokens"] += 1  # type: ignore[index]
    path = tmp_path / "worker-15.json"
    path.write_text(json.dumps(mismatch))
    paths.append(path)
    with pytest.raises(ValueError, match="token accounting"):
        aggregate_workers(paths, expected_arm="baseline")


def test_dspark_overlay_enables_the_observed_2304_wide_confidence_head(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "dspark"
    checkpoint.mkdir()
    source_config = {
        "architectures": ["Qwen3DSparkModel"],
        "hidden_size": 2_048,
        "markov_rank": 256,
        "dflash_config": {"use_confidence_head": True},
    }
    (checkpoint / "config.json").write_text(json.dumps(source_config))
    header = {
        "confidence_head.proj.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "confidence_head.proj.weight": {
            "dtype": "F32",
            "shape": [1, 2_304],
            "data_offsets": [4, 9_220],
        },
    }
    encoded = json.dumps(header).encode("utf-8")
    (checkpoint / "model.safetensors").write_bytes(
        len(encoded).to_bytes(8, "little") + encoded + b"0" * 9_220
    )

    receipt = prepare_overlay(checkpoint)

    assert json.loads((checkpoint / "config.json").read_text()) == {
        **source_config,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    assert receipt["confidence_head_width"] == 2_304
    assert receipt["confidence_head_with_markov"] is True


def test_dspark_overlay_rejects_a_confidence_shape_that_matches_neither_mode(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "dspark"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3DSparkModel"],
                "hidden_size": 2_048,
                "markov_rank": 256,
                "dflash_config": {"use_confidence_head": True},
            }
        )
    )
    header = {
        "confidence_head.proj.weight": {
            "dtype": "F32",
            "shape": [1, 999],
            "data_offsets": [0, 3_996],
        }
    }
    encoded = json.dumps(header).encode("utf-8")
    (checkpoint / "model.safetensors").write_bytes(
        len(encoded).to_bytes(8, "little") + encoded + b"0" * 3_996
    )

    with pytest.raises(ValueError, match="confidence head width"):
        prepare_overlay(checkpoint)


def test_renderer_emits_one_independent_four_node_barrier_job(tmp_path: Path) -> None:
    contract = ExperimentContract()
    arm = next(arm for arm in build_arms() if arm.key == "dspark_adaptive_k7")
    script = render_arm_sbatch(
        contract,
        arm,
        source_root="/home/sna/q30-vllm029",
        source_commit="a" * 40,
        container_image="/lustre/containers/vllm029.sqsh",
        result_dir="/lustre/results/adaptive",
    )
    path = tmp_path / "adaptive.sbatch"
    path.write_text(script)

    assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0
    assert "#SBATCH --nodes=4" in script
    assert "#SBATCH --gpus-per-node=4" in script
    assert "#SBATCH --account=coreai_dlalgo_nemorl" in script
    assert "#SBATCH --ntasks-per-node=4" in script
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --segment=4" in script
    assert "--ntasks=16" in script
    assert "--ntasks-per-node=4" in script
    assert "CUDA_VISIBLE_DEVICES=\"${SLURM_LOCALID}\"" in script
    assert "experiments.vllm_029_q30_dspark_adaptive.runtime" in script
    assert "experiments.vllm_029_q30_dspark_adaptive.aggregate" in script
    assert "prepare_dspark_overlay" in script
    assert contract.target_path in script
    assert contract.drafter_path in script
    assert "python3 -c" in script
    assert "python3 -m experiments.vllm_029_q30_dspark_adaptive.runtime" in script
    assert "python -m" not in script
    assert 'cp -aL "${TARGET_SOURCE}/." "${NODE_TARGET}/"' in script
    assert 'cp -aL "${DRAFTER_SOURCE}/." "${NODE_DRAFTER}/"' in script
    for cache_variable in (
        "TMPDIR",
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        assert f"export {cache_variable}=" in script
    assert "q30-vllm029-${SLURM_JOB_ID}" in script
    assert 'worker=$(printf "%02d" "${SLURM_PROCID}")' in script
    assert 'runtime-${worker}' in script
    assert "export CUDA_HOME=/usr/local/cuda-13.0" in script
    assert 'test -x "${CUDA_HOME}/bin/ptxas"' in script
    assert "--dependency" not in script

    probe_match = re.search(r"bash -lc (.+)\n\nexport RESULT_DIR", script)
    assert probe_match is not None
    bash_lc_argv = shlex.split(probe_match.group(1))
    assert len(bash_lc_argv) == 1
    python_argv = shlex.split(bash_lc_argv[0])
    assert python_argv[:2] == ["python3", "-c"]
    assert "vllm.__version__ == '0.29.0'" in python_argv[2]


def test_baseline_smoke_renderer_uses_one_worker_with_full_worker_contract(
    tmp_path: Path,
) -> None:
    contract = ExperimentContract()
    script = render_baseline_smoke_sbatch(
        contract,
        source_root="/home/sna/q30-vllm029",
        source_commit="a" * 40,
        container_image="/lustre/containers/vllm029.sqsh",
        result_dir="/lustre/results/baseline-smoke",
    )
    path = tmp_path / "baseline-smoke.sbatch"
    path.write_text(script)

    assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=4" in script
    assert "#SBATCH --account=coreai_dlalgo_nemorl" in script
    assert "--arm baseline" in script
    assert "--worker-index 0" in script
    assert "worker-00.json" in script
    assert "python3 -m experiments.vllm_029_q30_dspark_adaptive.runtime" in script
    assert "python -m" not in script
    assert 'cp -aL "${TARGET_SOURCE}/." "${NODE_TARGET}/"' in script
    assert "experiments.vllm_029_q30_dspark_adaptive.aggregate" not in script
    export = "export SOURCE_ROOT RESULT_DIR NODE_TARGET NODE_DRAFTER NODE_PROMPTS"
    assert export in script
    assert script.index(export) < script.index("srun --nodes=1")


def test_container_smoke_launcher_uses_available_python3_entrypoint() -> None:
    script = Path(
        "experiments/vllm_029_q30_dspark_adaptive/"
        "smoke_vllm029_container.sbatch"
    ).read_text()

    assert "python3 -m experiments.vllm_029_q30_dspark_adaptive.smoke" in script
    assert "python -m" not in script
    assert "export CUDA_HOME=/usr/local/cuda-13.0" in script
    assert 'test -x "${CUDA_HOME}/bin/ptxas"' in script


def test_renderer_cli_emits_canary_and_all_four_matrix_arms(tmp_path: Path) -> None:
    output_dir = tmp_path / "rendered"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.vllm_029_q30_dspark_adaptive.render",
            "--source-root",
            "/home/sna/q30-vllm029",
            "--source-commit",
            "a" * 40,
            "--container-image",
            "/lustre/containers/vllm029.sqsh",
            "--result-root",
            "/lustre/results/vllm029",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    assert {path.name for path in output_dir.glob("*.sbatch")} == {
        "baseline_smoke.sbatch",
        "adaptive_smoke.sbatch",
        "baseline.sbatch",
        "dspark_k5.sbatch",
        "dspark_k7.sbatch",
        "dspark_adaptive_k7.sbatch",
    }
    for path in output_dir.glob("*.sbatch"):
        assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0

    adaptive_smoke = (output_dir / "adaptive_smoke.sbatch").read_text()
    assert "#SBATCH --nodes=1" in adaptive_smoke
    assert "--arm dspark_adaptive_k7" in adaptive_smoke
    assert "prepare_dspark_overlay" in adaptive_smoke
    assert "worker-00.json" in adaptive_smoke
