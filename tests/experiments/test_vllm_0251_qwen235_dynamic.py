import json
from pathlib import Path

import pytest

from experiments.vllm_0251_drafter_matrix.calibrate_dynamic_sd import (
    derive_schedule,
    load_profile,
)
from experiments.vllm_0251_drafter_matrix.dynamic_profile_worker import (
    build_server_command,
)
from experiments.vllm_0251_drafter_matrix.matrix import (
    build_runtime_command as build_matrix_runtime_command,
    load_dynamic_schedule,
    resolve_run,
)
from experiments.vllm_0251_drafter_matrix.profile_dynamic_sd import (
    build_jobs,
    build_runtime_command,
    build_sbatch_command,
    get_profile_spec,
    snapshot_path,
)


def _write_qwen235_profile(path: Path) -> Path:
    batch_sizes = [1, 4, 8, 16, 32, 48, 64]
    k_values = [0, 1, 2, 3, 4, 5]
    payload = {
        "schema_version": 2,
        "calibration_status": "complete",
        "model_key": "qwen235",
        "target_revision": "8efa61729e24bd65b1d152b5ab5409052aa80e65",
        "drafter_revision": "3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87",
        "runtime_vllm": "0.25.1",
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
        "dataset_name": "OpenMathInstruct-2",
        "dataset_revision": "469216e3f46f4dacf476b382e192485ea51a143e",
        "prompt_template_sha256": "a" * 64,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_model_len": 8192,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 128,
        "profile_max_batch_size": 64,
        "enable_prefix_caching": True,
        "moe_backend": "triton",
        "cudagraph_capture_sizes": [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            192,
            256,
            320,
            384,
        ],
        "target_tensor_parallel_size": 8,
        "draft_tensor_parallel_size": 1,
        "num_batches_per_point": 20,
        "batch_sizes": batch_sizes,
        "k_values": k_values,
        "acceptance_rate_per_pos": [0.6, 0.4, 0.25, 0.15, 0.1],
        "rows": [
            {
                "batch_size": batch_size,
                "k": k,
                "median_itl_ms": 1.0 + 0.1 * k + 0.01 * batch_size,
                "completed_batches": 20,
            }
            for batch_size in batch_sizes
            for k in k_values
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_qwen235_profile_uses_matched_tp8_k0_k5_contract(tmp_path: Path) -> None:
    profile = get_profile_spec("qwen235")

    assert profile.target_repo_id == "Qwen/Qwen3-235B-A22B"
    assert profile.target_revision == "8efa61729e24bd65b1d152b5ab5409052aa80e65"
    assert (
        profile.drafter_repo_id
        == "RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3"
    )
    assert profile.drafter_revision == "3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87"
    assert profile.k_values == (0, 1, 2, 3, 4, 5)
    assert profile.batch_sizes == (1, 4, 8, 16, 32, 48, 64)
    assert (profile.nodes, profile.segment, profile.target_tensor_parallel_size) == (
        2,
        2,
        8,
    )
    assert profile.profile_max_batch_size == 64
    assert profile.max_num_seqs == 128
    assert profile.max_num_batched_tokens == 2048
    assert profile.moe_backend == "triton"
    assert 192 in profile.cudagraph_capture_sizes
    assert profile.cudagraph_capture_sizes[-2:] == (320, 384)

    jobs = build_jobs(tmp_path / "profile", profile=profile)
    assert [job.k for job in jobs] == [0, 1, 2, 3, 4, 5]
    assert [job.job_name for job in jobs] == [
        f"nemorl-qwen235-dynamicsd-k{k}" for k in range(6)
    ]


def test_qwen235_profile_commands_preserve_multinode_topology(tmp_path: Path) -> None:
    profile = get_profile_spec("qwen235")
    repo_dir = tmp_path / "repo"
    profile_root = tmp_path / "profile"
    hf_home = tmp_path / "hf"
    target = snapshot_path(hf_home, profile.target_repo_id, profile.target_revision)
    drafter = snapshot_path(hf_home, profile.drafter_repo_id, profile.drafter_revision)
    job = build_jobs(profile_root, profile=profile)[5]

    runtime = build_runtime_command(
        job,
        profile=profile,
        repo_dir=repo_dir,
        profile_root=profile_root,
        target_snapshot=target,
        drafter_snapshot=drafter,
        prompt_template=repo_dir / "examples/prompts/cot.txt",
    )
    scheduler = build_sbatch_command(
        job, profile=profile, repo_dir=repo_dir, mode="test-only"
    )

    assert "CUDA_VISIBLE_DEVICES=0,1" not in runtime
    assert (
        "PYTHONPATH=/tmp/nemorl-v0251-qwen235-dynamicsd-k5/profile/"
        f"lib/python3.13/site-packages:{repo_dir}"
    ) in runtime
    assert runtime[runtime.index("--model-key") + 1] == "qwen235"
    assert runtime[runtime.index("--target-tp") + 1] == "8"
    assert runtime[runtime.index("--max-k") + 1] == "5"
    assert runtime[runtime.index("--max-num-seqs") + 1] == "128"
    assert runtime[runtime.index("--profile-max-batch-size") + 1] == "64"
    assert runtime[runtime.index("--moe-backend") + 1] == "triton"
    assert runtime[runtime.index("--cudagraph-capture-sizes") + 1 :] == (
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
        "128",
        "192",
        "256",
        "320",
        "384",
    )
    assert "--nodes=2" in scheduler
    assert "--segment=2" in scheduler
    assert not any(part.startswith("--gres") for part in scheduler)


def test_qwen235_server_does_not_hide_ray_worker_gpus(tmp_path: Path) -> None:
    command = build_server_command(
        5,
        tmp_path / "target",
        tmp_path / "drafter",
        8100,
        served_model_name="qwen235-profile",
        target_tensor_parallel_size=8,
        max_model_len=8192,
        max_num_seqs=128,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
        distributed_executor_backend="ray",
        moe_backend="triton",
        cudagraph_capture_sizes=(
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            192,
            256,
            320,
            384,
        ),
    )
    joined = " ".join(command)

    assert "CUDA_VISIBLE_DEVICES=0,1" not in command
    assert "--tensor-parallel-size 8" in joined
    assert "--distributed-executor-backend ray" in joined
    assert "--moe-backend triton" in joined
    assert "--enable-prefix-caching" in command
    assert '"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,192,256,320,384]' in joined


def test_calibrator_accepts_complete_k0_k5_profile(tmp_path: Path) -> None:
    profile = load_profile(_write_qwen235_profile(tmp_path / "profile.json"))
    schedule = derive_schedule(profile)

    assert profile.k_values == (0, 1, 2, 3, 4, 5)
    assert profile.max_num_seqs == 128
    assert profile.profile_max_batch_size == 64
    assert profile.enable_prefix_caching is True
    assert profile.moe_backend == "triton"
    assert profile.cudagraph_capture_sizes[-2:] == (320, 384)
    assert len(profile.acceptance_rate_per_pos) == 5
    assert schedule.max_num_speculative_tokens == 5
    assert schedule.ranges[0].start_batch == 1
    assert schedule.ranges[-1].end_batch == 64


def test_calibrator_rejects_qwen235_profile_without_k5_endpoint_capture(
    tmp_path: Path,
) -> None:
    profile_path = _write_qwen235_profile(tmp_path / "profile.json")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["cudagraph_capture_sizes"].remove(384)
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing=\[384\]"):
        load_profile(profile_path)


def test_qwen235_dynamic_k5_variant_requires_range64_and_captures_to384(
    tmp_path: Path,
) -> None:
    profile = load_profile(_write_qwen235_profile(tmp_path / "profile.json"))
    schedule = derive_schedule(profile)
    schedule_path = tmp_path / "schedule.json"
    from experiments.vllm_0251_drafter_matrix.calibrate_dynamic_sd import (
        write_schedule,
    )

    write_schedule(profile, schedule, schedule_path)
    loaded = load_dynamic_schedule(schedule_path)
    run = resolve_run(
        "qwen235",
        "eagle3_thinking_dynamic_k5_cg384",
        "smoke2",
        "lyris",
        dynamic_schedule=loaded,
    )

    assert (
        "policy.generation.vllm_kwargs.compilation_config."
        "cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,192,256,320,384]"
    ) in run.hydra_overrides
    assert (
        "++policy.generation.vllm_kwargs.speculative_config."
        "num_speculative_tokens_per_batch_size=" in "\n".join(run.hydra_overrides)
    )


def test_qwen235_dynamic_schedule_transport_requires_explicit_opt_in(
    tmp_path: Path,
) -> None:
    profile = load_profile(_write_qwen235_profile(tmp_path / "profile.json"))
    schedule = derive_schedule(profile)
    schedule_path = tmp_path / "schedule.json"
    from experiments.vllm_0251_drafter_matrix.calibrate_dynamic_sd import (
        write_schedule,
    )

    write_schedule(profile, schedule, schedule_path)
    loaded = load_dynamic_schedule(schedule_path)

    with pytest.raises(ValueError, match="transport"):
        resolve_run(
            "qwen235",
            "eagle3_thinking_dynamic_k5_cg384",
            "smoke2",
            "lyris",
            dynamic_schedule=loaded,
            max_osl=32768,
        )

    transported = resolve_run(
        "qwen235",
        "eagle3_thinking_dynamic_k5_cg384",
        "smoke2",
        "lyris",
        dynamic_schedule=loaded,
        max_osl=32768,
        allow_dynamic_schedule_transport=True,
    )
    assert transported.dynamic_schedule_transport is True


def test_qwen235_dynamic_variant_rejects_schedule_beyond_active_batch64(
    tmp_path: Path,
) -> None:
    profile_path = _write_qwen235_profile(tmp_path / "profile.json")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["max_num_seqs"] = 256
    payload["profile_max_batch_size"] = 256
    payload["cudagraph_capture_sizes"] = [
        1,
        256,
        512,
        768,
        1024,
        1280,
        1536,
    ]
    payload["batch_sizes"] = [1, 4, 16, 32, 64, 128, 192, 256]
    payload["rows"] = [
        {
            "batch_size": batch_size,
            "k": k,
            "median_itl_ms": 1.0 + 0.1 * k + 0.01 * batch_size,
            "completed_batches": 20,
        }
        for batch_size in payload["batch_sizes"]
        for k in payload["k_values"]
    ]
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    profile = load_profile(profile_path)
    schedule = derive_schedule(profile)
    schedule_path = tmp_path / "schedule.json"
    from experiments.vllm_0251_drafter_matrix.calibrate_dynamic_sd import (
        write_schedule,
    )

    write_schedule(profile, schedule, schedule_path)

    with pytest.raises(ValueError, match="profiled batch size 64"):
        resolve_run(
            "qwen235",
            "eagle3_thinking_dynamic_k5_cg384",
            "smoke2",
            "lyris",
            dynamic_schedule=load_dynamic_schedule(schedule_path),
        )


def test_qwen235_dynamic_k3_optimizer_offload_ab_changes_only_offload_mode(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "qwen235-k3-schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calibration_status": "seed",
                "model_key": "qwen235",
                "target_revision": "8efa61729e24bd65b1d152b5ab5409052aa80e65",
                "drafter_revision": "3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87",
                "source_runtime_vllm": "0.25.1",
                "target_runtime_vllm": "0.25.1",
                "target_cuda_graph_mode": "FULL_AND_PIECEWISE",
                "profile_sha256": "f" * 64,
                "ranges": [[1, 64, 3]],
            }
        ),
        encoding="utf-8",
    )
    schedule = load_dynamic_schedule(schedule_path)
    runs = {
        mode: resolve_run(
            "qwen235",
            "eagle3_thinking_dynamic_k123_cg256",
            "smoke5",
            "lyris",
            dynamic_schedule=schedule,
            optimizer_offload_mode=mode,
        )
        for mode in ("pageable", "coalesced-pinned")
    }

    def without_offload_overrides(overrides: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            item
            for item in overrides
            if "policy.use_pinned_optimizer_offload" not in item
            and "policy.use_coalesced_optimizer_offload" not in item
        )

    assert without_offload_overrides(
        runs["pageable"].hydra_overrides
    ) == without_offload_overrides(runs["coalesced-pinned"].hydra_overrides)
    for mode, run in runs.items():
        command = build_matrix_runtime_command(
            run,
            tmp_path / "repo",
            tmp_path / "runs" / mode,
            f"qwen235-dynamic-k3-{mode}",
        )
        assert "NRL_REFIT_OFFLOAD_DIAGNOSTICS=1" in command
