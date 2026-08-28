from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "experiments/vllm_028_nemotron_bf16_matrix"
EXPECTED_SHAPES = [(1000, 10000), (10000, 1000)]
EXPECTED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 128, 512]
EXPECTED_METHODS = [
    "baseline",
    "mtp_static_k1",
    "mtp_static_k2",
    "mtp_static_k3",
    "mtp_static_k5",
    "mtp_dynamic_max_k5",
]
PINNED_CONTAINER = (
    "nvcr.io/nvidia/vllm-openai:0.28.0@sha256:"
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)
SUPER_CHECKPOINT = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
    "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
    "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e"
)
ULTRA_CHECKPOINT = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
    "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
    "snapshots/624ba927cfbef0427354998700de3d51173c8c04"
)
EXPECTED_DYNAMIC_SCHEDULE = "1:4:5,5:16:3,17:64:2,65:128:1,129:512:0"


def load_module(module_name: str) -> ModuleType:
    path = PACKAGE_ROOT / f"{module_name}.py"
    assert path.is_file(), f"vLLM 0.28 Nemotron BF16 matrix module is not implemented: {path}"
    spec = importlib.util.spec_from_file_location(
        f"vllm028_nemotron_bf16_{module_name}",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_contract_matrix() -> dict[str, Any]:
    return load_module("contract").build_contract_matrix()


def build_submission_plan(
    *,
    data_parallel_size: int = 1,
    container_image: str = PINNED_CONTAINER,
) -> list[dict[str, Any]]:
    contract = build_contract_matrix()
    return load_module("launcher").build_submission_plan(
        contract,
        data_parallel_size=data_parallel_size,
        container_image=container_image,
    )


def assert_arg_value(argv: list[str], flag: str, expected: str) -> None:
    assert argv[argv.index(flag) + 1] == expected


def legacy_args(
    *,
    model_key: str = "super",
    runner_key: str = "mrv1",
    method_key: str = "baseline",
    batch_sizes: list[int] | None = None,
    max_num_seqs: int = 512,
    speculative_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_key": model_key,
        "runner_key": runner_key,
        "method_key": method_key,
        "isl": 1000,
        "osl": 10000,
        "batch_sizes": batch_sizes or [1, 32, 128],
        "max_num_seqs": max_num_seqs,
        "speculative_config": speculative_config,
        "result_path": "/workspace/results/result.json",
    }


def test_build_contract_matrix_enumerates_only_the_required_shapes_and_batch_sizes() -> None:
    matrix = build_contract_matrix()

    assert matrix["schema_version"] == 1
    assert [(shape["isl"], shape["osl"]) for shape in matrix["shapes"]] == EXPECTED_SHAPES
    assert matrix["batch_sizes"] == EXPECTED_BATCH_SIZES


def test_build_contract_matrix_pins_models_methods_and_dp1_topology() -> None:
    matrix = build_contract_matrix()
    models = {model["key"]: model for model in matrix["models"]}

    assert matrix["method_order"] == EXPECTED_METHODS
    assert set(models) == {"super", "ultra"}
    assert models["super"]["runtime_topology"] == {
        "tensor_parallel_size": 2,
        "node_count": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
    }
    assert models["ultra"]["runtime_topology"] == {
        "tensor_parallel_size": 8,
        "node_count": 2,
        "data_parallel_size": 1,
        "enable_expert_parallel": True,
    }


def test_build_contract_matrix_records_required_cuda_graph_gates() -> None:
    matrix = build_contract_matrix()

    assert matrix["cuda_graph_modes"] == {
        "mrv1_gate": "PIECEWISE",
        "mrv2_canary": "FULL_AND_PIECEWISE",
    }


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        ("2:512:5", "start"),
        ("1:8:5,10:512:4", "contiguous"),
        ("1:256:5", "512"),
        ("1:512:6", "max"),
    ],
)
def test_validate_dynamic_schedule_requires_gap_free_coverage_from_1_through_512(
    schedule: str,
    message: str,
) -> None:
    contract = load_module("contract")

    with pytest.raises(ValueError, match=message):
        contract.validate_dynamic_schedule(schedule, max_k=5)


def test_validate_dynamic_schedule_accepts_k0_rows_within_max_k() -> None:
    contract = load_module("contract")

    rows = contract.validate_dynamic_schedule(
        "1:1:5,2:4:3,5:127:1,128:512:0",
        max_k=5,
    )

    assert rows == [
        {"start": 1, "end": 1, "k": 5},
        {"start": 2, "end": 4, "k": 3},
        {"start": 5, "end": 127, "k": 1},
        {"start": 128, "end": 512, "k": 0},
    ]


def test_dynamic_k_provenance_does_not_treat_offered_batch_as_effective_batch() -> None:
    benchmark = load_module("benchmark")
    resolver = getattr(benchmark, "resolve_k_provenance", None)

    assert resolver is not None
    assert resolver("mtp_dynamic_max_k5", 512) == {
        "effective_k": None,
        "requested_batch_schedule_k": 0,
        "k_selection_basis": "active_scheduled_batch",
    }


def test_static_k_provenance_remains_exact() -> None:
    benchmark = load_module("benchmark")
    resolver = getattr(benchmark, "resolve_k_provenance", None)

    assert resolver is not None
    assert resolver("mtp_static_k5", 512) == {
        "effective_k": 5,
        "k_selection_basis": "static",
    }


def test_build_submission_plan_pins_runtime_flags_and_version_provenance() -> None:
    plan = build_submission_plan()
    first = plan[0]

    assert first["runtime"] == {
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 32768,
        "max_num_seqs": 512,
        "ignore_eos": True,
    }
    assert first["runtime_provenance"] == {
        "vllm_version": "0.28.0",
        "vllm_branch": "release",
        "vllm_commit": "2cf0a69",
        "container_digest": PINNED_CONTAINER.split("@", 1)[1],
    }


def test_execute_submission_plan_dry_run_never_calls_sbatch() -> None:
    launcher = load_module("launcher")
    plan = build_submission_plan()
    sbatch_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    rendered = launcher.execute_submission_plan(
        plan,
        dry_run=True,
        sbatch_runner=lambda *args, **kwargs: sbatch_calls.append((args, kwargs)),
    )

    assert rendered
    assert sbatch_calls == []


@pytest.mark.parametrize(
    ("data_parallel_size", "container_image", "message"),
    [
        (2, PINNED_CONTAINER, "data_parallel_size"),
        (1, "nvcr.io/nvidia/vllm-openai:0.28.0", "container"),
    ],
)
def test_build_submission_plan_rejects_dp_greater_than_one_and_unpinned_containers(
    data_parallel_size: int,
    container_image: str,
    message: str,
) -> None:
    launcher = load_module("launcher")

    with pytest.raises(ValueError, match=message):
        launcher.build_submission_plan(
            build_contract_matrix(),
            data_parallel_size=data_parallel_size,
            container_image=container_image,
        )


def test_validate_result_payload_requires_tokens_ok_and_per_run_spec_metrics() -> None:
    results = load_module("results")
    record = results.validate_result_payload(
        {
            "schema_version": 1,
            "status": "complete",
            "config": {
                "model_key": "ultra",
                "method_key": "mtp_dynamic_max_k5",
                "isl": 1000,
                "osl": 10000,
                "batch_size": 512,
                "effective_k": 5,
            },
            "summary": {
                "tokens_ok": True,
                "spec_metrics": {
                    "draft_tokens": 4096,
                    "accepted_tokens": 2048,
                    "acceptance_rate": 0.5,
                },
            },
        }
    )

    assert record["summary"]["tokens_ok"] is True
    assert record["summary"]["spec_metrics"] == {
        "draft_tokens": 4096,
        "accepted_tokens": 2048,
        "acceptance_rate": 0.5,
    }


def test_validate_result_payload_rejects_k0_canaries_with_nonzero_draft_tokens() -> None:
    results = load_module("results")

    with pytest.raises(ValueError, match="K=0"):
        results.validate_result_payload(
            {
                "schema_version": 1,
                "status": "complete",
                "config": {
                    "model_key": "super",
                    "method_key": "mtp_dynamic_max_k5",
                    "isl": 10000,
                    "osl": 1000,
                    "batch_size": 512,
                    "effective_k": 0,
                    "canary_kind": "mrv2_full_and_piecewise",
                },
                "summary": {
                    "tokens_ok": True,
                    "spec_metrics": {
                        "draft_tokens": 7,
                        "accepted_tokens": 0,
                        "acceptance_rate": 0.0,
                    },
                },
            }
        )


def test_enrich_result_payload_adds_expected_and_actual_token_totals_per_row() -> None:
    benchmark = load_module("benchmark")
    payload = {
        "status": "complete",
        "rows": [
            {"batch_size": 1, "output_tokens": 30000},
            {"batch_size": 32, "output_tokens": 960000},
        ],
    }

    enriched = benchmark.enrich_result_payload(
        payload,
        osl=10000,
        repeats=3,
        runtime_provenance={
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
        },
    )

    assert enriched["rows"] == [
        {
            "batch_size": 1,
            "output_tokens": 30000,
            "expected_output_tokens": 30000,
            "actual_output_tokens": 30000,
            "tokens_ok": True,
        },
        {
            "batch_size": 32,
            "output_tokens": 960000,
            "expected_output_tokens": 960000,
            "actual_output_tokens": 960000,
            "tokens_ok": True,
        },
    ]


def test_enrich_result_payload_hoists_runtime_provenance_to_top_level() -> None:
    benchmark = load_module("benchmark")

    enriched = benchmark.enrich_result_payload(
        {"status": "complete", "rows": [{"batch_size": 128, "output_tokens": 128000}]},
        osl=1000,
        repeats=1,
        runtime_provenance={
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": PINNED_CONTAINER.split("@", 1)[1],
        },
    )

    assert enriched["runtime_provenance"] == {
        "vllm_version": "0.28.0",
        "vllm_branch": "release",
        "vllm_commit": "2cf0a69",
        "container_digest": PINNED_CONTAINER.split("@", 1)[1],
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"status": "running", "rows": [{"batch_size": 1, "output_tokens": 1000}]}, "complete"),
        ({"status": "complete", "rows": [{"batch_size": 1, "output_tokens": 999}]}, "tokens"),
    ],
)
def test_enrich_result_payload_rejects_incomplete_or_mismatched_rows(
    payload: dict[str, Any],
    message: str,
) -> None:
    benchmark = load_module("benchmark")

    with pytest.raises(ValueError, match=message):
        benchmark.enrich_result_payload(
            payload,
            osl=1000,
            repeats=1,
            runtime_provenance={"vllm_version": "0.28.0"},
        )


def test_render_sbatch_emits_vllm028_runtime_guards_and_exact_generation_flags() -> None:
    launcher = load_module("launcher")
    plan_row = {
        "model_key": "super",
        "runner_key": "mrv1",
        "method_key": "mtp_static_k5",
        "shape": {"isl": 1000, "osl": 10000},
        "batch_size": 32,
        "runtime": {
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "ignore_eos": True,
        },
        "runtime_provenance": {
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": PINNED_CONTAINER.split("@", 1)[1],
        },
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 5},
    }

    script = launcher.render_sbatch(
        plan_row,
        experiment_dir=Path("/workspace/exp"),
        result_dir=Path("/workspace/results/super-k5"),
    )

    assert "python3 - <<'PY'" in script
    assert "assert version('vllm') == '0.28.0'" in script
    assert "export VLLM_USE_V2_MODEL_RUNNER=0" in script
    assert "--no-enable-prefix-caching" in script
    assert "--enable-chunked-prefill" in script
    assert "--max-num-batched-tokens 32768" in script
    assert "--max-num-seqs 512" in script
    assert "--isl 1000" in script
    assert "--osl 10000" in script
    assert "--ignore-eos" in script
    assert '--speculative-config-json \'{"method":"mtp","num_speculative_tokens":5}\'' in script
    assert "#SBATCH --job-name=coreai_dlalgo_llm-v028.super-mrv1-mtp_static_k5-bs32" in script
    assert "sbatch " not in script


@pytest.mark.parametrize(
    ("plan_row", "needles"),
    [
        (
            {
                "model_key": "super",
                "runner_key": "mrv1",
                "method_key": "baseline",
                "shape": {"isl": 1000, "osl": 10000},
                "batch_size": 1,
                "runtime": {
                    "enable_prefix_caching": False,
                    "enable_chunked_prefill": True,
                    "max_num_batched_tokens": 32768,
                    "max_num_seqs": 512,
                    "ignore_eos": True,
                },
                "runtime_provenance": {
                    "vllm_version": "0.28.0",
                    "vllm_branch": "release",
                    "vllm_commit": "2cf0a69",
                    "container_digest": PINNED_CONTAINER.split("@", 1)[1],
                },
                "speculative_config": None,
            },
            [
                "#SBATCH --nodes=1",
                "--tensor-parallel-size 2",
                "export VLLM_USE_V2_MODEL_RUNNER=0",
            ],
        ),
        (
            {
                "model_key": "ultra",
                "runner_key": "mrv2",
                "method_key": "mtp_dynamic_max_k5",
                "shape": {"isl": 10000, "osl": 1000},
                "batch_size": 128,
                "runtime": {
                    "enable_prefix_caching": False,
                    "enable_chunked_prefill": True,
                    "max_num_batched_tokens": 32768,
                    "max_num_seqs": 512,
                    "ignore_eos": True,
                },
                "runtime_provenance": {
                    "vllm_version": "0.28.0",
                    "vllm_branch": "release",
                    "vllm_commit": "2cf0a69",
                    "container_digest": PINNED_CONTAINER.split("@", 1)[1],
                },
                "speculative_config": {
                    "method": "mtp",
                    "num_speculative_tokens": 5,
                    "num_speculative_tokens_per_batch_size": [[1, 512, 5]],
                },
            },
            [
                "#SBATCH --nodes=2",
                "--tensor-parallel-size 8",
                "--enable-expert-parallel",
                "export VLLM_USE_V2_MODEL_RUNNER=1",
            ],
        ),
    ],
)
def test_render_sbatch_encodes_topology_and_runner_specific_flags(
    plan_row: dict[str, Any],
    needles: list[str],
) -> None:
    launcher = load_module("launcher")

    script = launcher.render_sbatch(
        plan_row,
        experiment_dir=Path("/workspace/exp"),
        result_dir=Path("/workspace/results/run"),
    )

    assert all(needle in script for needle in needles)


def test_render_sbatch_uses_node_local_raid_scratch_for_caches() -> None:
    launcher = load_module("launcher")
    plan_row = {
        "model_key": "super",
        "runner_key": "mrv1",
        "method_key": "baseline",
        "shape": {"isl": 1000, "osl": 10000},
        "batch_size": 1,
        "runtime": {
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "ignore_eos": True,
        },
        "runtime_provenance": {
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": PINNED_CONTAINER.split("@", 1)[1],
        },
        "speculative_config": None,
    }

    script = launcher.render_sbatch(
        plan_row,
        experiment_dir=Path("/workspace/exp"),
        result_dir=Path("/workspace/results/run"),
    )

    assert "/raid/scratch" in script


def test_select_smoke_plan_rows_keeps_only_bs1_bs32_bs128_baseline_k5_and_dynamic_for_both_runners() -> None:
    launcher = load_module("launcher")
    plan = [
        {"runner_key": runner_key, "method_key": method_key, "batch_size": batch_size}
        for runner_key in ("mrv1", "mrv2")
        for method_key in ("baseline", "mtp_static_k1", "mtp_static_k5", "mtp_dynamic_max_k5")
        for batch_size in (1, 2, 32, 128, 512)
    ]

    smoke_rows = launcher.select_smoke_plan_rows(plan)

    assert smoke_rows == [
        {"runner_key": "mrv1", "method_key": "baseline", "batch_size": 1},
        {"runner_key": "mrv1", "method_key": "baseline", "batch_size": 32},
        {"runner_key": "mrv1", "method_key": "baseline", "batch_size": 128},
        {"runner_key": "mrv1", "method_key": "mtp_static_k5", "batch_size": 1},
        {"runner_key": "mrv1", "method_key": "mtp_static_k5", "batch_size": 32},
        {"runner_key": "mrv1", "method_key": "mtp_static_k5", "batch_size": 128},
        {"runner_key": "mrv1", "method_key": "mtp_dynamic_max_k5", "batch_size": 1},
        {"runner_key": "mrv1", "method_key": "mtp_dynamic_max_k5", "batch_size": 32},
        {"runner_key": "mrv1", "method_key": "mtp_dynamic_max_k5", "batch_size": 128},
        {"runner_key": "mrv2", "method_key": "baseline", "batch_size": 1},
        {"runner_key": "mrv2", "method_key": "baseline", "batch_size": 32},
        {"runner_key": "mrv2", "method_key": "baseline", "batch_size": 128},
        {"runner_key": "mrv2", "method_key": "mtp_static_k5", "batch_size": 1},
        {"runner_key": "mrv2", "method_key": "mtp_static_k5", "batch_size": 32},
        {"runner_key": "mrv2", "method_key": "mtp_static_k5", "batch_size": 128},
        {"runner_key": "mrv2", "method_key": "mtp_dynamic_max_k5", "batch_size": 1},
        {"runner_key": "mrv2", "method_key": "mtp_dynamic_max_k5", "batch_size": 32},
        {"runner_key": "mrv2", "method_key": "mtp_dynamic_max_k5", "batch_size": 128},
    ]


@pytest.mark.parametrize(
    ("model_key", "runner_key", "expected_model", "expected_tp", "expects_ep", "expected_cudagraph"),
    [
        ("super", "mrv1", SUPER_CHECKPOINT, "2", False, "PIECEWISE"),
        ("ultra", "mrv2", ULTRA_CHECKPOINT, "8", True, "FULL_AND_PIECEWISE"),
    ],
)
def test_build_legacy_benchmark_argv_maps_models_and_runner_settings_to_pinned_legacy_cli(
    model_key: str,
    runner_key: str,
    expected_model: str,
    expected_tp: str,
    expects_ep: bool,
    expected_cudagraph: str,
) -> None:
    benchmark = load_module("benchmark")

    argv = benchmark.build_legacy_benchmark_argv(
        legacy_args(model_key=model_key, runner_key=runner_key)
    )

    assert argv[:2] == [
        "python3",
        str(ROOT / "experiments/vllm_024_dynamicsd/benchmark.py"),
    ]
    assert_arg_value(argv, "--model", expected_model)
    assert_arg_value(argv, "--tensor-parallel-size", expected_tp)
    assert ("--enable-expert-parallel" in argv) is expects_ep
    assert_arg_value(argv, "--cudagraph-mode", expected_cudagraph)


def test_build_legacy_benchmark_argv_passes_explicit_engine_capacity_instead_of_max_batch_size() -> None:
    benchmark = load_module("benchmark")

    argv = benchmark.build_legacy_benchmark_argv(
        legacy_args(batch_sizes=[1, 32, 128], max_num_seqs=512)
    )

    assert_arg_value(argv, "--max-num-seqs", "512")


def test_legacy_driver_parser_accepts_and_preserves_explicit_max_num_seqs() -> None:
    path = ROOT / "experiments/vllm_024_dynamicsd/benchmark.py"
    spec = importlib.util.spec_from_file_location("vllm024_driver_capacity_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "--model",
            SUPER_CHECKPOINT,
            "--mode",
            "baseline",
            "--max-num-seqs",
            "512",
            "--output",
            "/tmp/result.json",
        ]
    )

    assert args.max_num_seqs == 512


def test_build_legacy_benchmark_argv_pins_common_runtime_flags_for_smoke() -> None:
    benchmark = load_module("benchmark")

    argv = benchmark.build_legacy_benchmark_argv(legacy_args())

    assert_arg_value(argv, "--max-model-len", "11256")
    assert_arg_value(argv, "--dtype", "bfloat16")
    assert_arg_value(argv, "--kv-cache-dtype", "fp8")
    assert_arg_value(argv, "--gpu-memory-utilization", "0.9")
    assert "--no-enable-prefix-caching" in argv
    assert "--enable-chunked-prefill" in argv
    assert_arg_value(argv, "--warmup-repeats", "1")
    assert_arg_value(argv, "--measure-repeats", "1")


def test_build_legacy_benchmark_argv_maps_baseline_method_without_static_or_dynamic_overrides() -> None:
    benchmark = load_module("benchmark")

    argv = benchmark.build_legacy_benchmark_argv(legacy_args(method_key="baseline"))

    assert_arg_value(argv, "--mode", "baseline")
    assert "--static-k" not in argv
    assert "--dynamic-schedule" not in argv


def test_build_legacy_benchmark_argv_maps_static_k5_to_mtp_static_with_exact_k() -> None:
    benchmark = load_module("benchmark")

    argv = benchmark.build_legacy_benchmark_argv(legacy_args(method_key="mtp_static_k5"))

    assert_arg_value(argv, "--mode", "mtp_static")
    assert_arg_value(argv, "--static-k", "5")


def test_build_legacy_benchmark_argv_maps_dynamic_method_to_exact_contiguous_schedule() -> None:
    benchmark = load_module("benchmark")

    argv = benchmark.build_legacy_benchmark_argv(
        legacy_args(method_key="mtp_dynamic_max_k5")
    )

    assert_arg_value(argv, "--mode", "mtp_dynamic")
    assert_arg_value(argv, "--dynamic-schedule", EXPECTED_DYNAMIC_SCHEDULE)


def test_build_legacy_benchmark_argv_rejects_speculative_config_inconsistent_with_method() -> None:
    benchmark = load_module("benchmark")

    with pytest.raises(ValueError, match="speculative"):
        benchmark.build_legacy_benchmark_argv(
            legacy_args(
                method_key="baseline",
                speculative_config={"method": "mtp", "num_speculative_tokens": 5},
            )
        )


def test_stage_vllm028_container_artifact_pins_official_digest_release_and_safe_staging() -> None:
    stage_file = PACKAGE_ROOT / "stage_vllm028_container.sbatch"

    assert stage_file.is_file(), f"missing staging artifact: {stage_file}"
    text = stage_file.read_text(encoding="utf-8")

    assert "41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4" in text
    assert "2cf0a69" in text
    assert "#SBATCH --partition=gb200-backfill" in text
    assert "#SBATCH --account=coreai_dlalgo_llm" in text
    assert ".sqsh" in text
    assert ".metadata.json" in text
    assert "ln -sfn" in text
    assert "find /lustre" not in text
    assert "find ${LUSTRE_ROOT}" not in text
    assert "ls -R" not in text


def test_stage_super_bf16_checkpoint_artifact_verifies_exact_revision_snapshot_and_shard_count() -> None:
    stage_file = PACKAGE_ROOT / "stage_super_bf16_checkpoint.sbatch"

    assert stage_file.is_file(), f"missing checkpoint staging artifact: {stage_file}"
    text = stage_file.read_text(encoding="utf-8")

    assert "NVIDIA-Nemotron-3-Super-120B-A12B-BF16" in text
    assert "d51eab0d1f979ebc26b546e634a04f450d99158e" in text
    assert "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e" in text
    assert "config.json" in text
    assert "50" in text
    assert "/raid/scratch" in text
    assert "HF_HOME" in text
    assert "HF_HUB_CACHE" in text
    assert "/lustre" in text


def test_render_sbatch_wraps_benchmark_in_srun_with_staged_sqsh_mounts_and_lustre_result_path() -> None:
    launcher = load_module("launcher")
    plan_row = {
        "model_key": "super",
        "runner_key": "mrv1",
        "method_key": "baseline",
        "shape": {"isl": 1000, "osl": 10000},
        "batch_size": 32,
        "runtime": {
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "ignore_eos": True,
        },
        "runtime_provenance": {
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": PINNED_CONTAINER.split("@", 1)[1],
        },
        "speculative_config": None,
    }

    script = launcher.render_sbatch(
        plan_row,
        experiment_dir=Path("/workspace/exp"),
        result_dir=Path("/lustre/results/super-baseline"),
    )

    assert "srun " in script
    assert ".sqsh" in script
    assert "--container-mounts" in script
    assert "/workspace/exp" in script
    assert "/lustre" in script
    assert "/raid/scratch" in script
    assert "--output /lustre/results/super-baseline/result.json" in script


def test_render_sbatch_uses_ray_for_ultra_multi_node_runs() -> None:
    launcher = load_module("launcher")
    plan_row = {
        "model_key": "ultra",
        "runner_key": "mrv2",
        "method_key": "mtp_dynamic_max_k5",
        "shape": {"isl": 10000, "osl": 1000},
        "batch_size": 128,
        "runtime": {
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "ignore_eos": True,
        },
        "runtime_provenance": {
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": PINNED_CONTAINER.split("@", 1)[1],
        },
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 5,
            "num_speculative_tokens_per_batch_size": [[1, 512, 5]],
        },
    }

    script = launcher.render_sbatch(
        plan_row,
        experiment_dir=Path("/workspace/exp"),
        result_dir=Path("/lustre/results/ultra-dynamic"),
    )

    assert "--distributed-executor-backend ray" in script


def test_submit_smoke_dry_run_renders_canary_rows_without_calling_sbatch_and_declares_staging_dependencies() -> None:
    submit_smoke = load_module("submit_smoke")
    sbatch_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    rendered = submit_smoke.render_dry_run(
        sbatch_runner=lambda *args, **kwargs: sbatch_calls.append((args, kwargs))
    )

    assert rendered
    assert any("mrv2_canary" in row for row in rendered)
    assert any("stage_vllm028_container" in row for row in rendered)
    assert any("stage_super_bf16_checkpoint" in row for row in rendered)
    assert sbatch_calls == []


def _legacy_k0_payload(*, legacy_num_draft_tokens: int | float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "config": {
            "model_key": "ultra",
            "method_key": "mtp_dynamic_max_k5",
            "isl": 1000,
            "osl": 10000,
            "batch_size": 512,
            "effective_k": 0,
        },
        "summary": {
            "tokens_ok": True,
            "spec_decode_metrics": {
                "num_draft_tokens": legacy_num_draft_tokens,
                "num_accepted_tokens": 0,
                "acceptance_rate": 0.0,
            },
        },
    }


def test_smoke_rows_and_rendered_dry_run_include_bs512_dynamic_k0_canaries_for_both_models_and_runners() -> None:
    submit_smoke = load_module("submit_smoke")

    rows = submit_smoke._smoke_rows()
    rendered = submit_smoke.render_dry_run(sbatch_runner=lambda *args, **kwargs: None)

    expected_pairs = {
        ("super", "mrv1"),
        ("super", "mrv2"),
        ("ultra", "mrv1"),
        ("ultra", "mrv2"),
    }
    actual_pairs = {
        (str(row["model_key"]), str(row["runner_key"]))
        for row in rows
        if row["method_key"] == "mtp_dynamic_max_k5" and row["batch_size"] == 512
    }

    assert actual_pairs == expected_pairs
    assert all(
        any(
            token in rendered_row
            for rendered_row in rendered
        )
        for token in (
            "coreai_dlalgo_llm-v028.super-mrv1-mtp_dynamic_max_k5-bs512",
            "coreai_dlalgo_llm-v028.super-mrv2-mtp_dynamic_max_k5-bs512",
            "coreai_dlalgo_llm-v028.ultra-mrv1-mtp_dynamic_max_k5-bs512",
            "coreai_dlalgo_llm-v028.ultra-mrv2-mtp_dynamic_max_k5-bs512",
        )
    )


def test_validate_result_payload_rejects_legacy_bs512_dynamic_k0_canary_with_nonzero_draft_tokens() -> None:
    results = load_module("results")

    with pytest.raises(ValueError, match="K=0"):
        results.validate_result_payload(_legacy_k0_payload(legacy_num_draft_tokens=7))


def test_validate_result_payload_accepts_legacy_bs512_dynamic_k0_canary_with_zero_draft_tokens() -> None:
    results = load_module("results")

    validated = results.validate_result_payload(_legacy_k0_payload(legacy_num_draft_tokens=0))

    assert (
        validated["summary"]["spec_decode_metrics"]["num_draft_tokens"] == 0
    )


def test_package_provides_logs_artifact_and_render_sbatch_logs_under_experiment_dir() -> None:
    launcher = load_module("launcher")
    logs_dir = PACKAGE_ROOT / "logs"
    plan_row = {
        "model_key": "super",
        "runner_key": "mrv1",
        "runner_gate_key": "mrv1_gate",
        "method_key": "baseline",
        "shape": {"isl": 1000, "osl": 10000},
        "batch_size": 32,
        "runtime": {
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "ignore_eos": True,
        },
        "runtime_provenance": {
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": PINNED_CONTAINER.split("@", 1)[1],
        },
        "speculative_config": None,
    }

    script = launcher.render_sbatch(
        plan_row,
        experiment_dir=PACKAGE_ROOT,
        result_dir=Path("/lustre/results/super-baseline"),
    )

    assert logs_dir.is_dir()
    assert f"#SBATCH --output={PACKAGE_ROOT / 'logs'}/" in script
    assert "#SBATCH --output=/lustre/results/super-baseline/" not in script


def test_benchmark_wrapper_plans_raw_and_temporary_outputs_before_atomic_canonical_publish() -> None:
    benchmark_path = PACKAGE_ROOT / "benchmark.py"
    text = benchmark_path.read_text(encoding="utf-8")

    assert ".raw.json" in text
    assert ".tmp" in text
    assert "replace(" in text or ".replace(" in text
    assert "_validate_result_contract(" in text
    replace_index = text.index("replace(") if "replace(" in text else text.index(".replace(")
    validate_index = text.index("_validate_result_contract(")
    assert validate_index < replace_index


def test_stage_super_checkpoint_verifier_counts_only_nonempty_50_shards() -> None:
    text = (PACKAGE_ROOT / "stage_super_bf16_checkpoint.sbatch").read_text(
        encoding="utf-8"
    )

    assert "-size +0c" in text or "st_size > 0" in text


def test_stage_vllm028_container_artifact_pins_exact_ray248_sidecar_with_provenance_and_validation() -> None:
    text = (PACKAGE_ROOT / "stage_vllm028_container.sbatch").read_text(
        encoding="utf-8"
    )

    assert "ray[cgraph,default]==2.48.0" in text
    assert "ray" in text and "aarch64" in text
    assert "sha256" in text
    assert "metadata" in text.lower() or "provenance" in text.lower()
    assert "ln -sfn" in text
    assert "import ray" in text
    assert "ray.__version__ == '2.48.0'" in text


def test_stage_vllm028_container_materializes_rootfs_before_starting_without_squashfuse() -> None:
    text = (PACKAGE_ROOT / "stage_vllm028_container.sbatch").read_text(
        encoding="utf-8"
    )

    import_index = text.index("enroot import")
    create_index = text.index("enroot create")
    start_index = text.index("enroot start")

    assert import_index < create_index < start_index
    assert 'enroot start "${PARTIAL_IMAGE}"' not in text


def test_rendered_ultra_sbatch_stages_and_mounts_exact_ray_bundle_on_all_nodes() -> None:
    launcher = load_module("launcher")
    plan_row = {
        "model_key": "ultra",
        "runner_key": "mrv2",
        "runner_gate_key": "mrv2_canary",
        "method_key": "mtp_dynamic_max_k5",
        "shape": {"isl": 10000, "osl": 1000},
        "batch_size": 128,
        "runtime": {
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "ignore_eos": True,
        },
        "runtime_provenance": {
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": PINNED_CONTAINER.split("@", 1)[1],
        },
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 5,
            "num_speculative_tokens_per_batch_size": [[1, 512, 5]],
        },
    }

    script = launcher.render_sbatch(
        plan_row,
        experiment_dir=PACKAGE_ROOT,
        result_dir=Path("/lustre/results/ultra-dynamic"),
    )

    assert "/raid/scratch" in script
    assert "RAY" in script
    assert "2.48.0" in script
    assert "PYTHONPATH" in script
    assert "--container-mounts" in script
    assert "ray" in script and ("sidecar" in script or "bundle" in script)
    assert "import ray" in script
    assert "ray.__version__ == '2.48.0'" in script
    assert "srun" in script and ("cp " in script or "rsync " in script or "tar " in script)


def test_rendered_ultra_ray_version_check_preserves_string_literal_through_shell_parsing() -> None:
    launcher = load_module("launcher")
    row = next(
        row
        for row in build_submission_plan()
        if row["model_key"] == "ultra"
        and row["runner_key"] == "mrv1"
        and row["method_key"] == "baseline"
        and row["shape"]["isl"] == 10000
        and row["shape"]["osl"] == 1000
        and row["batch_size"] == 1
    )
    script = launcher.render_sbatch(
        row,
        experiment_dir=PACKAGE_ROOT,
        result_dir=Path("/lustre/results/ultra-baseline"),
    )
    stage_line = next(line for line in script.splitlines() if "tar -xzf" in line)

    parsed = subprocess.run(
        [
            "bash",
            "-c",
            "srun() { printf '%s\\n' \"$@\"; }; "
            "RAY_SITE_PACKAGES=/raid/ray; CONTAINER_IMAGE=/tmp/image.sqsh; "
            f"{stage_line}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'ray.__version__ == "2.48.0"' in parsed.stdout


def test_rendered_super_sbatch_does_not_require_ray_bundle() -> None:
    launcher = load_module("launcher")
    plan_row = {
        "model_key": "super",
        "runner_key": "mrv1",
        "runner_gate_key": "mrv1_gate",
        "method_key": "baseline",
        "shape": {"isl": 1000, "osl": 10000},
        "batch_size": 32,
        "runtime": {
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "ignore_eos": True,
        },
        "runtime_provenance": {
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": PINNED_CONTAINER.split("@", 1)[1],
        },
        "speculative_config": None,
    }

    script = launcher.render_sbatch(
        plan_row,
        experiment_dir=PACKAGE_ROOT,
        result_dir=Path("/lustre/results/super-baseline"),
    )

    assert "ray[cgraph,default]==2.48.0" not in script
    assert "ray.__version__ == '2.48.0'" not in script
    assert "RAY_BUNDLE" not in script


def test_rendered_ultra_sbatch_uses_real_lustre_result_backed_ray_sync_dir() -> None:
    launcher = load_module("launcher")
    plan_row = {
        "model_key": "ultra",
        "runner_key": "mrv2",
        "runner_gate_key": "mrv2_canary",
        "method_key": "mtp_dynamic_max_k5",
        "shape": {"isl": 10000, "osl": 1000},
        "batch_size": 128,
        "runtime": {
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "ignore_eos": True,
        },
        "runtime_provenance": {
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": PINNED_CONTAINER.split("@", 1)[1],
        },
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 5,
            "num_speculative_tokens_per_batch_size": [[1, 512, 5]],
        },
    }

    script = launcher.render_sbatch(
        plan_row,
        experiment_dir=PACKAGE_ROOT,
        result_dir=Path("/lustre/results/ultra-dynamic"),
    )

    assert 'export RAY_SYNC_DIR="/lustre/results/ultra-dynamic/ray-sync-${SLURM_JOB_ID}"' in script
    assert "{result_dir}" not in script
    assert 'export RAY_SYNC_DIR="/raid/scratch' not in script


def test_committed_ray248_lock_is_python312_aarch64_and_hash_pinned() -> None:
    lock_path = PACKAGE_ROOT / "ray248-aarch64.lock"
    text = lock_path.read_text(encoding="utf-8")

    assert lock_path.is_file()
    assert "--python-version 3.12" in text or "Python 3.12" in text
    assert "aarch64-manylinux_2_39" in text or "aarch64" in text
    assert "ray[cgraph,default]==2.48.0" in text
    assert "--hash=sha256:" in text
    assert "# via ray" in text


def test_stage_vllm028_container_uses_committed_lock_with_require_hashes_and_records_manifest() -> None:
    text = (PACKAGE_ROOT / "stage_vllm028_container.sbatch").read_text(
        encoding="utf-8"
    )

    assert "ray248-aarch64.lock" in text
    assert "--require-hashes" in text
    assert "--only-binary" in text
    assert "lock_sha256" in text or "LOCK_SHA256" in text
    assert "pip freeze" in text or "installed_packages" in text or "manifest" in text


def test_run_multinode_ray_uses_job_specific_raid_scratch_temp_root_and_head_temp_dir() -> None:
    text = (PACKAGE_ROOT / "run_multinode_ray.sh").read_text(encoding="utf-8")

    assert "RAY_TEMP_ROOT" in text
    assert "/raid/scratch" in text
    assert "SLURM_JOB_ID" in text
    assert "--temp-dir" in text


def test_run_multinode_ray_guards_temp_root_cleanup() -> None:
    text = (PACKAGE_ROOT / "run_multinode_ray.sh").read_text(encoding="utf-8")

    assert "trap" in text
    assert "rm -rf" in text
    assert '"/raid/scratch/${USER}' in text or '"/raid/scratch/${USER}/' in text


def test_repo_root_safe_stage_submission_helper_exports_absolute_ray_lock_declares_afterok_and_is_documented() -> None:
    submit_smoke = load_module("submit_smoke")
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    sbatch_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    rendered = submit_smoke.build_stage_submission_plan(
        dry_run=True,
        sbatch_runner=lambda *args, **kwargs: sbatch_calls.append((args, kwargs)),
    )

    lock_path = (PACKAGE_ROOT / "ray248-aarch64.lock").resolve()
    assert rendered[0]["argv"] == [
        "sbatch",
        f"--export=ALL,RAY_LOCK_PATH={lock_path}",
        str(PACKAGE_ROOT / "stage_vllm028_container.sbatch"),
    ]
    assert rendered[1]["argv"] == [
        "sbatch",
        "--dependency=afterok:stage_vllm028_container",
        str(PACKAGE_ROOT / "stage_super_bf16_checkpoint.sbatch"),
    ]
    assert sbatch_calls == []
    assert "build_stage_submission_plan" in readme
    assert "SLURM_SUBMIT_DIR" not in readme
