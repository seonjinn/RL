from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/vllm_024_dynamicsd"


def load_benchmark_module() -> ModuleType:
    path = EXPERIMENT / "benchmark.py"
    spec = importlib.util.spec_from_file_location("vllm024_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sync_rollout_module() -> ModuleType:
    path = EXPERIMENT / "benchmark_sync_rollout.py"
    sys.path.insert(0, str(EXPERIMENT))
    spec = importlib.util.spec_from_file_location("vllm024_sync_rollout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sync_rollout_core_module() -> ModuleType:
    path = EXPERIMENT / "sync_rollout_core.py"
    spec = importlib.util.spec_from_file_location("vllm024_sync_rollout_core", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sync_summary_module() -> ModuleType:
    path = EXPERIMENT / "summarize_sync_rollout.py"
    spec = importlib.util.spec_from_file_location("vllm024_sync_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset_materializer_module() -> ModuleType:
    path = EXPERIMENT / "materialize_math_prompts.py"
    spec = importlib.util.spec_from_file_location("vllm024_math_prompts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_long_context_materializer_module() -> ModuleType:
    path = EXPERIMENT / "materialize_long_context_model_views.py"
    spec = importlib.util.spec_from_file_location(
        "vllm024_long_context_materializer", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_speedbench_dataset_module() -> ModuleType:
    path = EXPERIMENT / "speedbench_dataset.py"
    spec = importlib.util.spec_from_file_location("vllm024_speedbench_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_dry(script_name: str, **env_overrides: str) -> str:
    env = {
        "PATH": "/usr/bin:/bin",
        "DRY_RUN": "true",
        **env_overrides,
    }
    completed = subprocess.run(
        ["bash", str(EXPERIMENT / script_name)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def load_model_method_matrix() -> dict[str, Any]:
    return json.loads(
        (EXPERIMENT / "model_method_matrix.json").read_text(encoding="utf-8")
    )


def get_matrix_model(matrix: dict[str, Any], model_key: str) -> dict[str, Any]:
    return next(item for item in matrix["models"] if item["key"] == model_key)


def get_matrix_profile(model: dict[str, Any], profile_key: str) -> dict[str, Any]:
    return next(item for item in model["profiles"] if item["key"] == profile_key)


def extract_manifest_rows(output: str) -> list[str]:
    return [
        line
        for line in output.splitlines()
        if line.startswith(("SUPPORTED\t", "UNSUPPORTED\t", "INTEGRATION\t"))
    ]


def extract_sync_variants(output: str, marker: str = "[DRY-RUN]") -> list[str]:
    prefix = f"{marker} sync_variant="
    return [line.split("=", 1)[1] for line in output.splitlines() if line.startswith(prefix)]


def extract_manifest_path(output: str) -> Path:
    line = next(line for line in output.splitlines() if line.startswith("manifest="))
    return Path(line.split("=", 1)[1])


def extract_run_benchmark_script(output: str, variant: str) -> str:
    start = f"# BEGIN run_benchmark.sh {variant}\n"
    end = f"# END run_benchmark.sh {variant}\n"
    assert start in output
    assert end in output
    return output.split(start, 1)[1].split(end, 1)[0]


def fake_speedbench_row(
    category: str,
    idx: int,
    *,
    nominal_isl: int = 1024,
    actual_tokenizer_isl: int | None = None,
    turns: tuple[str, ...] | None = None,
    masked: bool = False,
) -> dict[str, Any]:
    row_turns = turns or (f"{category} prompt {idx}",)
    return {
        "question_id": f"{category}-{idx:03d}",
        "category": category,
        "sub_category": f"{category}-subtype",
        "turns": list(row_turns),
        "source": "nvidia/SPEED-Bench",
        "src_id": f"src-{category}-{idx:03d}",
        "difficulty": "hard" if category == "high_entropy" else None,
        "multiturn": len(row_turns) > 1,
        "nominal_isl": nominal_isl,
        "actual_tokenizer_isl": actual_tokenizer_isl,
        "masked": masked,
    }


def fake_speedbench_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in ("low_entropy", "mixed", "high_entropy"):
        for idx in range(24):
            turns = (f"{category} prompt {idx}",)
            if category == "mixed" and idx == 0:
                turns = (
                    "user: describe the rollout barrier",
                    "assistant: the barrier waits for the tail request",
                    "user: keep the multi-turn context intact",
                )
            rows.append(fake_speedbench_row(category, idx, turns=turns))
    return rows


def test_dynamic_schedule_and_speculative_configs() -> None:
    benchmark = load_benchmark_module()
    schedule = benchmark.parse_dynamic_schedule(
        "1:16:5,17:32:4,33:64:3,65:128:1,129:512:0"
    )

    assert schedule == [
        [1, 16, 5],
        [17, 32, 4],
        [33, 64, 3],
        [65, 128, 1],
        [129, 512, 0],
    ]
    assert benchmark.build_speculative_config(
        mode="baseline",
        draft_model="unused",
        static_k=5,
        dynamic_schedule=schedule,
        draft_tensor_parallel_size=1,
        draft_attention_backend="",
        suffix_max_cached_requests=10000,
        suffix_max_spec_factor=1.0,
        suffix_min_token_prob=0.1,
    ) is None
    assert benchmark.build_speculative_config(
        mode="static",
        draft_model="draft",
        static_k=5,
        dynamic_schedule=schedule,
        draft_tensor_parallel_size=1,
        draft_attention_backend="",
        suffix_max_cached_requests=10000,
        suffix_max_spec_factor=1.0,
        suffix_min_token_prob=0.1,
    ) == {
        "method": "eagle3",
        "model": "draft",
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
    }
    assert benchmark.build_speculative_config(
        mode="dynamic",
        draft_model="draft",
        static_k=5,
        dynamic_schedule=schedule,
        draft_tensor_parallel_size=1,
        draft_attention_backend="",
        suffix_max_cached_requests=10000,
        suffix_max_spec_factor=1.0,
        suffix_min_token_prob=0.1,
    ) == {
        "method": "eagle3",
        "model": "draft",
        "num_speculative_tokens": 5,
        "num_speculative_tokens_per_batch_size": schedule,
        "draft_tensor_parallel_size": 1,
    }
    assert benchmark.build_speculative_config(
        mode="mtp_static",
        draft_model="",
        static_k=5,
        dynamic_schedule=schedule,
        draft_tensor_parallel_size=1,
        draft_attention_backend="",
        suffix_max_cached_requests=10000,
        suffix_max_spec_factor=1.0,
        suffix_min_token_prob=0.1,
    ) == {
        "method": "mtp",
        "num_speculative_tokens": 5,
    }
    assert benchmark.build_speculative_config(
        mode="mtp_dynamic",
        draft_model="",
        static_k=5,
        dynamic_schedule=schedule,
        draft_tensor_parallel_size=1,
        draft_attention_backend="",
        suffix_max_cached_requests=10000,
        suffix_max_spec_factor=1.0,
        suffix_min_token_prob=0.1,
    ) == {
        "method": "mtp",
        "num_speculative_tokens": 5,
        "num_speculative_tokens_per_batch_size": schedule,
    }


@pytest.mark.parametrize(
    ("mode", "draft_model", "static_k", "expected"),
    (
        (
            "suffix",
            "",
            32,
            {
                "method": "suffix",
                "num_speculative_tokens": 32,
                "suffix_decoding_max_tree_depth": 32,
                "suffix_decoding_max_cached_requests": 10000,
                "suffix_decoding_max_spec_factor": 1.0,
                "suffix_decoding_min_token_prob": 0.1,
            },
        ),
        (
            "pard",
            "amd/PARD-Qwen3-0.6B",
            12,
            {
                "method": "draft_model",
                "model": "amd/PARD-Qwen3-0.6B",
                "num_speculative_tokens": 12,
                "draft_tensor_parallel_size": 1,
                "parallel_drafting": True,
            },
        ),
        (
            "pard2",
            "amd/PARD2-Qwen3-8B",
            15,
            {
                "method": "pard2",
                "model": "amd/PARD2-Qwen3-8B",
                "num_speculative_tokens": 15,
                "draft_tensor_parallel_size": 1,
                "parallel_drafting": True,
            },
        ),
        (
            "dflash",
            "z-lab/Qwen3-8B-DFlash-b16",
            15,
            {
                "method": "dflash",
                "model": "z-lab/Qwen3-8B-DFlash-b16",
                "num_speculative_tokens": 15,
                "draft_tensor_parallel_size": 1,
                "attention_backend": "FLASH_ATTN",
            },
        ),
    ),
)
def test_extended_speculative_configs(
    mode: str, draft_model: str, static_k: int, expected: dict[str, object]
) -> None:
    benchmark = load_benchmark_module()

    assert benchmark.build_speculative_config(
        mode=mode,
        draft_model=draft_model,
        static_k=static_k,
        dynamic_schedule=[[1, 16, 5]],
        draft_tensor_parallel_size=1,
        draft_attention_backend="FLASH_ATTN" if mode == "dflash" else "",
        suffix_max_cached_requests=10000,
        suffix_max_spec_factor=1.0,
        suffix_min_token_prob=0.1,
    ) == expected


@pytest.mark.parametrize(
    "schedule",
    (
        "2:16:5",
        "1:16:5,16:32:4",
        "1:0:5",
        "1:16:-1",
    ),
)
def test_dynamic_schedule_rejects_gaps_overlaps_and_invalid_values(
    schedule: str,
) -> None:
    benchmark = load_benchmark_module()

    with pytest.raises(ValueError):
        benchmark.parse_dynamic_schedule(schedule)


def test_dynamic_schedule_allows_gaps_that_inherit_the_previous_k() -> None:
    benchmark = load_benchmark_module()

    assert benchmark.parse_dynamic_schedule("1:16:5,18:32:4") == [
        [1, 16, 5],
        [18, 32, 4],
    ]


def test_stage_image_dry_run_is_lyris_safe_and_reproducible() -> None:
    output = run_dry("stage_image.sh", CLUSTER="lyris")

    assert "vllm/vllm-openai:v0.24.0-aarch64-ubuntu2404" in output
    assert "vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh" in output
    assert "--partition=gb200" in output
    assert "--account=coreai_dlalgo_llm" in output
    assert "--segment=1" in output
    assert "--container-save=" in output
    assert "vllm.__version__" in output
    assert "sha256sum" in output
    assert "--gres" not in output


def test_math_dataset_rows_are_normalized_without_solutions_in_prompts() -> None:
    materializer = load_dataset_materializer_module()
    dapo = materializer.normalize_row(
        "dapo_math_17k",
        {
            "prompt": [{"role": "user", "content": "Solve DAPO problem"}],
            "reward_model": {"ground_truth": "42", "style": "math"},
            "extra_info": {"index": "dapo-1"},
        },
        source_row=7,
    )
    openmath = materializer.normalize_row(
        "openmathinstruct2",
        {
            "problem": "Solve OpenMath problem",
            "generated_solution": "must not enter the prompt",
            "expected_answer": "7",
            "problem_source": "math",
        },
        source_row=9,
    )

    assert dapo["messages"] == [{"role": "user", "content": "Solve DAPO problem"}]
    assert dapo["expected_answer"] == "42"
    assert openmath["messages"] == [
        {"role": "user", "content": "Solve OpenMath problem"}
    ]
    assert openmath["expected_answer"] == "7"
    assert "generated_solution" not in openmath


def test_dataset_stage_dry_run_pins_revisions_and_avoids_gres() -> None:
    output = run_dry("stage_math_datasets.sh", CLUSTER="lyris")

    assert "BytedTsinghua-SIA/DAPO-Math-17k" in output
    assert "65877096c24ffa7abc4e4fa5edb95cf3413a5674" in output
    assert "nvidia/OpenMathInstruct-2" in output
    assert "469216e3f46f4dacf476b382e192485ea51a143e" in output
    assert "--streaming" in output
    assert "--segment=1" in output
    assert "--gres" not in output


def test_matrix_dry_run_uses_model_runner_v1_and_matched_graph_mode() -> None:
    output = run_dry(
        "submit_matrix.sh",
        CLUSTER="lyris",
        TEMPERATURES="0 1",
        VARIANTS="baseline static dynamic",
    )

    assert output.count("[DRY-RUN] variant=") == 6
    assert "[DRY-RUN] variant=baseline temperature=0" in output
    assert "[DRY-RUN] variant=static temperature=1" in output
    assert "[DRY-RUN] variant=dynamic temperature=1" in output
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in output
    assert "VLLM_USE_V1" not in output
    assert "cudagraph_mode=PIECEWISE" in output
    assert "num_speculative_tokens_per_batch_size" in output
    assert "--segment=1" in output
    assert "--gres" not in output
    assert "--batch-sizes 1 2" in output
    assert "--temperature 0" in output
    assert "--temperature 1" in output


def test_matrix_dry_run_supports_multinode_native_mtp() -> None:
    output = run_dry(
        "submit_matrix.sh",
        CLUSTER="ptyche",
        VARIANTS="baseline mtp_static mtp_dynamic",
        TEMPERATURES="1",
        NODES="2",
        SEGMENT="2",
        TP="8",
        DRAFT_MODEL="",
        DISTRIBUTED_EXECUTOR_BACKEND="ray",
        ENABLE_EXPERT_PARALLEL="true",
    )

    assert output.count("[DRY-RUN] variant=") == 3
    assert "#SBATCH --nodes=2" in output
    assert "#SBATCH --segment=2" in output
    assert "--ntasks=2" in output
    assert "/workspace/experiment/run_multinode_ray.sh" in output
    assert "--distributed-executor-backend 'ray'" in output
    assert "--enable-expert-parallel" in output
    assert "--tensor-parallel-size '8'" in output
    assert "--mode 'mtp_static'" in output
    assert "--mode 'mtp_dynamic'" in output
    assert "method=mtp" in output
    assert "--draft-model ''" in output
    assert "Qwen3-32B-speculator.eagle3" not in output
    assert "--gres" not in output


def test_nemotron_ultra_bf16_mtp_matrix_matches_official_tp8_profile() -> None:
    output = run_dry(
        "submit_nemotron_ultra_bf16_mtp_matrix.sh",
        CLUSTER="ptyche",
        STATIC_K_VALUES="5",
        TEMPERATURES="0 1",
        RUN_ID="ultra-bf16-test",
    )

    assert output.count("[DRY-RUN] variant=") == 6
    assert "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" in output
    assert "snapshots/624ba927cfbef0427354998700de3d51173c8c04" in output
    assert "#SBATCH --nodes=2" in output
    assert "#SBATCH --segment=2" in output
    assert "--tensor-parallel-size '8'" in output
    assert "--distributed-executor-backend 'ray'" in output
    assert "--enable-expert-parallel" in output
    assert "--dtype bfloat16" in output
    assert "--kv-cache-dtype 'fp8'" in output
    assert "--moe-backend 'flashinfer_trtllm'" in output
    assert "--cudagraph-mode 'PIECEWISE'" in output
    assert "--disable-fuse-allreduce-rms" in output
    assert "--model-loader-num-threads '96'" in output
    assert "--distributed-timeout-seconds '3600'" in output
    assert "--mamba-ssm-cache-dtype 'float16'" in output
    assert "--mamba-backend 'flashinfer'" in output
    assert "--enable-mamba-cache-stochastic-rounding" in output
    assert "--mamba-cache-philox-rounds '5'" in output
    assert "--mode 'mtp_static'" in output
    assert "--mode 'mtp_dynamic'" in output
    assert "--static-k '5'" in output
    assert "--temperature 0" in output
    assert "--top-p 1.0" in output
    assert "--temperature 1" in output
    assert "--top-p 0.95" in output
    assert "--gres" not in output


def test_extended_qwen8_matrix_preserves_legacy_methodology() -> None:
    output = run_dry(
        "submit_qwen8_extended_methods_matrix.sh",
        CLUSTER="lyris",
        SMOKE="false",
        DOMAINS="Math SWE",
        TEMPERATURES="0.0 1.0",
    )

    assert output.count("[DRY-RUN] variant=") == 20
    assert "models--Qwen--Qwen3-8B/snapshots/b968826" in output
    assert "models--amd--PARD-Qwen3-0.6B/snapshots/f9f650" in output
    assert "models--amd--PARD2-Qwen3-8B/snapshots/67a151" in output
    assert "models--z-lab--Qwen3-8B-DFlash-b16" in output
    assert "--mode 'suffix'" in output
    assert "--mode 'pard'" in output
    assert "--mode 'pard2'" in output
    assert "--mode 'dflash'" in output
    assert "--static-k '32'" in output
    assert "--static-k '12'" in output
    assert "--static-k '15'" in output
    assert "--isl '4096'" in output
    assert "--osl '32768'" in output
    assert "--batch-sizes 1 2 4 8 16 32" in output
    assert "--max-model-len '40960'" in output
    assert "--enforce-eager" in output
    assert "--attention-backend 'TRITON_ATTN'" in output
    assert "--draft-attention-backend 'FLASH_ATTN'" in output
    assert "PYTHONPATH=" in output
    assert "--gres" not in output


def test_extended_qwen8_matrix_allows_matched_piecewise_cuda_graphs() -> None:
    output = run_dry(
        "submit_qwen8_extended_methods_matrix.sh",
        CLUSTER="lyris",
        SMOKE="true",
        DOMAINS="Math",
        METHODS="baseline suffix",
        TEMPERATURES="1.0",
        RUN_ID="q8-cg",
        ENFORCE_EAGER="false",
        CUDAGRAPH_MODE="PIECEWISE",
    )

    assert output.count("[DRY-RUN] variant=") == 2
    assert "enforce_eager=false" in output
    assert "cudagraph_mode=PIECEWISE" in output
    assert "--cudagraph-mode 'PIECEWISE'" in output
    assert "--enforce-eager" not in output
    assert "/q8-cg_cg-on-piecewise/" in output
    assert "matrix_cg-on-piecewise" in output


def test_long_context_model_view_owns_only_config_and_metadata(
    tmp_path: Path,
) -> None:
    materializer = load_long_context_materializer_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "max_position_embeddings": 40960,
                "rope_scaling": None,
                "rope_theta": 1_000_000,
            }
        ),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "views" / "qwen3-8b"

    metadata = materializer.materialize_model_view(
        source=source,
        destination=destination,
        max_position_embeddings=131072,
        rope_factor=4.0,
    )

    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["Qwen3ForCausalLM"]
    assert config["max_position_embeddings"] == 131072
    assert config["rope_parameters"] == {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "rope_theta": 1_000_000,
    }
    assert "rope_scaling" not in config
    assert not (destination / "config.json").is_symlink()
    assert (destination / "model.safetensors").is_symlink()
    assert (destination / "model.safetensors").resolve() == (
        source / "model.safetensors"
    ).resolve()
    assert (destination / "tokenizer.json").is_symlink()
    assert metadata["source"] == str(source.resolve())
    assert metadata["max_position_embeddings"] == 131072
    assert json.loads(
        (destination / ".long_context_view.json").read_text(encoding="utf-8")
    ) == metadata


def test_long_context_model_view_rejects_unsafe_overwrite(tmp_path: Path) -> None:
    materializer = load_long_context_materializer_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "view"
    destination.mkdir()
    (destination / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        materializer.materialize_model_view(
            source=source,
            destination=destination,
            max_position_embeddings=131072,
            rope_factor=4.0,
        )

    assert (destination / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_qwen8_long_context_matrix_renders_supported_profiles() -> None:
    output = run_dry(
        "submit_qwen8_long_context_matrix.sh",
        CLUSTER="lyris",
        PROFILES="64k 128k",
        DOMAINS="Math SWE",
        TEMPERATURES="0.0 1.0",
    )

    assert output.count("[DRY-RUN] variant=") == 40
    assert "context_profile=64k isl=4096 osl=65536 total=69632" in output
    assert "context_profile=128k isl=4096 osl=126976 total=131072" in output
    assert "long-context-models/yarn4/qwen3-8b" in output
    assert "long-context-models/yarn4/pard-qwen3-0.6b" in output
    assert "long-context-models/yarn4/pard2-qwen3-8b" in output
    assert "long-context-models/yarn4/qwen3-8b-dflash-b16" in output
    assert "--osl '65536'" in output
    assert "--max-model-len '69632'" in output
    assert "--osl '126976'" in output
    assert "--max-model-len '131072'" in output
    assert output.count("--batch-sizes 1") == 40
    assert output.count("--warmup-repeats '0'") == 40
    assert output.count("--measure-repeats '1'") == 40
    assert "--segment=1" in output
    assert "--gres" not in output


def test_extended_assets_stage_dry_run_is_pinned_and_lustre_only() -> None:
    output = run_dry("stage_extended_method_assets.sh", CLUSTER="lyris")
    worker = (
        EXPERIMENT / "stage_extended_method_assets_in_container.sh"
    ).read_text(encoding="utf-8")

    assert "z-lab/Qwen3-8B-DFlash-b16" in worker
    assert "arctic-inference==0.1.1" in worker
    assert "grpcio-tools" in worker
    assert "cmake ninja" in worker
    assert "ee0da84ab9e04ac7610e28580af62c365e898389" in output
    assert "6a97dab2f17c0a3c031065329f092c4f61108a6f" in output
    assert "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038" in output
    assert "vllm024_pard2_target_features.patch" in worker
    assert "angelslim_lightweight_imports.patch" in worker
    assert "angelslim_fixed_length.patch" in worker
    assert "angelslim_split_run_modes.patch" in worker
    assert "angelslim_distributed_timeout.patch" in worker
    assert "angelslim_compact_result_transport.patch" in worker
    assert "angelslim_dflare_transport.py" in worker
    assert "datasets==4.4.1" in worker
    assert "stage_extended_method_assets_in_container.sh" in output
    assert "git clone" not in worker
    assert "urllib.request" in worker
    assert "/home/" not in worker
    assert "--segment=1" in output
    assert "--gres" not in output


def test_angelslim_distributed_timeout_covers_long_context_rank_skew() -> None:
    patch = (
        EXPERIMENT / "patches/angelslim_distributed_timeout.patch"
    ).read_text(encoding="utf-8")

    assert "import datetime" in patch
    assert "timeout=datetime.timedelta(hours=6)" in patch


def test_angelslim_compact_transport_patch_stages_cpu_only_results() -> None:
    worker = (
        EXPERIMENT / "stage_extended_method_assets_in_container.sh"
    ).read_text(encoding="utf-8")
    patch = (
        EXPERIMENT / "patches/angelslim_compact_result_transport.patch"
    ).read_text(encoding="utf-8")
    import_marker = (
        "from angelslim_dflare_transport import "
        "compact_response_map, write_rank_partial"
    )
    append_marker = "responses.append(compact_response_map(response))"
    partial_marker = "write_rank_partial(args.output_json, _dist_rank(), responses)"

    assert "${angelslim_source}/tools/angelslim_dflare_transport.py" in worker
    assert "angelslim_compact_result_transport.patch" in worker
    assert import_marker in worker
    assert append_marker in worker
    assert partial_marker in worker
    assert 'state_count=$((has_import + has_compact_append + has_partial_write))' in worker
    assert 'if [[ "${state_count}" == "3" ]]; then' in worker
    assert 'elif [[ "${state_count}" == "0" ]]; then' in worker
    assert "partial AngelSlim compact transport patch state" in worker
    assert import_marker in patch
    assert "compact_response_map" in patch
    assert "write_rank_partial" in patch
    assert "responses.append(response)" in patch
    assert append_marker in patch
    assert partial_marker in patch
    assert patch.index(partial_marker) < (
        patch.index("if _dist_size() > 1:")
    )


def test_angelslim_matrix_keeps_native_results_separate() -> None:
    output = run_dry(
        "submit_angelslim_matrix.sh",
        CLUSTER="lyris",
        METHODS="dflash dflare",
        DOMAINS="Math SWE",
        TEMPERATURES="0.0 1.0",
    )

    assert output.count("[DRY-RUN] native_method=") == 8
    assert "--draft-arch 'dflash'" in output
    assert "--draft-arch 'dflare'" in output
    assert "--block-size '16'" in output
    assert "--input-length '4096'" in output
    assert "--ignore-eos" in output
    assert "backend=angelslim_transformers_native" in output
    assert "angelslim_runtime" in output
    assert "--segment=1" in output
    assert "--gres" not in output


def test_angelslim_matrix_supports_spec_only_execution() -> None:
    output = run_dry(
        "submit_angelslim_matrix.sh",
        CLUSTER="lyris",
        METHODS="dflare",
        DOMAINS="Math",
        TEMPERATURES="0.0",
        RUN_MODE="spec",
    )

    assert output.count("[DRY-RUN] native_method=") == 1
    assert "run_mode=spec" in output
    assert "--run-mode 'spec'" in output
    assert "q8-math-dflare-spec-t0p0" in output
    assert "dflare_spec_t0p0" in output


def test_angelslim_long_context_dflare_renders_parallel_spec_jobs() -> None:
    output = run_dry(
        "submit_angelslim_long_context_dflare.sh",
        CLUSTER="lyris",
        PROFILES="64k 128k",
        DOMAINS="Math SWE",
        TEMPERATURES="0.0 1.0",
    )

    assert output.count("[DRY-RUN] native_method=dflare") == 8
    assert "context_profile=64k isl=4096 osl=65536 total=69632" in output
    assert "context_profile=128k isl=4096 osl=126976 total=131072" in output
    assert "long-context-models/yarn4/qwen3-8b" in output
    assert "long-context-models/yarn4/qwen3-8b-dflare" in output
    assert output.count("--run-mode 'spec'") == 8
    assert output.count("--max-new-tokens '65536'") == 4
    assert output.count("--max-new-tokens '126976'") == 4
    assert output.count("#SBATCH --partition=gb200") == 8
    assert "--segment=1" in output
    assert "--gres" not in output


def test_angelslim_long_context_dflare_supports_exact_baseline_jobs() -> None:
    output = run_dry(
        "submit_angelslim_long_context_dflare.sh",
        CLUSTER="lyris",
        PROFILES="64k",
        DOMAINS="Math SWE",
        TEMPERATURES="0.0 1.0",
        RUN_MODE="baseline",
        PARTITION="gb200-backfill",
        TIME_LIMIT="08:00:00",
    )

    assert output.count("[DRY-RUN] native_method=dflare") == 4
    assert output.count("--run-mode 'baseline'") == 4
    assert output.count("#SBATCH --partition=gb200-backfill") == 4
    assert output.count("#SBATCH --time=08:00:00") == 4
    assert "_qwen8_dflare_baseline" in output


def test_nsys_dry_run_profiles_one_steady_state_measurement() -> None:
    output = run_dry(
        "submit_nsys.sh",
        CLUSTER="lyris",
        VARIANTS="baseline dynamic",
        PROFILE_BATCH_SIZE="16",
    )

    assert output.count("[DRY-RUN] nsys_variant=") == 2
    assert "--capture-range=cudaProfilerApi" in output
    assert "--capture-range-end=stop" in output
    assert "--sample=none" in output
    assert "--cpuctxsw=none" in output
    assert "--batch-sizes 16" in output
    assert "--measure-repeats 1" in output
    assert "--cuda-profiler-range" in output
    assert "--segment=1" in output
    assert "--gres" not in output


def test_resolve_request_plan_is_deterministic_and_exact() -> None:
    core = load_sync_rollout_core_module()
    plan = core.load_request_plan(EXPERIMENT / "profiles/swe_sync_32k.json")

    first = core.resolve_request_plan(
        plan,
        prompt_ids=[f"p{i}" for i in range(16)],
        samples_per_prompt=4,
        seed_start=7,
    )
    second = core.resolve_request_plan(
        plan,
        prompt_ids=[f"p{i}" for i in range(16)],
        samples_per_prompt=4,
        seed_start=7,
    )

    assert first == second
    assert sum(request.max_tokens for request in first) == 589824
    assert {request.max_tokens for request in first} == {4096, 8192, 16384, 32768}
    assert Counter(request.max_tokens for request in first) == {
        4096: 32,
        8192: 16,
        16384: 12,
        32768: 4,
    }
    by_prompt = {request.prompt_id: request.max_tokens for request in first}
    assert Counter(by_prompt.values()) == {
        4096: 8,
        8192: 4,
        16384: 3,
        32768: 1,
    }


def test_resolve_request_plan_uses_rollout_batch_index_for_unique_seed_ranges() -> None:
    core = load_sync_rollout_core_module()
    plan = core.load_request_plan(EXPERIMENT / "profiles/swe_sync_32k.json")

    first = core.resolve_request_plan(
        plan,
        prompt_ids=[f"p{i}" for i in range(16)],
        samples_per_prompt=4,
        seed_start=7,
        rollout_batch_index=0,
    )
    second = core.resolve_request_plan(
        plan,
        prompt_ids=[f"p{i}" for i in range(16)],
        samples_per_prompt=4,
        seed_start=7,
        rollout_batch_index=1,
    )

    assert [request.max_tokens for request in second] == [
        request.max_tokens for request in first
    ]
    assert [request.seed for request in first] == list(range(7, 71))
    assert [request.seed for request in second] == list(range(71, 135))


def test_request_plan_hash_is_stable_for_equivalent_json(tmp_path: Path) -> None:
    core = load_sync_rollout_core_module()
    path = EXPERIMENT / "profiles/swe_sync_32k.json"
    plan = core.load_request_plan(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    reordered = {
        "buckets": list(reversed(payload["buckets"])),
        "max_model_len": payload["max_model_len"],
        "name": payload["name"],
    }
    rewritten = tmp_path / "rewritten_swe_sync_32k.json"
    rewritten.write_text(json.dumps(reordered, indent=2) + "\n", encoding="utf-8")
    rewritten_plan = core.load_request_plan(rewritten)

    assert rewritten_plan.plan_hash == plan.plan_hash


def test_request_plan_validates_context_overflow() -> None:
    core = load_sync_rollout_core_module()

    core.validate_context_window(
        prompt_tokens=4096,
        output_cap=32768,
        max_model_len=36864,
    )
    with pytest.raises(
        ValueError,
        match="context overflow: prompt=4097 output=32768 max=36864",
    ):
        core.validate_context_window(
            prompt_tokens=4097,
            output_cap=32768,
            max_model_len=36864,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("ignore_eos", "true", "ignore_eos must be a boolean"),
        ("max_tokens", 4096.5, "max_tokens must be an integer"),
        ("weight", 8.5, "weight must be an integer"),
        ("max_model_len", 36864.5, "max_model_len must be an integer"),
    ),
)
def test_load_request_plan_rejects_invalid_json_types(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    core = load_sync_rollout_core_module()
    payload = json.loads(
        (EXPERIMENT / "profiles/swe_sync_32k.json").read_text(encoding="utf-8")
    )
    if field == "max_model_len":
        payload[field] = value
    else:
        payload["buckets"][0][field] = value
    invalid_path = tmp_path / "invalid_request_plan.json"
    invalid_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(TypeError, match=match):
        core.load_request_plan(invalid_path)


@pytest.mark.parametrize(
    ("value", "expected_exception", "match"),
    (
        (123, TypeError, "name must be a string"),
        ([], TypeError, "name must be a string"),
        ({}, TypeError, "name must be a string"),
        ("", ValueError, "name must be a non-empty string"),
    ),
)
def test_load_request_plan_rejects_invalid_name_values(
    tmp_path: Path,
    value: object,
    expected_exception: type[Exception],
    match: str,
) -> None:
    core = load_sync_rollout_core_module()
    payload = json.loads(
        (EXPERIMENT / "profiles/swe_sync_32k.json").read_text(encoding="utf-8")
    )
    payload["name"] = value
    invalid_path = tmp_path / "invalid_name_request_plan.json"
    invalid_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(expected_exception, match=match):
        core.load_request_plan(invalid_path)


def test_load_request_plan_reads_expected_swe_sync_64k_profile() -> None:
    core = load_sync_rollout_core_module()
    plan = core.load_request_plan(EXPERIMENT / "profiles/swe_sync_64k.json")

    assert [bucket.max_tokens for bucket in plan.buckets] == [4096, 8192, 16384, 65536]
    assert [bucket.weight for bucket in plan.buckets] == [8, 4, 3, 1]


def test_summarize_barrier_tail_uses_max_minus_median() -> None:
    core = load_sync_rollout_core_module()

    summary = core.summarize_barrier_tail([1.0, 2.0, 10.0])

    assert summary["median_s"] == 2.0
    assert summary["tail_gap_s"] == 8.0


def test_sync_rollout_expands_prompt_samples_with_unique_seeds() -> None:
    sync_rollout = load_sync_rollout_module()

    requests = sync_rollout.expand_prompt_samples(
        [[10, 11], [20, 21]],
        samples_per_prompt=3,
        seed_start=100,
    )

    assert [request[0] for request in requests] == [
        [10, 11],
        [10, 11],
        [10, 11],
        [20, 21],
        [20, 21],
        [20, 21],
    ]
    assert [request[1] for request in requests] == [100, 101, 102, 103, 104, 105]


def test_sync_rollout_renders_chat_template_before_tokenizing() -> None:
    sync_rollout = load_sync_rollout_module()

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert messages == [{"role": "user", "content": "solve it"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "rendered chat prompt"

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "rendered chat prompt"
            assert add_special_tokens is False
            return [10, 20, 30]

    assert sync_rollout.tokenize_prompt(FakeTokenizer(), "solve it", 2) == [20, 30]


def test_tokenize_prompt_rejects_truncation_when_disabled() -> None:
    sync_rollout = load_sync_rollout_module()

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "x"
            assert add_special_tokens is True
            return list(range(32))

    with pytest.raises(ValueError, match="prompt exceeds max_prompt_tokens"):
        sync_rollout.tokenize_prompt(
            FakeTokenizer(),
            "x",
            16,
            allow_truncation=False,
        )


def test_sync_rollout_request_plan_controls_sampling_and_provenance() -> None:
    sync_rollout = load_sync_rollout_module()
    core = load_sync_rollout_core_module()
    plan = core.load_request_plan(EXPERIMENT / "profiles/swe_sync_32k.json")
    prompt_records = [
        sync_rollout.PromptRecord(
            prompt_id=f"prompt-{index}",
            token_ids=[index + 1],
            prompt_sha256=f"hash-{index}",
            source_prompt_sha256=f"source-hash-{index}",
        )
        for index in range(4)
    ]

    requests = sync_rollout.prepare_rollout_requests(
        prompt_records,
        request_plan=plan,
        samples_per_prompt=2,
        seed_start=100,
        rollout_batch_index=0,
        max_model_len=plan.max_model_len,
    )

    assert [request.prompt_id for request in requests] == [
        "prompt-0",
        "prompt-0",
        "prompt-1",
        "prompt-1",
        "prompt-2",
        "prompt-2",
        "prompt-3",
        "prompt-3",
    ]
    assert [request.seed for request in requests] == list(range(100, 108))
    assert [request.max_tokens for request in requests] == [
        4096,
        4096,
        4096,
        4096,
        8192,
        8192,
        16384,
        16384,
    ]
    assert all(request.ignore_eos for request in requests)

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    params = sync_rollout.build_sampling_params(
        FakeSamplingParams,
        requests,
        temperature=1.0,
        top_p=0.95,
    )

    assert [param.kwargs["max_tokens"] for param in params] == [
        request.max_tokens for request in requests
    ]
    assert [param.kwargs["min_tokens"] for param in params] == [
        request.min_tokens for request in requests
    ]
    assert [param.kwargs["ignore_eos"] for param in params] == [
        request.ignore_eos for request in requests
    ]
    assert [param.kwargs["seed"] for param in params] == list(range(100, 108))


def test_sync_rollout_hashes_actual_tokenized_prompt_and_preserves_source_hash(
    tmp_path: Path,
) -> None:
    sync_rollout = load_sync_rollout_module()
    prompt_jsonl = tmp_path / "prompts.jsonl"
    prompt_jsonl.write_text(
        json.dumps(
            {
                "id": "swe-1",
                "prompt_sha256": "source-provided-hash",
                "messages": [{"role": "user", "content": "fix bug"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert messages == [{"role": "user", "content": "fix bug"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "rendered prompt with assistant prefix"

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "rendered prompt with assistant prefix"
            assert add_special_tokens is False
            return [101, 202, 303]

    batches = sync_rollout.load_prompt_batches(
        FakeTokenizer(),
        prompt_jsonl=prompt_jsonl,
        prompt_offset=0,
        num_prompts=1,
        rollout_batches=1,
        max_prompt_tokens=16,
    )

    record = batches[0][0]
    assert record.prompt_id == "swe-1"
    assert record.prompt_sha256 == sync_rollout.token_hash([101, 202, 303])
    assert record.source_prompt_sha256 == "source-provided-hash"


def test_sync_rollout_response_jsonl_and_bucket_stats_preserve_provenance(
    tmp_path: Path,
) -> None:
    sync_rollout = load_sync_rollout_module()
    request = sync_rollout.RolloutRequest(
        prompt_id="prompt-0",
        prompt_sha256="prompt-hash",
        source_prompt_sha256="source-prompt-hash",
        sample_index=0,
        seed=7,
        prompt_token_ids=[1, 2, 3],
        max_tokens=4096,
        min_tokens=4096,
        ignore_eos=True,
    )

    class FakeCandidate:
        token_ids = [10, 11, 12]
        text = "patched answer"
        finish_reason = "length"

    class FakeOutput:
        outputs = [FakeCandidate()]

    responses_path = tmp_path / "responses.jsonl"
    sync_rollout.write_response_jsonl(
        responses_path,
        batch_index=0,
        requests=[request],
        outputs=[FakeOutput()],
        append=False,
    )

    rows = [
        json.loads(line)
        for line in responses_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "batch_index": 0,
            "prompt_id": "prompt-0",
            "prompt_sha256": "prompt-hash",
            "source_prompt_sha256": "source-prompt-hash",
            "sample_index": 0,
            "seed": 7,
            "max_tokens": 4096,
            "min_tokens": 4096,
            "ignore_eos": True,
            "finish_reason": "length",
            "output_tokens": 3,
            "output_token_hash": sync_rollout.token_hash([10, 11, 12]),
            "text": "patched answer",
        }
    ]
    assert sync_rollout.bucket_statistics([request], [[10, 11, 12]]) == [
        {
            "max_tokens": 4096,
            "request_count": 1,
            "output_tokens": 3,
            "completion_length": {
                "min": 3,
                "mean": 3.0,
                "p50": 3,
                "p90": 3,
                "p99": 3,
                "max": 3,
            },
        }
    ]


def test_sync_rollout_exact_output_work_emits_counts_and_rejects_underfill() -> None:
    sync_rollout = load_sync_rollout_module()
    forced = sync_rollout.RolloutRequest(
        prompt_id="prompt-0",
        prompt_sha256="prompt-hash",
        source_prompt_sha256=None,
        sample_index=0,
        seed=7,
        prompt_token_ids=[1, 2, 3],
        max_tokens=4,
        min_tokens=4,
        ignore_eos=True,
    )
    flexible = sync_rollout.RolloutRequest(
        prompt_id="prompt-1",
        prompt_sha256="prompt-hash-1",
        source_prompt_sha256=None,
        sample_index=0,
        seed=8,
        prompt_token_ids=[4, 5, 6],
        max_tokens=4,
        min_tokens=0,
        ignore_eos=False,
    )

    assert sync_rollout.exact_output_work([forced, flexible], [[1, 2, 3, 4], [9]]) == {
        "planned_output_tokens": [4, 4],
        "actual_output_tokens": [4, 1],
        "forced_output_mask": [True, False],
    }
    with pytest.raises(ValueError, match="forced output length mismatch"):
        sync_rollout.exact_output_work([forced], [[1, 2, 3]])


def test_sync_rollout_exact_output_work_treats_equal_min_max_as_forced() -> None:
    sync_rollout = load_sync_rollout_module()
    request = sync_rollout.RolloutRequest(
        prompt_id="prompt-0",
        prompt_sha256="prompt-hash",
        source_prompt_sha256=None,
        sample_index=0,
        seed=7,
        prompt_token_ids=[1, 2, 3],
        max_tokens=4,
        min_tokens=4,
        ignore_eos=False,
    )

    assert sync_rollout.exact_output_work([request], [[1, 2, 3, 4]]) == {
        "planned_output_tokens": [4],
        "actual_output_tokens": [4],
        "forced_output_mask": [True],
    }
    with pytest.raises(ValueError, match="forced output length mismatch"):
        sync_rollout.exact_output_work([request], [[1, 2, 3]])


def test_sync_rollout_generated_runner_executes_with_hostile_paths(
    tmp_path: Path,
) -> None:
    pwned = tmp_path / "pwned"
    hostile = f"{tmp_path}/odd ' \" $DOLLAR $(touch {pwned}) path"
    output = run_dry(
        "submit_sync_rollout.sh",
        CLUSTER="lyris",
        VARIANTS="baseline",
        MODEL=f"{hostile}/target",
        DRAFT_MODEL=f"{hostile}/draft",
        REQUEST_PLAN=f"{hostile}/host-plan.json",
        REQUEST_PLAN_IN_CONTAINER=f"{hostile}/container-plan.json",
        RESOLVED_REQUEST_PLAN_OUTPUT=f"{hostile}/resolved.json",
        RESPONSE_OUTPUT=f"{hostile}/responses.jsonl",
        RUNTIME_IMAGE_SHA256="runtime-sha-hostile",
    )
    script = extract_run_benchmark_script(output, "baseline")
    run_script = tmp_path / "run_benchmark.sh"
    run_script.write_text(script, encoding="utf-8")
    run_script.chmod(0o755)
    argv_path = tmp_path / "argv.txt"
    stub_python = tmp_path / "python-stub"
    stub_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ": >\"${STUB_ARGV_OUT:?}\"\n"
        "for arg in \"$@\"; do printf '%s\\n' \"$arg\" >>\"${STUB_ARGV_OUT}\"; done\n",
        encoding="utf-8",
    )
    stub_python.chmod(0o755)

    subprocess.run(
        [str(run_script)],
        cwd=tmp_path,
        env={
            "PATH": "/usr/bin:/bin",
            "BENCHMARK_PYTHON": str(stub_python),
            "CHECK_VLLM_VERSION": "false",
            "STUB_ARGV_OUT": str(argv_path),
            "DOLLAR": "expanded-if-unsafe",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    argv = argv_path.read_text(encoding="utf-8").splitlines()

    def value_after(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert argv[0] == "/workspace/experiment/benchmark_sync_rollout.py"
    assert value_after("--model") == f"{hostile}/target"
    assert value_after("--draft-model") == f"{hostile}/draft"
    assert value_after("--request-plan") == f"{hostile}/container-plan.json"
    assert value_after("--resolved-request-plan-output") == f"{hostile}/resolved.json"
    assert value_after("--response-output") == f"{hostile}/responses.jsonl"
    assert value_after("--runtime-image-sha256") == "runtime-sha-hostile"
    assert "$DOLLAR" in value_after("--model")
    assert "$(touch " in value_after("--model")
    assert not pwned.exists()


def test_sync_rollout_dry_run_models_barriered_rl_sampling() -> None:
    output = run_dry(
        "submit_sync_rollout.sh",
        CLUSTER="lyris",
        VARIANTS="baseline static dynamic",
    )

    assert output.count("[DRY-RUN] sync_variant=") == 3
    assert "--temperature 1.0" in output
    assert "--top-p 0.9" in output
    assert "--num-prompts 4" in output
    assert "--samples-per-prompt 2" in output
    assert "--rollout-batches 2" in output
    assert "--engine-max-num-seqs 64" in output
    assert "sync_barrier=LLM.generate_return" in output
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in output
    assert "VLLM_USE_V1" not in output
    assert "cudagraph_mode=PIECEWISE" in output
    assert "--segment=1" in output
    assert "--gres" not in output


def test_sync_rollout_dry_run_supports_native_mtp_tp8() -> None:
    output = run_dry(
        "submit_sync_rollout.sh",
        CLUSTER="ptyche",
        VARIANTS="baseline mtp_static mtp_dynamic",
        DRAFT_MODEL="",
        NODES="2",
        SEGMENT="2",
        TP="8",
        DISTRIBUTED_EXECUTOR_BACKEND="ray",
        ENABLE_EXPERT_PARALLEL="true",
        KV_CACHE_DTYPE="fp8",
        MAMBA_SSM_CACHE_DTYPE="float16",
        MAMBA_BACKEND="flashinfer",
    )

    assert output.count("[DRY-RUN] sync_variant=") == 3
    assert "#SBATCH --nodes=2" in output
    assert "#SBATCH --segment=2" in output
    assert "args+=(--mode mtp_static)" in output
    assert "args+=(--mode mtp_dynamic)" in output
    assert "args+=(--draft-model '')" in output
    assert "Qwen3-32B-speculator.eagle3" not in output
    assert "args+=(--distributed-executor-backend ray)" in output
    assert "args+=(--enable-expert-parallel)" in output
    assert "args+=(--kv-cache-dtype fp8)" in output
    assert "args+=(--mamba-ssm-cache-dtype float16)" in output
    assert "args+=(--mamba-backend flashinfer)" in output


def test_swe_sync_rollout_matrix_renders_request_plan_and_response_outputs() -> None:
    output = run_dry(
        "submit_swe_sync_rollout_matrix.sh",
        CLUSTER="lyris",
        SMOKE="true",
        MODELS="qwen32",
        REQUEST_PROFILES="32k",
        TEMPERATURES="0.0",
        VARIANTS="baseline dynamic",
        RUN_ID="swe-test",
    )

    assert "swe_sync_model=qwen32" in output
    assert "request_profile=32k" in output
    assert "request_plan_hash=" in output
    assert output.count("[DRY-RUN] sync_variant=") == 2
    assert "args+=(--request-plan /workspace/experiment/profiles/swe_sync_32k.json)" in output
    assert "args+=(--response-output " in output
    assert "responses.jsonl" in output
    assert "--runtime-image-sha256" in output
    assert "args+=(--max-model-len 36864)" in output
    assert "args+=(--num-prompts 16)" in output
    assert "args+=(--samples-per-prompt 1)" in output
    assert "requests_per_rollout_batch=16" in output
    assert "args+=(--max-prompt-tokens 4096)" in output
    assert "args+=(--max-new-tokens 32768)" in output
    assert "args+=(--temperature 0.0)" in output
    assert "args+=(--top-p 0.95)" in output
    assert "swebench_verified_prompts_all.jsonl" in output
    assert "long-context-models/yarn4" not in output
    assert "materialize_long_context_model_views.py" not in output
    assert "pinned RL math dataset" not in output


def test_model_method_matrix_has_unique_large_model_profile_keys() -> None:
    matrix = load_model_method_matrix()

    assert matrix["schema_version"] == 1
    seen: set[tuple[str, str, str]] = set()
    total = 0
    for model in matrix["models"]:
        for profile in model["profiles"]:
            for method_key in matrix["method_order"]:
                key = (model["key"], profile["key"], method_key)
                assert key not in seen
                seen.add(key)
                total += 1

    assert len(seen) == total
    qwen235 = get_matrix_model(matrix, "qwen235b")
    profile64 = get_matrix_profile(qwen235, "64k")
    ultra = get_matrix_model(matrix, "ultra")

    assert qwen235["topology"]["target_tp"] == 8
    assert profile64["context_policy"] == "yarn4_64k"
    assert profile64["rope_factor"] == 4.0
    assert profile64["max_position_embeddings"] == 131072
    assert ultra["topology"]["nodes"] == 2
    assert ultra["topology"]["segment"] == 2
    assert ultra["topology"]["model_loader_threads"] == 96


def test_large_model_matrix_rejects_qwen8_only_dflash() -> None:
    matrix = load_model_method_matrix()
    qwen32 = get_matrix_model(matrix, "qwen32")

    assert qwen32["methods"]["dflash"]["status"] == "unsupported"
    assert qwen32["methods"]["dflash"]["reason_code"] == "qwen3_8b_public_asset_only"


def test_large_model_matrix_marks_pard_and_pard2_per_approved_compatibility() -> None:
    matrix = load_model_method_matrix()
    qwen30 = get_matrix_model(matrix, "qwen30ba3b")
    qwen32 = get_matrix_model(matrix, "qwen32")
    qwen235 = get_matrix_model(matrix, "qwen235b")
    ultra = get_matrix_model(matrix, "ultra")

    assert qwen30["methods"]["pard"]["status"] == "integration"
    assert qwen30["methods"]["pard"]["reason_code"] == "runner_support_missing"
    assert qwen32["methods"]["pard2"]["status"] == "unsupported"
    assert qwen32["methods"]["pard2"]["reason_code"] == "exact_target_checkpoint_missing"
    assert qwen235["methods"]["pard2"]["status"] == "unsupported"
    assert qwen235["methods"]["pard2"]["reason_code"] == "not_validated"
    assert ultra["methods"]["eagle3"]["status"] == "unsupported"
    assert ultra["methods"]["mtp_static"]["status"] == "supported"
    assert ultra["methods"]["mtp_dynamic"]["status"] == "supported"


def test_swe_sync_rollout_64k_uses_matched_yarn_target_and_draft_views() -> None:
    output = run_dry(
        "submit_swe_sync_rollout_matrix.sh",
        CLUSTER="lyris",
        SMOKE="true",
        MODELS="qwen32",
        REQUEST_PROFILES="64k",
        TEMPERATURES="0.0",
        VARIANTS="baseline static",
        RUN_ID="swe64-test",
    )

    assert "request_profile=64k" in output
    assert "materialize_long_context_model_views.py" in output
    assert "--max-position-embeddings 131072" in output
    assert "--rope-factor 4.0" in output
    assert "--model-view qwen32-target=" in output
    assert "--model-view qwen32-eagle3-draft=" in output
    assert "long-context-models/yarn4/qwen32-target" in output
    assert "long-context-models/yarn4/qwen32-eagle3-draft" in output
    assert "args+=(--model /lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/long-context-models/yarn4/qwen32-target)" in output
    assert "args+=(--draft-model /lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/long-context-models/yarn4/qwen32-eagle3-draft)" in output
    assert "args+=(--max-model-len 69632)" in output
    assert "args+=(--max-new-tokens 65536)" in output


def test_swe_sync_rollout_non_smoke_defaults_to_primary_four_samples() -> None:
    output = run_dry(
        "submit_swe_sync_rollout_matrix.sh",
        CLUSTER="lyris",
        SMOKE="false",
        MODELS="qwen32",
        REQUEST_PROFILES="32k",
        TEMPERATURES="0.0",
        VARIANTS="baseline",
        RUN_ID="swe-full-default",
    )

    assert "--num-prompts 16" in output
    assert "--samples-per-prompt 4" in output
    assert "--rollout-batches 3" in output
    assert "--samples-per-prompt 16" not in output


def test_swe_sync_rollout_full_contract_override_uses_sixteen_samples() -> None:
    output = run_dry(
        "submit_swe_sync_rollout_matrix.sh",
        CLUSTER="lyris",
        SMOKE="false",
        FULL_CONTRACT="true",
        MODELS="qwen32",
        REQUEST_PROFILES="32k",
        TEMPERATURES="0.0",
        VARIANTS="baseline",
        RUN_ID="swe-full-contract",
    )

    assert "full_contract=true" in output
    assert "--num-prompts 16" in output
    assert "--samples-per-prompt 16" in output
    assert "--rollout-batches 3" in output


def test_sync_rollout_rejects_shared_explicit_outputs_for_multi_variant_runs() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "DRY_RUN": "true",
        "CLUSTER": "lyris",
        "VARIANTS": "baseline dynamic",
        "RESPONSE_OUTPUT": "/tmp/shared-responses.jsonl",
        "RESOLVED_REQUEST_PLAN_OUTPUT": "/tmp/shared-plan.json",
    }
    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "submit_sync_rollout.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "RESPONSE_OUTPUT must be auto or contain {variant}" in completed.stderr


def test_sync_rollout_accepts_variant_placeholders_and_shell_escapes_new_args() -> None:
    output = run_dry(
        "submit_sync_rollout.sh",
        CLUSTER="lyris",
        VARIANTS="baseline dynamic",
        RESPONSE_OUTPUT="/tmp/o'hara/{variant}/responses.jsonl",
        RESOLVED_REQUEST_PLAN_OUTPUT="/tmp/o'hara/{variant}/plan.json",
    )

    assert "/tmp/o\\'hara/baseline/responses.jsonl" in output
    assert "/tmp/o\\'hara/dynamic/responses.jsonl" in output
    assert "args+=(--response-output /tmp/o\\'hara/baseline/responses.jsonl)" in output
    assert "args+=(--resolved-request-plan-output /tmp/o\\'hara/dynamic/plan.json)" in output
    assert "--response-output '/tmp/o'hara" not in output
    assert "--resolved-request-plan-output '/tmp/o'hara" not in output


def test_swe_sync_rollout_dry_run_does_not_call_sbatch_or_mutate_dirs(
    tmp_path: Path,
) -> None:
    view_root = tmp_path / "views"
    result_root = tmp_path / "results"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    sbatch = stub_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'called\\n' >>\"${SBATCH_LOG:?}\"\n"
        "exit 64\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "submit_swe_sync_rollout_matrix.sh")],
        cwd=ROOT,
        env={
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "SBATCH_LOG": str(sbatch_log),
            "CLUSTER": "lyris",
            "DRY_RUN": "true",
            "TEST_ONLY": "false",
            "REQUIRE_GIT_PULL": "false",
            "MODELS": "qwen32",
            "REQUEST_PROFILES": "64k",
            "TEMPERATURES": "0.0",
            "VARIANTS": "baseline",
            "RUN_ID": "plan-mode",
            "LONG_CONTEXT_VIEW_ROOT": str(view_root),
            "RESULT_ROOT": str(result_root),
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[DRY-RUN]" in completed.stdout
    assert "materialize_long_context_model_views.py" in completed.stdout
    assert str(view_root / "qwen32-target") in completed.stdout
    assert str(view_root / "qwen32-eagle3-draft") in completed.stdout
    assert "run_benchmark.sh" in completed.stdout
    manifest = extract_manifest_path(completed.stdout)
    manifest_lines = extract_manifest_rows(completed.stdout)

    assert not sbatch_log.exists()
    assert not view_root.exists()
    assert not result_root.exists()
    assert not manifest.exists()
    assert manifest.parent != result_root / "plan-mode"
    assert any(
        line.startswith("SUPPORTED\tqwen32\t64k\tbaseline\tbaseline\t0.0\t")
        for line in manifest_lines
    )
    assert any("\tINTEGRATION\t" not in line for line in manifest_lines[1:])
    assert any(
        line.startswith(
            "INTEGRATION\tqwen32\t64k\tpard\t-\t0.0\t-\trunner_support_missing\t"
        )
        for line in manifest_lines
    )
    assert any(
        line.startswith(
            "UNSUPPORTED\tqwen32\t64k\tdflash\t-\t0.0\t-\tqwen3_8b_public_asset_only\t"
        )
        for line in manifest_lines
    )


def test_swe_sync_rollout_test_only_invokes_sbatch_and_cleans_temp_artifacts(
    tmp_path: Path,
) -> None:
    view_root = tmp_path / "views"
    result_root = tmp_path / "results"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    sbatch = stub_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "saw_test_only=false\n"
        "script_path=\n"
        "for arg in \"$@\"; do\n"
        "  printf 'arg=%s\\n' \"$arg\" >>\"${SBATCH_LOG:?}\"\n"
        "  if [[ \"$arg\" == '--test-only' ]]; then saw_test_only=true; fi\n"
        "  script_path=\"$arg\"\n"
        "done\n"
        "if [[ \"${saw_test_only}\" != true ]]; then exit 64; fi\n"
        "test -f \"${script_path}\"\n"
        "test -f \"$(dirname \"${script_path}\")/run_benchmark.sh\"\n"
        "test -x \"$(dirname \"${script_path}\")/run_benchmark.sh\"\n"
        "grep -q 'run_benchmark.sh' \"${script_path}\"\n"
        "printf 'script=%s\\n' \"${script_path}\" >>\"${SBATCH_LOG}\"\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "submit_swe_sync_rollout_matrix.sh")],
        cwd=ROOT,
        env={
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "SBATCH_LOG": str(sbatch_log),
            "CLUSTER": "lyris",
            "DRY_RUN": "false",
            "TEST_ONLY": "true",
            "REQUIRE_GIT_PULL": "false",
            "MODELS": "qwen32",
            "REQUEST_PROFILES": "64k",
            "TEMPERATURES": "0.0",
            "VARIANTS": "baseline",
            "RUN_ID": "test-only",
            "LONG_CONTEXT_VIEW_ROOT": str(view_root),
            "RESULT_ROOT": str(result_root),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    log_lines = sbatch_log.read_text(encoding="utf-8").splitlines()
    script_paths = [
        Path(line.split("=", 1)[1])
        for line in log_lines
        if line.startswith("script=")
    ]

    assert "[TEST-ONLY] python3" in completed.stdout
    assert "[TEST-ONLY] sync_variant=baseline" in completed.stdout
    assert "arg=--test-only" in log_lines
    assert len(script_paths) == 1
    assert not script_paths[0].exists()
    assert not script_paths[0].parent.exists()
    assert not view_root.exists()
    manifest = extract_manifest_path(completed.stdout)
    assert not result_root.exists()
    assert not manifest.exists()
    assert "INTEGRATION\tqwen32\t64k\tpard\t-\t0.0" in completed.stdout


def test_swe_sync_rollout_exits_on_unknown_model_without_reusing_state() -> None:
    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "submit_swe_sync_rollout_matrix.sh")],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "true",
            "CLUSTER": "lyris",
            "REQUIRE_GIT_PULL": "false",
            "MODELS": "qwen32 doesnotexist",
            "REQUEST_PROFILES": "32k",
            "TEMPERATURES": "0.0",
            "VARIANTS": "baseline",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "doesnotexist" in completed.stderr
    assert extract_sync_variants(completed.stdout) == ["baseline"]
    assert "swe_sync_model=doesnotexist" not in completed.stdout
    assert "/doesnotexist/" not in completed.stdout


def test_swe_sync_rollout_exits_on_unknown_profile_without_reusing_state() -> None:
    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "submit_swe_sync_rollout_matrix.sh")],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "true",
            "CLUSTER": "lyris",
            "REQUIRE_GIT_PULL": "false",
            "MODELS": "qwen32",
            "REQUEST_PROFILES": "32k missing",
            "TEMPERATURES": "0.0",
            "VARIANTS": "baseline",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "missing" in completed.stderr
    assert extract_sync_variants(completed.stdout) == ["baseline"]
    assert "request_profile=missing" not in completed.stdout
    assert "/missing/" not in completed.stdout


def test_swe_sync_rollout_submits_exact_supported_variants_only() -> None:
    output = run_dry(
        "submit_swe_sync_rollout_matrix.sh",
        CLUSTER="lyris",
        MODELS="qwen32",
        REQUEST_PROFILES="32k",
        TEMPERATURES="0.0",
        VARIANTS="baseline static dynamic",
    )

    assert extract_sync_variants(output) == ["baseline", "static", "dynamic"]
    assert "[DRY-RUN] sync_variant=pard" not in output
    assert "[DRY-RUN] sync_variant=pard2" not in output


def test_sync_rollout_smoke_false_prompt_requirement_is_domain_neutral() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "DRY_RUN": "true",
        "CLUSTER": "lyris",
        "SMOKE": "false",
        "PROMPT_JSONL": "",
    }
    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "submit_sync_rollout.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "SMOKE=false requires PROMPT_JSONL" in completed.stderr
    assert "math dataset" not in completed.stderr


def test_nemotron_sync_rl_wrapper_covers_ultra_and_super_bf16() -> None:
    output = run_dry(
        "submit_nemotron_sync_rl_mtp_matrix.sh",
        CLUSTER="ptyche",
        MODELS="ultra super",
        RUN_ID="sync-bf16-test",
    )

    assert output.count("[DRY-RUN] sync_variant=") == 6
    assert "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" in output
    assert "NVIDIA-Nemotron-3-Super-120B-A12B-BF16" in output
    assert "#SBATCH --nodes=2" in output
    assert "#SBATCH --segment=2" in output
    assert "args+=(--tensor-parallel-size 8)" in output
    assert "#SBATCH --nodes=1" in output
    assert "#SBATCH --segment=1" in output
    assert "args+=(--tensor-parallel-size 2)" in output
    assert "args+=(--temperature 1.0)" in output
    assert "args+=(--top-p 0.95)" in output
    assert "args+=(--samples-per-prompt 4)" in output
    assert "args+=(--rollout-batches 2)" in output
    assert "args+=(--mode mtp_static)" in output
    assert "args+=(--mode mtp_dynamic)" in output
    assert "--gres" not in output


def test_nemotron_sync_rl_wrapper_records_unsupported_matrix_rows(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "nemotron-sync"
    output = run_dry(
        "submit_nemotron_sync_rl_mtp_matrix.sh",
        CLUSTER="ptyche",
        MODELS="ultra",
        RUN_ID="sync-bf16-matrix",
        RESULT_ROOT=str(result_root),
    )
    manifest = extract_manifest_path(output)
    manifest_lines = extract_manifest_rows(output)

    assert output.count("[DRY-RUN] sync_variant=") == 3
    assert not result_root.exists()
    assert not manifest.exists()
    assert any(
        line.startswith(
            f"SUPPORTED\tultra\tsync_rl_math\tmtp_static\tmtp_static\t{result_root / 'ultra' / 'mtp_static'}\t"
        )
        for line in manifest_lines
    )
    assert any(
        line.startswith(
            "UNSUPPORTED\tultra\tsync_rl_math\teagle3\t-\t-\tnemotron_baseline_native_mtp_only\t"
        )
        for line in manifest_lines
    )
    assert any(
        line.startswith(
            "UNSUPPORTED\tultra\tsync_rl_math\tdflash\t-\t-\tqwen3_8b_public_asset_only\t"
        )
        for line in manifest_lines
    )
    assert any(
        line.startswith(
            f"SUPPORTED\tultra\tsync_rl_math\tbaseline\tbaseline\t{result_root / 'ultra' / 'baseline'}\t"
        )
        for line in manifest_lines
    )


def test_nemotron_sync_rl_wrapper_exits_on_unknown_model_without_reusing_state() -> None:
    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "submit_nemotron_sync_rl_mtp_matrix.sh")],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "true",
            "CLUSTER": "ptyche",
            "REQUIRE_GIT_PULL": "false",
            "MODELS": "ultra bogus",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "bogus" in completed.stderr
    assert extract_sync_variants(completed.stdout) == [
        "baseline",
        "mtp_static",
        "mtp_dynamic",
    ]
    assert "/bogus/" not in completed.stdout


def test_nemotron_sync_rl_wrapper_uses_matrix_defaults_and_allows_env_overrides(
    tmp_path: Path,
) -> None:
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()
    matrix = load_model_method_matrix()
    ultra = get_matrix_model(matrix, "ultra")
    ultra_profile = get_matrix_profile(ultra, "sync_rl_math")
    ultra_profile["smoke"]["num_prompts"] = 7
    ultra_profile["smoke"]["samples_per_prompt"] = 5
    ultra_profile["smoke"]["rollout_batches"] = 6
    ultra_profile["smoke"]["max_prompt_tokens"] = 1111
    ultra_profile["smoke"]["max_new_tokens"] = 222
    ultra_profile["smoke"]["engine_max_num_seqs"] = 17
    ultra_profile["smoke"]["time_limit"] = "03:03:03"
    ultra_profile["full"]["num_prompts"] = 23
    ultra_profile["full"]["samples_per_prompt"] = 19
    ultra_profile["full"]["rollout_batches"] = 4
    ultra_profile["full"]["max_prompt_tokens"] = 3333
    ultra_profile["full"]["max_new_tokens"] = 4444
    ultra_profile["full"]["engine_max_num_seqs"] = 71
    ultra_profile["full"]["time_limit"] = "09:09:09"
    (wrapper_dir / "model_method_matrix.json").write_text(
        json.dumps(matrix), encoding="utf-8"
    )
    for name in (
        "submit_nemotron_sync_rl_mtp_matrix.sh",
        "submit_sync_rollout.sh",
    ):
        (wrapper_dir / name).write_text(
            (EXPERIMENT / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
        (wrapper_dir / name).chmod(0o755)

    smoke = subprocess.run(
        ["bash", str(wrapper_dir / "submit_nemotron_sync_rl_mtp_matrix.sh")],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "true",
            "CLUSTER": "ptyche",
            "REQUIRE_GIT_PULL": "false",
            "MODELS": "ultra",
            "RESULT_ROOT": str(tmp_path / "smoke-root"),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    full = subprocess.run(
        ["bash", str(wrapper_dir / "submit_nemotron_sync_rl_mtp_matrix.sh")],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "true",
            "CLUSTER": "ptyche",
            "REQUIRE_GIT_PULL": "false",
            "MODELS": "ultra",
            "SMOKE": "false",
            "PROMPT_JSONL": "/tmp/prompts.jsonl",
            "RESULT_ROOT": str(tmp_path / "full-root"),
            "NUM_PROMPTS": "99",
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert "args+=(--num-prompts 7)" in smoke.stdout
    assert "args+=(--samples-per-prompt 5)" in smoke.stdout
    assert "args+=(--rollout-batches 6)" in smoke.stdout
    assert "args+=(--max-prompt-tokens 1111)" in smoke.stdout
    assert "args+=(--max-new-tokens 222)" in smoke.stdout
    assert "args+=(--engine-max-num-seqs 17)" in smoke.stdout
    assert "#SBATCH --time=03:03:03" in smoke.stdout
    assert "args+=(--num-prompts 99)" in full.stdout
    assert "args+=(--samples-per-prompt 19)" in full.stdout
    assert "args+=(--rollout-batches 4)" in full.stdout
    assert "args+=(--max-prompt-tokens 3333)" in full.stdout
    assert "args+=(--max-new-tokens 4444)" in full.stdout
    assert "args+=(--engine-max-num-seqs 71)" in full.stdout
    assert "#SBATCH --time=09:09:09" in full.stdout


def test_nemorl_perfcfg_dry_run_preserves_per_engine_recipe_shapes() -> None:
    output = run_dry(
        "submit_nemorl_perfcfg_sync_matrix.sh",
        CLUSTER="lyris",
        SMOKE="false",
        MODELS="qwen30ba3b qwen32 qwen235b",
    )

    assert "grpo-qwen3-30ba3b-4n4g.yaml" in output
    assert "per_engine_prompts=4" in output
    assert "target_tp=1" in output
    assert "grpo-qwen3-32b-4n4g.yaml" in output
    assert "per_engine_prompts=8" in output
    assert "target_tp=2" in output
    assert "grpo-qwen3-235b-32n4g.yaml" in output
    assert "per_engine_prompts=1" in output
    assert "target_tp=8" in output
    assert "#SBATCH --nodes=2" in output
    assert "#SBATCH --segment=2" in output
    assert "args+=(--distributed-executor-backend ray)" in output
    assert "args+=(--max-model-len 8192)" in output
    assert "args+=(--max-new-tokens 8192)" in output
    assert "args+=(--samples-per-prompt 32)" in output
    assert "args+=(--top-p 1.0)" in output
    assert "moe_backend=triton" in output
    assert "--max-num-batched-tokens" not in output
    assert "--gres" not in output


def test_legacy_0619_replay_dry_run_preserves_strict_contract() -> None:
    output = run_dry(
        "submit_legacy_0619_replay_matrix.sh",
        CLUSTER="lyris",
        SMOKE="false",
        MODELS="qwen32",
        DOMAINS="Math",
        BATCH_SIZES="1",
        RUN_ID="contract",
    )

    assert output.count("[DRY-RUN] variant=") == 6
    assert "math_500_data_prompts_qmath_20260617.jsonl" in output
    assert "--tensor-parallel-size '2'" in output
    assert "--throughput-gpu-count '4'" in output
    assert "--isl '4096'" in output
    assert "--osl '32768'" in output
    assert "--batch-sizes 1" in output
    assert "--max-model-len '40960'" in output
    assert "--max-num-batched-tokens '131072'" in output
    assert "--static-k '3'" in output
    assert "enforce_eager=false" in output
    assert "cudagraph_mode=PIECEWISE" in output
    assert "--cudagraph-mode 'PIECEWISE'" in output
    assert "--enforce-eager" not in output
    assert "--attention-backend 'TRITON_ATTN'" in output
    assert "--disable-custom-all-reduce" not in output
    assert "/contract_cg-on-piecewise/" in output
    assert ".0619-math-qwen32-b1-cg-on-piecewise-baseline-" in output
    assert "--tag 'matrix_cg-on-piecewise_baseline_" in output
    assert "--temperature 0.0" in output
    assert "--temperature 1.0" in output
    assert "[DRY-RUN] variant=suffix" not in output
    assert "[DRY-RUN] variant=pard" not in output
    assert "--gres" not in output


def test_legacy_0619_replay_allows_explicit_cuda_graph_off_override() -> None:
    output = run_dry(
        "submit_legacy_0619_replay_matrix.sh",
        CLUSTER="lyris",
        SMOKE="false",
        MODELS="qwen32",
        DOMAINS="Math",
        BATCH_SIZES="1",
        TEMPERATURES="0.0",
        VARIANTS="baseline",
        RUN_ID="legacy",
        ENFORCE_EAGER="true",
        CUDAGRAPH_MODE="NONE",
        DISABLE_CUSTOM_ALL_REDUCE="true",
    )

    assert "enforce_eager=true" in output
    assert "cudagraph_mode=NONE" in output
    assert "--cudagraph-mode 'NONE'" in output
    assert "--enforce-eager" in output
    assert "--disable-custom-all-reduce" in output
    assert "/legacy_cg-off-none/" in output
    assert ".0619-math-qwen32-b1-cg-off-none-baseline-" in output
    assert "--tag 'matrix_cg-off-none_baseline_" in output


def test_legacy_0619_replay_rejects_eager_with_non_none_cuda_graph_mode() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "DRY_RUN": "true",
        "CLUSTER": "lyris",
        "ENFORCE_EAGER": "true",
        "CUDAGRAPH_MODE": "PIECEWISE",
    }
    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "submit_legacy_0619_replay_matrix.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert (
        "ENFORCE_EAGER=true requires CUDAGRAPH_MODE=NONE"
        in completed.stderr
    )
    assert completed.stdout == ""


def write_sync_summary_result(
    matrix_root: Path,
    variant: str,
    *,
    planned: list[int] | None = None,
    actual: list[int] | None = None,
    forced: list[bool] | None = None,
    output_hashes: list[str] | None = None,
    request_plan_hash: str = "plan-sha",
    total_output_tokens: int = 10000,
) -> None:
    strict_config = {
        "runtime_image_sha256": "image-sha",
        "model_config_hash": "model-sha",
        "prompt_set_hash": "prompt-sha",
        "request_plan_hash": request_plan_hash,
        "cudagraph_mode": "PIECEWISE",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    result_dir = matrix_root / variant
    result_dir.mkdir()
    rollout_batches = []
    if planned is not None and actual is not None and forced is not None:
        rollout_batches.append(
            {
                "planned_output_tokens": planned,
                "actual_output_tokens": actual,
                "forced_output_mask": forced,
                "output_token_hashes": output_hashes or [],
            }
        )
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "config": {"mode": variant, **strict_config},
                "rollout_batches": rollout_batches,
                "summary": {
                    "total_rollout_time_s": 100.0,
                    "output_tok_s_per_gpu": 100.0,
                    "total_output_tokens": total_output_tokens,
                    "spec_decode_metrics": {},
                },
            }
        ),
        encoding="utf-8",
    )


def test_sync_rollout_summary_reports_baseline_and_static_relative_speedups(
    tmp_path: Path,
) -> None:
    summary_module = load_sync_summary_module()
    variants = {
        "baseline": (100.0, 100.0),
        "static": (80.0, 125.0),
        "dynamic": (70.0, 150.0),
    }
    for variant, (rollout_time, throughput) in variants.items():
        result_dir = tmp_path / variant
        result_dir.mkdir()
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "config": {
                        "mode": variant,
                        "temperature": 1.0,
                        "top_p": 0.9,
                    },
                    "summary": {
                        "total_rollout_time_s": rollout_time,
                        "output_tok_s_per_gpu": throughput,
                        "total_output_tokens": 10000,
                        "spec_decode_metrics": {
                            "acceptance_rate": 0.5,
                            "mean_acceptance_length": 3.0,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    rows = summary_module.build_summary(tmp_path)
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["baseline"]["throughput_speedup_vs_baseline"] == 1.0
    assert by_variant["dynamic"]["throughput_speedup_vs_baseline"] == 1.5
    assert by_variant["dynamic"]["rollout_time_reduction_vs_baseline_pct"] == 30.0
    assert by_variant["dynamic"]["throughput_speedup_vs_static"] == 1.2
    assert by_variant["dynamic"]["rollout_time_reduction_vs_static_pct"] == 12.5


def test_sync_rollout_summary_supports_native_mtp_variants(tmp_path: Path) -> None:
    summary_module = load_sync_summary_module()
    variants = {
        "baseline": (100.0, 100.0),
        "mtp_static": (75.0, 140.0),
        "mtp_dynamic": (65.0, 160.0),
    }
    for variant, (rollout_time, throughput) in variants.items():
        result_dir = tmp_path / variant
        result_dir.mkdir()
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "config": {
                        "mode": variant,
                        "temperature": 1.0,
                        "top_p": 0.95,
                    },
                    "summary": {
                        "total_rollout_time_s": rollout_time,
                        "output_tok_s_per_gpu": throughput,
                        "total_output_tokens": 10000,
                        "spec_decode_metrics": {},
                    },
                }
            ),
            encoding="utf-8",
        )

    rows = summary_module.build_summary(tmp_path)
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["mtp_dynamic"]["throughput_speedup_vs_baseline"] == 1.6
    assert (
        by_variant["mtp_dynamic"]["rollout_time_reduction_vs_baseline_pct"]
        == 35.0
    )
    assert by_variant["mtp_dynamic"]["throughput_speedup_vs_static"] == 1.142857


def test_sync_rollout_summary_rejects_mismatched_request_plan_hash(
    tmp_path: Path,
) -> None:
    summary_module = load_sync_summary_module()
    strict_config = {
        "runtime_image_sha256": "image-sha",
        "model_config_hash": "model-sha",
        "prompt_set_hash": "prompt-sha",
        "request_plan_hash": "plan-sha",
        "cudagraph_mode": "PIECEWISE",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    for variant in ("baseline", "static", "dynamic"):
        result_dir = tmp_path / variant
        result_dir.mkdir()
        config = {"mode": variant, **strict_config}
        if variant == "dynamic":
            config["request_plan_hash"] = "other-plan"
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "config": config,
                    "summary": {
                        "total_rollout_time_s": 100.0,
                        "output_tok_s_per_gpu": 100.0,
                        "total_output_tokens": 10000,
                        "spec_decode_metrics": {},
                    },
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="request_plan_hash"):
        summary_module.build_summary(tmp_path)


def test_sync_rollout_summary_allows_different_hashes_with_equal_exact_work(
    tmp_path: Path,
) -> None:
    summary_module = load_sync_summary_module()
    strict_config = {
        "runtime_image_sha256": "image-sha",
        "model_config_hash": "model-sha",
        "prompt_set_hash": "prompt-sha",
        "request_plan_hash": "plan-sha",
        "cudagraph_mode": "PIECEWISE",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    output_hashes = {
        "baseline": ["a", "b"],
        "static": ["c", "d"],
        "dynamic": ["e", "f"],
    }
    for variant in ("baseline", "static", "dynamic"):
        result_dir = tmp_path / variant
        result_dir.mkdir()
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "config": {"mode": variant, **strict_config},
                    "rollout_batches": [
                        {
                            "planned_output_tokens": [4, 4],
                            "actual_output_tokens": [4, 4],
                            "output_token_hashes": output_hashes[variant],
                        }
                    ],
                    "summary": {
                        "total_rollout_time_s": 100.0,
                        "output_tok_s_per_gpu": 100.0,
                        "total_output_tokens": 10000,
                        "spec_decode_metrics": {},
                    },
                }
            ),
            encoding="utf-8",
        )

    rows = summary_module.build_summary(tmp_path)
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["dynamic"]["exact_output_work_match_vs_baseline"] is True
    assert by_variant["dynamic"]["exact_output_hash_match_vs_baseline"] is False


def test_sync_rollout_summary_allows_unforced_underfill_with_matching_forced_work(
    tmp_path: Path,
) -> None:
    summary_module = load_sync_summary_module()
    forced_mask = [True, False, True]
    for variant, hashes, unforced_actual in (
        ("baseline", ["a", "b", "c"], 1),
        ("static", ["d", "e", "f"], 2),
        ("dynamic", ["g", "h", "i"], 3),
    ):
        write_sync_summary_result(
            tmp_path,
            variant,
            planned=[4, 4, 8],
            actual=[4, unforced_actual, 8],
            forced=forced_mask,
            output_hashes=hashes,
            total_output_tokens=12 + unforced_actual,
        )

    rows = summary_module.build_summary(tmp_path)
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["dynamic"]["exact_output_work_match_vs_baseline"] is True
    assert by_variant["dynamic"]["exact_output_hash_match_vs_baseline"] is False


def test_sync_rollout_summary_rejects_forced_planned_work_mismatch(
    tmp_path: Path,
) -> None:
    summary_module = load_sync_summary_module()
    write_sync_summary_result(
        tmp_path,
        "baseline",
        planned=[4, 4, 8],
        actual=[4, 1, 8],
        forced=[True, False, True],
        output_hashes=["same-a", "same-b", "same-c"],
    )
    write_sync_summary_result(
        tmp_path,
        "static",
        planned=[4, 4, 8],
        actual=[4, 2, 8],
        forced=[True, False, True],
        output_hashes=["same-a", "same-b", "same-c"],
    )
    write_sync_summary_result(
        tmp_path,
        "dynamic",
        planned=[4, 4, 7],
        actual=[4, 1, 7],
        forced=[True, False, True],
        output_hashes=["same-a", "same-b", "same-c"],
    )

    with pytest.raises(ValueError, match="exact forced output work mismatch"):
        summary_module.build_summary(tmp_path)


def test_sync_rollout_summary_rejects_identical_hashes_with_underlength_work(
    tmp_path: Path,
) -> None:
    summary_module = load_sync_summary_module()
    strict_config = {
        "runtime_image_sha256": "image-sha",
        "model_config_hash": "model-sha",
        "prompt_set_hash": "prompt-sha",
        "request_plan_hash": "plan-sha",
        "cudagraph_mode": "PIECEWISE",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    for variant in ("baseline", "static", "dynamic"):
        result_dir = tmp_path / variant
        result_dir.mkdir()
        actual_tokens = [4, 4]
        if variant == "dynamic":
            actual_tokens = [4, 3]
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "config": {"mode": variant, **strict_config},
                    "rollout_batches": [
                        {
                            "planned_output_tokens": [4, 4],
                            "actual_output_tokens": actual_tokens,
                            "forced_output_mask": [True, True],
                            "output_token_hashes": ["same-a", "same-b"],
                        }
                    ],
                    "summary": {
                        "total_rollout_time_s": 100.0,
                        "output_tok_s_per_gpu": 100.0,
                        "total_output_tokens": sum(actual_tokens),
                        "spec_decode_metrics": {},
                    },
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="actual output length"):
        summary_module.build_summary(tmp_path)


def test_speedbench_dataset_batches_balance_entropy_classes() -> None:
    adapter = load_speedbench_dataset_module()
    records = adapter.build_records(
        fake_speedbench_rows(),
        dataset_config="throughput_1k",
    )

    batches = adapter.select_sync_overlay_rows(records, seed=1234)

    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [16, 16, 16]
    assert [adapter.count_categories(batch) for batch in batches] == [
        {"low_entropy": 6, "mixed": 5, "high_entropy": 5},
        {"low_entropy": 5, "mixed": 6, "high_entropy": 5},
        {"low_entropy": 5, "mixed": 5, "high_entropy": 6},
    ]
    assert len(
        {
            record.question_id
            for batch in batches
            for record in batch
        }
    ) == 48
    assert {
        record.category
        for batch in batches
        for record in batch
    } == {"low_entropy", "mixed", "high_entropy"}


def test_speedbench_dataset_selection_is_deterministic_for_a_seed() -> None:
    adapter = load_speedbench_dataset_module()
    records = adapter.build_records(
        fake_speedbench_rows(),
        dataset_config="throughput_1k",
        actual_tokenizer_isl=1117,
    )

    selected_once = adapter.select_sync_overlay_rows(records, seed=99)
    selected_twice = adapter.select_sync_overlay_rows(records, seed=99)
    selected_other = adapter.select_sync_overlay_rows(records, seed=100)

    def digest(
        batches: tuple[tuple[Any, ...], ...],
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(record.canonical_hash for record in batch)
            for batch in batches
        )

    assert digest(selected_once) == digest(selected_twice)
    assert digest(selected_once) != digest(selected_other)
    assert {
        record.actual_tokenizer_isl
        for batch in selected_once
        for record in batch
    } == {1117}


def test_speedbench_dataset_preserves_multi_turns_and_rejects_masked_rows() -> None:
    adapter = load_speedbench_dataset_module()

    records = adapter.build_records(
        [
            fake_speedbench_row(
                "mixed",
                0,
                turns=(
                    "user: first turn",
                    "assistant: second turn",
                    "user: final turn",
                ),
            )
        ],
        dataset_config="qualitative",
    )

    assert records[0].turns == (
        "user: first turn",
        "assistant: second turn",
        "user: final turn",
    )
    assert records[0].multiturn is True
    assert records[0].nominal_isl is None

    with pytest.raises(ValueError, match="masked row"):
        adapter.build_records(
            [fake_speedbench_row("mixed", 1, masked=True)],
            dataset_config="throughput_1k",
        )


def test_speedbench_dataset_manifest_pins_revisions_and_checksums(
    tmp_path: Path,
) -> None:
    adapter = load_speedbench_dataset_module()
    prepared_root = tmp_path / "prepared"
    speed_root = prepared_root / "speed"
    source_root = tmp_path / "sources"
    expected_configs = {
        "qualitative",
        "throughput_1k",
        "throughput_2k",
        "throughput_8k",
        "throughput_16k",
        "throughput_32k",
    }
    for config_name in expected_configs:
        parquet = speed_root / config_name / "test.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(f"{config_name}-payload".encode("utf-8"))
    dataset_license_root = source_root / "speedbench"
    dataset_license_root.mkdir(parents=True, exist_ok=True)
    (dataset_license_root / "License.pdf").write_text("dataset license\n", encoding="utf-8")
    (dataset_license_root / "README.md").write_text("dataset readme\n", encoding="utf-8")
    modelopt_license = source_root / "modelopt-LICENSE"
    modelopt_license.write_text("apache license\n", encoding="utf-8")

    manifest = adapter.build_prepared_manifest(
        speed_root,
        dataset_license_root=dataset_license_root,
        modelopt_license_path=modelopt_license,
    )
    prepared_entries = {
        entry["config_name"]: entry
        for entry in manifest["prepared_configs"]
    }

    assert manifest["dataset"]["id"] == "nvidia/SPEED-Bench"
    assert manifest["dataset"]["revision"] == "487aa718444e816458d1a0a52bfce7a454285cf4"
    assert manifest["model_optimizer"]["revision"] == (
        "43fee0cd70fa9e5f85782d52a4bd8ad9c8b88446"
    )
    assert manifest["dataset"]["license_files"] == [
        {
            "relative_path": "License.pdf",
            "sha256": adapter.sha256_file(dataset_license_root / "License.pdf"),
        },
        {
            "relative_path": "README.md",
            "sha256": adapter.sha256_file(dataset_license_root / "README.md"),
        },
    ]
    assert manifest["model_optimizer"]["license_files"] == [
        {
            "relative_path": "modelopt-LICENSE",
            "sha256": adapter.sha256_file(modelopt_license),
        }
    ]
    assert manifest["parquet_files"] == [
        {
            "relative_path": "qualitative/test.parquet",
            "sha256": adapter.sha256_file(speed_root / "qualitative" / "test.parquet"),
        },
        {
            "relative_path": "throughput_16k/test.parquet",
            "sha256": adapter.sha256_file(speed_root / "throughput_16k" / "test.parquet"),
        },
        {
            "relative_path": "throughput_1k/test.parquet",
            "sha256": adapter.sha256_file(speed_root / "throughput_1k" / "test.parquet"),
        },
        {
            "relative_path": "throughput_2k/test.parquet",
            "sha256": adapter.sha256_file(speed_root / "throughput_2k" / "test.parquet"),
        },
        {
            "relative_path": "throughput_32k/test.parquet",
            "sha256": adapter.sha256_file(speed_root / "throughput_32k" / "test.parquet"),
        },
        {
            "relative_path": "throughput_8k/test.parquet",
            "sha256": adapter.sha256_file(speed_root / "throughput_8k" / "test.parquet"),
        },
    ]
    assert set(prepared_entries) == expected_configs
    assert prepared_entries["qualitative"]["nominal_isl"] is None
    assert prepared_entries["throughput_32k"]["nominal_isl"] == 32768
    assert prepared_entries["throughput_1k"]["actual_tokenizer_isl"] is None
    assert prepared_entries["throughput_1k"]["relative_path"] == (
        "throughput_1k/test.parquet"
    )
    assert prepared_entries["throughput_1k"]["sha256"] == adapter.sha256_file(
        speed_root / "throughput_1k" / "test.parquet"
    )
    assert all(
        not str(entry["relative_path"]).startswith("/")
        for entry in prepared_entries.values()
    )
    checksums_path = tmp_path / "checksums.sha256"
    checksum_lines = adapter.write_checksum_file(speed_root, checksums_path)
    assert checksum_lines == (
        f"{adapter.sha256_file(speed_root / 'qualitative' / 'test.parquet')}  qualitative/test.parquet",
        f"{adapter.sha256_file(speed_root / 'throughput_16k' / 'test.parquet')}  throughput_16k/test.parquet",
        f"{adapter.sha256_file(speed_root / 'throughput_1k' / 'test.parquet')}  throughput_1k/test.parquet",
        f"{adapter.sha256_file(speed_root / 'throughput_2k' / 'test.parquet')}  throughput_2k/test.parquet",
        f"{adapter.sha256_file(speed_root / 'throughput_32k' / 'test.parquet')}  throughput_32k/test.parquet",
        f"{adapter.sha256_file(speed_root / 'throughput_8k' / 'test.parquet')}  throughput_8k/test.parquet",
    )
    assert all(not line.split("  ", 1)[1].startswith("/") for line in checksum_lines)
    assert checksums_path.read_text(encoding="utf-8").splitlines() == list(checksum_lines)


def test_speedbench_dataset_manifest_rejects_missing_or_unexpected_parquet_sets(
    tmp_path: Path,
) -> None:
    adapter = load_speedbench_dataset_module()
    speed_root = tmp_path / "prepared" / "speed"
    dataset_license_root = tmp_path / "sources" / "speedbench"
    dataset_license_root.mkdir(parents=True, exist_ok=True)
    (dataset_license_root / "License.pdf").write_text("dataset license\n", encoding="utf-8")
    (dataset_license_root / "README.md").write_text("dataset readme\n", encoding="utf-8")
    modelopt_license = tmp_path / "sources" / "modelopt-LICENSE"
    modelopt_license.parent.mkdir(parents=True, exist_ok=True)
    modelopt_license.write_text("apache license\n", encoding="utf-8")

    for config_name in (
        "qualitative",
        "throughput_1k",
        "throughput_2k",
        "throughput_8k",
        "throughput_16k",
    ):
        parquet = speed_root / config_name / "test.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_text(config_name, encoding="utf-8")

    with pytest.raises(ValueError, match="missing expected parquet"):
        adapter.build_prepared_manifest(
            speed_root,
            dataset_license_root=dataset_license_root,
            modelopt_license_path=modelopt_license,
        )

    extra = speed_root / "throughput_32k" / "test.parquet"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("throughput_32k", encoding="utf-8")
    stray = speed_root / "unexpected" / "test.parquet"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text("stray", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected parquet"):
        adapter.build_prepared_manifest(
            speed_root,
            dataset_license_root=dataset_license_root,
            modelopt_license_path=modelopt_license,
        )


def test_speedbench_dataset_stage_dry_run_pins_revisions_and_respects_licenses() -> None:
    output = run_dry(
        "stage_speedbench.sh",
        CLUSTER="lyris",
        REQUIRE_GIT_PULL="false",
    )

    assert "nvidia/SPEED-Bench" in output
    assert "487aa718444e816458d1a0a52bfce7a454285cf4" in output
    assert "NVIDIA/Model-Optimizer" in output
    assert "43fee0cd70fa9e5f85782d52a4bd8ad9c8b88446" in output
    assert "examples/specdec_bench/prepare_data.py" in output
    assert "License.pdf" in output
    assert "LICENSE" in output
    assert "prepared_manifest.json" in output
    assert '--prepared-root "$SPEED_PREPARED_ROOT"' in output
    assert "sha256sum" in output
    assert "--segment=1" in output
    assert "--gres" not in output


def test_speedbench_stage_dry_run_skips_git_pull_and_leaves_roots_unchanged(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "stage-root"
    stage_root.mkdir()
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    git_log = tmp_path / "git.log"
    git_stub = stub_bin / "git"
    git_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'git-called\\n' >>\"${GIT_LOG:?}\"\n"
        "exit 99\n",
        encoding="utf-8",
    )
    git_stub.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "stage_speedbench.sh")],
        cwd=ROOT,
        env={
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "GIT_LOG": str(git_log),
            "CLUSTER": "lyris",
            "DRY_RUN": "true",
            "DATASET_ROOT": str(stage_root),
            "RUN_ID": "dry-run-review",
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[DRY-RUN]" in completed.stdout
    assert "manifest=" in completed.stdout
    assert not git_log.exists()
    assert not (stage_root / "dry-run-review").exists()


def test_speedbench_stage_test_only_uses_temp_render_files_and_cleans_up(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "stage-root"
    container = tmp_path / "container.sqsh"
    container.write_text("image", encoding="utf-8")
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    git_log = tmp_path / "git.log"

    (stub_bin / "git").write_text(
        "#!/usr/bin/env bash\n"
        "printf 'git-called\\n' >>\"${GIT_LOG:?}\"\n"
        "exit 99\n",
        encoding="utf-8",
    )
    (stub_bin / "git").chmod(0o755)
    (stub_bin / "sbatch").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "saw_test_only=false\n"
        "script_path=\n"
        "for arg in \"$@\"; do\n"
        "  printf 'arg=%s\\n' \"$arg\" >>\"${SBATCH_LOG:?}\"\n"
        "  if [[ \"$arg\" == '--test-only' ]]; then saw_test_only=true; fi\n"
        "  script_path=\"$arg\"\n"
        "done\n"
        "[[ \"$saw_test_only\" == true ]]\n"
        "test -f \"$script_path\"\n"
        "grep -q -- '--prepared-root' \"$script_path\"\n"
        "grep -q -- '/prepared/speed' \"$script_path\"\n"
        "printf 'script=%s\\n' \"$script_path\" >>\"${SBATCH_LOG}\"\n",
        encoding="utf-8",
    )
    (stub_bin / "sbatch").chmod(0o755)

    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "stage_speedbench.sh")],
        cwd=ROOT,
        env={
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "CLUSTER": "lyris",
            "TEST_ONLY": "true",
            "DATASET_ROOT": str(stage_root),
            "RUN_ID": "test-only-review",
            "CONTAINER_IMAGE": str(container),
            "SBATCH_LOG": str(sbatch_log),
            "GIT_LOG": str(git_log),
        },
        check=True,
        capture_output=True,
        text=True,
    )

    log_lines = sbatch_log.read_text(encoding="utf-8").splitlines()
    script_paths = [
        Path(line.split("=", 1)[1])
        for line in log_lines
        if line.startswith("script=")
    ]

    assert "[TEST-ONLY]" in completed.stdout
    assert not git_log.exists()
    assert "arg=--test-only" in log_lines
    assert len(script_paths) == 1
    assert not script_paths[0].exists()
    assert not script_paths[0].parent.exists()
    assert not (stage_root / "test-only-review").exists()


def test_speedbench_stage_test_only_keeps_hostile_newlines_out_of_sbatch_directives(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "newline-marker"
    hostile_root = (
        f"{tmp_path}/safe-root\n#SBATCH --comment=owned\n$(touch {marker})"
    )
    container = tmp_path / "container.sqsh"
    container.write_text("image", encoding="utf-8")
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    captured_script = tmp_path / "captured.sbatch"
    captured_output_arg = tmp_path / "captured-output.txt"

    (stub_bin / "sbatch").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "script_path=${!#}\n"
        "cp \"$script_path\" \"${CAPTURED_SCRIPT:?}\"\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$arg\" == --output=* ]]; then\n"
        "    printf '%s' \"$arg\" >\"${CAPTURED_OUTPUT_ARG:?}\"\n"
        "  fi\n"
        "done\n",
        encoding="utf-8",
    )
    (stub_bin / "sbatch").chmod(0o755)

    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "stage_speedbench.sh")],
        cwd=ROOT,
        env={
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "CLUSTER": "lyris",
            "TEST_ONLY": "true",
            "DATASET_ROOT": hostile_root,
            "RUN_ID": "newline-review",
            "CONTAINER_IMAGE": str(container),
            "CAPTURED_SCRIPT": str(captured_script),
            "CAPTURED_OUTPUT_ARG": str(captured_output_arg),
        },
        check=True,
        capture_output=True,
        text=True,
    )

    script_text = captured_script.read_text(encoding="utf-8")
    directive_lines = [
        line for line in script_text.splitlines() if line.startswith("#SBATCH")
    ]

    assert "[TEST-ONLY]" in completed.stdout
    assert "#SBATCH --output=" not in script_text
    assert "#SBATCH --comment=owned" not in directive_lines
    assert captured_output_arg.read_text(encoding="utf-8").startswith("--output=")
    assert "\n#SBATCH --comment=owned\n" in captured_output_arg.read_text(
        encoding="utf-8"
    )
    assert not marker.exists()


def test_speedbench_stage_rejects_invalid_scheduler_identifiers(
    tmp_path: Path,
) -> None:
    pwned = tmp_path / "pwned"
    completed = subprocess.run(
        ["bash", str(EXPERIMENT / "stage_speedbench.sh")],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "CLUSTER": "lyris",
            "DRY_RUN": "true",
            "ACCOUNT": f"coreai$(touch {pwned})",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "invalid scheduler identifier" in completed.stderr
    assert not pwned.exists()


def test_speedbench_stage_generated_runner_executes_with_hostile_paths(
    tmp_path: Path,
) -> None:
    pwned = tmp_path / "pwned"
    hostile_root = f"{tmp_path}/odd ' \" $DOLLAR $(touch {pwned}) path"
    hostile_hf = f"{hostile_root}/hf home"
    container = tmp_path / "container.sqsh"
    container.write_text("image", encoding="utf-8")
    output = run_dry(
        "stage_speedbench.sh",
        CLUSTER="lyris",
        REQUIRE_GIT_PULL="false",
        DATASET_ROOT=hostile_root,
        HF_HOME=hostile_hf,
        RUN_ID="hostile-review",
        CONTAINER_IMAGE=str(container),
    )
    script = output[output.index("#!/usr/bin/env bash\n") :]
    run_script = tmp_path / "submit.sbatch"
    run_script.write_text(script, encoding="utf-8")
    run_script.chmod(0o755)

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    argv_path = tmp_path / "srun-argv.txt"
    (stub_bin / "srun").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ": >\"${SRUN_ARGV_OUT:?}\"\n"
        "for arg in \"$@\"; do printf '%s\\n' \"$arg\" >>\"${SRUN_ARGV_OUT}\"; done\n"
        "exit 0\n",
        encoding="utf-8",
    )
    (stub_bin / "srun").chmod(0o755)

    subprocess.run(
        [str(run_script)],
        cwd=tmp_path,
        env={
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "SRUN_ARGV_OUT": str(argv_path),
            "DOLLAR": "expanded-if-unsafe",
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert "$DOLLAR" in argv_path.read_text(encoding="utf-8")
    assert "$(touch " in argv_path.read_text(encoding="utf-8")
    assert not pwned.exists()


def test_speedbench_stage_uses_mktemp_and_no_fixed_tmp_cleanup() -> None:
    text = (EXPERIMENT / "stage_speedbench.sh").read_text(encoding="utf-8")

    assert "mktemp -d" in text
    assert "rm -rf /tmp/" not in text


def test_scripts_do_not_depend_on_home_storage() -> None:
    for script_name in (
        "stage_image.sh",
        "stage_math_datasets.sh",
        "stage_speedbench.sh",
        "stage_extended_method_assets.sh",
        "stage_ray_site.sh",
        "submit_matrix.sh",
        "submit_nsys.sh",
        "submit_nemorl_perfcfg_sync_matrix.sh",
        "submit_legacy_0619_replay_matrix.sh",
        "submit_qwen8_extended_methods_matrix.sh",
        "submit_qwen8_long_context_matrix.sh",
        "submit_angelslim_matrix.sh",
        "submit_angelslim_long_context_dflare.sh",
        "submit_sync_rollout.sh",
        "submit_swe_sync_rollout_matrix.sh",
    ):
        text = (EXPERIMENT / script_name).read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "/lustre/fsw/coreai_dlalgo_llm/users/sna" in text
