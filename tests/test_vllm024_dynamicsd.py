from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
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

    assert "${angelslim_source}/tools/angelslim_dflare_transport.py" in worker
    assert "angelslim_compact_result_transport.patch" in worker
    assert "compact_response_map' \"${angelslim_source}/tools/dflash_benchmark.py\"" in worker
    assert "write_rank_partial' \"${angelslim_source}/tools/dflash_benchmark.py\"" in worker
    assert "from angelslim_dflare_transport import" in patch
    assert "compact_response_map" in patch
    assert "write_rank_partial" in patch
    assert "responses.append(response)" in patch
    assert "responses.append(compact_response_map(response))" in patch
    assert "write_rank_partial(args.output_json, _dist_rank(), responses)" in patch
    assert patch.index("write_rank_partial(args.output_json, _dist_rank(), responses)") < (
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
    assert "--distributed-executor-backend 'ray'" in output
    assert "--max-model-len '8192'" in output
    assert "--max-new-tokens 8192" in output
    assert "--samples-per-prompt 32" in output
    assert "--top-p 1.0" in output
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
    assert "--enforce-eager" in output
    assert "--attention-backend 'TRITON_ATTN'" in output
    assert "--disable-custom-all-reduce" in output
    assert "--temperature 0.0" in output
    assert "--temperature 1.0" in output
    assert "[DRY-RUN] variant=suffix" not in output
    assert "[DRY-RUN] variant=pard" not in output
    assert "--gres" not in output


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


def test_scripts_do_not_depend_on_home_storage() -> None:
    for script_name in (
        "stage_image.sh",
        "stage_math_datasets.sh",
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
    ):
        text = (EXPERIMENT / script_name).read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "/lustre/fsw/coreai_dlalgo_llm/users/sna" in text
