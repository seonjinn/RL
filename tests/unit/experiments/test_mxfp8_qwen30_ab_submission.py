from pathlib import Path

import yaml


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments/mxfp8_adaptive_rollout_v0251"


def test_qwen30_performance_config_defines_matched_cuda_graph_workload() -> None:
    config_path = EXPERIMENT / "configs/eval_qwen3_30ba3b_performance.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["eval"] == {
        "num_tests_per_prompt": 32,
        "save_path": "${oc.env:CANARY_OUTPUT_DIR}",
        "seed": 42,
    }
    generation = config["generation"]
    assert generation["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert generation["max_new_tokens"] == 4096
    assert generation["num_prompts_per_step"] == 64
    assert generation["temperature"] == 1.0
    assert generation["top_p"] == 1.0
    assert generation["top_k"] == -1

    vllm_cfg = generation["vllm_cfg"]
    assert vllm_cfg["precision"] == "fp8"
    assert vllm_cfg["is_mx"] is True
    assert vllm_cfg["enforce_eager"] is False
    assert vllm_cfg["tensor_parallel_size"] == 1
    assert vllm_cfg["pipeline_parallel_size"] == 1
    assert vllm_cfg["expert_parallel_size"] == 1
    assert vllm_cfg["quantization_ignored_layer_kws"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def test_qwen30_performance_config_uses_eight_engines_and_adaptive_contract() -> None:
    config_path = EXPERIMENT / "configs/eval_qwen3_30ba3b_performance.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generation = config["generation"]

    assert generation["vllm_kwargs"] == {
        "attention_backend": "FLASH_ATTN",
        "enable_chunked_prefill": True,
        "linear_backend": "${oc.env:NEMORL_MXFP8_LINEAR_BACKEND}",
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 256,
    }
    resources = generation["colocated"]["resources"]
    assert resources == {"gpus_per_node": 4, "num_nodes": 2}
    assert resources["gpus_per_node"] * resources["num_nodes"] == 8
    assert config["cluster"] == {"gpus_per_node": 4, "num_nodes": 2}

    env_vars = generation["vllm_cfg"]["env_vars"]
    for name in (
        "VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK",
        "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE",
        "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256",
        "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64",
        "VLLM_MXFP8_DENSE_TRTLLM_LAYOUT",
        "VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M",
    ):
        assert name in env_vars

    dataset = config["data"]["dataset_name"]
    assert dataset.endswith("/data/qwen_trace_math.jsonl")
    dataset_path = EXPERIMENT / "data/qwen_trace_math.jsonl"
    assert len(dataset_path.read_text(encoding="utf-8").splitlines()) == 64


def test_qwen30_cuda_graph_trace_matches_performance_workload() -> None:
    performance = yaml.safe_load(
        (EXPERIMENT / "configs/eval_qwen3_30ba3b_performance.yaml").read_text(
            encoding="utf-8"
        )
    )
    trace = yaml.safe_load(
        (EXPERIMENT / "configs/eval_qwen3_30ba3b_cuda_graph_trace.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert trace["generation"]["vllm_cfg"]["enforce_eager"] is False
    assert (
        trace["generation"]["vllm_kwargs"] == performance["generation"]["vllm_kwargs"]
    )
    assert trace["generation"]["colocated"] == performance["generation"]["colocated"]
    trace_env = trace["generation"]["vllm_cfg"]["env_vars"]
    for name in (
        "VLLM_MXFP8_DENSE_SHAPE_TRACE",
        "VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR",
        "VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX",
    ):
        assert name in trace_env


def test_qwen30_ab_submitter_binds_committed_artifacts_and_pulls_first() -> None:
    submitter_path = EXPERIMENT / "submit_qwen30_ab_ptyche.sh"
    submitter = submitter_path.read_text(encoding="utf-8")

    assert "run_ab.sh run" in submitter
    assert "eval_qwen3_30ba3b_performance.yaml" in submitter
    assert "data/qwen3_30ba3b_cg_output_shmoo_2501234_2501236_2501238" in submitter
    assert "TACTIC_ARTIFACT_DIR" in submitter
    assert 'TACTIC_FILE="$artifact_dir/exact_tactics.json"' in submitter
    assert (
        "88ea9238c8ce06d3b174b9cae928e4dbfc0d0a5ed4e9d2086c9d0f79ef4d3211" in submitter
    )
    assert 'base64 < "$LAYER_ALLOWLIST_FILE"' in submitter
    assert "sha256sum --check" in submitter

    assert 'git -C "$NEMO_RL_REPO_ROOT" diff --quiet' in submitter
    assert 'git -C "$NEMO_RL_REPO_ROOT" diff --cached --quiet' in submitter
    assert 'git -C "$NEMO_RL_REPO_ROOT" pull --ff-only' in submitter
    assert submitter.index('git -C "$NEMO_RL_REPO_ROOT" pull --ff-only') < (
        submitter.index("lock_sha=")
    )
    assert 'git -C "$CUSTOM_VLLM_SOURCE" rev-parse HEAD' in submitter
    assert 'git -C "$CUSTOM_VLLM_SOURCE" diff --quiet' in submitter
    assert 'git -C "$CUSTOM_VLLM_SOURCE" diff --cached --quiet' in submitter

    assert "--nodes=2" in submitter
    assert "--time=05:00:00" in submitter
    assert "--segment=2" in submitter
    assert "--dependency=" in submitter
    assert "args+=(--test-only)" in submitter
    assert "afterok" not in submitter
    assert "run_trace.sh" not in submitter


def test_qwen30_ab_submitter_validates_inputs_and_submits_without_dependency() -> None:
    submitter_path = EXPERIMENT / "submit_qwen30_ab_ptyche.sh"
    submitter = submitter_path.read_text(encoding="utf-8")

    assert "run_ab.sh run" in submitter
    assert "eval_qwen3_30ba3b_performance.yaml" in submitter
    assert "--nodes=2" in submitter
    assert "--time=05:00:00" in submitter
    assert "--segment=2" in submitter
    assert "--dependency=" in submitter
    assert "args+=(--test-only)" in submitter
    assert "afterok" not in submitter
