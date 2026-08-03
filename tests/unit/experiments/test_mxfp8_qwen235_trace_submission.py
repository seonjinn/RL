import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments/mxfp8_adaptive_rollout_v0251"


def test_qwen235_trace_config_defines_two_tp4_ep4_cuda_graph_engines() -> None:
    config_path = EXPERIMENT / "configs/eval_qwen3_235ba22b_32k_cuda_graph_trace.yaml"
    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["eval"] == {
        "num_tests_per_prompt": 1,
        "save_path": "${oc.env:CANARY_OUTPUT_DIR}",
        "seed": 42,
    }
    generation = config["generation"]
    assert generation["model_name"] == "Qwen/Qwen3-235B-A22B"
    assert generation["max_new_tokens"] == 32768
    assert generation["ignore_eos"] is True
    assert generation["stop_token_ids"] == []
    assert generation["num_prompts_per_step"] == 64

    vllm_cfg = generation["vllm_cfg"]
    assert vllm_cfg["precision"] == "fp8"
    assert vllm_cfg["is_mx"] is True
    assert vllm_cfg["tensor_parallel_size"] == 4
    assert vllm_cfg["pipeline_parallel_size"] == 1
    assert vllm_cfg["expert_parallel_size"] == 4
    assert vllm_cfg["enforce_eager"] is False
    assert "num_replicas" not in vllm_cfg
    assert vllm_cfg["quantization_ignored_layer_kws"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        ".mlp.gate",
    ]

    assert generation["vllm_kwargs"] == {
        "attention_backend": "FLASH_ATTN",
        "enable_chunked_prefill": True,
        "linear_backend": "${oc.env:NEMORL_MXFP8_LINEAR_BACKEND}",
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 32,
    }
    resources = generation["colocated"]["resources"]
    assert resources == {"gpus_per_node": 4, "num_nodes": 2}
    assert resources["gpus_per_node"] * resources["num_nodes"] == 8
    assert config["cluster"] == {"gpus_per_node": 4, "num_nodes": 2}

    env_vars = vllm_cfg["env_vars"]
    for name in (
        "VLLM_MXFP8_DENSE_SHAPE_TRACE",
        "VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR",
        "VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX",
    ):
        assert name in env_vars


def test_qwen235_trace_submitter_requires_clean_pinned_provenance() -> None:
    submitter_path = EXPERIMENT / "submit_qwen235_32k_trace_ptyche.sh"
    assert submitter_path.is_file()
    submitter = submitter_path.read_text(encoding="utf-8")

    assert "eval_qwen3_235ba22b_32k_cuda_graph_trace.yaml" in submitter
    assert "run_qwen235_trace_gate.sh" in submitter
    assert "nemorl-qwen235-mxfp8-32k-shape-trace" in submitter
    assert "coreai_dlalgo_llm-nemorl.qwen235-mxfp8-32k-trace" in submitter
    assert "HF_HOME=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf" in submitter
    assert 'HF_HUB_CACHE="$HF_HOME/hub"' in submitter
    assert "HF_DATASETS_CACHE=/home/sna/.cache/hf-datasets-canary" in submitter
    assert 'mkdir -p "$HF_DATASETS_CACHE"' in submitter
    assert "models--Qwen--Qwen3-235B-A22B" in submitter
    assert "CANARY_EXPECTED_REQUESTS=64" in submitter
    assert "CANARY_EXPECTED_TOKENS_PER_RESPONSE=32768" in submitter
    assert "NRL_VLLM_ASYNC_TIMEOUT_SECONDS=14400" in submitter

    expected_commit_default = (
        'EXPECTED_NEMO_RL_COMMIT=${EXPECTED_NEMO_RL_COMMIT:-$(git -C '
        '"$NEMO_RL_REPO_ROOT" rev-parse HEAD)}'
    )
    assert expected_commit_default in submitter
    assert "status --porcelain --untracked-files=all" in submitter
    assert 'require_clean_repo "$NEMO_RL_REPO_ROOT"' in submitter
    assert 'require_clean_repo "$CUSTOM_VLLM_SOURCE"' in submitter
    assert 'git -C "$NEMO_RL_REPO_ROOT" pull --ff-only' in submitter
    assert submitter.index(expected_commit_default) < submitter.index(
        'git -C "$NEMO_RL_REPO_ROOT" pull --ff-only'
    ) < submitter.index("lock_sha=")
    assert 'git -C "$CUSTOM_VLLM_SOURCE" rev-parse HEAD' in submitter
    assert 'actual_nemo_rl_commit=$(git -C "$NEMO_RL_REPO_ROOT" rev-parse HEAD)' in submitter
    assert '"$actual_nemo_rl_commit" != "$EXPECTED_NEMO_RL_COMMIT"' in submitter
    assert '"$CANARY_RESULT_ROOT/provenance.txt"' in submitter
    assert submitter.index("actual_nemo_rl_commit=") < submitter.index(
        '"$CANARY_RESULT_ROOT/provenance.txt"'
    ) < submitter.index("sbatch")

    assert "--nodes=2" in submitter
    assert "--time=05:00:00" in submitter
    assert "--segment=2" in submitter
    assert "--dependency=" in submitter
    assert "args+=(--test-only)" in submitter
    assert "afterok" not in submitter


def test_qwen235_performance_config_and_submitter_define_matched_three_arm_run() -> None:
    config_path = EXPERIMENT / "configs/eval_qwen3_235ba22b_performance.yaml"
    submitter_path = EXPERIMENT / "submit_qwen235_ab_ptyche.sh"
    assert config_path.is_file()
    assert submitter_path.is_file()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generation = config["generation"]
    assert generation["model_name"] == "Qwen/Qwen3-235B-A22B"
    assert generation["max_new_tokens"] == 4096
    assert generation["num_prompts_per_step"] == 64
    assert generation["vllm_cfg"]["tensor_parallel_size"] == 4
    assert generation["vllm_cfg"]["expert_parallel_size"] == 4
    assert generation["vllm_cfg"]["enforce_eager"] is False
    assert generation["vllm_kwargs"]["max_num_seqs"] == 32
    assert generation["vllm_kwargs"]["max_num_batched_tokens"] == 16384
    assert generation["vllm_kwargs"]["enable_chunked_prefill"] is True

    submitter = submitter_path.read_text(encoding="utf-8")
    assert "eval_qwen3_235ba22b_performance.yaml" in submitter
    assert "run_ab.sh run" in submitter
    assert "qwen235_tp4ep4_8x4_fix3_20260802" in submitter
    assert "2b8121d1b56ccb44a4ee9bdb10adc5e355f58bf21e79079eadeb2ac7494bf417" in submitter
    assert "models--Qwen--Qwen3-235B-A22B" in submitter
    assert "HF_DATASETS_CACHE=/home/sna/.cache/hf-datasets-canary" in submitter
    assert "--nodes=2" in submitter
    assert "--time=05:00:00" in submitter
    assert "--segment=2" in submitter
    assert "--dependency=" in submitter
    assert "args+=(--test-only)" in submitter
    assert "afterok" not in submitter


def test_qwen235_forced_32k_config_and_submitter_require_exact_outputs() -> None:
    config_path = EXPERIMENT / "configs/eval_qwen3_235ba22b_forced_32k_performance.yaml"
    submitter_path = EXPERIMENT / "submit_qwen235_forced_32k_ab_ptyche.sh"
    assert config_path.is_file()
    assert submitter_path.is_file()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["eval"]["num_tests_per_prompt"] == 1
    generation = config["generation"]
    assert generation["max_new_tokens"] == 32768
    assert generation["ignore_eos"] is True
    assert generation["stop_token_ids"] == []
    assert generation["num_prompts_per_step"] == 64
    assert generation["vllm_cfg"]["max_model_len"] == 36864
    assert generation["vllm_cfg"]["enforce_eager"] is False
    assert generation["vllm_kwargs"]["max_num_seqs"] == 32
    assert generation["vllm_kwargs"]["max_num_batched_tokens"] == 16384
    assert generation["vllm_kwargs"]["enable_chunked_prefill"] is True

    submitter = submitter_path.read_text(encoding="utf-8")
    assert "eval_qwen3_235ba22b_forced_32k_performance.yaml" in submitter
    assert "run_ab.sh pair" in submitter
    assert "CANARY_EXPECTED_REQUESTS=64" in submitter
    assert "CANARY_EXPECTED_TOKENS_PER_RESPONSE=32768" in submitter
    assert "NRL_VLLM_ASYNC_TIMEOUT_SECONDS=14400" in submitter
    assert "qwen235_tp4ep4_8x4_fix3_20260802" in submitter
    assert "--nodes=2" in submitter
    assert "--time=05:00:00" in submitter
    assert "--segment=2" in submitter
    assert "--dependency=" in submitter
    assert "args+=(--test-only)" in submitter
    assert "afterok" not in submitter


def test_qwen235_qkvo_trace_uses_separate_quantization_scope() -> None:
    config_path = (
        EXPERIMENT / "configs/eval_qwen3_235ba22b_qkvo_32k_eager_trace.yaml"
    )
    submitter_path = EXPERIMENT / "submit_qwen235_qkvo_32k_trace_ptyche.sh"
    assert config_path.is_file()
    assert submitter_path.is_file()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generation = config["generation"]
    assert generation["model_name"] == "Qwen/Qwen3-235B-A22B"
    assert generation["max_new_tokens"] == 32768
    assert generation["ignore_eos"] is True
    assert generation["stop_token_ids"] == []
    assert generation["vllm_cfg"]["quantization_ignored_layer_kws"] == [
        ".mlp.gate",
        "lm_head",
    ]
    assert generation["vllm_cfg"]["enforce_eager"] is True
    assert generation["vllm_kwargs"]["max_num_batched_tokens"] == 16384

    submitter = submitter_path.read_text(encoding="utf-8")
    assert "eval_qwen3_235ba22b_qkvo_32k_eager_trace.yaml" in submitter
    assert "run_qwen235_qkvo_trace_gate.sh" in submitter
    assert "nemorl-qwen235-mxfp8-qkvo-32k-shape-trace" in submitter
    assert "coreai_dlalgo_llm-nemorl.qwen235-mxfp8-qkvo-32k-trace" in submitter
    assert "CANARY_EXPECTED_REQUESTS=64" in submitter
    assert "CANARY_EXPECTED_TOKENS_PER_RESPONSE=32768" in submitter
    assert "CANARY_EXPECTED_TRACE_WORKERS=8" in submitter
    assert "--time=05:00:00" in submitter
    assert "--segment=2" in submitter
    assert "--dependency=" in submitter
    assert "args+=(--test-only)" in submitter


def test_qwen235_qkvo_performance_uses_cuda_graph_and_qualified_artifacts() -> None:
    config_path = (
        EXPERIMENT / "configs/eval_qwen3_235ba22b_qkvo_32k_performance.yaml"
    )
    submitter_path = EXPERIMENT / "submit_qwen235_qkvo_32k_ab_ptyche.sh"
    assert config_path.is_file()
    assert submitter_path.is_file()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generation = config["generation"]
    assert generation["model_name"] == "Qwen/Qwen3-235B-A22B"
    assert generation["max_new_tokens"] == 32768
    assert generation["ignore_eos"] is True
    assert generation["stop_token_ids"] == []
    assert generation["num_prompts_per_step"] == 64
    assert generation["vllm_cfg"]["quantization_ignored_layer_kws"] == [
        ".mlp.gate",
        "lm_head",
    ]
    assert generation["vllm_cfg"]["enforce_eager"] is False
    assert generation["vllm_kwargs"]["max_num_seqs"] == 32
    assert generation["vllm_kwargs"]["max_num_batched_tokens"] == 16384
    assert generation["vllm_kwargs"]["enable_chunked_prefill"] is True

    submitter = submitter_path.read_text(encoding="utf-8")
    assert "eval_qwen3_235ba22b_qkvo_32k_performance.yaml" in submitter
    assert "run_ab.sh pair" in submitter
    assert "qwen3_235ba22b_qkvo_shmoo_2508282_2508292" in submitter
    assert "CANARY_EXPECTED_REQUESTS=64" in submitter
    assert "CANARY_EXPECTED_TOKENS_PER_RESPONSE=32768" in submitter
    assert "--nodes=2" in submitter
    assert "--time=05:00:00" in submitter
    assert "--segment=2" in submitter
    assert "--dependency=" in submitter
    assert "args+=(--test-only)" in submitter


def _run_qkvo_trace_gate(
    tmp_path: Path,
    prefixes: list[str],
    *,
    k: int = 4096,
    n_logical: int = 1536,
    n_physical: int = 1536,
    trace_cap: int = 16384,
) -> subprocess.CompletedProcess[str]:
    fake_root = tmp_path / "repo"
    trace_script = fake_root / "experiments/mxfp8_adaptive_rollout_v0251/run_trace.sh"
    trace_script.parent.mkdir(parents=True)
    trace_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'mkdir -p "$CANARY_RESULT_ROOT/trace" "$SHAPE_TRACE_DIR"\n'
        'printf \'%s\\n\' "$TRACE_SUMMARY" > '
        '"$CANARY_RESULT_ROOT/trace/shape_summary.json"\n'
        'printf \'%s\\n\' "$TRACE_RECORDS" > "$SHAPE_TRACE_DIR/trace.jsonl"\n',
        encoding="utf-8",
    )
    trace_script.chmod(0o755)
    result_root = tmp_path / "result"
    trace_dir = result_root / "trace/raw"
    gate = EXPERIMENT / "run_qwen235_qkvo_trace_gate.sh"
    records = "\n".join(
        json.dumps(
            {
                "event": "mxfp8_dense_shape",
                "family": (
                    "QKV"
                    if ".qkv_proj" in prefixes[worker % len(prefixes)]
                    else "O"
                    if ".o_proj" in prefixes[worker % len(prefixes)]
                    else "OtherDense"
                ),
                "hostname": f"node-{worker % 2}",
                "k": k,
                "layout": "8x4",
                "m": 1,
                "n_logical": n_logical,
                "n_physical": n_physical,
                "pid": 100 + worker,
                "prefix": prefixes[worker % len(prefixes)],
            }
        )
        for worker in range(8)
    )

    return subprocess.run(
        ["bash", str(gate)],
        check=False,
        capture_output=True,
        env=os.environ
        | {
            "NEMO_RL_REPO_ROOT": str(fake_root),
            "CANARY_RESULT_ROOT": str(result_root),
            "SHAPE_TRACE_DIR": str(trace_dir),
            "TRACE_SUMMARY": json.dumps(
                {"eligible": True, "record_count": 8, "unique_signature_count": 1}
            ),
            "TRACE_RECORDS": records,
            "SHAPE_TRACE_MAX": str(trace_cap),
            "CANARY_EXPECTED_TRACE_WORKERS": "8",
        },
        text=True,
    )


def test_qwen235_qkvo_trace_gate_requires_fused_qkv_and_o_projection(
    tmp_path: Path,
) -> None:
    result = _run_qkvo_trace_gate(
        tmp_path,
        [
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.o_proj",
        ],
    )

    assert result.returncode == 0, result.stderr
    coverage = json.loads(
        (tmp_path / "result/trace/qkvo_coverage.json").read_text(encoding="utf-8")
    )
    assert coverage["qkv_prefix_count"] == 1
    assert coverage["o_prefix_count"] == 1
    assert coverage["hostname_count"] == 2
    assert coverage["worker_count"] == 8
    assert (tmp_path / "result/trace/qkvo_manifest.json").is_file()
    assert (tmp_path / "result/trace/shmoo/shapes_8x4.txt").is_file()


def test_qwen235_qkvo_trace_gate_rejects_missing_o_projection(tmp_path: Path) -> None:
    result = _run_qkvo_trace_gate(
        tmp_path, ["model.layers.0.self_attn.qkv_proj"]
    )

    assert result.returncode != 0
    assert "missing MXFP8 trace families: o_proj" in result.stderr


def test_qwen235_qkvo_trace_gate_rejects_trace_cap_reached(tmp_path: Path) -> None:
    result = _run_qkvo_trace_gate(
        tmp_path,
        [
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.o_proj",
        ],
        trace_cap=8,
    )

    assert result.returncode != 0
    assert "trace cap reached (8/8)" in result.stderr


def test_qwen235_qkvo_trace_gate_rejects_invalid_physical_signature(
    tmp_path: Path,
) -> None:
    result = _run_qkvo_trace_gate(
        tmp_path,
        [
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.o_proj",
        ],
        n_physical=1500,
    )

    assert result.returncode != 0
    assert "invalid MXFP8 physical signature" in result.stderr


def _run_trace_gate(tmp_path: Path, summary: dict[str, object]) -> subprocess.CompletedProcess[str]:
    fake_root = tmp_path / "repo"
    trace_script = fake_root / "experiments/mxfp8_adaptive_rollout_v0251/run_trace.sh"
    trace_script.parent.mkdir(parents=True)
    trace_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$CANARY_RESULT_ROOT/trace\"\n"
        "printf '%s\\n' \"$TRACE_SUMMARY\" > \"$CANARY_RESULT_ROOT/trace/shape_summary.json\"\n",
        encoding="utf-8",
    )
    trace_script.chmod(0o755)
    result_root = tmp_path / "result"
    gate = EXPERIMENT / "run_qwen235_trace_gate.sh"
    assert gate.is_file()

    return subprocess.run(
        ["bash", str(gate)],
        check=False,
        capture_output=True,
        env=os.environ
        | {
            "NEMO_RL_REPO_ROOT": str(fake_root),
            "CANARY_RESULT_ROOT": str(result_root),
            "TRACE_SUMMARY": json.dumps(summary),
        },
        text=True,
    )


def test_qwen235_trace_gate_accepts_eligible_shape_summary(tmp_path: Path) -> None:
    result = _run_trace_gate(
        tmp_path,
        {"eligible": True, "record_count": 1, "unique_signature_count": 1},
    )

    assert result.returncode == 0, result.stderr


def test_qwen235_trace_gate_rejects_empty_shape_summary(tmp_path: Path) -> None:
    result = _run_trace_gate(
        tmp_path,
        {"eligible": False, "record_count": 0, "unique_signature_count": 0},
    )

    assert result.returncode != 0
    assert "Qwen235 trace gate failed" in result.stderr
