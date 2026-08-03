import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments/mxfp8_adaptive_rollout_v0251"


def test_qwen235_trace_config_defines_two_tp4_ep4_cuda_graph_engines() -> None:
    config_path = EXPERIMENT / "configs/eval_qwen3_235ba22b_32k_cuda_graph_trace.yaml"
    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["eval"] == {
        "num_tests_per_prompt": 8,
        "save_path": "${oc.env:CANARY_OUTPUT_DIR}",
        "seed": 42,
    }
    generation = config["generation"]
    assert generation["model_name"] == "Qwen/Qwen3-235B-A22B"
    assert generation["max_new_tokens"] == 32768
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
    assert "models--Qwen--Qwen3-235B-A22B" in submitter

    assert "EXPECTED_NEMO_RL_COMMIT:?set EXPECTED_NEMO_RL_COMMIT" in submitter
    assert "status --porcelain --untracked-files=all" in submitter
    assert 'require_clean_repo "$NEMO_RL_REPO_ROOT"' in submitter
    assert 'require_clean_repo "$CUSTOM_VLLM_SOURCE"' in submitter
    assert 'git -C "$NEMO_RL_REPO_ROOT" pull --ff-only' in submitter
    assert submitter.index('git -C "$NEMO_RL_REPO_ROOT" pull --ff-only') < (
        submitter.index("lock_sha=")
    )
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
