import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments/mxfp8_adaptive_rollout_v0251"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_dataset(path: Path, rows: list[dict[str, str]]) -> str:
    encoded = "".join(
        json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        for row in rows
    )
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode()).hexdigest()


def _write_eval(
    directory: Path,
    rewards: list[float],
    *,
    prompts: list[str] | None = None,
) -> None:
    if prompts is None:
        prompts = [f"problem-{index}" for index in range(len(rewards))]
    evaluation_data = [
        {
            "sample_index": index,
            "prompt": prompt,
            "response": f"answer-{index}",
            "reward": reward,
        }
        for index, (prompt, reward) in enumerate(zip(prompts, rewards, strict=True))
    ]
    _write_json(
        directory / "evaluation_data.json", {"evaluation_data": evaluation_data}
    )
    _write_json(
        directory / "config.json",
        {
            "model_name": "Qwen/Qwen3-235B-A22B",
            "dataset_name": "/tmp/gsm8k_test.jsonl",
            "metric": "pass@k",
            "k_value": 1,
            "num_tests_per_prompt": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
        },
    )


def _run_gate(
    tmp_path: Path,
    baseline_rewards: list[float],
    adaptive_rewards: list[float],
    *,
    adaptive_prompts: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    baseline_dir = tmp_path / "baseline"
    adaptive_dir = tmp_path / "adaptive"
    _write_eval(baseline_dir, baseline_rewards)
    _write_eval(adaptive_dir, adaptive_rewards, prompts=adaptive_prompts)

    rows = [
        {
            "input": f"problem-{index}",
            "output": str(index),
            "sample_id": f"gsm8k-test-{index:04d}",
        }
        for index in range(len(baseline_rewards))
    ]
    dataset_path = tmp_path / "gsm8k_test.jsonl"
    dataset_sha256 = _write_dataset(dataset_path, rows)
    manifest_path = tmp_path / "gsm8k_test.manifest.json"
    _write_json(
        manifest_path,
        {
            "dataset_id": "openai/gsm8k",
            "dataset_config": "main",
            "split": "test",
            "revision": "pinned-revision",
            "row_count": len(rows),
            "jsonl_sha256": dataset_sha256,
        },
    )

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.mxfp8_adaptive_rollout_v0251.gsm8k_correctness_gate",
            "--baseline-dir",
            str(baseline_dir),
            "--adaptive-dir",
            str(adaptive_dir),
            "--dataset",
            str(dataset_path),
            "--manifest",
            str(manifest_path),
            "--expected-rows",
            str(len(rows)),
            "--output",
            str(tmp_path / "report.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_materializer_writes_deterministic_jsonl_and_sha256_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            [
                json.dumps({"question": "one?", "answer": "work #### 1"}) + "\n",
                json.dumps({"question": "two?", "answer": "work #### 2"}) + "\n",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "gsm8k_test.jsonl"
    manifest = tmp_path / "gsm8k_test.manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.mxfp8_adaptive_rollout_v0251.materialize_gsm8k",
            "--source-jsonl",
            str(source),
            "--expected-rows",
            "2",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {"input": "one?", "output": "1", "sample_id": "gsm8k-test-0000"},
        {"input": "two?", "output": "2", "sample_id": "gsm8k-test-0001"},
    ]
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    assert provenance["row_count"] == 2
    assert provenance["jsonl_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert provenance["revision"]


def test_correctness_gate_accepts_matched_non_regressing_results(
    tmp_path: Path,
) -> None:
    result = _run_gate(tmp_path, [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0])

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["baseline_accuracy"] == 0.5
    assert report["adaptive_accuracy"] == 0.75
    assert report["paired"]["adaptive_gains"] == 1
    assert report["paired"]["adaptive_losses"] == 0


def test_correctness_gate_rejects_statistically_significant_paired_regression(
    tmp_path: Path,
) -> None:
    result = _run_gate(tmp_path, [1.0] * 6, [0.0] * 6)

    assert result.returncode != 0
    assert "statistically significant accuracy regression" in result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["paired"]["one_sided_p_value"] == 0.015625


def test_correctness_gate_rejects_misaligned_samples(tmp_path: Path) -> None:
    result = _run_gate(
        tmp_path,
        [1.0, 0.0],
        [1.0, 0.0],
        adaptive_prompts=["problem-1", "problem-0"],
    )

    assert result.returncode != 0
    assert "prompt mismatch" in result.stderr


def test_correctness_gate_rejects_malformed_reward(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    adaptive_dir = tmp_path / "adaptive"
    _write_eval(baseline_dir, [1.0])
    _write_eval(adaptive_dir, [1.0])
    payload = json.loads(
        (adaptive_dir / "evaluation_data.json").read_text(encoding="utf-8")
    )
    payload["evaluation_data"][0]["reward"] = 0.5
    _write_json(adaptive_dir / "evaluation_data.json", payload)

    dataset_path = tmp_path / "gsm8k_test.jsonl"
    dataset_sha256 = _write_dataset(
        dataset_path,
        [{"input": "problem-0", "output": "0", "sample_id": "gsm8k-test-0000"}],
    )
    manifest_path = tmp_path / "gsm8k_test.manifest.json"
    _write_json(
        manifest_path,
        {
            "row_count": 1,
            "jsonl_sha256": dataset_sha256,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.mxfp8_adaptive_rollout_v0251.gsm8k_correctness_gate",
            "--baseline-dir",
            str(baseline_dir),
            "--adaptive-dir",
            str(adaptive_dir),
            "--dataset",
            str(dataset_path),
            "--manifest",
            str(manifest_path),
            "--expected-rows",
            "1",
            "--output",
            str(tmp_path / "report.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "non-binary reward" in result.stderr


def test_qwen235_gsm8k_config_and_wrapper_are_greedy_matched_and_opt_in() -> None:
    config_path = EXPERIMENT / "configs/eval_qwen3_235ba22b_gsm8k_correctness.yaml"
    wrapper_path = EXPERIMENT / "run_qwen235_gsm8k_correctness.sh"
    assert config_path.is_file()
    assert wrapper_path.is_file()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["eval"]["num_tests_per_prompt"] == 1
    assert config["eval"]["metric"] == "pass@k"
    assert config["eval"]["k_value"] == 1
    assert config["generation"]["model_name"] == "Qwen/Qwen3-235B-A22B"
    assert config["generation"]["temperature"] == 0.0
    assert config["generation"]["top_p"] == 1.0
    assert config["generation"]["top_k"] == -1
    assert config["data"]["dataset_name"] == "${oc.env:GSM8K_JSONL}"

    wrapper = wrapper_path.read_text(encoding="utf-8")
    assert "NEMORL_ENABLE_QWEN235_GSM8K_CORRECTNESS" in wrapper
    assert "materialize_gsm8k" in wrapper
    assert 'run_arm.sh" baseline' in wrapper
    assert 'run_arm.sh" adaptive' in wrapper
    assert 'run_arm.sh" trtllm_default' not in wrapper
    assert "gsm8k_correctness_gate" in wrapper
    assert "GSM8K_EXPECTED_SHA256" in wrapper

    disabled = subprocess.run(
        ["bash", str(wrapper_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env=os.environ
        | {
            "NEMORL_ENABLE_QWEN235_GSM8K_CORRECTNESS": "0",
        },
        text=True,
    )
    assert disabled.returncode == 2
    assert "explicit opt-in" in disabled.stderr


def test_qwen235_gsm8k_submitter_is_pinned_and_dependency_free() -> None:
    submitter_path = EXPERIMENT / "submit_qwen235_gsm8k_correctness_ptyche.sh"
    assert submitter_path.is_file()
    submitter = submitter_path.read_text(encoding="utf-8")

    assert "run_qwen235_gsm8k_correctness.sh" in submitter
    assert "NEMORL_ENABLE_QWEN235_GSM8K_CORRECTNESS=1" in submitter
    assert "qwen235_tp4ep4_8x4_fix1_20260802" in submitter
    assert "bf1630d7327d58b6742ab0359c5993b59dc28c4ec96d9008c9fe0a1e399c189e" in submitter
    assert "models--Qwen--Qwen3-235B-A22B" in submitter
    assert "HF_DATASETS_CACHE=/home/sna/.cache/hf-datasets-canary" in submitter
    assert "--nodes=2" in submitter
    assert "--time=05:00:00" in submitter
    assert "--segment=2" in submitter
    assert "--dependency=" in submitter
    assert "args+=(--test-only)" in submitter
    assert "afterok" not in submitter
    assert "status --porcelain --untracked-files=all" in submitter
    assert 'git -C "$NEMO_RL_REPO_ROOT" pull --ff-only' in submitter
    assert 'require_clean_repo "$NEMO_RL_REPO_ROOT"' in submitter
    assert 'require_clean_repo "$CUSTOM_VLLM_SOURCE"' in submitter
