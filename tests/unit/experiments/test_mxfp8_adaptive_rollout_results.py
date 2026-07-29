# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).parents[3]
PARSER_PATH = REPO_ROOT / "experiments" / "mxfp8_adaptive_rollout" / "parse_results.py"
NEMO_COMMIT = "8" * 40
VLLM_COMMIT = "b" * 40
CONTAINER_DIGEST = "sha256:" + "c" * 64
CONFIG_HASH = "d" * 64


def _load_parser() -> ModuleType:
    assert PARSER_PATH.is_file(), f"missing experiment parser: {PARSER_PATH}"
    spec = importlib.util.spec_from_file_location(
        "mxfp8_adaptive_rollout_results", PARSER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_log(*, arm: str = "original", repeat: int = 1) -> str:
    metadata = {
        "arm": arm,
        "repeat": repeat,
        "vllm_commit": VLLM_COMMIT,
        "nemo_rl_commit": NEMO_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "config_hash": CONFIG_HASH if arm == "adaptive" else "none",
        "tensor_parallel_size": 1,
        "seed": 42,
        "num_samples": 8,
        "generation_num_gpus": 4,
    }
    return f"""\
MXFP8_AB_METADATA {json.dumps(metadata, sort_keys=True)}
========================= Step 2/3 =========================
  • Mean Generation Length: 12.5000
  • Total step time: 9.00s
  • generation: 4.00s (44.4%)
  • timing/rollout/run_rollouts: 3.50s (38.9%)
========================= Step 3/3 =========================
  • Mean Generation Length: 10.0000
  • Total step time: 8.00s
  • generation: 3.00s (37.5%)
"""


def test_parse_log_extracts_step_metrics_and_provenance() -> None:
    parser = _load_parser()

    records = parser.parse_log(_literal_log())

    assert len(records) == 2
    first = records[0]
    assert first.step == 2
    assert first.arm == "original"
    assert first.repeat == 1
    assert first.rollout_wall_time_s == pytest.approx(3.5)
    assert first.generation_time_s == pytest.approx(4.0)
    assert first.total_step_time_s == pytest.approx(9.0)
    assert first.output_tokens == 100
    assert first.output_tokens_per_second_per_gpu == pytest.approx(100 / 3.5 / 4)
    assert first.vllm_commit == VLLM_COMMIT
    assert first.nemo_rl_commit == NEMO_COMMIT
    assert first.container_digest == CONTAINER_DIGEST
    assert first.config_hash == "none"
    assert first.tensor_parallel_size == 1
    assert first.seed == 42

    second = records[1]
    assert second.step == 3
    assert second.rollout_wall_time_s == pytest.approx(3.0)
    assert second.generation_time_s == pytest.approx(3.0)
    assert second.output_tokens == 80


def test_parse_log_rejects_missing_required_provenance() -> None:
    parser = _load_parser()
    metadata = {
        "arm": "original",
        "repeat": 1,
        "vllm_commit": VLLM_COMMIT,
        "nemo_rl_commit": NEMO_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "tensor_parallel_size": 1,
        "seed": 42,
        "num_samples": 8,
        "generation_num_gpus": 4,
    }
    log = f"""\
MXFP8_AB_METADATA {json.dumps(metadata, sort_keys=True)}
========================= Step 1/1 =========================
  • Mean Generation Length: 12.5000
  • Total step time: 9.00s
  • generation: 4.00s (44.4%)
"""

    with pytest.raises(ValueError, match="config_hash"):
        parser.parse_log(log)


def test_validate_ab_pair_allows_only_the_json_environment_key_to_differ() -> None:
    parser = _load_parser()
    common = {
        "nemo_rl_commit": NEMO_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "checkpoint": "Qwen/Qwen3-30B-A3B",
        "topology": {
            "num_nodes": 4,
            "gpus_per_node": 4,
            "tensor_parallel_size": 1,
        },
    }
    original = {
        **common,
        "resolved_config": {
            "grpo": {"seed": 42},
            "policy": {
                "generation": {
                    "vllm_cfg": {
                        "precision": "fp8",
                        "is_mx": True,
                        "env_vars": {"KEEP_ME": "same"},
                    }
                }
            },
        },
    }
    adaptive = {
        **common,
        "resolved_config": {
            "grpo": {"seed": 42},
            "policy": {
                "generation": {
                    "vllm_cfg": {
                        "precision": "fp8",
                        "is_mx": True,
                        "env_vars": {
                            "KEEP_ME": "same",
                            "VLLM_MXFP8_DENSE_CONFIG_FILE": (
                                "qwen3_30ba3b_tp1_v0202_qualified.json"
                            ),
                        },
                    }
                }
            },
        },
    }

    parser.validate_ab_pair(original, adaptive)

    adaptive["resolved_config"]["grpo"]["seed"] = 43
    with pytest.raises(ValueError, match="resolved Hydra config"):
        parser.validate_ab_pair(original, adaptive)


def test_validate_ab_pair_rejects_provenance_mismatch() -> None:
    parser = _load_parser()
    original = {
        "nemo_rl_commit": NEMO_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "container_digest": CONTAINER_DIGEST,
        "checkpoint": "Qwen/Qwen3-30B-A3B",
        "topology": {"num_nodes": 4, "gpus_per_node": 4},
        "resolved_config": {"grpo": {"seed": 42}},
    }
    adaptive = {**original, "container_digest": "sha256:" + "e" * 64}

    with pytest.raises(ValueError, match="container_digest"):
        parser.validate_ab_pair(original, adaptive)


def test_write_summaries_is_stable_across_input_order(tmp_path: Path) -> None:
    parser = _load_parser()
    original = parser.parse_log(_literal_log(arm="original", repeat=1))[0]
    adaptive = parser.parse_log(_literal_log(arm="adaptive", repeat=1))[0]
    first_json_path = tmp_path / "first-summary.json"
    first_csv_path = tmp_path / "first-summary.csv"
    second_json_path = tmp_path / "second-summary.json"
    second_csv_path = tmp_path / "second-summary.csv"

    parser.write_summaries([adaptive, original], first_json_path, first_csv_path)
    first_json = first_json_path.read_bytes()
    first_csv = first_csv_path.read_bytes()
    parser.write_summaries([original, adaptive], second_json_path, second_csv_path)

    assert second_json_path.read_bytes() == first_json
    assert second_csv_path.read_bytes() == first_csv
    assert [row["arm"] for row in json.loads(first_json)] == [
        "original",
        "adaptive",
    ]
    assert first_json.endswith(b"\n")
    assert first_csv.startswith(
        b"step,arm,repeat,rollout_wall_time_s,generation_time_s,"
    )
    assert first_csv.endswith(b"\n")


def test_write_summaries_refuses_to_overwrite_existing_output(
    tmp_path: Path,
) -> None:
    parser = _load_parser()
    record = parser.parse_log(_literal_log())[0]
    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"
    json_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        parser.write_summaries([record], json_path, csv_path)

    assert json_path.read_text(encoding="utf-8") == "existing\n"
    assert not csv_path.exists()


def test_not_applicable_result_names_zero_hit_and_ultra_tp4_fallback() -> None:
    parser = _load_parser()

    result = parser.not_applicable_result(
        "trace files contain zero eligible dense MXFP8 records"
    )

    assert result == {
        "fallback": {
            "model": "Nemotron 3 Ultra",
            "tensor_parallel_size": 4,
        },
        "reason": "trace files contain zero eligible dense MXFP8 records",
        "status": "not-applicable",
        "workload": "Qwen/Qwen3-30B-A3B",
    }
