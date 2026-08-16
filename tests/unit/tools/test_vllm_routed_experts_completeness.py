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
from types import ModuleType, SimpleNamespace


def _load_diagnostic() -> ModuleType:
    diagnostic_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "model_diagnostics"
        / "6.vllm_routed_experts_completeness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_routed_experts_completeness", diagnostic_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_route_semantics_reject_duplicate_experts_on_moe_layers() -> None:
    diagnostic = _load_diagnostic()
    validator = getattr(
        diagnostic,
        "_find_route_semantic_failures",
        lambda *_args, **_kwargs: [],
    )
    routes = [
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [1, 3], [0, 0]],
    ]

    failures = validator(routes, moe_layer_indices=[1], num_experts=4)

    assert failures == [
        {
            "token": 0,
            "layer": 1,
            "reason": "duplicate_expert_ids",
            "route": [0, 0],
        }
    ]


def test_route_semantics_reject_expert_ids_outside_model_range() -> None:
    diagnostic = _load_diagnostic()
    routes = [
        [[0, 0], [1, 4], [0, 0]],
    ]

    failures = diagnostic._find_route_semantic_failures(
        routes, moe_layer_indices=[1], num_experts=4
    )

    assert failures == [
        {
            "token": 0,
            "layer": 1,
            "reason": "expert_id_out_of_range",
            "route": [1, 4],
            "num_experts": 4,
        }
    ]


def test_route_semantics_ignore_dense_layer_placeholders() -> None:
    diagnostic = _load_diagnostic()
    routes = [
        [[0, 0], [1, 3], [0, 0]],
    ]

    failures = diagnostic._find_route_semantic_failures(
        routes, moe_layer_indices=[1], num_experts=4
    )

    assert failures == []


def test_route_contract_uses_only_hybrid_moe_layers() -> None:
    diagnostic = _load_diagnostic()
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(hybrid_override_pattern="MEME", num_experts=4)
            )
        )
    )
    resolver = getattr(diagnostic, "_resolve_route_contract", lambda _llm: None)

    assert resolver(llm) == ([1, 3], 4)


def test_request_check_rejects_semantically_invalid_routes_with_complete_count() -> (
    None
):
    diagnostic = _load_diagnostic()
    request_output = SimpleNamespace(
        prompt_token_ids=[11],
        prompt_routed_experts=None,
        num_cached_tokens=0,
        outputs=[
            SimpleNamespace(
                token_ids=[12],
                routed_experts=[[[0, 0], [0, 0], [0, 0]]],
            )
        ],
    )
    checker = getattr(
        diagnostic,
        "_check_request_output",
        lambda *_args, **_kwargs: [],
    )

    failures = checker(
        7,
        request_output,
        moe_layer_indices=[1],
        num_experts=4,
    )

    assert failures == [
        {
            "sample": 7,
            "segment": "completion",
            "token": 0,
            "layer": 1,
            "reason": "duplicate_expert_ids",
            "route": [0, 0],
        }
    ]


def test_route_semantic_failures_are_bounded() -> None:
    diagnostic = _load_diagnostic()
    routes = [[[0, 0]] for _ in range(30)]

    failures = diagnostic._find_route_semantic_failures(
        routes,
        moe_layer_indices=[0],
        num_experts=4,
        max_failures=5,
    )

    assert len(failures) == 5


def test_summary_writer_publishes_json_atomically(tmp_path: Path) -> None:
    diagnostic = _load_diagnostic()
    output_path = tmp_path / "result.json"
    writer = getattr(diagnostic, "_write_summary", lambda *_args: None)

    writer({"num_failures": 0}, output_path)

    assert json.loads(output_path.read_text()) == {"num_failures": 0}
    assert not output_path.with_suffix(".json.partial").exists()
