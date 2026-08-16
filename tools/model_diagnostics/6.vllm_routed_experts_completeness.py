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
"""Check whether vLLM returns complete and valid routed-expert outputs.

This is a standalone diagnostic for MoE models and vLLM builds that support
``enable_return_routed_experts``. It checks route counts, expert-id ranges, and
top-k uniqueness on MoE layers.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_extra_kwarg(raw: str) -> tuple[str, Any]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"Expected --llm-kwarg KEY=VALUE, got {raw!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def _route_count(route_tensor: Any) -> int:
    if route_tensor is None:
        return 0
    return len(route_tensor)


def _find_route_semantic_failures(
    route_tensor: Any,
    *,
    moe_layer_indices: list[int],
    num_experts: int,
    max_failures: int = 20,
) -> list[dict[str, Any]]:
    routes = route_tensor.tolist() if hasattr(route_tensor, "tolist") else route_tensor
    failures = []
    for token_idx, token_routes in enumerate(routes):
        for layer_idx in moe_layer_indices:
            route = list(token_routes[layer_idx])
            if len(set(route)) != len(route):
                failures.append(
                    {
                        "token": token_idx,
                        "layer": layer_idx,
                        "reason": "duplicate_expert_ids",
                        "route": route,
                    }
                )
            elif any(expert_id < 0 or expert_id >= num_experts for expert_id in route):
                failures.append(
                    {
                        "token": token_idx,
                        "layer": layer_idx,
                        "reason": "expert_id_out_of_range",
                        "route": route,
                        "num_experts": num_experts,
                    }
                )
            if len(failures) >= max_failures:
                return failures
    return failures


def _resolve_route_contract(llm: Any) -> tuple[list[int], int]:
    hf_config = llm.llm_engine.model_config.hf_config
    num_experts = next(
        (
            value
            for name in ("num_experts", "n_routed_experts", "num_local_experts")
            if isinstance((value := getattr(hf_config, name, None)), int) and value > 0
        ),
        None,
    )
    if num_experts is None:
        raise ValueError("Could not determine the routed-expert count from the model")

    pattern = getattr(hf_config, "hybrid_override_pattern", None)
    if pattern is not None:
        moe_layer_indices = [
            layer_idx
            for layer_idx, layer_type in enumerate(pattern)
            if layer_type == "E"
        ]
    else:
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
        if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
            raise ValueError("Could not determine the model layer count")
        moe_layer_indices = list(range(num_hidden_layers))

    if not moe_layer_indices:
        raise ValueError("The model configuration does not contain MoE layers")
    return moe_layer_indices, num_experts


def _check_request_output(
    sample_idx: int,
    request_output: Any,
    *,
    moe_layer_indices: list[int],
    num_experts: int,
) -> list[dict[str, Any]]:
    completion_output = request_output.outputs[0]
    prompt_route_tensor = getattr(request_output, "prompt_routed_experts", None)
    completion_route_tensor = getattr(completion_output, "routed_experts", None)
    prompt_routes = _route_count(prompt_route_tensor)
    completion_routes = _route_count(completion_route_tensor)
    actual_routes = prompt_routes + completion_routes
    valid_length = len(request_output.prompt_token_ids) + len(
        completion_output.token_ids
    )
    expected_routes = max(valid_length - 1, 0)
    max_allowed_routes = expected_routes + 1

    failures = []
    if actual_routes < expected_routes or actual_routes > max_allowed_routes:
        failures.append(
            {
                "sample": sample_idx,
                "reason": "route_count_mismatch",
                "prompt_routes": prompt_routes,
                "completion_routes": completion_routes,
                "actual_routes": actual_routes,
                "expected_routes": expected_routes,
                "max_allowed_routes": max_allowed_routes,
                "prompt_tokens": len(request_output.prompt_token_ids),
                "completion_tokens": len(completion_output.token_ids),
                "num_cached_tokens": getattr(request_output, "num_cached_tokens", None),
            }
        )

    for segment, route_tensor in (
        ("prompt", prompt_route_tensor),
        ("completion", completion_route_tensor),
    ):
        if route_tensor is None:
            continue
        for failure in _find_route_semantic_failures(
            route_tensor,
            moe_layer_indices=moe_layer_indices,
            num_experts=num_experts,
        ):
            failures.append({"sample": sample_idx, "segment": segment, **failure})
    return failures


def _write_summary(summary: dict[str, Any], output_path: Path) -> None:
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text(f"{json.dumps(summary, indent=2, sort_keys=True)}\n")
    partial_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prompt-repeat", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--llm-kwarg",
        action="append",
        default=[],
        type=_parse_extra_kwarg,
        help="Extra vLLM LLM kwarg as KEY=VALUE. VALUE may be JSON.",
    )
    args = parser.parse_args()

    from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat

    # Must run before vLLM pulls in tool_parsers (openai<2.25 NamespaceTool compat).
    ensure_vllm_source_compat()

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_return_routed_experts": True,
    }
    if args.enable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    for key, value in args.llm_kwarg:
        llm_kwargs[key] = value

    llm = LLM(**llm_kwargs)
    moe_layer_indices, num_experts = _resolve_route_contract(llm)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
    )

    shared_prefix = "Solve the following math problem carefully. " * args.prompt_repeat
    prompts = [
        f"{shared_prefix}\nProblem {idx}: What is {idx} plus {idx + 1}?"
        for idx in range(args.num_prompts)
    ]
    outputs = llm.generate(prompts, sampling_params)

    failures = []
    for idx, request_output in enumerate(outputs):
        failures.extend(
            _check_request_output(
                idx,
                request_output,
                moe_layer_indices=moe_layer_indices,
                num_experts=num_experts,
            )
        )

    summary = {
        "model": args.model,
        "num_outputs": len(outputs),
        "num_failures": len(failures),
        "moe_layer_indices": moe_layer_indices,
        "num_experts": num_experts,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "failures": failures[:20],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output is not None:
        _write_summary(summary, args.output)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
