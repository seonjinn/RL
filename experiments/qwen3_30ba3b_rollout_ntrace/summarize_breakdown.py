"""Build a compact, conserved summary from per-rank ntrace breakdowns."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _stack_category(path: tuple[str, ...]) -> str:
    label = " > ".join(path).lower()
    if any(
        token in label
        for token in (
            "moe_runner.py",
            "trtllm_fp8_moe.py",
            "trtllm_fp8_block_scale_moe",
        )
    ):
        return "moe"
    if any(token in label for token in ("attention", "fmha", "flash_attn")):
        return "attention"
    if any(token in label for token in ("sampler", "topk_topp", "sample_tokens")):
        return "sampling"
    if any(
        token in label
        for token in ("reshape_and_cache", "slot_mapping", "kv_cache")
    ):
        return "kv_cache"
    if "no python stack" in label:
        return "no_python_stack"
    if any(
        token in label
        for token in ("__vllm_inlined_submods__", "execution_fn", "cuda_graph.py")
    ):
        return "compiled_model_unresolved"
    return "runtime_other"


def _raw_kernel_category(name: str) -> str:
    lower = name.lower()
    if name.startswith("bmm_MxE4m3_"):
        return "expert_fc1_bmm"
    if name.startswith("bmm_Bfloat16_"):
        return "expert_fc2_bmm"
    if name.startswith("bmm_"):
        return "expert_bmm_other"
    if "moe::dev::routing" in lower or "moe::dev::finalize" in lower:
        return "moe_routing_finalize"
    if "fmha" in lower or "attention" in lower:
        return "attention"
    if "mxfp8_quantize" in lower or "quantize" in lower:
        return "quantization_layout"
    if "reshape_and_cache" in lower or "slot_mapping" in lower:
        return "kv_cache"
    if "softmax" in lower or "topk" in lower or "sample" in lower:
        return "sampling"
    if "triton_" in lower:
        return "triton_fused"
    return "other"


def _rank_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    num_iterations = int(data["num_iterations"])
    step_s = float(data["time_avg_ns"]) / 1e9
    idle_s = float(data["time_no_further_nvtx_range_avg_ns"]) / 1e9

    kernel_classes_ns: Counter[str] = Counter()
    stack_categories_ns: Counter[str] = Counter()
    stack_categories_per_iter_ns = [Counter() for _ in range(num_iterations)]
    stack: list[tuple[dict[str, Any], tuple[str, ...]]] = [(data, ())]
    while stack:
        node, parent_path = stack.pop()
        node_path = parent_path + (str(node.get("name", "?")),)
        breakdown = node.get("kernel_breakdown_avg", {})
        kernel_classes_ns.update(
            {
                str(category): float(duration_ns)
                for category, duration_ns in breakdown.items()
            }
        )
        active_ns = sum(
            float(duration_ns)
            for category, duration_ns in breakdown.items()
            if category != "Idle"
        )
        if active_ns:
            stack_categories_ns[_stack_category(node_path)] += active_ns
        stack_category = _stack_category(node_path)
        for category, durations_ns in node.get(
            "kernel_breakdown_per_iter", {}
        ).items():
            if category == "Idle":
                continue
            for index, duration_ns in enumerate(durations_ns):
                stack_categories_per_iter_ns[index][stack_category] += float(
                    duration_ns
                )
        stack.extend((child, node_path) for child in node.get("children", []))

    raw_other_ns: Counter[str] = Counter()
    for name, stats in data["instance_stats_global"].get("Other", {}).items():
        instances = stats["instances"]
        duration_ns = (
            int(instances["count"]) * float(instances["avg_ns"]) / num_iterations
        )
        raw_other_ns[_raw_kernel_category(name)] += duration_ns

    active_s = step_s - idle_s
    step_per_iter_s = [float(value) / 1e9 for value in data["time_per_iter_ns"]]
    idle_per_iter_s = [
        float(value) / 1e9
        for value in data["time_no_further_nvtx_range_per_iter_ns"]
    ]
    return {
        "rank": int(path.parent.name.removeprefix("rank")),
        "num_iterations": num_iterations,
        "step_s": step_s,
        "active_s": active_s,
        "idle_s": idle_s,
        "idle_pct": 100.0 * idle_s / step_s,
        "kernel_classes_s": {
            key: value / 1e9 for key, value in sorted(kernel_classes_ns.items())
        },
        "stack_categories_s": {
            key: value / 1e9 for key, value in sorted(stack_categories_ns.items())
        },
        "raw_other_categories_s": {
            key: value / 1e9 for key, value in sorted(raw_other_ns.items())
        },
        "iterations": [
            {
                "index": index,
                "label": data["iteration_labels"][index],
                "step_s": step_per_iter_s[index],
                "active_s": step_per_iter_s[index] - idle_per_iter_s[index],
                "idle_s": idle_per_iter_s[index],
                "idle_pct": 100.0
                * idle_per_iter_s[index]
                / step_per_iter_s[index],
                "stack_categories_s": {
                    key: value / 1e9
                    for key, value in sorted(
                        stack_categories_per_iter_ns[index].items()
                    )
                },
            }
            for index in range(num_iterations)
        ],
    }


def _metric_summary(ranks: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(rank[key]) for rank in ranks]
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "max_over_min": max(values) / min(values),
    }


def _iteration_summary(
    ranks: list[dict[str, Any]], index: int
) -> dict[str, Any]:
    iterations = [rank["iterations"][index] for rank in ranks]
    stack_keys = sorted(
        set().union(*(iteration["stack_categories_s"] for iteration in iterations))
    )
    return {
        "index": index,
        "label": iterations[0]["label"],
        "step_s": _metric_summary(iterations, "step_s"),
        "active_s": _metric_summary(iterations, "active_s"),
        "idle_s": _metric_summary(iterations, "idle_s"),
        "idle_pct": _metric_summary(iterations, "idle_pct"),
        "stack_categories_s": {
            key: statistics.fmean(
                iteration["stack_categories_s"].get(key, 0.0)
                for iteration in iterations
            )
            for key in stack_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_root", type=Path)
    args = parser.parse_args()

    analysis_root = args.analysis_root.resolve()
    paths = [
        analysis_root / "per_rank" / f"rank{rank}" / "ntrace_breakdown.json"
        for rank in range(8)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing breakdowns:\n" + "\n".join(missing))

    ranks = [_rank_summary(path) for path in paths]
    payload = {
        "ranks": ranks,
        "aggregate": {
            "step_s": _metric_summary(ranks, "step_s"),
            "active_s": _metric_summary(ranks, "active_s"),
            "idle_s": _metric_summary(ranks, "idle_s"),
            "idle_pct": _metric_summary(ranks, "idle_pct"),
        },
        "iterations": [
            _iteration_summary(ranks, index)
            for index in range(ranks[0]["num_iterations"])
        ],
        "notes": {
            "stack_categories": (
                "Conserved self time classified from each breakdown node's full "
                "Python stack path."
            ),
            "raw_other_categories": (
                "Additive raw-kernel instance time used to split ntrace's Other "
                "class; it is diagnostic and is not a conserved wall-time total."
            ),
        },
    }
    (analysis_root / "rollout_bottleneck_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    columns = (
        "rank",
        "step_s",
        "active_s",
        "idle_s",
        "idle_pct",
        "moe_stack_s",
        "expert_fc1_bmm_raw_s",
        "expert_fc2_bmm_raw_s",
        "attention_raw_s",
        "quantization_layout_raw_s",
    )
    lines = ["\t".join(columns)]
    for rank in ranks:
        stack_categories = rank["stack_categories_s"]
        raw_categories = rank["raw_other_categories_s"]
        values = (
            rank["rank"],
            rank["step_s"],
            rank["active_s"],
            rank["idle_s"],
            rank["idle_pct"],
            stack_categories.get("moe", 0.0),
            raw_categories.get("expert_fc1_bmm", 0.0),
            raw_categories.get("expert_fc2_bmm", 0.0),
            raw_categories.get("attention", 0.0),
            raw_categories.get("quantization_layout", 0.0),
        )
        lines.append("\t".join(str(value) for value in values))
    (analysis_root / "rollout_bottleneck_summary.tsv").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
