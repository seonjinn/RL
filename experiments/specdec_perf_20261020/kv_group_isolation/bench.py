"""Measure the per-step cost of vLLM KV cache grouping for Q35 SpecDec drafters.

One arm per process, one GPU each. The engine runs in-process so the patches
below reach the scheduler and the model runner, as in NeMo-RL's TP=1 workers.
The group_size arm replaces vLLM's group-size heuristic. It changes only the
KV block layout, not the math.
"""

import argparse
import importlib
import inspect
import json
import os
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "Qwen/Qwen3.5-35B-A3B-Base"
ASSETS = "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/qwen35-35ba3b"
# Same capture list as configs/qwen3_30ba3b/capture_k3_s128.yaml (spec arms only)
SPEC_CAPTURE_SIZES = [1, 2, 4] + list(range(8, 129, 4)) + list(range(136, 513, 8))
COT = (
    "Think step-by-step to solve the following problem. "
    "Output your answer inside of \\boxed{{}} tags.:\n{}\n\nLet's think step-by-step"
)

ARMS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "baseline_triton": {"attention_backend": "TRITON_ATTN"},
    "dflash_k3": {"spec": ("dflash", 3)},
    "dflash_k3_g5": {"spec": ("dflash", 3), "group_size": 5},
    "dspark_k4": {"spec": ("dspark", 4)},
}


def patch_group_size(group_size: int) -> None:
    import vllm.v1.core.kv_cache_utils as kcu

    src = textwrap.dedent(inspect.getsource(kcu._get_kv_cache_groups_uniform_page_size))
    swaps = {
        "group_size = min_num_layers\n": f"group_size = {group_size}\n",
        "if max_num_layers < min_num_layers * 1.5:": "if False:",
    }
    for old, new in swaps.items():
        assert src.count(old) == 1, (
            f"vLLM grouping source changed, cannot patch: {old!r}"
        )
        src = src.replace(old, new)
    exec(compile(src, kcu.__file__ or "kv_cache_utils.py", "exec"), vars(kcu))


def record_kv_groups(stats: dict[str, Any]) -> None:
    import vllm.v1.core.kv_cache_utils as kcu

    original = kcu._get_kv_cache_groups_uniform_page_size

    def wrapped(kv_cache_spec):
        groups = original(kv_cache_spec)
        stats["kv_cache_groups"] = len(groups)
        stats["layers_per_group"] = sorted({len(g.layer_names) for g in groups})
        stats["group_spec_types"] = sorted(
            {type(g.kv_cache_spec).__name__ for g in groups}
        )
        return groups

    kcu._get_kv_cache_groups_uniform_page_size = wrapped


def time_calls(timers: dict[str, list[float]]) -> None:
    """Wrap attention metadata builds and model runner steps with CPU timers."""
    import vllm.v1.worker.gpu.attn_utils as au

    build = au.build_attn_metadata

    def timed_build(*args, **kwargs):
        t = time.perf_counter()
        out = build(*args, **kwargs)
        timers["build_attn_metadata"].append(time.perf_counter() - t)
        return out

    for name, module in list(sys.modules.items()):
        if (
            name.startswith("vllm.v1.worker.gpu")
            and getattr(module, "build_attn_metadata", None) is build
        ):
            setattr(module, "build_attn_metadata", timed_build)

    # Patch both runners; the V2 runner (vllm.v1.worker.gpu) is the expected one
    for module_name in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
        runner_cls = importlib.import_module(module_name).GPUModelRunner
        for method in ("execute_model", "sample_tokens"):
            fn = getattr(runner_cls, method, None)
            if fn is None:
                continue

            def make(fn=fn, method=method, module_name=module_name):
                def timed(self, *args, **kwargs):
                    t = time.perf_counter()
                    out = fn(self, *args, **kwargs)
                    timers[method].append(time.perf_counter() - t)
                    timers[f"runner:{module_name}"].append(0.0)
                    return out

                return timed

            setattr(runner_cls, method, make())


def load_prompts(n: int) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        texts = [row["problem"] for row in ds.select(range(n))]
        source = "openmathinstruct-2"
    except Exception as e:  # noqa: BLE001
        print(f"dataset load failed ({e!r}); using synthetic problems", flush=True)
        texts = [
            f"Find the number of positive integers n < {100 + 7 * i} such that "
            f"n^2 + {i % 13 + 2}n is divisible by {i % 7 + 3}."
            for i in range(n)
        ]
        source = "synthetic"
    print(f"prompts: {n} from {source}", flush=True)
    return [COT.format(t) for t in texts]


def spec_counters(llm) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for m in llm.get_metrics():
        if m.name.startswith("vllm:spec_decode_num_"):
            value = getattr(m, "value", None)
            if isinstance(value, (int, float)):
                out[m.name.removeprefix("vllm:spec_decode_")] += value
    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    arm = ARMS[args.arm]

    stats: dict[str, Any] = {"arm": args.arm}
    if "group_size" in arm:
        patch_group_size(arm["group_size"])
    record_kv_groups(stats)

    import vllm
    from vllm import LLM, SamplingParams

    stats["vllm_version"] = vllm.__version__
    kwargs: dict[str, Any] = dict(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        max_num_seqs=128,
        enable_prefix_caching=False,
        enforce_eager=False,
        moe_backend="flashinfer_trtllm",
        disable_log_stats=False,
        seed=0,
    )
    if "attention_backend" in arm:
        kwargs["attention_backend"] = arm["attention_backend"]
    if "spec" in arm:
        method, k = arm["spec"]
        kwargs["speculative_config"] = {
            "method": method,
            "model": f"{ASSETS}/{method}",
            "num_speculative_tokens": k,
            "draft_tensor_parallel_size": 1,
            "attention_backend": "TRITON_ATTN",
        }
        kwargs["compilation_config"] = {"cudagraph_capture_sizes": SPEC_CAPTURE_SIZES}

    t = time.perf_counter()
    llm = LLM(**kwargs)
    stats["init_s"] = time.perf_counter() - t
    print(
        f"[{args.arm}] kv groups: {stats.get('kv_cache_groups')} {stats.get('layers_per_group')}",
        flush=True,
    )

    prompts = load_prompts(max(args.batches))
    warm = SamplingParams(temperature=1.0, max_tokens=128, ignore_eos=True)
    llm.generate(prompts[:32], warm, use_tqdm=False)

    timers: dict[str, list[float]] = defaultdict(list)
    time_calls(timers)
    phases = []
    for n in args.batches:
        timers.clear()
        params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, ignore_eos=True
        )
        before = spec_counters(llm)
        t = time.perf_counter()
        outs = llm.generate(prompts[:n], params, use_tqdm=False)
        wall = time.perf_counter() - t
        after = spec_counters(llm)
        delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        steps = len(timers["execute_model"])
        md = timers["build_attn_metadata"]
        phase = {
            "batch": n,
            "wall_s": wall,
            "output_tokens": tokens,
            "tok_per_s": tokens / wall,
            "engine_steps": steps,
            "step_ms": 1e3 * wall / max(steps, 1),
            "tokens_per_req_step": tokens / n / max(steps, 1),
            "execute_model_cpu_ms": 1e3 * sum(timers["execute_model"]) / max(steps, 1),
            "sample_tokens_cpu_ms": 1e3 * sum(timers["sample_tokens"]) / max(steps, 1),
            "attn_md_calls_per_step": len(md) / max(steps, 1),
            "attn_md_cpu_ms_per_step": 1e3 * sum(md) / max(steps, 1),
            "runner": sorted(k.removeprefix("runner:") for k in timers if k.startswith("runner:")),
        }
        if delta.get("num_drafts"):
            phase["mean_accept_len"] = (
                1 + delta.get("num_accepted_tokens", 0.0) / delta["num_drafts"]
            )
        phases.append(phase)
        print(f"[{args.arm}] {json.dumps(phase)}", flush=True)

    stats["phases"] = phases
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
