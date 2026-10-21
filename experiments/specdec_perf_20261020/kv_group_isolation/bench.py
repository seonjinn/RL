"""Measure the per-step cost of vLLM KV cache grouping for Q35 SpecDec drafters.

One arm per process, one GPU each. The engine runs in-process so the patches
below reach the scheduler and the model runner, as in NeMo-RL's TP=1 workers.
The group_size arms replace vLLM's group-size heuristic and the layout arms
pick the physical KV cache layout. Both change only where KV blocks live, not
the math. The gdn_reuse arms load the vLLM 0.29 backport of upstream GDN
metadata reuse; the greedy run checks that it leaves the tokens unchanged.
The *_verify arms also rebuild every reused GDN group with build() and compare
it with update_block_table(). "arm@N" runs the same arm again under another name,
which measures the token agreement between two processes.
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
CONFIGS = Path(__file__).resolve().parent.parent / "configs"
GDN_REUSE = Path(__file__).resolve().parent.parent / "vllm_gdn_metadata_reuse"
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
    # Block-outermost layouts route vLLM to its packed grouping planner
    "baseline_blhnc": {"layout": "BLHNC"},
    "dflash_k3_blhnc": {"spec": ("dflash", 3), "layout": "BLHNC"},
    "dspark_k4_blhnc": {"spec": ("dspark", 4), "layout": "BLHNC"},
    "dflash_k7": {"spec": ("dflash", 7)},
    "dflash_k7_blhnc": {"spec": ("dflash", 7), "layout": "BLHNC"},
    "baseline_reuse": {"gdn_reuse": True},
    "dflash_k3_reuse": {"spec": ("dflash", 3), "gdn_reuse": True},
    "dflash_k7_reuse": {"spec": ("dflash", 7), "gdn_reuse": True},
    "dspark_k4_reuse": {"spec": ("dspark", 4), "gdn_reuse": True},
}
ARMS |= {f"{name}_verify": {**arm, "verify": True} for name, arm in ARMS.items() if arm.get("gdn_reuse")}


def capture_sizes(k: int) -> list[int]:
    """CUDA graph sizes of the GRPO spec arm with the same K."""
    import yaml

    cfg = yaml.safe_load((CONFIGS / f"qwen3_30ba3b/capture_k{k}_s128.yaml").read_text())
    return cfg["policy"]["generation"]["vllm_kwargs"]["compilation_config"][
        "cudagraph_capture_sizes"
    ]


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

    # The dispatcher covers both the uniform-page-size and the packed planner
    original = kcu.get_kv_cache_groups

    def wrapped(vllm_config, kv_cache_spec):
        groups = original(vllm_config, kv_cache_spec)
        stats["kv_cache_groups"] = len(groups)
        stats["layers_per_group"] = sorted({len(g.layer_names) for g in groups})
        stats["group_spec_types"] = sorted(
            {type(g.kv_cache_spec).__name__ for g in groups}
        )
        return groups

    kcu.get_kv_cache_groups = wrapped


def verify_gdn_reuse(stats: dict[str, Any]) -> None:
    """Compare update_block_table() with a full build() for every reused GDN group.

    The reference build runs on the reusing group's own builder, whose buffers the
    reused metadata never reads. It is cloned before update_block_table() rewrites
    the state-index buffers that both share.
    """
    import dataclasses

    import torch
    from vllm.v1.attention.backends import gdn_attn

    cls = gdn_attn.GDNAttentionMetadataBuilder
    build, update = cls.build, cls.update_block_table
    signature = inspect.signature(build)
    last: dict[str, Any] = {}
    record: dict[str, Any] = {
        "checks": 0,
        "skipped_capturing": 0,
        "mismatches": defaultdict(int),
        "examples": [],
    }
    stats["gdn_reuse_verify"] = record

    def fields(metadata) -> dict[str, Any]:
        return {f.name: getattr(metadata, f.name) for f in dataclasses.fields(metadata)}

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {k: clone(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(clone(v) for v in value)
        return value

    def same(a, b) -> bool:
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return (
                isinstance(a, torch.Tensor)
                and isinstance(b, torch.Tensor)
                and (a.shape, a.dtype, a.device) == (b.shape, b.dtype, b.device)
                and torch.equal(a, b)
            )
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b

    def brief(value) -> Any:
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "head": value.flatten()[:16].tolist()}
        return repr(value)[:200]

    def recording_build(self, *args, **kwargs):
        last["args"] = signature.bind(self, *args, **kwargs).arguments
        return build(self, *args, **kwargs)

    def verified_update(self, metadata, blk_table, slot_mapping):
        # Comparing needs a host sync, which a graph capture forbids
        if torch.cuda.is_current_stream_capturing():
            record["skipped_capturing"] += 1
            return update(self, metadata, blk_table, slot_mapping)
        args = dict(last["args"], self=self)
        args["common_attn_metadata"] = dataclasses.replace(
            args["common_attn_metadata"],
            block_table_tensor=blk_table,
            slot_mapping=slot_mapping,
        )
        expected = clone(fields(build(**args)))
        out = update(self, metadata, blk_table, slot_mapping)
        record["checks"] += 1
        for name, value in fields(out).items():
            if same(expected[name], value):
                continue
            record["mismatches"][name] += 1
            if len(record["examples"]) < 8:
                record["examples"].append(
                    {
                        "field": name,
                        "num_prefills": out.num_prefills,
                        "num_decodes": out.num_decodes,
                        "num_spec_decodes": out.num_spec_decodes,
                        "expected": brief(expected[name]),
                        "actual": brief(value),
                    }
                )
        return out

    cls.build = recording_build
    cls.update_block_table = verified_update


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
    for module_name in (
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu_model_runner",
    ):
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


def top2_logprobs(completion) -> list[list[float | None]]:
    """Per position: the chosen token's logprob and the best other token's."""
    out = []
    for token, candidates in zip(completion.token_ids, completion.logprobs):
        others = [lp.logprob for t, lp in candidates.items() if t != token]
        out.append([candidates[token].logprob, max(others, default=None)])
    return out


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
    parser.add_argument("--arm", required=True, help="an ARMS name, optionally with @<run>")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--greedy-only", action="store_true", help="skip the throughput phases")
    args = parser.parse_args()
    arm_name = args.arm.split("@")[0]
    if arm_name not in ARMS:
        parser.error(f"unknown arm {arm_name!r}; choose from {sorted(ARMS)}")
    arm = ARMS[arm_name]

    stats: dict[str, Any] = {"arm": args.arm}
    if arm.get("gdn_reuse"):
        sys.path.insert(0, str(GDN_REUSE))
        import gdn_metadata_reuse

        assert gdn_metadata_reuse.install(), "this vLLM needs no GDN reuse backport"
    if arm.get("verify"):
        verify_gdn_reuse(stats)
    if "layout" in arm:
        os.environ["VLLM_KV_CACHE_LAYOUT"] = arm["layout"]
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
        kwargs["compilation_config"] = {"cudagraph_capture_sizes": capture_sizes(k)}

    t = time.perf_counter()
    llm = LLM(**kwargs)
    stats["init_s"] = time.perf_counter() - t
    # Read back what the engine resolved; the vLLM log has the same numbers
    cache_config = getattr(
        getattr(llm.llm_engine, "vllm_config", None), "cache_config", None
    )
    stats["kv_cache_layout"] = getattr(cache_config, "kv_cache_layout", None)
    stats["num_gpu_blocks"] = getattr(cache_config, "num_gpu_blocks", None)
    if arm.get("gdn_reuse"):
        stats["gdn_reuse_patched"] = list(gdn_metadata_reuse.PATCHED)
        assert set(stats["gdn_reuse_patched"]) == set(gdn_metadata_reuse.PATCHES), (
            f"backport patched only {stats['gdn_reuse_patched']}"
        )
    print(
        f"[{args.arm}] kv groups: {stats.get('kv_cache_groups')} {stats.get('layers_per_group')}"
        f" layout {stats['kv_cache_layout']} blocks {stats['num_gpu_blocks']}",
        flush=True,
    )

    prompts = load_prompts(max(args.batches))
    warm = SamplingParams(temperature=1.0, max_tokens=128, ignore_eos=True)
    llm.generate(prompts[:32], warm, use_tqdm=False)

    # Two greedy runs: their agreement is the noise floor for cross-arm token checks.
    # The top-2 logprobs tell a near-tie flip from a wrong computation where runs split.
    greedy = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True, logprobs=2)
    runs = [llm.generate(prompts[:16], greedy, use_tqdm=False) for _ in range(2)]
    stats["greedy_token_ids"] = [[list(o.outputs[0].token_ids) for o in outs] for outs in runs]
    stats["greedy_logprobs"] = [[top2_logprobs(o.outputs[0]) for o in outs] for outs in runs]

    timers: dict[str, list[float]] = defaultdict(list)
    time_calls(timers)
    phases = []
    for n in [] if args.greedy_only else args.batches:
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
            "runner": sorted(
                k.removeprefix("runner:") for k in timers if k.startswith("runner:")
            ),
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
