"""Benchmark vLLM DynamicSD under a synchronous RL rollout scenario.

Two modes share one engine build:

- profile: fixed-size batches (ignore_eos, fixed OSL) swept over --batch-sizes.
  Run once per K (via --speculative-config) to build the batch-size -> optimal-K
  lookup table that vLLM 0.24 DynamicSD consumes
  (speculative_config.num_speculative_tokens_per_batch_size).
- rollout: GRPO-style synchronous rollout. Each step submits
  num_prompts_per_step x num_generations_per_prompt sequences at once and waits
  for all of them (barrier), so the in-flight batch drains from N*G down to the
  long tail. DynamicSD should raise K as the batch drains.

Sampling defaults to temperature=1.0 to match the GRPO rollout regime.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def load_json_arg(value: str) -> dict[str, Any]:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def extract_prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("messages", "prompt", "conversation"):
        messages = row.get(key)
        if not isinstance(messages, list):
            continue
        prompt_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", ""))
            if role == "assistant":
                break
            content = message.get("content")
            if role and isinstance(content, str):
                prompt_messages.append({"role": role, "content": content})
        if prompt_messages:
            return prompt_messages

    for key in ("prompt", "question", "problem", "input", "data"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [{"role": "user", "content": value}]

    raise ValueError(f"could not extract prompt from row keys={sorted(row)}")


def normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    elif hasattr(value, "input_ids"):
        value = getattr(value, "input_ids")

    if value is None:
        raise ValueError("tokenizer did not return input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()

    ids = list(value)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(token_id) for token_id in ids]


def tokenize_prompt(tokenizer: Any, row: dict[str, Any], token_limit: int) -> list[int]:
    messages = extract_prompt_messages(row)
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    except Exception:
        text = "\n".join(message["content"] for message in messages)
        ids = tokenizer.encode(text, add_special_tokens=True)

    ids = normalize_token_ids(ids)
    if token_limit > 0 and len(ids) > token_limit:
        ids = ids[-token_limit:]
    if not ids:
        raise ValueError("tokenized prompt is empty")
    return ids


def load_prompt_pool(
    tokenizer: Any,
    prompt_jsonl: str | None,
    count: int,
    token_limit: int,
    offset: int,
) -> list[list[int]]:
    if not prompt_jsonl:
        return [list(range(min(token_limit, 1024))) for _ in range(count)]

    rows: list[dict[str, Any]] = []
    with Path(prompt_jsonl).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < offset or not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= count:
                break
    if not rows:
        raise ValueError(f"no prompts loaded from {prompt_jsonl}")
    pool = [tokenize_prompt(tokenizer, row, token_limit) for row in rows]
    while len(pool) < count:
        pool.append(pool[len(pool) % len(rows)])
    return pool


def get_vllm_metrics_snapshot(llm: Any) -> list[Any]:
    try:
        metrics = llm.get_metrics()
        if metrics is not None:
            return list(metrics)
    except Exception:
        pass
    return []


def read_spec_decode_metrics(llm: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "num_accepted_tokens_per_pos": [],
    }
    available = False
    for metric in get_vllm_metrics_snapshot(llm):
        name = getattr(metric, "name", "")
        if name == "vllm:spec_decode_num_drafts":
            metrics["num_drafts"] += int(getattr(metric, "value", 0))
            available = True
        elif name == "vllm:spec_decode_num_draft_tokens":
            metrics["num_draft_tokens"] += int(getattr(metric, "value", 0))
            available = True
        elif name == "vllm:spec_decode_num_accepted_tokens":
            metrics["num_accepted_tokens"] += int(getattr(metric, "value", 0))
            available = True
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values = list(getattr(metric, "values", []) or [])
            per_pos = metrics["num_accepted_tokens_per_pos"]
            if len(per_pos) < len(values):
                per_pos.extend([0] * (len(values) - len(per_pos)))
            for idx, value in enumerate(values):
                per_pos[idx] += int(value)
            available = True
    metrics["metrics_available"] = available
    return metrics


def diff_spec_decode_metrics(
    current: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    if not current.get("metrics_available"):
        return {}
    diff: dict[str, Any] = {
        "num_drafts": max(0, current["num_drafts"] - baseline.get("num_drafts", 0)),
        "num_draft_tokens": max(
            0, current["num_draft_tokens"] - baseline.get("num_draft_tokens", 0)
        ),
        "num_accepted_tokens": max(
            0, current["num_accepted_tokens"] - baseline.get("num_accepted_tokens", 0)
        ),
    }
    cur_pos = list(current.get("num_accepted_tokens_per_pos", []))
    base_pos = list(baseline.get("num_accepted_tokens_per_pos", []))
    per_pos = []
    for idx in range(max(len(cur_pos), len(base_pos))):
        cur = cur_pos[idx] if idx < len(cur_pos) else 0
        base = base_pos[idx] if idx < len(base_pos) else 0
        per_pos.append(max(0, cur - base))
    diff["num_accepted_tokens_per_pos"] = per_pos

    if diff["num_draft_tokens"] > 0:
        diff["acceptance_rate"] = diff["num_accepted_tokens"] / diff["num_draft_tokens"]
    if diff["num_drafts"] > 0:
        diff["mean_acceptance_length"] = (
            1.0 + diff["num_accepted_tokens"] / diff["num_drafts"]
        )
        diff["acceptance_rate_per_pos"] = [
            accepted / diff["num_drafts"] for accepted in per_pos
        ]
    return diff


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))
    return ordered[idx]


def summarize_lengths(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {}
    return {
        "count": len(lengths),
        "mean": statistics.fmean(lengths),
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "min": min(lengths),
        "total": sum(lengths),
    }


def extract_request_timing(
    outputs: list[Any], monotonic_anchor: float
) -> list[dict[str, Any]]:
    """Per-request finish times relative to batch start, for the drain curve.

    vLLM 0.24 v1 RequestOutput.metrics is a RequestStateStats whose
    first_token_ts / last_token_ts are absolute monotonic-clock timestamps, so
    the anchor must come from time.monotonic() taken just before generate().
    Rows without metrics are kept with token counts only.
    """
    rows: list[dict[str, Any]] = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        first_token = getattr(metrics, "first_token_ts", None) if metrics else None
        last_token = getattr(metrics, "last_token_ts", None) if metrics else None
        row: dict[str, Any] = {
            "request_id": str(getattr(output, "request_id", "")),
            "output_tokens": sum(len(o.token_ids) for o in output.outputs),
        }
        if first_token is not None:
            row["first_token_s"] = first_token - monotonic_anchor
        if last_token is not None:
            row["finished_s"] = last_token - monotonic_anchor
        rows.append(row)
    return rows


def build_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "disable_log_stats": False,
    }
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    kernel_config: dict[str, Any] = {}
    if args.moe_backend:
        kernel_config["moe_backend"] = args.moe_backend
    if args.disable_flashinfer_autotune:
        kernel_config["enable_flashinfer_autotune"] = False
    if kernel_config:
        kwargs["kernel_config"] = kernel_config
    if args.cudagraph_capture_sizes:
        kwargs["compilation_config"] = {
            "cudagraph_capture_sizes": sorted(set(args.cudagraph_capture_sizes))
        }
    if args.speculative_config:
        kwargs["speculative_config"] = load_json_arg(args.speculative_config)
    return LLM(**kwargs)


def make_flusher(args: argparse.Namespace, results: list[dict[str, Any]]):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list, type(None)))
    }
    if args.speculative_config:
        config["speculative_config_resolved"] = load_json_arg(args.speculative_config)

    def flush(partial: bool) -> None:
        payload = {"config": config, "partial": partial, "results": results}
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    return flush


def run_profile(args: argparse.Namespace, llm: Any) -> None:
    from vllm import SamplingParams, TokensPrompt

    tokenizer = llm.get_tokenizer()
    pool_size = max(args.batch_sizes) * args.repeats
    pool = load_prompt_pool(
        tokenizer, args.prompt_jsonl, pool_size, args.isl_cap, args.prompt_offset
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.osl,
        min_tokens=args.osl,
        ignore_eos=True,
        detokenize=False,
    )

    results: list[dict[str, Any]] = []
    flush = make_flusher(args, results)

    warmup_prompts = [TokensPrompt(prompt_token_ids=pool[0])]
    llm.generate(warmup_prompts, sampling, use_tqdm=False)

    for batch_size in args.batch_sizes:
        wall_times: list[float] = []
        spec_totals: dict[str, Any] = {}
        cursor = 0
        for _ in range(args.repeats):
            batch = [
                TokensPrompt(prompt_token_ids=pool[(cursor + i) % len(pool)])
                for i in range(batch_size)
            ]
            cursor += batch_size
            before = read_spec_decode_metrics(llm)
            t_start = time.perf_counter()
            llm.generate(batch, sampling, use_tqdm=False)
            wall = time.perf_counter() - t_start
            after = read_spec_decode_metrics(llm)
            wall_times.append(wall)
            spec_diff = diff_spec_decode_metrics(after, before)
            for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
                spec_totals[key] = spec_totals.get(key, 0) + spec_diff.get(key, 0)

        mean_wall = statistics.fmean(wall_times)
        total_output_tokens = batch_size * args.osl
        row: dict[str, Any] = {
            "mode": "profile",
            "batch_size": batch_size,
            "repeats": args.repeats,
            "wall_times_s": wall_times,
            "mean_wall_s": mean_wall,
            "output_tok_s": total_output_tokens / mean_wall,
            # includes prefill (full generate() wall / OSL); apples-to-apples
            # across K at a fixed BS, but not decode-only ITL
            "wall_ms_per_output_token": mean_wall * 1000.0 / args.osl,
        }
        if spec_totals.get("num_drafts", 0) > 0:
            row["mean_acceptance_length"] = 1.0 + (
                spec_totals["num_accepted_tokens"] / spec_totals["num_drafts"]
            )
            row["acceptance_rate"] = (
                spec_totals["num_accepted_tokens"] / spec_totals["num_draft_tokens"]
            )
        row["spec_totals"] = spec_totals
        results.append(row)
        flush(partial=True)
        print(
            f"[profile] bs={batch_size} wall={mean_wall:.2f}s "
            f"tok/s={row['output_tok_s']:.1f} "
            f"AL={row.get('mean_acceptance_length', 0.0):.3f}",
            flush=True,
        )
    flush(partial=False)


def run_rollout(args: argparse.Namespace, llm: Any) -> None:
    from vllm import SamplingParams, TokensPrompt

    tokenizer = llm.get_tokenizer()
    pool = load_prompt_pool(
        tokenizer,
        args.prompt_jsonl,
        args.num_prompts_per_step * args.num_steps,
        args.isl_cap,
        args.prompt_offset,
    )

    results: list[dict[str, Any]] = []
    flush = make_flusher(args, results)

    warmup_sampling = SamplingParams(
        temperature=args.temperature, max_tokens=64, ignore_eos=True, detokenize=False
    )
    llm.generate(
        [TokensPrompt(prompt_token_ids=pool[0])], warmup_sampling, use_tqdm=False
    )

    for step in range(args.num_steps):
        base = step * args.num_prompts_per_step
        # Submit G explicit copies per prompt (instead of SamplingParams(n=G))
        # so vLLM reports per-generation finish times: with n=G the parent
        # request finishes only when all G children do, which hides the drain
        # tail at parent granularity. Distribution is identical; each copy
        # gets its own seed.
        prompts = [
            TokensPrompt(prompt_token_ids=pool[base + i])
            for i in range(args.num_prompts_per_step)
            for _ in range(args.num_generations_per_prompt)
        ]
        sampling = [
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                seed=(args.seed + step * len(prompts) + idx)
                if args.per_request_seed
                else None,
                detokenize=False,
            )
            for idx in range(len(prompts))
        ]
        before = read_spec_decode_metrics(llm)
        t_start = time.perf_counter()
        monotonic_anchor = time.monotonic()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        wall = time.perf_counter() - t_start
        after = read_spec_decode_metrics(llm)

        lengths = [len(o.token_ids) for output in outputs for o in output.outputs]
        token_ids = None
        if args.save_token_ids:
            token_ids = [
                list(o.token_ids) for output in outputs for o in output.outputs
            ]
        total_tokens = sum(lengths)
        row: dict[str, Any] = {
            "mode": "rollout",
            "step": step,
            "num_prompts": args.num_prompts_per_step,
            "num_generations_per_prompt": args.num_generations_per_prompt,
            "num_sequences": len(lengths),
            "wall_s": wall,
            "output_tok_s": total_tokens / wall if wall > 0 else 0.0,
            "output_lengths": summarize_lengths(lengths),
            "spec_decode": diff_spec_decode_metrics(after, before),
            "request_timing": extract_request_timing(outputs, monotonic_anchor),
        }
        if token_ids is not None:
            row["token_ids"] = token_ids
        results.append(row)
        flush(partial=True)
        spec = row["spec_decode"]
        print(
            f"[rollout] step={step} seqs={len(lengths)} wall={wall:.1f}s "
            f"tok/s={row['output_tok_s']:.1f} "
            f"len_p50={row['output_lengths'].get('p50', 0)} "
            f"len_max={row['output_lengths'].get('max', 0)} "
            f"AL={spec.get('mean_acceptance_length', 0.0):.3f}",
            flush=True,
        )
    flush(partial=False)


def load_trajectories(path: str, max_trajs: int, max_turns: int) -> list[list[dict]]:
    trajs = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            msgs = json.loads(line).get("messages", [])
            if sum(1 for m in msgs if m.get("role") == "assistant") >= 2:
                trajs.append(msgs[: 2 * max_turns + 1])
            if len(trajs) >= max_trajs:
                break
    if not trajs:
        raise ValueError(f"no multi-turn trajectories in {path}")
    return trajs


def run_replay(args: argparse.Namespace, llm: Any) -> None:
    """Teacher-forced multi-turn replay: for each recorded assistant turn,
    generate with the recorded prior conversation as prompt; the recorded
    (not generated) turn feeds the next prefix. All variants see identical
    prefix sequences; growing shared prefixes exercise prefix caching and
    copy-span density like an agentic rollout."""
    from vllm import SamplingParams, TokensPrompt

    tokenizer = llm.get_tokenizer()
    trajs = load_trajectories(
        args.replay_jsonl, args.replay_trajectories, args.replay_max_turns
    )
    results: list[dict[str, Any]] = []
    flush = make_flusher(args, results)

    for ti, msgs in enumerate(trajs):
        turn_idx = 0
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            prefix = msgs[:i]
            try:
                ids = tokenizer.apply_chat_template(
                    prefix, tokenize=True, add_generation_prompt=True
                )
            except Exception:
                continue
            ids = normalize_token_ids(ids)
            if len(ids) > args.max_model_len - args.replay_turn_max_tokens:
                break
            ref_len = len(
                tokenizer.encode(m.get("content") or "", add_special_tokens=False)
            )
            sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=min(max(ref_len, 16), args.replay_turn_max_tokens),
                seed=args.seed + ti * 1000 + turn_idx
                if args.per_request_seed
                else None,
                detokenize=False,
            )
            before = read_spec_decode_metrics(llm)
            t0 = time.perf_counter()
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=ids)] * args.replay_copies,
                sampling,
                use_tqdm=False,
            )
            wall = time.perf_counter() - t0
            after = read_spec_decode_metrics(llm)
            gen_tokens = sum(len(o.token_ids) for out in outs for o in out.outputs)
            results.append(
                {
                    "mode": "replay",
                    "trajectory": ti,
                    "turn": turn_idx,
                    "prefix_tokens": len(ids),
                    "ref_turn_tokens": ref_len,
                    "wall_s": wall,
                    "gen_tokens": gen_tokens,
                    "output_tok_s": gen_tokens / wall if wall > 0 else 0.0,
                    "spec_decode": diff_spec_decode_metrics(after, before),
                }
            )
            turn_idx += 1
        flush(partial=True)
        print(f"[replay] traj={ti} turns={turn_idx}", flush=True)
    flush(partial=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("profile", "rollout", "replay"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--speculative-config")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--disable-flashinfer-autotune", action="store_true")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--moe-backend", default=None)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--cudagraph-capture-sizes", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prompt-jsonl")
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--isl-cap", type=int, default=4096)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)

    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=2)

    parser.add_argument("--num-prompts-per-step", type=int, default=16)
    parser.add_argument("--num-generations-per-prompt", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--per-request-seed", action="store_true")
    parser.add_argument("--replay-jsonl")
    parser.add_argument("--replay-trajectories", type=int, default=8)
    parser.add_argument("--replay-max-turns", type=int, default=20)
    parser.add_argument("--replay-copies", type=int, default=8)
    parser.add_argument("--replay-turn-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--save-token-ids",
        action="store_true",
        help="store generated token ids per output (parity checks; small runs only)",
    )

    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    llm = build_llm(args)
    if args.mode == "profile":
        run_profile(args, llm)
    elif args.mode == "replay":
        run_replay(args, llm)
    else:
        run_rollout(args, llm)


if __name__ == "__main__":
    main()
