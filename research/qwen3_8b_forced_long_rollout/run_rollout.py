# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Run pinned Qwen3-8B forced-long offline rollouts with or without DFlash."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, TypedDict


MODEL_CONTEXT_TOKENS = 40960
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
SAMPLING_TOP_K = -1


class ManifestRecord(TypedDict):
    logical_step: int
    source_id: str
    prompt_sha256: str


def validate_runtime_versions(
    versions: dict[str, str], *, expected_vllm: str
) -> dict[str, str]:
    actual_vllm = versions.get("vllm")
    if actual_vllm != expected_vllm:
        raise RuntimeError(
            f"vLLM runtime mismatch: expected {expected_vllm}, got {actual_vllm}"
        )
    return versions


def load_manifest(
    path: Path, *, expected_sha256: str, count: int
) -> list[ManifestRecord]:
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"manifest SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    document = json.loads(payload)
    records = document.get("records") if isinstance(document, dict) else None
    if not isinstance(records, list) or len(records) != count:
        raise ValueError(f"manifest must contain exactly {count} records")
    expected_steps = list(range(1, count + 1))
    if [record.get("logical_step") for record in records] != expected_steps:
        raise ValueError("manifest logical steps must be contiguous and ordered")
    source_ids = [record.get("source_id") for record in records]
    if not all(isinstance(source_id, str) and source_id for source_id in source_ids):
        raise ValueError("manifest source IDs must be non-empty strings")
    if len(set(source_ids)) != count:
        raise ValueError("manifest source IDs must be unique")
    hashes = [record.get("prompt_sha256") for record in records]
    if not all(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for digest in hashes
    ):
        raise ValueError("manifest prompt SHA256 values must be lowercase hex")
    return records


def sampling_kwargs(*, min_tokens: int, max_tokens: int) -> dict[str, Any]:
    if not 0 <= min_tokens <= max_tokens:
        raise ValueError("min_tokens must satisfy 0 <= min_tokens <= max_tokens")
    return {
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": SAMPLING_TOP_P,
        "top_k": SAMPLING_TOP_K,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "ignore_eos": False,
    }


def validate_prompt_lengths(
    lengths: list[int], *, max_input_tokens: int, max_output_tokens: int
) -> None:
    if not lengths:
        raise ValueError("prompt batch must not be empty")
    longest = max(lengths)
    if longest > max_input_tokens:
        raise ValueError(
            f"prompt length {longest} exceeds configured input limit {max_input_tokens}"
        )
    if longest + max_output_tokens > MODEL_CONTEXT_TOKENS:
        raise ValueError(
            f"prompt length {longest} plus output {max_output_tokens} exceeds "
            f"model context {MODEL_CONTEXT_TOKENS}"
        )


def required_decode_capture_sizes(
    *, max_num_seqs: int, speculative_tokens: tuple[int, ...]
) -> list[int]:
    sizes = {1, 2, 4}
    for speculative_horizon in speculative_tokens:
        width = speculative_horizon + 1
        sizes.update(batch_size * width for batch_size in range(1, max_num_seqs + 1))
    return sorted(sizes)


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    engine = config["engine"]
    required_capture_sizes = required_decode_capture_sizes(
        max_num_seqs=engine["max_num_seqs"], speculative_tokens=(5, 7)
    )
    configured_capture_sizes = engine["compilation_config"]["cudagraph_capture_sizes"]
    if configured_capture_sizes != required_capture_sizes:
        raise ValueError(
            "CUDA Graph capture sizes must exactly cover the shared K5/K7 decode shapes"
        )
    return config


def _prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("DAPOMath row is missing prompt messages")
    content = prompt[0].get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("DAPOMath prompt content is empty")
    return content


def _load_prompts(
    config: dict[str, Any], records: list[ManifestRecord], tokenizer: Any
) -> tuple[list[str], list[int]]:
    from datasets import load_dataset

    dataset_config = config["dataset"]
    dataset = load_dataset(
        dataset_config["repo_id"],
        revision=dataset_config["revision"],
        split=dataset_config["split"],
    ).select([record["logical_step"] - 1 for record in records])
    prompts = []
    for row, record in zip(dataset, records, strict=True):
        prompt_text = _prompt_text(row)
        source_id = row.get("extra_info", {}).get("index")
        if source_id != record["source_id"]:
            raise ValueError(
                f"manifest source ID mismatch at step {record['logical_step']}"
            )
        prompt_sha256 = hashlib.sha256(prompt_text.encode()).hexdigest()
        if prompt_sha256 != record["prompt_sha256"]:
            raise ValueError(
                f"manifest prompt SHA256 mismatch at step {record['logical_step']}"
            )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )
    lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    validate_prompt_lengths(
        lengths,
        max_input_tokens=config["generation"]["max_input_tokens"],
        max_output_tokens=config["generation"]["max_tokens"],
    )
    return prompts, lengths


def _raw_spec_counters(llm: Any) -> dict[str | tuple[str, int], float]:
    counters: dict[str | tuple[str, int], float] = {}
    get_metrics = getattr(llm, "get_metrics", None)
    if get_metrics is None:
        return counters
    for metric in get_metrics():
        name = getattr(metric, "name", "")
        if "spec_decode" not in name:
            continue
        values = getattr(metric, "values", None)
        if isinstance(values, list):
            for position, value in enumerate(values, start=1):
                counters[name, position] = float(value)
        elif hasattr(metric, "value"):
            counters[name] = float(metric.value)
    return counters


def _spec_delta(
    before: dict[str | tuple[str, int], float],
    after: dict[str | tuple[str, int], float],
) -> dict[str, float]:
    delta = {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in set(before) | set(after)
    }
    drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    metrics = {
        "spec/num_drafts": drafts,
        "spec/num_draft_tokens": draft_tokens,
        "spec/num_accepted_tokens": accepted,
        "spec/acceptance_length": 1.0 + accepted / drafts if drafts else 1.0,
        "spec/acceptance_rate": accepted / draft_tokens if draft_tokens else 0.0,
    }
    for key, value in delta.items():
        if isinstance(key, tuple) and drafts:
            _, position = key
            metrics[f"spec/acceptance_rate_pos{position}"] = value / drafts
    return metrics


def _engine(config: dict[str, Any], arm: str) -> Any:
    from vllm import LLM

    model = config["model"]
    engine = config["engine"]
    target_model = os.environ.get("NRL_TARGET_MODEL", model["target_repo_id"])
    kwargs: dict[str, Any] = {
        "model": target_model,
        "revision": model["target_revision"],
        "tokenizer_revision": model["target_revision"],
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": MODEL_CONTEXT_TOKENS,
        "gpu_memory_utilization": engine["gpu_memory_utilization"],
        "max_num_seqs": engine["max_num_seqs"],
        "max_num_batched_tokens": engine["max_num_batched_tokens"],
        "enforce_eager": False,
        "seed": config["seed"],
        "trust_remote_code": True,
        "compilation_config": engine["compilation_config"],
    }
    if arm == "dflash_k5":
        draft_model = os.environ.get("NRL_DRAFT_MODEL", model["draft_repo_id"])
        kwargs["speculative_config"] = {
            "method": "dflash",
            "model": draft_model,
            "revision": model["draft_revision"],
            "num_speculative_tokens": 5,
            "draft_tensor_parallel_size": 1,
        }
    elif arm != "baseline":
        raise ValueError(f"unknown arm: {arm}")
    return LLM(**kwargs)


def _wandb_run(
    config: dict[str, Any], arm: str, start: int, end: int, rank: int
) -> Any:
    import wandb

    wandb_config = config["wandb"]
    return wandb.init(
        project=wandb_config["project"],
        entity=os.environ.get("WANDB_ENTITY"),
        name=f"{wandb_config['name_prefix']}-{arm}-r{rank}-{start}-{end}",
        group=wandb_config["group"],
        job_type=f"{arm}-rank-{rank}",
        tags=[*wandb_config["tags"], arm, f"rank-{rank}"],
        config={
            "arm": arm,
            "range": [start, end],
            "rank": rank,
            "target_revision": config["model"]["target_revision"],
            "draft_revision": config["model"]["draft_revision"],
            "dataset_revision": config["dataset"]["revision"],
            "manifest_sha256": config["dataset"]["manifest_sha256"],
            "min_tokens": config["generation"]["min_tokens"],
            "max_tokens": config["generation"]["max_tokens"],
            "temperature": SAMPLING_TEMPERATURE,
            "top_p": SAMPLING_TOP_P,
            "top_k": SAMPLING_TOP_K,
            "max_model_len": MODEL_CONTEXT_TOKENS,
            "thinking": True,
        },
    )


def _write_record(stream: Any, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def _run_batches(
    *,
    llm: Any,
    prompts: list[str],
    prompt_lengths: list[int],
    logical_ids: list[int],
    dataset_indices: list[int],
    batch_size: int,
    min_tokens: int,
    max_tokens: int,
    seed: int,
    output_path: Path,
    wandb_run: Any,
    phase: str,
) -> dict[str, float]:
    from vllm import SamplingParams
    from wandb import Table

    completed = 0
    output_lengths: list[int] = []
    started = time.perf_counter()
    with output_path.open("a") as stream:
        for offset in range(0, len(prompts), batch_size):
            batch_prompts = prompts[offset : offset + batch_size]
            batch_ids = logical_ids[offset : offset + batch_size]
            batch_indices = dataset_indices[offset : offset + batch_size]
            batch_prompt_lengths = prompt_lengths[offset : offset + batch_size]
            params = [
                SamplingParams(
                    **sampling_kwargs(min_tokens=min_tokens, max_tokens=max_tokens),
                    seed=seed + logical_id,
                )
                for logical_id in batch_ids
            ]
            counters_before = _raw_spec_counters(llm)
            batch_started = time.perf_counter()
            outputs = llm.generate(batch_prompts, params, use_tqdm=False)
            batch_elapsed = time.perf_counter() - batch_started
            counters_after = _raw_spec_counters(llm)

            table_rows: list[list[Any]] = []
            batch_output_tokens = 0
            for logical_id, dataset_index, prompt_length, output in zip(
                batch_ids,
                batch_indices,
                batch_prompt_lengths,
                outputs,
                strict=True,
            ):
                completion = output.outputs[0]
                token_ids = list(completion.token_ids)
                output_length = len(token_ids)
                if min_tokens and not min_tokens <= output_length <= max_tokens:
                    raise RuntimeError(
                        f"forced-long response {logical_id} has {output_length} tokens"
                    )
                batch_output_tokens += output_length
                output_lengths.append(output_length)
                record = {
                    "logical_response_id": logical_id,
                    "dataset_index": dataset_index,
                    "prompt_tokens": prompt_length,
                    "output_tokens": output_length,
                    "finish_reason": completion.finish_reason,
                    "output_token_sha256": hashlib.sha256(
                        json.dumps(token_ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "text": completion.text,
                }
                _write_record(stream, record)
                table_rows.append(
                    [logical_id, dataset_index, prompt_length, output_length]
                )

            completed += len(outputs)
            metrics = {
                f"{phase}/completed_responses": completed,
                f"{phase}/logical_response_id_max": max(batch_ids),
                f"{phase}/batch_generation_s": batch_elapsed,
                f"{phase}/batch_output_tokens": batch_output_tokens,
                f"{phase}/output_tok_per_s_per_gpu": batch_output_tokens
                / batch_elapsed,
                f"{phase}/output_tokens_mean": statistics.fmean(
                    output_lengths[-len(outputs) :]
                ),
                f"{phase}/output_tokens_min": min(output_lengths[-len(outputs) :]),
                f"{phase}/output_tokens_max": max(output_lengths[-len(outputs) :]),
                **_spec_delta(counters_before, counters_after),
                f"{phase}/responses": Table(
                    columns=[
                        "logical_response_id",
                        "dataset_index",
                        "prompt_tokens",
                        "output_tokens",
                    ],
                    data=table_rows,
                ),
            }
            wandb_run.log(
                metrics,
                step=0 if phase == "natural_probe" else max(batch_ids),
            )

    elapsed = time.perf_counter() - started
    return {
        "responses": float(completed),
        "elapsed_s": elapsed,
        "output_tokens": float(sum(output_lengths)),
        "output_tok_per_s_per_gpu": sum(output_lengths) / elapsed,
        "output_tokens_min": float(min(output_lengths)),
        "output_tokens_p50": float(statistics.median(output_lengths)),
        "output_tokens_max": float(max(output_lengths)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "dflash_k5"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--natural-probe", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    runtime_versions = validate_runtime_versions(
        {"vllm": importlib.metadata.version("vllm")},
        expected_vllm=config["runtime"]["vllm_version"],
    )
    print(f"runtime_versions={json.dumps(runtime_versions, sort_keys=True)}")

    from transformers import AutoTokenizer

    manifest = load_manifest(
        Path(config["dataset"]["manifest_path"]),
        expected_sha256=config["dataset"]["manifest_sha256"],
        count=config["dataset"]["count"],
    )
    selected_records = [
        manifest[index]
        for index in range(args.start, args.end)
        if (index - args.start) % args.world_size == args.rank
    ]
    if not selected_records:
        raise ValueError("selected response range is empty")
    logical_ids = [record["logical_step"] for record in selected_records]
    dataset_indices = [logical_id - 1 for logical_id in logical_ids]

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("NRL_TARGET_MODEL", config["model"]["target_repo_id"]),
        revision=config["model"]["target_revision"],
    )
    prompts, prompt_lengths = _load_prompts(config, selected_records, tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "config": config,
        "arm": args.arm,
        "start": args.start,
        "end": args.end,
        "rank": args.rank,
        "world_size": args.world_size,
        "logical_ids": logical_ids,
        "dataset_indices": dataset_indices,
        "prompt_token_lengths": prompt_lengths,
        "prompt_token_length_sha256": hashlib.sha256(
            json.dumps(prompt_lengths, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    llm = _engine(config, args.arm)
    run = _wandb_run(config, args.arm, args.start, args.end, args.rank)
    try:
        summaries: dict[str, dict[str, float]] = {}
        if args.natural_probe:
            summaries["natural_probe"] = _run_batches(
                llm=llm,
                prompts=prompts[: args.batch_size],
                prompt_lengths=prompt_lengths[: args.batch_size],
                logical_ids=logical_ids[: args.batch_size],
                dataset_indices=dataset_indices[: args.batch_size],
                batch_size=args.batch_size,
                min_tokens=0,
                max_tokens=config["generation"]["max_tokens"],
                seed=config["seed"],
                output_path=args.output_dir / "natural.jsonl",
                wandb_run=run,
                phase="natural_probe",
            )
        summaries["forced_long"] = _run_batches(
            llm=llm,
            prompts=prompts,
            prompt_lengths=prompt_lengths,
            logical_ids=logical_ids,
            dataset_indices=dataset_indices,
            batch_size=args.batch_size,
            min_tokens=config["generation"]["min_tokens"],
            max_tokens=config["generation"]["max_tokens"],
            seed=config["seed"],
            output_path=args.output_dir / "forced_long.jsonl",
            wandb_run=run,
            phase="forced_long",
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
