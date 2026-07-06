#!/usr/bin/env python3
"""Run SPEED-Bench official and AsyncLLM synchronous-rollout overlay cohorts."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

from benchmark import (
    DEFAULT_DYNAMIC_SCHEDULE,
    build_speculative_config,
    diff_spec_decode_counters,
    parse_dynamic_schedule,
    read_spec_decode_counters,
    runtime_metadata,
    sum_spec_decode_counters,
    write_json_atomic,
)
from benchmark_sync_rollout import model_config_hash
from speedbench_dataset import SpeedBenchRecord, select_sync_overlay_rows
from sync_rollout_core import RequestPlan, load_request_plan, resolve_request_plan


ACCEPTANCE_LIMITATION = (
    "vLLM exposes draft-position acceptance counters. Completion-position "
    "windows report contributor counts only; they are not output-position "
    "acceptance without additional instrumentation."
)

MODELOPT_PINNED_COMMIT = "43fee0cd70fa9e5f85782d52a4bd8ad9c8b88446"
MODELOPT_RUN_PY_SHA256 = (
    "1b82c76f4beba534a3b6b1545122adb9a1e81da8a7ba50c4d49a4284fc26f356"
)
MODELOPT_INSTRUMENTATION_PATCH_SHA256 = (
    "dd6d436d3b05459cf00ea49a98e1ea00fd6a9a62f56124db7a080252c189913b"
)
MODELOPT_PATCHED_RUN_PY_SHA256 = (
    "75eb4a928127333d2b305b32584d7c282b57af193c2e08036d661b07d55c779c"
)
MODELOPT_TIMING_SIDECAR = "task5_timing_total_tokens.json"
MODELOPT_RESOLVED_CONFIG_SIDECAR = "task5_resolved_vllm_config.json"
MODELOPT_INSTRUMENTATION_SIDECAR = "task5_instrumentation.json"

MODELOPT_INSTRUMENTATION_IMPORT = """import dataclasses
import json
from pathlib import Path
"""

MODELOPT_INSTRUMENTATION_HELPERS = r'''

def _task5_jsonable(value, _seen=None):
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if not isinstance(value, type) and dataclasses.is_dataclass(value):
        return _task5_jsonable(dataclasses.asdict(value), _seen)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    object_id = id(value)
    if object_id in _seen:
        return "<recursive>"
    if isinstance(value, dict):
        _seen.add(object_id)
        try:
            return {
                str(k): _task5_jsonable(v, _seen)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        finally:
            _seen.remove(object_id)
    if isinstance(value, (list, tuple)):
        _seen.add(object_id)
        try:
            return [_task5_jsonable(item, _seen) for item in value]
        finally:
            _seen.remove(object_id)
    if isinstance(value, set):
        _seen.add(object_id)
        try:
            return [_task5_jsonable(item, _seen) for item in sorted(value, key=repr)]
        finally:
            _seen.remove(object_id)
    if hasattr(value, "__dict__"):
        _seen.add(object_id)
        try:
            return {
                str(k): _task5_jsonable(v, _seen)
                for k, v in sorted(vars(value).items())
                if not str(k).startswith("_")
            }
        finally:
            _seen.remove(object_id)
    return repr(value)


def _task5_attr_path(value, path):
    current = value
    for key in path:
        if current is None:
            return None
        current = getattr(current, key, None)
    return current


def _task5_resolved_vllm_config(model):
    async_model = getattr(model, "model", None)
    for path in (
        ("vllm_config",),
        ("engine", "vllm_config"),
        ("llm_engine", "vllm_config"),
        ("engine_core", "vllm_config"),
    ):
        config = _task5_attr_path(async_model, path)
        if config is not None:
            return config
    return None


def _task5_sampling_config(model):
    config = getattr(model, "sampling_config", None)
    if config is None:
        return None
    return {
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "top_k": getattr(config, "top_k", None),
    }


def _task5_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _task5_write_resolved_vllm_config_sidecar(save_dir, model, serving_config):
    if save_dir is None:
        return
    payload = {
        "schema_version": 1,
        "serving_config": _task5_jsonable(serving_config),
        "engine_args": _task5_jsonable(getattr(model, "engine_args", None)),
        "vllm_config": _task5_jsonable(_task5_resolved_vllm_config(model)),
        "sampling_kwargs": _task5_jsonable(getattr(model, "sampling_kwargs", None)),
        "sampling_config": _task5_jsonable(_task5_sampling_config(model)),
    }
    _task5_write_json(Path(save_dir) / "task5_resolved_vllm_config.json", payload)


def _task5_write_timing_total_tokens_sidecar(save_dir, metrics_list):
    if save_dir is None:
        return
    timing_metric = None
    for metric in metrics_list:
        if getattr(metric, "name", None) == "timing" and hasattr(metric, "total_tokens"):
            timing_metric = metric
            break
    if timing_metric is None:
        raise RuntimeError("Task5 instrumentation could not find Timing metric")
    total_tokens = list(getattr(timing_metric, "total_tokens"))
    if not total_tokens or any((not isinstance(value, int)) or value <= 0 for value in total_tokens):
        raise RuntimeError("Task5 instrumentation saw invalid Timing.total_tokens")
    _task5_write_json(
        Path(save_dir) / "task5_timing_total_tokens.json",
        {
            "schema_version": 1,
            "source": "Timing.total_tokens",
            "total_tokens": total_tokens,
            "turn_count": len(total_tokens),
        },
    )
'''

MODELOPT_INSTRUMENTATION_PATCH = (
    MODELOPT_INSTRUMENTATION_IMPORT + MODELOPT_INSTRUMENTATION_HELPERS
)


@dataclass(frozen=True, slots=True)
class OverlayPrompt:
    prompt_id: str
    prompt_token_ids: list[int]
    prompt_sha256: str
    source_prompt_sha256: str | None
    category: str
    dataset_config: str
    turn_count: int
    multiturn: bool

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)


@dataclass(frozen=True, slots=True)
class OverlayRequest:
    request_id: str
    prompt_id: str
    prompt_sha256: str
    source_prompt_sha256: str | None
    category: str
    sample_index: int
    seed: int
    prompt_token_ids: list[int]
    max_tokens: int
    min_tokens: int
    ignore_eos: bool


@dataclass(frozen=True, slots=True)
class CompletedRequest:
    request: OverlayRequest
    output: Any
    output_token_ids: list[int]
    ttft_s: float
    finished_at_s: float
    completion_time_s: float
    finish_reason: str


@dataclass(frozen=True, slots=True)
class OverlaySelection:
    prompts: tuple[OverlayPrompt, ...]
    dataset_config: str
    prepared_manifest_hash: str
    parquet_hash: str
    prompt_set_hash: str


def token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def modelopt_instrumentation_jsonable(
    value: Any,
    _seen: set[int] | None = None,
) -> Any:
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if not isinstance(value, type) and is_dataclass(value):
        return modelopt_instrumentation_jsonable(asdict(cast(Any, value)), _seen)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    object_id = id(value)
    if object_id in _seen:
        return "<recursive>"
    if isinstance(value, Mapping):
        _seen.add(object_id)
        try:
            return {
                str(key): modelopt_instrumentation_jsonable(item, _seen)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        finally:
            _seen.remove(object_id)
    if isinstance(value, (list, tuple)):
        _seen.add(object_id)
        try:
            return [modelopt_instrumentation_jsonable(item, _seen) for item in value]
        finally:
            _seen.remove(object_id)
    if isinstance(value, set):
        _seen.add(object_id)
        try:
            return [
                modelopt_instrumentation_jsonable(item, _seen)
                for item in sorted(value, key=repr)
            ]
        finally:
            _seen.remove(object_id)
    if hasattr(value, "__dict__"):
        _seen.add(object_id)
        try:
            return {
                str(key): modelopt_instrumentation_jsonable(item, _seen)
                for key, item in sorted(vars(value).items())
                if not str(key).startswith("_")
            }
        finally:
            _seen.remove(object_id)
    return repr(value)


def _replace_anchor_once(text: str, anchor: str, replacement: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise ValueError(f"ModelOpt instrumentation anchor mismatch: {anchor!r}")
    return text.replace(anchor, replacement, 1)


def _instrument_modelopt_run_py_source(source: str) -> str:
    source = _replace_anchor_once(
        source,
        "import argparse\nimport asyncio\n",
        "import argparse\nimport asyncio\n" + MODELOPT_INSTRUMENTATION_IMPORT,
    )
    source = _replace_anchor_once(
        source,
        "import yaml\n",
        "import yaml\n" + MODELOPT_INSTRUMENTATION_HELPERS,
    )
    source = _replace_anchor_once(
        source,
        (
            "        dump_env(args, args.save_dir, "
            'overrides={"serving_config": model.get_serving_config()})\n'
        ),
        (
            "        task5_serving_config = model.get_serving_config()\n"
            "        dump_env(args, args.save_dir, "
            'overrides={"serving_config": task5_serving_config})\n'
            "        _task5_write_resolved_vllm_config_sidecar(\n"
            "            args.save_dir,\n"
            "            model,\n"
            "            task5_serving_config,\n"
            "        )\n"
        ),
    )
    source = _replace_anchor_once(
        source,
        "    runner.clear_metrics()\n",
        (
            "    _task5_write_timing_total_tokens_sidecar(args.save_dir, metrics_list)\n"
            "    runner.clear_metrics()\n"
        ),
    )
    return source


def stage_instrumented_modelopt_source(
    modelopt_root: Path,
    staged_root: Path,
    *,
    save_dir: Path,
) -> dict[str, Any]:
    source_run_py = modelopt_root / "examples/specdec_bench/run.py"
    if not source_run_py.is_file():
        raise ValueError(f"missing pinned ModelOpt run.py: {source_run_py}")
    source = source_run_py.read_text(encoding="utf-8")
    source_hash = sha256_text(source)
    if source_hash != MODELOPT_RUN_PY_SHA256:
        raise ValueError(
            "ModelOpt run.py source hash mismatch: "
            f"{source_hash} != {MODELOPT_RUN_PY_SHA256}"
        )
    patch_hash = sha256_text(MODELOPT_INSTRUMENTATION_PATCH)
    if patch_hash != MODELOPT_INSTRUMENTATION_PATCH_SHA256:
        raise ValueError(
            "ModelOpt Task5 instrumentation patch hash mismatch: "
            f"{patch_hash} != {MODELOPT_INSTRUMENTATION_PATCH_SHA256}"
        )
    patched_source = _instrument_modelopt_run_py_source(source)
    patched_hash = sha256_text(patched_source)
    if patched_hash != MODELOPT_PATCHED_RUN_PY_SHA256:
        raise ValueError(
            "ModelOpt instrumented run.py source hash mismatch: "
            f"{patched_hash} != {MODELOPT_PATCHED_RUN_PY_SHA256}"
        )
    if staged_root.exists():
        shutil.rmtree(staged_root)
    shutil.copytree(modelopt_root, staged_root, symlinks=True)
    staged_run_py = staged_root / "examples/specdec_bench/run.py"
    staged_run_py.write_text(patched_source, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "modelopt_commit": MODELOPT_PINNED_COMMIT,
        "source_run_py": str(source_run_py),
        "staged_run_py": str(staged_run_py),
        "source_sha256": source_hash,
        "patch_sha256": patch_hash,
        "patched_source_sha256": patched_hash,
        "timing_sidecar": MODELOPT_TIMING_SIDECAR,
        "resolved_config_sidecar": MODELOPT_RESOLVED_CONFIG_SIDECAR,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(save_dir / MODELOPT_INSTRUMENTATION_SIDECAR, metadata)
    return metadata


def prompt_set_hash(prompt_hashes: Sequence[str]) -> str:
    payload = json.dumps(list(prompt_hashes), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def collect_runtime_metadata() -> dict[str, Any]:
    try:
        return runtime_metadata()
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_available": False,
        }


def _string_field(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected non-empty string field {field_name!r}")
    return value


def _optional_string_field(row: Mapping[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected optional string field {field_name!r}")
    return value


def _bool_field(row: Mapping[str, Any], field_name: str, default: bool) -> bool:
    value = row.get(field_name, default)
    if type(value) is not bool:
        raise ValueError(f"expected boolean field {field_name!r}")
    return value


def _turn_count(row: Mapping[str, Any]) -> int:
    turns = row.get("turns")
    if turns is None:
        return 1
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise ValueError("turns must be a sequence of strings")
    return len(turns)


def _prompt_token_ids(row: Mapping[str, Any]) -> list[int]:
    for field_name in ("prompt_token_ids", "input_ids", "token_ids"):
        value = row.get(field_name)
        if value is None:
            continue
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or not value
            or any(type(token_id) is not int for token_id in value)
        ):
            raise ValueError(f"{field_name} must be a non-empty integer sequence")
        return list(value)
    raise ValueError("prepared SPEED-Bench row must include preserved token IDs")


def overlay_prompt_from_prepared_row(
    row: Mapping[str, Any],
    *,
    max_prompt_tokens: int | None = None,
) -> OverlayPrompt:
    del max_prompt_tokens
    token_ids = _prompt_token_ids(row)
    return OverlayPrompt(
        prompt_id=_string_field(row, "question_id"),
        prompt_token_ids=token_ids,
        prompt_sha256=token_hash(token_ids),
        source_prompt_sha256=_optional_string_field(row, "canonical_hash"),
        category=_string_field(row, "category"),
        dataset_config=_string_field(row, "dataset_config"),
        turn_count=_turn_count(row),
        multiturn=_bool_field(row, "multiturn", _turn_count(row) > 1),
    )


def _speedbench_messages(turns: Sequence[str]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for index, turn in enumerate(turns):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append({"role": role, "content": str(turn)})
    if messages and messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": ""})
    return messages


def tokenize_speedbench_record(tokenizer: Any, record: SpeedBenchRecord) -> list[int]:
    messages = _speedbench_messages(record.turns)
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    else:
        token_ids = tokenizer.encode("\n".join(record.turns), add_special_tokens=True)
    result = list(token_ids)
    if not result or any(type(token_id) is not int for token_id in result):
        raise ValueError("tokenizer returned invalid SPEED-Bench token IDs")
    return result


def _record_to_overlay_prompt(
    record: SpeedBenchRecord,
    token_ids: Sequence[int],
) -> OverlayPrompt:
    return overlay_prompt_from_prepared_row(
        {
            **asdict(record),
            "prompt_token_ids": list(token_ids),
        }
    )


def _manifest_config_entry(
    manifest: Mapping[str, Any],
    dataset_config: str,
) -> dict[str, Any]:
    for entry in manifest.get("prepared_configs", []):
        if entry.get("config_name") == dataset_config:
            return dict(entry)
    raise ValueError(f"prepared manifest missing config {dataset_config!r}")


def _verify_checksum_file(
    checksums_path: Path,
    *,
    relative_path: str,
    expected_sha256: str,
) -> None:
    found = False
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, path_text = line.split("  ", 1)
        if path_text == relative_path:
            found = True
            if digest != expected_sha256:
                raise ValueError(f"checksum mismatch for {relative_path}")
    if not found:
        raise ValueError(f"missing checksum entry for {relative_path}")


def read_prepared_speedbench_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # pyright: ignore[reportMissingImports]

        return [
            dict(row)
            for row in pd.read_parquet(path).to_dict(orient="records")
        ]
    except Exception as exc:
        raise ValueError(f"failed to read prepared parquet: {path}") from exc


def resolve_prepared_dataset_path(
    *,
    prepared_root: Path,
    prepared_manifest: Path,
    prepared_checksums: Path,
    dataset_config: str,
) -> Path:
    manifest = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    entry = _manifest_config_entry(manifest, dataset_config)
    relative_path = str(entry["relative_path"])
    parquet_path = prepared_root / relative_path
    actual_hash = sha256_file(parquet_path)
    if actual_hash != entry.get("sha256"):
        raise ValueError(f"prepared parquet hash mismatch for {relative_path}")
    _verify_checksum_file(
        prepared_checksums,
        relative_path=relative_path,
        expected_sha256=actual_hash,
    )
    return parquet_path


def build_overlay_from_prepared_parquet(
    *,
    prepared_root: Path,
    prepared_manifest: Path,
    prepared_checksums: Path,
    dataset_config: str,
    tokenizer: Any,
    seed: int,
) -> OverlaySelection:
    from speedbench_dataset import build_records

    parquet_path = resolve_prepared_dataset_path(
        prepared_root=prepared_root,
        prepared_manifest=prepared_manifest,
        prepared_checksums=prepared_checksums,
        dataset_config=dataset_config,
    )
    actual_hash = sha256_file(parquet_path)
    records = build_records(
        read_prepared_speedbench_rows(parquet_path),
        dataset_config=dataset_config,
    )
    selected_batches = select_sync_overlay_rows(records, seed=seed)
    prompts = tuple(
        _record_to_overlay_prompt(record, tokenize_speedbench_record(tokenizer, record))
        for batch in selected_batches
        for record in batch
    )
    if len(prompts) != 48 or len({prompt.prompt_id for prompt in prompts}) != 48:
        raise ValueError("SPEED-Bench overlay requires exactly 48 unique prompts")
    return OverlaySelection(
        prompts=prompts,
        dataset_config=dataset_config,
        prepared_manifest_hash=sha256_file(prepared_manifest),
        parquet_hash=actual_hash,
        prompt_set_hash=prompt_set_hash([prompt.prompt_sha256 for prompt in prompts]),
    )


def build_overlay_prompt_batches(
    records: Sequence[SpeedBenchRecord],
    *,
    prepared_token_ids_by_question_id: Mapping[str, Sequence[int]],
    seed: int,
) -> tuple[tuple[OverlayPrompt, ...], ...]:
    selected = select_sync_overlay_rows(records, seed=seed)
    return tuple(
        tuple(
            _record_to_overlay_prompt(
                record,
                prepared_token_ids_by_question_id[record.question_id],
            )
            for record in batch
        )
        for batch in selected
    )


def expand_overlay_barrier_batches(
    prompts: Sequence[OverlayPrompt],
    *,
    active_concurrency: int,
    rollout_batches: int,
) -> tuple[tuple[OverlayPrompt, ...], ...]:
    if len(prompts) != 48 or len({prompt.prompt_id for prompt in prompts}) != 48:
        raise ValueError("expected exactly 48 unique SPEED-Bench overlay prompts")
    if active_concurrency <= 0 or rollout_batches <= 0:
        raise ValueError("active_concurrency and rollout_batches must be positive")
    requests_per_batch = max(16, active_concurrency)
    batches: list[tuple[OverlayPrompt, ...]] = []
    for batch_index in range(rollout_batches):
        segment_index = batch_index % 3
        base = list(prompts[segment_index * 16 : (segment_index + 1) * 16])
        batch = [
            base[index % len(base)]
            for index in range(requests_per_batch)
        ]
        batches.append(tuple(batch))
    return tuple(batches)


def _prompt_by_id(prompts: Sequence[OverlayPrompt]) -> dict[str, OverlayPrompt]:
    return {prompt.prompt_id: prompt for prompt in prompts}


def prepare_overlay_requests(
    prompts: Sequence[OverlayPrompt],
    *,
    request_plan: RequestPlan,
    samples_per_prompt: int,
    seed_start: int,
    rollout_batch_index: int,
    max_model_len: int,
) -> list[OverlayRequest]:
    alias_to_prompt = {
        f"{index}:{prompt.prompt_id}": prompt
        for index, prompt in enumerate(prompts)
    }
    resolved = resolve_request_plan(
        request_plan,
        prompt_ids=list(alias_to_prompt),
        samples_per_prompt=samples_per_prompt,
        seed_start=seed_start,
        prompt_token_lengths=[len(prompt.prompt_token_ids) for prompt in prompts],
        rollout_batch_index=rollout_batch_index,
        max_model_len=max_model_len,
    )
    requests: list[OverlayRequest] = []
    for item in resolved:
        prompt = alias_to_prompt[item.prompt_id]
        requests.append(
            OverlayRequest(
                request_id=(
                    f"speedbench-{rollout_batch_index}-"
                    f"{item.prompt_id}-{item.sample_index}"
                ),
                prompt_id=prompt.prompt_id,
                prompt_sha256=prompt.prompt_sha256,
                source_prompt_sha256=prompt.source_prompt_sha256,
                category=prompt.category,
                sample_index=item.sample_index,
                seed=item.seed,
                prompt_token_ids=list(prompt.prompt_token_ids),
                max_tokens=item.max_tokens,
                min_tokens=item.min_tokens,
                ignore_eos=item.ignore_eos,
            )
        )
    return requests


def validate_request_plan_exact_work(
    requests: Sequence[OverlayRequest],
    prompts: Sequence[OverlayPrompt],
    *,
    samples_per_prompt: int,
) -> dict[str, int]:
    expected_count = len(prompts) * samples_per_prompt
    if len(requests) != expected_count:
        raise ValueError(
            "request-plan exact-work mismatch: "
            f"expected={expected_count} actual={len(requests)}"
        )
    return {
        "expected_requests": expected_count,
        "actual_requests": len(requests),
        "unique_prompts": len({prompt.prompt_id for prompt in prompts}),
    }


def build_prompt_shape_warmup_requests(
    prompts: Sequence[OverlayPrompt],
    *,
    samples_per_prompt: int,
    seed_start: int,
    max_tokens: int,
) -> list[OverlayRequest]:
    requests: list[OverlayRequest] = []
    seed = seed_start
    for prompt in prompts:
        for sample_index in range(samples_per_prompt):
            requests.append(
                OverlayRequest(
                    request_id=f"warmup-{prompt.prompt_id}-{sample_index}",
                    prompt_id=prompt.prompt_id,
                    prompt_sha256=prompt.prompt_sha256,
                    source_prompt_sha256=prompt.source_prompt_sha256,
                    category=prompt.category,
                    sample_index=sample_index,
                    seed=seed,
                    prompt_token_ids=list(prompt.prompt_token_ids),
                    max_tokens=max_tokens,
                    min_tokens=0,
                    ignore_eos=False,
                )
            )
            seed += 1
    return requests


def first_candidate(output: Any) -> Any:
    return output.outputs[0]


def candidate_token_ids(output: Any) -> list[int]:
    return list(first_candidate(output).token_ids)


def finish_reason(output: Any) -> str:
    return str(getattr(first_candidate(output), "finish_reason", "unknown"))


async def run_one_request_async(
    engine: Any,
    request: OverlayRequest,
    sampling_params: Any,
    *,
    batch_started_at_s: float,
    clock: Callable[[], float],
) -> CompletedRequest:
    first_output_at_s: float | None = None
    final_output: Any | None = None
    final_token_ids: list[int] = []
    final_time_s = batch_started_at_s
    async for output in engine.generate(
        prompt={"prompt_token_ids": request.prompt_token_ids},
        sampling_params=sampling_params,
        request_id=request.request_id,
    ):
        now_s = clock()
        if first_output_at_s is None:
            first_output_at_s = now_s
        final_output = output
        final_token_ids = candidate_token_ids(output)
        final_time_s = now_s
        if bool(getattr(output, "finished", False)):
            break
    if first_output_at_s is None or final_output is None:
        raise RuntimeError(f"AsyncLLM request produced no output: {request.request_id}")
    return CompletedRequest(
        request=request,
        output=final_output,
        output_token_ids=final_token_ids,
        ttft_s=round(first_output_at_s - batch_started_at_s, 6),
        finished_at_s=final_time_s,
        completion_time_s=round(final_time_s - batch_started_at_s, 6),
        finish_reason=finish_reason(final_output),
    )


async def run_overlay_batch_async(
    engine: Any,
    requests: Sequence[OverlayRequest],
    *,
    sampling_params_by_request: Mapping[str, Any],
    active_concurrency: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    batch_started_at_s = clock()
    semaphore = (
        asyncio.Semaphore(active_concurrency)
        if active_concurrency is not None and active_concurrency > 0
        else None
    )

    async def run_limited_request(request: OverlayRequest) -> CompletedRequest:
        if semaphore is None:
            return await run_one_request_async(
                engine,
                request,
                sampling_params_by_request[request.request_id],
                batch_started_at_s=batch_started_at_s,
                clock=clock,
            )
        async with semaphore:
            return await run_one_request_async(
                engine,
                request,
                sampling_params_by_request[request.request_id],
                batch_started_at_s=batch_started_at_s,
                clock=clock,
            )

    completed = await asyncio.gather(
        *(
            run_limited_request(request)
            for request in requests
        )
    )
    barrier_finished_at_s = clock()
    max_request_finished_at_s = max(
        (item.finished_at_s for item in completed),
        default=batch_started_at_s,
    )
    barrier_finished_at_s = max(barrier_finished_at_s, max_request_finished_at_s)
    output_token_ids = [item.output_token_ids for item in completed]
    return {
        "sync_barrier": "AsyncLLM.gather",
        "request_count": len(completed),
        "batch_started_at_s": batch_started_at_s,
        "barrier_finished_at_s": barrier_finished_at_s,
        "barrier_time_s": round(barrier_finished_at_s - batch_started_at_s, 6),
        "ttft_s": [item.ttft_s for item in completed],
        "completion_time_s": [item.completion_time_s for item in completed],
        "output_token_ids": output_token_ids,
        "output_token_hashes": [token_hash(token_ids) for token_ids in output_token_ids],
        "prompt_token_ids": [
            list(item.request.prompt_token_ids) for item in completed
        ],
        "finish_reasons": {
            reason: [item.finish_reason for item in completed].count(reason)
            for reason in sorted({item.finish_reason for item in completed})
        },
        "requests": [request_provenance(item.request) for item in completed],
    }


def draft_position_acceptance_rates(
    metrics: Mapping[str, Any],
) -> list[dict[str, float | int]]:
    proposal_count = float(metrics.get("num_drafts", 0.0) or 0.0)
    accepted_by_pos = list(metrics.get("num_accepted_tokens_per_pos", []) or [])
    rows: list[dict[str, float | int]] = []
    for position, accepted_value in enumerate(accepted_by_pos):
        accepted = float(accepted_value)
        if proposal_count and accepted > proposal_count:
            raise ValueError(
                f"draft-position accepted tokens exceed proposal denominator at "
                f"position {position}: accepted={accepted} proposals={proposal_count}"
            )
        rows.append(
            {
                "draft_position": position,
                "accepted_tokens": accepted,
                "proposal_count": proposal_count,
                "acceptance_rate": (
                    round(accepted / proposal_count, 6)
                    if proposal_count
                    else 0.0
                ),
            }
        )
    return rows


def completion_position_length_windows(
    *,
    output_token_ids: Sequence[Sequence[int]],
    window_size: int,
) -> list[dict[str, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    max_length = max((len(tokens) for tokens in output_token_ids), default=0)
    windows: list[dict[str, int]] = []
    for start in range(0, max_length, window_size):
        end = min(start + window_size, max_length) - 1
        contributor_count = sum(
            1
            for position in range(start, end + 1)
            for tokens in output_token_ids
            if len(tokens) > position
        )
        windows.append(
            {
                "start_pos": start,
                "end_pos": end,
                "contributor_count": contributor_count,
            }
        )
    return windows


def validate_spec_decode_counter_gate(mode: str, metrics: Mapping[str, Any]) -> None:
    if mode == "baseline":
        return
    if not metrics.get("metrics_available") or not metrics.get("active"):
        raise RuntimeError(
            f"SpecDec counters are unavailable or inactive for mode={mode}"
        )


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 6)


def summarize_overlay_latencies(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    ttft = [
        float(value)
        for row in rows
        for value in list(row.get("ttft_s", []) or [])
    ]
    completion = [
        float(value)
        for row in rows
        for value in list(row.get("completion_time_s", []) or [])
    ]
    barriers = [float(row.get("barrier_time_s", 0.0) or 0.0) for row in rows]
    median_barrier = statistics.median(barriers) if barriers else 0.0
    max_barrier = max(barriers, default=0.0)
    return {
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p90_s": percentile(ttft, 0.90),
        "ttft_p99_s": percentile(ttft, 0.99),
        "completion_p50_s": percentile(completion, 0.50),
        "completion_p90_s": percentile(completion, 0.90),
        "completion_p99_s": percentile(completion, 0.99),
        "barrier_tail_gap_s": round(max_barrier - median_barrier, 6),
    }


def _schedule_k_for_concurrency(
    concurrency: int,
    schedule: Sequence[Sequence[int]],
) -> int:
    current_k = 0
    for start, end, k_value in schedule:
        if concurrency < start:
            return current_k
        current_k = k_value
        if start <= concurrency <= end:
            return k_value
    return current_k


def active_concurrency_k_tier_reachability(
    concurrencies: Iterable[int],
    dynamic_schedule: str,
) -> list[dict[str, int | bool]]:
    schedule = parse_dynamic_schedule(dynamic_schedule)
    return [
        {
            "concurrency": int(concurrency),
            "k": _schedule_k_for_concurrency(int(concurrency), schedule),
            "reachable": True,
        }
        for concurrency in concurrencies
    ]


def require_k_tier_reachability(
    concurrencies: Iterable[int],
    dynamic_schedule: str,
) -> None:
    schedule = parse_dynamic_schedule(dynamic_schedule)
    required = {int(item[2]) for item in schedule if int(item[2]) > 0}
    reached = {
        int(item["k"])
        for item in active_concurrency_k_tier_reachability(
            concurrencies,
            dynamic_schedule,
        )
        if int(item["k"]) > 0
    }
    missing = sorted(required - reached)
    if missing:
        raise ValueError(f"K-tier not reached by active concurrency plan: {missing}")


def build_official_speedbench_command(
    *,
    model: str,
    tokenizer: str,
    modelopt_root: Path,
    dataset_path: Path,
    save_dir: Path,
    variant: str,
    tensor_parallel_size: int,
    active_concurrency: int,
    max_model_len: int,
    max_new_tokens: int,
    draft_model: str = "",
    static_k: int = 0,
    temperature: float | None = None,
) -> list[str]:
    if variant in ("dynamic", "mtp_dynamic"):
        raise ValueError(
            "pinned ModelOpt run.py does not support scheduled dynamic "
            f"speculation for official mode={variant}"
        )
    algorithm = {
        "baseline": "NONE",
        "static": "EAGLE3",
        "mtp_static": "MTP",
    }[variant]
    draft_length = 0 if variant == "baseline" else static_k
    command = [
        "python3",
        str(modelopt_root / "examples/specdec_bench/run.py"),
        "--tokenizer",
        tokenizer,
        "--dataset",
        "speed",
        "--dataset_path",
        str(dataset_path),
        "--engine",
        "VLLM",
        "--speculative_algorithm",
        algorithm,
        "--model_dir",
        model,
        "--max_seq_len",
        str(max_model_len),
        "--output_length",
        str(max_new_tokens),
        "--draft_length",
        str(draft_length),
        "--tp_size",
        str(tensor_parallel_size),
        "--concurrency",
        str(active_concurrency),
        "--trust_remote_code",
        "--save_dir",
        str(save_dir),
    ]
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if draft_model:
        command.extend(["--draft_model_dir", draft_model])
    return command


def _require_runtime_sha(value: str) -> str:
    if not value.strip() or value == "unknown":
        raise ValueError("runtime_image_sha256 must be set and must not be unknown")
    return value


def method_for_mode(mode: str) -> str:
    if mode == "baseline":
        return "baseline"
    if mode.startswith("mtp_"):
        return "mtp"
    return "eagle3"


def _required_provenance_value(value: str | None, field_name: str) -> str:
    if value is None or not value or value == "unknown":
        raise ValueError(f"{field_name} must be present and must not be unknown")
    return value


def _load_modelopt_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required ModelOpt output file: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError(f"{path.name} must contain one result object")
        return dict(payload[0])
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"{path.name} must contain a JSON object")


def _load_modelopt_sidecar(save_dir: Path, filename: str) -> dict[str, Any]:
    return _load_modelopt_json_object(save_dir / filename)


def _positive_float(mapping: Mapping[str, Any], key: str, *, file_name: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{file_name} missing numeric {key}") from exc
    if value <= 0.0:
        raise ValueError(f"{file_name} {key} must be positive")
    return value


def _stats_mean(mapping: Mapping[str, Any], key: str, *, file_name: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{file_name} missing statistics object {key}")
    return _positive_float(value, "mean", file_name=f"{file_name}:{key}")


def _required_string(mapping: Mapping[str, Any], key: str, *, file_name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{file_name} missing string {key}")
    return value


def _required_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    file_name: str,
    minimum: int,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{file_name} missing integer {key}")
    if value < minimum:
        raise ValueError(f"{file_name} {key} must be >= {minimum}")
    return value


def _required_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    file_name: str,
    minimum: float,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{file_name} missing numeric {key}")
    number = float(value)
    if number < minimum:
        raise ValueError(f"{file_name} {key} must be >= {minimum}")
    return number


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
    *,
    file_name: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{file_name} missing object {key}")
    return value


_MISSING = object()


def _lookup_path(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _required_official_field(
    resolved_config: Mapping[str, Any],
    field_name: str,
    paths: Sequence[Sequence[str]],
) -> Any:
    for path in paths:
        value = _lookup_path(resolved_config, path)
        if value is not _MISSING and value is not None:
            return value
    path_text = " or ".join(".".join(path) for path in paths)
    raise ValueError(
        f"official {field_name} missing from instrumented resolved config "
        f"({path_text})"
    )


def _official_sampling_number(
    configuration: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    field_name: str,
) -> float:
    if field_name == "temperature":
        raw_value = configuration.get("temperature")
        if raw_value is not None:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError("configuration.json temperature missing numeric value")
            return float(raw_value)
    runtime_params = configuration.get("runtime_params")
    sampling_kwargs: Any = {}
    if isinstance(runtime_params, Mapping):
        sampling_kwargs = runtime_params.get("sampling_kwargs", {})
    if not isinstance(sampling_kwargs, Mapping):
        raise ValueError("configuration.json runtime_params.sampling_kwargs must be an object")
    value = sampling_kwargs.get(field_name)
    if value is None:
        value = _lookup_path(resolved_config, ("sampling_config", field_name))
    if value is _MISSING or value is None:
        value = _lookup_path(resolved_config, ("sampling_kwargs", field_name))
    if value is _MISSING or value is None:
        raise ValueError(f"official {field_name} missing from instrumented sampling config")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"official {field_name} missing numeric value")
    number = float(value)
    if field_name == "top_p" and not 0.0 < number <= 1.0:
        raise ValueError("official top_p must be in (0, 1]")
    if field_name == "temperature" and number < 0.0:
        raise ValueError("official temperature must be >= 0")
    return number


def _timing_total_tokens_from_sidecar(sidecar: Mapping[str, Any]) -> list[int]:
    if sidecar.get("source") != "Timing.total_tokens":
        raise ValueError(f"{MODELOPT_TIMING_SIDECAR} source must be Timing.total_tokens")
    raw_values = sidecar.get("total_tokens")
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"{MODELOPT_TIMING_SIDECAR} missing total_tokens")
    total_tokens: list[int] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{MODELOPT_TIMING_SIDECAR} total_tokens must be positive integers"
            )
        total_tokens.append(value)
    return total_tokens


def _validate_modelopt_instrumentation_sidecar(
    instrumentation: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected: dict[str, Any] = {
        "schema_version": 1,
        "modelopt_commit": MODELOPT_PINNED_COMMIT,
        "source_sha256": MODELOPT_RUN_PY_SHA256,
        "patch_sha256": MODELOPT_INSTRUMENTATION_PATCH_SHA256,
        "patched_source_sha256": MODELOPT_PATCHED_RUN_PY_SHA256,
        "timing_sidecar": MODELOPT_TIMING_SIDECAR,
        "resolved_config_sidecar": MODELOPT_RESOLVED_CONFIG_SIDECAR,
    }
    for field_name, expected_value in expected.items():
        value = instrumentation.get(field_name)
        if value is None or value == "" or value == "unknown":
            raise ValueError(
                f"{MODELOPT_INSTRUMENTATION_SIDECAR} missing {field_name}"
            )
        if value != expected_value:
            raise ValueError(
                f"{MODELOPT_INSTRUMENTATION_SIDECAR} {field_name} mismatch: "
                f"{value} != {expected_value}"
            )
    return instrumentation


def _cudagraph_mode(compilation_config: Any) -> str:
    if not isinstance(compilation_config, Mapping):
        raise ValueError("official compilation_config must be an object")
    mode = compilation_config.get("cudagraph_mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("official cudagraph_mode missing from compilation_config")
    return mode


def parse_modelopt_official_outputs(
    save_dir: Path,
    *,
    expected_dataset_path: Path,
) -> dict[str, Any]:
    timing = _load_modelopt_json_object(save_dir / "timing.json")
    acceptance = _load_modelopt_json_object(save_dir / "acceptance_rate.json")
    specbench = _load_modelopt_json_object(save_dir / "specbench_results.json")
    configuration = _load_modelopt_json_object(save_dir / "configuration.json")
    timing_sidecar = _load_modelopt_sidecar(save_dir, MODELOPT_TIMING_SIDECAR)
    resolved_config_sidecar = _load_modelopt_sidecar(
        save_dir,
        MODELOPT_RESOLVED_CONFIG_SIDECAR,
    )
    instrumentation = _validate_modelopt_instrumentation_sidecar(
        _load_modelopt_sidecar(save_dir, MODELOPT_INSTRUMENTATION_SIDECAR)
    )

    output_tps = _positive_float(timing, "Output TPS", file_name="timing.json")
    output_tps_per_gpu = _positive_float(
        timing,
        "Output TPS/gpu",
        file_name="timing.json",
    )
    mean_output_tokens = _stats_mean(
        timing,
        "Number of Output Tokens",
        file_name="timing.json",
    )
    total_tokens_by_turn = _timing_total_tokens_from_sidecar(timing_sidecar)
    request_al = specbench.get("Request_AL")
    if not isinstance(request_al, Mapping) or not request_al:
        raise ValueError("specbench_results.json missing Request_AL")
    request_count = len(request_al)
    total_output_tokens = sum(total_tokens_by_turn)
    if total_output_tokens <= 0:
        raise ValueError("timing.json computed total output tokens must be positive")
    if abs((sum(total_tokens_by_turn) / len(total_tokens_by_turn)) - mean_output_tokens) > 1e-3:
        raise ValueError("timing.json Number of Output Tokens mean does not match raw totals")
    configured_dataset_path = _required_string(
        configuration,
        "dataset_path",
        file_name="configuration.json",
    )
    if str(configured_dataset_path) != str(expected_dataset_path):
        raise ValueError(
            "configuration.json dataset_path mismatch: "
            f"{configured_dataset_path} != {expected_dataset_path}"
        )
    dataset = _required_string(configuration, "dataset", file_name="configuration.json")
    if dataset != "speed":
        raise ValueError("configuration.json dataset must be speed")
    engine = _required_string(configuration, "engine", file_name="configuration.json")
    if engine != "VLLM":
        raise ValueError("configuration.json engine must be VLLM")
    serving_config = _required_mapping(
        resolved_config_sidecar,
        "serving_config",
        file_name=MODELOPT_RESOLVED_CONFIG_SIDECAR,
    )
    vllm_config = _required_mapping(
        resolved_config_sidecar,
        "vllm_config",
        file_name=MODELOPT_RESOLVED_CONFIG_SIDECAR,
    )
    engine_args = _required_mapping(
        resolved_config_sidecar,
        "engine_args",
        file_name=MODELOPT_RESOLVED_CONFIG_SIDECAR,
    )
    resolved_config = {
        "serving_config": serving_config,
        "engine_args": engine_args,
        "vllm_config": vllm_config,
        "sampling_config": resolved_config_sidecar.get("sampling_config"),
        "sampling_kwargs": resolved_config_sidecar.get("sampling_kwargs"),
    }
    compilation_config = _required_official_field(
        resolved_config,
        "compilation_config",
        (
            ("vllm_config", "compilation_config"),
            ("engine_args", "compilation_config"),
        ),
    )
    config_values = {
        "dataset_path": configured_dataset_path,
        "dataset": dataset,
        "engine": engine,
        "serving_config": serving_config,
        "engine_args": engine_args,
        "vllm_config": vllm_config,
        "resolved_config_sidecar": resolved_config_sidecar,
        "timing_sidecar": timing_sidecar,
        "instrumentation": instrumentation,
        "speculative_algorithm": _required_string(
            configuration,
            "speculative_algorithm",
            file_name="configuration.json",
        ),
        "model_dir": _required_string(
            configuration,
            "model_dir",
            file_name="configuration.json",
        ),
        "temperature": _official_sampling_number(
            configuration,
            resolved_config,
            "temperature",
        ),
        "max_seq_len": _required_int(
            configuration,
            "max_seq_len",
            file_name="configuration.json",
            minimum=1,
        ),
        "output_length": _required_int(
            configuration,
            "output_length",
            file_name="configuration.json",
            minimum=1,
        ),
        "resolved_max_model_len": _required_official_field(
            resolved_config,
            "max_model_len",
            (
                ("vllm_config", "model_config", "max_model_len"),
                ("engine_args", "max_model_len"),
                ("serving_config", "max_model_len"),
            ),
        ),
        "resolved_tensor_parallel_size": _required_official_field(
            resolved_config,
            "tensor_parallel_size",
            (
                ("vllm_config", "parallel_config", "tensor_parallel_size"),
                ("engine_args", "tensor_parallel_size"),
                ("serving_config", "tensor_parallel_size"),
            ),
        ),
        "resolved_pipeline_parallel_size": _required_official_field(
            resolved_config,
            "pipeline_parallel_size",
            (
                ("vllm_config", "parallel_config", "pipeline_parallel_size"),
                ("engine_args", "pipeline_parallel_size"),
                ("serving_config", "pipeline_parallel_size"),
            ),
        ),
        "dtype": _required_official_field(
            resolved_config,
            "dtype",
            (("vllm_config", "model_config", "dtype"), ("engine_args", "dtype")),
        ),
        "kv_cache_dtype": _required_official_field(
            resolved_config,
            "kv_cache_dtype",
            (
                ("vllm_config", "cache_config", "cache_dtype"),
                ("engine_args", "kv_cache_dtype"),
            ),
        ),
        "top_p": _official_sampling_number(configuration, resolved_config, "top_p"),
        "compilation_config": compilation_config,
        "cudagraph_mode": _cudagraph_mode(compilation_config),
        "max_num_batched_tokens": _required_official_field(
            resolved_config,
            "max_num_batched_tokens",
            (
                ("vllm_config", "scheduler_config", "max_num_batched_tokens"),
                ("engine_args", "max_num_batched_tokens"),
            ),
        ),
        "gpu_memory_utilization": _required_official_field(
            resolved_config,
            "gpu_memory_utilization",
            (
                ("vllm_config", "cache_config", "gpu_memory_utilization"),
                ("engine_args", "gpu_memory_utilization"),
            ),
        ),
        "distributed_executor_backend": _required_official_field(
            resolved_config,
            "distributed_executor_backend",
            (
                ("vllm_config", "parallel_config", "distributed_executor_backend"),
                ("engine_args", "distributed_executor_backend"),
            ),
        ),
        "distributed_timeout_seconds": _required_official_field(
            resolved_config,
            "distributed_timeout_seconds",
            (
                ("vllm_config", "parallel_config", "distributed_timeout_seconds"),
                ("engine_args", "distributed_timeout_seconds"),
            ),
        ),
        "enable_expert_parallel": _required_official_field(
            resolved_config,
            "enable_expert_parallel",
            (
                ("vllm_config", "parallel_config", "enable_expert_parallel"),
                ("engine_args", "enable_expert_parallel"),
            ),
        ),
        "model_loader_extra_config": _required_official_field(
            resolved_config,
            "model_loader_extra_config",
            (
                ("vllm_config", "load_config", "model_loader_extra_config"),
                ("engine_args", "model_loader_extra_config"),
            ),
        ),
        "mamba_ssm_cache_dtype": _required_official_field(
            resolved_config,
            "mamba_ssm_cache_dtype",
            (
                ("vllm_config", "mamba_config", "mamba_ssm_cache_dtype"),
                ("engine_args", "mamba_ssm_cache_dtype"),
            ),
        ),
        "mamba_backend": _required_official_field(
            resolved_config,
            "mamba_backend",
            (
                ("vllm_config", "mamba_config", "mamba_backend"),
                ("engine_args", "mamba_backend"),
            ),
        ),
        "enable_mamba_cache_stochastic_rounding": _required_official_field(
            resolved_config,
            "enable_mamba_cache_stochastic_rounding",
            (
                (
                    "vllm_config",
                    "mamba_config",
                    "enable_mamba_cache_stochastic_rounding",
                ),
                ("engine_args", "enable_mamba_cache_stochastic_rounding"),
            ),
        ),
        "mamba_cache_philox_rounds": _required_official_field(
            resolved_config,
            "mamba_cache_philox_rounds",
            (
                ("vllm_config", "mamba_config", "mamba_cache_philox_rounds"),
                ("engine_args", "mamba_cache_philox_rounds"),
            ),
        ),
        "moe_backend": _required_official_field(
            resolved_config,
            "moe_backend",
            (
                ("vllm_config", "kernel_config", "moe_backend"),
                ("engine_args", "kernel_config", "moe_backend"),
            ),
        ),
        "draft_length": _required_int(
            configuration,
            "draft_length",
            file_name="configuration.json",
            minimum=0,
        ),
        "tp_size": _required_int(
            configuration,
            "tp_size",
            file_name="configuration.json",
            minimum=1,
        ),
        "concurrency": _required_int(
            configuration,
            "concurrency",
            file_name="configuration.json",
            minimum=1,
        ),
        "save_dir": _required_string(
            configuration,
            "save_dir",
            file_name="configuration.json",
        ),
    }
    average_al = _positive_float(
        specbench,
        "Average_AL",
        file_name="specbench_results.json",
    )
    acceptance_average_al = _positive_float(
        acceptance,
        "Average_AL",
        file_name="acceptance_rate.json",
    )
    if abs(acceptance_average_al - average_al) > 1e-9:
        raise ValueError("acceptance_rate.json and specbench_results.json Average_AL mismatch")
    return {
        "timing": timing,
        "acceptance": acceptance,
        "specbench": specbench,
        "configuration": configuration,
        "timing_sidecar": timing_sidecar,
        "resolved_config_sidecar": resolved_config_sidecar,
        "instrumentation": instrumentation,
        "configuration_values": config_values,
        "output_tps": output_tps,
        "output_tps_per_gpu": output_tps_per_gpu,
        "total_output_tokens": total_output_tokens,
        "total_tokens_by_turn": total_tokens_by_turn,
        "total_rollout_time_s": round(total_output_tokens / output_tps, 6),
        "average_al": average_al,
        "request_count": request_count,
    }


def adapt_official_speedbench_output(
    *,
    save_dir: Path,
    output: Path,
    model: str,
    draft_model: str,
    variant: str,
    dataset_config: str,
    tensor_parallel_size: int,
    active_concurrency: int,
    max_model_len: int,
    max_new_tokens: int,
    static_k: int,
    temperature: float | None,
    runtime_image_sha256: str,
    model_config_hash_value: str | None,
    prepared_manifest_hash: str | None,
    prepared_dataset_path: Path,
) -> dict[str, Any]:
    official = parse_modelopt_official_outputs(
        save_dir,
        expected_dataset_path=prepared_dataset_path,
    )
    configuration = official["configuration"]
    configuration_values = official["configuration_values"]
    expected_algorithm = {
        "baseline": "NONE",
        "static": "EAGLE3",
        "mtp_static": "MTP",
    }.get(variant)
    if expected_algorithm is None:
        raise ValueError(
            "pinned ModelOpt run.py does not support scheduled dynamic "
            f"speculation for official mode={variant}"
        )
    if configuration_values["speculative_algorithm"] != expected_algorithm:
        raise ValueError(
            "configuration.json speculative_algorithm mismatch: "
            f"{configuration_values['speculative_algorithm']} != {expected_algorithm}"
        )
    if variant == "baseline":
        configured_draft_model = configuration.get("draft_model_dir")
        official_draft_model = (
            configured_draft_model
            if isinstance(configured_draft_model, str) and configured_draft_model.strip()
            else "none"
        )
    else:
        official_draft_model = _required_string(
            configuration,
            "draft_model_dir",
            file_name="configuration.json",
        )
        if int(configuration_values["draft_length"]) < 1:
            raise ValueError("configuration.json draft_length must be >= 1")
    runtime_sha = _require_runtime_sha(runtime_image_sha256)
    model_hash = _required_provenance_value(
        str(
            (
                configuration.get("checkpoint", {})
                if isinstance(configuration.get("checkpoint"), Mapping)
                else {}
            ).get("index_sha256")
            or model_config_hash_value
            or ""
        ),
        "model_config_hash",
    )
    manifest_hash = _required_provenance_value(
        prepared_manifest_hash,
        "prepared_manifest_hash",
    )
    instrumentation = configuration_values["instrumentation"]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "runtime": collect_runtime_metadata(),
        "config": {
            "cohort": "official",
            "official_runner": "ModelOpt examples/specdec_bench/run.py",
            "upstream_turn_loop_preserved": True,
            "mode": variant,
            "method": method_for_mode(variant),
            "model": str(configuration_values["model_dir"]),
            "draft_model": str(official_draft_model),
            "dataset_config": dataset_config,
            "dataset_path": str(configuration_values["dataset_path"]),
            "active_concurrency": int(configuration_values["concurrency"]),
            "tensor_parallel_size": int(configuration_values["resolved_tensor_parallel_size"]),
            "pipeline_parallel_size": int(configuration_values["resolved_pipeline_parallel_size"]),
            "dtype": configuration_values["dtype"],
            "kv_cache_dtype": configuration_values["kv_cache_dtype"],
            "max_model_len": int(configuration_values["resolved_max_model_len"]),
            "max_new_tokens": int(configuration_values["output_length"]),
            "static_k": int(configuration_values["draft_length"]),
            "temperature": float(configuration_values["temperature"]),
            "top_p": float(configuration_values["top_p"]),
            "sampling_protocol": "official-modelopt",
            "runtime_image_sha256": runtime_sha,
            "model_config_hash": model_hash,
            "prepared_manifest_hash": manifest_hash,
            "request_plan_hash": "official-upstream-modelopt",
            "prompt_set_hash": f"official-upstream:{configuration_values['dataset_path']}",
            "cudagraph_mode": configuration_values["cudagraph_mode"],
            "compilation_config": configuration_values["compilation_config"],
            "sampling": {
                "temperature": float(configuration_values["temperature"]),
                "top_p": float(configuration_values["top_p"]),
            },
            "max_num_batched_tokens": configuration_values["max_num_batched_tokens"],
            "gpu_memory_utilization": configuration_values["gpu_memory_utilization"],
            "samples_per_prompt": 1,
            "rollout_batches": 1,
            "distributed_executor_backend": configuration_values[
                "distributed_executor_backend"
            ],
            "distributed_timeout_seconds": configuration_values[
                "distributed_timeout_seconds"
            ],
            "enable_expert_parallel": configuration_values["enable_expert_parallel"],
            "model_loader_extra_config": configuration_values["model_loader_extra_config"],
            "mamba_ssm_cache_dtype": configuration_values["mamba_ssm_cache_dtype"],
            "mamba_backend": configuration_values["mamba_backend"],
            "enable_mamba_cache_stochastic_rounding": configuration_values[
                "enable_mamba_cache_stochastic_rounding"
            ],
            "mamba_cache_philox_rounds": configuration_values["mamba_cache_philox_rounds"],
            "moe_backend": configuration_values["moe_backend"],
            "official_configuration": configuration,
            "official_serving_config": configuration_values["serving_config"],
            "official_engine_args": configuration_values["engine_args"],
            "official_vllm_config": configuration_values["vllm_config"],
            "official_instrumentation": instrumentation,
            "official_instrumentation_schema_version": instrumentation[
                "schema_version"
            ],
            "official_instrumentation_modelopt_commit": instrumentation[
                "modelopt_commit"
            ],
            "official_instrumentation_source_sha256": instrumentation["source_sha256"],
            "official_instrumentation_patch_sha256": instrumentation["patch_sha256"],
            "official_instrumentation_patched_source_sha256": instrumentation[
                "patched_source_sha256"
            ],
        },
        "official_output": {
            "save_dir": str(save_dir),
            "timing": official["timing"],
            "acceptance_rate": official["acceptance"],
            "specbench_results": official["specbench"],
            "timing_total_tokens": official["timing_sidecar"],
            "resolved_vllm_config": official["resolved_config_sidecar"],
            "instrumentation": official["instrumentation"],
        },
        "summary": {
            "total_rollout_time_s": official["total_rollout_time_s"],
            "total_output_tokens": official["total_output_tokens"],
            "output_tok_s": official["output_tps"],
            "output_tok_s_per_gpu": official["output_tps_per_gpu"],
            "spec_decode_metrics": {
                "mean_acceptance_length": official["average_al"],
                "acceptance_rate": None,
                "acceptance_rate_unavailable_reason": (
                    "ModelOpt SpecBench Average_AL is mean accepted length, "
                    "not a scalar acceptance rate"
                ),
                "acceptance_length_histogram": official["specbench"].get(
                    "Acceptance_Length_Histogram",
                    {},
                ),
                "conditional_acceptance_rate": official["specbench"].get(
                    "Conditional_Acceptance_Rate",
                    {},
                ),
                "joint_acceptance_rate": official["specbench"].get(
                    "Joint_Acceptance_Rate",
                    {},
                ),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, payload)
    return payload


def run_official_speedbench(
    *,
    model: str,
    tokenizer: str,
    modelopt_root: Path,
    dataset_config: str,
    prepared_dataset_path: Path,
    output: Path,
    variant: str,
    tensor_parallel_size: int,
    active_concurrency: int,
    max_model_len: int,
    max_new_tokens: int,
    static_k: int,
    temperature: float | None,
    runtime_image_sha256: str,
    model_config_hash: str | None,
    prepared_manifest_hash: str | None,
    draft_model: str = "",
) -> dict[str, Any]:
    save_dir = output.with_suffix(".modelopt")
    staged_modelopt_root = Path(str(save_dir) + ".instrumented_source")
    stage_instrumented_modelopt_source(
        modelopt_root,
        staged_modelopt_root,
        save_dir=save_dir,
    )
    command = build_official_speedbench_command(
        model=model,
        tokenizer=tokenizer,
        modelopt_root=staged_modelopt_root,
        dataset_path=prepared_dataset_path,
        save_dir=save_dir,
        variant=variant,
        tensor_parallel_size=tensor_parallel_size,
        active_concurrency=active_concurrency,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        draft_model=draft_model,
        static_k=static_k,
        temperature=temperature,
    )
    subprocess.run(command, check=True)
    return adapt_official_speedbench_output(
        save_dir=save_dir,
        output=output,
        model=model,
        draft_model=draft_model,
        variant=variant,
        dataset_config=dataset_config,
        tensor_parallel_size=tensor_parallel_size,
        active_concurrency=active_concurrency,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        static_k=static_k,
        temperature=temperature,
        runtime_image_sha256=runtime_image_sha256,
        model_config_hash_value=model_config_hash,
        prepared_manifest_hash=prepared_manifest_hash,
        prepared_dataset_path=prepared_dataset_path,
    )


def request_provenance(request: OverlayRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "prompt_id": request.prompt_id,
        "prompt_sha256": request.prompt_sha256,
        "source_prompt_sha256": request.source_prompt_sha256,
        "category": request.category,
        "sample_index": request.sample_index,
        "seed": request.seed,
        "prompt_tokens": len(request.prompt_token_ids),
        "max_tokens": request.max_tokens,
        "min_tokens": request.min_tokens,
        "ignore_eos": request.ignore_eos,
    }


def length_statistics(lengths: Sequence[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    ordered = sorted(lengths)

    def percentile(quantile: float) -> float:
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def build_sampling_params(
    sampling_params_cls: Any,
    requests: Sequence[OverlayRequest],
    *,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    return {
        request.request_id: sampling_params_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            ignore_eos=request.ignore_eos,
            seed=request.seed,
            logprobs=0,
        )
        for request in requests
    }


def build_compilation_config(args: argparse.Namespace) -> dict[str, Any]:
    compilation_config: dict[str, Any] = {
        "cudagraph_mode": args.cudagraph_mode,
    }
    if args.disable_fuse_allreduce_rms:
        compilation_config["pass_config"] = {"fuse_allreduce_rms": False}
    return compilation_config


def build_async_engine_kwargs(
    args: argparse.Namespace,
    speculative_config: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.active_concurrency,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "enable_expert_parallel": args.enable_expert_parallel,
        "seed": args.seed,
        "disable_log_stats": False,
        "compilation_config": build_compilation_config(args),
    }
    if speculative_config is not None:
        kwargs["speculative_config"] = copy.deepcopy(speculative_config)
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.distributed_executor_backend:
        kwargs["distributed_executor_backend"] = args.distributed_executor_backend
    if args.distributed_timeout_seconds is not None:
        kwargs["distributed_timeout_seconds"] = args.distributed_timeout_seconds
    if args.model_loader_num_threads > 0:
        kwargs["model_loader_extra_config"] = {
            "enable_multithread_load": True,
            "num_threads": args.model_loader_num_threads,
        }
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.moe_backend:
        kwargs["kernel_config"] = {"moe_backend": args.moe_backend}
    if args.mamba_ssm_cache_dtype:
        kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype
    if args.mamba_backend:
        kwargs["mamba_backend"] = args.mamba_backend
    if args.enable_mamba_cache_stochastic_rounding:
        kwargs["enable_mamba_cache_stochastic_rounding"] = True
    if args.mamba_cache_philox_rounds is not None:
        kwargs["mamba_cache_philox_rounds"] = args.mamba_cache_philox_rounds
    return kwargs


async def run_overlay(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import SamplingParams  # pyright: ignore[reportMissingImports]
    from vllm.engine.arg_utils import AsyncEngineArgs  # pyright: ignore[reportMissingImports]
    from vllm.v1.engine.async_llm import AsyncLLM  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    overlay = build_overlay_from_prepared_parquet(
        prepared_root=args.prepared_root,
        prepared_manifest=args.prepared_manifest,
        prepared_checksums=args.prepared_checksums,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    prompt_batches = expand_overlay_barrier_batches(
        overlay.prompts,
        active_concurrency=args.active_concurrency,
        rollout_batches=args.rollout_batches,
    )
    request_plan = load_request_plan(args.request_plan)
    if request_plan.max_model_len != args.max_model_len:
        raise ValueError(
            f"--max-model-len must match request plan: got {args.max_model_len}, "
            f"expected {request_plan.max_model_len}"
        )
    dynamic_schedule = parse_dynamic_schedule(args.dynamic_schedule)
    speculative_config = build_speculative_config(
        mode=args.mode,
        draft_model=args.draft_model,
        static_k=args.static_k,
        dynamic_schedule=dynamic_schedule,
    )
    engine_kwargs = build_async_engine_kwargs(args, speculative_config)
    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLM.from_engine_args(engine_args)
    rows: list[dict[str, Any]] = []
    total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size
    max_new_tokens = max(bucket.max_tokens for bucket in request_plan.buckets)
    try:
        warmup_requests = build_prompt_shape_warmup_requests(
            prompt_batches[0],
            samples_per_prompt=args.samples_per_prompt,
            seed_start=args.seed - (args.active_concurrency * args.samples_per_prompt),
            max_tokens=args.warmup_max_tokens,
        )
        warmup_params = build_sampling_params(
            SamplingParams,
            warmup_requests,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        await run_overlay_batch_async(
            engine,
            warmup_requests,
            sampling_params_by_request=warmup_params,
            active_concurrency=args.active_concurrency,
        )
        for batch_index, prompt_batch in enumerate(prompt_batches):
            requests = prepare_overlay_requests(
                prompt_batch,
                request_plan=request_plan,
                samples_per_prompt=args.samples_per_prompt,
                seed_start=args.seed,
                rollout_batch_index=batch_index,
                max_model_len=args.max_model_len,
            )
            if args.request_plan_exact_work:
                validate_request_plan_exact_work(
                    requests,
                    prompt_batch,
                    samples_per_prompt=args.samples_per_prompt,
                )
            before = read_spec_decode_counters(engine)
            params = build_sampling_params(
                SamplingParams,
                requests,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            row = await run_overlay_batch_async(
                engine,
                requests,
                sampling_params_by_request=params,
                active_concurrency=args.active_concurrency,
            )
            metrics = diff_spec_decode_counters(read_spec_decode_counters(engine), before)
            validate_spec_decode_counter_gate(args.mode, metrics)
            output_lengths = [len(tokens) for tokens in row["output_token_ids"]]
            exact_work = (
                validate_request_plan_exact_work(
                    requests,
                    prompt_batch,
                    samples_per_prompt=args.samples_per_prompt,
                )
                if args.request_plan_exact_work
                else None
            )
            row.update(
                {
                    "batch_index": batch_index,
                    "output_tokens": sum(output_lengths),
                    "output_tok_s": sum(output_lengths) / row["barrier_time_s"],
                    "output_tok_s_per_gpu": (
                        sum(output_lengths) / row["barrier_time_s"] / total_gpus
                    ),
                    "completion_length": length_statistics(output_lengths),
                    "spec_decode_metrics": metrics,
                    "draft_position_acceptance": draft_position_acceptance_rates(
                        metrics
                    ),
                    "completion_position_windows": (
                        completion_position_length_windows(
                            output_token_ids=row["output_token_ids"],
                            window_size=args.acceptance_window_size,
                        )
                    ),
                    "acceptance_limitation": ACCEPTANCE_LIMITATION,
                    "request_plan_exact_work": exact_work,
                }
            )
            rows.append(row)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    total_time_s = sum(float(row["barrier_time_s"]) for row in rows)
    total_output_tokens = sum(int(row["output_tokens"]) for row in rows)
    latency_summary = summarize_overlay_latencies(rows)
    runtime_sha = _require_runtime_sha(args.runtime_image_sha256)
    compilation_config = build_compilation_config(args)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "runtime": collect_runtime_metadata(),
        "config": {
            "cohort": "overlay",
            "scenario": "speedbench_sync_overlay",
            "sync_barrier": "AsyncLLM.gather",
            "mode": args.mode,
            "method": method_for_mode(args.mode),
            "model": args.model,
            "draft_model": args.draft_model or "none",
            "tokenizer": tokenizer_path,
            "speculative_config": speculative_config,
            "request_plan": str(args.request_plan),
            "request_plan_hash": request_plan.plan_hash,
            "prepared_root": str(args.prepared_root),
            "prepared_manifest": str(args.prepared_manifest),
            "prepared_checksums": str(args.prepared_checksums),
            "prepared_manifest_hash": overlay.prepared_manifest_hash,
            "prepared_parquet_hash": overlay.parquet_hash,
            "dataset_config": overlay.dataset_config,
            "prompt_set_hash": overlay.prompt_set_hash,
            "unique_prompt_count": len({prompt.prompt_id for prompt in overlay.prompts}),
            "active_concurrency": args.active_concurrency,
            "samples_per_prompt": args.samples_per_prompt,
            "rollout_batches": args.rollout_batches,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "total_gpus": total_gpus,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_new_tokens": max_new_tokens,
            "max_num_seqs": args.active_concurrency,
            "max_num_batched_tokens": args.max_num_batched_tokens
            if args.max_num_batched_tokens is not None
            else "auto",
            "enable_expert_parallel": args.enable_expert_parallel,
            "distributed_executor_backend": args.distributed_executor_backend or "auto",
            "distributed_timeout_seconds": args.distributed_timeout_seconds
            if args.distributed_timeout_seconds is not None
            else "auto",
            "model_loader_extra_config": engine_kwargs.get(
                "model_loader_extra_config",
                "auto",
            ),
            "attention_backend": args.attention_backend or "auto",
            "moe_backend": args.moe_backend or "auto",
            "cudagraph_mode": args.cudagraph_mode,
            "compilation_config": compilation_config,
            "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype or "auto",
            "mamba_backend": args.mamba_backend or "auto",
            "enable_mamba_cache_stochastic_rounding": (
                args.enable_mamba_cache_stochastic_rounding
            ),
            "mamba_cache_philox_rounds": args.mamba_cache_philox_rounds
            if args.mamba_cache_philox_rounds is not None
            else "auto",
            "runtime_image_sha256": runtime_sha,
            "model_config_hash": model_config_hash(args.model),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_protocol": "sync-rl-overlay-user",
            "sampling": {"temperature": args.temperature, "top_p": args.top_p},
            "seed": args.seed,
        },
        "rollout_batches": rows,
        "summary": {
            "total_rollout_time_s": total_time_s,
            "total_output_tokens": total_output_tokens,
            "output_tok_s": total_output_tokens / total_time_s if total_time_s else 0.0,
            "output_tok_s_per_gpu": (
                total_output_tokens / total_time_s / total_gpus if total_time_s else 0.0
            ),
            "spec_decode_metrics": sum_spec_decode_counters(
                [row["spec_decode_metrics"] for row in rows]
            ),
            **latency_summary,
        },
    }
    write_json_atomic(args.output, payload)
    return payload


def run_official(args: argparse.Namespace) -> int:
    if args.prepared_manifest is None:
        raise ValueError("--prepared-manifest is required for official cohort")
    if args.prepared_checksums is None:
        raise ValueError("--prepared-checksums is required for official cohort")
    tokenizer = args.tokenizer or args.model
    prepared_dataset_path = resolve_prepared_dataset_path(
        prepared_root=args.prepared_root,
        prepared_manifest=args.prepared_manifest,
        prepared_checksums=args.prepared_checksums,
        dataset_config=args.dataset_config,
    )
    prepared_manifest_hash = sha256_file(args.prepared_manifest)
    command = build_official_speedbench_command(
        model=args.model,
        tokenizer=tokenizer,
        modelopt_root=args.modelopt_root,
        dataset_path=prepared_dataset_path,
        save_dir=args.output.with_suffix(".modelopt"),
        variant=args.mode,
        tensor_parallel_size=args.tensor_parallel_size,
        active_concurrency=args.active_concurrency,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        draft_model=args.draft_model,
        static_k=args.static_k,
        temperature=args.temperature,
    )
    if args.print_official_command:
        print(json.dumps(command))
        return 0
    run_official_speedbench(
        model=args.model,
        tokenizer=tokenizer,
        modelopt_root=args.modelopt_root,
        dataset_config=args.dataset_config,
        prepared_dataset_path=prepared_dataset_path,
        output=args.output,
        variant=args.mode,
        tensor_parallel_size=args.tensor_parallel_size,
        active_concurrency=args.active_concurrency,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        static_k=args.static_k,
        temperature=args.temperature,
        runtime_image_sha256=args.runtime_image_sha256,
        model_config_hash=model_config_hash(args.model),
        prepared_manifest_hash=prepared_manifest_hash,
        draft_model=args.draft_model,
    )
    return 0


def resolve_cohort_sampling_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.cohort == "overlay":
        if args.temperature is None:
            args.temperature = 1.0
        if args.top_p is None:
            args.top_p = 1.0
    elif args.cohort == "official":
        pass
    else:
        raise ValueError(f"unsupported cohort: {args.cohort}")
    if args.temperature is not None and args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("official", "overlay"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--draft-model", default="")
    parser.add_argument(
        "--mode",
        choices=("baseline", "static", "dynamic", "mtp_static", "mtp_dynamic"),
        required=True,
    )
    parser.add_argument("--static-k", type=int, default=5)
    parser.add_argument("--dynamic-schedule", default=DEFAULT_DYNAMIC_SCHEDULE)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--distributed-executor-backend", default="")
    parser.add_argument("--distributed-timeout-seconds", type=int)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--model-loader-num-threads", type=int, default=0)
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--moe-backend", default="")
    parser.add_argument("--disable-fuse-allreduce-rms", action="store_true")
    parser.add_argument("--mamba-ssm-cache-dtype", default="")
    parser.add_argument("--mamba-backend", default="")
    parser.add_argument(
        "--enable-mamba-cache-stochastic-rounding",
        action="store_true",
    )
    parser.add_argument("--mamba-cache-philox-rounds", type=int)
    parser.add_argument("--active-concurrency", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--rollout-batches", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--acceptance-window-size", type=int, default=16)
    parser.add_argument("--cudagraph-mode", default="PIECEWISE")
    parser.add_argument("--request-plan", type=Path)
    parser.add_argument("--request-plan-exact-work", action="store_true")
    parser.add_argument("--runtime-image-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modelopt-root", type=Path, default=Path("/workspace/modelopt"))
    parser.add_argument("--prepared-root", type=Path, default=Path("/workspace/speedbench/prepared/speed"))
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--prepared-checksums", type=Path)
    parser.add_argument("--dataset-config", default="throughput_1k")
    parser.add_argument("--print-official-command", action="store_true")
    return parser


def parse_speedbench_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return resolve_cohort_sampling_defaults(build_parser().parse_args(argv))


def main() -> None:
    args = parse_speedbench_args()
    if args.cohort == "official":
        raise SystemExit(run_official(args))
    if args.prepared_manifest is None:
        raise ValueError("--prepared-manifest is required for overlay cohort")
    if args.prepared_checksums is None:
        raise ValueError("--prepared-checksums is required for overlay cohort")
    if args.request_plan is None:
        raise ValueError("--request-plan is required for overlay cohort")
    asyncio.run(run_overlay(args))


if __name__ == "__main__":
    main()
