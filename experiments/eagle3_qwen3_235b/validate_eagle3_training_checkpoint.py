#!/usr/bin/env python3
"""Validate a trained ModelOpt Eagle3 checkpoint before export.

This gate is intentionally filesystem-first. It does not instantiate the
235B verifier or draft model, but it verifies that the checkpoint directory has
the HF/ModelOpt files that ``scripts/export_hf_checkpoint.py`` needs before the
expensive export/conversion job starts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--modelopt-dir", type=Path, default=ROOT / "Model-Optimizer")
    parser.add_argument(
        "--reference-arch",
        type=Path,
        default=EXP / "qwen3_235b_thinking_eagle3_architecture.json",
    )
    parser.add_argument("--expected-base-model", default=None)
    parser.add_argument("--min-weight-bytes", type=int, default=1)
    parser.add_argument("--require-trainer-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-modelopt-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-modelopt-state-load",
        action="store_true",
        help="Fail if modelopt_state.pth cannot be loaded with torch/ModelOpt.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid JSON: {exc}"


def file_summary(paths: list[Path]) -> dict[str, Any]:
    total_bytes = 0
    files: list[dict[str, Any]] = []
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        total_bytes += size
        files.append({"name": path.name, "bytes": size})
    return {"count": len(paths), "total_bytes": total_bytes, "files": files[:16]}


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def parse_safetensors_header(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size < 8:
        return {"status": "fail", "error": "file is smaller than safetensors header prefix", "bytes": size}
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        header_bytes = handle.read(header_size)
    header_end = 8 + header_size
    if header_size <= 0:
        return {"status": "fail", "error": "header size is not positive", "bytes": size, "header_size": header_size}
    if header_end > size:
        return {"status": "fail", "error": "header extends past end of file", "bytes": size, "header_size": header_size}
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as exc:
        return {"status": "fail", "error": f"header JSON is invalid: {exc}", "bytes": size, "header_size": header_size}
    if not isinstance(header, dict):
        return {"status": "fail", "error": "header JSON is not an object", "bytes": size, "header_size": header_size}

    data_region_size = size - header_end
    tensor_count = 0
    used_offsets: list[tuple[int, int, str]] = []
    problems: list[str] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        tensor_count += 1
        if not isinstance(spec, dict):
            problems.append(f"{name}: tensor spec is not an object")
            continue
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if not isinstance(dtype, str):
            problems.append(f"{name}: dtype is missing")
        if not isinstance(shape, list) or not all(isinstance(item, int) and item >= 0 for item in shape):
            problems.append(f"{name}: shape is invalid")
            shape = []
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            problems.append(f"{name}: data_offsets is invalid")
            continue
        start, end = offsets
        if end > data_region_size:
            problems.append(f"{name}: data_offsets extends past data region")
        used_offsets.append((start, end, name))
        dtype_size = DTYPE_BYTES.get(dtype)
        if dtype_size is not None and isinstance(shape, list):
            expected_bytes = product(shape) * dtype_size
            if expected_bytes != end - start:
                problems.append(f"{name}: byte size {end - start} != expected {expected_bytes}")

    if tensor_count == 0:
        problems.append("no tensor entries found")
    sorted_offsets = sorted(used_offsets)
    for (left_start, left_end, left_name), (right_start, _right_end, right_name) in zip(
        sorted_offsets,
        sorted_offsets[1:],
    ):
        if right_start < left_end:
            problems.append(f"{left_name} overlaps {right_name}")

    return {
        "status": "fail" if problems else "pass",
        "bytes": size,
        "header_size": header_size,
        "tensor_count": tensor_count,
        "data_region_size": data_region_size,
        "problems": problems[:16],
    }


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def inspect_directory(checks: list[dict[str, Any]], checkpoint_dir: Path) -> dict[str, Any]:
    if not checkpoint_dir.exists():
        add(checks, "paths", "checkpoint directory", "missing", f"path is not visible: {checkpoint_dir}", path=str(checkpoint_dir))
        return {"checkpoint_dirs": []}
    if not checkpoint_dir.is_dir():
        add(checks, "paths", "checkpoint directory", "fail", f"path is not a directory: {checkpoint_dir}", path=str(checkpoint_dir))
        return {"checkpoint_dirs": []}
    immediate_files = [path for path in checkpoint_dir.iterdir() if path.is_file()]
    checkpoint_dirs = sorted(
        [path for path in checkpoint_dir.iterdir() if path.is_dir() and checkpoint_step(path) >= 0],
        key=checkpoint_step,
    )
    total_bytes = sum(path.stat().st_size for path in immediate_files)
    add(
        checks,
        "paths",
        "checkpoint directory",
        "pass" if immediate_files or checkpoint_dirs else "incomplete",
        "checkpoint directory is visible"
        if immediate_files or checkpoint_dirs
        else "checkpoint directory exists but has no files or checkpoint-* children",
        path=str(checkpoint_dir),
        immediate_file_count=len(immediate_files),
        immediate_total_bytes=total_bytes,
        checkpoint_dirs=[path.name for path in checkpoint_dirs[-8:]],
    )
    return {"checkpoint_dirs": [str(path) for path in checkpoint_dirs]}


def check_config(checks: list[dict[str, Any]], checkpoint_dir: Path, reference_arch: Path) -> dict[str, Any] | None:
    config_path = checkpoint_dir / "config.json"
    config, error = load_json(config_path)
    if error:
        add(checks, "config", "HF checkpoint config", "missing", error, path=str(config_path))
        return None
    add(
        checks,
        "config",
        "HF checkpoint config",
        "pass",
        "config.json is readable",
        path=str(config_path),
        model_type=config.get("model_type"),
        hidden_size=config.get("hidden_size"),
        vocab_size=config.get("vocab_size"),
        num_hidden_layers=config.get("num_hidden_layers"),
    )

    ref, ref_error = load_json(reference_arch)
    if ref_error:
        add(checks, "config", "Qwen3 reference compatibility", "warn", ref_error, reference_arch=str(reference_arch))
        return config
    verifier = ref.get("verifier_summary") or {}
    problems: list[str] = []
    if verifier.get("hidden_size") and config.get("hidden_size") != verifier.get("hidden_size"):
        problems.append(f"hidden_size {config.get('hidden_size')} != reference {verifier.get('hidden_size')}")
    if verifier.get("num_hidden_layers") and config.get("num_hidden_layers") not in {None, verifier.get("num_hidden_layers")}:
        problems.append(
            f"num_hidden_layers {config.get('num_hidden_layers')} != reference {verifier.get('num_hidden_layers')}"
        )
    status = "fail" if problems else "pass"
    add(
        checks,
        "config",
        "Qwen3 reference compatibility",
        status,
        "; ".join(problems) if problems else "checkpoint config is compatible with Qwen3 verifier dimensions",
        model_type=config.get("model_type"),
        expected_model_type=ref.get("model_type"),
        verifier_hidden_size=verifier.get("hidden_size"),
        verifier_num_hidden_layers=verifier.get("num_hidden_layers"),
    )
    return config


def check_weight_index(checks: list[dict[str, Any]], checkpoint_dir: Path, index_name: str) -> None:
    index_path = checkpoint_dir / index_name
    if not index_path.exists():
        return
    payload, error = load_json(index_path)
    if error:
        add(checks, "weights", index_name, "fail", error, path=str(index_path))
        return
    weight_map = payload.get("weight_map") or {}
    referenced = sorted(set(str(value) for value in weight_map.values()))
    missing = [name for name in referenced if not (checkpoint_dir / name).exists()]
    if missing:
        add(
            checks,
            "weights",
            index_name,
            "fail",
            "weight index references missing shard files",
            path=str(index_path),
            missing=missing[:16],
            referenced_count=len(referenced),
        )
        return
    add(
        checks,
        "weights",
        index_name,
        "pass",
        "weight index references visible shard files",
        path=str(index_path),
        referenced_count=len(referenced),
        tensor_count=len(weight_map),
    )


def check_weights(checks: list[dict[str, Any]], checkpoint_dir: Path, min_weight_bytes: int) -> list[Path]:
    safetensors = sorted(checkpoint_dir.glob("*.safetensors"))
    bin_weights = sorted(checkpoint_dir.glob("pytorch_model*.bin"))
    weights = safetensors + bin_weights
    summary = file_summary(weights)
    if not weights:
        checkpoint_dirs = sorted(
            [path for path in checkpoint_dir.iterdir() if path.is_dir() and checkpoint_step(path) >= 0],
            key=checkpoint_step,
        ) if checkpoint_dir.exists() and checkpoint_dir.is_dir() else []
        latest = str(checkpoint_dirs[-1]) if checkpoint_dirs else None
        add(
            checks,
            "weights",
            "HF checkpoint weights",
            "missing",
            "no weight files matching *.safetensors or pytorch_model*.bin",
            path=str(checkpoint_dir),
            latest_nested_checkpoint=latest,
        )
        return weights
    if summary["total_bytes"] < min_weight_bytes:
        add(
            checks,
            "weights",
            "HF checkpoint weights",
            "fail",
            f"weight files total fewer than {min_weight_bytes} bytes",
            **summary,
        )
    else:
        add(checks, "weights", "HF checkpoint weights", "pass", "weight files are present and non-empty", **summary)

    parsed = [{"name": path.name, **parse_safetensors_header(path)} for path in safetensors]
    failures = [item for item in parsed if item.get("status") != "pass"]
    if failures:
        add(
            checks,
            "weights",
            "safetensors structure",
            "fail",
            "one or more safetensors files have invalid headers, offsets, or tensor metadata",
            failures=failures[:8],
            checked=len(parsed),
        )
    elif parsed:
        add(
            checks,
            "weights",
            "safetensors structure",
            "pass",
            "safetensors headers, offsets, and tensor metadata are valid",
            checked=len(parsed),
            tensors=sum(int(item.get("tensor_count") or 0) for item in parsed),
        )
    check_weight_index(checks, checkpoint_dir, "model.safetensors.index.json")
    check_weight_index(checks, checkpoint_dir, "pytorch_model.bin.index.json")
    return weights


def check_trainer_state(checks: list[dict[str, Any]], checkpoint_dir: Path, require: bool) -> dict[str, Any] | None:
    path = checkpoint_dir / "trainer_state.json"
    payload, error = load_json(path)
    if error:
        add(
            checks,
            "training",
            "trainer state",
            "missing" if require else "warn",
            error,
            path=str(path),
        )
        return None
    global_step = int(payload.get("global_step") or 0)
    status = "pass" if global_step > 0 else "incomplete"
    add(
        checks,
        "training",
        "trainer state",
        status,
        "trainer_state.json records a positive global_step"
        if status == "pass"
        else "trainer_state.json exists but global_step is not positive",
        path=str(path),
        global_step=global_step,
        best_model_checkpoint=payload.get("best_model_checkpoint"),
    )
    return payload


def check_training_args(checks: list[dict[str, Any]], checkpoint_dir: Path) -> None:
    path = checkpoint_dir / "training_args.bin"
    if path.exists() and path.stat().st_size > 0:
        add(checks, "training", "training arguments", "pass", "training_args.bin is present", path=str(path), bytes=path.stat().st_size)
    else:
        add(checks, "training", "training arguments", "warn", "training_args.bin is not visible; export may still work", path=str(path))


def torch_envelope(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
    if prefix.startswith(b"PK\x03\x04"):
        return {"format": "torch_zip", "prefix_hex": prefix.hex()}
    if prefix.startswith(b"\x80"):
        return {"format": "pickle", "prefix_hex": prefix.hex()}
    return {"format": "unknown", "prefix_hex": prefix.hex()}


def as_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {}


def extract_modes(state: dict[str, Any]) -> list[dict[str, Any]]:
    modes: list[dict[str, Any]] = []
    for entry in state.get("modelopt_state_dict") or []:
        if isinstance(entry, (list, tuple)) and entry:
            mode = str(entry[0])
            config = as_plain_dict(entry[1]) if len(entry) > 1 else {}
            metadata = as_plain_dict(entry[2]) if len(entry) > 2 else {}
            modes.append({"mode": mode, "config": config, "metadata": metadata})
        elif isinstance(entry, dict):
            mode = str(entry.get("mode") or entry.get("name") or "<unknown>")
            config = as_plain_dict(entry.get("config") or {})
            metadata = as_plain_dict(entry.get("metadata") or {})
            modes.append({"mode": mode, "config": config, "metadata": metadata})
        else:
            modes.append({"mode": f"<unrecognized:{type(entry).__name__}>", "config": {}, "metadata": {}})
    return modes


def load_modelopt_state(path: Path, modelopt_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    sys.path.insert(0, str(modelopt_dir))
    try:
        from modelopt.torch.opt.conversion import load_modelopt_state as modelopt_load_state  # type: ignore
    except Exception:
        try:
            import torch  # type: ignore
        except Exception as exc:
            return None, f"torch/ModelOpt import unavailable: {exc}"
        try:
            return torch.load(path, map_location="cpu", weights_only=False), None
        except TypeError:
            try:
                return torch.load(path, map_location="cpu"), None
            except Exception as exc:
                return None, f"torch.load failed: {exc}"
        except Exception as exc:
            return None, f"torch.load failed: {exc}"
    try:
        return modelopt_load_state(path), None
    except Exception as exc:
        return None, f"ModelOpt load_modelopt_state failed: {exc}"


def check_eagle_state_arch(
    checks: list[dict[str, Any]],
    modes: list[dict[str, Any]],
    reference_arch: Path,
) -> None:
    eagle_modes = [item for item in modes if item.get("mode") == "eagle"]
    if not eagle_modes:
        add(checks, "modelopt", "Eagle mode", "fail", "modelopt_state.pth does not contain an eagle mode", modes=[item.get("mode") for item in modes])
        return
    add(checks, "modelopt", "Eagle mode", "pass", "modelopt_state.pth contains an eagle mode", modes=[item.get("mode") for item in modes])
    ref, error = load_json(reference_arch)
    if error:
        add(checks, "modelopt", "Eagle architecture state", "warn", error, reference_arch=str(reference_arch))
        return
    expected = ref.get("eagle_architecture_config") or {}
    config = eagle_modes[-1].get("config") or {}
    actual = config.get("eagle_architecture_config") or {}
    if not isinstance(actual, dict):
        add(checks, "modelopt", "Eagle architecture state", "fail", "eagle_architecture_config is not a dict")
        return
    keys = [
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
        "use_aux_hidden_state",
        "use_input_layernorm_in_first_layer",
        "use_last_layernorm",
        "eagle_aux_hidden_state_layer_ids",
    ]
    mismatches = {
        key: {"actual": actual.get(key), "expected": expected.get(key)}
        for key in keys
        if expected.get(key) is not None and actual.get(key) != expected.get(key)
    }
    offline = config.get("eagle_offline")
    if offline is not True:
        mismatches["eagle_offline"] = {"actual": offline, "expected": True}
    if mismatches:
        add(
            checks,
            "modelopt",
            "Eagle architecture state",
            "fail",
            "ModelOpt Eagle state does not match the Qwen3-235B Eagle3 reference",
            mismatches=mismatches,
        )
        return
    add(
        checks,
        "modelopt",
        "Eagle architecture state",
        "pass",
        "ModelOpt Eagle state matches the Qwen3-235B Eagle3 reference",
        compared_keys=keys,
        eagle_offline=offline,
    )


def check_modelopt_state(checks: list[dict[str, Any]], checkpoint_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    path = checkpoint_dir / "modelopt_state.pth"
    if not path.exists():
        add(
            checks,
            "modelopt",
            "modelopt_state.pth",
            "missing" if args.require_modelopt_state else "warn",
            f"not visible: {path}",
            path=str(path),
        )
        return []
    size = path.stat().st_size
    envelope = torch_envelope(path)
    if size <= 0:
        add(checks, "modelopt", "modelopt_state.pth", "fail", "modelopt_state.pth is empty", path=str(path), bytes=size)
        return []
    if envelope["format"] == "unknown":
        add(checks, "modelopt", "modelopt_state.pth", "fail", "modelopt_state.pth is not a recognizable torch/pickle file", path=str(path), bytes=size, **envelope)
        return []
    add(checks, "modelopt", "modelopt_state.pth", "pass", "ModelOpt state file is present and non-empty", path=str(path), bytes=size, **envelope)

    state, error = load_modelopt_state(path, args.modelopt_dir)
    if error:
        add(
            checks,
            "modelopt",
            "ModelOpt state load",
            "fail" if args.require_modelopt_state_load else "warn",
            error,
            modelopt_dir=str(args.modelopt_dir),
        )
        return []
    if not isinstance(state, dict):
        add(checks, "modelopt", "ModelOpt state load", "fail", f"loaded state is not a dict: {type(state).__name__}")
        return []
    modes = extract_modes(state)
    if not isinstance(state.get("modelopt_state_dict"), list) or not isinstance(state.get("modelopt_version"), str):
        add(
            checks,
            "modelopt",
            "ModelOpt state schema",
            "fail",
            "loaded state does not have the expected modelopt_state_dict/modelopt_version schema",
            keys=sorted(state.keys()),
        )
        return modes
    add(
        checks,
        "modelopt",
        "ModelOpt state load",
        "pass",
        "ModelOpt state loaded and has the expected schema",
        modelopt_version=state.get("modelopt_version"),
        modes=[item.get("mode") for item in modes],
    )
    check_eagle_state_arch(checks, modes, args.reference_arch)
    return modes


def check_tokenizer_assets(checks: list[dict[str, Any]], checkpoint_dir: Path) -> None:
    candidates = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    present = [name for name in candidates if (checkpoint_dir / name).exists()]
    if present:
        add(checks, "tokenizer", "tokenizer assets", "pass", "tokenizer-related assets are present", present=present)
    else:
        add(
            checks,
            "tokenizer",
            "tokenizer assets",
            "warn",
            "no tokenizer assets found in the checkpoint directory; export loads the model but resume/debug may need tokenizer files",
        )


def check_expected_base_model(checks: list[dict[str, Any]], config: dict[str, Any] | None, expected: str | None) -> None:
    if not expected or config is None:
        return
    name = config.get("_name_or_path") or config.get("name_or_path")
    if not name:
        add(checks, "config", "expected base model", "warn", "config does not record _name_or_path", expected_base_model=expected)
    elif str(name) == expected:
        add(checks, "config", "expected base model", "pass", "config _name_or_path matches expected base model", expected_base_model=expected)
    else:
        add(checks, "config", "expected base model", "warn", "config _name_or_path differs from expected base model", expected_base_model=expected, recorded=name)


def overall_status(checks: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in checks}
    if statuses & {"fail"}:
        return "fail"
    if statuses & {"missing", "incomplete"}:
        return "incomplete"
    return "pass"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Training Checkpoint Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"Checkpoint: `{payload['checkpoint_dir']}`",
        "",
        "| area | check | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['area']} | {check['name']} | {check['status'].upper()} | {str(check['detail']).replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    directory = inspect_directory(checks, args.checkpoint_dir)
    config = check_config(checks, args.checkpoint_dir, args.reference_arch)
    check_expected_base_model(checks, config, args.expected_base_model)
    weights = check_weights(checks, args.checkpoint_dir, args.min_weight_bytes)
    trainer_state = check_trainer_state(checks, args.checkpoint_dir, args.require_trainer_state)
    check_training_args(checks, args.checkpoint_dir)
    modes = check_modelopt_state(checks, args.checkpoint_dir, args)
    check_tokenizer_assets(checks, args.checkpoint_dir)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall_status(checks),
        "checkpoint_dir": str(args.checkpoint_dir),
        "modelopt_dir": str(args.modelopt_dir),
        "reference_arch": str(args.reference_arch),
        "expected_base_model": args.expected_base_model,
        "checkpoint_dirs": directory.get("checkpoint_dirs") or [],
        "weight_files": [str(path) for path in weights],
        "trainer_global_step": int(trainer_state.get("global_step") or 0) if trainer_state else None,
        "modelopt_modes": [item.get("mode") for item in modes],
        "checks": checks,
    }
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if args.fail_on_error and payload["overall_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
