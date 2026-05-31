#!/usr/bin/env python3
"""Validate exported HF and vLLM Eagle3 draft artifacts.

This check runs after ModelOpt export/conversion and before the trained-draft
RL smoke sweep. It is filesystem-only: no Slurm, no GPU, and no model loading.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from compare_eagle3_configs import compare_configs, load_json


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--vllm-draft-dir", type=Path, required=True)
    parser.add_argument("--verifier-config-dir", type=Path, required=True)
    parser.add_argument("--reference-arch", type=Path, default=EXP / "qwen3_235b_thinking_eagle3_architecture.json")
    parser.add_argument("--export-config-compare-json", type=Path)
    parser.add_argument("--vllm-config-compare-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def read_json_if_present(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "not provided"
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def check_safetensors_structure(paths: list[Path], label: str, checks: list[dict[str, Any]]) -> None:
    if not paths:
        return
    parsed = [{"name": path.name, **parse_safetensors_header(path)} for path in paths]
    failures = [item for item in parsed if item.get("status") != "pass"]
    if failures:
        add(
            checks,
            "safetensors",
            label,
            "fail",
            "one or more safetensors files have invalid headers, offsets, or tensor metadata",
            failures=failures[:8],
            checked=len(parsed),
        )
        return
    add(
        checks,
        "safetensors",
        label,
        "pass",
        "safetensors headers, offsets, and tensor metadata are valid",
        checked=len(parsed),
        tensors=sum(int(item.get("tensor_count") or 0) for item in parsed),
        files=parsed[:8],
    )


def check_config_file(path: Path, label: str, checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not path.exists():
        add(checks, "config", label, "missing", f"missing {path}", path=str(path))
        return None
    try:
        payload = load_json(path)
    except Exception as exc:
        add(checks, "config", label, "fail", f"cannot parse JSON: {exc}", path=str(path))
        return None
    add(checks, "config", label, "pass", "config JSON is readable", path=str(path))
    return payload


def check_weights(directory: Path, label: str, checks: list[dict[str, Any]], *, require_safetensors: bool) -> None:
    if not directory.exists():
        add(checks, "weights", label, "missing", f"directory is not visible: {directory}", path=str(directory))
        return
    safetensors = sorted(directory.glob("*.safetensors"))
    bin_weights = sorted(directory.glob("pytorch_model*.bin")) if not require_safetensors else []
    weights = safetensors + bin_weights
    summary = file_summary(weights)
    if not weights:
        expected = "*.safetensors" if require_safetensors else "*.safetensors or pytorch_model*.bin"
        add(checks, "weights", label, "missing", f"no weight files matching {expected}", path=str(directory))
    elif summary["total_bytes"] <= 0:
        add(checks, "weights", label, "fail", "weight files exist but total byte size is zero", **summary)
    else:
        add(checks, "weights", label, "pass", "weight files are present and non-empty", **summary)
        check_safetensors_structure(safetensors, f"{label} safetensors structure", checks)


def check_compare(
    *,
    draft_config: Path,
    verifier_config: Path,
    reference_arch: Path,
    compare_json: Path | None,
    expected_kind: str,
    label: str,
    checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        payload = compare_configs(
            draft_config=draft_config,
            verifier_config=verifier_config,
            reference_arch=reference_arch,
        )
    except Exception as exc:
        add(checks, "config_compare", label, "fail", f"cannot compute config comparison: {exc}", compare_json=str(compare_json) if compare_json else None)
        return None

    provided_payload, error = read_json_if_present(compare_json)
    source = "computed"
    if provided_payload is not None:
        source = f"computed; provided={compare_json}"
        provided_failures = int(provided_payload.get("failure_count") or 0)
        if provided_payload.get("status") != "passed" or provided_failures:
            add(
                checks,
                "config_compare",
                f"{label} recorded report",
                "fail",
                "provided config comparison JSON is not pass-level",
                compare_json=str(compare_json),
                status=provided_payload.get("status"),
                failure_count=provided_failures,
            )
        elif provided_payload.get("config_kind") != expected_kind:
            add(
                checks,
                "config_compare",
                f"{label} recorded report",
                "fail",
                "provided config comparison JSON has the wrong config kind",
                compare_json=str(compare_json),
                config_kind=provided_payload.get("config_kind"),
                expected_kind=expected_kind,
            )
    elif compare_json is not None and error:
        try:
            write_json(compare_json, payload)
            provided_payload = payload
            source = f"computed; wrote={compare_json}"
            add(
                checks,
                "config_compare",
                f"{label} recorded report",
                "pass",
                "wrote computed config comparison JSON",
                compare_json=str(compare_json),
                report_status=payload.get("status"),
                failure_count=int(payload.get("failure_count") or 0),
            )
        except Exception as exc:
            add(
                checks,
                "config_compare",
                f"{label} recorded report",
                "fail",
                f"cannot write computed config comparison JSON: {exc}",
                compare_json=str(compare_json),
                original_error=error,
            )

    failure_count = int(payload.get("failure_count") or 0)
    status = payload.get("status")
    kind = payload.get("config_kind")
    if status == "passed" and failure_count == 0 and kind == expected_kind:
        add(checks, "config_compare", label, "pass", "config comparison passed", source=source, config_kind=kind, check_count=len(payload.get("checks") or []))
    elif status == "passed" and failure_count == 0:
        add(checks, "config_compare", label, "fail", "config comparison kind does not match expected artifact type", source=source, config_kind=kind, expected_kind=expected_kind)
    else:
        add(checks, "config_compare", label, "fail", "config comparison failed", source=source, status=status, failure_count=failure_count, config_kind=kind)
    return payload


def check_vllm_contract(config: dict[str, Any] | None, checks: list[dict[str, Any]]) -> None:
    if config is None:
        return
    problems: list[str] = []
    if config.get("speculators_model_type") != "eagle3":
        problems.append("speculators_model_type is not eagle3")
    if config.get("architectures") != ["Eagle3Speculator"]:
        problems.append("architectures is not ['Eagle3Speculator']")
    if not isinstance(config.get("transformer_layer_config"), dict):
        problems.append("transformer_layer_config is missing")
    verifier = (config.get("speculators_config") or {}).get("verifier") or {}
    if not verifier.get("name_or_path"):
        problems.append("speculators_config.verifier.name_or_path is missing")
    if config.get("target_hidden_size") is None:
        problems.append("target_hidden_size is missing")

    if problems:
        add(checks, "vllm_contract", "vLLM one-checkpoint Eagle3 contract", "fail", "; ".join(problems))
    else:
        add(
            checks,
            "vllm_contract",
            "vLLM one-checkpoint Eagle3 contract",
            "pass",
            "vLLM config advertises an Eagle3 speculator with verifier metadata",
            verifier_name_or_path=verifier.get("name_or_path"),
            target_hidden_size=config.get("target_hidden_size"),
        )


def check_hf_contract(config: dict[str, Any] | None, checks: list[dict[str, Any]]) -> None:
    if config is None:
        return
    problems: list[str] = []
    if config.get("num_hidden_layers") != 1:
        problems.append("num_hidden_layers is not 1")
    if config.get("hidden_size") is None:
        problems.append("hidden_size is missing")
    if config.get("vocab_size") is None:
        problems.append("vocab_size is missing")
    if not (config.get("eagle_aux_hidden_state_layer_ids") or (config.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")):
        problems.append("aux hidden-state layer ids are missing")
    if problems:
        add(checks, "hf_contract", "HF Eagle3 draft contract", "fail", "; ".join(problems))
    else:
        add(checks, "hf_contract", "HF Eagle3 draft contract", "pass", "HF draft config has one draft layer and Eagle3 aux-layer metadata")


def overall_status(checks: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in checks}
    if statuses & {"fail"}:
        return "fail"
    if statuses & {"missing", "incomplete"}:
        return "incomplete"
    return "pass"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Export Artifact Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
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

    verifier_config = args.verifier_config_dir / "config.json"
    export_config = args.export_dir / "config.json"
    vllm_config = args.vllm_draft_dir / "config.json"

    check_config_file(verifier_config, "verifier config", checks)
    check_config_file(args.reference_arch, "reference architecture", checks)
    hf_config = check_config_file(export_config, "HF exported draft config", checks)
    vllm_payload = check_config_file(vllm_config, "vLLM draft config", checks)

    check_weights(args.export_dir, "HF exported draft weights", checks, require_safetensors=False)
    check_weights(args.vllm_draft_dir, "vLLM draft safetensors", checks, require_safetensors=True)
    check_hf_contract(hf_config, checks)
    check_vllm_contract(vllm_payload, checks)

    export_compare = None
    vllm_compare = None
    if export_config.exists() and verifier_config.exists() and args.reference_arch.exists():
        export_compare = check_compare(
            draft_config=export_config,
            verifier_config=args.verifier_config_dir,
            reference_arch=args.reference_arch,
            compare_json=args.export_config_compare_json,
            expected_kind="hf_draft",
            label="HF export config comparison",
            checks=checks,
        )
    if vllm_config.exists() and verifier_config.exists() and args.reference_arch.exists():
        vllm_compare = check_compare(
            draft_config=vllm_config,
            verifier_config=args.verifier_config_dir,
            reference_arch=args.reference_arch,
            compare_json=args.vllm_config_compare_json,
            expected_kind="vllm_one_checkpoint",
            label="vLLM draft config comparison",
            checks=checks,
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall_status(checks),
        "export_dir": str(args.export_dir),
        "vllm_draft_dir": str(args.vllm_draft_dir),
        "verifier_config_dir": str(args.verifier_config_dir),
        "reference_arch": str(args.reference_arch),
        "export_config_compare_json": str(args.export_config_compare_json) if args.export_config_compare_json else None,
        "vllm_config_compare_json": str(args.vllm_config_compare_json) if args.vllm_config_compare_json else None,
        "config_comparisons": {
            "export": export_compare,
            "vllm": vllm_compare,
        },
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
