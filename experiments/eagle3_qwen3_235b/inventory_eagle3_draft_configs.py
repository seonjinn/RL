#!/usr/bin/env python3
"""Inventory Eagle3 draft model configs and compare them to a reference arch."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_KEYS = (
    "hidden_size",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "head_dim",
    "rms_norm_eps",
    "rope_theta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Draft config files or directories")
    parser.add_argument(
        "--reference-arch",
        type=Path,
        default=Path("experiments/eagle3_qwen3_235b/qwen3_235b_thinking_eagle3_architecture.json"),
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser.parse_args()


def get_nested(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def iter_config_files(roots: Iterable[Path], max_depth: int) -> tuple[list[Path], list[dict[str, str]], list[dict[str, Any]]]:
    files: list[Path] = []
    seen: set[Path] = set()
    warnings: list[dict[str, str]] = []
    root_statuses: list[dict[str, Any]] = []

    def add(path: Path) -> None:
        try:
            if not path.is_file() or path.name != "config.json":
                return
        except OSError as exc:
            warnings.append({"path": str(path), "error": str(exc)})
            return
        try:
            resolved = path.resolve()
        except OSError as exc:
            warnings.append({"path": str(path), "error": str(exc)})
            resolved = path
        if resolved not in seen:
            seen.add(resolved)
            files.append(path)

    for root in roots:
        try:
            if root.is_file():
                root_statuses.append({"path": str(root), "status": "file"})
                add(root)
                continue
            if not root.is_dir():
                exists = root.exists()
                status = "unsupported" if exists else "missing"
                message = "root is not a file or directory" if exists else "root is not visible"
                root_statuses.append({"path": str(root), "status": status, "error": message})
                warnings.append({"path": str(root), "error": message})
                continue
        except OSError as exc:
            root_statuses.append({"path": str(root), "status": "error", "error": str(exc)})
            warnings.append({"path": str(root), "error": str(exc)})
            continue
        root_statuses.append({"path": str(root), "status": "directory"})

        try:
            root_depth = len(root.resolve().parts)
        except OSError as exc:
            warnings.append({"path": str(root), "error": str(exc)})
            root_depth = len(root.parts)

        def on_walk_error(exc: OSError) -> None:
            warnings.append({"path": getattr(exc, "filename", str(root)), "error": str(exc)})

        for dirpath, dirnames, filenames in os.walk(root, onerror=on_walk_error):
            path = Path(dirpath)
            try:
                depth = len(path.resolve().parts) - root_depth
            except OSError as exc:
                warnings.append({"path": str(path), "error": str(exc)})
                depth = max_depth + 1
            if depth >= max_depth:
                dirnames[:] = []
            if depth <= max_depth and "config.json" in filenames:
                add(path / "config.json")
    return files, warnings, root_statuses


def aux_layers(config: dict[str, Any]) -> Any:
    return (
        get_nested(config, "eagle_config.eagle_aux_hidden_state_layer_ids")
        or get_nested(config, "eagle_architecture_config.eagle_aux_hidden_state_layer_ids")
        or config.get("eagle_aux_hidden_state_layer_ids")
    )


def source_model(config: dict[str, Any]) -> Any:
    candidates = [
        "base_model_name_or_path",
        "model_name_or_path",
        "_name_or_path",
        "architectures",
        "eagle_config.verifier",
        "eagle_config.base_model",
        "verifier",
    ]
    for key in candidates:
        value = get_nested(config, key) if "." in key else config.get(key)
        if value not in (None, "", []):
            return value
    return None


def reference_config(path: Path) -> dict[str, Any]:
    data = load_json(path)
    return data.get("eagle_architecture_config", data)


def summarize_config(path: Path, reference: dict[str, Any]) -> dict[str, Any]:
    try:
        cfg = load_json(path)
    except Exception as exc:
        return {"path": str(path), "status": "error", "error": str(exc)}

    summary: dict[str, Any] = {
        "path": str(path),
        "status": "ok",
        "parent": str(path.parent),
        "source_model": source_model(cfg),
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "aux_layers": aux_layers(cfg),
        "rope_scaling": cfg.get("rope_scaling", get_nested(cfg, "eagle_config.rope_scaling")),
    }
    for key in DEFAULT_KEYS:
        summary[key] = cfg.get(key, get_nested(cfg, f"eagle_config.{key}"))

    mismatches: dict[str, dict[str, Any]] = {}
    for key in (
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
    ):
        expected = reference.get(key)
        if expected is not None and summary.get(key) != expected:
            mismatches[key] = {"actual": summary.get(key), "expected": expected}
    expected_aux = reference.get("eagle_aux_hidden_state_layer_ids")
    if expected_aux is not None and summary["aux_layers"] != expected_aux:
        mismatches["aux_layers"] = {"actual": summary["aux_layers"], "expected": expected_aux}
    summary["reference_mismatches"] = mismatches
    summary["matches_reference"] = not mismatches
    return summary


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Draft Config Inventory",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        f"Configs scanned: **{payload['configs_scanned']}**",
        "",
    ]
    if payload.get("warnings"):
        lines += ["## Warnings", ""]
        for warning in payload["warnings"]:
            lines.append(f"- `{warning['path']}`: {warning['error']}")
        lines.append("")
    if payload.get("root_statuses"):
        lines += [
            "## Roots",
            "",
            "| status | root | detail |",
            "| --- | --- | --- |",
        ]
        for root in payload["root_statuses"]:
            detail = root.get("error") or "-"
            lines.append(f"| {root.get('status')} | `{root.get('path')}` | {detail} |")
        lines.append("")
    lines += [
        "| match | layers | heads | kv | mlp | rope | aux | config |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in payload["configs"]:
        if item.get("status") != "ok":
            lines.append(f"| ERROR | - | - | - | - | - | - | `{item['path']}` |")
            continue
        match = "yes" if item.get("matches_reference") else "no"
        lines.append(
            f"| {match} | {item.get('num_hidden_layers')} | {item.get('num_attention_heads')} | "
            f"{item.get('num_key_value_heads')} | {item.get('intermediate_size')} | "
            f"{item.get('rope_theta')} | `{item.get('aux_layers')}` | `{item['path']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    reference = reference_config(args.reference_arch)
    files, warnings, root_statuses = iter_config_files(args.roots, args.max_depth)
    configs = [summarize_config(path, reference) for path in files]
    configs.sort(key=lambda item: (not item.get("matches_reference", False), item.get("path", "")))
    if configs and not warnings:
        overall_status = "pass"
        recommendation = "Use matching configs as architecture references and inspect mismatches before reuse."
    elif configs:
        overall_status = "warn"
        recommendation = "Some roots could not be scanned; use visible configs only as reference material."
    elif warnings:
        overall_status = "warn"
        recommendation = "Draft artifact roots were inaccessible or unreadable; keep Hayate artifacts as non-blocking reference evidence."
    else:
        overall_status = "missing"
        recommendation = "No draft config.json files were found under the requested roots."
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall_status,
        "recommendation": recommendation,
        "roots": [str(root) for root in args.roots],
        "root_statuses": root_statuses,
        "reference_arch": str(args.reference_arch),
        "configs_scanned": len(configs),
        "warnings": warnings,
        "configs": configs,
    }

    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
