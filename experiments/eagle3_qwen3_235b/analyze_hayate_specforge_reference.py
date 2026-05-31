#!/usr/bin/env python3
"""Summarize accessible Hayate/Hiso SpecForge artifacts for Qwen3 Eagle3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_SPECFORGE_DIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge"
)
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
EXP_DIR = Path(__file__).resolve().parent
BUNDLED_REFERENCE = EXP_DIR / "hayate_specforge_reference.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specforge-dir", type=Path, default=DEFAULT_SPECFORGE_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument("--architecture-json", type=Path)
    parser.add_argument("--bundled-reference", type=Path, default=BUNDLED_REFERENCE)
    parser.add_argument(
        "--disable-bundled-fallback",
        action="store_true",
        help="Report missing_reference instead of using the repo-captured SpecForge snapshot.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def file_summary(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "status": "missing"}
    return {"path": str(path), "status": "present", "size": stat.st_size}


def config_record(path: Path) -> dict[str, Any]:
    data = load_json(path)
    record = file_summary(path)
    if data is None:
        record["json_status"] = "missing_or_invalid"
        return record
    keys = [
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rope_theta",
        "vocab_size",
        "draft_vocab_size",
    ]
    record["json_status"] = "valid"
    record["fields"] = {key: data.get(key) for key in keys}
    record["eagle_config"] = data.get("eagle_config", {})
    return record


def read_text(path: Path, limit: int = 20000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:limit]


def extract_flags(text: str, flags: list[str]) -> dict[str, str | None]:
    words = text.replace("\\\n", " ").split()
    values: dict[str, str | None] = {}
    for flag in flags:
        values[flag] = None
        for i, word in enumerate(words):
            if word == flag and i + 1 < len(words):
                values[flag] = words[i + 1]
                break
            prefix = f"{flag}="
            if word.startswith(prefix):
                values[flag] = word[len(prefix) :]
                break
    return values


def compare_235b(spec_config: dict[str, Any] | None, verifier: dict[str, Any] | None, arch: dict[str, Any] | None) -> dict[str, Any]:
    if not spec_config or not verifier:
        return {"status": "missing_inputs"}
    spec_aux = (spec_config.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
    arch_aux = None
    if arch:
        arch_aux = (arch.get("eagle_architecture_config") or {}).get("eagle_aux_hidden_state_layer_ids")
    fields = [
        ("aux_layers", spec_aux, arch_aux),
        ("hidden_size", spec_config.get("hidden_size"), verifier.get("hidden_size")),
        ("intermediate_size", spec_config.get("intermediate_size"), verifier.get("intermediate_size")),
        ("max_position_embeddings", spec_config.get("max_position_embeddings"), verifier.get("max_position_embeddings")),
        ("num_attention_heads", spec_config.get("num_attention_heads"), verifier.get("num_attention_heads")),
        ("num_key_value_heads", spec_config.get("num_key_value_heads"), verifier.get("num_key_value_heads")),
        ("rope_theta", spec_config.get("rope_theta"), verifier.get("rope_theta")),
        ("vocab_size", spec_config.get("vocab_size"), verifier.get("vocab_size")),
    ]
    rows = [
        {"field": field, "specforge": spec_value, "current": current_value, "match": spec_value == current_value}
        for field, spec_value, current_value in fields
    ]
    return {
        "status": "reference_only" if any(not row["match"] for row in rows) else "matches_current",
        "rows": rows,
        "conclusion": (
            "SpecForge 235B config is useful for aux-layer sanity, but it is not a direct config source for Thinking-2507."
        ),
    }


def fallback_payload(
    args: argparse.Namespace,
    verifier_config: Path,
    architecture_json: Path,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    comparison = snapshot.get("qwen3_235b_comparison") if isinstance(snapshot.get("qwen3_235b_comparison"), dict) else {}
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "reference_only" if comparison.get("status") in {"reference_only", "matches_current"} else "missing_reference",
        "source": "bundled_reference_snapshot",
        "live_specforge_visible": False,
        "requested_specforge_dir": str(args.specforge_dir),
        "specforge_dir": str(snapshot.get("specforge_dir") or args.specforge_dir),
        "inspected_paths": [{"path": str(args.specforge_dir), "exists": args.specforge_dir.exists()}],
        "bundled_reference": {
            "path": str(args.bundled_reference),
            "generated_at": snapshot.get("generated_at"),
            "specforge_dir": snapshot.get("specforge_dir"),
        },
        "readme_claim": snapshot.get(
            "readme_claim",
            "SpecForge README identifies it as an SGLang ecosystem project for SGLang-compatible draft training.",
        ),
        "configs": snapshot.get("configs") or {},
        "examples": snapshot.get("examples") or {},
        "outputs": snapshot.get("outputs") or {},
        "current_verifier_config": str(verifier_config),
        "current_architecture_json": str(architecture_json),
        "qwen3_235b_comparison": comparison or {"status": "missing_inputs"},
        "warnings": [
            "Live SpecForge path is not visible from this host; using the repo-captured Hayate/Hiso SpecForge snapshot as reference evidence only."
        ],
    }


def main() -> int:
    args = parse_args()
    verifier_config = args.verifier_config or args.artifact_root / "verifier_config/config.json"
    architecture_json = args.architecture_json or args.artifact_root / "architecture/eagle3_architecture.json"

    specforge = args.specforge_dir
    if not specforge.exists() and not args.disable_bundled_fallback:
        snapshot = load_json(args.bundled_reference)
        if snapshot:
            data = fallback_payload(args, verifier_config, architecture_json, snapshot)
            markdown = render_markdown(data)
            if args.json_out:
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if args.markdown_out:
                args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
                args.markdown_out.write_text(markdown, encoding="utf-8")
            print(markdown)
            return 0

    configs = {}
    for name in [
        "qwen3-235B-A22B-eagle3.json",
        "qwen3-30B-A3B-eagle3.json",
        "qwen3-8b-eagle3.json",
        "qwen3-8b-eagle3_long.json",
        "qwen3-coder-480B-A35B-instruct-eagle3.json",
    ]:
        configs[name] = config_record(specforge / "configs" / name)

    examples = {}
    for name in [
        "run_qwen3_moe_eagle3_online.sh",
        "run_qwen3_dense_eagle3_online_30b-moe_large_qwen_data_long.sh",
        "run_qwen3_dense_eagle3_online_8b_dapo.sh",
    ]:
        text = read_text(specforge / "examples" / name)
        examples[name] = {
            "path": str(specforge / "examples" / name),
            "status": "present" if text else "missing",
            "flags": extract_flags(
                text,
                [
                    "--target-model-path",
                    "--draft-model-config",
                    "--train-data-path",
                    "--output-dir",
                    "--num-epochs",
                    "--draft-global-batch-size",
                    "--draft-micro-batch-size",
                    "--batch-size",
                    "--learning-rate",
                    "--max-length",
                    "--chat-template",
                    "--tp-size",
                    "--ttt-length",
                ],
            ),
        }

    outputs = {}
    for rel in [
        "outputs/Qwen3-30B-A3B-eagle3-base/epoch_9",
        "outputs/Qwen3-8B-eagle3-long/epoch_9",
        "outputs/Qwen3-8B-eagle3-long/epoch_9/with-embed",
    ]:
        directory = specforge / rel
        files = {}
        if directory.exists():
            for child in sorted(directory.iterdir()):
                if child.is_file():
                    files[child.name] = file_summary(child)
        outputs[rel] = {"path": str(directory), "status": "present" if directory.exists() else "missing", "files": files}

    verifier = load_json(verifier_config)
    architecture = load_json(architecture_json)
    spec_235b = load_json(specforge / "configs/qwen3-235B-A22B-eagle3.json")
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "reference_only" if specforge.exists() else "missing_reference",
        "source": "live_specforge_path" if specforge.exists() else "missing_reference",
        "live_specforge_visible": specforge.exists(),
        "requested_specforge_dir": str(args.specforge_dir),
        "specforge_dir": str(specforge),
        "inspected_paths": [{"path": str(specforge), "exists": specforge.exists()}],
        "readme_claim": "SpecForge README identifies it as an SGLang ecosystem project for SGLang-compatible draft training.",
        "configs": configs,
        "examples": examples,
        "outputs": outputs,
        "current_verifier_config": str(verifier_config),
        "current_architecture_json": str(architecture_json),
        "qwen3_235b_comparison": compare_235b(spec_235b, verifier, architecture),
    }

    markdown = render_markdown(data)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Hayate SpecForge Reference",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Source: `{data.get('source', 'unknown')}`",
        f"Live SpecForge visible: `{data.get('live_specforge_visible')}`",
        f"SpecForge dir: `{data['specforge_dir']}`",
        "",
        data["readme_claim"],
        "",
    ]
    if data.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in data["warnings"])
        lines.append("")
    lines.extend(
        [
            "## Qwen3-235B Config Comparison",
        "",
        "| field | SpecForge | current verifier/architecture | match |",
        "| --- | --- | --- | --- |",
        ]
    )
    for row in data["qwen3_235b_comparison"].get("rows", []):
        lines.append(f"| {row['field']} | `{row['specforge']}` | `{row['current']}` | {row['match']} |")
    lines.extend(
        [
            "",
            data["qwen3_235b_comparison"].get("conclusion", ""),
            "",
            "## Example Training Flags",
            "",
        ]
    )
    for name, record in data["examples"].items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("| flag | value |")
        lines.append("| --- | --- |")
        for flag, value in record["flags"].items():
            lines.append(f"| `{flag}` | `{value}` |")
        lines.append("")
    lines.extend(["## Output Inventories", ""])
    for rel, record in data["outputs"].items():
        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append(f"Status: `{record['status']}`")
        lines.append("")
        if record["files"]:
            lines.append("| file | size |")
            lines.append("| --- | ---: |")
            for file_name, summary in record["files"].items():
                lines.append(f"| `{file_name}` | {summary.get('size', 0)} |")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
