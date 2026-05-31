#!/usr/bin/env python3
"""Materialize static Qwen3 Eagle3 inputs without launching GPU work.

This prepares the local verifier metadata, Eagle3 architecture overrides, and
answer-only-loss chat template that the later hidden-state/train/export path
expects. It intentionally does not create rollout data or model weights.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import derive_eagle3_architecture
import prepare_qwen3_generation_template


REQUIRED_MODEL_FILES = ("config.json", "tokenizer_config.json")
OPTIONAL_MODEL_FILES = ("generation_config.json",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Thinking-2507")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Use a local directory containing config/tokenizer JSON files instead of HF raw URLs.",
    )
    parser.add_argument("--verifier-config-dir", type=Path)
    parser.add_argument("--architecture-dir", type=Path)
    parser.add_argument("--template-out", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-markdown", type=Path)
    parser.add_argument("--download-timeout", type=float, default=30.0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing static files.")
    parser.add_argument("--aux-layers")
    parser.add_argument("--eagle-decoder-type", choices=("auto", "llama", "kimik2"), default="auto")
    parser.add_argument("--copy-rope-scaling", action="store_true")
    parser.add_argument("--no-template-patch", action="store_true")
    parser.add_argument("--skip-template-validation", action="store_true")
    parser.add_argument("--allow-missing-transformers", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--model-or-tokenizer", help="Tokenizer/model path for full mask validation.")
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(checks: list[dict[str, Any]], name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"name": name, "status": status, "detail": detail, **evidence})


def status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def model_raw_url(model: str, revision: str, filename: str) -> str:
    safe_model = urllib.parse.quote(model, safe="/")
    safe_revision = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/{safe_model}/raw/{safe_revision}/{filename}"


def copy_or_download(
    *,
    filename: str,
    source_dir: Path | None,
    destination_dir: Path,
    model: str,
    revision: str,
    force: bool,
    timeout: float,
    required: bool,
    checks: list[dict[str, Any]],
) -> Path | None:
    destination = destination_dir / filename
    if destination.exists() and destination.stat().st_size > 0 and not force:
        add(
            checks,
            f"{filename} materialized",
            "pass",
            "existing non-empty file kept",
            path=str(destination),
            sha256=sha256(destination),
            bytes=destination.stat().st_size,
        )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source_dir is not None:
            source = source_dir / filename
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copyfile(source, destination)
            source_label = str(source)
        else:
            source_label = model_raw_url(model, revision, filename)
            with urllib.request.urlopen(source_label, timeout=timeout) as response:
                destination.write_bytes(response.read())
        if destination.stat().st_size <= 0:
            raise ValueError("downloaded/copied file is empty")
        add(
            checks,
            f"{filename} materialized",
            "pass",
            f"materialized from {source_label}",
            path=str(destination),
            source=source_label,
            sha256=sha256(destination),
            bytes=destination.stat().st_size,
        )
        return destination
    except (FileNotFoundError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        status = "fail" if required else "warn"
        add(
            checks,
            f"{filename} materialized",
            status,
            f"could not materialize {filename}: {exc}",
            path=str(destination),
        )
        if required:
            return None
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def derive_architecture_outputs(
    config_path: Path,
    architecture_dir: Path,
    args: argparse.Namespace,
    checks: list[dict[str, Any]],
) -> dict[str, Path] | None:
    json_out = architecture_dir / "eagle3_architecture.json"
    env_out = architecture_dir / "eagle3_architecture.env"
    dotlist_out = architecture_dir / "eagle3_architecture.dotlist"
    try:
        cfg = derive_eagle3_architecture.load_json(config_path)
        arch_args = argparse.Namespace(
            aux_layers=args.aux_layers,
            eagle_decoder_type=args.eagle_decoder_type,
            copy_rope_scaling=args.copy_rope_scaling,
            include_hidden_size=True,
        )
        model_type = str(cfg.get("model_type") or "")
        decoder_type = derive_eagle3_architecture.infer_decoder_type(model_type, args.eagle_decoder_type)
        arch = derive_eagle3_architecture.derive_architecture(cfg, arch_args)
        warnings: list[str] = []
        if (
            model_type
            and decoder_type == "llama"
            and model_type.lower() not in derive_eagle3_architecture.LLAMA_LIKE_MODEL_TYPES
        ):
            warnings.append(
                f"model_type={model_type!r} is not in the known LLaMA-like allowlist; "
                "verify ModelOpt supports this verifier before training."
            )
        if derive_eagle3_architecture.get_rope_type(cfg) != "default" and not args.copy_rope_scaling:
            warnings.append(
                "verifier has non-default rope_scaling; generated training arch keeps "
                "rope_type=default and carries only rope_theta, matching the Qwen3 path."
            )
        payload: dict[str, Any] = {
            "source": str(config_path),
            "model_type": model_type or None,
            "eagle_decoder_type": decoder_type,
            "aux_layer_rule": "sorted({1, max(0, num_hidden_layers // 2 - 1), max(0, num_hidden_layers - 4)})",
            "aux_layer_indexing": "0-based transformer layer ids",
            "verifier_summary": {
                "num_hidden_layers": cfg.get("num_hidden_layers"),
                "hidden_size": cfg.get("hidden_size"),
                "num_attention_heads": cfg.get("num_attention_heads"),
                "num_key_value_heads": cfg.get("num_key_value_heads"),
                "intermediate_size": cfg.get("intermediate_size"),
                "rope_theta": derive_eagle3_architecture.get_rope_theta(cfg),
                "rope_scaling_type": derive_eagle3_architecture.get_rope_type(cfg),
            },
            "eagle_architecture_config": arch,
            "warnings": warnings,
            "verifier_identity_fields": {
                "hidden_size": cfg.get("hidden_size"),
                "vocab_size": cfg.get("vocab_size"),
                "max_position_embeddings": cfg.get("max_position_embeddings"),
            },
        }
        write_json(json_out, payload)
        write_text(env_out, derive_eagle3_architecture.render_env(arch, decoder_type, cfg))
        write_text(dotlist_out, derive_eagle3_architecture.render_dotlist(arch, decoder_type))
        status = "warn" if warnings else "pass"
        add(
            checks,
            "Eagle3 architecture derived",
            status,
            "architecture/env/dotlist outputs written" if not warnings else "; ".join(warnings),
            json=str(json_out),
            env=str(env_out),
            dotlist=str(dotlist_out),
            aux_layers=arch.get("eagle_aux_hidden_state_layer_ids"),
        )
        return {"json": json_out, "env": env_out, "dotlist": dotlist_out}
    except Exception as exc:
        add(checks, "Eagle3 architecture derived", "fail", f"architecture derivation failed: {exc}")
        return None


def prepare_template(
    tokenizer_config: Path,
    template_out: Path,
    no_patch: bool,
    checks: list[dict[str, Any]],
) -> bool:
    try:
        data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        template = data.get("chat_template")
        if not isinstance(template, str) or not template:
            raise ValueError(f"{tokenizer_config} has no non-empty chat_template")
        strategy = "copied"
        if not no_patch:
            template, strategy = prepare_qwen3_generation_template.patch_assistant_branch(template)
        template_out.parent.mkdir(parents=True, exist_ok=True)
        template_out.write_text(template, encoding="utf-8")
        if not prepare_qwen3_generation_template.has_generation_tags(template):
            add(
                checks,
                "Qwen3 generation template prepared",
                "fail",
                "template was written but lacks generation/endgeneration tags",
                path=str(template_out),
                strategy=strategy,
            )
            return False
        add(
            checks,
            "Qwen3 generation template prepared",
            "pass",
            "template has generation/endgeneration tags",
            path=str(template_out),
            strategy=strategy,
            bytes=template_out.stat().st_size,
            sha256=sha256(template_out),
        )
        return True
    except Exception as exc:
        add(checks, "Qwen3 generation template prepared", "fail", f"template preparation failed: {exc}")
        return False


def run_template_validation(
    *,
    script_dir: Path,
    model_or_tokenizer: str,
    template_out: Path,
    report_json: Path,
    args: argparse.Namespace,
    checks: list[dict[str, Any]],
) -> None:
    if args.skip_template_validation:
        add(
            checks,
            "assistant mask validation",
            "warn",
            "full Transformers mask validation skipped by request",
            path=str(report_json),
        )
        return
    command = [
        sys.executable,
        str(script_dir / "validate_chat_template_loss_mask.py"),
        "--model-or-tokenizer",
        model_or_tokenizer,
        "--chat-template",
        str(template_out),
        "--json-out",
        str(report_json),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.allow_missing_transformers:
        command.append("--allow-missing-transformers")
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    payload: dict[str, Any] = {}
    if report_json.exists():
        try:
            payload = json.loads(report_json.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    validation_status = str(payload.get("status") or "").lower()
    if completed.returncode == 0 and validation_status == "pass":
        status = "pass"
        detail = "Transformers assistant-token mask validation passed"
    elif completed.returncode == 0 and validation_status == "warning":
        status = "warn"
        detail = str(payload.get("error") or payload.get("reason") or "assistant mask validation warning")
    else:
        status = "fail"
        detail = f"assistant mask validation failed with returncode {completed.returncode}"
    add(
        checks,
        "assistant mask validation",
        status,
        detail,
        path=str(report_json),
        command=" ".join(command),
        stdout_tail=completed.stdout[-2000:],
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 Static Input Materialization",
        "",
        f"Overall: **{str(payload['overall_status']).upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"Artifact root: `{payload['artifact_root']}`",
        f"Model: `{payload['model']}`",
        f"Revision: `{payload['revision']}`",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in payload["checks"]:
        detail = str(check.get("detail") or "").replace("\n", " ")
        lines.append(f"| {check.get('name')} | {str(check.get('status')).upper()} | {detail} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- verifier config dir: `{payload['outputs']['verifier_config_dir']}`",
            f"- architecture JSON: `{payload['outputs']['architecture_json']}`",
            f"- architecture env: `{payload['outputs']['architecture_env']}`",
            f"- architecture dotlist: `{payload['outputs']['architecture_dotlist']}`",
            f"- chat template: `{payload['outputs']['chat_template']}`",
            f"- template mask validation: `{payload['outputs']['template_mask_validation_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root
    verifier_config_dir = args.verifier_config_dir or artifact_root / "verifier_config"
    architecture_dir = args.architecture_dir or artifact_root / "architecture"
    template_out = args.template_out or artifact_root / "templates/qwen3_generation_template.jinja2"
    report_json = args.report_json or artifact_root / "reports/qwen3_static_inputs.json"
    report_markdown = args.report_markdown or artifact_root / "reports/qwen3_static_inputs.md"
    template_validation_json = template_out.with_suffix(".mask_validation.json")
    script_dir = Path(__file__).resolve().parent

    checks: list[dict[str, Any]] = []
    for filename in REQUIRED_MODEL_FILES:
        copy_or_download(
            filename=filename,
            source_dir=args.source_dir,
            destination_dir=verifier_config_dir,
            model=args.model,
            revision=args.revision,
            force=args.force,
            timeout=args.download_timeout,
            required=True,
            checks=checks,
        )
    for filename in OPTIONAL_MODEL_FILES:
        copy_or_download(
            filename=filename,
            source_dir=args.source_dir,
            destination_dir=verifier_config_dir,
            model=args.model,
            revision=args.revision,
            force=args.force,
            timeout=args.download_timeout,
            required=False,
            checks=checks,
        )

    config_path = verifier_config_dir / "config.json"
    tokenizer_config_path = verifier_config_dir / "tokenizer_config.json"
    architecture_outputs = None
    if config_path.exists() and config_path.stat().st_size > 0:
        architecture_outputs = derive_architecture_outputs(config_path, architecture_dir, args, checks)
    else:
        add(checks, "Eagle3 architecture derived", "fail", f"missing verifier config: {config_path}")

    if tokenizer_config_path.exists() and tokenizer_config_path.stat().st_size > 0:
        template_ready = prepare_template(tokenizer_config_path, template_out, args.no_template_patch, checks)
    else:
        template_ready = False
        add(checks, "Qwen3 generation template prepared", "fail", f"missing tokenizer config: {tokenizer_config_path}")

    if template_ready:
        model_or_tokenizer = args.model_or_tokenizer or args.model
        run_template_validation(
            script_dir=script_dir,
            model_or_tokenizer=model_or_tokenizer,
            template_out=template_out,
            report_json=template_validation_json,
            args=args,
            checks=checks,
        )

    counts = status_counts(checks)
    if counts.get("fail", 0):
        overall_status = "fail"
    elif counts.get("warn", 0):
        overall_status = "warn"
    else:
        overall_status = "pass"
    payload = {
        "overall_status": overall_status,
        "generated_at": utc_now(),
        "artifact_root": str(artifact_root),
        "model": args.model,
        "revision": args.revision,
        "source_dir": str(args.source_dir) if args.source_dir else None,
        "counts": counts,
        "outputs": {
            "verifier_config_dir": str(verifier_config_dir),
            "architecture_json": str((architecture_outputs or {}).get("json", architecture_dir / "eagle3_architecture.json")),
            "architecture_env": str((architecture_outputs or {}).get("env", architecture_dir / "eagle3_architecture.env")),
            "architecture_dotlist": str((architecture_outputs or {}).get("dotlist", architecture_dir / "eagle3_architecture.dotlist")),
            "chat_template": str(template_out),
            "template_mask_validation_json": str(template_validation_json),
        },
        "checks": checks,
    }
    write_json(report_json, payload)
    write_text(report_markdown, render_markdown(payload))
    print(render_markdown(payload))
    return 1 if overall_status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
