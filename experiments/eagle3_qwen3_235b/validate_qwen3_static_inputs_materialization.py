#!/usr/bin/env python3
"""Validate Qwen3 static-input materialization on a lightweight host."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = None
    if env:
        import os

        merged = os.environ.copy()
        merged.update(env)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def add(checks: list[dict[str, Any]], name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"name": name, "status": status, "detail": detail, "evidence": evidence})


def status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    lines = [
        "# Qwen3 Static Input Materialization Validation",
        "",
        f"Overall: **{str(payload['overall_status']).upper()}**",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['name']} | {str(check['status']).upper()} | {check['detail']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fake_chat_template() -> str:
    return """{%- set ns = namespace(last_query_index=0) -%}
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}
{%- elif message['role'] == 'user' -%}
{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
{%- elif message['role'] == 'assistant' -%}
{{ '<|im_start|>assistant\n' }}
{%- if loop.index0 > ns.last_query_index -%}
{{ message['content'] }}
{%- else -%}
{{ message['content'] }}
{%- endif -%}
{{ '<|im_end|>\n' }}
{%- endif -%}
{%- endfor -%}
"""


def seed_source(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "num_hidden_layers": 8,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "intermediate_size": 4096,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": fake_chat_template()}, indent=2) + "\n",
        encoding="utf-8",
    )
    (source / "generation_config.json").write_text(json.dumps({"temperature": 0.6}) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_materializer(root: Path, checks: list[dict[str, Any]]) -> None:
    source = root / "source"
    artifact = root / "artifact"
    seed_source(source)
    report = artifact / "reports/qwen3_static_inputs.json"
    completed = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/materialize_qwen3_static_inputs.py",
            "--artifact-root",
            str(artifact),
            "--source-dir",
            str(source),
            "--model",
            "Fake/Qwen3",
            "--revision",
            "local",
            "--skip-template-validation",
            "--force",
            "--report-json",
            str(report),
            "--report-markdown",
            str(artifact / "reports/qwen3_static_inputs.md"),
        ]
    )
    problems: list[str] = []
    if completed.returncode != 0:
        problems.append(f"materializer returned {completed.returncode}")
    if not report.exists():
        problems.append("materializer did not write report JSON")
        payload: dict[str, Any] = {}
    else:
        payload = read_json(report)
    for relative in [
        "verifier_config/config.json",
        "verifier_config/tokenizer_config.json",
        "verifier_config/generation_config.json",
        "architecture/eagle3_architecture.json",
        "architecture/eagle3_architecture.env",
        "architecture/eagle3_architecture.dotlist",
        "templates/qwen3_generation_template.jinja2",
    ]:
        if not (artifact / relative).exists():
            problems.append(f"missing output: {relative}")
    if payload.get("overall_status") != "warn":
        problems.append("skip-template-validation run should report overall_status=warn")
    if (payload.get("counts") or {}).get("fail", 0) != 0:
        problems.append("materializer reported failed checks")
    arch_path = artifact / "architecture/eagle3_architecture.json"
    if arch_path.exists():
        arch = read_json(arch_path)
        aux_layers = ((arch.get("eagle_architecture_config") or {}).get("eagle_aux_hidden_state_layer_ids"))
        if aux_layers != [1, 3, 4]:
            problems.append(f"unexpected aux layers: {aux_layers}")
    template_path = artifact / "templates/qwen3_generation_template.jinja2"
    if template_path.exists():
        text = template_path.read_text(encoding="utf-8")
        if "{% generation %}" not in text or "{% endgeneration %}" not in text:
            problems.append("template lacks generation tags")
    add(
        checks,
        "standalone materializer creates static inputs",
        "fail" if problems else "pass",
        "materializer contract failed" if problems else "materializer wrote verifier, architecture, and template outputs",
        problems=problems,
        stdout_tail=completed.stdout[-3000:],
        report=str(report),
    )


def validate_bootstrap_integration(root: Path, checks: list[dict[str, Any]]) -> None:
    source = root / "bootstrap_source"
    artifact = root / "bootstrap_artifact"
    seed_source(source)
    completed = run(
        [
            "bash",
            "experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh",
        ],
        env={
            "ARTIFACT_ROOT": str(artifact),
            "BASE_MODEL": "Fake/Qwen3",
            "PREP_DRY_RUN": "false",
            "RUN_STATIC_INPUT_PREP": "true",
            "STATIC_INPUT_SOURCE_DIR": str(source),
            "STATIC_INPUT_SKIP_TEMPLATE_VALIDATION": "true",
            "RUN_PROVENANCE": "false",
            "RUN_TEMPLATE_PREP": "false",
            "RUN_ARCH_DERIVE": "false",
            "RUN_DATA_PREP": "false",
            "RUN_PREFLIGHT": "false",
            "RUN_PIPELINE": "false",
            "RUN_PIPELINE_SUBMIT_PREFLIGHT": "false",
            "RUN_TRAINING_SCALE_PLAN": "false",
            "RUN_AUDIT": "false",
            "RUN_NEXT_ACTION_PLAN": "false",
        },
    )
    problems: list[str] = []
    if completed.returncode != 0:
        problems.append(f"bootstrap returned {completed.returncode}")
    for relative in [
        "verifier_config/config.json",
        "architecture/eagle3_architecture.env",
        "templates/qwen3_generation_template.jinja2",
        "reports/qwen3_static_inputs.json",
    ]:
        if not (artifact / relative).exists():
            problems.append(f"missing bootstrap output: {relative}")
    add(
        checks,
        "bootstrap can materialize static inputs",
        "fail" if problems else "pass",
        "bootstrap static-input integration failed" if problems else "bootstrap static-input step is wired",
        problems=problems,
        stdout_tail=completed.stdout[-3000:],
        artifact=str(artifact),
    )


def main() -> int:
    args = parse_args()
    root = Path(tempfile.mkdtemp(prefix="eagle3_static_inputs_"))
    checks: list[dict[str, Any]] = []
    try:
        validate_materializer(root, checks)
        validate_bootstrap_integration(root, checks)
    finally:
        if args.keep_temp:
            add(checks, "temporary directory retained", "warn", str(root))
        else:
            shutil.rmtree(root, ignore_errors=True)
    counts = status_counts(checks)
    overall_status = "fail" if counts.get("fail", 0) else "pass"
    payload = {"overall_status": overall_status, "counts": counts, "checks": checks}
    write_json(args.json_out, payload)
    write_markdown(args.markdown_out, payload)
    print((args.markdown_out.read_text(encoding="utf-8") if args.markdown_out else json.dumps(payload, indent=2)))
    return 1 if overall_status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
