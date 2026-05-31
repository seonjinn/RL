#!/usr/bin/env python3
"""Compare an Eagle3 draft config against a verifier/reference config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--verifier-config", type=Path, required=True)
    parser.add_argument("--reference-arch", type=Path, default=None)
    parser.add_argument("--expected-aux-layers", default="1,46,90")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_nested(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def aux_layers(draft: dict[str, Any]) -> list[int] | None:
    value = get_nested(draft, "eagle_config.eagle_aux_hidden_state_layer_ids")
    if value is not None:
        return value
    return draft.get("eagle_aux_hidden_state_layer_ids")


def is_vllm_one_checkpoint(draft: dict[str, Any]) -> bool:
    return (
        draft.get("speculators_model_type") == "eagle3"
        or draft.get("architectures") == ["Eagle3Speculator"]
        or isinstance(draft.get("transformer_layer_config"), dict)
    )


def draft_value(draft: dict[str, Any], key: str, vllm: bool) -> Any:
    if not vllm:
        return draft.get(key)
    if key in {"hidden_size", "vocab_size", "num_hidden_layers"}:
        return get_nested(draft, f"transformer_layer_config.{key}")
    value = get_nested(draft, f"transformer_layer_config.{key}")
    return value if value is not None else draft.get(key)


def check(
    name: str,
    actual: Any,
    expected: Any,
    results: list[dict[str, Any]],
    *,
    required: bool = True,
) -> None:
    if actual is None and not required:
        status = "SKIP"
    else:
        status = "OK" if actual == expected else "MISMATCH"
    results.append({"status": status, "name": name, "actual": actual, "expected": expected})


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compare_configs(
    *,
    draft_config: Path,
    verifier_config: Path,
    reference_arch: Path | None = None,
    expected_aux_layers: str = "1,46,90",
) -> dict[str, Any]:
    draft = load_json(draft_config)
    verifier = load_json(verifier_config)
    reference_arch_path = reference_arch
    reference = load_json(reference_arch_path) if reference_arch_path else {}
    reference_fields = reference.get("eagle_architecture_config", reference)
    vllm = is_vllm_one_checkpoint(draft)

    expected_aux = [int(x) for x in expected_aux_layers.split(",") if x.strip()]
    results: list[dict[str, Any]] = []

    if vllm:
        check("vllm.speculators_model_type", draft.get("speculators_model_type"), "eagle3", results)
        check("vllm.target_hidden_size", draft.get("target_hidden_size"), verifier.get("hidden_size"), results)
        check(
            "vllm.verifier.name_or_path",
            get_nested(draft, "speculators_config.verifier.name_or_path"),
            str(verifier_config),
            results,
        )

    for key in ["hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads"]:
        check(key, draft_value(draft, key, vllm), verifier.get(key), results)
    check("draft_num_hidden_layers", draft_value(draft, "num_hidden_layers", vllm), 1, results)
    check("aux_layers", aux_layers(draft), expected_aux, results, required=not vllm)

    for key in ["intermediate_size", "head_dim", "rms_norm_eps", "rope_theta"]:
        if key in reference_fields:
            check(f"reference.{key}", draft_value(draft, key, vllm), reference_fields[key], results)

    if vllm:
        rope_type = get_nested(draft, "transformer_layer_config.rope_scaling.rope_type")
    else:
        rope_type = get_nested(draft, "rope_scaling.rope_type", get_nested(draft, "rope_scaling.type"))
    ref_rope_type = get_nested(reference_fields, "rope_scaling.rope_type")
    if ref_rope_type is not None:
        check("reference.rope_scaling.rope_type", rope_type, ref_rope_type, results)

    failures = [row for row in results if row["status"] == "MISMATCH"]
    payload = {
        "status": "failed" if failures else "passed",
        "config_kind": "vllm_one_checkpoint" if vllm else "hf_draft",
        "draft_config": str(draft_config),
        "verifier_config": str(verifier_config),
        "reference_arch": str(reference_arch_path) if reference_arch_path else None,
        "checks": results,
        "failure_count": len(failures),
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = compare_configs(
        draft_config=args.draft_config,
        verifier_config=args.verifier_config,
        reference_arch=args.reference_arch,
        expected_aux_layers=args.expected_aux_layers,
    )
    write_json(args.json_out, payload)

    for row in payload["checks"]:
        print(
            f"{row['status']:8} {row['name']}: "
            f"actual={row['actual']!r} expected={row['expected']!r}"
        )

    if payload["failure_count"]:
        raise SystemExit(f"{payload['failure_count']} config checks failed")


if __name__ == "__main__":
    main()
