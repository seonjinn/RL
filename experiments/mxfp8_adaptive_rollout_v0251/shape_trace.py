from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_SIGNATURE_FIELDS = ("m", "n_logical", "n_physical", "k", "layout")


def summarize_shape_trace(trace_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") != "mxfp8_dense_shape":
                raise ValueError(f"unexpected event in {path}:{line_number}")
            missing = [field for field in _SIGNATURE_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"missing signature fields in {path}:{line_number}: {missing}"
                )
            records.append(record)

    unique: dict[tuple[int, int, int, int, str], dict[str, Any]] = {}
    for record in records:
        key = (
            int(record["m"]),
            int(record["n_logical"]),
            int(record["n_physical"]),
            int(record["k"]),
            str(record["layout"]),
        )
        unique.setdefault(
            key,
            {
                "m": key[0],
                "n_logical": key[1],
                "n_physical": key[2],
                "k": key[3],
                "layout": key[4],
            },
        )
    signatures = [unique[key] for key in sorted(unique)]
    return {
        "eligible": bool(signatures),
        "record_count": len(records),
        "unique_signature_count": len(signatures),
        "signatures": signatures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize_shape_trace(args.trace_dir)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
