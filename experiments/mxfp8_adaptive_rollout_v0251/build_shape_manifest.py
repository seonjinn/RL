from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypedDict, cast

_EVENT = "mxfp8_dense_shape"
_LAYOUTS = ("8x4", "128x4")
_REQUIRED_FIELDS = (
    "event",
    "family",
    "hostname",
    "k",
    "layout",
    "m",
    "n_logical",
    "n_physical",
    "pid",
    "prefix",
)


class ShapeTraceError(ValueError):
    pass


class ProvenanceEntry(TypedDict):
    source_file: str
    hostname: str
    pid: int
    prefix: str
    family: str
    record_count: int


class SignatureEntry(TypedDict):
    m: int
    n_logical: int
    n_physical: int
    k: int
    layout: str
    record_count: int
    prefixes: list[str]
    provenance: list[ProvenanceEntry]


class LayoutCount(TypedDict):
    record_count: int
    unique_signature_count: int


class ShapeManifest(TypedDict):
    schema_version: int
    input_file_count: int
    record_count: int
    unique_signature_count: int
    duplicate_record_count: int
    layout_counts: dict[str, LayoutCount]
    signatures: list[SignatureEntry]


@dataclass(frozen=True, order=True)
class _Signature:
    m: int
    n_logical: int
    n_physical: int
    k: int
    layout: str


@dataclass(frozen=True, order=True)
class _Provenance:
    source_file: str
    hostname: str
    pid: int
    prefix: str
    family: str


@dataclass(frozen=True)
class _Record:
    signature: _Signature
    provenance: _Provenance


def _error(location: str, message: str) -> ShapeTraceError:
    return ShapeTraceError(f"{location}: {message}")


def _positive_int(record: dict[str, object], field: str, location: str) -> int:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(location, f"{field} must be a positive integer, got {value!r}")
    return value


def _nonempty_string(record: dict[str, object], field: str, location: str) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise _error(location, f"{field} must be a non-empty string, got {value!r}")
    return value


def _parse_record(value: object, source_file: str, location: str) -> _Record:
    if not isinstance(value, dict):
        raise _error(location, "record must be a JSON object")
    record = cast(dict[str, object], value)
    missing = [field for field in _REQUIRED_FIELDS if field not in record]
    if missing:
        raise _error(location, f"missing required fields: {', '.join(missing)}")

    event = _nonempty_string(record, "event", location)
    if event != _EVENT:
        raise _error(location, f"unexpected event {event!r}; expected {_EVENT!r}")

    family = _nonempty_string(record, "family", location)
    hostname = _nonempty_string(record, "hostname", location)
    prefix = _nonempty_string(record, "prefix", location)
    layout = _nonempty_string(record, "layout", location)
    if layout not in _LAYOUTS:
        raise _error(
            location,
            f"unsupported layout {layout!r}; expected one of {', '.join(_LAYOUTS)}",
        )

    m = _positive_int(record, "m", location)
    n_logical = _positive_int(record, "n_logical", location)
    n_physical = _positive_int(record, "n_physical", location)
    k = _positive_int(record, "k", location)
    pid = _positive_int(record, "pid", location)
    if n_physical < n_logical:
        raise _error(
            location,
            "n_physical must be greater than or equal to n_logical, "
            f"got {n_physical} < {n_logical}",
        )
    if k % 32 != 0:
        raise _error(location, f"k must be divisible by 32 for MXFP8, got {k}")

    return _Record(
        signature=_Signature(m, n_logical, n_physical, k, layout),
        provenance=_Provenance(source_file, hostname, pid, prefix, family),
    )


def _source_labels(paths: Sequence[Path]) -> dict[Path, str]:
    common_parent = Path(
        os.path.commonpath([str(path.resolve().parent) for path in paths])
    )
    return {
        path: path.resolve().relative_to(common_parent).as_posix() for path in paths
    }


def _validated_paths(trace_paths: Iterable[Path]) -> tuple[Path, ...]:
    paths = tuple(Path(path) for path in trace_paths)
    if not paths:
        raise ShapeTraceError("at least one trace JSONL file is required")

    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            raise ShapeTraceError(f"duplicate input trace file: {path}")
        seen.add(resolved)
        if not path.is_file():
            raise ShapeTraceError(
                f"input trace file does not exist or is not a file: {path}"
            )
    return paths


def build_shape_manifest(trace_paths: Iterable[Path]) -> ShapeManifest:
    paths = _validated_paths(trace_paths)
    source_labels = _source_labels(paths)
    signature_counts: Counter[_Signature] = Counter()
    signature_prefixes: defaultdict[_Signature, set[str]] = defaultdict(set)
    provenance_counts: defaultdict[_Signature, Counter[_Provenance]] = defaultdict(
        Counter
    )
    prefix_metadata: dict[str, tuple[str, int, int, int]] = {}

    for path in sorted(paths, key=lambda item: source_labels[item]):
        source_file = source_labels[path]
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ShapeTraceError(f"{path}: trace file is not valid UTF-8") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _error(location, f"invalid JSON: {exc.msg}") from exc
            parsed = _parse_record(value, source_file, location)
            signature = parsed.signature
            provenance = parsed.provenance

            metadata = (
                provenance.family,
                signature.n_logical,
                signature.n_physical,
                signature.k,
            )
            previous = prefix_metadata.setdefault(provenance.prefix, metadata)
            if previous != metadata:
                raise _error(
                    location,
                    f"inconsistent dimensions for prefix {provenance.prefix!r}: "
                    f"expected {previous}, got {metadata}",
                )

            signature_counts[signature] += 1
            signature_prefixes[signature].add(provenance.prefix)
            provenance_counts[signature][provenance] += 1

    signatures: list[SignatureEntry] = []
    for signature in sorted(signature_counts):
        provenance = [
            ProvenanceEntry(
                source_file=item.source_file,
                hostname=item.hostname,
                pid=item.pid,
                prefix=item.prefix,
                family=item.family,
                record_count=count,
            )
            for item, count in sorted(provenance_counts[signature].items())
        ]
        signatures.append(
            SignatureEntry(
                m=signature.m,
                n_logical=signature.n_logical,
                n_physical=signature.n_physical,
                k=signature.k,
                layout=signature.layout,
                record_count=signature_counts[signature],
                prefixes=sorted(signature_prefixes[signature]),
                provenance=provenance,
            )
        )

    layout_counts: dict[str, LayoutCount] = {}
    for layout in _LAYOUTS:
        layout_signatures = [item for item in signatures if item["layout"] == layout]
        if layout_signatures:
            layout_counts[layout] = LayoutCount(
                record_count=sum(item["record_count"] for item in layout_signatures),
                unique_signature_count=len(layout_signatures),
            )

    record_count = sum(signature_counts.values())
    return ShapeManifest(
        schema_version=1,
        input_file_count=len(paths),
        record_count=record_count,
        unique_signature_count=len(signatures),
        duplicate_record_count=record_count - len(signatures),
        layout_counts=layout_counts,
        signatures=signatures,
    )


def write_shape_outputs(
    manifest: ShapeManifest, output_path: Path, shmoo_dir: Path
) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")

    shmoo_dir.mkdir(parents=True, exist_ok=True)
    for layout in _LAYOUTS:
        rows = sorted(
            {
                (entry["m"], entry["n_physical"], entry["k"])
                for entry in manifest["signatures"]
                if entry["layout"] == layout
            }
        )
        path = shmoo_dir / f"shapes_{layout}.txt"
        if rows:
            path.write_text(
                ";".join(f"{m},{n_physical},{k}" for m, n_physical, k in rows) + "\n",
                encoding="utf-8",
            )
        elif path.exists():
            path.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize dense MXFP8 JSONL traces into exact Qwen shapes."
    )
    parser.add_argument("trace_files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shmoo-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_shape_manifest(args.trace_files)
    write_shape_outputs(manifest, args.output, args.shmoo_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
