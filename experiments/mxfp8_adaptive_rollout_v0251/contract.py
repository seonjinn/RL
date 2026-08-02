from __future__ import annotations

import argparse
import base64
import hashlib
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Arm = Literal["baseline", "trace", "adaptive"]


@dataclass(frozen=True)
class AdaptiveInputs:
    tactic_file: Path
    tactic_sha256: str
    layer_allowlist_b64: str
    switch_m: int = 256


@dataclass(frozen=True)
class TraceInputs:
    trace_dir: Path
    trace_max: int = 8192


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_base64(value: str) -> None:
    try:
        decoded = base64.b64decode(value, validate=True).decode("ascii")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("layer allowlist must be base64-encoded ASCII") from error
    if not decoded.strip():
        raise ValueError("layer allowlist must not be empty")


def build_arm_environment(
    arm: Arm,
    *,
    runtime_root: Path,
    adaptive: AdaptiveInputs | None = None,
    trace: TraceInputs | None = None,
) -> dict[str, str]:
    runtime_root = runtime_root.resolve()
    if not (runtime_root / "vllm").is_dir():
        raise ValueError(f"vLLM runtime overlay is not a directory: {runtime_root}")

    env = {
        "PYTHONPATH": str(runtime_root),
        "VLLM_SUBPROCESS_PYTHONPATH": str(runtime_root),
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": "PYTHONPATH",
        "VLLM_FLASHINFER_MOE_BACKEND": "latency",
        "VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK": "1",
        "NEMORL_MXFP8_LINEAR_BACKEND": "flashinfer_cutedsl",
    }
    if arm == "baseline":
        if adaptive is not None or trace is not None:
            raise ValueError("baseline arm must not receive adaptive or trace inputs")
        return env
    if arm == "trace":
        if adaptive is not None:
            raise ValueError("trace arm must not receive adaptive inputs")
        if trace is None:
            raise ValueError("trace arm requires trace inputs")
        if trace.trace_max <= 0:
            raise ValueError("trace_max must be positive")
        env.update(
            {
                "NEMORL_MXFP8_LINEAR_BACKEND": "flashinfer_trtllm",
                "VLLM_MXFP8_DENSE_SHAPE_TRACE": "1",
                "VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR": str(
                    trace.trace_dir.resolve()
                ),
                "VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX": str(trace.trace_max),
            }
        )
        return env
    if arm != "adaptive":
        raise ValueError(f"unsupported arm: {arm}")
    if trace is not None:
        raise ValueError("adaptive arm must not receive trace inputs")
    if adaptive is None:
        raise ValueError("adaptive arm requires a tactic table and allowlist")
    if adaptive.switch_m <= 0:
        raise ValueError("adaptive switch_m must be positive")

    tactic_file = adaptive.tactic_file.resolve()
    if not tactic_file.is_file():
        raise ValueError(f"tactic table is not a file: {tactic_file}")
    actual_sha256 = _sha256(tactic_file)
    expected_sha256 = adaptive.tactic_sha256.strip().lower()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "tactic table SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    _validate_base64(adaptive.layer_allowlist_b64)

    env.update(
        {
            "NEMORL_MXFP8_LINEAR_BACKEND": "flashinfer_trtllm",
            "VLLM_MXFP8_DENSE_TRTLLM_LAYOUT": "adaptive",
            "VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M": str(adaptive.switch_m),
            "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE": str(tactic_file),
            "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256": actual_sha256,
            "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64": (
                adaptive.layer_allowlist_b64
            ),
        }
    )
    return env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=("baseline", "trace", "adaptive"), required=True
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--tactic-file", type=Path)
    parser.add_argument("--tactic-sha256")
    parser.add_argument("--layer-allowlist-b64")
    parser.add_argument("--switch-m", type=int, default=256)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-max", type=int, default=8192)
    parser.add_argument("--shell", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    adaptive = None
    trace = None
    if args.arm == "adaptive":
        if not all(
            (args.tactic_file, args.tactic_sha256, args.layer_allowlist_b64)
        ):
            raise SystemExit("adaptive arm requires tactic file, SHA256, and allowlist")
        adaptive = AdaptiveInputs(
            tactic_file=args.tactic_file,
            tactic_sha256=args.tactic_sha256,
            layer_allowlist_b64=args.layer_allowlist_b64,
            switch_m=args.switch_m,
        )
    elif args.arm == "trace":
        if args.trace_dir is None:
            raise SystemExit("trace arm requires trace directory")
        trace = TraceInputs(trace_dir=args.trace_dir, trace_max=args.trace_max)
    env = build_arm_environment(
        args.arm,
        runtime_root=args.runtime_root,
        adaptive=adaptive,
        trace=trace,
    )
    for key, value in sorted(env.items()):
        if args.shell:
            print(f"export {key}={shlex.quote(value)}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
