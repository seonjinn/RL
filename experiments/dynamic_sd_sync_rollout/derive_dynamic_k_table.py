"""Derive the DynamicSD batch-size -> K lookup table from profile runs.

Input: profile-mode JSONs from sync_rollout_dynamic_sd.py, one per K
(baseline run = K 0). For every profiled batch size the K with the highest
measured output tok/s wins; contiguous batch-size ranges sharing a winner are
merged into the [[bs_lo, bs_hi, K], ...] shape vLLM 0.24 expects in
speculative_config.num_speculative_tokens_per_batch_size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_profile(path: Path) -> tuple[int, dict[str, Any], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    spec = config.get("speculative_config_resolved") or {}
    k = int(spec.get("num_speculative_tokens", 0)) if spec else 0
    rows = {
        int(row["batch_size"]): row
        for row in payload.get("results", [])
        if row.get("mode") == "profile"
    }
    return k, spec, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profiles", nargs="+", help="profile JSON paths (all K incl. baseline)"
    )
    parser.add_argument(
        "--extend-to",
        type=int,
        default=None,
        help="upper bound of the last range (>= max profiled BS; default: max "
        "profiled BS; vLLM carries the last K forward anyway)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output speculative_config JSON with the dynamic table",
    )
    parser.add_argument("--grid-csv", help="optional CSV dump of the full BS x K grid")
    parser.add_argument(
        "--max-capture-tokens",
        type=int,
        default=512,
        help="cudagraph capture budget: cap K so bs*(K+1) <= this "
        "(vLLM default max_cudagraph_capture_size; 0 disables the cap)",
    )
    args = parser.parse_args()

    by_k: dict[int, dict[int, dict[str, Any]]] = {}
    spec_base: dict[str, Any] = {}
    for path_str in args.profiles:
        k, spec, rows = load_profile(Path(path_str))
        if k in by_k:
            raise ValueError(f"duplicate profile for K={k}: {path_str}")
        by_k[k] = rows
        if k > 0 and not spec_base:
            spec_base = dict(spec)

    if not spec_base:
        raise ValueError("need at least one K>0 profile to supply the draft config")

    if 0 not in by_k:
        print(
            "WARNING: no K=0 baseline profile provided; speculation can never "
            "be disabled in the derived table"
        )

    batch_sizes = sorted({bs for rows in by_k.values() for bs in rows})
    grid: list[dict[str, Any]] = []
    optimal: dict[int, int] = {}
    for bs in batch_sizes:
        candidates = {
            k: rows[bs]["output_tok_s"] for k, rows in by_k.items() if bs in rows
        }
        if not candidates:
            continue
        missing = sorted(set(by_k) - set(candidates))
        if missing:
            print(
                f"WARNING: bs={bs} missing profiles for K={missing}; "
                "optimum picked from a subset"
            )
        best_k = max(candidates, key=lambda k: candidates[k])
        optimal[bs] = best_k
        for k, tok_s in sorted(candidates.items()):
            row = by_k[k][bs]
            grid.append(
                {
                    "batch_size": bs,
                    "k": k,
                    "output_tok_s": tok_s,
                    "wall_ms_per_output_token": row.get("wall_ms_per_output_token"),
                    "mean_acceptance_length": row.get("mean_acceptance_length"),
                    "optimal": k == best_k,
                }
            )

    profiled = sorted(optimal)
    extend_to = args.extend_to if args.extend_to is not None else profiled[-1]
    if extend_to < profiled[-1]:
        raise ValueError(
            f"--extend-to {extend_to} < max profiled batch size {profiled[-1]}; "
            "this would emit an inverted range that vLLM rejects"
        )
    # Dense bs -> K, carrying the profiled optimum forward between grid points,
    # then capped so bs*(K+1) never exceeds the cudagraph capture budget. The
    # profiled grid is too coarse to see the eager-fallback cliff between
    # points (e.g. K=5 optimal at BS=64 but 86*(5+1) > 512 at BS=86), so the
    # hardware constraint must be enforced analytically.
    dense: list[int] = [0] * (extend_to + 1)
    for bs in range(1, extend_to + 1):
        anchor = max((p for p in profiled if p <= bs), default=profiled[0])
        k = optimal[anchor]
        if args.max_capture_tokens > 0:
            k = min(k, max(0, args.max_capture_tokens // bs - 1))
        dense[bs] = k
    ranges: list[list[int]] = []
    for bs in range(1, extend_to + 1):
        if ranges and ranges[-1][2] == dense[bs]:
            ranges[-1][1] = bs
        else:
            ranges.append([bs, bs, dense[bs]])

    spec_out = dict(spec_base)
    spec_out["num_speculative_tokens"] = max(
        (k for k in by_k if k > 0), default=spec_base.get("num_speculative_tokens", 3)
    )
    spec_out["num_speculative_tokens_per_batch_size"] = ranges

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec_out, indent=2), encoding="utf-8")

    if args.grid_csv:
        import csv

        with Path(args.grid_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(grid[0]))
            writer.writeheader()
            writer.writerows(grid)

    print(json.dumps({"ranges": ranges, "optimal_per_bs": optimal}, indent=2))


if __name__ == "__main__":
    main()
