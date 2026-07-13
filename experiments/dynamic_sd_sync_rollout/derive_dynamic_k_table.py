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
        default=256,
        help="upper bound of the last range (set to max_num_seqs)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output speculative_config JSON with the dynamic table",
    )
    parser.add_argument("--grid-csv", help="optional CSV dump of the full BS x K grid")
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

    batch_sizes = sorted({bs for rows in by_k.values() for bs in rows})
    grid: list[dict[str, Any]] = []
    optimal: dict[int, int] = {}
    for bs in batch_sizes:
        candidates = {
            k: rows[bs]["output_tok_s"] for k, rows in by_k.items() if bs in rows
        }
        if not candidates:
            continue
        best_k = max(candidates, key=lambda k: candidates[k])
        optimal[bs] = best_k
        for k, tok_s in sorted(candidates.items()):
            row = by_k[k][bs]
            grid.append(
                {
                    "batch_size": bs,
                    "k": k,
                    "output_tok_s": tok_s,
                    "itl_ms_per_token": row.get("itl_ms_per_token"),
                    "mean_acceptance_length": row.get("mean_acceptance_length"),
                    "optimal": k == best_k,
                }
            )

    ranges: list[list[int]] = []
    profiled = sorted(optimal)
    for idx, bs in enumerate(profiled):
        hi = profiled[idx + 1] - 1 if idx + 1 < len(profiled) else args.extend_to
        k = optimal[bs]
        if ranges and ranges[-1][2] == k:
            ranges[-1][1] = hi
        else:
            ranges.append([bs, hi, k])
    if ranges and ranges[0][0] > 1:
        ranges[0][0] = 1

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
