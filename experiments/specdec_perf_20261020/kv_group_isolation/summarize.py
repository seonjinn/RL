"""Print bench.py results as markdown tables: python summarize.py <dir or json> ..."""

import json
import sys
from pathlib import Path


def load(paths: list[str]) -> list[dict]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    return [json.loads(f.read_text()) for f in files]


def greedy_match(a: list[list[int]], b: list[list[int]]) -> str:
    """Identical sequences and mean matched prefix length."""
    same = sum(x == y for x, y in zip(a, b))
    prefix = []
    for x, y in zip(a, b):
        n = next(
            (i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))
        )
        prefix.append(n)
    return f"{same}/{len(a)} seqs, prefix {sum(prefix) / len(prefix):.0f}/{len(a[0])}"


def main() -> None:
    runs = load(sys.argv[1:])
    batches = sorted({ph["batch"] for r in runs for ph in r["phases"]})
    cols = " | ".join(f"B{b}" for b in batches)
    sep = "|---" * (5 + 4 * len(batches)) + "|"

    def per_batch(r: dict, key: str, fmt: str) -> str:
        by_batch = {ph["batch"]: ph for ph in r["phases"]}
        return " | ".join(
            format(by_batch[b][key], fmt) if key in by_batch.get(b, {}) else "-"
            for b in batches
        )

    print(
        f"| Arm | Groups | Layers/group | Layout | GPU blocks | tok/s {cols} | step ms {cols}"
        f" | attn-md CPU ms {cols} | accept {cols} |"
    )
    print(sep)
    for r in runs:
        print(
            f"| {r['arm']} | {r.get('kv_cache_groups', '-')} | {r.get('layers_per_group', '-')}"
            f" | {r.get('kv_cache_layout') or '-'} | {r.get('num_gpu_blocks') or '-'}"
            f" | {per_batch(r, 'tok_per_s', '.0f')} | {per_batch(r, 'step_ms', '.1f')}"
            f" | {per_batch(r, 'attn_md_cpu_ms_per_step', '.2f')}"
            f" | {per_batch(r, 'mean_accept_len', '.2f')} |"
        )

    # Greedy tokens: run-to-run agreement within an arm, then each *_reuse arm vs its base
    greedy = {r["arm"]: r["greedy_token_ids"] for r in runs if "greedy_token_ids" in r}
    if greedy:
        print("\n| Arm | Run 1 vs run 2 | vs base arm |\n|---|---|---|")
        for name, (first, second) in greedy.items():
            base = (
                greedy.get(name.removesuffix("_reuse"))
                if name.endswith("_reuse")
                else None
            )
            vs_base = greedy_match(base[0], first) if base else "-"
            print(f"| {name} | {greedy_match(first, second)} | {vs_base} |")


if __name__ == "__main__":
    main()
