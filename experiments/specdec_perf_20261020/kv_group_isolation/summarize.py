"""Print bench.py results as markdown tables: python summarize.py <dir or json> ..."""

import json
import sys
from pathlib import Path


def load(paths: list[str]) -> list[dict]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    return [json.loads(f.read_text()) for f in files]


def matched_prefix(x: list[int], y: list[int]) -> int:
    return next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))


def greedy_match(a: list[list[int]], b: list[list[int]]) -> str:
    """Identical sequences and mean matched prefix length."""
    same = sum(x == y for x, y in zip(a, b))
    prefix = [matched_prefix(x, y) for x, y in zip(a, b)]
    return f"{same}/{len(a)} seqs, prefix {sum(prefix) / len(prefix):.0f}/{len(a[0])}"


def logprob_split(a: dict, b: dict) -> str:
    """Largest chosen-token logprob gap before the runs split, and the top-2 margin at the split.

    Bit-identical logprobs up to a split with a large margin point at a logic difference.
    Drifting logprobs and near-tie splits point at numeric noise.
    """
    drift, margins = 0.0, []
    for x, y, lx, ly in zip(
        a["greedy_token_ids"][0], b["greedy_token_ids"][0], a["greedy_logprobs"][0], b["greedy_logprobs"][0]
    ):
        n = matched_prefix(x, y)
        drift = max([drift] + [abs(p[0] - q[0]) for p, q in zip(lx[:n], ly[:n])])
        if n < min(len(x), len(y)) and lx[n][1] is not None:
            margins.append(lx[n][0] - lx[n][1])
    split = f"max split margin {max(margins):.3f}" if margins else "no split"
    return f"max dlogprob {drift:.1e}, {split}"


def reference_run(name: str, names: set[str]) -> str | None:
    """arm@N compares with arm; arm_reuse and arm_reuse_verify compare with arm."""
    base = name.split("@")[0]
    if base == name:
        base = name.removesuffix("_verify").removesuffix("_reuse")
    return base if base != name and base in names else None


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

    # Greedy tokens: run-to-run agreement within a process, then each run vs its reference arm
    greedy = {r["arm"]: r for r in runs if "greedy_token_ids" in r}
    if greedy:
        print("\n| Run | Run 1 vs run 2 | Reference | vs reference | logprobs vs reference |")
        print("|---|---|---|---|---|")
        for name, r in greedy.items():
            first, second = r["greedy_token_ids"]
            ref = reference_run(name, set(greedy))
            vs_ref = greedy_match(greedy[ref]["greedy_token_ids"][0], first) if ref else "-"
            lp = (
                logprob_split(greedy[ref], r)
                if ref and "greedy_logprobs" in r and "greedy_logprobs" in greedy[ref]
                else "-"
            )
            print(f"| {name} | {greedy_match(first, second)} | {ref or '-'} | {vs_ref} | {lp} |")

    verified = [r for r in runs if "gdn_reuse_verify" in r]
    if verified:
        print("\n| Run | Checks | Skipped in capture | Mismatched fields |\n|---|---|---|---|")
        for r in verified:
            v = r["gdn_reuse_verify"]
            print(
                f"| {r['arm']} | {v['checks']} | {v['skipped_capturing']}"
                f" | {json.dumps(v['mismatches']) if v['mismatches'] else 'none'} |"
            )


if __name__ == "__main__":
    main()
