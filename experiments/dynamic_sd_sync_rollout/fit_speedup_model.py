"""Fit an analytic speculative-decoding cost model to the profile grids.

Model per (model, bench):
    t_step(B, K) = alpha + beta * B * (K + 1) + K * (gamma + delta * B)
    tok_s(B, K)  = B * AL(K) / t_step(B, K)

alpha: fixed per-step cost (kernel launches, sampler)
beta:  per-processed-token cost of the target forward (verify batch)
gamma/delta: per-draft-iteration fixed / per-sequence drafter cost
AL(K): measured mean acceptance length (AL(0) = 1)

Fit alpha..delta by least squares on measured t_step = B*AL/tok_s over the
grid, then report prediction quality and whether the model reproduces the
measured optimal K per batch size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fit_group(g: pd.DataFrame) -> dict | None:
    g = g[g["sample_method"].isin(["-", "greedy"])].copy()
    al = {
        int(k): sub["mean_acceptance_length"].mean()
        for k, sub in g[g["k"] > 0].groupby("k")
    }
    al[0] = 1.0
    if len(al) < 3:
        return None
    g["al"] = g["k"].map(al)
    g["t_step"] = g["batch_size"] * g["al"] / g["output_tok_s"]

    B, K = g["batch_size"].to_numpy(float), g["k"].to_numpy(float)
    X = np.column_stack([np.ones_like(B), B * (K + 1), K, K * B])
    y = g["t_step"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef = np.maximum(coef, 0)  # physical costs are non-negative
    pred_t = X @ coef
    g["pred_tok_s"] = g["batch_size"] * g["al"] / np.maximum(pred_t, 1e-9)
    rel_err = np.abs(g["pred_tok_s"] - g["output_tok_s"]) / g["output_tok_s"]

    agree = total = 0
    regrets = []
    for bs, sub in g.groupby("batch_size"):
        if sub["k"].nunique() < 2:
            continue
        total += 1
        best_k = sub.loc[sub["output_tok_s"].idxmax(), "k"]
        pred_k = sub.loc[sub["pred_tok_s"].idxmax(), "k"]
        agree += int(best_k == pred_k)
        best = sub["output_tok_s"].max()
        at_pred = sub.loc[sub["k"] == pred_k, "output_tok_s"].max()
        regrets.append(at_pred / best)
    return {
        "alpha": coef[0],
        "beta": coef[1],
        "gamma": coef[2],
        "delta": coef[3],
        "n_points": len(g),
        "median_rel_err_pct": 100 * float(np.median(rel_err)),
        "p90_rel_err_pct": 100 * float(np.quantile(rel_err, 0.9)),
        "optimal_k_agreement": f"{agree}/{total}",
        "mean_regret_pct": 100 * (1 - float(np.mean(regrets))) if regrets else None,
        "worst_regret_pct": 100 * (1 - float(np.min(regrets))) if regrets else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.profile_csv)
    rows = []
    for (model, bench), g in df.groupby(["model", "bench"]):
        fit = fit_group(g)
        if fit is None:
            continue
        rows.append({"model": model, "bench": bench, **fit})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
