"""Render seaborn figures (house style) from summarize_results.py CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

EDGE = "#192133"
VARIANT_ORDER = [
    "baseline",
    "fixed_k1",
    "fixed_k2",
    "fixed_k3",
    "fixed_k5",
    "fixed_k3_prob",
    "dynamic",
    "dynamic_prob",
    "suffix",
]


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, linestyle="--", dashes=(6, 6), linewidth=1.1, axis="y", zorder=0)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_linewidth(1.4)
        ax.spines[side].set_color("black")


def finish(fig, ax, out_base: Path, ncol: int | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol or len(labels),
            fontsize=10,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_base}.png")


def slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def plot_profile_grids(profile_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(profile_csv)
    df["K"] = "K=" + df["k"].astype(str)
    df.loc[df["variant"] == "baseline", "K"] = "K=0 (off)"
    greedy = df[df["sample_method"].isin(["-", "greedy"])]
    for (model, bench), group in greedy.groupby(["model", "bench"]):
        hue_order = sorted(
            group["K"].unique(), key=lambda s: int(s.split("=")[1].split()[0])
        )
        order = sorted(group["batch_size"].unique())
        fig, ax = plt.subplots(figsize=(max(5.5, 0.62 * len(order)), 2.4))
        sns.barplot(
            data=group,
            x="batch_size",
            y="output_tok_s",
            hue="K",
            order=order,
            hue_order=hue_order,
            palette=sns.color_palette("Paired", n_colors=len(hue_order)),
            edgecolor=EDGE,
            linewidth=2.0,
            zorder=10,
            ax=ax,
        )
        style_axes(ax, "Concurrent batch size", "Output tokens/s")
        finish(fig, ax, out_dir / f"profile_tok_s_{slug(model)}_{bench}")


def plot_profile_speedup(profile_csv: Path, out_dir: Path) -> None:
    """Per-GPU throughput speedup vs the K=0 baseline at every batch size."""
    df = pd.read_csv(profile_csv)
    df = df[df["sample_method"].isin(["-", "greedy"])]
    base = df[df["k"] == 0][["model", "bench", "batch_size", "output_tok_s_per_gpu"]]
    base = base.rename(columns={"output_tok_s_per_gpu": "baseline_tok_s_per_gpu"})
    merged = df[df["k"] > 0].merge(base, on=["model", "bench", "batch_size"])
    if merged.empty:
        return
    merged["speedup"] = (
        merged["output_tok_s_per_gpu"] / merged["baseline_tok_s_per_gpu"]
    )
    merged["K"] = "K=" + merged["k"].astype(str)
    for (model, bench), group in merged.groupby(["model", "bench"]):
        hue_order = sorted(group["K"].unique(), key=lambda s: int(s.split("=")[1]))
        order = sorted(group["batch_size"].unique())
        fig, ax = plt.subplots(figsize=(max(5.5, 0.62 * len(order)), 2.4))
        sns.barplot(
            data=group,
            x="batch_size",
            y="speedup",
            hue="K",
            order=order,
            hue_order=hue_order,
            palette=sns.color_palette("Paired", n_colors=len(hue_order)),
            edgecolor=EDGE,
            linewidth=2.0,
            zorder=10,
            ax=ax,
        )
        ax.axhline(y=1, linestyle="--", linewidth=1.1, color="black")
        style_axes(ax, "Concurrent batch size", "Tokens/s/GPU speedup vs no-SD")
        finish(fig, ax, out_dir / f"profile_speedup_per_gpu_{slug(model)}_{bench}")


def plot_rollout_tok_s_per_gpu(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    if df.empty or "mean_output_tok_s_per_gpu" not in df.columns:
        return
    df["setting"] = df["model"] + "\n" + df["bench"]
    hue_order = [v for v in VARIANT_ORDER if v in set(df["variant"])]
    order = sorted(df["setting"].unique())
    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(order)), 2.4))
    sns.barplot(
        data=df,
        x="setting",
        y="mean_output_tok_s_per_gpu",
        hue="variant",
        order=order,
        hue_order=hue_order,
        palette=sns.color_palette("Paired", n_colors=len(hue_order)),
        edgecolor=EDGE,
        linewidth=2.0,
        zorder=10,
        ax=ax,
    )
    style_axes(ax, "", "Rollout tokens/s/GPU")
    finish(fig, ax, out_dir / "rollout_tok_s_per_gpu")


def plot_acceptance(profile_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(profile_csv)
    df = df[(df["k"] > 0) & df["mean_acceptance_length"].notna()]
    if df.empty:
        return
    agg = (
        df.groupby(["model", "bench", "k", "sample_method"])["mean_acceptance_length"]
        .mean()
        .reset_index()
    )
    agg["K"] = "K=" + agg["k"].astype(str)
    agg["setting"] = agg["model"] + "\n" + agg["bench"]
    hue_order = sorted(agg["K"].unique(), key=lambda s: int(s.split("=")[1]))
    greedy = agg[agg["sample_method"] == "greedy"]
    order = sorted(greedy["setting"].unique())
    fig, ax = plt.subplots(figsize=(max(4.5, 1.0 * len(order)), 2.4))
    sns.barplot(
        data=greedy,
        x="setting",
        y="mean_acceptance_length",
        hue="K",
        order=order,
        hue_order=hue_order,
        palette=sns.color_palette("Paired", n_colors=len(hue_order)),
        edgecolor=EDGE,
        linewidth=2.0,
        zorder=10,
        ax=ax,
    )
    style_axes(ax, "", "Mean acceptance length")
    finish(fig, ax, out_dir / "profile_acceptance_length")


def plot_rollout_speedup(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    base = df[df["variant"] == "baseline"][["model", "bench", "mean_step_wall_s"]]
    base = base.rename(columns={"mean_step_wall_s": "baseline_wall_s"})
    merged = df.merge(base, on=["model", "bench"])
    merged["speedup"] = merged["baseline_wall_s"] / merged["mean_step_wall_s"]
    merged["setting"] = merged["model"] + "\n" + merged["bench"]
    hue_order = [v for v in VARIANT_ORDER if v in set(merged["variant"])]
    order = sorted(merged["setting"].unique())
    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(order)), 2.4))
    sns.barplot(
        data=merged,
        x="setting",
        y="speedup",
        hue="variant",
        order=order,
        hue_order=hue_order,
        palette=sns.color_palette("Paired", n_colors=len(hue_order)),
        edgecolor=EDGE,
        linewidth=2.0,
        zorder=10,
        ax=ax,
    )
    ax.axhline(y=1, linestyle="--", linewidth=1.1, color="black")
    style_axes(ax, "", "Rollout-step speedup vs baseline")
    finish(fig, ax, out_dir / "rollout_speedup")


def plot_drain_curves(drain_csv: Path, out_dir: Path) -> None:
    if not drain_csv.exists():
        return
    df = pd.read_csv(drain_csv)
    if df.empty:
        return
    df = df[df["step"] > 0]  # step 0 pays warmup/compile noise
    for (model, bench), group in df.groupby(["model", "bench"]):
        variants = [v for v in VARIANT_ORDER if v in set(group["variant"])]
        palette = sns.color_palette("Paired", n_colors=len(variants))
        fig, ax = plt.subplots(figsize=(5.5, 2.4))
        for color, variant in zip(palette, variants):
            sub = group[group["variant"] == variant]
            steps = sub["step"].nunique() or 1
            times = sub["finished_s"].sort_values().to_numpy()
            total = len(times) / steps
            remaining = [total - (i + 1) / steps for i in range(len(times))]
            ax.plot(
                times, remaining, label=variant, linewidth=2.2, color=color, zorder=10
            )
        style_axes(ax, "Time within rollout step (s)", "Sequences still running")
        finish(
            fig,
            ax,
            out_dir / f"drain_{slug(model)}_{bench}",
            ncol=min(4, len(variants)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    plot_profile_grids(args.data_dir / "profile_grid.csv", args.out_dir)
    plot_profile_speedup(args.data_dir / "profile_grid.csv", args.out_dir)
    plot_acceptance(args.data_dir / "profile_grid.csv", args.out_dir)
    summary = args.data_dir / "rollout_summary.csv"
    if summary.exists():
        plot_rollout_speedup(summary, args.out_dir)
        plot_rollout_tok_s_per_gpu(summary, args.out_dir)
    plot_drain_curves(args.data_dir / "drain_curves.csv", args.out_dir)


if __name__ == "__main__":
    main()
