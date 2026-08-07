"""Publication plots backed by explicit component and run evidence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EDGE_COLOR = "#192133"
PLOT_NAMES = (
    "mxfp8_moe_tactic_audit_micro_speedup",
    "mxfp8_moe_tactic_audit_tactic_cache_shares",
    "mxfp8_moe_tactic_audit_end_to_end",
    "mxfp8_moe_tactic_audit_step_variation",
)


def _style(ax: Axes, ylabel: str) -> None:
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, linestyle="--", dashes=(6, 6), linewidth=1.1, axis="y", zorder=0)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_linewidth(2.0)
        ax.spines[side].set_color("black")


def _save(fig: Figure, base: Path, caption: str) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(base.with_suffix(f".{extension}"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def _bars(ax: Axes, data: pd.DataFrame, *, x: str, y: str, ylabel: str) -> None:
    order = list(data[x])
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=x,
        order=order,
        hue_order=order,
        palette=sns.color_palette("Paired", n_colors=len(order)),
        edgecolor=EDGE_COLOR,
        linewidth=2.0,
        legend=False,
        zorder=10,
        ax=ax,
    )
    _style(ax, ylabel)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.3g", padding=3, fontsize=10)


def write_complete_plots(
    output_dir: Path,
    *,
    component_speedups: Sequence[tuple[str, float]],
    tactic_change_share: float,
    cache_hit_share: float,
    normalized_throughput: float,
    normalized_total_step_time: float,
    per_step: Sequence[tuple[str, str, int, float, float]],
    metadata_caption: str,
) -> None:
    """Write four 600-DPI PNG/PDF figures from complete executed evidence."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    micro = pd.DataFrame(component_speedups, columns=["Component", "Speedup"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _bars(ax, micro, x="Component", y="Speedup", ylabel="Per-profile component speedup")
    ax.axhline(1.0, linestyle="--", linewidth=1.1, color="black", zorder=2)
    _save(fig, output_dir / PLOT_NAMES[0], metadata_caption)

    shares = pd.DataFrame(
        (
            ("Tactic change", tactic_change_share),
            ("Cache hit", cache_hit_share),
            ("Fallback", 1.0 - cache_hit_share),
        ),
        columns=["Evidence", "Share"],
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _bars(ax, shares, x="Evidence", y="Share", ylabel="Share")
    ax.set_ylim(0, 1.12)
    _save(fig, output_dir / PLOT_NAMES[1], metadata_caption)

    end_to_end = pd.DataFrame(
        (
            ("tok/s/GPU", normalized_throughput),
            ("Total step time", normalized_total_step_time),
        ),
        columns=["Metric", "Candidate / Stock"],
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _bars(ax, end_to_end, x="Metric", y="Candidate / Stock", ylabel="Candidate / Stock")
    ax.axhline(1.0, linestyle="--", linewidth=1.1, color="black", zorder=2)
    _save(
        fig,
        output_dir / PLOT_NAMES[2],
        metadata_caption + "; total step time: lower is better",
    )

    rows = []
    for run_id, arm, step, tokens, seconds in per_step:
        label = f"{arm}/{run_id} S{step}"
        rows.extend(
            (
                (label, arm.title(), "tok/s/GPU", tokens),
                (label, arm.title(), "Total step s", seconds),
            )
        )
    frame = pd.DataFrame(rows, columns=["Step", "Arm", "Metric", "Value"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for ax, metric in zip(axes, ("tok/s/GPU", "Total step s"), strict=True):
        subset = cast(pd.DataFrame, frame[frame["Metric"] == metric])
        sns.barplot(
            data=subset,
            x="Step",
            y="Value",
            hue="Arm",
            hue_order=["Stock", "Candidate"],
            palette=sns.color_palette("Paired", n_colors=2),
            edgecolor=EDGE_COLOR,
            linewidth=2.0,
            errorbar=None,
            zorder=10,
            ax=ax,
        )
        _style(ax, metric)
        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(container, fmt="%.3g", padding=2, fontsize=8)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.tick_params(axis="x", rotation=65, labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=11,
    )
    _save(fig, output_dir / PLOT_NAMES[3], metadata_caption + "; raw steps 3-8")


def write_unavailable_plots(output_dir: Path, state: str) -> None:
    """Render explicit placeholders without fabricated performance values."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    for name in PLOT_NAMES:
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ax.text(
            0.5,
            0.5,
            f"{state}\nNo performance values reported",
            ha="center",
            va="center",
            fontsize=13,
        )
        ax.set_axis_off()
        _save(fig, output_dir / name, "MXFP8 MoE tactic audit")
