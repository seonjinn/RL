"""Paper-ready plots for the MXFP8 MoE tactic audit."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns


EDGE_COLOR = "#192133"
PLOT_NAMES = (
    "mxfp8_moe_tactic_audit_micro_speedup",
    "mxfp8_moe_tactic_audit_tactic_cache_shares",
    "mxfp8_moe_tactic_audit_end_to_end",
    "mxfp8_moe_tactic_audit_step_variation",
)


def _style_axis(ax: Axes, ylabel: str) -> None:
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, linestyle="--", dashes=(6, 6), linewidth=1.1, axis="y", zorder=0)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_linewidth(2.0)
        ax.spines[side].set_color("black")


def _save(fig: Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{extension}"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def _bar(ax: Axes, data: pd.DataFrame, *, x: str, y: str, ylabel: str) -> None:
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
    _style_axis(ax, ylabel)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.3g", padding=3, fontsize=10)


def write_complete_plots(
    output_dir: Path,
    *,
    micro_speedups: Sequence[tuple[str, float]],
    tactic_change_share: float,
    cache_hit_share: float,
    normalized_generation_throughput: float,
    normalized_step_speed: float,
    step_values: Sequence[tuple[int, float, float]],
) -> None:
    """Write the four required 600-DPI PNG/PDF plot pairs."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    micro = pd.DataFrame(micro_speedups, columns=["Kernel", "Speedup"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _bar(ax, micro, x="Kernel", y="Speedup", ylabel="Call-weighted micro speedup")
    ax.axhline(1.0, linestyle="--", linewidth=1.1, color="black", zorder=2)
    _save(fig, output_dir / PLOT_NAMES[0])

    shares = pd.DataFrame(
        [
            ("Tactic change", tactic_change_share),
            ("Cache hit", cache_hit_share),
            ("Fallback", 1.0 - cache_hit_share),
        ],
        columns=["Evidence", "Share"],
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _bar(ax, shares, x="Evidence", y="Share", ylabel="Share")
    ax.set_ylim(0, 1.12)
    _save(fig, output_dir / PLOT_NAMES[1])

    end_to_end = pd.DataFrame(
        [
            ("Generation tok/s/GPU", normalized_generation_throughput),
            ("Step speed", normalized_step_speed),
        ],
        columns=["Metric", "Stock normalized"],
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _bar(ax, end_to_end, x="Metric", y="Stock normalized", ylabel="Stock normalized")
    ax.axhline(1.0, linestyle="--", linewidth=1.1, color="black", zorder=2)
    _save(fig, output_dir / PLOT_NAMES[2])

    variation_rows = [
        (f"Step {step}", "Stock", stock_value)
        for step, stock_value, _ in step_values
    ] + [
        (f"Step {step}", "Candidate", candidate_value)
        for step, _, candidate_value in step_values
    ]
    variation = pd.DataFrame(variation_rows, columns=["Step", "Arm", "tok/s/GPU"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(
        data=variation,
        x="Step",
        y="tok/s/GPU",
        hue="Arm",
        hue_order=["Stock", "Candidate"],
        palette=sns.color_palette("Paired", n_colors=2),
        edgecolor=EDGE_COLOR,
        linewidth=2.0,
        zorder=10,
        ax=ax,
    )
    _style_axis(ax, "Generation tok/s/GPU")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    fig.legend(handles, labels, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=11)
    _save(fig, output_dir / PLOT_NAMES[3])


def write_incomplete_plots(output_dir: Path) -> None:
    """Render explicit non-numeric placeholders when collection fails closed."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    for plot_name in PLOT_NAMES:
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ax.text(0.5, 0.5, "INCOMPLETE EVIDENCE\nNo performance values reported", ha="center", va="center", fontsize=13)
        ax.set_axis_off()
        _save(fig, output_dir / plot_name)
