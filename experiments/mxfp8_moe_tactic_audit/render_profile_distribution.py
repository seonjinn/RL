"""Render the FC1/FC2 profile-level tactic improvement distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EDGE_COLOR = "#192133"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _profile_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for profile in summary["profiles"]:
        rows.append(
            {
                "Profile": (
                    f"T={profile['total_tokens']}, "
                    f"Mmax={profile['expert_m_max']}, "
                    f"{profile['skew_class']}"
                ),
                "Gain (%)": (profile["speedup"] - 1.0) * 100.0,
                "Skew": profile["skew_class"],
            }
        )
    return pd.DataFrame(rows).sort_values("Gain (%)", ascending=False)


def render(summary_path: Path, output: Path) -> None:
    summary = json.loads(summary_path.read_text())
    profiles = _profile_frame(summary)
    distribution = pd.DataFrame(summary["gain_distribution"])
    distribution["Selected GPU-time share (%)"] = (
        distribution["selected_gpu_time_share"] * 100.0
    )

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    sns.barplot(
        data=profiles,
        y="Profile",
        x="Gain (%)",
        hue="Skew",
        hue_order=["median-skew", "high-skew"],
        palette=sns.color_palette("Paired", n_colors=2),
        edgecolor=EDGE_COLOR,
        linewidth=2.0,
        dodge=False,
        zorder=10,
        ax=axes[0],
    )
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.1, zorder=2)
    axes[0].set_title("Gain by routed-row profile", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("FC1+FC2 kernel improvement (%)")
    axes[0].set_ylabel("")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    for patch in axes[0].patches:
        width = patch.get_width()
        axes[0].annotate(
            f"{width:.2f}%",
            (width, patch.get_y() + patch.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
        )

    sns.barplot(
        data=distribution,
        x="range",
        y="Selected GPU-time share (%)",
        hue="range",
        palette=sns.color_palette("Paired", n_colors=len(distribution)),
        edgecolor=EDGE_COLOR,
        linewidth=2.0,
        legend=False,
        zorder=10,
        ax=axes[1],
    )
    axes[1].set_title("Improvement distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Kernel improvement range")
    axes[1].set_ylabel("Share of selected GPU time (%)")
    axes[1].tick_params(axis="x", rotation=30)
    for patch in axes[1].patches:
        height = patch.get_height()
        axes[1].annotate(
            f"{height:.1f}%",
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for ax in axes:
        ax.grid(True, axis="y", linestyle="--", dashes=(6, 6), linewidth=1.0, zorder=0)
        ax.tick_params(labelsize=9)
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_linewidth(1.8)
            ax.spines[side].set_color("black")

    fig.text(
        0.5,
        0.01,
        "T: total tokens; Mmax: maximum routed rows for one expert. N and K are fixed by the FC1/FC2 projections.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = _parse_args()
    render(args.summary, args.output)
