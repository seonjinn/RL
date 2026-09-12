from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text


WIDTH_PX = 1480
HEIGHT_PX = 830
DPI = 100
REQUIRED_LABELS = [
    "Target context (shared)",
    "EAGLE-3",
    "DFlash",
    "DSpark",
    "AUTOREGRESSIVE",
    "PARALLEL",
    "SEMI-AUTOREGRESSIVE",
    "Confidence scheduler",
    "Target verifier",
    "Accepted prefix",
]
SOURCE_URLS = [
    "https://arxiv.org/abs/2503.01840",
    "https://arxiv.org/abs/2602.06036",
    "https://arxiv.org/abs/2607.05147",
]


@dataclass(frozen=True)
class BoxAudit:
    patch: FancyBboxPatch
    texts: tuple[Text, ...]


def add_panel(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=2.5,
        facecolor="#FFFFFF",
        edgecolor=edgecolor,
        transform=ax.transAxes,
        zorder=2,
    )
    patch.set_path_effects(
        [path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.1), path_effects.Normal()]
    )
    ax.add_patch(patch)


def add_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str | None,
    facecolor: str,
    edgecolor: str,
    title_size: float = 14,
    subtitle_size: float = 9.5,
    linewidth: float = 1.7,
    shadow: bool = False,
) -> BoxAudit:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.011",
        linewidth=linewidth,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
        zorder=3,
    )
    if shadow:
        patch.set_path_effects(
            [path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.1), path_effects.Normal()]
        )
    ax.add_patch(patch)

    center_x = x + width / 2
    title_artist = ax.text(
        center_x,
        y + height * (0.62 if subtitle else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#17202A",
        transform=ax.transAxes,
        zorder=4,
    )
    texts: list[Text] = [title_artist]
    if subtitle:
        texts.append(
            ax.text(
                center_x,
                y + height * 0.29,
                subtitle,
                ha="center",
                va="center",
                fontsize=subtitle_size,
                color="#4B5563",
                transform=ax.transAxes,
                zorder=4,
            )
        )
    return BoxAudit(patch=patch, texts=tuple(texts))


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#667085",
    curve: float = 0,
    width: float = 1.8,
    mutation_scale: float = 14,
    zorder: float = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=width,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=0,
            shrinkB=0,
            transform=ax.transAxes,
            zorder=zorder,
        )
    )


def add_token_row(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    edgecolor: str,
    facecolor: str,
    arrows: bool,
) -> list[BoxAudit]:
    labels = ("t+1", "t+2", "t+K")
    token_width = 0.052
    token_height = 0.057
    gap = 0.026
    audits: list[BoxAudit] = []
    for index, label in enumerate(labels):
        token_x = x + index * (token_width + gap)
        audits.append(
            add_box(
                ax,
                x=token_x,
                y=y,
                width=token_width,
                height=token_height,
                title=label,
                subtitle=None,
                facecolor=facecolor,
                edgecolor=edgecolor,
                title_size=11,
                linewidth=1.6,
            )
        )
        if arrows and index < len(labels) - 1:
            add_arrow(
                ax,
                (token_x + token_width + 0.003, y + token_height / 2),
                (token_x + token_width + gap - 0.003, y + token_height / 2),
                color=edgecolor,
                width=1.5,
                mutation_scale=11,
                zorder=4,
            )
    return audits


def audit_text_fit(fig: plt.Figure, audits: list[BoxAudit], padding_px: float = 6) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow: list[str] = []
    for audit in audits:
        box = audit.patch.get_window_extent(renderer)
        for text in audit.texts:
            bounds = text.get_window_extent(renderer)
            if not (
                bounds.x0 >= box.x0 + padding_px
                and bounds.x1 <= box.x1 - padding_px
                and bounds.y0 >= box.y0 + padding_px
                and bounds.y1 <= box.y1 - padding_px
            ):
                overflow.append(text.get_text())
    return overflow


def audit_canvas_text(fig: plt.Figure, ax: plt.Axes, padding_px: float = 6) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    overflow: list[str] = []
    for text in ax.texts:
        bounds = text.get_window_extent(renderer)
        if not (
            bounds.x0 >= canvas.x0 + padding_px
            and bounds.x1 <= canvas.x1 - padding_px
            and bounds.y0 >= canvas.y0 + padding_px
            and bounds.y1 <= canvas.y1 - padding_px
        ):
            overflow.append(text.get_text())
    return overflow


def render(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.945,
        "How Three Speculative Drafters Differ",
        fontsize=27,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.895,
        "The proposal path changes. The target model remains the final correctness boundary.",
        fontsize=13.5,
        color="#5B6472",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    audits: list[BoxAudit] = []
    audits.append(
        add_box(
            ax,
            x=0.27,
            y=0.79,
            width=0.46,
            height=0.075,
            title="Target context (shared)",
            subtitle="fused hidden states condition every drafter",
            facecolor="#F2F4F7",
            edgecolor="#667085",
            title_size=14,
            subtitle_size=9.8,
        )
    )

    card_y = 0.325
    card_h = 0.39
    card_w = 0.28
    card_x = [0.04, 0.36, 0.68]
    colors = [
        ("#4472C4", "#EEF4FC"),
        ("#F28E2B", "#FFF4E8"),
        ("#76B900", "#F2F8E9"),
    ]
    names = ["EAGLE-3", "DFlash", "DSpark"]
    badges = ["AUTOREGRESSIVE", "PARALLEL", "SEMI-AUTOREGRESSIVE"]

    for x, (edge, fill), name, badge in zip(card_x, colors, names, badges, strict=True):
        add_panel(ax, x=x, y=card_y, width=card_w, height=card_h, edgecolor=edge)
        ax.text(
            x + card_w / 2,
            0.675,
            name,
            ha="center",
            va="center",
            fontsize=19,
            fontweight="bold",
            color="#17202A",
            transform=ax.transAxes,
            zorder=4,
        )
        audits.append(
            add_box(
                ax,
                x=x + 0.053,
                y=0.605,
                width=0.174,
                height=0.045,
                title=badge,
                subtitle=None,
                facecolor=fill,
                edgecolor=edge,
                title_size=10.5,
                linewidth=1.5,
            )
        )
        add_arrow(
            ax,
            (0.5, 0.79),
            (x + card_w / 2, card_y + card_h),
            color="#98A2B3",
            curve=0.1 if x == card_x[0] else (-0.1 if x == card_x[2] else 0),
            width=1.6,
        )

    audits.extend(
        add_token_row(
            ax,
            x=card_x[0] + 0.036,
            y=0.495,
            edgecolor=colors[0][0],
            facecolor=colors[0][1],
            arrows=True,
        )
    )
    ax.text(
        card_x[0] + card_w / 2,
        0.415,
        "Previous draft token feeds the next",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#344054",
        transform=ax.transAxes,
    )
    ax.text(
        card_x[0] + card_w / 2,
        0.365,
        "K draft passes",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=colors[0][0],
        transform=ax.transAxes,
    )

    audits.extend(
        add_token_row(
            ax,
            x=card_x[1] + 0.036,
            y=0.495,
            edgecolor=colors[1][0],
            facecolor=colors[1][1],
            arrows=False,
        )
    )
    ax.text(
        card_x[1] + card_w / 2,
        0.425,
        "Whole block predicted together",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#344054",
        transform=ax.transAxes,
    )
    ax.text(
        card_x[1] + card_w / 2,
        0.375,
        "1 block-diffusion pass",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=colors[1][0],
        transform=ax.transAxes,
    )

    dspark_x = card_x[2]
    audits.append(
        add_box(
            ax,
            x=dspark_x + 0.055,
            y=0.515,
            width=0.17,
            height=0.052,
            title="Parallel backbone",
            subtitle=None,
            facecolor=colors[2][1],
            edgecolor=colors[2][0],
            title_size=10.8,
            linewidth=1.5,
        )
    )
    audits.append(
        add_box(
            ax,
            x=dspark_x + 0.055,
            y=0.43,
            width=0.17,
            height=0.052,
            title="Light sequential head",
            subtitle=None,
            facecolor="#F8FBEF",
            edgecolor=colors[2][0],
            title_size=10.4,
            linewidth=1.5,
        )
    )
    audits.append(
        add_box(
            ax,
            x=dspark_x + 0.055,
            y=0.345,
            width=0.17,
            height=0.052,
            title="Confidence scheduler",
            subtitle=None,
            facecolor="#F8FBEF",
            edgecolor=colors[2][0],
            title_size=10.6,
            linewidth=1.5,
        )
    )
    add_arrow(
        ax,
        (dspark_x + card_w / 2, 0.515),
        (dspark_x + card_w / 2, 0.482),
        color=colors[2][0],
        zorder=4,
    )
    add_arrow(
        ax,
        (dspark_x + card_w / 2, 0.43),
        (dspark_x + card_w / 2, 0.397),
        color=colors[2][0],
        zorder=4,
    )

    verifier_x = 0.24
    verifier_y = 0.105
    verifier_w = 0.49
    verifier_h = 0.105
    audits.append(
        add_box(
            ax,
            x=verifier_x,
            y=verifier_y,
            width=verifier_w,
            height=verifier_h,
            title="Target verifier",
            subtitle="verify in parallel • accept the matching prefix",
            facecolor="#FDEDEF",
            edgecolor="#D92D4B",
            title_size=16,
            subtitle_size=10,
            linewidth=2.3,
            shadow=True,
        )
    )
    audits.append(
        add_box(
            ax,
            x=0.79,
            y=0.118,
            width=0.17,
            height=0.079,
            title="Accepted prefix",
            subtitle="+ corrected token",
            facecolor="#EAF7FA",
            edgecolor="#168AAD",
            title_size=12.5,
            subtitle_size=9,
            linewidth=2,
        )
    )

    verifier_targets = [verifier_x + 0.1, verifier_x + verifier_w / 2, verifier_x + verifier_w - 0.1]
    for x, target_x in zip(card_x, verifier_targets, strict=True):
        add_arrow(
            ax,
            (x + card_w / 2, card_y),
            (target_x, verifier_y + verifier_h),
            color="#667085",
            width=1.9,
        )
    add_arrow(
        ax,
        (verifier_x + verifier_w, verifier_y + verifier_h / 2),
        (0.79, 0.1575),
        color="#168AAD",
        width=2.3,
    )

    visible_text = {text.get_text() for text in ax.texts}
    missing = [label for label in REQUIRED_LABELS if label not in visible_text]
    if missing:
        raise RuntimeError(f"Required labels not rendered: {missing}")

    overflow = audit_text_fit(fig, audits) + audit_canvas_text(fig, ax)
    if overflow:
        raise RuntimeError(f"Text overflow detected: {overflow}")

    fig.savefig(
        output,
        dpi=DPI,
        facecolor="white",
        edgecolor="none",
        metadata={"Software": "Matplotlib"},
    )
    plt.close(fig)

    receipt = {
        "width_px": WIDTH_PX,
        "height_px": HEIGHT_PX,
        "text_overflow_count": len(overflow),
        "required_labels": REQUIRED_LABELS,
        "source_urls": SOURCE_URLS,
        "renderer": "Matplotlib",
    }
    output.with_suffix(".layout.json").write_text(json.dumps(receipt, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the compact SpecDec method comparison figure.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
