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
    "Target context",
    "EAGLE-3",
    "DFlash",
    "DSpark",
    "AUTOREGRESSIVE",
    "PARALLEL",
    "SEMI-AUTOREGRESSIVE",
    "Confidence scheduler",
    "Target verify",
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
        boxstyle="round,pad=0.006,rounding_size=0.014",
        linewidth=2.5,
        facecolor="#FFFFFF",
        edgecolor=edgecolor,
        transform=ax.transAxes,
        zorder=1,
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
    title_size: float = 13,
    subtitle_size: float = 9,
    linewidth: float = 1.6,
) -> BoxAudit:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.01",
        linewidth=linewidth,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
        zorder=3,
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
                y + height * 0.28,
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
    width: float = 1.8,
    mutation_scale: float = 13,
    zorder: float = 2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=width,
            color=color,
            shrinkA=0,
            shrinkB=0,
            transform=ax.transAxes,
            zorder=zorder,
        )
    )


def add_method_header(
    ax: plt.Axes,
    *,
    center_x: float,
    method: str,
    category: str,
    edgecolor: str,
    facecolor: str,
) -> list[BoxAudit]:
    ax.text(
        center_x,
        0.79,
        method,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#17202A",
        transform=ax.transAxes,
        zorder=4,
    )
    return [
        add_box(
            ax,
            x=center_x - 0.095,
            y=0.725,
            width=0.19,
            height=0.044,
            title=category,
            subtitle=None,
            facecolor=facecolor,
            edgecolor=edgecolor,
            title_size=10.2,
            linewidth=1.5,
        )
    ]


def add_context_and_output(
    ax: plt.Axes,
    *,
    center_x: float,
) -> list[BoxAudit]:
    audits = [
        add_box(
            ax,
            x=center_x - 0.09,
            y=0.635,
            width=0.18,
            height=0.055,
            title="Target context",
            subtitle=None,
            facecolor="#F2F4F7",
            edgecolor="#667085",
            title_size=11.2,
        ),
        add_box(
            ax,
            x=center_x - 0.09,
            y=0.235,
            width=0.18,
            height=0.065,
            title="Target verify",
            subtitle="parallel check",
            facecolor="#FDEDEF",
            edgecolor="#D92D4B",
            title_size=11.5,
            subtitle_size=8.5,
        ),
        add_box(
            ax,
            x=center_x - 0.09,
            y=0.115,
            width=0.18,
            height=0.062,
            title="Accepted prefix",
            subtitle="+ corrected token",
            facecolor="#EAF7FA",
            edgecolor="#168AAD",
            title_size=11.2,
            subtitle_size=8.4,
        ),
    ]
    add_arrow(ax, (center_x, 0.235), (center_x, 0.177), color="#168AAD", width=2)
    return audits


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
    token_width = 0.051
    token_height = 0.056
    gap = 0.025
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
                title_size=10.5,
                linewidth=1.5,
            )
        )
        if arrows and index < len(labels) - 1:
            add_arrow(
                ax,
                (token_x + token_width + 0.003, y + token_height / 2),
                (token_x + token_width + gap - 0.003, y + token_height / 2),
                color=edgecolor,
                width=1.4,
                mutation_scale=10,
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
        0.95,
        "Three Ways to Draft the Next K Tokens",
        fontsize=27,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.9,
        "Each method has its own proposal path. Every path ends with target-model verification.",
        fontsize=13.5,
        color="#5B6472",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    panel_y = 0.07
    panel_h = 0.78
    panel_w = 0.295
    panel_x = [0.03, 0.3525, 0.675]
    centers = [x + panel_w / 2 for x in panel_x]
    colors = [
        ("#4472C4", "#EEF4FC"),
        ("#F28E2B", "#FFF4E8"),
        ("#76B900", "#F2F8E9"),
    ]

    audits: list[BoxAudit] = []
    for x, center_x, (edge, fill), method, category in zip(
        panel_x,
        centers,
        colors,
        ("EAGLE-3", "DFlash", "DSpark"),
        ("AUTOREGRESSIVE", "PARALLEL", "SEMI-AUTOREGRESSIVE"),
        strict=True,
    ):
        add_panel(ax, x=x, y=panel_y, width=panel_w, height=panel_h, edgecolor=edge)
        audits.extend(
            add_method_header(
                ax,
                center_x=center_x,
                method=method,
                category=category,
                edgecolor=edge,
                facecolor=fill,
            )
        )
        audits.extend(add_context_and_output(ax, center_x=center_x))

    eagle_center = centers[0]
    ax.add_patch(
        FancyBboxPatch(
            (panel_x[0] + 0.032, 0.458),
            0.231,
            0.088,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            facecolor="none",
            edgecolor=colors[0][0],
            transform=ax.transAxes,
            zorder=2.5,
        )
    )
    audits.extend(
        add_token_row(
            ax,
            x=panel_x[0] + 0.045,
            y=0.475,
            edgecolor=colors[0][0],
            facecolor=colors[0][1],
            arrows=True,
        )
    )
    first_eagle_token_x = panel_x[0] + 0.045 + 0.051 / 2
    add_arrow(ax, (eagle_center, 0.635), (first_eagle_token_x, 0.531), color=colors[0][0])
    add_arrow(ax, (eagle_center, 0.458), (eagle_center, 0.3), color=colors[0][0])
    ax.text(
        eagle_center,
        0.405,
        "one token at a time",
        ha="center",
        va="center",
        fontsize=11,
        color="#344054",
        transform=ax.transAxes,
    )
    ax.text(
        eagle_center,
        0.36,
        "K draft passes",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=colors[0][0],
        transform=ax.transAxes,
    )

    dflash_center = centers[1]
    audits.append(
        add_box(
            ax,
            x=dflash_center - 0.105,
            y=0.455,
            width=0.21,
            height=0.105,
            title="Block-diffusion drafter",
            subtitle="t+1  •  t+2  •  …  •  t+K",
            facecolor=colors[1][1],
            edgecolor=colors[1][0],
            title_size=11.5,
            subtitle_size=9.5,
            linewidth=1.8,
        )
    )
    add_arrow(ax, (dflash_center, 0.635), (dflash_center, 0.56), color=colors[1][0])
    add_arrow(ax, (dflash_center, 0.455), (dflash_center, 0.3), color=colors[1][0])
    ax.text(
        dflash_center,
        0.395,
        "whole block at once",
        ha="center",
        va="center",
        fontsize=11,
        color="#344054",
        transform=ax.transAxes,
    )
    ax.text(
        dflash_center,
        0.35,
        "1 parallel draft pass",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=colors[1][0],
        transform=ax.transAxes,
    )

    dspark_center = centers[2]
    dspark_boxes = [
        (0.535, "Parallel backbone"),
        (0.455, "Light sequential head"),
        (0.375, "Confidence scheduler"),
    ]
    for y, title in dspark_boxes:
        audits.append(
            add_box(
                ax,
                x=dspark_center - 0.1,
                y=y,
                width=0.2,
                height=0.052,
                title=title,
                subtitle=None,
                facecolor=colors[2][1],
                edgecolor=colors[2][0],
                title_size=10.5,
                linewidth=1.5,
            )
        )
    add_arrow(ax, (dspark_center, 0.635), (dspark_center, 0.587), color=colors[2][0])
    add_arrow(ax, (dspark_center, 0.535), (dspark_center, 0.507), color=colors[2][0], zorder=4)
    add_arrow(ax, (dspark_center, 0.455), (dspark_center, 0.427), color=colors[2][0], zorder=4)
    add_arrow(ax, (dspark_center, 0.375), (dspark_center, 0.3), color=colors[2][0])
    visible_text = [text.get_text() for text in ax.texts]
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
        "layout": "1x3",
        "panel_count": 3,
        "target_verifier_count": visible_text.count("Target verify"),
        "text_overflow_count": len(overflow),
        "required_labels": REQUIRED_LABELS,
        "source_urls": SOURCE_URLS,
        "renderer": "Matplotlib",
    }
    output.with_suffix(".layout.json").write_text(json.dumps(receipt, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the 1x3 SpecDec drafter comparison figure.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
