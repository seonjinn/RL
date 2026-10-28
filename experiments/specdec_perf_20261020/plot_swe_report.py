"""Draw the SWE RL SpecDec plots from the report data tables.

    python plot_swe_report.py [report/data]

Reads swe_variation.csv, swe_sd2x2_ratios.csv, swe_sd2x2_arms.csv and
swe_sampler_steps.csv. Writes report/plots/swe_speedup.png (model-call and
step speedup per run), report/plots/swe_sampler_steps.png (step time per step
for each sampler), report/plots/swe_trajectory.png (2x2 trajectory phases and
step) and report/plots/swe_drafter_domain.png (frozen against online drafters).
"""

import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from plot_report import (  # noqa: E402
    CEILING,
    DRAFTERS,
    GRID,
    INK,
    INK_MUTED,
    SURFACE,
    style,
)
from plot_swe_workload import read  # noqa: E402

HERE = Path(__file__).parent
CONTROL = "#9b9a95"
# (group header, [(label, drafter, online, source, key)]). Source "var" keys
# swe_variation.csv by run; source "2x2" keys swe_sd2x2_ratios.csv by id.
SPEEDUP_GROUPS = [
    (
        "Thinking drafter, in_order sampler",
        [
            (
                "DFlash K3 frozen",
                "dflash",
                False,
                "var",
                "sd1022-sc-r16-frozen-dflash-k3",
            ),
            (
                "DFlash K3 online",
                "dflash",
                True,
                "var",
                "sd1022-sc-r16-online-dflash-k3-lr1e4-tp4",
            ),
            (
                "DSpark K4 frozen",
                "dspark",
                False,
                "var",
                "sd1022-sc-r16-frozen-dspark-k4-r2",
            ),
            (
                "DSpark K4 online",
                "dspark",
                True,
                "var",
                "sd1022-sc-r16-online-dspark-k4-lr1e4-tp4",
            ),
        ],
    ),
    (
        "SWE-trained drafter, in_order sampler",
        [
            (
                "DFlash K7",
                "dflash",
                False,
                "var",
                "sd1023-sc-r16-regen-frozen-dflash-k7",
            ),
            (
                "DSpark K8",
                "dspark",
                False,
                "var",
                "sd1023-sc-r16-regen-frozen-dspark-k8",
            ),
        ],
    ),
    (
        "SWE-trained drafter, eval timeout 600 s",
        [
            (
                "DFlash K7",
                "dflash",
                False,
                "var",
                "sd1024-sc-r16-eval600-regen-frozen-dflash-k7",
            ),
            (
                "DSpark K8",
                "dspark",
                False,
                "var",
                "sd1023-sc-r16-eval600-regen-frozen-dspark-k8",
            ),
        ],
    ),
    (
        "SWE-trained drafter, ready-first sampler",
        [
            (
                "DFlash K7",
                "dflash",
                False,
                "var",
                "sd1024-sc-r16-readyfirst-regen-frozen-dflash-k7",
            ),
            (
                "DSpark K8",
                "dspark",
                False,
                "var",
                "sd1023-sc-r16-readyfirst-regen-frozen-dspark-k8",
            ),
        ],
    ),
    (
        "SWE-trained drafter, ready-first + eval 450 s",
        [
            (
                "DFlash K7",
                "dflash",
                False,
                "var",
                "sd1024-sc-r16-rf-eval450-drop4-regen-frozen-dflash-k7",
            ),
            (
                "DSpark K8",
                "dspark",
                False,
                "var",
                "sd1024-sc-r16-rf-eval450-drop4-regen-frozen-dspark-k8",
            ),
            ("DSpark K8, 2x2 L0 (40 steps)", "dspark", False, "2x2", "Pooled-b0-d0"),
            ("DSpark K8, 2x2 L5 (40 steps)", "dspark", False, "2x2", "Pooled-b5-d5"),
            (
                "DSpark K8, GBS 256",
                "dspark",
                False,
                "var",
                "sd1026-sc-r16-gbs256-rf-eval450-drop4-regen-frozen-dspark-k8",
            ),
        ],
    ),
]
SAMPLERS = [
    ("in_order", "in_order"),
    ("eval600", "eval timeout 600 s"),
    ("readyfirst", "ready-first"),
    ("rf_eval450", "ready-first + eval 450 s"),
]
TIMEOUT_STEPS = [2, 7, 10, 15]
ARMS_2X2 = [
    ("b0", "L0, no drafter"),
    ("d0", "L0, DSpark K8"),
    ("b5", "L5 fixes, no drafter"),
    ("d5", "L5 fixes, DSpark K8"),
]
TRAJ_PHASES = [
    (("pre_s", "connect_s", "init_s"), "Startup", "#eda100"),
    (("model_s",), "Model calls", "#2a78d6"),
    (("exec_s",), "Tool execution", "#1baf7a"),
    (("final_eval_s",), "Final eval", "#6250d6"),
]
# (group, label, drafter, frozen run, online run), all in swe_variation.csv.
DOMAIN_ROWS = [
    (
        "Thinking",
        "DFlash K3",
        "dflash",
        "sd1022-sc-r16-frozen-dflash-k3",
        "sd1022-sc-r16-online-dflash-k3-lr1e4-tp4",
    ),
    (
        "Thinking",
        "DSpark K4",
        "dspark",
        "sd1022-sc-r16-frozen-dspark-k4-r2",
        "sd1022-sc-r16-online-dspark-k4-lr1e4-tp4",
    ),
    (
        "SWE-trained",
        "DFlash K3",
        "dflash",
        "sd1023-sc-r16-regen-frozen-dflash-k3",
        "sd1023-sc-r16-regen-online-dflash-k3-lr1e4-sw8k",
    ),
    (
        "SWE-trained",
        "DFlash K7",
        "dflash",
        "sd1023-sc-r16-regen-frozen-dflash-k7",
        "sd1023-sc-r16-regen-online-dflash-k7-lr1e4-sw8k",
    ),
    (
        "SWE-trained",
        "DSpark K4",
        "dspark",
        "sd1023-sc-r16-regen-frozen-dspark-k4",
        "sd1023-sc-r16-regen-online-dspark-k4-lr1e4-sw8k",
    ),
    (
        "SWE-trained",
        "DSpark K8",
        "dspark",
        "sd1023-sc-r16-regen-frozen-dspark-k8",
        "sd1023-sc-r16-regen-online-dspark-k8-lr1e4-sw8k",
    ),
]


def fmt_ratio(v: float) -> str:
    # Half-up from the CSV digits, so 1.325 prints 1.33x as in the README tables.
    return f"{Decimal(repr(v)).quantize(Decimal('0.01'), ROUND_HALF_UP)}x"


def drafter_marker(ax: Axes, x: float, y: float, drafter: str, online: bool) -> None:
    _, color, marker = DRAFTERS[drafter]
    ax.scatter(
        x,
        y,
        s=60,
        marker=marker,
        color=SURFACE if online else color,
        edgecolors=color if online else SURFACE,
        linewidths=1.5,
        zorder=4,
    )


def drafter_legend(online: bool) -> list[Artist]:
    handles: list[Artist] = [
        Line2D([], [], marker=m, linestyle="", color=c, markersize=8, label=n)
        for n, c, m in DRAFTERS.values()
    ]
    if online:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor=SURFACE,
                markeredgecolor=INK_MUTED,
                markersize=8,
                label="Online (hollow)",
            )
        )
    return handles


def speedup_row(source: str, key: str, var: dict, sd2x2: dict) -> dict[str, float]:
    if source == "2x2":
        r = sd2x2[key]
        return {
            "calls": float(r["model_ratio"]),
            "step": float(r["step_ratio"]),
            "lo": float(r["step_ratio_p05"]),
            "hi": float(r["step_ratio_p95"]),
        }
    r = var[key]
    return {
        "calls": float(r["model_call_ratio"]),
        "step": float(r["step_ratio"]),
        "lo": float(r["step_ratio_p05"]),
        "hi": float(r["step_ratio_p95"]),
    }


def plot_speedup(data_dir: Path, path: Path) -> None:
    var = {r["run"]: r for r in read(data_dir / "swe_variation.csv")}
    sd2x2 = {r["id"]: r for r in read(data_dir / "swe_sd2x2_ratios.csv")}
    slots = sum(len(rows) + 1 for _, rows in SPEEDUP_GROUPS)
    fig, ax = plt.subplots(figsize=(9, 0.32 * slots + 1.6), facecolor=SURFACE)
    style(ax)
    x_min, x_max = 0.6, 2.3
    y = 0
    ticks: list[tuple[int, str]] = []
    for header, rows in SPEEDUP_GROUPS:
        ax.text(
            x_min + 0.02,
            y,
            header,
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=INK_MUTED,
        )
        y += 1
        for label, drafter, online, source, key in rows:
            v = speedup_row(source, key, var, sd2x2)
            ax.barh(y, v["calls"] - 1, left=1, height=0.55, color=CEILING, zorder=1)
            color = DRAFTERS[drafter][1]
            ax.plot([v["lo"], v["hi"]], [y, y], color=color, linewidth=1.5, zorder=3)
            drafter_marker(ax, v["step"], y, drafter, online)
            ax.text(
                max(v["calls"], v["hi"]) + 0.03,
                y,
                f"calls {fmt_ratio(v['calls'])}, step {fmt_ratio(v['step'])}",
                va="center",
                fontsize=8,
                color=INK if v["lo"] > 1 else INK_MUTED,
                fontweight="bold" if v["lo"] > 1 else "normal",
            )
            ticks.append((y, label))
            y += 1
    ax.set_yticks([t for t, _ in ticks], [label for _, label in ticks], fontsize=9)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(x_min, x_max + 0.32)
    ax.axvline(1, color=INK_MUTED, linewidth=1, zorder=2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("Speedup over the same-config control (x)", color=INK)
    handles = drafter_legend(online=True)
    handles.append(
        Line2D([], [], color=INK_MUTED, linewidth=1.5, label="Step, 90% interval")
    )
    handles.append(Rectangle((0, 0), 1, 1, color=CEILING, label="Model-call speedup"))
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0, 1),
        labelcolor=INK,
    )
    ax.set_title(
        "Q30 SWE: model calls run 2x faster, the step needs the ready-first sampler",
        loc="left",
        fontsize=11,
        color=INK,
        pad=24,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_sampler_steps(data_dir: Path, path: Path) -> None:
    series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for r in read(data_dir / "swe_sampler_steps.csv"):
        series[(r["sampler"], r["arm"])].append((int(r["step"]), float(r["step_s"])))
    fig, axes = plt.subplots(
        1, len(SAMPLERS), figsize=(12, 3.6), sharey=True, facecolor=SURFACE
    )
    for ax, (sampler, title) in zip(axes, SAMPLERS):
        style(ax)
        for arm, name, color in (
            ("control", "Control", CONTROL),
            ("dspark_k8", "DSpark K8", DRAFTERS["dspark"][1]),
        ):
            pts = sorted(series[(sampler, arm)])
            ys = [s for _, s in pts]
            ax.plot(
                [step for step, _ in pts],
                ys,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3.5,
                label=f"{name}, mean {sum(ys) / len(ys):.0f} s",
            )
        if sampler == "in_order":
            for step in TIMEOUT_STEPS:
                ax.axvline(step, color=GRID, linewidth=6, zorder=0)
            title = f"{title} (shaded: 1200 s eval timeout)"
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("Step", color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    axes[0].set_ylim(0, 1650)
    axes[0].set_ylabel("Step time (s)", color=INK)
    fig.suptitle(
        "Q30 SWE step time: in_order waits for the slowest eval, ready-first does not",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def pooled_arms(data_dir: Path) -> dict[str, dict[str, float]]:
    """Trajectory phases weighted by trajectories, step columns by steps, over R1 and R2."""
    out: dict[str, dict[str, float]] = {}
    rows = read(data_dir / "swe_sd2x2_arms.csv")
    for arm, _ in ARMS_2X2:
        reps = [r for r in rows if r["arm"] == arm]
        n = sum(int(r["trajectories"]) for r in reps)
        steps = sum(int(r["steps"]) for r in reps)
        agg: dict[str, float] = {}
        for key in reps[0]:
            if key.endswith("_s") and key not in ("step_s", "exposed_gen_s"):
                agg[key] = sum(int(r["trajectories"]) * float(r[key]) for r in reps) / n
        for key in ("step_s", "exposed_gen_s"):
            agg[key] = sum(int(r["steps"]) * float(r[key]) for r in reps) / steps
        out[arm] = agg
    return out


def stacked(
    ax: Axes, y: int, parts: list[tuple[float, str, str]], total: float, first: bool
) -> None:
    left = 0.0
    for value, name, color in parts:
        ax.barh(
            y,
            value,
            left=left,
            color=color,
            height=0.6,
            edgecolor=SURFACE,
            linewidth=2,
            label=name if first else None,
        )
        left += value
    ax.text(total + 3, y, f"{total:.0f} s", va="center", fontsize=8, color=INK)


def plot_trajectory(data_dir: Path, path: Path) -> None:
    arms = pooled_arms(data_dir)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 3.8), sharey=True, facecolor=SURFACE
    )
    for y, (arm, _) in enumerate(ARMS_2X2):
        a = arms[arm]
        parts = [
            (sum(a[k] for k in keys), name, color) for keys, name, color in TRAJ_PHASES
        ]
        other = a["traj_mean_s"] - sum(v for v, _, _ in parts)
        parts.append((max(other, 0.0), "Other agent time", GRID))
        stacked(ax1, y, parts, a["traj_mean_s"], y == 0)
        gen = a["exposed_gen_s"]
        stacked(
            ax2,
            y,
            [
                (gen, "Exposed generation", "#2a78d6"),
                (a["step_s"] - gen, "Training and other", GRID),
            ],
            a["step_s"],
            y == 0,
        )
    ax1.set_yticks(range(len(ARMS_2X2)), [label for _, label in ARMS_2X2])
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean trajectory time (s)", color=INK)
    ax1.set_title(
        "Trajectory: fixes cut 21 s with DSpark K8", loc="left", fontsize=10, color=INK
    )
    ax2.set_xlabel("Mean step time (s)", color=INK)
    ax2.set_title(
        "Step: with DSpark K8, the cut does not reach the step",
        loc="left",
        fontsize=10,
        color=INK,
    )
    for ax in (ax1, ax2):
        style(ax)
        ax.set_xlim(0, 215)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(
            frameon=False,
            fontsize=8,
            ncol=3,
            loc="lower left",
            bbox_to_anchor=(0, 1.08),
            labelcolor=INK,
        )
    fig.suptitle(
        "Q30 SWE 2x2, ready-first + eval 450 s, R1 and R2 pooled",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_drafter_domain(data_dir: Path, path: Path) -> None:
    var = {r["run"]: r for r in read(data_dir / "swe_variation.csv")}
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    style(ax)
    labels = []
    for y, (group, label, drafter, frozen, online) in enumerate(DOMAIN_ROWS):
        f = float(var[frozen]["model_call_ratio"])
        o = float(var[online]["model_call_ratio"])
        ax.annotate(
            "",
            xy=(o, y),
            xytext=(f, y),
            arrowprops={"arrowstyle": "-|>", "color": INK_MUTED, "lw": 1.2},
            zorder=2,
        )
        drafter_marker(ax, f, y, drafter, online=False)
        drafter_marker(ax, o, y, drafter, online=True)
        ax.text(
            max(f, o) + 0.05,
            y,
            f"{fmt_ratio(f)} to {fmt_ratio(o)}",
            va="center",
            fontsize=8,
            color=INK,
        )
        labels.append(f"{group} {label}")
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.9, 2.45)
    ax.axvline(1, color=INK_MUTED, linewidth=1, zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("Model-call speedup over the control (x)", color=INK)
    ax.legend(
        handles=drafter_legend(online=True),
        frameon=False,
        fontsize=8,
        loc="upper right",
        labelcolor=INK,
    )
    ax.set_title(
        "Online training at lr 1e-4: lifts the Thinking drafter, hurts the SWE drafter",
        loc="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report" / "data"
    plots = HERE / "report" / "plots"
    plot_speedup(data_dir, plots / "swe_speedup.png")
    plot_sampler_steps(data_dir, plots / "swe_sampler_steps.png")
    plot_trajectory(data_dir, plots / "swe_trajectory.png")
    plot_drafter_domain(data_dir, plots / "swe_drafter_domain.png")


if __name__ == "__main__":
    main()
