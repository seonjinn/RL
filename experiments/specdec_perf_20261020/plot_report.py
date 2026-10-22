"""Draw the report plots and the best-arm table from the dashboard run summary.

    python plot_report.py [report/data/runs.json]

Writes report/plots/ceiling.png, report/plots/k_sweep.png and
report/data/best_arms.csv. runs.json holds one row per run with the paired
deltas against the control of its cohort (arm / control - 1, steps 3-20).
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).parent
MIN_PAIRED = 8
DRAFTERS = {"dflash": ("DFlash", "#2a78d6", "o"), "dspark": ("DSpark", "#eb6834", "D")}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
CEILING = "#d9d8d3"
SHORT = {
    "Qwen3-30B-A3B": "Q30",
    "Qwen3-235B-A22B": "Q235",
    "Nemotron 3.5 Nano": "N35",
    "Qwen3.5-35B-A3B": "Q35",
}
WORKLOAD = {"default": "4K", "40K": "DAPO 40K", "32K": "32K"}
# Prefer one cohort per model so every K shares a control; Q35 uses stock vLLM (no GDN reuse backport).
K_SWEEP_COHORT = {"Q30": "ksweep", "Q235": "ksweep", "N35": "", "Q35": "gmu08"}
# N35 K3/K4 and K7/K8 ran in different cohorts, each against its own control.
MIXED_COHORT = {"N35"}


def speedup(delta: float | None) -> float | None:
    return None if delta is None else 1 / (1 + delta)


def e2e(r: dict) -> float:
    return 1 / (1 + r["pair_step_delta"])


def workload(r: dict) -> str:
    label = WORKLOAD.get(r["seqlen"], r["seqlen"])
    return f"DAPO {label}" if "dapo" in r["run"] and "DAPO" not in label else label


def is_stock_frozen(r: dict) -> bool:
    return (
        r["drafter"] in DRAFTERS
        and r["drafter_mode"] == "frozen"
        and "gdnreuse" not in r["run"]
    )


def paired(r: dict) -> bool:
    return (r.get("pair_steps") or 0) >= MIN_PAIRED and r.get(
        "pair_step_delta"
    ) is not None


def best_arms(runs: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    best: dict[tuple[str, str, str, str], dict] = {}
    for r in runs:
        if r["model"] not in SHORT or not is_stock_frozen(r) or not paired(r):
            continue
        # Key on the workload, not seqlen: plain 32K math and DAPO 32K share seqlen "32K".
        key = (r["model"], workload(r), r["mode"], r["drafter"])
        if key not in best or r["pair_step_delta"] < best[key]["pair_step_delta"]:
            best[key] = r
    return best


def gen_share(arm: dict, by_run: dict[str, dict]) -> float | None:
    ctl = by_run.get(arm["ref"])
    if ctl is None or not ctl.get("gen_s") or not ctl.get("step_s"):
        return None
    return ctl["gen_s"] / ctl["step_s"]


def write_best_csv(best: dict, by_run: dict[str, dict], path: Path) -> None:
    fields = [
        "model",
        "workload",
        "mode",
        "drafter",
        "k",
        "e2e",
        "gen",
        "train",
        "logprob",
        "refit",
    ]
    fields += [
        "accept_len",
        "gen_share",
        "ceiling",
        "paired_steps",
        "run",
        "job",
        "control",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (model, _, mode, drafter), r in sorted(best.items()):
            share = gen_share(r, by_run)
            row = {
                "model": SHORT[model],
                "workload": workload(r),
                "mode": mode,
                "drafter": DRAFTERS[drafter][0],
                "k": r["k"],
                "accept_len": r.get("accept_len"),
                "gen_share": None if share is None else round(share, 3),
                "ceiling": None if share is None else round(1 / (1 - share), 3),
                "paired_steps": r["pair_steps"],
                "run": r["run"],
                "job": r["job"],
                "control": r["ref"],
            }
            for col, key in [
                ("e2e", "pair_step_delta"),
                ("gen", "pair_gen_delta"),
                ("train", "pair_train_delta"),
                ("logprob", "pair_logprob_delta"),
                ("refit", "pair_refit_delta"),
            ]:
                v = speedup(r.get(key))
                row[col] = None if v is None else round(v, 3)
            w.writerow(row)


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)


def plot_ceiling(best: dict, by_run: dict[str, dict], path: Path) -> None:
    rows: dict[tuple[str, str, str], dict[str, dict]] = {}
    for (model, load, mode, drafter), r in best.items():
        rows.setdefault((model, load, mode), {})[drafter] = r
    ceilings: dict[tuple[str, str, str], float] = {}
    labels: dict[tuple[str, str, str], str] = {}
    for key, arms in rows.items():
        ref_arm = arms.get("dspark") or arms["dflash"]
        share = gen_share(ref_arm, by_run)
        if share is not None:
            ceilings[key] = 1 / (1 - share)
            labels[key] = f"{SHORT[key[0]]} {workload(ref_arm)} {key[2]}"
    order = sorted(ceilings, key=lambda k: ceilings[k])
    x_max = 2.8

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.6), facecolor=SURFACE)
    style(ax)
    for y, key in enumerate(order):
        ceil = ceilings[key]
        ax.barh(y, min(ceil, x_max) - 1, left=1, height=0.5, color=CEILING, zorder=1)
        right = max([min(ceil, x_max)] + [e2e(r) for r in rows[key].values()])
        ax.text(
            right + 0.05,
            y,
            f"{ceil:.2f}x" + (" (off scale)" if ceil > x_max else ""),
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )
        for drafter, r in rows[key].items():
            _, color, marker = DRAFTERS[drafter]
            ax.scatter(
                e2e(r),
                y,
                s=60,
                color=color,
                marker=marker,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=3,
            )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[key] for key in order], fontsize=9)
    ax.set_xlim(0.9, x_max + 0.45)
    ax.axvline(1, color=INK_MUTED, linewidth=1, zorder=2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("E2E step speedup over control (x)", color=INK)
    handles: list[Artist] = [
        Line2D(
            [], [], marker=m, linestyle="", color=c, markersize=8, label=f"{n} (best K)"
        )
        for n, c, m in DRAFTERS.values()
    ]
    handles.append(
        Rectangle((0, 0), 1, 1, color=CEILING, label="Generation-free ceiling")
    )
    ax.legend(
        handles=handles, loc="lower right", frameon=False, fontsize=8, labelcolor=INK
    )
    ax.set_title(
        "Measured E2E speedup against the generation-free ceiling",
        loc="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_k_sweep(runs: list[dict], path: Path) -> None:
    models = [m for m in SHORT if SHORT[m] in K_SWEEP_COHORT]
    fig, axes = plt.subplots(
        1, len(models), figsize=(11, 3.2), sharey=True, facecolor=SURFACE
    )
    for ax, model in zip(axes, models):
        style(ax)
        cohort = K_SWEEP_COHORT[SHORT[model]]
        for drafter, (name, color, marker) in DRAFTERS.items():
            points: dict[int, float] = {}
            for r in runs:
                if (
                    r["model"] != model
                    or r["seqlen"] != "default"
                    or r["mode"] != "sync"
                    or r["drafter"] != drafter
                    or not is_stock_frozen(r)
                    or not paired(r)
                    or cohort not in r["run"]
                    or "dapo" in r["run"]
                ):
                    continue
                points[r["k"]] = max(points.get(r["k"], 0.0), e2e(r))
            ks = sorted(points)
            linestyle = "--" if SHORT[model] in MIXED_COHORT else "-"
            ax.plot(
                ks,
                [points[k] for k in ks],
                color=color,
                linewidth=2,
                linestyle=linestyle,
                zorder=2,
            )
            ax.scatter(
                ks,
                [points[k] for k in ks],
                s=50,
                color=color,
                marker=marker,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=3,
                label=name,
            )
        ax.axhline(1, color=INK_MUTED, linewidth=1, zorder=1)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_xticks(range(3, 9))
        ax.set_xlabel("K (draft tokens)", color=INK)
        suffix = " (two cohorts)" if SHORT[model] in MIXED_COHORT else ""
        ax.set_title(
            f"{SHORT[model]} 4K sync{suffix}", loc="left", fontsize=10, color=INK
        )
    axes[0].set_ylabel("E2E step speedup (x)", color=INK)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.suptitle(
        "E2E speedup by K: the block-size K (7 DFlash, 8 DSpark) never wins clearly",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report/data/runs.json"
    runs = json.loads(src.read_text())
    by_run = {r["run"]: r for r in runs}
    best = best_arms(runs)
    plots = HERE / "report/plots"
    plots.mkdir(parents=True, exist_ok=True)
    write_best_csv(best, by_run, HERE / "report/data/best_arms.csv")
    plot_ceiling(best, by_run, plots / "ceiling.png")
    plot_k_sweep(runs, plots / "k_sweep.png")


if __name__ == "__main__":
    main()
