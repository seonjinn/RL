"""Draw the SWE workload plots from swe_workload_compare.py output.

    python plot_swe_workload.py [report/data]

Reads swe_workload_by_call.csv, workload_wandb.csv and swe_gbs_scaling.csv.
Writes report/plots/swe_turns.png (calls per trajectory and context per call),
report/plots/workload_step.png (one GRPO step, math against SWE) and
report/plots/swe_gbs_step.png (one SWE step at GBS 32 and 256).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_report import GRID, INK, INK_MUTED, SURFACE, style  # noqa: E402

HERE = Path(__file__).parent
# Baseline runs pooled per model. SpecDec arms run the same agent loop.
POOLS = {
    "Q30 SWE (V2)": (["q30_v2_rf_base", "q30_v2_l0_r1", "q30_v2_l0_r2"], "#2a78d6"),
    "Q235 SWE": (["q235_v10_base"], "#eb6834"),
}
MAX_CALL = 80
MIN_ALIVE = 20
STEP_ROWS = [
    ("Q30 math 4K", "math q30 sync"),
    ("Q235 math 8K", "math q235 sync"),
    ("Q30 SWE V2", "swe q30 v2 rf450"),
    ("Q235 SWE", "swe q235 v10"),
]
GBS_ROWS = [
    ("GBS 32 control", "32", "baseline"),
    ("GBS 32 DSpark K8", "32", "dspark-k8"),
    ("GBS 256 control", "256", "baseline"),
    ("GBS 256 DSpark K8", "256", "dspark-k8"),
]
PHASES = [
    ("gen_s", "Generation (exposed)", "#2a78d6"),
    ("logprob_s", "Logprob", "#9b9a95"),
    ("train_s", "Training", "#eb6834"),
    ("refit_s", "Refit", "#52514e"),
]


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def pooled(rows: list[dict], runs: list[str]) -> dict[int, tuple[int, float]]:
    """Per call index: trajectories that reach it and their mean context."""
    n: dict[int, int] = defaultdict(int)
    ctx: dict[int, float] = defaultdict(float)
    for r in rows:
        if r["run"] in runs:
            k, i = int(r["trajectories"]), int(r["call"])
            n[i] += k
            ctx[i] += k * float(r["context_mean"])
    return {i: (n[i], ctx[i] / n[i]) for i in sorted(n)}


def plot_turns(data_dir: Path, path: Path) -> None:
    rows = read(data_dir / "swe_workload_by_call.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for label, (runs, color) in POOLS.items():
        alive = pooled(rows, runs)
        total = alive[1][0]
        ends = {i: n - alive.get(i + 1, (0, 0.0))[0] for i, (n, _) in alive.items()}
        mean_calls = sum(i * e for i, e in ends.items()) / total
        xs = list(range(1, MAX_CALL + 1))
        share = [100 * ends.get(i, 0) / total for i in xs]
        share[-1] += 100 * sum(e for i, e in ends.items() if i > MAX_CALL) / total
        ax1.step(
            xs,
            share,
            where="mid",
            color=color,
            linewidth=2,
            label=f"{label}: {total} trajectories, mean {mean_calls:.1f}",
        )
        idx = [i for i in xs if alive.get(i, (0, 0.0))[0] >= MIN_ALIVE]
        ax2.plot(
            idx,
            [alive[i][1] / 1000 for i in idx],
            color=color,
            linewidth=2,
            label=label,
        )
    ax1.axvline(1, color=INK, linewidth=1, linestyle="--", label="math GRPO: 1 call")
    ax1.set_xlabel(f"LLM calls per trajectory ({MAX_CALL}+ in the last bin)", color=INK)
    ax1.set_ylabel("Trajectories (%)", color=INK)
    ax1.set_title("Turns per rollout", color=INK, loc="left")
    ax2.axhline(0.11, color=INK, linewidth=1, linestyle="--", label="math prompt: 0.1K")
    ax2.set_xlabel(
        f"Call index (indices that {MIN_ALIVE}+ trajectories reach)", color=INK
    )
    ax2.set_ylabel("Mean prompt per call (K tokens)", color=INK)
    ax2.set_title("Each call sends the whole history again", color=INK, loc="left")
    for ax in (ax1, ax2):
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def phase_bars(path: Path, rows: list[tuple[str, dict, str]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=SURFACE)
    for y, (_, r, note) in enumerate(rows):
        left = 0.0
        for key, phase, color in PHASES:
            v = float(r[key]) if r[key] != "nan" else 0.0
            ax.barh(
                y,
                v,
                left=left,
                color=color,
                height=0.6,
                edgecolor=SURFACE,
                linewidth=2,
                label=phase if y == 0 else None,
            )
            left += v
        step = float(r["step_s"])
        ax.barh(
            y,
            max(step - left, 0.0),
            left=left,
            color=GRID,
            height=0.6,
            edgecolor=SURFACE,
            linewidth=2,
            label="Other" if y == 0 else None,
        )
        ax.annotate(note, (step + 10, y), va="center", fontsize=8, color=INK_MUTED)
    ax.set_yticks(range(len(rows)), [name for name, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Mean step time, steps 2-20 (s)", color=INK)
    ax.set_title(title, color=INK, loc="left")
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=5,
        loc="lower right",
        bbox_to_anchor=(1, 1),
        labelcolor=INK,
    )
    ax.set_xlim(0, max(float(r["step_s"]) for _, r, _ in rows) * 1.75)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_step(data_dir: Path, path: Path) -> None:
    by_run = {r["run"]: r for r in read(data_dir / "workload_wandb.csv")}
    rows = []
    for name, run in STEP_ROWS:
        r = by_run[run]
        turns = round(float(r["turns"]))
        note = (
            f"{float(r['step_s']):.0f} s, {float(r['samples']):.0f} samples x {turns} "
            f"{'call' if turns == 1 else 'calls'}, "
            f"{float(r['gen_tokens_per_step']) / 1e6:.2f}M output tokens"
        )
        rows.append((name, r, note))
    phase_bars(path, rows, "One GRPO step: math against SWE")


def plot_gbs(data_dir: Path, path: Path) -> None:
    by_run = {(r["gbs"], r["run"]): r for r in read(data_dir / "swe_gbs_scaling.csv")}
    rows = []
    for name, gbs, run in GBS_ROWS:
        r = by_run[(gbs, run)]
        note = (
            f"{float(r['step_s']):.0f} s, {float(r['samples']):.0f} samples, "
            f"{float(r['samples_per_s']):.2f} samples/s"
        )
        rows.append((name, r, note))
    phase_bars(path, rows, "Q30 SWE step: GBS 32 and 256")


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report" / "data"
    plots = HERE / "report" / "plots"
    plot_turns(data_dir, plots / "swe_turns.png")
    plot_step(data_dir, plots / "workload_step.png")
    plot_gbs(data_dir, plots / "swe_gbs_step.png")


if __name__ == "__main__":
    main()
