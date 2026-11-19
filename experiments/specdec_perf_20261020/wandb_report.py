"""Build the W&B report for the Q30/Q235 math and Q30 SWE SpecDec results.

Usage: python wandb_report.py [--publish] [--dry-run]
Needs `wandb` and `wandb-workspaces`, and a W&B API key. Saves a draft unless --publish.
"""

import argparse
from dataclasses import dataclass

import wandb_workspaces.reports.v2 as wr

ENTITY = "nvidia"
PROJECT = "sna-async-grpo-gb200"
CONTROL_COLOR = "#8c8c8c"
X = "nemo_rl/step"

STEP = "timing/train/total_step_time"
GEN = "timing/train/generation"
EXPOSED_GEN = "timing/train/exposed_generation"
ACCEPT = "train/vllm/spec_acceptance_length"
REWARD = "train/reward"
MODEL_CALL = "train/swe_agents_train/total_model_call_time/mean"
FINAL_EVAL_MAX = "train/swe_agents_train/final_eval_time/max"


@dataclass(frozen=True)
class Cohort:
    title: str
    controls: tuple[str, ...]
    arms: tuple[str, ...]
    metrics: tuple[tuple[str, str], ...]


MATH_SYNC = (
    (STEP, "Step time (s)"),
    (GEN, "Generation (s)"),
    (ACCEPT, "Acceptance length"),
    (REWARD, "Reward"),
)
MATH_ASYNC = (
    (STEP, "Step time (s)"),
    (EXPOSED_GEN, "Exposed generation (s)"),
    (ACCEPT, "Acceptance length"),
    (REWARD, "Reward"),
)
SWE = (
    (STEP, "Step time (s)"),
    (MODEL_CALL, "Model calls per trajectory (s)"),
    (EXPOSED_GEN, "Exposed generation (s)"),
    (ACCEPT, "Acceptance length"),
)

Q30_MATH = [
    Cohort(
        "Q30 OpenMath 4K sync, K sweep",
        ("u4dfifzd",),
        ("e0rbf6qi", "a6tkn96x", "ushbu743", "39u4hgay", "skw8hq4y", "yt4txoru"),
        MATH_SYNC,
    ),
    Cohort(
        "Q30 OpenMath 4K async, K sweep",
        ("w3khdyvq",),
        ("0n288t0z", "0v0u23i6", "9ag16r1t", "2uisfuxp", "rmxee20f", "wgibatgs"),
        MATH_ASYNC,
    ),
    Cohort("Q30 OpenMath 32K sync", ("mtq03gjw",), ("w6wwvie8", "pcxm3sep"), MATH_SYNC),
    Cohort(
        "Q30 OpenMath 32K async", ("pl4om4av",), ("2thidk2d", "dmkw12v7"), MATH_ASYNC
    ),
    Cohort("Q30 DAPO 4K sync", ("eu22cy77",), ("u0rpswlc", "w841miy0"), MATH_SYNC),
    Cohort("Q30 DAPO 4K async", ("sff1oxfz",), ("m6ptfo17", "qcieowm5"), MATH_ASYNC),
    Cohort("Q30 DAPO 32K sync", ("tjld3pb4",), ("3ljkmdv4", "fp42idvu"), MATH_SYNC),
    Cohort("Q30 DAPO 32K async", ("qp7roui9",), ("jkqj9yeu", "p4th87d2"), MATH_ASYNC),
    Cohort("Q30 DAPO 40K sync", ("nnabm1a1",), ("fkls57tv", "51jxxxjj"), MATH_SYNC),
]

Q235_MATH = [
    Cohort(
        "Q235 OpenMath 8K sync, K sweep",
        ("nx1k0nig",),
        ("zoj3cy96", "3rxkurwl", "oj47xnqq", "cypbqdxt", "b13a2gei", "2x566mtk"),
        MATH_SYNC,
    ),
    Cohort(
        "Q235 OpenMath 8K async, K sweep",
        ("scj79oey",),
        ("b8lac1vq", "bvsu2zv0", "nerem33s", "nl3tvc8h", "2hbxe91k", "ukp1myw1"),
        MATH_ASYNC,
    ),
]

SWE_COHORTS = {
    "thinking": Cohort(
        "Q30 SWE, Thinking drafter, in_order sampler",
        ("1lxfxvr1",),
        ("6vtyjlwj", "7fj6qias", "hsq3r2mb", "yxj0300h"),
        SWE,
    ),
    "swe_trained": Cohort(
        "Q30 SWE, SWE-trained drafter, in_order sampler",
        ("wcve38fi",),
        ("puem1egq", "2nwt2a86", "aftqrcnj", "ulofqx3g"),
        SWE,
    ),
    "ready_first": Cohort(
        "Q30 SWE, ready-first sampler",
        ("v8qxbpu0", "jwhmbw4i"),
        ("b9p1bw4k", "nkkzmwb6"),
        SWE,
    ),
    "gbs256": Cohort("Q30 SWE, GBS 256", ("jzwzpphm",), ("c4k4aiq5",), SWE),
    "staleness2": Cohort(
        "Q30 SWE, staleness 2 (two repeats)",
        ("iax111vg", "pbeh824e"),
        ("j2wdc3e7", "8c63wpzw"),
        SWE,
    ),
}


def md(text: str) -> wr.MarkdownBlock:
    return wr.MarkdownBlock(text=text.strip())


def panel_grid(cohort: Cohort, columns: int = 2) -> wr.PanelGrid:
    width = 24 // columns
    ids = cohort.controls + cohort.arms
    id_list = ", ".join(f"'{i}'" for i in ids)
    runset = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name=cohort.title,
        filters=f"ID in [{id_list}]",
        run_settings={i: wr.RunSettings(color=CONTROL_COLOR) for i in cohort.controls},
    )
    panels = [
        wr.LinePlot(
            title=title,
            x=X,
            y=[key],
            title_x="Trainer step",
            title_y=title,
            smoothing_type="none",
            legend_position="south",
            layout=wr.Layout(
                x=(n % columns) * width, y=(n // columns) * 8, w=width, h=8
            ),
        )
        for n, (key, title) in enumerate(cohort.metrics)
    ]
    return wr.PanelGrid(runsets=[runset], panels=panels)


INTRO = """
DFlash and DSpark add speculative decoding (SpecDec) with trained drafters to NeMo RL GRPO. This report shows how much of the RL step SpecDec saves on Qwen3-30B-A3B (Q30) and Qwen3-235B-A22B (Q235) math GRPO, and on Q30 SWE RL.

**Setup.** OCI-HSG GB200, NeMo RL latest main plus the draft co-training PRs (#3701-#3757), vLLM `flashinfer_trtllm` MoE backend, CUDA graphs on, 20 steps, temperature 1.0. Each arm uses the shipped performance recipe and adds only the SpecDec settings. Each arm runs against a control of its own cohort (gray in all charts). Math speedups use the steps 3-20 that both runs logged. Speedup is control time divided by arm time. All drafters are frozen unless the row says online.

**How to read the charts.** The x axis is the trainer step (`nemo_rl/step`). Sync runs show the generation time. Async runs show the exposed generation, the part of generation that training does not hide.
"""

HEADLINE = """
| Model | Workload | Mode | Best arm | E2E step speedup | Generation speedup | Acceptance length | E2E ceiling |
|---|---|---|---|---|---|---|---|
| Q30 | OpenMath 4K | sync | DSpark K6 / DFlash K7 | 1.21x / 1.21x | 1.81x / 1.74x | 3.85 / 3.86 | 1.61x |
| Q30 | OpenMath 4K | async | DSpark K8 / DFlash K7 | 1.04x / 1.04x (noise) | train-bound | 4.12 / 3.82 | 1.01x |
| Q30 | OpenMath 32K | sync | DSpark K8 / DFlash K7 | 1.55x / 1.30x | 1.75x / 1.39x | 3.57 / 2.88 | 5.33x |
| Q30 | OpenMath 32K | async | DSpark K8 / DFlash K7 | **1.60x** / 1.40x | 3.13x / 2.05x | 3.56 / 2.78 | 1.76x |
| Q30 | DAPO 4K | sync | DFlash K7 / DSpark K8 | 1.15x / 1.14x | 1.63x / 1.55x | 3.85 / 4.14 | 1.53x |
| Q30 | DAPO 32K | sync | DSpark K8 / DFlash K7 | 1.55x / 1.33x | 1.77x / 1.43x | 3.64 / 2.99 | 5.41x |
| Q30 | DAPO 32K | async | DSpark K8 / DFlash K7 | 1.46x / 1.41x | 2.46x / 2.16x | 3.62 / 2.98 | 2.17x |
| Q30 | DAPO 40K | sync | DSpark K5 / DFlash K5 | 1.52x / 1.52x | 2.27x / 2.28x | 3.19 / 3.28 | 2.57x |
| Q235 | OpenMath 8K | sync | DSpark K4 / DFlash K3 | 1.30x / 1.29x | 1.65x / 1.63x | 2.73 / 2.48 | 2.09x |
| Q235 | OpenMath 8K | async | DSpark K4 / DFlash K3 | **1.39x** / 1.36x | 3.03x / 2.94x | 2.70 / 2.46 | 1.37x |
| Q30 | SWE RL, GBS 32, ready-first | async | DSpark K8 (SWE-trained) | **1.23x** [1.19, 1.28] | 2.01x model calls | 3.31 | - |
| Q30 | SWE RL, GBS 256, ready-first | async | DSpark K8 (SWE-trained) | 1.17x [1.14, 1.20] | 2.04x model calls | - | 1.54x |

The E2E ceiling is the step speedup if generation took zero time (control step / (control step - control generation)). SWE intervals are 90% bootstrap intervals over the 20 paired steps.
"""

TAKEAWAYS = """
- **The gain follows the generation share, not the drafter.** Long outputs (Q30 32K and 40K) and the large model (Q235) give 1.3-1.6x on the step. OpenMath 4K sync gives 1.14-1.21x.
- **Async 4K math is train-bound.** Exposed generation is about 1% of the step, so SpecDec has nothing to save (0.99-1.04x).
- **Async runs with a small exposed share get most of their ceiling.** Q235 async gets 0.99-1.06 of its 1.37x ceiling. Q30 32K async DSpark K8 gets 0.87 of 1.76x.
- **DSpark beats DFlash on long outputs.** On 32K sync, DSpark gives 1.55x against 1.30-1.33x, with acceptance length 3.6 against 2.9-3.0.
- **The best K is 2-4 below the block size (8) on math.** On SWE with an SWE-trained drafter, the block-size K wins.
- **On SWE, measure model calls, not the step.** A drafter trained on SWE rollouts doubles model-call speed (2.0x). The step gain shows only with the ready-first sampler, because the in_order step waits for 1200 s final-eval timeouts.
"""

Q30_MATH_TEXT = """
Each panel set is one cohort: one control (gray) and the drafter arms that ran against it. The OpenMath 4K panels hold the full K sweep (DFlash K3/K5/K7, DSpark K4/K6/K8).

- **4K sync:** the K curve is flat (within 2% from K3 to K8). Best: DSpark K6 1.21x, DFlash K5 1.20x in this cohort.
- **4K async:** all arms sit at the control step time. Exposed generation is near zero.
- **32K and 40K:** generation is 61-82% of the sync control step. DSpark K8 gives 1.55x on OpenMath 32K sync and DAPO 32K sync. The 32K and DAPO 40K sync controls crashed after step 14, 12 and 16, so their comparisons use 14, 12 and 16 paired steps.
- In every frozen arm, policy training, logprob and refit stay within 1-4% of the control. A frozen drafter adds no training work.
"""

Q235_MATH_TEXT = """
Q235 uses the 8K OpenMath recipe: 16 nodes sync, 32 nodes async 1-off.

| Mode | DFlash K3 / K5 / K7 | DSpark K4 / K6 / K8 |
|---|---|---|
| sync | 1.29x / 1.29x / 1.26x | 1.30x / 1.28x / 1.27x |
| async | 1.36x / 1.35x / 1.33x | 1.39x / 1.38x / 1.33x |

The acceptance length goes up with K (DFlash 2.46 to 2.86, DSpark 2.70 to 3.03), but the block-size K is 3-5% slower. At temperature 1.0 the per-token acceptance is 0.25-0.41, so the last draft slots rarely land but still cost verify tokens. **Async Q235 removes almost all exposed generation: DSpark K4 reaches 1.39x against a 1.37x ceiling.** The sync K3 and K4 arms are reruns of two arms that died at startup.
"""

SWE_THINKING_TEXT = """
Q30 SWE RL runs an OpenHands agent in NeMo Gym on the SWE blend (SWE-Gym, SWE-bench, SWE-rebench tasks), 131K max length, GBS 32 (4 prompts x 8), 9 nodes. Model calls take about one fifth of a trajectory. The step waits for the slowest final eval (patch tests in a container).

**With the general Thinking drafter, DSpark gives 1.31x on model calls. DFlash gives 1.01x.** Online co-training (draft lr 1e-4, TP4) lifts DFlash K3 from 1.01x to 1.30x. The step ratios (0.90-1.07x) all have 90% intervals that include 1. Final-eval time windows set the step, not the drafter.

| Arm | Model-call ratio [90%] | Step ratio [90%] | Acceptance length |
|---|---|---|---|
| frozen DFlash K3 | 1.01x [0.99, 1.03] | 0.90x [0.73, 1.11] | 1.28 |
| frozen DSpark K4 | 1.31x [1.28, 1.34] | 0.90x [0.74, 1.10] | 1.76 |
| online DFlash K3 | 1.30x [1.24, 1.36] | 1.05x [0.92, 1.23] | 1.66 |
| online DSpark K4 | 1.31x [1.26, 1.36] | 1.07x [0.95, 1.24] | 1.76 |
"""

SWE_TRAINED_TEXT = """
**A drafter trained on SWE rollouts doubles model-call speed.** These DFlash and DSpark drafters learned from SWE trajectories that Q30 generated again (block size 8, 8K sliding-window attention). The control and arms use the V2 single-controller path.

| Arm | Model-call ratio [90%] | Step ratio [90%] | Acceptance length |
|---|---|---|---|
| frozen DFlash K3 | 1.83x [1.78, 1.87] | 1.06x [0.99, 1.19] | 2.48 |
| frozen DFlash K7 | 2.06x [2.03, 2.10] | 0.98x [0.77, 1.16] | 3.22 |
| frozen DSpark K4 | 1.91x [1.89, 1.93] | 1.10x [1.02, 1.23] | 2.75 |
| frozen DSpark K8 | 2.00x [1.96, 2.03] | 1.03x [0.95, 1.17] | 3.67 |

On SWE the block-size K beats the small K, the opposite of math. Online co-training at lr 1e-4 makes these drafters 0.05-0.18x worse, so keep an SWE-fit drafter frozen. **The step does not move: four 1200 s final-eval timeouts take 54-66% of the 20-step time in every run.**
"""

SWE_RF_TEXT = """
**The ready-first sampler turns the model-call gain into a step gain.** Ready-first trains on the groups that finish first and moves a slow group to a later step (`max_staleness_versions: 1`). The step spread drops from 105-121% to 18%, so 20 steps can resolve the gain.

| Pair | Model-call ratio [90%] | Step ratio [90%] | Steps faster |
|---|---|---|---|
| in_order, DSpark K8 | 2.00x [1.96, 2.03] | 1.03x [0.95, 1.17] | 13/20 |
| ready-first, DSpark K8 | 2.01x [1.99, 2.04] | **1.23x [1.19, 1.28]** | **20/20** |
| ready-first + eval 450 s, DSpark K8 | 1.98x [1.94, 2.02] | 1.22x [1.08, 1.38] | 14/20 |

In the eval-450 pair the seconds map one to one: model calls save 35.0 s per trajectory, exposed generation saves 35.4 s and the step saves 31.9 s.
"""

SWE_GBS_TEXT = """
**GBS 256 triples SWE throughput, and DSpark K8 keeps its 2x model-call gain.** The same 9 nodes run 32 prompts x 8. Training becomes the largest phase (48-55% of the step) on one training node.

| Q30 V2, steps 2-20 | GBS 32 control | GBS 32 DSpark K8 | GBS 256 control | GBS 256 DSpark K8 |
|---|---|---|---|---|
| Step (s) | 175.3 | 144.1 | 457.9 | 392.9 |
| Exposed generation (s) | 123.9 | 88.8 | 161.2 | 99.7 |
| Samples per second | 0.18 | 0.22 | 0.55 | 0.64 |
| Model-call ratio | 1 | 1.98x | 1 | 2.04x |
| Step ratio [90%] | 1 | 1.22x [1.08, 1.38] | 1 | 1.17x [1.14, 1.20] |

**DSpark K8 at GBS 256 gives 3.5x the samples per second of the GBS 32 control on the same nodes.** The batch gives 3.0x and the drafter 1.17x on top.
"""

SWE_STALE_TEXT = """
**Staleness 2 and SpecDec multiply.** `max_staleness_versions: 2` with a 3 x PPS in-flight pool cuts the GBS 32 step by 1.67x without a drafter. DSpark K8 still gives 1.23x [1.07, 1.41] on top (R1 1.19x, R2 1.27x).

| Q30 V2 GBS 32, steps 2-20 | Staleness 1 | Staleness 2 | Ratio [90%] |
|---|---|---|---|
| Control step (s) | 175.7 | 105.2 | 1.67x [1.46, 1.92] |
| DSpark K8 step (s) | 141.2 | 85.6 | 1.65x [1.47, 1.84] |

Staleness 2 with DSpark K8 is 2.05x faster than the staleness 1 control (175.7 s to 85.6 s).
"""

CLOSING = """
**SpecDec pays off where decode sets the step time, and only there.** On math, the generation share of the control sets the ceiling, and DSpark at a K 2-4 below the block size gets 0.4-1.0 of it. On SWE, an SWE-trained drafter doubles model-call speed, but the gain reaches the step only when the sampler stops waiting for slow final evals.

Source: branch `sna/draft-cotrain-specdec-perf-20261020` on github.com/seonjinn/RL, `experiments/specdec_perf_20261020/report/README.md`. This report is generated by `experiments/specdec_perf_20261020/wandb_report.py`.
"""


def build() -> wr.Report:
    blocks: list = [
        wr.TableOfContents(),
        md(INTRO),
        wr.H1("Summary"),
        md(HEADLINE),
        md(TAKEAWAYS),
        wr.HorizontalRule(),
        wr.H1("Qwen3-30B-A3B math GRPO"),
        md(Q30_MATH_TEXT),
    ]
    for cohort in Q30_MATH:
        blocks += [wr.H3(cohort.title), panel_grid(cohort)]
    blocks += [
        wr.HorizontalRule(),
        wr.H1("Qwen3-235B-A22B math GRPO"),
        md(Q235_MATH_TEXT),
    ]
    for cohort in Q235_MATH:
        blocks += [wr.H3(cohort.title), panel_grid(cohort)]
    blocks += [wr.HorizontalRule(), wr.H1("Qwen3-30B-A3B SWE RL")]
    for key, text in (
        ("thinking", SWE_THINKING_TEXT),
        ("swe_trained", SWE_TRAINED_TEXT),
        ("ready_first", SWE_RF_TEXT),
        ("gbs256", SWE_GBS_TEXT),
        ("staleness2", SWE_STALE_TEXT),
    ):
        cohort = SWE_COHORTS[key]
        blocks += [wr.H2(cohort.title), md(text), panel_grid(cohort)]
    blocks += [wr.HorizontalRule(), wr.H1("Key takeaway"), md(CLOSING)]
    return wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Speculative Decoding in GRPO RL: Qwen3-30B-A3B and Qwen3-235B-A22B Math, Qwen3-30B-A3B SWE",
        description="DFlash/DSpark SpecDec step, generation and model-call speedups on GB200, latest main plus the draft co-training PRs.",
        width="fluid",
        blocks=blocks,
    )


CORE_MATH_SYNC = (
    (STEP, "Step time (s)"),
    (GEN, "Generation (s)"),
    (ACCEPT, "Acceptance length"),
)
CORE_MATH_ASYNC = (
    (STEP, "Step time (s)"),
    (EXPOSED_GEN, "Exposed generation (s)"),
    (ACCEPT, "Acceptance length"),
)
CORE_SWE = (
    (STEP, "Step time (s)"),
    (MODEL_CALL, "Model calls per trajectory (s)"),
    (EXPOSED_GEN, "Exposed generation (s)"),
)

CORE_Q30_MATH = [
    # The DSpark K6 and DFlash K7 headline numbers come from two cohorts, so this panel holds both controls.
    Cohort(
        "OpenMath 4K sync",
        ("u4dfifzd", "mkoug4lw"),
        ("skw8hq4y", "00gp00ai"),
        CORE_MATH_SYNC,
    ),
    Cohort(
        "OpenMath 4K async (train-bound)",
        ("w3khdyvq",),
        ("wgibatgs", "9ag16r1t"),
        CORE_MATH_ASYNC,
    ),
    Cohort(
        "OpenMath 32K sync", ("mtq03gjw",), ("pcxm3sep", "w6wwvie8"), CORE_MATH_SYNC
    ),
    Cohort(
        "OpenMath 32K async", ("pl4om4av",), ("dmkw12v7", "2thidk2d"), CORE_MATH_ASYNC
    ),
    Cohort("DAPO 4K sync", ("eu22cy77",), ("u0rpswlc", "w841miy0"), CORE_MATH_SYNC),
    Cohort("DAPO 32K sync", ("tjld3pb4",), ("fp42idvu", "3ljkmdv4"), CORE_MATH_SYNC),
    Cohort("DAPO 32K async", ("qp7roui9",), ("p4th87d2", "jkqj9yeu"), CORE_MATH_ASYNC),
    Cohort("DAPO 40K sync", ("nnabm1a1",), ("51jxxxjj", "fkls57tv"), CORE_MATH_SYNC),
]

CORE_Q235_MATH = [
    Cohort("OpenMath 8K sync", ("nx1k0nig",), ("cypbqdxt", "zoj3cy96"), CORE_MATH_SYNC),
    Cohort(
        "OpenMath 8K async", ("scj79oey",), ("nl3tvc8h", "b8lac1vq"), CORE_MATH_ASYNC
    ),
]

CORE_SWE_COHORTS = [
    Cohort("GBS 32, ready-first", ("v8qxbpu0",), ("b9p1bw4k",), CORE_SWE),
    Cohort("GBS 256, ready-first", ("jzwzpphm",), ("c4k4aiq5",), CORE_SWE),
]

CORE_INTRO = """
This report keeps only the headline results of DFlash and DSpark speculative decoding (SpecDec) in NeMo RL GRPO. Each chart shows one control (gray) and the best arm of each drafter. The full study, with every K and every SWE cohort, is in the companion report "Speculative Decoding in GRPO RL: Qwen3-30B-A3B and Qwen3-235B-A22B Math, Qwen3-30B-A3B SWE".

**Setup.** OCI-HSG GB200, NeMo RL latest main plus the draft co-training PRs (#3701-#3757), CUDA graphs on, 20 steps, temperature 1.0, frozen drafters. Math speedups use steps 3-20. Speedup is control time divided by arm time. Sync charts show generation time. Async charts show exposed generation, the part of generation that training does not hide.
"""

CORE_SWE_TEXT = """
Q30 SWE RL runs an OpenHands agent in NeMo Gym, 131K max length, 9 nodes, with a DSpark K8 drafter trained on SWE rollouts. Model calls are about one fifth of a trajectory.

**The ready-first sampler is the condition for a step gain.** With the default in_order sampler, the step waits for slow final evals (1200 s timeouts), and the 2.0x model-call gain does not show in the step (1.03x [0.95, 1.17]). Ready-first trains on the groups that finish first, so the model-call gain reaches the step.

| Ready-first, DSpark K8 | Model-call ratio [90%] | Step ratio [90%] | Steps faster |
|---|---|---|---|
| GBS 32 | 2.01x [1.99, 2.04] | **1.23x [1.19, 1.28]** | 20/20 |
| GBS 256 | 2.04x | **1.17x [1.14, 1.20]** | 19/20 |

The 90% intervals are bootstrap intervals over paired steps (GBS 256 uses steps 2-20). At GBS 256, DSpark K8 raises throughput from 0.55 to 0.64 samples per second.
"""

CORE_CLOSING = """
**SpecDec pays off where decode sets the step time, and only there.** Long outputs (Q30 32K and 40K) and the large model (Q235) give 1.3-1.6x on the math step. Async 4K math is train-bound, so SpecDec saves nothing there. On SWE, an SWE-trained DSpark drafter doubles model-call speed, and the ready-first sampler turns that into a 1.17-1.23x step gain.
"""


def build_core() -> wr.Report:
    blocks: list = [
        wr.TableOfContents(),
        md(CORE_INTRO),
        wr.H1("Summary"),
        md(HEADLINE),
        wr.HorizontalRule(),
    ]
    for heading, cohorts in (
        ("Qwen3-30B-A3B math GRPO", CORE_Q30_MATH),
        ("Qwen3-235B-A22B math GRPO", CORE_Q235_MATH),
    ):
        blocks.append(wr.H1(heading))
        for cohort in cohorts:
            blocks += [wr.H3(cohort.title), panel_grid(cohort, columns=3)]
    blocks += [wr.HorizontalRule(), wr.H1("Qwen3-30B-A3B SWE RL"), md(CORE_SWE_TEXT)]
    for cohort in CORE_SWE_COHORTS:
        blocks += [wr.H3(cohort.title), panel_grid(cohort, columns=3)]
    blocks += [wr.HorizontalRule(), wr.H1("Key takeaway"), md(CORE_CLOSING)]
    return wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Speculative Decoding in GRPO RL: Headline Results",
        description="Best DFlash/DSpark arms for Q30 and Q235 math GRPO and Q30 SWE RL with the ready-first sampler, on GB200.",
        width="fluid",
        blocks=blocks,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--core", action="store_true", help="build the headline-only report"
    )
    args = parser.parse_args()
    report = build_core() if args.core else build()
    if args.dry_run:
        print(f"{len(report.blocks)} blocks built")
        return
    report.save(draft=not args.publish)
    print(report.url)


if __name__ == "__main__":
    main()
