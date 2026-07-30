#!/usr/bin/env python3
"""Render the Mamba/MoE TE CUDA Graph result ledger as static HTML."""

import argparse
import csv
import html
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from collect_results import CORRECTNESS_FIELDS, aggregate_performance, steady_state_rows

REPO_ROOT = Path(__file__).parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "experiments"
    / "cuda_graph"
    / "results"
    / "mamba_moe_te_graph_20260729_results.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "cuda_graph"
    / "results"
    / "mamba_moe_te_graph_20260729_report.html"
)
DEFAULT_MCORE_SHA = "100047b517ea91526dc465448fcb3b37b2598388"
DEFAULT_MODEL_SNAPSHOT = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
    "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/"
    "snapshots/97ab8012882a655dc38df4fee47422aca9caca07"
)
DEFAULT_TOKENIZER_SNAPSHOT = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
    "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/"
    "snapshots/2d59de1cbd51c0adf384eb906b766d1aee0e0517"
)

CORRECTNESS_COLUMNS = (
    ("reward_mean", "Reward / accuracy"),
    ("generation_kl_error", "Generation KL error"),
    ("token_mult_prob_error", "Token multiplication probability error"),
    ("policy_kl_error", "Policy KL error"),
    ("js_divergence_error", "JS divergence error"),
    ("sampling_importance_ratio", "Sampling importance ratio"),
    (
        "num_masked_seqs_by_logprob_error",
        "Masked sequences by logprob error",
    ),
    ("policy_loss", "Policy loss"),
    ("grad_norm", "Gradient norm"),
)

TERMINAL_SLURM_FAILURE_STATUSES = frozenset(
    {
        "BOOT_FAIL",
        "DEADLINE",
        "NODE_FAIL",
        "OOM",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)


def escape(value: object) -> str:
    """Escape one value before inserting it in HTML."""
    return html.escape(str(value), quote=True)


def scope_label(scope: str) -> str:
    """Make baseline, whole-layer TE, and config variants unambiguous."""
    if scope in {"baseline-no-cg", "no_cg", "[no_cg]"}:
        return "No-CG baseline (CUDA graphs disabled)"
    if scope in {"whole-layer", "[]", "te-whole-layer"}:
        return "TE whole-layer capture (empty module list)"
    if any(
        marker in scope
        for marker in ("overlap-", "moe-act-", "moe_act", "shared-expert")
    ):
        return f"{scope} (configuration variant; graph scope unchanged)"
    return scope


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read an optional normalized result CSV."""
    if not path.is_file():
        return []
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def table(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[tuple[str, str]],
) -> str:
    """Render selected result fields or an explicit pending message."""
    if not rows:
        return '<p class="pending">No collected rows yet.</p>'
    headers = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = []
        for field, _ in columns:
            value = (
                scope_label(row.get(field, ""))
                if field == "scope"
                else row.get(field, "")
            )
            cells.append(f"<td>{escape(value)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _has_value(row: Mapping[str, str], fields: Sequence[str]) -> bool:
    return any(row.get(field, "") not in {"", None} for field in fields)


def _has_nonzero_exit_code(exit_code: str) -> bool:
    fields = exit_code.strip().split(":")
    if not exit_code.strip() or len(fields) not in {1, 2}:
        return False
    try:
        return any(int(field) != 0 for field in fields)
    except ValueError:
        return False


def _is_failure_row(row: Mapping[str, str]) -> bool:
    status = row.get("status", "")
    normalized_status = status.upper().replace("-", "_")
    status_parts = normalized_status.replace(":", " ").split()
    return (
        row.get("failure", "") != ""
        or _has_nonzero_exit_code(row.get("exit_code", ""))
        or any(
            marker in status.lower()
            for marker in ("fail", "error", "invalid", "cancel")
        )
        or any(part in TERMINAL_SLURM_FAILURE_STATUSES for part in status_parts)
    )


def render_html(
    rows: Sequence[Mapping[str, str]],
    *,
    te_version: str,
    te_source_commit: str,
    te_overlay_sha256: str,
    nemo_rl_sha: str = "__REQUIRED_NEMO_RL_SHA__",
    bridge_sha: str = "__REQUIRED_MEGATRON_BRIDGE_SHA__",
    mcore_sha: str = DEFAULT_MCORE_SHA,
    container_sha256: str = "__REQUIRED_CONTAINER_SHA256__",
    model_snapshot: str = DEFAULT_MODEL_SNAPSHOT,
    tokenizer_snapshot: str = DEFAULT_TOKENIZER_SNAPSHOT,
) -> str:
    """Build a self-contained report with correctness and experiment ledgers."""
    smoke_rows = [
        row for row in rows if row.get("status", "").lower().startswith("smoke:")
    ]
    performance_fields = (
        "e2e_step_time",
        "e2e_tokens_per_sec_per_gpu",
        "generation_time",
        "generation_tokens_per_sec_per_gpu",
        "policy_training_time",
        "policy_training_tokens_per_sec_per_gpu",
        "logprob_time",
        "logprob_tokens_per_sec_per_gpu",
    )
    performance_rows = [row for row in rows if _has_value(row, performance_fields)]
    steady_state_performance = aggregate_performance(
        steady_state_rows(performance_rows)
    )
    accuracy_rows = [row for row in rows if _has_value(row, CORRECTNESS_FIELDS)]
    failure_rows = [row for row in rows if _is_failure_row(row)]
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    correctness = """
<table>
<thead><tr><th>Task</th><th>Evidence</th><th>Verified status</th></tr></thead>
<tbody>
<tr><td>MCore Task 1</td><td>Slurm 2471224</td><td>66 passed</td></tr>
<tr><td>MCore Task 2</td><td>Slurm 2471343</td><td>29 + 3 passed</td></tr>
<tr><td>MCore Task 3</td><td>Slurm 2471570</td><td>38 + 3 passed</td></tr>
<tr><td>MCore Task 4</td><td>Slurm 2471681</td><td>43 + 23 passed</td></tr>
<tr><td>MCore Task 5</td><td>Slurm 2471820, 2471877, 2471888, 2471988</td><td>Final job 2471988 completed with exit 0 on 4xGB200: every rank reported 2 passed / 108 deselected. The packed Mamba parity passed in 74.33s; MoE 5→3→5 passed in 6.96s; total test time was 82.78s. Earlier job 2471820 passed packed Mamba and exposed an invalid topk test configuration before graph execution. Job 2471877 reached forward and exposed a stale post-reset telemetry assertion, not a production CUDA Graph failure. Focused MoE job 2471888 completed with each of four ranks reporting 1 passed. Its <code>routing_map.sum</code> token-count oracle is valid for this EP1/TP1 test only and is not generalized to EP&gt;1 post-communication counts. Fixes <code>1f1ca8fd7f36a5121b362ebed4a233a3a1ebee8b</code> and final MCore head <code>100047b517ea91526dc465448fcb3b37b2598388</code> are recorded.</td></tr>
<tr><td>NeMo-RL Task 6</td><td>Host verification</td><td>37 host tests + Pyrefly passed</td></tr>
<tr><td>NeMo-RL Task 7</td><td>Slurm 2472646</td><td>138 passed integration tests with exit 0 on the pinned nightly container.</td></tr>
<tr><td>NeMo-RL Task 9</td><td>Slurm 2475736, 2475881</td><td>Native TE <code>4a18653fc7274b10e33cd786b91be6261c523dc0</code> wheel build and GB200 validation both completed with exit 0. Wheel SHA256: <code>029fdbcb3fc0aa17b1a4f7398f56040204307d4bc839d318feda1677c98fff5e</code>. TE Python, PyTorch native extension, and core library resolved from the immutable wheel prefix; five static checks plus graph-safe MoE aux-loss CUDA Graph capture/replay reported 6 passed in 3.58s.</td></tr>
</tbody>
</table>
"""
    smoke = table(
        smoke_rows,
        (
            ("scope", "Launcher"),
            ("job_id", "Job"),
            ("status", "Status"),
            ("geometry_key", "Geometry key"),
            ("capture_count", "Captures"),
            ("replay_count", "Replays"),
            ("cache_hit_count", "Cache hits"),
            ("eviction_count", "Evictions"),
            ("fallback_count", "Fallbacks"),
        ),
    )
    performance = table(
        performance_rows,
        (
            ("scope", "Launcher"),
            ("job_id", "Job"),
            ("e2e_step_time", "E2E step time"),
            ("e2e_tokens_per_sec_per_gpu", "E2E tokens/s/GPU"),
            ("generation_time", "Generation time"),
            (
                "generation_tokens_per_sec_per_gpu",
                "Generation tokens/s/GPU",
            ),
            ("policy_training_time", "Policy time"),
            (
                "policy_training_tokens_per_sec_per_gpu",
                "Policy tokens/s/GPU",
            ),
            ("logprob_time", "Logprob time"),
            ("logprob_tokens_per_sec_per_gpu", "Logprob tokens/s/GPU"),
            ("peak_allocated_gib", "Peak allocated GiB"),
            ("peak_reserved_gib", "Peak reserved GiB"),
        ),
    )
    steady_state = table(
        steady_state_performance,
        (
            ("scope", "Launcher"),
            ("job_id", "Job"),
            ("sample_count", "Samples"),
            ("valid", "Valid"),
            ("invalid_reason", "Invalid reason"),
            ("e2e_step_time_median", "E2E time median"),
            ("e2e_step_time_p95", "E2E time p95"),
            ("e2e_tokens_per_sec_per_gpu_median", "E2E tokens/s/GPU median"),
            ("e2e_tokens_per_sec_per_gpu_p95", "E2E tokens/s/GPU p95"),
            (
                "e2e_tokens_per_sec_per_gpu_ratio_to_baseline",
                "E2E throughput / baseline",
            ),
            ("generation_time_median", "Generation time median"),
            ("generation_time_p95", "Generation time p95"),
            (
                "generation_tokens_per_sec_per_gpu_median",
                "Generation tokens/s/GPU median",
            ),
            ("generation_tokens_per_sec_per_gpu_p95", "Generation tokens/s/GPU p95"),
            (
                "generation_tokens_per_sec_per_gpu_ratio_to_baseline",
                "Generation throughput / baseline",
            ),
            ("policy_training_time_median", "Policy time median"),
            ("policy_training_time_p95", "Policy time p95"),
            (
                "policy_training_tokens_per_sec_per_gpu_median",
                "Policy tokens/s/GPU median",
            ),
            ("policy_training_tokens_per_sec_per_gpu_p95", "Policy tokens/s/GPU p95"),
            (
                "policy_training_tokens_per_sec_per_gpu_ratio_to_baseline",
                "Policy throughput / baseline",
            ),
            ("logprob_time_median", "Logprob time median"),
            ("logprob_time_p95", "Logprob time p95"),
            ("logprob_tokens_per_sec_per_gpu_median", "Logprob tokens/s/GPU median"),
            ("logprob_tokens_per_sec_per_gpu_p95", "Logprob tokens/s/GPU p95"),
            (
                "logprob_tokens_per_sec_per_gpu_ratio_to_baseline",
                "Logprob throughput / baseline",
            ),
        ),
    )
    correctness_deltas = table(
        steady_state_performance,
        (
            ("scope", "Launcher"),
            ("job_id", "Job"),
            *(
                (f"{field}_delta", f"{label} delta")
                for field, label in CORRECTNESS_COLUMNS
            ),
        ),
    )
    accuracy = table(
        accuracy_rows,
        (
            ("scope", "Launcher"),
            ("job_id", "Job"),
            *CORRECTNESS_COLUMNS,
        ),
    )
    failures = table(
        failure_rows,
        (
            ("scope", "Launcher"),
            ("job_id", "Job"),
            ("status", "Failure"),
            ("failure", "Failure detail"),
            ("exit_code", "Exit code"),
            ("elapsed", "Elapsed"),
            ("completed_steps", "Completed steps"),
            ("step", "Step"),
            ("fallback_count", "Fallbacks"),
        ),
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mamba/MoE Transformer Engine CUDA Graph study</title>
<style>
:root {{ color-scheme: dark; }}
body {{ background: #0e1420; color: #e6edf7; font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1440px; padding: 0 1rem; }}
h1, h2 {{ color: #8ed6ff; }}
section {{ background: #172130; border: 1px solid #2a3b52; border-radius: .6rem; margin: 1rem 0; padding: 1rem; }}
.legend {{ display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); }}
.legend div {{ background: #101a28; border-left: .25rem solid #70d6a7; padding: .75rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #2a3b52; padding: .55rem; text-align: left; vertical-align: top; }}
th {{ color: #a8ddff; }} code {{ color: #9de4ba; }} .table-wrap {{ overflow-x: auto; }} .pending {{ color: #aeb9c9; }}
</style>
</head>
<body>
<h1>Mamba/MoE Transformer Engine CUDA Graph study</h1>
<p>Generated {escape(generated_at)}. No experiment submission is performed by this renderer.</p>
<div class="legend">
<div><strong>No-CG baseline (CUDA graphs disabled)</strong><br>The baseline uses <code>cuda_graph_impl=none</code>.</div>
<div><strong>TE whole-layer capture (empty module list)</strong><br>An empty TE scope means whole-layer capture; it is never the no-CG baseline.</div>
<div><strong>MoE configuration variants</strong><br><code>moe_act</code> recompute and shared-expert overlap are configuration variants; graph scope is unchanged.</div>
</div>
<section id="correctness"><h2>Correctness</h2>{correctness}</section>
<section id="smoke"><h2>Smoke</h2>{smoke}</section>
<section id="performance"><h2>Performance</h2>{performance}</section>
<section id="steady-state-performance"><h2>Steady-state performance (steps 6–20)</h2>{steady_state}</section>
<section id="accuracy"><h2>Accuracy</h2>{accuracy}</section>
<section id="correctness-deltas"><h2>Correctness deltas (steps 6–20)</h2>{correctness_deltas}</section>
<section id="failures"><h2>Failures</h2>{failures}</section>
<section id="provenance"><h2>Provenance</h2>
<table><tbody>
<tr><th>NeMo-RL SHA</th><td><code>{escape(nemo_rl_sha)}</code></td></tr>
<tr><th>Megatron-Bridge SHA</th><td><code>{escape(bridge_sha)}</code></td></tr>
<tr><th>Megatron-LM SHA</th><td><code>{escape(mcore_sha)}</code></td></tr>
<tr><th>Nightly container SHA256</th><td><code>{escape(container_sha256)}</code></td></tr>
<tr><th>Transformer Engine version</th><td><code>{escape(te_version)}</code></td></tr>
<tr><th>Transformer Engine source commit</th><td><code>{escape(te_source_commit)}</code></td></tr>
<tr><th>Transformer Engine overlay SHA256</th><td><code>{escape(te_overlay_sha256)}</code></td></tr>
<tr><th>Model snapshot</th><td><code>{escape(model_snapshot)}</code></td></tr>
<tr><th>Tokenizer snapshot</th><td><code>{escape(tokenizer_snapshot)}</code></td></tr>
</tbody></table>
</section>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """Parse local CSV and provenance inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nemo-rl-sha", default="__REQUIRED_NEMO_RL_SHA__")
    parser.add_argument("--bridge-sha", default="__REQUIRED_MEGATRON_BRIDGE_SHA__")
    parser.add_argument(
        "--mcore-sha",
        default=DEFAULT_MCORE_SHA,
    )
    parser.add_argument("--container-sha256", default="__REQUIRED_CONTAINER_SHA256__")
    parser.add_argument("--te-version", required=True)
    parser.add_argument("--te-source-commit", required=True)
    parser.add_argument("--te-overlay-sha256", required=True)
    parser.add_argument("--model-snapshot", default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--tokenizer-snapshot", default=DEFAULT_TOKENIZER_SNAPSHOT)
    return parser.parse_args()


def main() -> None:
    """Render the available local result ledger."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_html(
            read_rows(args.input),
            te_version=args.te_version,
            te_source_commit=args.te_source_commit,
            te_overlay_sha256=args.te_overlay_sha256,
            nemo_rl_sha=args.nemo_rl_sha,
            bridge_sha=args.bridge_sha,
            mcore_sha=args.mcore_sha,
            container_sha256=args.container_sha256,
            model_snapshot=args.model_snapshot,
            tokenizer_snapshot=args.tokenizer_snapshot,
        )
    )


if __name__ == "__main__":
    main()
