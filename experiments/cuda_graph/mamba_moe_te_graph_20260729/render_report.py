#!/usr/bin/env python3
"""Render the Mamba/MoE TE CUDA Graph result ledger as static HTML."""

import argparse
import csv
import html
import json
import re
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
DEFAULT_CALL_COVERAGE = (
    REPO_ROOT
    / "experiments"
    / "cuda_graph"
    / "results"
    / "cg_call_coverage_jobs_2479812_2479813.json"
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


def cuda_graph_coverage_label(scope: str) -> str:
    """Describe requested model and NeMo-RL phase coverage for one scope."""
    normalized = scope.lower().replace("_", "-")
    if normalized in {"baseline-no-cg", "no-cg", "[no-cg]"}:
        return "No CUDA Graph; all phases eager"
    if normalized in {"whole-layer", "[]", "te-whole-layer"}:
        return (
            "Policy training: graphable whole-layer regions; "
            "logprob/generation: eager"
        )

    tokens = set(normalized.split("-"))
    modules = []
    if "attn" in tokens:
        modules.append("attention path (pre-attn LN + attention + BDA)")
    if "mlp" in tokens:
        modules.append("dense MLP path (pre-MLP LN + MLP + BDA)")
    if "mamba" in tokens:
        modules.append("full Mamba layer (norm + mixer + BDA)")

    eager_boundary = ""
    if "router" in tokens:
        if "preprocess" in tokens:
            modules.append(
                "MoE pre-MLP LN + shared expert (when configured/non-overlapped) + "
                "router/dispatcher preprocess"
            )
            eager_boundary = "; dispatch/experts/combine/BDA eager"
        else:
            modules.append(
                "MoE pre-MLP LN + shared expert (when configured/non-overlapped) + router"
            )
            eager_boundary = "; preprocess/dispatch/experts/combine/BDA eager"
    elif "moe" in tokens:
        modules.append(
            "full MoE path "
            "(pre-MLP LN + router/dispatch/experts/postprocess; drop-and-pad only)"
        )

    if not modules:
        return (
            "Configuration variant; coverage follows parent graph scope; "
            "logprob/generation: eager"
        )
    return (
        f"Policy training: {' + '.join(modules)}{eager_boundary}; "
        "logprob/generation: eager"
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read an optional normalized result CSV."""
    if not path.is_file():
        return []
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def read_call_coverage(path: Path) -> dict[str, dict[str, object]]:
    """Read an optional dynamic Nsight CUDA Graph call summary."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("CUDA Graph call coverage payload must be an object")
    for name, summary in payload.items():
        if not isinstance(summary, dict):
            raise ValueError(
                f"CUDA Graph call coverage variant {name} must be an object"
            )
    return payload


def _coverage_evidence(summary: Mapping[str, object]) -> tuple[str, str]:
    profiles = summary.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        return "", ""
    first_profile = profiles[0]
    if not isinstance(first_profile, dict):
        return "", ""
    path = str(first_profile.get("path", ""))
    job_match = re.search(r"/(\d+)-logs/", path)
    range_match = re.search(r"_(\d+):(\d+)_", path)
    job_id = job_match.group(1) if job_match else ""
    profiled_step = f"Step {range_match.group(1)}" if range_match else ""
    return job_id, profiled_step


def dynamic_call_coverage_table(
    coverage: Mapping[str, Mapping[str, object]],
) -> str:
    """Render dynamic CUDA Graph launch evidence from policy-worker profiles."""
    if not coverage:
        return '<p class="pending">No dynamic CUDA Graph call profiles yet.</p>'
    labels = {
        "baseline": "No-CG baseline",
        "positive": "PR5672 TE partial graph: moe_router",
    }
    rows = []
    for name in ("baseline", "positive", *sorted(set(coverage) - {"baseline", "positive"})):
        if name not in coverage:
            continue
        summary = coverage[name]
        job_id, profiled_step = _coverage_evidence(summary)
        profile_count = int(summary.get("profile_count", 0))
        profiles_with_launches = int(
            summary.get("profiles_with_cuda_graph_launches", 0)
        )
        worker_coverage = float(summary.get("profile_cuda_graph_coverage_pct", 0))
        graph_launches = int(summary.get("total_cuda_graph_launch_calls", 0))
        total_cuda_api_calls = int(summary.get("total_cuda_api_calls", 0))
        graph_api_share = float(
            summary.get("cuda_graph_launch_share_of_cuda_api_calls_pct", 0)
        )
        minimum = int(summary.get("cuda_graph_launch_calls_min", 0))
        median = float(summary.get("cuda_graph_launch_calls_median", 0))
        maximum = int(summary.get("cuda_graph_launch_calls_max", 0))
        cells = (
            labels.get(name, name),
            job_id,
            profiled_step,
            f"{profiles_with_launches} / {profile_count} ({worker_coverage}%)",
            f"{graph_launches:,}",
            f"{minimum} / {median} / {maximum}",
            f"{total_cuda_api_calls:,}",
            f"{graph_api_share}%",
        )
        rows.append(f"<tr>{''.join(f'<td>{escape(cell)}</td>' for cell in cells)}</tr>")
    headers = (
        "Variant",
        "Job",
        "Profiled policy step",
        "Workers with graph launches",
        "Total graph launches",
        "Graph launches/worker min / median / max",
        "All CUDA API calls",
        "Graph launch / CUDA API calls",
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{''.join(f'<th>{escape(header)}</th>' for header in headers)}"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<p>The final percentage uses every CUDA runtime/driver API call as its "
        "denominator; it is not graph-eligible model-module call coverage.</p>"
    )


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
            if field == "scope":
                value = scope_label(row.get(field, ""))
            elif field == "cuda_graph_coverage":
                value = cuda_graph_coverage_label(row.get("scope", ""))
            else:
                value = row.get(field, "")
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
    call_coverage: Mapping[str, Mapping[str, object]] | None = None,
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
            ("cuda_graph_coverage", "CUDA Graph coverage"),
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
            ("cuda_graph_coverage", "CUDA Graph coverage"),
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
            ("cuda_graph_coverage", "CUDA Graph coverage"),
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
            ("cuda_graph_coverage", "CUDA Graph coverage"),
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
            ("cuda_graph_coverage", "CUDA Graph coverage"),
            ("job_id", "Job"),
            *CORRECTNESS_COLUMNS,
        ),
    )
    failures = table(
        failure_rows,
        (
            ("scope", "Launcher"),
            ("cuda_graph_coverage", "CUDA Graph coverage"),
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
    dynamic_coverage = dynamic_call_coverage_table(call_coverage or {})

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
<div><strong>CUDA Graph coverage</strong><br>Coverage names the requested policy-training regions. Logprob and vLLM generation remain eager; job status and replay telemetry determine whether requested coverage was actually achieved.</div>
</div>
<section id="correctness"><h2>Correctness</h2>{correctness}</section>
<section id="dynamic-call-coverage"><h2>Dynamic CUDA Graph call evidence</h2>{dynamic_coverage}</section>
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
    parser.add_argument(
        "--call-coverage",
        type=Path,
        default=DEFAULT_CALL_COVERAGE,
    )
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
            call_coverage=read_call_coverage(args.call_coverage),
        )
    )


if __name__ == "__main__":
    main()
