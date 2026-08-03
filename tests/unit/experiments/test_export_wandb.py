from __future__ import annotations

import builtins
import importlib.util
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
MODULE_PATH = EXPERIMENT_DIR / "export_wandb.py"
CANONICAL_TAGS = (
    "timing/train/total_step_time",
    "timing/train/generation",
    "timing/train/policy_training",
    "timing/train/policy_and_reference_logprobs",
    "performance/tokens_per_sec_per_gpu",
    "performance/generation_tokens_per_sec_per_gpu",
    "performance/policy_training_tokens_per_sec_per_gpu",
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    "cuda_graph/capture_count",
    "cuda_graph/replay_count",
    "cuda_graph/cache_hits",
    "cuda_graph/cache_misses",
    "cuda_graph/cache_evictions",
    "cuda_graph/fallback_count",
    "cuda_graph/graph_calls",
    "cuda_graph/eligible_calls",
    "cuda_graph/logical_tokens",
    "cuda_graph/padded_tokens",
    "cuda_graph/capacity_tokens",
    "cuda_graph/coverage",
    "cuda_graph/capacity_utilization",
    "cuda_graph/padding_utilization",
    "train/reward",
    "train/gen_kl_error",
    "train/token_mult_prob_error",
    "train/policy_kl_error",
    "train/js_divergence_error",
    "train/sampling_importance_ratio",
    "train/num_masked_seqs_by_logprob_error",
    "train/loss",
    "train/grad_norm",
)
PROVENANCE = {
    "nemo_rl_commit": "1" * 40,
    "bridge_commit": "2" * 40,
    "mcore_commit": "3" * 40,
    "te_commit": "4" * 40,
    "te_version": "2.16.0.dev0",
    "container_sha256": "5" * 64,
}
IDENTITY = {
    "model": "qwen3_30ba3b",
    "dispatcher": "hybridep",
    "scope": "attn,moe_router",
    "mode": "nemorl",
    "cluster": "oci-hsg",
    "profile": "oci-hsg-gb200",
    "phase": "smoke",
    "steps": 5,
    "repeat": 0,
    "run_group": "qwen30-smoke",
    "job_id": "2475000",
    "router_replay": "off",
}


class FakeRun:
    def __init__(self, rows: Iterable[Mapping[str, object]]) -> None:
        self._rows = list(rows)
        self.scan_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def scan_history(
        self, *args: object, **kwargs: object
    ) -> Iterable[Mapping[str, object]]:
        self.scan_calls.append((args, kwargs))
        return iter(self._rows)

    @property
    def history(self) -> object:
        raise AssertionError("history must not be accessed")

    @property
    def summary(self) -> object:
        raise AssertionError("summary must not be accessed")


def _load_exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_wandb", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXPERIMENT_DIR))
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        sys.path.pop(0)
    return module


def _partial_rows(
    *,
    include_graph: bool = True,
    omitted_tag: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, tag in enumerate(CANONICAL_TAGS, start=1):
        if tag == omitted_tag or (tag.startswith("cuda_graph/") and not include_graph):
            continue
        source_tag = (
            "train/cuda_graph/cache_hit_count"
            if tag == "cuda_graph/cache_hits"
            else (
                "train/cuda_graph/cache_miss_count"
                if tag == "cuda_graph/cache_misses"
                else tag
            )
        )
        for step in range(1, 6):
            rows.append(
                {"optimizer_step": step, source_tag: float(index * 1000 + step)}
            )
    return rows


def _export(
    exporter: ModuleType, run: FakeRun, output: Path, **overrides: object
) -> None:
    arguments = {
        **IDENTITY,
        "optimizer_step_keys": ("optimizer_step",),
        "status": "passed",
        "provenance": PROVENANCE,
        "parity": None,
        "output": output,
        **overrides,
    }
    exporter.export_run(run, **arguments)


def test_module_import_requires_neither_wandb_nor_tensorboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_optional(name: str, *args: object, **kwargs: object) -> object:
        if name == "wandb" or name.startswith(("wandb.", "tensorboard.")):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_optional)
    _load_exporter()


def test_exporter_coalesces_unfiltered_partial_rows_and_later_values(
    tmp_path: Path,
) -> None:
    exporter = _load_exporter()
    rows = _partial_rows()
    rows.append({"optimizer_step": 3, "train/reward": 999.0})
    run = FakeRun(rows)
    output = tmp_path / "results.jsonl"

    _export(exporter, run, output)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert run.scan_calls == [((), {})]
    assert len(records) == 5
    assert records[2]["metrics"]["train/reward"] == 999.0
    assert records[0]["metrics"]["cuda_graph/cache_hits"] == 11001.0
    assert records[0]["parity"] == {}


def test_exporter_allows_baseline_without_graph_rows(tmp_path: Path) -> None:
    exporter = _load_exporter()
    run = FakeRun(_partial_rows(include_graph=False))
    output = tmp_path / "results.jsonl"

    _export(exporter, run, output, scope="baseline_no_cg")

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert {record["graph_telemetry_status"] for record in records} == {
        "not_applicable"
    }
    assert not any(tag.startswith("cuda_graph/") for tag in records[0]["metrics"])


@pytest.mark.parametrize(
    "missing_tag",
    ("cuda_graph/eligible_calls", "cuda_graph/cache_misses", "train/gen_kl_error"),
)
def test_exporter_rejects_incomplete_nonbaseline_and_preserves_output(
    tmp_path: Path,
    missing_tag: str,
) -> None:
    exporter = _load_exporter()
    output = tmp_path / "results.jsonl"
    output.write_text('{"previous":true}\n')

    with pytest.raises(ValueError, match=missing_tag):
        _export(exporter, FakeRun(_partial_rows(omitted_tag=missing_tag)), output)
    assert output.read_text() == '{"previous":true}\n'


@pytest.mark.parametrize("bad_value", (math.nan, math.inf, -math.inf))
def test_exporter_rejects_nonfinite_metrics_and_preserves_output(
    tmp_path: Path,
    bad_value: float,
) -> None:
    exporter = _load_exporter()
    rows = _partial_rows()
    rows.append({"optimizer_step": 2, "train/reward": bad_value})
    output = tmp_path / "results.jsonl"
    output.write_text('{"previous":true}\n')

    with pytest.raises(ValueError, match="non-finite metric"):
        _export(exporter, FakeRun(rows), output)
    assert output.read_text() == '{"previous":true}\n'


@pytest.mark.parametrize(
    ("row", "keys", "message"),
    (
        ({"optimizer_step": True, "train/reward": 1.0}, ("optimizer_step",), "integer"),
        ({"optimizer_step": 1.5, "train/reward": 1.0}, ("optimizer_step",), "integer"),
        (
            {"optimizer_step": math.inf, "train/reward": 1.0},
            ("optimizer_step",),
            "integer",
        ),
        ({"train/reward": 1.0}, ("optimizer_step",), "missing optimizer step"),
        (
            {"optimizer_step": 1, "other_step": 2, "train/reward": 1.0},
            ("optimizer_step", "other_step"),
            "conflicting optimizer step",
        ),
    ),
)
def test_exporter_rejects_bad_optimizer_step_identity_before_output(
    tmp_path: Path,
    row: dict[str, object],
    keys: tuple[str, ...],
    message: str,
) -> None:
    exporter = _load_exporter()
    output = tmp_path / "results.jsonl"
    output.write_text('{"previous":true}\n')

    with pytest.raises(ValueError, match=message):
        _export(
            exporter,
            FakeRun([row]),
            output,
            optimizer_step_keys=keys,
        )
    assert output.read_text() == '{"previous":true}\n'


def test_exporter_does_not_assume_wandb_internal_step(tmp_path: Path) -> None:
    exporter = _load_exporter()
    output = tmp_path / "results.jsonl"
    row = {"_step": 1, "train/reward": 1.0}

    with pytest.raises(ValueError, match="missing optimizer step"):
        _export(exporter, FakeRun([row]), output)


@pytest.mark.parametrize(
    ("overrides", "error"),
    (
        (
            {"provenance": {**PROVENANCE, "mcore_commit": "bad"}},
            "provenance.mcore_commit",
        ),
        ({"parity": {"router_topk_parity": 1}}, "router_topk_parity"),
    ),
)
def test_context_validation_precedes_wandb_scan(
    tmp_path: Path,
    overrides: dict[str, object],
    error: str,
) -> None:
    exporter = _load_exporter()
    run = FakeRun(_partial_rows())

    with pytest.raises((TypeError, ValueError), match=error):
        _export(exporter, run, tmp_path / "results.jsonl", **overrides)
    assert run.scan_calls == []


def test_wandb_output_matches_shared_canonical_export(tmp_path: Path) -> None:
    exporter = _load_exporter()
    rows = _partial_rows()
    wandb_output = tmp_path / "wandb.jsonl"
    shared_output = tmp_path / "shared.jsonl"
    _export(exporter, FakeRun(rows), wandb_output)

    identity, provenance, parity = exporter.resolve_export_context(
        **IDENTITY,
        provenance=PROVENANCE,
        parity=None,
    )
    exporter.export_scalar_values(
        exporter.coalesce_history(
            rows, optimizer_step_keys=("optimizer_step",), steps=5
        ),
        identity=identity,
        status="passed",
        provenance=provenance,
        parity=parity,
        output=shared_output,
    )

    assert wandb_output.read_bytes() == shared_output.read_bytes()
