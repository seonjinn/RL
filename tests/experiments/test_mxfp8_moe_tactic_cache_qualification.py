from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.mxfp8_moe_tactic_audit.qualify_cache import (
    BucketAudit,
    CacheProvenance,
    QualificationDecision,
    audit_bucket,
    build_candidate_cache,
    qualify_bucket,
    select_cache_path,
)
from experiments.mxfp8_moe_tactic_audit.schema import TacticMeasurement, TacticPair


MOE_OP = "flashinfer::trtllm_fp8_block_scale_moe"
MOE_RUNNER = "MoERunner"


def _cache_key(bucket: int) -> str:
    return str((MOE_OP, MOE_RUNNER, ((bucket, 2048),), ()))


def _bucket(
    weighted_gain: float,
    cv: float,
    worst_regression: float,
    *,
    all_correct: bool = True,
) -> BucketAudit:
    return BucketAudit(
        cache_key=_cache_key(16),
        stock=TacticPair(1, 2),
        candidate=TacticPair(3, 4),
        weighted_gain=weighted_gain,
        max_cv=cv,
        worst_high_weight_regression=worst_regression,
        all_correct=all_correct,
    )


@pytest.mark.parametrize(
    "gain,cv,worst_regression",
    [
        (0.024, 0.02, 0.009),
        (0.02, 0.03, 0.01),
    ],
)
def test_promotes_only_robust_two_percent_gain(
    gain: float, cv: float, worst_regression: float
) -> None:
    decision = qualify_bucket(_bucket(gain, cv, worst_regression))

    assert decision.promoted
    assert decision.selected == TacticPair(3, 4)
    assert decision.reason == "candidate passed qualification gates"


@pytest.mark.parametrize(
    "gain,cv,worst_regression,reason",
    [
        (0.019, 0.02, 0.0, "weighted gain below 2%"),
        (0.03, 0.031, 0.0, "coefficient of variation above 3%"),
        (0.03, 0.02, 0.011, "high-weight regression above 1%"),
    ],
)
def test_rejects_unqualified_candidate(
    gain: float,
    cv: float,
    worst_regression: float,
    reason: str,
) -> None:
    decision = qualify_bucket(_bucket(gain, cv, worst_regression))

    assert not decision.promoted
    assert decision.selected == TacticPair(1, 2)
    assert decision.reason == reason


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_qualification_metrics(value: float) -> None:
    decision = qualify_bucket(_bucket(value, 0.01, 0.0))

    assert not decision.promoted
    assert decision.reason == "qualification metrics are not finite"


def test_rejects_candidate_when_any_row_fails_correctness() -> None:
    decision = qualify_bucket(_bucket(0.10, 0.01, 0.0, all_correct=False))

    assert not decision.promoted
    assert decision.reason == "candidate failed correctness checks"


def _measurement(
    signature_key: str,
    tactic: TacticPair,
    median_us: float,
    *,
    cv: float = 0.01,
    finite: bool = True,
    deterministic: bool = True,
    cosine_similarity: float = 1.0,
    failure: str | None = None,
) -> TacticMeasurement:
    return TacticMeasurement(
        signature_key=signature_key,
        tactic=tactic,
        median_us=median_us,
        p95_us=median_us * 1.01,
        cv=cv,
        warmups=3,
        repetitions=10,
        finite=finite,
        deterministic=deterministic,
        max_abs_error=0.0,
        cosine_similarity=cosine_similarity,
        failure=failure,
    )


def test_audit_uses_stock_denominator_and_selected_profile_weights() -> None:
    stock = TacticPair(1, 2)
    candidate = TacticPair(3, 4)
    measurements = (
        _measurement("heavy", stock, 100.0),
        _measurement("heavy", candidate, 95.0, cv=0.02),
        _measurement("light", stock, 200.0),
        _measurement("light", candidate, 202.0, cv=0.03),
    )

    audit = audit_bucket(
        cache_key=_cache_key(16),
        stock=stock,
        candidate=candidate,
        profile_weights={"heavy": 0.6, "light": 0.4},
        measurements=measurements,
    )

    assert audit.weighted_gain == pytest.approx(0.05)
    assert audit.max_cv == 0.03
    assert audit.worst_high_weight_regression == pytest.approx(0.01)
    assert audit.all_correct
    assert qualify_bucket(audit).promoted


def test_audit_ignores_sub_five_percent_profiles_for_worst_regression() -> None:
    stock = TacticPair(1, 2)
    candidate = TacticPair(3, 4)
    measurements = (
        _measurement("heavy", stock, 100.0),
        _measurement("heavy", candidate, 95.0),
        _measurement("tail", stock, 100.0),
        _measurement("tail", candidate, 150.0),
    )

    audit = audit_bucket(
        cache_key=_cache_key(16),
        stock=stock,
        candidate=candidate,
        profile_weights={"heavy": 0.96, "tail": 0.04},
        measurements=measurements,
    )

    assert audit.worst_high_weight_regression == pytest.approx(-0.05)
    assert qualify_bucket(audit).promoted


@pytest.mark.parametrize(
    "candidate_overrides",
    [
        {"finite": False},
        {"deterministic": False},
        {"cosine_similarity": 0.998},
        {"failure": "kernel failed"},
    ],
)
def test_audit_rejects_failed_nondeterministic_or_micro_incorrect_rows(
    candidate_overrides: dict[str, object],
) -> None:
    stock = TacticPair(1, 2)
    candidate = TacticPair(3, 4)
    candidate_row = _measurement(
        "profile",
        candidate,
        90.0,
        **candidate_overrides,  # type: ignore[arg-type]
    )

    audit = audit_bucket(
        cache_key=_cache_key(16),
        stock=stock,
        candidate=candidate,
        profile_weights={"profile": 1.0},
        measurements=(
            _measurement("profile", stock, 100.0),
            candidate_row,
        ),
    )

    assert not audit.all_correct
    assert not qualify_bucket(audit).promoted


def _install_fake_flashinfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    package_root = tmp_path / "fake_site"
    package = package_root / "flashinfer"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="ascii")
    (package / "autotuner.py").write_text(
        """\
from abc import ABC, abstractmethod
import json
import os


def _tactic(value):
    return tuple(value) if isinstance(value, list) else value


class AutoTuner:
    _instance = None

    def __init__(self):
        self._file_configs = {}
        self.profiling_cache = {}

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_configs(self, path):
        _record("load_configs", str(path))
        if os.environ.get("FAKE_FLASHINFER_FAIL_STAGE") == "load":
            raise RuntimeError("injected load failure")
        with open(path, encoding="ascii") as cache_file:
            payload = json.load(cache_file)
        metadata = payload.pop("_metadata", None)
        expected = os.environ.get("FAKE_FLASHINFER_RUNTIME", "runtime-a")
        if metadata is not None and metadata.get("runtime_marker") != expected:
            return False
        self._file_configs.update(
            {key: (value[0], _tactic(value[1])) for key, value in payload.items()}
        )
        return True

    def save_configs(self, path):
        _record("save_configs", str(path))
        if os.environ.get("FAKE_FLASHINFER_FAIL_STAGE") == "save":
            raise RuntimeError("injected save failure")
        metadata = {"runtime_marker": os.environ.get("FAKE_FLASHINFER_RUNTIME", "runtime-a")}
        try:
            with open(path, encoding="ascii") as cache_file:
                metadata = json.load(cache_file).get("_metadata", metadata)
        except FileNotFoundError:
            pass
        payload = {"_metadata": metadata}
        for key in sorted(self._file_configs):
            runner, tactic = self._file_configs[key]
            payload[key] = [runner, list(tactic) if isinstance(tactic, tuple) else tactic]
        with open(path, "w", encoding="ascii") as cache_file:
            json.dump(payload, cache_file, indent=2)

    def search_cache(self, custom_op, runners, input_shapes, tuning_config, inputs=None):
        extras = runners[0].get_cache_key_extras(inputs) if inputs is not None else ()
        key = str(
            (
                custom_op,
                runners[0].__class__.__name__,
                tuple(tuple(shape) for shape in input_shapes),
                extras,
            )
        )
        _record("search_cache", key)
        if key in self._file_configs:
            runner_name, tactic = self._file_configs[key]
            runner_id = next(
                (
                    index
                    for index, runner in enumerate(runners)
                    if runner.__class__.__name__ == runner_name
                ),
                0,
            )
            return True, runner_id, tactic, None
        return False, 0, -1, None

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        input_shapes = tuple(value.size() for value in inputs)
        key = str(
            (
                custom_op,
                runners[0].__class__.__name__,
                tuple(tuple(shape) for shape in input_shapes),
                runners[0].get_cache_key_extras(inputs),
            )
        )
        _record("choose_one", key)
        _, runner_id, tactic, _ = self.search_cache(
            custom_op, runners, input_shapes, tuning_config, inputs=inputs
        )
        return runners[runner_id], tactic


class TunableRunner(ABC):
    @abstractmethod
    def get_valid_tactics(self, inputs, profile):
        return [-1]

    def get_cache_key_extras(self, inputs):
        return ()

    @abstractmethod
    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        raise NotImplementedError


class TuningConfig:
    pass


def _record(operation, key):
    log_path = os.environ.get("FAKE_FLASHINFER_OPERATION_LOG")
    if log_path:
        with open(log_path, "a", encoding="ascii") as log_file:
            log_file.write(
                json.dumps({"pid": os.getpid(), "operation": operation, "key": key})
                + "\\n"
            )
""",
        encoding="ascii",
    )
    monkeypatch.syspath_prepend(str(package_root))
    current_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(package_root)
    if current_pythonpath:
        pythonpath = os.pathsep.join((pythonpath, current_pythonpath))
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("FAKE_FLASHINFER_RUNTIME", "runtime-a")
    for module_name in tuple(sys.modules):
        if module_name == "flashinfer" or module_name.startswith("flashinfer."):
            del sys.modules[module_name]
    return package_root


def _write_stock_cache(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "_metadata": {"runtime_marker": "runtime-a", "stock_only": "preserve-me"},
        _cache_key(16): [MOE_RUNNER, [1, 2]],
        _cache_key(32): [MOE_RUNNER, [5, 6]],
        str((f"{MOE_OP}_similar", MOE_RUNNER, ((16, 2048),), ())): [
            MOE_RUNNER,
            [7, 8],
        ],
        str(("other::op", "OtherRunner", ((16, 2048),), ())): ["OtherRunner", 9],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")
    return payload


def _provenance(tmp_path: Path) -> CacheProvenance:
    trace_a = tmp_path / "rank-0.jsonl"
    trace_b = tmp_path / "rank-1.jsonl"
    profiles = tmp_path / "selected_profiles.json"
    shmoo = tmp_path / "shmoo.jsonl"
    trace_a.write_text("trace-a\n", encoding="ascii")
    trace_b.write_text("trace-b\n", encoding="ascii")
    profiles.write_text("profiles\n", encoding="ascii")
    shmoo.write_text("shmoo\n", encoding="ascii")
    return CacheProvenance(
        trace_paths=(trace_a, trace_b),
        selected_profiles=profiles,
        shmoo_results=shmoo,
        model_revision="qwen3-30b-a3b-revision",
        container="nvcr.io/nvidia/nemo-rl@sha256:container",
        vllm_commit="a76062edee3a3ac23d47a93c7ce466f06a19111f",
        flashinfer_version="0.6.13",
        cuda_version="13.0",
        gpu_name="NVIDIA GB200",
        tp_size=1,
        ep_size=1,
        dp_size=16,
        cuda_graph_mode="enabled",
    )


def test_build_candidate_uses_autotuner_and_preserves_nonpromoted_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_flashinfer(tmp_path, monkeypatch)
    operation_log = tmp_path / "operations.jsonl"
    monkeypatch.setenv("FAKE_FLASHINFER_OPERATION_LOG", str(operation_log))
    stock_path = tmp_path / "stock.json"
    stock_payload = _write_stock_cache(stock_path)
    candidate_dir = tmp_path / "candidate"
    decisions = (
        QualificationDecision(
            cache_key=_cache_key(16),
            selected=TacticPair(3, 4),
            promoted=True,
            reason="candidate passed qualification gates",
        ),
        QualificationDecision(
            cache_key=_cache_key(32),
            selected=TacticPair(5, 6),
            promoted=False,
            reason="weighted gain below 2%",
        ),
    )

    manifest = build_candidate_cache(
        stock_path,
        decisions,
        candidate_dir,
        provenance=_provenance(tmp_path),
    )

    candidate_path = candidate_dir / "autotune_configs.json"
    candidate_payload = json.loads(candidate_path.read_text(encoding="ascii"))
    assert candidate_payload[_cache_key(16)] == [MOE_RUNNER, [3, 4]]
    for key, value in stock_payload.items():
        if key != _cache_key(16):
            assert candidate_payload[key] == value
    assert manifest.promoted_entries == 1
    assert manifest.retained_entries == 3
    assert manifest.stock_sha256 != manifest.candidate_sha256

    decisions_payload = json.loads(
        (candidate_dir / "qualification_decisions.json").read_text(encoding="ascii")
    )
    assert decisions_payload["cache_manifest_sha256"] == hashlib.sha256(
        (candidate_dir / "cache_manifest.json").read_bytes()
    ).hexdigest()
    assert decisions_payload["decisions"][0]["selected"] == {"gemm1": 3, "gemm2": 4}
    assert decisions_payload["decisions"][0]["stock"] == {"gemm1": 1, "gemm2": 2}

    manifest_payload = json.loads(
        (candidate_dir / "cache_manifest.json").read_text(encoding="ascii")
    )
    assert manifest_payload["schema_version"] == 1
    assert manifest_payload["stock_sha256"] == manifest.stock_sha256
    assert manifest_payload["candidate_sha256"] == manifest.candidate_sha256
    expected_runtime = {
        "cuda_graph_mode": "enabled",
        "cuda_version": "13.0",
        "dp_size": "16",
        "flashinfer_version": "0.6.13",
        "gpu_name": "NVIDIA GB200",
        "model_revision": "qwen3-30b-a3b-revision",
        "container": "nvcr.io/nvidia/nemo-rl@sha256:container",
        "ep_size": "1",
        "tp_size": "1",
        "vllm_commit": "a76062edee3a3ac23d47a93c7ce466f06a19111f",
    }
    source_fingerprints = manifest_payload["source_fingerprints"]
    assert {
        key: source_fingerprints[key] for key in expected_runtime
    } == expected_runtime
    assert all(
        len(manifest_payload["source_fingerprints"][field]) == 64
        for field in (
            "trace_set_sha256",
            "selected_profiles_sha256",
            "shmoo_results_sha256",
        )
    )

    operations = [json.loads(line) for line in operation_log.read_text().splitlines()]
    assert {event["pid"] for event in operations}.isdisjoint({os.getpid()})
    for cache_key in (_cache_key(16), _cache_key(32), _cache_key(17)):
        assert any(
            event["operation"] == "search_cache" and event["key"] == cache_key
            for event in operations
        )
        assert any(
            event["operation"] == "choose_one" and event["key"] == cache_key
            for event in operations
        )


def test_build_candidate_supports_all_exact_moe_keys_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_flashinfer(tmp_path, monkeypatch)
    operation_log = tmp_path / "operations.jsonl"
    monkeypatch.setenv("FAKE_FLASHINFER_OPERATION_LOG", str(operation_log))
    stock_path = tmp_path / "stock.json"
    stock_path.write_text(
        json.dumps(
            {
                "_metadata": {"runtime_marker": "runtime-a"},
                _cache_key(16): [MOE_RUNNER, [1, 2]],
                _cache_key(32): [MOE_RUNNER, [5, 6]],
            },
            indent=2,
        ),
        encoding="ascii",
    )
    decisions = (
        QualificationDecision(
            cache_key=_cache_key(16),
            selected=TacticPair(3, 4),
            promoted=True,
            reason="candidate passed qualification gates",
        ),
        QualificationDecision(
            cache_key=_cache_key(32),
            selected=TacticPair(7, 8),
            promoted=True,
            reason="candidate passed qualification gates",
        ),
    )

    manifest = build_candidate_cache(
        stock_path,
        decisions,
        tmp_path / "candidate",
        provenance=_provenance(tmp_path),
    )

    assert manifest.promoted_entries == 2
    assert manifest.retained_entries == 0
    operations = [json.loads(line) for line in operation_log.read_text().splitlines()]
    for cache_key in (_cache_key(16), _cache_key(32), _cache_key(17)):
        assert any(
            event["operation"] == "search_cache" and event["key"] == cache_key
            for event in operations
        )
        assert any(
            event["operation"] == "choose_one" and event["key"] == cache_key
            for event in operations
        )


def test_cache_subprocess_timeout_removes_partial_candidate_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stock_path = tmp_path / "stock.json"
    _write_stock_cache(stock_path)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    candidate_path = candidate_dir / "autotune_configs.json"
    manifest_path = candidate_dir / "cache_manifest.json"
    candidate_path.write_text("stale candidate\n", encoding="ascii")
    manifest_path.write_text("stale manifest\n", encoding="ascii")

    def time_out(*args: object, **kwargs: object) -> None:
        request = json.loads(str(kwargs["input"]))
        Path(request["candidate_path"]).write_text("partial\n", encoding="ascii")
        manifest_path.write_text("partial manifest\n", encoding="ascii")
        assert kwargs["timeout"] == 2.5
        raise subprocess.TimeoutExpired(cmd="cache-subprocess", timeout=2.5)

    monkeypatch.setattr(subprocess, "run", time_out)

    with pytest.raises(
        RuntimeError,
        match="cache subprocess timed out after 2.5 seconds",
    ):
        build_candidate_cache(
            stock_path,
            (),
            candidate_dir,
            provenance=_provenance(tmp_path),
            subprocess_timeout_seconds=2.5,
        )

    assert not candidate_path.exists()
    assert not manifest_path.exists()


def test_parent_tuner_state_is_unchanged_on_child_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_flashinfer(tmp_path, monkeypatch)
    autotuner = importlib.import_module("flashinfer.autotuner")
    tuner = autotuner.AutoTuner.get()
    tuner._file_configs["parent-file-config"] = ("ParentRunner", 17)
    tuner.profiling_cache["parent-profile-config"] = (0, 23, None)
    expected_file_configs = tuner._file_configs.copy()
    expected_profiling_cache = tuner.profiling_cache.copy()
    stock_path = tmp_path / "stock.json"
    _write_stock_cache(stock_path)
    build_candidate_cache(
        stock_path,
        (
            QualificationDecision(
                cache_key=_cache_key(16),
                selected=TacticPair(3, 4),
                promoted=True,
                reason="candidate passed qualification gates",
            ),
        ),
        tmp_path / "candidate",
        provenance=_provenance(tmp_path),
    )

    assert tuner._file_configs == expected_file_configs
    assert tuner.profiling_cache == expected_profiling_cache

    monkeypatch.setenv("FAKE_FLASHINFER_FAIL_STAGE", "save")
    with pytest.raises(RuntimeError, match="cache subprocess failed"):
        build_candidate_cache(
            stock_path,
            (
                QualificationDecision(
                    cache_key=_cache_key(16),
                    selected=TacticPair(3, 4),
                    promoted=True,
                    reason="candidate passed qualification gates",
                ),
            ),
            tmp_path / "failed-candidate",
            provenance=_provenance(tmp_path),
        )

    assert tuner._file_configs == expected_file_configs
    assert tuner.profiling_cache == expected_profiling_cache


def test_fingerprint_mismatch_selects_stock_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_flashinfer(tmp_path, monkeypatch)
    stock_path = tmp_path / "stock.json"
    _write_stock_cache(stock_path)
    candidate_dir = tmp_path / "candidate"
    provenance = _provenance(tmp_path)
    build_candidate_cache(
        stock_path,
        (),
        candidate_dir,
        provenance=provenance,
    )
    runtime = dict(provenance.runtime_fingerprints())

    selected = select_cache_path(
        stock_path=stock_path,
        candidate_path=candidate_dir / "autotune_configs.json",
        manifest_path=candidate_dir / "cache_manifest.json",
        runtime_fingerprints=runtime,
    )
    mismatch = dict(runtime, gpu_name="NVIDIA H100")
    rejected = select_cache_path(
        stock_path=stock_path,
        candidate_path=candidate_dir / "autotune_configs.json",
        manifest_path=candidate_dir / "cache_manifest.json",
        runtime_fingerprints=mismatch,
    )

    assert selected == candidate_dir / "autotune_configs.json"
    assert rejected == stock_path


def test_build_rejects_nonexact_or_absent_promoted_moe_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_flashinfer(tmp_path, monkeypatch)
    stock_path = tmp_path / "stock.json"
    _write_stock_cache(stock_path)
    invalid = QualificationDecision(
        cache_key=str((f"{MOE_OP}_similar", MOE_RUNNER, ((16, 2048),), ())),
        selected=TacticPair(3, 4),
        promoted=True,
        reason="candidate passed qualification gates",
    )
    absent = QualificationDecision(
        cache_key=_cache_key(64),
        selected=TacticPair(3, 4),
        promoted=True,
        reason="candidate passed qualification gates",
    )

    with pytest.raises(ValueError, match="exact FlashInfer MoE file key"):
        build_candidate_cache(
            stock_path,
            (invalid,),
            tmp_path / "invalid-candidate",
            provenance=_provenance(tmp_path),
        )
    with pytest.raises(ValueError, match="not present in stock cache"):
        build_candidate_cache(
            stock_path,
            (absent,),
            tmp_path / "absent-candidate",
            provenance=_provenance(tmp_path),
        )
