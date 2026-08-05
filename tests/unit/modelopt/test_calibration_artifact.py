from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from examples.modelopt import export_nvfp4_calibration
from examples.modelopt.export_nvfp4_calibration import collect_nvfp4_input_amax
from nemo_rl.modelopt import calibration_artifact
from nemo_rl.modelopt.calibration_artifact import (
    load_nvfp4_calibration,
    normalize_quant_cfg_identity,
    save_nvfp4_calibration,
)
from nemo_rl.modelopt.models.generation.nvfp4_refit import NVFP4Calibration


@pytest.fixture
def metadata() -> dict[str, str | int]:
    return {
        "model_id": "nvidia/Nemotron-3-Super-120B-A12B",
        "model_revision": "0123456789abcdef",
        "quant_cfg": "examples/modelopt/quant_configs/nvfp4_experts.yaml",
        "dataset": "cnn_dailymail",
        "sample_count": 16,
        "sequence_length": 1024,
        "seed": 1234,
    }


def _json_metadata(metadata: dict[str, str | int]) -> dict[str, str]:
    return {key: json.dumps(value) for key, value in metadata.items()}


def _load(
    path: Path,
    metadata: dict[str, str | int],
    *,
    expected_projection_names: set[str] | None = None,
) -> NVFP4Calibration:
    return load_nvfp4_calibration(
        path,
        model_id=str(metadata["model_id"]),
        model_revision=str(metadata["model_revision"]),
        quant_cfg=str(metadata["quant_cfg"]),
        expected_projection_names=expected_projection_names,
    )


class _FakeQuantizer(torch.nn.Module):
    def __init__(self, value: float, *, enabled: bool = True) -> None:
        super().__init__()
        self.is_enabled = enabled
        self._amax = torch.tensor(value)


class _FakeFusedExperts(torch.nn.Module):
    def __init__(self, *, num_experts: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_dim = 3
        self.gate_up_proj = torch.nn.Parameter(torch.ones((num_experts, 6, 4)))
        self.down_proj = torch.nn.Parameter(torch.ones((num_experts, 4, 3)))
        self.act_fn = torch.nn.SiLU()
        self.gate_up_proj_input_quantizer = _FakeQuantizer(12.0)
        self.down_proj_input_quantizer = _FakeQuantizer(24.0)


def _model_with_fused_experts(experts: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].mlp = torch.nn.Module()
    model.layers[0].mlp.experts = experts
    return model


_MISSING_COMMIT = object()


def _install_fake_exporter_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_commit: object,
    tokenizer_commit: object,
) -> dict[str, str | None]:
    seen: dict[str, str | None] = {}

    class FakeProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2, 2)))
            self.input_quantizer = _FakeQuantizer(12.0)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace()
            if model_commit is not _MISSING_COMMIT:
                self.config._commit_hash = model_commit
            self.up_proj = FakeProjection()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(
            model_id: str,
            *,
            revision: str,
            dtype: torch.dtype,
            device_map: str,
        ) -> torch.nn.Module:
            del model_id, dtype, device_map
            seen["model_revision"] = revision
            return FakeModel()

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    algorithms_utils = types.ModuleType("nemo_rl.algorithms.utils")
    algorithms_utils.set_seed = lambda seed: None
    worker_utils = types.ModuleType("nemo_rl.modelopt.models.policy.workers.utils")

    def fake_get_tokenizer(
        model_id: str,
        *,
        max_seq_len: int,
        revision: str | None = None,
    ) -> object:
        del model_id, max_seq_len
        seen["tokenizer_revision"] = revision
        init_kwargs = {}
        if tokenizer_commit is not _MISSING_COMMIT:
            init_kwargs["_commit_hash"] = tokenizer_commit
        return types.SimpleNamespace(init_kwargs=init_kwargs)

    worker_utils.get_tokenizer = fake_get_tokenizer
    worker_utils.quantize_model = lambda **kwargs: None
    modelopt_utils = types.ModuleType("nemo_rl.modelopt.utils")
    modelopt_utils.resolve_nvfp4_real_quant_mode = lambda quant_cfg: "w4a4"
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "nemo_rl.algorithms.utils", algorithms_utils)
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.modelopt.models.policy.workers.utils",
        worker_utils,
    )
    monkeypatch.setitem(sys.modules, "nemo_rl.modelopt.utils", modelopt_utils)
    return seen


def _run_fake_exporter(tmp_path: Path, *, revision: str = "release-tag") -> Path:
    quant_cfg_path = tmp_path / "nvfp4.yaml"
    quant_cfg_path.write_text("quant_cfg: nvfp4\n")
    output_path = tmp_path / "calibration.safetensors"
    export_nvfp4_calibration.main(
        [
            "--model",
            "org/model",
            "--model-revision",
            revision,
            "--quant-cfg",
            str(quant_cfg_path),
            "--dataset",
            "cnn_dailymail",
            "--sample-count",
            "1",
            "--sequence-length",
            "16",
            "--seed",
            "1234",
            "--output",
            str(output_path),
        ]
    )
    return output_path


def test_normalize_quant_cfg_identity_resolves_paths_and_preserves_symbolic_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("configs/nvfp4.yaml")
    config_path.parent.mkdir()
    config_path.write_text("quant_cfg: nvfp4\n")

    assert normalize_quant_cfg_identity(str(config_path)) == str(config_path.resolve())
    assert normalize_quant_cfg_identity("NVFP4_DEFAULT_CFG") == "NVFP4_DEFAULT_CFG"


def test_normalize_quant_cfg_identity_resolves_project_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "repo"
    config_path = project_root / "examples/modelopt/quant_configs/nvfp4.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("quant_cfg: nvfp4\n")
    worker_dir = tmp_path / "ray-worker"
    worker_dir.mkdir()
    monkeypatch.chdir(worker_dir)
    monkeypatch.setattr(calibration_artifact, "_PROJECT_ROOT", project_root)

    assert normalize_quant_cfg_identity(
        "examples/modelopt/quant_configs/nvfp4.yaml"
    ) == str(config_path.resolve())


def test_collect_input_amax_uses_enabled_hf_projection_names() -> None:
    class FakeQuantizer:
        def __init__(self, value: float, *, enabled: bool) -> None:
            self.is_enabled = enabled
            self._amax = torch.tensor(value, requires_grad=False)

    class FakeProjection(torch.nn.Module):
        def __init__(self, value: float, *, enabled: bool = True) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2, 2)))
            self.input_quantizer = FakeQuantizer(value, enabled=enabled)

    model = torch.nn.Module()
    model.add_module("gate_proj", FakeProjection(12.0))
    model.add_module("up_proj", FakeProjection(24.0))
    model.add_module("down_proj", FakeProjection(48.0, enabled=False))

    input_amax = collect_nvfp4_input_amax(model)

    assert list(input_amax) == ["gate_proj.weight", "up_proj.weight"]
    assert torch.equal(input_amax["gate_proj.weight"], torch.tensor(12.0))
    assert torch.equal(input_amax["up_proj.weight"], torch.tensor(24.0))
    assert all(not value.requires_grad for value in input_amax.values())


def test_collect_input_amax_expands_fused_experts_to_exact_hf_names() -> None:
    input_amax = collect_nvfp4_input_amax(
        _model_with_fused_experts(_FakeFusedExperts())
    )

    assert list(input_amax) == [
        "layers.0.mlp.experts.0.gate_proj.weight",
        "layers.0.mlp.experts.0.up_proj.weight",
        "layers.0.mlp.experts.0.down_proj.weight",
        "layers.0.mlp.experts.1.gate_proj.weight",
        "layers.0.mlp.experts.1.up_proj.weight",
        "layers.0.mlp.experts.1.down_proj.weight",
    ]
    for expert_idx in range(2):
        prefix = f"layers.0.mlp.experts.{expert_idx}"
        assert torch.equal(input_amax[f"{prefix}.gate_proj.weight"], torch.tensor(12.0))
        assert torch.equal(input_amax[f"{prefix}.up_proj.weight"], torch.tensor(12.0))
        assert torch.equal(input_amax[f"{prefix}.down_proj.weight"], torch.tensor(24.0))


def test_collect_input_amax_rejects_inconsistent_fused_expert_shapes() -> None:
    experts = _FakeFusedExperts()
    experts.down_proj = torch.nn.Parameter(torch.ones((3, 4, 3)))

    with pytest.raises(RuntimeError, match="num_experts.*shape"):
        collect_nvfp4_input_amax(_model_with_fused_experts(experts))


def test_collect_input_amax_rejects_missing_fused_expert_quantizer() -> None:
    experts = _FakeFusedExperts()
    del experts.down_proj_input_quantizer

    with pytest.raises(RuntimeError, match="down_proj_input_quantizer"):
        collect_nvfp4_input_amax(_model_with_fused_experts(experts))


@pytest.mark.parametrize(
    ("enabled", "amax"),
    [
        (False, torch.tensor(24.0)),
        (True, torch.tensor([24.0])),
        (True, torch.tensor(float("nan"))),
    ],
    ids=("disabled", "non-scalar", "nonfinite"),
)
def test_collect_input_amax_rejects_invalid_fused_expert_quantizer(
    enabled: bool,
    amax: torch.Tensor,
) -> None:
    experts = _FakeFusedExperts()
    experts.down_proj_input_quantizer.is_enabled = enabled
    experts.down_proj_input_quantizer._amax = amax

    with pytest.raises(RuntimeError, match="down_proj_input_quantizer"):
        collect_nvfp4_input_amax(_model_with_fused_experts(experts))


def test_exporter_resolves_model_commit_and_pins_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_revision = "0123456789abcdef0123456789abcdef01234567"
    seen = _install_fake_exporter_dependencies(
        monkeypatch,
        model_commit=resolved_revision,
        tokenizer_commit=resolved_revision,
    )
    monkeypatch.chdir(tmp_path)
    output_path = _run_fake_exporter(tmp_path)

    assert seen == {
        "model_revision": "release-tag",
        "tokenizer_revision": resolved_revision,
    }
    with safe_open(output_path, framework="pt", device="cpu") as artifact:
        assert artifact.metadata()["model_revision"] == json.dumps(resolved_revision)
        assert artifact.metadata()["quant_cfg"] == json.dumps(
            str((tmp_path / "nvfp4.yaml").resolve())
        )


@pytest.mark.parametrize(
    "model_commit",
    [_MISSING_COMMIT, None, "release-tag", "0123456789abcdef"],
    ids=("missing", "none", "mutable", "short"),
)
def test_exporter_rejects_missing_or_nonimmutable_model_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_commit: object,
) -> None:
    _install_fake_exporter_dependencies(
        monkeypatch,
        model_commit=model_commit,
        tokenizer_commit=_MISSING_COMMIT,
    )

    with pytest.raises(RuntimeError, match="resolved immutable commit SHA"):
        _run_fake_exporter(tmp_path)


def test_exporter_rejects_mismatched_tokenizer_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_exporter_dependencies(
        monkeypatch,
        model_commit="0123456789abcdef0123456789abcdef01234567",
        tokenizer_commit="89abcdef0123456789abcdef0123456789abcdef",
    )

    with pytest.raises(RuntimeError, match="tokenizer.*commit.*does not match"):
        _run_fake_exporter(tmp_path)


def test_exporter_accepts_tokenizer_without_resolved_commit_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_revision = "0123456789abcdef0123456789abcdef01234567"
    _install_fake_exporter_dependencies(
        monkeypatch,
        model_commit=resolved_revision,
        tokenizer_commit=_MISSING_COMMIT,
    )

    output_path = _run_fake_exporter(tmp_path)

    with safe_open(output_path, framework="pt", device="cpu") as artifact:
        assert artifact.metadata()["model_revision"] == json.dumps(resolved_revision)


def test_round_trip_preserves_exact_hf_projection_names_and_metadata(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    path = tmp_path / "calibration.safetensors"
    input_amax = {
        "model.layers.0.mlp.gate_proj.weight": torch.tensor(12.0),
        "model.layers.0.mlp.up_proj.input_quantizer._amax": torch.tensor(24.0),
    }

    save_nvfp4_calibration(path, input_amax, **metadata)

    with safe_open(path, framework="pt", device="cpu") as artifact:
        assert artifact.keys() == [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ]
        artifact_metadata = artifact.metadata()
        assert {key: artifact_metadata[key] for key in metadata} == _json_metadata(
            metadata
        )
        quant_cfg_path = Path(str(metadata["quant_cfg"]))
        assert artifact_metadata["quant_cfg_sha256"] == json.dumps(
            hashlib.sha256(quant_cfg_path.read_bytes()).hexdigest()
        )

    calibration = _load(path, metadata)
    assert list(calibration.input_amax) == [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    ]
    assert torch.equal(
        calibration.input_amax["model.layers.0.mlp.gate_proj.weight"],
        torch.tensor(12.0),
    )
    assert torch.equal(
        calibration.input_amax["model.layers.0.mlp.up_proj.weight"],
        torch.tensor(24.0),
    )


def test_load_rejects_missing_required_metadata(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    path = tmp_path / "missing-metadata.safetensors"
    incomplete = _json_metadata(metadata)
    del incomplete["seed"]
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
        path,
        metadata=incomplete,
    )

    with pytest.raises(ValueError, match="missing required metadata.*seed"):
        _load(path, metadata)


def test_load_rejects_duplicate_normalized_projection_names(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    path = tmp_path / "duplicate-names.safetensors"
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor(1.0),
            "model.layers.0.mlp.up_proj.input_quantizer._amax": torch.tensor(2.0),
        },
        path,
        metadata=_json_metadata(metadata),
    )

    with pytest.raises(ValueError, match="duplicate normalized.*up_proj.weight"):
        _load(path, metadata)


def test_load_rejects_missing_and_unexpected_exact_projection_names(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    path = tmp_path / "wrong-projections.safetensors"
    save_file(
        {
            "layers.0.mlp.gate_proj.weight": torch.tensor(1.0),
            "model.layers.0.mlp.up_proj.weight": torch.tensor(2.0),
        },
        path,
        metadata=_json_metadata(metadata),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"projection names do not match.*missing .*model\.layers\.0\.mlp\."
            r"gate_proj\.weight.*unexpected .*layers\.0\.mlp\.gate_proj\.weight"
        ),
    ):
        _load(
            path,
            metadata,
            expected_projection_names={
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            },
        )


@pytest.mark.parametrize(
    ("identity_key", "unexpected", "match"),
    [
        ("model_id", "other/model", "model_id"),
        ("model_revision", "other-revision", "model_revision"),
        ("quant_cfg", "NVFP4_DEFAULT_CFG", "quant_cfg"),
    ],
)
def test_load_rejects_unexpected_artifact_identity(
    tmp_path: Path,
    metadata: dict[str, str | int],
    identity_key: str,
    unexpected: str,
    match: str,
) -> None:
    path = tmp_path / f"wrong-{identity_key}.safetensors"
    artifact_metadata = dict(metadata)
    artifact_metadata[identity_key] = unexpected
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
        path,
        metadata=_json_metadata(artifact_metadata),
    )

    with pytest.raises(ValueError, match=f"{match}.*does not match"):
        _load(path, metadata)


def test_load_accepts_identical_quant_cfg_at_different_paths(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    artifact_quant_cfg = tmp_path / "artifact-snapshot/nvfp4.yaml"
    expected_quant_cfg = tmp_path / "runtime-snapshot/nvfp4.yaml"
    for quant_cfg in (artifact_quant_cfg, expected_quant_cfg):
        quant_cfg.parent.mkdir()
        quant_cfg.write_text("quant_cfg: nvfp4\n")

    artifact_metadata = dict(metadata)
    artifact_metadata["quant_cfg"] = str(artifact_quant_cfg.resolve())
    expected_metadata = dict(metadata)
    expected_metadata["quant_cfg"] = str(expected_quant_cfg.resolve())
    path = tmp_path / "calibration.safetensors"
    save_nvfp4_calibration(
        path,
        {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
        **artifact_metadata,
    )

    calibration = _load(path, expected_metadata)

    assert list(calibration.input_amax) == ["model.layers.0.mlp.up_proj.weight"]


def test_load_rejects_different_quant_cfg_contents_at_different_paths(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    artifact_quant_cfg = tmp_path / "artifact-snapshot/nvfp4.yaml"
    expected_quant_cfg = tmp_path / "runtime-snapshot/nvfp4.yaml"
    artifact_quant_cfg.parent.mkdir()
    expected_quant_cfg.parent.mkdir()
    artifact_quant_cfg.write_text("quant_cfg: w4a4\n")
    expected_quant_cfg.write_text("quant_cfg: w4a16\n")

    artifact_metadata = dict(metadata)
    artifact_metadata["quant_cfg"] = str(artifact_quant_cfg.resolve())
    expected_metadata = dict(metadata)
    expected_metadata["quant_cfg"] = str(expected_quant_cfg.resolve())
    path = tmp_path / "calibration.safetensors"
    save_nvfp4_calibration(
        path,
        {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
        **artifact_metadata,
    )

    with pytest.raises(ValueError, match="quant_cfg.*does not match"):
        _load(path, expected_metadata)


def test_load_rejects_quant_cfg_mutated_after_artifact_creation(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    artifact_quant_cfg = tmp_path / "artifact-snapshot/nvfp4.yaml"
    expected_quant_cfg = tmp_path / "runtime-snapshot/nvfp4.yaml"
    artifact_quant_cfg.parent.mkdir()
    expected_quant_cfg.parent.mkdir()
    artifact_quant_cfg.write_text("quant_cfg: w4a4\n")
    expected_quant_cfg.write_text("quant_cfg: w4a16\n")

    artifact_metadata = dict(metadata)
    artifact_metadata["quant_cfg"] = str(artifact_quant_cfg.resolve())
    expected_metadata = dict(metadata)
    expected_metadata["quant_cfg"] = str(expected_quant_cfg.resolve())
    path = tmp_path / "calibration.safetensors"
    save_nvfp4_calibration(
        path,
        {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
        **artifact_metadata,
    )
    artifact_quant_cfg.write_text("quant_cfg: w4a16\n")

    with pytest.raises(ValueError, match="quant_cfg.*does not match"):
        _load(path, expected_metadata)


def test_load_rejects_same_quant_cfg_path_mutated_after_artifact_creation(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    quant_cfg = tmp_path / "nvfp4.yaml"
    quant_cfg.write_text("quant_cfg: w4a4\n")
    artifact_metadata = dict(metadata)
    artifact_metadata["quant_cfg"] = str(quant_cfg.resolve())
    path = tmp_path / "calibration.safetensors"
    save_nvfp4_calibration(
        path,
        {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
        **artifact_metadata,
    )
    quant_cfg.write_text("quant_cfg: w4a16\n")

    with pytest.raises(ValueError, match="quant_cfg.*does not match"):
        _load(path, artifact_metadata)


def test_save_rejects_file_backed_quant_cfg_hash_failure(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    quant_cfg = tmp_path / "nvfp4.yaml"
    quant_cfg.write_text("quant_cfg: w4a4\n")
    artifact_metadata = dict(metadata)
    artifact_metadata["quant_cfg"] = str(quant_cfg.resolve())

    with (
        patch.object(Path, "open", side_effect=OSError("injected read failure")),
        pytest.raises(ValueError, match="Could not hash.*quantization config"),
    ):
        save_nvfp4_calibration(
            tmp_path / "calibration.safetensors",
            {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
            **artifact_metadata,
        )


def test_save_rejects_missing_file_backed_quant_cfg(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    artifact_metadata = dict(metadata)
    artifact_metadata["quant_cfg"] = str((tmp_path / "missing.yaml").resolve())

    with pytest.raises(ValueError, match="Could not resolve.*quantization config"):
        save_nvfp4_calibration(
            tmp_path / "calibration.safetensors",
            {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
            **artifact_metadata,
        )


def test_save_accepts_symbolic_quant_cfg_with_yaml_suffix(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    symbolic_quant_cfg = "general/ptq/nvfp4_default-fp8_kv.yaml"
    artifact_metadata = dict(metadata)
    artifact_metadata["quant_cfg"] = symbolic_quant_cfg
    path = tmp_path / "calibration.safetensors"

    save_nvfp4_calibration(
        path,
        {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
        **artifact_metadata,
    )

    with safe_open(path, framework="pt", device="cpu") as artifact:
        assert "quant_cfg_sha256" not in artifact.metadata()
    _load(path, artifact_metadata)


@pytest.mark.parametrize(
    "invalid_amax",
    [
        torch.empty(0),
        torch.tensor([1.0]),
        torch.tensor(0.0),
        torch.tensor(-1.0),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
    ],
    ids=("empty", "non-scalar", "zero", "negative", "nan", "infinite"),
)
def test_load_rejects_invalid_input_amax(
    tmp_path: Path,
    metadata: dict[str, str | int],
    invalid_amax: torch.Tensor,
) -> None:
    path = tmp_path / "invalid-amax.safetensors"
    save_file(
        {"model.layers.0.mlp.up_proj.weight": invalid_amax},
        path,
        metadata=_json_metadata(metadata),
    )

    with pytest.raises(ValueError, match="scalar input amax|finite and positive"):
        _load(path, metadata)


def test_save_rejects_empty_artifact_and_alias_collisions(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        save_nvfp4_calibration(tmp_path / "empty.safetensors", {}, **metadata)

    duplicate = {
        "model.layers.0.mlp.up_proj.weight": torch.tensor(1.0),
        "model.layers.0.mlp.up_proj.input_quantizer._amax": torch.tensor(2.0),
    }
    with pytest.raises(ValueError, match="duplicate normalized.*up_proj.weight"):
        save_nvfp4_calibration(
            tmp_path / "duplicate.safetensors", duplicate, **metadata
        )
