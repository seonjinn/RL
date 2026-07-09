# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.vllm import patches, vllm_worker


def _llama_draft_loader_source(load_statement: str) -> str:
    return (
        "class DraftModel:\n"
        "    def __init__(self, loader):\n"
        "        self.loader = loader\n"
        "\n"
        "    def load_weights(self, weights):\n"
        "        model_weights = dict(weights)\n"
        "        includes_draft_id_mapping = (\n"
        "            'draft_id_to_target_id' in model_weights\n"
        "        )\n"
        "        includes_embed_tokens = any(\n"
        "            'embed_tokens' in name for name in model_weights\n"
        "        )\n"
        "        skip_substrs = []\n"
        "        if not includes_draft_id_mapping:\n"
        "            skip_substrs.append('draft_id_to_target_id')\n"
        "        if not includes_embed_tokens:\n"
        "            skip_substrs.append('embed_tokens')\n"
        "        loader = self.loader\n"
        f"{load_statement}"
    )


def _run_patched_llama_loader(
    source: str,
    loaded_weights: set[str],
    checkpoint_weights: set[str],
) -> tuple[set[str], object]:
    namespace: dict[str, object] = {}
    exec(source, namespace)

    class Loader:
        def load_weights(self, _weights) -> set[str]:
            return set(loaded_weights)

    model = namespace["DraftModel"](Loader())
    receipt = model.load_weights((name, object()) for name in checkpoint_weights)
    return receipt, model


def test_online_eagle_patch_marks_dummy_refit_head_as_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer = tmp_path / "llm_base_proposer.py"
    proposer.write_text(
        "class Proposer:\n"
        "    def load_model(self):\n"
        "        self.model = self._get_model()\n"
        "\n"
        "        self._maybe_share_embeddings(None)\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(proposer))

    patches._patch_vllm_online_eagle_head_ownership(MagicMock())

    patched = proposer.read_text()
    assert "online_refit_uses_dummy_drafter" in patched
    assert "self.model.has_own_lm_head = True" in patched
    assert "self.model.has_own_embed_tokens = False" in patched


def test_ray_worker_patch_rejects_unknown_vllm_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("class RayDistributedExecutor: pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(ray_executor))

    with pytest.raises(RuntimeError, match="vLLM source layout changed"):
        patches._patch_vllm_init_workers_ray("/venv/bin/python", None)


def test_ray_worker_patch_forwards_draft_cudagraph_runtime_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("        self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(ray_executor))
    monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)

    patches._patch_vllm_init_workers_ray("/venv/bin/python", None)

    copied_names = set(
        name
        for name in patches.os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"].split(",")
        if name
    )
    assert "NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH" in copied_names


def test_static_draft_loader_patch_uses_draft_load_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_model = tmp_path / "draft_model.py"
    draft_model.write_text(
        "        return replace(\n"
        "            base,\n"
        "            quant_config=None,\n"
        "            parallel_config=replace(\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(draft_model))

    patches._patch_vllm_draft_model_load_config(MagicMock())

    patched = draft_model.read_text()
    assert "load_config=spec.draft_load_config or base.load_config" in patched


def test_v2_eagle_patch_uses_draft_load_config_and_preserves_online_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eagle_utils = tmp_path / "utils.py"
    eagle_utils.write_text(
        "    speculative_config = vllm_config.speculative_config\n"
        "    assert speculative_config is not None\n"
        "    draft_model_config = speculative_config.draft_model_config\n"
        '    with set_model_tag("eagle_head"):\n'
        "        eagle_model = get_model(\n"
        "            vllm_config=vllm_config, model_config=draft_model_config\n"
        "        )\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(eagle_utils))

    patches._patch_vllm_v2_eagle_load_config_and_ownership(MagicMock())

    patched = eagle_utils.read_text()
    assert "load_config=draft_load_config or vllm_config.load_config" in patched
    assert "online_refit_uses_dummy_drafter" in patched
    assert "eagle_model.has_own_lm_head = True" in patched
    assert "eagle_model.has_own_embed_tokens = False" in patched


def test_v2_dflash_patch_uses_draft_load_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dflash_utils = tmp_path / "utils.py"
    dflash_utils.write_text(
        '    with set_model_tag("dflash_head"):\n'
        "        dflash_model = get_model(\n"
        "            vllm_config=draft_vllm_config, model_config=draft_model_config\n"
        "        )\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(dflash_utils))

    patches._patch_vllm_v2_dflash_load_config(MagicMock())

    patched = dflash_utils.read_text()
    assert "load_config=speculative_config.draft_load_config" in patched
    assert "or vllm_config.load_config" in patched


def test_v2_dflash_patch_skips_when_dflash_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(
        patches,
        "_get_vllm_file",
        MagicMock(side_effect=RuntimeError("missing dflash module")),
    )

    patches._patch_vllm_v2_dflash_load_config(logger)

    logger.info.assert_called_once()
    assert "not installed" in logger.info.call_args.args[0]


def test_qwen3_draft_loader_patch_returns_loaded_parameter_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eagle_model = tmp_path / "qwen3_eagle3.py"
    eagle_model.write_text("        loader.load_weights(model_weights.items())\n")
    dflash_model = tmp_path / "qwen3_dflash.py"
    dflash_model.write_text(
        "        loader.load_weights(model_weights.items())\n"
        "        self.model._build_fused_kv_buffers()\n"
    )

    def get_vllm_file(path: str) -> str:
        if path.endswith("qwen3_eagle3.py"):
            return str(eagle_model)
        if path.endswith("qwen3_dflash.py"):
            return str(dflash_model)
        raise AssertionError(f"Unexpected vLLM path: {path}")

    monkeypatch.setattr(patches, "_get_vllm_file", get_vllm_file)

    patches._patch_vllm_qwen3_draft_loader_results(MagicMock())

    assert (
        "return loader.load_weights(model_weights.items())" in eagle_model.read_text()
    )
    assert "self.has_own_lm_head = any(" in eagle_model.read_text()
    assert 'name.startswith("lm_head.")' in eagle_model.read_text()
    dflash_source = dflash_model.read_text()
    assert (
        "loaded_weights = loader.load_weights(model_weights.items())" in dflash_source
    )
    assert "self.model._build_fused_kv_buffers()" in dflash_source
    assert "return loaded_weights" in dflash_source


def test_qwen3_draft_loader_patch_skips_missing_dflash_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eagle_model = tmp_path / "qwen3_eagle3.py"
    eagle_model.write_text("        loader.load_weights(model_weights.items())\n")
    logger = MagicMock()

    def get_vllm_file(path: str) -> str:
        if path.endswith("qwen3_eagle3.py"):
            return str(eagle_model)
        if path.endswith("qwen3_dflash.py"):
            raise RuntimeError("missing dflash module")
        raise AssertionError(f"Unexpected vLLM path: {path}")

    monkeypatch.setattr(patches, "_get_vllm_file", get_vllm_file)

    patches._patch_vllm_qwen3_draft_loader_results(logger)

    assert "self.has_own_lm_head = any(" in eagle_model.read_text()
    assert any("not installed" in call.args[0] for call in logger.info.call_args_list)


def test_qwen3_draft_loader_patch_skips_when_models_are_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(
        patches,
        "_get_vllm_file",
        MagicMock(side_effect=RuntimeError("missing optional model")),
    )

    patches._patch_vllm_qwen3_draft_loader_results(logger)

    assert logger.info.call_count == 2
    assert all("not installed" in call.args[0] for call in logger.info.call_args_list)


def test_llama_draft_loader_patch_returns_loaded_parameter_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama_model = tmp_path / "llama_eagle3.py"
    llama_model.write_text("        loader.load_weights(model_weights.items())\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(llama_model))

    patches._patch_vllm_llama_draft_loader_result(MagicMock())

    patched = llama_model.read_text()
    assert "self.has_own_embed_tokens = includes_embed_tokens" in patched
    assert "self.has_own_lm_head = any(" in patched
    assert 'name.startswith("lm_head.")' in patched
    assert "loaded_weights = loader.load_weights(model_weights.items())" in patched
    assert "return loaded_weights | intentional_default_params" in patched


def test_llama_draft_loader_patch_tracks_only_intentional_sparse_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama_model = tmp_path / "llama_eagle3.py"
    llama_model.write_text(
        _llama_draft_loader_source(
            "        loader.load_weights(model_weights.items())\n"
        )
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(llama_model))

    patches._patch_vllm_llama_draft_loader_result(MagicMock())

    loaded, model = _run_patched_llama_loader(
        llama_model.read_text(),
        loaded_weights={"model.layers.0.weight"},
        checkpoint_weights={"model.layers.0.weight"},
    )
    assert loaded == {
        "draft_id_to_target_id",
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.weight",
    }
    assert model.has_own_embed_tokens is False
    assert model.has_own_lm_head is False

    model_parameters = loaded | {"model.layers.1.weight"}
    assert model_parameters - loaded == {"model.layers.1.weight"}


def test_llama_draft_loader_patch_does_not_mask_failed_owned_weight_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama_model = tmp_path / "llama_eagle3.py"
    llama_model.write_text(
        _llama_draft_loader_source(
            "        loader.load_weights(model_weights.items())\n"
        )
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(llama_model))

    patches._patch_vllm_llama_draft_loader_result(MagicMock())

    loaded, model = _run_patched_llama_loader(
        llama_model.read_text(),
        loaded_weights={"model.layers.0.weight"},
        checkpoint_weights={
            "draft_id_to_target_id",
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.0.weight",
        },
    )
    assert model.has_own_embed_tokens is True
    assert model.has_own_lm_head is True
    assert {
        "draft_id_to_target_id",
        "lm_head.weight",
        "model.embed_tokens.weight",
    }.isdisjoint(loaded)


def test_llama_draft_loader_patch_upgrades_legacy_receipt_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama_model = tmp_path / "llama_eagle3.py"
    llama_model.write_text(
        _llama_draft_loader_source(
            "        self.has_own_lm_head = any(\n"
            '            name.startswith("lm_head.") for name in model_weights\n'
            "        )\n"
            "        return loader.load_weights(model_weights.items())\n"
        )
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(llama_model))

    patches._patch_vllm_llama_draft_loader_result(MagicMock())
    patched = llama_model.read_text()
    patches._patch_vllm_llama_draft_loader_result(MagicMock())

    assert "intentional_default_params" in patched
    assert llama_model.read_text() == patched


def test_llama_draft_loader_patch_rejects_unknown_source_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama_model = tmp_path / "llama_eagle3.py"
    llama_model.write_text("class DraftModel: pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(llama_model))

    with pytest.raises(RuntimeError, match="vLLM source layout changed"):
        patches._patch_vllm_llama_draft_loader_result(MagicMock())


def test_llama_draft_loader_patch_skips_when_model_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(
        patches,
        "_get_vllm_file",
        MagicMock(side_effect=RuntimeError("missing optional model")),
    )

    patches._patch_vllm_llama_draft_loader_result(logger)

    logger.info.assert_called_once()
    assert "not installed" in logger.info.call_args.args[0]


def test_qwen3_draft_loader_patch_upgrades_legacy_receipt_only_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eagle_model = tmp_path / "qwen3_eagle3.py"
    eagle_model.write_text(
        "        return loader.load_weights(model_weights.items())\n"
    )
    dflash_model = tmp_path / "qwen3_dflash.py"
    dflash_model.write_text(
        "        loaded_weights = loader.load_weights(model_weights.items())\n"
        "        self.model._build_fused_kv_buffers()\n"
        "        return loaded_weights\n"
    )

    def get_vllm_file(path: str) -> str:
        if path.endswith("qwen3_eagle3.py"):
            return str(eagle_model)
        if path.endswith("qwen3_dflash.py"):
            return str(dflash_model)
        raise AssertionError(f"Unexpected vLLM path: {path}")

    monkeypatch.setattr(patches, "_get_vllm_file", get_vllm_file)

    patches._patch_vllm_qwen3_draft_loader_results(MagicMock())

    assert "self.has_own_lm_head = any(" in eagle_model.read_text()


def test_missing_probabilistic_draft_row_patch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_runner = tmp_path / "gpu_model_runner.py"
    model_runner.write_text(
        "            if row_idx is None:\n"
        "                logger.warning(\n"
        '                    "Missing cached draft probabilities for request %s; "\n'
        '                    "falling back to legacy speculative rejection behavior.",\n'
        "                    req_id,\n"
        "                )\n"
        "                return None\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(model_runner))

    patches._patch_vllm_missing_draft_probs_fail_closed(MagicMock())

    patched = model_runner.read_text()
    assert "raise RuntimeError(" in patched
    assert "missing q(token)" in patched
    assert "falling back to legacy" not in patched

    patches._patch_vllm_missing_draft_probs_fail_closed(MagicMock())

    assert model_runner.read_text() == patched


def test_medusa_loader_patch_uses_draft_load_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medusa = tmp_path / "medusa.py"
    medusa.write_text(
        "            self.model = get_model(\n"
        "                vllm_config=self.vllm_config,\n"
        "                model_config=self.spec_config.draft_model_config,\n"
        "            )\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(medusa))

    patches._patch_vllm_medusa_load_config(MagicMock())

    patched = medusa.read_text()
    assert "load_config=self.spec_config.draft_load_config" in patched


def test_draft_model_cudagraph_patch_initializes_generic_proposer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_runner = tmp_path / "gpu_model_runner.py"
    model_runner.write_text(
        "        if self.speculative_config and (\n"
        "            self.speculative_config.use_eagle()\n"
        "            or self.speculative_config.uses_extract_hidden_states()\n"
        "        ):\n"
        "            assert isinstance(\n"
        "                self.drafter,\n"
        "                EagleProposer\n"
        "                | DFlashProposer\n"
        "                | ExtractHiddenStatesProposer\n"
        "                | Gemma4Proposer,\n"
        "            )\n"
        "            self.drafter.initialize_cudagraph_keys(cudagraph_mode)\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(model_runner))

    patches._patch_vllm_draft_model_cudagraph_keys(MagicMock())

    patched = model_runner.read_text()
    assert "self.speculative_config.uses_draft_model()" in patched
    assert "NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH" in patched
    assert "| DraftModelProposer" in patched


def test_draft_model_cudagraph_patch_supports_nightly_without_gemma4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_runner = tmp_path / "gpu_model_runner.py"
    model_runner.write_text(
        "        if self.speculative_config and (\n"
        "            self.speculative_config.use_eagle()\n"
        "            or self.speculative_config.uses_extract_hidden_states()\n"
        "        ):\n"
        "            assert isinstance(\n"
        "                self.drafter,\n"
        "                EagleProposer | DFlashProposer | ExtractHiddenStatesProposer,\n"
        "            )\n"
        "            self.drafter.initialize_cudagraph_keys(cudagraph_mode)\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(model_runner))

    patches._patch_vllm_draft_model_cudagraph_keys(MagicMock())

    patched = model_runner.read_text()
    assert "self.speculative_config.uses_draft_model()" in patched
    assert "NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH" in patched
    assert "DraftModelProposer" in patched


@pytest.mark.parametrize(
    (
        "speculative_config",
        "expect_specdec_patches",
        "expect_probabilistic_guard",
        "expect_draft_cg_patch",
    ),
    [
        (None, False, False, False),
        ({}, False, False, False),
        ({"method": "eagle3"}, True, False, False),
        (
            {"method": "eagle3", "draft_sample_method": "probabilistic"},
            True,
            True,
            False,
        ),
        (
            {
                "method": "eagle3",
                "rejection_sample_method": "synthetic",
                "draft_sample_method": "probabilistic",
            },
            True,
            False,
            False,
        ),
        ({"method": "draft_model"}, True, False, True),
    ],
)
def test_apply_patches_only_installs_required_specdec_patches(
    monkeypatch: pytest.MonkeyPatch,
    speculative_config: dict[str, object] | None,
    expect_specdec_patches: bool,
    expect_probabilistic_guard: bool,
    expect_draft_cg_patch: bool,
) -> None:
    logger = MagicMock()
    vllm_module = ModuleType("vllm")
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = lambda _name: logger
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    always_patch_names = (
        "_patch_vllm_init_workers_ray",
        "_patch_vllm_hermes_tool_parser_thread_safety",
    )
    specdec_patch_names = (
        "_patch_vllm_llama_eagle3_own_lm_head",
        "_patch_vllm_online_eagle_head_ownership",
        "_patch_vllm_draft_model_load_config",
        "_patch_vllm_v2_eagle_load_config_and_ownership",
        "_patch_vllm_v2_dflash_load_config",
        "_patch_vllm_qwen3_draft_loader_results",
        "_patch_vllm_llama_draft_loader_result",
        "_patch_vllm_medusa_load_config",
    )
    patch_mocks = {}
    for name in (*always_patch_names, *specdec_patch_names):
        patch_mocks[name] = MagicMock()
        monkeypatch.setattr(patches, name, patch_mocks[name])
    probabilistic_guard = MagicMock()
    monkeypatch.setattr(
        patches,
        "_patch_vllm_missing_draft_probs_fail_closed",
        probabilistic_guard,
    )
    draft_cg_patch = MagicMock()
    monkeypatch.setattr(
        patches,
        "_patch_vllm_draft_model_cudagraph_keys",
        draft_cg_patch,
        raising=False,
    )
    if expect_draft_cg_patch:
        monkeypatch.setenv("NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH", "true")
    patches._apply_vllm_patches(
        "/venv/bin/python", speculative_config=speculative_config
    )

    patch_mocks["_patch_vllm_init_workers_ray"].assert_called_once_with(
        "/venv/bin/python", None
    )
    patch_mocks["_patch_vllm_hermes_tool_parser_thread_safety"].assert_called_once_with(
        logger
    )
    for name in specdec_patch_names:
        if expect_specdec_patches:
            patch_mocks[name].assert_called_once_with(logger)
        else:
            patch_mocks[name].assert_not_called()
    if expect_draft_cg_patch:
        draft_cg_patch.assert_called_once_with(logger)
    else:
        draft_cg_patch.assert_not_called()
    if expect_probabilistic_guard:
        probabilistic_guard.assert_called_once_with(logger)
    else:
        probabilistic_guard.assert_not_called()


@pytest.mark.parametrize(
    "speculative_config",
    [None, {}, {"method": "eagle3", "num_speculative_tokens": 3}],
)
def test_worker_forwards_speculative_config_to_patch_runner(
    monkeypatch: pytest.MonkeyPatch,
    speculative_config: dict[str, object] | None,
) -> None:
    config = {
        "model_name": "test-model",
        "vllm_cfg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "precision": "bfloat16",
        },
    }
    if speculative_config is not None:
        config["vllm_kwargs"] = {"speculative_config": speculative_config}
    apply_patches = MagicMock()
    monkeypatch.setattr(vllm_worker, "_apply_vllm_patches", apply_patches)
    worker = object.__new__(vllm_worker.BaseVllmGenerationWorker)

    worker._init_config(config, None, 1.0, None, None)

    apply_patches.assert_called_once_with(
        worker.py_executable,
        extra_env_vars=None,
        speculative_config=speculative_config,
    )
