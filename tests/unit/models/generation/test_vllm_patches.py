# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from enum import Enum
from pathlib import Path
from textwrap import dedent, indent
from types import ModuleType, SimpleNamespace
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


def _probabilistic_draft_temperature_source() -> str:
    return (
        "    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)\n"
        "    # consistent with sampler.py's _SAMPLING_EPS threshold\n"
        "    temperature = sampling_metadata.temperature\n"
        "    # Avoid division by zero if there are greedy requests.\n"
        "    if not sampling_metadata.all_random:\n"
        "        is_greedy = temperature < _SAMPLING_EPS\n"
        "        temperature = torch.where(is_greedy, 1.0, temperature)\n"
        "    logits.div_(temperature.view(-1, 1))\n"
    )


def _run_patched_parallel_probabilistic_draft_temperature(
    source: str,
    logits_count: int,
    temperature_count: int,
) -> None:
    namespace: dict[str, object] = {"_SAMPLING_EPS": 1e-5}
    exec(
        f"def sample(logits, sampling_metadata):\n{source}    return None\n",
        namespace,
    )

    class Logits:
        shape = (logits_count,)

    class Temperature:
        def numel(self) -> int:
            return temperature_count

    class SamplingMetadata:
        temperature = Temperature()
        all_random = False

    namespace["sample"](Logits(), SamplingMetadata())


def test_parallel_probabilistic_draft_temperature_patch_expands_temperatures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer = tmp_path / "llm_base_proposer.py"
    proposer.write_text(_probabilistic_draft_temperature_source())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(proposer))

    patches._patch_vllm_parallel_probabilistic_draft_temperature(MagicMock())

    patched = proposer.read_text()
    assert "temperature_count = temperature.numel()" in patched
    assert "logits_count % temperature_count != 0" in patched
    assert "temperature.repeat_interleave(logits_count // temperature_count)" in patched
    assert "if logits_count <= 0 or temperature_count <= 0:" in patched
    assert patched.index("if logits_count <= 0") < patched.index(
        "if temperature_count != logits_count"
    )

    patches._patch_vllm_parallel_probabilistic_draft_temperature(MagicMock())

    assert proposer.read_text() == patched


def test_parallel_probabilistic_draft_temperature_patch_preserves_request_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    proposer = tmp_path / "llm_base_proposer.py"
    proposer.write_text(_probabilistic_draft_temperature_source())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(proposer))

    patches._patch_vllm_parallel_probabilistic_draft_temperature(MagicMock())

    namespace: dict[str, object] = {"_SAMPLING_EPS": 1e-5, "torch": torch}
    exec(
        "def sample(logits, sampling_metadata):\n"
        f"{proposer.read_text()}"
        "    return logits.softmax(dim=-1, dtype=torch.float32)\n",
        namespace,
    )

    class SamplingMetadata:
        temperature = torch.tensor([0.0, 2.0])
        all_random = False

    logits = torch.tensor(
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    )
    expected_temperature = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    expected = (logits / expected_temperature.view(-1, 1)).softmax(
        dim=-1, dtype=torch.float32
    )

    actual = namespace["sample"](logits.clone(), SamplingMetadata())

    torch.testing.assert_close(actual, expected)

    class EqualShapeSamplingMetadata:
        temperature = torch.tensor([0.5, 2.0])
        all_random = True

    equal_shape_logits = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    equal_shape_expected = (
        equal_shape_logits / EqualShapeSamplingMetadata.temperature.view(-1, 1)
    ).softmax(dim=-1, dtype=torch.float32)

    equal_shape_actual = namespace["sample"](
        equal_shape_logits.clone(), EqualShapeSamplingMetadata()
    )

    torch.testing.assert_close(equal_shape_actual, equal_shape_expected)


def test_parallel_probabilistic_draft_temperature_patch_rejects_non_divisible_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer = tmp_path / "llm_base_proposer.py"
    proposer.write_text(_probabilistic_draft_temperature_source())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(proposer))

    patches._patch_vllm_parallel_probabilistic_draft_temperature(MagicMock())

    with pytest.raises(RuntimeError, match="not divisible"):
        _run_patched_parallel_probabilistic_draft_temperature(
            proposer.read_text(), logits_count=5, temperature_count=2
        )


@pytest.mark.parametrize(
    ("logits_count", "temperature_count"),
    [(0, 0), (0, 1), (1, 0)],
)
def test_parallel_probabilistic_draft_temperature_patch_rejects_non_positive_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logits_count: int,
    temperature_count: int,
) -> None:
    proposer = tmp_path / "llm_base_proposer.py"
    proposer.write_text(_probabilistic_draft_temperature_source())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(proposer))

    patches._patch_vllm_parallel_probabilistic_draft_temperature(MagicMock())

    with pytest.raises(RuntimeError, match="parallel draft logits"):
        _run_patched_parallel_probabilistic_draft_temperature(
            proposer.read_text(), logits_count, temperature_count
        )


def test_parallel_probabilistic_draft_temperature_patch_rejects_partial_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer = tmp_path / "llm_base_proposer.py"
    proposer.write_text(
        "    temperature_count = temperature.numel()\n"
        f"{_probabilistic_draft_temperature_source()}"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(proposer))

    with pytest.raises(RuntimeError, match="incomplete"):
        patches._patch_vllm_parallel_probabilistic_draft_temperature(MagicMock())


def test_parallel_probabilistic_draft_temperature_patch_rejects_unknown_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer = tmp_path / "llm_base_proposer.py"
    proposer.write_text("class Proposer: pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(proposer))

    with pytest.raises(RuntimeError, match="vLLM source layout changed"):
        patches._patch_vllm_parallel_probabilistic_draft_temperature(MagicMock())


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
    assert "NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS" in copied_names


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
    monkeypatch.delenv("NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS", raising=False)
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
        "        if self._draft_probs is None or self._draft_prob_req_ids is None:\n"
        "            return None\n"
        "\n"
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
    assert "has no cached q(token)" in patched
    assert "any(spec_decode_metadata.num_draft_tokens)" in patched
    assert "not self.input_batch.sampling_metadata.all_greedy" in patched
    assert "falling back to legacy" not in patched

    patches._patch_vllm_missing_draft_probs_fail_closed(MagicMock())

    assert model_runner.read_text() == patched


def test_missing_probabilistic_draft_cache_patch_executes_fail_closed_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_runner = tmp_path / "gpu_model_runner.py"
    model_runner.write_text(
        "class Runner:\n"
        "    def get_draft_probs(self, spec_decode_metadata):\n"
        "        if self._draft_probs is None or self._draft_prob_req_ids is None:\n"
        "            return None\n"
        "\n"
        "        for req_id in ():\n"
        "            if row_idx is None:\n"
        "                logger.warning(\n"
        '                    "Missing cached draft probabilities for request %s; "\n'
        '                    "falling back to legacy speculative rejection behavior.",\n'
        "                    req_id,\n"
        "                )\n"
        "                return None\n"
        "        return object()\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(model_runner))

    patches._patch_vllm_missing_draft_probs_fail_closed(MagicMock())

    namespace: dict[str, object] = {}
    exec(model_runner.read_text(), namespace)

    class SamplingMetadata:
        all_greedy = False

    class InputBatch:
        sampling_metadata = SamplingMetadata()

    class SpecDecodeMetadata:
        num_draft_tokens = [1, 0]

    runner = namespace["Runner"]()
    runner._draft_probs = None
    runner._draft_prob_req_ids = None
    runner.input_batch = InputBatch()

    with pytest.raises(RuntimeError, match="has no cached q\(token\)"):
        runner.get_draft_probs(SpecDecodeMetadata())

    runner.input_batch.sampling_metadata.all_greedy = True
    assert runner.get_draft_probs(SpecDecodeMetadata()) is None


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


def test_piecewise_specdec_cudagraph_patch_aligns_capture_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compilation = tmp_path / "compilation.py"
    compilation.write_text(
        "        if (\n"
        "            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
        "            and uniform_decode_query_len > 1\n"
        "        ):\n"
        "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
        "                uniform_decode_query_len,\n"
        "                tensor_parallel_size,\n"
        "            )\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(compilation))

    patches._patch_vllm_piecewise_specdec_cudagraph_alignment(MagicMock())

    patched = compilation.read_text()
    assert "cudagraph_mode != CUDAGraphMode.NONE" in patched
    assert "cudagraph_mode.decode_mode() == CUDAGraphMode.FULL" not in patched

    patches._patch_vllm_piecewise_specdec_cudagraph_alignment(MagicMock())
    assert compilation.read_text() == patched


def test_cudagraph_dispatch_metrics_records_graph_and_fallback_calls() -> None:
    class Mode(Enum):
        NONE = 0
        PIECEWISE = 1

    class Dispatcher:
        keys_initialized = True
        cudagraph_mode = Mode.PIECEWISE
        compilation_config = SimpleNamespace(max_cudagraph_capture_size=64)

        def dispatch(self, num_tokens: int, **_kwargs):
            if num_tokens <= 64:
                return Mode.PIECEWISE, SimpleNamespace(num_tokens=64)
            return Mode.NONE, SimpleNamespace(num_tokens=num_tokens)

    patches._install_vllm_cudagraph_dispatch_metrics(Dispatcher)
    dispatcher = Dispatcher()

    dispatcher.dispatch(32)
    dispatcher.dispatch(96)

    assert dispatcher._nrl_cudagraph_dispatch_metrics == {
        "calls_none": 1,
        "calls_piecewise": 1,
        "unpadded_tokens_none": 96,
        "unpadded_tokens_piecewise": 32,
        "padded_tokens_none": 96,
        "padded_tokens_piecewise": 64,
        "fallback_oversize": 1,
    }


def test_cudagraph_dispatch_metrics_classifies_missing_capture_key() -> None:
    class Mode(Enum):
        NONE = 0
        PIECEWISE = 1

    class Dispatcher:
        keys_initialized = True
        cudagraph_mode = Mode.PIECEWISE
        compilation_config = SimpleNamespace(max_cudagraph_capture_size=64)

        def dispatch(self, num_tokens: int, **_kwargs):
            return Mode.NONE, SimpleNamespace(num_tokens=num_tokens)

    patches._install_vllm_cudagraph_dispatch_metrics(Dispatcher)
    patches._install_vllm_cudagraph_dispatch_metrics(Dispatcher)
    dispatcher = Dispatcher()

    dispatcher.dispatch(32)

    assert dispatcher._nrl_cudagraph_dispatch_metrics["calls_none"] == 1
    assert dispatcher._nrl_cudagraph_dispatch_metrics["fallback_missing_key"] == 1


def test_cudagraph_dispatch_metrics_classifies_uninitialized_v1_dispatcher() -> None:
    class Mode(Enum):
        NONE = 0

    class Dispatcher:
        keys_initialized = False
        cudagraph_mode = Mode.NONE
        compilation_config = SimpleNamespace(max_cudagraph_capture_size=64)

        def dispatch(self, num_tokens: int, **_kwargs):
            return Mode.NONE, SimpleNamespace(num_tokens=num_tokens)

    patches._install_vllm_cudagraph_dispatch_metrics(Dispatcher)
    dispatcher = Dispatcher()

    dispatcher.dispatch(32)

    assert dispatcher._nrl_cudagraph_dispatch_metrics["fallback_uninitialized"] == 1


def test_cudagraph_dispatch_metrics_supports_v2_graph_manager() -> None:
    class Mode(Enum):
        NONE = 0
        PIECEWISE = 1

    class Manager:
        _graphs_captured = True
        cudagraph_mode = Mode.PIECEWISE
        compilation_config = SimpleNamespace(max_cudagraph_capture_size=64)

        def dispatch(
            self,
            num_reqs: int,
            num_tokens: int,
            uniform_token_count: int | None,
            num_active_loras: int,
        ):
            del num_reqs, uniform_token_count, num_active_loras
            mode = Mode.PIECEWISE if num_tokens <= 64 else Mode.NONE
            return SimpleNamespace(cg_mode=mode, num_tokens=num_tokens)

    patches._install_vllm_cudagraph_dispatch_metrics(Manager)
    manager = Manager()

    manager.dispatch(8, 32, 4, 0)
    manager.dispatch(8, 96, 4, 0)

    assert manager._nrl_cudagraph_dispatch_metrics["calls_piecewise"] == 1
    assert manager._nrl_cudagraph_dispatch_metrics["calls_none"] == 1
    assert manager._nrl_cudagraph_dispatch_metrics["fallback_oversize"] == 1


def test_cudagraph_dispatch_metrics_source_patch_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = tmp_path / "cudagraph_dispatcher.py"
    manager = tmp_path / "cudagraph_utils.py"
    dispatcher.write_text("class CudagraphDispatcher:\n    pass\n")
    manager.write_text("class CudaGraphManager:\n    pass\n")
    vllm_files = {
        "v1/cudagraph_dispatcher.py": dispatcher,
        "v1/worker/gpu/cudagraph_utils.py": manager,
    }
    monkeypatch.setattr(patches, "_get_vllm_file", lambda path: str(vllm_files[path]))
    monkeypatch.delenv("NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS", raising=False)

    patches._patch_vllm_cudagraph_dispatch_metrics(MagicMock())
    patched = dispatcher.read_text()
    manager_patched = manager.read_text()

    assert "NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS" in patched
    assert "_install_vllm_cudagraph_dispatch_metrics" in patched
    assert "_install_vllm_cudagraph_dispatch_metrics(CudaGraphManager)" in (
        manager_patched
    )

    patches._patch_vllm_cudagraph_dispatch_metrics(MagicMock())
    assert dispatcher.read_text() == patched
    assert manager.read_text() == manager_patched


def _write_vllm_024_tail_gate_sources(tmp_path: Path) -> dict[str, Path]:
    sources = {
        "config/speculative.py": dedent(
            '''\
                # dynamic speculative decoding control
                num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None
                """Batch-size schedule used to dynamically choose speculative-token count.

                Each entry is ``(range_start, range_end, num_speculative_tokens)`` with an
                inclusive batch-size range.
                """

                # params generated in the post-init stage
            '''
        ),
        "v1/core/sched/output.py": dedent(
            """\
                # Dynamic speculative decoding: optimal K chosen by scheduler.
                # Number of spec tokens to schedule for the next step.
                num_spec_tokens_to_schedule: int = 0

                @classmethod
                def make_empty(cls) -> "SchedulerOutput":
            """
        ),
        "v1/worker/gpu/model_runner.py": dedent(
            """\
                @torch.inference_mode()
                def execute_model(
                    self,
                    scheduler_output: SchedulerOutput,
                    intermediate_tensors: IntermediateTensors | None = None,
                    dummy_run: bool = False,
                    skip_attn_for_dummy_run: bool = False,
                    is_profile: bool = False,
                ) -> ModelRunnerOutput | IntermediateTensors | None:
                    if not dummy_run:
            """
        )
        + dedent(
            """\
                    finished_req_ids = scheduler_output.finished_req_ids
                    self.execute_model_state = ExecuteModelState(
                        input_batch=input_batch,
                        attn_metadata=attn_metadata,
                        slot_mappings_by_layer=slot_mappings_by_layer,
                        hidden_states=hidden_states,
                        aux_hidden_states=aux_hidden_states,
                        finished_req_ids=finished_req_ids,
                    )
            """
        )
        + dedent(
            """\
                    input_batch = self.execute_model_state.input_batch
                    attn_metadata = self.execute_model_state.attn_metadata
                    slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
                    hidden_states = self.execute_model_state.hidden_states
                    aux_hidden_states = self.execute_model_state.aux_hidden_states
                    finished_req_ids = self.execute_model_state.finished_req_ids
                    self.execute_model_state = None
            """
        )
        + dedent(
            """\
                        draft_tokens = self.speculator.propose(
                            input_batch,
                            attn_metadata,
                            slot_mappings_by_layer,
                            spec_hidden_states,
                            aux_hidden_states,
                            num_sampled,
                            num_rejected,
                            self.req_states.last_sampled_tokens,
                            self.req_states.next_prefill_tokens,
                            self.sampler.sampling_states.temperature.gpu,
                            self.sampler.sampling_states.seeds.gpu,
                            mm_inputs=mm_inputs,
                        )
                        self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens

                    if self.num_speculative_steps > 0:
                        # Spec-decode and diffusion LLMs both use draft tokens but the latter does
                        # not have a speculator (i.e. self.speculator is None)
                        self.draft_tokens_handler.set_draft_tokens(
                            input_batch,
                            self.req_states.draft_tokens[input_batch.idx_mapping],
                        )
            """
        )
        + dedent(
            """\
                class ExecuteModelState(NamedTuple):
                    input_batch: InputBatch
                    attn_metadata: dict[str, Any] | None
                    slot_mappings_by_layer: dict[str, torch.Tensor] | None
                    hidden_states: torch.Tensor | None
                    aux_hidden_states: list[torch.Tensor] | None
                    finished_req_ids: set[str]
            """
        ),
        "v1/worker/gpu/spec_decode/autoregressive/speculator.py": dedent(
            """\
                def propose(
                    self,
                    input_batch: InputBatch,
                    attn_metadata: dict[str, Any],
                    slot_mappings: dict[str, torch.Tensor],
                    # [num_tokens, hidden_size]
                    last_hidden_states: torch.Tensor,
                    # num_layers x [num_tokens, hidden_size]
                    aux_hidden_states: list[torch.Tensor] | None,
                    # [num_reqs]
                    num_sampled: torch.Tensor,
                    # [num_reqs]
                    num_rejected: torch.Tensor,
                    # [max_num_reqs]
                    last_sampled: torch.Tensor,
                    # [max_num_reqs]
                    next_prefill_tokens: torch.Tensor,
                    # [max_num_reqs]
                    temperature: torch.Tensor,
                    # [max_num_reqs]
                    seeds: torch.Tensor,
                    num_tokens_across_dp: torch.Tensor | None = None,
                    dummy_run: bool = False,
                    skip_attn_for_dummy_run: bool = False,
                    mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
                    is_profile: bool = False,
                ) -> torch.Tensor:
                    num_tokens = input_batch.num_tokens_after_padding
            """
        )
        + dedent(
            """\
                    self.draft_max_seq_len = min(
                        max_seq_len + self.num_speculative_steps, self.max_model_len
                    )
            """
        )
        + dedent(
            """\
                        self._prefill(
                            num_reqs,
                            prefill_batch_desc.num_tokens,
                            attn_metadata,
                            slot_mappings,
                            num_tokens_across_dp=num_tokens_across_dp,
                            cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                            mm_inputs=mm_inputs,
                        )

                    if self.num_speculative_steps == 1:
                        # Early exit.
                        return self.draft_tokens[:num_reqs, :1]

                    # Prepare the inputs for the decode steps.
                    prepare_decode_inputs(
            """
        )
        + dedent(
            """\
                    # Generate the remaining num_speculative_steps - 1 draft tokens.
                    self._multi_step_decode(
                        num_reqs,
                        dummy_run and skip_attn_for_dummy_run,
                        decode_batch_desc,
                        num_tokens_across_dp,
                    )

                    return self.draft_tokens[:num_reqs]
            """
        ),
        "v1/worker/gpu_model_runner.py": dedent(
            """\
                @torch.inference_mode()
                def execute_model(
                    self,
                    scheduler_output: "SchedulerOutput",
                    intermediate_tensors: IntermediateTensors | None = None,
                ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
                    if self.execute_model_state is not None:
                        raise RuntimeError(
                            "State error: sample_tokens() must be called "
                            "after execute_model() returns None."
                        )

                    if self.routed_experts_initialized:
            """
        ),
    }
    for relative_path in (
        "config/speculative.py",
        "v1/core/sched/output.py",
        "v1/worker/gpu_model_runner.py",
    ):
        sources[relative_path] = indent(sources[relative_path], "    ")
    v2_source, execute_state = sources["v1/worker/gpu/model_runner.py"].split(
        "class ExecuteModelState", 1
    )
    v2_execute, v2_body = v2_source.split(
        "finished_req_ids = scheduler_output.finished_req_ids", 1
    )
    sources["v1/worker/gpu/model_runner.py"] = (
        indent(v2_execute, "    ")
        + indent(
            "finished_req_ids = scheduler_output.finished_req_ids" + v2_body,
            "        ",
        )
        + "class ExecuteModelState"
        + execute_state
    )
    speculator_source = sources[
        "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    ]
    speculator_signature, speculator_body = speculator_source.split(
        "self.draft_max_seq_len", 1
    )
    sources["v1/worker/gpu/spec_decode/autoregressive/speculator.py"] = indent(
        speculator_signature, "    "
    ) + indent("self.draft_max_seq_len" + speculator_body, "        ")
    paths = {}
    for relative_path, source in sources.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        paths[relative_path] = path
    return paths


def _install_tail_gate_source_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    paths = _write_vllm_024_tail_gate_sources(tmp_path)
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda relative_path: str(paths[relative_path])
    )
    return paths


def test_runtime_tail_gate_patch_applies_vllm_024_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _install_tail_gate_source_fixture(tmp_path, monkeypatch)

    patches._patch_vllm_runtime_tail_gating(MagicMock())

    config = paths["config/speculative.py"].read_text()
    assert 'sd_tail_gate_mode: str = "off"' in config
    assert "sd_tail_gate_threshold: int | None = None" in config
    assert "sd_tail_gate_consecutive_checks: int = 10" in config
    assert "sd_tail_gate_margin: float = 0.05" in config
    assert "sd_tail_gate_config_path: str | None = None" in config
    assert 'sd_tail_gate_off_mode: str = "advance_only"' in config

    scheduler_output = paths["v1/core/sched/output.py"].read_text()
    assert scheduler_output.count("num_spec_tokens_to_schedule: int = 0") == 1
    for field in (
        "tail_gate_state",
        "tail_gate_tick",
        "tail_gate_active_requests",
        "tail_gate_decode_active_requests",
        "tail_gate_mean_sequence_length",
        "tail_gate_predicted_speedup_sum",
        "tail_gate_predicted_speedup_count",
        "tail_gate_expected_accept_length",
        "tail_gate_just_activated",
    ):
        assert field in scheduler_output

    model_runner = paths["v1/worker/gpu/model_runner.py"].read_text()
    assert model_runner.index("runtime_num_spec_tokens") < model_runner.index(
        "if not dummy_run:"
    )
    assert "runtime_num_spec_tokens not in" in model_runner
    assert 'self.speculative_config.method not in ("eagle", "eagle3")' in model_runner
    assert "self.speculative_config.use_eagle()" not in model_runner
    assert "num_spec_tokens_to_schedule=runtime_num_spec_tokens" in model_runner
    assert "self.execute_model_state.num_spec_tokens_to_schedule" in model_runner
    assert "num_speculative_tokens=runtime_num_spec_tokens" in model_runner
    assert "self.req_states.draft_tokens[input_batch.idx_mapping] = 0" in model_runner
    assert "draft_tokens_for_handler" in model_runner
    assert "_nrl_tail_gate_metrics" in model_runner
    assert "vllm:spec_decode_tail_gate_decode_active_requests_sum" in model_runner

    speculator = paths[
        "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    ].read_text()
    assert "num_speculative_tokens: int | None = None" in speculator
    assert "runtime_num_spec_tokens not in" in speculator
    k0_branch = speculator.index("if runtime_num_spec_tokens == 0:")
    assert speculator.index("self._prefill(") < k0_branch
    assert k0_branch < speculator.index("prepare_decode_inputs(")
    assert k0_branch < speculator.index("self._multi_step_decode(")
    assert "return self.draft_tokens[:num_reqs, :0]" in speculator

    v1_model_runner = paths["v1/worker/gpu_model_runner.py"].read_text()
    assert "_nrl_tail_gate_metrics" in v1_model_runner
    assert "vllm:spec_decode_tail_gate_decode_active_requests_sum" in v1_model_runner
    assert "num_spec_tokens_to_schedule" in v1_model_runner
    assert "propose_draft_token_ids" not in v1_model_runner


def test_runtime_tail_gate_patch_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _install_tail_gate_source_fixture(tmp_path, monkeypatch)
    patches._patch_vllm_runtime_tail_gating(MagicMock())
    patched = {relative_path: path.read_text() for relative_path, path in paths.items()}

    patches._patch_vllm_runtime_tail_gating(MagicMock())

    assert {
        relative_path: path.read_text() for relative_path, path in paths.items()
    } == (patched)


@pytest.mark.parametrize(
    ("changed_path", "old_anchor"),
    [
        ("config/speculative.py", "# params generated in the post-init stage"),
        (
            "v1/core/sched/output.py",
            '@classmethod\n    def make_empty(cls) -> "SchedulerOutput":',
        ),
        ("v1/worker/gpu/model_runner.py", "if not dummy_run:"),
        (
            "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
            "if self.num_speculative_steps == 1:",
        ),
        ("v1/worker/gpu_model_runner.py", "if self.routed_experts_initialized:"),
    ],
)
def test_runtime_tail_gate_patch_validates_all_sources_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_path: str,
    old_anchor: str,
) -> None:
    paths = _install_tail_gate_source_fixture(tmp_path, monkeypatch)
    changed_source = (
        paths[changed_path].read_text().replace(old_anchor, f"changed {old_anchor}", 1)
    )
    assert changed_source != paths[changed_path].read_text()
    paths[changed_path].write_text(changed_source)
    original = {
        relative_path: path.read_text() for relative_path, path in paths.items()
    }

    with pytest.raises(RuntimeError, match="vLLM 0.24 source layout changed"):
        patches._patch_vllm_runtime_tail_gating(MagicMock())

    assert {
        relative_path: path.read_text() for relative_path, path in paths.items()
    } == (original)


@pytest.mark.parametrize(
    (
        "speculative_config",
        "expect_specdec_patches",
        "expect_probabilistic_patches",
        "expect_draft_cg_patch",
        "expect_tail_gate_patch",
    ),
    [
        (None, False, False, False, False),
        ({}, False, False, False, False),
        ({"method": "eagle3"}, True, False, False, False),
        (
            {"method": "eagle3", "draft_sample_method": "probabilistic"},
            True,
            True,
            False,
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
            False,
        ),
        ({"method": "draft_model"}, True, False, True, False),
        (
            {"method": "eagle3", "sd_tail_gate_mode": "threshold"},
            True,
            False,
            False,
            True,
        ),
    ],
)
def test_apply_patches_only_installs_required_specdec_patches(
    monkeypatch: pytest.MonkeyPatch,
    speculative_config: dict[str, object] | None,
    expect_specdec_patches: bool,
    expect_probabilistic_patches: bool,
    expect_draft_cg_patch: bool,
    expect_tail_gate_patch: bool,
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
        "_patch_vllm_piecewise_specdec_cudagraph_alignment",
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
    parallel_probabilistic_temperature_patch = MagicMock()
    monkeypatch.setattr(
        patches,
        "_patch_vllm_parallel_probabilistic_draft_temperature",
        parallel_probabilistic_temperature_patch,
        raising=False,
    )
    draft_cg_patch = MagicMock()
    monkeypatch.setattr(
        patches,
        "_patch_vllm_draft_model_cudagraph_keys",
        draft_cg_patch,
        raising=False,
    )
    tail_gate_patch = MagicMock()
    monkeypatch.setattr(
        patches,
        "_patch_vllm_runtime_tail_gating",
        tail_gate_patch,
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
    if expect_tail_gate_patch:
        tail_gate_patch.assert_called_once_with(logger)
    else:
        tail_gate_patch.assert_not_called()
    if expect_probabilistic_patches:
        probabilistic_guard.assert_called_once_with(logger)
        parallel_probabilistic_temperature_patch.assert_called_once_with(logger)
    else:
        probabilistic_guard.assert_not_called()
        parallel_probabilistic_temperature_patch.assert_not_called()


def test_apply_patches_installs_cudagraph_metrics_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = lambda _name: logger
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    monkeypatch.setenv("NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS", "true")
    monkeypatch.setattr(patches, "_patch_vllm_init_workers_ray", MagicMock())
    metrics_patch = MagicMock()
    monkeypatch.setattr(
        patches, "_patch_vllm_cudagraph_dispatch_metrics", metrics_patch
    )
    monkeypatch.setattr(
        patches, "_patch_vllm_hermes_tool_parser_thread_safety", MagicMock()
    )

    patches._apply_vllm_patches("/venv/bin/python", speculative_config=None)

    metrics_patch.assert_called_once_with(logger)


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
