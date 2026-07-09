# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.vllm import patches


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
