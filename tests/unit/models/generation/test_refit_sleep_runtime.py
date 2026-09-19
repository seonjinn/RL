# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for runtime provenance safety."""

import pytest

from tests.functional.refit_sleep_runtime import _environment


def test_provenance_records_runtime_settings_not_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "private-token")
    monkeypatch.setenv("HF_TOKEN", "private-token")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "10.0")
    monkeypatch.setenv("NUM_OF_TOKENS_PER_CHUNK_COMBINE_API", "128")
    monkeypatch.setenv("NRL_FORCE_REBUILD_VENVS", "false")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/opt/nemo_rl_venv")
    recorded = _environment()
    assert "VLLM_API_KEY" not in recorded
    assert "HF_TOKEN" not in recorded
    assert recorded["NCCL_DEBUG"] == "INFO"
    assert recorded["TORCH_CUDA_ARCH_LIST"] == "10.0"
    assert recorded["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == "128"
    assert recorded["NRL_FORCE_REBUILD_VENVS"] == "false"
    assert recorded["UV_PROJECT_ENVIRONMENT"] == "/opt/nemo_rl_venv"
