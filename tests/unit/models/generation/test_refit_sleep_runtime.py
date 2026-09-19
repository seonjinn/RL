# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for runtime provenance and functional-gate lifecycle safety."""

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from tests.functional.refit_sleep_runtime import _environment, managed_ray_session
from tests.functional.refit_sleep_utils import (
    failure_deadline,
    incomplete_receiver_manifest,
    initialize_refit_manifest,
)


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


def test_failure_manifest_reuses_metadata_exported_while_policy_is_resident() -> None:
    events: list[str] = []

    class Policy:
        resident = True

        def prepare_refit_info(self, *, refit_payload_mode: str) -> dict[str, Any]:
            assert self.resident, "weight metadata cannot be exported after offload"
            events.append(f"export:{refit_payload_mode}")
            return {"model.weight": (torch.Size([2, 2]), torch.bfloat16)}

    class Generation:
        sleeping = True

        def get_refit_payload_mode(self) -> str:
            return "raw"

        def prepare_refit_info(self, manifest: dict[str, Any]) -> None:
            assert self.sleeping or len(manifest) == 2
            events.append(
                "install-failure" if len(manifest) == 2 else "install-initial"
            )

    policy = Policy()
    generation = Generation()
    initialized_manifest = initialize_refit_manifest(policy, generation)
    policy.resident = False
    generation.sleeping = False

    events.append("destructive-sleep")
    with failure_deadline(1):
        events.append("watchdog-started")
        generation.prepare_refit_info(
            incomplete_receiver_manifest(initialized_manifest)
        )

    assert events == [
        "export:raw",
        "install-initial",
        "destructive-sleep",
        "watchdog-started",
        "install-failure",
    ]


def test_managed_ray_session_spans_two_qwen_fixture_lifecycles() -> None:
    events: list[str] = []

    class Ray:
        initialized = False

        def is_initialized(self) -> bool:
            return self.initialized

        def shutdown(self) -> None:
            assert self.initialized
            events.append("ray-shutdown")
            self.initialized = False

    ray = Ray()

    def initialize() -> None:
        assert not ray.initialized
        events.append("ray-init")
        ray.initialized = True

    @contextmanager
    def model_lifecycle(name: str) -> Iterator[None]:
        assert ray.initialized
        events.append(f"{name}-start")
        try:
            yield
        finally:
            assert ray.initialized
            events.append(f"{name}-stop")

    with managed_ray_session(ray_module=ray, initialize=initialize):
        with model_lifecycle("abc"):
            for state in ("candidate", "fresh-B", "fresh-C"):
                assert ray.initialized
                events.append(state)
        with model_lifecycle("failure"):
            assert ray.initialized

    assert events == [
        "ray-init",
        "abc-start",
        "candidate",
        "fresh-B",
        "fresh-C",
        "abc-stop",
        "failure-start",
        "failure-stop",
        "ray-shutdown",
    ]
