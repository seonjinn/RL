# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence, cast

import torch

from nemo_rl.models.dflash2_contract import (
    DFlash2CheckpointContract,
    inspect_dflash2_checkpoint_if_present,
)

SpeculatorType = Literal["eagle3", "dflash", "dflash2", "dspark"]


class SpeculatorRuntimeError(RuntimeError):
    """The runtime cannot safely apply a speculative-model refit."""


class RunnerFamily(str, Enum):
    """Capability used to access the live speculative model."""

    ACCESSOR = "get_draft_model"
    DRAFTER = "drafter.model"
    SPECULATOR = "speculator.model"


def validate_speculator_runtime_contract(
    *,
    speculator_type: str,
    num_speculative_tokens: int | None,
) -> SpeculatorType:
    """Validate variant-specific serving geometry before runner discovery."""
    if speculator_type not in ("eagle3", "dflash", "dflash2", "dspark"):
        raise SpeculatorRuntimeError(f"unsupported speculator_type={speculator_type!r}")
    if speculator_type == "dflash2" and num_speculative_tokens != 7:
        raise SpeculatorRuntimeError(
            "DFlash2 requires runtime num_speculative_tokens=7"
        )
    return cast(SpeculatorType, speculator_type)


def validate_vllm_speculative_startup(
    speculative_config: Mapping[str, object] | None,
) -> DFlash2CheckpointContract | None:
    """Recognize and validate a static DFlash2 checkpoint before vLLM starts."""
    if speculative_config is None:
        return None
    method = speculative_config.get("method")
    if method == "dflash2":
        raise SpeculatorRuntimeError(
            "vLLM serves DFlash2 checkpoints with method='dflash', not 'dflash2'"
        )
    if method not in (None, "dflash"):
        return None
    model = speculative_config.get("model")
    if method is None and model is None:
        return None
    if not isinstance(model, str) or not model.strip():
        raise SpeculatorRuntimeError(
            "vLLM DFlash speculative_config requires a non-empty model"
        )
    revision = speculative_config.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise SpeculatorRuntimeError(
            "vLLM DFlash speculative_config revision must be a string"
        )
    contract = inspect_dflash2_checkpoint_if_present(model, revision=revision)
    if contract is None:
        return None
    num_speculative_tokens = speculative_config.get("num_speculative_tokens")
    validate_speculator_runtime_contract(
        speculator_type="dflash2",
        num_speculative_tokens=(
            num_speculative_tokens
            if isinstance(num_speculative_tokens, int)
            and not isinstance(num_speculative_tokens, bool)
            else None
        ),
    )
    return contract


def resolve_vllm_speculator_type(speculative_config: object) -> str | None:
    """Resolve DFlash2 from the loaded draft architecture used by vLLM."""
    if speculative_config is None:
        return None
    if isinstance(speculative_config, Mapping):
        method = speculative_config.get("method")
        draft_model_config = speculative_config.get("draft_model_config")
    else:
        method = getattr(speculative_config, "method", None)
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if not isinstance(method, str):
        return None
    hf_config = getattr(draft_model_config, "hf_config", None)
    if isinstance(draft_model_config, Mapping):
        hf_config = draft_model_config.get("hf_config")
    if isinstance(hf_config, Mapping):
        architectures = hf_config.get("architectures")
    else:
        architectures = getattr(hf_config, "architectures", None)
    if (
        method == "dflash"
        and isinstance(architectures, Sequence)
        and not isinstance(architectures, (str, bytes))
        and any(
            architecture in ("DFlash2DraftModel", "Qwen3DFlash2DraftModel")
            for architecture in architectures
        )
    ):
        return "dflash2"
    return method


def validate_vllm_refit_boundary(
    speculative_config: object,
    *,
    state_dict_names: Sequence[str],
) -> str | None:
    """Allow target refits while rejecting unsupported DFlash2 draft refits."""
    speculator_type = resolve_vllm_speculator_type(speculative_config)
    if speculator_type != "dflash2":
        return speculator_type
    if isinstance(speculative_config, Mapping):
        num_speculative_tokens = speculative_config.get("num_speculative_tokens")
    else:
        num_speculative_tokens = getattr(
            speculative_config, "num_speculative_tokens", None
        )
    validate_speculator_runtime_contract(
        speculator_type="dflash2",
        num_speculative_tokens=(
            num_speculative_tokens
            if isinstance(num_speculative_tokens, int)
            and not isinstance(num_speculative_tokens, bool)
            else None
        ),
    )
    if any(name.startswith("draft.") for name in state_dict_names):
        raise SpeculatorRuntimeError(
            "DFlash2 live refit is not implemented; refusing draft weight updates"
        )
    return speculator_type


@dataclass(frozen=True, slots=True)
class WeightComponentManifest:
    """Ordered transport and loading contract for one model component."""

    component: Literal["target", "draft"]
    ordered_names: tuple[str, ...]
    byte_count: int
    owner_ranks: tuple[int, ...]
    loader: str
    finalizer: str


@dataclass(frozen=True, slots=True)
class ModelUpdateManifest:
    """Transport-neutral target and optional draft update description."""

    target: WeightComponentManifest
    draft: WeightComponentManifest | None

    @classmethod
    def from_state_dict_info(
        cls,
        state_dict_info: Mapping[str, tuple[Sequence[int], torch.dtype]],
        *,
        target_owner_ranks: tuple[int, ...],
        draft_owner_ranks: tuple[int, ...],
    ) -> ModelUpdateManifest:
        target_names = tuple(
            name for name in state_dict_info if not name.startswith("draft.")
        )
        draft_names = tuple(
            name for name in state_dict_info if name.startswith("draft.")
        )

        def byte_count(names: tuple[str, ...]) -> int:
            total = 0
            for name in names:
                shape, dtype = state_dict_info[name]
                total += int(torch.Size(shape).numel()) * dtype.itemsize
            return total

        target = WeightComponentManifest(
            component="target",
            ordered_names=target_names,
            byte_count=byte_count(target_names),
            owner_ranks=target_owner_ranks,
            loader="target.load_weights",
            finalizer="process_weights_after_loading",
        )
        draft = None
        if draft_names:
            draft = WeightComponentManifest(
                component="draft",
                ordered_names=draft_names,
                byte_count=byte_count(draft_names),
                owner_ranks=draft_owner_ranks,
                loader="draft.load_weights",
                finalizer="process_weights_after_loading",
            )
        return cls(target=target, draft=draft)


class ModelUpdateCoverage:
    """Validate exact input coverage while allowing explicit non-owner skips."""

    def __init__(self, manifest: ModelUpdateManifest, *, rank: int) -> None:
        self._manifest = manifest
        self._rank = rank
        self._expected = set(manifest.target.ordered_names)
        if manifest.draft is not None:
            self._expected.update(manifest.draft.ordered_names)
        self._covered: set[str] = set()

    @property
    def has_draft(self) -> bool:
        return self._manifest.draft is not None

    def _record(self, names: Sequence[str]) -> None:
        incoming: set[str] = set()
        duplicate: set[str] = set()
        for name in names:
            if name in incoming:
                duplicate.add(name)
            incoming.add(name)
        duplicate.update(incoming & self._covered)
        unexpected = incoming - self._expected
        if duplicate:
            raise SpeculatorRuntimeError(
                f"duplicate keys ({len(duplicate)}): {sorted(duplicate)[:8]}"
            )
        if unexpected:
            raise SpeculatorRuntimeError(
                f"unexpected keys ({len(unexpected)}): {sorted(unexpected)[:8]}"
            )
        self._covered.update(incoming)

    def record_loaded(self, names: Sequence[str]) -> None:
        self._record(names)

    def record_owner_skip(
        self, names: Sequence[str], *, component: Literal["target", "draft"]
    ) -> None:
        manifest = self._manifest.target
        if component == "draft":
            manifest = self._manifest.draft
        if manifest is None or self._rank in manifest.owner_ranks:
            raise SpeculatorRuntimeError(
                f"rank {self._rank} cannot skip owned {component} weights"
            )
        allowed = set(manifest.ordered_names)
        invalid = set(names) - allowed
        if invalid:
            raise SpeculatorRuntimeError(
                f"unexpected {component} owner skips: {sorted(invalid)[:8]}"
            )
        self._record(names)

    def require_complete(self) -> None:
        missing = self._expected - self._covered
        if missing:
            raise SpeculatorRuntimeError(
                f"missing keys ({len(missing)}): {sorted(missing)[:8]}"
            )


@dataclass(slots=True)
class DraftRuntimeAdapter:
    """Capability-driven access to a live vLLM speculative model."""

    speculator_type: SpeculatorType
    vllm_version: str
    runner_family: RunnerFamily
    model: Any | None
    pp_rank: int
    pp_size: int
    is_owner: bool

    @classmethod
    def resolve(
        cls,
        model_runner: Any,
        *,
        speculator_type: str,
        vllm_version: str,
        num_speculative_tokens: int | None = None,
        pp_rank: int,
        pp_size: int,
    ) -> DraftRuntimeAdapter:
        resolved_speculator_type = validate_speculator_runtime_contract(
            speculator_type=speculator_type,
            num_speculative_tokens=num_speculative_tokens,
        )
        if pp_size <= 0 or not 0 <= pp_rank < pp_size:
            raise SpeculatorRuntimeError(
                f"invalid pipeline rank {pp_rank} for PP={pp_size}"
            )
        if resolved_speculator_type == "dflash2":
            raise SpeculatorRuntimeError(
                f"speculator_type=dflash2, vLLM={vllm_version}: DFlash2 is "
                "recognized, but live refit is not implemented"
            )

        accessor = getattr(model_runner, "get_draft_model", None)
        if callable(accessor):
            family = RunnerFamily.ACCESSOR
            model = accessor()
        elif hasattr(model_runner, "drafter"):
            family = RunnerFamily.DRAFTER
            model = getattr(getattr(model_runner, "drafter"), "model", None)
        elif hasattr(model_runner, "speculator"):
            family = RunnerFamily.SPECULATOR
            model = getattr(getattr(model_runner, "speculator"), "model", None)
        else:
            raise SpeculatorRuntimeError(
                f"speculator_type={speculator_type}, vLLM={vllm_version}: no "
                "get_draft_model, drafter.model, or speculator.model capability"
            )

        if resolved_speculator_type in ("dflash", "dspark") and pp_size != 1:
            raise SpeculatorRuntimeError(
                f"speculator_type={speculator_type}, vLLM={vllm_version}, "
                f"runner_family={family.value}: PP={pp_size} is not supported"
            )

        is_owner = pp_rank == pp_size - 1
        if is_owner and model is None:
            raise SpeculatorRuntimeError(
                f"speculator_type={speculator_type}, vLLM={vllm_version}, "
                f"runner_family={family.value}: owner draft model is unavailable"
            )
        return cls(
            speculator_type=resolved_speculator_type,
            vllm_version=vllm_version,
            runner_family=family,
            model=model,
            pp_rank=pp_rank,
            pp_size=pp_size,
            is_owner=is_owner,
        )

    @property
    def owner_ranks(self) -> tuple[int, ...]:
        return (self.pp_size - 1,)
