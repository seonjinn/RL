# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve body-only drafter attention without executing checkpoint code."""

import logging
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, StrictBool

_Window = Annotated[int, Field(gt=0, strict=True)]


class _DFlashAttentionMetadata(BaseModel, extra="ignore"):
    use_swa: StrictBool | None = None
    swa_window_size: _Window | None = None
    causal: StrictBool | None = None


class _CheckpointAttentionMetadata(BaseModel, extra="ignore"):
    sliding_window: _Window | None = None
    use_sliding_window: StrictBool | None = None
    dflash_config: _DFlashAttentionMetadata | None = None

    def effective_window(self) -> int | None:
        draft = self.dflash_config
        if draft is not None and draft.causal is True:
            raise ValueError("causal draft blocks are not supported")
        if draft is not None and draft.use_swa is True:
            if self.use_sliding_window is False:
                raise ValueError("checkpoint sliding-window enable flags disagree")
            if draft.swa_window_size is None:
                raise ValueError("checkpoint use_swa requires swa_window_size")
            if (
                self.sliding_window is not None
                and self.sliding_window != draft.swa_window_size
            ):
                raise ValueError(
                    "checkpoint sliding_window and swa_window_size disagree"
                )
            return draft.swa_window_size
        if draft is not None and draft.use_swa is False:
            if self.use_sliding_window is True:
                raise ValueError("checkpoint sliding-window enable flags disagree")
            return None
        if self.use_sliding_window is False:
            return None
        if self.use_sliding_window is True and self.sliding_window is None:
            raise ValueError("checkpoint use_sliding_window requires sliding_window")
        return self.sliding_window


def _checkpoint_config_path(model_name: str, model_revision: str | None) -> Path | None:
    source = Path(model_name)
    if source.is_dir() or source.is_file():
        config_path = (source if source.is_dir() else source.parent) / "config.json"
        return config_path if config_path.is_file() else None
    if source.is_absolute() or model_name.startswith("."):
        raise FileNotFoundError(f"draft checkpoint path does not exist: {model_name}")

    # Hugging Face is optional for callers using only local checkpoint exports.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    try:
        return Path(hf_hub_download(model_name, "config.json", revision=model_revision))
    except RemoteEntryNotFoundError as error:
        if error.response is None or error.response.status_code != 404:
            raise
        # Legacy weight-only repositories have no attention metadata.
        return None


def resolve_draft_sliding_window(
    *,
    model_name: str | None,
    model_revision: str | None,
    sliding_window: int | None,
    block_size: int,
) -> int | None:
    """Inherit checkpoint attention, rejecting explicit incompatible overrides.

    None means inherit from config.json, or full context when metadata is absent.
    A positive explicit window is allowed for random/weight-only initialization;
    when checkpoint metadata exists it must agree. Authentication, network and
    malformed-config errors are not converted into a full-context fallback.
    """
    window = sliding_window
    if model_name is not None:
        config_path = _checkpoint_config_path(model_name, model_revision)
        if config_path is not None:
            metadata = _CheckpointAttentionMetadata.model_validate_json(
                config_path.read_text()
            )
            window = metadata.effective_window()
            if sliding_window is not None and sliding_window != window:
                raise ValueError(
                    f"draft sliding_window={sliding_window} disagrees with checkpoint "
                    f"attention window={window}: {config_path}"
                )
    if window is not None and (type(window) is not int or window < block_size):
        raise ValueError(
            "checkpoint sliding_window must be an integer >= draft block_size"
        )
    logging.getLogger(__name__).info(
        "[draft] attention resolved: model=%s sliding_window=%s training_block_size=%d",
        model_name,
        window,
        block_size,
    )
    return window
