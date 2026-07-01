# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import os
import stat
import tempfile
from pathlib import Path


_PATCH_MARKER = "NRL_DRAFT_MODEL_CUDAGRAPH_INIT_PATCH"

_VLLM_020_DRAFT_INIT = (
    "        # Initialize drafter's cudagraph dispatcher if using spec decode.\n"
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

_VLLM_020_DRAFT_INIT_WITH_DRAFT_MODEL = (
    "        # Initialize drafter's cudagraph dispatcher if using spec decode.\n"
    f"        # {_PATCH_MARKER}: include generic draft-model proposers.\n"
    "        if self.speculative_config and (\n"
    "            self.speculative_config.use_eagle()\n"
    "            or self.speculative_config.use_dflash()\n"
    "            or self.speculative_config.uses_draft_model()\n"
    "            or self.speculative_config.uses_extract_hidden_states()\n"
    "        ):\n"
    "            assert isinstance(\n"
    "                self.drafter,\n"
    "                EagleProposer\n"
    "                | DFlashProposer\n"
    "                | DraftModelProposer\n"
    "                | ExtractHiddenStatesProposer,\n"
    "            )\n"
    "            self.drafter.initialize_cudagraph_keys(cudagraph_mode)\n"
)


def atomic_replace_text(file_path: str | Path, new_content: str) -> None:
    """Atomically replace a text file while preserving its permission bits."""
    path = Path(file_path)
    file_mode = stat.S_IMODE(path.stat().st_mode)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        os.fchmod(temp_fd, file_mode)
        with os.fdopen(temp_fd, "w") as temp_file:
            temp_file.write(new_content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def requires_draft_model_cudagraph_support(
    method: str | None, enforce_eager: bool, has_draft_model: bool = False
) -> bool:
    """Return whether a generic PARD proposer needs CUDA Graph verification."""
    is_generic_draft = method in {"draft_model", "pard2"} or (
        method is None and has_draft_model
    )
    return not enforce_eager and is_generic_draft


def _get_draft_cudagraph_init_block(source: str) -> str | None:
    block_start = source.find(
        "# Initialize drafter's cudagraph dispatcher if using spec decode."
    )
    if block_start < 0:
        return None

    terminal = "self.drafter.initialize_cudagraph_keys(cudagraph_mode)"
    block_end = source.find(terminal, block_start)
    if block_end < 0:
        return None
    return source[block_start : block_end + len(terminal)]


def ensure_draft_model_cudagraph_support(source: str) -> tuple[str, bool]:
    """Return vLLM source with verified generic draft-model CUDA Graph setup.

    Unknown source layouts fail closed. A performance run must not silently
    continue with a PARD drafter outside the captured CUDA Graph path.
    """
    init_block = _get_draft_cudagraph_init_block(source)
    executable_block = (
        "\n".join(line.split("#", 1)[0] for line in init_block.splitlines())
        if init_block is not None
        else ""
    )
    native_support = init_block is not None and all(
        token in executable_block
        for token in (
            "self.speculative_config.uses_draft_model()",
            "DraftModelProposer",
            "self.drafter.initialize_cudagraph_keys(cudagraph_mode)",
        )
    )
    if native_support:
        return source, False

    if _PATCH_MARKER in source:
        raise RuntimeError(
            "The vLLM draft-model CUDA Graph patch marker exists, but the "
            "required DraftModelProposer initialization is incomplete."
        )

    if _VLLM_020_DRAFT_INIT not in source:
        raise RuntimeError(
            "Cannot verify generic draft-model CUDA Graph initialization in "
            "this vLLM source version. Refusing to run PARD with a possible "
            "eager drafter fallback."
        )

    return source.replace(
        _VLLM_020_DRAFT_INIT,
        _VLLM_020_DRAFT_INIT_WITH_DRAFT_MODEL,
        1,
    ), True
