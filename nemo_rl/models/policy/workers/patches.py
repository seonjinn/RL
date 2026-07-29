# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from fcntl import LOCK_EX, LOCK_UN, flock
from importlib.util import find_spec


_THD_CONTEXT_PARALLEL_PATH = (
    "pytorch/attention/dot_product_attention/context_parallel.py"
)
_THD_CONTEXT_PARALLEL_UNPATCHED = """\
        if ctx.qkv_format == "thd" and not ctx.use_fused_attention:
            dq[cu_seqlens_q_padded[-1] :].fill_(0)
            dk[cu_seqlens_kv_padded[-1] :].fill_(0)
            dv[cu_seqlens_kv_padded[-1] :].fill_(0)
"""
_THD_CONTEXT_PARALLEL_PATCHED = """\
        if (
            ctx.qkv_format == "thd"
            and not ctx.use_fused_attention
            and cu_seqlens_q_padded is not None
            and cu_seqlens_kv_padded is not None
        ):
            if is_graph_capturing():
                q_pad_mask = (
                    torch.arange(dq.shape[0], device=dq.device)
                    >= cu_seqlens_q_padded[-1]
                )
                kv_pad_mask = (
                    torch.arange(dk.shape[0], device=dk.device)
                    >= cu_seqlens_kv_padded[-1]
                )
                dq[q_pad_mask] = 0
                dk[kv_pad_mask] = 0
                dv[kv_pad_mask] = 0
            else:
                dq[cu_seqlens_q_padded[-1] :].fill_(0)
                dk[cu_seqlens_kv_padded[-1] :].fill_(0)
                dv[cu_seqlens_kv_padded[-1] :].fill_(0)
"""
_THD_CONTEXT_PARALLEL_PATCH_MARKER = "dq[q_pad_mask] = 0"


def _get_transformer_engine_file(relative_path: str) -> str:
    """Return absolute path to a Transformer Engine file or raise if it cannot be found.

    The relative_path should be a POSIX-style path under the transformer_engine
    package root, e.g. "pytorch/triton/permutation.py".
    """
    spec = find_spec("transformer_engine")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Transformer Engine package not found while attempting to patch "
            f"'{relative_path}'. Ensure `transformer-engine` is installed and "
            "available in this environment."
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))

    if not os.path.exists(file_path):
        raise RuntimeError(
            "Failed to locate expected Transformer Engine file to patch. "
            f"Looked for '{relative_path}' at '{file_path}'. "
            "This likely indicates an unexpected Transformer Engine installation "
            "layout or version mismatch."
        )

    return file_path


def _patch_thd_context_parallel_cuda_graph(*, required: bool) -> None:
    """Backport Transformer Engine's graph-safe THD CP gradient tail masking."""
    context_parallel_file = _get_transformer_engine_file(_THD_CONTEXT_PARALLEL_PATH)
    lock_file_path = f"{context_parallel_file}.nemo_rl_patch.lock"
    with open(lock_file_path, "a+") as lock_file:
        flock(lock_file.fileno(), LOCK_EX)
        try:
            with open(context_parallel_file) as source_file:
                content = source_file.read()
            if (
                _THD_CONTEXT_PARALLEL_UNPATCHED not in content
                and _THD_CONTEXT_PARALLEL_PATCH_MARKER in content
            ):
                return
            if _THD_CONTEXT_PARALLEL_UNPATCHED not in content:
                message = (
                    "Packed Transformer Engine CUDA Graph training found an "
                    "unsupported Transformer Engine context_parallel.py. "
                    "Upgrade to a build containing TransformerEngine #2898."
                )
                if required:
                    raise RuntimeError(message)
                print(message)
                return

            patched_content = content.replace(
                _THD_CONTEXT_PARALLEL_UNPATCHED,
                _THD_CONTEXT_PARALLEL_PATCHED,
                1,
            )
            with open(context_parallel_file, "w") as source_file:
                source_file.write(patched_content)
                source_file.flush()
        finally:
            flock(lock_file.fileno(), LOCK_UN)

    print(
        "Applied Transformer Engine #2898 THD context-parallel CUDA Graph "
        f"fix to {context_parallel_file}."
    )


def apply_transformer_engine_thd_context_parallel_patch(
    *, required: bool = False
) -> None:
    """Apply the THD CP graph fix, failing closed when a graph request needs it."""
    try:
        _patch_thd_context_parallel_cuda_graph(required=required)
    except (OSError, RuntimeError) as error:
        if required:
            raise RuntimeError(
                "Failed to prepare Transformer Engine for packed THD CUDA Graph "
                "training."
            ) from error
        print(
            f"Error checking/patching Transformer Engine THD context parallel: {error}"
        )


def apply_transformer_engine_patch() -> None:
    """Apply patch from https://github.com/NVIDIA/TransformerEngine/pull/2286/files.

    This locates the target file via importlib metadata instead of importing
    `transformer_engine`, to avoid side effects during initialization. If the
    permutation module has already been imported, it will be reloaded so that
    the patched source takes effect.
    """
    try:
        perm_file = _get_transformer_engine_file("pytorch/triton/permutation.py")

        with open(perm_file, "r") as f:
            content = f.read()

        if "get_int_dtype = triton.constexpr_function(get_int_dtype)" not in content:
            print(f"Applying Triton fix to {perm_file}...")

            # 1. Replace the usage
            old_usage = "idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"
            new_usage = "idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"

            # 2. Insert the definition before the first @triton.jit
            jit_anchor = "@triton.jit"

            new_definition = (
                "\n\n"
                "get_int_dtype = core.get_int_dtype\n"
                "get_int_dtype = triton.constexpr_function(get_int_dtype)\n"
            )

            new_content = None
            if old_usage in content:
                temp_content = content.replace(old_usage, new_usage)

                if jit_anchor in temp_content:
                    new_content = temp_content.replace(
                        jit_anchor, new_definition + jit_anchor, 1
                    )

            if new_content:
                try:
                    with open(perm_file, "w") as f:
                        f.write(new_content)
                    print("Successfully patched transformer_engine permutation.py.")
                except OSError as e:
                    print(
                        f"Could not write patch to transformer_engine (permission denied?): {e}"
                    )

        # If the permutation module is already imported in this process,
        # reload it so that the patched source takes effect for subsequent use.
        import importlib
        import sys

        perm_module_name = "transformer_engine.pytorch.triton.permutation"
        if perm_module_name in sys.modules:
            importlib.reload(sys.modules[perm_module_name])

    except Exception as e:
        print(f"Error checking/patching transformer_engine: {e}")
