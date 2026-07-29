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
import sys
from fcntl import LOCK_EX, LOCK_UN, flock
from importlib.util import find_spec


_TE_UTILS_PATH = "pytorch/utils.py"
_TE_WEAK_REF_FLOAT64_ANCHOR = """\
    torch.float32: "<f4",
"""
_TE_WEAK_REF_FLOAT64_PATCHED = """\
    torch.float32: "<f4",
    torch.float64: "<f8",
"""
_TE_WEAK_REF_FLOAT64_MARKER = 'torch.float64: "<f8",'
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


def _update_loaded_transformer_engine_weak_ref_float64_mapping() -> None:
    """Update an already-imported TE utils module without reloading its classes."""
    utils_module = sys.modules.get("transformer_engine.pytorch.utils")
    torch_module = sys.modules.get("torch")
    if utils_module is None or torch_module is None:
        return

    dtype_mapping = getattr(utils_module, "_torch_dtype_to_np_typestr_dict", None)
    if isinstance(dtype_mapping, dict):
        dtype_mapping[torch_module.float64] = "<f8"


def _patch_transformer_engine_weak_ref_float64(*, required: bool) -> None:
    """Allow TE CUDA Graph weak references to preserve float64 router outputs."""
    utils_file = _get_transformer_engine_file(_TE_UTILS_PATH)
    lock_file_path = f"{utils_file}.nemo_rl_patch.lock"
    patched = False
    with open(lock_file_path, "a+") as lock_file:
        flock(lock_file.fileno(), LOCK_EX)
        try:
            with open(utils_file) as source_file:
                content = source_file.read()
            if _TE_WEAK_REF_FLOAT64_MARKER not in content:
                if _TE_WEAK_REF_FLOAT64_ANCHOR not in content:
                    message = (
                        "Transformer Engine CUDA Graph training with an fp64 MoE "
                        "router found an unsupported Transformer Engine utils.py."
                    )
                    if required:
                        raise RuntimeError(message)
                    print(message)
                    return

                patched_content = content.replace(
                    _TE_WEAK_REF_FLOAT64_ANCHOR,
                    _TE_WEAK_REF_FLOAT64_PATCHED,
                    1,
                )
                with open(utils_file, "w") as source_file:
                    source_file.write(patched_content)
                    source_file.flush()
                patched = True
        finally:
            flock(lock_file.fileno(), LOCK_UN)

    _update_loaded_transformer_engine_weak_ref_float64_mapping()
    if patched:
        print(
            "Applied Transformer Engine float64 CUDA Graph weak-reference "
            f"support to {utils_file}."
        )


def apply_transformer_engine_weak_ref_float64_patch(*, required: bool = False) -> None:
    """Patch TE weak references, failing closed for fp64 router graph requests."""
    try:
        _patch_transformer_engine_weak_ref_float64(required=required)
    except (OSError, RuntimeError) as error:
        if required:
            raise RuntimeError(
                "Failed to prepare Transformer Engine float64 weak-reference "
                "support for MoE router CUDA Graph training."
            ) from error
        print(
            "Error checking/patching Transformer Engine float64 weak-reference "
            f"support: {error}"
        )


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
