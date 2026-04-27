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
import threading
from importlib.util import find_spec


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


def _patch_transformer_engine_permutation() -> None:
    perm_file = _get_transformer_engine_file("pytorch/triton/permutation.py")

    with open(perm_file, "r") as f:
        content = f.read()

    if "get_int_dtype = triton.constexpr_function(get_int_dtype)" in content:
        return

    print(f"Applying Triton fix to {perm_file}...")

    old_usage = "idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"
    new_usage = (
        "idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"
    )
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
            new_content = temp_content.replace(jit_anchor, new_definition + jit_anchor, 1)

    if new_content:
        with open(perm_file, "w") as f:
            f.write(new_content)
        print("Successfully patched transformer_engine permutation.py.")


def _patch_transformer_engine_graph_nvtx() -> None:
    graph_file = _get_transformer_engine_file("pytorch/graph.py")

    with open(graph_file, "r") as f:
        content = f.read()

    marker = 'torch.cuda.nvtx.range_push("te_graph/fwd_graph_replay")'
    if marker in content:
        return

    replacements = [
        (
            """                if cuda_graph_stream != torch.cuda.current_stream():
                    if need_copy_idx:
                        cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                    with cuda_graph_stream:
                        for i in need_copy_idx:
                            static_input_surface[i].copy_(inputs[i])
                        fwd_graph.replay()
                    if cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(cuda_graph_stream)
                else:
                    for i in need_copy_idx:
                        static_input_surface[i].copy_(inputs[i])
                    fwd_graph.replay()
""",
            """                if cuda_graph_stream != torch.cuda.current_stream():
                    if need_copy_idx:
                        torch.cuda.nvtx.range_push("te_graph/fwd_wait_for_inputs")
                        cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                        torch.cuda.nvtx.range_pop()
                    with cuda_graph_stream:
                        if need_copy_idx:
                            torch.cuda.nvtx.range_push("te_graph/fwd_static_input_copy")
                            for i in need_copy_idx:
                                static_input_surface[i].copy_(inputs[i])
                            torch.cuda.nvtx.range_pop()
                        torch.cuda.nvtx.range_push("te_graph/fwd_graph_replay")
                        fwd_graph.replay()
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("te_graph/fwd_wait_for_completion")
                    if cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(cuda_graph_stream)
                    torch.cuda.nvtx.range_pop()
                else:
                    if need_copy_idx:
                        torch.cuda.nvtx.range_push("te_graph/fwd_static_input_copy")
                        for i in need_copy_idx:
                            static_input_surface[i].copy_(inputs[i])
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("te_graph/fwd_graph_replay")
                    fwd_graph.replay()
                    torch.cuda.nvtx.range_pop()
""",
        ),
        (
            """                if ctx.cuda_graph_stream != torch.cuda.current_stream():
                    if grad_copy_pairs:
                        ctx.cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                    with ctx.cuda_graph_stream:
                        for g, grad in grad_copy_pairs:
                            g.copy_(grad)
                        bwd_graph.replay()
                    if ctx.cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(ctx.cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(ctx.cuda_graph_stream)
                else:
                    for g, grad in grad_copy_pairs:
                        g.copy_(grad)
                    bwd_graph.replay()
""",
            """                if ctx.cuda_graph_stream != torch.cuda.current_stream():
                    if grad_copy_pairs:
                        torch.cuda.nvtx.range_push("te_graph/bwd_wait_for_grads")
                        ctx.cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                        torch.cuda.nvtx.range_pop()
                    with ctx.cuda_graph_stream:
                        if grad_copy_pairs:
                            torch.cuda.nvtx.range_push("te_graph/bwd_static_grad_copy")
                            for g, grad in grad_copy_pairs:
                                g.copy_(grad)
                            torch.cuda.nvtx.range_pop()
                        torch.cuda.nvtx.range_push("te_graph/bwd_graph_replay")
                        bwd_graph.replay()
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("te_graph/bwd_wait_for_completion")
                    if ctx.cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(ctx.cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(ctx.cuda_graph_stream)
                    torch.cuda.nvtx.range_pop()
                else:
                    if grad_copy_pairs:
                        torch.cuda.nvtx.range_push("te_graph/bwd_static_grad_copy")
                        for g, grad in grad_copy_pairs:
                            g.copy_(grad)
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("te_graph/bwd_graph_replay")
                    bwd_graph.replay()
                    torch.cuda.nvtx.range_pop()
""",
        ),
    ]

    new_content = content
    for old, new in replacements:
        if old not in new_content:
            raise RuntimeError(
                "Failed to locate expected Transformer Engine graph snippet for NVTX patch. "
                "This likely indicates a TE version mismatch."
            )
        new_content = new_content.replace(old, new, 1)

    with open(graph_file, "w") as f:
        f.write(new_content)
    print("Successfully patched transformer_engine graph.py with NVTX replay ranges.")


def _install_transformer_engine_graph_runtime_nvtx_patch() -> None:
    """Install a runtime monkey patch for TE graph replay internals.

    We patch PyTorch CUDA methods instead of editing TE source so the patch keeps
    working across minor TE version/layout changes inside the container. The wrappers
    only annotate calls whose stack originates from ``transformer_engine/pytorch/graph.py``.
    """
    if not os.environ.get("NRL_NSYS_WORKER_PATTERNS"):
        return

    import torch

    if getattr(torch.cuda.CUDAGraph.replay, "_nrl_te_nvtx_patched", False):
        return

    graph_file = _get_transformer_engine_file("pytorch/graph.py")
    thread_state = threading.local()
    thread_state.depth = 0

    def _find_te_graph_phase():
        frame = sys._getframe(1)
        while frame is not None:
            code = frame.f_code
            filename = code.co_filename
            if filename == graph_file:
                name = code.co_name
                if name == "forward":
                    return "fwd"
                if name == "backward":
                    return "bwd"
                return "generic"
            frame = frame.f_back
        return None

    def _wrap_nvtx(label, fn):
        def wrapped(*args, **kwargs):
            phase = _find_te_graph_phase()
            if phase is None:
                return fn(*args, **kwargs)

            depth = getattr(thread_state, "depth", 0)
            thread_state.depth = depth + 1
            try:
                torch.cuda.nvtx.range_push(f"te_graph/{phase}_{label}")
                try:
                    return fn(*args, **kwargs)
                finally:
                    torch.cuda.nvtx.range_pop()
            finally:
                thread_state.depth -= 1

        wrapped._nrl_te_nvtx_patched = True
        return wrapped

    torch.cuda.CUDAGraph.replay = _wrap_nvtx(
        "graph_replay", torch.cuda.CUDAGraph.replay
    )
    torch.cuda.Stream.wait_stream = _wrap_nvtx(
        "wait_stream", torch.cuda.Stream.wait_stream
    )
    torch.cuda.Stream.wait_event = _wrap_nvtx(
        "wait_event", torch.cuda.Stream.wait_event
    )
    torch.Tensor.copy_ = _wrap_nvtx("copy", torch.Tensor.copy_)
    print("Installed runtime NVTX monkey patch for transformer_engine graph replay.")


def apply_transformer_engine_patch():
    """Apply patch from https://github.com/NVIDIA/TransformerEngine/pull/2286/files.

    This locates the target file via importlib metadata instead of importing
    `transformer_engine`, to avoid side effects during initialization. If the
    permutation module has already been imported, it will be reloaded so that
    the patched source takes effect.
    """
    try:
        try:
            _patch_transformer_engine_permutation()
        except OSError as e:
            print(
                f"Could not write permutation patch to transformer_engine (permission denied?): {e}"
            )

        _install_transformer_engine_graph_runtime_nvtx_patch()

        # If the permutation module is already imported in this process,
        # reload it so that the patched source takes effect for subsequent use.
        import importlib
        import sys

        perm_module_name = "transformer_engine.pytorch.triton.permutation"
        if perm_module_name in sys.modules:
            importlib.reload(sys.modules[perm_module_name])
        graph_module_name = "transformer_engine.pytorch.graph"
        if graph_module_name in sys.modules:
            importlib.reload(sys.modules[graph_module_name])

    except Exception as e:
        print(f"Error checking/patching transformer_engine: {e}")
