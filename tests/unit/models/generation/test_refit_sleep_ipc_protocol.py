# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real ZMQ drain/ACK regression using the production receive loop on CPU.

AST isolation avoids importing the Linux-only vLLM extension on macOS. The
manifest and receive loop are compiled verbatim; only CUDA rebuild/loading
and finalization are replaced. This is not evidence of a GPU refit.
"""

import ast
import gc
import logging
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import zmq

from nemo_rl.models.policy.utils import IPCProtocol, calculate_aligned_size
from tests.functional.refit_sleep_utils import incomplete_receiver_manifest


def _cpu_receive_extension() -> Any:
    path = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/vllm/vllm_backend.py"
    )
    source = ast.parse(path.read_text())
    body = [
        node
        for node in source.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name
        in {
            "_format_refit_key_error",
            "IPCWeightManifestError",
            "_IPCWeightManifest",
        }
    ]
    extension = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "VllmInternalWorkerExtension"
    )
    methods = [
        node
        for node in extension.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_reset_refit_runtime_coverage",
            "update_weights_via_ipc_zmq",
        }
    ]
    assert len(body) == 3 and len(methods) == 2
    for method in methods:
        method.decorator_list = []
    body.append(
        ast.ClassDef(
            name="CPUReceiver", bases=[], keywords=[], body=methods, decorator_list=[]
        )
    )
    namespace = {
        "Iterable": Iterable,
        "Sequence": Sequence,
        "torch": torch,
        "gc": gc,
        "logger": logging.getLogger(__name__),
        "IPCProtocol": IPCProtocol,
        "calculate_aligned_size": calculate_aligned_size,
        "rebuild_cuda_tensor_from_ipc": lambda handle, index: handle,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["CPUReceiver"]()


def test_missing_manifest_drains_data_and_complete_ack_on_both_sides() -> None:
    receiver = _cpu_receive_extension()
    receiver._nrl_refit_reconstructs_all_runtime_weights = True
    receiver.state_dict_info = incomplete_receiver_manifest(
        {"model.weight": (torch.Size([1]), torch.float32)}
    )
    receiver.device = SimpleNamespace(index=0)
    receiver.maybe_init_zmq = lambda: None
    receiver._synchronize_before_ipc_data_ack = lambda: None
    receiver._weight_update_errors_are_fatal = lambda: False
    loaded = []

    def load(weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        loaded.extend(name for name, _ in weights)
        return set(loaded)

    @contextmanager
    def lifecycle(transport: str) -> Iterator[Any]:
        assert transport == "ipc"
        yield lambda: pytest.fail("incomplete source manifest must not finalize")

    receiver._load_weights = load
    receiver._weight_update_lifecycle = lifecycle
    used_bytes = calculate_aligned_size(torch.float32.itemsize)
    payload = torch.zeros(used_bytes, dtype=torch.uint8)
    with zmq.Context() as context:
        receiver.zmq_socket = context.socket(zmq.REP)
        receiver.zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)
        receiver.zmq_socket.setsockopt(zmq.SNDTIMEO, 1000)
        receiver.zmq_socket.setsockopt(zmq.LINGER, 0)
        receiver.zmq_socket.bind("inproc://task6-manifest")

        def sender() -> list[bytes]:
            with context.socket(zmq.REQ) as socket:
                socket.setsockopt(zmq.RCVTIMEO, 1000)
                socket.setsockopt(zmq.SNDTIMEO, 1000)
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect("inproc://task6-manifest")
                replies = []
                # The real sender keeps streaming after each ACK and only
                # releases the transaction after the final COMPLETE ACK.
                socket.send_pyobj((payload, ["model.weight"], used_bytes))
                replies.append(socket.recv())
                socket.send_pyobj(IPCProtocol.COMPLETE)
                replies.append(socket.recv())
                return replies

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                sender_result = pool.submit(sender)
                assert receiver.update_weights_via_ipc_zmq() is False
                assert sender_result.result(timeout=3) == [b"ack", b"ack"]
            assert loaded == ["model.weight"]
            assert receiver._nrl_refit_reconstructs_all_runtime_weights is False
        finally:
            receiver.zmq_socket.close(linger=0)
