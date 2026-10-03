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

"""CPU checks for refit diagnostic boundaries, without importing GPU packages."""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, Optional
from unittest.mock import patch


def load_stream_method() -> Callable[..., None]:
    repo = Path(os.environ.get("REPO", str(Path(__file__).resolve().parents[2])))
    source = repo / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
    tree = ast.parse(source.read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    method = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "stream_weights_via_ipc_zmq"
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"Optional": Optional}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[method.name]


class RefitMemoryLoggingTest(unittest.TestCase):
    def run_stream(self, *, fail: bool) -> list[str]:
        events: list[str] = []
        weight = object()
        worker = SimpleNamespace(
            maybe_init_zmq=lambda: events.append("init"),
            _log_gpu_mem=lambda tag: events.append(tag),
            _iter_params_with_optional_kv_scales=lambda **_: iter(
                [("linear.weight", weight)]
            ),
            zmq_socket=object(),
            rank=0,
            cfg={},
        )

        def stream(**kwargs: object) -> None:
            events.append("transfer")
            self.assertEqual(kwargs["buffer_size_bytes"], 4096)
            self.assertIs(kwargs["zmq_socket"], worker.zmq_socket)
            self.assertEqual(
                list(kwargs["params_generator"]), [("linear.weight", weight)]
            )
            if fail:
                raise RuntimeError("receiver failed")

        utils = ModuleType("nemo_rl.models.policy.utils")
        utils.stream_weights_via_ipc_zmq_impl = stream
        with patch.dict(sys.modules, {utils.__name__: utils}):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "receiver failed"):
                    load_stream_method()(worker, buffer_size_bytes=4096)
            else:
                load_stream_method()(worker, buffer_size_bytes=4096)
        return events

    def test_success_records_entry_and_exit_without_changing_payload(self) -> None:
        self.assertEqual(
            self.run_stream(fail=False),
            ["init", "refit_ipc_enter", "transfer", "refit_ipc_exit"],
        )

    def test_failure_records_exit_and_preserves_the_exception(self) -> None:
        self.assertEqual(
            self.run_stream(fail=True),
            ["init", "refit_ipc_enter", "transfer", "refit_ipc_exit"],
        )


if __name__ == "__main__":
    unittest.main()
