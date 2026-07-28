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

from unittest.mock import MagicMock

import pytest


def test_te_cuda_graph_lifecycle_captures_after_three_successful_steps() -> None:
    from nemo_rl.models.megatron.cuda_graph_lifecycle import TECudaGraphLifecycle

    helper = MagicMock()
    helper.capture_finished.return_value = False
    helper.graphs_created.return_value = True
    lifecycle = TECudaGraphLifecycle(helper=helper, warmup_steps=3)

    for _ in range(3):
        assert lifecycle.ready_to_capture() is False
        assert lifecycle.capture_if_ready() is False
        lifecycle.record_successful_step()

    assert lifecycle.ready_to_capture() is True
    assert lifecycle.capture_if_ready() is True
    assert lifecycle.graphs_created() is True
    assert lifecycle.capture_if_ready() is False
    helper.create_cudagraphs.assert_called_once_with()


def test_te_cuda_graph_lifecycle_rejects_empty_capture() -> None:
    from nemo_rl.models.megatron.cuda_graph_lifecycle import TECudaGraphLifecycle

    helper = MagicMock()
    helper.capture_finished.return_value = False
    helper.graphs_created.return_value = False
    lifecycle = TECudaGraphLifecycle(helper=helper, warmup_steps=0)

    with pytest.raises(RuntimeError, match="no graphable layers"):
        lifecycle.capture_if_ready()


def test_te_cuda_graph_lifecycle_deletes_created_graphs_once() -> None:
    from nemo_rl.models.megatron.cuda_graph_lifecycle import TECudaGraphLifecycle

    helper = MagicMock()
    helper.graphs_created.return_value = True
    lifecycle = TECudaGraphLifecycle(helper=helper, warmup_steps=3)

    lifecycle.close()
    lifecycle.close()

    helper.delete_cuda_graphs.assert_called_once_with()
