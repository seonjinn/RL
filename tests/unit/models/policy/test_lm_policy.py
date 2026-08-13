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

from typing import Any

import pytest
import torch

from nemo_rl.models.policy.lm_policy import Policy


class _WorkerGroup:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def run_all_workers_sharded_data(self, *args: Any, **kwargs: Any) -> object:
        return object()

    def get_all_worker_results(self, futures: object) -> list[dict[str, Any]]:
        return self._results

    def shutdown(self, cleanup_method: str) -> bool:
        return True


def test_policy_train_preserves_full_cuda_graph_worker_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = {
        "full_cuda_graph_warmup_calls": 3,
        "full_cuda_graph_capture_calls": 1,
        "full_cuda_graph_replay_calls": 3,
        "full_cuda_graph_reset_calls": 0,
        "full_cuda_graph_storage_signature_sha256": "a" * 64,
        "full_cuda_graph_validation_warmup_calls": 3,
        "full_cuda_graph_validation_capture_calls": 1,
        "full_cuda_graph_validation_replay_calls": 3,
        "full_cuda_graph_validation_reset_calls": 0,
    }
    worker_result = {
        "global_loss": torch.tensor(0.25),
        "grad_norm": torch.tensor([1.5]),
        "all_mb_metrics": {"num_valid_samples": [2]},
        **evidence,
    }
    policy: Any = Policy.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 2, "train_micro_batch_size": 1}
    policy.flops_tracker = None
    policy.worker_group = _WorkerGroup([worker_result])
    monkeypatch.setattr(policy, "_shard_for_train", lambda data, batch_size: [data])
    monkeypatch.setattr(policy, "_report_sharded_payload", lambda data, label: None)

    result = policy.train(data=object(), loss_fn=object())

    assert {key: result[key] for key in evidence} == evidence
