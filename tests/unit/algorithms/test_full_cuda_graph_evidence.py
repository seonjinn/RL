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

from pathlib import Path
from typing import Any

import pytest


_COUNTER_FIELDS = (
    "full_cuda_graph_warmup_calls",
    "full_cuda_graph_capture_calls",
    "full_cuda_graph_replay_calls",
    "full_cuda_graph_reset_calls",
)
_DIGEST_FIELD = "full_cuda_graph_storage_signature_sha256"


def _evidence(
    *,
    warmup: Any = 1,
    capture: Any = 1,
    replay: Any = 2,
    reset: Any = 0,
    digest: Any = "a" * 64,
) -> dict[str, Any]:
    return {
        _COUNTER_FIELDS[0]: warmup,
        _COUNTER_FIELDS[1]: capture,
        _COUNTER_FIELDS[2]: replay,
        _COUNTER_FIELDS[3]: reset,
        _DIGEST_FIELD: digest,
    }


def test_full_cuda_graph_evidence_tracker_preserves_exact_values_and_deltas() -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    tracker = FullCudaGraphEvidenceTracker()
    first_metrics: dict[str, Any] = {"loss": 0.5}
    first = _evidence()
    train_results = {
        **first,
        "full_cuda_graph_rank_local_digest": "must-not-leak",
    }

    tracker.preserve(train_results, first_metrics)

    assert {field: first_metrics[field] for field in first} == first
    assert "full_cuda_graph_rank_local_digest" not in first_metrics
    assert {
        f"{field}_delta": first_metrics[f"{field}_delta"] for field in _COUNTER_FIELDS
    } == {
        "full_cuda_graph_warmup_calls_delta": 1,
        "full_cuda_graph_capture_calls_delta": 1,
        "full_cuda_graph_replay_calls_delta": 2,
        "full_cuda_graph_reset_calls_delta": 0,
    }
    assert all(type(first_metrics[field]) is int for field in _COUNTER_FIELDS)
    assert type(first_metrics[_DIGEST_FIELD]) is str

    second_metrics: dict[str, Any] = {"loss": 0.4}
    second = _evidence(warmup=1, capture=1, replay=5, reset=0)
    tracker.preserve(second, second_metrics)

    assert {field: second_metrics[field] for field in second} == second
    assert {
        f"{field}_delta": second_metrics[f"{field}_delta"] for field in _COUNTER_FIELDS
    } == {
        "full_cuda_graph_warmup_calls_delta": 0,
        "full_cuda_graph_capture_calls_delta": 0,
        "full_cuda_graph_replay_calls_delta": 3,
        "full_cuda_graph_reset_calls_delta": 0,
    }


def test_full_cuda_graph_evidence_tracker_leaves_disabled_payload_unchanged() -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    tracker = FullCudaGraphEvidenceTracker()
    metrics = {"loss": 0.5, "logprob_tokens_per_second": 123.0}
    original = metrics.copy()

    tracker.preserve({"loss": 0.5}, metrics)

    assert metrics == original
    assert all("full_cuda_graph" not in field for field in metrics)


@pytest.mark.parametrize(
    "train_results",
    [
        {"full_cuda_graph_warmup_calls": 1},
        _evidence(capture=True),
        _evidence(replay=-1),
        _evidence(digest="A" * 64),
    ],
)
def test_full_cuda_graph_evidence_tracker_rejects_partial_or_malformed_values(
    train_results: dict[str, Any],
) -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    with pytest.raises(ValueError, match="full-iteration CUDA graph"):
        FullCudaGraphEvidenceTracker().preserve(train_results, {})


@pytest.mark.parametrize(
    "second",
    [
        _evidence(replay=1),
        _evidence(digest="b" * 64),
        {},
    ],
)
def test_full_cuda_graph_evidence_tracker_rejects_unstable_run_evidence(
    second: dict[str, Any],
) -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    tracker = FullCudaGraphEvidenceTracker()
    tracker.preserve(_evidence(), {})

    with pytest.raises(ValueError, match="full-iteration CUDA graph"):
        tracker.preserve(second, {})


def test_policy_train_reducers_wire_evidence_after_ordinary_metric_reduction() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    expected_calls = {
        "grpo.py": 2,
        "grpo_sync.py": 1,
        "sft.py": 1,
        "ppo.py": 1,
    }

    for filename, call_count in expected_calls.items():
        source = (repo_root / "nemo_rl" / "algorithms" / filename).read_text()
        assert source.count("FullCudaGraphEvidenceTracker()") == call_count
        assert source.count(
            "full_cuda_graph_evidence.preserve(train_results, metrics)"
        ) == (call_count)

        search_from = 0
        for _ in range(call_count):
            preserve_at = source.index(
                "full_cuda_graph_evidence.preserve(train_results, metrics)",
                search_from,
            )
            reduction_at = source.rfind(
                'metrics.update(train_results["all_mb_metrics"])',
                search_from,
                preserve_at,
            )
            assert reduction_at != -1
            search_from = preserve_at + 1

    combined_source = "".join(
        (repo_root / "nemo_rl" / "algorithms" / filename).read_text()
        for filename in expected_calls
    )
    assert "full_cuda_graph_logprob" not in combined_source
