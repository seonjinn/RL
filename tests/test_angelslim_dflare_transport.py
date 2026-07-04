from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path

import pytest

from experiments.vllm_024_dynamicsd.angelslim_dflare_transport import (
    CompactResponse,
    compact_response_map,
    write_rank_partial,
)


class FakeCudaTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeResponse:
    def __init__(
        self,
        *,
        time_per_output_token: float,
        acceptance_lengths: list[int],
        num_input_tokens: int,
        output_shape: tuple[int, int],
    ) -> None:
        self.time_per_output_token = time_per_output_token
        self.acceptance_lengths = acceptance_lengths
        self.num_input_tokens = num_input_tokens
        self.output_ids = FakeCudaTensor(output_shape)


def test_compact_response_map_retains_metrics_and_drops_output_ids() -> None:
    responses = {
        1: FakeResponse(
            time_per_output_token=0.25,
            acceptance_lengths=[1, 1, 1],
            num_input_tokens=3,
            output_shape=(1, 7),
        ),
        4: FakeResponse(
            time_per_output_token=0.125,
            acceptance_lengths=[4, 3, 2],
            num_input_tokens=3,
            output_shape=(1, 9),
        ),
    }

    compact = compact_response_map(responses)

    assert compact == {
        1: CompactResponse(
            time_per_output_token=0.25,
            acceptance_lengths=[1, 1, 1],
            num_input_tokens=3,
            num_output_tokens=4,
        ),
        4: CompactResponse(
            time_per_output_token=0.125,
            acceptance_lengths=[4, 3, 2],
            num_input_tokens=3,
            num_output_tokens=6,
        ),
    }
    with pytest.raises(FrozenInstanceError):
        compact[1].num_output_tokens = 99

    payload = {block_size: asdict(response) for block_size, response in compact.items()}
    serialized = json.dumps(payload, sort_keys=True)

    assert '"output_ids"' not in serialized
    assert '"num_output_tokens": 4' in serialized
    assert '"time_per_output_token": 0.125' in serialized


def test_write_rank_partial_creates_expected_json_file(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    responses = [
        {
            1: CompactResponse(
                time_per_output_token=0.25,
                acceptance_lengths=[1, 1],
                num_input_tokens=3,
                num_output_tokens=4,
            )
        }
    ]

    partial_path = write_rank_partial(result_path, rank=2, responses=responses)

    assert partial_path == tmp_path / "result.json.rank2.partial.json"
    assert partial_path.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [partial_path.name]
    assert json.loads(partial_path.read_text(encoding="utf-8")) == [
        {
            "1": {
                "acceptance_lengths": [1, 1],
                "num_input_tokens": 3,
                "num_output_tokens": 4,
                "time_per_output_token": 0.25,
            }
        }
    ]
