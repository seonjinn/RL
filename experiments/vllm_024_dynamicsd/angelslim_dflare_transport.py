from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CompactResponse:
    time_per_output_token: float
    acceptance_lengths: list[int]
    num_input_tokens: int
    num_output_tokens: int


def _shape_num_tokens(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("output_ids must expose a 2D shape")
    return int(shape[1])


def _compact_response(response: Any) -> CompactResponse:
    num_input_tokens = int(getattr(response, "num_input_tokens"))
    total_tokens = _shape_num_tokens(getattr(response, "output_ids"))
    return CompactResponse(
        time_per_output_token=float(getattr(response, "time_per_output_token")),
        acceptance_lengths=[
            int(length) for length in getattr(response, "acceptance_lengths")
        ],
        num_input_tokens=num_input_tokens,
        num_output_tokens=total_tokens - num_input_tokens,
    )


def compact_response_map(responses: Mapping[int, Any]) -> dict[int, CompactResponse]:
    return {
        int(block_size): _compact_response(response)
        for block_size, response in responses.items()
    }


def _json_ready_responses(
    responses: Sequence[Mapping[int, CompactResponse]],
) -> list[dict[str, dict[str, int | float | list[int]]]]:
    return [
        {
            str(int(block_size)): asdict(response)
            for block_size, response in response_map.items()
        }
        for response_map in responses
    ]


def write_rank_partial(
    path: Path, rank: int, responses: Sequence[Mapping[int, CompactResponse]]
) -> Path:
    partial_path = path.with_name(f"{path.name}.rank{int(rank)}.partial.json")
    temporary_path = partial_path.with_name(f"{partial_path.name}.tmp")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(
        json.dumps(_json_ready_responses(responses), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(partial_path)
    return partial_path
