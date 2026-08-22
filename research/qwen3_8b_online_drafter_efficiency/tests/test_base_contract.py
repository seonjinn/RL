import json
from pathlib import Path

import pytest

from research.qwen3_8b_online_drafter_efficiency.base_contract import (
    assert_terminal_green_base,
    load_base_contract,
)


def test_base_requires_two_green_receipts_on_the_current_head(tmp_path: Path) -> None:
    head = "a" * 40
    path = tmp_path / "base_contract.json"
    path.write_text(
        json.dumps(
            {
                "product_head": head,
                "container_sha256": "b" * 64,
                "full_gate": {
                    "job_id": 1,
                    "head": head,
                    "result": "PASS",
                    "result_path": "/durable/full/result.json",
                },
                "packed_e2e": {
                    "job_id": 2,
                    "head": head,
                    "result": "PASS",
                    "result_path": "/durable/packed/result.json",
                },
            }
        )
    )
    contract = load_base_contract(path)
    assert_terminal_green_base(contract, current_head=head)
    with pytest.raises(RuntimeError, match="base head drift"):
        assert_terminal_green_base(contract, current_head="c" * 40)
