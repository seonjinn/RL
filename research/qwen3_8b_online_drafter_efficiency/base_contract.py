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
"""Record and validate the terminal-GREEN base for online drafter efficiency work."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Literal, cast


@dataclass(frozen=True, slots=True)
class GateReceipt:
    """Durable evidence for a passing gate run."""

    job_id: int
    head: str
    result: Literal["PASS"]
    result_path: str


@dataclass(frozen=True, slots=True)
class BaseContract:
    """The product revision and green gate receipts that define the baseline."""

    product_head: str
    container_sha256: str
    full_gate: GateReceipt
    packed_e2e: GateReceipt


def _expect_object(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must have string keys")
    return cast(dict[str, object], value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _expect_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _load_gate_receipt(value: object, *, field_name: str) -> GateReceipt:
    payload = _expect_object(value, field_name=field_name)
    result = _expect_string(payload.get("result"), field_name=f"{field_name}.result")
    if result != "PASS":
        raise ValueError(f"{field_name}.result must be PASS")
    return GateReceipt(
        job_id=_expect_int(payload.get("job_id"), field_name=f"{field_name}.job_id"),
        head=_expect_string(payload.get("head"), field_name=f"{field_name}.head"),
        result=cast(Literal["PASS"], result),
        result_path=_expect_string(
            payload.get("result_path"), field_name=f"{field_name}.result_path"
        ),
    )


def _validate_receipt(
    receipt: GateReceipt, *, product_head: str, field_name: str
) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", receipt.head):
        raise ValueError(f"{field_name}.head must be a 40-character lowercase SHA")
    if receipt.result != "PASS":
        raise ValueError(f"{field_name}.result must be PASS")
    if not receipt.result_path.strip() or not receipt.result_path.startswith("/"):
        raise ValueError(f"{field_name}.result_path must be a nonempty durable path")
    if receipt.head != product_head:
        raise ValueError(f"{field_name}.head must equal product_head")


def _validate_contract(contract: BaseContract) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", contract.product_head):
        raise ValueError("product_head must be a 40-character lowercase SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", contract.container_sha256):
        raise ValueError("container_sha256 must be a 64-character lowercase SHA256")
    _validate_receipt(
        contract.full_gate, product_head=contract.product_head, field_name="full_gate"
    )
    _validate_receipt(
        contract.packed_e2e, product_head=contract.product_head, field_name="packed_e2e"
    )


def load_base_contract(path: Path) -> BaseContract:
    """Load and validate a base contract from ``path``."""
    payload = _expect_object(json.loads(path.read_text()), field_name="base contract")
    contract = BaseContract(
        product_head=_expect_string(
            payload.get("product_head"), field_name="product_head"
        ),
        container_sha256=_expect_string(
            payload.get("container_sha256"), field_name="container_sha256"
        ),
        full_gate=_load_gate_receipt(payload.get("full_gate"), field_name="full_gate"),
        packed_e2e=_load_gate_receipt(
            payload.get("packed_e2e"), field_name="packed_e2e"
        ),
    )
    _validate_contract(contract)
    return contract


def assert_terminal_green_base(
    contract: BaseContract,
    *,
    current_head: str,
) -> None:
    """Raise unless the recorded terminal-GREEN base matches ``current_head``."""
    _validate_contract(contract)
    if current_head != contract.product_head:
        raise RuntimeError(
            f"base head drift: current={current_head} recorded={contract.product_head}"
        )


def _parse_receipt(receipt_json: str, *, field_name: str) -> GateReceipt:
    return _load_gate_receipt(json.loads(receipt_json), field_name=field_name)


def _record_base_contract(args: argparse.Namespace) -> None:
    full_gate = _parse_receipt(args.full_gate_receipt, field_name="full_gate")
    contract = BaseContract(
        product_head=full_gate.head,
        container_sha256=args.container_sha256,
        full_gate=full_gate,
        packed_e2e=_parse_receipt(args.packed_e2e_receipt, field_name="packed_e2e"),
    )
    _validate_contract(contract)
    args.output.write_text(json.dumps(asdict(contract), indent=2) + "\n")


def _validate_base_contract(args: argparse.Namespace) -> None:
    assert_terminal_green_base(
        load_base_contract(args.contract), current_head=args.current_head
    )
    print("base_contract=PASS")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--full-gate-receipt", required=True)
    record_parser.add_argument("--packed-e2e-receipt", required=True)
    record_parser.add_argument("--container-sha256", required=True)
    record_parser.add_argument("--output", required=True, type=Path)
    record_parser.set_defaults(handler=_record_base_contract)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--contract", required=True, type=Path)
    validate_parser.add_argument("--current-head", required=True)
    validate_parser.set_defaults(handler=_validate_base_contract)
    return parser


def main() -> None:
    """Run the base-contract command-line interface."""
    args = _build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
