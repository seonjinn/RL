#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfileArm:
    config_name: str
    update_probe_enabled: bool


_ARMS = {
    "fixed-control": ProfileArm(
        config_name="qwen3_8b_dflash_fixed_dense_control",
        update_probe_enabled=False,
    ),
    "online-current": ProfileArm(
        config_name="qwen3_8b_dflash_online_cp1",
        update_probe_enabled=True,
    ),
    "online-probe-off": ProfileArm(
        config_name="qwen3_8b_dflash_online_cp1",
        update_probe_enabled=False,
    ),
}


def resolve_arm(name: str) -> ProfileArm:
    try:
        return _ARMS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_ARMS))
        raise ValueError(
            f"Unsupported profile arm {name!r}; expected one of: {choices}"
        ) from error


def validate_profile_receipt(result_dir: Path) -> list[Path]:
    reports = sorted(result_dir.glob("policy-rank-*.nsys-rep"))
    if not reports:
        raise RuntimeError(f"no Nsight reports found in {result_dir}")
    empty = [report for report in reports if report.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"empty Nsight reports: {empty}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(_ARMS))
    parser.add_argument("--validate-result-dir", type=Path)
    args = parser.parse_args()

    if args.arm is not None:
        arm = resolve_arm(args.arm)
        print(arm.config_name, str(arm.update_probe_enabled).lower())
    if args.validate_result_dir is not None:
        reports = validate_profile_receipt(args.validate_result_dir)
        print(f"profile_reports={len(reports)}")


if __name__ == "__main__":
    main()
