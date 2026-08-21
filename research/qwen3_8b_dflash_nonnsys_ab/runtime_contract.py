#!/usr/bin/env python3

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentArm:
    config_path: str
    draft_training_enabled: bool


_ARMS = {
    "fixed": ExperimentArm(
        config_path="research/qwen3_8b_dflash_fixed_dense_control/config.yaml",
        draft_training_enabled=False,
    ),
    "online": ExperimentArm(
        config_path="research/qwen3_8b_dflash_online_cp1/config.yaml",
        draft_training_enabled=True,
    ),
}


def resolve_arm(name: str) -> ExperimentArm:
    try:
        return _ARMS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_ARMS))
        raise ValueError(
            f"Unsupported A/B arm {name!r}; expected one of: {choices}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(_ARMS))
    args = parser.parse_args()
    arm = resolve_arm(args.arm)
    print(arm.config_path, str(arm.draft_training_enabled).lower())


if __name__ == "__main__":
    main()
