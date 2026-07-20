# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


NEMO_GYM_SUBPROCESS_OPENAI_VERSION = "2.7.2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", type=Path, required=True)
    args, entrypoint_args = parser.parse_known_args()

    from nemo_gym import global_config  # pyright: ignore[reportMissingImports]

    global_config.openai_version = NEMO_GYM_SUBPROCESS_OPENAI_VERSION
    print(f"NeMo Gym subprocess OpenAI version: {NEMO_GYM_SUBPROCESS_OPENAI_VERSION}")
    sys.argv = [str(args.entrypoint), *entrypoint_args]
    runpy.run_path(str(args.entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
