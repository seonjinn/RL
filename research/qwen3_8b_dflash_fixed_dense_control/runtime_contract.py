#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def validate_ray_temp_root(path: Path) -> None:
    value = str(path)
    if re.fullmatch(r"/raid/scratch/r/\d+", value) is None:
        raise ValueError(
            "Ray temp root must use the proven short node-local path "
            "/raid/scratch/r/$SLURM_JOB_ID"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-temp-root", type=Path, required=True)
    args = parser.parse_args()
    validate_ray_temp_root(args.ray_temp_root)


if __name__ == "__main__":
    main()
