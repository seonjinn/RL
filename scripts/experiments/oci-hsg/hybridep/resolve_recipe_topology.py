#!/usr/bin/env python3

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

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, cast


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers  # noqa: E402


@dataclass(frozen=True)
class RecipeTopology:
    num_nodes: int
    gpus_per_node: int
    config_segment_size: int | None
    resolved_config_sha256: str


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def resolve_topology(config_path: Path) -> RecipeTopology:
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(config_path), resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("resolved recipe must be a mapping")
    resolved_config = cast(dict[str, Any], resolved)
    cluster_raw = resolved_config.get("cluster")
    if not isinstance(cluster_raw, dict):
        raise ValueError("resolved recipe must contain a cluster mapping")
    cluster = cast(dict[str, Any], cluster_raw)

    config_segment_size = cluster.get("segment_size")
    if config_segment_size is not None:
        config_segment_size = _positive_int(config_segment_size, "cluster.segment_size")

    encoded = json.dumps(
        resolved_config,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return RecipeTopology(
        num_nodes=_positive_int(cluster.get("num_nodes"), "cluster.num_nodes"),
        gpus_per_node=_positive_int(
            cluster.get("gpus_per_node"), "cluster.gpus_per_node"
        ),
        config_segment_size=config_segment_size,
        resolved_config_sha256=sha256(encoded).hexdigest(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve scheduler-relevant topology from a NeMo-RL recipe."
    )
    parser.add_argument("config_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        topology = resolve_topology(args.config_path)
    except Exception as error:
        print(f"Failed to resolve recipe topology: {error}", file=sys.stderr)
        return 2

    config_segment = (
        "null"
        if topology.config_segment_size is None
        else str(topology.config_segment_size)
    )
    print(
        "\t".join(
            (
                str(topology.num_nodes),
                str(topology.gpus_per_node),
                config_segment,
                topology.resolved_config_sha256,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
