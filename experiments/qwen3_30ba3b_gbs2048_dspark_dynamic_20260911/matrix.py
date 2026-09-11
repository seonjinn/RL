#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class Arm:
    name: str
    k: int | None
    status: str
    reason: str = ""


ARMS = (
    Arm("baseline", None, "ready"),
    Arm("dspark_k3", 3, "ready"),
    Arm("dspark_k5", 5, "ready"),
    Arm("dspark_k7", 7, "ready"),
    Arm(
        "dspark_dynamic",
        None,
        "blocked-unproven",
        "vLLM 0.25.1 scheduler-selected K has no verified reduction in DSpark draft work",
    ),
)


if __name__ == "__main__":
    print(json.dumps([asdict(arm) for arm in ARMS], indent=2))
