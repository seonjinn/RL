"""Copy a NeMo-Gym SWE jsonl and drop the rows of the given instance ids.

    python filter_swe_instances.py <in.jsonl> <out.jsonl> <instance_id> [<instance_id> ...]

The SWE GRPO data loader does not shuffle, so the dropped rows are replaced by the
next rows in file order and every other step keeps its prompts.
"""

import json
import sys
from pathlib import Path


def instance_id(line: str) -> str:
    return json.loads(line)["responses_create_params"]["metadata"]["instance_id"]


def main() -> None:
    src, dst, drop = Path(sys.argv[1]), Path(sys.argv[2]), set(sys.argv[3:])
    if not drop:
        sys.exit("give at least one instance id")
    kept, dropped = 0, []
    with src.open() as fin, dst.open("w") as fout:
        for n, line in enumerate(fin, 1):
            iid = instance_id(line)
            if iid in drop:
                dropped.append((n, iid))
                continue
            fout.write(line)
            kept += 1
    for n, iid in dropped:
        print(f"dropped line {n}: {iid}")
    missing = drop - {iid for _, iid in dropped}
    if missing:
        sys.exit(f"not found: {sorted(missing)}")
    print(f"wrote {dst} ({kept} rows)")


if __name__ == "__main__":
    main()
