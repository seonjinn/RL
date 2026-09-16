"""Stage the user-confirmed post-trained target from a pinned HF revision."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from launch import ARTIFACTS, TARGET, TARGET_REPO, TARGET_REVISION


def validate_checkpoint(root: Path) -> dict[str, int]:
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError("unexpected target architecture")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards:
        raise ValueError("empty checkpoint index")
    for name in [*shards, "tokenizer.json", "tokenizer_config.json"]:
        if Path(name).name != name:
            raise ValueError("checkpoint member escapes target directory")
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            raise FileNotFoundError(root / name)
    return {
        "shards": len(shards),
        "weight_bytes": sum((root / name).stat().st_size for name in shards),
    }


def main() -> None:
    job = os.environ["SLURM_JOB_ID"]
    if not job.isdigit():
        raise ValueError("invalid job id")
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite existing target: {TARGET}")
    node_root = Path(f"/raid/scratch/sna/q35-stage-{job}")
    node_root.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HOME"] = str(node_root / "hf")
    os.environ["XDG_CACHE_HOME"] = str(node_root / "cache")
    from huggingface_hub import snapshot_download

    source = Path(
        snapshot_download(
            repo_id=TARGET_REPO,
            revision=TARGET_REVISION,
            local_dir=node_root / "checkpoint",
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"],
            max_workers=4,
            token=False,
        )
    )
    counts = validate_checkpoint(source)
    temporary = TARGET.with_name(f"{TARGET.name}.staging-{job}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary, ignore=shutil.ignore_patterns(".cache"))
    if validate_checkpoint(temporary) != counts:
        raise ValueError("staged target manifest differs from downloaded snapshot")
    receipt = {
        "repo_id": TARGET_REPO,
        "revision": TARGET_REVISION,
        "job_id": job,
        **counts,
    }
    (temporary / "stage-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.rename(TARGET)
    (ARTIFACTS / f"stage-{job}-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({"target": str(TARGET), **receipt}), flush=True)


if __name__ == "__main__":
    main()
