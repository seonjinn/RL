import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    hf_home = Path(os.environ["HF_HOME"]).resolve()
    model_ids = [
        model_id.strip()
        for model_id in os.environ["MODEL_IDS"].split(",")
        if model_id.strip()
    ]
    if not model_ids:
        raise ValueError("MODEL_IDS must contain at least one Hugging Face repository")

    for model_id in model_ids:
        snapshot = Path(snapshot_download(repo_id=model_id, cache_dir=hf_home / "hub"))
        config = snapshot / "config.json"
        weights = sorted(snapshot.glob("*.safetensors")) + sorted(snapshot.glob("*.bin"))
        if not config.is_file() or not weights:
            raise RuntimeError(
                f"Incomplete snapshot for {model_id}: config={config.is_file()} weights={len(weights)}"
            )
        print(
            json.dumps(
                {
                    "model_id": model_id,
                    "snapshot": str(snapshot),
                    "weights": len(weights),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
