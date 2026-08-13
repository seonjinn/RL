#!/usr/bin/env bash

set -euo pipefail

: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT to the tested NeMo-RL commit}"
: "${HF_HOME:?Set HF_HOME to the shared Hugging Face cache root}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B}"
DATASET_ID="${DATASET_ID:-nvidia/OpenMathInstruct-2}"
DATASET_SPLIT="${DATASET_SPLIT:-train_1M}"
SOURCE_DIR="${SOURCE_DIR:-$(git rev-parse --show-toplevel)}"

cd "${SOURCE_DIR}"
test "$(git rev-parse HEAD)" = "${EXPECTED_SOURCE_COMMIT}"
test -z "$(git status --short)"

unset UV_NO_EDITABLE
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

uv run --frozen python - "${MODEL_ID}" "${DATASET_ID}" "${DATASET_SPLIT}" <<'PY'
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

model_id, dataset_id, dataset_split = sys.argv[1:]
snapshot = Path(snapshot_download(model_id, local_files_only=True))
config = AutoConfig.from_pretrained(model_id, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

index_paths = sorted(snapshot.glob("*.safetensors.index.json"))
if not index_paths:
    raise RuntimeError(f"No safetensors index found in {snapshot}")

weight_shards: set[str] = set()
for index_path in index_paths:
    index = json.loads(index_path.read_text())
    weight_shards.update(index.get("weight_map", {}).values())

if not weight_shards:
    raise RuntimeError(f"No weight shards listed by {index_paths}")

missing_shards = [
    shard
    for shard in sorted(weight_shards)
    if not (snapshot / shard).is_file() or (snapshot / shard).stat().st_size == 0
]
if missing_shards:
    raise RuntimeError(f"Missing or empty weight shards: {missing_shards}")

dataset = load_dataset(dataset_id, split=dataset_split)
first_row = dataset[0]
required_columns = {"problem", "expected_answer"}
missing_columns = sorted(required_columns - set(first_row))
if missing_columns:
    raise RuntimeError(f"Dataset row is missing required columns: {missing_columns}")

print(
    json.dumps(
        {
            "cuda": torch.version.cuda,
            "dataset_id": dataset_id,
            "dataset_rows": len(dataset),
            "dataset_split": dataset_split,
            "model_hidden_size": config.hidden_size,
            "model_id": model_id,
            "model_snapshot": str(snapshot),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_size": len(tokenizer),
            "weight_shards": len(weight_shards),
        },
        sort_keys=True,
    )
)
PY
