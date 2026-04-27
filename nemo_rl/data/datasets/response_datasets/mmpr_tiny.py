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

"""MMPR-Tiny multimodal dataset (and its `mmpr_miniscule` deterministic
sample) wired for Super's response-dataset registry.

Layout expected on disk under ``cache_dir``:

    <cache_dir>/
        mmpr_tiny.parquet
        MMPR-Tiny/images/<image-files>...
        .mmpr_ready                   # marker file (optional)

When the marker / parquet / images are not present we fall back to the
upstream HF download path (mirrors Omni's
``nemo-rl-recipes/nemo_rl/data/datasets/response_datasets/mmpr_tiny.py``),
so the same dataset class works both with the pre-staged
``mmpr_miniscule`` corpus and a fresh HF pull.
"""

import os
import shutil
import zipfile
from typing import Any, Optional

import pandas as pd
from datasets import Dataset, Features, Sequence, Value

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.interfaces import TaskDataSpec


def format_mmpr_tiny_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """Format the MMPR-Tiny dataset into an OpenAI-API-like message log.

    Each row carries one image path (already resolved to an absolute
    filesystem path by ``prepare_mmpr_tiny_dataset``) and a question.
    The ``vlm_hf_data_processor`` opens the image lazily when the
    message log is materialized.
    """
    user_content = [
        {
            "type": "image",
            "image": example["images"][0],
        },
        {
            "type": "text",
            "text": str(example["question"]).replace("<image>", ""),
        },
    ]

    assistant_content = str(example["answer"])

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_name": "mmpr_tiny",
    }


def _ensure_mmpr_cached(cache_dir: str) -> None:
    """Download and extract MMPR-Tiny images if not already cached.

    Thread/process-safe via an atomic ready marker plus an exclusive
    file lock; in the typical Super workflow ``cache_dir`` already
    points at ``mmpr_miniscule/processed`` (data already on disk) and
    this short-circuits on the existing marker / parquet / images
    without any download.
    """
    images_dir = os.path.join(cache_dir, "MMPR-Tiny", "images")
    parquet_path = os.path.join(cache_dir, "mmpr_tiny.parquet")
    ready_marker = os.path.join(cache_dir, ".mmpr_ready")

    if os.path.exists(ready_marker):
        return

    if os.path.exists(images_dir) and os.path.exists(parquet_path):
        with open(ready_marker, "w") as f:
            f.write("ready\n")
        return

    os.makedirs(cache_dir, exist_ok=True)
    lock_file = os.path.join(cache_dir, ".mmpr_download.lock")

    import fcntl

    try:
        with open(lock_file, "w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

            if os.path.exists(ready_marker):
                return

            print(f"Downloading MMPR-Tiny to {cache_dir}...", flush=True)
            from huggingface_hub import hf_hub_download

            zip_path = hf_hub_download(
                "OpenGVLab/MMPR-Tiny", "images.zip", repo_type="dataset"
            )
            with zipfile.ZipFile(zip_path, "r") as zf:
                temp = os.path.join(cache_dir, "_temp")
                zf.extractall(temp)
                shutil.move(os.path.join(temp, "images"), images_dir)
                os.rmdir(temp)

            pq = hf_hub_download(
                "OpenGVLab/MMPR-Tiny",
                "mmpr_tiny.parquet",
                repo_type="dataset",
            )
            shutil.copy(pq, parquet_path)

            with open(ready_marker, "w") as f:
                f.write("ready\n")
            print(f"MMPR-Tiny cached successfully at {cache_dir}", flush=True)
    finally:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError:
                pass


def prepare_mmpr_tiny_dataset(
    split: str = "train",
    task_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    val_size: int = 500,
):
    """Load and prepare the MMPR-Tiny / mmpr_miniscule dataset.

    Args:
        split: ``train`` is the only meaningful split for the source
            corpus; ``test`` is mapped to the validation slice for
            registry compatibility.
        task_name: Task-name column to attach (defaults to
            ``mmpr_tiny``).
        cache_dir: Directory containing ``mmpr_tiny.parquet`` and
            ``MMPR-Tiny/images/``.
        val_size: Number of leading rows to carve off for validation.
            Capped at 10% of the corpus to keep the train slice large
            on tiny corpora like ``mmpr_miniscule`` (128 rows -> max
            12 val rows).
    """
    if task_name is None:
        task_name = "mmpr_tiny"

    if cache_dir is None:
        raise ValueError(
            "cache_dir is required for MMPR-Tiny "
            "(point at e.g. .../mmpr_miniscule/processed)"
        )

    _ensure_mmpr_cached(cache_dir)

    df = pd.read_parquet(os.path.join(cache_dir, "mmpr_tiny.parquet"))

    df["images"] = df["images"].str[0].apply(
        lambda x: [os.path.join(cache_dir, x["path"])]
    )
    df["question"] = df["prompt"].apply(
        lambda p: next(
            (m["content"] for m in p if m.get("role") == "user"), ""
        )
    )
    df["answer"] = df["reward_model"].apply(
        lambda r: r.get("ground_truth", "")
    )
    df = df[["images", "question", "answer"]]
    df = df.assign(task_name=task_name)

    features = Features(
        {
            "images": Sequence(Value("string")),
            "question": Value("string"),
            "answer": Value("string"),
            "task_name": Value("string"),
        }
    )

    full_dataset = Dataset.from_pandas(
        df, preserve_index=False, features=features
    )

    total_size = len(full_dataset)
    if val_size > 0:
        val_size = min(val_size, max(1, total_size // 10))
        val_dataset = full_dataset.select(range(val_size))
        train_dataset = full_dataset.select(range(val_size, total_size))
    else:
        train_dataset = full_dataset
        val_dataset = None

    return {"train": train_dataset, "validation": val_dataset}


class MMPRTinyDataset(RawDataset):
    """Wrapper around the MMPR-Tiny / mmpr_miniscule corpus.

    Args:
        split: ``train`` selects the train slice; ``test`` returns the
            validation slice (kept for registry symmetry).
        prompt_file: Optional system prompt file passed through to
            ``TaskDataSpec``.
        cache_dir: Directory containing ``mmpr_tiny.parquet`` and
            ``MMPR-Tiny/images/`` (typically the pre-staged
            ``mmpr_miniscule/processed`` directory).
        val_size: Number of leading rows to use for validation.
    """

    def __init__(
        self,
        split: str = "train",
        prompt_file: Optional[str] = None,
        cache_dir: Optional[str] = None,
        val_size: int = 500,
        **kwargs,
    ):
        if split not in ("train", "test"):
            raise ValueError(
                f"Invalid split: {split}. Please use 'train' or 'test'."
            )
        self.task_name = "mmpr_tiny"

        slices = prepare_mmpr_tiny_dataset(
            split=split,
            task_name=self.task_name,
            cache_dir=cache_dir,
            val_size=val_size,
        )
        self.formatted_ds = slices

        # Surface ``self.dataset`` and ``self.val_dataset`` matching
        # Super's registry convention. ``setup_response_data`` in
        # ``nemo_rl/data/utils.py`` will pick up ``self.val_dataset``
        # automatically (see "validation dataset from train dataset"
        # branch), so a single ``data.train`` entry in YAML is enough --
        # no separate ``data.validation`` block needed for the
        # tiny-corpus smoke.
        self.dataset = (
            slices["train"]
            if split == "train"
            else (slices.get("validation") or slices["train"])
        )
        self.val_dataset = slices.get("validation")

        self.task_spec = TaskDataSpec(
            task_name="MMPR-Tiny",
            prompt_file=prompt_file,
        )
