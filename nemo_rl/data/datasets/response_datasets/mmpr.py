# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import hashlib
import json
import os
from typing import Any, Optional

from datasets import Dataset, load_from_disk

from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import strip_image_tokens_from_text


def _mmpr_cache_dir(data_path: str, task_name: str) -> str:
    """Return the disk cache path for the prepared MMPR dataset.

    Cache key is derived from (absolute data_path, recipe mtime, task_name).
    Updating the recipe JSON automatically invalidates the cache.
    """
    cache_root = os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.expanduser("~/.cache/huggingface/datasets"),
    )
    try:
        recipe_mtime = int(os.path.getmtime(data_path))
    except OSError:
        recipe_mtime = 0
    cache_key = hashlib.sha256(
        f"{os.path.abspath(data_path)}|{recipe_mtime}|{task_name}".encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(cache_root, f"mmpr_prepared_{cache_key}")


def format_mmpr_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """Format the MMPR dataset into an OpenAI-API-like message log for MPO training.

    Expected MMPR format:
    {
        "image": path or list of paths,
        "question": str,
        "chosen_response": str,  # Preferred response
        "rejected_response": str,  # Non-preferred response
        ...
    }
    """
    images = example["image"]
    if isinstance(images, str):
        images = [images]

    user_content = [
        {
            "type": "text",
            "text": strip_image_tokens_from_text(str(example["question"])),
        },
    ]

    for img in images:
        user_content.append({
            "type": "image",
            "image": img,
        })

    # For MPO (and other preference-optimization losses), we need both chosen and rejected responses
    # MMPR typically provides preference pairs
    chosen_content = strip_image_tokens_from_text(
        str(example.get("chosen_response", example.get("chosen", "")))
    )
    rejected_content = strip_image_tokens_from_text(
        str(example.get("rejected_response", example.get("rejected", "")))
    )

    ret = {
        "context": [{"role": "user", "content": user_content}],
        "completions": [
            {
                "rank": 0,
                "completion": [
                    {"role": "assistant", "content": chosen_content.replace("<image>", "image")}
                ],
            },
            {
                "rank": 1,
                "completion": [
                    {"role": "assistant", "content": rejected_content.replace("<image>", "image")}
                ],
            },
        ],
    }
    if example.get("system", None) is not None:
        ret["context"].insert(0, {"role": "system", "content": example["system"]})
    return ret

# def process_mmpr_example(example: dict[str, Any]) -> dict[str, Any]:
#     """Process an MMPR example."""
#     thinking_mode = False
#     if "final answer:" in example["chosen"].lower():
#         index_chosen = example["chosen"].lower().find("final answer:")
#         example["chosen"] = "<think>" + example["chosen"][:index_chosen] + "</think>" + example["chosen"][index_chosen:]
#         index_rejected = example["rejected"].lower().find("final answer:")
#         example["rejected"] = "<think>" + example["rejected"][:index_rejected] + "</think>" + example["rejected"][index_rejected:]
#         thinking_mode = True
#     elif "\\boxed" in example["chosen"]:
#         index_chosen = example["chosen"].find("\\boxed")
#         example["chosen"] = "<think>" + example["chosen"][:index_chosen] + "</think>" + example["chosen"][index_chosen:]
#         index_rejected = example["rejected"].find("\\boxed")
#         example["rejected"] = "<think>" + example["rejected"][:index_rejected] + "</think>" + example["rejected"][index_rejected:]
#         thinking_mode = True
#     else:
#         example["chosen"] = "<think></think>" + example["chosen"] 
#         example["rejected"] = "<think></think>" + example["rejected"] 
#         thinking_mode = False

#     if thinking_mode:
#         example["system"] = "/think"
#     else:
#         example["system"] = "/no_think"
#     return example

def prepare_mmpr_dataset(data_path: str, split: str, task_name: Optional[str] = None):
    """Prepare the MMPR dataset for training.

    The full preparation pipeline (JSONL parsing, per-row ``os.path.exists``
    image checks for ~2.8M rows, shuffle + select + add_column) takes several
    minutes on lustre. We cache the resulting train/validation datasets to
    disk under ``$HF_DATASETS_CACHE`` so subsequent runs reload in seconds.
    The cache key is derived from the recipe's absolute path + mtime, so any
    edit to the recipe automatically invalidates the cache.
    """
    if task_name is None:
        task_name = "mmpr"

    cache_dir = _mmpr_cache_dir(data_path, task_name)
    train_cache = os.path.join(cache_dir, "train")
    val_cache = os.path.join(cache_dir, "validation")
    if os.path.exists(os.path.join(train_cache, "dataset_info.json")) and os.path.exists(
        os.path.join(val_cache, "dataset_info.json")
    ):
        print(f"[MMPR] Loading cached prepared dataset from {cache_dir}")
        train_dataset = load_from_disk(train_cache)
        val_dataset = load_from_disk(val_cache)
        print(
            f"[MMPR] Cache hit: {len(train_dataset)} train / {len(val_dataset)} val samples"
        )
        return {"train": train_dataset, "validation": val_dataset}

    print(
        f"[MMPR] Cache miss at {cache_dir}; preparing from scratch "
        "(this may take several minutes the first time)"
    )

    try:
        print(f"Loading MMPR dataset from {data_path}...")

        # data_path is a JSON recipe file (e.g. meta_commercial.json)
        full_dataset = None
        with open(data_path, "r") as f:
            meta_data = json.load(f)
        root = os.path.dirname(os.path.dirname(data_path))

        dataset = []
        for dataset_name, dataset_info in meta_data.items():
                # if "_".join(dataset_name.split("_")[-2:]) not in ["correctness_rules", "format_rules", "direct_rules"]:
                #     continue
                # elif dataset_name.split("_")[0] in ["ai2d", "chartqa", "CLEVR", "cocorem","docvqa", "dvqa", "gaokao", "geo170k","geometry3k", "geomverse","geoqa+", "geos" \
                # "MathV360K", "mavis", "unigeo", "super", "vqav2"]:
            # if dataset_name.split("_")[0] in ["ai2d"]:
            image_root = root + "/" + dataset_info["root"]
            annotation_file = root + "/" + dataset_info["annotation"]
            with open(annotation_file, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    if "<think>" not in rec["chosen"]:
                        rec["chosen"] = "<think></think>\n\n" + rec["chosen"] 
                        rec["rejected"] = "<think></think>\n\n" + rec["rejected"] 
                        rec["system"] = ""

                    #rec = process_mmpr_example(json.loads(line))
                    if isinstance(rec["image"], str):
                        rec["image"] = [os.path.join(image_root, rec["image"])]
                    else:
                        rec["image"] = [os.path.join(image_root, image_path) for image_path in rec["image"]]
                    if len(rec["image"]) != 1 or not os.path.exists(rec["image"][0]):
                        continue
                    dataset.append(rec)
        full_dataset = Dataset.from_list(dataset).shuffle(seed=42)
        # Create a small validation set from train data
        train_size = len(full_dataset)
        val_size = min(2000, train_size // 10)
        val_dataset = full_dataset.select(range(val_size))
        train_dataset = full_dataset.select(range(val_size, train_size))

        print(f"Successfully loaded MMPR dataset with {len(train_dataset)} training samples")

    except Exception as e:
        print(f"Error loading MMPR dataset: {e}")
        raise

    # Add task_name column
    train_dataset = train_dataset.add_column("task_name", [task_name] * len(train_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    # Persist to disk so future runs skip the slow JSONL + os.path.exists pass.
    # save_to_disk forces flatten_indices once here, so the loaded dataset is
    # already materialized and downstream consumers (e.g. add_column) won't
    # trigger another flatten_indices progress bar.
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[MMPR] Saving prepared dataset to cache at {cache_dir}")
        train_dataset.save_to_disk(train_cache)
        val_dataset.save_to_disk(val_cache)
        print(f"[MMPR] Cache write complete")
    except Exception as cache_err:
        # Caching is best-effort; a failure here should not break training.
        print(f"[MMPR] WARNING: failed to write dataset cache: {cache_err}")

    return {
        "train": train_dataset,
        "validation": val_dataset,
    }


class MMPRDataset:
    """Dataset class for MMPR (Multimodal Preference Ranking) dataset.

    This dataset contains multimodal preference pairs for MPO training.
    Each example includes an image, a question, and preferred/rejected response pairs.

    Args:
        split: The split of the dataset to use ('train', 'validation', 'test')
        prompt_file: Optional file containing custom prompts for the dataset
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        prompt_file: Optional[str] = None,
        **kwargs,
    ):
        if split not in ["train", "validation", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train', 'validation', or 'test'."
            )

        self.task_name = "mmpr"
        self.formatted_ds = prepare_mmpr_dataset(
            data_path=data_path, split=split, task_name=self.task_name
        )

        self.task_spec = TaskDataSpec(
            task_name="MMPR",
            prompt_file=prompt_file,
        )