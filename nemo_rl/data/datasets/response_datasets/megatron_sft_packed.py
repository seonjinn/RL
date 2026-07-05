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

from functools import partial
from typing import Any, Optional

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path
from nemo_rl.data.megatron_sft_packed import megatron_sft_packed_preprocessor


class MegatronSFTPackedDataset(RawDataset):
    """Load Megatron-LM offline-packed SFT JSONL records."""

    def __init__(
        self,
        data_path: str,
        chat_key: str = "messages",
        subset: Optional[str] = None,
        split: Optional[str] = None,
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        self.chat_key = chat_key
        self.task_name = "megatron_sft_packed"
        self.dataset = load_dataset_from_path(data_path, subset, split)
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        messages = [message for message in data[self.chat_key]]
        assert messages and messages[0]["role"] == "system", (
            "Megatron SFT packed records are expected to start with a system message"
        )
        assert messages[-1]["role"] == "assistant", (
            "Megatron SFT packed records are expected to end with an assistant message"
        )
        return {"packed_messages": messages, "task_name": self.task_name}

    def set_processor(self):
        self.processor = partial(
            megatron_sft_packed_preprocessor,
            prompt_format=self.data_config.get("megatron_sft_prompt_format", None),
            pad_token=self.data_config.get("megatron_sft_pad_token", None),
            assistant_prefix_len=int(
                self.data_config.get("megatron_sft_assistant_prefix_len", 0)
            ),
            context_parallel_size=int(
                self.data_config.get("megatron_sft_context_parallel_size", 1)
            ),
        )
