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

import base64
from io import BytesIO
from typing import Any, Optional, Union

import requests
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from nemo_rl.utils.vlm_debug import debug_enabled, write_stage


def _tensor_shape(tensor: Optional[torch.Tensor]) -> list[int] | None:
    return list(tensor.shape) if tensor is not None else None


def _tensor_shape_list(
    tensors: list[Optional[torch.Tensor]],
) -> list[list[int] | None]:
    return [_tensor_shape(tensor) for tensor in tensors]


class PackedTensor:
    """Wrapper around a list of torch tensors and a dimension along which to pack the tensors.

    This class is used to wrap a list of tensors along with a `dim_to_pack` parameter.
    It can be used for data that can be packed along different dimensions (such as multimodal data).

    `dim_to_pack` is used to specify the dimension along which to pack the tensors.

    The list of tensors can be returned as a single packed tensor by calling `as_tensor` which will concatenate the tensors along the `dim_to_pack` dimension.
    """

    def __init__(
        self,
        tensors: Union[torch.Tensor, list[Optional[torch.Tensor]], list[None]],
        dim_to_pack: int,
    ) -> None:
        assert tensors is not None, "Input tensors to PackedTensor cannot be None"

        if isinstance(tensors, torch.Tensor):
            self.tensors: list[Optional[torch.Tensor]] = [tensors]
        elif isinstance(tensors, list):
            assert len(tensors) > 0, (
                "Input tensors to PackedTensor must be a non-empty list"
            )
            self.tensors: list[Optional[torch.Tensor]] = tensors
        else:
            raise ValueError(
                f"Unsupported type for input tensors to PackedTensor: {type(tensors)}"
            )
        self.dim_to_pack = dim_to_pack
        if debug_enabled():
            write_stage(
                "packed_tensor_create",
                {
                    "dim_to_pack": self.dim_to_pack,
                    "logical_length": len(self.tensors),
                    "tensor_shapes": _tensor_shape_list(self.tensors),
                },
            )

    def as_tensor(
        self, device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        if device is not None:
            # Move only non-None tensors to device, preserve Nones
            for i, item in enumerate(self.tensors):
                if item is not None:
                    self.tensors[i] = item.to(device)
        non_none_tensors = [t for t in self.tensors if t is not None]
        if len(non_none_tensors) == 0:
            return None
        padding_applied: list[list[int]] = []
        if len(non_none_tensors) > 1:
            ndim = non_none_tensors[0].ndim
            pack_dim = self.dim_to_pack % ndim
            has_mismatch = any(
                non_none_tensors[0].shape[d] != tensor.shape[d]
                for tensor in non_none_tensors[1:]
                for d in range(ndim)
                if d != pack_dim
            )
            if has_mismatch:
                max_shape = [
                    max(tensor.shape[d] for tensor in non_none_tensors)
                    for d in range(ndim)
                ]
                padded_tensors = []
                for tensor in non_none_tensors:
                    pad: list[int] = []
                    pad_per_dim = [0] * ndim
                    for d in reversed(range(ndim)):
                        pad_amount = 0 if d == pack_dim else max_shape[d] - tensor.shape[d]
                        pad.extend([0, pad_amount])
                        pad_per_dim[d] = pad_amount
                    padded_tensors.append(torch.nn.functional.pad(tensor, pad, value=0))
                    padding_applied.append(pad_per_dim)
                non_none_tensors = padded_tensors

        result = torch.cat(non_none_tensors, dim=self.dim_to_pack).to(device)
        if debug_enabled():
            write_stage(
                "packed_tensor_as_tensor",
                {
                    "dim_to_pack": self.dim_to_pack,
                    "logical_length": len(self.tensors),
                    "input_tensor_shapes": _tensor_shape_list(self.tensors),
                    "non_none_tensor_shapes": [
                        list(tensor.shape) for tensor in non_none_tensors
                    ],
                    "padding_applied": padding_applied or None,
                    "output_shape": list(result.shape),
                },
            )
        return result

    def __len__(self) -> int:
        # this is the number of tensors in this data wrapper
        return len(self.tensors)

    def to(self, device: str | torch.device) -> "PackedTensor":
        self.tensors = [
            item.to(device) if item is not None else None for item in self.tensors
        ]
        return self

    def slice(self, indices: Union[list[int], torch.Tensor]) -> "PackedTensor":
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        tensors = [self.tensors[i] for i in idx]
        return PackedTensor(tensors, self.dim_to_pack)

    @classmethod
    def empty_like(cls, other: "PackedTensor") -> "PackedTensor":
        """Return a new PackedTensor with same length and dim_to_pack as `other`, with all entries None."""
        return cls([None] * len(other.tensors), other.dim_to_pack)

    @classmethod
    def concat(cls, from_packed_tensors: list["PackedTensor"]) -> "PackedTensor":
        """Concatenate a list of PackedTensor objects into a single PackedTensor.

        The underlying tensors from the PackedTensors are combined into a single list of tensors and used to create a new PackedTensor.

        Each batch must have the same dim_to_pack.

        Example:
        ```{doctest}
        >>> import torch
        >>> from nemo_rl.data.multimodal_utils import PackedTensor
        >>> p1 = PackedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])], dim_to_pack=0)
        >>> p2 = PackedTensor([torch.tensor([7, 8, 9])], dim_to_pack=0)
        >>> p3 = PackedTensor.concat([p1, p2])
        >>> p3.tensors
        [tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9])]
        >>> p3.as_tensor()
        tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>>
        ```
        """
        dim_to_packs = [batch.dim_to_pack for batch in from_packed_tensors]
        assert len(set(dim_to_packs)) == 1, (
            "All packed tensors must have the same dim_to_pack"
        )
        # concatenate the tensors
        tensors = []
        for packed_tensor in from_packed_tensors:
            tensors.extend(packed_tensor.tensors)
        dim_to_pack = dim_to_packs[0]
        return cls(tensors, dim_to_pack)

    @classmethod
    def flattened_concat(
        cls, from_packed_tensors: list["PackedTensor"]
    ) -> "PackedTensor":
        """Given a list of PackedTensor objects, flattens each PackedTensor and then concatenates them into a single PackedTensor.

        Each PackedTensor is first flattened by packing along the PackedTensor's `dim_to_pack` dimension. Then, the resulting flattened tensors are used to create a new PackedTensor.

        This is different from `PackedTensor.concat` which simply extends the underlying list of tensors. This is important because the `slice` and `__len__` methods operate on the underlying list of tensors. Note, however, that calling `as_tensor` on the resulting PackedTensor will result in the same tensor as `concat`.

        Each batch must have the same dim_to_pack.

        Example:
        ```{doctest}
        >>> import torch
        >>> from nemo_rl.data.multimodal_utils import PackedTensor
        >>> p1 = PackedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])], dim_to_pack=0)
        >>> p2 = PackedTensor([torch.tensor([7, 8, 9])], dim_to_pack=0)
        >>> p3 = PackedTensor.flattened_concat([p1, p2])
        >>> p3.tensors
        [tensor([1, 2, 3, 4, 5, 6]), tensor([7, 8, 9])]
        >>> p3.as_tensor()
        tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>>
        ```
        """
        dim_to_packs = [batch.dim_to_pack for batch in from_packed_tensors]
        assert len(set(dim_to_packs)) == 1, (
            "All packed tensors must have the same dim_to_pack"
        )
        tensors = [p.as_tensor() for p in from_packed_tensors]
        return cls(tensors, from_packed_tensors[0].dim_to_pack)


def get_multimodal_keys_from_processor(processor) -> list[str]:
    """Get keys of the multimodal data that can be used as model inputs.

    This will be used in the data_processor function to determine which keys to use as model inputs.
    """
    if isinstance(processor, PreTrainedTokenizerBase):
        return []

    all_keys = set()
    processor_model_input_names = list(getattr(processor, "model_input_names", []))
    image_processor_model_input_names = []
    if hasattr(processor, "image_processor"):
        image_processor_model_input_names = list(
            getattr(processor.image_processor, "model_input_names", [])
        )
        all_keys.update(image_processor_model_input_names)
    if hasattr(processor, "video_processor"):
        all_keys.update(processor.video_processor.model_input_names)
    if hasattr(processor, "feature_extractor"):
        all_keys.update(processor.feature_extractor.model_input_names)
    all_keys.update(processor_model_input_names)
    tokenizer_model_input_names = list(processor.tokenizer.model_input_names)
    all_keys.difference_update(set(tokenizer_model_input_names))
    multimodal_keys = list(all_keys)
    if debug_enabled():
        write_stage(
            "multimodal_keys",
            {
                "processor_class": type(processor).__name__,
                "processor_model_input_names": processor_model_input_names,
                "image_processor_model_input_names": image_processor_model_input_names,
                "tokenizer_model_input_names": tokenizer_model_input_names,
                "multimodal_keys": sorted(multimodal_keys),
            },
        )
    return multimodal_keys


def get_dim_to_pack_along(processor, key: str) -> int:
    """Special considerations for packing certain keys from certain processors.

    In most cases, the packed items are along dim 0
    """
    if processor.__class__.__name__ == "SmolVLMProcessor":
        return 1
    # return zero by default
    return 0


def resolve_to_image(image_path_or_image: str | Image.Image) -> Image.Image:
    """Resolve the image path to a PIL.Image object.

    image_path can be either:
    - path to local file
    - url to image
    - base64 encoded image
    """
    if isinstance(image_path_or_image, Image.Image):
        return image_path_or_image

    if image_path_or_image.startswith(("http://", "https://")):
        # Handle URL
        response = requests.get(image_path_or_image)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    elif image_path_or_image.startswith("data:"):
        # Handle base64 encoded image
        # Format: data:image/jpeg;base64,/9j/4AAQSkZJRg...
        header, encoded = image_path_or_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(BytesIO(image_data)).convert("RGB")
    else:
        # Handle local file path
        return Image.open(image_path_or_image).convert("RGB")
