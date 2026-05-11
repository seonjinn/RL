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

from typing import Any, Optional
import math
import os

import torch
from einops import rearrange
from megatron.core.packed_seq_params import PackedSeqParams
from nemo_rl.data.multimodal_utils import PackedTensor


def _get_num_embeddings_from_sizes(
    imgs_sizes: torch.Tensor,
    patch_dim: int,
    downsample_ratio: float,
    class_token_len: int = 0,
) -> torch.Tensor:
    patches_per_image = (imgs_sizes[:, 0] // patch_dim) * (imgs_sizes[:, 1] // patch_dim)
    seq_len = patches_per_image + class_token_len
    return (seq_len * (downsample_ratio ** 2)).int()


def is_llava_model(model) -> bool:
    """Check if the model is a LLaVA model.

    Args:
        model: The model to check

    Returns:
        True if the model is a LLaVA model, False otherwise
    """
    # Handle wrapped models (e.g., DDP, Float16Module)
    actual_model = model
    while hasattr(actual_model, 'module'):
        actual_model = actual_model.module
    # Check for core LLaVAModel
    try:
        from megatron.core.models.multimodal.llava_model import LLaVAModel
        if isinstance(actual_model, LLaVAModel):
            return True
    except ImportError:
        pass
    # Check for MIMO-based LLaVA models (Megatron-Bridge)
    # These wrap a LLaVA-style model but aren't a direct LLaVAModel subclass.
    # Detect by checking for the llava_model attribute or image token handling.
    if hasattr(actual_model, 'llava_model'):
        return True
    if hasattr(actual_model, 'img_start_token_id') and hasattr(actual_model, 'img_end_token_id'):
        return True
    config = getattr(actual_model, 'config', None)
    if config is not None and hasattr(config, 'img_start_token_id') and hasattr(config, 'img_end_token_id'):
        return True
    return False


def _get_image_token_index(model) -> Optional[int]:
    """Extract the single image placeholder token ID used by Megatron LLaVA."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    for obj in [inner, getattr(inner, "config", None)]:
        if obj is None:
            continue
        image_token_index = getattr(obj, "image_token_index", None)
        if image_token_index is not None:
            return int(image_token_index)
    return None


def _get_model_config(model) -> tuple[int, float, int, bool, Optional[int]]:
    """Return the vision expansion parameters used by Megatron LLaVA."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    patch_dim = getattr(getattr(inner, "vision_model", None), "patch_dim", 16)
    downsample_ratio = getattr(inner, "downsample_ratio", None)
    if downsample_ratio is None:
        pixel_shuffle = getattr(inner, "_pixel_shuffle", False)
        conv_merging = getattr(inner, "_use_conv_merging", False)
        downsample_ratio = 1.0
        if pixel_shuffle:
            downsample_ratio *= 0.5
        if conv_merging:
            downsample_ratio *= 0.5
        if not pixel_shuffle and not conv_merging:
            downsample_ratio = 0.5

    drop_vision_class_token = getattr(inner, "_drop_vision_class_token", True)
    class_token_len = 0 if drop_vision_class_token else getattr(inner, "_class_token_len", 1)
    dynamic_resolution = getattr(inner, "_dynamic_resolution", False)
    static_img_seq_len = getattr(inner, "img_seq_len", None) if not dynamic_resolution else None
    return patch_dim, float(downsample_ratio), int(class_token_len), bool(dynamic_resolution), static_img_seq_len


def _split_imgs_sizes_by_sample(
    imgs_sizes,
    image_counts: list[int],
) -> list[Optional[torch.Tensor]]:
    if imgs_sizes is None:
        return [None for _ in image_counts]

    if isinstance(imgs_sizes, PackedTensor):
        tensors = imgs_sizes.tensors
        if imgs_sizes._dedup_indices is not None:
            tensors = [imgs_sizes.tensors[i] for i in imgs_sizes._dedup_indices]
        return [t for t in tensors]

    if torch.is_tensor(imgs_sizes):
        if imgs_sizes.dim() == 3 and imgs_sizes.shape[0] == len(image_counts):
            return [imgs_sizes[b] for b in range(len(image_counts))]
        if imgs_sizes.dim() == 2:
            per_sample = []
            offset = 0
            for count in image_counts:
                per_sample.append(imgs_sizes[offset : offset + count])
                offset += count
            return per_sample

    return [None for _ in image_counts]


def _compute_vision_expansion_per_sample(
    collapsed_ids: torch.Tensor,
    data_dict: dict,
    model,
) -> list[int]:
    """Compute actual Megatron image expansion, not the vLLM token count."""
    image_token_index = _get_image_token_index(model)
    if image_token_index is None:
        return [0 for _ in range(collapsed_ids.shape[0])]

    image_counts = [
        int((collapsed_ids[b] == image_token_index).sum().item())
        for b in range(collapsed_ids.shape[0])
    ]
    patch_dim, downsample_ratio, class_token_len, dynamic_resolution, static_img_seq_len = _get_model_config(model)
    imgs_sizes_per_sample = _split_imgs_sizes_by_sample(data_dict.get("imgs_sizes"), image_counts)

    expansions = []
    for sizes, image_count in zip(imgs_sizes_per_sample, image_counts):
        if image_count == 0:
            expansions.append(0)
        elif sizes is not None and torch.is_tensor(sizes) and sizes.numel() > 0:
            sizes = sizes.to(dtype=torch.int32, device=collapsed_ids.device).view(-1, 2)
            embeds = _get_num_embeddings_from_sizes(
                sizes, patch_dim, downsample_ratio, class_token_len
            )
            expansions.append(max(0, int(embeds.sum().item()) - image_count))
        elif not dynamic_resolution and static_img_seq_len is not None:
            expansions.append(max(0, image_count * (int(static_img_seq_len) - 1)))
        else:
            expansions.append(0)
    return expansions


def compute_vision_expansion(
    imgs_sizes_per_sample: list,
    num_image_placeholders_per_sample: list[int],
    patch_dim: int,
    downsample_ratio: float,
    class_token_len: int = 0,
    num_frames_per_sample: Optional[list] = None,
    temporal_patch_size: int = 1,
) -> list[int]:
    """Compute per-sample vision token expansion (extra tokens after model expansion).

    Each image placeholder in collapsed input_ids expands to N vision embeddings
    during the Megatron forward pass.  The "expansion" for a sample is:
        total_vision_embeds - num_image_placeholders
    which equals the number of extra tokens the model produces beyond the
    collapsed sequence length.

    When ``temporal_patch_size > 1`` (conv3d), RADIO groups T consecutive
    video frames into one tubelet.  imgs_sizes still has one entry per raw
    frame, but the model produces ``num_frames / T`` tubelet embeddings per
    video.  The ``num_frames_per_sample`` list tells this function how many
    raw frames each sample has so the embedding count can be divided by T.
    """
    expansions = []
    for b_idx, (b_imgs_sizes, n_placeholders) in enumerate(zip(
        imgs_sizes_per_sample, num_image_placeholders_per_sample
    )):
        if b_imgs_sizes is None or n_placeholders == 0:
            expansions.append(0)
            continue
        if not isinstance(b_imgs_sizes, torch.Tensor):
            b_imgs_sizes = torch.tensor(b_imgs_sizes, dtype=torch.int32)
        if b_imgs_sizes.numel() == 0:
            expansions.append(0)
            continue
        embeds = _get_num_embeddings_from_sizes(
            b_imgs_sizes, patch_dim, downsample_ratio, class_token_len
        )
        total_embeds = int(embeds.sum().item())
        if temporal_patch_size > 1 and num_frames_per_sample is not None:
            b_num_frames = num_frames_per_sample[b_idx]
            if b_num_frames is not None:
                if not isinstance(b_num_frames, torch.Tensor):
                    b_num_frames = torch.tensor(b_num_frames, dtype=torch.int32)
                has_video = (b_num_frames > 1).any().item() if b_num_frames.numel() > 0 else False
                if has_video:
                    per_item_nf = b_num_frames.tolist()
                    per_item_embeds = embeds.tolist()
                    reduced_total = 0
                    frame_offset = 0
                    for nf in per_item_nf:
                        item_embeds = sum(per_item_embeds[frame_offset:frame_offset + nf])
                        if nf > 1:
                            num_tubelets = math.ceil(nf / temporal_patch_size)
                            per_frame_embeds = item_embeds // nf
                            reduced_total += per_frame_embeds * num_tubelets
                        else:
                            reduced_total += item_embeds
                        frame_offset += nf
                    total_embeds = reduced_total
        expansions.append(max(0, total_embeds - n_placeholders))
    return expansions


def compute_expanded_lengths(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    imgs_sizes,
    image_token_id: Optional[int],
    patch_dim: int = 16,
    downsample_ratio: float = 0.5,
    class_token_len: int = 0,
    num_frames=None,
    temporal_patch_size: int = 1,
    max_length: Optional[int] = None,
    img_start_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Return per-sample expanded length for vision-expansion-aware packing.

    The expanded length accounts for the difference between vLLM image
    placeholder tokens and actual Megatron vision embeddings::

        expanded_len = input_len + vision_expansion

    where ``vision_expansion`` is computed by :func:`compute_vision_expansion`.
    The result is clamped to *max_length* when provided so that it never
    exceeds the bin-packer capacity.
    """
    batch_size = input_ids.shape[0]
    expanded = input_lengths.clone().to(torch.int64)

    if image_token_id is None or imgs_sizes is None:
        if max_length is not None:
            expanded.clamp_(max=max_length)
        return expanded

    if hasattr(imgs_sizes, "tensors"):
        if getattr(imgs_sizes, "_dedup_indices", None) is not None:
            imgs_sizes_per_sample = [imgs_sizes.tensors[j] for j in imgs_sizes._dedup_indices]
        else:
            imgs_sizes_per_sample = imgs_sizes.tensors
    elif isinstance(imgs_sizes, torch.Tensor):
        if imgs_sizes.dim() == 2:
            imgs_sizes_per_sample = [imgs_sizes[b] for b in range(batch_size)]
        else:
            imgs_sizes_per_sample = [None] * batch_size
    else:
        imgs_sizes_per_sample = [None] * batch_size

    if img_start_token_id is not None:
        num_image_placeholders = [
            int((input_ids[b] == img_start_token_id).sum().item())
            for b in range(batch_size)
        ]
    else:
        num_image_placeholders = [
            int((input_ids[b] == image_token_id).sum().item())
            for b in range(batch_size)
        ]

    _cel_collapse_savings = [0] * batch_size
    if img_start_token_id is not None:
        for b in range(batch_size):
            raw_image_count = int((input_ids[b] == image_token_id).sum().item())
            _cel_collapse_savings[b] = max(0, raw_image_count - num_image_placeholders[b])

    if num_frames is not None:
        if hasattr(num_frames, "tensors"):
            if getattr(num_frames, "_dedup_indices", None) is not None:
                nf_per_sample = [num_frames.tensors[j] for j in num_frames._dedup_indices]
            else:
                nf_per_sample = num_frames.tensors
        elif isinstance(num_frames, torch.Tensor):
            nf_per_sample = [num_frames[b] for b in range(batch_size)]
        else:
            nf_per_sample = [None] * batch_size
    else:
        nf_per_sample = None

    expansions = compute_vision_expansion(
        imgs_sizes_per_sample,
        num_image_placeholders,
        patch_dim,
        downsample_ratio,
        class_token_len,
        num_frames_per_sample=nf_per_sample,
        temporal_patch_size=temporal_patch_size,
    )

    for b in range(batch_size):
        expanded[b] = int(input_lengths[b].item()) - _cel_collapse_savings[b] + expansions[b]

    if max_length is not None:
        expanded.clamp_(max=max_length)

    return expanded


def _trim_image_data_for_truncated_sample(
    new_data_dict: dict,
    b: int,
    surviving_image_count: int,
) -> None:
    """Trim per-sample image-data tensors to the first ``surviving_image_count`` images.

    Text truncation in ``truncate_for_expanded_budget`` can drop trailing
    ``<img>...</img>`` groups from ``input_ids[b]``. Without a matching trim
    on the image-data side, downstream ``_prepare_image_data`` will build a
    ``num_image_tiles`` whose length is the ORIGINAL image count, while
    Megatron's ``_preprocess_data`` counts placeholders post-truncation — and
    the ``num_image_tiles.split(num_images_per_sample, dim=0)`` crashes with
    ``split_with_sizes expects split_sizes to sum exactly to N``.
    """
    tiles_to_keep_pvf: Optional[int] = None
    if "pixel_values_flat" in new_data_dict and "image_num_patches" in new_data_dict:
        inp = new_data_dict["image_num_patches"]
        if isinstance(inp, PackedTensor):
            inp_per_sample = (
                [inp.tensors[j] for j in inp._dedup_indices]
                if inp._dedup_indices is not None
                else list(inp.tensors)
            )
            if 0 <= b < len(inp_per_sample) and inp_per_sample[b] is not None:
                tiles_to_keep_pvf = int(
                    inp_per_sample[b][:surviving_image_count].sum().item()
                )

    for key in ("pixel_values", "imgs_sizes", "image_num_patches", "num_frames"):
        if key not in new_data_dict:
            continue
        pt = new_data_dict[key]
        if not isinstance(pt, PackedTensor):
            continue
        per_sample = (
            [pt.tensors[j] for j in pt._dedup_indices]
            if pt._dedup_indices is not None
            else list(pt.tensors)
        )
        if not (0 <= b < len(per_sample)):
            continue
        old = per_sample[b]
        if old is None:
            continue
        if surviving_image_count <= 0:
            per_sample[b] = None
        elif old.shape[0] > surviving_image_count:
            per_sample[b] = old[:surviving_image_count]
        else:
            continue
        new_data_dict[key] = PackedTensor(per_sample, dim_to_pack=pt.dim_to_pack)

    if tiles_to_keep_pvf is not None and "pixel_values_flat" in new_data_dict:
        pvf = new_data_dict["pixel_values_flat"]
        if isinstance(pvf, PackedTensor):
            pvf_per_sample = (
                [pvf.tensors[j] for j in pvf._dedup_indices]
                if pvf._dedup_indices is not None
                else list(pvf.tensors)
            )
            if 0 <= b < len(pvf_per_sample) and pvf_per_sample[b] is not None:
                old_pvf = pvf_per_sample[b]
                if tiles_to_keep_pvf <= 0:
                    pvf_per_sample[b] = None
                elif old_pvf.shape[0] > tiles_to_keep_pvf:
                    pvf_per_sample[b] = old_pvf[:tiles_to_keep_pvf]
                else:
                    return
                new_data_dict["pixel_values_flat"] = PackedTensor(
                    pvf_per_sample, dim_to_pack=pvf.dim_to_pack
                )


def truncate_for_expanded_budget(
    data_dict: dict,
    max_seq_length: int,
    patch_dim: int = 16,
    downsample_ratio: float = 0.5,
    class_token_len: int = 0,
    pad_token_id: int = 0,
    image_token_id: Optional[int] = None,
    img_start_token_id: Optional[int] = None,
    temporal_patch_size: int = 1,
) -> tuple[dict, torch.Tensor]:
    """Truncate collapsed sequences so the expanded (post-vision) length fits the budget.

    Mirrors Megatron's ``_truncate_to_decoder_seq_len``: the expanded sequence
    length is ``collapsed_len - num_images + total_vision_embeds``, and this must
    be <= ``max_seq_length``.  When a sample exceeds the budget, text tokens are
    removed from the right (response side).

    Args:
        data_dict: Training batch with ``input_ids``, ``input_lengths``, and
            optionally ``imgs_sizes`` (PackedTensor), ``token_mask``,
            ``advantages``, ``generation_logprobs``, ``sample_mask``.
        max_seq_length: Maximum allowed **expanded** sequence length.
        patch_dim: Vision patch dimension (must match model).
        downsample_ratio: Pixel-shuffle downsample ratio (must match model).
        class_token_len: Number of class tokens per image (usually 0).
        pad_token_id: Token ID used for padding.
        image_token_id: The image placeholder token ID.  If ``None``, no
            truncation is performed (text-only data).
        img_start_token_id: The ``<img>`` token ID.  When provided, counts
            ``<img>`` groups (post-collapse placeholders) instead of raw
            ``<image>`` tokens for expansion calculation.
        temporal_patch_size: Conv3d temporal patch size (T).  When > 1,
            video frame embeddings are reduced by factor T.

    Returns:
        (data_dict, truncated_mask) where ``truncated_mask`` is a bool tensor
        of shape ``(batch_size,)`` indicating which samples were truncated.
    """
    input_ids = data_dict["input_ids"]
    input_lengths = data_dict.get("input_lengths")
    batch_size = input_ids.shape[0]
    truncated_mask = torch.zeros(batch_size, dtype=torch.bool)

    if image_token_id is None or input_lengths is None:
        return data_dict, truncated_mask

    imgs_sizes = data_dict.get("imgs_sizes")
    has_imgs = imgs_sizes is not None

    if has_imgs and hasattr(imgs_sizes, "tensors"):
        if getattr(imgs_sizes, "_dedup_indices", None) is not None:
            imgs_sizes_per_sample = [imgs_sizes.tensors[j] for j in imgs_sizes._dedup_indices]
        else:
            imgs_sizes_per_sample = imgs_sizes.tensors
    elif has_imgs and isinstance(imgs_sizes, torch.Tensor):
        imgs_sizes_per_sample = []
        idx = 0
        for b in range(batch_size):
            n = int((input_ids[b] == image_token_id).sum().item())
            if n > 0 and idx + n <= imgs_sizes.shape[0]:
                imgs_sizes_per_sample.append(imgs_sizes[idx : idx + n])
                idx += n
            else:
                imgs_sizes_per_sample.append(None)
    else:
        imgs_sizes_per_sample = [None] * batch_size

    if img_start_token_id is not None:
        num_image_placeholders = [
            int((input_ids[b] == img_start_token_id).sum().item()) for b in range(batch_size)
        ]
    else:
        num_image_placeholders = [
            int((input_ids[b] == image_token_id).sum().item()) for b in range(batch_size)
        ]

    num_frames = data_dict.get("num_frames")
    num_frames_per_sample: Optional[list] = None
    if temporal_patch_size > 1 and num_frames is not None:
        if hasattr(num_frames, "tensors"):
            if getattr(num_frames, "_dedup_indices", None) is not None:
                num_frames_per_sample = [num_frames.tensors[j] for j in num_frames._dedup_indices]
            else:
                num_frames_per_sample = num_frames.tensors
        elif isinstance(num_frames, torch.Tensor):
            num_frames_per_sample = [num_frames] * batch_size
        elif isinstance(num_frames, list):
            num_frames_per_sample = num_frames

    expansions = compute_vision_expansion(
        imgs_sizes_per_sample,
        num_image_placeholders,
        patch_dim,
        downsample_ratio,
        class_token_len,
        num_frames_per_sample=num_frames_per_sample,
        temporal_patch_size=temporal_patch_size,
    )

    collapse_savings = [0] * batch_size
    if img_start_token_id is not None:
        for b in range(batch_size):
            raw_image_count = int((input_ids[b] == image_token_id).sum().item())
            collapse_savings[b] = max(0, raw_image_count - num_image_placeholders[b])

    samples_to_truncate = []
    for b in range(batch_size):
        valid_len = int(input_lengths[b].item())
        expanded_len = valid_len - collapse_savings[b] + expansions[b]
        if expanded_len > max_seq_length:
            max_collapsed_len = max(0, max_seq_length - expansions[b] + collapse_savings[b])
            if max_collapsed_len < valid_len:
                samples_to_truncate.append((b, valid_len, max_collapsed_len, expansions[b], expanded_len))

    if not samples_to_truncate:
        return data_dict, truncated_mask

    new_data_dict = data_dict.copy()
    new_data_dict["input_ids"] = new_data_dict["input_ids"].clone()
    new_data_dict["input_lengths"] = new_data_dict["input_lengths"].clone()
    for key in ("token_mask", "advantages", "generation_logprobs"):
        if key in new_data_dict and new_data_dict[key].dim() >= 2:
            new_data_dict[key] = new_data_dict[key].clone()

    group_count_tok_id = (
        img_start_token_id if img_start_token_id is not None else image_token_id
    )

    for b, valid_len, max_collapsed_len, expansion, expanded_len in samples_to_truncate:
        truncated_mask[b] = True
        new_data_dict["input_ids"][b, max_collapsed_len:] = pad_token_id
        new_data_dict["input_lengths"][b] = max_collapsed_len

        for key in ("token_mask", "advantages", "generation_logprobs"):
            if key in new_data_dict and new_data_dict[key].dim() >= 2:
                new_data_dict[key][b, max_collapsed_len:] = 0

        if group_count_tok_id is not None:
            pre_ids = data_dict["input_ids"][b, :valid_len]
            post_ids = new_data_dict["input_ids"][b, :max_collapsed_len]
            original_n_imgs = int((pre_ids == group_count_tok_id).sum().item())
            surviving_n_imgs = int((post_ids == group_count_tok_id).sum().item())
            if surviving_n_imgs < original_n_imgs:
                _trim_image_data_for_truncated_sample(
                    new_data_dict, b, surviving_n_imgs
                )

    return new_data_dict, truncated_mask


def collapse_multimodal_tokens(data_dict: Any, model: Any) -> Any:
    """Collapse N image tokens to 1 token per image for Megatron LLaVA forward pass.

    vLLM uses N tokens per image (1:1 token-to-embedding), while Megatron uses 1 token
    per image/tile (1:N via imgs_sizes). This collapses <img><image>×N</img> to <img><image></img>.

    Processes the full padded sequence (not just valid content) so that after model forward,
    output length matches padded input length. Padding tokens (zeros) won't match image token
    IDs, so only content region gets collapsed while padding is preserved.
    """
    image_token_ids = _get_image_token_ids(model)
    if image_token_ids is None or "pixel_values" not in data_dict:
        return data_dict

    input_ids = data_dict["input_ids"]
    input_lengths = data_dict.get("input_lengths")
    img_start_id, img_end_id = image_token_ids
    batch_size = input_ids.shape[0]

    # Check if pixel_values key exists without image tokens. This happens when
    # all samples in a micro-batch were discarded (overlong).
    img_start_count = (input_ids == img_start_id).sum().item()
    img_end_count = (input_ids == img_end_id).sum().item()

    if img_start_count == 0 and img_end_count == 0 and "pixel_values" in data_dict:
        # Drop the stale multimodal keys and treat the batch as text-only.
        del data_dict["pixel_values"]
        if "imgs_sizes" in data_dict:
            del data_dict["imgs_sizes"]
        return data_dict

    original_seq_len = input_ids.shape[1]
    has_imgs_sizes = "imgs_sizes" in data_dict

    collapsed_list = []
    new_lengths = []
    tokens_removed_per_sample = []
    all_keep_masks = []

    for b in range(batch_size):
        # Process full padded sequence, not just valid content
        # Padding tokens (zeros) won't match image token IDs, so only content gets collapsed
        sample = input_ids[b]
        full_len = sample.shape[0]
        valid_len = input_lengths[b].item() if input_lengths is not None else full_len

        keep_mask = torch.ones(full_len, dtype=torch.bool, device=input_ids.device)
        for start_pos in (sample == img_start_id).nonzero(as_tuple=True)[0]:
            end_matches = (sample[start_pos:] == img_end_id).nonzero(as_tuple=True)[0]
            if len(end_matches) == 0:
                raise ValueError(
                    "Malformed multimodal token sequence: found <img> token without a "
                    f"matching </img> token (batch_index={b}, start_pos={start_pos.item()})."
                )
            end_pos = end_matches[0] + start_pos
            keep_mask[start_pos + 2 : end_pos] = False

        collapsed_list.append(sample[keep_mask])
        all_keep_masks.append(keep_mask)
        tokens_removed = full_len - keep_mask.sum().item()
        tokens_removed_per_sample.append(tokens_removed)
        # Actual content length = original content - tokens removed (from content region)
        new_lengths.append(valid_len - tokens_removed)

    max_collapsed_len = max(len(c) for c in collapsed_list)
    collapsed_ids = torch.zeros(
        batch_size, max_collapsed_len, dtype=input_ids.dtype, device=input_ids.device
    )
    for b, collapsed in enumerate(collapsed_list):
        collapsed_ids[b, : len(collapsed)] = collapsed

    new_data_dict = data_dict.copy()
    new_data_dict["input_ids"] = collapsed_ids
    if input_lengths is not None:
        new_data_dict["input_lengths"] = torch.tensor(
            new_lengths, dtype=input_lengths.dtype, device=input_lengths.device
        )
    stored_tokens_removed_per_sample = list(tokens_removed_per_sample)

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    if not getattr(inner, "_dynamic_resolution", True):
        static_img_seq_len = getattr(inner, "img_seq_len", None)
        if static_img_seq_len is not None:
            # Preserve the local physical removal count for collapsed lengths,
            # but store the fixed static-resolution expansion delta because
            # downstream packing/SP code already consumes this tensor in
            # expansion space.
            static_img_seq_len = int(static_img_seq_len)
            for b in range(batch_size):
                num_images = int((collapsed_ids[b] == img_start_id).sum().item())
                stored_tokens_removed_per_sample[b] = (
                    num_images * (static_img_seq_len - 1)
                )

    new_data_dict["tokens_removed_per_sample"] = torch.tensor(
        stored_tokens_removed_per_sample, dtype=torch.int64, device=input_ids.device
    )
    new_data_dict["_collapse_keep_mask"] = torch.stack(all_keep_masks)
    new_data_dict["vision_expansion_per_sample"] = torch.tensor(
        _compute_vision_expansion_per_sample(collapsed_ids, data_dict, model),
        dtype=torch.int64,
        device=input_ids.device,
    )

    return new_data_dict


def _get_image_token_ids(model) -> Optional[tuple[int, int]]:
    """Extract <img> and </img> token IDs from Megatron model."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    for obj in [inner, getattr(inner, "config", None)]:
        if obj is None:
            continue
        start = getattr(obj, "img_start_token_id", None)
        end = getattr(obj, "img_end_token_id", None)
        if start is not None and end is not None:
            return start, end
    return None


def _get_sound_token_index(model) -> Optional[int]:
    """Extract the sound placeholder token index from Megatron model."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model
    return getattr(inner, "sound_token_index", None)


def _get_sound_feature_extractor(model):
    """Get the FastConformer feature extractor from the model's sound config."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model
    sound_model = getattr(inner, "sound_model", None)
    if sound_model is None:
        return None
    sound_config = getattr(sound_model, "config", None)
    if sound_config is None:
        return None
    from megatron.core.models.huggingface.fastconformer.feature_extraction_fastconformer import (
        FastConformerFeatureExtractor,
    )
    num_mel_bins = getattr(sound_config, "num_mel_bins", 128)
    sampling_rate = getattr(sound_config, "sampling_rate", 16000)
    hop_length = getattr(sound_config, "hop_length", 160)
    win_length = getattr(sound_config, "win_length", 400)
    n_fft = getattr(sound_config, "n_fft", 512)
    return FastConformerFeatureExtractor(
        feature_size=num_mel_bins,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
    )


def prepare_multimodal_data(multimodal_data: dict, model, device: torch.device) -> None:
    """Prepare pixel_values and sound_clips for Megatron forward pass."""
    _prepare_image_data(multimodal_data, model, device)
    _prepare_sound_data(multimodal_data, model, device)


def _prepare_image_data(multimodal_data: dict, model, device: torch.device) -> None:
    """Prepare pixel_values for Megatron forward (patchification for dynamic resolution)."""
    if "pixel_values" not in multimodal_data:
        # LLaVAModel requires images, imgs_sizes, and num_image_tiles; pass empty tensors
        # num_image_tiles must be empty to match images count, even if input_ids has image tokens
        multimodal_data["images"] = torch.empty(0, dtype=torch.bfloat16, device=device)
        multimodal_data["imgs_sizes"] = torch.empty(0, 2, dtype=torch.int32, device=device)
        multimodal_data["num_image_tiles"] = torch.empty(0, dtype=torch.int, device=device)
        return

    images = multimodal_data.pop("pixel_values").to(torch.bfloat16)

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    dynamic_res = getattr(inner, "_dynamic_resolution", False)
    has_imgs_sizes = "imgs_sizes" in multimodal_data
    imgs_sizes = multimodal_data.get("imgs_sizes")

    if dynamic_res and has_imgs_sizes:
        patch_dim = getattr(inner.vision_model, "patch_dim", 16)
        # imgs_sizes contains actual pixel dimensions for cropping
        # RADIO uses these to compute patch counts for position encoding
        # LLaVAModel._preprocess_data applies pixel_shuffle reduction internally
        images, num_tiles, vision_params = _patchify_for_dynamic_resolution(
            images, multimodal_data["imgs_sizes"], patch_dim
        )
        multimodal_data["num_image_tiles"] = num_tiles
        multimodal_data["vision_packed_seq_params"] = vision_params

        # When temporal compression is enabled (video_temporal_patch_size > 1),
        # RADIO requires num_frames to distinguish images (1 frame) from videos.
        # For image-only data, default to 1 frame per image.
        temporal_patch_size = getattr(inner, "_video_temporal_patch_size", 1)
        if temporal_patch_size > 1 and "num_frames" not in multimodal_data:
            num_images = len(multimodal_data["imgs_sizes"])
            multimodal_data["num_frames"] = torch.ones(num_images, dtype=torch.int32, device=device)
    elif dynamic_res and not has_imgs_sizes:
        raise AssertionError(
            "dynamic_resolution=True but imgs_sizes not provided in multimodal_data. "
            "The data pipeline must supply imgs_sizes when dynamic_resolution is enabled, "
            "otherwise the model output length will not match the input length."
        )

    multimodal_data["images"] = images


def _prepare_sound_data(multimodal_data: dict, model, device: torch.device) -> None:
    """Prepare sound_clips for Megatron forward.

    Raw waveforms are split from the flat concatenated tensor, padded into a 2D batch,
    then converted to log-mel spectrograms via FastConformerFeatureExtractor.
    The BridgeSoundEncoder expects mel features [batch, frames, mel_bins], not raw audio.
    """
    if "sound_clips" not in multimodal_data:
        return

    flat_waveform = multimodal_data.pop("sound_clips")
    lengths = multimodal_data.pop("sound_length")

    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int32)
    lengths = lengths.to(device)

    if lengths.numel() == 0 or flat_waveform.numel() == 0:
        return

    clips = torch.split(flat_waveform, lengths.tolist())
    max_len = int(lengths.max().item())
    num_clips = len(clips)
    padded = torch.zeros(num_clips, max_len, dtype=torch.float32, device=device)
    for i, clip in enumerate(clips):
        padded[i, : clip.shape[0]] = clip.to(dtype=torch.float32, device=device)

    feature_extractor = _get_sound_feature_extractor(model)
    if feature_extractor is not None:
        result = feature_extractor(
            raw_speech=padded,
            audio_lengths=lengths.long(),
            sampling_rate=feature_extractor.sampling_rate,
            device=str(device),
        )
        mel_features = result["input_features"].to(dtype=torch.bfloat16, device=device)
        hop_length = feature_extractor.hop_length
        mel_lengths = torch.tensor(
            [int(wl) // hop_length for wl in lengths.tolist()],
            dtype=torch.int32, device=device,
        )
        max_canonical_len = int(mel_lengths.max().item())
        mel_features = mel_features[:, :max_canonical_len, :]
        multimodal_data["sound_clips"] = mel_features
        multimodal_data["sound_length"] = mel_lengths
    else:
        multimodal_data["sound_clips"] = padded.to(dtype=torch.bfloat16)
        multimodal_data["sound_length"] = lengths


def _patchify_for_dynamic_resolution(
    images: torch.Tensor,
    imgs_sizes: torch.Tensor,
    patch_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, PackedSeqParams]:
    """Convert images to packed patches for dynamic resolution RADIO vision encoder."""

    def to_patches(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        img = img[:, :h, :w]
        py, px = h // patch_dim, w // patch_dim
        return rearrange(
            img, "c (py yy) (px xx) -> (py px) (c yy xx)", py=py, yy=patch_dim, px=px, xx=patch_dim
        )

    patches_list = [to_patches(img, *imgs_sizes[i].tolist()) for i, img in enumerate(images)]

    cu_seqlens = [0]
    for p in patches_list:
        cu_seqlens.append(cu_seqlens[-1] + p.shape[0])

    max_seqlen = max(p.shape[0] for p in patches_list)
    return (
        torch.cat(patches_list, dim=0).unsqueeze(0),
        torch.ones(len(images), dtype=torch.int, device=images.device),
        PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=images.device),
            cu_seqlens_kv=torch.tensor(cu_seqlens, dtype=torch.int32, device=images.device),
            max_seqlen_q=torch.tensor(max_seqlen, dtype=torch.int32, device=images.device),
            max_seqlen_kv=torch.tensor(max_seqlen, dtype=torch.int32, device=images.device),
        ),
    )


def remap_expanded_logits_to_collapsed(
    expanded_logits: torch.Tensor,
    collapsed_input_ids: torch.Tensor,
    model,
    multimodal_data: dict,
) -> torch.Tensor:
    """Map Megatron's image-expanded logits back to collapsed token positions."""
    image_token_index = _get_image_token_index(model)
    if image_token_index is None:
        return expanded_logits

    batch_size, collapsed_len = collapsed_input_ids.shape
    expanded_len = expanded_logits.shape[1]
    if expanded_len == collapsed_len:
        return expanded_logits

    imgs_sizes = multimodal_data.get("imgs_sizes")
    if imgs_sizes is None or imgs_sizes.numel() == 0:
        return expanded_logits

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    patch_dim = getattr(inner.vision_model, "patch_dim", 16)
    pixel_shuffle = getattr(inner, "_pixel_shuffle", False)
    conv_merging = getattr(inner, "_use_conv_merging", False)
    drop_cls = getattr(inner, "_drop_vision_class_token", True)
    cls_len = 0 if drop_cls else getattr(inner, "_class_token_len", 1)

    per_img_embeds = torch.prod(
        imgs_sizes // patch_dim, dim=-1, dtype=torch.int32
    ) + cls_len
    if pixel_shuffle:
        per_img_embeds = (per_img_embeds * (0.5 ** 2)).int()
    if conv_merging:
        per_img_embeds = (per_img_embeds * (0.5 ** 2)).int()

    result_list = []
    image_offset = 0
    for b in range(batch_size):
        ids = collapsed_input_ids[b]
        img_positions = (ids == image_token_index).nonzero(as_tuple=True)[0]

        if len(img_positions) == 0:
            result_list.append(expanded_logits[b, :collapsed_len])
            continue

        mapping = torch.arange(collapsed_len, device=ids.device, dtype=torch.long)
        for local_img_idx, img_pos in enumerate(img_positions):
            embed_idx = image_offset + local_img_idx
            if embed_idx >= len(per_img_embeds):
                break
            expansion = int(per_img_embeds[embed_idx].item()) - 1
            mapping[img_pos:] += expansion
        image_offset += len(img_positions)

        mapping = mapping.clamp(0, expanded_len - 1)
        result_list.append(expanded_logits[b, mapping])

    return torch.stack(result_list)
