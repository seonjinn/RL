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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
)
from megatron.core.utils import StragglerDetector
from megatron.training.utils import get_ltor_masks_and_position_ids

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
from nemo_rl.models.megatron.common import (
    _round_up_to_multiple,
    _vlm_sp_repad_collapsed,
)
from nemo_rl.models.megatron.multimodal import (
    collapse_multimodal_tokens,
    is_llava_model,
)


@dataclass
class ProcessedInputs:
    """Processed microbatch inputs used for model forward pass."""

    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    cu_seqlens_padded: Optional[torch.Tensor]
    mtp_loss_mask: Optional[torch.Tensor] = None
    use_llava_handoff: bool = False


@dataclass
class ProcessedMicrobatch:
    """Container for a processed microbatch ready for model forward pass.

    This dataclass holds both the original data dictionary and the processed
    tensors needed for the Megatron model forward pass.

    Attributes:
        data_dict: The original BatchedDataDict containing raw batch data
        input_ids: Processed input token IDs (may be packed for sequence packing)
        input_ids_cp_sharded: Context-parallel sharded input token IDs
        attention_mask: Attention mask tensor (None for packed sequences)
        position_ids: Position IDs tensor (None for packed sequences)
        packed_seq_params: PackedSeqParams for sequence packing (None if not packing)
        cu_seqlens_padded: Padded cumulative sequence lengths (None if not packing)
        use_llava_handoff: True when the model is a LLaVA-style model, in which case
            ``input_ids`` (full, unsharded) must be forwarded to the model and CP
            sharding is applied internally by ``LLaVAModel._preprocess_data``.
    """

    data_dict: BatchedDataDict[Any]
    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    cu_seqlens_padded: Optional[torch.Tensor]
    mtp_loss_mask: Optional[torch.Tensor] = None
    use_llava_handoff: bool = False


def make_processed_microbatch_iterator(
    raw_iterator: Iterator[BatchedDataDict[Any]],
    cfg: dict[str, Any],
    seq_length_key: Optional[str],
    pad_individual_seqs_to_multiple_of: int,
    pad_packed_seq_to_multiple_of: int,
    straggler_timer: StragglerDetector,
    pad_full_seq_to: Optional[int],
    model: Any = None,
) -> Iterator[ProcessedMicrobatch]:
    """Wrap a raw microbatch iterator to yield processed microbatches.

    This function takes a raw iterator that yields BatchedDataDict objects and
    wraps it to yield ProcessedMicrobatch objects that contain both the original
    data and the processed tensors ready for model forward pass.

    Args:
        raw_iterator: Iterator yielding raw BatchedDataDict microbatches
        cfg: Configuration dictionary containing sequence_packing settings
        seq_length_key: Key for sequence length in data dict (required for packing)
        pad_individual_seqs_to_multiple_of: Padding multiple for individual sequences
        pad_packed_seq_to_multiple_of: Padding multiple for packed sequences
        pad_full_seq_to: Target length for full sequence padding (optional)
        model: Optional Megatron model. When provided and detected as a LLaVA-style
            model (``is_llava_model``), the raw microbatch is run through
            ``collapse_multimodal_tokens`` before packing/CP sharding so that the
            packed sequence layout matches the collapsed token space the LLaVA
            ``_preprocess_data`` step expects. For text-only or non-LLaVA models
            this is a no-op and the original behaviour is preserved.

    Yields:
        ProcessedMicrobatch objects containing processed tensors ready for model forward
    """
    pack_sequences = cfg["sequence_packing"]["enabled"]
    use_llava_handoff = bool(model is not None and is_llava_model(model))

    for data_dict in raw_iterator:
        data_dict = data_dict.to("cuda")

        # VLM token collapse: shrink N image tokens per image to 1 collapsed slot
        # so packing/CP shard math operates in the collapsed (vLLM-style) length
        # space. The model re-expands internally via ``LLaVAModel._preprocess_data``.
        # Safe to call unconditionally: returns ``data_dict`` unchanged for
        # text-only or non-LLaVA models.
        if model is not None:
            data_dict = collapse_multimodal_tokens(data_dict, model)

        tokens_removed_per_sample = data_dict.pop("tokens_removed_per_sample", None)

        processed_inputs = process_microbatch(
            data_dict=data_dict,
            seq_length_key=seq_length_key,
            pad_individual_seqs_to_multiple_of=pad_individual_seqs_to_multiple_of,
            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
            pad_full_seq_to=pad_full_seq_to,
            pack_sequences=pack_sequences,
            straggler_timer=straggler_timer,
            tokens_removed_per_sample=tokens_removed_per_sample,
            use_llava_handoff=use_llava_handoff,
            policy_cfg=cfg,
        )

        yield ProcessedMicrobatch(
            data_dict=data_dict,
            input_ids=processed_inputs.input_ids,
            input_ids_cp_sharded=processed_inputs.input_ids_cp_sharded,
            attention_mask=processed_inputs.attention_mask,
            position_ids=processed_inputs.position_ids,
            packed_seq_params=processed_inputs.packed_seq_params,
            cu_seqlens_padded=processed_inputs.cu_seqlens_padded,
            mtp_loss_mask=processed_inputs.mtp_loss_mask,
            use_llava_handoff=processed_inputs.use_llava_handoff,
        )


def get_microbatch_iterator(
    data: BatchedDataDict[Any],
    cfg: dict[str, Any],
    mbs: int,
    straggler_timer: StragglerDetector,
    seq_length_key: Optional[str] = None,
    model: Any = None,
) -> Tuple[Iterator[ProcessedMicrobatch], int, int, int, int]:
    """Create a processed microbatch iterator from a batch of data.

    This function creates an iterator that yields ProcessedMicrobatch objects,
    which contain both the original data dictionary and the processed tensors
    ready for model forward pass.

    Args:
        data: The batch data to create microbatches from
        cfg: Configuration dictionary
        mbs: Microbatch size
        seq_length_key: Key for sequence lengths in data dict (auto-detected if None)

    Returns:
        Tuple containing the iterator and metadata
        - iterator: Iterator yielding ProcessedMicrobatch objects
        - data_iterator_len: Number of microbatches in the iterator
        - micro_batch_size: Size of each microbatch
        - seq_dim_size: Sequence length dimension size
        - padded_seq_length: Padded sequence length for pipeline parallelism (may differ from seq_length)
    """
    micro_batch_size = mbs
    pad_factor = 1
    pad_full_seq_to = None
    pad_packed_seq_to_multiple_of = 1

    _, seq_dim_size = get_and_validate_seqlen(data)

    # Auto-detect seq_length_key if not provided
    if seq_length_key is None and cfg["sequence_packing"]["enabled"]:
        seq_length_key = "input_lengths"

    if cfg["dynamic_batching"]["enabled"]:
        raw_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
        data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
    elif cfg["sequence_packing"]["enabled"]:
        raw_iterator = data.make_microbatch_iterator_for_packable_sequences()
        data_iterator_len, pack_seq_dim_size = (
            data.get_microbatch_iterator_for_packable_sequences_len()
        )
        (
            pad_factor,
            pad_packed_seq_to_multiple_of,
            pad_full_seq_to,
        ) = _get_pack_sequence_parameters_for_megatron(
            cfg["megatron_cfg"],
            pack_seq_dim_size,
        )
        micro_batch_size = 1
    else:
        raw_iterator = data.make_microbatch_iterator(mbs)
        data_iterator_len = data.size // mbs

    # Wrap the raw iterator with processing
    processed_iterator = make_processed_microbatch_iterator(
        raw_iterator=raw_iterator,
        cfg=cfg,
        seq_length_key=seq_length_key,
        pad_individual_seqs_to_multiple_of=pad_factor,
        pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
        pad_full_seq_to=pad_full_seq_to,
        straggler_timer=straggler_timer,
        model=model,
    )

    # Compute padded sequence length for pipeline parallelism
    padded_seq_length = pad_full_seq_to if pad_full_seq_to is not None else seq_dim_size

    return (
        processed_iterator,
        data_iterator_len,
        micro_batch_size,
        seq_dim_size,
        padded_seq_length,
    )


def process_microbatch(
    data_dict: BatchedDataDict[Any],
    seq_length_key: Optional[str] = None,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_full_seq_to: Optional[int] = None,
    pack_sequences: bool = False,
    straggler_timer: Optional[StragglerDetector] = None,
    tokens_removed_per_sample: Optional[torch.Tensor] = None,
    use_llava_handoff: bool = False,
    policy_cfg: Optional[dict[str, Any]] = None,
) -> ProcessedInputs:
    """Process a microbatch for Megatron model forward pass.

    Args:
        data_dict: The (possibly multimodal-collapsed) microbatch data.
        seq_length_key: Key in ``data_dict`` containing actual sequence lengths.
        pad_individual_seqs_to_multiple_of: Per-sequence padding factor for packing.
        pad_packed_seq_to_multiple_of: Packed-sequence padding factor.
        pad_full_seq_to: Optional total length the packed sequence is padded to.
        pack_sequences: Whether to pack sequences for sequence-packed training.
        straggler_timer: Optional Megatron straggler detector.
        tokens_removed_per_sample: Per-sample count of image tokens removed by
            ``collapse_multimodal_tokens``. Used to (a) track expanded sequence
            boundaries that the LLaVA model will see internally and (b) re-pad
            collapsed input ids for sequence-parallel alignment when SP is on.
            ``None`` for text-only or non-LLaVA paths.
        use_llava_handoff: True when the model is a LLaVA-style model. In that
            case the model expects the unsharded ``input_ids`` and applies CP
            sharding internally via ``_preprocess_data``.
        policy_cfg: Optional policy config used to look up ``sequence_parallel``
            and ``tensor_model_parallel_size`` for VLM SP repadding.
    """
    ctx = straggler_timer(bdata=True) if straggler_timer is not None else nullcontext()
    with ctx:
        input_ids = data_dict["input_ids"]
        attention_mask = None
        position_ids = None
        packed_seq_params = None

        original_batch_size = input_ids.shape[0]
        original_seq_length = input_ids.shape[1]
        seq_lengths = None  # Will be set if using packed sequences
        cu_seqlens = None
        cu_seqlens_padded = None
        mtp_loss_mask = None

        if pack_sequences:
            # For packed sequences with padded input, we need sequence lengths
            assert seq_length_key is not None, (
                "seq_length_key must be provided for packed sequences"
            )
            assert seq_length_key in data_dict, (
                f"{seq_length_key} not found in data_dict"
            )

            # Get sequence lengths and context parallel size
            seq_lengths = data_dict[seq_length_key]

            # Pack sequences
            (
                input_ids,
                input_ids_cp_sharded,
                packed_seq_params,
                cu_seqlens,
                cu_seqlens_padded,
            ) = _pack_sequences_for_megatron(
                input_ids,
                seq_lengths,
                pad_individual_seqs_to_multiple_of,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
                cp_rank=get_context_parallel_rank(),
                cp_size=get_context_parallel_world_size(),
                tokens_removed_per_sample=tokens_removed_per_sample,
                skip_local_cp_sharding=use_llava_handoff,
            )

            # VLM + sequence packing: the LLaVA model re-expands collapsed image
            # tokens internally during ``_preprocess_data``, so the attention
            # mechanism must see the *expanded* cu_seqlens for every sequence in
            # the pack, not just the trailing one. This mirrors Omni's bookkeeping
            # in ``forward_step_arbitrary_loss``.
            if tokens_removed_per_sample is not None and cu_seqlens_padded is not None:
                n_seqs = cu_seqlens_padded.shape[0] - 1
                cumulative_removed = torch.zeros(
                    n_seqs + 1,
                    dtype=torch.int32,
                    device=cu_seqlens_padded.device,
                )
                cumulative_removed[1:] = (
                    tokens_removed_per_sample[:n_seqs].to(torch.int32).cumsum(0)
                )
                cu_seqlens_padded_expanded = (
                    cu_seqlens_padded.clone() + cumulative_removed
                )
                # Single clone aliased into all four PackedSeqParams fields:
                # TE relies on ``cu_seqlens_q is cu_seqlens_kv`` identity to
                # detect self-attention. Distinct clones break that.
                cu_seqlens_for_attn = cu_seqlens_padded_expanded.clone()
                packed_seq_params.cu_seqlens_q = cu_seqlens_for_attn
                packed_seq_params.cu_seqlens_kv = cu_seqlens_for_attn
                packed_seq_params.cu_seqlens_q_padded = cu_seqlens_for_attn
                packed_seq_params.cu_seqlens_kv_padded = cu_seqlens_for_attn
                expanded_slot_lengths = (
                    cu_seqlens_padded_expanded[1:] - cu_seqlens_padded_expanded[:-1]
                )
                packed_seq_params.max_seqlen_q = int(
                    expanded_slot_lengths.max().item()
                )
                packed_seq_params.max_seqlen_kv = int(
                    expanded_slot_lengths.max().item()
                )

            # Pack pre-computed mtp_loss_mask the same way as input_ids
            if "mtp_loss_mask" in data_dict:
                (
                    _,
                    mtp_loss_mask,
                    _,
                    _,
                    _,
                ) = _pack_sequences_for_megatron(
                    data_dict["mtp_loss_mask"],
                    seq_lengths,
                    pad_individual_seqs_to_multiple_of,
                    pad_packed_seq_to_multiple_of,
                    pad_full_seq_to,
                    cp_rank=get_context_parallel_rank(),
                    cp_size=get_context_parallel_world_size(),
                )

            # For packed sequences, position_ids and attention_mask are typically None
            # The PackedSeqParams handles all necessary sequence information
            position_ids = None
            attention_mask = None
        else:
            # VLM + (sequence-parallel or context-parallel) without packing:
            # LLaVA re-expands collapsed image tokens internally, producing a
            # wider activation tensor that must be divisible by the active
            # ``_calc_shard_factor`` -- ``tp_size`` (for SP), ``cp_size * 2``
            # (for CP load balancing), or their LCM when both are on. Re-pad
            # the collapsed input_ids so that the LLaVA-expanded length aligns.
            if policy_cfg is not None and tokens_removed_per_sample is not None:
                megatron_cfg = policy_cfg.get("megatron_cfg", {}) or {}
                sp = bool(megatron_cfg.get("sequence_parallel", False))
                tp_size = int(megatron_cfg.get("tensor_model_parallel_size", 1))
                cp_size = int(megatron_cfg.get("context_parallel_size", 1))
                divisor = 1
                if sp and tp_size > 1:
                    divisor = tp_size
                if cp_size > 1:
                    cp_factor = cp_size * 2
                    if divisor > 1:
                        from math import gcd
                        divisor = (divisor * cp_factor) // gcd(divisor, cp_factor)
                    else:
                        divisor = cp_factor
                if divisor > 1:
                    input_ids = _vlm_sp_repad_collapsed(
                        input_ids, tokens_removed_per_sample, divisor
                    )
                    data_dict["input_ids"] = input_ids
            input_ids_cp_sharded = input_ids
            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                data=input_ids,
                eod_token=0,  # used for loss_mask, which we don't use
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
            )
            if "mtp_loss_mask" in data_dict:
                mtp_loss_mask = data_dict["mtp_loss_mask"]
    return ProcessedInputs(
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids_cp_sharded,
        attention_mask=attention_mask,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
        cu_seqlens_padded=cu_seqlens_padded,
        mtp_loss_mask=mtp_loss_mask,
        use_llava_handoff=use_llava_handoff,
    )


def process_global_batch(
    data: BatchedDataDict[Any],
    loss_fn: LossFunction,
    dp_group: torch.distributed.ProcessGroup,
    *,
    batch_idx: int,
    batch_size: int,
) -> dict[str, Any]:
    """Process a global batch and compute normalization factors.

    Args:
        data: Full dataset to extract a batch from
        loss_fn: Loss function (used to check loss type for token-level validation)
        dp_group: Data parallel process group for all-reduce
        batch_idx: Index of batch to extract
        batch_size: Size of batch to extract

    Returns:
        Dictionary containing:
        - batch: The extracted batch
        - global_valid_seqs: Number of valid sequences across all ranks
        - global_valid_toks: Number of valid tokens across all ranks
    """
    batch = data.get_batch(batch_idx=batch_idx, batch_size=batch_size)

    assert "sample_mask" in batch, "sample_mask must be present in the data!"

    # Get the normalization factor for the loss
    local_valid_seqs = torch.sum(batch["sample_mask"])

    if "token_mask" not in batch:
        local_valid_toks = local_valid_seqs * batch["input_ids"].shape[1]
    else:
        local_valid_toks = torch.sum(
            batch["token_mask"][:, 1:] * batch["sample_mask"].unsqueeze(-1)
        )

    to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
    torch.distributed.all_reduce(to_reduce, group=dp_group)
    global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

    if hasattr(loss_fn, "loss_type") and loss_fn.loss_type == LossType.TOKEN_LEVEL:
        assert "token_mask" in batch, (
            "token_mask must be present in the data when using token-level loss"
        )

    return {
        "batch": batch,
        "global_valid_seqs": global_valid_seqs,
        "global_valid_toks": global_valid_toks,
    }


def _pack_sequences_for_megatron(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_packed_seq_to: Optional[int] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    tokens_removed_per_sample: Optional[torch.Tensor] = None,
    skip_local_cp_sharding: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    PackedSeqParams,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """Pack sequences for Megatron model processing with optional context parallelism.

    Args:
        input_ids: Input token IDs [batch_size, seq_length]
        seq_lengths: Actual sequence lengths for each sample [batch_size]
        pad_individual_seqs_to_multiple_of: Pad individual sequences to a multiple of this value
        pad_packed_seq_to_multiple_of: Pad packed sequences to a multiple of this value
        pad_packed_seq_to: Pad packed sequences to this value (before CP)
            - The three parameters above can be calculated using _get_pack_sequence_parameters_for_megatron, we do not recommend users to set these parameters manually.
        cp_size: Context parallelism size
        tokens_removed_per_sample: Per-sample count of image tokens removed by
            ``collapse_multimodal_tokens``. When provided, per-sequence padding
            ensures that expanded lengths (collapsed + removed) are multiples
            of ``pad_individual_seqs_to_multiple_of``. ``None`` for non-VLM paths.
            Currently raises ``NotImplementedError`` if combined with PP+CP>1.
        skip_local_cp_sharding: If True, return the unsharded packed input as
            ``input_ids_cp_sharded`` so the LLaVA model can apply CP sharding
            internally after rebuilding embeddings.

    Returns:
        Tuple of:
        - packed_input_ids: Packed input tensor [1, T]
        - input_ids_cp_sharded: Sharded input tensor [cp_size, T // cp_size]
        - packed_seq_params: PackedSeqParams object
        - cu_seqlens: Cumulative sequence lengths
        - cu_seqlens_padded: Padded cumulative sequence lengths
    """
    if (
        tokens_removed_per_sample is not None
        and pad_packed_seq_to is not None
        and cp_size > 1
    ):
        raise NotImplementedError(
            "PP > 1 with VLM sequence packing and CP > 1 is not yet supported. "
            "Per-microbatch VLM token expansion produces variable expanded "
            "lengths that break PP's uniform sequence-length requirement."
        )

    if tokens_removed_per_sample is not None:
        assert tokens_removed_per_sample.shape[0] >= input_ids.shape[0], (
            f"tokens_removed_per_sample has {tokens_removed_per_sample.shape[0]} "
            f"entries but batch_size is {input_ids.shape[0]}"
        )

    batch_size = input_ids.shape[0]
    needs_padding = (
        pad_individual_seqs_to_multiple_of > 1
        or pad_packed_seq_to_multiple_of > 1
        or pad_packed_seq_to is not None
    )

    if pad_packed_seq_to is not None:
        assert pad_packed_seq_to % pad_packed_seq_to_multiple_of == 0, (
            f"pad_packed_seq_to ({pad_packed_seq_to}) is not a multiple of "
            f"pad_packed_seq_to_multiple_of ({pad_packed_seq_to_multiple_of})."
        )

    pad_factor = pad_individual_seqs_to_multiple_of

    # --- Loop 1: build cu_seqlens (collapsed) and padded_seq_lens (collapsed,
    # but VLM-aware so the *expanded* length is divisible by pad_factor). The
    # padded_seq_lens list is shared with Loop 2 to avoid recomputation drift.
    cu_seqlens = [0]
    valid_tokens: list[torch.Tensor] = []
    padded_seq_lens: list[int] = []
    for b in range(batch_size):
        seq_len = (
            seq_lengths[b].item() if torch.is_tensor(seq_lengths[b]) else seq_lengths[b]
        )
        valid_tokens.append(input_ids[b, :seq_len])
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
        if needs_padding:
            removed = (
                int(tokens_removed_per_sample[b].item())
                if tokens_removed_per_sample is not None
                else 0
            )
            # VLM-aware: pad collapsed length so (collapsed_padded + removed)
            # is a multiple of pad_factor. removed=0 degenerates to the
            # standard text-only formula.
            padded_seq_len = (
                _round_up_to_multiple(seq_len + removed, pad_factor) - removed
            )
            padded_seq_lens.append(padded_seq_len)

    # --- Post-loop: adjust the last sequence so the packed-total alignment
    # holds (PP requires fixed total via pad_packed_seq_to; FP8 requires the
    # total to be a multiple of pad_packed_seq_to_multiple_of).
    if needs_padding and batch_size > 0:
        running = sum(padded_seq_lens[:-1]) if batch_size > 1 else 0
        if pad_packed_seq_to is not None:
            # PP > 1: target collapsed total is fixed. VLM+CP>1 is guarded
            # above, so this is non-VLM or CP=1 VLM.
            padded_seq_lens[-1] = pad_packed_seq_to - running
        elif pad_packed_seq_to_multiple_of > 1:
            if tokens_removed_per_sample is not None:
                # VLM + FP8: align total in *expanded* space, then derive
                # collapsed padding.
                running_removed = (
                    sum(
                        int(tokens_removed_per_sample[b].item())
                        for b in range(batch_size - 1)
                    )
                    if batch_size > 1
                    else 0
                )
                running_expanded = running + running_removed
                last_removed = int(
                    tokens_removed_per_sample[batch_size - 1].item()
                )
                last_expanded = padded_seq_lens[-1] + last_removed
                total_expanded = _round_up_to_multiple(
                    running_expanded + last_expanded,
                    pad_packed_seq_to_multiple_of,
                )
                padded_seq_lens[-1] = (
                    total_expanded - running_expanded - last_removed
                )
            else:
                # Non-VLM: align collapsed total (original behaviour).
                current = padded_seq_lens[-1]
                new_total = _round_up_to_multiple(
                    running + current, pad_packed_seq_to_multiple_of
                )
                padded_seq_lens[-1] = new_total - running

    # --- Build cu_seqlens_padded from padded_seq_lens ---
    cu_seqlens_padded = None
    if needs_padding:
        cu_seqlens_padded_list = [0]
        for psl in padded_seq_lens:
            cu_seqlens_padded_list.append(cu_seqlens_padded_list[-1] + psl)
        cu_seqlens_padded = torch.tensor(
            cu_seqlens_padded_list, dtype=torch.int32, device=input_ids.device
        )

    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=input_ids.device)

    # --- VLM assertion: verify expanded slot alignment ---
    if tokens_removed_per_sample is not None and pad_factor > 1 and needs_padding:
        for b in range(batch_size):
            removed = int(tokens_removed_per_sample[b].item())
            expanded_slot = padded_seq_lens[b] + removed
            assert expanded_slot % pad_factor == 0, (
                f"[VLM-pack] expanded slot {b} = {expanded_slot} "
                f"(collapsed_padded={padded_seq_lens[b]}, removed={removed}) "
                f"not aligned to pad_factor={pad_factor}"
            )

    # --- Calculate max sequence length (padded if using CP/SP) ---
    if needs_padding:
        seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
        max_seqlen = seq_lens_padded.max().item()
    else:
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

    # --- Loop 2: build padded token tensors. Uses padded_seq_lens[b] from
    # Loop 1 verbatim so both loops agree on per-sequence padding.
    if pad_factor > 1:
        all_input_ids = []
        padded_tokens = []
        for b in range(batch_size):
            seq_len = (
                seq_lengths[b].item()
                if torch.is_tensor(seq_lengths[b])
                else seq_lengths[b]
            )
            padded_seq_len = padded_seq_lens[b]
            seq_tokens = input_ids[b, :seq_len]
            if padded_seq_len > seq_len:
                seq_tokens = torch.nn.functional.pad(
                    seq_tokens, (0, padded_seq_len - seq_len), value=0
                )
            all_input_ids.append(seq_tokens)

            # Skip local CP sharding when (a) the downstream model will shard
            # after rebuilding embeddings (LLaVA handoff) or (b) VLM has
            # already moved alignment decisions into expanded space, in which
            # case CP sharding is the model's responsibility.
            if (
                cp_size > 1
                and not skip_local_cp_sharding
                and tokens_removed_per_sample is None
            ):
                seq_tokens = _get_tokens_on_this_cp_rank(
                    seq_tokens, cp_rank, cp_size, seq_dim=0
                )

            padded_tokens.append(seq_tokens)

        packed_input_ids = torch.cat(padded_tokens, dim=0).unsqueeze(0)
        all_input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(0)
    else:
        # No individual padding, just concatenate valid tokens
        # For 'thd' format, the shape should be [1, T] where T is total tokens
        packed_input_ids = torch.cat(valid_tokens, dim=0).unsqueeze(0)
        all_input_ids = packed_input_ids
        if needs_padding:
            if pad_packed_seq_to is not None:
                pad_len = pad_packed_seq_to - packed_input_ids.shape[1]
            elif pad_packed_seq_to_multiple_of > 1:
                current_seq_len = packed_input_ids.shape[1]
                pad_this_seq_to = _round_up_to_multiple(
                    current_seq_len, pad_packed_seq_to_multiple_of
                )
                pad_len = pad_this_seq_to - current_seq_len
            else:
                pad_len = 0
            if pad_len > 0:
                packed_input_ids = torch.nn.functional.pad(
                    packed_input_ids, (0, pad_len), value=0
                )
                all_input_ids = torch.nn.functional.pad(
                    all_input_ids, (0, pad_len), value=0
                )

    if cu_seqlens_padded is None:
        cu_seqlens_padded = cu_seqlens.clone()

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=int(max_seqlen),
        max_seqlen_kv=int(max_seqlen),
        qkv_format="thd",
    )

    return (
        all_input_ids.contiguous(),
        packed_input_ids.contiguous(),
        packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
    )


def _get_pack_sequence_parameters_for_megatron(
    megatron_cfg: dict,
    max_seq_len_in_batch: int,
):
    """Get pack sequence parameters for Megatron model processing with optional context parallelism.

    Args:
        megatron_cfg: Megatron configuration
        max_seq_len_in_batch: Maximum sequence length in batch

    Returns:
        Tuple of:
        - pad_individual_seqs_to_multiple_of: Pad individual sequences to a multiple of this value
        - pad_packed_seq_to_multiple_of: Pad packed sequences to a multiple of this value
        - pad_packed_seq_to: Pad packed sequences to this value (before CP)
    """
    tp_size = megatron_cfg["tensor_model_parallel_size"]
    sp = megatron_cfg["sequence_parallel"]
    pp_size = megatron_cfg["pipeline_model_parallel_size"]
    cp_size = megatron_cfg["context_parallel_size"]
    fp8_cfg = megatron_cfg.get("fp8_cfg", None) or {}
    use_fp8 = fp8_cfg.get("enabled", False)

    # individual sequence needs to be splitted to CP domain, and to TP domain when SP is enabled.
    pad_individual_seqs_to_multiple_of = 1
    if cp_size > 1:
        pad_individual_seqs_to_multiple_of *= cp_size * 2
    if tp_size > 1 and sp:
        pad_individual_seqs_to_multiple_of *= tp_size

    # packed sequence length, after sharding to TP and CP domains, needs to be divisible
    # by a recipe-dependent divisor:
    #   blockwise FP8 : 128  (cublas block size)
    #   MXFP8         :  32  (MXFP8 block size)
    #   other FP8     :  16
    #   HybridEP+flex : 128  (MAX_NUM_OF_TOKENS_PER_RANK must be divisible by
    #                         NUM_OF_TOKENS_PER_CHUNK=128)
    # When multiple constraints apply, take the max (128 is a multiple of 32/16).
    divisor = 1
    if use_fp8:
        if fp8_cfg["fp8_recipe"] == "blockwise":
            divisor = max(divisor, 128)
        elif fp8_cfg["fp8_recipe"] == "mxfp8":
            divisor = max(divisor, 32)
        else:
            divisor = max(divisor, 16)
    if (
        megatron_cfg.get("moe_token_dispatcher_type") == "flex"
        and megatron_cfg.get("moe_flex_dispatcher_backend") == "hybridep"
    ):
        divisor = max(divisor, 128)

    if divisor > 1:
        pad_packed_seq_to_multiple_of = divisor
        if cp_size > 1:
            pad_packed_seq_to_multiple_of *= cp_size * 2
        if tp_size > 1 and sp:
            pad_packed_seq_to_multiple_of *= tp_size
    else:
        pad_packed_seq_to_multiple_of = 1

    # when PP is used, all sequences must have the same length, so we need to pad the packed sequence to the max sequence length in the batch.
    if pp_size > 1:
        pad_packed_seq_to = max_seq_len_in_batch
    else:
        pad_packed_seq_to = None

    # make sure the pad_packed_seq_to is a multiple of the pad_packed_seq_to_multiple_of
    if pad_packed_seq_to is not None:
        pad_packed_seq_to = _round_up_to_multiple(
            pad_packed_seq_to, pad_packed_seq_to_multiple_of
        )

    return (
        pad_individual_seqs_to_multiple_of,
        pad_packed_seq_to_multiple_of,
        pad_packed_seq_to,
    )


def _unpack_sequences_from_megatron(
    output_tensor: torch.Tensor,
    seq_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: Optional[torch.Tensor],
    original_batch_size: int,
    original_seq_length: int,
) -> torch.Tensor:
    """Unpack sequences from Megatron output format.

    Args:
        output_tensor: Packed output tensor [1, T, vocab_size]
        seq_lengths: Actual sequence lengths for each sample
        cu_seqlens: Cumulative sequence lengths
        cu_seqlens_padded: Padded cumulative sequence lengths (if CP was used)
        original_batch_size: Original batch size
        original_seq_length: Original maximum sequence length

    Returns:
        Unpacked output tensor [batch_size, seq_length, vocab_size]
    """
    # Remove the batch dimension to get [T, vocab_size]
    output_tensor = output_tensor.squeeze(0)

    # Create a padded output tensor with original shape
    vocab_size = output_tensor.shape[-1]
    unpacked_output = torch.zeros(
        (original_batch_size, original_seq_length, vocab_size),
        dtype=output_tensor.dtype,
        device=output_tensor.device,
    )

    # Get context parallel size to determine which cu_seqlens to use
    cp_size = get_context_parallel_world_size()

    # Fill in the unpacked output tensor with valid tokens
    for b in range(original_batch_size):
        # Get actual sequence length for this sample
        seq_len = (
            seq_lengths[b].item() if torch.is_tensor(seq_lengths[b]) else seq_lengths[b]
        )

        if cp_size > 1 and cu_seqlens_padded is not None:
            # When using CP, we need to account for padding
            # Calculate the padded sequence boundaries
            pad_factor = cp_size * 2
            padded_seq_len = ((seq_len + pad_factor - 1) // pad_factor) * pad_factor
            start_idx = cu_seqlens_padded[b].item()

            # Only copy the valid tokens (not the padding)
            unpacked_output[b, :seq_len] = output_tensor[
                start_idx : start_idx + seq_len
            ]
        else:
            # No CP, use regular cu_seqlens
            start_idx = cu_seqlens[b].item()
            end_idx = cu_seqlens[b + 1].item()

            # Copy the valid tokens to the unpacked tensor
            unpacked_output[b, :seq_len] = output_tensor[start_idx:end_idx]

    return unpacked_output


def get_and_validate_seqlen(data: BatchedDataDict[Any]):
    # dim 1 is always assumed to be the sequence dim, sanity check this here
    sequence_dim = 1
    seq_dim_size = data["input_ids"].shape[sequence_dim]
    for k, v in data.items():
        if torch.is_tensor(v) and len(v.shape) > 1:
            assert v.shape[sequence_dim] == seq_dim_size, (
                f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape} for key {k}"
            )
    return sequence_dim, seq_dim_size
