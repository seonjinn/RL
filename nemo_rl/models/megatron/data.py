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

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Iterator, Optional, Tuple

import torch
from megatron.bridge.training.utils.packed_seq_utils import (
    get_packed_seq_cp_partition_indices,
)
from megatron.core.packed_seq_params import PackedSeqParams, pad_sequence_for_thd
from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import StragglerDetector

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
from nemo_rl.models.megatron.common import _round_up_to_multiple
from nemo_rl.utils.r3_trace import (
    r3_trace_verify_forward_enabled,
    trace_cp_routed_experts,
)

_MICROBATCH_GENERATIONS = count(1)


def _next_microbatch_generation() -> int:
    """Return one process-local, strictly advancing microbatch generation."""
    return next(_MICROBATCH_GENERATIONS)


@dataclass
class ProcessedInputs:
    """Processed microbatch inputs used for model forward pass."""

    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    cu_seqlens_padded: Optional[torch.Tensor]
    cu_seqlens: Optional[torch.Tensor] = None
    structural_padding_mask: Optional[torch.Tensor] = None
    structural_padding_mask_cp_sharded: Optional[torch.Tensor] = None
    packed_geometry: Optional["PackedGeometry"] = None
    mtp_loss_mask: Optional[torch.Tensor] = None
    routed_experts: Optional[torch.Tensor] = None
    routed_experts_cp_sharded: Optional[torch.Tensor] = None


@dataclass
class ProcessedMicrobatch:
    """Container for a processed microbatch ready for model forward pass.

    This dataclass holds both the original data dictionary and the processed
    tensors needed for the Megatron model forward pass.

    Attributes:
        data_dict: The original BatchedDataDict containing raw batch data
        input_ids: Processed input token IDs (may be packed for sequence packing)
        input_ids_cp_sharded: Model-forward token IDs. Usually CP-sharded; models
            that insert media before CP selection receive the full packed THD row.
        attention_mask: Attention mask tensor (None for packed sequences)
        position_ids: Position IDs tensor (None for packed sequences)
        packed_seq_params: PackedSeqParams for sequence packing (None if not packing)
        cu_seqlens: Compact logical boundaries for real packed sequences
        cu_seqlens_padded: Padded cumulative sequence lengths (None if not packing)
        structural_padding_mask: Global physical-order structural padding mask
        structural_padding_mask_cp_sharded: CP-local structural padding mask
        packed_geometry: Immutable packed token and sequence-capacity accounting
        mtp_loss_mask: Pre-computed MTP loss mask (token_mask × sample_mask).
            None when MTP is disabled or token/sample masks are absent.
        routed_experts: Optional token-aligned routed expert ids
        routed_experts_cp_sharded: Context-parallel sharded routed expert ids
    """

    data_dict: BatchedDataDict[Any]
    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    cu_seqlens_padded: Optional[torch.Tensor]
    cu_seqlens: Optional[torch.Tensor] = None
    structural_padding_mask: Optional[torch.Tensor] = None
    structural_padding_mask_cp_sharded: Optional[torch.Tensor] = None
    packed_geometry: Optional["PackedGeometry"] = None
    mtp_loss_mask: Optional[torch.Tensor] = None
    routed_experts: Optional[torch.Tensor] = None
    routed_experts_cp_sharded: Optional[torch.Tensor] = None
    microbatch_generation: int = 0

    def __post_init__(self) -> None:
        if type(self.microbatch_generation) is not int:
            raise TypeError("microbatch_generation must be an int.")
        if self.microbatch_generation < 0:
            raise ValueError("microbatch_generation must be nonnegative.")


def _update_replay_identity_tensor(
    digest: Any,
    name: str,
    tensor: Optional[torch.Tensor],
) -> None:
    digest.update(name.encode("utf-8"))
    if tensor is None:
        digest.update(b":none;")
        return
    if not torch.is_tensor(tensor):
        raise TypeError(f"Replay identity field {name} must be a Tensor or None.")
    value = tensor.detach()
    if value.device.type != "cpu":
        value = value.cpu()
    value = value.contiguous()
    digest.update(f":{value.dtype}:{tuple(value.shape)}:".encode("utf-8"))
    if value.numel() > 0:
        digest.update(value.view(torch.uint8).numpy().tobytes())
    digest.update(b";")


def _processed_microbatch_replay_identity(microbatch: ProcessedMicrobatch) -> str:
    """Return a compact content identity without retaining payload tensors."""
    digest = hashlib.sha256()
    for name in (
        "input_ids_cp_sharded",
        "routed_experts_cp_sharded",
        "structural_padding_mask_cp_sharded",
        "cu_seqlens",
        "cu_seqlens_padded",
    ):
        _update_replay_identity_tensor(digest, name, getattr(microbatch, name))

    geometry = microbatch.packed_geometry
    digest.update(b"packed_geometry:")
    if geometry is None:
        digest.update(b"none;")
    else:
        for name in (
            "logical_tokens",
            "padded_tokens",
            "capacity_tokens",
            "real_sequence_count",
            "cu_seqlens_capacity_entries",
        ):
            value = getattr(geometry, name)
            if type(value) is not int:
                raise TypeError(
                    f"Replay identity geometry field {name} must be an int."
                )
            digest.update(f"{name}:{value};".encode("utf-8"))
    return digest.hexdigest()


class ReplayableProcessedMicrobatchIterator(Iterator[ProcessedMicrobatch]):
    """Reprocess graph microbatches without retaining their CUDA payloads."""

    def __init__(
        self,
        factory: Callable[[Callable[[], int]], Iterator[ProcessedMicrobatch]],
        generation_provider: Optional[Callable[[], int]],
    ) -> None:
        self._factory = factory
        self._generation_provider = (
            _next_microbatch_generation
            if generation_provider is None
            else generation_provider
        )
        self._generations: list[int] = []
        self._identities: list[str] = []
        self._exhausted = False
        self._replay_created = False
        self._iterator = factory(self._record_generation)

    def _record_generation(self) -> int:
        generation = self._generation_provider()
        self._generations.append(generation)
        return generation

    def __iter__(self) -> "ReplayableProcessedMicrobatchIterator":
        return self

    def __next__(self) -> ProcessedMicrobatch:
        try:
            microbatch = next(self._iterator)
        except StopIteration:
            self._exhausted = True
            raise
        self._identities.append(_processed_microbatch_replay_identity(microbatch))
        return microbatch

    def replay(self) -> Iterator[ProcessedMicrobatch]:
        """Create the one schedule iterator with the preflight generations."""
        if not self._exhausted:
            raise RuntimeError(
                "CUDA graph microbatch preflight must exhaust its iterator before "
                "schedule replay."
            )
        if self._replay_created:
            raise RuntimeError(
                "CUDA graph microbatch schedule replay may be created only once."
            )
        self._replay_created = True
        generations = tuple(self._generations)
        identities = tuple(self._identities)
        if len(generations) != len(identities):
            raise RuntimeError(
                "CUDA graph microbatch preflight generation and identity counts "
                f"differ: {len(generations)} != {len(identities)}."
            )

        def replay_iterator() -> Iterator[ProcessedMicrobatch]:
            generation_index = 0

            def replay_generation() -> int:
                nonlocal generation_index
                if generation_index >= len(generations):
                    raise RuntimeError(
                        "CUDA graph microbatch replay produced more inputs than "
                        "preflight."
                    )
                generation = generations[generation_index]
                generation_index += 1
                return generation

            replay_source = iter(self._factory(replay_generation))
            for index, (generation, identity) in enumerate(
                zip(generations, identities, strict=True)
            ):
                try:
                    microbatch = next(replay_source)
                except StopIteration as error:
                    raise RuntimeError(
                        "CUDA graph microbatch replay produced fewer inputs than "
                        f"preflight: {index} != {len(generations)}."
                    ) from error
                if microbatch.microbatch_generation != generation:
                    raise RuntimeError(
                        "CUDA graph microbatch replay generation differs at index "
                        f"{index}: {microbatch.microbatch_generation} != {generation}."
                    )
                replay_identity = _processed_microbatch_replay_identity(microbatch)
                if replay_identity != identity:
                    raise RuntimeError(
                        "CUDA graph microbatch replay identity differs at index "
                        f"{index}."
                    )
                if index == len(generations) - 1:
                    try:
                        next(replay_source)
                    except StopIteration:
                        pass
                    else:
                        raise RuntimeError(
                            "CUDA graph microbatch replay produced more inputs than "
                            "preflight."
                        )
                yield microbatch

        return replay_iterator()


@dataclass(frozen=True)
class PackedGeometry:
    """Immutable logical, physical, and fixed-capacity packed geometry."""

    logical_tokens: int
    padded_tokens: int
    capacity_tokens: int
    real_sequence_count: int
    cu_seqlens_capacity_entries: int


@dataclass(frozen=True)
class _PackedSequenceOutput:
    """Internal rich result while the public packer keeps its legacy tuple."""

    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    packed_seq_params: PackedSeqParams
    cu_seqlens: torch.Tensor
    cu_seqlens_padded: torch.Tensor
    structural_padding_mask: torch.Tensor
    structural_padding_mask_cp_sharded: torch.Tensor
    packed_geometry: PackedGeometry


def _validate_cuda_graph_training_inputs(
    inputs: ProcessedInputs,
    *,
    global_token_capacity: int,
    thd_max_packed_sequences: int,
    tp_rank: int = 0,
    tp_size: int = 1,
    sequence_parallel: bool = False,
) -> None:
    """Validate Task 8 fixed THD geometry before yielding a training input."""
    if global_token_capacity < 1:
        raise ValueError("CUDA graph training token capacity must be positive.")
    if thd_max_packed_sequences < 2:
        raise ValueError(
            "thd_max_packed_sequences must reserve one real and one dummy sequence."
        )
    cp_rank = get_context_parallel_rank()
    cp_size = get_context_parallel_world_size()
    if global_token_capacity % cp_size != 0:
        raise ValueError(
            "CUDA graph training token capacity must be divisible by context "
            f"parallel size ({cp_size})."
        )
    local_token_capacity = global_token_capacity // cp_size
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"Invalid TP rank/size: rank={tp_rank}, size={tp_size}.")
    router_token_capacity = local_token_capacity
    if sequence_parallel:
        if router_token_capacity % tp_size != 0:
            raise ValueError(
                "CP-local router token capacity must divide evenly across TP/SP ranks."
            )
        router_token_capacity //= tp_size

    geometry = inputs.packed_geometry
    if geometry is None:
        raise ValueError("CUDA graph training requires Task 8 packed geometry.")
    if geometry.capacity_tokens != global_token_capacity:
        raise ValueError(
            "Packed geometry token capacity does not match "
            "sequence_packing.train_mb_tokens."
        )
    if not 1 <= geometry.real_sequence_count <= thd_max_packed_sequences - 1:
        raise ValueError(
            f"Packed THD real sequence count ({geometry.real_sequence_count}) exceeds "
            f"the configured bound ({thd_max_packed_sequences - 1})."
        )
    expected_entries = thd_max_packed_sequences + 1
    if geometry.cu_seqlens_capacity_entries != expected_entries:
        raise ValueError(
            "Packed THD cumulative-entry capacity does not match "
            f"thd_max_packed_sequences + 1 ({expected_entries})."
        )
    if not (
        0
        <= geometry.logical_tokens
        <= geometry.padded_tokens
        <= geometry.capacity_tokens
    ):
        raise ValueError(
            "Packed geometry must satisfy logical_tokens <= padded_tokens <= "
            "capacity_tokens."
        )

    if inputs.input_ids.shape[-1] != global_token_capacity:
        raise ValueError(
            "Global packed input shape does not match the fixed CUDA graph token "
            f"capacity: {inputs.input_ids.shape[-1]} != {global_token_capacity}."
        )
    if inputs.input_ids_cp_sharded.shape[-1] != local_token_capacity:
        raise ValueError(
            "CP-local packed input shape does not match the fixed CUDA graph token "
            f"capacity: {inputs.input_ids_cp_sharded.shape[-1]} != "
            f"{local_token_capacity}."
        )
    if (
        inputs.structural_padding_mask is None
        or inputs.structural_padding_mask.dtype != torch.bool
        or inputs.structural_padding_mask.shape != inputs.input_ids.shape
    ):
        raise ValueError(
            "Global structural padding mask must be bool and match the fixed packed input."
        )
    if (
        inputs.structural_padding_mask_cp_sharded is None
        or inputs.structural_padding_mask_cp_sharded.dtype != torch.bool
        or inputs.structural_padding_mask_cp_sharded.shape
        != inputs.input_ids_cp_sharded.shape
    ):
        raise ValueError(
            "CP-local structural padding mask must be bool and match the fixed packed input."
        )

    packed_seq_params = inputs.packed_seq_params
    if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
        raise ValueError("CUDA graph training requires THD PackedSeqParams.")
    for field_name in (
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
    ):
        cumulative = getattr(packed_seq_params, field_name, None)
        if (
            not isinstance(cumulative, torch.Tensor)
            or cumulative.dim() != 1
            or cumulative.numel() != expected_entries
        ):
            raise ValueError(
                f"{field_name} must have the fixed CUDA graph entry capacity "
                f"({expected_entries})."
            )
    if packed_seq_params.total_tokens != local_token_capacity:
        raise ValueError(
            "PackedSeqParams.total_tokens must equal the CP-local fixed token capacity."
        )
    if packed_seq_params.pad_between_seqs is not True:
        raise ValueError(
            "Fixed THD CUDA graph metadata requires pad_between_seqs=True."
        )
    if (
        not isinstance(packed_seq_params.seq_idx, torch.Tensor)
        or packed_seq_params.seq_idx.shape != inputs.input_ids.shape
    ):
        raise ValueError(
            "Packed Mamba seq_idx must match the global fixed token shape consumed "
            "after Mamba context-parallel all-to-all."
        )

    expected_real_entries = geometry.real_sequence_count + 1
    for field_name in ("cu_seqlens", "cu_seqlens_padded"):
        cumulative = getattr(inputs, field_name)
        if (
            not isinstance(cumulative, torch.Tensor)
            or cumulative.dim() != 1
            or cumulative.numel() != expected_real_entries
        ):
            raise ValueError(
                f"Real {field_name} must contain exactly one endpoint per real "
                "sequence plus the origin."
            )

    sample_ids = packed_seq_params.seq_aux_loss_sample_ids
    if not isinstance(sample_ids, torch.Tensor):
        raise ValueError("Packed sequence auxiliary-loss sample IDs must be present.")
    if sample_ids.dim() != 1 or sample_ids.numel() != router_token_capacity:
        raise ValueError(
            "Packed sequence auxiliary-loss sample IDs must match the router-local "
            f"token capacity ({router_token_capacity})."
        )
    if sample_ids.dtype != torch.int64:
        raise ValueError("Packed sequence auxiliary-loss sample IDs must use int64.")
    if sample_ids.device != inputs.input_ids.device:
        raise ValueError(
            "Packed sequence auxiliary-loss sample IDs must share the input device."
        )
    if not sample_ids.is_contiguous():
        raise ValueError(
            "Packed sequence auxiliary-loss sample IDs must be contiguous."
        )

    num_samples = packed_seq_params.seq_aux_loss_num_samples
    if not isinstance(num_samples, torch.Tensor):
        raise ValueError(
            "Packed sequence auxiliary-loss sample count must be a Tensor scalar."
        )
    if num_samples.shape != torch.Size([]):
        raise ValueError(
            "Packed sequence auxiliary-loss sample count must be a scalar."
        )
    if num_samples.dtype != torch.int64:
        raise ValueError("Packed sequence auxiliary-loss sample count must use int64.")
    if num_samples.device != inputs.input_ids.device:
        raise ValueError(
            "Packed sequence auxiliary-loss sample count must share the input device."
        )
    if not num_samples.is_contiguous():
        raise ValueError(
            "Packed sequence auxiliary-loss sample count must be contiguous."
        )

    sample_count = int(num_samples)
    expected_max_samples = thd_max_packed_sequences - 1
    max_samples = packed_seq_params.seq_aux_loss_max_samples
    if (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples != expected_max_samples
    ):
        raise ValueError(
            "Packed sequence auxiliary-loss static sample capacity must equal "
            f"thd_max_packed_sequences - 1 ({expected_max_samples})."
        )
    if not 1 <= sample_count <= expected_max_samples:
        raise ValueError(
            "Packed sequence auxiliary-loss sample count must satisfy "
            f"1 <= N <= {expected_max_samples}."
        )
    if sample_count != geometry.real_sequence_count:
        raise ValueError(
            "Packed sequence auxiliary-loss sample count must equal packed geometry "
            f"real_sequence_count ({geometry.real_sequence_count})."
        )
    if bool(torch.any(sample_ids < 0)):
        raise ValueError(
            "Packed sequence auxiliary-loss sample IDs must be nonnegative."
        )
    if bool(torch.any(sample_ids >= sample_count)):
        raise ValueError(
            "Packed sequence auxiliary-loss sample IDs must be less than N."
        )

    expected_sample_ids = _build_packed_seq_aux_loss_sample_ids(
        inputs.cu_seqlens_padded,
        capacity_tokens=global_token_capacity,
        real_sequence_count=geometry.real_sequence_count,
        cp_rank=cp_rank,
        cp_size=cp_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
        sequence_parallel=sequence_parallel,
    )
    if not torch.equal(sample_ids, expected_sample_ids):
        raise ValueError(
            "Packed sequence auxiliary-loss sample IDs do not match the exact "
            "router order and dummy ownership."
        )


def make_processed_microbatch_iterator(
    raw_iterator: Iterator[BatchedDataDict[Any]],
    cfg: dict[str, Any],
    seq_length_key: Optional[str],
    pad_individual_seqs_to_multiple_of: int,
    pad_packed_seq_to_multiple_of: int,
    straggler_timer: StragglerDetector,
    pad_full_seq_to: Optional[int],
    delegate_pack_to_model: bool = False,
    thd_max_packed_sequences: Optional[int] = None,
    for_cuda_graph_training: bool = False,
    delegate_mtp_loss_mask_to_model: bool = False,
    model_slices_context_parallel_inputs: bool = False,
    microbatch_generation_provider: Optional[Callable[[], int]] = None,
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
        for_cuda_graph_training: Validate fixed Task 8 geometry before every yield.

    Yields:
        ProcessedMicrobatch objects containing processed tensors ready for model forward
    """
    pack_sequences = cfg["sequence_packing"]["enabled"]
    generation_provider = (
        _next_microbatch_generation
        if microbatch_generation_provider is None
        else microbatch_generation_provider
    )
    if thd_max_packed_sequences is not None and not for_cuda_graph_training:
        raise ValueError(
            "Fixed THD sequence capacity is training-graph-only; set "
            "for_cuda_graph_training=True."
        )
    tp_rank = 0
    tp_size = 1
    sequence_parallel = False
    if for_cuda_graph_training:
        megatron_cfg = cfg["megatron_cfg"]
        configured_tp_size = megatron_cfg["tensor_model_parallel_size"]
        sequence_parallel = megatron_cfg["sequence_parallel"]
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if tp_size != configured_tp_size:
            raise ValueError(
                "Initialized tensor model parallel world size "
                f"({tp_size}) does not match megatron_cfg.tensor_model_parallel_size "
                f"({configured_tp_size})."
            )
        if not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"Initialized tensor model parallel rank ({tp_rank}) is outside "
                f"[0, {tp_size})."
            )

    for data_dict in raw_iterator:
        # Move to GPU
        data_dict = data_dict.to("cuda")

        # Process the microbatch
        processed_inputs = process_microbatch(
            data_dict=data_dict,
            seq_length_key=seq_length_key,
            pad_individual_seqs_to_multiple_of=pad_individual_seqs_to_multiple_of,
            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
            pad_full_seq_to=pad_full_seq_to,
            pack_sequences=pack_sequences,
            delegate_pack_to_model=delegate_pack_to_model,
            thd_max_packed_sequences=thd_max_packed_sequences,
            delegate_mtp_loss_mask_to_model=delegate_mtp_loss_mask_to_model,
            model_slices_context_parallel_inputs=model_slices_context_parallel_inputs,
            straggler_timer=straggler_timer,
            tp_rank=tp_rank,
            tp_size=tp_size,
            sequence_parallel=sequence_parallel,
        )

        if for_cuda_graph_training:
            if pad_full_seq_to is None or thd_max_packed_sequences is None:
                raise ValueError(
                    "CUDA graph training requires fixed token and sequence capacities."
                )
            _validate_cuda_graph_training_inputs(
                processed_inputs,
                global_token_capacity=pad_full_seq_to,
                thd_max_packed_sequences=thd_max_packed_sequences,
                tp_rank=tp_rank,
                tp_size=tp_size,
                sequence_parallel=sequence_parallel,
            )

        microbatch_generation = generation_provider()
        if type(microbatch_generation) is not int:
            raise TypeError("microbatch generation provider must return an int.")
        if microbatch_generation < 0:
            raise ValueError(
                "microbatch generation provider returned a negative value."
            )

        yield ProcessedMicrobatch(
            data_dict=data_dict,
            input_ids=processed_inputs.input_ids,
            input_ids_cp_sharded=processed_inputs.input_ids_cp_sharded,
            attention_mask=processed_inputs.attention_mask,
            position_ids=processed_inputs.position_ids,
            packed_seq_params=processed_inputs.packed_seq_params,
            cu_seqlens=processed_inputs.cu_seqlens,
            cu_seqlens_padded=processed_inputs.cu_seqlens_padded,
            structural_padding_mask=processed_inputs.structural_padding_mask,
            structural_padding_mask_cp_sharded=(
                processed_inputs.structural_padding_mask_cp_sharded
            ),
            packed_geometry=processed_inputs.packed_geometry,
            mtp_loss_mask=processed_inputs.mtp_loss_mask,
            routed_experts=processed_inputs.routed_experts,
            routed_experts_cp_sharded=processed_inputs.routed_experts_cp_sharded,
            microbatch_generation=microbatch_generation,
        )


def get_microbatch_iterator(
    data: BatchedDataDict[Any],
    cfg: dict[str, Any],
    mbs: int,
    straggler_timer: StragglerDetector,
    seq_length_key: Optional[str] = None,
    delegate_pack_to_model: bool = False,
    thd_max_packed_sequences: Optional[int] = None,
    for_cuda_graph_training: bool = False,
    delegate_mtp_loss_mask_to_model: bool = False,
    model_slices_context_parallel_inputs: bool = False,
    microbatch_generation_provider: Optional[Callable[[], int]] = None,
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
        for_cuda_graph_training: Use canonical fixed training graph geometry. Evaluation
            and logprob callers leave this false and retain eager geometry.

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

    if not isinstance(for_cuda_graph_training, bool):
        raise TypeError("for_cuda_graph_training must be a bool.")
    if for_cuda_graph_training:
        if cfg["sequence_packing"]["enabled"] is not True:
            raise ValueError(
                "CUDA graph training requires sequence_packing.enabled=true."
            )
        if cfg["dynamic_batching"]["enabled"] is not False:
            raise ValueError(
                "CUDA graph training requires dynamic_batching.enabled=false."
            )
        if delegate_pack_to_model:
            raise ValueError(
                "CUDA graph training cannot use delegate_pack_to_model=True because "
                "NeMo-RL must own the fixed physical token order."
            )
        configured_sequence_capacity = cfg["megatron_cfg"].get(
            "thd_max_packed_sequences"
        )
        if (
            isinstance(configured_sequence_capacity, bool)
            or not isinstance(configured_sequence_capacity, int)
            or configured_sequence_capacity < 2
        ):
            raise ValueError(
                "CUDA graph training requires thd_max_packed_sequences >= 2."
            )
        if (
            thd_max_packed_sequences is not None
            and thd_max_packed_sequences != configured_sequence_capacity
        ):
            raise ValueError(
                "Explicit thd_max_packed_sequences differs from the canonical "
                "megatron_cfg value."
            )
        thd_max_packed_sequences = configured_sequence_capacity
    elif thd_max_packed_sequences is not None:
        raise ValueError(
            "Fixed THD sequence capacity is training-graph-only; set "
            "for_cuda_graph_training=True."
        )

    _, seq_dim_size = get_and_validate_seqlen(data)

    # Auto-detect seq_length_key if not provided
    if seq_length_key is None and cfg["sequence_packing"]["enabled"]:
        seq_length_key = "input_lengths"

    if cfg["dynamic_batching"]["enabled"]:
        raw_iterator_factory = data.make_microbatch_iterator_with_dynamic_shapes
        data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
    elif cfg["sequence_packing"]["enabled"]:
        raw_iterator_factory = data.make_microbatch_iterator_for_packable_sequences
        data_iterator_len, pack_seq_dim_size = (
            data.get_microbatch_iterator_for_packable_sequences_len()
        )
        (
            pad_factor,
            pad_packed_seq_to_multiple_of,
            pad_full_seq_to,
        ) = _get_pack_sequence_parameters_for_megatron(
            cfg["megatron_cfg"],
            cfg["make_sequence_length_divisible_by"],
            pack_seq_dim_size,
        )
        if for_cuda_graph_training:
            train_mb_tokens = cfg["sequence_packing"].get("train_mb_tokens")
            if (
                isinstance(train_mb_tokens, bool)
                or not isinstance(train_mb_tokens, int)
                or train_mb_tokens < 1
            ):
                raise ValueError(
                    "CUDA graph training requires a positive "
                    "sequence_packing.train_mb_tokens."
                )
            cp_size = cfg["megatron_cfg"]["context_parallel_size"]
            if train_mb_tokens % cp_size != 0:
                raise ValueError(
                    "sequence_packing.train_mb_tokens must be divisible by "
                    f"context_parallel_size ({cp_size})."
                )
            if train_mb_tokens % pad_factor != 0:
                raise ValueError(
                    "sequence_packing.train_mb_tokens must be divisible by the "
                    f"individual sequence alignment ({pad_factor})."
                )
            if train_mb_tokens % pad_packed_seq_to_multiple_of != 0:
                raise ValueError(
                    "sequence_packing.train_mb_tokens must be divisible by the packed "
                    f"alignment ({pad_packed_seq_to_multiple_of})."
                )
            pad_full_seq_to = train_mb_tokens
        micro_batch_size = 1
    else:
        raw_iterator_factory = lambda: data.make_microbatch_iterator(mbs)
        data_iterator_len = data.size // mbs

    def processed_iterator_factory(
        generation_provider: Callable[[], int],
    ) -> Iterator[ProcessedMicrobatch]:
        return make_processed_microbatch_iterator(
            raw_iterator=raw_iterator_factory(),
            cfg=cfg,
            seq_length_key=seq_length_key,
            pad_individual_seqs_to_multiple_of=pad_factor,
            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
            pad_full_seq_to=pad_full_seq_to,
            straggler_timer=straggler_timer,
            delegate_pack_to_model=delegate_pack_to_model,
            thd_max_packed_sequences=thd_max_packed_sequences,
            for_cuda_graph_training=for_cuda_graph_training,
            delegate_mtp_loss_mask_to_model=delegate_mtp_loss_mask_to_model,
            model_slices_context_parallel_inputs=model_slices_context_parallel_inputs,
            microbatch_generation_provider=generation_provider,
        )

    if for_cuda_graph_training:
        processed_iterator: Iterator[ProcessedMicrobatch] = (
            ReplayableProcessedMicrobatchIterator(
                processed_iterator_factory,
                microbatch_generation_provider,
            )
        )
    else:
        generation_provider = (
            _next_microbatch_generation
            if microbatch_generation_provider is None
            else microbatch_generation_provider
        )
        processed_iterator = processed_iterator_factory(generation_provider)

    # Compute padded sequence length for pipeline parallelism
    padded_seq_length = pad_full_seq_to if pad_full_seq_to is not None else seq_dim_size

    return (
        processed_iterator,
        data_iterator_len,
        micro_batch_size,
        seq_dim_size,
        padded_seq_length,
    )


def get_ltor_masks_and_position_ids(*args: Any, **kwargs: Any) -> Any:
    """Lazy proxy for `megatron.training.utils.get_ltor_masks_and_position_ids`.

    The underlying import is deferred to call time so that importing this module does
    not pull in `megatron.training` -> modelopt -> transformers -> torchvision, which
    can crash on a duplicate torchvision ``roi_align` meta-kernel registration in the mcore venv.
    """
    from megatron.training.utils import get_ltor_masks_and_position_ids as _impl

    return _impl(*args, **kwargs)


def process_microbatch(
    data_dict: BatchedDataDict[Any],
    seq_length_key: Optional[str] = None,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_full_seq_to: Optional[int] = None,
    pack_sequences: bool = False,
    delegate_pack_to_model: bool = False,
    thd_max_packed_sequences: Optional[int] = None,
    delegate_mtp_loss_mask_to_model: bool = False,
    model_slices_context_parallel_inputs: bool = False,
    straggler_timer: Optional[StragglerDetector] = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    sequence_parallel: bool = False,
) -> ProcessedInputs:
    """Process a microbatch for Megatron model forward pass."""
    ctx = straggler_timer(bdata=True) if straggler_timer is not None else nullcontext()
    with ctx:
        input_ids = data_dict["input_ids"]
        attention_mask = None
        position_ids = None
        packed_seq_params = None
        routed_experts = (
            data_dict["routed_experts"] if "routed_experts" in data_dict else None
        )
        token_identity_cp_sharded = None
        if routed_experts is not None and routed_experts.dim() != 4:
            raise ValueError(
                "routed_experts must have shape [batch, seq, num_moe_layers, topk] "
                f"before Megatron packing; got {tuple(routed_experts.shape)}"
            )
        routed_experts_cp_sharded = routed_experts

        original_batch_size = input_ids.shape[0]
        original_seq_length = input_ids.shape[1]
        seq_lengths = None  # Will be set if using packed sequences
        cu_seqlens = None
        cu_seqlens_padded = None
        structural_padding_mask = None
        structural_padding_mask_cp_sharded = None
        packed_geometry = None
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

            if delegate_pack_to_model:
                if thd_max_packed_sequences is not None:
                    raise ValueError(
                        "Fixed THD graph capacity cannot use "
                        "delegate_pack_to_model=True because NeMo-RL cannot "
                        "reproduce the model's internal physical token order."
                    )
                has_mtp_loss_mask = "mtp_loss_mask" in data_dict
                assert not has_mtp_loss_mask or delegate_mtp_loss_mask_to_model, (
                    "MTP training requires a self-packing VLM that advertises "
                    "model_owns_mtp_loss_mask_packing"
                )
                # VLM path: model (e.g. mbridge Qwen3VL) does its own
                # preprocess_packed_seqs; NeMo-RL must NOT pre-pack + CP-shard,
                # or the double-processing produces shape mismatches downstream
                # (GDN/RoPE/MoE). We only pad each sequence individually and
                # hand the model [B, max_seq] + bool attention_mask + cu_seqlens.
                if routed_experts is not None:
                    # Router replay needs routed_experts CP-sharded into the
                    # model's local token order, but a self-packing model packs
                    # and CP-shards internally, so NeMo-RL cannot build a matching
                    # layout here. Fail loudly rather than feed misaligned routes.
                    raise NotImplementedError(
                        "Router replay (routed_experts) is not supported with "
                        "models that pack and context-parallel shard internally "
                        "(delegate_pack_to_model=True)."
                    )
                (
                    input_ids,
                    input_ids_cp_sharded,
                    attention_mask,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _prepare_vlm_batch_for_megatron(
                    input_ids,
                    seq_lengths,
                    pad_individual_seqs_to_multiple_of,
                    pad_full_seq_to=pad_full_seq_to,
                )
                if has_mtp_loss_mask:
                    source_mtp_loss_mask = data_dict["mtp_loss_mask"]
                    assert source_mtp_loss_mask.ndim == 2
                    assert (
                        source_mtp_loss_mask.shape[0] == input_ids_cp_sharded.shape[0]
                    )
                    mtp_loss_mask = source_mtp_loss_mask.new_zeros(
                        input_ids_cp_sharded.shape
                    )
                    copied_length = min(
                        source_mtp_loss_mask.shape[1],
                        input_ids_cp_sharded.shape[1],
                    )
                    mtp_loss_mask[:, :copied_length] = source_mtp_loss_mask[
                        :, :copied_length
                    ]
                    mtp_loss_mask = mtp_loss_mask * attention_mask.to(
                        dtype=mtp_loss_mask.dtype
                    )
                position_ids = None
            else:
                cp_rank = get_context_parallel_rank()
                cp_size = get_context_parallel_world_size()
                if (
                    model_slices_context_parallel_inputs
                    and "mtp_loss_mask" in data_dict
                ):
                    raise NotImplementedError(
                        "Nemotron Omni caller-packed THD inputs do not yet support MTP. "
                        "Disable MTP for the Nano image/text path."
                    )
                token_identity = None
                if routed_experts is not None and r3_trace_verify_forward_enabled():
                    token_identity = _make_r3_trace_token_identity(
                        input_ids, seq_lengths
                    )

                # Pack sequences on main's per-sequence zigzag CP layout.
                if model_slices_context_parallel_inputs:
                    if thd_max_packed_sequences is not None:
                        raise ValueError(
                            "Fixed THD graph capacity does not yet support models "
                            "that slice context-parallel inputs internally."
                        )
                    (
                        input_ids,
                        local_input_ids,
                        _packed_seq_params,
                        cu_seqlens,
                        cu_seqlens_padded,
                    ) = _pack_sequences_for_megatron(
                        input_ids,
                        seq_lengths,
                        pad_individual_seqs_to_multiple_of,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                        cp_rank=cp_rank,
                        cp_size=cp_size,
                    )
                    packed_seq_params = PackedSeqParams(
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_kv=cu_seqlens,
                        cu_seqlens_q_padded=cu_seqlens_padded,
                        cu_seqlens_kv_padded=cu_seqlens_padded,
                        max_seqlen_q=int(
                            (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1])
                            .max()
                            .item()
                        ),
                        max_seqlen_kv=int(
                            (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1])
                            .max()
                            .item()
                        ),
                        # TE's default inference excludes the final boundary, so
                        # it misses trailing-only padding for a single sequence.
                        # CP zigzag can move that padding to a rank-local seam.
                        pad_between_seqs=not torch.equal(cu_seqlens, cu_seqlens_padded),
                        qkv_format="thd",
                        total_tokens=input_ids.shape[1],
                    )
                    # This field is the model-forward input. For this capability
                    # the model needs the full THD row so it can insert media
                    # before selecting its CP-owned embeddings.
                    input_ids_cp_sharded = input_ids
                else:
                    packed_output = _pack_sequences_for_megatron_with_geometry(
                        input_ids,
                        seq_lengths,
                        pad_individual_seqs_to_multiple_of,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                        cp_rank=cp_rank,
                        cp_size=cp_size,
                        thd_max_packed_sequences=thd_max_packed_sequences,
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        sequence_parallel=sequence_parallel,
                    )
                    input_ids = packed_output.input_ids
                    local_input_ids = packed_output.input_ids_cp_sharded
                    input_ids_cp_sharded = local_input_ids
                    packed_seq_params = packed_output.packed_seq_params
                    cu_seqlens = packed_output.cu_seqlens
                    cu_seqlens_padded = packed_output.cu_seqlens_padded
                    structural_padding_mask = packed_output.structural_padding_mask
                    structural_padding_mask_cp_sharded = (
                        packed_output.structural_padding_mask_cp_sharded
                    )
                    packed_geometry = packed_output.packed_geometry
                # routed_experts and the R3 trace token identity ride the SAME
                # per-seq zigzag CP sharding as input_ids, re-derived from
                # cu_seqlens_padded.
                if routed_experts is not None:
                    (
                        routed_experts,
                        routed_experts_cp_sharded,
                        _token_identity_packed,
                        token_identity_cp_sharded,
                    ) = _shard_routed_experts_for_cp(
                        routed_experts,
                        token_identity,
                        seq_lengths,
                        cu_seqlens,
                        cu_seqlens_padded,
                        cp_rank,
                        cp_size,
                    )
                    routed_experts = _pad_routed_experts_tail(
                        routed_experts,
                        target_tokens=input_ids.shape[1],
                    )
                    routed_experts_cp_sharded = _pad_routed_experts_tail(
                        routed_experts_cp_sharded,
                        target_tokens=input_ids_cp_sharded.shape[1],
                    )
                    token_identity_cp_sharded = _pad_token_aligned_tail(
                        token_identity_cp_sharded,
                        target_tokens=input_ids_cp_sharded.shape[1],
                        value=0,
                    )
                    if model_slices_context_parallel_inputs:
                        cp_partition_indices = get_packed_seq_cp_partition_indices(
                            packed_seq_params,
                            total_tokens=input_ids.shape[1],
                            cp_size=get_context_parallel_world_size(),
                            cp_rank=get_context_parallel_rank(),
                            device=input_ids.device,
                        )
                        routed_experts_cp_sharded = routed_experts.index_select(
                            1, cp_partition_indices
                        ).contiguous()
                        if _token_identity_packed is not None:
                            token_identity_cp_sharded = (
                                _token_identity_packed.index_select(
                                    1, cp_partition_indices
                                ).contiguous()
                            )
                if (
                    routed_experts_cp_sharded is not None
                    and routed_experts_cp_sharded.dim() != 4
                ):
                    raise ValueError(
                        "CP-sharded routed_experts must have shape [1, tokens, "
                        "num_moe_layers, topk] after Megatron packing; got "
                        f"{tuple(routed_experts_cp_sharded.shape)}"
                    )
                verified_token_count = _verify_r3_trace_cp_token_alignment(
                    source_input_ids=data_dict["input_ids"],
                    source_routed_experts=data_dict.get("routed_experts"),
                    input_ids_cp_sharded=(
                        local_input_ids
                        if model_slices_context_parallel_inputs
                        else input_ids_cp_sharded
                    ),
                    routed_experts_cp_sharded=routed_experts_cp_sharded,
                    token_identity_cp_sharded=token_identity_cp_sharded,
                )
                trace_cp_routed_experts(
                    routed_experts_cp_sharded=routed_experts_cp_sharded,
                    token_identity_cp_sharded=token_identity_cp_sharded,
                    input_ids_cp_sharded=(
                        local_input_ids
                        if model_slices_context_parallel_inputs
                        else input_ids_cp_sharded
                    ),
                    cp_token_identity_verified_count=verified_token_count,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )

                # Pack pre-computed mtp_loss_mask the same way as input_ids
                if "mtp_loss_mask" in data_dict:
                    mtp_loss_mask = _pack_token_aligned_tensor_for_megatron(
                        data_dict["mtp_loss_mask"],
                        seq_lengths,
                        cu_seqlens_padded,
                        cp_rank=cp_rank,
                        cp_size=cp_size,
                        local_capacity=input_ids_cp_sharded.shape[1],
                    )

                # For packed sequences, position_ids and attention_mask are typically None
                # The PackedSeqParams handles all necessary sequence information
                position_ids = None
                attention_mask = None
        else:
            if routed_experts is not None:
                if "input_lengths" not in data_dict:
                    raise ValueError(
                        "routed_experts requires input_lengths when sequence packing "
                        "is disabled so padding rows can be repaired before router "
                        "replay."
                    )
                routed_experts = _fill_routed_experts_padding(
                    routed_experts,
                    data_dict["input_lengths"],
                )
                routed_experts_cp_sharded = routed_experts
                if r3_trace_verify_forward_enabled():
                    token_identity_cp_sharded = _make_r3_trace_token_identity(
                        input_ids,
                        data_dict["input_lengths"],
                    )
            input_ids_cp_sharded = input_ids
            verified_token_count = _verify_r3_trace_cp_token_alignment(
                source_input_ids=data_dict["input_ids"],
                source_routed_experts=data_dict.get("routed_experts"),
                input_ids_cp_sharded=input_ids_cp_sharded,
                routed_experts_cp_sharded=routed_experts_cp_sharded,
                token_identity_cp_sharded=token_identity_cp_sharded,
            )
            trace_cp_routed_experts(
                routed_experts_cp_sharded=routed_experts_cp_sharded,
                token_identity_cp_sharded=token_identity_cp_sharded,
                input_ids_cp_sharded=input_ids_cp_sharded,
                cp_token_identity_verified_count=verified_token_count,
                cp_rank=get_context_parallel_rank(),
                cp_size=get_context_parallel_world_size(),
            )
            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                data=input_ids,
                eod_token=0,  # used for loss_mask, which we don't use
                pad_token=0,  # used for loss_mask, which we don't use
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                pad_mask_loss=False,
            )
            if "mtp_loss_mask" in data_dict:
                mtp_loss_mask = data_dict["mtp_loss_mask"]
    return ProcessedInputs(
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids_cp_sharded,
        attention_mask=attention_mask,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
        cu_seqlens=cu_seqlens,
        cu_seqlens_padded=cu_seqlens_padded,
        structural_padding_mask=structural_padding_mask,
        structural_padding_mask_cp_sharded=structural_padding_mask_cp_sharded,
        packed_geometry=packed_geometry,
        mtp_loss_mask=mtp_loss_mask,
        routed_experts=routed_experts,
        routed_experts_cp_sharded=routed_experts_cp_sharded,
    )


def _make_r3_trace_token_identity(
    input_ids: torch.Tensor,
    seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build debug-only ``[batch_idx, token_pos, valid]`` token identities."""
    batch_size, seq_len = input_ids.shape[:2]
    batch_idx = torch.arange(
        batch_size,
        dtype=torch.int32,
        device=input_ids.device,
    ).view(batch_size, 1, 1)
    token_pos = torch.arange(
        seq_len,
        dtype=torch.int32,
        device=input_ids.device,
    ).view(1, seq_len, 1)
    if seq_lengths is None:
        valid = torch.ones(
            batch_size,
            seq_len,
            1,
            dtype=torch.int32,
            device=input_ids.device,
        )
    else:
        valid = (
            token_pos.expand(batch_size, seq_len, 1)
            < seq_lengths.to(device=input_ids.device, dtype=torch.int32).view(
                batch_size,
                1,
                1,
            )
        ).to(dtype=torch.int32)
    return torch.cat(
        (
            batch_idx.expand(batch_size, seq_len, 1),
            token_pos.expand(batch_size, seq_len, 1),
            valid,
        ),
        dim=-1,
    )


def _verify_r3_trace_cp_token_alignment(
    *,
    source_input_ids: torch.Tensor,
    source_routed_experts: Optional[torch.Tensor],
    input_ids_cp_sharded: torch.Tensor,
    routed_experts_cp_sharded: Optional[torch.Tensor],
    token_identity_cp_sharded: Optional[torch.Tensor],
) -> Optional[int]:
    """Verify debug identities line up with CP-local tokens and routed experts."""
    if not r3_trace_verify_forward_enabled() or token_identity_cp_sharded is None:
        return None
    if source_routed_experts is None or routed_experts_cp_sharded is None:
        raise RuntimeError(
            "R3 forward verifier expected routed_experts and token identity tensors "
            "to be present together."
        )
    if token_identity_cp_sharded.shape[-1] != 3:
        raise RuntimeError(
            "R3 token identity must have trailing [batch_idx, token_pos, valid] "
            f"dimension; got {tuple(token_identity_cp_sharded.shape)}"
        )

    flat_identity = token_identity_cp_sharded.reshape(-1, 3).to(dtype=torch.long)
    flat_tokens = input_ids_cp_sharded.reshape(-1)
    flat_routed = routed_experts_cp_sharded.reshape(
        -1,
        *routed_experts_cp_sharded.shape[2:],
    )
    if (
        flat_identity.shape[0] != flat_tokens.shape[0]
        or flat_identity.shape[0] != flat_routed.shape[0]
    ):
        raise RuntimeError(
            "R3 token identity, input_ids, and routed_experts CP slices have "
            "different token counts: "
            f"identity={flat_identity.shape[0]} tokens={flat_tokens.shape[0]} "
            f"routed={flat_routed.shape[0]}"
        )

    valid_mask = flat_identity[:, 2] == 1
    checked = int(valid_mask.sum().item())
    if checked == 0:
        return 0

    source_rows = flat_identity[valid_mask, 0]
    source_cols = flat_identity[valid_mask, 1]
    expected_tokens = source_input_ids[source_rows, source_cols].to(
        device=flat_tokens.device,
        dtype=flat_tokens.dtype,
    )
    actual_tokens = flat_tokens[valid_mask]
    if not bool(torch.equal(actual_tokens, expected_tokens)):
        raise RuntimeError(
            "R3 CP token identity verifier found input_ids that do not match "
            "their source [batch_idx, token_pos] identities."
        )

    expected_routed = source_routed_experts[source_rows, source_cols].to(
        device=flat_routed.device,
        dtype=flat_routed.dtype,
    )
    actual_routed = flat_routed[valid_mask]
    if not bool(torch.equal(actual_routed, expected_routed)):
        raise RuntimeError(
            "R3 CP token identity verifier found routed_experts that do not match "
            "their source [batch_idx, token_pos] identities."
        )

    return checked


def _fill_routed_experts_padding(
    routed_experts: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """Replace materialized jagged padding with a valid dummy top-k route."""
    if routed_experts.dim() != 4:
        raise ValueError(
            "routed_experts must have shape [batch, seq, num_moe_layers, topk]; "
            f"got {tuple(routed_experts.shape)}"
        )
    if seq_lengths.shape != (routed_experts.shape[0],):
        raise ValueError(
            "seq_lengths must have one entry per routed_experts row; "
            f"got {tuple(seq_lengths.shape)} for batch={routed_experts.shape[0]}"
        )

    seq_lengths = seq_lengths.to(device=routed_experts.device, dtype=torch.long)
    seq_positions = torch.arange(
        routed_experts.shape[1],
        device=routed_experts.device,
    ).unsqueeze(0)
    padding_mask = seq_positions >= seq_lengths.unsqueeze(1)
    if not bool(padding_mask.any().item()):
        return routed_experts

    repaired = routed_experts.clone()
    default_route = torch.arange(
        routed_experts.shape[-1],
        dtype=routed_experts.dtype,
        device=routed_experts.device,
    ).view(1, 1, 1, routed_experts.shape[-1])
    default_routes = default_route.expand_as(repaired)
    repaired[padding_mask] = default_routes[padding_mask]
    return repaired


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


def _prepare_vlm_batch_for_megatron(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    pad_individual_seqs_to_multiple_of: int,
    pad_full_seq_to: Optional[int] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PackedSeqParams,
    Optional[torch.Tensor],
    torch.Tensor,
]:
    """Prepare a [B, max_seq] batch for a model that does its own packing + CP sharding.

    Used with mbridge VLM wrappers (e.g. Qwen3VL). The model's forward calls
    preprocess_packed_seqs internally, which re-packs + CP-shards from
    attention_mask. So NeMo-RL must NOT pre-pack / CP-shard; it only:
      * pads each sequence (along dim 1) to pad_individual_seqs_to_multiple_of,
      * builds a bool attention_mask describing real token validity,
      * builds cu_seqlens_padded describing full (pre-shard) packed layout,
      * hands everything to the model as [B, max_seq].

    When ``pad_full_seq_to`` is set (PP>1 requires a constant total packed
    length across microbatches), the last sequence's effective length is
    extended so ``sum(padded_lens) == pad_full_seq_to``. These extra positions
    are treated as "valid" by the model (so mbridge's internal packing stays
    consistent) but should be masked out at the loss layer via token_mask.

    Returns:
        - input_ids: packed [1, T] view for downstream logprob/loss target slicing
        - input_ids_cp_sharded: [B, padded_max_seq] for the model forward
        - attention_mask: [B, padded_max_seq] bool (True for valid tokens)
        - packed_seq_params: PackedSeqParams(qkv_format="thd", cu_seqlens_*=padded)
        - cu_seqlens: [B+1] compact logical boundaries used by packed loss
        - cu_seqlens_padded: [B+1] int32 matching packed_seq_params
    """
    batch_size, _ = input_ids.shape
    device = input_ids.device
    align = max(1, pad_individual_seqs_to_multiple_of)

    # One CPU-GPU sync per call via .tolist(); per-seq arithmetic runs on CPU
    # ints (fast) instead of .item() in a loop (which sync'd per seq).
    if torch.is_tensor(seq_lengths):
        lengths_list = seq_lengths.tolist()
    else:
        lengths_list = list(seq_lengths)
    logical_cu_vals = [0]
    for length in lengths_list:
        logical_cu_vals.append(logical_cu_vals[-1] + length)
    cu_seqlens = torch.tensor(
        logical_cu_vals,
        dtype=torch.int32,
        device=device,
    )
    padded_lens = [_round_up_to_multiple(L, align) for L in lengths_list]

    # PP>1: force sum(padded_lens) to a fixed value so every microbatch produces
    # the same decoder-side packed length. We mirror _pack_sequences_for_megatron
    # by absorbing the deficit into the LAST sequence's effective length. The
    # extra positions look valid to the model but are zero-ed out at the loss
    # layer via token_mask (consistent with the non-VLM path).
    if pad_full_seq_to is not None and batch_size > 0:
        natural_sum = sum(padded_lens)
        deficit = pad_full_seq_to - natural_sum
        assert deficit >= 0, (
            f"pad_full_seq_to ({pad_full_seq_to}) < natural padded sum "
            f"({natural_sum}); increase pad_full_seq_to."
        )
        assert deficit % align == 0, (
            f"pad_full_seq_to deficit ({deficit}) must be a multiple of "
            f"pad_individual_seqs_to_multiple_of ({align})."
        )
        if deficit > 0:
            lengths_list[-1] += deficit
            padded_lens[-1] += deficit

    padded_max = max(padded_lens) if padded_lens else 0

    # Row-pad input_ids to padded_max so all sequences live in one rectangular tensor.
    if input_ids.shape[1] < padded_max:
        pad_amt = padded_max - input_ids.shape[1]
        input_ids_2d = torch.nn.functional.pad(input_ids, (0, pad_amt), value=0)
    elif input_ids.shape[1] > padded_max:
        input_ids_2d = input_ids[:, :padded_max].contiguous()
    else:
        input_ids_2d = input_ids

    # Vectorised attention_mask: positions < padded length, broadcast over batch.
    # We use padded_lens (not raw lengths) so mbridge's preprocess_packed_seqs,
    # which recomputes seqlens from attention_mask.sum, sees the same packed
    # total as our cu_seqlens_padded. Otherwise a mismatch between raw length
    # and align-padded length leads to GDN's cu_seqlens vs total_seq_len check
    # firing. Tokens in the padded tail are masked out at the loss layer.
    padded_lens_tensor = torch.tensor(padded_lens, dtype=torch.long, device=device)
    positions = torch.arange(padded_max, device=device)
    attention_mask = positions.unsqueeze(0) < padded_lens_tensor.unsqueeze(1)

    # Build cu_seqlens on CPU then H2D once.
    cu_vals = [0]
    for p in padded_lens:
        cu_vals.append(cu_vals[-1] + p)
    cu_seqlens_padded = torch.tensor(cu_vals, dtype=torch.int32, device=device)

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=padded_max,
        max_seqlen_kv=padded_max,
    )

    # Packed (unsharded) view for downstream logprob / loss code that slices
    # per-sequence targets via cu_seqlens_padded.
    packed_segments = [input_ids_2d[i, :p] for i, p in enumerate(padded_lens)]
    packed_input_ids = (
        torch.cat(packed_segments, dim=0).unsqueeze(0)
        if packed_segments
        else input_ids_2d.new_zeros((1, 0))
    )

    # input_ids_cp_sharded keeps the [B, max_seq] layout: the model (mbridge
    # Qwen3VL) runs its own preprocess_packed_seqs to pack + CP-shard.
    # input_ids is the packed (but not CP-sharded) view for target/logprob
    # post-processing, which uses cu_seqlens_padded to slice per sequence.
    return (
        packed_input_ids,
        input_ids_2d,
        attention_mask,
        packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
    )


def _pack_sequences_for_megatron(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_packed_seq_to: Optional[int] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    thd_max_packed_sequences: Optional[int] = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    sequence_parallel: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    PackedSeqParams,
    torch.Tensor,
    torch.Tensor,
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

    Returns:
        Tuple of:
        - packed_input_ids: Packed input tensor [1, T]
        - input_ids_cp_sharded: Sharded input tensor [cp_size, T // cp_size]
        - packed_seq_params: PackedSeqParams object
        - cu_seqlens: Cumulative sequence lengths
        - cu_seqlens_padded: Padded cumulative sequence lengths
    """
    output = _pack_sequences_for_megatron_with_geometry(
        input_ids=input_ids,
        seq_lengths=seq_lengths,
        pad_individual_seqs_to_multiple_of=pad_individual_seqs_to_multiple_of,
        pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
        pad_packed_seq_to=pad_packed_seq_to,
        cp_rank=cp_rank,
        cp_size=cp_size,
        thd_max_packed_sequences=thd_max_packed_sequences,
        tp_rank=tp_rank,
        tp_size=tp_size,
        sequence_parallel=sequence_parallel,
    )
    return (
        output.input_ids,
        output.input_ids_cp_sharded,
        output.packed_seq_params,
        output.cu_seqlens,
        output.cu_seqlens_padded,
    )


def _pack_sequences_for_megatron_with_geometry(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_packed_seq_to: Optional[int] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    thd_max_packed_sequences: Optional[int] = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    sequence_parallel: bool = False,
) -> _PackedSequenceOutput:
    """Pack THD tensors while preserving real and model-facing geometry."""
    if input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
        )
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid CP rank/size: rank={cp_rank}, size={cp_size}.")
    if pad_individual_seqs_to_multiple_of < 1:
        raise ValueError("pad_individual_seqs_to_multiple_of must be positive.")
    if pad_packed_seq_to_multiple_of < 1:
        raise ValueError("pad_packed_seq_to_multiple_of must be positive.")

    lengths = _validate_packed_sequence_lengths(input_ids, seq_lengths)
    real_sequence_count = len(lengths)
    if thd_max_packed_sequences is not None:
        if thd_max_packed_sequences < 2:
            raise ValueError(
                "thd_max_packed_sequences must reserve at least one real sequence "
                "and one dummy sequence."
            )
        if real_sequence_count > thd_max_packed_sequences - 1:
            raise ValueError(
                f"Packed THD real sequence count ({real_sequence_count}) exceeds "
                f"the configured bound ({thd_max_packed_sequences - 1})."
            )
        if pad_packed_seq_to is None:
            raise ValueError(
                "Fixed THD sequence capacity requires an explicit pad_full_seq_to "
                "token target."
            )

    logical_endpoints = [0]
    natural_physical_endpoints = [0]
    for seq_len in lengths:
        logical_endpoints.append(logical_endpoints[-1] + seq_len)
        physical_len = _round_up_to_multiple(
            seq_len,
            pad_individual_seqs_to_multiple_of,
        )
        if cp_size > 1 and physical_len % (2 * cp_size) != 0:
            raise ValueError(
                f"Packed sequence physical length ({physical_len}) must be divisible "
                f"by 2 * context parallel size ({2 * cp_size})."
            )
        natural_physical_endpoints.append(natural_physical_endpoints[-1] + physical_len)

    natural_padded_tokens = natural_physical_endpoints[-1]
    if pad_packed_seq_to is not None:
        if pad_packed_seq_to % pad_packed_seq_to_multiple_of != 0:
            raise ValueError(
                f"pad_packed_seq_to ({pad_packed_seq_to}) is not a multiple of "
                f"pad_packed_seq_to_multiple_of ({pad_packed_seq_to_multiple_of})."
            )
        capacity_tokens = int(pad_packed_seq_to)
    else:
        capacity_tokens = _round_up_to_multiple(
            natural_padded_tokens,
            pad_packed_seq_to_multiple_of,
        )
    if natural_padded_tokens > capacity_tokens:
        raise ValueError(
            f"Natural packed THD occupancy ({natural_padded_tokens}) exceeds "
            f"token capacity ({capacity_tokens})."
        )
    if capacity_tokens % cp_size != 0:
        raise ValueError(
            f"Packed THD token capacity ({capacity_tokens}) must be divisible by "
            f"context parallel size ({cp_size})."
        )

    fixed_capacity = thd_max_packed_sequences is not None
    physical_endpoints = list(natural_physical_endpoints)
    if not fixed_capacity and capacity_tokens > natural_padded_tokens:
        physical_endpoints[-1] += capacity_tokens - natural_padded_tokens
        last_physical_len = physical_endpoints[-1] - physical_endpoints[-2]
        if cp_size > 1 and last_physical_len % (2 * cp_size) != 0:
            raise ValueError(
                f"Packed sequence physical length ({last_physical_len}) must be "
                f"divisible by 2 * context parallel size ({2 * cp_size})."
            )

    model_physical_endpoints = (
        natural_physical_endpoints if fixed_capacity else physical_endpoints
    )
    physical_lengths = [
        model_physical_endpoints[index + 1] - model_physical_endpoints[index]
        for index in range(real_sequence_count)
    ]
    max_physical_length = max(physical_lengths)
    cu_seqlens = torch.tensor(
        logical_endpoints,
        dtype=torch.int32,
        device=input_ids.device,
    )
    cu_seqlens_padded = torch.tensor(
        model_physical_endpoints,
        dtype=torch.int32,
        device=input_ids.device,
    )
    all_input_ids, input_ids_cp_sharded = _pack_token_aligned_tensors(
        input_ids,
        lengths,
        cu_seqlens_padded,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )

    base_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_physical_length,
        max_seqlen_kv=max_physical_length,
        total_tokens=(None if fixed_capacity else input_ids_cp_sharded.shape[1]),
        pad_between_seqs=not torch.equal(cu_seqlens, cu_seqlens_padded),
    )

    full_structural_mask, local_structural_mask = _build_packed_structural_padding_mask(
        seq_lengths,
        cu_seqlens_padded,
        capacity_tokens=capacity_tokens,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )

    if fixed_capacity:
        dummy_tokens = capacity_tokens - natural_padded_tokens
        if cp_size > 1 and dummy_tokens % (2 * cp_size) != 0:
            raise ValueError(
                f"Fixed THD dummy padding length ({dummy_tokens}) must be divisible "
                f"by 2 * context parallel size ({2 * cp_size})."
            )
        (
            input_ids_cp_sharded,
            _,
            _,
            _,
            packed_seq_params,
            mcore_tail_mask,
        ) = pad_sequence_for_thd(
            tokens=input_ids_cp_sharded,
            labels=None,
            loss_mask=None,
            position_ids=None,
            packed_seq_params=base_params,
            target_len=capacity_tokens // cp_size,
            max_num_seqs=thd_max_packed_sequences,
            context_parallel_size=cp_size,
        )
        if input_ids_cp_sharded is None:
            raise RuntimeError("MCore THD padding unexpectedly removed input tokens.")
        all_input_ids = _pad_token_aligned_tail(
            all_input_ids,
            target_tokens=capacity_tokens,
            value=0,
        )
        if mcore_tail_mask.shape != local_structural_mask.shape:
            raise ValueError(
                "MCore tail mask shape does not match the CP-local structural mask: "
                f"{tuple(mcore_tail_mask.shape)} != "
                f"{tuple(local_structural_mask.shape)}."
            )
        local_structural_mask = local_structural_mask | mcore_tail_mask
    else:
        packed_seq_params = base_params

    if fixed_capacity:
        local_capacity = input_ids_cp_sharded.shape[1]
        packed_seq_params.total_tokens = local_capacity
        packed_seq_params.seq_idx = _build_packed_seq_idx(
            cu_seqlens_padded,
            capacity_tokens=capacity_tokens,
            real_sequence_count=real_sequence_count,
        )
        if packed_seq_params.seq_idx.shape != all_input_ids.shape:
            raise ValueError(
                "Packed Mamba seq_idx shape does not match global post-all-to-all tokens: "
                f"{tuple(packed_seq_params.seq_idx.shape)} != "
                f"{tuple(all_input_ids.shape)}."
            )
        packed_seq_params.seq_aux_loss_sample_ids = (
            _build_packed_seq_aux_loss_sample_ids(
                cu_seqlens_padded,
                capacity_tokens=capacity_tokens,
                real_sequence_count=real_sequence_count,
                cp_rank=cp_rank,
                cp_size=cp_size,
                tp_rank=tp_rank,
                tp_size=tp_size,
                sequence_parallel=sequence_parallel,
            )
        )
        packed_seq_params.seq_aux_loss_num_samples = torch.tensor(
            real_sequence_count,
            dtype=torch.int64,
            device=input_ids.device,
        )
        packed_seq_params.seq_aux_loss_max_samples = thd_max_packed_sequences - 1

    geometry = PackedGeometry(
        logical_tokens=logical_endpoints[-1],
        padded_tokens=natural_padded_tokens,
        capacity_tokens=capacity_tokens,
        real_sequence_count=real_sequence_count,
        cu_seqlens_capacity_entries=int(packed_seq_params.cu_seqlens_q.numel()),
    )
    return _PackedSequenceOutput(
        input_ids=all_input_ids.contiguous(),
        input_ids_cp_sharded=input_ids_cp_sharded.contiguous(),
        packed_seq_params=packed_seq_params,
        cu_seqlens=cu_seqlens,
        cu_seqlens_padded=cu_seqlens_padded,
        structural_padding_mask=full_structural_mask,
        structural_padding_mask_cp_sharded=local_structural_mask,
        packed_geometry=geometry,
    )


def _validate_packed_sequence_lengths(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> list[int]:
    """Validate one logical length per input row and return Python integers."""
    if seq_lengths.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(
            f"seq_lengths must use an integer dtype, got {seq_lengths.dtype}."
        )
    if seq_lengths.dim() != 1 or seq_lengths.numel() != input_ids.shape[0]:
        raise ValueError(
            "seq_lengths must have exactly one entry per input row; "
            f"got {tuple(seq_lengths.shape)} for batch={input_ids.shape[0]}."
        )
    lengths = [int(length) for length in seq_lengths.tolist()]
    if not lengths:
        raise ValueError("Packed THD input must contain at least one real sequence.")
    for index, seq_len in enumerate(lengths):
        if seq_len < 0 or seq_len > input_ids.shape[1]:
            raise ValueError(
                f"Sequence length at index {index} ({seq_len}) is outside "
                f"[0, {input_ids.shape[1]}]."
            )
    return lengths


def _pack_token_aligned_tensors(
    tensor: torch.Tensor,
    seq_lengths: list[int],
    cu_seqlens_padded: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a token-aligned tensor in global and exact per-sequence CP order."""
    global_parts = []
    local_parts = []
    for index, seq_len in enumerate(seq_lengths):
        physical_len = int(cu_seqlens_padded[index + 1] - cu_seqlens_padded[index])
        if physical_len < seq_len:
            raise ValueError(
                f"Physical packed length ({physical_len}) is smaller than logical "
                f"length ({seq_len}) at sequence {index}."
            )
        part = tensor[index, :seq_len]
        if physical_len > seq_len:
            padding = part.new_zeros((physical_len - seq_len, *part.shape[1:]))
            part = torch.cat((part, padding), dim=0)
        global_parts.append(part)
        local_parts.append(
            _get_tokens_on_this_cp_rank(part, cp_rank, cp_size, seq_dim=0)
            if cp_size > 1
            else part
        )
    return (
        torch.cat(global_parts, dim=0).unsqueeze(0),
        torch.cat(local_parts, dim=0).unsqueeze(0),
    )


def _build_packed_structural_padding_mask(
    seq_lengths: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    *,
    capacity_tokens: int,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build global and CP-local THD structural masks in token packing order."""
    if capacity_tokens < 0:
        raise ValueError("Packed THD token capacity cannot be negative.")
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid CP rank/size: rank={cp_rank}, size={cp_size}.")
    if seq_lengths.dim() != 1:
        raise ValueError("seq_lengths must be one-dimensional.")
    if seq_lengths.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(
            f"seq_lengths must use an integer dtype, got {seq_lengths.dtype}."
        )
    if cu_seqlens_padded.dim() != 1:
        raise ValueError("cu_seqlens_padded must be one-dimensional.")
    if cu_seqlens_padded.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(
            "cu_seqlens_padded must use an integer dtype, got "
            f"{cu_seqlens_padded.dtype}."
        )
    if cu_seqlens_padded.numel() != seq_lengths.numel() + 1:
        raise ValueError(
            "cu_seqlens_padded must contain one boundary per real sequence plus "
            "the initial zero."
        )
    if int(cu_seqlens_padded[0]) != 0:
        raise ValueError("cu_seqlens_padded must start at zero.")

    full_parts = []
    local_parts = []
    for index, seq_len_tensor in enumerate(seq_lengths):
        seq_len = int(seq_len_tensor)
        start = int(cu_seqlens_padded[index])
        end = int(cu_seqlens_padded[index + 1])
        physical_len = end - start
        if seq_len < 0 or physical_len < 0 or physical_len < seq_len:
            raise ValueError(
                f"Invalid logical/physical packed bounds at sequence {index}: "
                f"logical={seq_len}, physical={physical_len}."
            )
        if cp_size > 1 and physical_len % (2 * cp_size) != 0:
            raise ValueError(
                f"Packed sequence physical length ({physical_len}) must be divisible "
                f"by 2 * context parallel size ({2 * cp_size})."
            )
        mask = torch.arange(physical_len, device=seq_lengths.device) >= seq_len
        full_parts.append(mask)
        local_parts.append(
            _get_tokens_on_this_cp_rank(mask, cp_rank, cp_size, seq_dim=0)
            if cp_size > 1
            else mask
        )

    padded_tokens = int(cu_seqlens_padded[-1])
    dummy_tokens = capacity_tokens - padded_tokens
    if dummy_tokens < 0:
        raise ValueError(
            f"Packed THD physical occupancy ({padded_tokens}) exceeds capacity "
            f"({capacity_tokens})."
        )
    if cp_size > 1 and dummy_tokens % (2 * cp_size) != 0:
        raise ValueError(
            f"Fixed THD dummy padding length ({dummy_tokens}) must be divisible by "
            f"2 * context parallel size ({2 * cp_size})."
        )
    dummy_mask = torch.ones(
        dummy_tokens,
        dtype=torch.bool,
        device=seq_lengths.device,
    )
    full_parts.append(dummy_mask)
    local_parts.append(dummy_mask[: dummy_tokens // cp_size])
    return (
        torch.cat(full_parts).unsqueeze(0).contiguous(),
        torch.cat(local_parts).unsqueeze(0).contiguous(),
    )


def _build_packed_seq_idx(
    cu_seqlens_padded: torch.Tensor,
    *,
    capacity_tokens: int,
    real_sequence_count: int,
) -> torch.Tensor:
    """Build Mamba sequence IDs in the global order restored by its CP all-to-all."""
    global_parts = []
    for sequence_id in range(real_sequence_count):
        physical_len = int(
            cu_seqlens_padded[sequence_id + 1] - cu_seqlens_padded[sequence_id]
        )
        sequence_ids = torch.full(
            (physical_len,),
            sequence_id,
            dtype=torch.int32,
            device=cu_seqlens_padded.device,
        )
        global_parts.append(sequence_ids)
    dummy_tokens = capacity_tokens - int(cu_seqlens_padded[-1])
    global_parts.append(
        torch.full(
            (dummy_tokens,),
            real_sequence_count,
            dtype=torch.int32,
            device=cu_seqlens_padded.device,
        )
    )
    return torch.cat(global_parts).unsqueeze(0).contiguous()


def _build_packed_seq_aux_loss_sample_ids(
    cu_seqlens_padded: torch.Tensor,
    *,
    capacity_tokens: int,
    real_sequence_count: int,
    cp_rank: int,
    cp_size: int,
    tp_rank: int,
    tp_size: int,
    sequence_parallel: bool,
) -> torch.Tensor:
    local_parts: list[torch.Tensor] = []
    for sample_id in range(real_sequence_count):
        physical_len = int(
            cu_seqlens_padded[sample_id + 1] - cu_seqlens_padded[sample_id]
        )
        ids = torch.full(
            (physical_len,),
            sample_id,
            dtype=torch.int64,
            device=cu_seqlens_padded.device,
        )
        local_parts.append(
            _get_tokens_on_this_cp_rank(ids, cp_rank, cp_size, seq_dim=0)
            if cp_size > 1
            else ids
        )

    dummy_tokens = capacity_tokens - int(cu_seqlens_padded[-1])
    dummy = torch.zeros(
        (dummy_tokens,), dtype=torch.int64, device=cu_seqlens_padded.device
    )
    if cp_size > 1 and dummy_tokens > 0:
        dummy = _get_tokens_on_this_cp_rank(dummy, cp_rank, cp_size, seq_dim=0)
    local = torch.cat((*local_parts, dummy), dim=0)

    if sequence_parallel:
        if tp_size < 1 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"Invalid TP rank/size: rank={tp_rank}, size={tp_size}.")
        if local.numel() % tp_size != 0:
            raise ValueError(
                "CP-local sample IDs must divide evenly across TP/SP ranks."
            )
        width = local.numel() // tp_size
        local = local.narrow(0, tp_rank * width, width)
    return local.contiguous()


def _pad_token_aligned_tail(
    tensor: Optional[torch.Tensor],
    *,
    target_tokens: int,
    value: int,
) -> Optional[torch.Tensor]:
    """Right-pad token dimension 1 without changing existing token semantics."""
    if tensor is None:
        return None
    current_tokens = tensor.shape[1]
    if current_tokens > target_tokens:
        raise ValueError(
            f"Token-aligned tensor length ({current_tokens}) exceeds capacity "
            f"({target_tokens})."
        )
    if current_tokens == target_tokens:
        return tensor
    padding = tensor.new_full(
        (tensor.shape[0], target_tokens - current_tokens, *tensor.shape[2:]),
        value,
    )
    return torch.cat((tensor, padding), dim=1)


def _pad_routed_experts_tail(
    routed_experts: Optional[torch.Tensor],
    *,
    target_tokens: int,
) -> Optional[torch.Tensor]:
    """Pad router replay rows with valid, distinct top-k expert IDs."""
    if routed_experts is None:
        return None
    current_tokens = routed_experts.shape[1]
    if current_tokens > target_tokens:
        raise ValueError(
            f"Routed expert tensor length ({current_tokens}) exceeds capacity "
            f"({target_tokens})."
        )
    if current_tokens == target_tokens:
        return routed_experts
    topk = routed_experts.shape[-1]
    default_route = torch.arange(
        topk,
        dtype=routed_experts.dtype,
        device=routed_experts.device,
    ).view(1, 1, 1, topk)
    padding = default_route.expand(
        routed_experts.shape[0],
        target_tokens - current_tokens,
        routed_experts.shape[2],
        topk,
    )
    return torch.cat((routed_experts, padding), dim=1)


def _pack_token_aligned_tensor_for_megatron(
    tensor: torch.Tensor,
    seq_lengths: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    local_capacity: int,
) -> torch.Tensor:
    """Pack one zero-padded side input in exactly the model token order."""
    lengths = _validate_packed_sequence_lengths(tensor, seq_lengths)
    _, local = _pack_token_aligned_tensors(
        tensor,
        lengths,
        cu_seqlens_padded,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )
    padded = _pad_token_aligned_tail(local, target_tokens=local_capacity, value=0)
    if padded is None:
        raise RuntimeError("Packed token-aligned tensor unexpectedly became None.")
    return padded


def _shard_routed_experts_for_cp(
    routed_experts: Optional[torch.Tensor],  # [B, S, L, K] or None
    token_identity: Optional[
        torch.Tensor
    ],  # [B, S, 3] or None (R3 forward verifier, debug)
    seq_lengths: torch.Tensor,  # [B]
    cu_seqlens: torch.Tensor,  # [B+1] valid cumulative (from _pack_sequences_for_megatron)
    cu_seqlens_padded: Optional[
        torch.Tensor
    ],  # [B+1] padded cumulative (None when no padding)
    cp_rank: int,
    cp_size: int,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """CP-shard routed_experts / token_identity onto main's per-seq packed layout.

    Mirrors _pack_sequences_for_megatron's per-sequence zigzag for input_ids: each
    sequence is padded to its padded length (from cu_seqlens_padded, so boundaries are
    IDENTICAL to input_ids) then sharded with _get_tokens_on_this_cp_rank(seq_dim=0).
    routed_experts pad rows use arange(topk) (a valid top-k route; mcore
    _validate_replay_tensor rejects 0/dup/-1). token_identity pads with 0 (verifier skips).
    No roll (these are per-token, not next-token targets).
    Returns (routed_packed, routed_cp_sharded, identity_packed, identity_cp_sharded),
    each [1, T(/cp), ...] or None.

    This additive helper is the only routed_experts-specific CP code path.
    """
    batch_size = seq_lengths.shape[0]

    all_routed = [] if routed_experts is not None else None
    cp_routed = [] if routed_experts is not None else None
    all_identity = [] if token_identity is not None else None
    cp_identity = [] if token_identity is not None else None
    topk = routed_experts.shape[-1] if routed_experts is not None else None

    for b in range(batch_size):
        seq_len = int(seq_lengths[b])
        if cu_seqlens_padded is not None:
            padded_len = int(cu_seqlens_padded[b + 1] - cu_seqlens_padded[b])
        else:
            padded_len = int(cu_seqlens[b + 1] - cu_seqlens[b])

        if routed_experts is not None:
            # [seq_len, num_moe_layers, topk] padded to the SAME padded_len boundary as
            # input_ids so the routes ride the same per-seq zigzag.
            re = routed_experts[b, :seq_len]
            if padded_len > seq_len:
                # mcore _validate_replay_tensor rejects zero/duplicate/-1 routes,
                # so pad each row with a valid top-k route arange(topk).
                default_route = torch.arange(
                    topk,
                    dtype=re.dtype,
                    device=re.device,
                ).view(1, 1, topk)
                pad_rows = default_route.expand(
                    padded_len - seq_len,
                    re.shape[1],
                    topk,
                )
                re = torch.cat((re, pad_rows), dim=0)
            all_routed.append(re)
            re_cp = (
                _get_tokens_on_this_cp_rank(re, cp_rank, cp_size, seq_dim=0)
                if cp_size > 1
                else re
            )
            cp_routed.append(re_cp)

        if token_identity is not None:
            # [seq_len, 3] padded to the SAME padded_len boundary
            id = token_identity[b, :seq_len]
            if padded_len > seq_len:
                # pad rows with valid=0 so the verifier skips them
                id = torch.nn.functional.pad(
                    id,
                    (0, 0, 0, padded_len - seq_len),
                    value=0,
                )
            all_identity.append(id)
            id_cp = (
                _get_tokens_on_this_cp_rank(id, cp_rank, cp_size, seq_dim=0)
                if cp_size > 1
                else id
            )
            cp_identity.append(id_cp)

    routed_packed = (
        torch.cat(all_routed, dim=0).unsqueeze(0)
        if routed_experts is not None
        else None
    )
    routed_cp_sharded = (
        torch.cat(cp_routed, dim=0).unsqueeze(0) if routed_experts is not None else None
    )
    identity_packed = (
        torch.cat(all_identity, dim=0).unsqueeze(0)
        if token_identity is not None
        else None
    )
    identity_cp_sharded = (
        torch.cat(cp_identity, dim=0).unsqueeze(0)
        if token_identity is not None
        else None
    )
    return routed_packed, routed_cp_sharded, identity_packed, identity_cp_sharded


def _get_pack_sequence_parameters_for_megatron(
    megatron_cfg: dict,
    pad_individual_seqs_to_multiple_of: int,
    max_seq_len_in_batch: int,
):
    """Get pack sequence parameters for Megatron model processing with optional context parallelism.

    Args:
        megatron_cfg: Megatron configuration
        pad_individual_seqs_to_multiple_of: Pad individual sequences to a multiple of this value
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
    minimum_pad_factor = 1
    if cp_size > 1:
        minimum_pad_factor *= cp_size * 2
    if tp_size > 1 and sp:
        minimum_pad_factor *= tp_size
    assert pad_individual_seqs_to_multiple_of % minimum_pad_factor == 0, (
        f"make_sequence_length_divisible_by ({pad_individual_seqs_to_multiple_of}) is not a multiple of minimum_pad_factor ({minimum_pad_factor}).\n"
        f"Please set policy.make_sequence_length_divisible_by to a multiple of {minimum_pad_factor}.\n"
        f"    - If CP is enabled, the minimum pad factor is `cp_size * 2`.\n"
        f"    - If TP+SP is enabled, the minimum pad factor is `tp_size`.\n"
        f"    - If both are enabled, the minimum pad factor is `cp_size * 2 * tp_size`."
    )

    # packed sequence length, after sharding to TP and CP domains, needs to be divisible
    # by a recipe-dependent divisor:
    #   blockwise FP8 : 128  (cublas block size)
    #   MXFP8         :  32  (MXFP8 block size)
    #   other FP8     :  16
    #   HybridEP+flex : 128  (MAX_NUM_OF_TOKENS_PER_RANK must be divisible by
    #                         NUM_OF_TOKENS_PER_CHUNK=128 in deep_ep JIT kernels)
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
