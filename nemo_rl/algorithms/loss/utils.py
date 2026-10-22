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

import torch
import torch.distributed

from nemo_rl.algorithms.loss.interfaces import MetricNormalizer
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
from nemo_rl.utils.sequence_lengths import CpuIntTuple, to_cpu_int_tuple


def rescale_loss_metrics(
    metrics: dict[str, Any],
    normalizers: dict[str, MetricNormalizer],
    *,
    token_factor: float,
    sequence_factor: float,
) -> dict[str, Any]:
    """Change global loss denominators while preserving raw counts and extrema.

    Metrics absent from ``normalizers`` are returned unchanged. Unlike
    split-API normalization of raw sums, this only adjusts denominators
    explicitly advertised by the loss.
    """
    factors = {
        MetricNormalizer.TOKENS: token_factor,
        MetricNormalizer.SEQUENCES: sequence_factor,
        MetricNormalizer.NONE: 1.0,
    }
    return {
        key: value * factors[normalizers[key]] if key in normalizers else value
        for key, value in metrics.items()
    }


def map_teacher_logits_to_draft_vocab(
    teacher_logits: torch.Tensor,
    d2t: Optional[torch.Tensor],
    vocab_parallel_rank: Optional[int] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Restrict full-vocab teacher logits to the draft vocabulary via ``d2t``.

    ``d2t`` maps draft-vocab index ``i`` to target-vocab index ``i + d2t[i]``.
    Under tensor parallelism the teacher logits arrive vocab-sharded, so they
    are gathered to the full vocab and re-sliced to this rank's shard of the
    draft vocabulary (the draft output layer is sharded the same way). No-op
    when ``d2t`` is None (full-vocab drafts).
    """
    if d2t is None:
        return teacher_logits
    reverse_mapping = (
        torch.arange(len(d2t), device=teacher_logits.device, dtype=d2t.dtype) + d2t
    )
    if vocab_parallel_group is not None:
        from megatron.core.tensor_parallel import (
            gather_from_tensor_model_parallel_region,
        )

        teacher_logits = gather_from_tensor_model_parallel_region(
            teacher_logits, vocab_parallel_group
        )
        tp_size = torch.distributed.get_world_size(vocab_parallel_group)
        local_draft_size = len(d2t) // tp_size
        assert vocab_parallel_rank is not None
        start_index = vocab_parallel_rank * local_draft_size
        end_index = (vocab_parallel_rank + 1) * local_draft_size
        reverse_mapping = reverse_mapping[start_index:end_index]
    return teacher_logits[:, :, reverse_mapping]


def roll_packed_seq_dim(
    tensor: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    seq_dim: int,
) -> torch.Tensor:
    """Left-shift a packed tensor by one along ``seq_dim`` within each segment.

    Equivalent to a per-sequence ``torch.roll(shifts=-1)`` over the packed
    layout: one global roll followed by zeroing each segment's final slot (the
    only positions where the global roll would leak the next segment's first
    row). Segment boundaries come from ``cu_seqlens_padded``, the physical
    offsets of the packed layout.
    """
    rolled = torch.roll(tensor, shifts=-1, dims=seq_dim)
    boundary_index = (cu_seqlens_padded[1:] - 1).to(
        dtype=torch.long, device=rolled.device
    )
    index: list[Any] = [slice(None)] * rolled.dim()
    index[seq_dim] = boundary_index
    rolled[tuple(index)] = 0
    return rolled


def pack_rolled_draft_token_mask(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
) -> torch.Tensor:
    """Build the packed draft-loss mask ``[1, T_packed]`` from unpacked masks.

    Mirrors the non-packed DRAFT prepare (``token_mask`` left-shifted by one,
    scaled by ``sample_mask``), laid out at each sequence's padded offset.
    Each sequence's last real slot (whose shifted target would cross the
    boundary) and all padding slots stay zero.

    Reuses ``_pack_input_ids`` for the padded-offset layout (including its
    clamp for bin-alignment padding absorbed into the last sequence's
    effective length) and ``roll_packed_seq_dim`` for the boundary-safe
    per-segment left shift.
    """
    packed = _pack_input_ids(
        token_mask * sample_mask.unsqueeze(-1),
        to_cpu_int_tuple(cu_seqlens),
        to_cpu_int_tuple(cu_seqlens_padded),
    )
    return roll_packed_seq_dim(packed, cu_seqlens_padded, seq_dim=1)


def _project_hidden_states_per_teacher(
    payload: torch.Tensor,
    weight_by_index: dict[int, torch.Tensor],
    teacher_index: torch.Tensor,
) -> torch.Tensor:
    """Project each sample's hidden states through its own teacher's LM head.

    ``payload`` is ``[B, S, H]``; ``teacher_index`` is ``[B]`` int, naming which
    loaded teacher LM head produced each row (see ``OPD_FULL_TEACHER_INDEX_FIELD``).
    Grouped by teacher so this costs one matmul per distinct teacher present in
    the microbatch, not a per-sample loop -- when sequence packing is on, each
    call already has ``B == 1`` (one sequence at a time), so it takes the
    single-teacher path and never materializes the scatter buffer.

    Raises:
        ValueError: If ``teacher_index`` does not carry one entry per payload
            row, if a row's teacher_index has no loaded LM head, or if an LM
            head disagrees with the payload or with the other teachers'.
    """
    teacher_index = teacher_index.to(device=payload.device)
    if int(teacher_index.shape[0]) != int(payload.shape[0]):
        raise ValueError(
            "opd_full teacher_index must carry one entry per payload row: got "
            f"{tuple(teacher_index.shape)} against payload {tuple(payload.shape)}."
        )

    def _weight_for(idx: int) -> torch.Tensor:
        weight = weight_by_index.get(idx)
        if weight is None:
            raise ValueError(
                f"opd_full row tagged teacher_index={idx}, but no teacher LM "
                f"head is loaded for that index (loaded: {sorted(weight_by_index)})."
            )
        if int(payload.shape[-1]) != int(weight.shape[1]):
            raise ValueError(
                "Teacher hidden states do not match the loaded teacher LM head "
                f"for teacher_index={idx}: payload width {payload.shape[-1]} vs "
                f"LM-head input width {weight.shape[1]}."
            )
        return weight

    present = [int(idx) for idx in teacher_index.unique().tolist()]
    if len(present) == 1:
        # The sequence-packing path lands here on every call. Projecting
        # straight into the result skips the scatter buffer below, which would
        # otherwise double the peak ``[B, S, V_local]`` allocation for nothing.
        weight = _weight_for(present[0])
        return torch.matmul(payload.to(dtype=weight.dtype), weight.t())

    # One buffer holds every group, so the groups have to agree on the
    # vocabulary shard and dtype. They do by construction -- every shard is
    # requested at the student's own width and dtype -- but say so here, since
    # the alternative is an opaque RuntimeError out of the masked assignment.
    reference_index, reference = next(iter(weight_by_index.items()))
    teacher_logits = torch.empty(
        payload.shape[0],
        payload.shape[1],
        int(reference.shape[0]),
        dtype=reference.dtype,
        device=payload.device,
    )
    for idx in present:
        weight = _weight_for(idx)
        if (int(weight.shape[0]), weight.dtype) != (
            int(reference.shape[0]),
            reference.dtype,
        ):
            raise ValueError(
                "opd_full teacher LM heads must share one vocabulary shard and "
                f"dtype to be projected together: teacher_index={idx} is "
                f"[{weight.shape[0]}, ...] {weight.dtype}, but "
                f"teacher_index={reference_index} is [{reference.shape[0]}, ...] "
                f"{reference.dtype}."
            )
        mask = teacher_index == idx
        teacher_logits[mask] = torch.matmul(
            payload[mask].to(dtype=weight.dtype), weight.t()
        )
    return teacher_logits


def reconstruct_opd_full_teacher_logits(
    payload: torch.Tensor,
    *,
    teacher_payload: str,
    student_logits: torch.Tensor,
    vocab_parallel_rank: Optional[int],
    context_parallel_group: Optional[torch.distributed.ProcessGroup],
    teacher_output_layer_weight_by_index: Optional[dict[int, torch.Tensor]] = None,
    teacher_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Turn the transported teacher payload into this rank's teacher logit shard.

    The payload is canonical ``[B, S, D]`` (already TP/CP-gathered on the teacher
    side), so it is re-sharded onto the student's own CP window here, exactly as
    ``from_parallel_logits_to_logprobs`` does for the student logits.

    Args:
        payload: Teacher payload ``[B, S, D]`` -- hidden states or full logits.
        teacher_payload: ``"hidden_states"`` or ``"logits"``.
        student_logits: Student logits ``[B, S_local, V_local]``, used for the
            target device, dtype-independent shapes, and CP geometry.
        vocab_parallel_rank: This rank's vocabulary-parallel rank.
        context_parallel_group: Context-parallel process group, if any.
        teacher_output_layer_weight_by_index: ``[V_local, H_teacher]`` teacher
            LM-head shards for the hidden-state path, keyed by the stable index
            rows are tagged with. Routed by ``teacher_index`` whenever that
            column is present.
        teacher_index: ``[B]`` int, this microbatch's per-row teacher index
            (see ``OPD_FULL_TEACHER_INDEX_FIELD``). Required once
            ``teacher_output_layer_weight_by_index`` holds more than one entry,
            and validated against the loaded heads whenever present.

    Returns:
        Teacher logits ``[B, S_local, V_local]`` aligned with ``student_logits``.

    Raises:
        ValueError: If the payload and student shard cannot be aligned, if the
            hidden-state path is used without a teacher LM-head shard, or if
            several shards are loaded without a ``teacher_index`` to route by.
    """
    vocab_shard_size = int(student_logits.shape[-1])

    # Every narrowing below happens before the device copy, and before the
    # hidden-state path widens the payload to the vocabulary.
    if teacher_payload == "hidden_states":
        if not teacher_output_layer_weight_by_index:
            raise ValueError(
                "opd_full hidden-state reconstruction requires a loaded teacher "
                "output-layer weight shard on the training worker."
            )
        if len(teacher_output_layer_weight_by_index) > 1 and teacher_index is None:
            # Refuse to guess rather than fall back to one arbitrary shard: that
            # keeps every shape right and silently distills the whole microbatch
            # toward whichever teacher the dict happens to yield first.
            raise ValueError(
                f"opd_full has {len(teacher_output_layer_weight_by_index)} "
                "teacher LM-head shards loaded, but this microbatch carries "
                "no per-row teacher index (OPD_FULL_TEACHER_INDEX_FIELD); "
                "there is no way to tell which teacher produced each row."
            )
        # Every loaded shard, not just the ones this microbatch routes to: the
        # widths are uniform by construction (the worker rejects a teacher whose
        # hidden size disagrees at load time), so a mismatch here is a real
        # inconsistency and should not depend on which rows happened to arrive.
        for idx, weight in sorted(teacher_output_layer_weight_by_index.items()):
            if int(payload.shape[-1]) != int(weight.shape[1]):
                raise ValueError(
                    "Teacher hidden states do not match the loaded teacher LM head "
                    f"for teacher_index={idx}: payload width {payload.shape[-1]} vs "
                    f"LM-head input width {weight.shape[1]}."
                )
    else:
        assert vocab_parallel_rank is not None, (
            "vocab_parallel_rank is required to slice the opd_full logits payload"
        )
        vocab_start_index = vocab_parallel_rank * vocab_shard_size
        vocab_end_index = vocab_start_index + vocab_shard_size
        if int(payload.shape[-1]) < vocab_end_index:
            raise ValueError(
                "Teacher logits payload is narrower than this rank's vocabulary "
                f"window: payload width {payload.shape[-1]} vs required "
                f"{vocab_end_index}."
            )
        payload = payload[..., vocab_start_index:vocab_end_index]

    # Narrow to this rank's sequence window *before* the payload is widened to
    # the vocabulary. Projecting first would do cp_size times the matmul and
    # allocate a full-sequence [B, S, V_local] tensor that the divergence
    # kernel's chunking cannot bound.
    cp_size = (
        1
        if context_parallel_group is None
        else torch.distributed.get_world_size(context_parallel_group)
    )
    target_seq_len = int(student_logits.shape[1]) * cp_size
    pad_len = target_seq_len - int(payload.shape[1])
    if pad_len < 0:
        raise ValueError(
            "Teacher payload is longer than the student forward window: "
            f"{payload.shape[1]} vs {target_seq_len}."
        )
    if pad_len > 0:
        # Zero-padding survives the projection (the LM head has no bias), so the
        # padded positions carry the same uniform logits either way. They are
        # masked out downstream regardless.
        payload = torch.nn.functional.pad(payload, (0, 0, 0, pad_len))
    if cp_size > 1:
        cp_rank = torch.distributed.get_rank(context_parallel_group)
        payload = _get_tokens_on_this_cp_rank(payload, cp_rank, cp_size, seq_dim=1)

    # contiguous() before the copy: the vocabulary slice above is a view, and the
    # result is handed to save_for_backward. Without it the whole-vocabulary
    # payload storage stays resident on the device until backward completes.
    payload = payload.contiguous().to(device=student_logits.device)

    if teacher_payload == "hidden_states":
        # Validated above, before any narrowing ran.
        assert teacher_output_layer_weight_by_index is not None
        if teacher_index is not None:
            # Route by tag whenever the column is present: a tag naming an
            # unloaded teacher must fail loud, not take the only head.
            teacher_logits = _project_hidden_states_per_teacher(
                payload, teacher_output_layer_weight_by_index, teacher_index
            )
        else:
            weight = next(iter(teacher_output_layer_weight_by_index.values()))
            teacher_logits = torch.matmul(
                payload.to(dtype=weight.dtype),
                weight.t(),
            )
    else:
        teacher_logits = payload

    if int(teacher_logits.shape[-1]) != vocab_shard_size:
        raise ValueError(
            "Reconstructed teacher logits must match the student vocabulary shard "
            f"width; got {teacher_logits.shape[-1]} vs {vocab_shard_size}."
        )
    return teacher_logits


def roll_packed_left_cp(
    tensor: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Left-shift by one WITHIN each packed subsequence, zigzag-CP-aware.

    ``tensor``'s dim 0 is the rank-local packed (THD) dimension and
    ``cu_seqlens_local`` its subsequence boundaries. Without CP each
    subsequence rolls independently with its tail zero-filled. Under CP the
    local row of every subsequence is the zigzag pair ``[chunk_r,
    chunk_{2cp-1-r}]`` (the ``_get_tokens_on_this_cp_rank`` layout), so each
    chunk rolls locally and its tail element is the head of the globally NEXT
    chunk: chunk ``r``'s successor lives on rank ``r+1`` (or is this rank's
    own back chunk when ``r == cp-1``), chunk ``2cp-1-r``'s successor on rank
    ``r-1`` (or is the global sequence tail, zero, when ``r == 0``). All
    subsequences' boundary elements travel in ONE batched isend/irecv.
    """
    cp_size = 1 if cp_group is None else cp_group.size()
    cu = cu_seqlens_local.to(dtype=torch.long)
    seq_lens = cu[1:] - cu[:-1]

    # Global roll first, then patch every chunk-tail row (which the global
    # roll filled with the next chunk's head IN LOCAL LAYOUT — wrong across
    # subsequence/zigzag boundaries).
    rolled = torch.roll(tensor, shifts=-1, dims=0)

    if cp_size == 1:
        rolled[cu[1:] - 1] = 0
        return rolled

    cp_rank = cp_group.rank()
    global_ranks = torch.distributed.get_process_group_ranks(cp_group)
    next_rank = global_ranks[(cp_rank + 1) % cp_size]
    prev_rank = global_ranks[(cp_rank - 1) % cp_size]

    half = seq_lens // 2
    front_head = cu[:-1]
    back_head = cu[:-1] + half
    front_tail = back_head - 1
    back_tail = cu[1:] - 1

    send_to_prev = tensor[front_head]  # prev's front-chunk tails need these
    send_to_next = tensor[back_head]  # next's back-chunk tails need these
    recv_from_next = torch.empty_like(send_to_prev)
    recv_from_prev = torch.empty_like(send_to_next)

    ops = []
    if cp_rank > 0:
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend, send_to_prev, prev_rank, cp_group
            )
        )
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_from_prev, prev_rank, cp_group
            )
        )
    if cp_rank < cp_size - 1:
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend, send_to_next, next_rank, cp_group
            )
        )
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_from_next, next_rank, cp_group
            )
        )
    if ops:
        for req in torch.distributed.batch_isend_irecv(ops):
            req.wait()
    if cp_rank == cp_size - 1:
        # Chunk cp-1's global successor is chunk cp — this rank's own back chunk.
        recv_from_next = tensor[back_head]
    if cp_rank == 0:
        # Chunk 2cp-1 is the global sequence tail.
        recv_from_prev = torch.zeros_like(recv_from_prev)

    rolled[front_tail] = recv_from_next
    rolled[back_tail] = recv_from_prev
    return rolled


def packed_zigzag_token_coords(
    cu_seqlens_global: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each rank-local packed token to ``(subseq index, global position)``.

    ``cu_seqlens_global`` holds the PRE-CP padded boundaries. The local
    layout is the per-subsequence zigzag (`_get_tokens_on_this_cp_rank`):
    rank ``r`` holds chunks ``r`` and ``2*cp - 1 - r`` of every subsequence.
    Returns two ``[T_local]`` int64 tensors: the packed subsequence index
    (== the pre-packing batch row) and the token's position WITHIN its
    subsequence in the original (un-sharded) order. Pure local arithmetic,
    no communication — the basis for gathering per-token masks/labels from
    unpacked ``[B, S]`` tensors.
    """
    cu = cu_seqlens_global.to(dtype=torch.long)
    lens_local = (cu[1:] - cu[:-1]) // cp_size
    num_seqs = lens_local.numel()
    device = cu.device
    cu_local = torch.zeros(num_seqs + 1, dtype=torch.long, device=device)
    cu_local[1:] = torch.cumsum(lens_local, dim=0)
    seq_index = torch.repeat_interleave(
        torch.arange(num_seqs, device=device), lens_local
    )
    pos_local = (
        torch.arange(int(cu_local[-1].item()), device=device) - cu_local[:-1][seq_index]
    )
    if cp_size == 1:
        return seq_index, pos_local
    half = (lens_local // 2)[seq_index]
    is_front = pos_local < half
    pos_global = torch.where(
        is_front,
        cp_rank * half + pos_local,
        (2 * cp_size - 1 - cp_rank) * half + (pos_local - half),
    )
    return seq_index, pos_global


def packed_zigzag_local_index(
    seq_idx: torch.Tensor,
    pos: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Rank-local THD row of ``(subseq, global in-seq position)``.

    The caller guarantees this rank owns each position's zigzag chunk
    (``pos // half in {cp_rank, 2*cp - 1 - cp_rank}``).
    """
    cu = cu_seqlens_local.to(dtype=torch.long)
    half = ((cu[1:] - cu[:-1]) // 2)[seq_idx]
    front_off = pos - cp_rank * half
    is_front = (front_off >= 0) & (front_off < half)
    back_off = pos - (2 * cp_size - 1 - cp_rank) * half
    return cu[:-1][seq_idx] + torch.where(is_front, front_off, half + back_off)


def packed_zigzag_successor_halo(
    tensor: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    halo: int,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """First ``halo`` rows of each local chunk's globally NEXT chunk.

    ``tensor``'s dim 0 is the rank-local zigzag THD row. Returns
    ``[num_seqs, 2, halo, ...]`` (index 0 = front chunk's successor, 1 = back
    chunk's). Same neighbor pattern as :func:`roll_packed_left_cp`: front
    chunk ``r``'s successor is rank ``r+1``'s front chunk head (``r == cp-1``:
    this rank's own back chunk), back chunk ``2cp-1-r``'s successor is rank
    ``r-1``'s back chunk head (``r == 0``: past the sequence end, zeros).
    Requires ``halo <= min(half)`` — a successor chunk must cover the halo.
    """
    cp_size = 1 if cp_group is None else cp_group.size()
    cu = cu_seqlens_local.to(dtype=torch.long)
    half = (cu[1:] - cu[:-1]) // 2
    if halo > int(half.min().item()):
        raise ValueError(
            f"successor halo {halo} exceeds the smallest local chunk "
            f"{int(half.min().item())}; shorten gamma or lower CP for this "
            "sequence length."
        )
    num_seqs = half.numel()
    row_shape = tensor.shape[1:]
    head_index = torch.arange(halo, device=tensor.device).view(1, -1) + cu[:-1].view(
        -1, 1
    )
    front_head = tensor[head_index.reshape(-1)].reshape(num_seqs, halo, *row_shape)
    back_head = tensor[(head_index + half.view(-1, 1)).reshape(-1)].reshape(
        num_seqs, halo, *row_shape
    )

    if cp_size == 1:
        return torch.stack([back_head, torch.zeros_like(back_head)], dim=1)

    cp_rank = cp_group.rank()
    global_ranks = torch.distributed.get_process_group_ranks(cp_group)
    next_rank = global_ranks[(cp_rank + 1) % cp_size]
    prev_rank = global_ranks[(cp_rank - 1) % cp_size]

    recv_front_halo = torch.empty_like(front_head)
    recv_back_halo = torch.empty_like(back_head)
    ops = []
    if cp_rank > 0:
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend, front_head.contiguous(), prev_rank, cp_group
            )
        )
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_back_halo, prev_rank, cp_group
            )
        )
    if cp_rank < cp_size - 1:
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend, back_head.contiguous(), next_rank, cp_group
            )
        )
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_front_halo, next_rank, cp_group
            )
        )
    if ops:
        for req in torch.distributed.batch_isend_irecv(ops):
            req.wait()
    if cp_rank == cp_size - 1:
        recv_front_halo = back_head
    if cp_rank == 0:
        recv_back_halo = torch.zeros_like(back_head)
    return torch.stack([recv_front_halo, recv_back_halo], dim=1)


def draft_pass_token_mask(token_mask: torch.Tensor, ttt_pass: int) -> torch.Tensor:
    """Per-pass draft loss mask (before the sample_mask factor).

    Pass ``d`` position ``i`` predicts token ``x_{i+d+1}``, so its validity is
    ``token_mask[i + d + 1]``: a left shift by ``d + 1`` that also drops the
    out-of-bounds tail. Returned shape is ``[B, S - d - 1]``, aligned with the
    ``student[:, :S-d-1]`` / ``teacher[:, d:S-1]`` slices used by
    ``DraftCrossEntropyLossFn``.
    """
    return token_mask[:, ttt_pass + 1 :]


def compute_draft_pass_valid_counts(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    ttt_steps: int,
) -> torch.Tensor:
    """Local per-pass valid-token counts for the draft TTT loss.

    Returns a fp32 tensor of shape ``[ttt_steps]`` where entry ``d-1`` is this
    rank's ``#valid pass-d targets``. The caller must all-reduce it over the
    data-parallel group (and only DP: every CP/TP rank computes the loss over
    the full sequence/vocab, so reducing over those groups would double-count).

    The loss consumes the vector twice: the gradient denominator is
    ``sum_d alpha_d * counts[d-1]``, and the per-pass diagnostic metrics divide
    by ``counts[d-1]`` so that the driver's sum-over-microbatches aggregation
    (grpo.py logs mb-metric lists with ``np.sum``) reconstructs the global
    per-token mean instead of summing per-microbatch means.
    """
    counts = token_mask.new_zeros((ttt_steps,), dtype=torch.float32)
    for ttt_pass in range(1, ttt_steps + 1):
        pass_mask = draft_pass_token_mask(token_mask, ttt_pass) * sample_mask.unsqueeze(
            -1
        )
        counts[ttt_pass - 1] = pass_mask.sum().float()
    return counts


def block_draft_slot_mask(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    anchors: torch.Tensor,
    anchor_valid: torch.Tensor,
    *,
    gamma: int,
) -> torch.Tensor:
    """Validity mask ``[B, N, gamma]`` for DFlash/DSpark block draft slots.

    Slot ``j`` of a block anchored at ``p`` predicts ``x_{p + 1 + j}``; it is
    valid when the block is real (``anchor_valid``), the label position is in
    bounds, the label token is trained (``token_mask``), and the sample is
    valid. Anchors near the sequence end keep their in-bounds slots instead of
    being rejected outright (see the CP design note's packing issue ③).

    Validity is PREFIX-CONTIGUOUS along the slot axis: once a slot is invalid
    every later slot is too. A speculative block at serving time never spans a
    user/tool/prefill hole, so an anchor whose labels cross such a hole must
    not train the slots beyond it (multi-turn data would otherwise resurrect
    slots on the far side of the hole).
    """
    batch_size, num_anchors = anchors.shape
    seq_idx = (
        torch.arange(batch_size, device=anchors.device)
        .unsqueeze(1)
        .expand(-1, num_anchors)
        .reshape(-1)
    )
    return block_draft_slot_mask_packed(
        token_mask,
        sample_mask,
        seq_idx,
        anchors.reshape(-1),
        anchor_valid.reshape(-1),
        gamma=gamma,
    ).reshape(batch_size, num_anchors, gamma)


def block_draft_slot_mask_packed(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    seq_idx: torch.Tensor,
    anchors: torch.Tensor,
    anchor_valid: torch.Tensor,
    *,
    gamma: int,
) -> torch.Tensor:
    """Flat-block variant of :func:`block_draft_slot_mask`.

    ``seq_idx``/``anchors``/``anchor_valid`` are ``[NB]`` (one row per block,
    e.g. this CP rank's owned subset of a packed microbatch); the masks stay
    unpacked ``[B, S]``/``[B]``. Returns ``[NB, gamma]``.
    """
    seq_len = token_mask.shape[1]
    label_pos = (
        anchors.unsqueeze(-1)
        + 1
        + torch.arange(gamma, device=anchors.device).view(1, -1)
    )
    in_bounds = label_pos < seq_len
    label_token_mask = token_mask[
        seq_idx.unsqueeze(-1).expand(-1, gamma), label_pos.clamp(max=seq_len - 1)
    ]
    slot_mask = (
        anchor_valid.unsqueeze(-1)
        & in_bounds
        & (label_token_mask > 0.5)
        & (sample_mask[seq_idx].unsqueeze(-1) > 0.5)
    )
    return torch.cumprod(slot_mask.to(torch.int32), dim=-1).bool()


def compute_block_draft_slot_valid_counts(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    anchors: torch.Tensor,
    anchor_valid: torch.Tensor,
    *,
    gamma: int,
) -> torch.Tensor:
    """Local per-slot valid counts ``[gamma]`` for the block draft loss.

    The block analogue of :func:`compute_draft_pass_valid_counts` (same
    DP-only all-reduce contract and the same denominator/metric consumption
    in the loss): entry ``j`` counts this rank's valid slot-``j`` CE targets.
    """
    slot_mask = block_draft_slot_mask(
        token_mask, sample_mask, anchors, anchor_valid, gamma=gamma
    )
    return slot_mask.float().sum(dim=(0, 1))


def _pack_input_ids(
    input_ids: torch.Tensor,
    cu_seqlens_q: CpuIntTuple,
    cu_seqlens_q_padded: CpuIntTuple,
    cp_rank: int = 0,
    cp_size: int = 1,
    roll_shift: int = 0,
) -> torch.Tensor:
    """Pack input_ids from [B, S] to [1, T_packed // CP] using sequence boundaries.

    Each sequence is individually padded to its padded length (from
    cu_seqlens_q_padded), optionally rolled, and CP-sharded at that padded
    length before being placed into the packed output.  This matches how
    Megatron packs and CP-shards sequences in _pack_sequences_for_megatron.

    Args:
        input_ids: Unpacked input IDs [B, S].
        cu_seqlens_q: Unpadded cumulative sequence lengths [B+1].
        cu_seqlens_q_padded: Padded cumulative sequence lengths [B+1].
        cp_rank: Context parallelism rank.
        cp_size: Context parallelism size.
        roll_shift: If non-zero, roll each padded sequence by this amount
            before CP-sharding.  Use -1 to build shifted targets for
            next-token prediction.
    """
    batch_size = input_ids.shape[0]
    total_packed_len = cu_seqlens_q_padded[-1] // cp_size
    packed = torch.zeros(
        total_packed_len, dtype=input_ids.dtype, device=input_ids.device
    )
    for i in range(batch_size):
        actual_len = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        padded_len = cu_seqlens_q_padded[i + 1] - cu_seqlens_q_padded[i]
        packed_start = cu_seqlens_q_padded[i]
        seq = torch.zeros(padded_len, dtype=input_ids.dtype, device=input_ids.device)
        # The packer absorbs bin-level alignment padding into the last
        # sequence's effective length (see _get_pack_sequence_parameters_for_megatron),
        # so cu_seqlens can exceed the unpacked row width. Copy only real
        # tokens; the tail stays zero and is excluded from the loss by token_mask.
        copy_len = min(actual_len, input_ids.shape[1])
        seq[:copy_len] = input_ids[i, :copy_len]
        if roll_shift != 0:
            seq = seq.roll(shifts=roll_shift, dims=0)
        sharded = _get_tokens_on_this_cp_rank(seq, cp_rank, cp_size, seq_dim=0)
        packed[packed_start // cp_size : (packed_start + padded_len) // cp_size] = (
            sharded
        )
    return packed.unsqueeze(0)
