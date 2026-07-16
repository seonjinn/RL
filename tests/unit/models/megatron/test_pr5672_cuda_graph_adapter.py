import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams
from nemo_rl.models.megatron.data import process_microbatch
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


def test_packed_seq_params_pad_cu_seqlens_to_cuda_graph_sequence_budget() -> None:
    result = process_microbatch(
        data_dict={
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "input_lengths": torch.tensor([3, 2]),
        },
        seq_length_key="input_lengths",
        pack_sequences=True,
        pad_full_seq_to=8,
        cuda_graph_max_packed_seqs=4,
    )

    assert torch.equal(
        result.cu_seqlens_padded, torch.tensor([0, 3, 8], dtype=torch.int32)
    )
    assert torch.equal(
        result.packed_seq_params.cu_seqlens_q,
        torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
    )
    assert torch.equal(
        result.packed_seq_params.cu_seqlens_q_padded,
        torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
    )


def test_packed_seq_params_rejects_more_sequences_than_cuda_graph_budget() -> None:
    with pytest.raises(ValueError, match="cuda_graph_max_packed_seqs"):
        process_microbatch(
            data_dict={
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]]),
                "input_lengths": torch.tensor([3, 2, 1]),
            },
            seq_length_key="input_lengths",
            pack_sequences=True,
            pad_full_seq_to=8,
            cuda_graph_max_packed_seqs=2,
        )


def test_cuda_graph_sample_packed_seq_params_matches_bucket_and_tensor_shape() -> None:
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
        max_seqlen_q=8,
        max_seqlen_kv=8,
        local_cp_size=1,
    )

    sample = MegatronPolicyWorkerImpl._make_cuda_graph_sample_packed_seq_params(
        packed_seq_params, 16
    )

    assert sample.qkv_format == "thd"
    assert sample.max_seqlen_q == 16
    assert sample.max_seqlen_kv == 16
    assert sample.local_cp_size == 1
    assert torch.equal(sample.cu_seqlens_q, torch.tensor([0, 16, 16, 16, 16]))
    assert torch.equal(sample.cu_seqlens_q_padded, torch.tensor([0, 16, 16, 16, 16]))
