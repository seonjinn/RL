import torch


def test_pr5672_sample_packed_seq_params_matches_cuda_graph_bucket():
    from megatron.core.packed_seq_params import PackedSeqParams
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    cu_seqlens = torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=8,
        max_seqlen_kv=8,
    )

    sample = MegatronPolicyWorkerImpl._make_cuda_graph_sample_packed_seq_params(
        packed_seq_params, 4096
    )

    expected = torch.tensor([0, 4096, 4096, 4096, 4096], dtype=torch.int32)
    assert sample.qkv_format == "thd"
    assert sample.max_seqlen_q == 4096
    assert sample.max_seqlen_kv == 4096
    assert torch.equal(sample.cu_seqlens_q, expected)
    assert torch.equal(sample.cu_seqlens_q_padded, expected)
