from types import SimpleNamespace

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


def test_pr5672_parameter_move_invalidates_cuda_graphs():
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    class GraphModule:
        def __init__(self) -> None:
            self.cuda_graphs = [object()]

    worker = object.__new__(MegatronPolicyWorkerImpl)
    graph_module = GraphModule()
    worker._get_cg_modules = lambda: [graph_module]
    worker._cuda_graph_helper = object()
    worker._cuda_graph_bucket_helpers = {4096: object()}
    worker._cuda_graph_bucket_graphs = {4096: {graph_module: graph_module.cuda_graphs}}
    worker._cuda_graph_active_bucket = 4096
    worker._cuda_graph_saved_graphs = {graph_module: graph_module.cuda_graphs}
    worker._cuda_graph_captured_seq_length = 4096
    worker._cuda_graph_train_steps = 9
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(cuda_graph_warmup_steps=3)
    )

    worker._invalidate_cuda_graphs_after_parameter_move()

    assert graph_module.cuda_graphs == []
    assert worker._cuda_graph_helper is None
    assert worker._cuda_graph_bucket_helpers == {}
    assert worker._cuda_graph_bucket_graphs == {}
    assert worker._cuda_graph_active_bucket is None
    assert worker._cuda_graph_saved_graphs == {}
    assert worker._cuda_graph_captured_seq_length is None
    assert worker._cuda_graph_train_steps == 3
