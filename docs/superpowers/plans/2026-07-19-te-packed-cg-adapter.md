# TE Packed Partial CUDA Graph Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Port the PR5672 Transformer Engine packed-THD partial CUDA-graph contract into the current NeMo-RL CUDA-graph branch without changing Qwen3-30B-A3B training semantics.

**Architecture:** Keep cuda_graph_impl=transformer_engine and graph only the selected partial module scope. The NeMo-RL data path supplies fixed-shape graph-facing packed metadata while retaining real packed boundaries for loss; the worker supplies a TE capture sample and clears every graph before refit relocates parameters.

**Tech Stack:** Python 3.13, PyTorch, Ray, NeMo-RL, Megatron-Core, Transformer Engine, pytest, SLURM/Ptyche.

## Global Constraints

- Work only in /Users/sna/CudaGraph_PR/RL-pr5672-adapter-ptyche-20260719 on experiment/te-packed-cg-adapter-20260719.
- Preserve all non-CUDA-graph changes from parent 00b4cfed7297366ec5c87174fdcacdfee498f330.
- Pin MCore to bed605f292f926090f5f43ba5e30fb024c2306dc; never use a moving PR branch.
- Use cuda_graph_impl: transformer_engine; do not integrate PR5783 local THD or PR4359.
- Production scope is attn and router dtype remains FP64. Router, all-to-all dispatch, and experts stay eager.
- Keep three initial warmup steps. After refit, capture on the next eligible training call because old parameter pointers are invalid.
- cuda_graph_max_packed_seqs is workload-specific. Graph PSP shape is Nmax + 1; an oversized batch fails clearly.
- Disable checkpoint saving in all benchmark and convergence jobs.
- Run tests and GPU jobs in the Ptyche Linux container. macOS cannot run this lockfile because it targets Linux.

---

## File Map

| File | Responsibility |
|---|---|
| 3rdparty/Megatron-LM-workspace/Megatron-LM | MCore PR5672 API accepting sample_packed_seq_params. |
| nemo_rl/models/megatron/data.py | Fixed-shape graph PSP and original loss boundaries. |
| nemo_rl/models/megatron/train.py | Sends original boundaries to the sequence-packing loss wrapper. |
| nemo_rl/models/policy/workers/megatron_policy_worker.py | Capture sample, iterator peek, graph invalidation before CPU offload. |
| tests/unit/models/megatron/test_megatron_data.py | Static PSP and packed-boundary tests. |
| tests/unit/models/megatron/test_train.py | Loss-boundary forwarding tests. |
| tests/unit/models/policy/test_megatron_worker.py | Capture sample and parameter-relocation tests. |
| tests/unit/models/megatron/test_megatron_setup.py | Production recipe contract test. |
| examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nocg-adapter.yaml | Comparable packed no-CG adapter baseline. |
| examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn.yaml | FP64 production attention recipe. |
| examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router-w3.yaml | FP32 router diagnostic recipe. |
| experiments/cuda_graph/launch_qwen30_moe_cg_comparison_ptyche.sh | Ptyche smoke and benchmark launcher. |

## Task 1: Pin and verify the MCore TE adapter

**Files:**
- Modify: 3rdparty/Megatron-LM-workspace/Megatron-LM
- Test: 3rdparty/Megatron-LM-workspace/Megatron-LM/tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py

**Consumes:** NeMo-RL will call TECudaGraphHelper with sample_packed_seq_params.

**Produces:** The MCore helper flattens dynamic packed tensor fields into TE graph kwargs and rebuilds PSP on replay.

- [ ] **Step 1: Write a failing API assertion**

~~~python
import inspect
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

assert "sample_packed_seq_params" in inspect.signature(TECudaGraphHelper).parameters
~~~

Run:

~~~bash
uv run --locked --extra mcore python -c 'import inspect; from megatron.core.transformer.cuda_graphs import TECudaGraphHelper; assert "sample_packed_seq_params" in inspect.signature(TECudaGraphHelper).parameters'
~~~

Expected: FAIL on parent MCore because the helper lacks the keyword.

- [ ] **Step 2: Pin the tested MCore commit**

~~~bash
git -C 3rdparty/Megatron-LM-workspace/Megatron-LM fetch \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-vs-pr5783-ptyche-20260716/3rdparty/Megatron-LM-workspace/Megatron-LM \
  bed605f292f926090f5f43ba5e30fb024c2306dc
git -C 3rdparty/Megatron-LM-workspace/Megatron-LM checkout --detach \
  bed605f292f926090f5f43ba5e30fb024c2306dc
~~~

- [ ] **Step 3: Verify MCore**

~~~bash
uv run --locked --extra mcore python -c 'import inspect; from megatron.core.transformer.cuda_graphs import TECudaGraphHelper; assert "sample_packed_seq_params" in inspect.signature(TECudaGraphHelper).parameters'
uv run --locked --extra mcore pytest \
  3rdparty/Megatron-LM-workspace/Megatron-LM/tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py -q
~~~

Expected: API assertion and focused MCore test pass in Ptyche's container.

- [ ] **Step 4: Commit**

~~~bash
git add 3rdparty/Megatron-LM-workspace/Megatron-LM
git commit -s -m "build: pin MCore TE packed graph adapter"
~~~

## Task 2: Staticize graph PSP while preserving loss boundaries

**Files:**
- Modify: nemo_rl/models/megatron/data.py:39-50, 116-150, 842-1060, 1295-1350
- Modify: nemo_rl/models/megatron/train.py:200-280, 395-500
- Test: tests/unit/models/megatron/test_megatron_data.py
- Test: tests/unit/models/megatron/test_train.py

**Consumes:** cuda_graph_pr5672_thd, cuda_graph_max_packed_seqs, and Task 1.

**Produces:** A graph-facing endpoint-padded PSP and independent real loss boundaries.

- [ ] **Step 1: Write failing data tests**

Append these tests to tests/unit/models/megatron/test_megatron_data.py:

~~~python
def test_pr5672_graph_psp_padding_keeps_loss_boundaries_unpadded():
    from nemo_rl.models.megatron.data import _pack_sequences_for_megatron

    input_ids = torch.tensor([[11, 12, 0, 0], [21, 22, 23, 0]], device="cuda")
    lengths = torch.tensor([2, 3], device="cuda")
    _, _, psp, loss_cu, loss_cu_padded = _pack_sequences_for_megatron(
        input_ids=input_ids,
        seq_lengths=lengths,
        pad_packed_seq_to=8,
        cu_seqlens_pad_to_entries=5,
    )

    assert loss_cu.tolist() == [0, 2, 5]
    assert loss_cu_padded.tolist() == [0, 2, 5]
    assert psp.cu_seqlens_q.tolist() == [0, 2, 5, 5, 5]
    assert psp.cu_seqlens_q_padded.tolist() == [0, 2, 5, 5, 5]


def test_pr5672_graph_psp_rejects_too_many_sequences():
    from nemo_rl.models.megatron.data import _pack_sequences_for_megatron

    with pytest.raises(AssertionError, match="increase policy.megatron_cfg.cuda_graph_max_packed_seqs"):
        _pack_sequences_for_megatron(
            input_ids=torch.tensor([[1], [2], [3]], device="cuda"),
            seq_lengths=torch.tensor([1, 1, 1], device="cuda"),
            pad_packed_seq_to=3,
            cu_seqlens_pad_to_entries=3,
        )
~~~

Append a LossPostProcessor test to tests/unit/models/megatron/test_train.py that calls it with loss_cu_seqlens=torch.tensor([0, 2, 5]) and verifies SequencePackingLossWrapper uses that tensor, not a graph PSP ending in repeated 5.

- [ ] **Step 2: Confirm failure**

~~~bash
uv run --locked --extra mcore pytest \
  tests/unit/models/megatron/test_megatron_data.py::test_pr5672_graph_psp_padding_keeps_loss_boundaries_unpadded \
  tests/unit/models/megatron/test_megatron_data.py::test_pr5672_graph_psp_rejects_too_many_sequences -q
~~~

Expected: FAIL because the parent does not retain unpadded boundaries or activate fixed PSPs for TE PR5672.

- [ ] **Step 3: Implement the data and loss contract**

In data.py retain cu_seqlens in ProcessedInputs and ProcessedMicrobatch, and activate static PSP entry count for either local PR5783 or TE PR5672:

~~~python
uses_static_packed_seq_graph_inputs = (
    megatron_cfg.get("cuda_graph_impl") == "local"
    and megatron_cfg.get("cuda_graph_pr5783_thd", False)
) or (
    megatron_cfg.get("cuda_graph_impl") == "transformer_engine"
    and megatron_cfg.get("cuda_graph_pr5672_thd", False)
)
if pack_sequences and uses_static_packed_seq_graph_inputs:
    cu_seqlens_pad_to_entries = (
        megatron_cfg.get("cuda_graph_max_packed_seqs") or 64
    ) + 1
~~~

Carry cu_seqlens through process_microbatch and the iterator. Preserve endpoint-repeat padding only in the PackedSeqParams copies.

In train.py add optional loss_cu_seqlens and loss_cu_seqlens_padded arguments to LossPostProcessor.__call__ and call it with:

~~~python
post_processing_fn_wrapped = post_processing_fn(
    data_dict=data_dict,
    packed_seq_params=packed_seq_params,
    loss_cu_seqlens=cu_seqlens,
    loss_cu_seqlens_padded=cu_seqlens_padded,
    global_valid_seqs=global_valid_seqs,
    global_valid_toks=global_valid_toks,
)
~~~

Use explicit values in SequencePackingLossWrapper, falling back to PSP only when no explicit loss boundaries are supplied.

- [ ] **Step 4: Run focused tests**

~~~bash
uv run --locked --extra mcore pytest \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_train.py -q
~~~

Expected: PASS in Ptyche's container.

- [ ] **Step 5: Commit**

~~~bash
git add nemo_rl/models/megatron/data.py nemo_rl/models/megatron/train.py \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_train.py
git commit -s -m "feat: preserve loss boundaries for TE packed graphs"
~~~

## Task 3: Supply capture PSP and invalidate stale graphs

**Files:**
- Modify: nemo_rl/models/policy/workers/megatron_policy_worker.py:20-55, 490-840, 980-1060, 2560-2595
- Test: tests/unit/models/policy/test_megatron_worker.py

**Consumes:** Task 1 MCore API and Task 2 graph-facing PSP.

**Produces:** TE capture gets a fixed-shape PSP and refit cannot replay graphs with old GPU parameter addresses.

- [ ] **Step 1: Write failing worker tests**

Append to tests/unit/models/policy/test_megatron_worker.py:

~~~python
def test_pr5672_sample_packed_seq_params_keeps_shape_and_metadata():
    from megatron.core.packed_seq_params import PackedSeqParams
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    actual = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32),
        max_seqlen_q=8,
        max_seqlen_kv=8,
    )
    sample = MegatronPolicyWorkerImpl._make_cuda_graph_sample_packed_seq_params(actual, 16)

    assert sample.qkv_format == "thd"
    assert sample.max_seqlen_q == 16
    assert sample.cu_seqlens_q.tolist() == [0, 16, 16, 16, 16]


def test_pr5672_parameter_move_invalidates_all_graph_state():
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    worker = object.__new__(MegatronPolicyWorkerImpl)
    module = SimpleNamespace(cuda_graphs=[object()])
    worker.model = SimpleNamespace(modules=lambda: [module])
    worker._cuda_graph_helper = object()
    worker._cuda_graph_bucket_helpers = {4096: object()}
    worker._cuda_graph_bucket_graphs = {4096: {module: module.cuda_graphs}}
    worker._cuda_graph_active_bucket = 4096
    worker._cuda_graph_saved_graphs = {module: module.cuda_graphs}
    worker._cuda_graph_captured_seq_length = 4096
    worker._cuda_graph_train_steps = 99
    worker.megatron_cfg = SimpleNamespace(model=SimpleNamespace(cuda_graph_warmup_steps=3))

    worker._invalidate_cuda_graphs_after_parameter_move()

    assert module.cuda_graphs == []
    assert worker._cuda_graph_helper is None
    assert worker._cuda_graph_bucket_helpers == {}
    assert worker._cuda_graph_bucket_graphs == {}
    assert worker._cuda_graph_active_bucket is None
    assert worker._cuda_graph_saved_graphs == {}
    assert worker._cuda_graph_captured_seq_length is None
    assert worker._cuda_graph_train_steps == 3
~~~

- [ ] **Step 2: Confirm failure**

~~~bash
uv run --locked --extra mcore pytest \
  tests/unit/models/policy/test_megatron_worker.py::test_pr5672_sample_packed_seq_params_keeps_shape_and_metadata \
  tests/unit/models/policy/test_megatron_worker.py::test_pr5672_parameter_move_invalidates_all_graph_state -q
~~~

Expected: FAIL because neither method exists.

- [ ] **Step 3: Implement capture and invalidation**

Import tee and PackedSeqParams. Add:

~~~python
@staticmethod
def _make_cuda_graph_sample_packed_seq_params(
    packed_seq_params: PackedSeqParams, seq_length: int
) -> PackedSeqParams:
    cu_seqlens = packed_seq_params.cu_seqlens_q
    if cu_seqlens is None:
        raise ValueError("PR #5672 CUDA graphs require PackedSeqParams.cu_seqlens_q.")
    sample_cu_seqlens = torch.full_like(cu_seqlens, seq_length)
    sample_cu_seqlens[0] = 0
    return PackedSeqParams(
        qkv_format=packed_seq_params.qkv_format,
        cu_seqlens_q=sample_cu_seqlens,
        cu_seqlens_kv=sample_cu_seqlens,
        cu_seqlens_q_padded=sample_cu_seqlens,
        cu_seqlens_kv_padded=sample_cu_seqlens,
        max_seqlen_q=seq_length,
        max_seqlen_kv=seq_length,
        local_cp_size=packed_seq_params.local_cp_size,
        cp_group=packed_seq_params.cp_group,
    )
~~~

Thread the optional PSP through _maybe_capture_cuda_graphs and _capture_all_buckets, passing the sample as sample_packed_seq_params to every TECudaGraphHelper.

On the first non-evaluation global batch, peek without consuming:

~~~python
data_iterator, sample_iterator = tee(data_iterator)
cuda_graph_sample_packed_seq_params = next(sample_iterator).packed_seq_params
if cuda_graph_sample_packed_seq_params is None:
    raise ValueError(
        "PR #5672 CUDA graphs require sequence packing to produce PackedSeqParams."
    )
~~~

Add _invalidate_cuda_graphs_after_parameter_move that clears module cuda_graphs, the single helper, all bucket helpers/graphs, active bucket, saved graphs, and captured sequence length. Set _cuda_graph_train_steps to cuda_graph_warmup_steps. Call it immediately before:

~~~python
self.model = self.move_model(self.model, "cpu")
~~~

- [ ] **Step 4: Run worker validation**

~~~bash
uv run --locked --extra mcore pytest tests/unit/models/policy/test_megatron_worker.py -q
python -m py_compile nemo_rl/models/policy/workers/megatron_policy_worker.py
~~~

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/policy/test_megatron_worker.py
git commit -s -m "feat: capture TE packed graphs with fresh parameters"
~~~

## Task 4: Encode the production recipe contract

**Files:**
- Create: examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nocg-adapter.yaml
- Modify: examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn.yaml
- Modify: examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router-w3.yaml
- Modify: experiments/cuda_graph/launch_qwen30_moe_cg_comparison_ptyche.sh
- Test: tests/unit/models/megatron/test_megatron_setup.py

**Consumes:** Tasks 1-3.

**Produces:** A comparable packed no-CG adapter baseline, a production-safe FP64 attention recipe, and separately labelled FP32 router diagnostics.

- [ ] **Step 1: Write the failing recipe test**

~~~python
def test_pr5672_qwen30_packed_attention_recipe_uses_te_static_thd():
    register_omegaconf_resolvers()
    recipe_dir = Path(__file__).parents[4] / "examples/configs/recipes/llm/performance"
    no_cg = load_config(recipe_dir / "grpo-qwen3-30ba3b-4n4g-nocg-adapter.yaml")
    attn = load_config(recipe_dir / "grpo-qwen3-30ba3b-4n4g-cg-attn.yaml")
    OmegaConf.resolve(no_cg)
    OmegaConf.resolve(attn)

    assert no_cg.checkpointing.enabled is False
    assert no_cg.policy.megatron_cfg.cuda_graph_impl == "none"
    assert no_cg.policy.megatron_cfg.moe_router_dtype == "fp64"
    assert attn.checkpointing.enabled is False
    assert attn.policy.megatron_cfg.moe_router_dtype == "fp64"
    cfg = attn.policy.megatron_cfg
    assert cfg.cuda_graph_impl == "transformer_engine"
    assert cfg.cuda_graph_scope == "attn"
    assert cfg.cuda_graph_pr5672_thd is True
    assert cfg.cuda_graph_packed_seq is True
    assert cfg.cuda_graph_max_packed_seqs == 64
    assert cfg.cuda_graph_warmup_steps == 3
    assert list(cfg.cuda_graph_buckets) == [4096]

    expected_packing = {
        "enabled": True,
        "train_mb_tokens": 4096,
        "logprob_mb_tokens": 4096,
        "algorithm": "modified_first_fit_decreasing",
        "sequence_length_round": 64,
    }
    for key, value in expected_packing.items():
        assert no_cg.policy.sequence_packing[key] == value
        assert attn.policy.sequence_packing[key] == value
~~~

- [ ] **Step 2: Confirm failure**

~~~bash
uv run --locked --extra mcore pytest \
  tests/unit/models/megatron/test_megatron_setup.py::test_pr5672_qwen30_packed_attention_recipe_uses_te_static_thd -q
~~~

Expected: FAIL because the dedicated comparable adapter no-CG baseline does not exist.

- [ ] **Step 3: Update recipes and launcher**

Set the production attention recipe to:

~~~yaml
cuda_graph_impl: transformer_engine
cuda_graph_scope: attn
cuda_graph_warmup_steps: 3
cuda_graph_packed_seq: true
cuda_graph_pr5672_thd: true
cuda_graph_max_packed_seqs: 64
cuda_graph_buckets:
- 4096
~~~

For the MoE diagnostic recipe, add cuda_graph_pr5672_thd: true and retain FP32 router override only in launcher invocation. Add a comment that this condition is diagnostic and cannot establish production accuracy.

Create the dedicated `adapter-nocg` recipe with the same enabled packing, 4096-token train/logprob budgets, `modified_first_fit_decreasing`, and 64-token rounding as `adapter-attn`. Add ADAPTER_WORKTREE and adapter-* handling to the launcher. Map only adapter-nocg to the dedicated no-CG recipe and adapter-attn to the production attention recipe. Preserve --test-only as the default and require SUBMIT=1 for submission.

- [ ] **Step 4: Run recipe and launcher checks**

~~~bash
uv run --locked --extra mcore pytest \
  tests/unit/models/megatron/test_megatron_setup.py::test_pr5672_qwen30_packed_attention_recipe_uses_te_static_thd -q
ADAPTER_WORKTREE=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-adapter-ptyche-20260719 \
  CONDITION=adapter-nocg STEPS=20 \
  ./experiments/cuda_graph/launch_qwen30_moe_cg_comparison_ptyche.sh
ADAPTER_WORKTREE=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-adapter-ptyche-20260719 \
  CONDITION=adapter-attn STEPS=20 \
  ./experiments/cuda_graph/launch_qwen30_moe_cg_comparison_ptyche.sh
~~~

Expected: recipe test passes and Slurm accepts each dry-run submission.

- [ ] **Step 5: Commit**

~~~bash
git add examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nocg-adapter.yaml \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn.yaml \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router-w3.yaml \
  experiments/cuda_graph/launch_qwen30_moe_cg_comparison_ptyche.sh \
  tests/unit/models/megatron/test_megatron_setup.py
git commit -s -m "test: add reproducible TE packed graph recipes"
~~~

## Task 5: Validate safety, performance, and accuracy on Ptyche

**Files:**
- Create: experiments/cuda_graph/report/te_packed_adapter_20260719.md
- Read: experiments/cuda_graph/logs/<job>-logs/driver_command.sh
- Read: experiments/cuda_graph/logs/<job>-logs/ray-driver-*

**Consumes:** Pushed commits from Tasks 1-4.

**Produces:** A reproducible decision report separating FP64 production results from FP32 router diagnostics.

- [ ] **Step 1: Push and stage the exact worktree**

~~~bash
git push -u origin experiment/te-packed-cg-adapter-20260719
ssh sna-mfa@login-ptyche '
  cd /lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-adapter-ptyche-20260719 &&
  git pull --ff-only &&
  git submodule update --init --recursive &&
  git status --short
'
~~~

Expected: clean remote worktree with recorded parent and MCore SHAs.

- [ ] **Step 2: Run an 8-step blocking smoke only for diagnosis**

~~~bash
CUDA_LAUNCH_BLOCKING=1 CONDITION=adapter-attn STEPS=8 \
  RUN_TAG=adapter-attn-blocking-smoke8 SUBMIT=1 \
  ./experiments/cuda_graph/launch_qwen30_moe_cg_comparison_ptyche.sh
~~~

Expected: capture after step 3, refit/offload, reload, and replay complete without illegal memory access. Do not use this job for performance.

- [ ] **Step 3: Run the FP64 production performance pair**

Submit `adapter-nocg` (the dedicated packed no-CG baseline) and `adapter-attn` for 20 steps with identical seed, data, checkpoint, 4n4g topology, and validation cadence. Enforce equal packing/token budgets in both recipes: enabled packing, train_mb_tokens=4096, logprob_mb_tokens=4096, algorithm=modified_first_fit_decreasing, and sequence_length_round=64. Run both conditions from the adapter worktree and its shared converted Megatron checkpoint namespace. Disable checkpoints.

- [ ] **Step 4: Run separately labelled FP32 router diagnostics**

After the FP64 pair passes, run adapter-moe-router and adapter-attn-moe-router with policy.megatron_cfg.moe_router_dtype=fp32. Mark both rows diagnostic-fp32; exclude them from all accuracy claims.

- [ ] **Step 5: Run FP64 accuracy sign-off and write the report**

Run `adapter-nocg` FP64 and `adapter-attn` FP64 for three identical seeds, 40 steps per seed, and 1,024 fixed validation samples. Report accuracy, reward, GenKL, policy loss, ratio/clip diagnostics, NaN count, total step time, E2E TPS/GPU, policy TPS/GPU, logprob TPS/GPU, and generation TPS/GPU.

Use performance steps 4-19 excluding validation steps. Record RL commit, MCore SHA, container, model, topology, packing, Nmax, warmup, scope, router dtype, sample count, and included steps.

- [ ] **Step 6: Extend the FP64 performance pair to Qwen3-235B-A22B**

After the Qwen3-30B-A3B smoke passes, run the same no-CG versus PR5672 `attn` comparison on the native Ptyche `grpo-qwen3-235b-16n4g` topology (16 nodes x 4 GPUs). Keep the base recipe's FP64 router, eager router/dispatch/experts, sequence packing, token budget, seed, and validation cadence identical; disable checkpointing and external telemetry. Do not use the MXFP8-rollout, async, 16n8g, or 32n4g overlays for this first comparison.

Before submitting, verify the model snapshot, OpenMath cache/access, converted Megatron checkpoint cache, and the direct MCore `TECudaGraphHelper` API on Ptyche. Use `cuda_graph_buckets=[8192]`, three warmup steps, and `cuda_graph_max_packed_seqs=512` for the first five-step smoke and the 20-step pair. Record the high-water actual packed-sequence count from train, logprob, and validation; only lower Nmax after that evidence. Report the 235B pair as performance-only unless it also completes the separately specified three-seed accuracy protocol.

- [ ] **Step 7: Commit the report**

~~~bash
git add experiments/cuda_graph/report/te_packed_adapter_20260719.md
git commit -s -m "docs: report TE packed graph validation"
git push
~~~

## Plan Self-Review

- Spec coverage: Tasks 1-3 implement MCore, data/loss, and lifecycle contracts; Task 4 fixes recipe reproducibility; Task 5 validates safety, performance, and accuracy.
- Scope: FP64 router graph, full iteration graph, and generation/logprob optimization are deliberately excluded.
- Type consistency: PackedSeqParams is optional only at capture; loss receives optional original boundary tensors; worker-state names match the current class.
- Platform: Linux/CUDA commands are explicitly assigned to Ptyche.
