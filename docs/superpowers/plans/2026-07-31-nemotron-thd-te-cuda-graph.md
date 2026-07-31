# Nemotron Packed THD Transformer Engine CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide correctness-preserving Transformer Engine partial CUDA Graph
training for every model-compatible `attn`, `mlp`, `mamba`, `moe`,
`moe_router`, and `moe_preprocess` combination used by Nemotron 3 Nano,
Super, and Ultra packed-THD model paths.

**Architecture:** Start from the latest NeMo-RL, Megatron-Bridge, and
Megatron-LM main branches. Port PR 5672's packed attention adapter, preserve a
single physical token capacity through every graph/eager boundary, make MoE
dispatchers own their replay state, and use bounded schedule-specific graph
banks. Carry a structural padding mask from NeMo-RL through HybridModel,
including sequence-parallel and MTP paths, so padding never changes routing,
capacity, gradients, or reported accuracy.

**Tech Stack:** Python 3.12, PyTorch distributed, Transformer Engine,
Megatron-LM/MCore, Megatron-Bridge, NeMo-RL, Ray, SLURM, pytest, uv, W&B, and
static HTML reporting.

## Global Constraints

- Work only in `/Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-20260731`.
- NeMo-RL starts at `d152853a91fc5e1e1f66fc06e3a7e5ff5fb6ef7e`.
- Megatron-Bridge starts at `3bc95ef5fa4a76b7155fb090bdcfa1bf643bd56f`.
- Megatron-LM starts at `b19b1f47cf7e289607f3be480c5f06c6ada25b16`.
- Do not merge Megatron-LM `dev` wholesale.
- Port PR 5672 commits `4cb58d5d6`, `1ba1418b8`, and `6ff66f0a0`.
- Apply PR 5542 commit `d1384c2d9`; manually port the reviewed PR 5724 and
  PR 5541 invariants because their dev merge commits conflict with main.
- Manually port PR 4359 commit `7f9175207`'s main-compatible
  `padding_mask`/`PackedSeqParams` MLP plumbing; do not cherry-pick its branch.
- Do not port PR 5668: latest main already implements its HybridEP
  uneven-input invariant through PR 5008. Add a regression test instead.
- Do not port PR 5401: its z-loss-safe behavior is already in the selected
  baseline.
- Transformer Engine must be
  `a6c70f4dc84ae9d4a7d0a057c990ac1dc925d480` or newer. Use
  a nightly-container native artifact built from
  `869f99c47d5773e3dbf4a85d4cc8679c4e050089` or a later verified commit.
- The selected MCore main has no Transformer Engine submodule. Do not recreate
  it or change the dependency to trigger a per-job native build; verify the
  immutable container digest, TE version, and TE build revision at preflight.
- `padding_mask` is `torch.bool`, shaped `[batch, local_sequence]`, with
  `True` meaning padding. It is not NeMo-RL's loss `token_mask`.
- Hidden states remain at fixed physical capacity through graph replay,
  eager dispatch, expert compute, combine, residual, and BDA.
- Do not repair a narrowed output with zero padding, truncation, or prefix
  reconstruction.
- A replay signature mismatch raises before any rank launches a graph or
  collective. Rank-local eager fallback is forbidden.
- Full `moe` capture is enabled only with statically padded expert capacity
  and verified zero valid-token drop.
- This implementation uses append-dummy THD tail padding. A non-dummy
  `extend_last` request fails validation rather than importing the dev-only
  scheduler/config surface.
- Warmup is exactly three globally successful optimizer updates.
- Graph-bank capacity is two schedule keys. Capture, switch, eviction, and
  reset occur only at a drained optimizer-step boundary.
- NeMo-RL checkpoints are disabled in experiment jobs.
- Use W&B project `sna-cg-study`; never commit API keys or tokens.
- Do not build native Transformer Engine separately in each job.
- Commit and push MCore, then Bridge, then NeMo-RL. Use signed-off commits;
  MCore commits also use repository-required GPG signing.

---

## File and Responsibility Map

### Megatron-LM

- `megatron/core/packed_seq_params.py`: split/rebuild graph-safe packed
  metadata; retain real and physical cumulative lengths.
- `megatron/core/model_parallel_config.py`: expose fixed THD sequence-count
  and tail-padding settings.
- `megatron/core/transformer/transformer_config.py`: validate graph capacity,
  scope, dispatcher, and whole-MoE requirements.
- `megatron/core/transformer/cuda_graphs.py`: packed sample kwargs, graphable
  leaf discovery, MTP HybridStack traversal, and graph-bank construction.
- `megatron/core/transformer/module.py`: replay guard integration.
- `megatron/core/transformer/transformer_layer.py`: attention adapter,
  structural-mask static input, partial-MoE replay continuation, and
  dispatcher-state capture/restore.
- `megatron/core/ssm/mamba_layer.py`: packed Mamba graph adapter using
  capacity-sized `seq_idx`.
- `megatron/core/transformer/moe/cuda_graph_replay.py`: typed dispatcher-owned
  non-Tensor replay state.
- `megatron/core/transformer/moe/token_dispatcher.py`: allgather, alltoall,
  Flex/HybridEP, DeepEP, and NCCL-EP replay contracts.
- `megatron/core/transformer/moe/router.py`: exclude structural padding from
  routing loss, bias counts, and supported dispatch capacity.
- `megatron/core/transformer/te_cuda_graph_bank.py`: own, fingerprint,
  activate, and reset schedule-specific TE graphs.
- `megatron/core/models/hybrid/hybrid_model.py`: scatter the structural mask
  with sequence parallelism and pass it into MTP.
- `megatron/core/models/common/utils.py`: suppress delayed shared-expert
  weight gradients only when captured.

### Megatron-Bridge

- `3rdparty/Megatron-LM`: pin the verified MCore branch.
- `tests/unit_tests/recipes/nemotronh/test_nemotron_3_nano.py`: validate Nano
  topology and graph override compatibility.
- `tests/unit_tests/recipes/nemotronh/test_nemotron_3_super.py`: validate
  latent-MoE/MTP topology and graph override compatibility.
- `tests/unit_tests/recipes/nemotronh/test_nemotron_3_ultra.py`: validate
  Flex/HybridEP topology while leaving production recipe defaults unchanged.

### NeMo-RL

- `nemo_rl/models/megatron/data.py`: construct and CP-shard the structural
  padding mask.
- `nemo_rl/models/megatron/train.py`: pass the mask to HybridModel.
- `nemo_rl/models/megatron/setup.py`: validate capacity, scope, runtime, and
  dispatcher prerequisites.
- `nemo_rl/models/megatron/cuda_graph_lifecycle.py`: two-entry, three-warmup
  graph-bank lifecycle.
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`: activate graph
  banks at step boundaries and return telemetry.
- `nemo_rl/models/policy/lm_policy.py`: aggregate graph and packing metrics.
- `nemo_rl/algorithms/grpo.py` and `nemo_rl/algorithms/grpo_sync.py`: emit
  graph, performance, and correctness metrics.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/`: persistent launch,
  collection, comparison, and HTML-report surface.

---

### Task 1: Establish the Reproducible Three-Repository Baseline

**Files:**
- Inspect: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/pyproject.toml`

**Interfaces:**
- Consumes: exact source commits from Global Constraints.
- Produces: MCore branch `sj/thd-cg-hybrid-nemotron-20260731` and Bridge branch
  `sna/thd-cg-hybrid-nemotron-20260731`, both based on official main, plus an
  external-runtime contract deferred to the persistent preflight harness.

- [ ] **Step 1: Verify the isolated worktree and remotes**

```bash
git status --short
git submodule status --recursive
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge remote -v
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM remote -v
```

Expected: only the committed design and plan are present; no unrelated dirty
files exist.

- [ ] **Step 2: Create the Bridge and MCore branches from official main**

```bash
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge switch \
  -c sna/thd-cg-hybrid-nemotron-20260731 upstream/main
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM switch \
  -c sj/thd-cg-hybrid-nemotron-20260731 upstream/main
```

- [ ] **Step 3: Verify the external Transformer Engine runtime contract**

```bash
test ! -e \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/third_party/TransformerEngine
rg -n "transformer-engine" \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/pyproject.toml
```

Expected: the removed submodule is absent and MCore declares Transformer
Engine as an external package. Task 11 adds a persistent
`validate_te_runtime.py` that checks the immutable nightly container before
any GPU job.

- [ ] **Step 4: Record source provenance**

```bash
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  rev-parse HEAD
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  diff --check
```

Expected: Bridge is `3bc95ef5`, MCore is `b19b1f47`, and `diff --check`
passes. No source commit is expected in this task because the runtime is
container-owned; the first MCore implementation commit is Task 2.

### Task 2: Port PR 5672 and Fixed-Capacity THD Metadata

**Files:**
- Modify: `megatron/core/packed_seq_params.py`
- Modify: `megatron/core/model_parallel_config.py`
- Modify: `megatron/core/transformer/transformer_config.py`
- Modify: `megatron/core/transformer/cuda_graphs.py`
- Modify: `megatron/core/transformer/transformer_layer.py`
- Test: `tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`
- Test: `tests/unit_tests/test_sequence_packing.py`

**Interfaces:**
- Consumes: `PackedSeqParams` and TE keyword-argument capture.
- Produces:
  `split_packed_seq_params_for_cuda_graph(PackedSeqParams | None)`,
  `build_packed_seq_params_from_cuda_graph_kwargs(...)`, and fixed-capacity
  cumulative metadata whose entry count is
  `thd_max_packed_sequences + 1`.

- [ ] **Step 1: Add red tests for graph metadata and physical capacity**

Add tests with these assertions:

```python
def test_packed_graph_static_metadata_keeps_pad_between_seqs() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        tokens_per_sample=8,
        pad_between_seqs=True,
    )
    tensor_kwargs, static = split_packed_seq_params_for_cuda_graph(params)
    rebuilt = build_packed_seq_params_from_cuda_graph_kwargs(dict(tensor_kwargs), static)
    assert rebuilt.pad_between_seqs is True
    assert rebuilt.tokens_per_sample == 8


def test_packed_graph_rejects_changed_static_pad_between_seqs() -> None:
    layer = _TransformerLayerCudaGraphStub()
    captured = PackedSeqParams(qkv_format="thd", pad_between_seqs=True, tokens_per_sample=8)
    tensor_kwargs, static = split_packed_seq_params_for_cuda_graph(captured)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static, tensor_kwargs)
    replay = PackedSeqParams(qkv_format="thd", pad_between_seqs=False, tokens_per_sample=8)
    with pytest.raises(AssertionError, match="pad_between_seqs"):
        layer._flatten_te_cuda_graph_packed_seq_params(
            {"packed_seq_params": replay}
        )


def test_thd_graph_metadata_has_fixed_sequence_capacity() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7, 12], dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor([0, 3, 7, 12], dtype=torch.int32, device="cuda"),
        cu_seqlens_q_padded=torch.tensor(
            [0, 4, 8, 12], dtype=torch.int32, device="cuda"
        ),
        cu_seqlens_kv_padded=torch.tensor(
            [0, 4, 8, 12], dtype=torch.int32, device="cuda"
        ),
        max_seqlen_q=5,
        max_seqlen_kv=5,
        pad_between_seqs=True,
    )
    padded = pad_sequence_for_thd(
        tokens=torch.ones(12, 1, dtype=torch.long, device="cuda"),
        labels=None,
        loss_mask=None,
        position_ids=None,
        packed_seq_params=params,
        target_len=16,
        max_num_seqs=7,
    )
    graph_params = padded[4]
    assert graph_params.cu_seqlens_q.numel() == 8
    assert graph_params.cu_seqlens_q_padded.numel() == 8
    assert graph_params.cu_seqlens_q[-1] <= graph_params.cu_seqlens_q_padded[-1]
```

- [ ] **Step 2: Run the red tests**

```bash
uv run pytest -q \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/test_sequence_packing.py
```

Expected: failures show missing PR 5672 helpers and missing fixed-capacity THD
behavior.

- [ ] **Step 3: Port PR 5672 without importing its branch history**

Apply commits in order:

```bash
git cherry-pick 4cb58d5d6
git cherry-pick 1ba1418b8
git cherry-pick 6ff66f0a0
```

Then add `"pad_between_seqs"` to
`PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS`. Add `"tokens_per_sample"` as
well, because whole-layer MoE uses it to restore the batch dimension. Static
replay validation must compare both values with capture instead of rebuilding
them as `None`.

- [ ] **Step 4: Manually port the PR 5724/5541 invariants**

Implement the following exact contract:

```python
def get_thd_padding_kwargs(
    config: ModelParallelConfig,
) -> tuple[int | None, int | None, int | None]:
    alignment = config.pad_packed_seq_alignment
    target_len = config.pad_packed_seq_to
    max_sequences = config.thd_max_packed_sequences
    if config.cuda_graph_impl == "transformer_engine":
        assert max_sequences is not None, (
            "THD Transformer Engine CUDA graphs require "
            "thd_max_packed_sequences."
        )
    return alignment, target_len, max_sequences
```

Preserve separate `cu_seqlens_q`/`cu_seqlens_kv` logical endpoints and
`cu_seqlens_q_padded`/`cu_seqlens_kv_padded` physical endpoints. For graph
capture, set `pad_between_seqs=True` and pad all four tensors to the configured
entry capacity. Append the capacity tail as one dummy sequence and require a
zigzag-CP dummy tail to be divisible by `2 * context_parallel_size`.
`thd_max_packed_sequences` counts real plus dummy sequences, so MCore Tensor
entry capacity is `thd_max_packed_sequences + 1`; NeMo-RL
`max_sequences_per_bin` counts real sequences, so its Tensor entry capacity is
`max_sequences_per_bin + 2`.

Do not port PR 5541's `HyperConnectionHybridLayer` hunk: that class is absent
from the selected main baseline. `TransformerLayer` remains the reconstruction
boundary for Nano, Super, and Ultra. These workloads retain main's implicit
zigzag CP behavior; do not import the dev-only `cp_partition_mode` scheduler
surface as part of this capacity port.

- [ ] **Step 5: Run focused tests and format**

```bash
uv run pytest -q \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/test_sequence_packing.py
uv run isort megatron tests
uv run black megatron tests
git diff --check
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit the semantic port**

```bash
git add megatron/core/packed_seq_params.py \
  megatron/core/model_parallel_config.py \
  megatron/core/transformer/transformer_config.py \
  megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/transformer_layer.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/test_sequence_packing.py
git commit -s -S -m "feat: preserve fixed-capacity THD graph metadata"
```

### Task 3: Add the Packed Mamba Graph Adapter

**Files:**
- Modify: `megatron/core/packed_seq_params.py`
- Modify: `megatron/core/ssm/mamba_layer.py`
- Modify: `megatron/core/transformer/cuda_graphs.py`
- Test: `tests/unit_tests/ssm/test_mamba_layer.py`
- Test: `tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`
- Test: `tests/unit_tests/transformer/test_cuda_graphs.py`

**Interfaces:**
- Consumes: capacity-sized `PackedSeqParams.seq_idx`.
- Produces: Mamba-only graph kwargs that do not leak into the attention
  adapter.

- [ ] **Step 1: Add red tests for packed Mamba signatures**

```python
def test_mamba_graph_uses_capacity_sized_seq_idx() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        total_tokens=16,
    )
    tensor_kwargs, static = split_mamba_packed_seq_params_for_cuda_graph(params)
    assert tensor_kwargs["_mamba_packed_seq_params_seq_idx"].shape == (1, 16)
    assert "cu_seqlens_kv" not in tensor_kwargs
    assert static["qkv_format"] == "thd"
    assert static["total_tokens"] == 16


def test_mamba_replay_rejects_seq_idx_shape_change() -> None:
    class _MambaLayerCudaGraphStub:
        _set_te_cuda_graph_mamba_packed_seq_params_static_metadata = (
            MambaLayer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata
        )
        _flatten_te_cuda_graph_mamba_packed_seq_params = (
            MambaLayer._flatten_te_cuda_graph_mamba_packed_seq_params
        )

    layer = _MambaLayerCudaGraphStub()
    captured = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        total_tokens=16,
    )
    tensor_kwargs, static = split_mamba_packed_seq_params_for_cuda_graph(captured)
    layer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata(
        static, tensor_kwargs
    )
    replay = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        total_tokens=12,
    )
    with pytest.raises(AssertionError, match="total_tokens"):
        layer._flatten_te_cuda_graph_mamba_packed_seq_params(
            {"packed_seq_params": replay}
        )
```

- [ ] **Step 2: Verify the tests fail**

```bash
uv run pytest -q \
  tests/unit_tests/ssm/test_mamba_layer.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/transformer/test_cuda_graphs.py -k "mamba and packed"
```

Expected: missing Mamba split/flatten helpers.

- [ ] **Step 3: Implement the Mamba-only adapter**

Use these distinct fields:

```python
MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS = ("seq_idx",)
MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS = (
    "qkv_format",
    "local_cp_size",
    "cp_group",
    "total_tokens",
)
```

`MambaLayer.get_layer_static_inputs()` adds the captured capacity-sized
`seq_idx`; replay validates field presence, shape, dtype, device, layout, and
stride. `total_tokens` is replay-validated static metadata because a changed
value requires recapture, but it is never flattened into the attention
adapter. Replay reuses the current precomputed `seq_idx` Tensor storage and
never reconstructs it from `total_tokens` inside the graph callable. Preserve
the existing local-Mamba-graph and inference-context behavior.

- [ ] **Step 4: Run and commit**

```bash
uv run pytest -q \
  tests/unit_tests/ssm/test_mamba_layer.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/transformer/test_cuda_graphs.py -k "mamba or packed"
git diff --check
git add megatron/core/packed_seq_params.py \
  megatron/core/ssm/mamba_layer.py \
  megatron/core/transformer/cuda_graphs.py \
  tests/unit_tests/ssm/test_mamba_layer.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git commit -s -S -m "feat: support packed Mamba TE graph replay"
```

### Task 4: Propagate Structural Padding Through HybridModel and Routing

**Files:**
- Modify: `megatron/core/models/hybrid/hybrid_model.py`
- Modify: `megatron/core/transformer/transformer_layer.py`
- Modify: `megatron/core/transformer/moe/router.py`
- Test: `tests/unit_tests/models/test_hybrid_moe_model.py`
- Test: `tests/unit_tests/transformer/moe/test_routers.py`
- Test: `tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`

**Interfaces:**
- Consumes: `padding_mask: Tensor[bool]` with `True` for padding.
- Produces: a layer-local mask whose sequence dimension matches SP/CP-local
  hidden states, including MTP heads.

- [ ] **Step 1: Add red mask-order and routing tests**

```python
def test_hybrid_model_scatter_padding_mask_matches_hidden_states() -> None:
    model = HybridModel.__new__(HybridModel)
    model.config = SimpleNamespace(sequence_parallel=True)
    model.pg_collection = SimpleNamespace(tp=get_tensor_model_parallel_group())
    mask = torch.tensor([[False, True, False, True]], device="cuda")
    observed = model._scatter_padding_mask_to_sequence_parallel(mask)
    expected = (
        tensor_parallel.scatter_to_sequence_parallel_region(
            mask.transpose(0, 1).contiguous(),
            group=model.pg_collection.tp,
        )
        .transpose(0, 1)
        .contiguous()
    )
    assert observed.shape == expected.shape
    assert torch.equal(observed, expected)


def test_hybrid_mtp_receives_padding_mask() -> None:
    hidden = torch.randn(4, 1, 8, device="cuda")
    model = SimpleNamespace(mtp=Mock(return_value=hidden))
    mask = torch.tensor([[False, False, True, True]], device="cuda")
    packed_seq_params = PackedSeqParams(qkv_format="thd")
    HybridModel._forward_mtp(
        model,
        input_ids=torch.ones(1, 4, dtype=torch.long, device="cuda"),
        position_ids=torch.arange(4, device="cuda").unsqueeze(0),
        hidden_states=hidden,
        attention_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=packed_seq_params,
        padding_mask=mask,
        embedding=Mock(),
    )
    assert torch.equal(model.mtp.call_args.kwargs["padding_mask"], mask)
    assert (
        model.mtp.call_args.kwargs["packed_seq_params"] is packed_seq_params
    )


def test_hybridep_dropless_removes_padding_routes() -> None:
    container = MoEModelTestContainer(
        tp_size=1,
        ep_size=1,
        pp_size=1,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
    )
    router = container.moe_layer.router.cuda()
    padding = torch.tensor([[False], [True], [False], [True]], device="cuda")
    probs, routing_map = router(
        torch.randn(4, 1, router.config.hidden_size, device="cuda"), padding
    )
    assert not routing_map[padding.reshape(-1)].any()
    assert torch.count_nonzero(probs[padding.reshape(-1)]) == 0
```

- [ ] **Step 2: Run the red tests**

```bash
uv run pytest -q \
  tests/unit_tests/models/test_hybrid_moe_model.py \
  tests/unit_tests/transformer/moe/test_routers.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  -k "padding_mask or padding_routes"
```

- [ ] **Step 3: Implement the model-local mask transform**

Mirror GPTModel's SP transform exactly on the embedding-owning stage. PP
intermediate stages have neither `input_ids` nor a full global mask to
validate/scatter:

```python
if padding_mask is not None and input_ids is not None:
    assert padding_mask.shape == input_ids.shape
if (
    padding_mask is not None
    and input_ids is not None
    and self.config.sequence_parallel
):
    padding_mask = (
        tensor_parallel.scatter_to_sequence_parallel_region(
            padding_mask.transpose(0, 1).contiguous(),
            group=self.pg_collection.tp,
        )
        .transpose(0, 1)
        .contiguous()
    )
```

Pass the transformed mask to both `self.decoder(...)` and `self.mtp(...)`.
When fixed THD graph capacity is enabled,
`TransformerLayer.get_layer_static_inputs()` creates a bool mask shaped
`[micro_batch_size, slen_per_cptp]`.

Extract the existing MTP call without changing its behavior:

```python
def _forward_mtp(
    self,
    *,
    input_ids,
    position_ids,
    hidden_states,
    attention_mask,
    inference_params,
    rotary_pos_emb,
    packed_seq_params,
    padding_mask,
    embedding,
):
    return self.mtp(
        input_ids=input_ids,
        position_ids=position_ids,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
        embedding=embedding,
    )
```

- [ ] **Step 4: Preserve mask and packed metadata across every graph boundary**

Port the reviewed PR 4359 `7f9175207` invariant in main-native form:

```python
hidden_states = self._forward_mlp(
    hidden_states,
    padding_mask=padding_mask,
    packed_seq_params=packed_seq_params,
)
```

Task 2 commit `cd170e14e` already uses that call in
`_te_cuda_graph_capture` and has the capture regression; do not duplicate or
rewrite it. Add the missing THD static bool mask. In attention-only replay,
retain the original `padding_mask` and rebuilt `packed_seq_params` before
clearing graph kwargs, then pass them into the eager MLP/router tail. In full
and partial MoE capture, preserve Task 2's graph-static `tokens_per_sample` so
`_maybe_unflatten_for_moe()` observes the same batch geometry as eager
execution.

- [ ] **Step 5: Apply PR 5542 and retain fail-closed capacity behavior**

```bash
git cherry-pick d1384c2d95c4fb18a892c524aa9991441e83b9db
```

For dropless Flex/HybridEP, zero padding rows before dispatch. For full-MoE
drop-and-pad, enable the path only after the test proves masked padding does
not consume a valid token's capacity. Other unverified sparse-dispatch
backends reject full-MoE packed capture in config validation. Replace the
dev-only `deepepv2` negative test case from PR 5542 with current main's
`ncclep` backend name.

- [ ] **Step 6: Run and commit**

```bash
uv run pytest -q \
  tests/unit_tests/models/test_hybrid_moe_model.py \
  tests/unit_tests/transformer/moe/test_routers.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git diff --check
git add megatron/core/models/hybrid/hybrid_model.py \
  megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/moe/router.py \
  tests/unit_tests/models/test_hybrid_moe_model.py \
  tests/unit_tests/transformer/moe/test_routers.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git commit -s -S -m "fix: exclude packed padding from hybrid routing"
```

### Task 5: Make MoE Dispatchers Own Replay State

**Files:**
- Create: `megatron/core/transformer/moe/cuda_graph_replay.py`
- Modify: `megatron/core/transformer/moe/token_dispatcher.py`
- Modify: `megatron/core/transformer/transformer_layer.py`
- Test: `tests/unit_tests/transformer/moe/test_token_dispatcher.py`
- Test: `tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py`
- Test: `tests/unit_tests/transformer/test_cuda_graphs.py`

**Interfaces:**
- Consumes: graph input and preprocessed hidden-state tensors.
- Produces:
  `snapshot_cudagraph_replay_state`,
  `restore_cudagraph_replay_state`, and
  `validate_cudagraph_continuation`, indexed by captured graph runner.

- [ ] **Step 1: Define red contract tests against real dispatcher classes**

Add real-dispatcher tests with these contracts:

- AlltoAll snapshots and restores exact `hidden_shape`,
  `hidden_shape_before_permute`, scalar capacity, and output-count geometry.
- fixed-capacity Flex/HybridEP snapshots and restores manager
  `_original_num_tokens`, `_padded_num_tokens`, capacity,
  `num_permuted_tokens`, and an immutable tuple form of any CPU
  `tokens_per_expert` required by drop-and-pad.
- exact input and continuation signatures reject shape, dtype, device, layout,
  or stride changes even when `.numel()` is unchanged.
- two mocked graph runners for one layer restore different states in
  `test_cuda_graphs.py`; a single mutable layer-wide state is forbidden.
- packed AllGather with cleared padding routes fails capability validation
  because eager permutation narrows expert input.
- Flex/DeepEP and Flex/NCCL-EP fail before capture. NCCL-EP static mode remains
  disabled until Task 6 graph-bank activation owns its external-buffer
  bootstrap/reset lifecycle.

- [ ] **Step 2: Run the red tests**

```bash
uv run pytest -q \
  tests/unit_tests/transformer/moe/test_token_dispatcher.py \
  tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  -k "replay_state or continuation"
```

- [ ] **Step 3: Add the typed state**

```python
@dataclass(frozen=True)
class TensorReplaySignature:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    stride: tuple[int, ...]


@dataclass(frozen=True)
class AlltoAllCudaGraphState:
    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size
    capacity: int | None
    num_out_tokens: int | None


@dataclass(frozen=True)
class HybridEPCudaGraphState:
    original_num_tokens: int
    padded_num_tokens: int
    capacity: int | None
    num_permuted_tokens: int
    tokens_per_expert: tuple[int, ...] | None


@dataclass(frozen=True)
class MoECudaGraphReplayState:
    dispatcher_kind: Literal["alltoall", "flex-hybridep"]
    input_signature: TensorReplaySignature
    flattened_input_shape: torch.Size
    topology_fingerprint: tuple[tuple[str, object], ...]
    backend_state: AlltoAllCudaGraphState | HybridEPCudaGraphState
```

The base dispatcher methods raise `NotImplementedError`, and capability
validation runs before any rank starts graph capture. AlltoAll and
fixed-capacity Flex/HybridEP implement snapshot/restore. The topology
fingerprint includes backend identity, TP/EP sizes, router top-k, drop-and-pad,
rank/expert capacity, and HybridEP uneven-input policy. Per-invocation
stream/event/handle state is never snapshotted.

- [ ] **Step 4: Integrate state with partial replay**

Capture state after `dispatch_preprocess` produces graph outputs but before
the graph is committed to its runner. `TransformerLayer` records one immutable
state per captured graph index. After TE replay restores Tensor
`cudagraph_attrs`, it restores the state paired with that exact runner before
calling eager dispatch/expert/combine. After combine it validates exact output
shape, dtype, device, layout, and stride against the physical-capacity input;
HybridEP additionally verifies returned rows equal restored
`original_num_tokens`. No output-size repair, rank-local fallback, prefix
reconstruction, or hidden final truncation is present.

- [ ] **Step 5: Run and commit**

```bash
uv run pytest -q \
  tests/unit_tests/transformer/moe/test_token_dispatcher.py \
  tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git diff --check
git add megatron/core/transformer/moe/cuda_graph_replay.py \
  megatron/core/transformer/moe/token_dispatcher.py \
  megatron/core/transformer/transformer_layer.py \
  tests/unit_tests/transformer/moe/test_token_dispatcher.py \
  tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git commit -s -S -m "fix: restore dispatcher-owned TE graph state"
```

### Task 6: Add Owned Graph Banks and Exact Replay Signatures

**Files:**
- Create: `megatron/core/transformer/te_cuda_graph_bank.py`
- Modify: `megatron/core/transformer/cuda_graphs.py`
- Modify: `megatron/core/transformer/module.py`
- Modify: `megatron/core/transformer/transformer_layer.py`
- Test: `tests/unit_tests/transformer/test_te_cuda_graph_bank.py`
- Test: `tests/unit_tests/transformer/test_cuda_graphs.py`

**Interfaces:**
- Consumes: a fresh one-shot `TECudaGraphHelper`, normalized microbatch count,
  packed signature, structural-mask signature, and Task 5 per-runner
  dispatcher states.
- Produces: `TECudaGraphBank`, `TECudaGraphBankManager.capture(...)`,
  `.activate()`, and `.reset()`.

- [ ] **Step 1: Forward-port the old bank tests as red tests**

Port only the tests and public behavior from the 20260729 branch, beginning
with commit `e97dd5e3e`. Bring the test-local `_FakeGraph`, `_FakeMoELayer`,
`_FakeHelper`, and `_make_manager` fixtures with the assertions; do not copy
cluster overlays or runtime workarounds. Adapt them to these exact cases:

```python
def test_graph_bank_activation_restores_dispatcher_states() -> None:
    layer = _FakeMoELayer("moe")
    manager = _make_manager([layer], runtime_num_microbatches=lambda: 5)
    first_states = tuple(_make_alltoall_state(16) for _ in range(5))
    layer._te_cuda_graph_dispatcher_replay_states = first_states
    first = manager.capture(
        _FakeHelper([layer], [[_FakeGraph(f"first-{i}") for i in range(5)]]),
        num_microbatches=5,
    )
    manager._runtime_num_microbatches = lambda: 3
    second_states = tuple(_make_alltoall_state(24) for _ in range(3))
    layer._te_cuda_graph_dispatcher_replay_states = second_states
    second = manager.capture(
        _FakeHelper([layer], [[_FakeGraph(f"second-{i}") for i in range(3)]]),
        num_microbatches=3,
    )
    manager._runtime_num_microbatches = lambda: 5
    first.activate()
    assert layer._te_cuda_graph_dispatcher_replay_states is first_states
    manager._runtime_num_microbatches = lambda: 3
    second.activate()
    assert layer._te_cuda_graph_dispatcher_replay_states is second_states
```

`_make_alltoall_state()` constructs the full Task 5
`TensorReplaySignature`, flattened shape, topology fingerprint, and
`AlltoAllCudaGraphState`; do not use the obsolete sparse state constructor.
Add RED tests for:

- exact graph-list, callable, manual-hook, packed-attention, Mamba, mask,
  ordered MoE Tensor schema, and per-runner dispatcher-state identity;
- runtime-count mismatch before activation, including the invalid 5→3
  activation shown above without resetting the provider;
- capture/activation rollback, helper one-shot behavior, inactive/idempotent
  reset, and unique-callable reset;
- live continuation/delayed work blocking every transition;
- forward and backward-dw replay guards;
- real `MambaLayer` whole/partial helper routing, closing Task 3's deferred
  production-path test.

- [ ] **Step 2: Verify the bank tests fail**

```bash
uv run pytest -q \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py -k "graph_bank"
```

- [ ] **Step 3: Port and tighten the bank implementation**

Use this fingerprint:

```python
@dataclass(frozen=True)
class TECudaGraphBankFingerprint:
    num_microbatches: int
    layer_ids: tuple[int, ...]
    graph_identities: tuple[tuple[int, ...], ...]
    graph_counts: tuple[int, ...]
    cuda_graph_modules: tuple[str, ...]
    packed_input_signatures: tuple[tuple[int, tuple[object, ...]], ...]
    padding_mask_signatures: tuple[
        tuple[int, TensorReplaySignature | None], ...
    ]
    moe_attribute_schema: tuple[tuple[int, tuple[str, ...]], ...]
    dispatcher_state_signatures: tuple[
        tuple[int, tuple[MoECudaGraphReplayState | None, ...]], ...
    ]
```

`packed_input_signatures` freezes the generic attention metadata/key set and
the separate Mamba static/`seq_idx` signature per layer; it never mixes their
field lists or freezes dynamic contents. `padding_mask_signatures` represents
absence explicitly and preserves per-layer shape/dtype/device/layout/stride.
Dispatcher-state tuple length must equal the corresponding graph count and be
paired with `graph_identities`.

`capture()` accepts an uncaptured helper, verifies the helper's normalized
schedule count against the requested/runtime count, installs empty
manager-owned lists, populates them once, snapshots every contract, and then
restores the previous active installation. Capture failure restores all
capture globals and the previous bank. Helper reuse fails.

Activation first verifies the model is drained and the runtime count matches,
then installs graph-list identity, manual hooks, generic/Mamba metadata, mask
surface, ordered MoE Tensor schema, dispatcher states, and replay guard as one
transaction. Capture currently precedes `cuda_graph_set_manual_hooks()`, so
hook setup must refresh the owning bank contract (or move under the manager)
before any later activation.

Forward and backward-dw replay guards validate the registered active bank,
runtime count, exact installed list/callable identity, and selected graph
index in constant hot-path work. Capture, activation, reset, and eviction call
the supplied drained callback and reject live partial-MoE continuation,
delayed-wgrad/communication, activation-checkpoint, or shared-expert state.
Reset is idempotent, synchronizes at the drained boundary, releases graph
references, resets each owned callable identity once, and never resets graphs
owned by another bank.

- [ ] **Step 4: Run and commit**

```bash
uv run pytest -q \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git diff --check
git add megatron/core/transformer/te_cuda_graph_bank.py \
  megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/module.py \
  megatron/core/transformer/transformer_layer.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git commit -s -S -m "feat: own schedule-specific TE graph banks"
```

### Task 7: Cover Hybrid MTP and Shared-Expert Weight Gradients

**Files:**
- Modify: `megatron/core/transformer/cuda_graphs.py`
- Modify: `megatron/core/transformer/transformer_layer.py`
- Modify: `megatron/core/models/common/fine_grained_callables.py`
- Modify: `megatron/core/models/common/utils.py`
- Test: `tests/unit_tests/transformer/test_cuda_graphs.py`
- Test: `tests/unit_tests/transformer/test_submodule_callables.py`

**Interfaces:**
- Consumes: an MTP `mtp_model_layer` that is either a single graphable layer or
  a nested `HybridStack`.
- Produces:
  `_iter_graphable_te_leaves(module, config)` and explicit
  `shared_expert_wgrad_captured` metadata.

- [ ] **Step 1: Add red nested-MTP discovery tests**

```python
def test_te_discovery_descends_into_hybrid_mtp_stack() -> None:
    attention = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(attention)
    attention.self_attention = torch.nn.Identity()
    attention.cross_attention = IdentityOp()
    attention.mlp = IdentityOp()

    moe = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(moe)
    moe.self_attention = IdentityOp()
    moe.cross_attention = IdentityOp()
    moe.mlp = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(moe.mlp)

    mtp_stack = HybridStack.__new__(HybridStack)
    torch.nn.Module.__init__(mtp_stack)
    mtp_stack.layers = torch.nn.ModuleList([attention, moe])
    config = SimpleNamespace(
        cuda_graph_modules=[
            CudaGraphModule.attn,
            CudaGraphModule.moe_router,
            CudaGraphModule.moe_preprocess,
        ]
    )

    assert list(_iter_graphable_te_leaves(mtp_stack, config)) == [
        attention,
        moe,
    ]


def test_delayed_shared_expert_wgrad_runs_when_not_captured() -> None:
    wrapper = _BackwardDWWrapper.__new__(_BackwardDWWrapper)
    wrapper.layer = SimpleNamespace(cuda_graphs=[object()])
    wrapper.cuda_graph_modules = [CudaGraphModule.moe_router]
    wrapper.shared_expert_wgrad_captured = False
    wrapper.shared_expert_dw_callable = Mock()
    wrapper.attn_dw_callable = Mock()
    wrapper.graphed_backward_dw_callable = Mock()

    wrapper.backward_dw()

    wrapper.shared_expert_dw_callable.assert_called_once_with()
    wrapper.graphed_backward_dw_callable.assert_called_once_with()
    wrapper.attn_dw_callable.assert_called_once_with()
```

- [ ] **Step 2: Run the red tests**

```bash
uv run pytest -q \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_submodule_callables.py \
  -k "mtp or shared_expert"
```

- [ ] **Step 3: Implement one leaf-expansion path**

Use `_iter_graphable_te_leaves` in all three places:

```python
def _iter_graphable_te_leaves(module, config):
    if isinstance(module, HybridStack):
        for layer in module.layers:
            if _layer_is_graphable(layer, config):
                yield layer
        return
    if _layer_is_graphable(module, config):
        yield module
```

Discovery, static sample membership checks, and current-microbatch assignment
must consume the same returned leaf sequence. A requested scope with zero
matching leaves raises during helper construction.

- [ ] **Step 4: Make MTP overlap scheduling fail closed**

Ordinary non-overlapped MTP graph discovery supports a nested `HybridStack`.
The existing expert-parallel overlap scheduler in
`models/common/fine_grained_callables.py` assumes that `mtp_model_layer` is a
single `TransformerLayer`; it must not silently build an incomplete schedule
for a `HybridStack`. Add a validation in `build_mtp_layer_callables` that
raises a descriptive `RuntimeError` for packed hybrid MTP with
`overlap_moe_expert_parallel_comm=True`. The Super correctness and performance
matrix exercises MTP with that overlap disabled.

- [ ] **Step 5: Make delayed-wgrad ownership explicit**

Set `shared_expert_wgrad_captured=True` only when the selected submodules
actually include the non-overlapped shared expert. AlltoAll overlap restores
its state-machine transition; Flex overlap remains eager.

- [ ] **Step 6: Run and commit**

```bash
uv run pytest -q \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_submodule_callables.py
git diff --check
git add megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/transformer_layer.py \
  megatron/core/models/common/fine_grained_callables.py \
  megatron/core/models/common/utils.py \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_submodule_callables.py
git commit -s -S -m "fix: graph hybrid MTP and shared-expert ownership"
```

### Task 8: Add NeMo-RL Structural Padding Mask Plumbing

**Files:**
- Modify: `nemo_rl/models/megatron/data.py`
- Modify: `nemo_rl/models/megatron/train.py`
- Test: `tests/unit/models/megatron/test_megatron_data.py`
- Test: `tests/unit/models/megatron/test_train.py`

**Interfaces:**
- Consumes: logical sequence lengths and physical `cu_seqlens_padded`.
- Produces:
  `_build_packed_structural_padding_mask(...) -> tuple[Tensor, Tensor]`,
  `PackedGeometry`, and a
  `ProcessedMicrobatch.structural_padding_mask_cp_sharded` in CP-local order.

- [ ] **Step 1: Add red physical-layout tests**

```python
def test_structural_mask_marks_internal_and_capacity_padding() -> None:
    full, local = _build_packed_structural_padding_mask(
        seq_lengths=torch.tensor([3, 2]),
        cu_seqlens_padded=torch.tensor([0, 4, 8]),
        cp_rank=0,
        cp_size=1,
    )
    assert torch.equal(
        full,
        torch.tensor([[False, False, False, True, False, False, True, True]]),
    )
    assert torch.equal(local, full)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_structural_mask_uses_same_zigzag_cp_order_as_tokens(cp_rank) -> None:
    packed_tokens, cp_tokens, params, _, padded = _pack_sequences_for_megatron(
        input_ids=torch.arange(16).reshape(2, 8),
        seq_lengths=torch.tensor([5, 3]),
        pad_individual_seqs_to_multiple_of=4,
        pad_packed_seq_to_multiple_of=16,
        cp_rank=cp_rank,
        cp_size=2,
    )
    full_mask, local_mask = _build_packed_structural_padding_mask(
        torch.tensor([5, 3]), padded, cp_rank=cp_rank, cp_size=2
    )
    assert local_mask.shape == cp_tokens.shape
    expected_parts = []
    for sample_idx, seq_len in enumerate((5, 3)):
        physical_len = int(padded[sample_idx + 1] - padded[sample_idx])
        sample_mask = torch.arange(physical_len) >= seq_len
        expected_parts.append(
            _get_tokens_on_this_cp_rank(
                sample_mask, cp_rank, 2, seq_dim=0
            )
        )
    assert torch.equal(local_mask, torch.cat(expected_parts).unsqueeze(0))
    assert full_mask.sum() == 8
```

- [ ] **Step 2: Run the red tests**

```bash
uv run pytest -q \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_train.py \
  -k "structural_mask or padding_mask"
```

- [ ] **Step 3: Implement mask construction and propagation**

```python
def _build_packed_structural_padding_mask(
    seq_lengths: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_masks = []
    local_masks = []
    for sample_idx, seq_len_tensor in enumerate(seq_lengths):
        seq_len = int(seq_len_tensor)
        physical_len = int(
            cu_seqlens_padded[sample_idx + 1] - cu_seqlens_padded[sample_idx]
        )
        mask = torch.arange(
            physical_len, device=seq_lengths.device
        ) >= seq_len
        full_masks.append(mask)
        local_masks.append(
            _get_tokens_on_this_cp_rank(mask, cp_rank, cp_size, seq_dim=0)
            if cp_size > 1
            else mask
        )
    return (
        torch.cat(full_masks).unsqueeze(0).contiguous(),
        torch.cat(local_masks).unsqueeze(0).contiguous(),
    )
```

Add this typed geometry:

```python
@dataclass(frozen=True)
class PackedGeometry:
    logical_tokens: int
    padded_tokens: int
    capacity_tokens: int
    real_sequence_count: int
    cu_seqlens_capacity_entries: int
```

Add `cu_seqlens`, `structural_padding_mask`,
`structural_padding_mask_cp_sharded`, and `packed_geometry` to
`ProcessedInputs` and `ProcessedMicrobatch`. Copy them in the iterator. Pass
the CP-sharded mask through `forward_with_post_processing_fn`, then call
`model(..., padding_mask=structural_padding_mask_cp_sharded)`. Loss wrappers
continue to consume actual `cu_seqlens`; fixed graph sentinel entries never
become loss boundaries. Reject packed TE training graphs when
`delegate_pack_to_model=True`, because NeMo-RL cannot construct the model's
internal physical ordering.

- [ ] **Step 4: Run and commit**

```bash
uv run pytest -q \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_train.py
git diff --check
git add nemo_rl/models/megatron/data.py \
  nemo_rl/models/megatron/train.py \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_train.py
git commit -s -m "fix: carry packed structural padding into MCore"
```

### Task 9: Forward-Port NeMo-RL Graph Lifecycle and Telemetry

**Files:**
- Create: `nemo_rl/models/megatron/cuda_graph_lifecycle.py`
- Modify: `nemo_rl/data/packing/algorithms.py`
- Modify: `nemo_rl/distributed/batched_data_dict.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/algorithms/utils.py`
- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `pyrefly.toml`
- Test: `tests/unit/data/packing/test_algorithms.py`
- Test: `tests/unit/models/megatron/test_cuda_graph_lifecycle.py`
- Test: `tests/unit/models/megatron/test_megatron_setup.py`
- Test: `tests/unit/models/policy/test_policy_validation.py`
- Test: `tests/unit/models/policy/test_megatron_worker.py`
- Test: `tests/unit/models/policy/test_lm_policy.py`
- Test: `tests/unit/algorithms/test_utils.py`
- Test: `tests/unit/algorithms/test_grpo.py`

**Interfaces:**
- Consumes: MCore `TECudaGraphBank`, normalized pipeline microbatch count, and
  per-step packed geometry.
- Produces: graph-bank lifecycle outcomes and serializable `cuda_graph/*`
  metrics.

- [ ] **Step 1: Port lifecycle tests before code**

Use the 20260729 tests and retain these exact outcomes:

```python
class _Bank:
    def __init__(self) -> None:
        self.activate_count = 0
        self.reset_count = 0

    def activate(self) -> None:
        self.activate_count += 1

    def reset(self) -> None:
        self.reset_count += 1


def _ensure(
    lifecycle: TECudaGraphLifecycle,
    num_microbatches: int,
) -> TECudaGraphEnsureResult:
    key = TECudaGraphScheduleKey(num_microbatches=num_microbatches)
    return lifecycle.ensure_active(key, _Bank)


def test_lifecycle_waits_for_three_successful_optimizer_steps() -> None:
    lifecycle = TECudaGraphLifecycle(capacity=2, warmup_steps=3)
    key = TECudaGraphScheduleKey(num_microbatches=5)
    for _ in range(2):
        lifecycle.record_optimizer_step(successful=True)
        assert lifecycle.ensure_active(key, _Bank).status == "warming"
    lifecycle.record_optimizer_step(successful=True)
    assert lifecycle.ensure_active(key, _Bank).status == "captured"


def test_lifecycle_is_two_entry_lru() -> None:
    lifecycle = TECudaGraphLifecycle(capacity=2, warmup_steps=0)
    _ensure(lifecycle, 3)
    _ensure(lifecycle, 5)
    _ensure(lifecycle, 3)
    result = _ensure(lifecycle, 7)
    assert result.evicted_key == TECudaGraphScheduleKey(5)
```

- [ ] **Step 2: Run the red lifecycle and worker tests**

```bash
uv run pytest -q \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/policy/test_lm_policy.py \
  tests/unit/algorithms/test_grpo.py
```

- [ ] **Step 3: Port the lifecycle without old runtime workarounds**

Forward-port `TECudaGraphScheduleKey`, `TECudaGraphEnsureResult`, and
`TECudaGraphLifecycle` from the reviewed outer commits `c72565364` and
`c6eaec655`. Port selected lifecycle and telemetry hunks only; do not port old
TE overlay builders, Python symlink workarounds, output padding, checked
results, or hard-coded cluster paths.

- [ ] **Step 4: Return typed step telemetry**

```python
@dataclass(frozen=True)
class CudaGraphStepMetrics:
    capture_count: int
    replay_count: int
    cache_hit_count: int
    eviction_count: int
    fallback_count: int
    graph_calls: int
    eligible_calls: int
    logical_tokens: int
    padded_tokens: int
    capacity_tokens: int
```

Worker sync and split APIs return the dataclass as a plain numeric mapping.
Keep this mapping separate from `all_mb_metrics`. After existing CP/TP/PP
deduplication, `LMPolicy` verifies that lifecycle counters and geometry keys
match across DP representatives. It selects one lifecycle-counter value,
sums logical/padded/capacity tokens and graph/eligible calls across DP, then
recomputes `coverage = graph_calls / eligible_calls` and
`capacity_utilization = logical_tokens / capacity_tokens`. It never averages
ratios or sums replicated capture counts. Both synchronous `train()` and split
`finish_train_step()` return the same schema. GRPO's two training paths and
GRPO-sync's training path log the result under `cuda_graph/*`.
`fallback_count` must remain zero; no code path increments it and continues.

- [ ] **Step 5: Run and commit**

```bash
uv run pytest -q \
  tests/unit/data/packing/test_algorithms.py \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_policy_validation.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/policy/test_lm_policy.py \
  tests/unit/algorithms/test_utils.py \
  tests/unit/algorithms/test_grpo.py
git diff --check
git add nemo_rl/models/megatron/cuda_graph_lifecycle.py \
  nemo_rl/data/packing/algorithms.py \
  nemo_rl/distributed/batched_data_dict.py \
  nemo_rl/models/megatron/setup.py \
  nemo_rl/models/policy/__init__.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/models/policy/lm_policy.py \
  nemo_rl/algorithms/utils.py \
  nemo_rl/algorithms/grpo.py \
  nemo_rl/algorithms/grpo_sync.py \
  pyrefly.toml \
  tests/unit/data/packing/test_algorithms.py \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_policy_validation.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/policy/test_lm_policy.py \
  tests/unit/algorithms/test_utils.py \
  tests/unit/algorithms/test_grpo.py
git commit -s -m "feat: expose bounded TE graph lifecycle metrics"
```

### Task 10: Validate Bridge Nemotron Topologies and Pin MCore

**Files:**
- Modify: `3rdparty/Megatron-LM`
- Modify: `tests/unit_tests/recipes/nemotronh/test_nemotron_3_nano.py`
- Modify: `tests/unit_tests/recipes/nemotronh/test_nemotron_3_super.py`
- Modify: `tests/unit_tests/recipes/nemotronh/test_nemotron_3_ultra.py`

**Interfaces:**
- Consumes: verified MCore branch.
- Produces: Bridge branch with unchanged production recipe defaults and
  tested experiment overrides.

- [ ] **Step 1: Add topology assertions**

```python
def test_nano_accepts_te_hybrid_scope_override() -> None:
    cfg = nemotron_3_nano_pretrain_config()
    set_cuda_graph_modules(
        cfg.model,
        ["attn", "mamba", "moe_router", "moe_preprocess"],
    )
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"


def test_super_scope_override_keeps_mtp_and_latent_moe() -> None:
    cfg = nemotron_3_super_pretrain_config()
    set_cuda_graph_modules(
        cfg.model,
        ["attn", "mamba", "moe_router", "moe_preprocess"],
    )
    assert cfg.model.mtp_num_layers > 0
    assert cfg.model.moe_latent_size is not None


def test_ultra_default_remains_graph_disabled() -> None:
    cfg = nemotron_3_ultra_pretrain_config()
    assert cfg.model.cuda_graph_impl == "none"
```

- [ ] **Step 2: Run Bridge recipe tests**

```bash
uv run pytest -q \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_nano.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_super.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_ultra.py
```

- [ ] **Step 3: Commit and push MCore first**

```bash
git -C 3rdparty/Megatron-LM status --short
git -C 3rdparty/Megatron-LM push -u origin sj/thd-cg-hybrid-nemotron-20260731
git add 3rdparty/Megatron-LM \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_nano.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_super.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_ultra.py
git commit -s -m "build: pin packed THD CUDA graph MCore"
git push -u origin sna/thd-cg-hybrid-nemotron-20260731
```

### Task 11: Build the Persistent Scope Matrix and HTML Report

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_smoke_matrix.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_performance_matrix.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_accuracy_soak.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/collect_results.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/analyze_cuda_graph_calls.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_tensorboard.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_te_runtime.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_nemorl_scope.sub`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_scope.sub`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/nano.env`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/super.env`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/ultra.env`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/ptyche.env.example`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/oci-hsg.env.example`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/lyris.env.example`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/00_baseline_no_cg.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/01_whole_layer.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/02_moe.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/03_moe_router.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/04_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/05_mamba.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/06_mamba_moe.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/07_mamba_moe_router.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/08_mamba_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/09_mlp.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/10_mlp_moe.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/11_mlp_moe_router.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/12_mlp_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/13_mlp_mamba.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/14_mlp_mamba_moe.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/15_mlp_mamba_moe_router.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/16_mlp_mamba_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/17_attn.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/18_attn_moe.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/19_attn_moe_router.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/20_attn_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/21_attn_mamba.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/22_attn_mamba_moe.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/23_attn_mamba_moe_router.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/24_attn_mamba_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/25_attn_mlp.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/26_attn_mlp_moe.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/27_attn_mlp_moe_router.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/28_attn_mlp_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/29_attn_mlp_mamba.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/30_attn_mlp_mamba_moe.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/31_attn_mlp_mamba_moe_router.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/32_attn_mlp_mamba_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/attn_mamba_router_preprocess_overlap_false.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/moe_overlap_false_moe_act_false.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/moe_overlap_false_moe_act_true.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/moe_overlap_true_moe_act_false.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/moe_overlap_true_moe_act_true.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/router_preprocess_overlap_false_moe_act_false.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/router_preprocess_overlap_false_moe_act_true.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/router_preprocess_overlap_true_moe_act_false.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/router_preprocess_overlap_true_moe_act_true.sh`
- Test: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`
- Test: `tests/unit/experiments/test_analyze_cuda_graph_calls.py`
- Test: `tests/unit/experiments/test_export_tensorboard.py`
- Test: `tests/unit/experiments/test_validate_te_runtime.py`

**Interfaces:**
- Consumes: model, scope, dispatcher, step count, and cluster profile.
- Produces: reproducible SLURM jobs, normalized CSV/JSON, and one static HTML
  report.

- [ ] **Step 1: Add launcher red tests**

```python
def test_scope_matrix_contains_all_32_te_rows_and_baseline() -> None:
    rows = load_scope_matrix()
    assert len(rows) == 33
    assert rows[0].scope == ()
    assert rows[-1].scope == (
        "attn",
        "mlp",
        "mamba",
        "moe_router",
        "moe_preprocess",
    )


def test_launcher_disables_checkpoints_and_uses_three_warmups() -> None:
    command = render_scope_command(model="nano", scope=("attn",), steps=20)
    assert "checkpointing.enabled=false" in command
    assert "cuda_graph_warmup_steps=3" in command
    assert "logger.wandb.project='sna-cg-study'" in command


def test_launcher_requires_verified_native_te_runtime() -> None:
    command = render_scope_command(model="nano", scope=("attn",), steps=20)
    assert "validate_te_runtime.py" in command
```

- [ ] **Step 2: Run the red harness tests**

```bash
uv run pytest -q \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  tests/unit/experiments/test_analyze_cuda_graph_calls.py \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_validate_te_runtime.py
```

- [ ] **Step 3: Implement the exact scope matrix**

The matrix has dense subsets of `attn`, `mlp`, and `mamba`, crossed with
`none`, `moe`, `moe_router`, and `moe_router+moe_preprocess`. Add a separate
no-CG baseline. `scope_matrix.py` classifies each row as runnable,
model-incompatible, capacity-blocked, dependency-blocked, or submitted before
calling `sbatch`. The `variants/` files independently cross shared-expert
overlap and `moe_act` so those configuration dimensions do not masquerade as
graph-scope names.

- [ ] **Step 4: Implement the reusable launcher**

`run_scope.sh` accepts:

```text
MODEL=nano|super|ultra
SCOPE=baseline|comma-separated modules
STEPS=5|20|100
CLUSTER=ptyche|oci-hsg
MODE=nemorl|mcore
```

It sources one model file, uses `NRL_FORCE_REBUILD_VENVS=true`, disables
checkpointing, writes under
`exp_logs/nemotron_thd_te_graph_20260731/<run-name>`, and never embeds
credentials. Ultra refuses submission until its external model path, data,
judge, and launch profile are all present.

Before Python environment rebuild or model launch, the job runs
`validate_te_runtime.py` inside the immutable nightly container. The validator
records the container digest, installed TE distribution versions, and native
TE build revision, and rejects an unverifiable revision or one older than
`869f99c47`. It never clones or builds Transformer Engine.

Model selectors are fixed to:

```text
Nano: examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml
Super: examples/configs/recipes/llm/grpo-nemotron3-super-120BA12B-8n4g-megatron.yaml
Ultra NeMo provenance: examples/nemo_gym/nemotron-3-ultra/student_rlvr1.yaml
Ultra Bridge fixture: src/megatron/bridge/recipes/nemotronh/h100/nemotron_3_ultra.py
```

All production submissions use the SLURM `batch` partition, not `backfill`.

- [ ] **Step 5: Implement collection and reporting**

The report table includes:

```text
model, dispatcher, scope, status, steps,
E2E step time and tokens/s/GPU,
generation step time and tokens/s/GPU,
policy-training step time and tokens/s/GPU,
logprob step time and tokens/s/GPU,
graph calls, eligible calls, graph coverage,
logical/padded/capacity tokens, utilization,
reward, policy loss, gen_kl_error, multi_logprob_error,
router top-k parity, expert-count parity, grad norm,
NaN/Inf status, source commits, TE commit, job id
```

- [ ] **Step 6: Run tests and commit**

```bash
uv run pytest -q \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  tests/unit/experiments/test_analyze_cuda_graph_calls.py \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_validate_te_runtime.py
git diff --check
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731 \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  tests/unit/experiments/test_analyze_cuda_graph_calls.py \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_validate_te_runtime.py
git commit -s -m "test: add Nemotron THD CUDA graph matrix"
```

### Task 12: Run Local and Distributed Correctness Gates

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_scope.sub`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/mcore_correctness.json`

**Interfaces:**
- Consumes: pushed MCore/Bridge/NeMo branches and immutable container/TE
  provenance.
- Produces: eager-versus-graph forward, backward, router, gradient, and
  optimizer parity evidence.

- [ ] **Step 1: Run CPU/unit suites**

```bash
uv run pytest -q \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_train.py \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/policy/test_lm_policy.py \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
```

- [ ] **Step 2: Run MCore focused suites in a compatible GPU container**

```bash
uv run python \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_te_runtime.py
uv run pytest -q \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/moe/test_token_dispatcher.py \
  tests/unit_tests/transformer/moe/test_routers.py \
  tests/unit_tests/models/test_hybrid_moe_model.py
```

- [ ] **Step 3: Commit and push the outer dependency pointer**

```bash
git add 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git commit -s -m "build: pin Nemotron THD CUDA graph Bridge"
git push -u origin experiment/thd-cg-hybrid-nemotron-20260731
```

- [ ] **Step 4: Preflight and submit distributed parity jobs**

Run SLURM `--test-only` first. Then submit independent rows in parallel for:

```text
GPT dense: attn, mlp, attn+mlp
Nano: mamba; attn; moe_router; moe_router+moe_preprocess;
      attn+mamba+moe_router+moe_preprocess
Super: the Nano rows plus latent-MoE shared expert and MTP enabled
Ultra fixture: Flex/HybridEP router/preprocess and combined hybrid scope
```

Each job alternates two in-capacity occupancies after capture for 8-20 replay
iterations. It compares valid-token output, total/router loss, exact top-k
indices, every parameter gradient, optimizer-updated parameters, padding-row
gradients, expert counts, and valid-token drop count.

- [ ] **Step 5: Monitor every submitted job for five minutes**

```bash
squeue -j "$JOB_IDS" -o "%.18i %.2t %.10M %.40j"
sacct -j "$JOB_IDS" --format=JobID,State,Elapsed,ExitCode
```

Capture the first failing stack trace and stop dependent performance
submissions when a correctness gate fails.

### Task 13: Run NeMo-RL Scope, Performance, and Accuracy Experiments

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/results.csv`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/results.json`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/report.html`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md`

**Interfaces:**
- Consumes: Task 12 correctness-passing scopes.
- Produces: five-step smoke, paired 20-step performance/accuracy, and paired
  100-step stability results.

- [ ] **Step 1: Check scheduling before submission**

```bash
experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_smoke_matrix.sh \
  --cluster=ptyche --test-only
```

Use the best available account only after FairShare and `sbatch --test-only`
succeed.

- [ ] **Step 2: Submit all independent five-step rows in parallel**

```bash
experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_smoke_matrix.sh \
  --cluster=ptyche --models=nano,super --steps=5
```

Rows unsupported by model topology are recorded without submitting a job.
Ultra submits only when its resolved external profile passes preflight.

- [ ] **Step 3: Submit paired 20-step attribution rows**

```bash
experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_performance_matrix.sh \
  --cluster=ptyche --models=nano,super --steps=20 --repeats=3
```

Run baseline, each single axis, router/preprocess, and the best combined
correctness-passing scope. Random seeds, generated-token budgets, packing
capacity, and topology are identical within each pair.

- [ ] **Step 4: Submit paired 100-step stability rows**

```bash
experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_accuracy_soak.sh \
  --cluster=ptyche --models=nano,super --steps=100
```

Include attention-only and best combined scope because upstream issue 5966
reports possible late attention-graph gradient instability.

- [ ] **Step 5: Collect and render**

```bash
uv run python experiments/cuda_graph/nemotron_thd_te_graph_20260731/collect_results.py
uv run python experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py
```

The renderer fails if a completed comparison lacks timing, throughput,
coverage, provenance, or accuracy fields.

- [ ] **Step 6: Verify final artifacts and commit**

```bash
uv run pytest -q tests/unit/experiments
git diff --check
git status --short
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731
git commit -s -m "results: report Nemotron THD CUDA graph study"
git push
```

The task is complete only when no correctness-passing row has fallback,
capacity mismatch, padding influence, NaN/Inf, router divergence, or
unexplained accuracy drift.
