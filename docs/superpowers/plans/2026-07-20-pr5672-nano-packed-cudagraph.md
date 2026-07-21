# PR5672 Nano Packed CUDA Graph Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Extend PR5672 packed THD Transformer Engine graphs with a safe Mamba metadata ABI and reject unsupported Nano graph scopes before capture.

**Architecture:** A dedicated MCore branch starts at PR5672 head 6ff66f0a000ee65efa4f322c17871a3938f33427. Attention keeps PR5672's four dynamic cumulative-length tensors. Mamba adds dynamic seq_idx, but not total_tokens, and rebuilds a normal PackedSeqParams object only inside the captured callable. NeMo-RL rejects Nano packed-CP attention and TE FP64 router graph requests before construction and logs requested/effective scopes.

**Tech Stack:** Python 3.13, PyTorch, Transformer Engine, Megatron-LM, NeMo-RL, pytest, Ray, SLURM/Ptyche.

## Global Constraints

- Worktree: /Users/sna/CudaGraph_PR/RL-pr5672-nano-extension-20260720, branch experiment/pr5672-nano-packed-support-20260720.
- Create an independent MCore worktree from PR5672 head. Push MCore to seonjinn before pinning the NeMo-RL submodule.
- Port no whole Ultra branch. Preserve only required packed-replay safety behavior.
- Do not modify Transformer Engine. The known Nano packed CP2 attention and FP64 router limitations must fail before make_graphed_callables.
- Keep the attention gate Nano-only; do not block Qwen packed CP experiments.
- Do not change production FP64 router precision. FP32 router experiments are diagnostic-only.
- Use exactly three warmup steps and disable training checkpoint saving.
- Run all GPU tests and jobs in the Ptyche Linux container. Commit with -s and push only to the seonjinn forks.

---

## File Map

| File | Responsibility |
| --- | --- |
| 3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/packed_seq_params.py | Mamba packed graph fields and flatten/rebuild helpers. |
| 3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/transformer/cuda_graphs.py | Mamba graph sample kwargs. |
| 3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/ssm/mamba_layer.py | Mamba capture/replay adaptation. |
| 3rdparty/Megatron-LM-workspace/Megatron-LM/tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py | ABI tests. |
| 3rdparty/Megatron-LM-workspace/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py | GPU Mamba parity test. |
| nemo_rl/models/megatron/setup.py | Nano attention and FP64 router preflight. |
| nemo_rl/models/policy/workers/megatron_policy_worker.py | Scope provenance logging. |
| tests/unit/models/megatron/test_megatron_setup.py | Preflight unit tests. |
| experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh | Nano launcher contract. |
| tests/unit/experiments/test_nanov3_cuda_graph_launcher.py | Launcher tests. |

### Task 1: Create and validate the PR5672 MCore base

**Files:**
- Modify: 3rdparty/Megatron-LM-workspace/Megatron-LM only after MCore passes.
- Test: tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py.

**Consumes:** upstream/pr5672-latest at 6ff66f0a000ee65efa4f322c17871a3938f33427.

**Produces:** a clean experiment/pr5672-nano-packed-support-20260720 MCore branch.

- [x] **Step 1: Create a MCore worktree**

~~~
git -C /Users/sna/CudaGraph_PR/RL-pr5672-adapter-ptyche-20260719/3rdparty/Megatron-LM-workspace/Megatron-LM fetch upstream pull/5672/head:refs/remotes/upstream/pr5672-latest
git -C /Users/sna/CudaGraph_PR/RL-pr5672-adapter-ptyche-20260719/3rdparty/Megatron-LM-workspace/Megatron-LM worktree add -b experiment/pr5672-nano-packed-support-20260720 /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720 upstream/pr5672-latest
git -C /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720 remote set-url origin git@github.com:seonjinn/Megatron-LM.git
git -C /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720 rev-parse HEAD
~~~

Expected: 6ff66f0a000ee65efa4f322c17871a3938f33427.

- [x] **Step 2: Run the unmodified PR5672 test**

~~~
cd /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
~~~

Expected: PASS. Stop and diagnose environment/base failures before adding code.

- [x] **Step 3: Restore and verify the packed RoPE CP regression before Mamba work**

Append this failing test to the same test file:

~~~python
def test_te_cuda_graph_rotary_sample_preserves_packed_cp_layout():
    from megatron.core.transformer.cuda_graphs import _get_te_cuda_graph_rotary_pos_emb

    cp_group = object()
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 8]),
        cu_seqlens_kv=torch.IntTensor([0, 8]),
        max_seqlen_q=8,
        max_seqlen_kv=8,
        cp_group=cp_group,
    )

    class Rotary:
        def get_rotary_seq_len(self, unused, decoder, transformer_input, config, params):
            assert params is packed_seq_params
            return 8

        def __call__(self, seq_len, *, packed_seq, cp_group):
            return seq_len, packed_seq, cp_group

    class Model:
        position_embedding_type = "rope"
        rotary_pos_emb = Rotary()
        decoder = object()

    assert _get_te_cuda_graph_rotary_pos_emb(
        Model(), torch.ones(8, 1, 4), object(),
        packed_seq_params
    ) == (8, True, cp_group)
~~~

Run it before the repair:

~~~
cd /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py -k rotary_sample_preserves
~~~

Expected: FAIL because PR5672 builds the rotary sample inline without the packed and CP arguments. Extract the existing inline rotary-length logic into these two helpers in cuda_graphs.py and replace the inline call with the second helper:

~~~python
def _get_te_cuda_graph_rotary_seq_len(
    transformer_module, transformer_input, config, packed_seq_params
):
    return transformer_module.rotary_pos_emb.get_rotary_seq_len(
        None, transformer_module.decoder, transformer_input, config, packed_seq_params
    )


def _get_te_cuda_graph_rotary_pos_emb(
    transformer_module, transformer_input, config, packed_seq_params, rotary_pos_emb_cache=None
):
    rotary_seq_len = _get_te_cuda_graph_rotary_seq_len(
        transformer_module, transformer_input, config, packed_seq_params
    )
    if rotary_pos_emb_cache is None:
        rotary_pos_emb_cache = {}
    if rotary_seq_len not in rotary_pos_emb_cache:
        rotary_pos_emb_cache[rotary_seq_len] = transformer_module.rotary_pos_emb(
            rotary_seq_len,
            packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == "thd",
            cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
        )
    return rotary_pos_emb_cache[rotary_seq_len]
~~~

Use the precise keyword name accepted by the current rotary embedding implementation (packed_seq on the present local regression), then rerun the complete packed test file. Commit this minimal prerequisite with:

~~~
git add megatron/core/transformer/cuda_graphs.py tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git commit -s -m "fix: preserve packed RoPE CP layout in TE CUDA graphs"
~~~

### Task 2: Add the Mamba packed metadata ABI

**Files:**
- Modify: megatron/core/packed_seq_params.py.
- Test: tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py.

**Consumes:** PR5672 attention fields and Tensor-only TE graph inputs.

**Produces:** Mamba-specific split/build functions with seq_idx dynamic and total_tokens omitted.

- [x] **Step 1: Write failing ABI tests**

Append:

~~~python
def test_mamba_packed_cuda_graph_uses_seq_idx_not_total_tokens():
    from megatron.core.packed_seq_params import split_mamba_packed_seq_params_for_cuda_graph

    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 2, 5]),
        cu_seqlens_kv=torch.IntTensor([0, 2, 5]),
        cu_seqlens_q_padded=torch.IntTensor([0, 2, 5]),
        cu_seqlens_kv_padded=torch.IntTensor([0, 2, 5]),
        max_seqlen_q=3,
        max_seqlen_kv=3,
        total_tokens=5,
    )
    tensor_kwargs, static_metadata = split_mamba_packed_seq_params_for_cuda_graph(params)

    assert "_mamba_packed_seq_params_seq_idx" in tensor_kwargs
    assert tensor_kwargs["_mamba_packed_seq_params_seq_idx"] is params.seq_idx
    assert "total_tokens" not in static_metadata


def test_mamba_packed_cuda_graph_rebuild_uses_supplied_seq_idx():
    from megatron.core.packed_seq_params import (
        build_mamba_packed_seq_params_from_cuda_graph_kwargs,
        split_mamba_packed_seq_params_for_cuda_graph,
    )

    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 2, 5]),
        cu_seqlens_kv=torch.IntTensor([0, 2, 5]),
        max_seqlen_q=3,
        max_seqlen_kv=3,
        total_tokens=5,
    )
    tensor_kwargs, static_metadata = split_mamba_packed_seq_params_for_cuda_graph(params)
    kwargs = {"hidden_states": torch.ones(5, 1, 4), **tensor_kwargs}

    rebuilt = build_mamba_packed_seq_params_from_cuda_graph_kwargs(kwargs, static_metadata)

    assert rebuilt.total_tokens is None
    assert rebuilt.seq_idx is params.seq_idx
    assert set(kwargs) == {"hidden_states"}
~~~

- [x] **Step 2: Confirm red**

~~~
cd /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py -k mamba_packed_cuda_graph
~~~

Expected: FAIL on missing import.

- [x] **Step 3: Implement the smallest Mamba ABI**

Add after the existing PR5672 constants and helpers:

~~~python
MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX = "_mamba_packed_seq_params_"
MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS = (
    *PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS,
    "seq_idx",
)
MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS = (
    *PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS,
)


def split_mamba_packed_seq_params_for_cuda_graph(
    packed_seq_params: PackedSeqParams | None,
) -> tuple[dict[str, Tensor | None], dict[str, object]]:
    if packed_seq_params is None:
        return {}, {}
    tensor_kwargs = {}
    for field_name in MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS:
        value = getattr(packed_seq_params, field_name)
        if value is not None and not isinstance(value, Tensor):
            raise TypeError(
                f"PackedSeqParams.{field_name} must be a Tensor or None for Mamba CUDA graphs, got {type(value).__name__}."
            )
        if value is not None:
            tensor_kwargs[_cuda_graph_packed_seq_params_key(
                field_name, MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
            )] = value
    static_metadata = {}
    for field_name in MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS:
        value = getattr(packed_seq_params, field_name)
        if isinstance(value, Tensor):
            raise TypeError(
                f"PackedSeqParams.{field_name} is static Mamba CUDA graph metadata and must not be a Tensor."
            )
        static_metadata[field_name] = value
    return tensor_kwargs, static_metadata


def has_mamba_packed_seq_params_cuda_graph_kwargs(kwargs: Mapping[str, object]) -> bool:
    return any(
        _cuda_graph_packed_seq_params_key(
            field_name, MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
        ) in kwargs
        for field_name in MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS
    )


def build_mamba_packed_seq_params_from_cuda_graph_kwargs(
    kwargs: MutableMapping[str, object], static_metadata: Mapping[str, object] | None
) -> PackedSeqParams | None:
    params_kwargs = dict(static_metadata or {})
    found_tensor_field = False
    for field_name in MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS:
        key = _cuda_graph_packed_seq_params_key(
            field_name, MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
        )
        if key not in kwargs:
            continue
        found_tensor_field = True
        value = kwargs.pop(key)
        if value is not None and not isinstance(value, Tensor):
            raise TypeError(
                f"Flattened Mamba PackedSeqParams field {key} must be a Tensor or None, got {type(value).__name__}."
            )
        params_kwargs[field_name] = value
    if not params_kwargs and not found_tensor_field:
        return None
    return PackedSeqParams(**params_kwargs)
~~~

- [x] **Step 4: Confirm green and commit**

~~~
cd /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git add megatron/core/packed_seq_params.py tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git commit -s -m "feat: add Mamba packed CUDA graph inputs"
~~~

Expected: complete PR5672 packed-ABI suite PASS.

### Task 3: Capture and replay the Mamba ABI through TE

**Files:**
- Modify: megatron/core/transformer/cuda_graphs.py.
- Modify: megatron/core/ssm/mamba_layer.py.
- Test: tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py.

**Consumes:** Task 2 helpers and GraphableMegatronModule's existing Tensor-only assertion.

**Produces:** captured Mamba receives flattened sample tensors; replay flattens runtime params and base replay receives tensors only.

- [x] **Step 1: Write the failing sample test**

~~~python
class _MambaLayerCudaGraphStub:
    def _set_te_cuda_graph_mamba_packed_seq_params_static_metadata(
        self, static_metadata, tensor_kwarg_names
    ):
        self.static_metadata = dict(static_metadata)
        self.tensor_kwarg_names = tuple(sorted(tensor_kwarg_names))


def test_te_cuda_graph_mamba_sample_has_flattened_packed_inputs():
    from megatron.core.transformer.cuda_graphs import (
        _add_mamba_packed_seq_params_to_te_cuda_graph_sample_kwargs,
    )

    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 2, 5]),
        cu_seqlens_kv=torch.IntTensor([0, 2, 5]),
        max_seqlen_q=3,
        max_seqlen_kv=3,
        total_tokens=5,
    )
    layer = _MambaLayerCudaGraphStub()
    sample_kwargs = {"attention_mask": None}

    _add_mamba_packed_seq_params_to_te_cuda_graph_sample_kwargs(
        layer, sample_kwargs, params
    )

    assert "_mamba_packed_seq_params_seq_idx" in sample_kwargs
    assert layer.static_metadata["qkv_format"] == "thd"
    assert "total_tokens" not in layer.static_metadata


def test_mamba_replay_flattens_packed_seq_params_before_tensor_gate():
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.packed_seq_params import split_mamba_packed_seq_params_for_cuda_graph

    layer = _MambaLayerCudaGraphStub()
    layer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata = (
        MambaLayer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata.__get__(layer)
    )
    layer._flatten_te_cuda_graph_mamba_packed_seq_params = (
        MambaLayer._flatten_te_cuda_graph_mamba_packed_seq_params.__get__(layer)
    )
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 2, 5]),
        cu_seqlens_kv=torch.IntTensor([0, 2, 5]),
        max_seqlen_q=3,
        max_seqlen_kv=3,
        total_tokens=5,
    )
    tensor_kwargs, static_metadata = split_mamba_packed_seq_params_for_cuda_graph(params)
    layer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata(
        static_metadata, tensor_kwargs
    )
    kwargs = {"packed_seq_params": params}

    layer._flatten_te_cuda_graph_mamba_packed_seq_params(kwargs)

    assert "packed_seq_params" not in kwargs
    assert kwargs["_mamba_packed_seq_params_seq_idx"] is params.seq_idx
~~~

- [x] **Step 2: Confirm red**

~~~
cd /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py -k mamba_sample
~~~

Expected: FAIL on missing sample helper.

- [x] **Step 3: Add the Mamba sample helper and selection**

In cuda_graphs.py, import MambaLayer and the Task 2 helpers. Add:

~~~python
def _add_mamba_packed_seq_params_to_te_cuda_graph_sample_kwargs(
    layer, sample_kwargs, sample_packed_seq_params
):
    if sample_packed_seq_params is None:
        return
    tensor_kwargs, static_metadata = split_mamba_packed_seq_params_for_cuda_graph(
        sample_packed_seq_params
    )
    duplicate_keys = set(sample_kwargs) & set(tensor_kwargs)
    assert not duplicate_keys, (
        "Mamba PackedSeqParams CUDA graph Tensor kwargs overlap with existing sample kwargs: "
        f"{', '.join(sorted(duplicate_keys))}."
    )
    layer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata(
        static_metadata, tensor_kwargs.keys()
    )
    sample_kwargs.update(tensor_kwargs)
~~~

In TECudaGraphHelper._get_sample_arguments(), derive:

~~~python
contains_mamba = (
    isinstance(layer, MambaLayer)
    and CudaGraphModule.mamba in self.config.cuda_graph_modules
)
~~~

Then add this branch immediately after the existing contains_self_attn branch:

~~~python
elif contains_mamba:
    _add_mamba_packed_seq_params_to_te_cuda_graph_sample_kwargs(
        layer, static_inputs, self.sample_packed_seq_params
    )
~~~

- [x] **Step 4: Adapt Mamba capture and replay**

In mamba_layer.py, import the Task 2 helpers. Add the following Mamba methods; no change to GraphableMegatronModule:

~~~python
def _set_te_cuda_graph_mamba_packed_seq_params_static_metadata(
    self, static_metadata, tensor_kwarg_names
):
    self._te_cuda_graph_mamba_packed_seq_params_static_metadata = dict(static_metadata)
    self._te_cuda_graph_mamba_packed_seq_params_tensor_kwarg_names = tuple(
        sorted(tensor_kwarg_names)
    )


def _rebuild_te_cuda_graph_mamba_packed_seq_params(self, kwargs):
    if not has_mamba_packed_seq_params_cuda_graph_kwargs(kwargs):
        return
    assert kwargs.get("packed_seq_params") is None
    kwargs["packed_seq_params"] = build_mamba_packed_seq_params_from_cuda_graph_kwargs(
        kwargs, self._te_cuda_graph_mamba_packed_seq_params_static_metadata
    )


def _flatten_te_cuda_graph_mamba_packed_seq_params(self, kwargs):
    packed_seq_params = kwargs.pop("packed_seq_params", None)
    expected_static_metadata = getattr(
        self, "_te_cuda_graph_mamba_packed_seq_params_static_metadata", None
    )
    if packed_seq_params is None:
        assert expected_static_metadata is None
        return
    tensor_kwargs, static_metadata = split_mamba_packed_seq_params_for_cuda_graph(
        packed_seq_params
    )
    assert static_metadata == expected_static_metadata
    assert tuple(sorted(tensor_kwargs)) == (
        self._te_cuda_graph_mamba_packed_seq_params_tensor_kwarg_names
    )
    assert not (set(kwargs) & set(tensor_kwargs))
    kwargs.update(tensor_kwargs)


def _te_cuda_graph_capture(self, *args, **kwargs):
    self._rebuild_te_cuda_graph_mamba_packed_seq_params(kwargs)
    return self.forward(*args, **kwargs)


def _te_cuda_graph_replay(self, *args, **kwargs):
    assert kwargs.get("inference_context") is None
    self._flatten_te_cuda_graph_mamba_packed_seq_params(kwargs)
    return super()._te_cuda_graph_replay(*args, **kwargs)
~~~

Use the current TransformerLayer packed helper's descriptive errors for missing static metadata, changed tensor names, and duplicate kwarg names.

- [x] **Step 5: Confirm green, commit, and push**

~~~
cd /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git add megatron/core/transformer/cuda_graphs.py megatron/core/ssm/mamba_layer.py tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git commit -s -m "feat: support packed Mamba TE CUDA graphs"
git push -u origin experiment/pr5672-nano-packed-support-20260720
~~~

Expected: PASS and pushed immutable MCore commit.

### Task 4: Prove CP2 packed Mamba graph correctness on GPU

**Files:**
- Modify: tests/unit_tests/transformer/test_cuda_graphs.py.

**Consumes:** Task 3 Mamba graph path.

**Produces:** CP2 eager/graph parity coverage for two document-boundary layouts in one token bucket.

- [ ] **Step 1: Write a failing GPU parity test**

~~~python
def test_packed_mamba_te_cuda_graph_replay_matches_eager():
    eager_output, eager_grads = _run_packed_mamba(
        cuda_graph_impl="none", packed_lengths=[8, 24]
    )
    graph_output, graph_grads = _run_packed_mamba(
        cuda_graph_impl="transformer_engine", packed_lengths=[8, 24]
    )
    replay_output, replay_grads = _run_packed_mamba(
        cuda_graph_impl="transformer_engine", packed_lengths=[12, 20]
    )

    torch.testing.assert_close(graph_output, eager_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(graph_grads, eager_grads, rtol=1e-3, atol=1e-3)
    assert replay_output.shape == graph_output.shape
    assert all(torch.isfinite(grad).all() for grad in replay_grads)
~~~

The helper must use CP2, cuda_graph_modules=[CudaGraphModule.mamba], total_tokens=32, and real PackedSeqParams. It invokes TECudaGraphHelper with the first packed sample before the replay invocation.

- [ ] **Step 2: Run the focused GPU parity test**

~~~
cd /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720
uv run python -m torch.distributed.run --nproc-per-node 2 -m pytest -q tests/unit_tests/transformer/test_cuda_graphs.py -k packed_mamba_te_cuda_graph_replay_matches_eager
~~~

Expected: PASS. The prior Nano job established the baseline failure before Task 3; if a later TE/Mamba CP operation is capture-unsafe, preserve that exact failure and stop Mamba promotion rather than changing the test to CP1.

- [ ] **Step 3: Commit GPU coverage**

~~~
git -C /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720 add tests/unit_tests/transformer/test_cuda_graphs.py
git -C /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720 commit -s -m "test: cover packed Mamba TE graph replay"
git -C /Users/sna/CudaGraph_PR/Megatron-LM-pr5672-nano-extension-20260720 push
~~~

### Task 5: Preflight Nano unsupported scopes and log provenance

**Files:**
- Modify: nemo_rl/models/megatron/setup.py.
- Modify: nemo_rl/models/policy/workers/megatron_policy_worker.py.
- Test: tests/unit/models/megatron/test_megatron_setup.py.

**Consumes:** Nano CP2 packed attention failure, checkpoint-derived FP64 router dtype, and MCore normalized cuda_graph_modules.

**Produces:** failure before model construction and a requested/effective scope log line.

- [ ] **Step 1: Write failing validation tests**

~~~python
def test_nanov3_packed_cp_attention_scope_is_rejected_before_capture():
    from nemo_rl.models.megatron.setup import _enforce_packed_seq_cuda_graph_consistency

    config = {
        "model_name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
        "sequence_packing": {"enabled": True},
        "megatron_cfg": {
            "cuda_graph_impl": "transformer_engine",
            "cuda_graph_scope": ["attn"],
            "cuda_graph_packed_seq": True,
            "context_parallel_size": 2,
        },
    }
    with pytest.raises(ValueError, match="Nano packed CP attention"):
        _enforce_packed_seq_cuda_graph_consistency(config)


def test_te_fp64_router_scope_is_rejected_before_graph_creation():
    from nemo_rl.models.megatron.setup import _validate_te_cuda_graph_model_scope

    model_cfg = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=["moe_router"],
        moe_router_dtype="fp64",
    )
    with pytest.raises(ValueError, match="FP64 MoE router"):
        _validate_te_cuda_graph_model_scope(model_cfg)
~~~

- [ ] **Step 2: Confirm red**

~~~
cd /Users/sna/CudaGraph_PR/RL-pr5672-nano-extension-20260720
uv run --group test pytest -q tests/unit/models/megatron/test_megatron_setup.py -k 'nanov3_packed_cp_attention_scope or te_fp64_router_scope'
~~~

Expected: FAIL because neither guard exists.

- [ ] **Step 3: Implement deterministic preflight**

Add these helpers to setup.py and make _cuda_graph_scope_includes_attention call _cuda_graph_scope_values:

~~~python
def _cuda_graph_scope_values(scope: object) -> set[str]:
    if scope in (None, "", [], "full"):
        return {"full"}
    values = scope if isinstance(scope, list) else str(scope).split(",")
    return {str(value).strip() for value in values if str(value).strip()}


def _is_nanov3_model(config: PolicyConfig) -> bool:
    return str(config.get("model_name", "")).lower().startswith(
        "nvidia/nvidia-nemotron-3-nano-"
    )


def _validate_te_cuda_graph_model_scope(model_cfg: Any) -> None:
    graph_modules = {str(module.value if hasattr(module, "value") else module)
                     for module in model_cfg.cuda_graph_modules}
    if (
        model_cfg.cuda_graph_impl == "transformer_engine"
        and {"moe_router", "moe_preprocess"} & graph_modules
        and model_cfg.moe_router_dtype == "fp64"
    ):
        raise ValueError(
            "FP64 MoE router CUDA graphs are unsupported by the installed Transformer Engine; "
            "keep the router eager or use a separately labelled FP32 diagnostic recipe."
        )
~~~

At the end of _enforce_packed_seq_cuda_graph_consistency(), after packing is enabled, add:

~~~python
if (
    _is_nanov3_model(config)
    and megatron_cfg.get("cuda_graph_impl") == "transformer_engine"
    and megatron_cfg.get("context_parallel_size", 1) > 1
    and _cuda_graph_scope_includes_attention(megatron_cfg.get("cuda_graph_scope"))
):
    raise ValueError(
        "Nano packed CP attention CUDA graphs are unsupported by the installed Transformer "
        "Engine. Remove 'attn' from policy.megatron_cfg.cuda_graph_scope until the TE "
        "packed-CP backward capture fix is available."
    )
~~~

Call _validate_te_cuda_graph_model_scope(model_cfg) immediately after model_cfg.__post_init__() in setup_model_config().

- [ ] **Step 4: Add requested/effective logging**

Immediately before validate_and_set_config(), save the raw request. After self.megatron_cfg = runtime_config.megatron_cfg, emit:

~~~python
requested_cuda_graph_scope = config["megatron_cfg"].get("cuda_graph_scope")
effective_cuda_graph_scope = [
    module.value if hasattr(module, "value") else str(module)
    for module in self.megatron_cfg.model.cuda_graph_modules
]
if self.rank == 0:
    logger.info(
        "CUDA graph scope requested=%s effective=%s impl=%s",
        requested_cuda_graph_scope,
        effective_cuda_graph_scope,
        self.megatron_cfg.model.cuda_graph_impl,
    )
~~~

- [ ] **Step 5: Confirm green and commit**

~~~
cd /Users/sna/CudaGraph_PR/RL-pr5672-nano-extension-20260720
uv run --group test pytest -q tests/unit/models/megatron/test_megatron_setup.py tests/unit/experiments/test_nanov3_cuda_graph_launcher.py
git add nemo_rl/models/megatron/setup.py nemo_rl/models/policy/workers/megatron_policy_worker.py tests/unit/models/megatron/test_megatron_setup.py
git commit -s -m "fix: preflight unsupported Nano CUDA graph scopes"
~~~

Expected: PASS. Existing Qwen attention tests remain valid.

### Task 6: Pin MCore, validate the Nano launcher, and run experiments

**Files:**
- Modify: 3rdparty/Megatron-LM-workspace/Megatron-LM.
- Modify: experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh.
- Test: tests/unit/experiments/test_nanov3_cuda_graph_launcher.py.

**Consumes:** pushed MCore branch and Task 5 preflight.

**Produces:** provenance-correct Nano jobs and a matched no-CG/Mamba evaluation matrix.

- [ ] **Step 1: Write failing launcher tests**

~~~python
def test_nanov3_launcher_rejects_dense_mlp_scope():
    result = _run_launcher("mlp")
    assert result.returncode == 2
    assert "Nano has no dense MLP-only layers" in result.stderr


def test_nanov3_launcher_records_requested_scope():
    result = _run_launcher("mamba")
    assert result.returncode == 0, result.stderr
    assert "requested_scope=mamba" in result.stdout
    assert "cuda_graph_warmup_steps=3" in result.stdout
~~~

- [ ] **Step 2: Confirm red, implement launcher rules, and confirm green**

Replace the mlp case with:

~~~bash
  mlp)
    echo 'Nano has no dense MLP-only layers; use mamba or a supported MoE diagnostic scope' >&2
    exit 2
    ;;
~~~

After the scope case, add:

~~~bash
echo "requested_scope=$(printf %s "$SCOPE_CASE")"
~~~

Then run:

~~~
cd /Users/sna/CudaGraph_PR/RL-pr5672-nano-extension-20260720
python3 -m pytest -q tests/unit/experiments/test_nanov3_cuda_graph_launcher.py
~~~

Expected: PASS. Keep attn and router launcher cases so NeMo's explicit preflight reason is tested without hiding it in shell code.

- [ ] **Step 3: Pin MCore and push the integration branch**

~~~
cd /Users/sna/CudaGraph_PR/RL-pr5672-nano-extension-20260720
git submodule update --init --recursive
git -C 3rdparty/Megatron-LM-workspace/Megatron-LM fetch origin experiment/pr5672-nano-packed-support-20260720
git -C 3rdparty/Megatron-LM-workspace/Megatron-LM checkout --detach origin/experiment/pr5672-nano-packed-support-20260720
git add 3rdparty/Megatron-LM-workspace/Megatron-LM experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh tests/unit/experiments/test_nanov3_cuda_graph_launcher.py
git commit -s -m "feat: integrate Nano packed CUDA graph validation"
git push seonjinn HEAD:experiment/pr5672-nano-packed-support-20260720
~~~

- [ ] **Step 4: Submit in validation order on Ptyche**

Run git pull, recursive submodule initialization, and scheduler --test-only in the fresh remote worktree. Submit no-CG five steps, then Mamba five steps; observe each for five minutes. Run attn and moe-router only as preflight checks: they must fail before make_graphed_callables. Do not schedule a 20-step rejected scope.

After Mamba smoke passes, submit matched jobs with the same seed, Nano topology, max packed sequences 512, bucket [8192], warmup 3, and checkpointing disabled:

~~~bash
SCOPE_CASE=nocg STEPS=20 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh
SCOPE_CASE=mamba STEPS=20 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh
SCOPE_CASE=nocg STEPS=40 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh
SCOPE_CASE=mamba STEPS=40 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh
~~~

For post-warmup steps, report median E2E, generation, logprob, and policy-training time plus tokens/s/GPU. For 40 steps compare reward, accuracy, policy loss, KL, clip ratio, NaN/invalid count, and completion rate. Do not present FP32-router diagnostics as production convergence evidence.
