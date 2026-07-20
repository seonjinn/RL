# Qwen3-235B TE Packed CUDA Graph Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, fair Qwen3-235B-A22B baseline-versus-PR5672 Transformer Engine packed-attention CUDA-graph experiment on Ptyche, then validate safety and performance.

**Architecture:** Two overlays inherit the native 16n4g Qwen3-235B recipe: a no-CG baseline and a PR5672 TE-attention treatment. They have matching sequence packing, FP64 router, disabled training checkpointing, and one converted Megatron cache. A launcher accepts only these conditions and dry-runs unless `SUBMIT=1`.

**Tech Stack:** Hydra/OmegaConf, NeMo-RL, Megatron-Core `bed605f292f926090f5f43ba5e30fb024c2306dc`, Transformer Engine, Slurm/Ray on Ptyche.

## Global Constraints

- Start on `grpo-qwen3-235b-16n4g.yaml`: 16 nodes x 4 B200 GPUs, PP4 and HybridEP.
- Exclude 16n8g, 32n4g, MXFP8-rollout, and async overlays.
- Graph attention only. Router, dispatch, and experts stay eager with the inherited FP64 router.
- The treatment uses warmup 3, bucket 8192, and Nmax 512; both recipes disable training checkpoints.
- Use one adapter worktree and one conversion-cache namespace. The first 20-step pair is performance-only.

---

## Task 1: Create matched Qwen3-235B recipes and a safe launcher

**Files:**

- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-nocg-adapter.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-cg-attn-adapter.yaml`
- Create: `experiments/cuda_graph/launch_qwen235_cg_comparison_ptyche.sh`
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`

**Produces:** `adapter-nocg` and `adapter-attn` with identical 235B semantics apart from CUDA Graph.

- [ ] **Step 1: Write the failing resolved-config test**

Use `register_omegaconf_resolvers()`, `load_config`, and `OmegaConf.resolve` to load both new recipes. Assert both resolve to 16 nodes, 4 GPUs/node, segment 16, FP64 router, disabled checkpointing, and matching packing:

```python
for field in ("enabled", "train_mb_tokens", "logprob_mb_tokens",
              "algorithm", "sequence_length_round"):
    assert no_cg.policy.sequence_packing[field] == attention.policy.sequence_packing[field]
assert no_cg.policy.megatron_cfg.cuda_graph_impl == "none"
assert attention.policy.megatron_cfg.cuda_graph_impl == "transformer_engine"
assert attention.policy.megatron_cfg.cuda_graph_scope == "attn"
assert attention.policy.megatron_cfg.cuda_graph_pr5672_thd is True
assert attention.policy.megatron_cfg.cuda_graph_packed_seq is True
assert attention.policy.megatron_cfg.cuda_graph_warmup_steps == 3
assert attention.policy.megatron_cfg.cuda_graph_max_packed_seqs == 512
assert list(attention.policy.megatron_cfg.cuda_graph_buckets) == [8192]
```

- [ ] **Step 2: Confirm the test is red**

Run:

```bash
python3 -c 'from pathlib import Path; assert not (Path("examples/configs/recipes/llm/performance") / "grpo-qwen3-235b-16n4g-nocg-adapter.yaml").exists()'
```

Expected: success before implementation. The locked pytest is intentionally deferred to Ptyche because the lockfile supports Linux only.

- [ ] **Step 3: Add the recipe overlays**

Both overlays inherit `./grpo-qwen3-235b-16n4g.yaml`, have unique result/log names, set `checkpointing.enabled: false`, and explicitly retain this packing contract:

```yaml
policy:
  sequence_packing:
    enabled: true
    train_mb_tokens: 8192
    logprob_mb_tokens: 8192
    algorithm: modified_first_fit_decreasing
    sequence_length_round: 64
```

The baseline additionally sets:

```yaml
policy:
  megatron_cfg:
    cuda_graph_impl: none
```

The attention overlay instead sets:

```yaml
policy:
  megatron_cfg:
    cuda_graph_impl: transformer_engine
    cuda_graph_scope: attn
    cuda_graph_warmup_steps: 3
    cuda_graph_packed_seq: true
    cuda_graph_pr5672_thd: true
    cuda_graph_max_packed_seqs: 512
    cuda_graph_buckets:
    - 8192
```

- [ ] **Step 4: Add the Ptyche launcher**

Accept only `CONDITION=adapter-nocg|adapter-attn`; default to `STEPS=20`; use the adapter worktree, `GPUS_PER_NODE=4`, `--nodes=16 --segment=16 --exclusive`, account `coreai_dlalgo_llm`, and partition `batch`. Set one cache root:

```bash
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/qwen3-235b-a22b-adapter-20260720"
CHECKPOINT_READY_FILE="${CHECKPOINT_DIR}/Qwen/Qwen3-235B-A22B/iter_0000000/run_config.yaml"
```

Require the ready file before attention but allow no-CG to create it. Check the HF token, worktree, and config. Export direct-MCore-first `PYTHONPATH`, `NRL_MEGATRON_CHECKPOINT_DIR`, and HF caches. Override only steps, validation period, telemetry, log directory, and W&B name. Use `--time=04:00:00`, which fits Ptyche batch's five-hour maximum for the 20-step pair. Run `sbatch --test-only` first; require `SUBMIT=1` for `q235` submission. Do not create dependencies or enable checkpoint saves.

- [ ] **Step 5: Verify and commit**

Run:

```bash
ruff check tests/unit/models/megatron/test_megatron_setup.py
ruff format --check tests/unit/models/megatron/test_megatron_setup.py
bash -n experiments/cuda_graph/launch_qwen235_cg_comparison_ptyche.sh
git diff --check
git add examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-nocg-adapter.yaml \
  examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-cg-attn-adapter.yaml \
  experiments/cuda_graph/launch_qwen235_cg_comparison_ptyche.sh \
  tests/unit/models/megatron/test_megatron_setup.py
git commit -s -m "test: add Qwen3 235B packed graph comparison"
```

Expected: static checks pass; execute locked pytest on Ptyche in Task 2.

## Task 1.5: Preserve configured CUDA-graph buckets with pipeline parallelism

**Files:**

- Modify: `nemo_rl/models/megatron/data.py`
- Modify: `tests/unit/models/megatron/test_megatron_data.py`

**Consumes:** The Qwen3-235B treatment's PP4 configuration, `cuda_graph_buckets: [8192]`, and the worker's requirement that the packed sequence length equal an active bucket before capture/replay.

**Produces:** A PP>1 packed CUDA-graph step pads an eligible non-bucket batch to the selected configured bucket rather than silently disabling graph replay.

- [ ] **Step 1: Write the failing PP+bucket test**

Directly exercise `_get_pack_sequence_parameters_for_megatron` with pipeline model parallel size 4, CUDA-graph packed padding enabled, `cuda_graph_buckets: [8192]`, and a natural `max_seq_len_in_batch` such as 4096. Assert the returned packed target is 8192. Add a second case with a minimum fill ratio that rejects the 8192 bucket and assert the PP fallback target remains the natural 4096 length.

- [ ] **Step 2: Confirm RED**

Run the focused test. Before the fix, the eligible PP case returns 4096 because the PP branch preempts `_select_cuda_graph_bucket`.

```bash
uv run --locked --extra mcore pytest \
  tests/unit/models/megatron/test_megatron_data.py -k pp_cuda_graph_bucket -q
```

- [ ] **Step 3: Select a bucket before PP fallback**

In `_get_pack_sequence_parameters_for_megatron`, retain the fixed-shape PP invariant but, when `cuda_graph_pad_packed_seq` and non-empty `cuda_graph_buckets` are configured, call `_select_cuda_graph_bucket(max_seq_len_in_batch, cuda_graph_buckets, min_fill_ratio)`. If it returns a bucket, set `pad_packed_seq_to` to that bucket and set `is_cg_step=True`; otherwise retain the PP natural-length padding and set `is_cg_step=False`. Do not alter the PR5783 special path, non-PP behavior, or no-bucket PP behavior.

- [ ] **Step 4: Verify and commit**

Run the focused test and the existing packed-data tests, then:

```bash
ruff check nemo_rl/models/megatron/data.py tests/unit/models/megatron/test_megatron_data.py
ruff format --check nemo_rl/models/megatron/data.py tests/unit/models/megatron/test_megatron_data.py
git diff --check
git add nemo_rl/models/megatron/data.py tests/unit/models/megatron/test_megatron_data.py
git commit -s -m "fix: honor CUDA graph buckets with pipeline parallelism"
```

## Task 2: Stage, smoke-test, and benchmark on Ptyche

**Files:**

- Create: `experiments/cuda_graph/report/qwen3_235b_te_packed_adapter_20260720.md`
- Read: `experiments/cuda_graph/logs/<job>-logs/driver_command.sh`
- Read: `experiments/cuda_graph/logs/<job>-logs/ray-driver-*`

**Consumes:** Task 1 pushed to `seonjinn/experiment/te-packed-cg-adapter-20260719` and a valid Ptyche Kerberos ticket.

**Produces:** Safety evidence and a 20-step no-CG/TE-attention comparison, or a precise external blocker.

- [ ] **Step 1: Preflight and stage one fresh worktree**

Verify `klist -s`, Slurm access, Qwen3-235B HF snapshot, OpenMath access/cache, container, and conversion-cache path. Stage the exact fork branch in the fresh adapter worktree, then run `git pull --ff-only`, `git submodule update --init --recursive`, `uv lock --check`, and a Python signature check that `TECudaGraphHelper.__init__` exposes `sample_packed_seq_params`.

Run:

```bash
uv run --locked --extra mcore pytest \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_train.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/megatron/test_megatron_setup.py -q
```

- [ ] **Step 2: Run conversion/no-CG and a blocking diagnostic smoke**

Run `adapter-nocg` long enough to create the conversion `run_config.yaml`. Then run five-step `CUDA_LAUNCH_BLOCKING=1 adapter-attn`; verify three warmups, capture, refit/offload, reload and replay without illegal memory access. Remove or disable training checkpoint artifacts if a regression attempts to save them.

- [ ] **Step 3: Run the matched performance pair**

Run 20-step `adapter-nocg` and `adapter-attn` with identical seed, model/data snapshot, topology, validation cadence, packing values, and conversion cache. Do not add Slurm dependencies. Exclude validation plus warmup/capture from the steady-state window. Record E2E, policy, logprob, and generation timings/throughputs, and the high-water real packed sequence count across train/logprob/validation.

- [ ] **Step 4: Report and commit**

Record commit/MCore SHA, container, topology, router dtype, packing, Nmax, bucket, included steps, jobs/states, preflight, safety, and metrics. Label the 20-step pair performance-only. Commit the report signed and push.

## Plan Self-Review

- Spec coverage: Task 1 enforces a single controlled 235B A/B comparison; Task 2 handles authentication-dependent staging, safety, performance, and reporting.
- Scope: no full-layer/MoE graph, MXFP8, async, 16n8g/32n4g, or 235B accuracy claim.
- Reproducibility: resolved config test enforces identical workload semantics outside the graph settings.
- Platform: locked tests and Slurm work happen only on Ptyche after Kerberos preflight.
