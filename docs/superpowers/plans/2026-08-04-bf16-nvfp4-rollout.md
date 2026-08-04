# BF16 Training with NVFP4 Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a plain BF16 Megatron policy with real W4A16 or W4A4 NVFP4 vLLM rollout through legacy refit and NCCL-Reshard.

**Architecture:** The ordinary `MegatronPolicyWorker` remains the BF16 source. The vLLM ModelOpt generation extension classifies incoming manifests and serializes eligible BF16 weights with the pinned Megatron-Bridge NVFP4 exporter before entering vLLM's existing layerwise reload lifecycle. W4A4 reads fixed input `amax` from a provenance-checked calibration artifact. NCCL-Reshard adds ordered multi-component transforms and group/model finalization without putting quantization into the trainer.

**Tech Stack:** Python 3.12, PyTorch, ModelOpt, Megatron-Bridge, vLLM 0.25.1, safetensors, Hydra-style YAML configs, pytest, Ray, NCCL-Reshard/xferdtensor, SLURM on GCP-NRT B200.

## Global Constraints

- Base branch is `upstream/main` commit `aa5c74edec9657f3aca3c570255970e24fa6768d` in `/Users/sna/MXFP8_generation/.worktrees/nemorl-bf16-nvfp4-rollout`.
- `policy.quant_cfg` stays `null`; policy training, optimizer, checkpoint, forward, and backward remain BF16.
- Do not select or modify `MegatronQuantPolicyWorker` for the rollout-only recipes.
- W4A16 and W4A4 generation use real ModelOpt NVFP4 kernels, not fake quantization.
- W4A4 must fail without a provenance-checked input-`amax` artifact; dummy calibration and silent W4A16 fallback are forbidden.
- The rollout recipes set `loss_fn.force_on_policy_ratio: false` and `loss_fn.use_importance_sampling_correction: true`.
- Use Megatron-Bridge's pinned `QuantMeta`, `get_modelopt_quant_exporter()`, `quantize_nvfp4_weight()`, and `compute_nvfp4_input_scale()` APIs; do not hand-roll FP4 nibble packing or scale formulas.
- Preserve the existing prepacked QARL W4A16/W4A4 path and its tests.
- vLLM expert parallelism remains unsupported for fused ModelOpt MoE refit.
- Add type hints to every new function and use project configuration and testing conventions.
- Follow red-green-refactor for every production behavior and commit with DCO sign-off.
- GPU jobs use containers, committed code, experiment directories, `batch`, four hours, and five minutes of early monitoring.

---

### Task 1: Rollout-Only Recipe Contract

**Files:**
- Create: `tests/test_nvfp4_rollout_recipes.py`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.yaml`

**Interfaces:**
- Consumes: existing `load_config()` recipe composition and the Qwen3-30B-A3B four-node performance recipe.
- Produces: two configs with plain BF16 policy selection and real-quant vLLM generation selection.

- [ ] **Step 1: Write failing recipe tests**

Add parametrized tests that load both recipes and assert:

```python
assert config["policy"]["quant_cfg"] is None
assert config["loss_fn"]["force_on_policy_ratio"] is False
assert config["loss_fn"]["use_importance_sampling_correction"] is True
assert config["policy"]["generation"]["real_quant"] is True
assert config["policy"]["generation"]["quant_cfg"] == expected_quant_cfg
assert config["cluster"]["num_nodes"] == 4
assert config["cluster"]["gpus_per_node"] == 4
```

Also call `resolve_policy_worker_cls()` and `resolve_generation_worker_cls()` with the composed sections and assert plain Megatron policy plus `VllmQuantGenerationWorker`.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_nvfp4_rollout_recipes.py -q
```

Expected: both recipe paths are missing.

- [ ] **Step 3: Add the recipes**

W4A16 uses:

```yaml
defaults: ./grpo-qwen3-30ba3b-4n4g.yaml
loss_fn:
  force_on_policy_ratio: false
  use_importance_sampling_correction: true
checkpointing:
  checkpoint_dir: results/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout
policy:
  quant_cfg: null
  generation:
    quant_cfg: examples/modelopt/quant_configs/nvfp4_experts_weightonly.yaml
    real_quant: true
    real_quant_ignore: &nvfp4_rollout_ignore
      - lm_head
      - '*output_layer*'
      - '*mlp.gate*'
      - '*router*'
      - '*self_attention*'
      - '*self_attn*'
logger:
  log_dir: logs/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout
  wandb:
    project: nemo-rl
    name: grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout
```

W4A4 uses `nvfp4_experts.yaml`, distinct result/log names, and:

```yaml
policy:
  generation:
    real_quant_calibration_path: null
```

The null default makes the checked-in recipe composable while setup validation requires a CLI override before execution.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 test and the existing MXFP8 recipe test module. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_nvfp4_rollout_recipes.py examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.yaml examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.yaml
git commit -s -m "test(perf): add BF16 to NVFP4 rollout recipes"
```

### Task 2: Canonical BF16-to-NVFP4 Serializer

**Files:**
- Create: `nemo_rl/modelopt/models/generation/nvfp4_refit.py`
- Create: `tests/unit/models/generation/test_nvfp4_refit.py`

**Interfaces:**
- Produces: `NVFP4RefitMode`, `NVFP4Calibration`, `nvfp4_refit_group()`, and `serialize_bf16_nvfp4_group()`.
- Consumes: Megatron-Bridge's canonical `QuantMeta` and NVFP4 exporter.

- [ ] **Step 1: Write failing serializer tests**

Define the desired API in tests:

```python
NVFP4RefitMode = Literal["w4a16", "w4a4"]

@dataclass(frozen=True)
class NVFP4Calibration:
    input_amax: Mapping[str, torch.Tensor]

def nvfp4_refit_group(name: str) -> tuple[str, tuple[str, ...]]: ...

def serialize_bf16_nvfp4_group(
    tensors: Mapping[str, torch.Tensor],
    *,
    mode: NVFP4RefitMode,
    calibration: NVFP4Calibration | None,
) -> list[tuple[str, torch.Tensor]]: ...
```

Tests cover singleton down projection, paired expert gate/up, ignored-name pass-through exclusion, exact output names, `uint8` packed values, E4M3 block scales, FP32 global scales, and absence/presence of input scale by mode. Gate/up siblings must use a shared maximum weight `amax`; W4A4 must use the matching calibration `input_amax` and reject missing, zero, NaN, or infinite values.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/unit/models/generation/test_nvfp4_refit.py -q
```

Expected: import failure for the missing module.

- [ ] **Step 3: Implement grouping and canonical serialization**

Use `get_modelopt_quant_exporter("w4a16_nvfp4")` or `get_modelopt_quant_exporter("nvfp4")` to obtain the pinned qformat. Compute one positive finite group `weight_amax`, construct `QuantMeta(qformat=qformat, block_size=16, weight_amax=shared_amax, input_amax=...)`, and call `quantize_nvfp4_weight()` for every logical tensor. Reject groups whose K dimension is not divisible by 16.

The group function returns:

```text
...experts.{expert_id}.gate_proj.weight -> (...experts.{expert_id}.w13, (gate, up))
...experts.{expert_id}.up_proj.weight   -> (...experts.{expert_id}.w13, (gate, up))
...experts.{expert_id}.down_proj.weight -> (...experts.{expert_id}.w2, (down,))
other eligible 2-D weight                -> (name, (name,))
```

The `w13`/`w2` value above is an internal staging and completeness key, not an
emitted checkpoint tensor name. Serialization calls the canonical exporter for
each original logical HF name and emits complete per-expert
`gate_proj`/`up_proj`/`down_proj` families. Task 3 sends those receiver-generated
names directly to vLLM's standard expert loader; only the existing prepacked
QARL path uses fused `w13_weight`/`w2_weight` tensors and
`_batch_fused_modelopt_moe_weights()`.

- [ ] **Step 4: Verify GREEN and parity**

Run the new tests plus Megatron-Bridge's ModelOpt conversion tests that exercise `quantize_nvfp4_weight`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/modelopt/models/generation/nvfp4_refit.py tests/unit/models/generation/test_nvfp4_refit.py
git commit -s -m "feat(modelopt): serialize BF16 weights for NVFP4 refit"
```

### Task 3: Legacy W4A16 Receiver Refit

**Files:**
- Modify: `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`
- Modify: `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`

**Interfaces:**
- Consumes: Task 2 grouping and serializer.
- Produces: BF16-manifest classification and staged W4A16 receiver loading without changing the prepacked QARL path.

- [ ] **Step 1: Write failing source-classification and reload tests**

Add tests proving:

- all eligible entries with BF16 `.weight` tensors classify as `bf16`;
- complete packed weight/scale families classify as `modelopt`;
- mixed BF16/packed manifests fail before loading;
- gate/up split across two `_load_weights()` calls are cloned into owned staging and loaded only when complete;
- completion calls the real-quant layerwise finalizer once;
- a second refit preserves runtime parameter and kernel identity;
- ignored tensors pass through unchanged.

- [ ] **Step 2: Verify RED**

Run the exact new test names. Expected: the backend still treats every real-quant payload as prepacked.

- [ ] **Step 3: Implement BF16 manifest handling**

In `prepare_refit_info()`, record `self._nrl_real_quant_source` and initialize an empty owned staging map for BF16 mode. In `_load_weights()`, filter ignored tensors, clone incomplete group members before the transport can reuse its buffer, serialize each complete group, and delegate the emitted checkpoint-layout family to `super()._load_weights()`. Before the lifecycle finalizer runs, reject any incomplete groups with their missing names.

Keep `_batch_fused_modelopt_moe_weights()` only for prepacked QARL payloads. Receiver-generated per-expert names must flow directly through vLLM's standard expert weight loader.

- [ ] **Step 4: Verify GREEN and existing QARL compatibility**

Run all tests in `test_vllm_modelopt_real_quant_config.py` and the new serializer tests. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/modelopt/models/generation/vllm_quant_backend.py tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
git commit -s -m "feat(modelopt): load BF16 refits into W4A16 rollout"
```

### Task 4: W4A4 Calibration Artifact

**Files:**
- Create: `nemo_rl/modelopt/calibration_artifact.py`
- Create: `examples/modelopt/export_nvfp4_calibration.py`
- Modify: `nemo_rl/models/generation/vllm/config.py`
- Modify: `nemo_rl/modelopt/models/generation/vllm_quant_worker.py`
- Modify: `tests/unit/models/generation/test_nvfp4_refit.py`
- Create: `tests/unit/modelopt/test_calibration_artifact.py`

**Interfaces:**
- Produces: `save_nvfp4_calibration()`, `load_nvfp4_calibration()`, and `real_quant_calibration_path` configuration.
- Artifact format: safetensors tensors named by HF projection plus JSON metadata under `model_id`, `model_revision`, `quant_cfg`, `dataset`, `sample_count`, `sequence_length`, and `seed`.

- [ ] **Step 1: Write failing artifact tests**

Use temporary safetensors files to assert exact round-trip tensor names and metadata. Reject missing metadata, duplicate normalized names, unexpected model/config identity, empty tensors, and nonpositive/nonfinite `amax`. Add config propagation tests proving a configured artifact becomes an absolute worker-visible path and W4A16 ignores it. Source-aware enforcement is deferred to Task 5 because only `prepare_refit_info()` can distinguish a BF16 manifest from prepacked QARL W4A4.

- [ ] **Step 2: Verify RED**

Run the new artifact tests. Expected: missing module and config key.

- [ ] **Step 3: Implement artifact I/O and validation**

Keep safetensors parsing in `calibration_artifact.py`. Return the Task 2 `NVFP4Calibration` object only after model id, revision, and quant config match. Thread the resolved artifact path through `_configure_quant_engine_kwargs()` as an absolute worker environment value. Task 5 loads it once in the vLLM extension after `prepare_refit_info()` classifies the source manifest, requires it only for BF16-source W4A4, and leaves prepacked QARL W4A4 unchanged.

- [ ] **Step 4: Implement the standalone exporter**

The CLI accepts exact flags:

```text
--model
--model-revision
--quant-cfg
--dataset
--sample-count
--sequence-length
--seed
--output
```

It initializes a temporary ModelOpt calibration model, executes the existing calibration utilities on real data, gathers named enabled input quantizer `amax` values, writes the artifact, validates it by reopening, and exits without constructing an optimizer or training checkpoint.

- [ ] **Step 5: Verify GREEN**

Run artifact, config, and existing quant-worker tests. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/modelopt/calibration_artifact.py examples/modelopt/export_nvfp4_calibration.py nemo_rl/models/generation/vllm/config.py nemo_rl/modelopt/models/generation/vllm_quant_worker.py tests/unit/modelopt/test_calibration_artifact.py tests/unit/models/generation/test_nvfp4_refit.py
git commit -s -m "feat(modelopt): add W4A4 rollout calibration artifacts"
```

### Task 5: Legacy W4A4 Receiver Refit

**Files:**
- Modify: `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`
- Modify: `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`

**Interfaces:**
- Consumes: Task 4 validated calibration and Task 2 W4A4 serializer.
- Produces: repeated BF16-to-W4A4 refit with fixed destination input scales.

- [ ] **Step 1: Write failing W4A4 lifecycle tests**

Test a complete gated expert family, exact `[E, 2]` W13 and `[E]` W2 input-scale layout, fixed input-scale values across two weight refits, missing calibration keys, and successful ModelOpt finalization with stable runtime tensor addresses.

- [ ] **Step 2: Verify RED**

Run the new W4A4 tests. Expected: BF16 source mode has no calibration wired into serialization.

- [ ] **Step 3: Load fixed scales and serialize W4A4 groups**

Select Task 2 mode `w4a4`, resolve each group's input `amax` from the loaded artifact, emit the canonical input-scale names once into destination checkpoint state, and exclude static input scales from subsequent per-step transport payload requirements. Preserve QARL behavior, where input scales continue to arrive from the actor each refit.

- [ ] **Step 4: Verify GREEN**

Run the full ModelOpt generation unit module, artifact tests, and serializer tests. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/modelopt/models/generation/vllm_quant_backend.py tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
git commit -s -m "feat(modelopt): load BF16 refits into W4A4 rollout"
```

### Task 6: Multi-Component NCCL Transform Plan

**Files:**
- Modify: `nemo_rl/weight_sync/refit_transforms.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/weight_sync/test_refit_transforms.py`
- Modify: `tests/unit/weight_sync/test_nccl_reshard_utils.py`

**Interfaces:**
- Produces: separate wire and destination component descriptions, completion keys, and model finalization scope.
- Consumes: BF16 policy source metadata and Task 2 NVFP4 component descriptions.

- [ ] **Step 1: Write failing transform-plan tests**

Specify one BF16 `weight` wire component for receiver conversion. W4A16 destination components are `weight`, `weight_scale`, and `weight_scale_2`. W4A4 adds `input_scale` with `source="calibration"`; it is destination state rather than NCCL payload. Assert role-aware destination placement: packed values and block scales follow projection sharding, while global and input scales are replicated. Assert gate/up share one W13 completion key and W2 has another.

- [ ] **Step 2: Verify RED**

Run focused weight-sync tests. Expected: current single-buffer `RefitCtx` and parameter-only finalization cannot represent the plan.

- [ ] **Step 3: Generalize the transform contract**

Use these typed fields so the transfer representation is not confused with the
post-transform checkpoint representation:

```python
@dataclass(frozen=True)
class TransformComponentSpec:
    role: str
    global_shape: tuple[int, ...]
    dtype_name: str

@dataclass(frozen=True)
class DestinationComponentSpec:
    role: str
    global_shape: tuple[int, ...]
    dtype_name: str
    source: Literal["codec", "calibration"]

@dataclass
class RefitCtx:
    transfer_tensors: tuple[torch.Tensor, ...]
    extra: dict[str, Any]

@dataclass(frozen=True)
class RefitTransformPlan:
    transform_id: str
    wire_components: tuple[TransformComponentSpec, ...]
    destination_components: tuple[DestinationComponentSpec, ...]
    completion_key: str
    finalize_scope: Literal["parameter", "model"]
```

Retain compatibility access for identity and existing MXFP8 transforms while
converting their internals to explicit wire and destination tuples. Extend
serialization/agreement validation to compare every component field,
completion key, and finalization scope.

- [ ] **Step 4: Make policy metadata source-consistent**

For rollout-only NVFP4, metadata describes raw BF16 source shards and destination transform intent; it must not claim packed NVFP4 shapes while `_iter_local_hf_param_shards()` yields BF16. Preserve gate/up as one logical completion group through expert grouping.

- [ ] **Step 5: Verify GREEN and compatibility**

Run all weight-sync tests, including BF16 identity, matching blockwise FP8, BF16-to-MXFP8, and the new NVFP4 plans. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/weight_sync/refit_transforms.py nemo_rl/weight_sync/nccl_reshard_utils.py nemo_rl/models/policy/workers/megatron_policy_worker.py tests/unit/weight_sync/test_refit_transforms.py tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "feat(refit): represent multi-component NCCL transforms"
```

### Task 7: NVFP4 NCCL Receiver and Validation

**Files:**
- Modify: `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Modify: `tests/unit/models/generation/test_nccl_reshard_backend.py`
- Modify: `tests/unit/weight_sync/test_nccl_reshard_utils.py`

**Interfaces:**
- Consumes: Task 6 plans and Task 2 serializer.
- Produces: destination-local BF16-to-NVFP4 NCCL conversion under one ModelOpt reload lifecycle.

- [ ] **Step 1: Write failing receiver tests**

Test ordered component receipt, scratch checkpoint-layout tensors, one finalizer after bulk and misc paths, incomplete group rejection, W4A4 static scale completeness, second-refit identity, and validator acceptance for the two supported combinations. Retain rejection for DTensor policy, non-vLLM generation, ModelOpt MoE EP, non-block-aligned K, and missing W4A4 artifact.

- [ ] **Step 2: Verify RED**

Run the focused generation and validator tests. Expected: validator rejects `real_quant` and the receiver bypasses ModelOpt's lifecycle.

- [ ] **Step 3: Implement destination conversion**

Receive resharded BF16 shards into owned checkpoint-layout scratch, serialize complete groups with Task 2, and call `_load_weights()` inside `_weight_update_lifecycle("collective")`. Do not write collective bytes directly into post-processed Marlin or fused-MoE runtime parameters. Finalize once after bulk and misc updates and synchronize before returning completion.

- [ ] **Step 4: Replace blanket validation with capability checks**

Accept only Megatron BF16 storage plus vLLM W4A16/W4A4 real quant under the restrictions in Global Constraints. Emit actionable errors naming the unsupported source format, target format, sharding, backend, or missing artifact.

- [ ] **Step 5: Verify GREEN**

Run all NCCL-Reshard, ModelOpt generation, serializer, and artifact unit tests. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/modelopt/models/generation/vllm_quant_backend.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/models/generation/test_nccl_reshard_backend.py tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "feat(refit): support BF16 to NVFP4 NCCL reshard"
```

### Task 8: Functional Tests and GCP-NRT Smoke Runs

**Files:**
- Create: `tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.sh`
- Create: `tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.sh`
- Create: `experiments/bf16-nvfp4-rollout/README.md`
- Create: `experiments/bf16-nvfp4-rollout/PLAN.md`
- Create after runs: `experiments/bf16-nvfp4-rollout/report.md`

**Interfaces:**
- Consumes: Tasks 1-7 and the GCP-NRT NeMo-RL nightly container/model paths already documented in the workspace.
- Produces: reproducible commands and legacy/NCCL two-step evidence for both modes.

- [ ] **Step 1: Add failing functional assertions**

Each test-suite script runs two steps with checkpointing disabled and verifies logs contain real ModelOpt NVFP4 detection, the expected W4A16 or W4A4 quantization method, two completed refits, and two completed GRPO steps. It rejects QARL policy-worker selection and any incomplete reload, manifest, NaN, or NCCL agreement error.

- [ ] **Step 2: Run local static verification**

Run recipe tests, `bash -n` on both scripts, formatting, lint, and the focused unit suites from Tasks 1-7.

- [ ] **Step 3: Commit and push before submission**

Commit scripts and experiment plan with DCO sign-off, push `sna/bf16-nvfp4-rollout` to `fork`, then use the remote clone to pull that exact commit.

- [ ] **Step 4: Generate the W4A4 calibration artifact**

Submit the standalone exporter on GCP-NRT using the Qwen3-30B-A3B checkpoint and record the exact checkpoint revision, quant config, dataset, sample count, sequence length, seed, output path, file size, and SHA256 in `experiments/bf16-nvfp4-rollout/report.md`.

- [ ] **Step 5: Submit and monitor four runs**

Run W4A16 legacy, W4A4 legacy, W4A16 NCCL, and W4A4 NCCL with `grpo.max_num_steps=2`, `checkpointing.enabled=false`, unique W&B names under an `sna-` project, and four-hour `batch` allocations. Monitor each for five minutes and record job id, commit, container, config, overrides, log path, and W&B URL.

- [ ] **Step 6: Validate outputs and write the report**

Require two completed refits and steps. Compare reward, KL, importance ratio, generation, refit, and E2E times. Record failures exactly rather than treating initialization as success.

- [ ] **Step 7: Commit the report**

```bash
git add experiments/bf16-nvfp4-rollout tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.sh tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.sh
git commit -s -m "test(modelopt): validate BF16 to NVFP4 rollout"
```

### Task 9: Whole-Branch Verification

**Files:**
- Modify only files required to fix review findings.

**Interfaces:**
- Consumes: completed Tasks 1-8.
- Produces: a review-ready branch with reproducible evidence.

- [ ] **Step 1: Run the complete affected unit test set**

Run recipe, ModelOpt policy/generation, serializer, artifact, vLLM backend, refit-transform, and NCCL-Reshard tests from a clean environment.

- [ ] **Step 2: Run formatting and static checks**

Use the repository-prescribed formatter, linter, type checker, and `git diff --check`. Do not broaden edits outside the affected modules.

- [ ] **Step 3: Request whole-branch code review**

Review against the design spec and this plan, emphasizing BF16 trainer invariance, canonical packing, W4A4 scale provenance, transport-buffer lifetime, grouped MoE completeness, and second-refit identity.

- [ ] **Step 4: Resolve findings and reverify**

Fix every Critical or Important finding, rerun its focused regression test, then rerun the full affected test set.

- [ ] **Step 5: Final state capture**

Update session memory and the experiment report with branch SHA, test commands/results, GPU job results, remaining limitations, and exact next action for a future PR split.
