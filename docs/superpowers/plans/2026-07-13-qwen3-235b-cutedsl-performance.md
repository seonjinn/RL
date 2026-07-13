# Qwen3-235B CuTeDSL Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a fail-closed Qwen3-235B-A22B 16-node/64-GB200 CuTeDSL OFF/ON benchmark that preserves the official NeMo-RL workload, separates timing from Nsight profiling, and reports component-level speedups with causal kernel attribution.

**Architecture:** Add one policy-MXFP8 overlay on the official BF16-rollout recipe and describe the model/workload/topology with canonical typed JSON profiles. Reuse the existing matrix engine through the profile interface, store 235B results under a separate experiment namespace, and submit three timing jobs plus two independent profile-only jobs. Bind submissions, manifests, timing results, and profile evidence with the same profile SHA and compatibility identity so mixed 30B/235B or mixed-software aggregates fail closed.

**Tech Stack:** NeMo-RL, Megatron-Core, Transformer Engine MXFP8/CuTeDSL, Hydra/OmegaConf YAML, Pydantic v2, Bash, SLURM/Pyxis, Ray, Python JSON/hashlib, pytest, Ruff, Nsight Systems, static HTML.

## Global Constraints

- Base recipe: `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml`.
- Model: `Qwen/Qwen3-235B-A22B`; dataset: `nvidia/OpenMathInstruct-2`, split `train_1M`.
- Hardware/topology: 16 nodes, 4 GB200 GPUs/node, segment size 16, TP2/PP4/CP2/EP16/ETP1.
- Workload: 16 prompts × 32 generations, GBS512, MBS1, logprob batch 1, maximum total sequence length 8192.
- Rollout remains asynchronous BF16 vLLM TP8 with GPU-memory utilization 0.4; never inherit `grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml`.
- Policy overlay requires router fp32, TE op fuser, GLU interleave 32, MXFP8 `e4m3`, and `fp8_param=false`.
- Accepted OFF/ON arms differ only in `policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP` equal to `"0"` or `"1"`.
- Both arms keep `cuda_graph_impl=none`, `overlap_moe_expert_parallel_comm=false`, `high_priority_a2a_comm_stream=false`, and `delay_wgrad_compute=false`.
- Validation and checkpoint saving are disabled identically in timing arms.
- Existing-Ray runs require `triton_cache_scope=job_node_local` before any 16-node submission.
- Functional and pilot results are never performance evidence.
- Accepted timing uses three replicas in ON/OFF, OFF/ON, ON/OFF order, each with five warmup and twenty measured updates.
- Nsight runs are two independent two-update jobs, one ON and one OFF, and never share Slurm status with timing jobs.
- No speedup claim is valid until timing, workload equivalence, compatibility identity, and profile attribution all pass.
- Every remote submission uses one reviewed/pushed source SHA and one pinned image SHA.
- Every new production Python or shell file starts with the repository's 2026 NVIDIA Apache-2.0 header; files under `tests/` are exempt.

---

## File Map

- Create `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml`: policy-only MXFP8/CuTeDSL overlay.
- Create `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/lib/model_profile.py`: typed profile loader, canonicalizer, SHA, and shell export CLI.
- Create `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/model_profiles/qwen3_30ba3b_4n4g.json`: typed compatibility profile for the existing official 30B wrapper.
- Create `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/model_profiles/qwen3_235b_16n4g.json`: exact 235B workload/topology/artifact profile.
- Modify `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/prepare_hf_cache.py`: consume a typed model profile and record file counts and bytes.
- Modify `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh`: accept profile/stage, namespace results, and emit role-aware submission schema.
- Modify `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`: validate the entire profile and execute exactly one run role.
- Modify `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py`: bind three timing jobs and two profile jobs with one compatibility identity.
- Modify `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py`: render dynamic model labels, profile-only state, and namespaced aggregates.
- Create `experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/submit_nemo2606_16n4g_performance.sh`: thin model-specific staged wrapper.
- Create `experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/evaluate_pilot.py`: conservative duration projection.
- Create `experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/README.md`: commands, stages, incidents, and acceptance state.
- Create `experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/report/public/index.html`: generated report target, not hand-edited.
- Modify the five existing CuTeDSL test modules and add `tests/test_qwen3_235b_cutedsl_recipe.py`.

### Task 1: Policy-MXFP8 CuTeDSL overlay

**Files:**
- Create: `tests/test_qwen3_235b_cutedsl_recipe.py`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml`

**Interfaces:**
- Produces: Hydra recipe `grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml` with unchanged BF16 rollout and exact policy prerequisites.

- [ ] **Step 1: Write a RED resolved-config contract**

```python
RECIPE = PROJECT_ROOT / "examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml"


def test_qwen3_235b_policy_mxfp8_cutedsl_contract() -> None:
    config = OmegaConf.to_container(load_config(RECIPE), resolve=True)
    policy = config["policy"]
    megatron = policy["megatron_cfg"]
    generation = policy["generation"]

    assert policy["model_name"] == "Qwen/Qwen3-235B-A22B"
    assert config["cluster"] == {"num_nodes": 16, "gpus_per_node": 4, "segment_size": 16}
    assert [megatron[key] for key in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
    )] == [2, 4, 2, 16, 1]
    assert policy["train_global_batch_size"] == 512
    assert policy["train_micro_batch_size"] == 1
    assert policy["logprob_batch_size"] == 1
    assert config["grpo"]["num_prompts_per_step"] == 16
    assert config["grpo"]["num_generations_per_prompt"] == 32
    assert policy["max_total_sequence_length"] == 8192
    assert generation["vllm_cfg"]["tensor_parallel_size"] == 8
    assert generation["vllm_cfg"]["precision"] == "bfloat16"
    assert generation["vllm_cfg"]["gpu_memory_utilization"] == 0.4
    assert megatron["moe_router_dtype"] == "fp32"
    assert megatron["use_transformer_engine_op_fuser"] is True
    assert megatron["moe_mlp_glu_interleave_size"] == 32
    assert megatron["fp8_cfg"] == {
        **megatron["fp8_cfg"],
        "enabled": True,
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "fp8_param": False,
    }
    assert megatron["env_vars"]["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == "1"
    assert megatron["cuda_graph_impl"] == "none"
    assert megatron["overlap_moe_expert_parallel_comm"] is False
    assert megatron["high_priority_a2a_comm_stream"] is False
    assert megatron["delay_wgrad_compute"] is False
```

- [ ] **Step 2: Confirm the missing overlay is RED**

Run: `uv run pytest -q tests/test_qwen3_235b_cutedsl_recipe.py`

Expected: FAIL with `FileNotFoundError` for the new recipe.

- [ ] **Step 3: Add the minimal overlay**

```yaml
defaults: ./grpo-qwen3-235b-16n4g.yaml
checkpointing:
  checkpoint_dir: results/grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl
policy:
  megatron_cfg:
    moe_router_dtype: fp32
    use_transformer_engine_op_fuser: true
    moe_mlp_glu_interleave_size: 32
    cuda_graph_impl: none
    overlap_moe_expert_parallel_comm: false
    high_priority_a2a_comm_stream: false
    delay_wgrad_compute: false
    fp8_cfg:
      enabled: true
      fp8: e4m3
      fp8_recipe: mxfp8
      fp8_param: false
    env_vars:
      NVTE_CUTEDSL_FUSED_GROUPED_MLP: "1"
logger:
  log_dir: logs/grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl
  wandb:
    name: grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl
```

Do not add any `policy.generation` keys, which keeps rollout inherited from the BF16 base.

- [ ] **Step 4: Prove resolved values and base isolation**

Run: `uv run pytest -q tests/test_qwen3_235b_cutedsl_recipe.py tests/test_mxfp8_rollout_recipes.py`

Expected: PASS, including the existing rollout-recipe tests.

- [ ] **Step 5: Commit the recipe**

```bash
git add examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml tests/test_qwen3_235b_cutedsl_recipe.py
git commit -s -m "feat: add Qwen3-235B policy MXFP8 recipe"
```

### Task 2: Canonical typed model profiles

**Files:**
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/lib/model_profile.py`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/model_profiles/qwen3_30ba3b_4n4g.json`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/model_profiles/qwen3_235b_16n4g.json`
- Modify: `tests/test_nemo2606_multinode_factorial_harness.py`

**Interfaces:**
- Produces: frozen, `extra="forbid"` Pydantic v2 `ArtifactProfile`, `TopologyProfile`, `WorkloadProfile`, and `ModelProfile` user-facing schemas.
- Produces: `load_model_profile(path: Path) -> ModelProfile`.
- Produces: `canonical_profile_json(profile: ModelProfile) -> str` and `profile_sha256(profile: ModelProfile) -> str`.
- CLI: `model_profile.py shell --profile PATH`, emitting shell-quoted `CUTEDSL_PROFILE_*` assignments.

- [ ] **Step 1: Write profile parser and exact-value tests**

```python
def test_qwen3_235b_profile_is_exact_and_canonical() -> None:
    module = load_model_profile_module()
    path = EXPERIMENT_DIR / "model_profiles/qwen3_235b_16n4g.json"
    profile = module.load_model_profile(path)
    assert profile.profile_id == "qwen3_235b_16n4g"
    assert profile.recipe == "examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml"
    assert profile.result_namespace == "cutedsl_qwen3_235b_pre_tyche_16n4g"
    assert profile.topology.model_dump() == {
        "num_nodes": 16,
        "gpus_per_node": 4,
        "segment_size": 16,
        "tp": 2,
        "pp": 4,
        "cp": 2,
        "ep": 16,
        "etp": 1,
    }
    assert profile.workload.model_dump() == {
        "train_global_batch_size": 512,
        "train_micro_batch_size": 1,
        "logprob_batch_size": 1,
        "max_total_sequence_length": 8192,
        "sequence_packing_enabled": True,
        "num_prompts_per_step": 16,
        "num_generations_per_prompt": 32,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", module.profile_sha256(profile))


def test_profile_loader_rejects_unknown_keys_and_path_escape(tmp_path: Path) -> None:
    payload = valid_235b_profile_dict()
    payload["unknown"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_model_profile_module().load_model_profile(path)
    payload = valid_235b_profile_dict()
    payload["result_namespace"] = "../escape"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="result_namespace"):
        load_model_profile_module().load_model_profile(path)
```

- [ ] **Step 2: Confirm RED import/profile files**

Run: `uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k 'profile_is_exact or profile_loader_rejects'`

Expected: FAIL because the module and profile JSON files do not exist.

- [ ] **Step 3: Implement strict Pydantic schemas and canonicalization**

```python
class StrictProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactProfile(StrictProfile):
    model_repo_id: str
    dataset_repo_id: str
    dataset_repo_type: Literal["dataset"]
    dataset_split: str

    @field_validator("model_repo_id", "dataset_repo_id", "dataset_split")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact identifiers must not be empty")
        return value


class TopologyProfile(StrictProfile):
    num_nodes: int
    gpus_per_node: int
    segment_size: int
    tp: int
    pp: int
    cp: int
    ep: int
    etp: int

    @field_validator("num_nodes", "gpus_per_node", "segment_size", "tp", "pp", "cp", "ep", "etp")
    @classmethod
    def require_positive(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("topology values must be positive integers")
        return value

    @model_validator(mode="after")
    def validate_parallelism(self) -> "TopologyProfile":
        if self.segment_size > self.num_nodes:
            raise ValueError("segment_size must not exceed num_nodes")
        if self.ep > self.num_nodes * self.gpus_per_node:
            raise ValueError("ep must not exceed world size")
        return self


class WorkloadProfile(StrictProfile):
    train_global_batch_size: int
    train_micro_batch_size: int
    logprob_batch_size: int
    max_total_sequence_length: int
    sequence_packing_enabled: bool
    num_prompts_per_step: int
    num_generations_per_prompt: int

    @field_validator(
        "train_global_batch_size",
        "train_micro_batch_size",
        "logprob_batch_size",
        "max_total_sequence_length",
        "num_prompts_per_step",
        "num_generations_per_prompt",
    )
    @classmethod
    def require_positive(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("workload values must be positive integers")
        return value

    @model_validator(mode="after")
    def validate_batch_cardinality(self) -> "WorkloadProfile":
        cardinality = self.num_prompts_per_step * self.num_generations_per_prompt
        if cardinality != self.train_global_batch_size:
            raise ValueError("prompt-generation cardinality must equal global batch size")
        return self


class ModelProfile(StrictProfile):
    schema_version: Literal[1]
    profile_id: str
    display_name: str
    recipe: str
    result_namespace: str
    artifacts: ArtifactProfile
    topology: TopologyProfile
    workload: WorkloadProfile

    @field_validator("profile_id", "result_namespace")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9_]+", value) is None:
            raise ValueError("profile identifiers must match [a-z0-9_]+")
        return value

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        if not value.strip() or len(value) > 128:
            raise ValueError("display_name must contain 1 to 128 characters")
        return value

    @field_validator("recipe")
    @classmethod
    def validate_recipe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value.startswith("examples/configs/recipes/") or ".." in path.parts:
            raise ValueError("recipe must be a contained recipe path")
        return value


def canonical_profile_json(profile: ModelProfile) -> str:
    return json.dumps(profile.model_dump(), sort_keys=True, separators=(",", ":"))


def profile_sha256(profile: ModelProfile) -> str:
    return hashlib.sha256(canonical_profile_json(profile).encode()).hexdigest()


def load_model_profile(path: Path) -> ModelProfile:
    try:
        return ModelProfile.model_validate_json(path.read_text())
    except ValidationError as error:
        raise ValueError(f"Invalid model profile {path.name}: {error}") from error
```

Use the exact validators above with no call-site defaults; do not weaken strict parsing to coerce strings or booleans into integers.

- [ ] **Step 4: Add the exact 235B JSON profile**

```json
{
  "schema_version": 1,
  "profile_id": "qwen3_235b_16n4g",
  "display_name": "Qwen3-235B-A22B",
  "recipe": "examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml",
  "result_namespace": "cutedsl_qwen3_235b_pre_tyche_16n4g",
  "artifacts": {
    "model_repo_id": "Qwen/Qwen3-235B-A22B",
    "dataset_repo_id": "nvidia/OpenMathInstruct-2",
    "dataset_repo_type": "dataset",
    "dataset_split": "train_1M"
  },
  "topology": {
    "num_nodes": 16,
    "gpus_per_node": 4,
    "segment_size": 16,
    "tp": 2,
    "pp": 4,
    "cp": 2,
    "ep": 16,
    "etp": 1
  },
  "workload": {
    "train_global_batch_size": 512,
    "train_micro_batch_size": 1,
    "logprob_batch_size": 1,
    "max_total_sequence_length": 8192,
    "sequence_packing_enabled": true,
    "num_prompts_per_step": 16,
    "num_generations_per_prompt": 32
  }
}
```

Add the 30B profile with its existing official 4n4g resolved values so the old wrapper has no hidden defaults.

- [ ] **Step 5: Run parser, format, and regression tests**

Run: `uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k 'model_profile or profile_is_exact' && uv run ruff check experiments/cutedsl_qwen3_30ba3b_oci_1n4g/lib/model_profile.py tests/test_nemo2606_multinode_factorial_harness.py`

Expected: PASS and no Ruff findings.

- [ ] **Step 6: Commit typed profiles**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/lib/model_profile.py experiments/cutedsl_qwen3_30ba3b_oci_1n4g/model_profiles tests/test_nemo2606_multinode_factorial_harness.py
git commit -s -m "feat: define typed CuTeDSL model profiles"
```

### Task 3: Profile-aware HF cache and full config validation

**Files:**
- Modify: `tests/test_cutedsl_hf_cache.py`
- Modify: `tests/test_nemo2606_multinode_factorial_harness.py`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/prepare_hf_cache.py`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`

**Interfaces:**
- Changes: `prepare_cache(profile: ModelProfile, hf_home: Path, shared_manifest: Path, snapshot_download: SnapshotDownload, load_dataset: DatasetLoad) -> dict[str, Any]`.
- CLI adds required `--model-profile PATH`.
- Manifest adds `model_profile`, `model_profile_sha256`, `topology`, `workload`, and `triton_cache_scope`.

- [ ] **Step 1: Parameterize cache tests for 30B and 235B**

```python
@pytest.mark.parametrize(
    ("profile_name", "expected_model"),
    [
        ("qwen3_30ba3b_4n4g.json", "Qwen/Qwen3-30B-A3B"),
        ("qwen3_235b_16n4g.json", "Qwen/Qwen3-235B-A22B"),
    ],
)
def test_prepare_cache_uses_profile_artifacts(
    tmp_path: Path, profile_name: str, expected_model: str
) -> None:
    profile = load_profile(PROFILE_DIR / profile_name)
    manifest = PREPARER.prepare_cache(
        profile,
        tmp_path / "hf",
        tmp_path / "manifest.json",
        fake_snapshot_download,
        fake_load_dataset,
    )
    assert manifest["profile_id"] == profile.profile_id
    assert manifest["profile_sha256"] == profile_sha256(profile)
    assert manifest["repositories"]["model"]["repo_id"] == expected_model
    assert manifest["repositories"]["dataset"]["split"] == "train_1M"
    assert manifest["repositories"]["dataset"]["num_rows"] == 1_000_000
    assert manifest["repositories"]["model"]["file_count"] > 0
    assert manifest["repositories"]["model"]["total_bytes"] > 0
```

Add tests that a 30B manifest cannot verify under the 235B profile, offline verification happens in a fresh subprocess, and the cache-capacity check rejects `required_bytes + 20 GiB > free_bytes` before download.

Also retain and extend the shared-conversion contract:

```python
def test_235b_megatron_conversion_uses_job_shared_checkpoint_root() -> None:
    source = MATRIX_PAYLOAD.read_text()
    create_root = 'mkdir -p "${MEGATRON_CHECKPOINT_ROOT}"'
    export_root = 'export NRL_MEGATRON_CHECKPOINT_DIR="${MEGATRON_CHECKPOINT_ROOT}"'
    assert 'MEGATRON_CHECKPOINT_ROOT="${CONTAINER_RUNTIME_DIR}/megatron_checkpoints"' in source
    assert source.index(create_root) < source.index(export_root)
    assert '"megatron_checkpoint_scope": "job_shared"' in source
    assert 'find "${MEGATRON_CHECKPOINT_ROOT}" -name run_config.yaml -type f -print -quit' in source
```

- [ ] **Step 2: Observe RED signature and hard-coded model failures**

Run: `uv run pytest -q tests/test_cutedsl_hf_cache.py -k 'profile or capacity'`

Expected: FAIL because `prepare_cache` has no profile parameter and still uses `MODEL_REPO_ID` globals.

- [ ] **Step 3: Replace artifact globals with profile values**

```python
def _repositories(profile: ModelProfile) -> tuple[tuple[str, str, str | None], ...]:
    return (
        ("model", profile.artifacts.model_repo_id, None),
        (
            "dataset",
            profile.artifacts.dataset_repo_id,
            profile.artifacts.dataset_repo_type,
        ),
    )


def _snapshot_size(snapshot: Path) -> tuple[int, int]:
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"Hugging Face snapshot contains no files: {snapshot.name}")
    return len(files), sum(path.stat().st_size for path in files)
```

Include `profile_id`, `profile_sha256`, `file_count`, and `total_bytes` in schema version 2. Read dataset repo/split from the profile. Name shared manifests with `f"nemo2606_{profile.profile_id}_{profile_sha256(profile)[:12]}.json"`.

- [ ] **Step 4: Validate all fixed fields against the resolved recipe**

In the matrix embedded Python, construct:

```python
actual_topology = {
    "num_nodes": config["cluster"]["num_nodes"],
    "gpus_per_node": config["cluster"]["gpus_per_node"],
    "segment_size": config["cluster"]["segment_size"],
    "tp": megatron["tensor_model_parallel_size"],
    "pp": megatron["pipeline_model_parallel_size"],
    "cp": megatron["context_parallel_size"],
    "ep": megatron["expert_model_parallel_size"],
    "etp": megatron["expert_tensor_parallel_size"],
}
actual_workload = {
    "train_global_batch_size": policy["train_global_batch_size"],
    "train_micro_batch_size": policy["train_micro_batch_size"],
    "logprob_batch_size": policy["logprob_batch_size"],
    "max_total_sequence_length": policy["max_total_sequence_length"],
    "sequence_packing_enabled": bool(policy["sequence_packing"]),
    "num_prompts_per_step": config["grpo"]["num_prompts_per_step"],
    "num_generations_per_prompt": config["grpo"]["num_generations_per_prompt"],
}
assert actual_topology == model_profile["topology"]
assert actual_workload == model_profile["workload"]
```

Retain exact invariant checks for grouped GEMM, router fp32, op fuser, interleave 32, policy MXFP8, BF16 rollout, and all four disabled full-CG/A2A selectors.

Record `megatron_checkpoint_scope=job_shared` in the manifest. After policy initialization and before the first accepted measured update, require exactly one converted checkpoint tree containing `iter_0000000/run_config.yaml`; NeMo-RL's rank-zero conversion remains the only writer and every node reads the exported shared root. A missing or multiple conversion tree fails before timing evidence is accepted.

- [ ] **Step 5: Run cache and config tests**

Run: `uv run pytest -q tests/test_cutedsl_hf_cache.py tests/test_nemo2606_multinode_factorial_harness.py -k 'profile or topology or workload or offline or capacity'`

Expected: PASS for both profiles; mixed identities fail with deterministic messages.

- [ ] **Step 6: Commit cache/config generalization**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/prepare_hf_cache.py experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch tests/test_cutedsl_hf_cache.py tests/test_nemo2606_multinode_factorial_harness.py
git commit -s -m "feat: bind CuTeDSL runs to model profiles"
```

### Task 4: Separate functional, pilot, timing, and profile run roles

**Files:**
- Modify: `tests/test_nemo2606_multinode_factorial_harness.py`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`
- Create: `experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/submit_nemo2606_16n4g_performance.sh`
- Create: `experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/evaluate_pilot.py`

**Interfaces:**
- Submitter CLI: `--model-profile PATH --stage preflight|functional|pilot|timing|profile [--submission-group ID] [--test-only]`.
- Matrix env: `CUTEDSL_BENCHMARK_RUN_ROLE=functional|pilot|timing|profile`, and for profile `CUTEDSL_BENCHMARK_PROFILE_ARM=on|off`.
- Submission schema version 2 has `record_type`, `run_role`, `profile_id`, `profile_sha256`, `job_id`, `submission_group`, `factorial_context`, and role-specific order/arm.
- Pilot API: `project_full_paired_seconds(observed_seconds: float, pilot_updates_per_arm: int = 10, accepted_updates_per_arm: int = 25) -> float`.

- [ ] **Step 1: Write submission-shape and role-isolation tests**

```python
def test_235b_timing_stage_submits_three_unprofiled_jobs(mock_sbatch: Path) -> None:
    calls = run_submitter(stage="timing", mock_sbatch=mock_sbatch)
    assert [call["timing_order"] for call in calls] == ["on,off", "off,on", "on,off"]
    assert {call["run_role"] for call in calls} == {"timing"}
    assert {call["profile_enabled"] for call in calls} == {"0"}
    assert {call["num_nodes"] for call in calls} == {"16"}
    assert {call["gpus_per_node"] for call in calls} == {"4"}


def test_235b_profile_stage_submits_two_independent_jobs(mock_sbatch: Path) -> None:
    calls = run_submitter(stage="profile", mock_sbatch=mock_sbatch, timing_complete=True)
    assert [call["profile_arm"] for call in calls] == ["on", "off"]
    assert {call["run_role"] for call in calls} == {"profile"}
    assert {call["timing_order"] for call in calls} == {""}
    assert len({call["job_id"] for call in calls}) == 2


def test_role_artifacts_are_disjoint() -> None:
    source = MATRIX_PAYLOAD.read_text()
    assert 'RUN_ROLE="${CUTEDSL_BENCHMARK_RUN_ROLE:?"}' in source
    assert 'timing_summary.json must not exist for profile role' in source
    assert 'kernel_attribution.json must not exist for timing role' in source


@pytest.mark.parametrize("selector", ["NEMO2606_FULL_CG_ENABLED", "NEMO2606_A2A_ENABLED"])
def test_235b_profile_rejects_full_cg_and_a2a_before_sbatch(
    selector: str, mock_sbatch: Path
) -> None:
    completed = run_submitter(
        stage="functional",
        mock_sbatch=mock_sbatch,
        extra_env={selector: "1"},
        check=False,
    )
    assert completed.returncode == 1
    assert mock_sbatch.read_text() == ""
    assert "Qwen3-235B profile requires full-CG=0 and A2A=0" in completed.stderr
```

- [ ] **Step 2: Confirm the current combined-job behavior is RED**

Run: `uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k '235b_timing_stage or 235b_profile_stage or role_artifacts'`

Expected: FAIL because the current first timing replica appends ON/OFF profiling to the same Slurm job.

- [ ] **Step 3: Add strict role dispatch to the matrix payload**

```bash
RUN_ROLE="${CUTEDSL_BENCHMARK_RUN_ROLE:?CUTEDSL_BENCHMARK_RUN_ROLE is required}"
case "${RUN_ROLE}" in
    functional|pilot|timing|profile) ;;
    *) echo "[ERROR] Invalid run role: ${RUN_ROLE}" >&2; exit 1 ;;
esac
readonly RUN_ROLE
```

Role contract:

```bash
case "${RUN_ROLE}" in
    functional) timing_arms=(on); WARMUP_UPDATES=0; MEASURED_UPDATES=3 ;;
    pilot) timing_arms=(off on); WARMUP_UPDATES=5; MEASURED_UPDATES=5 ;;
    timing) IFS=',' read -r -a timing_arms <<< "${CUTEDSL_BENCHMARK_ORDER:?}" ;;
    profile)
        timing_arms=()
        PROFILE_ARM="${CUTEDSL_BENCHMARK_PROFILE_ARM:?}"
        [[ "${PROFILE_ARM}" == on || "${PROFILE_ARM}" == off ]] || exit 1
        ;;
esac
```

Timing roles skip every profile block and assert no `profiles/` or `kernel_attribution.json`. Profile role calls `run_profile_arm "${PROFILE_ARM}" 0`, generates a one-arm `kernel_attribution.json`, skips timing/summarizer code, asserts no `timing_summary.json`, and sets `performance_eligible=false`.

- [ ] **Step 4: Emit five independent accepted records**

Timing JSONL records are built from the canonical profile rather than a copied digest:

```python
timing_record = {
    "schema_version": 2,
    "record_type": "timing",
    "run_role": "timing",
    "replicate_index": 0,
    "timing_order": "on,off",
    "profile_enabled": False,
    "job_id": timing_job_id,
    "submission_group": submission_group,
    "factorial_context": "g0a0",
    "profile_id": profile.profile_id,
    "profile_sha256": profile_sha256(profile),
}
```

Profile JSONL records use the same computed identity:

```python
profile_record = {
    "schema_version": 2,
    "record_type": "profile",
    "run_role": "profile",
    "profile_arm": "on",
    "profile_enabled": True,
    "job_id": profile_on_job_id,
    "submission_group": submission_group,
    "factorial_context": "g0a0",
    "profile_id": profile.profile_id,
    "profile_sha256": profile_sha256(profile),
}
```

`--stage profile` must read the existing group JSONL, require three distinct completed timing records, and invoke `collect_cutedsl_ab_replicates.py --timing-only` before calling `sbatch`. It appends exactly ON and OFF profile records to that same file. Do not use an `afterok` dependency; profile scheduling and failure remain independent of timing status.

- [ ] **Step 5: Add the conservative pilot projection**

```python
FIVE_HOURS_SECONDS = 18_000.0
ACCEPTED_LIMIT_SECONDS = 16_200.0


def project_full_paired_seconds(
    observed_seconds: float,
    pilot_updates_per_arm: int = 10,
    accepted_updates_per_arm: int = 25,
) -> float:
    if not math.isfinite(observed_seconds) or observed_seconds <= 0:
        raise ValueError("observed_seconds must be finite and positive")
    return observed_seconds * accepted_updates_per_arm / pilot_updates_per_arm
```

The CLI reads the pilot manifest's monotonic start/end seconds, writes `pilot_projection.json`, and exits 0 only when projected seconds are `<= 16_200`. If it exceeds 4.5 hours, no paired timing job may be submitted; the submitter prints the predeclared six-job arm-separated scheme and exits 3 before accepted data exists.

- [ ] **Step 6: Add the thin 235B wrapper**

```bash
#!/bin/bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
PROFILE="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g/model_profiles/qwen3_235b_16n4g.json"
exec "${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh" \
    --model-profile "${PROFILE}" "$@"
```

No 235B topology value is duplicated in this wrapper.

- [ ] **Step 7: Run role, submission, pilot, and Bash tests**

Run: `uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k 'stage or run_role or pilot or qwen3_235b' && bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh && bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch && bash -n experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/submit_nemo2606_16n4g_performance.sh`

Expected: PASS; timing mock has exactly three jobs and profile mock exactly two.

- [ ] **Step 8: Commit role separation**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch experiments/cutedsl_qwen3_235b_pre_tyche_16n4g tests/test_nemo2606_multinode_factorial_harness.py
git commit -s -m "feat: separate CuTeDSL timing and profile jobs"
```

### Task 5: Collector compatibility identity and standalone profile attribution

**Files:**
- Modify: `tests/test_cutedsl_replicate_collector.py`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py`

**Interfaces:**
- Produces: `ProfileRun(job_id: str, run_id: str, result_dir: Path, arm: Literal["on", "off"])`.
- Produces: `_partition_submission_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]`.
- Produces: `_compatibility_identity(manifest: dict[str, Any]) -> dict[str, Any]`.
- CLI adds `--timing-only`; final aggregate schema becomes version 3 with `profile_runs.on` and `profile_runs.off`.

- [ ] **Step 1: Split fixtures into three timing directories and two profile directories**

```python
def test_collector_accepts_three_timing_and_two_profile_jobs(tmp_path: Path) -> None:
    result_root, submission = write_qwen235_group(
        tmp_path,
        timing_jobs=("100", "101", "102"),
        profile_jobs={"on": "200", "off": "201"},
    )
    completed = run_collector(result_root, submission)
    assert completed.returncode == 0, completed.stderr
    aggregate = json.loads((result_root / "aggregate/aggregate.json").read_text())
    assert aggregate["schema_version"] == 3
    assert aggregate["replicate_count"] == 3
    assert aggregate["profile_runs"] == {
        "on": {"job_id": "200", "run_id": "200"},
        "off": {"job_id": "201", "run_id": "201"},
    }
```

Add rejection tests for missing/duplicate profile arm, profile job ID equal to a timing job ID, different submission group/context, source/image/profile/base-config/artifact/topology mismatch, failed status, timing summary in a profile job, or profile artifacts in a timing job.

- [ ] **Step 2: Confirm existing designated-replica logic is RED**

Run: `uv run pytest -q tests/test_cutedsl_replicate_collector.py -k 'three_timing_and_two_profile or profile_arm or mixed_profile'`

Expected: FAIL because the collector requires exactly one profile-enabled timing replica and searches its directory for both arms.

- [ ] **Step 3: Add exact record partitioning**

```python
@dataclass(frozen=True)
class ProfileRun:
    job_id: str
    run_id: str
    result_dir: Path
    arm: Literal["on", "off"]


def _partition_submission_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    timing = [record for record in records if record.get("record_type") == "timing"]
    profile_records = [
        record for record in records if record.get("record_type") == "profile"
    ]
    if len(timing) < 3:
        raise CollectorError("at least three timing records are required")
    by_arm = {record.get("profile_arm"): record for record in profile_records}
    if len(profile_records) != 2 or set(by_arm) != {"on", "off"}:
        raise CollectorError("exactly one ON and one OFF profile record are required")
    all_ids = [record.get("job_id") for record in records]
    if len(all_ids) != len(set(all_ids)):
        raise CollectorError("timing and profile job IDs must be distinct")
    return timing, by_arm
```

Require schema version 2 and equal `submission_group`, `factorial_context`, `profile_id`, and `profile_sha256` across every record.

- [ ] **Step 4: Define compatibility identity and normalized operational fields**

```python
def _compatibility_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_sha": manifest["source_sha"],
        "upstream_sha": manifest["upstream_sha"],
        "image_sha256": manifest["image_sha256"],
        "model_profile": manifest["model_profile"],
        "model_profile_sha256": manifest["model_profile_sha256"],
        "recipe": manifest["recipe"],
        "base_config_sha256": manifest["base_config_sha256"],
        "artifact_revisions": manifest["artifact_revisions"],
        "topology": manifest["topology"],
        "fixed_config_evidence": manifest["fixed_config_evidence"],
        "feature_context": manifest["feature_context"],
        "triton_cache_scope": manifest["triton_cache_scope"],
    }
```

Normalize only `grpo.max_num_steps`, paths, logger names, run role, timing order, and the CuTeDSL selector when computing the association hash. Never normalize model, topology, rollout, batch, sequence, or precision fields.

- [ ] **Step 5: Validate one-arm profile evidence**

For each `ProfileRun`, require successful job status, `run_role=profile`, `performance_eligible=false`, exactly one profile summary, one nonempty kernel-evidence file, at least one `.nsys-rep`, and selector/config evidence matching its arm. Across the pair, require ON fused GLU/dGLU/quant/grouped counts greater than zero, OFF fused counts zero, and OFF baseline expert GEMM count greater than zero with grouped-MoE config true.

- [ ] **Step 6: Preserve timing-only validation for staged submission**

`--timing-only` validates three successful timing records, workload equivalence, complete component series, compatibility identity, and absence of profile artifacts, then writes `aggregate/timing_gate.json` with `passed=true`. It does not compute a publishable causal aggregate and exits nonzero if any profile record already exists.

- [ ] **Step 7: Run full collector tests**

Run: `uv run pytest -q tests/test_cutedsl_replicate_collector.py`

Expected: PASS; old schema fixtures either migrate explicitly or fail with `schema_version 2 required`, never silently mix.

- [ ] **Step 8: Commit collector changes**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py tests/test_cutedsl_replicate_collector.py
git commit -s -m "feat: bind timing to standalone profile evidence"
```

### Task 6: Dynamic, sanitized 235B HTML report

**Files:**
- Modify: `tests/test_cutedsl_report.py`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py`
- Create: `experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/README.md`

**Interfaces:**
- Renderer CLI adds required `--result-root PATH` for aggregate refresh and derives display name from `model_profile.display_name`.
- Profile-only run HTML must show `Not performance evidence` and suppress speedup cards.
- Final aggregate renders PolicyTraining, E2E, Logprob, Refit, Generation, and generation-finalization rows from aggregate schema 3.

- [ ] **Step 1: Write dynamic-label, role, and sanitization tests**

```python
def test_qwen235_report_uses_profile_label_and_separate_profile_jobs(tmp_path: Path) -> None:
    run_dir = write_qwen235_report_fixture(tmp_path)
    render(run_dir)
    html = (run_dir / "index.html").read_text()
    assert "Qwen3-235B-A22B" in html
    assert "Qwen3 30B-A3B" not in html
    assert "Profile ON job 200" in html
    assert "Profile OFF job 201" in html
    assert "PolicyTraining tokens/s/GPU" in html


def test_profile_only_report_suppresses_performance_claim(tmp_path: Path) -> None:
    run_dir = write_profile_only_fixture(tmp_path, arm="on")
    render(run_dir)
    html = (run_dir / "index.html").read_text()
    assert "Not performance evidence" in html
    assert "Geometric-mean speedup" not in html
```

Extend the existing path/credential tests to profile manifests and cache-diagnostic summaries.

- [ ] **Step 2: Confirm the literal 30B label is RED**

Run: `uv run pytest -q tests/test_cutedsl_report.py -k 'qwen235_report or profile_only_report'`

Expected: FAIL because the renderer contains literal 30B labels and expects profile artifacts under a timing job.

- [ ] **Step 3: Render model and role dynamically**

```python
def _display_name(manifest: dict[str, Any]) -> str:
    profile = manifest.get("model_profile")
    if isinstance(profile, dict) and isinstance(profile.get("display_name"), str):
        return profile["display_name"]
    return manifest["artifact_revisions"]["model"]["repo_id"]


def _role_banner(manifest: dict[str, Any]) -> str:
    if manifest.get("run_role") == "profile":
        return '<p class="status inconclusive">Not performance evidence</p>'
    return ""
```

Read profile job links from aggregate schema 3, not timing job directories. Render public job IDs and relative artifact paths only.

- [ ] **Step 4: Render component metrics and eligibility gates**

For each component, render raw ON/OFF medians, per-replica paired effect, geometric-mean speedup, bootstrap interval, and direction consistency. Put PolicyTraining first. Render `neutral/inconclusive` for refit or generation when direction differs across replicas. Suppress all speedup cards when workload, timing, compatibility, or attribution gates fail.

- [ ] **Step 5: Document exact staged commands and incidents**

The 235B README must include:

```bash
./experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/submit_nemo2606_16n4g_performance.sh --stage preflight --test-only
./experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/submit_nemo2606_16n4g_performance.sh --stage functional
PILOT_APPROVED_GROUP=$(./experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/submit_nemo2606_16n4g_performance.sh --stage pilot | sed -n 's/^submission_group=//p')
./experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/submit_nemo2606_16n4g_performance.sh --stage timing --submission-group "${PILOT_APPROVED_GROUP}"
TIMING_COMPLETE_GROUP="${PILOT_APPROVED_GROUP}"
./experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/submit_nemo2606_16n4g_performance.sh --stage profile --submission-group "${TIMING_COMPLETE_GROUP}"
```

The wrapper must print exactly one line prefixed by `submission_group=` so command substitution is deterministic.

- [ ] **Step 6: Run renderer/report tests**

Run: `uv run pytest -q tests/test_cutedsl_report.py && uv run ruff check experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py tests/test_cutedsl_report.py`

Expected: PASS and no private storage path, hostname, IP, token, or worker-log body in staged public HTML.

- [ ] **Step 7: Commit reporting support**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/README.md tests/test_cutedsl_report.py
git commit -s -m "feat: report Qwen3-235B CuTeDSL results"
```

### Task 7: Local verification, review, push, and staged cluster execution

**Files:**
- Modify only files already listed when a test, review, or remote functional result proves a defect.
- Generate: `experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/report/public/index.html` after accepted data exists.

**Interfaces:**
- Produces: one source SHA, one image SHA, pinned model/dataset revisions, three accepted timing replicas, two profile jobs, aggregate schema 3, and sanitized HTML.

- [ ] **Step 1: Run complete local verification**

```bash
uv run pytest -q \
  tests/test_qwen3_235b_cutedsl_recipe.py \
  tests/test_cutedsl_policy_recipe.py \
  tests/test_cutedsl_hf_cache.py \
  tests/test_nemo2606_multinode_factorial_harness.py \
  tests/test_cutedsl_replicate_collector.py \
  tests/test_cutedsl_report.py
uv run ruff check \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g \
  experiments/cutedsl_qwen3_235b_pre_tyche_16n4g \
  tests/test_qwen3_235b_cutedsl_recipe.py
bash -n ray.sub
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh
bash -n experiments/cutedsl_qwen3_235b_pre_tyche_16n4g/submit_nemo2606_16n4g_performance.sh
```

Expected: all tests pass, Ruff is clean, Bash emits no syntax errors, and existing 30B regressions remain green.

- [ ] **Step 2: Request two-stage independent review**

Use `superpowers:requesting-code-review` against both approved specs and both implementation plans. First review spec compliance, then review code quality/safety. Resolve findings and rerun Step 1 before any push.

- [ ] **Step 3: Push only the feature branch**

```bash
git status --short
git push fork sna/nemo-2606-cutedsl-a2a-factorial-20260712
```

Expected: a clean feature worktree and no write to a default branch.

- [ ] **Step 4: Fast-forward Pre-Tyche and run scheduler/cache preflight**

Pull the feature branch in the existing remote worktree, verify recursive submodule cleanliness and the pinned nightly image SHA, then run the 16-node `--stage preflight --test-only` command. Run the locked HF cache warmup and fresh-process offline verification. Acceptance requires exact 40-character model/dataset revisions, one million dataset rows, bounded file count/bytes, and at least 20 GiB remaining after the recorded cache size.

- [ ] **Step 5: Run and monitor the three-update functional gate**

Submit `--stage functional`. Monitor Slurm, Ray head, and all workers for at least five minutes and through completion. Require generation, refit, reference/policy logprob, PolicyTraining, and next offload to complete; require `job_node_local`; reject Triton JSON errors, OOM, actor loss, or model/topology drift. Do not add this job to performance aggregation.

- [ ] **Step 6: Run the five-plus-five duration pilot**

Submit `--stage pilot`, collect the monotonic wall-clock interval, and run `evaluate_pilot.py`. If projection is `<= 16_200` seconds, record `paired_timing_approved=true`. If it is larger, record `arm_separated_required=true` and stop before accepted timing submission; implement the predeclared six-job arm-separated schema in a separately reviewed change.

- [ ] **Step 7: Submit and monitor three accepted timing replicas**

Submit `--stage timing` under the pilot-approved group. Verify the order ON/OFF, OFF/ON, ON/OFF; five warmups and twenty measured updates per arm; no `.nsys-rep`; identical profile SHA/source/image/revisions/base config; complete E2E, Logprob, PolicyTraining, Refit, Generation, finalization, and memory series. Monitor each job for five minutes and to terminal state.

- [ ] **Step 8: Gate timing and submit two independent profile jobs**

Run the collector with `--timing-only`. Only after it writes `timing_gate.json` with `passed=true`, submit `--stage profile` for ON and OFF. Each job runs exactly two updates and one selector. Require nonempty `.nsys-rep` files, actual CUDA kernel-stat rows, ON fused GLU/dGLU/quant/grouped evidence, and OFF zero fused counts plus baseline expert-GEMM/config evidence.

- [ ] **Step 9: Collect, render, and verify the final result**

Run the final collector, then renderer with the explicit namespaced result root. Check three replicas, two profile jobs, workload token-delta limits, bootstrap intervals, order sensitivity, and eligibility gates. Visually inspect the HTML and run the automated public-artifact sanitization test before committing it.

- [ ] **Step 10: Commit and push reproducible evidence**

Commit only small manifests, aggregate JSON/CSV, incident summaries, config/script snapshots, and static HTML. Do not commit checkpoints, Hugging Face blobs, raw `.nsys-rep`, full worker logs, private paths, or secrets. Push the feature branch and summarize component effects only after `performance_claim_eligible=true`.
