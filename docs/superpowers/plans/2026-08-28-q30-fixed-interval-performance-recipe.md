# Q30 Fixed-Interval Performance Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 200-step Qwen3-30B-A3B Math GRPO studies for DFlash and DSpark with drafter updates every 5, 10, or 20 policy steps while preserving the official 4n4g performance recipe.

**Architecture:** Each experiment config inherits `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` and overlays only 200 steps, local target/tokenizer paths, cadence observability, and one drafter plus its update schedule. The launcher provides output/W&B locations but does not override workload, parallelism, vLLM scheduling, or CUDA Graph settings.

**Tech Stack:** NeMo-RL, Hydra/OmegaConf, Pydantic, Bash, pytest, SLURM, W&B.

**Spec:** User-approved request in this session: fixed intervals 5/10/20, 200 steps, official performance recipe unchanged except for DFlash/DSpark drafter configuration.

## Global Constraints

- Use the isolated worktree `/Users/sna/Nemo-RL_Qwen3_Roadmap/.worktrees/dflash-dspark-cadence-latest-main-20260826`.
- Keep product source and its SHA immutable in the launcher.
- Use `nemotron_n3_post`, 4 nodes × 4 GPUs, and W&B project `sna-specdec`.
- Never write the W&B API key to a file, config, manifest, receipt, or log.
- Scheduler dry-run must pass before each actual submission.
- Monitor the first interval canary through Step 2 before submitting the remaining matrix.

---

### Task 1: Encode the six interval contracts

**Files:**
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_contract.py`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dflash-fixed5.yaml`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dflash-fixed10.yaml`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dflash-fixed20.yaml`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dspark-fixed5.yaml`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dspark-fixed10.yaml`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dspark-fixed20.yaml`

**Interfaces:**
- Consumes: the official Q30 4n4g recipe and `DraftUpdateScheduleConfig`.
- Produces: six JSON/YAML overlays named `<drafter>-fixed<interval>` with `fixed_interval` equal to 5, 10, or 20.

- [x] **Step 1: Write the failing contract**

```python
INTERVALS = (5, 10, 20)
VARIANTS = tuple(
    f"{drafter}-fixed{interval}"
    for drafter in ("dflash", "dspark")
    for interval in INTERVALS
)

def test_configs_only_overlay_interval_drafter_fields():
    for variant in VARIANTS:
        config = config_for(variant)
        assert config["grpo"] == {"max_num_steps": 200}
        assert "cluster" not in config
        assert "megatron_cfg" not in config["policy"]
        assert set(config["policy"]["generation"]["vllm_kwargs"]) == {
            "speculative_config"
        }
```

- [x] **Step 2: Verify RED**

Run: `uv run --no-project --with pytest --with pydantic python -m pytest -q experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_contract.py`

Expected: FAIL because fixed5/fixed20 files do not exist and fixed10 still overrides the workload.

- [x] **Step 3: Write minimal overlays**

Each config must contain the same target/tokenizer and drafter geometry as the existing arm, plus:

```json
"grpo": {"max_num_steps": 200},
"policy": {
  "draft": {
    "enabled": true,
    "optimizer": {"lr": 5e-6, "min_lr": 5e-7, "weight_decay": 0.01},
    "update_schedule": {
      "mode": "fixed",
      "action": "sparse_update",
      "fixed_interval": 5
    }
  }
}
```

Use 10 and 20 for their respective arms. Do not encode GBS, TP/EP, sequence length, cluster, data-plane, checkpoint policy, max OSL, vLLM `max_num_seqs`, or compilation configuration.

- [x] **Step 4: Verify GREEN**

Run the command from Step 2 and expect all contract tests to pass.

### Task 2: Preserve performance runtime in composition and launcher

**Files:**
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/verify_composed_configs.py`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh`

**Interfaces:**
- Consumes: the six overlays from Task 1.
- Produces: fail-closed composed-config validation and six allowlisted SLURM variants.

- [x] **Step 1: Add failing launcher assertions**

Assert rendered drivers contain none of `max_num_seqs=`, `compilation_config.backend=`, `compilation_config.cudagraph_mode=`, or `compilation_config.cudagraph_capture_sizes=`.

- [x] **Step 2: Verify RED**

Run the contract test and expect failure on the existing launcher overrides.

- [x] **Step 3: Remove workload/runtime overrides and validate inherited values**

The verifier must assert prompts 64, generations 32, GBS 2048, max sequence 4096, TP1/EP16/PP1/CP1, sequence parallel false, packing/fused loss true, validation period 10, checkpointing disabled, and the Triton MoE backend. The launcher must retain only cadence result location plus logger/output overrides.

- [x] **Step 4: Verify GREEN**

Run contract tests, `ruff check`, `bash -n`, and `git diff --check`; expect zero failures.

### Task 3: Publish, validate, and submit the 200-step matrix

**Files:**
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/README.md` if present; otherwise create it.

**Interfaces:**
- Consumes: verified configs and launcher.
- Produces: committed source, scheduler receipts, job IDs, W&B runs, and durable logs.

- [x] **Step 1: Document the exact inherited recipe and cadence matrix**

Record all six variants and clarify that fixed interval N means an online drafter update every N policy steps, while frozen fixed is a separate study.

- [ ] **Step 2: Commit and push exact files**

Run `git commit -s` with a focused message and push the current branch.

- [ ] **Step 3: Update the remote harness and run scheduler validation**

Fast-forward the clean remote harness to the exact commit, run state-dict/composition preflight, and invoke `--test-only dflash-fixed5`.

- [ ] **Step 4: Submit and monitor the canary**

Pipe the local W&B key through stdin, submit `dflash-fixed5`, and monitor CUDA Graph, Step 1, and Step 2 gates without exposing the secret.

- [ ] **Step 5: Submit the remaining five 200-step arms**

Only after the canary crosses Step 2 without OOM, run scheduler validation and submit DFlash fixed10/fixed20 and DSpark fixed5/fixed10/fixed20.
