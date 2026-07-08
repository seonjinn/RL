# NeMo-RL vLLM 0.24 Upgrade Validation

This experiment validates the vLLM 0.24 upgrade with unchanged upstream
NeMo-RL performance recipes on AWS-DFW and Pre-Tyche. Cluster-specific
overrides only set the run length, disable checkpoint writes, preserve CUDA
Graph execution, and separate logs and W&B runs.

| Label | Upstream recipe | Nodes | GPUs/node | Segment | Max sequence |
|---|---|---:|---:|---:|---:|
| qwen30ba3b | `grpo-qwen3-30ba3b-4n4g.yaml` | 4 | 4 | 4 | 4,096 |
| qwen32b | `grpo-qwen3-32b-4n4g.yaml` | 4 | 4 | 4 | 4,096 |
| qwen235b | `grpo-qwen3-235b-16n4g.yaml` | 16 | 4 | 16 | 8,192 |

Run scheduler validation before submission:

```bash
experiments/vllm_024_upgrade/submit_performance_step10.sh test-only all
experiments/vllm_024_upgrade/submit_performance_step10.sh submit all
```

The launcher reads the W&B key from `WANDB_API_KEY` or from the file named by
`WANDB_API_KEY_FILE`. It can also read the private `.netrc` created by the
cluster-setup skill when `WANDB_NETRC_HOME` is set. It never stores the key in
the repository or job logs. Set `USE_GRES=true` on OCI-HSG and AWS-DFW; keep it
false on Pre-Tyche.

## Eagle-3 DynamicSD on AWS-DFW

The DynamicSD cohort compares target-only decoding, fixed Eagle-3 K5/K7/K9, and
Eagle-3 with scheduler-batch-size-dependent K for Qwen3-30B-A3B, Qwen3-32B, and
Qwen3-235B-A22B.
All three variants use the same upstream performance recipe, vLLM 0.24,
`temperature=1.0`, `top_p=1.0`, and PIECEWISE CUDA Graphs. Keep this cohort
separate from historical eager-mode or full-CUDA-graph measurements.

Render all fifteen commands locally:

```bash
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  dry-run all all
```

On AWS-DFW, validate scheduling and run the two-step gate before the full
matrix:

```bash
MAX_STEPS=2 RUN_TAG=vllm024-dynamicsd-smoke-20260707 \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  test-only all all

MAX_STEPS=2 RUN_TAG=vllm024-dynamicsd-smoke-20260707 \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit all all
```

Each smoke must reach Step 2 policy training, expose positive draft and
acceptance counters for both Eagle-3 variants, and exit successfully. Then
submit the 20-step matrix:

```bash
MAX_STEPS=20 RUN_TAG=vllm024-dynamicsd-step20-20260707 \
experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit all all
```

Collect the completed W&B histories with:

```bash
uv run --no-sync python \
  experiments/vllm_024_upgrade/summarize_eagle3_dynamicsd.py \
  --manifest experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/submissions.tsv \
  --entity nvidia \
  --project nemorl-vllm024-dynamicsd-aws-dfw \
  --output-dir experiments/vllm_024_upgrade/runs/vllm024-dynamicsd-step20-20260707/summary
```

Final means use Steps 2-20. Every SpecDec row is matched only to the baseline
with the same model, recipe, graph mode, sampling, topology, container, and
commit. The summary reports generation and E2E time/throughput speedups,
acceptance rate, mean accepted length, job IDs, W&B links, and reward/response
length/KL health gates.

## Lyris Recipe and 32K Long-Tail Gate

The Lyris cohort uses the same three upstream synchronous performance recipes
and compares only the matched `baseline`, `eagle3_k5`, and `dynamic` triplet.
The `recipe` profile preserves every recipe-owned sequence limit. The
`longtail32k` profile sets `max_new_tokens=32768` and total/model length to
`36864`. Both profiles explicitly set vLLM `max_num_batched_tokens=16384`,
which includes scheduled draft tokens, and use draft TP1. Lyris jobs use the
`coreai_dlalgo_llm` account, the `gb200` partition, no `--gres`, and one segment
per allocated node.

Set the cluster paths once in the dedicated Lyris worktree:

```bash
export LYRIS_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna
export REPO_DIR=${LYRIS_ROOT}/RL-vllm024-dynamicsd-20260708
export CONTAINER=${LYRIS_ROOT}/containers/nemo_rl_nightly.sqsh
export HF_HOME=${LYRIS_ROOT}/hf_home
export WANDB_API_KEY_FILE=${LYRIS_ROOT}/.secrets/wandb_api_key
export WANDB_PROJECT=nemorl-vllm024-dynamicsd-lyris
export ACCOUNT=coreai_dlalgo_llm
export PARTITION=gb200
export USE_GRES=false
```

Run scheduler validation for each profile before allocating GPUs:

```bash
PROFILE=recipe MAX_STEPS=1 RUN_TAG=vllm024-dynamicsd-lyris-recipe-smoke-20260708 \
  experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  test-only all core

PROFILE=longtail32k MAX_STEPS=1 \
  RUN_TAG=vllm024-dynamicsd-lyris-longtail32k-smoke-20260708 \
  experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  test-only all core
```

Submit the one-step smoke cohorts with the same run tags by replacing
`test-only` with `submit`. A model/profile triplet can advance only after all
three jobs exit successfully, the SpecDec jobs report positive
`spec_num_drafts`, `spec_num_draft_tokens`, and `spec_num_accepted_tokens`, the
resolved vLLM config reports `max_num_batched_tokens=16384`, and logs contain no
CUDA Graph fallback error. Use a distinct final run tag for each passing
profile:

```bash
PROFILE=recipe MAX_STEPS=20 RUN_TAG=vllm024-dynamicsd-lyris-recipe-step20-20260708 \
  experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit all core

PROFILE=longtail32k MAX_STEPS=20 \
  RUN_TAG=vllm024-dynamicsd-lyris-longtail32k-step20-20260708 \
  experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit all core
```

Keep the recipe and long-tail manifests separate. The collector rejects any
baseline/SpecDec comparison whose profile, sequence limits, scheduler token
budget, topology, container, commit, or recipe differs.
