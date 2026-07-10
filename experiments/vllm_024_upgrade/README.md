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

## Tail-Gated Eagle-3 Matrix on Lyris

The tail-gated matrix keeps the Qwen3-30B-A3B and Qwen3-32B four-node upstream
performance recipes matched. It exposes only three Model Runner V1 arms
(`baseline_v1`, `always_on_v1_k5`, `stock_dynamic_v1`) and four Model Runner V2
arms (`baseline_v2`, `always_on_v2_k5`, `fastrl_threshold_v2_k5`,
`efficient_roofline_v2_k5`). V1 uses PIECEWISE graphs; V2 uses
FULL_AND_PIECEWISE graphs. Do not compare performance across these runner
cohorts.

Render or validate the complete matrix before any submit:

```bash
experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh dry-run all all
experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh test-only all all
```

The launcher pins 64 prompts by 32 generations, train GBS 512, 4,096-token
sequence and output limits, a 4,128-token engine limit, and a 16,384-token
scheduler budget. It records runner, graph, gate, calibration, W&B, container,
commit, recipe, and Slurm provenance in the Lustre `submissions.tsv` manifest.
`submit` runs Slurm validation before `sbatch`, requires a clean and already
pushed `sna/` branch, and never pushes a remote itself.

Create a roofline input from measured component latencies before running the
roofline arm. The CSV must carry a single model/TP/cluster/container identity,
the EfficientRollout constants, and `B,S,K,T_T,T_D,T_V` columns:

```bash
uv run --no-sync python experiments/vllm_024_upgrade/calibrate_tail_gate.py \
  --input /lustre/.../tail_gate_measurements.csv \
  --output-dir /lustre/.../tail_gate_calibrations
```

The converter writes a deterministic EfficientRollout-compatible JSON and a
SHA256 sidecar for each calibration identity.

## Baseline/SpecDec Token and Logprob Parity on Lyris

The parity gate runs Qwen3-32B target-only and Eagle-3 K5 through the NeMo-RL
`VllmGeneration` adapter with vLLM 0.24, standard rejection sampling, PIECEWISE
CUDA Graphs, target TP2, draft TP1, chunked prefill enabled, prefix caching
disabled, and a 16,384-token scheduler budget. Greedy jobs use one sample per
prompt. Sampled jobs use 64 independent samples per prompt at temperature 1.0
and top-p 1.0. Both variants use NeMo-RL's deterministic topology-derived
worker seed; the runner does not inject a second vLLM seed keyword.

Render or validate the four-job matrix before submission:

```bash
experiments/vllm_024_upgrade/submit_generation_parity.sh dry-run all all
experiments/vllm_024_upgrade/submit_generation_parity.sh test-only all all
```

Submit after the launcher commit is present on a remote branch:

```bash
experiments/vllm_024_upgrade/submit_generation_parity.sh submit all all
```

Each batch appends and flushes generated token IDs and chosen-token logprobs to
`samples.jsonl`, so a timeout still leaves valid completed rows. `metadata.json`
records the resolved generation contract, commit, SLURM job ID, throughput, and
SpecDec counters. Reward parity remains a separate matched GRPO smoke gate; the
standalone parity producer intentionally does not synthesize an RL reward.
