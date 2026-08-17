# Synchronous CUDA IPC BF16 versus MXFP8 rollout

This experiment compares two synchronous, colocated NeMo-RL runs while keeping
policy training in BF16:

- BF16 rollout
- routed-expert FC1/FC2 MXFP8 rollout

Both arms use FlashInfer TRTLLM MoE kernels and request CUDA Graph capture.
Colocation plus `policy.generation.refit_transport=null` selects the legacy
synchronous CUDA IPC/ZMQ weight-update path. It does not instantiate the newer
`IPCWeightSynchronizer`; `ipc` is not a valid public config value.

The first gate uses one Qwen3-30B-A3B step. A successful gate is followed by a
20-step A/B. Nemotron3 Nano uses the same contract and can run in parallel after
the common smoke gate passes.

## Local contract test

```bash
python3 -m pytest experiments/sync_ipc_bf16_mxfp8_ab/test_launcher.py -q
```

## Render one resolved launch

```bash
ACTION=render MODEL=qwen30 ARM=mxfp8 MAX_STEPS=20 \
  bash experiments/sync_ipc_bf16_mxfp8_ab/submit.sh
```

Cluster submission requires `REPO`, `CONTAINER`, `HF_HOME`, `BASE`, `PARTITION`,
and `SLURM_ACCOUNT` to describe the selected reproducible environment. Keep
`USE_GRES=false` for exclusive-node Lyris/Ptyche partitions; set it to `true`
only on clusters that require an explicit GPU GRES request.

## Analyze completed runs

Copy each run's `metrics.json` and `metadata.env` under
`results/<model>/<arm>/`, then run:

```bash
uv run --no-project --with seaborn --with pandas --with matplotlib \
  python experiments/sync_ipc_bf16_mxfp8_ab/analyze_results.py \
  --results-root experiments/sync_ipc_bf16_mxfp8_ab/results \
  --report-root experiments/sync_ipc_bf16_mxfp8_ab/report
```

The completed 20-step comparison is summarized in [the report](report/README.md).
