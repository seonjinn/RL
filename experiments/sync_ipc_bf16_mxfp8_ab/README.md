# Synchronous CUDA IPC BF16 versus MXFP8 rollout

This experiment compares two synchronous, colocated NeMo-RL runs while keeping
policy training in BF16:

- BF16 rollout
- routed-expert FC1/FC2 MXFP8 rollout

Both arms use FlashInfer TRTLLM MoE kernels and CUDA Graphs. Colocation plus
`policy.generation.refit_transport=null` selects `IPCWeightSynchronizer`; `ipc`
is not a valid public config value.

The first gate uses one Qwen3-30B-A3B step. A successful gate is followed by a
sequential 20-step A/B. Nemotron3 Nano runs only after the Qwen pair validates
the common IPC path.

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
