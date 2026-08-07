# PR 3477 Nemotron 3 Nano smoke

This experiment validates the exact PR 3477 head with BF16 Megatron training,
MXFP8 vLLM rollout, non-colocated workers, and NCCL reshard refit.

The first gate is a two-step smoke on four GCP-NRT B200 nodes. The topology is
two trainer nodes plus two generation nodes, with eight GPUs per node. vLLM
uses TP1 to avoid further partitioning Nano's 1856-wide expert intermediate
dimension.

Initialize all workspaces before submission:

```bash
git submodule update --init --recursive
```

Run scheduling validation before submission:

```bash
ACTION=test-only MAX_STEPS=2 ./experiments/pr3477_nano_smoke/submit_gcp_nrt.sh
```

Submit only after the test-only request succeeds:

```bash
ACTION=submit MAX_STEPS=2 ./experiments/pr3477_nano_smoke/submit_gcp_nrt.sh
```

## Exact-head result

- PR parent: `6f57c1b79504245fc8211028e504465045315f34`
- Experiment branch: `sna/pr3477-nano-smoke-20260807`
- GCP-NRT job: `502962`
- W&B: <https://wandb.ai/nvidia/sna-pr3477-nano-smoke/runs/eyk4cm10>
- Result: failed during initial vLLM model loading, before NCCL reshard refit

All 16 generation workers failed in
`process_weights_after_loading_mxfp8_moe()` when FlashInfer's
`shuffle_matrix_sf_a()` asserted that the row dimension must be divisible by
the 128-row epilogue tile. Nemotron 3 Nano uses a non-gated expert intermediate
size of 1856, and `1856 % 128 == 64`. TP1 therefore does not remove the need for
the existing vLLM intermediate-size padding patch. This is a Nano MXFP8 model
loading prerequisite rather than evidence that the new NCCL reshard transfer
itself failed.
