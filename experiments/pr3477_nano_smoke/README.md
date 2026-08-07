# PR 3477 Nemotron 3 Nano smoke

This experiment validates the exact PR 3477 head with BF16 Megatron training,
MXFP8 vLLM rollout, non-colocated workers, and NCCL reshard refit.

The first gate is a two-step smoke on four GCP-NRT B200 nodes. The topology is
two trainer nodes plus two generation nodes, with eight GPUs per node. vLLM
uses TP1 because Nano's 1856-wide expert intermediate dimension is not
compatible with the MXFP8 grouping requirements at TP4.

Initialize the Gym workspace before submission:

```bash
git submodule update --init --recursive 3rdparty/Gym-workspace/Gym
```

Run scheduling validation before submission:

```bash
ACTION=test-only MAX_STEPS=2 ./experiments/pr3477_nano_smoke/submit_gcp_nrt.sh
```

Submit only after the test-only request succeeds:

```bash
ACTION=submit MAX_STEPS=2 ./experiments/pr3477_nano_smoke/submit_gcp_nrt.sh
```
