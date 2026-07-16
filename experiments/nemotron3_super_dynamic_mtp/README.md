# Nemotron-3 Super Dynamic Native MTP

This experiment compares native MTP against a matched non-speculative control
using the Nemotron-3 Super 32-node, 4-GPU performance recipe added by NeMo-RL
PR #3207.

## Matrix

| Variant | Speculative decoding | Draft depth |
|---|---|---|
| `mtp_off` | Disabled | n/a |
| `native_mtp_k5` | Native checkpoint MTP | Fixed K=5 |
| `dynamic_native_mtp_k5` | Native checkpoint MTP | K=5 for BS 1-64, K=3 for BS 65-128, K=1 for BS 129-256 |

All three variants preserve the PR recipe's model, dataset, batch sizes,
parallelism, and 8K maximum sequence length. They use vLLM 0.24, CUDA Graphs
in `PIECEWISE` mode, the Triton MoE backend, disabled checkpoint writes, and
the same 32-node topology.

The checkpoint has one physical Nemotron-H MTP pattern. K greater than one
reuses that pattern autoregressively, matching vLLM's native MTP behavior. The
initial dynamic schedule is based on standalone K sweeps and must be refined
with the RL rollout concurrency distribution.

The training recipe keeps `mtp_num_layers: 0`. The target policy is refit after
updates, while the checkpoint MTP weights remain static. Acceptance can
therefore decline as the policy moves away from the original checkpoint.

## Run

Run a scheduling check before submission:

```bash
CLUSTER=ptyche MODE=test-only VARIANT=mtp_off \
  experiments/nemotron3_super_dynamic_mtp/submit_nemotron3_super_dynamic_mtp.sh
```

Submit each matched variant:

```bash
for variant in mtp_off native_mtp_k5 dynamic_native_mtp_k5; do
  CLUSTER=ptyche MODE=submit VARIANT="${variant}" MAX_STEPS=2 \
    experiments/nemotron3_super_dynamic_mtp/submit_nemotron3_super_dynamic_mtp.sh
done
```

For final performance results, set `MAX_STEPS=20` and compare steps 2-20.
Report E2E step time and throughput, generation time and throughput,
acceptance rate, mean accepted length, and the generation-time ratio.
