# Nano policy-training Nsight comparison

This experiment explains why routed-expert MXFP8 training does not reach the
1.5x kernel-level speedup observed in a separate workload.

## Comparison

The primary pair changes only policy-training precision:

| Arm | Policy training | Rollout | Purpose |
| --- | --- | --- | --- |
| `bf16-train-mxfp8-rollout` | BF16 | MXFP8 | Policy control |
| `mxfp8-train-mxfp8-rollout` | Routed-expert MXFP8, `fp8_param: false` | MXFP8 | Candidate |

Both arms use the same Nano model, seed, topology, data order, rollout backend,
CUDA Graph setting, and NCCL Reshard refit path. Nsight captures policy steps 3
and 4. The run continues through step 6 so capture shutdown does not overlap the
job exit.

The BF16-training plus BF16-rollout arm is useful as an end-to-end reference,
but it is not the policy-kernel control because it can produce a different
training batch.

## Submit

Run from this repository checkout on OCI-HSG:

```bash
export REPO=$PWD
export CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_6916783.sqsh
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
export WANDB_HOME=$HOME
export RESULT_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/results/nano-policy-nsys
export SLURM_ACCOUNT=nemotron_n4_post

ACTION=test-only ./research/nano_policy_nsys/submit_matched.sh
ACTION=submit ./research/nano_policy_nsys/submit_matched.sh
```

The launcher keeps source in `/home`, caches in `/raid/scratch`, and only the
durable logs and `.nsys-rep` files in `/lustre`.

## Analysis gates

1. Confirm both arms ran the same source SHA and configuration except training
   precision.
2. Compare policy input token counts and sequence-length distributions before
   interpreting elapsed time.
3. Use all policy ranks to separate useful compute from collective wait time.
4. Break active GPU time into routed-expert GEMM, non-expert GEMM, Mamba,
   communication, optimizer, quantization, and uncategorized kernels.
5. Report both measured policy throughput and the Amdahl upper bound implied by
   the measured routed-expert share.

If token counts differ, throughput is the headline metric. Raw policy latency is
reported only with the token-count difference beside it.
