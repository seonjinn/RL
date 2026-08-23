# MXFP8 fused refit spike

This experiment tests whether the remaining synchronous MXFP8 refit cost can
be reduced by combining or removing work between trainer-side quantization and
generation-side expert layout preparation.

The first run profiles step 3 of the full PR 3294 Qwen3-30B-A3B synchronous,
colocated CUDA IPC path on GB200. CUDA Graphs remain enabled. Nsight Systems
captures policy and vLLM workers, and ntrace consumes the resulting report for
kernel and call-stack attribution.

The implementation gate is deliberately narrow:

- preserve the live vLLM parameter addresses used by CUDA Graph replay;
- add no full-model prepared-weight copy;
- keep extra steady-state GPU storage below 4 GiB per generation rank;
- retain bitwise parity with the current batched expert shuffle;
- proceed to a 20-step A/B only if the isolated refit path improves by at least
  10% or 0.5 seconds.

```bash
ACTION=test-only ./experiments/mxfp8_fused_refit_spike/submit_oci_hsg.sh
ACTION=submit ./experiments/mxfp8_fused_refit_spike/submit_oci_hsg.sh
```

