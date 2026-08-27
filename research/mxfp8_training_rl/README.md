# MXFP8 training in RL

This experiment validates MXFP8 training compute with BF16 parameter storage
(`fp8_param: false`) and MXFP8 vLLM rollout. It covers Qwen3-30B-A3B and
Nemotron-3 Nano on GB200.

Run a two-step smoke test first:

```bash
MODEL=qwen30 MAX_STEPS=2 ACTION=test-only ./submit_oci_hsg.sh
MODEL=qwen30 MAX_STEPS=2 ACTION=submit ./submit_oci_hsg.sh

MODEL=nano MAX_STEPS=2 ACTION=test-only ./submit_oci_hsg.sh
MODEL=nano MAX_STEPS=2 ACTION=submit ./submit_oci_hsg.sh
```

After both jobs complete, use `MAX_STEPS=20` for performance measurements. The
steady-state summary should average steps 2 through 19.

The launcher requires `REPO`, `CONTAINER`, `HF_HOME`, `WANDB_HOME`,
`RESULT_ROOT`, and `SLURM_ACCOUNT`. It stores source in `/home`, worker virtual
environments and JIT caches in `/raid/scratch`, and durable logs in `/lustre`.
