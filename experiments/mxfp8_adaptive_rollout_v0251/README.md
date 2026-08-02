# NeMo-RL vLLM 0.25.1 MXFP8 Safe Adaptive canary

This experiment answers two separate questions:

1. Can current NeMo-RL main load and run the corrected custom vLLM 0.25.1
   Safe Adaptive backend with CUDA Graph enabled?
2. Does the offline table add work to each rollout request?

The table is loaded once during worker initialization. Generation performs an
exact dictionary lookup and never runs the offline shmoo. The first canary
compares CuTeDSL and Safe Adaptive in the same two-node allocation. A refit
canary is added only after this gate passes because backend-specific prepared
weights may need rebuilding after a weight update.

The first gate deliberately uses eight checked-in prompts, a 4K model length,
and 128 generated tokens. It preserves the TP8/EP8 async actor and CUDA Graph
paths while minimizing the cost of detecting import, layout, and engine-start
failures. It is not a production throughput measurement.

## Required environment

```bash
export NEMO_RL_REPO_ROOT=$(pwd -P)
export CUSTOM_VLLM_SOURCE=/home/sna/mxfp8-safe-backend/vllm-v0251-safe-backend
export EXPECTED_VLLM_COMMIT=658d7b1571a914bee7df48f717c2a428ee7c45ad
export MODEL_PATH=/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8
export TACTIC_FILE=/home/sna/mxfp8-safe-backend/vllm-benchmark-v0251-safe/experiments/sweep/data/microbench/mxfp8_v0251_safe_backend_artifacts_20260801_r6_robust/exact_tactics.json
export TACTIC_SHA256=d5681371ea2476c3732d58089148e13123165b9e740d3e32ddec98d6eca40a1d
export LAYER_ALLOWLIST_B64=MTI4MCw4MTkyCjIwNDgsODE5Mgo0Mzg0LDgxOTIKODE5MiwxMDI0CjgxOTIsMTI4MAo4MTkyLDIwNDgK
export CANARY_RESULT_ROOT=/home/sna/results/nemorl-v0251-mxfp8-safe-adaptive/$(date +%Y%m%d_%H%M%S)
```

The launcher derives separate driver and actor venv paths from the `uv.lock`
hash and custom vLLM commit. Both arms reuse those same versioned environments.
It also initializes the pinned shallow workspace submodules before resolving the
locked environment.

The locked driver environment must be prepared once in the matching container.
The launcher passes its Ray binary through `RAY_CLI` so the head, worker, and
driver processes use one version, preventing a container-Ray versus lockfile-Ray
version mismatch.

Submit through the repository `ray.sub` launcher on Ptyche with two nodes,
four GPUs per node, account `coreai_dlalgo_llm`, partition `36x2-a01r`, and
`--segment=2`. The immutable production form should replace
`CUSTOM_VLLM_SOURCE` with a custom wheel baked into the NeMo-RL container.

```bash
bash experiments/mxfp8_adaptive_rollout_v0251/submit_ptyche.sh test-only
bash experiments/mxfp8_adaptive_rollout_v0251/submit_ptyche.sh submit
```
