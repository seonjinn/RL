# Q35 KV cache grouping isolation

On Q35 at gmu 0.8, each DFlash K3 step costs about 2.4x a baseline step, while DSpark K4 costs about 1.8x. The two drafters have the same size and depth. DFlash has one full-attention layer that cannot share a KV spec with the target, so vLLM 0.29's group-size heuristic falls to 1 and splits the KV pool into 46 groups. DSpark gets 8 groups and the baseline gets 4. vLLM builds attention metadata once per group on every step.

`bench.py` runs vLLM in-process on one GPU per arm. It uses the container's vLLM worker venv and the same engine settings as the GRPO arms. The prompts are OpenMathInstruct-2 with the GRPO CoT template and `ignore_eos`, so every request decodes 2048 tokens. The arms:

| Arm | Change |
|---|---|
| `dflash_k3` | DFlash K3 as in the GRPO runs |
| `dflash_k3_g5` | same, with vLLM's KV group size forced to 5 (10 groups). Only the block layout changes. |
| `dspark_k4` | DSpark K4 as in the GRPO runs |
| `baseline`, `baseline_triton` | no SpecDec, with the default backend and with TRITON_ATTN |
| `dflash_k7` | DFlash K7 as in the GRPO runs |
| `*_blhnc` | the same arm with `VLLM_KV_CACHE_LAYOUT=BLHNC`, which uses vLLM's packed grouping planner. On the hybrid GDN target it fails at startup, because this layout needs dense, unpadded pages. |
| `*_reuse` | the same arm with the vLLM 0.29 backport of the upstream GDN metadata reuse ([`../vllm_gdn_metadata_reuse`](../vllm_gdn_metadata_reuse/README.md)) |
| `*_reuse_verify` | the `*_reuse` arm. For each reused GDN group it also runs the full `build()` and compares every metadata field with `update_block_table()`. |
| `<arm>@<N>` | the same arm again in a new process, under another name |

Spec arms capture the CUDA graph sizes of the GRPO config with the same K (`../configs/qwen3_30ba3b/capture_k<K>_s128.yaml`).

Each arm also decodes 16 prompts greedily twice and records the top-2 logprobs. `summarize.py` compares the two runs, which gives the in-process noise floor. It also compares each `*_reuse`, `*_reuse_verify` and `<arm>@<N>` run with its base arm. Where two runs split, a near-tie top-2 margin points at numeric noise, and a large margin points at a logic difference.

If `dflash_k3_g5` closes most of the step-time gap to `dspark_k4`, the grouping is the cause.

Usage: `cd` to this directory on OCI-HSG and run `sbatch run.sbatch [lane ...]`. Each lane is a quoted, space-separated list of arms that run in order on one GPU, for example `sbatch run.sbatch "dflash_k3_blhnc dflash_k7_blhnc" "baseline_blhnc baseline"`. The per-arm JSON and logs go to `results/run_<ts>_<jobid>/`. `BENCH_ARGS=--greedy-only` skips the throughput phases. `python summarize.py <results dir>` prints the tables.