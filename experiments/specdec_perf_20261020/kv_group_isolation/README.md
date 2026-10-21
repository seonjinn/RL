# Q35 KV cache grouping isolation

On Q35 at gmu 0.8, each DFlash K3 step costs about 2.4x a baseline step, while DSpark K4 costs about 1.8x. The two drafters have the same size and depth. DFlash has one full-attention layer that cannot share a KV spec with the target, so vLLM 0.29's group-size heuristic falls to 1 and splits the KV pool into 46 groups. DSpark gets 8 groups and the baseline gets 4. vLLM builds attention metadata once per group on every step.

`bench.py` runs vLLM in-process on one GPU per arm. It uses the container's vLLM worker venv and the same engine settings as the GRPO arms. The prompts are OpenMathInstruct-2 with the GRPO CoT template and `ignore_eos`, so every request decodes 2048 tokens. The arms:

| Arm | Change |
|---|---|
| `dflash_k3` | DFlash K3 as in the GRPO runs |
| `dflash_k3_g5` | same, with vLLM's KV group size forced to 5 (10 groups). Only the block layout changes. |
| `dspark_k4` | DSpark K4 as in the GRPO runs |
| `baseline`, `baseline_triton` | no SpecDec, with the default backend and with TRITON_ATTN |

If `dflash_k3_g5` closes most of the step-time gap to `dspark_k4`, the grouping is the cause.

Usage: `cd` to this directory on OCI-HSG and run `sbatch run.sbatch`. The per-arm JSON and logs go to `results/run_<ts>_<jobid>/`.