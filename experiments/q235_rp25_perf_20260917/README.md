# Qwen3-235B RP25 Speculative-Decoding Performance Study

This experiment compares the official Qwen3-235B-A22B Math GRPO performance
recipe against matched frozen DFlash, DSpark, and EAGLE-3 drafters on Lyris.

## Controlled baseline

- Recipe: `grpo-qwen3-235b-16n4g.yaml`
- Topology: 16 nodes, 4 GB200 GPUs per node, segment size 16
- Workload inherited without override: 16 prompts × 32 generations, 8192-token
  maximum sequence length, training TP2/PP4/CP2/EP16, generation TP8
- Runtime: BF16, the official recipe's `triton` MoE backend, `enforce_eager=false`,
  `FULL_AND_PIECEWISE`
- The launcher preserves the official performance wrapper's
  `NCCL_NVLS_ENABLE=0` workaround.
- It also sets `NRL_DISABLE_NUMA_MEMBIND=1`: CPU affinity remains enabled, but
  policy allocations may spill across the two Grace NUMA sockets. Without this,
  four TP8 vLLM workers placed about 320 GiB on one socket and OOM-killed the
  two policy ranks bound to that socket even though the node still had more
  than 400 GiB available on its other socket.
- It sets Ray's host-memory protection threshold to 98%. The default 95%
  threshold killed the DFlash K5 recovery run during Step 12 refit after total
  node usage exceeded the threshold by only 19.7 MB, while approximately
  44.6 GiB of physical host memory was still available. The memory monitor
  remains enabled; this does not disable host-memory protection.
- Baseline CUDA Graph sizes: `[1, 2, 4, 8, 16, 32, 64]`
- Baseline has no `max_num_seqs` override and no speculative decoder
- SpecDec arms use `max_num_seqs=64`. Their FAP CUDA Graph shapes cover powers
  of two request buckets through 64 multiplied by the target verification
  width `K+1`; DSpark additionally covers its draft query width `K`.
- Duration: 20 steps; final comparison window is steps 3–20

## Matched drafter matrix

The DFlash and DSpark exports were trained for the base Qwen3-235B-A22B target
with the PTV2-en B8 recipe through step 25391. B8 exports are used for K5/K7,
while B16 exports are used for K11/K13.

Lyris staging root:
`/lustre/fsw/coreai_dlalgo_llm/users/sna/drafters/q235-base-ptv2en-s25391-20260917`

| Arm | Export | K values |
|---|---|---|
| DFlash B8 | `dflash-b8` | 5, 7 |
| DSpark B8 | `dspark-b8` | 5, 7 |
| DFlash B16 | `dflash-b16` | 11, 13 |
| DSpark B16 | `dspark-b16` | 11, 13 |
| EAGLE-3 | RedHatAI Qwen3-235B-A22B speculator snapshot | 3, 5 |

Every non-baseline arm first runs as a one-step gate. Only gates that finish
with exit code zero and produce valid W&B metrics are promoted to 20 steps.
DSpark uses the source-verified vLLM #48167 FAP compatibility overlay. Megatron
dataset helpers and the vLLM overlay are built under `/raid/scratch`, not in the
shared source checkout.

## Commands

```bash
python3 experiments/q235_rp25_perf_20260917/launch.py \
  --site lyris --arm baseline --steps 20 --render

python3 experiments/q235_rp25_perf_20260917/launch.py \
  --site lyris --arm dflash_k5 --steps 1 --test-only

python3 experiments/q235_rp25_perf_20260917/launch.py \
  --site lyris --arm dflash_k5 --steps 1 --submit

# Matched September 16 nightly baseline on Ptyche
python3 experiments/q235_rp25_perf_20260917/launch.py --site ptyche --render
python3 experiments/q235_rp25_perf_20260917/launch.py --site ptyche --submit
```

Valid arms are `baseline`, `dflash_k5`, `dflash_k7`, `dspark_k5`,
`dspark_k7`, `dflash_b16_k11`, `dflash_b16_k13`, `dspark_b16_k11`,
`dspark_b16_k13`, `eagle3_k3`, and `eagle3_k5`. The launcher always runs
`sbatch --test-only` before a real submission and stores the rendered job,
overrides, source SHA, submodule receipt, and container recipe hash with the run
artifacts. W&B project is `nvidia/sna-specdec`, group
`q235-rp25-frozen-perf-20260917`.

The Ptyche site uses the staged immutable target directory
`Qwen3-235B-A22B-8efa61729e24bd65b1d152b5ab5409052aa80e65` and the
September 16 nightly container. It retains the official baseline workload and
does not add a `max_num_seqs` override.
