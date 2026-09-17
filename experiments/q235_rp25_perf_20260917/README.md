# Qwen3-235B RP25 Speculative-Decoding Performance Study

This experiment compares the official Qwen3-235B-A22B Math GRPO performance
recipe against the RP25 step-38600 drafters staged on OCI-HSG.

## Controlled baseline

- Recipe: `grpo-qwen3-235b-16n4g.yaml`
- Topology: 16 nodes, 4 GB200 GPUs per node, segment size 16
- Workload inherited without override: 16 prompts × 32 generations, 8192-token
  maximum sequence length, training TP2/PP4/CP2/EP16, generation TP8
- Runtime: BF16, `flashinfer_trtllm`, `enforce_eager=false`,
  `FULL_AND_PIECEWISE`
- Baseline CUDA Graph sizes: `[1, 2, 4, 8, 16, 32, 64]`
- Baseline has no `max_num_seqs` override and no speculative decoder
- Duration: 20 steps; final comparison window is steps 3–20

## OCI-HSG drafter receipt

Root:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/drafters/specdec_ptv23_q235_s38600`

- DFlash2: `sd2p3rp-q235-base-ptv3rp25-dflash2-b8-16n/exported-checkpoint-38600`
- DSpark: `sd2p3rp-q235-base-ptv3rp25-dspark-b8-16n/exported-checkpoint-38600`

Both exported checkpoints contain `config.json` and `model.safetensors`. DSpark
can use the current vLLM 0.25.1 path. DFlash2 requires an isolated runtime
compatibility gate before it can enter the matched performance matrix.

## Commands

```bash
python3 experiments/q235_rp25_perf_20260917/launch.py --render
python3 experiments/q235_rp25_perf_20260917/launch.py --test-only
python3 experiments/q235_rp25_perf_20260917/launch.py --submit

# Matched September 16 nightly baseline on Ptyche
python3 experiments/q235_rp25_perf_20260917/launch.py --site ptyche --render
python3 experiments/q235_rp25_perf_20260917/launch.py --site ptyche --submit
```

The launcher always runs `sbatch --test-only` before a real submission and
stores the rendered job, overrides, source SHA, submodule receipt, and
container recipe hash with the run artifacts.

The Ptyche site uses the staged immutable target directory
`Qwen3-235B-A22B-8efa61729e24bd65b1d152b5ab5409052aa80e65` and the
September 16 nightly container. It retains the official baseline workload and
does not add a `max_num_seqs` override.
