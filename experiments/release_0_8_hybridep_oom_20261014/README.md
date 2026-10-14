# Release 0.8 HybridEP OOM reproduction on CW-DFW

Source: `NVIDIA-NeMo/RL` commit `19666f86ca3b1988880b2b8b9107c575608c4ab0` on `r0.8.0`.

Container: `/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs` (provided for this investigation). The job records the container path and a separate CPU job records its SHA256.

Run the smoke job first, then the unchanged recipe with `hybridep` for 20 steps. If it fails in policy training, repeat with only `dispatcher=deepep` changed. The two recipe allocations are MiniMax M2.7 on 8 nodes × 8 H100 and Qwen3.5 35B A3B on 4 nodes × 8 H100. Keep each attempt's logs, including failures.

Both checkpoints are present under the shared `HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/nemo-rl/hf_home` provided for the experiment. The attempted prefetch through a personal cache failed because its HF token had expired. Run jobs from this checkout after `git pull --ff-only`.

```bash
mkdir -p /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/slurm
sbatch --test-only experiments/release_0_8_hybridep_oom_20261014/cw_image_smoke.sbatch
sbatch experiments/release_0_8_hybridep_oom_20261014/cw_image_smoke.sbatch
PARTITION=batch_short WALLTIME=02:00:00 bash experiments/release_0_8_hybridep_oom_20261014/cw_submit.sh qwen hybridep 2
PARTITION=batch_short WALLTIME=02:00:00 bash experiments/release_0_8_hybridep_oom_20261014/cw_submit.sh qwen deepep 2
bash experiments/release_0_8_hybridep_oom_20261014/cw_submit.sh minimax hybridep 2
bash experiments/release_0_8_hybridep_oom_20261014/cw_submit.sh minimax deepep 2
OPTIMIZER_FOREACH=false bash experiments/release_0_8_hybridep_oom_20261014/cw_submit.sh minimax hybridep 2
PARTITION=backfill WALLTIME=08:00:00 bash experiments/release_0_8_hybridep_oom_20261014/cw_submit.sh qwen hybridep 20
PARTITION=backfill WALLTIME=08:00:00 bash experiments/release_0_8_hybridep_oom_20261014/cw_submit.sh minimax hybridep 20
PARTITION=backfill WALLTIME=08:00:00 bash experiments/release_0_8_hybridep_oom_20261014/cw_submit.sh minimax deepep 20
POLICY_MAX_TOTAL_SEQUENCE_LENGTH=6144 GENERATION_MAX_NEW_TOKENS=4096 OPTIMIZER_FOREACH=false bash experiments/release_0_8_hybridep_oom_20261014/cw_submit.sh minimax hybridep 2
```

Set `DEPENDENCY=afterany:<job_id>` to run a control after its baseline. The submit script performs `sbatch --test-only` before each submission and disables checkpoint saving and W&B for the short reproduction. It leaves the model, batch sizes, sequence lengths, EP sizes, and activation checkpointing at the release values unless the explicit foreach control is selected. The MiniMax foreach control tests the first-step AdamW OOM seen in the original driver log; Qwen uses Transformer Engine FusedAdam and does not accept that control.

The user accepts partial progress within `batch`'s four-hour limit for 20-step runs. Qwen's first step took 820.52 seconds, so a complete 20-step run may exceed that limit. The optional MiniMax short-sequence control reduces the policy total length from 10,240 to 6,144 tokens and the generation maximum from 8,192 to 4,096 tokens, while retaining a 2,048-token prompt allowance. This is a memory mitigation experiment, not a reproduction of the original full-length recipe.

The driver runs from `/opt/nemo-rl` inside the pinned image. The image contains the full Gym submodule and its Git HEAD resolves to `19666f86ca3b1988880b2b8b9107c575608c4ab0`. The CW checkout stores the submit scripts and manifests; its Gym submodule is not initialized, so `uv run` from that checkout fails before training.

The image already contains the base and actor environments. The driver uses `uv run --no-sync` and leaves `NRL_FORCE_REBUILD_VENVS` unset; rebuilding in the Pyxis overlay filled `/root/.cache/uv` during the first CPU preflight.

The shared `HF_HOME` has an expired token. The submit script sets `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` so public model and dataset requests do not send it. CPU preflight `20215255` loaded both model tokenizers/configs and the two DAPO datasets with this setting. Global `HF_HUB_OFFLINE=1` loads the models but prevents `datasets.load_dataset()` from resolving the DAPO datasets, so it is unset for training.
