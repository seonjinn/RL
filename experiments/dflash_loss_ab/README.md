# DFlash Drafter Loss A/B + Qwen3-235B Pipeline Screening

Trains DFlash drafters with the vllm-project/speculators toolkit on AWS-DFW GB200 and
compares training losses by expected acceptance length (EAL) on a held-out split.
Motivated by the bebop-mtp claim that TV loss beats CE for drafter training, and by the
need for a custom 235B drafter (public eagle3/dflash drafters collapse on agentic SWE
rollouts).

## Usage

All jobs run on AWS-DFW (`sna@aws-dfw-cs-001-login-01.nvidia.com`), base directory
`B=/lustre/fsw/portfolios/nemotron/users/sna/dflash_training` (venvs, speculators repo,
data, checkpoints live there; see `setup_speculators_dflash.sh` in
`../dynamic_sd_sync_rollout/` for environment bootstrap).

- `dflash_loss_ab.sbatch` - 30B-Thinking target, kl_div + nla arms, UltraChat 10K, online mode (1 node)
- `dflash_loss_ab2.sbatch` - same, ce + tv arms
- `dflash_235b_screen.sbatch` - 235B-Thinking target, 2-node (server TP4 + train x4), kl_div
- `dl_235b.sbatch` - 235B weights download on cpu partition

Submit with `sbatch --qos=interactive` (starts instantly, <=4 nodes) or normal QoS.

## Non-obvious details

- AWS-DFW forces `HF_HUB_CACHE` system-wide to `projects/nemotron_sw_post/users/$USER/hf_home`;
  `HF_HOME` is ignored for the hub cache.
- `batch` partition rejects jobs without `--gpus-per-node`; 4h hard wall.
- GB200 JIT needs `CUDA_HOME=/cm/shared/apps/cuda13.0/toolkit/13.0.2` plus `ninja`/`cmake`
  in the vllm venv; first sm_100 compile takes ~30 min (warm cache: ~3-7 min).
- System TMPDIR is a long lustre path that breaks ZMQ IPC (107-char limit); set `TMPDIR=/tmp`.
- Cross-node training needs `--vllm-endpoint http://<server-node>:8000/v1` AND
  `--hidden-states-path` on lustre (FileBackend default `/tmp/hidden_states` is node-local).
- Install the speculators repo editable (`-e speculators -e speculators/hs_connectors`);
  the PyPI release is behind the repo scripts.
