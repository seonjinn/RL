# Mixed Preference Optimization for VLMs in NeMo RL

[Mixed Preference Optimization (MPO)](https://arxiv.org/abs/2411.10442)
combines a DPO-style preference loss, an auxiliary SFT generation loss, and a
[BCO](https://arxiv.org/abs/2404.04656) quality term. This guide covers the
VLM (vision-language) variant ported on top of the existing omni / Megatron /
vLLM stack.

## Launch a VLM MPO Run

The script [`examples/run_vlm_mpo.py`](../../examples/run_vlm_mpo.py) launches
a VLM MPO experiment. For multi-node SLURM, use
[`scripts/vlm_mpo.sh`](../../scripts/vlm_mpo.sh), which reads `.env`
(`SBATCH_ACCOUNT`, `CONTAINER`, `MOUNTS`, `MPO_MODEL_NAME`, `MPO_DATA_PATH`),
allocates `NUM_NODES` nodes (default `4`), and submits the job:

```bash
bash scripts/vlm_mpo.sh
```

If you are not on the prebuilt sqsh container, override the runner so uv
installs the required extras:
`RUNNER='uv run --extra mcore --extra vllm' bash scripts/vlm_mpo.sh`.
For Ray / Slurm setup details see the [cluster documentation](../cluster.md).

**Reminder**: set `WANDB_API_KEY` and `HF_HOME` (or run `wandb login` /
`huggingface-cli login` on the submission host -- `vlm_mpo.sh` bind-mounts
`~/.netrc` into the container).

## Configuration

The reference config is
[`examples/omni/nanov3_mpo.yaml`](../../examples/omni/nanov3_mpo.yaml)
(`data.dataset_name: mmpr`). Override values via the CLI:

```bash
uv run --no-sync examples/run_vlm_mpo.py \
    --config examples/omni/nanov3_mpo.yaml \
    mpo.preference_loss_weight=0.9 \
    mpo.bco_loss_weight=0.1
```

Use `--no-sync` on the prebuilt sqsh container (`vlm_mpo.sh` already does
this internally) so uv reuses the baked `/opt/nemo_rl_venv` without
re-running `uv sync`. On a fresh checkout, replace with
`uv run --extra mcore --extra vllm`.

To use a different dataset, copy the YAML and change `data.dataset_name` to
one of the loaders under [Datasets](#datasets).

## MPO-Specific Parameters

All under the `mpo:` section of the YAML. Implementation:
[`MPOLossFn`](../../nemo_rl/algorithms/loss_functions.py).

- `mpo.reference_policy_kl_penalty` -- KL penalty strength (β), shared by the
  preference and BCO terms
- `mpo.preference_loss_weight` -- weight on the DPO-style preference loss
- `mpo.sft_loss_weight` -- weight on the auxiliary SFT loss
- `mpo.bco_loss_weight` -- weight on the BCO (binary-classifier) loss
- `mpo.preference_average_log_probs`, `mpo.sft_average_log_probs`,
  `mpo.quality_average_log_probs` -- whether to average per-token log-probs
  in each term
- `mpo.reward_shift_momentum` -- EMA momentum for the BCO reward shift δ
  (default `0.99`)
- `mpo.reward_shift` -- initial BCO reward shift δ

## Datasets

Selected via `data.dataset_name`:

- `mmpr` -- MMPR / MMPR-v1.2 image preference pairs
  ([`mmpr.py`](../../nemo_rl/data/datasets/response_datasets/mmpr.py))
- `blend_v1` -- blended MMPR-v1.2 subsets tagged for verifier types
  ([`blend_v1.py`](../../nemo_rl/data/datasets/response_datasets/blend_v1.py))
- `omni_dataset` -- image / video / audio QA, with optional preference
  completions
  ([`omni_dataset.py`](../../nemo_rl/data/datasets/response_datasets/omni_dataset.py))
- `video_dataset` -- video QA, sampled into `data.num_frames` frames
  ([`video_dataset.py`](../../nemo_rl/data/datasets/response_datasets/video_dataset.py))

For the exact on-disk schema each loader expects, read the docstring at the
top of the corresponding file.

## Evaluate the Trained Model

Refer to the [evaluation guide](eval.md) once training has produced a
checkpoint.
