# Nemotron3 Super Option B attention A/B

This experiment compares FlashInfer and Triton vLLM attention with the same
Nemotron3 Super MXFP8 training/rollout recipe. Both arms use Option B grouped
GEMM, 32 x 4 GB200 GPUs, Async-1off, GBS 256, NCCL Reshard refit, and 20 steps.
The only intended A/B variable is the attention backend. The rollout uses
TP=4 and EP=4 so the BF16 first/last expert weights have expert-dimension
destination shards supported by the current FlashInfer TRTLLM refit path.

The 2026-10-08 extension adds `async-bf16-bf16.yaml` and
`async-bf16-mxfp8.yaml` as precision controls. Submit either control with
`SUPER_ARM=bf16-bf16` or `SUPER_ARM=bf16-mxfp8` and choose `flashinfer` or
`triton` as the first launcher argument. The default `SUPER_ARM=option-b`
preserves the original MXFP8 training + rollout arm. Use one source archive
and nightly image across the new controls; compare completed steps 2-20 and
verify the realized attention backend and finite generation KL.

## Lyris reproduction

Use a clean checkout of the pushed branch. The source archive is an immutable
snapshot copied to node-local storage by the launcher. The launcher uses the
Automodel, Gym, and Megatron-Bridge submodules built into the container, not
submodule checkouts from the source archive.

```bash
git clone --branch sna/mxfp8-perf https://github.com/NVIDIA-NeMo/RL.git "$HOME/RL-super-optionb"
cd "$HOME/RL-super-optionb"
export SOURCE_COMMIT="$(git rev-parse HEAD)"
export RESULT_ROOT="/lustre/fsw/coreai_dlalgo_llm/users/$USER/experiments/super-optionb-attention"
export SOURCE_ARCHIVE="$RESULT_ROOT/source-${SOURCE_COMMIT:0:10}.tar"
export CONTAINER=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/mxfp8_backends_20261007/nemo_rl_nightly_vllm029_20261007_3265087.sqsh
mkdir -p "$RESULT_ROOT"
git archive --format=tar -o "$SOURCE_ARCHIVE" HEAD

# WANDB_API_KEY must already be exported securely; do not put it in the script.
bash experiments/super_option_b_attention_20261007/submit-lyris.sh flashinfer test-only
bash experiments/super_option_b_attention_20261007/submit-lyris.sh triton test-only
bash experiments/super_option_b_attention_20261007/submit-lyris.sh flashinfer
bash experiments/super_option_b_attention_20261007/submit-lyris.sh triton
```

The current launcher assumes the model is already cached under
`/lustre/fsw/coreai_dlalgo_llm/users/$USER/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/`.
Set `MEGATRON_CONVERSION_CACHE_DIR` to an existing completed HF-to-Megatron
conversion when reusing a compatible model checkpoint across experiment-only
source commits. The launcher validates its `run_config.yaml` before submission;
without this variable, it uses a commit-scoped conversion cache.
The image path above must be readable to the submitting user. `SLURM_ACCOUNT`
can override the default `coreai_dlalgo_llm` account; `RESULT_ROOT` and
`CONTAINER` must point to accessible paths. Each run writes its resolved
config, SLURM/Ray logs, and W&B metadata under `RESULT_ROOT`.

The first pair of runs (jobs `3269219` and `3269231`, commit `5db1b4563`)
failed before step 1 because rollout EP=1 produced an unsupported expert
`Shard(1)` in the BF16 TRTLLM refit path. Commit `56c8479ccb` sets rollout
EP=4 for both arms.

The EP=4 rerun completed 20/20 steps in both arms. The vLLM worker logs
confirm `FLASHINFER` and `TRITON_ATTN` selection, respectively. Step 2-20
mean E2E time was 32.56s for [FlashInfer](https://wandb.ai/nvidia/nemo-rl-mxfp8-training/runs/pzvp4a1h)
and 32.90s for [Triton](https://wandb.ai/nvidia/nemo-rl-mxfp8-training/runs/6ts03ajk).
Generation throughput was 617.96 versus 602.03 tokens/s/GPU. This is a
single-run, directional comparison; mean output length differed by 2.7%.
Triton step 3 logged `NaN` for `gen_kl_error`, `policy_kl_error`,
`js_divergence_error`, and `approx_entropy` despite 256 valid samples and
a finite loss. FlashInfer logged finite values for all four in steps 2-20.
This pair does not yet establish Triton numerical correctness or an
attention-backend speedup.

## Six-arm precision matrix

The Async-1off comparison uses GBS 256 on 32 x 4 GB200 GPUs with 20 steps.
Use `SUPER_ARM=bf16-bf16`, `bf16-mxfp8`, `mxfp8-default`, `option-b`,
`mxfp8-false-default`, or `mxfp8-false-option-b` with either attention backend.
All MXFP8-training arms keep the same routed-expert TE scope, first 2/last 6
BF16 layers, rollout EP4, and NCCL Reshard refit. The Option B arms additionally
enable grouped-tensor storage, the TE op fuser, and cuDNN/CuteDSL flags.
