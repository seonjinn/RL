# Draft co-training implementation comparison

Compares two online DFlash/DSpark drafter co-training implementations on the same model,
drafter checkpoint, K, data and 20 GRPO steps:

- ours: PR stack #3701-#3757, branch `sna/draft-cotrain-specdec-perf-20261020`
  (`experiments/specdec_perf_20261020`)
- teammate: PRs #4059/#4060 on latest main with #3701-#3715 and #3960 reverted, this branch
  (`sna/markx-draft-cotrain-20261021`)

The two cannot share a training layout. Ours validates TP2 + SP. The teammate's block drafts
require SP off, and Megatron MoE with TP>1 needs SP, so a MoE policy runs at TP1. Each arm is
therefore compared against a no-SpecDec control at its own layout, and against the frozen
drafter arm at the recipe layout (`specdec_perf_20261020` frozen_dflash_k7 / frozen_dspark_k8).

## Usage

    ACCOUNT=coreai_dlalgo_llm experiments/draft_impl_compare_20261021/submit.sh \
      experiments/draft_impl_compare_20261021/configs/qwen3_30ba3b/online_dflash_k7.yaml \
      q30-markx-online-dflash-k7 20

Run names start with `q30-markx-`. W&B project `sna-async-grpo-gb200`. Checkpointing stays off.