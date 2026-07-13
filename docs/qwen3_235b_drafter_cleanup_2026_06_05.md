# Qwen3-235B Drafter Cleanup Log

Date: 2026-06-05 PDT

## Policy

Delete only checkpoints whose benchmark result is already captured in
`docs/qwen3_235b_pard_math_local_checkpoint_gates.csv` and
`docs/qwen3_235b_pard_action_report_2026_06_05.md`.

Keep:

- public HF drafters such as `amd/PARD-Qwen3-0.6B` and public EAGLE3 models.
- the latest 1K CAT venreuse checkpoint until a better PARD/PARD-2 checkpoint
  is found.
- final internally trained 500K EAGLE3 checkpoints:
  - `eagle3_qwen3_235b_mixed_math_nonopenmath_500k_parallel/checkpoints_train_500k_layers94_mlen8193/0`
  - `eagle3_qwen3_30ba3b_mixed_math_nonopenmath_500k_parallel/checkpoints_train_500k_layers48_mlen8193/0`

## Deleted PARD Candidates

These are safe deletion candidates because their outcomes regressed, were only
compatibility smokes, or were superseded by later gates.

| Size | Path | Reason |
|---:|---|---|
| `9.2GB` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_cat_logprob_128_lr3e6` | 128-row CAT integration smoke; result retained. |
| `9.2GB` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_prefix_reward_1024_lr3e6_4chunk_fs1` | Prefix-reward 1K checkpoint; did not beat public PARD. |
| `9.2GB` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_weightedce_tpp_1024_8x128_lr3e6` | Weighted CE checkpoint; gate regressed to `0.869x`. |
| `3.4GB` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/interpolated_public_pard_prefix_reward_1024` | Interpolation alpha checkpoints; all below public historical PARD. |
| `9.2GB` | `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_pard_math/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_custom_uniform_compare128_v3` | Old custom/uniform smoke. |
| `9.2GB` | `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_pard_math/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_plain_compare128_current` | Old plain CE comparison. |
| `9.2GB` | `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_pard_math/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_prefix_reward_smoke128` | Old prefix-reward smoke. |
| `9.2GB` | `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_pard_math/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_prefix_reward_smoke128_v2` | Old prefix-reward smoke. |
| `9.2GB` | `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_pard_math/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_prefix_reward_smoke128_v3` | Old prefix-reward smoke. |
| `9.2GB` | `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_pard_math/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_teacher_chunks_partial450_retry_venvreuse_nopip` | Partial450 regressed; result retained. |

Estimated reclaim from this PARD-only deletion set: `86.2GB`.

Deletion status: completed on 2026-06-05 PDT. All 10 paths above were verified
removed after `rm -rf`. The retained comparison checkpoint
`PARD-Qwen3-0.6B_qwen235b_math_k5_cat_logprob_1024_8x128_lr3e6_venvreuse/checkpoint-16`
was verified present after cleanup.

## Deleted EAGLE3 Candidates

`/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_openmath_reasoning_cot_smoke128`
was about `47GB`. It was an internally trained early smoke artifact using
`openmath_reasoning_cot_conversations_50k_speculators.jsonl`, with multiple
`hidden_states*`, `vllm_tmp_hidden_states*`, and `checkpoints_train_smoke_128_*`
attempts from 2026-05-24. It was deleted after the user confirmed it was no
longer needed.

The following 500K chunk intermediate directories were also deleted. These were
not the final trained 500K checkpoints; they were openmath chunk/intermediate
hidden-state artifacts. Exact per-directory size was not captured because
`du -sh` timed out before deletion.

| Path | Status |
|---|---|
| `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_235b_openmath_500k_chunk000_50k` | deleted |
| `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_235b_openmath_500k_chunk001_50k` | deleted |
| `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_235b_openmath_500k_chunk002_50k` | deleted |
| `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_30ba3b_openmath_500k_chunk000_50k` | deleted |
| `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_30ba3b_openmath_500k_chunk001_50k` | deleted |
| `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_30ba3b_openmath_500k_chunk002_50k` | deleted |
| `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/version_probe` | deleted |

Confirmed retained after deleting the chunk intermediates:

- `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_235b_mixed_math_nonopenmath_500k_parallel/checkpoints_train_500k_layers94_mlen8193/0`
  - verified present on 2026-06-05 PDT, current size `3.2G`
- `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_30ba3b_mixed_math_nonopenmath_500k_parallel/checkpoints_train_500k_layers48_mlen8193/0`
  - verified present on 2026-06-05 PDT, current size `1.2G`

Confirmed reclaim with known sizes: `86.2GB + 47GB = 133.2GB`.
Additional reclaim from the 500K chunk intermediates is real but unquantified,
because exact sizes were not captured before deletion.

Final 500K checkpoints were intentionally not deleted because they may be useful
for later internally trained drafter comparisons.
