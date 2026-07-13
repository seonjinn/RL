# Qwen3-235B Recent Speculative Decoding Method Triage

Date: 2026-06-05 PDT

This note records the current method triage while the Qwen3-235B no-stop
full-GRPO `MAX_STEPS=20` pair is pending. The first DFlash compatibility smoke
has now produced an aligned Qwen3-235B checkpoint, so the next gate is whether
that checkpoint can load and move held-out OpenMath throughput in vLLM
standalone. The goal is to avoid repeatedly chasing methods that are interesting
but not yet actionable in the current vLLM / NeMo-RL Qwen3-235B stack.

## Latest Correction, 2026-06-06 PDT

This section supersedes older scheduler/build-chain notes below.

### Follow-up, 2026-06-06 14:xx PDT

- Qwen3-32B now has a completed dense Full-GRPO baseline/PARD pair at GBS 256:
  baseline `3195498` and public PARD K5 `3195499` both completed with exit
  `0:0`. PARD reduced generation time from `16.66s` to `14.21s` and improved
  generation-worker throughput from `355.47` to `416.75 tok/s/GPU`
  (`1.17-1.18x` generation speedup). E2E improved from `100.91` to
  `105.49 tok/s/GPU` (`1.045x`) because generation is only about `25-28%` of
  the step at this shape.
- Qwen3-30B-A3B no-local-spec retries still failed in Megatron MoE construction:
  `SequentialMLP` used `deepcopy(config)`, which recursively touched
  distributed `ProcessGroup` objects and raised
  `TypeError: cannot pickle 'torch._C._distributed_c10d.ProcessGroup' object`.
  This failure is independent of PARD quality and is directly relevant to
  Qwen3-235B-A22B because both are MoE models.
- Remote Megatron patch applied:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/experts.py`
  now uses `copy.copy(config)` instead of `deepcopy(config)` in the
  `SequentialMLP` branch that only needs to override top-level
  `ffn_hidden_size`. The patched file passed `python3 -m py_compile`.
  Reapply artifact:
  `experiments/eagle3_online/remote_patch_files/megatron_moe_sequential_mlp_shallowcopy_processgroup.patch`.
- Post-patch validation jobs were submitted:
  - Qwen3-30B-A3B baseline: `3195815`
  - Qwen3-30B-A3B local CAT/PARD-2-style K5: `3195816`
  These jobs are the immediate evidence gate for whether the MoE
  `ProcessGroup` fix is sufficient before interpreting Qwen3-235B results.
- Active Qwen3-235B Full-GRPO job `3195285` was still `PENDING (Priority)` when
  last reachable. Because it had not started, it should pick up the
  `SequentialMLP` shallow-copy patch before MoE policy workers are created.
- Local cluster access is temporarily blocked by DNS/VPN resolution:
  `oci-hsg-cs-001-vscode-02` resolves through public `8.8.8.8` as `NXDOMAIN`.
  This is not a job failure. Use
  `scripts/poll_qwen235b_pard_fullgrpo_current_status.sh` after internal DNS is
  restored to poll `3195285`, `3195815`, and `3195816`.
- If the remote checkout is refreshed before polling, reapply the MoE patch
  with `scripts/apply_remote_megatron_moe_pg_shallowcopy_patch.sh` first. The
  poll script now also reports whether the `SequentialMLP` shallow-copy patch is
  present in the active remote checkout.
- Source-status check remains unchanged: vLLM supports PARD through
  `draft_model` plus `parallel_drafting=true` with examples using
  `amd/PARD-Qwen3-0.6B`, while the AMD PARD repository still does not expose an
  official PARD-2 implementation/checkpoint. Keep labeling local CAT/D-PACE
  runs as PARD-2-style approximations, not official PARD-2.
- Reverified upstream while remote cluster access was unavailable:
  AMD-AGI/PARD `master` HEAD is still
  `77eee0a12a729aaa4cc38b2a30fd544e11a8173b`, and its README update line says
  the PARD-2 paper is released but code/model checkpoints will be released soon.
  The vLLM `v0.20.0` PARD docs still show the expected runtime form:
  `speculative_config={"model": "amd/PARD-Qwen3-0.6B",
  "method": "draft_model", "parallel_drafting": True}`.
- Current execution-facing decision summary:
  `docs/qwen3_235b_pard_operator_decision_20260606.md`. This table separates
  short synthetic, OpenMath, NeMo-RL generation-only, stop-after-generation, and
  still-unverified no-stop Full-GRPO evidence so the Qwen3-235B E2E claim is not
  overstated.
- Source-facing method table:
  `docs/qwen3_235b_2026_specdec_source_triage.md`. This records the current
  public-source status for PARD, PARD-2/CAT, DFlash, P-EAGLE, Eagle-3,
  DFlash-family follow-ups, SpecKV, and MoE-specific verification methods.

### Scheduler Update, 11:33 PDT

No new Qwen3-235B Full-GRPO E2E result is available yet. The active no-stop
jobs are still pending for priority with zero elapsed time. `scontrol show job`
now shows `Dependency=(null)` for every active job, so the current blocker is
32-node allocation priority rather than a dependency deadlock or launch/log
failure.

| Job | Mode | Current scheduler state |
|---:|---|---|
| `3186510` | baseline, real sampling, `MAX_STEPS=4` | `PENDING (Priority)`, planned start `13:35 PDT`, no driver log |
| `3186511` | local CAT/TPP-mask K5, real sampling, `MAX_STEPS=4` | `PENDING (Priority)`, planned start `13:35 PDT`, no driver log |
| `3192180` | 2K dynamic D-PACE K5, real sampling, `MAX_STEPS=4` | `PENDING (Priority)`, planned start `14:10 PDT`, no driver log |
| `3192438` | 2K dynamic D-PACE K3, real sampling, `MAX_STEPS=4` | `PENDING (Priority)`, planned start `14:27 PDT`, no driver log |
| `3186342` | baseline fixed-256 diagnostic, `MAX_STEPS=20` | `PENDING (Priority)`, planned start `12:04 PDT`, no driver log |
| `3186343` | local CAT/TPP-mask K5 fixed-256 diagnostic, `MAX_STEPS=20` | `PENDING (Priority)`, planned start `12:04 PDT`, no driver log |
| `3186344` | public PARD K5 fixed-256 diagnostic, `MAX_STEPS=20` | `PENDING (Priority)`, planned start `13:23 PDT`, no driver log |

This means current claims must remain limited to standalone and NeMo-RL
generation-gate results. Do not claim no-stop Full-GRPO E2E speedup until one
of these pairs reaches step metrics.

### Method Source Check, 2026-06-06

GitHub verification: cloned `https://github.com/AMD-AGI/PARD.git` master at
`77eee0a12a729aaa4cc38b2a30fd544e11a8173b`. The repo README says the PARD-2
paper was released on 2026-05-09 and that code/model checkpoints will be
released soon. The checked file list contains PARD training/inference files
(`pard/pard_train.py`, `pard/pard_infer.py`, `utils/vllm_infer.py`) but no
dedicated `pard2` implementation or PARD-2 checkpoint assets.

| Method | Current actionable status | Qwen3-235B decision |
|---|---|---|
| PARD (`https://arxiv.org/abs/2504.18583`, `https://github.com/AMD-AGI/PARD`) | Runnable now through vLLM `draft_model` with `parallel_drafting=true`. | Keep as the runtime substrate. It is the only branch with measured generation speedup inside the NeMo-RL launcher. |
| PARD-2 / CAT (`https://arxiv.org/abs/2605.08632`) | Official code/checkpoints are not in the AMD PARD repo yet. Local CAT/D-PACE trainers are approximations. | Continue local objective work, but label it as PARD-2-style rather than official PARD-2. |
| DFlash (`https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/`, `https://arxiv.org/abs/2602.06036`) | Runtime is now verified on source vLLM, but the current aligned Qwen3-235B checkpoint has only `~0.25-1.22%` OpenMath acceptance. | Do not spend NeMo-RL nodes on current DFlash. A future DFlash branch needs a better trained checkpoint first. |
| P-EAGLE (`https://arxiv.org/abs/2602.01469`) | Relevant to parallelizing EAGLE-style drafters, but no ready Qwen3-235B P-EAGLE checkpoint/path exists in this stack. | Secondary option after PARD objective work, not the immediate fix. |
| SpecKV (`https://arxiv.org/abs/2605.02888`) | Adaptive K/controller idea is relevant because K3 acceptance can be higher while K5 throughput wins. | Treat as a controller follow-up once PARD K3/K5 telemetry is stable; it does not solve drafter-domain mismatch alone. |
| MoE-Spec (`https://arxiv.org/abs/2602.16052`) | Directly relevant to Qwen3-235B-A22B because verification cost and expert bandwidth can erase speculative gains. | Systems follow-up if no-stop Full-GRPO shows weak E2E despite generation speedup. |

| Area | Latest state | Decision |
|---|---|---|
| PARD runtime | Works in vLLM standalone and NeMo-RL `VllmGeneration` gates with `draft_model`, `parallel_drafting=true`, target `TP=4`, draft `TP=4`. | Keep as primary Qwen3-235B path because it is the only current branch with measured generation speedup inside the NeMo-RL launcher. |
| PARD-2 official release | AMD PARD repo still says PARD-2 code/model checkpoints will be released soon. | Treat local CAT/D-PACE/LK-style checkpoints as PARD-2-style approximations, not official PARD-2 reproductions. |
| Best local OpenMath standalone | 2K dynamic D-PACE draft-probability CE K5, job `3190567`: `627.14 tok/s/GPU`, `1.296x`, `47.01%` K5 acceptance. | Current best local checkpoint; use it for NeMo-RL Full-GRPO validation. |
| K3 on best local checkpoint | 2K D-PACE K3, job `3193047`: `584.47 tok/s/GPU`, `1.207x`, `61.55%` K3 acceptance. | Do not assume higher aggregate K3 acceptance means better standalone throughput; K5 remains better in standalone. |
| NeMo-RL sync D-PACE K sweep | Same 2K D-PACE checkpoint in 1-node sync `VllmGeneration`, fixed 256 tokens: K1 job `3194117` reached `322.23 tok/s` (`1.250x`, `76.42%` acceptance), K2 job `3194118` reached `335.88 tok/s` (`1.303x`, `67.61%` acceptance), K3 job `3192349` reached `374.62 tok/s` (`1.454x`, `57.50%` acceptance), and K5 job `3192211` reached `355.44 tok/s` (`1.379x`, `44.25%` acceptance). | In the NeMo-RL generation path, K3 is currently the best systems tradeoff. High K1/K2 aggregate acceptance does not compensate for issuing fewer speculative tokens; no K2 Full-GRPO companion is justified unless E2E results contradict this. Keep K3 and K5 in Full-GRPO. |
| More data with same objective | 4K D-PACE K5 regressed to `1.212x`; 4K K3 regressed to `1.191x`. | Do not blindly scale teacher rows without improving objective/runtime. |
| Objective ablations | 2K accept-rate K5 reached `1.267x`; 2K hybrid D-PACE + accepted-prefix reward reached `1.241x`; D-PACE smoothing alpha `0.2`/`0.8` reached `1.243x`/`1.221x`. | Useful negative evidence. Simple accept-rate reward and alpha changes are not enough; best remains 2K D-PACE K5 with alpha `0.5`. |
| DFlash branch | Runtime support is now verified: support probe `3187927` completed with `dflash_ready=true` on `vllm-0.19.1rc1.dev315+g0b790a250`. OpenMath retry28 jobs `3189242`/`3189243` completed, but `dflash_openmath_reasoning_cot_smoke512_k5_aligned` has only `~0.25-1.22%` aggregate acceptance across K3/K5. | Do not promote this DFlash checkpoint to NeMo-RL. Next DFlash action would be better hidden-state extraction/training, not another runtime retry. |
| Full-GRPO E2E | Current no-stop sampling jobs `3186510`, `3186511`, `3192180`, and `3192438`, plus fixed-256 diagnostic jobs `3186342`, `3186343`, and `3186344`, are still `PENDING (Priority)` with no driver logs or E2E metrics at 11:33 PDT. `Dependency=(null)` on all active jobs; latest refresh shows planned starts `12:04-14:27 PDT`. See `docs/qwen3_235b_fullgrpo_scheduler_status_20260606.csv` for the latest refresh snapshot. | Do not claim NeMo-RL E2E benefit yet; only generation-gate benefit is proven. |

Training-sample guidance from completed local PARD-2-style runs: 1K-2K rows are
enough for objective ranking, 8K-16K is a reasonable next domain ablation only
after the objective improves, and 500K rows are not justified yet. The 0.6B
drafter fine-tune itself is cheap (`~5 min` for 2K rows, `~7 min` for 4K rows);
the expensive part is collecting Qwen3-235B teacher continuations/logprobs and
gating them. The measured teacher side is about `34-39 min` per 128-row chunk
on 4 GB200 GPUs with target `TP=4`; parallel chunking is therefore required for
larger datasets. The 4K regression is the key evidence that more rows alone are
not the current limiter.

DFlash artifact audit note: the aligned DFlash smoke directory contains stale
HuggingFace dataset metadata copied from the original 50K prepared data, but the
actual Arrow stream in `prepared_from_existing_512/data-00000-of-00001.arrow`
has `512` rows. The checkpoint config is valid DFlash
(`speculators_model_type=dflash`, `block_size=6`, `speculative_tokens=5`,
`mask_token_id=151670`), but
`val/full_acc_epoch=0.0326` and per-position validation accuracy is only about
`3.0-3.6%`. This matches the near-zero OpenMath acceptance and confirms the
current issue is checkpoint quality, not DFlash runtime registration.

## Immediate Path

| Method | Source | Current status for Qwen3-235B |
|---|---|---|
| PARD | https://arxiv.org/abs/2504.18583, https://github.com/AMD-AGI/PARD | Actionable now. vLLM can run `draft_model` with `parallel_drafting=true`; public `amd/PARD-Qwen3-0.6B` and our local CAT/TPP-mask checkpoint both load in direct gates. |
| PARD-2 / CAT | https://arxiv.org/abs/2605.08632, https://github.com/AMD-AGI/PARD | Most relevant next objective. The paper targets acceptance-length optimization via CAT, matching our failure mode. The official repo lists PARD-2 paper release but says code/checkpoints will be released soon, so our CAT/TPP-mask trainer remains a runnable approximation rather than a full PARD-2 implementation. |
| DFlash | https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/, https://arxiv.org/abs/2602.06036 | Runtime path is no longer the main blocker after support probe `3187927` and retry28 K3/K5 gates. The aligned smoke checkpoint loads and runs, yet OpenMath acceptance is only `~0.25-1.22%`, so it is not a NeMo-RL candidate without retraining. |
| DREAM-R | https://arxiv.org/abs/2605.28678 | Supports the diagnosis that reasoning-domain drafter alignment matters. Not a drop-in vLLM/NeMo-RL text-generation drafter path because it is a multimodal speculative-reasoning framework with SAPO/TBVM/FPSR. |

## Secondary Candidates

| Method | Source | Why it matters | Why it is not first |
|---|---|---|---|
| TreeFlash | https://arxiv.org/abs/2606.03819 | Addresses non-autoregressive block/tree drafter distribution drift by approximating AR conditioning while keeping one-shot draft complexity. | Requires a new drafter/runtime path; current vLLM stack already supports PARD more directly. |
| P-EAGLE | https://arxiv.org/abs/2602.01469 | Converts EAGLE to parallel multi-token prediction and focuses on long-context reasoning outputs; implemented in vLLM for some models. | No ready Qwen3-235B P-EAGLE drafter in this experiment stack. Potentially useful after PARD-2 if Qwen3-235B long-output EAGLE/PARD remains weak. |
| SpecVocab | https://arxiv.org/abs/2602.13836 | Reduces output-head drafting bottleneck through step-wise vocabulary subsets. | More relevant to draft-output cost than our current dominant issue of domain alignment plus MoE verification/runtime overhead. |
| SpecKV | https://arxiv.org/abs/2605.02888 | Adaptive speculation length from draft confidence/entropy. | Useful as a controller over K once we have stable K5/K12 telemetry; does not by itself fix drafter-domain mismatch. |
| MoE-Spec | https://arxiv.org/abs/2602.16052 | Directly relevant to Qwen3-235B-A22B because MoE verification can erase speculative decoding gains through expert bandwidth/memory pressure. | Training-free in principle, but would require verification-time expert budgeting changes inside the target MoE runtime. Treat as a systems follow-up if PARD K5 acceptance is high but E2E speedup remains low. |
| Attention Drift | https://arxiv.org/abs/2605.09992 | Explains long-context/template degradation in EAGLE3/MTP-style drafters via hidden-state drift. | Diagnostic insight more than an immediately runnable algorithm. It supports measuring per-depth acceptance and not over-scaling K when deeper acceptance decays. |
| Steering pretrained drafters | https://arxiv.org/abs/2511.09844 | Dynamic verifier-hidden-state steering aims to improve pretrained drafter acceptance under mismatch. | Interesting for a future target-aware PARD-style variant, but requires verifier-to-drafter hidden-state plumbing not present in the current stack. |

## Current Decision

Continue the PARD/PARD-2-style no-stop Full-GRPO validation because it remains
the nearest drop-in runtime path for NeMo-RL. The current local checkpoint to
validate is the 2K dynamic D-PACE K5 drafter, not the older 1K CAT/TPP-mask
checkpoint. Do not scale the same objective blindly: 4K D-PACE and the 2K
hybrid accepted-prefix reward ablation both regressed. DFlash should stay as a
secondary research branch, but the current aligned DFlash smoke checkpoint is
not a candidate for NeMo-RL because held-out OpenMath acceptance is near zero.

Active jobs as of 2026-06-06 11:33 PDT:

| Job | Mode | Current scheduler state |
|---:|---|---|
| `3186510` | Qwen3-235B baseline no-stop Full-GRPO, real sampling, `MAX_STEPS=4` | `PENDING (Priority)`, planned start `13:35 PDT`; no E2E logs yet |
| `3186511` | Qwen3-235B local CAT/TPP-mask K5 no-stop Full-GRPO, real sampling, `MAX_STEPS=4` | `PENDING (Priority)`, planned start `13:35 PDT`; no E2E logs yet |
| `3192180` | Qwen3-235B dynamic D-PACE K5 no-stop Full-GRPO, matched to baseline `3186510` | `PENDING (Priority)`, planned start `14:10 PDT`; no E2E logs yet |
| `3192438` | Qwen3-235B dynamic D-PACE K3 no-stop Full-GRPO, matched to baseline `3186510` | `PENDING (Priority)`, planned start `14:27 PDT`; no E2E logs yet |
| `3186342` | Qwen3-235B baseline no-stop Full-GRPO, fixed-256 diagnostic, `MAX_STEPS=20` | `PENDING (Priority)`, planned start `12:04 PDT`; no E2E logs yet |
| `3186343` | Qwen3-235B local CAT/TPP-mask K5 no-stop Full-GRPO, fixed-256 diagnostic, `MAX_STEPS=20` | `PENDING (Priority)`, planned start `12:04 PDT`; no E2E logs yet |
| `3186344` | Qwen3-235B public PARD K5 no-stop Full-GRPO, fixed-256 diagnostic, `MAX_STEPS=20` | `PENDING (Priority)`, planned start `13:23 PDT`; no E2E logs yet |
| `3178708` | Qwen3-235B DFlash train-only smoke, newly prepared 512-row data, existing 50K OpenMath hidden-state pool, `MAX_SAMPLES=512`, `BLOCK_SIZE=6`, `NOISE_STD=0`, explicit DFlash forward dtype cast | completed, exit `0:0`; checkpoint created; metrics all zero |
| `3178746` | Qwen3-235B DFlash train-only smoke, 512-row subset of original 50K prepared Arrow data, same existing hidden-state pool | completed, exit `0:0`; aligned checkpoint created |
| `3178867` | DFlash checkpoint vLLM container config probe, no target model load | completed, exit `0:0`; current vLLM container only supports `eagle3` Speculators config conversion |
| `3185614` | DFlash-capable vLLM source-build retry6, PR `38300` commit `0b790a2`, CUTLASS support-op fallback patch | running at 22:55 PDT |
| `3185615` | DFlash runtime support probe, `afterok:3185614` | dependency-pending |
| `3185616` / `3185618` / `3185621` | DFlash OpenMath standalone baseline / K3 / K5, `afterok:3185615` | dependency-pending |

Queue/config note: a smaller no-stop Full-GRPO job was not submitted after
the 11:33 PDT audit. The active Qwen3-235B E2E shape is `32n4g`, generation
`TP=4`, Megatron `TP=2`, `PP=8`, `CP=2`, `EP=16`, and GBS `256`. Reducing
nodes would change the training shard shape and weaken the E2E evidence; the
cheap validation path remains 1-node `VllmGeneration` direct/sync gates.

Note: the first DFlash smoke attempt `3178648` was cancelled after it revealed a
submit-script issue: `MAX_SAMPLES` did not limit train.py when `RUN_PREPARE` was
disabled, so it tried to train over the full 50K Arrow data. The replacement
job `3178667` created a separate 512-row prepared-data directory before
training, but failed on a DFlash dtype mismatch: hidden states reached the FC
projection as `float32` while DFlash weights were `bfloat16`. The current retry
`3178708` passes `--noise-std 0`, applies
`speculators_noise_zero_preserve_dtype_compat.patch`, and applies
`speculators_dflash_forward_dtype_compat.patch`. The latter casts
`hidden_states` and `verifier_last_hidden_states` to the DFlash module weight
dtype at the forward boundary. Retry `3178693` was cancelled after confirming
that the zero-noise patch alone did not remove the dtype mismatch.

`3178708` compatibility result:

- Completed in `00:02:54` with exit `0:0`.
- Created checkpoint:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/dflash_openmath_reasoning_cot_smoke512_k5/checkpoints/0`.
- Config records `speculators_model_type=dflash`, `block_size=6`, and
  `speculative_tokens=5`.
- `val_metrics.json` is all zero, and the logs show loaded-token-id versus
  input-id mismatch warnings. Treat this only as proof that the DFlash Qwen3-235B
  training path can run; do not benchmark this checkpoint until the
  prepared-data / hidden-state alignment issue is fixed.

Aligned retry:

- `3178746` uses `RUN_PREPARED_SUBSET=true`, selecting the first `512` rows from
  the original 50K prepared Arrow directory:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_openmath_reasoning_cot_50k`.
- This aligns with `hs_0..hs_511` from
  `hidden_states_layers93_mlen8193_50k`, avoiding the token-id mismatch caused
  by re-tokenizing a new 512-row JSONL.
- Completed in `00:02:51` with exit `0:0`.
- Created checkpoint:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/dflash_openmath_reasoning_cot_smoke512_k5_aligned/checkpoints/0`.
- Config records `speculators_model_type=dflash`, `block_size=6`,
  `speculative_tokens=5`, `num_hidden_layers=2`, `max_anchors=128`,
  `mask_token_id=151670`, and `dtype=bfloat16`.
- Validation metrics are nonzero but weak: `val/loss_epoch=4.457`,
  `val/full_acc_epoch=0.0326`, with per-position validation accuracy around
  `3.0-3.6%`. Treat this as a load/training compatibility checkpoint, not yet
  a performance checkpoint.

If K5 keeps the generation-only speedup but E2E remains weak, the next systems
hypothesis is MoE verification overhead, not acceptance alone. DFlash now
trains cleanly enough for a smoke, but the current vLLM `0.17.0` container
cannot load it for inference: probe `3178867` reports
`supported_speculators_types=['eagle3']` and rejects
`speculators_model_type=dflash`. The next gate is therefore a vLLM runtime
support fix or newer container with DFlash registered, then a held-out OpenMath
`ISL=1024`, `OSL=1024`, bs32 standalone benchmark before any NeMo-RL
integration.

Runtime-support follow-up on 2026-06-05 PDT:

- Official upstream direction is vLLM PR `#38300`, "add DFlash speculators
  support." The PR adds `register_speculator("dflash")`, DFlash config
  conversion, and DFlash model/proposer tests. It was merged into vLLM `main`
  as commit `0b790a2`.
- Local `.tmp_vllm_v020` contains the needed runtime files:
  `vllm/v1/spec_decode/dflash.py`,
  `vllm/model_executor/models/qwen3_dflash.py`, and
  `register_speculator("dflash")`.
- The current remote vLLM site
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_17_0_extract_py312`
  is missing `qwen3_dflash.py` and only registers `eagle3`.
- A reusable probe was added:
  `experiments/eagle3_qwen3_235b/probe_vllm_dflash_support.py`, submitted by
  `submit_qwen235b_dflash_vllm_support_probe.sh`.
- A DFlash-capable vLLM source-build wrapper was added:
  `experiments/eagle3_qwen3_235b/submit_vllm_native_source_build_dflash.sh`.
  It builds from `git+https://github.com/vllm-project/vllm.git@0b790a2` and
  fails the build unless DFlash config registration, Qwen3 DFlash model import,
  and DFlash proposer import all pass. Source-build job `3179079` was submitted
  after lowering the time limit to `04:00:00`, and dependent support probe
  `3179087` is queued with `afterok:3179079`.
- The first DFlash standalone performance gate is queued behind the support
  probe: baseline `3179107` and DFlash K5 `3179108`, OpenMath `ISL=1024`,
  `OSL=1024`, batch sizes `1 2 4 8 16 32`, target/draft `TP=4`.
- Poll helper: `scripts/poll_qwen235b_dflash_chain_status.sh`.
- Result plotter: `scripts/plot_qwen235b_dflash_openmath_gate.py`.
- A retry submit was attempted after adding the wrapper, but the login node's
  `/lustre/fs1` mount was in `Cannot send after transport endpoint shutdown`
  state. Because the checkpoint path and container image are both under
  `/lustre`, the next probe/benchmark must wait for filesystem recovery or use
  a non-Lustre staging path plus an accessible container.

Latest execution status at 20:36 PDT:

- PARD/PARD-2 remains the primary actionable path for Qwen3-235B full-GRPO
  because it already uses vLLM's `draft_model` parallel-drafting runtime and
  has produced the best in-launcher generation result so far.
- The active no-stop full-GRPO pair is baseline `3177855` and local CAT/TPP-mask
  K5 `3177856`, both `MAX_STEPS=20`, `stop_after_generation=false`, and still
  pending for priority. At 20:39 PDT, both had scheduler candidate
  `StartTime=2026-06-05T22:46:23`; stdout logs were still missing, which is
  expected before start. This is the pair needed before claiming E2E full-GRPO
  speedup. Poll helper: `scripts/poll_qwen235b_fullgrpo20_status.sh`.
- The best completed full-GRPO-related result is still the
  stop-after-generation pair `3175807` / `3175808`: `1.718x` generation
  throughput speedup with `53.5%` acceptance. Treat this as generation-segment
  proof, not E2E full-GRPO proof.
- The DFlash runtime chain was restarted after the first source build failed on
  a Torch header mismatch. New chain:
  source build `3179221`, support probe `3179225`, OpenMath baseline `3179227`,
  and DFlash K5 `3179228`. The retry build has the FP8 header shim enabled and
  is running; dependent jobs are pending on `afterok`. At 20:48 PDT, `3179221`
  had reached `17:22` elapsed with no failure markers, passing the previous
  `3179079` failure time of `16:50`. This suggests the shim cleared the
  immediate `Float8_e4m3fnuz.h` build blocker, though the source build is not
  complete yet.

Latest update at 20:54 PDT:

- Source-build retry `3179221` ultimately failed at `18:29`: the first shim
  fixed `Float8_e4m3fnuz.h`, but the same vLLM file also includes
  `torch/headeronly/util/Float8_e4m3fn.h`. The build shim now creates both
  wrapper headers and forwards them to the container's `c10/util` FP8 headers.
- The active DFlash chain is now source build `3179391`, support probe
  `3179395`, OpenMath baseline `3179397`, and DFlash K5 `3179398`. At 20:54
  PDT, `3179391` was running on `nvl72120-T18`; the probe and OpenMath jobs
  were dependency-pending. Poll helper and plotter defaults were updated to
  this chain.
- The active no-stop full-GRPO pair remains baseline `3177855` and K5
  `3177856`; both are still `PENDING (Priority)` with no stdout logs. There is
  still no completed no-stop E2E full-GRPO speedup number.
- A quick fallback audit of the existing vLLM `0.17.0` extracted site shows
  that DFlash is not just a missing `register_speculator("dflash")` entry. The
  site also lacks `SpeculativeConfig.use_dflash`, `DFlashProposer`,
  `copy_and_expand_dflash_inputs_kernel`, `llm_base_proposer.py` DFlash
  changes, `gpu_model_runner.py` DFlash dispatch, and the `DFlashDraftModel`
  model/registry entry. Therefore a small Python overlay is not a clean
  fallback; the source build remains the primary runtime path unless we decide
  to maintain a larger backport.

Latest update at 21:13 PDT:

- Source-build retry `3179391` failed at `16:16` on
  `torch/headeronly/util/BFloat16.h`. This confirms the mismatch is broader
  than FP8: vLLM's stable-ABI helper code expects PyTorch `torch/headeronly`
  wrappers that are absent from the NeMo Torch `2.8.0a0+nv25.05` container.
- The build wrapper now creates a minimal `torch/headeronly` compatibility tree
  for `BFloat16`, `Half`, the two FP8 types, `Exception`, `shim_utils`,
  `core/ScalarType`, and `core/Dispatch`. It forwards scalar types to `c10`
  and supplies the `THO_DISPATCH_*` / `STD_TORCH_CHECK` macros used by vLLM's
  stable libtorch helpers.
- New active chain: source build `3179564`, support probe `3179565`, OpenMath
  baseline `3179567`, and DFlash K5 `3179568`.

Latest update at 21:32 PDT:

- No new true full-GRPO E2E metric has completed. The active no-stop pair
  remains baseline `3177855` and local CAT/TPP-mask K5 `3177856`; both are
  still `PENDING (Priority)`, elapsed `0:00`, with no stdout logs. A follow-up
  21:33 PDT `scontrol` poll showed scheduler candidate starts of
  `2026-06-05T23:20:00` for `3177855` and `2026-06-05T23:40:19` for
  `3177856`; treat these as backfill estimates, not guaranteed starts.
- The best completed Qwen3-235B NeMo-RL result is still generation-only:
  stop-after-generation jobs `3175807` / `3175808` show `1.718x` generation
  throughput, `1.774x` policy-generate timer speedup, and `53.5%` acceptance.
- DFlash remains a secondary runtime path under validation. Source-build job
  `3179564` is still running at `19:08` elapsed with the broader
  `torch/headeronly` shim; support probe `3179565` and OpenMath baseline /
  DFlash jobs `3179567` / `3179568` are dependency-pending. No DFlash
  performance JSON exists yet.
- At 21:41 PDT, an overlap process check confirmed the DFlash vLLM source
  build is actively compiling rather than hung. It is slower than necessary
  because the current `srun env` did not forward `TORCH_CUDA_ARCH_LIST`, so the
  build is compiling multiple CUDA architectures. The wrapper has been patched
  for future retries to forward arch/build variables into the container, and
  the DFlash submit wrapper now defaults to `CMAKE_CUDA_ARCHITECTURES=100`.

Literature refresh at 21:36 PDT:

- PARD-2 is still the most directly actionable extension of the current local
  PARD path. The arXiv abstract says it changes the draft-model objective from
  token prediction toward consecutive acceptance length, and adds
  Confidence-Adaptive Token optimization for target-aligned weighting:
  <https://arxiv.org/abs/2605.08632>. This matches the local direction already
  under test: CAT/TPP-mask style training plus vLLM `parallel_drafting=true`.
- The public AMD PARD repository currently describes PARD-2 and reports strong
  speedups, but its update log says the PARD-2 code and model checkpoints will
  be released soon: <https://github.com/AMD-AGI/PARD>. Therefore the current
  local CAT/TPP-mask implementation is the practical bridge rather than waiting
  for a drop-in checkpoint.
- vLLM `v0.20.1` documentation now lists Parallel Draft Model / PARD as a
  first-class model-based speculation method and exposes `parallel_drafting` in
  `speculative_config`: <https://docs.vllm.ai/en/v0.20.1/features/speculative_decoding/>.
  This supports the strategy of keeping Qwen3-235B PARD experiments on the
  draft-model path while separately validating DFlash.
- DFlash is a newer block-diffusion alternative that drafts an entire block in
  one forward pass, with reported lossless acceleration above EAGLE-3:
  <https://arxiv.org/abs/2602.06036>. The vLLM Speculators docs describe it as
  block-parallel and under active validation:
  <https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/>.
  Because our current container vLLM does not support DFlash, the source-build
  chain is the necessary runtime gate before any Qwen3-235B measurement.
- DFlare and DDTree are worth tracking after DFlash works. DFlare adds
  layer-wise target-feature fusion for stronger block diffusion drafters
  (<https://arxiv.org/abs/2606.02091>), while DDTree builds a verification tree
  from DFlash block distributions (<https://arxiv.org/abs/2604.12989>). Both
  depend on a DFlash-like runtime substrate, so they are not immediate NeMo-RL
  implementation targets until DFlash support is working.
- D2SD is another June 2026 diffusion-drafter variant. It uses dual diffusion
  draft models and a confidence-guided prefix tree for better recovery after
  the first mismatch: <https://arxiv.org/abs/2606.04446>. It is relevant for
  future Qwen3-235B work, but its runtime shape is closer to DFlash/DDTree than
  to the current vLLM PARD path, so it should not displace the active
  CAT/TPP-mask PARD full-GRPO validation.

Latest update at 21:56 PDT:

- DFlash runtime validation is now chained off source-build retry4 `3180764`.
  The build log confirms the intended GB200-only path
  `CMAKE_CUDA_ARCHITECTURES=100` plus the injected torch header compatibility
  include directory.
- Downstream jobs are queued as `3180920` support probe, `3180951` OpenMath
  baseline, `3181077` OpenMath DFlash K=3, and `3180953` OpenMath DFlash K=5.
  All OpenMath jobs use `Qwen/Qwen3-235B-A22B`, the local DFlash checkpoint
  `dflash_openmath_reasoning_cot_smoke512_k5_aligned/checkpoints/0`,
  `ISL=1024`, `OSL=1024`, batch sizes `1 2 4 8 16 32`, `TP=4`, `draft_TP=4`,
  and `max_model_len=4096`.

Latest update at 22:19 PDT:

- No-stop Full-GRPO PARD validation still has no new E2E metric because all
  three jobs are still `PENDING (Priority)`:
  - baseline `3177855`, candidate start `2026-06-06T00:25:56` PDT
  - local CAT/TPP-mask PARD-2-style K5 `3177856`, candidate start
    `2026-06-06T01:28:49` PDT
  - public PARD K5 `3182758`, candidate start `2026-06-06T01:38:05` PDT
- Public PARD K5 `3182758` was added in the same worker32/full-GRPO shape as
  the local CAT checkpoint to separate "PARD-2-style checkpoint quality" from
  "full-GRPO runtime tail" effects. It shares baseline `3177855`, uses
  `amd/PARD-Qwen3-0.6B`, `parallel_drafting=true`, K5, `MAX_STEPS=20`,
  `stop_after_generation=false`, fixed decode `256`, generation `TP=4`, train
  `TP=2/PP=8/CP=2/EP=16`, and GBS `256`.
- The prior no-stop hard failure remains attributed to NeMo-RL/Megatron API
  compatibility, not to PARD-2. The remote `PackedSeqParams` guard is present
  and the container smoke passed. The active jobs are the first valid test of
  whether that fix lets the full training tail complete.
- `scripts/extract_qwen235b_tail2_metrics.py` now also extracts
  `E2E (Tokens/sec/gpu)`, `Generation Worker Group (Tokens/sec/gpu)`, and
  `Total step time` as latest/mean/count fields. The updated script was copied
  to the remote worktree and `python3 -m py_compile` passed there. Once logs
  exist, the same parser can report generation throughput speedup, generation
  step-time speedup, E2E throughput speedup, and E2E step-time speedup.
- DFlash runtime validation is now on retry5, not retry4: source build
  `3181912` is `RUNNING` in the vLLM wheel-build phase, with support probe
  `3181930` and OpenMath baseline / DFlash K3 / DFlash K5 jobs
  `3181932` / `3181956` / `3181937` pending on dependency. This retry builds
  vLLM commit `0b790a2`, disables `_C_stable_libtorch`, keeps the torch header
  compatibility shims, and targets GB200-only `CMAKE_CUDA_ARCHITECTURES=100`.

Latest update at 22:25 PDT:

- The no-stop Full-GRPO PARD/PARD-2 trio is still `PENDING (Priority)` and has
  not emitted logs. This means no new E2E speedup claim is valid yet.
- A matched-result summarizer was added and synced to the remote worktree:
  `scripts/summarize_qwen235b_fullgrpo_pard.py`. The full-GRPO poll helper now
  prints its table automatically. Once logs exist, it computes generation and
  E2E throughput/step-time speedups for local CAT/PARD-2-style K5 `3177856`
  and public PARD K5 `3182758` against baseline `3177855`.
- DFlash source-build retry5 `3181912` remains in vLLM wheel build at roughly
  `14:32` elapsed. No support-probe or OpenMath DFlash performance result is
  available yet.

Latest update at 22:46 PDT:

- PARD/PARD-2 Full-GRPO no-stop has been resubmitted with the corrected
  Qwen3-235B runtime-tail safeguards. The active jobs are baseline `3185585`,
  local CAT/TPP-mask PARD-2-style K5 `3185586`, and public PARD K5 `3185587`.
  All are `PENDING (Priority)` and have no logs yet.
- Cancelled superseded pending jobs:
  `3177855`, `3177856`, `3182758`, `3185571`, `3185572`, `3185573`.
- Fixes now in the remote runtime checkout: `PackedSeqParams.total_tokens`
  guard, SpecDec omitted-logprob repair path, greedy sampler-mismatch opt-in,
  and DDP overlap disabled for the Megatron logprob/reference tail. The remote
  Megatron train file already has the `temperature > 0.0` guard for greedy
  finite loss.
- Interpretation: this is still a throughput/E2E step-time diagnostic, not a
  valid learning-quality PARD-2 claim, because the run is greedy
  `temperature=0` with fixed 256-token decode.

Latest update at 22:55 PDT:

- PARD/PARD-2 Full-GRPO r3 remains queued, not failed:
  baseline `3185585`, local CAT/TPP-mask PARD-2-style K5 `3185586`, and public
  PARD K5 `3185587` are all `PENDING (Priority)` with no stdout/ray-driver
  logs.
- Public PARD/PARD-2 evidence was refreshed:
  AMD-AGI/PARD now exposes a public repo with PARD training/inference files,
  Qwen3 `PARD-Qwen3-0.6B` weights, and a README note that the PARD-2 paper was
  released on `2026-05-09` while PARD-2 code/checkpoints are still "released
  soon". The README reports PARD Qwen3 vLLM speedups and PARD-2 claims up to
  `6.94x`, but those are not yet drop-in verified for Qwen3-235B.
- DFlash retry5 `3181912` failed after successfully building the vLLM wheel:
  import probe hit `No module named 'triton.language.target_info'` and then
  `torch.ops._C` missing `cutlass_scaled_mm_supports_fp8` while importing the
  Qwen3-MoE path. Downstream probe/benchmark jobs `3181930`, `3181932`,
  `3181956`, and `3181937` were cancelled by dependency.
- DFlash retry6 was submitted after patching the source-build path so missing
  CUTLASS FP8/block-FP8/group-GEMM support ops return `False` instead of
  aborting import. Active chain:
  - build `3185614`, running at 22:55 PDT
  - support probe `3185615`, dependency-pending
  - OpenMath standalone baseline `3185616`, K3 `3185618`, K5 `3185621`,
    dependency-pending
- This keeps DFlash as the main non-PARD branch because vLLM Speculators docs
  describe DFlash as one-forward block drafting conditioned on target hidden
  states, and vLLM PR `38300` merged DFlash speculators config parsing,
  Qwen3-DFlash weight loading, and an E2E DFlash auto-detect test path.

Latest update at 23:00 PDT:

- DFlash retry6 `3185614` failed during the source-patch phase before wheel
  build. The local/source commit already had a `cutlass_group_gemm_supported`
  `AttributeError -> False` fallback, so the exact text anchor for an
  unnecessary group-GEMM patch was absent.
- DFlash retry7 removes that unnecessary group-GEMM exact-anchor patch and
  keeps only the FP8/block-FP8 missing support-op fallback. Active retry7
  chain:
  - build `3185715`
  - support probe `3185716`, `afterok:3185715`
  - OpenMath standalone baseline `3185717`, DFlash K3 `3185718`, DFlash K5
    `3185724`, all `afterok:3185716`
  `scripts/poll_qwen235b_dflash_chain_status.sh` now defaults to this retry7
  chain.

Latest update at 23:02 PDT:

- PARD/PARD-2 Full-GRPO r3 still has no result to parse. Baseline `3185585`,
  local CAT/TPP-mask PARD-2-style K5 `3185586`, and public PARD K5 `3185587`
  are all `PENDING (Priority)` with missing stdout logs. `scontrol` reports a
  candidate start of `2026-06-06T02:49:40` PDT for all three, but this is only
  a scheduler estimate.
- DFlash retry7 build `3185715` is `RUNNING` at `00:02:16` elapsed with no
  current failure marker. Support probe `3185716` and OpenMath standalone
  baseline/K3/K5 jobs `3185717` / `3185718` / `3185724` remain
  dependency-pending.

Latest update at 23:16 PDT:

- Active PARD/PARD-2-style fixed-decode diagnostic jobs remain queued, not
  failed: baseline `3185585`, local CAT/TPP-mask K5 `3185586`, and public PARD
  K5 `3185587` are all `PENDING (Priority)` with no logs. Treat these as
  fixed-work throughput/E2E timing diagnostics only.
- A separate real-sampling GRPO-tail smoke was submitted to avoid the
  `temperature=0`/fixed-length semantic issue:
  - baseline `3186018`
  - local CAT/TPP-mask PARD-2-style K5 `3186020`
  - public PARD K5 `3186021`
  These jobs use `temperature=1.0`, natural EOS, `max_new_tokens=256`,
  generation `TP=4`, and `MAX_STEPS=2`; all are currently `PENDING (Priority)`.
- DFlash retry7 build `3185715` is still running in vLLM wheel build; probe
  `3185716` and DFlash OpenMath baseline/K3/K5 jobs `3185717` / `3185718` /
  `3185724` remain dependency-pending.

Latest method evidence refresh at 23:19 PDT:

- PARD-2 remains aligned with our failure mode because its CAT objective shifts
  training from token accuracy toward accepted-prefix length. The paper page
  points to `https://github.com/AMD-AGI/PARD`, but for this project the runnable
  artifact is still best described as `local CAT/TPP-mask PARD-2-style`, not
  official PARD-2.
- vLLM Speculators' current decision guide lists Eagle-3, P-EAGLE, and DFlash as
  supported algorithms. This supports keeping DFlash and P-EAGLE/EAGLE-3.1 as
  the non-PARD branches rather than trying older Medusa/ngram-style methods for
  Qwen3-235B.
- P-EAGLE is relevant because it removes the K-forward-pass draft bottleneck
  from EAGLE-style drafters. The vLLM blog reports `1.05x-1.69x` over vanilla
  EAGLE-3 on B200/GPT-OSS workloads, while the paper reports `1.10x-1.36x`
  over autoregressive EAGLE-3 across GPT-OSS and Qwen3-Coder 30B. This is not a
  direct Qwen3-235B result, but it attacks the same overhead problem seen when
  EAGLE/PARD acceptance is nonzero but throughput gains are weak.
- EAGLE 3.1 is relevant to the existing Qwen3-235B EAGLE3 underperformance
  because it explicitly targets attention drift and long-context robustness. The
  vLLM announcement says EAGLE 3.1 can deliver up to `2x` longer acceptance
  length versus EAGLE 3 in long-context workloads and is integrated as a
  config-driven extension of the EAGLE3 path. No Qwen3-235B public EAGLE 3.1
  draft is available yet, so this is a training branch rather than a drop-in
  benchmark branch.
- DFlash remains worth pursuing because it predicts a block in one forward pass
  and vLLM issue `#38240` was closed through DFlash support work. A current
  caveat is that DFlash uses non-causal attention and has reported
  incompatibility with FP8 KV-cache paths; our Qwen3-235B DFlash chain should
  keep `kv_cache_dtype=auto`/bf16 unless upstream adds non-causal FP8 support.
