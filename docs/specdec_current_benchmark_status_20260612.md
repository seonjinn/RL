# SpecDec Current Benchmark Status - 2026-06-12

## Latest Update - 22:49 CEST

Refreshed:

- Lyris Qwen8/Qwen30 long-OSL SWE-Bench matrix
- Lyris SWE-Bench Verified OSL32K high-K sweep
- Lyris Qwen3-235B SWE-Bench method/K sweep
- Lyris Qwen3-235B SWE-Bench Verified OSL64K pilot launch
- Lyris Qwen3-235B SWE-Bench Verified OSL64K pilot live-start
- Lyris Qwen3-30B-A3B Verified OSL128K PARD K5 timeout transition
- Lyris Qwen3-30B-A3B Full OSL128K PARD K5 timeout transition
- Lyris Qwen3-8B Full OSL128K PARD-2 K5 timeout transition
- Lyris Qwen3-8B SWE-Bench Verified OSL128K CUDA diagnostics
- Lyris Qwen3-8B SWE-Bench Verified OSL96K baseline bracketing and dependent specdec jobs
- OCI NeMo-RL online PARD-2 / suffix Full-GRPO metrics
- Pre-submit NeMo-RL online drafter code audit
- OCI Qwen30B no-SpecDec Full-GRPO long-output batch-long baseline retry
- OCI standalone SWE-Bench smoke metrics
- Expected-performance artifacts:
  - `docs/lyris_specdec_expected_performance_20260612.html`
  - `docs/lyris_specdec_expected_performance_20260612.png`
  - `docs/lyris_specdec_expected_performance_raw_20260612.csv`
- Lyris official PARD-2 target-feature K=1 diagnostic
- Lyris Qwen3-235B OSL64K PARD K=3 diagnostic preflight and submission
- Lyris Qwen3-8B SWE-Bench Verified OSL96K suffix final row
- Lyris official PARD-2 target-feature smoke launcher review and corrected `r3` completion
- Lyris official PARD-2 target-feature Qwen8 OSL4096 K=1 diagnostic submission
- Lyris matching Qwen8 OSL4096 no-spec baseline submission
- OCI staged NeMo-RL official PARD-2 online smoke submission and retry after code review
- OCI and Lyris NeMo-RL official PARD-2 online 20-step status
- Post-terminal Lyris/OCI refresh and regenerated expected-performance artifacts
- Lyris Qwen3-8B / Qwen3-30B-A3B MATH500 OSL32K pre-submit review, submission, and refresh artifact
- Lyris official PARD-2 online import-fix retry
- First partial Lyris MATH500 OSL32K metric row
- Lyris MATH500 OSL32K and official online-PARD2 no-duplicate post-submit review
- Lyris Qwen3-8B MATH500 suffix K32 completion for BS1/BS2
- Lyris MATH500 / online PARD-2 follow-up poll with no new final rows
- Lyris Qwen3-30B-A3B MATH500 suffix K32 BS1 completion
- Lyris MATH500 / online PARD-2 follow-up poll with active-job progress
- Lyris MATH500 / online PARD-2 follow-up poll with no new final rows
- Lyris MATH500 / online PARD-2 post-submit code review with no duplicate submission
- Lyris online PARD-2 maskfix retry submission
- Lyris MATH500 new Eagle-3/Qwen30 suffix rows and regenerated expected-performance bundle
- Lyris MATH500 / online PARD-2 follow-up poll with no new final rows
- Lyris MATH500 / online PARD-2 follow-up poll with no new final rows after retry build progress
- Lyris MATH500 / online PARD-2 follow-up poll with dependency-build progress
- Lyris online PARD-2 W&B/TIS retries and Qwen30 MATH500 BS2 baseline completion
- Lyris MATH/SWE/Qwen235B refresh after TIS retry submission
- Lyris online PARD-2 TIS retry active compile confirmation
- Lyris online PARD-2 RayVirtualCluster port-range compatibility retry
- Lyris online PARD-2 vLLM Ray env-propagation compatibility retry
- Lyris online PARD-2 optional vLLM ViT backend patch compatibility retry

The expected-performance raw table has 187 data rows after the latest regeneration. The plot script defaults include the current long-OSL, 32K high-K sweep, Qwen3-235B final metrics, live partial rows, and the current MATH500 final rows. The MATH500 OSL32K metrics CSV now has 16 final metric rows, including the completed Qwen30 BS2 baseline. The refreshed long-OSL metrics CSV has 65 final benchmark rows, and the long-OSL status snapshot is 44 completed / 9 failed / 4 cancelled / 5 timeout. The Qwen3-235B status manifest tracks 38 jobs and is now terminal: 33 completed / 5 failed. At Qwen235B OSL64K, baseline/suffix/Eagle-3 have completed final rows; PARD K5, PARD K3, and native PARD-2 failed with CUDA illegal-memory / vLLM `EngineDeadError` class failures. The 15:29 CEST pre-submit review found that the remote OCI NeMo-RL checkout still only has the guard/fallback path for official PARD-2 online training, so I did not submit a NeMo-RL official online-PARD2 job from that checkout.

At 17:44 CEST, the corrected Lyris official PARD-2 online job `2107214` failed before GRPO setup after dependency build. Root cause was a staged source mismatch, not a PARD-2 runtime error: `nemo_rl/algorithms/advantage_estimator.py` and `grpo.py` imported `get_gdpo_reward_component_keys`, but the staged `nemo_rl/algorithms/utils.py` lacked that helper. I patched the overlay utility to provide a small numeric-order `reward1`, `reward2`, ... key extractor, reran local `bash -n` / `py_compile`, and reran the Lyris dry-run preflight. The dry-run passed remote `py_compile`, `scripts/test_vllm_draft_refit_target_proj.py`, and all `PARD2_OFFICIAL_PATCH_CHECKS`. Retry job `2107361` was submitted with run id `20260612_qwen8_pard2_official_online_lyris_gdpoimportfix`; it is running on `lyris0230`, the staged source contains the new helper, and the early log has passed the official-PARD2 patch checks without the previous import error.

At 17:34 CEST, I reviewed the Lyris MATH submit path before launching new jobs. Local `bash -n` passed for the Lyris standalone and official-PARD2 smoke launchers; local `py_compile` passed for `standalone_vllm_specdec_breakdown.py` and the OpenMath materializer; the remote Lyris copy of `standalone_vllm_specdec_breakdown.py` also compiled. The immediate compatibility issue was that the small local MATH prompt source uses a `data` field, so the prompt loader now accepts `data` in addition to `prompt`, `question`, `problem`, and `input`. I staged `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/data/math_500_data_prompts_20260612.jsonl` on Lyris and verified 20 parseable rows.

New Lyris Qwen3-8B MATH500 OSL32K jobs submitted under account `coreai_dlalgo_llm`, all currently `RUNNING`:

| Job | Method | K | Prompt count | Batch sizes | Logs |
| ---: | --- | ---: | ---: | --- | --- |
| `2107295` | baseline |  | 4 | `1 2` | `qwen8_math500_osl32k_lyris_baseline_offset0_n4_isl4096_osl32768_bs1_2_20260612_math` |
| `2107296` | suffix | 32 | 4 | `1 2` | `qwen8_math500_osl32k_lyris_suffix_offset0_n4_isl4096_osl32768_bs1_2_k32_arctic_20260612_math` |
| `2107297` | Eagle-3 | 3 | 4 | `1 2` | `qwen8_math500_osl32k_lyris_eagle3_offset0_n4_isl4096_osl32768_bs1_2_k3_20260612_math` |
| `2107298` | official PARD-2 | 3 | 4 | `1 2` | `qwen8_math500_osl32k_pard2_official_k3_20260612` |
| `2107302` | official PARD-2 | 5 | 4 | `1 2` | `qwen8_math500_osl32k_pard2_official_k5_20260612` |

Immediate launch log scan was clean: the baseline/suffix/Eagle-3 jobs reached vLLM engine initialization, and both official-PARD2 jobs passed all `PARD2_OFFICIAL_PATCH_CHECKS` before vLLM engine initialization. No `Traceback`, CUDA, OOM, `EngineDeadError`, or import error was present in the initial log tails.

At 17:40 CEST, I submitted the missing generic-PARD MATH comparison for Qwen3-30B-A3B after confirming there was no duplicate Qwen30 MATH/PARD run on Lyris. The same reviewed Lyris standalone launcher and staged MATH prompt file were used. New jobs:

| Job | Model | Method | K | Prompt count | Batch sizes | Logs |
| ---: | --- | --- | ---: | ---: | --- | --- |
| `2107332` | Qwen3-30B-A3B | baseline |  | 4 | `1 2` | `qwen30ba3b_math500_osl32k_lyris_baseline_offset0_n4_isl4096_osl32768_bs1_2_20260612_math` |
| `2107333` | Qwen3-30B-A3B | suffix | 32 | 4 | `1 2` | `qwen30ba3b_math500_osl32k_lyris_suffix_offset0_n4_isl4096_osl32768_bs1_2_k32_arctic_20260612_math` |
| `2107334` | Qwen3-30B-A3B | PARD | 5 | 4 | `1 2` | `qwen30ba3b_math500_osl32k_lyris_pard_offset0_n4_isl4096_osl32768_bs1_2_k5_20260612_math` |
| `2107335` | Qwen3-30B-A3B | PARD | 3 | 4 | `1 2` | `qwen30ba3b_math500_osl32k_pardk3_lyris_pard_offset0_n4_isl4096_osl32768_bs1_2_k3_20260612_math` |

The generated remote configs are valid: both PARD arms use `amd/PARD-Qwen3-0.6B`, `parallel_drafting=true`, and `num_speculative_tokens` set to K3/K5 respectively. I added `scripts/refresh_lyris_math500_osl32k_results.sh`, which now writes `docs/lyris_math500_osl32k_manifest_20260612.csv`, `docs/lyris_math500_osl32k_status_20260612.csv`, `docs/lyris_math500_osl32k_metrics_20260612.csv`, and `docs/lyris_math500_osl32k_status_20260612.md`. At 17:48 CEST, I fixed this refresh script for macOS Bash 3 compatibility by replacing `mapfile` with a read loop, then reran it successfully.

First partial MATH row extracted at 17:48 CEST:

| Model | Method | Batch | tok/s/GPU | Acceptance | Mean accept len | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B | suffix K32 | 1 | 216.02 | 84.51% | 6.88 | No speedup yet because the matching Qwen8 baseline row is still running. |

The same refresh shows all 9 MATH jobs still `RUNNING`; no final Qwen8 baseline, Qwen8 PARD-2, Qwen8 Eagle-3, or Qwen30 rows have completed yet.

At 17:51 CEST, I reran the preflight review before considering any additional Lyris submission. `bash -n` passed for the Lyris online/standalone launchers and MATH refresh script, and `py_compile` passed for the PARD-2 utility overlay, standalone vLLM runner, metrics extractor, and refit guard test. I did not submit a duplicate job because all reviewed MATH jobs are already running, and the online-PARD2 retry `2107361` is already active. The refreshed MATH status remains 9 `RUNNING` jobs with the same single completed row, Qwen3-8B suffix K32 BS1 at `216.02` tok/s/GPU and `84.51%` acceptance. A focused log scan found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in the current MATH job tails; Qwen30 suffix has begun BS1 and is showing high live acceptance in the first chunk, but no final Qwen30 row exists yet. `2107361` remains `RUNNING` on `lyris0230`, has passed all `PARD2_OFFICIAL_PATCH_CHECKS`, has installed the first dependency batch, and has not repeated the previous `get_gdpo_reward_component_keys` import failure; it is still building native dependencies and has not reached `SETUP COMPLETE`.

At 17:55 CEST, Qwen3-8B MATH500 suffix K32 job `2107296` completed successfully and wrote both batch-size rows. Refreshed `docs/lyris_math500_osl32k_metrics_20260612.csv` now has: BS1 `216.02` tok/s/GPU, `84.51%` acceptance, mean accept length `6.88`; BS2 `469.83` tok/s/GPU, `91.24%` acceptance, mean accept length `9.48`. Speedup remains blank because the matching Qwen3-8B no-spec baseline `2107295` is still running. The current MATH queue is 1 completed / 8 running. Qwen30 suffix has live progress through at least two BS1 chunks, but no final Qwen30 `breakdown.json` exists yet. Online PARD-2 retry `2107361` remains `RUNNING`; it is still before `SETUP COMPLETE`, with no new import/runtime error in the scanned driver log tail.

I also updated `scripts/plot_lyris_specdec_expected_performance.py` so the expected-performance HTML/PNG/CSV bundle includes `docs/lyris_math500_osl32k_metrics_20260612.csv` by default. MATH500 rows are now labelled with a `MATH500` shape prefix so they do not collide with SWE-Bench Full OSL32K rows that share the same model/ISL/OSL/batch. The regenerated raw table has 173 rows and includes both Qwen3-8B MATH500 suffix K32 batch rows; their speedup cells remain blank until the MATH baseline finishes.

Final no-submit decision for this pass: do not launch duplicate Lyris jobs. The reviewed MATH arms are already active except completed suffix `2107296`, and the online PARD-2 retry `2107361` is already active. Latest poll still shows the remaining 8 MATH jobs `RUNNING`; `2107361` has built several native dependencies after passing source checks, but is not yet at `SETUP COMPLETE`.

At 17:58 CEST, refreshed `docs/lyris_math500_osl32k_status_20260612.md` again. The MATH status is unchanged at 1 completed / 8 running, with 2 final metric rows: Qwen3-8B suffix K32 BS1/BS2. Qwen30 suffix has live progress through at least three BS1 chunks, but no final Qwen30 `breakdown.json` exists yet. Qwen8 Eagle-3 has live BS1 chunk progress at `52.35` tok/s/GPU and `33.82%` acceptance for chunk 1/4, but no final row yet. The online PARD-2 retry `2107361` remains before `SETUP COMPLETE`; it has built `flash-attn`, `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` after passing source checks, with no repeated import/runtime error in the scanned log tail. No additional job was submitted.

At 18:00 CEST, the MATH refresh found a new final row: Qwen3-30B-A3B suffix K32 BS1 from job `2107333`, `146.74` tok/s/GPU, `88.43%` acceptance, mean accept length `7.85`. The Qwen30 no-spec baseline `2107332` is still running, so the speedup cell remains blank. The expected-performance HTML/PNG/CSV bundle was regenerated; `docs/lyris_specdec_expected_performance_raw_20260612.csv` now has 174 rows and includes three MATH500 final rows: Qwen30 suffix BS1 plus Qwen8 suffix BS1/BS2. Remaining MATH jobs are still active, and no new job was submitted.

At 18:02 CEST, the MATH refresh still has 3 final rows and 8 running jobs. No new final `breakdown.json` appeared after the Qwen30 suffix BS1 row. Active-tail scan found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in the checked MATH job tails. Qwen8 Eagle-3 now has live BS1 chunks 1/4 and 2/4: chunk 1 `52.35` tok/s/GPU with `33.82%` acceptance, chunk 2 `90.43` tok/s/GPU with `79.49%` acceptance. Qwen30 suffix has the completed BS1 aggregate row and is still running for later work. Online PARD-2 retry `2107361` remains `RUNNING` and before `SETUP COMPLETE`, still in native dependency build. No duplicate job was submitted.

At 18:04 CEST, the MATH refresh still has 3 final rows and 8 running jobs. Qwen8 baseline has live progress for BS1 chunk 1/4 at `36.87` tok/s/GPU, but no aggregate baseline row exists yet, so MATH speedups remain blank. Qwen8 Eagle-3 remains live through BS1 chunk 2/4; Qwen30 suffix has only the final BS1 row so far. A focused log scan again found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in the checked MATH job tails. Online PARD-2 retry `2107361` is still `RUNNING` before `SETUP COMPLETE`, with no new error in the scanned driver log. No plot regeneration or new job submission was needed because no new final metric row appeared.

At 18:08 CEST, I reran the pre-submit/post-submit review before considering any additional Lyris job. Local `bash -n` passed for the Lyris online launcher, shared online launcher, PARD-2 vLLM site helper, standalone vLLM launcher, and MATH refresh script; `py_compile` passed for the standalone runner, metrics scripts, plot script, and PARD-2 utility overlay. Remote Lyris validation on the staged online-PARD2 checkout also passed `bash -n`, Python compile, and all `PARD2_OFFICIAL_PATCH_CHECKS`; the staged MATH prompt file still has 20 parseable rows with nonempty `data`. The refreshed MATH status is unchanged at 1 completed / 8 running, with the same 3 final suffix rows and no baseline aggregate yet. Active-tail scan remains clean for `Traceback`, CUDA/OOM, `EngineDeadError`, and import errors. Qwen30 PARD K5 has live BS1 chunk 1/4 at `41.98` tok/s/GPU and `52.32%` acceptance; Qwen30 suffix is running BS2 chunk 1/2. Lyris online PARD-2 retry `2107361` is still `RUNNING` on `lyris0230`, elapsed `00:21:28`, before `SETUP COMPLETE`, with no repeat of the GDPO helper import failure. I did not submit duplicates because the reviewed MATH and online-PARD2 arms are already active.

At 18:14 CEST, Lyris online PARD-2 retry `2107361` moved terminal `FAILED` after dependency build with another staged-source import mismatch: `mask_out_neg_inf_logprobs` was imported by the PARD-2 loss/train overlay but missing from the staged `nemo_rl.algorithms.utils`. I verified the helper in the Lyris base checkout, patched `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/algorithms/utils.py` with the same `-inf` logprob masking behavior, and reran validation. Local `py_compile`/`bash -n`, Lyris dry-run staging, remote refit guard test, and all `PARD2_OFFICIAL_PATCH_CHECKS` passed. Submitted maskfix retry `2107520` with run id `20260612_qwen8_pard2_official_online_lyris_maskfix`; it is `RUNNING` on `lyris0216`. Early `2107520` log has passed source checks and is rebuilding native dependencies, with no `ImportError`/`Traceback` in the scanned tail. MATH500 OSL32K remained unchanged at 3 final suffix rows and 8 running jobs, so I did not regenerate the expected-performance bundle.

At 18:19 CEST, MATH500 OSL32K refresh added two final rows and the expected-performance bundle was regenerated. Current MATH status is 2 completed / 7 running across 9 tracked jobs; final MATH rows are now Qwen8 suffix K32 BS1/BS2, Qwen8 Eagle-3 K3 BS1, and Qwen30 suffix K32 BS1/BS2. New rows: Qwen8 Eagle-3 K3 BS1 `74.38` tok/s/GPU with `62.94%` acceptance and mean accept length `2.89`; Qwen30 suffix K32 BS2 `303.81` tok/s/GPU with `92.52%` acceptance and mean accept length `9.81`. Matching Qwen8 and Qwen30 no-spec baselines are still running, so MATH speedups remain blank. `docs/lyris_specdec_expected_performance_raw_20260612.csv` now has 176 rows and includes all 5 MATH500 rows. Active MATH log scan remained clean for `Traceback`, CUDA/OOM, `EngineDeadError`, and import errors. Online PARD-2 retry `2107520` is still `RUNNING` on `lyris0216`, elapsed `00:06:38`, before `SETUP COMPLETE`, still in native dependency build and with no `ImportError`/`Traceback` in the scanned tail.

At 18:21 CEST, the MATH refresh still has 5 final rows and 7 running jobs. No new `breakdown.json` appeared, so the expected-performance bundle remains at 176 raw rows and was not regenerated again. Baselines are still running, so MATH speedup cells remain blank. Active-tail scan found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors. Live progress now includes Qwen8 baseline BS1 chunks 1/4 and 2/4 at about `36.9` tok/s/GPU, Qwen30 PARD K5 BS1 chunks 1/4 and 2/4 at `41.98` and `39.90` tok/s/GPU with `52.32%` and `47.50%` acceptance, and Qwen30 PARD K3 BS1 chunk 1/4 at `37.38` tok/s/GPU with `71.64%` acceptance. Online PARD-2 retry `2107520` is still `RUNNING` on `lyris0216`, elapsed `00:08:34`, before `SETUP COMPLETE`, still in native dependency build and with no new `ImportError`/`Traceback` in the scanned tail. No duplicate job was submitted.

At 18:23 CEST, the MATH refresh still has 5 final rows and 7 running jobs; expected-performance raw rows remain 176, so no regeneration was needed. Baselines are still running and MATH speedups remain blank. Active-tail scan again found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in checked MATH logs. Live progress is unchanged in shape: Qwen8 baseline BS1 chunk 2/4 is complete at about `36.82` tok/s/GPU, Qwen30 PARD K5 has BS1 chunks 1/4 and 2/4 at `41.98` and `39.90` tok/s/GPU with `52.32%` and `47.50%` acceptance, and Qwen30 PARD K3 has BS1 chunk 1/4 at `37.38` tok/s/GPU with `71.64%` acceptance. Online PARD-2 retry `2107520` remains `RUNNING` on `lyris0216`, elapsed `00:10:52`, before `SETUP COMPLETE`, still in native dependency build; the scanned tail shows `deep-ep` built and no `ImportError`/`Traceback`. No new job was submitted.

At 18:28 CEST, the MATH refresh still has 5 final rows and 7 running jobs; expected-performance raw rows remain 176, so no regeneration was needed. Baselines are still running and MATH speedups remain blank. Active-tail scan again found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in checked MATH logs. Live MATH progress remains partial: Qwen8 baseline has BS1 chunks 1/4 and 2/4 complete at about `36.9` tok/s/GPU, Qwen30 PARD K5 has BS1 chunks 1/4 and 2/4 complete at `41.98` and `39.90` tok/s/GPU with `52.32%` and `47.50%` acceptance, and Qwen30 PARD K3 has BS1 chunk 1/4 complete at `37.38` tok/s/GPU with `71.64%` acceptance. Online PARD-2 retry `2107520` remains `RUNNING` on `lyris0216`, elapsed `00:15:36`, before `SETUP COMPLETE`, still in native dependency build; the scanned tail shows `flash-attn`, `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` built, with no `ImportError`/`Traceback`. No new job was submitted.

The 14:11 CEST pre-submit overlay review found a separate hard blocker: the local vLLM `method=pard2` path is currently a native draft-model alias, not official PARD-2 target-feature conditioning. `DraftModelProposer` sets `pass_hidden_states_to_model=False`, PARD-2 does not enable auxiliary target-layer capture, Qwen3 does not accept the `hidden_states` kwarg, and there is no `target_proj` / `warp_model.bin` loader-injection path. Because of that, I did not submit any new official PARD-2 online jobs from `remote_patch_pard2_official/`. Existing rows labelled native PARD-2 or PARD-2 in the vLLM tables should be read as alias/draft-model-path measurements unless a later source patch explicitly adds target-feature conditioning.

At 14:42 CEST, a vLLM source patch for official PARD-2 target-feature conditioning was prepared at `experiments/eagle3_qwen3_235b/patches/vllm_pard2_official_target_feat.patch`. Local syntax and patch dry-run checks pass, but it still needs a Lyris/container smoke before any official PARD-2 online benchmark job is submitted or claimed.

At 14:54 CEST, the official PARD-2 target-feature runtime path had passed Lyris smokes, but acceptance stayed too low for SWE-Bench: K=3 job `2104955` accepted 0 of 381 draft tokens, and K=1 job `2104975` accepted 2 of 506 draft tokens (`0.395%`). Lowering K did not solve the acceptance issue, so no long official PARD-2 target-feature benchmark was submitted from that path.

After reviewing the Qwen3-235B OSL64K launcher and remote inputs, a narrower PARD K=3 diagnostic was submitted instead of rerunning the full matrix. Job `2104995` is running on `lyris0220` with `Qwen/Qwen3-235B-A22B`, SWE-Bench Verified, `ISL=4096`, `OSL=65536`, `BS=1`, `TP=4`, `fp8` KV cache, `amd/PARD-Qwen3-0.6B`, and `num_speculative_tokens=3`. The generated remote config is valid, the job passed model/drafter load, and live generation telemetry is now present. Early live rows show generation throughput around 10.8-18.2 tok/s and average draft acceptance around 34-53%, but no final `breakdown.json` exists yet.

At 15:16 CEST, the official PARD-2 target-feature smoke launcher completed as `2106641` with tag `pard2_official_target_feat_smoke_r3_20260612`. The prior runs failed before generation: `2104851` hit a broad method-alias patch conflict, and `2106398` passed patch/source checks but then failed Pydantic validation because the alias helper skipped the exact `SpeculativeMethod` Literal entry. The launcher now checks for the exact `    "pard2",` Literal line and the in-job source check verifies `literal_accepts_pard2=True` before model startup. `2106641` completed `0:0` on `lyris0098`, loaded target and drafter, selected PARD-2 target layers `(36, 29, 21, 13)`, and wrote `breakdown.json`: 21.50 tok/s/GPU, 0.00% acceptance, 381 drafted, 0 accepted.

At 15:29 CEST, I patched the local NeMo-RL submission helper so `DRAFT_FORMAT=pard2` maps to `SPECDEC_METHOD=pard2`, enables parallel drafting, and recomputes default PARD-2 loss/CAT settings after auto-selecting `POLICY_DRAFT_TYPE=pard2`. Local `bash -n`, `py_compile`, and `scripts/test_vllm_draft_refit_target_proj.py` pass. I did not sync this into the live OCI checkout because active/pending jobs use that shared tree and the official training overlay is still not installed there. Instead, I submitted a narrow Lyris static-vLLM diagnostic, job `2106663`, using the already-smoked official PARD-2 target-feature patch: Qwen3-8B target, `amd/PARD2-Qwen3-8B`, SWE-Bench Verified prompts, `ISL=2048`, `OSL=4096`, `K=1`, `prompt_count=4`, account `coreai_dlalgo_llm`.

At 15:49 CEST, `2106663` had completed `0:0` on `lyris0179`. Result: 22.93 tok/s/GPU, 54 accepted of 16,326 drafted tokens, `0.33%` acceptance, mean acceptance length `1.003`. Longer OSL did not improve official target-feature PARD-2 acceptance on SWE-Bench Verified. I submitted matching no-spec baseline `2106734` for the same Qwen8 Verified OSL4096 shape.

At 16:18 CEST, matching no-spec baseline `2106734` had completed `0:0` on `lyris0171`: 36.75 tok/s/GPU and 445.87 s latency for the same Qwen8 Verified OSL4096 shape. The official target-feature PARD-2 diagnostic is therefore `0.624x` versus no-spec baseline and should be treated as a negative result, not an optimization.

OCI staged NeMo-RL official online-PARD2 smoke `3278974` failed before GRPO because the vLLM patch-prep step used `/opt/nemo_rl_venv/bin/python`, which could not import `vllm`. I reviewed and patched the staged launcher path before resubmitting:

- `prepare_pard2_official_vllm_site.sh` now supports an explicit clean vLLM source site and uses static source-marker checks instead of importing vLLM/torch for validation.
- `submit_nemorl_online_draft_specdec.sh` wires through `PARD2_VLLM_SOURCE_SITE`.
- `submit_qwen8_pard2_official_online_smoke_20260612.sh` points at the clean base actor-venv vLLM source and preflights that path.

Validation before retry: local `bash -n` / `py_compile` passed, remote dry-run passed, and direct login-node patch prep built `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat` with all official PARD-2 source checks true. Resubmitted OCI job `3279154`: Qwen3-8B official PARD-2 online smoke, `K=1`, `MAX_STEPS=2`, account `coreai_dlalgo_llm`. Initial state: pending on OCI priority.

At 16:36 CEST, OCI job `3279154` had run and failed after clearing the previous launcher/vLLM setup gate. It built the patched vLLM site, all `PARD2_OFFICIAL_PATCH_CHECKS` were true, Ray connected, vLLM workers loaded target and drafter, and PARD-2 target layers `(36, 29, 21, 13)` were selected. The new failure happened during initial `prepare_refit_info`: `RuntimeError: Official PARD-2 draft is missing target_proj during export.` I patched `remote_patch_pard2_official/nemo_rl/models/megatron/draft/utils.py` so export finds `target_proj` across the same wrapper chain used by training. Local `py_compile` and remote dry-run passed. Resubmitted OCI retry `3279229` with run id `20260612_qwen8_pard2_official_online_smoke_targetprojfix`; initial state: pending on OCI priority.

Lyris Qwen3-235B-A22B SWE-Bench Verified OSL64K baseline `2104334` completed `0:0` on `lyris0162`: 2.09 tok/s/GPU, 8.36 tok/s total, 7839.24 s latency. Qwen8 OSL96K PARD-2 `2103214` and Qwen235B OSL64K PARD K3 diagnostic `2104995` were still running at the same poll.

## Lyris Long-OSL SWE-Bench

Files:

- Manifest: `docs/lyris_swebench_longosl_manifest_20260612.csv`
- Status: `docs/lyris_swebench_longosl_status_20260612.csv`
- Metrics: `docs/lyris_swebench_longosl_metrics_20260612.csv`
- Live progress: `docs/lyris_swebench_longosl_live_progress_20260612.csv`

Current tracked jobs: 62 total in the refreshed long-OSL manifest, including the original long-OSL matrix, Qwen8 OSL128K retries/diagnostics, cancelled FlashAttention follow-ups, and the OSL96K gate/follow-ups.

Refreshed accounting snapshot:

- RUNNING: 3
- COMPLETED: 42
- FAILED: 9
- CANCELLED: 4
- TIMEOUT: 4
- PENDING: 0

The Qwen3-8B OSL128K path is still the main instability to watch:

- `2102305`: Verified baseline failed with CUDA illegal memory access in Qwen3 rotary embedding after generation had been running.
- `2102585`: Verified baseline retry also failed after 19:57 with the same rotary-embedding CUDA illegal-memory class.
- `2102306`: Verified suffix K32 failed with CUDA illegal memory access in vLLM rejection sampling after generation had been running.
- `2102307`: Verified PARD-2 K5 hit the 5-hour walltime at `2026-06-12T03:15:43` with 0 completed rows, so there is no final PARD-2 speedup at Qwen8 Verified OSL128K.
- `2102541`: Verified suffix K8 retry failed with the same rejection-sampler CUDA illegal-memory path after generation had reached high acceptance.
- `2102341`: Full suffix K32 OSL128K failed early with the same class of runtime failure.
- `2102338`: Full baseline failed after 19:26 with the same baseline rotary-embedding failure class.
- `2102675`: Verified suffix K4 retry failed after 8:58 in the same rejection-sampler CUDA illegal-memory path. Lowering K from 32/8 to 4 did not resolve the OSL128K Qwen8 vLLM failure.
- `2102797`: Verified baseline-only diagnostic failed after 31:15. This was not speculative decoding: `speculative_config=None`, `ATTENTION_BACKEND=TRITON_ATTN`, and the stack is now in `triton_reshape_and_cache_flash.py` with `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`.

The follow-up baseline-only FlashAttention diagnostic `2102987` also failed, after 28:39 on `lyris0266`. It kept `speculative_config=None`, used `ATTENTION_BACKEND=FLASH_ATTN`, and exported `CUDA_LAUNCH_BLOCKING=1` plus `TORCH_SHOW_CPP_STACKTRACES=1`. The stack moved from the earlier TRITON reshape/cache path into `vllm_flash_attn/cute/interface.py` / CUTLASS DSL with `RuntimeError: CUDA Error: cudaErrorIllegalAddress`. This means Qwen8 OSL128K is not fixed by changing attention backend and should not get more 128K specdec retries until the baseline context-limit issue is isolated.

Cancelled dependency-held Qwen8 OSL128K FlashAttention replacement jobs:

- `2103047`: Verified baseline, no debug env, `afterok:2102987`, cancelled without running.
- `2103048`: Verified suffix K32, no debug env, `afterok:2102987`, cancelled without running.
- `2103050`: Full baseline, no debug env, `afterok:2102987`, cancelled without running.
- `2103051`: Full suffix K32, no debug env, `afterok:2102987`, cancelled without running.

Max-context bracketing result:

- `2103166`: Qwen3-8B SWE-Bench Verified OSL96K baseline-only, TRITON_ATTN, BS1, tracker `latest_lyris_swebench_verified_osl96k_qwen8_baseline_maxctxdiag_20260612_jobs.txt`; completed successfully after 1:29:25. Final throughput is 37.13 tok/s/GPU. This means Qwen8 is stable at OSL96K in the same baseline path that repeatedly failed at OSL128K.

Dependent OSL96K speculative follow-ups submitted after code/config review:

- `2103212`: Qwen3-8B SWE-Bench Verified OSL96K suffix K32, BS1, `afterok:2103166`.
- `2103213`: Qwen3-8B SWE-Bench Verified OSL96K Eagle-3 K3, BS1, `afterok:2103166`.
- `2103214`: Qwen3-8B SWE-Bench Verified OSL96K PARD-2 K5, BS1, `afterok:2103166`.
- Tracker: `latest_lyris_swebench_verified_osl96k_qwen8_specdec_after2103166_20260612_jobs.txt`.

The OSL96K non-speculative baseline succeeded. Current follow-up state:

| Job | Method | Node | Live gen tok/s | Live acceptance | Completed rows |
| ---: | --- | --- | ---: | ---: | ---: |
| `2103212` | suffix K32 | completed | 16.40 final | 21.31% final | 1 |
| `2103213` | Eagle-3 K3 | `lyris0062` | 7.0 | 0.0% | 0 |
| `2103214` | PARD-2 K5 | `lyris0155` | 8.1 | 0.0% | 0 |

Suffix K32 is now a final row: 16.40 tok/s/GPU, 0.442x vs the 37.13 tok/s/GPU baseline, 21.31% acceptance, and mean acceptance length 1.75. Eagle-3 K3 and PARD-2 K5 remain live partial rows only; no final `breakdown.json` exists yet for those two follow-ups.

Completed metric rows are now 64. New/important rows from this refresh:

- Full OSL128K Qwen30 PARD K5 `2102334` reached the 5-hour limit and accounting now marks it `TIMEOUT` at `2026-06-12T03:58:16`. It has no completed benchmark row, so no apples-to-apples speedup should be claimed for this arm.
- Full OSL128K Qwen8 PARD-2 K5 `2102349` reached the 5-hour limit and accounting now marks it `TIMEOUT` at `2026-06-12T03:58:17`. It has no completed benchmark row, so no apples-to-apples speedup should be claimed for this arm.
- Verified OSL128K Qwen8 PARD-2 K5 `2102307` reached the 5-hour limit and accounting now marks it `TIMEOUT` at `2026-06-12T03:15:43`. It has no completed benchmark row, so no apples-to-apples speedup should be claimed for this arm.
- Verified OSL16K Qwen8 suffix K32 BS2 is now included in the final table: 650.01 tok/s/GPU, 8.680x, 92.01% acceptance.
- Verified OSL128K Qwen30 PARD K5 `2102304` reached the 5-hour limit and accounting now marks it `TIMEOUT` at `2026-06-12T02:58:10`. It has no completed benchmark row, so no apples-to-apples speedup should be claimed for this arm. The live progress row showed only partial generation with 0 completed rows.
- Full OSL64K Qwen30 PARD K5 `2102327` finished: 13.66 tok/s/GPU, 0.693x vs baseline, 10.99% acceptance. This confirms PARD K5 is below baseline at Qwen30 Full OSL64K.
- Full OSL32K Qwen8 PARD-2 K5 `2102323` finished BS2: 33.52 tok/s/GPU, 0.447x vs baseline, 3.78% acceptance. The BS1 row remains 17.90 tok/s/GPU, 0.480x, 4.17% acceptance.
- Full OSL128K Qwen30 baseline `2102332` finished: 20.06 tok/s/GPU. This makes Full OSL128K Qwen30 suffix K32 `2102333` a measured 3.973x speedup at 79.70 tok/s/GPU with 97.16% acceptance.
- Full OSL128K Qwen8 Eagle-3 K3 `2102361` finished: 21.68 tok/s/GPU, 40.88% acceptance. The matching Qwen8 Full OSL128K baseline is failed/unstable, so this remains a raw throughput row rather than an apples-to-apples speedup.
- Full OSL64K Qwen8 PARD-2 K5 `2102330` is now terminally completed after 4:18:23. Its final row remains 12.92 tok/s/GPU, 0.354x vs baseline, 1.78% acceptance.
- Verified OSL96K Qwen8 baseline BS1 finished: 37.13 tok/s/GPU.
- Verified OSL96K Qwen8 suffix K32 BS1 finished: 16.40 tok/s/GPU, 0.442x, 21.31% acceptance, and mean acceptance length 1.75. This is a regression despite the longer context.
- Verified OSL16K Qwen30 baseline BS1 finished: 19.87 tok/s/GPU.
- Verified OSL16K Qwen30 suffix K32 BS1: 153.10 tok/s/GPU, 7.706x, 88.18% acceptance.
- Verified OSL16K Qwen30 PARD K5 BS1: 53.78 tok/s/GPU, 2.707x, 57.68% acceptance.
- Verified OSL16K Qwen30 baseline BS2 finished: 39.94 tok/s/GPU; suffix K32 is 7.803x and PARD K5 is 2.657x at BS2.
- Verified OSL64K Qwen8 baseline BS1 finished: 36.89 tok/s/GPU.
- Verified OSL64K Qwen8 suffix K32 BS1: 165.08 tok/s/GPU, 4.475x, 94.90% acceptance.
- Verified OSL64K Qwen8 PARD-2 K5 BS1 finished: 13.37 tok/s/GPU, 0.362x, 2.61% acceptance.
- Verified OSL64K Qwen30 baseline BS1 finished: 19.92 tok/s/GPU.
- Verified OSL64K Qwen30 suffix K32 BS1: 72.48 tok/s/GPU, 3.639x, 71.63% acceptance.
- Verified OSL64K Qwen30 PARD K5 BS1 finished: 15.20 tok/s/GPU, 0.763x, 14.17% acceptance.
- Full OSL16K Qwen8 baseline is now terminally completed; its BS1/BS2 rows remain 36.81 and 73.60 tok/s/GPU.
- Verified OSL128K Qwen30 baseline BS1 finished: 19.97 tok/s/GPU.
- Verified OSL128K Qwen30 suffix K32 BS1: 86.30 tok/s/GPU, 4.321x, 98.16% acceptance.
- Verified OSL128K Qwen8 Eagle-3 K3 wrote a row: 22.26 tok/s/GPU, 39.57% acceptance, but the matching Qwen8 128K baseline is failed/unstable so this is not an apples-to-apples speedup.
- Full OSL16K Qwen8 PARD-2 K5 BS1: 29.64 tok/s/GPU, 0.805x, 8.44% acceptance.
- Full OSL16K Qwen8 PARD-2 K5 BS2: 52.43 tok/s/GPU, 0.712x, 7.49% acceptance.
- Full OSL32K Qwen8 baseline BS1 finished: 37.29 tok/s/GPU.
- Full OSL32K Qwen8 suffix K32 BS1: 271.84 tok/s/GPU, 7.291x, 94.33% acceptance.
- Full OSL32K Qwen8 Eagle-3 K3 BS1: 65.18 tok/s/GPU, 1.748x, 55.57% acceptance.
- Full OSL32K Qwen8 PARD-2 K5 BS1: 17.90 tok/s/GPU, 0.480x, 4.17% acceptance.
- Full OSL32K Qwen8 baseline BS2 finished: 75.03 tok/s/GPU.
- Full OSL32K Qwen8 suffix K32 BS2: 475.81 tok/s/GPU, 6.342x, 92.73% acceptance.
- Full OSL32K Qwen8 Eagle-3 K3 BS2: 117.25 tok/s/GPU, 1.563x, 57.48% acceptance.
- Full OSL32K Qwen8 PARD-2 K5 BS2: 33.52 tok/s/GPU, 0.447x, 3.78% acceptance.
- Full OSL32K Qwen30 baseline BS1 finished: 19.81 tok/s/GPU.
- Full OSL32K Qwen30 suffix K32 BS1: 162.25 tok/s/GPU, 8.192x, 92.46% acceptance.
- Full OSL32K Qwen30 PARD K5 BS1 finished: 34.12 tok/s/GPU, 1.723x, 40.98% acceptance.
- Full OSL64K Qwen30 baseline BS1 finished: 19.70 tok/s/GPU.
- Full OSL64K Qwen30 suffix K32 BS1: 36.70 tok/s/GPU, 1.863x, 52.25% acceptance.
- Full OSL64K Qwen30 PARD K5 BS1: 13.66 tok/s/GPU, 0.693x, 10.99% acceptance.
- Full OSL128K Qwen30 suffix K32 BS1: 79.70 tok/s/GPU, 3.973x, 97.16% acceptance.
- Full OSL64K Qwen8 baseline BS1 finished: 36.48 tok/s/GPU.
- Full OSL64K Qwen8 suffix K32 BS1: 164.65 tok/s/GPU, 4.513x, 93.69% acceptance.
- Full OSL64K Qwen8 Eagle-3 K3 BS1: 20.59 tok/s/GPU, 0.564x, 17.28% acceptance.
- Verified OSL64K Qwen8 Eagle-3 K3 finished: 22.35 tok/s/GPU, 0.606x, 21.10% acceptance.

Key final rows:

| Dataset | OSL | Model | Method | BS | tok/s/GPU | Speedup | Acceptance |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| verified | 16K | Qwen3-8B | baseline | 1 | 37.11 | 1.000x |  |
| verified | 16K | Qwen3-8B | suffix K32 | 1 | 288.86 | 7.783x | 89.16% |
| verified | 16K | Qwen3-8B | Eagle-3 K3 | 1 | 98.81 | 2.662x | 72.98% |
| verified | 16K | Qwen3-8B | PARD-2 K5 | 1 | 33.08 | 0.891x | 11.79% |
| verified | 16K | Qwen3-8B | baseline | 2 | 74.88 | 1.000x |  |
| verified | 16K | Qwen3-8B | suffix K32 | 2 | 650.01 | 8.680x | 92.01% |
| verified | 16K | Qwen3-8B | Eagle-3 K3 | 2 | 177.97 | 2.377x | 71.97% |
| verified | 16K | Qwen3-30B-A3B | baseline | 1 | 19.87 | 1.000x |  |
| verified | 16K | Qwen3-30B-A3B | suffix K32 | 1 | 153.10 | 7.706x | 88.18% |
| verified | 16K | Qwen3-30B-A3B | PARD K5 | 1 | 53.78 | 2.707x | 57.68% |
| verified | 16K | Qwen3-30B-A3B | baseline | 2 | 39.94 | 1.000x |  |
| verified | 16K | Qwen3-30B-A3B | suffix K32 | 2 | 311.62 | 7.803x | 85.29% |
| verified | 16K | Qwen3-30B-A3B | PARD K5 | 2 | 106.13 | 2.657x | 64.08% |
| verified | 64K | Qwen3-8B | baseline | 1 | 36.89 | 1.000x |  |
| verified | 64K | Qwen3-8B | suffix K32 | 1 | 165.08 | 4.475x | 94.90% |
| verified | 64K | Qwen3-8B | PARD-2 K5 | 1 | 13.37 | 0.362x | 2.61% |
| verified | 96K | Qwen3-8B | baseline | 1 | 37.13 | 1.000x |  |
| verified | 64K | Qwen3-30B-A3B | baseline | 1 | 19.92 | 1.000x |  |
| verified | 64K | Qwen3-30B-A3B | suffix K32 | 1 | 72.48 | 3.639x | 71.63% |
| verified | 64K | Qwen3-30B-A3B | PARD K5 | 1 | 15.20 | 0.763x | 14.17% |
| verified | 128K | Qwen3-30B-A3B | baseline | 1 | 19.97 | 1.000x |  |
| verified | 128K | Qwen3-30B-A3B | suffix K32 | 1 | 86.30 | 4.321x | 98.16% |
| full | 16K | Qwen3-30B-A3B | baseline | 1 | 19.65 | 1.000x |  |
| full | 16K | Qwen3-30B-A3B | suffix K32 | 1 | 133.68 | 6.804x | 82.87% |
| full | 16K | Qwen3-30B-A3B | PARD K5 | 1 | 55.93 | 2.847x | 59.53% |
| full | 32K | Qwen3-30B-A3B | baseline | 1 | 19.81 | 1.000x |  |
| full | 32K | Qwen3-30B-A3B | suffix K32 | 1 | 162.25 | 8.192x | 92.46% |
| full | 32K | Qwen3-30B-A3B | PARD K5 | 1 | 34.12 | 1.723x | 40.98% |
| full | 32K | Qwen3-8B | baseline | 1 | 37.29 | 1.000x |  |
| full | 32K | Qwen3-8B | suffix K32 | 1 | 271.84 | 7.291x | 94.33% |
| full | 32K | Qwen3-8B | Eagle-3 K3 | 1 | 65.18 | 1.748x | 55.57% |
| full | 32K | Qwen3-8B | PARD-2 K5 | 1 | 17.90 | 0.480x | 4.17% |
| full | 32K | Qwen3-8B | Eagle-3 K3 | 2 | 117.25 | 1.563x | 57.48% |
| full | 32K | Qwen3-8B | PARD-2 K5 | 2 | 33.52 | 0.447x | 3.78% |
| full | 64K | Qwen3-30B-A3B | baseline | 1 | 19.70 | 1.000x |  |
| full | 64K | Qwen3-30B-A3B | suffix K32 | 1 | 36.70 | 1.863x | 52.25% |
| full | 64K | Qwen3-30B-A3B | PARD K5 | 1 | 13.66 | 0.693x | 10.99% |
| full | 64K | Qwen3-8B | baseline | 1 | 36.48 | 1.000x |  |
| full | 64K | Qwen3-8B | suffix K32 | 1 | 164.65 | 4.513x | 93.69% |
| full | 64K | Qwen3-8B | Eagle-3 K3 | 1 | 20.59 | 0.564x | 17.28% |
| full | 128K | Qwen3-30B-A3B | baseline | 1 | 20.06 | 1.000x |  |
| full | 128K | Qwen3-30B-A3B | suffix K32 | 1 | 79.70 | 3.973x | 97.16% |

Interpretation: suffix is the clear winner where matching baselines have completed, including Qwen8 Verified OSL16K at 7.783x BS1 / 8.680x BS2, Qwen8 Verified OSL64K at 4.475x, Qwen8 Full OSL32K at 7.291x BS1 / 6.342x BS2, Qwen8 Full OSL64K at 4.513x, Qwen30 Verified OSL64K at 3.639x, Qwen30 Verified OSL128K at 4.321x, Qwen30 Full OSL32K at 8.192x, Qwen30 Full OSL64K at 1.863x, and Qwen30 Full OSL128K at 3.973x. PARD is positive on Qwen30 Verified/Full OSL16K and Full OSL32K, but falls below baseline on Qwen30 Verified OSL64K and Full OSL64K; both Qwen30 Verified OSL128K PARD K5 and Full OSL128K PARD K5 timed out before final benchmark rows. PARD-2 K5 remains poor on Qwen8 SWE-Bench OSL16K, OSL32K, and OSL64K; the Full OSL32K BS2 row is 0.447x with 3.78% acceptance, the Full OSL64K row is 0.354x with 1.78% acceptance, and the Verified OSL64K row is 0.362x with 2.61% acceptance. Qwen8 Verified OSL128K and Full OSL128K PARD-2 K5 also timed out with no final rows. Eagle-3 degrades at Qwen8 OSL64K. Qwen8 OSL128K is not a K-selection issue: multiple baseline-only and suffix retries fail in vLLM CUDA kernels across TRITON and FlashAttention. Qwen8 OSL96K baseline is stable at 37.13 tok/s/GPU, so the active path is now waiting for the released OSL96K suffix/Eagle-3/PARD-2 jobs rather than launching more 128K retries.

## Lyris SWE-Bench Verified OSL32K High-K Sweep

Files:

- Status: `docs/lyris_specdec_32k_status_20260612.csv`
- Metrics: `docs/lyris_specdec_32k_metrics_20260612.csv`
- Summary: `docs/lyris_specdec_32k_status_20260612.md`

Current tracked jobs: 32 total.

- COMPLETED: 29
- TIMEOUT: 3

The three timeouts are Qwen3-8B PARD-2 K5/K9/K11 after 5h, but each wrote BS1/BS2 rows before timeout. They are all well below baseline, so extending those jobs just to collect BS4 is not a useful next submission.

Key final rows:

| Model | Method | BS | tok/s/GPU | Speedup | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B | baseline | 1 | 19.58 | 1.000x |  |
| Qwen3-30B-A3B | suffix K32 | 1 | 177.25 | 9.055x | 93.92% |
| Qwen3-30B-A3B | PARD K11 | 1 | 35.38 | 1.807x | 19.36% |
| Qwen3-30B-A3B | PARD K9 | 1 | 34.98 | 1.787x | 23.55% |
| Qwen3-30B-A3B | PARD K5 | 1 | 32.85 | 1.678x | 38.51% |
| Qwen3-8B | baseline | 1 | 37.76 | 1.000x |  |
| Qwen3-8B | suffix K5 | 1 | 149.99 | 3.972x | 96.52% |
| Qwen3-8B | Eagle-3 K5 | 1 | 75.31 | 1.995x | 44.06% |
| Qwen3-8B | Eagle-3 K9 | 1 | 73.52 | 1.947x | 26.40% |
| Qwen3-8B | Eagle-3 K11 | 1 | 71.67 | 1.898x | 21.80% |
| Qwen3-8B | PARD-2 K5 | 1 | 18.70 | 0.495x | 5.36% |
| Qwen3-8B | PARD-2 K9 | 1 | 18.01 | 0.477x | 2.20% |
| Qwen3-8B | PARD-2 K11 | 1 | 18.13 | 0.480x | 1.81% |
| Qwen3-14B | suffix K5 | 1 | 141.98 | 4.187x | 98.73% |
| Qwen3-14B | PARD-2 K11 | 1 | 16.74 | 0.494x | 1.58% |
| Qwen3-30B-A3B-Thinking-2507 | Eagle-3 K3 | 1 | 21.17 | 1.052x | 10.11% |

Interpretation: for 30B PARD, K11 has the best throughput on this 32K sweep but loses acceptance sharply; suffix K32 is still much stronger. For Qwen8 Eagle-3, K5 is the best high-K point across BS1/2/4, while K9/K11 reduce acceptance and throughput. PARD-2 does not improve with higher K on Qwen8/Qwen14; it remains below baseline and should not be expanded until the drafter/domain issue is fixed.

## Qwen3-235B SWE-Bench

Files:

- Status: `docs/lyris_qwen235b_suffix_status_20260612.csv`
- Metrics: `docs/lyris_qwen235b_suffix_metrics_20260612.csv`
- Summary: `docs/lyris_qwen235b_suffix_status_20260612.md`

Current tracked jobs: 37 total.

- COMPLETED: 29
- RUNNING: 6
- FAILED: 2
- PENDING: 0

The failed native PARD-2 probe jobs have successful retry coverage. Native PARD-2 OSL32K K1/K2/K3 all have final rows.

New Qwen3-235B SWE-Bench Verified OSL64K pilot jobs were submitted on Lyris after local syntax validation, remote prompt/container/HF-cache checks, and remote speculative-config review against the successful OSL32K setup:

| Job | Method | Config |
| ---: | --- | --- |
| `2104334` | baseline | TP4, fp8 KV, 1 prompt, OSL64K |
| `2104335` | suffix K32 | `method=suffix`, K32 |
| `2104336` | suffix K8 | `method=suffix`, K8 |
| `2104337` | PARD K5 | `amd/PARD-Qwen3-0.6B`, draft TP4 |
| `2104338` | Eagle-3 K3 | `nvidia/Qwen3-235B-A22B-Eagle3`, draft TP4 |
| `2104340` | native PARD-2 K1 | `method=pard2`, `amd/PARD2-Qwen3-8B`, draft TP4 |

All six are now running on Lyris `gb200`:

| Job | Method | Node | Live gen tok/s | Live acceptance | Completed rows |
| ---: | --- | --- | ---: | ---: | ---: |
| `2104334` | baseline | `lyris0162` | 8.2 |  | 0 |
| `2104335` | suffix K32 | `lyris0179` | 106.7 | 100.0% | 0 |
| `2104336` | suffix K8 | `lyris0006` | 60.4 | 100.0% | 0 |
| `2104337` | PARD K5 | `lyris0007` | 5.3 | 0.0% | 0 |
| `2104338` | Eagle-3 K3 | `lyris0009` | 4.4 | 0.8% | 0 |
| `2104340` | native PARD-2 K1 | `lyris0013` | 5.1 | 6.1% | 0 |

These are live partial signals only; no Qwen3-235B OSL64K final metric rows exist yet. The live OSL64K tails are volatile. The latest tail now has suffix K32 and K8 far above the live baseline, while PARD, Eagle-3, and native PARD-2 remain below it. Wait for final `breakdown.json` rows before treating this as the measured ranking.

Key final OSL32K rows:

| Method | BS | tok/s/GPU | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1 | 2.09 | 1.000x |  |
| suffix K32 | 1 | 12.60 | 6.042x | 82.63% |
| suffix K8 | 1 | 11.65 | 5.586x | 86.35% |
| suffix K4 | 1 | 7.01 | 3.360x | 75.43% |
| Eagle-3 K3 | 1 | 5.19 | 2.488x | 54.28% |
| PARD K5 | 1 | 3.19 | 1.529x | 18.39% |
| draft-model PARD-2 K3 | 1 | 1.87 | 0.895x | 4.64% |
| draft-model PARD-2 K5 | 1 | 1.89 | 0.906x | 2.86% |
| native PARD-2 K1 | 1 | 1.88 | 0.899x | 11.53% |
| native PARD-2 K2 | 1 | 1.90 | 0.909x | 6.65% |
| native PARD-2 K3 | 1 | 1.91 | 0.917x | 4.64% |

Interpretation: Qwen3-235B OSL32K is very favorable for suffix decoding. PARD and Eagle-3 are positive but behind suffix. Native PARD-2 K1 raises acceptance relative to K2/K3, but all native PARD-2 K1/K2/K3 rows remain below baseline at OSL32K because acceptance is too low to amortize draft overhead.

## OCI NeMo-RL Online Drafter

Files:

- `docs/qwen30ba3b_pard2_online_long_output_win2048_comparison_20260611.csv`
- `docs/qwen30ba3b_fullgrpo20_long_output_all_comparison_20260612.csv`
- `docs/qwen30ba3b_suffix_fullgrpo20_summary_20260611.csv`

Qwen30B PARD2-style online/refit vs static, step>=2:

| Variant | Steps | Acceptance | Step speedup vs static |
| --- | ---: | ---: | ---: |
| online_start1_pard2_win2048 | 19 | 46.54% | 1.001x |
| online_start10_pard2_win2048 | 19 | 47.78% | 0.989x |
| online_start5_int5_pard2_win2048 | 19 | 48.08% | 0.989x |

Against the no-spec long-output baseline, the original `3274971` timed out after 4h on OCI `batch` after producing 14 matched completed baseline rows. The clean `batch_long` retry `3277195` has now finished with 19 comparable `step>=2` rows, so it is the current baseline for the long-output comparison.

| Variant | Variant steps | Step speedup vs no-spec | E2E tok/s/GPU speedup |
| --- | ---: | ---: | ---: |
| no-spec batch-long retry | 19 | 1.011x | 1.003x |
| static PARD-2 | 19 | 1.818x | 1.964x |
| online start1 PARD-2 | 19 | 1.821x | 1.967x |
| online start10 PARD-2 | 19 | 1.797x | 1.950x |
| online start5 interval5 PARD-2 | 19 | 1.797x | 1.948x |
| suffix K32 long-output | 19 | 1.741x | 1.866x |

Qwen30B suffix full-GRPO20 is complete: 20/20 steps, 25.61% token acceptance, mean step time 70.715s, latest error none. SWERL suffix generation-only `3266737` is still pending on priority; the latest direct `squeue` check reports start `N/A`, so no ray-driver metrics exist yet.

Interpretation: these rows remain useful as PARD2-style/static-vs-refit comparisons, but the 13:49 CEST code audit shows they should not be treated as official PARD-2 online-drafter training evidence. The reviewed code supports generic PARD and a legacy/generic PARD2-style CAT fallback, while official PARD-2 online training is explicitly guarded because target feature capture, `target_proj` conditioning, and PARD-2 projection refit/export are not implemented.

Qwen8 PARD2-style initial-refit evidence:

| Run | Steps | Refits | Acceptance | Step time | Gen time | Gen worker tok/s/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static-equivalent 50-step | 49 | 0 | 47.54% | 32.68s | 17.20s | 140.70 |
| online interval-5 50-step | 49 | 9 | 47.16% | 32.93s | 16.91s | 142.77 |

Interpretation: for Qwen8, the initial-refit path fixes the static-equivalent/dummy-draft failure mode, but the reviewed code does not implement official PARD-2 online target-feature training. Treat these as generic PARD2-style/refit behavior until the official PARD-2 path is extended and validated.

## OCI Standalone SWE-Bench Smoke

Files:

- Original tracker: `latest_oci_swebench_longosl_submit_smoke_20260612_jobs.csv`
- Suffix retry tracker: `latest_oci_swebench_verified_osl16k_qwen8_suffix_retry_20260612_jobs.csv`
- Extracted metrics: `docs/oci_swebench_osl16k_smoke_metrics_20260612.csv`

Status:

- `3275051` Qwen30 baseline: RUNNING, no `breakdown.json` yet.
- `3275053` Qwen8 baseline: COMPLETED.
- `3275056` original Qwen8 suffix: FAILED due Python 3.13 arctic site used with Python 3.12 container.
- `3275303` Qwen8 suffix retry: COMPLETED after installing `arctic-inference==0.1.1` inside the job.
- `3275057` Qwen8 PARD-2: COMPLETED; BS1/BS2 rows have been written.
- `3275065` Qwen8 Eagle-3: COMPLETED.

Current completed/partial OCI rows:

| Method | BS | tok/s/GPU | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1 | 31.66 | 1.000x |  |
| suffix K32 | 1 | 260.25 | 8.220x | 89.16% |
| Eagle-3 K3 | 1 | 73.80 | 2.331x | 72.98% |
| PARD-2 K5 | 1 | 32.23 | 1.018x | 11.79% |
| baseline | 2 | 63.51 | 1.000x |  |
| suffix K32 | 2 | 624.41 | 9.831x | 92.01% |
| Eagle-3 K3 | 2 | 134.34 | 2.115x | 71.97% |
| PARD-2 K5 | 2 | 49.72 | 0.783x | 9.64% |

The OCI suffix retry validates the wrapper fix: the prior `_C` import failure is gone and suffix now completes. PARD-2 is not competitive in this SWE-Bench OSL16K setup because acceptance is too low, especially at BS2.

## Math / OpenMath

Files:

- High-batch PARD sweep: `docs/qwen3_235b_public_pard_openmath_k_sweep_20260608.csv`
- K-selection notes: `docs/qwen3_pard_k_selection_20260608.md`
- Local checkpoint gates: `docs/qwen3_235b_pard_math_local_checkpoint_gates.csv`

No active SLURM rows were found for the recorded Math/OpenMath job set in `squeue`. The checked OpenMath high-batch jobs are completed.

Key high-batch OpenMath rows, Qwen3-235B, ISL1024/OSL1024:

| Method | BS | tok/s/GPU | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| baseline | 64 | 807.19 | 1.000x |  |
| public PARD K3 | 64 | 1055.48 | 1.308x | 57.82% |
| public PARD K5 | 64 | 1016.29 | 1.259x | 45.01% |
| public PARD K7 | 64 | 965.73 | 1.196x | 35.18% |
| public PARD K9 | 64 | 957.51 | 1.186x | 28.90% |
| baseline | 128 | 1372.76 | 1.000x |  |
| public PARD K3 | 128 | 1708.11 | 1.244x | 58.30% |
| public PARD K5 | 128 | 1605.71 | 1.170x | 44.77% |
| public PARD K7 | 128 | 1528.92 | 1.114x | 34.91% |
| public PARD K9 | 128 | 1426.78 | 1.039x | 28.78% |

Interpretation: on real OpenMath, larger K is not automatically better. K3 gives the best high-batch throughput/acceptance in the public PARD sweep, while K5 remains the conservative choice when comparing against the NeMo-RL fixed-output gates and older K-selection notes. K7/K9 lose acceptance quickly.

## Submission Decision

Submitted targeted Lyris jobs after code/config review:

- `2102675`: Qwen3-8B SWE-Bench Verified OSL128K suffix K4, BS1, tracker `latest_lyris_swebench_verified_osl128k_qwen8_suffix_k4_retry_20260612_jobs.txt`.
- `2102797`: Qwen3-8B SWE-Bench Verified OSL128K baseline-only CUDA diagnostic, BS1, tracker `latest_lyris_swebench_verified_osl128k_qwen8_baseline_cuda_diag_20260612_jobs.txt`.
- `2102987`: Qwen3-8B SWE-Bench Verified OSL128K baseline-only FLASH_ATTN diagnostic, BS1, tracker `latest_lyris_swebench_verified_osl128k_qwen8_baseline_flashattn_diag_20260612_jobs.txt`.
- `2103047`: Qwen3-8B SWE-Bench Verified OSL128K FLASH_ATTN no-debug baseline, BS1, `afterok:2102987`, tracker `latest_lyris_swebench_verified_osl128k_qwen8_flashattn_nodebug_after2102987_20260612_jobs.txt`.
- `2103048`: Qwen3-8B SWE-Bench Verified OSL128K FLASH_ATTN no-debug suffix K32, BS1, `afterok:2102987`, same tracker as `2103047`.
- `2103050`: Qwen3-8B SWE-Bench Full OSL128K FLASH_ATTN no-debug baseline, BS1, `afterok:2102987`, tracker `latest_lyris_swebench_full_osl128k_qwen8_flashattn_nodebug_after2102987_20260612_jobs.txt`.
- `2103051`: Qwen3-8B SWE-Bench Full OSL128K FLASH_ATTN no-debug suffix K32, BS1, `afterok:2102987`, same tracker as `2103050`.
- `2103166`: Qwen3-8B SWE-Bench Verified OSL96K TRITON_ATTN baseline-only max-context diagnostic, BS1, tracker `latest_lyris_swebench_verified_osl96k_qwen8_baseline_maxctxdiag_20260612_jobs.txt`.
- `2103212`: Qwen3-8B SWE-Bench Verified OSL96K suffix K32, BS1, `afterok:2103166`, tracker `latest_lyris_swebench_verified_osl96k_qwen8_specdec_after2103166_20260612_jobs.txt`.
- `2103213`: Qwen3-8B SWE-Bench Verified OSL96K Eagle-3 K3, BS1, `afterok:2103166`, same tracker as `2103212`.
- `2103214`: Qwen3-8B SWE-Bench Verified OSL96K PARD-2 K5, BS1, `afterok:2103166`, same tracker as `2103212`.
- `2104334`: Qwen3-235B SWE-Bench Verified OSL64K baseline, BS1, tracker `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_suffix_k32_pilot_20260612_jobs.txt`.
- `2104335`: Qwen3-235B SWE-Bench Verified OSL64K suffix K32, BS1, same tracker as `2104334`.
- `2104336`: Qwen3-235B SWE-Bench Verified OSL64K suffix K8, BS1, tracker `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_suffix_k8_pilot_20260612_jobs.txt`.
- `2104337`: Qwen3-235B SWE-Bench Verified OSL64K PARD K5, BS1, tracker `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_pard_k5_pilot_20260612_jobs.txt`.
- `2104338`: Qwen3-235B SWE-Bench Verified OSL64K Eagle-3 K3, BS1, tracker `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_eagle3_k3_pilot_20260612_jobs.txt`.
- `2104340`: Qwen3-235B SWE-Bench Verified OSL64K native PARD-2 K1, BS1, tracker `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_pard2_native_k1_pilot_20260612_jobs.txt`.

Reasoning:

- The broad Lyris long-OSL matrix is already active under `coreai_dlalgo_llm`.
- The Qwen3-235B SWE-Bench OSL32K method/K sweep is completed.
- The Qwen3-235B SWE-Bench Verified OSL64K pilot was submitted after code/config review.
- The reviewed Lyris launcher writes the expected suffix config (`method=suffix`, `num_speculative_tokens=K`), uses `coreai_dlalgo_llm`, and installs `arctic-inference==0.1.1` in the job.
- The launcher now has opt-in `CUDA_LAUNCH_BLOCKING` and `TORCH_SHOW_CPP_STACKTRACES` passthrough for diagnostic jobs.
- Qwen8 OSL128K suffix K32 and K8 both failed in vLLM rejection sampling after generation started; K4 was the next narrow retry before launching a broader set.
- The no-debug FlashAttention jobs were intentionally dependency-held. Because `2102987` failed, SLURM cancelled `2103047`, `2103048`, `2103050`, and `2103051` without running them.
- `2103166` is a baseline-only stability bracket, not a benchmark claim. It tests whether Qwen8 becomes stable below 128K before we spend runs on suffix/PARD/PARD-2/Eagle at that context length.
- The OCI suffix failure has a completed successful retry.
- The Math/OpenMath jobs checked here are completed; there is no active failed Math job needing a retry.
- The diagnostic launcher path passed `bash -n`; the Python extraction/plot helpers passed `python3 -m py_compile`. The diagnostic uses `method=baseline`, so there is no suffix/PARD/PARD-2/Eagle-3 speculative config to validate for this run.

Result:

- `2102675` also failed in the vLLM rejection-sampler CUDA illegal-memory path.
- `2102585` baseline retry also failed in the Qwen3 rotary-embedding CUDA illegal-memory path.
- I did not submit another Qwen8 OSL128K suffix K because K4/K8/K32 all fail and the matching baseline is also unstable.
- 08:33 continuation decision: submitted `2102797` as a stability/debug run, not as a benchmark point. It was running on `lyris0126`; the log confirmed `speculative_config=None`.
- 08:39 continuation decision: no additional benchmark job submitted. The broad matrix remains active, and the debug run has not yet failed or produced a stack.
- 08:44 continuation decision: no additional benchmark job submitted. At that point `2102797` was running normally and had not reached the earlier 20-minute failure window yet.
- 08:56 continuation decision: still no additional benchmark job submitted. `2102797` has now passed the prior baseline failure window without a CUDA error, so the next useful action is to let it reach completion or fail with a more informative stack.
- 09:03 code-review decision: `bash -n` passed for the Lyris submit wrappers and refresh scripts, and the Python extraction/plot helpers passed `py_compile`. I did not resubmit the broad Lyris long-OSL wrapper because it would duplicate the already-active matrix; `2102797` remains the correct gate before any new Qwen8 OSL128K retry.
- 09:09 continuation decision: `2102797` failed after 31:15 in `triton_reshape_and_cache_flash`, still baseline-only. I patched and revalidated the launcher env passthrough, then submitted `2102987` as a non-duplicate FLASH_ATTN baseline diagnostic. The generated remote `run.sbatch` confirms `--attention-backend 'FLASH_ATTN'`, `CUDA_LAUNCH_BLOCKING=1`, and `TORCH_SHOW_CPP_STACKTRACES=1`. It is running on `lyris0266`.
- 09:14 continuation decision: `2102987` is still running at 3:46 elapsed with `AttentionBackendEnum.FLASH_ATTN`, ~26.9 generation tok/s, and no CUDA error. No additional Qwen8 OSL128K job submitted; this diagnostic needs to pass or fail the old failure window first.
- 09:21 code-review/submission decision: local `bash -n` passed for `submit_lyris_swebench32k_standalone_specdec.sh`, `submit_lyris_swebench_longosl_standalone_specdec_20260612.sh`, and the refresh scripts; `py_compile` passed for `standalone_vllm_specdec_breakdown.py`, `extract_vllm_standalone_breakdown_metrics.py`, and `plot_lyris_specdec_expected_performance.py`; local and remote benchmark-driver/sitecustomize checksums match. I submitted the four no-debug FLASH_ATTN baseline/suffix replacements as `afterok:2102987` jobs rather than duplicating the whole long-OSL matrix.
- 09:24 continuation state: `2102987` is running at 14:26 elapsed with ~26.2 tok/s generation and no CUDA error. `2103047`, `2103048`, `2103050`, and `2103051` are all PENDING with `afterok:2102987(unfulfilled)`.
- 09:30 continuation state: `2102987` is running at 19:43 elapsed with ~26.1 tok/s generation and no CUDA error. It has crossed the first old baseline failure point but still needs to reach or complete past the 31:15 `2102797` failure point.
- 09:40 continuation state: `2102987` failed after 28:39 with `RuntimeError: CUDA Error: cudaErrorIllegalAddress` in `vllm_flash_attn/cute/interface.py` via CUTLASS DSL. The dependent no-debug FlashAttention jobs were cancelled without running.
- 09:43 code-review/submission decision: syntax/import checks still passed. I submitted `2103166`, a single baseline-only Qwen8 Verified OSL96K TRITON_ATTN max-context diagnostic, to bracket the stable boundary between working 64K and failing 128K.
- 09:44 continuation state: `2103166` is running on `lyris0062` at 0:53 elapsed.
- 09:53 code-review/submission decision: local shell syntax and Python compile checks passed; remote checksums for `standalone_vllm_specdec_breakdown.py` and `specdec_breakdown_instrumentation/sitecustomize.py` matched local reviewed files; remote Python compile passed. I submitted OSL96K suffix/Eagle-3/PARD-2 follow-ups as `afterok:2103166` so they only run if the non-spec baseline succeeds.
- 10:00 refresh decision: added the newer 128K diagnostic/dependency trackers and the OSL96K baseline/specdec trackers to the long-OSL refresh defaults. The refreshed manifest tracks 62 jobs.
- 10:22 code-review/status decision: local `bash -n` and `py_compile` checks still pass; the local and Lyris benchmark-driver/sitecustomize checksums match; prompt JSONL files and the Lyris container are present. All 49 expected long-OSL combinations are already represented in the manifest, so I did not submit duplicates. The refreshed metric CSV has 50 completed benchmark rows and the expected-performance raw CSV has 157 rows.
- 10:22 continuation state: `2103166` is still running at 36:12 elapsed, with 37.2 tok/s generation, 8.3% GPU KV cache, and no error lines. `2103212`, `2103213`, and `2103214` remain PENDING on `afterok:2103166`.
- 10:27 refresh decision: Lyris long-OSL now has 30 completed / 16 running / 3 pending jobs and 51 metric rows. Full OSL64K Qwen30 baseline `2102325` completed, making suffix K32 `2102326` a measured 1.863x speedup. OCI no-spec long-output baseline `3274971` has 13 matched completed steps and remains running.
- 10:33 no-submit decision: bounded watch of `2103166` reached 49:47 elapsed with stable ~37 tok/s and no error lines. The OSL96K suffix/Eagle-3/PARD-2 jobs remain correctly held on `afterok:2103166`; no duplicate or manual dependency override was submitted.
- 10:45 code-review/submission decision: local and remote syntax checks passed for the OCI Qwen30B no-spec long-output wrapper/helper. The remote helper differs only by an extra `NRL_DEBUG_DRAFT_ROUNDTRIP` env passthrough, not sbatch behavior. Dry-run confirmed `--partition=batch_long --time=08:00:00`, account `coreai_dlalgo_llm`, no SpecDec, and 20 GRPO steps. Submitted retry `3277195`; status is `PENDING (Priority)`.
- 10:45 Lyris gate status: `2103166` remains `RUNNING` at 1:02:10 on `lyris0062`; `2103212`, `2103213`, and `2103214` remain `PENDING (Dependency)`.
- 10:50 refresh decision: regenerated Lyris long-OSL artifacts, OCI PARD-2/SWERL artifacts, and expected-performance HTML/PNG/CSV. Lyris now has 34 completed / 12 running / 9 failed / 4 cancelled / 3 pending tracked jobs, still 55 final metric rows. `2103166` remains `RUNNING` at 1:03:52; `3277195` remains `PENDING (Priority)` with start estimate `2026-06-12T03:21:29`. No new duplicate jobs were submitted.
- 10:53 live poll: `2103166` remains `RUNNING` at 1:08:25 on `lyris0062`; `2103212`, `2103213`, and `2103214` remain `PENDING (Dependency)`. OCI `3277195` remains `PENDING (Priority)`. No new recent `breakdown.json` files were written, so no additional metrics refresh or submission was warranted.
- 10:54 live poll: active Lyris long-OSL jobs remain running, including `2103166` at 1:10:38. The active logs have live throughput and no current error lines; no new active-job `breakdown.json` appeared. OCI `3277195` remains `PENDING (Priority)` with updated start estimate `2026-06-12T02:25:20`. No refresh or duplicate submission was warranted.
- 10:55 live poll: Lyris active set is unchanged; `2103166` is still `RUNNING` at 1:12:18 and OSL96K follow-ups remain dependency-held. Active jobs still show live throughput and the direct tail scan found no current error lines. OCI `3277195` remains `PENDING (Priority)` with start estimate `2026-06-12T02:25:20`. No table refresh or new submission was warranted.
- 11:00 refresh decision: `2102300` completed and wrote the Qwen8 Verified OSL64K PARD-2 K5 `breakdown.json`. Refreshed Lyris long-OSL artifacts and regenerated expected-performance HTML/PNG/CSV. Lyris now has 35 completed / 11 running / 9 failed / 4 cancelled / 3 pending tracked jobs and 56 final metric rows. The new PARD-2 row is 13.37 tok/s/GPU, 0.362x, 2.61% acceptance. `2103166` remains `RUNNING` at 1:14:51; `3277195` remains pending on OCI.
- 11:10 review/status decision: re-reviewed the Lyris wrapper/base helper and refresh scripts; `bash -n` and Python compile checks passed, and both Lyris/OCI SSH ControlMasters are active. The refreshed Lyris manifest has 36 completed / 10 running / 9 failed / 4 cancelled / 3 pending tracked jobs and still 56 final metric rows. `2103166` remains `RUNNING` at 1:24:15; `2103212`, `2103213`, and `2103214` remain correctly dependency-held on `afterok:2103166`, so no duplicate job was submitted. OCI `3277195` is now `RUNNING` on `nvl72042-T[05-08]`, but no clean no-spec step metrics are available yet.
- 11:18 refresh decision: `2103166` completed successfully and wrote the Qwen8 Verified OSL96K baseline row: 37.13 tok/s/GPU. Lyris now has 37 completed / 9 running / 9 failed / 4 cancelled / 3 pending tracked jobs and 58 final metric rows. The OSL96K suffix `2103212`, Eagle-3 `2103213`, and PARD-2 `2103214` jobs are released from dependency and priority-pending with estimated starts later today. Regenerated expected-performance HTML/PNG/CSV; raw table now has 160 rows. No duplicate jobs submitted. OCI `3277195` remains running but still has no clean no-spec step metrics.
- 11:25 code-review/no-submit decision: local shell syntax and Python compile checks passed again for the reviewed launch/refresh path. Lyris checkpoint prewarm `2100960` completed successfully, and the target OSL96K suffix/Eagle-3/PARD-2 jobs `2103212`/`2103213`/`2103214` are already submitted and priority-pending with estimated start `2026-06-12T05:05:00`. OCI no-spec `3277195` is running in Step 1/20 generation with no SpecDec, and SWERL `3266737` remains pending with start `N/A`. No duplicate jobs were submitted.
- 11:30 refresh decision: Lyris long-OSL artifacts now show 38 completed / 8 running / 9 failed / 4 cancelled / 3 pending tracked jobs and 59 final metric rows. `2102361` completed and added the Full OSL128K Qwen8 Eagle-3 K3 row: 21.68 tok/s/GPU, 40.88% acceptance, no valid speedup because the matching Qwen8 128K baseline is failed. Regenerated expected-performance HTML/PNG/CSV; raw table remains 160 rows. OCI `3277195` completed Step 1 cleanly but still has 0 step>=2 comparable rows, and SWERL `3266737` remains pending.
- 11:35 refresh decision: `2102332` completed and added the Full OSL128K Qwen30 baseline row: 20.06 tok/s/GPU. The existing Full OSL128K Qwen30 suffix K32 row is now a measured 3.973x speedup at 79.70 tok/s/GPU and 97.16% acceptance. Lyris long-OSL now has 39 completed / 7 running / 9 failed / 4 cancelled / 3 pending tracked jobs and 60 final metric rows. OSL96K follow-ups remain priority-pending, and OCI `3277195` is still in Step 2 generation.
- 11:46 code-review/submission decision: added a dedicated Qwen3-235B SWE-Bench Verified OSL64K Lyris wrapper and registered its trackers in the 235B refresh script. Local `bash -n` passed, remote prompt/container/HF-cache checks passed, and generated speculative configs matched the reviewed 32K setup: suffix K32/K8, PARD K5 with `amd/PARD-Qwen3-0.6B` draft TP4, Eagle-3 K3 with `nvidia/Qwen3-235B-A22B-Eagle3` draft TP4, and native PARD-2 K1 with `method=pard2` / `amd/PARD2-Qwen3-8B` draft TP4. Submitted `2104334`-`2104338` and `2104340`; all are pending on Lyris priority. The 235B manifest now has 37 jobs: 29 completed / 2 failed / 6 pending.
- 11:53 refresh decision: `2102327` completed and added Qwen30 Full OSL64K PARD K5 at 13.66 tok/s/GPU, 0.693x, 10.99% acceptance; `2102323` completed and added Qwen8 Full OSL32K PARD-2 K5 BS2 at 33.52 tok/s/GPU, 0.447x, 3.78% acceptance. Lyris long-OSL now has 41 completed / 5 running / 9 failed / 4 cancelled / 3 pending tracked jobs and 62 final metric rows. Expected-performance HTML/PNG/CSV were regenerated.
- 12:01 refresh decision: `2102304` Qwen30 Verified OSL128K PARD K5 hit the 5h walltime and accounting marks it `TIMEOUT`; it has only a live partial row with 0 completed rows, so it is not used as a final speedup. Lyris long-OSL now has 41 completed / 4 running / 9 failed / 4 cancelled / 1 timeout / 3 pending tracked jobs. Qwen3-235B OSL64K and Qwen8 OSL96K follow-ups remain priority-pending; OCI `3277195` remains running and SWERL suffix `3266737` remains pending. No duplicate jobs were submitted.
- 12:03 live poll: Qwen8 Verified OSL96K suffix `2103212` entered startup/prolog on `lyris0018`; no slurm output or `breakdown.json` exists yet. Eagle-3 `2103213` and PARD-2 `2103214` remain priority-pending. Qwen3-235B OSL64K jobs remain priority-pending. OCI `3277195` is still running, and SWERL suffix `3266737` remains pending.
- 12:09 refresh decision: OSL96K Qwen8 suffix/Eagle-3/PARD-2 jobs `2103212`/`2103213`/`2103214` are all running and have live partial rows but 0 completed rows. Qwen3-235B OSL64K jobs `2104334`-`2104338` and `2104340` are all running and have early live rows but no final OSL64K metrics. Lyris long-OSL now has 41 completed / 7 running / 9 failed / 4 cancelled / 1 timeout / 0 pending tracked jobs; Qwen3-235B status now has 29 completed / 6 running / 2 failed / 0 pending. No duplicate jobs were submitted.
- 12:15 refresh decision: `2102330` Qwen8 Full OSL64K PARD-2 K5 moved from running to completed, leaving Lyris long-OSL at 42 completed / 6 running / 9 failed / 4 cancelled / 1 timeout / 0 pending tracked jobs. The final metric count remains 62 because its `breakdown.json` row was already present; latest OSL96K and Qwen3-235B OSL64K rows remain live partials with 0 completed rows. OCI `3277195` is running, and SWERL suffix `3266737` remains pending with start `N/A`. No duplicate jobs were submitted.
- 12:21 refresh decision: `2102307` Qwen8 Verified OSL128K PARD-2 K5 hit the 5h walltime and accounting marks it `TIMEOUT`; it has 0 completed rows, so no final speedup is claimed. Lyris long-OSL now has 42 completed / 5 running / 9 failed / 4 cancelled / 2 timeout / 0 pending tracked jobs and 63 final metric rows. Expected-performance HTML/PNG/CSV were regenerated; the raw table now has 170 rows. Qwen3-235B OSL64K remains running with live partials only. OCI `3277195` now has 3 comparable no-spec `step>=2` rows but is still running.
- 12:36 refresh/poll decision: regenerated the expected-performance HTML/PNG/CSV after the latest Lyris refresh; the raw table remains 170 rows. Remaining Lyris jobs are still running. Current live OSL96K rows are suffix K32 11.6 gen tok/s / 8.9% acceptance, Eagle-3 K3 11.0 / 0.0%, and PARD-2 K5 9.4 / 0.2%, all with 0 completed rows. Qwen3-235B OSL64K live rows are volatile; the latest refresh has Eagle-3 K3 at 16.6 gen tok/s / 53.1% acceptance, PARD K5 at 9.3 / 7.8%, suffix K32 at 7.6 / 10.0%, suffix K8 at 6.6 / 5.2%, and native PARD-2 K1 at 7.3 / 5.7%, all with 0 completed rows. OCI `3277195` remains running at 1:31:15 elapsed, and SWERL suffix `3266737` remains pending with estimated start `2026-06-13T07:20:00`.
- 13:05 refresh/no-submit decision: `2102334` Qwen30 Full OSL128K PARD K5 and `2102349` Qwen8 Full OSL128K PARD-2 K5 both hit the 5h walltime with 0 completed rows. Lyris long-OSL now has 42 completed / 3 running / 9 failed / 4 cancelled / 4 timeout / 0 pending tracked jobs and still 63 final metric rows. I did not resubmit these arms: Qwen30 Full 128K PARD had 0.0% live acceptance and far-below-baseline live throughput before timeout, and Qwen8 Full 128K still lacks a stable matching baseline. Regenerated expected-performance HTML/PNG/CSV; the raw table remains 170 rows. Qwen3-235B OSL64K remains running with live partials only, and all current speculative live tails are below the live baseline. OCI `3277195` now has 6 comparable no-spec `step>=2` rows and remains running; SWERL suffix `3266737` remains pending with estimated start `2026-06-13T07:50:00`.
- 13:16 refresh/no-submit decision: regenerated Lyris long-OSL, Qwen3-235B, and OCI NeMo-RL status artifacts. Lyris long-OSL remains 42 completed / 3 running / 9 failed / 4 cancelled / 4 timeout / 0 pending tracked jobs with 63 final metric rows. Qwen3-235B remains 29 completed / 6 running / 2 failed with OSL64K live partials only, and all current OSL64K speculative live tails are below the live baseline. OCI `3277195` now has 7 comparable no-spec `step>=2` rows and remains running; SWERL suffix `3266737` remains pending with start `N/A`. No duplicate jobs were submitted.
- 13:25 refresh/no-submit decision: re-polled Lyris and OCI. Lyris long-OSL still has 42 completed / 3 running / 9 failed / 4 cancelled / 4 timeout / 0 pending tracked jobs and 63 final rows; Qwen3-235B still has 29 completed / 6 running / 2 failed and 42 final rows. The active Lyris jobs still have 0 completed rows, and OCI `3277195` still has 7 comparable no-spec rows. I did not submit replacements because the useful remaining experiments are already running and the completed evidence does not point to a new narrowly justified retry.
- 13:34 code-review/no-submit decision: added SSH keepalive and a 12s timeout around the final Qwen3-235B remote `breakdown.json` checks in `scripts/refresh_lyris_qwen235b_suffix_results.sh`, then refreshed Lyris long-OSL, Qwen3-235B, OCI, and regenerated the expected-performance HTML/PNG/CSV. Long-OSL remains 42 completed / 3 running / 9 failed / 4 cancelled / 4 timeout with 63 final rows; Qwen3-235B remains 29 completed / 6 running / 2 failed with 42 final rows. Qwen235B OSL64K suffix K8/K32 now look promising in live tails, but all six OSL64K jobs still have 0 completed rows. OCI `3277195` now has 8 comparable no-spec rows. No new jobs were submitted.
- 13:41 refresh/no-submit decision: re-ran the Lyris and OCI collectors and regenerated expected-performance HTML/PNG/CSV. Row counts are unchanged: long-OSL has 63 final rows, Qwen3-235B has 42 final rows, and the expected-performance raw table has 170 rows. Qwen235B OSL64K suffix K32/K8 live tails rose to 106.7 / 60.4 gen tok/s with 100% live acceptance, but all six OSL64K jobs still have 0 completed rows. Qwen8 OSL96K suffix/Eagle/PARD-2 also still have 0 completed rows. No new jobs were submitted.
- 13:49 code-audit/no-submit decision: reviewed the actual OCI NeMo-RL worktree fetched into `.tmp_remote_current_oci`. The code supports generic PARD online training and a legacy/generic PARD2-style fallback, but official PARD-2 online training raises `NotImplementedError` unless the fallback is explicitly enabled, because target hidden-state capture, `target_proj` conditioning, and PARD-2 projection refit/export are missing. SuffixDecoding is static vLLM-only in this path and has no trainable drafter/refit target. I did not submit new official PARD-2 or SuffixDecoding online-drafter jobs.

Next gates:

- Wait for Lyris official PARD-2 online job `2107214` to finish environment build and enter model setup/training; it is the remaining live Lyris NeMo-RL online run.
- Treat Qwen8 Verified OSL96K PARD-2 K5 `2103214` as terminal without a benchmark row: it timed out at 5h with a 0.0% acceptance tail.
- Treat Qwen235B OSL64K PARD/PARD-2 as blocked by CUDA illegal-address / vLLM `EngineDeadError` until a narrower runtime fix is identified. Completed Qwen235B OSL64K rows currently favor suffix K8 over suffix K32 and Eagle-3.
- Use the completed `3277195` batch-long no-spec retry as the current Qwen30 long-output baseline. It has 19 comparable rows; `3274971` is retained only as the older partial 4h baseline.

- 14:40 code-review/submission result: the official PARD-2 vLLM target-feature patch was fixed and smoked on Lyris. Earlier smoke attempts found real issues before long-job submission: `target_proj.weight` was created too early and failed normal draft checkpoint loading, then the generic parallel-drafting path required EAGLE-style `mask_hidden`. The fixed patch defers `target_proj` creation until after base drafter load and repeats the last target feature for PARD-2 mask slots. Smoke job `2104955` completed successfully on `lyris0096` and wrote `breakdown.json` with active `method=pard2`, `K=3`, Qwen3-8B target, `amd/PARD2-Qwen3-8B` draft, `output_tok_s_per_gpu=21.72`, `drafted=381`, `accepted=0`, `acceptance_rate=0.00%`. I did not submit a long official-PARD-2 benchmark from this path yet because the smoke is runtime-valid but shows no acceptance on the SWE-Bench Verified prompt.

- 16:52 CEST code-review/submission result: reviewed the official PARD-2 online launch path again before submitting. Local `bash -n` passed for `experiments/eagle3_online/prepare_pard2_official_vllm_site.sh`, `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh`, and `experiments/eagle3_online/submit_qwen8_pard2_official_online_smoke_20260612.sh`; local `py_compile` passed for the PARD/PARD-2 online overlay and official vLLM patch helpers. OCI smoke `3279229` completed `0:0`: it built the official patched vLLM site, loaded Qwen3-8B plus `amd/PARD2-Qwen3-8B`, selected PARD-2 target layers `(36, 29, 21, 13)`, exported 312 PARD draft weights including the `target_proj` path, completed both GRPO steps, and the driver log had no traceback/runtime/value errors. Acceptance remained low in the tiny smoke (`0/63` on one worker and `1/62` on the repeated worker line), so this is a code/runtime gate, not a performance claim. After the clean smoke and dry-run preflight, submitted OCI 20-step Qwen3-8B official PARD-2 online run `3279589` from isolated stage `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-online20-20260612`; initial state is `PENDING (Priority)`.

- 16:58 CEST refresh/status result: refreshed Lyris long-OSL and Qwen235B artifacts and regenerated `docs/lyris_specdec_expected_performance_20260612.{png,md,html}` plus `docs/lyris_specdec_expected_performance_raw_20260612.csv` (171 raw rows). New final Qwen8 Verified OSL96K Eagle-3 row is `13.3867 tok/s/GPU`, `0.3605x` vs the `37.1319 tok/s/GPU` baseline, `11.18%` acceptance; suffix K32 remains `16.4032 tok/s/GPU`, `0.4418x`, `21.31%` acceptance. PARD-2 K5 `2103214` is still running near walltime with live tail around `6.5 gen tok/s` and `0.0%` acceptance, so no final PARD-2 OSL96K speedup is claimed yet. Qwen235B SWE-Bench Verified OSL64K final rows now show suffix K8 as the best completed arm: baseline `2.0900 tok/s/GPU`, suffix K8 `4.3994` (`2.1050x`, `46.38%` acceptance), suffix K32 `3.8240` (`1.8297x`, `38.93%`), and Eagle-3 K3 `2.2884` (`1.0949x`, `21.84%`). Qwen235B PARD K5 `2104337` and native PARD-2 K1 `2104340` both failed with CUDA illegal-address / vLLM `EngineDeadError` during warmup/generation rather than a Python config error; PARD K3 diagnostic `2104995` remains running, so I did not submit a duplicate PARD retry. OCI online job `3279589` has started on `nvl72108-T17`; it passed `PARD2_OFFICIAL_PATCH_CHECKS`, built the official patched vLLM site, loaded with `SpeculativeConfig(method='pard2', model='amd/PARD2-Qwen3-8B', num_spec_tokens=1)`, selected target layers `(36, 29, 21, 13)`, and is initializing `lm_policy` workers.

- 17:16 CEST Lyris online submission result: reviewed the Lyris official PARD-2 online wrapper before submission. Local `bash -n`, local `py_compile`, `scripts/test_vllm_draft_refit_target_proj.py`, and remote dry-run/preflight passed. The Lyris wrapper stages an isolated checkout at `/lustre/fsw/coreai_dlalgo_llm/users/sna/SpecDec-RL-pard2-official-online-lyris-20260612`, uses the reviewed OCI online overlay plus `remote_patch_pard2_official/`, and points vLLM at the already-smoked official PARD-2 vLLM 0.20.2 source site rather than trying to patch the Lyris `RL-nightly-test` vLLM 0.17 tree. The first Lyris attempt `2107184` failed before GRPO because `uv` searched for Python `3.13.13`; this was a launch-environment issue, not a PARD-2 code failure. I fixed the wrapper to use `/opt/nemo_rl_venv/bin/python3`, reran the dry-run, and submitted corrected Lyris job `2107214` under `coreai_dlalgo_llm`. Initial poll: `RUNNING` on `lyris0108`, past the previous interpreter failure, building the `uv` environment. No model/performance result is available yet.

- 17:19 CEST poll result: OCI online job `3279589` completed successfully (`COMPLETED`, exit `0:0`, elapsed `00:18:00`) and ran all `20/20` GRPO steps with `Draft Training Enabled: True`. It repeatedly exported/refit 312 draft weights and the vLLM load result included `target_proj.weight`, so the official PARD-2 online plumbing is now a 20-step correctness pass. Acceptance is still poor: final-tail SpecDec metrics were around `2.2%` to `3.2%` average draft acceptance, with many earlier steps in the `0.0%` to `5.8%` range. Lyris corrected online job `2107214` remains `RUNNING` on `lyris0108`, elapsed `00:03:12` at the poll, and is still building the `uv` environment after passing `PARD2_OFFICIAL_PATCH_CHECKS`. Qwen8 Verified OSL96K PARD-2 K5 `2103214` hit the 5h walltime with no `breakdown.json`; the tail was stable around `6.4` generation tok/s and `0.0%` acceptance. Qwen235B Verified OSL64K PARD K3 diagnostic `2104995` failed after `02:26:29` with CUDA illegal memory access in Qwen3-MoE rotary embedding, followed by vLLM `EngineDeadError`; no final row was written.

- 17:25 CEST refresh result: reran `scripts/refresh_lyris_swebench_longosl_results.sh`, `scripts/refresh_lyris_qwen235b_suffix_results.sh`, `scripts/refresh_pard2_swerl_active_status.sh`, and regenerated `docs/lyris_specdec_expected_performance_20260612.{png,md,html}` plus raw CSV. Long-OSL status is now 44 completed / 9 failed / 4 cancelled / 5 timeout with 65 final metric rows. Qwen235B status is terminal at 33 completed / 5 failed with 46 final metric rows. The Qwen30 batch-long no-spec retry `3277195` is complete enough for comparison: 19 rows, mean step time `880.775s`, and it is now the no-spec denominator. Against it, static PARD-2 is `1.818x` step-speedup / `1.964x` generation tok/s/GPU speedup, online start1 PARD-2 is `1.821x` / `1.967x`, online start10 is `1.797x` / `1.950x`, online start5 interval5 is `1.797x` / `1.948x`, and suffix K32 long-output is `1.741x` / `1.866x`. Lyris online official PARD-2 `2107214` remains `RUNNING` on `lyris0108`, elapsed `00:09:17`; it has passed source checks but has not reached model setup yet.

- 18:40 CEST continuation result: refreshed Lyris MATH500, SWE-Bench long-OSL, and Qwen235B artifacts, then regenerated `docs/lyris_specdec_expected_performance_20260612.{png,md,html}` plus raw CSV. Current row counts are: MATH500 OSL32K `5` final rows, SWE-Bench long-OSL `65` final rows, Qwen235B `46` final rows, combined expected-performance raw table `176` rows. MATH500 still has no baseline final rows, so MATH speedups remain blank; the current MATH final rows are Qwen8 suffix K32 BS1/BS2, Qwen8 Eagle-3 K3 BS1, and Qwen30 suffix K32 BS1/BS2. Long-OSL SWE-Bench remains terminal at 44 completed / 9 failed / 4 cancelled / 5 timeout, and Qwen235B remains terminal at 33 completed / 5 failed. The key performance interpretation is unchanged: SuffixDecoding is the strongest completed long-output method for Qwen8/Qwen30 and for Qwen235B OSL32K/64K; PARD/PARD-2 performance is acceptable only at shorter/easier shapes, while long-context Qwen8 PARD-2 and Qwen235B PARD-2 suffer from very low acceptance or runtime failures. Lyris online PARD-2 `2107520` failed with a staged `load_dataloader_state` import mismatch after dependency build; I patched the staged GRPO overlay with a compatibility fallback, reran local/remote preflight, and submitted replacement job `2107786` under `coreai_dlalgo_llm`. Initial state: `RUNNING` on `lyris0044`.

- 18:43 CEST MATH/online refresh: Qwen8 MATH500 OSL32K Eagle-3 `2107297` completed and added the BS2 final row: `118.53` tok/s/GPU, `63.62%` acceptance, mean accept length `2.91`. MATH500 now has `6` final rows and `3 completed / 6 running` jobs; Qwen8/Qwen30 baselines are still running, so MATH speedup cells remain blank. Regenerated the expected-performance bundle again; the raw table now has `177` rows and includes all six MATH500 rows. Active MATH tails show Qwen8 baseline BS1 chunk 3/4 at about `36.77` tok/s/GPU, Qwen30 baseline BS1 chunk 1/4 at `20.19` tok/s/GPU, Qwen30 PARD K5 BS1 chunk 3/4 at `49.38` tok/s/GPU with `63.15%` acceptance, and Qwen30 PARD K3 BS1 chunk 3/4 at `39.82` tok/s/GPU with `77.70%` acceptance. No MATH tail scan found `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors. Lyris online PARD-2 replacement `2107786` remains `RUNNING` on `lyris0044`, passed source checks, installed the first dependency batch, and is building native dependencies without repeating the previous import errors.

- 18:45 CEST no-new-row poll: refreshed MATH500 again and row counts are unchanged: `6` final MATH rows, `3 completed / 6 running` MATH jobs, and `177` combined expected-performance raw rows. Qwen8 and Qwen30 MATH baselines are still running, so MATH speedup cells remain blank. Lyris online PARD-2 replacement `2107786` remains `RUNNING` on `lyris0044`, elapsed `00:05:31`, still in native dependency setup after source checks. No `Traceback`, `ImportError`, `ModuleNotFoundError`, or repeated staged-source helper failure appears in the scanned online-driver tail.

- 18:48 CEST MATH/online refresh: Qwen30 MATH500 OSL32K PARD K5 `2107334` completed its BS1 aggregate row: `42.33` tok/s/GPU, `52.15%` acceptance, mean accept length `3.61`. The matching Qwen30 baseline is still running, so no final speedup is claimed yet. MATH500 now has `7` final rows; after regenerating the expected-performance bundle, the raw table has `178` rows and includes the new Qwen30 PARD K5 MATH row. Active MATH tails also show Qwen8 PARD-2 official K3/K5 BS1 chunk 1/4 at only about `16.0` tok/s/GPU with `0.05%`/`0.03%` acceptance, which is consistent with the earlier low-acceptance PARD-2 finding. Online PARD-2 replacement `2107786` remains `RUNNING` on `lyris0044`, before `SETUP COMPLETE`, with no new actionable error in the scanned tail.

- 18:52 CEST review/status result: re-ran the local submission gates before relying on the submitted Lyris replacement. `py_compile` passed for the staged GRPO overlay, PARD-2 utility overlay, plotting helper, metrics extractor, and standalone vLLM runner; `bash -n` passed for the Lyris online launcher, shared online launcher, PARD-2 vLLM-site helper, refresh scripts, and standalone vLLM launcher; `git diff --check` passed on the touched launch/docs/report paths. Refreshed MATH500 OSL32K and regenerated the expected-performance bundle. MATH500 now has `8` final rows and the combined raw table has `179` rows. The new Qwen8 no-spec BS1 baseline is `36.79` tok/s/GPU, giving Qwen8 suffix K32 BS1 a measured `5.871x` speedup at `216.02` tok/s/GPU and `84.51%` acceptance, and Qwen8 Eagle-3 K3 BS1 a measured `2.021x` speedup at `74.38` tok/s/GPU and `62.94%` acceptance. Online PARD-2 replacement `2107786` is still `RUNNING` on `lyris0044`, elapsed about `00:11:54`, and the scanned driver tail still shows no repeated staged-source import failure or actionable traceback.

- 18:58 CEST MATH/report refresh: a targeted Lyris tail scan found that Qwen30 MATH500 OSL32K PARD K3 `2107335` had written its BS1 aggregate row after the prior refresh. Re-ran the MATH collector and regenerated `docs/lyris_specdec_expected_performance_20260612.{png,md,html}` plus raw CSV. MATH500 now has `9` final rows and the combined raw table has `180` rows. New row: Qwen30 PARD K3 BS1 `37.20` tok/s/GPU, `70.67%` acceptance, mean accept length `3.12`. On the same Qwen30 MATH slice, PARD K5 remains faster at `42.33` tok/s/GPU but lower acceptance (`52.15%`, mean accept length `3.61`); the matching Qwen30 baseline is still running, so no final Qwen30 MATH speedup is claimed yet. Long-OSL SWE-Bench and Qwen235B refreshes remain at `65` and `46` final rows respectively, so the larger interpretation is unchanged: suffix remains the strongest completed long-output method, while PARD/PARD-2 is weaker or unstable at the longest contexts. Online PARD-2 replacement `2107786` remains `RUNNING` on `lyris0044`, elapsed about `00:17:05`, still before `SETUP COMPLETE`, with no repeated staged-source import/config failure in the scanned tail.

- 19:02 CEST online/MATH poll: refreshed MATH500 again; final row count is unchanged at `9`, and the combined expected-performance raw CSV remains at `180` rows. Active MATH tails have no new aggregate beyond Qwen8 baseline BS1 and Qwen30 PARD K3/K5 BS1; Qwen30 baseline still has only live BS1 chunk 1/4 at about `20.19` tok/s/GPU, and Qwen8 official PARD-2 K3/K5 still only show BS1 chunk 1/4 around `16.0` tok/s/GPU with near-zero acceptance (`0.05%`/`0.03%`). Online PARD-2 replacement `2107786` remains `RUNNING` on `lyris0044`, elapsed about `00:21:54`. Raw driver tail shows the job is still in driver environment native-build setup: `flash-attn`, `deep-ep`, `transformer-engine-torch`, `nv-grouped-gemm`, and `mamba-ssm` built successfully; `causal-conv1d` has not yet printed a final built line. There is still no `SETUP COMPLETE`, model load, GRPO step, import/config traceback, runtime error, or OOM signal.

- 19:06 CEST online retry result: Lyris online PARD-2 job `2107786` failed after the driver env finished building native deps. Root cause was another staged-source compatibility mismatch: the overlaid newer `grpo.py` imports `nemo_rl.data_plane.interfaces.DataPlaneConfig`, while the Lyris base checkout has no `nemo_rl/data_plane` package. This symbol is only used as the optional `MasterConfig.data_plane` type, so I patched the staged GRPO overlay to fall back to `dict[str, Any]` when the import is unavailable. Local gates passed (`py_compile`, `bash -n`, `git diff --check`), and the Lyris dry-run/preflight passed with target-projection refit tests and `PARD2_OFFICIAL_PATCH_CHECKS`. Submitted replacement job `2108021` with run id `20260612_qwen8_pard2_official_online_lyris_dataplanefix`; initial poll shows it `RUNNING` on `lyris0217`, the remote staged `grpo.py` contains the `DataPlaneConfig` fallback, and the driver log is in env setup after passing PARD-2 patch checks. MATH500 remains unchanged at `9` final rows and the combined raw report remains `180` rows.

- 19:19 CEST online/MATH poll: replacement `2108021` remains `RUNNING` on `lyris0217`, elapsed about `00:12:51`. It is still in driver environment native-build setup after passing `PARD2_OFFICIAL_PATCH_CHECKS`; current built native packages include `flash-attn`, `deep-ep`, and `transformer-engine-torch`. There is still no repeated `data_plane` import failure, no `SETUP COMPLETE`, and no GRPO/model-load signal yet. MATH500 refresh is unchanged at `9` final rows, and the combined expected-performance raw report remains `180` rows with `9` MATH rows.

- 19:36 CEST online hydrafix submission: replacement `2108021` failed after env build with Hydra struct validation rejecting `policy.megatron_cfg.force_reconvert_from_hf=false`; the Lyris base config does not define that optional key. I reviewed and patched the submit path before resubmission: the shared launcher now passes `++policy.megatron_cfg.force_reconvert_from_hf=false`, and the Lyris wrapper now force-adds the optional nested DDP overlap flags with `++policy.megatron_cfg.distributed_data_parallel_config.*`. Validation passed: local `bash -n`, local `py_compile`, `git diff --check`, Lyris dry-run staging/preflight, vLLM draft-refit target-projection tests, `PARD2_OFFICIAL_PATCH_CHECKS`, and a targeted Hydra parse of the exact dry-run override vector against the staged Lyris config. Submitted replacement job `2108168` with run id `20260612_qwen8_pard2_official_online_lyris_hydrafix`, account `coreai_dlalgo_llm`; follow-up poll shows it `RUNNING` on `lyris0267`, in driver env setup before `SETUP COMPLETE`.

- 19:42 CEST benchmark refresh / online poll: refreshed MATH500, SWE-Bench long-OSL, and Qwen235B artifacts, then regenerated `docs/lyris_specdec_expected_performance_20260612.{png,md,html}` plus raw CSV. The combined raw report now has `181` rows, including `10` MATH500 rows. New MATH evidence: Qwen8 baseline BS2 completed at `73.79` tok/s/GPU, giving Qwen8 suffix K32 BS2 a measured `6.367x` speedup at `469.83` tok/s/GPU and Qwen8 Eagle-3 K3 BS2 a measured `1.606x` speedup at `118.53` tok/s/GPU. Qwen30 MATH baseline is still running, so Qwen30 MATH suffix/PARD speedup cells remain blank. SWE-Bench long-OSL remains `65` final rows and Qwen235B remains `46` final rows. Online PARD-2 hydrafix job `2108168` is `RUNNING` on `lyris0267` at about `00:05:40`, still in driver environment setup after `PARD2_OFFICIAL_PATCH_CHECKS`; no repeated Hydra/config/import/runtime failure appears in the scanned log, and it has not reached `SETUP COMPLETE` yet.

- 19:46 CEST MATH refresh / online poll: refreshed MATH500 again and regenerated the expected-performance bundle. MATH500 now has `12` final rows and the combined raw report has `183` rows. New rows: Qwen30 PARD K3 BS2 `68.87` tok/s/GPU, `71.95%` acceptance, mean accept length `3.16`; Qwen30 PARD K5 BS2 `61.57` tok/s/GPU, `43.45%` acceptance, mean accept length `3.17`. Qwen30 MATH baseline `2107332` is still running, so Qwen30 speedup cells remain blank. Online PARD-2 hydrafix `2108168` remains `RUNNING` on `lyris0267` at about `00:09:39`; it has passed `PARD2_OFFICIAL_PATCH_CHECKS` and is still building native dependencies, with `deep-ep` now built. No repeated Hydra/config/import/runtime failure appears in the scanned log, and it has not reached `SETUP COMPLETE` yet.

- 20:35 CEST online retry / MATH refresh: online PARD-2 `2108168` failed after env setup with a Ray version mismatch: Ray head `2.49.2`, driver `2.54.0`. I patched the Lyris wrapper to start Ray with a job-local `ray[default]==2.54.0` venv instead of the container's existing Ray, then submitted `2108268`. That replacement passed Ray connect and data setup, then failed at `grpo.setup` because the overlaid newer GRPO code expected `MasterConfig` attribute access while the Lyris runner passed a plain dict. I patched `.tmp_remote_current_oci/nemo_rl/algorithms/grpo.py` to convert dict configs with `MasterConfig(**master_config)` at setup entry, validated locally and with Lyris dry-run/preflight, and submitted replacement `2108402` with run id `20260612_qwen8_pard2_official_online_lyris_masterconfigfix`. Initial poll: `RUNNING` on `lyris0049`; staged `grpo.py` contains the conversion, Ray head is creating `ray[default]==2.54.0`, and the driver is in early setup after `PARD2_OFFICIAL_PATCH_CHECKS`. MATH500 advanced to `15` final rows and the combined raw report to `186` rows. New MATH evidence: Qwen30 baseline BS1 `20.11` tok/s/GPU, so Qwen30 BS1 speedups are now suffix K32 `7.296x`, PARD K5 `2.105x`, and PARD K3 `1.850x`; Qwen8 official PARD-2 BS1 final rows are poor, K3 `15.99` tok/s/GPU (`0.435x`, `0.0716%` acceptance) and K5 `16.10` tok/s/GPU (`0.438x`, `0.0430%` acceptance).

- 20:43 CEST refresh / online poll: refreshed MATH500, SWE-Bench long-OSL, and Qwen235B artifacts, then regenerated `docs/lyris_specdec_expected_performance_20260612.{png,md,html}` plus raw CSV. Counts are stable: MATH500 has `15` final metric rows, SWE-Bench long-OSL has `65`, Qwen235B has `46`, and the combined raw table has `186` data rows. The expected-performance table is grouped by model and batch size; Qwen30 MATH BS2 speedups remain blank because the Qwen30 BS2 baseline has not completed. `2108402` remains `RUNNING` on `lyris0049` at about `00:09:51`; it is still in driver environment/native dependency setup after `PARD2_OFFICIAL_PATCH_CHECKS`, with Ray 2.54.0 active and no new traceback visible.

- 20:44 CEST live poll: no new final benchmark rows or failures. Qwen30 MATH baseline `2107332` remains `RUNNING` on `lyris0178` at about `03:06:06`; its vLLM log shows two running requests around `40 tok/s`, so the missing Qwen30 BS2 baseline is still making progress. Online PARD-2 `2108402` remains `RUNNING` on `lyris0049` at about `00:12:33`; the driver has built `deep-ep` and `transformer-engine-torch`, is still before `SETUP COMPLETE`/GRPO setup, and no repeated Ray, `MasterConfig`, Hydra, import, CUDA/OOM, or traceback error appears in the scanned log. Local `py_compile`, `bash -n`, and `git diff --check` passed again.

- 20:52 CEST live monitor: no new terminal state and no new metric row. Online PARD-2 `2108402` remains `RUNNING` on `lyris0049` at about `00:20:08`; a four-poll monitor showed it still in native dependency setup after `PARD2_OFFICIAL_PATCH_CHECKS`, with `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` built but no `SETUP COMPLETE`, model load, GRPO step, or traceback yet. Qwen30 MATH baseline `2107332` remains `RUNNING` on `lyris0178` at about `03:13:41`, so the Qwen30 MATH BS2 speedup cells remain pending. `git diff --check` on the touched files passed.

- 21:00 CEST online retry: `2108402` reached Nemo-RL config/data setup and failed in logger setup because `logger.wandb_enabled=true` required a W&B API key on Lyris. This confirms the Ray 2.54 and `MasterConfig` fixes got past their earlier failure points. I patched the shared online launcher to make `logger.wandb_enabled` controlled by `WANDB_ENABLED`, and patched the Lyris wrapper to pass `WANDB_ENABLED=false`. Local validation passed (`bash -n`, `py_compile`, `git diff --check`), and the Lyris dry-run/preflight passed with vLLM draft-refit target-projection tests, `PARD2_OFFICIAL_PATCH_CHECKS`, and an exact dry-run command containing `logger.wandb_enabled=false`. Submitted replacement job `2108503` with run id `20260612_qwen8_pard2_official_online_lyris_wandbfix`; initial state is `RUNNING` on `lyris0126`, Ray head started with `ray[default]==2.54.0`, and the driver is in venv setup. Qwen30 MATH baseline `2107332` remains running, so no benchmark rows were regenerated in this step.

- 21:28 CEST online retry / MATH refresh: `2108503` reached config/data setup with `logger.wandb_enabled=false`, then failed in `ClippedPGLossFn` because the Lyris base config set `loss_fn.truncated_importance_sampling_type='tis'` while leaving `loss_fn.truncated_importance_sampling_ratio=None`. I patched the Lyris wrapper to explicitly preserve no-truncation behavior with `loss_fn.truncated_importance_sampling_type=null` and `loss_fn.truncated_importance_sampling_ratio=null`, reran local validation plus the Lyris dry-run/preflight, and submitted replacement job `2108658` with run id `20260612_qwen8_pard2_official_online_lyris_tisfix`. It is `RUNNING` on `lyris0213`, has passed `PARD2_OFFICIAL_PATCH_CHECKS`, and is still in dependency/environment setup before `SETUP COMPLETE`. Qwen30 MATH baseline `2107332` completed, so the MATH/expected-performance bundle was regenerated: Qwen30 MATH BS2 baseline is `40.12` tok/s/GPU, suffix K32 is `303.81` tok/s/GPU (`7.572x`, `92.52%` acceptance), PARD K3 is `68.87` tok/s/GPU (`1.716x`, `71.95%` acceptance), and PARD K5 is `61.57` tok/s/GPU (`1.535x`, `43.45%` acceptance).

- 21:39 CEST refresh / online poll: reran the MATH500, SWE-Bench long-OSL, and Qwen235B collectors, then regenerated `docs/lyris_specdec_expected_performance_20260612.{png,md,html}` plus raw CSV. Row counts are stable after regeneration: MATH500 has `16` final metric rows, SWE-Bench long-OSL has `65`, Qwen235B has `46`, and the combined raw table has `187` data rows. `2108658` remains `RUNNING` on `lyris0213`, elapsed about `00:14:42`; it is still in dependency/native setup after source checks, with `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` built. There is still no `SETUP COMPLETE`, model load, GRPO step, W&B error, TIS assertion, traceback, CUDA/OOM, or runtime failure visible in the scanned driver log.

- 21:43 CEST online poll: `2108658` remains `RUNNING` on `lyris0213`, elapsed about `00:18:31`. The Ray head is up with `ray==2.54.0`; the driver command includes `logger.wandb_enabled=false`, `loss_fn.truncated_importance_sampling_type=null`, and `loss_fn.truncated_importance_sampling_ratio=null`. A compute-node process scan shows active `nvcc`/`ptxas` work for `causal-conv1d` and `mamba-ssm`, so the long dependency stage is compile-bound rather than idle. No code patch or resubmission was made because there is no new failure to diagnose.

- 21:53 CEST online retry: `2108658` got through dependency build, Ray connection, actor environment creation, loaded the config, and reached data/cluster setup with the intended `logger.wandb_enabled=false` and TIS-null config. It then failed at `RayVirtualCluster(...)` because the newer GRPO overlay passed `port_range_low` / `port_range_high`, but the Lyris `RayVirtualCluster.__init__` signature does not accept those kwargs. I patched `.tmp_remote_current_oci/nemo_rl/algorithms/grpo.py` with a signature-aware `_create_ray_virtual_cluster` helper that keeps the port-range kwargs on newer APIs and drops them for older APIs. Local `py_compile` passed, the Lyris dry-run/preflight passed remote compile, vLLM draft-refit target-projection tests, and `PARD2_OFFICIAL_PATCH_CHECKS`, then I submitted replacement job `2108818` with run id `20260612_qwen8_pard2_official_online_lyris_portfix`. Initial state: `RUNNING` on `lyris0161`; the staged source contains the helper and all three GRPO cluster call sites use it.

- 22:01 CEST online poll: `2108818` remains `RUNNING` on `lyris0161`, elapsed about `00:07:45` at the poll. The driver log has passed `PARD2_OFFICIAL_PATCH_CHECKS`, is using the existing official PARD-2 vLLM site, installed the driver environment, and is in native dependency setup. No traceback, W&B error, TIS assertion, Ray version mismatch, `RayVirtualCluster` port-range error, CUDA/OOM signal, model-load failure, or GRPO-step failure is visible in the scanned tail yet.

- 22:05 CEST refresh / online poll: refreshed MATH500, SWE-Bench long-OSL, and Qwen235B artifacts, then regenerated `docs/lyris_specdec_expected_performance_20260612.{png,md,html}` plus raw CSV. Row counts remain stable: MATH500 has `16` final metric rows, SWE-Bench long-OSL has `65`, Qwen235B has `46`, and the combined raw table has `187` data rows. `2108818` remains `RUNNING` on `lyris0161`, elapsed about `00:11:59`; it is still in native dependency setup, with `deep-ep` and `transformer-engine-torch` built after `PARD2_OFFICIAL_PATCH_CHECKS`. No repeated W&B, TIS, Ray-version, `RayVirtualCluster`, import/config, CUDA/OOM, runtime, model-load, or GRPO-step failure is visible yet.

- 22:08 CEST online monitor: a three-poll monitor showed `2108818` still `RUNNING` on `lyris0161` through `00:14:32` elapsed. The log remains in native dependency setup with `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` built; a node process scan showed active `nvcc`/`cicc`/`ptxas` work for `mamba-ssm` and `causal-conv1d`. It has not reached `SETUP COMPLETE`, model load, or GRPO steps yet, and there is no new failure signature. No replacement job was submitted.

- 22:13 CEST online monitor: `2108818` remains `RUNNING` on `lyris0161` through `00:19:24` elapsed. The driver venv is `9.0G`, row counts remain unchanged (`16` MATH500, `65` SWE-Bench long-OSL, `46` Qwen235B, `187` combined raw rows), and a process scan still shows active CUDA compiler work for selective-scan / `mamba-ssm` plus `causal-conv1d`. The driver log has not reached `SETUP COMPLETE`, model load, or GRPO steps, and no repeated W&B, TIS, Ray-version, `RayVirtualCluster`, import/config, CUDA/OOM, runtime, model-load, or GRPO-step error is visible. No replacement job was submitted.

- 22:22 CEST online retry: `2108818` got past dependency build, Ray connection, data/cluster setup, actor environment creation, and vLLM worker initialization (`4/4` workers), then failed in the NeMo-RL vLLM init monkey-patch with `RuntimeError: Could not patch vLLM ADDITIONAL_ENV_VARS for SpecDec runtime env propagation.` Root cause: the official PARD-2 vLLM source site does not define the older `ADDITIONAL_ENV_VARS` constant; it uses the newer `vllm/ray/ray_env.py` path with `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` / `get_env_vars_to_copy`. I reviewed and patched `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_worker.py` so the SpecDec runtime env list is also merged into `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`, while retaining the strict failure if neither the old nor new propagation path exists. Local `py_compile` and `git diff --check` passed; the Lyris dry-run/preflight passed remote compile, vLLM draft-refit target-projection tests, and all `PARD2_OFFICIAL_PATCH_CHECKS`. Submitted replacement job `2108947` with run id `20260612_qwen8_pard2_official_online_lyris_envpropfix`, account `coreai_dlalgo_llm`. Initial state: `RUNNING` on `lyris0018`, in driver environment setup after source checks.

- 22:27 CEST online poll: `2108947` remains `RUNNING` on `lyris0018`, elapsed about `00:03:52` at the status poll. It has passed `PARD2_OFFICIAL_PATCH_CHECKS`, the official PARD-2 vLLM source-site check, and the main Python package install. The compute-node process scan shows active `uv`, `cicc`, `ptxas`, and `cc1plus` work for `mamba-ssm`, `causal-conv1d`, `deep-ep`, `nv-grouped-gemm`, and `transformer-engine`; the driver venv is `9.0G`. It has not reached `SETUP COMPLETE`, model load, vLLM init, or GRPO steps yet, and no new W&B, TIS, Ray-version, `RayVirtualCluster`, import/config, CUDA/OOM, runtime, model-load, or env-propagation error is visible.

- 22:32 CEST online poll: `2108947` remains `RUNNING` on `lyris0018`, elapsed about `00:09:33`. The filtered driver log now shows `deep-ep` built after the official PARD-2 source checks. It remains in the native dependency build stage before `SETUP COMPLETE`, model load, vLLM init, or GRPO steps. No new W&B, TIS, Ray-version, `RayVirtualCluster`, import/config, CUDA/OOM, runtime, model-load, traceback, or env-propagation error is visible, so no patch or resubmission was made.

- 22:38 CEST online process check: `2108947` remains `RUNNING` on `lyris0018`. The driver log now shows `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` built. A compute-node process scan shows active `cicc`/`ptxas` work for `mamba-ssm` selective-scan kernels and `causal-conv1d`; the job-local Ray 2.54.0 head/raylet are up. The driver command includes the reviewed overrides (`logger.wandb_enabled=false`, TIS null, `method=pard2`, `num_speculative_tokens=1`, `parallel_drafting=true`). The job has still not reached `SETUP COMPLETE`, model load, vLLM init, or GRPO steps, and no new failure signature is visible.

- 22:49 CEST online retry: `2108947` reached the previous vLLM actor-init boundary: all driver native packages built, Ray connected, config/data/compute-cluster setup succeeded, the actor venv was created, and `vllm_policy` workers initialized `4/4`. The previous `ADDITIONAL_ENV_VARS` env-propagation failure did not recur. It then failed in a different optional compatibility patch: `_patch_vllm_vit_flash_attn_backend()` called `_get_vllm_file("attention/layer.py")`, but the official PARD-2 vLLM source site does not have that old path. I patched `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_worker.py` so this ViT flash-attention workaround logs and skips when `vllm/attention/layer.py` is absent, while still applying the patch on vLLM layouts that have the file. Local `py_compile`, `bash -n`, and `git diff --check` passed; the Lyris dry-run passed remote refit `target_proj` tests and all `PARD2_OFFICIAL_PATCH_CHECKS`, and the staged remote source contains the skip. Submitted replacement job `2109062` with run id `20260612_qwen8_pard2_official_online_lyris_vitpatchfix`; initial state: `RUNNING` on `lyris0149`.

- 22:56 CEST online poll: `2109062` remains `RUNNING` on `lyris0149`, elapsed about `00:06:59`. It has passed `PARD2_OFFICIAL_PATCH_CHECKS` and the official PARD-2 vLLM site check, and is still in driver/native dependency setup. A compute-node process scan shows active CUDA compiler work for `mamba-ssm`, `causal-conv1d`, `nv-grouped-gemm`, and `transformer-engine`; the job-local Ray process is up and the launched command contains the reviewed W&B/TIS/PARD-2 overrides. It has not reached Nemo config load, vLLM actor initialization, or GRPO steps yet, and no new failure signature is visible.
