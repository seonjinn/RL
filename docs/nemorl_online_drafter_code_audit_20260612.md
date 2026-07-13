# NeMo-RL Online Drafter Code Audit - 2026-06-12

## Scope

Pre-submit review for launching more NeMo-RL online drafter jobs for PARD, PARD-2, and SuffixDecoding.

Source reviewed:

- Remote OCI worktree: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606`
- Fetched local audit copy: `.tmp_remote_current_oci/nemo_rl/...`
- Files reviewed:
  - `nemo_rl/models/megatron/setup.py`
  - `nemo_rl/models/megatron/draft/utils.py`
  - `nemo_rl/models/policy/workers/megatron_policy_worker.py`
  - `nemo_rl/algorithms/grpo.py`

The fetched remote files pass `python3 -m py_compile`.

## Submit Decision

Do not submit new official PARD-2 online-drafter or SuffixDecoding online-drafter jobs yet.

The current code is safe to continue using for static vLLM PARD-2, static vLLM SuffixDecoding, Eagle-3, and generic PARD online training within the constraints below. It is not safe to claim official PARD-2 online training from this code path.

## 14:11 CEST Addendum - Official PARD-2 Overlay Review

A local training-side overlay now exists under `remote_patch_pard2_official/` and it compiles with `python3 -m py_compile`. It adds target hidden-state capture, PARD-2 target-feature construction, `target_proj` attachment/export fields, and a guarded official-PARD-2 training branch for NeMo-RL.

Do not submit official PARD-2 online jobs from this overlay yet. The pre-submit runtime review found a vLLM evaluation/refit blocker: the current native `method=pard2` path in the local vLLM v0.20 snapshot is only a draft-model alias, not official PARD-2 target-feature conditioning.

Specific blockers:

- `.tmp_vllm_v020/vllm/v1/spec_decode/draft_model.py:24`: `DraftModelProposer` always constructs the base proposer with `pass_hidden_states_to_model=False`, so PARD-2 target hidden states are not sent to the drafter.
- `.tmp_vllm_v020/vllm/v1/worker/gpu_model_runner.py:532`: `method=pard2` routes through `uses_draft_model()` and therefore through `DraftModelProposer`.
- `.tmp_vllm_v020/vllm/v1/worker/gpu_model_runner.py:557`: auxiliary target hidden-state capture is enabled for DFlash/Eagle3/extract-hidden-states paths, but not for PARD-2.
- `.tmp_vllm_v020/vllm/v1/worker/gpu_model_runner.py:4961`: auxiliary layer selection reads `eagle_aux_hidden_state_layer_ids` / DFlash config only, not `pard2_target_layers`.
- `.tmp_vllm_v020/vllm/model_executor/models/qwen3.py:316`: `Qwen3ForCausalLM.forward()` does not accept a `hidden_states` kwarg and has no `target_proj` / `target_feat` injection path.
- `experiments/eagle3_qwen3_235b/patches/vllm_pard2_method_alias.patch` only registers `pard2` as a draft-model method and fixes proposer return shape; it does not implement `target_proj` loading or conditioning.

Result: an official PARD-2 online submission would train a more correct drafter-side path, but the generation/refit benchmark would still run the alias path unless vLLM is patched. That would make the benchmark result misleading.

## 14:42 CEST Addendum - vLLM Patch Prepared

Prepared a source patch for the vLLM side:

- `experiments/eagle3_qwen3_235b/patches/vllm_pard2_official_target_feat.patch`

The patch applies on top of `vllm_pard2_method_alias.patch` and adds target-feature plumbing for Qwen3 PARD-2: hidden-state passing, `pard2_target_dim` buffer sizing, `pard2_target_layers` auxiliary capture, `target_proj` construction, `warp_model.bin` loading, and embedding injection.

Local gates passed:

- `python3 -m py_compile` for the edited vLLM files.
- Reverse patch dry-run against the edited snapshot.
- Forward patch dry-run on a scratch alias-baseline copy.

Still do not submit official PARD-2 online benchmark jobs until this patch passes a Lyris/container smoke proving `target_proj.weight` is loaded and Qwen3 draft forward consumes target features without shape errors.

## Findings

- Generic PARD online training is implemented. The PARD builder supports target-independent token/K-slot draft training and exports train-owner draft weights for vLLM refit.
- Generic PARD online training currently requires target pipeline parallel size 1, draft pipeline parallel size 1, and draft tensor parallel size matching target tensor parallel size.
- PARD-2-style generic fallback exists, but it is explicitly not the official PARD-2 online training path.
- Official PARD-2 checkpoints are detected through PARD-2 config fields such as `pard2`, `spd_type=pard2`, `pard2_target_layers`, and `pard2_target_dim`.
- For official PARD-2 checkpoints, the reviewed code raises `NotImplementedError` unless `allow_generic_pard2_fallback=true`.
- The guard explains the missing pieces: target hidden state capture, `target_feat` injection through `target_proj`, and refit/export of `warp_model.bin` / `target_proj` weights into vLLM.
- Static vLLM PARD-2 evaluation remains supported by disabling online draft training and using the vLLM speculative config path.
- SuffixDecoding has no trainable draft model in this code path. Current support is static vLLM `method=suffix`; online SuffixDecoding-style training would require a new objective/data path and a refit target, not just a config flag.

## Evidence Pointers

- `.tmp_remote_current_oci/nemo_rl/models/megatron/draft/utils.py:1479`: PARD/PARD-2 builder path.
- `.tmp_remote_current_oci/nemo_rl/models/megatron/draft/utils.py:1496`: official PARD-2 detection.
- `.tmp_remote_current_oci/nemo_rl/models/megatron/draft/utils.py:1504`: `NotImplementedError` guard for official PARD-2 online training.
- `.tmp_remote_current_oci/nemo_rl/models/megatron/draft/utils.py:1738`: non-owner draft copies are skipped during export/refit.
- `.tmp_remote_current_oci/nemo_rl/models/megatron/setup.py:992`: train-owner metadata is attached to the draft model.
- `.tmp_remote_current_oci/nemo_rl/models/policy/workers/megatron_policy_worker.py:1219`: draft refit only exports PARD weights from the train owner.

## Patch Bundle Drift

The actual remote OCI worktree is ahead of the local tracked patch bundle under `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL`. In particular, the remote `draft/utils.py` contains the PARD-2 guard and train-owner refit logic, while the local bundle is stale or incomplete for these files.

Before implementing official PARD-2 online training, sync the reviewed remote source back into a tracked patch bundle or create a clean replacement patch branch. Otherwise future submissions may use a different code path than the one reviewed here.

## Required Work Before Official PARD-2 Online Submission

- Capture target hidden states/features at the PARD-2 target layers during rollout/training.
- Feed those features into the PARD-2 `target_proj` / `target_feat` path during draft training.
- Include `warp_model.bin`, `target_proj`, and any associated PARD-2 projection weights in the refit/export path.
- Add a small official PARD-2 online smoke test that fails if the generic fallback is used.
- Re-run static-equivalent vs online comparisons only after the above path is implemented.
- Patch vLLM so `method=pard2` passes target hidden states to the drafter, captures configured PARD-2 target layers, loads `target_proj`/`warp_model.bin`, and injects projected target features into the draft model embeddings.

## 14:40 CEST Addendum - vLLM Runtime Smoke Passed, Online Gate Still Closed

The vLLM-side official PARD-2 patch now passed a Lyris container smoke:

- Job `2104955`, tag `pard2_official_target_feat_smoke_retry4_20260612`, completed `0:0`.
- The patch loads `warp_model.bin`, attaches `target_proj` after the normal drafter checkpoint load, captures `pard2_target_layers`, and feeds target features into Qwen3 draft forward.
- The PARD-2 parallel-drafting masked slots now repeat the last target feature, matching the AMD PARD-2 inference implementation.
- Smoke output: `output_tok_s_per_gpu=21.72`, `acceptance_rate=0.00%`, `drafted=381`, `accepted=0`.

This clears the static vLLM runtime gate for official PARD-2 target-feature conditioning. It does not clear the NeMo-RL official online-training gate above: training/refit/export still needs target-feature capture and `warp_model.bin` / `target_proj` ownership handling in the online drafter path.

## 14:54 CEST Addendum - Low-K PARD-2 Diagnostic

A lower-K official PARD-2 target-feature diagnostic also completed:

- Job `2104975`, tag `pard2_official_target_feat_k1_swev4_20260612`, completed `0:0`.
- Setup: `Qwen/Qwen3-8B` target, `amd/PARD2-Qwen3-8B` draft, `method=pard2`, `K=1`, SWE-Bench Verified prompts, `ISL=2048`, `OSL=128`, `prompt_count=4`.
- Result: `output_tok_s_per_gpu=22.34`, `acceptance_rate=0.395%`, `drafted=506`, `accepted=2`.

This reinforces the split decision: the static official PARD-2 vLLM runtime path works, but current SWE-Bench acceptance is too low to justify claiming useful PARD-2 performance, and it still does not unblock official PARD-2 online drafter training in NeMo-RL.

## 15:12 CEST Addendum - Online Refit Guard

Reviewed the local official-PARD-2 training overlay again and added a narrow vLLM refit guard in:

- `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_backend.py`
- `scripts/test_vllm_draft_refit_target_proj.py`

The guard only activates when streamed draft weights contain `target_proj.*`. In that case, the vLLM drafter must report loaded weight names and must include every streamed `target_proj.*` key; otherwise refit raises immediately instead of silently dropping the PARD-2 projection.

Validation:

- `python3 scripts/test_vllm_draft_refit_target_proj.py` passed.
- `python3 -m py_compile` passed for the official-PARD-2 overlay, the vLLM backend refit patch, and the focused test.
- `bash -n` passed for the corrected Lyris PARD-2 target-feature smoke launcher.
- `scripts/test_pard2_target_feature_alignment.py` still needs a real torch environment; local Python does not have `torch`.

Submission status after review:

- Old smoke `2104851` failed before model execution due to a broad method-alias patch conflict.
- Follow-up smoke `2106398` passed patch application, Python compile, and the initial PARD-2 source checks, but failed before generation because the alias helper skipped the exact `SpeculativeMethod` Literal entry and Pydantic rejected `method=pard2`.
- The smoke launcher now checks for the exact `    "pard2",` Literal line and the in-job source check verifies `literal_accepts_pard2=True`.
- New smoke `2106641` ran on Lyris with tag `pard2_official_target_feat_smoke_r3_20260612` and completed `0:0` on `lyris0098`. It passed patch application, Python compile, and the improved source checks including `literal_accepts_pard2=True`, loaded target and drafter, selected PARD-2 target layers `(36, 29, 21, 13)`, and wrote `breakdown.json`.
- Smoke result: 21.50 tok/s/GPU, 0.00% acceptance, 381 drafted, 0 accepted.

This improves the online refit safety check, but it is still not a full official PARD-2 online-training pass. Do not claim NeMo-RL official PARD-2 online drafter training until a remote torch/vLLM training smoke proves target hidden-state capture, `target_proj` training/export, and vLLM refit together.

15:29 CEST pre-submit review:

- Patched the local shared launcher so `DRAFT_FORMAT=pard2` selects `SPECDEC_METHOD=pard2` instead of the older `draft_model` alias, enables PARD-style parallel drafting, and recomputes default `POLICY_DRAFT_LOSS=pard2` / CAT weighting after auto-selecting `POLICY_DRAFT_TYPE=pard2`.
- Local validation passed: `bash -n` for the online launcher and PARD-2 smoke launcher, `py_compile` for the official overlay and vLLM refit guard, and `scripts/test_vllm_draft_refit_target_proj.py`.
- Local `scripts/test_pard2_target_feature_alignment.py` still cannot run on this Mac because `torch` is not installed locally.
- OCI remote review showed `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606` still has the guard/fallback-only official PARD-2 path and does not have the new official overlay installed. I did not patch that live checkout in place because active/pending jobs are using it.
- No NeMo-RL official online-PARD2 benchmark was submitted from OCI in this pass. The submitted follow-up was a Lyris static-vLLM target-feature diagnostic, job `2106663`, to test longer OSL acceptance with the already-smoked official vLLM patch.

15:49 CEST progress:

- Added reusable official-PARD2 vLLM patch helpers under `experiments/eagle3_qwen3_235b/patches/` and `experiments/eagle3_online/prepare_pard2_official_vllm_site.sh`.
- Extended the NeMo-RL online launcher so `DRAFT_FORMAT=pard2` can build/check a patched vLLM source site from `PARD2_OFFICIAL_VLLM_PATCH_DIR` before GRPO starts.
- Added `experiments/eagle3_online/submit_qwen8_pard2_official_online_smoke_20260612.sh`, which stages a separate OCI checkout at `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612` rather than modifying the live checkout.
- Dry-run passed with `method=pard2`, `allow_generic_pard2_fallback=false`, official vLLM patch-site setup, and `NEMO_RL_PY_EXECUTABLES_SYSTEM=0`.
- Submitted OCI job `3278974`: Qwen3-8B official PARD-2 online smoke, `K=1`, `MAX_STEPS=2`, staged checkout, account `coreai_dlalgo_llm`. Initial state: pending on priority.

16:18 CEST retry after code review:

- `3278974` failed before GRPO with `ModuleNotFoundError: No module named 'vllm'` in `prepare_pard2_official_vllm_site.sh`; the failing interpreter was `/opt/nemo_rl_venv/bin/python`.
- Patched the helper to accept `PARD2_VLLM_SOURCE_SITE`, probe usable Python executables, and build from a clean mounted vLLM source tree instead of assuming the driver venv can import vLLM.
- Converted `check_pard2_official_patch.py` from import/`inspect` checks to static source-file marker checks. This avoids torch/vLLM import failures during patch preparation while still verifying `method=pard2`, hidden-state passing, `target_proj`, `warp_model.bin`, repeated target features, and `pard2_target_layers`.
- Updated the Qwen8 smoke wrapper to use the clean source tree at the base actor venv and to preflight that path before `sbatch`.
- Validation passed: local `bash -n`, local `py_compile`, remote dry-run, and direct login-node patch build/check. The direct build produced all `PARD2_OFFICIAL_PATCH_CHECKS=True`.
- Resubmitted OCI job `3279154`: Qwen3-8B official PARD-2 online smoke, `K=1`, `MAX_STEPS=2`, staged checkout, account `coreai_dlalgo_llm`. Current state at submission: pending on priority.

16:36 CEST target-projection export fix:

- `3279154` cleared the vLLM patch-prep gate and reached model setup: the patched vLLM site was built, all source checks were true, vLLM loaded Qwen3-8B plus `amd/PARD2-Qwen3-8B`, and PARD-2 target layers `(36, 29, 21, 13)` were selected.
- It failed during initial `prepare_refit_info` before training with `RuntimeError: Official PARD-2 draft is missing target_proj during export.`
- Root cause: `_attach_pard2_target_projection()` attaches `target_proj` to the draft module object returned by the provider, while `export_pard_weights_to_hf()` exported the final unwrapped module and only checked `getattr(unwrapped_draft_model, "target_proj", None)`.
- Patch: added `_get_pard2_target_projection()` in `remote_patch_pard2_official/nemo_rl/models/megatron/draft/utils.py` and changed export to find `target_proj` across the same wrapper chain used by the PARD-2 training hook.
- Validation passed: local `py_compile` for `draft/utils.py`, `draft/pard.py`, and `train.py`; remote dry-run passed after staging the fixed overlay.
- Resubmitted OCI job `3279229` with run id `20260612_qwen8_pard2_official_online_smoke_targetprojfix`. Current state at submission: pending on priority.

16:52 CEST reviewed submission:

- Re-ran the pre-submit review on the official PARD-2 online launch path.
- Local validation passed: `bash -n` for `prepare_pard2_official_vllm_site.sh`, `submit_nemorl_online_draft_specdec.sh`, and `submit_qwen8_pard2_official_online_smoke_20260612.sh`; `py_compile` for `remote_patch_pard2_official/nemo_rl/models/megatron/draft/utils.py`, `remote_patch_pard2_official/nemo_rl/models/megatron/draft/pard.py`, `remote_patch_pard2_official/nemo_rl/models/megatron/train.py`, and the official vLLM patch helpers.
- OCI smoke `3279229` completed successfully (`COMPLETED`, exit `0:0`). It reached `SETUP COMPLETE`, ran both `Step 1/2` and `Step 2/2`, kept `Draft Training Enabled: True`, exported PARD draft weights during refit, and had no `Traceback`, `RuntimeError`, `ValueError`, or `ERROR` lines in the driver log scan.
- The smoke is not a performance win: final SpecDec log lines showed very low acceptance (`0/63` and `1/62` accepted/drafted in the compact tail). Treat it as a correctness gate for official PARD-2 online plumbing only.
- Dry-run preflight passed for the isolated 20-step stage at `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-online20-20260612`.
- Submitted OCI job `3279589`: Qwen3-8B official PARD-2 online, `K=1`, 20 GRPO steps, `NUM_PROMPTS=4`, `NUM_GENERATIONS=4`, `MAX_NEW_TOKENS=256`, `MAX_MODEL_LEN=2048`, account `coreai_dlalgo_llm`. Initial state: `PENDING (Priority)`.

16:58 CEST live gate:

- OCI job `3279589` started on `nvl72108-T17`.
- Early driver log passed `PARD2_OFFICIAL_PATCH_CHECKS` with all checks true, rebuilt the official patched vLLM site, and loaded the vLLM engine with `SpeculativeConfig(method='pard2', model='amd/PARD2-Qwen3-8B', num_spec_tokens=1)`.
- vLLM selected PARD-2 target layers `(36, 29, 21, 13)` and began `lm_policy` worker initialization. No setup/training result is available yet.

17:16 CEST Lyris submission review:

- Added and reviewed `experiments/eagle3_online/submit_lyris_qwen8_pard2_official_online_20260612.sh` for the Lyris path. It stages an isolated checkout at `/lustre/fsw/coreai_dlalgo_llm/users/sna/SpecDec-RL-pard2-official-online-lyris-20260612`, keeps the account at `coreai_dlalgo_llm`, uses partition `gb200`, and intentionally emits no `--gres` or `--segment` flags.
- The Lyris wrapper uses the reviewed OCI online overlay plus `remote_patch_pard2_official/`, and points `PARD2_VLLM_SOURCE_SITE` at the already-smoked official PARD-2 vLLM 0.20.2 source site: `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/vllm-runs/pard2_official_target_feat_smoke_r3_20260612/patched_vllm_site`. This avoids applying the official PARD-2 vLLM 0.20 patch to Lyris `RL-nightly-test` vLLM 0.17.
- Validation before corrected submission passed: local `bash -n`, local `py_compile`, `scripts/test_vllm_draft_refit_target_proj.py`, remote wrapper dry-run, remote source static checks, and remote `PARD2_OFFICIAL_PATCH_CHECKS`.
- First Lyris attempt `2107184` failed before GRPO with `No interpreter found for Python 3.13.13`. The wrapper was fixed to run `uv` with `/opt/nemo_rl_venv/bin/python3`; this is a launch-environment correction, not a PARD-2 code-path change.
- Corrected Lyris job `2107214` was submitted after the dry-run passed. Initial state: `RUNNING` on `lyris0108`, past the previous interpreter failure and building the `uv` environment. No training/performance result is available yet.

17:19 CEST poll result:

- OCI 20-step official PARD-2 online run `3279589` completed successfully (`COMPLETED`, exit `0:0`, elapsed `00:18:00`). It reached `Step 20/20`, kept `Draft Training Enabled: True`, exported/refit 312 draft weights, and vLLM reported `target_proj.weight` in the loaded draft weights. This is now a 20-step correctness pass for the official PARD-2 online/refit path.
- The same run is not a performance win: final-tail SpecDec metrics were only around `2.2%` to `3.2%` average draft acceptance, with many step metrics between `0.0%` and `5.8%`.
- Lyris corrected job `2107214` remains `RUNNING` on `lyris0108`; it passed `PARD2_OFFICIAL_PATCH_CHECKS` and the existing official PARD-2 vLLM site check, then entered `uv` dependency build/install.
- Existing Lyris standalone jobs moved terminal: Qwen8 Verified OSL96K PARD-2 K5 `2103214` timed out at 5h with no `breakdown.json` and a `0.0%` acceptance tail; Qwen235B Verified OSL64K PARD K3 `2104995` failed after `02:26:29` with CUDA illegal memory access in Qwen3-MoE rotary embedding followed by vLLM `EngineDeadError`.

17:25 CEST refresh result:

- Reran the Lyris long-OSL refresh, Qwen235B refresh, OCI/SWERL refresh, and regenerated the expected-performance HTML/PNG/CSV artifacts.
- Qwen30 batch-long no-spec retry `3277195` now has 19 comparable `step>=2` rows and replaces the older partial `3274971` timeout as the long-output no-spec denominator. Static/online PARD-2 remain faster than no-spec in this Qwen30 long-output GRPO regime: static PARD-2 `1.818x` step speedup, online start1 `1.821x`, online start10 `1.797x`, online start5 interval5 `1.797x`, and suffix K32 `1.741x`.
- This refresh does not change the official-PARD-2 online assessment on SWE-style Qwen8: correctness/refit is working, but acceptance remains low in the completed OCI run. The Lyris official-PARD2 online job `2107214` is still the active live gate and has not reached model setup yet.

17:34 CEST Lyris MATH submission review:

- Reviewed the Lyris standalone vLLM submitter and the official PARD-2 target-feature smoke submitter before launching new jobs.
- Validation passed: local `bash -n` for both launchers, local `py_compile` for `standalone_vllm_specdec_breakdown.py` and `materialize_openmath_prompts.py`, remote Lyris `py_compile` for the staged benchmark script, and remote JSONL parse/count for the staged MATH prompt file.
- Fixed one prompt-compatibility issue before submission: `standalone_vllm_specdec_breakdown.py` now accepts the MATH prompt field `data` in addition to `prompt`, `question`, `problem`, and `input`.
- Submitted Qwen3-8B MATH500 OSL32K jobs on Lyris, account `coreai_dlalgo_llm`, `ISL=4096`, `OSL=32768`, `prompt_count=4`, batch sizes `1 2`: baseline `2107295`, suffix K32 `2107296`, Eagle-3 K3 `2107297`, official PARD-2 K3 `2107298`, and official PARD-2 K5 `2107302`.
- Immediate launch logs were clean: standard vLLM jobs reached engine initialization; both official-PARD2 jobs passed all `PARD2_OFFICIAL_PATCH_CHECKS`; no initial `Traceback`, CUDA/OOM, `EngineDeadError`, or import failure was present.

17:40 CEST Qwen30 MATH PARD submission:

- Confirmed no existing Lyris Qwen3-30B-A3B MATH500 OSL32K PARD run was present before submitting.
- Re-ran correct validation: local `bash -n` for the standalone launcher and MATH refresh script, local `py_compile` for the Python benchmark/extractor paths, remote Lyris `py_compile` for the staged benchmark script, and remote parse/count for the 20-row MATH prompt JSONL.
- Submitted Qwen3-30B-A3B MATH500 OSL32K jobs on Lyris, account `coreai_dlalgo_llm`, same prompt file and `ISL=4096`, `OSL=32768`, `prompt_count=4`, batch sizes `1 2`: baseline `2107332`, suffix K32 `2107333`, PARD K5 `2107334`, and PARD K3 `2107335`.
- Verified generated configs: PARD K3/K5 use `amd/PARD-Qwen3-0.6B`, `method=draft_model`, `parallel_drafting=true`, and the intended K values.
- Added `scripts/refresh_lyris_math500_osl32k_results.sh`; the 17:45 refresh writes the MATH manifest/status/metrics docs and currently shows 9 running jobs with no completed `breakdown.json` rows yet.
- At 17:48 CEST, fixed the refresh script for macOS Bash 3 compatibility by replacing `mapfile` with a read loop. The rerun extracted the first partial metric row: Qwen3-8B suffix K32 BS1, `216.02` tok/s/GPU, `84.51%` acceptance, mean acceptance length `6.88`. Matching baseline is still running, so no speedup is claimed yet.

17:44 CEST Lyris online PARD-2 retry:

- Lyris official online-PARD2 job `2107214` failed before GRPO setup, after dependency build, with `ImportError: cannot import name 'get_gdpo_reward_component_keys' from 'nemo_rl.algorithms.utils'`.
- This was a staged-source mismatch: newer `advantage_estimator.py` / `grpo.py` expected the helper, while the staged utility overlay did not provide it.
- Patched `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/algorithms/utils.py` with `get_gdpo_reward_component_keys(batch)`, returning `reward1`, `reward2`, ... keys in numeric order.
- Validation passed: local `bash -n`, local `py_compile`, dry-run Lyris staging, remote `py_compile`, `scripts/test_vllm_draft_refit_target_proj.py`, and all `PARD2_OFFICIAL_PATCH_CHECKS`.
- Submitted retry `2107361` with run id `20260612_qwen8_pard2_official_online_lyris_gdpoimportfix`; current state `RUNNING` on `lyris0230`, with early log past `PARD2_OFFICIAL_PATCH_CHECKS` and no repeat of the previous import error.

17:51 CEST post-submit review:

- Re-ran the relevant local gates before considering any additional submission: `bash -n` passed for the Lyris online launcher, shared online launcher, PARD-2 vLLM site helper, standalone vLLM launcher, and MATH refresh script; `py_compile` passed for the PARD-2 utility overlay, standalone vLLM benchmark, metrics extractor, and refit guard test.
- Refreshed `docs/lyris_math500_osl32k_status_20260612.md`; all 9 MATH500 OSL32K benchmark jobs remain `RUNNING`. The only completed row is still Qwen3-8B suffix K32 BS1 at `216.02` tok/s/GPU, `84.51%` acceptance, and mean accept length `6.88`; no speedup is claimed because the matched baseline is still running.
- Focused Lyris log scan found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in the current MATH job tails. Qwen30 suffix has started BS1 with live chunk output, but no final Qwen30 benchmark row exists yet.
- Online PARD-2 retry `2107361` remains `RUNNING` on `lyris0230`; it has passed all `PARD2_OFFICIAL_PATCH_CHECKS`, installed the first dependency batch, and has not repeated the previous `get_gdpo_reward_component_keys` import error. It is still building native dependencies and has not reached `SETUP COMPLETE`.
- No duplicate jobs were submitted in this pass because the reviewed jobs are already active.

17:55 CEST MATH refresh:

- Qwen3-8B MATH500 suffix K32 job `2107296` completed successfully. Refreshed metrics now include BS1 `216.02` tok/s/GPU, `84.51%` acceptance, mean accept length `6.88`, and BS2 `469.83` tok/s/GPU, `91.24%` acceptance, mean accept length `9.48`.
- The matching Qwen3-8B baseline `2107295` is still running, so suffix speedup remains uncomputed. Current MATH queue state is 1 completed / 8 running.
- Qwen30 suffix has live BS1 chunk progress, but no final Qwen30 row exists yet. Online PARD-2 retry `2107361` is still in dependency build and has not reached `SETUP COMPLETE`.
- Regenerated `docs/lyris_specdec_expected_performance_20260612.{png,md,html}` and `docs/lyris_specdec_expected_performance_raw_20260612.csv` after adding MATH500 metrics as a default source in `scripts/plot_lyris_specdec_expected_performance.py`. The raw table now has 173 rows and keeps MATH500 rows separate from SWE-Bench rows with a `MATH500` shape prefix.

17:58 CEST poll:

- Refreshed MATH status remains 1 completed / 8 running with the same 2 final rows. Qwen30 suffix has live progress through at least three BS1 chunks; Qwen8 Eagle-3 has live chunk-1 progress at `52.35` tok/s/GPU and `33.82%` acceptance. No new final `breakdown.json` exists yet.
- Lyris online PARD-2 retry `2107361` remains `RUNNING` and before `SETUP COMPLETE`. It has built several native dependencies after source checks, including `flash-attn`, `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm`, with no repeated import/runtime error in the scanned tail.
- No duplicate job was submitted.

18:00 CEST MATH refresh:

- Qwen3-30B-A3B MATH500 suffix K32 job `2107333` now has a final BS1 row: `146.74` tok/s/GPU, `88.43%` acceptance, mean accept length `7.85`.
- The matching Qwen30 baseline `2107332` is still running, so speedup is not computed yet.
- Regenerated the expected-performance bundle. `docs/lyris_specdec_expected_performance_raw_20260612.csv` now has 174 rows and includes three MATH500 final rows.

18:02 CEST poll:

- MATH status remains 3 final rows and 8 running jobs. No new final row appeared after Qwen30 suffix BS1.
- Active log scan found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in the checked MATH job tails. Qwen8 Eagle-3 has live BS1 chunk progress through chunk 2/4: `52.35` then `90.43` tok/s/GPU with acceptance moving from `33.82%` to `79.49%`.
- Online PARD-2 retry `2107361` remains before `SETUP COMPLETE`, still building native dependencies after the source checks. No duplicate job was submitted.

18:08 CEST post-submit review:

- Refreshed MATH status again: 9 tracked jobs, with Qwen3-8B suffix `2107296` completed and the remaining 8 jobs still `RUNNING`.
- Final MATH metric rows remain unchanged: Qwen3-8B suffix K32 BS1/BS2 and Qwen3-30B-A3B suffix K32 BS1. Matching no-spec baselines are still running, so MATH speedups remain blank.
- Re-ran local gates: `bash -n` passed for the Lyris online launcher, shared online launcher, PARD-2 vLLM site helper, standalone vLLM launcher, and MATH refresh script; `py_compile` passed for the standalone runner, metrics scripts, plot script, and PARD-2 utility overlay.
- Re-ran remote Lyris gates on the staged official-PARD2 online checkout: `bash -n`, Python compile, and `PARD2_OFFICIAL_PATCH_CHECKS` all passed. The staged MATH prompt file has 20 parseable rows, all with nonempty `data`.
- Active-tail scan again found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in checked MATH logs. Qwen30 PARD K5 has live BS1 chunk 1/4 at `41.98` tok/s/GPU with `52.32%` acceptance; Qwen30 suffix is progressing through BS2 chunk 1/2.
- Lyris online PARD-2 retry `2107361` is still `RUNNING` on `lyris0230`, elapsed `00:21:28`, before `SETUP COMPLETE`. It passed source checks and dependency build is still progressing, with no repeat of the previous GDPO import failure.
- No new job was submitted in this pass because the reviewed MATH and online-PARD2 arms are already active. Submitting duplicates now would mix repeat rows with the current apples-to-apples baseline/specdec matrix.

18:14 CEST online-PARD2 maskfix retry:

- Lyris online PARD-2 retry `2107361` failed after dependency build with another staged-source utility mismatch: `ImportError: cannot import name 'mask_out_neg_inf_logprobs' from 'nemo_rl.algorithms.utils'`.
- Verified the helper exists in the Lyris base checkout at `/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-nightly-test/nemo_rl/algorithms/utils.py` and is imported by the overlaid PARD-2 loss/train paths.
- Patched `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/algorithms/utils.py` to include the same top-k/top-p `-inf` logprob masking helper, alongside the earlier GDPO reward-key helper.
- Validation passed before resubmit: local `py_compile`, local `bash -n`, Lyris dry-run staging, remote refit guard test, and all `PARD2_OFFICIAL_PATCH_CHECKS`.
- Submitted Lyris maskfix retry `2107520` with run id `20260612_qwen8_pard2_official_online_lyris_maskfix`, account `coreai_dlalgo_llm`. It is `RUNNING` on `lyris0216`; the staged utility file contains both helper definitions.
- Early log for `2107520` has passed `PARD2_OFFICIAL_PATCH_CHECKS`, rebuilt the first dependency batch, and is building native dependencies. It has not reached `SETUP COMPLETE` yet, and no `ImportError`/`Traceback` appears in the scanned tail.
- MATH500 OSL32K status stayed at 3 final suffix rows and 8 running jobs, so no expected-performance regeneration was needed.

18:19 CEST MATH refresh and online-PARD2 poll:

- Refreshed MATH status: 9 tracked jobs, 2 completed / 7 running. Qwen3-8B suffix `2107296` and Qwen3-30B-A3B suffix `2107333` are completed.
- MATH metrics now have 5 final rows: Qwen8 suffix K32 BS1/BS2, Qwen8 Eagle-3 K3 BS1, and Qwen30 suffix K32 BS1/BS2.
- New rows from this refresh: Qwen8 Eagle-3 K3 BS1 `74.38` tok/s/GPU, `62.94%` acceptance, mean accept length `2.89`; Qwen30 suffix K32 BS2 `303.81` tok/s/GPU, `92.52%` acceptance, mean accept length `9.81`.
- Matching no-spec baselines are still running, so all MATH speedup cells remain blank.
- Regenerated the expected-performance bundle. `docs/lyris_specdec_expected_performance_raw_20260612.csv` now has 176 rows and includes all 5 MATH500 rows.
- Active-tail scan again found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in checked MATH logs. Qwen30 PARD K5 and K3 each have live BS1 chunk progress but no aggregate row yet.
- Online PARD-2 retry `2107520` remains `RUNNING` on `lyris0216`, elapsed `00:06:38`, still before `SETUP COMPLETE` in native dependency build. The scanned tail has no `ImportError`/`Traceback`.

18:21 CEST no-new-row poll:

- Refreshed MATH status remains 2 completed / 7 running with the same 5 final MATH rows and 176 expected-performance raw rows. No report regeneration was needed.
- MATH baselines are still running, so all MATH speedup cells remain blank.
- Active-tail scan found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in checked MATH logs.
- Live progress: Qwen8 baseline has BS1 chunks 1/4 and 2/4 at about `36.9` tok/s/GPU; Qwen30 PARD K5 has BS1 chunks 1/4 and 2/4 at `41.98` and `39.90` tok/s/GPU with `52.32%` and `47.50%` acceptance; Qwen30 PARD K3 has BS1 chunk 1/4 at `37.38` tok/s/GPU with `71.64%` acceptance.
- Online PARD-2 retry `2107520` remains `RUNNING` on `lyris0216`, elapsed `00:08:34`, still before `SETUP COMPLETE` in native dependency build. No new `ImportError`/`Traceback` appears in the scanned tail.
- No new job was submitted; the useful MATH and online-PARD2 arms are already active and clean.

18:23 CEST no-new-row poll:

- Refreshed MATH status remains 2 completed / 7 running with the same 5 final rows and 176 expected-performance raw rows. No expected-performance regeneration was needed.
- MATH baseline rows are still not final, so all MATH speedups remain blank.
- Active-tail scan found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in checked MATH logs.
- Live progress remains useful but partial: Qwen8 baseline BS1 chunk 2/4 is complete at about `36.82` tok/s/GPU; Qwen30 PARD K5 has BS1 chunks 1/4 and 2/4 at `41.98` and `39.90` tok/s/GPU with `52.32%` and `47.50%` acceptance; Qwen30 PARD K3 has BS1 chunk 1/4 at `37.38` tok/s/GPU with `71.64%` acceptance.
- Online PARD-2 retry `2107520` remains `RUNNING` on `lyris0216`, elapsed `00:10:52`, still before `SETUP COMPLETE` in native dependency build. The scanned tail shows `deep-ep` built and no `ImportError`/`Traceback`.
- No patch or resubmission was needed in this pass.

18:28 CEST no-new-row poll:

- Refreshed MATH status remains 2 completed / 7 running with the same 5 final rows and 176 expected-performance raw rows. No expected-performance regeneration was needed.
- MATH baseline rows are still not final, so all MATH speedups remain blank.
- Active-tail scan found no `Traceback`, CUDA/OOM, `EngineDeadError`, or import errors in checked MATH logs.
- Live MATH progress remains partial: Qwen8 baseline has BS1 chunks 1/4 and 2/4 complete at about `36.9` tok/s/GPU; Qwen30 PARD K5 has BS1 chunks 1/4 and 2/4 complete at `41.98` and `39.90` tok/s/GPU with `52.32%` and `47.50%` acceptance; Qwen30 PARD K3 has BS1 chunk 1/4 complete at `37.38` tok/s/GPU with `71.64%` acceptance.
- Online PARD-2 retry `2107520` remains `RUNNING` on `lyris0216`, elapsed `00:15:36`, still before `SETUP COMPLETE` in native dependency build. The scanned tail shows `flash-attn`, `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` built, with no `ImportError`/`Traceback`.
- No patch or resubmission was needed in this pass.

18:40 CEST load-dataloader-state retry:

- Lyris online PARD-2 retry `2107520` failed after dependency setup with another staged-source compatibility mismatch: `ImportError: cannot import name 'load_dataloader_state' from 'nemo_rl.data.utils'`.
- Root cause: the overlaid newer `grpo.py` imports `load_dataloader_state`, while the Lyris base `nemo_rl.data.utils` module does not define it. The checkpoint semantics in the same `grpo.py` save `train_dataloader.pt` for normal dataloaders and `train_dataloader_<task>.pt` for multiple dataloaders.
- Patched `.tmp_remote_current_oci/nemo_rl/algorithms/grpo.py` with a local fallback implementation that preserves those filenames and falls back to `train_dataloader.pt` for older checkpoints.
- Validation passed before resubmit: local `py_compile`, local `bash -n`, `git diff --check`, Lyris dry-run staging, remote Python compile of staged `grpo.py`/`utils.py`, remote refit guard test, and all `PARD2_OFFICIAL_PATCH_CHECKS`.
- Submitted replacement Lyris job `2107786` with run id `20260612_qwen8_pard2_official_online_lyris_loadfix`, account `coreai_dlalgo_llm`. Initial state: `RUNNING` on `lyris0044`; the staged source contains the fallback at `nemo_rl/algorithms/grpo.py`.

18:43 CEST loadfix poll:

- Replacement job `2107786` remains `RUNNING` on `lyris0044`, elapsed `00:03:17`.
- The current driver log has passed `PARD2_OFFICIAL_PATCH_CHECKS`, installed the first dependency batch, and is building native dependencies. It has not reached `SETUP COMPLETE` yet.
- No repeated `load_dataloader_state`, `mask_out_neg_inf_logprobs`, GDPO reward-key, or other import error appears in the scanned tail.

18:45 CEST loadfix poll:

- Replacement job `2107786` remains `RUNNING` on `lyris0044`, elapsed `00:05:31`.
- It is still in runtime setup/native dependency build after passing `PARD2_OFFICIAL_PATCH_CHECKS` and installing the first dependency batch.
- The scanned driver tail still has no `Traceback`, `ImportError`, `ModuleNotFoundError`, or repeat of the previous staged-source helper failures.

18:48 CEST loadfix poll:

- Replacement job `2107786` remains `RUNNING` on `lyris0044`, elapsed about `00:07:25`.
- It is still before `SETUP COMPLETE`, with the current tail showing dependency build progress and the already-passed `PARD2_OFFICIAL_PATCH_CHECKS`.
- The scanned driver tail still has no actionable `Traceback`, `RuntimeError`, `ImportError`, `ModuleNotFoundError`, or helper-mismatch failure.

18:52 CEST loadfix review/poll:

- Re-ran the local pre-submission gates against the current launcher and overlays: `py_compile`, `bash -n`, and `git diff --check` all passed with no output.
- Replacement job `2107786` remains `RUNNING` on `lyris0044`, elapsed about `00:11:54`.
- The log has passed `PARD2_OFFICIAL_PATCH_CHECKS`, installed the first dependency batch, and built native dependencies including `flash-attn`, `deep-ep`, and `transformer-engine-torch`.
- The scanned driver tail still has no repeated `load_dataloader_state`, `mask_out_neg_inf_logprobs`, GDPO reward-key, or other import mismatch, and no actionable traceback/runtime/OOM error.

18:58 CEST loadfix poll:

- Replacement job `2107786` remains `RUNNING` on `lyris0044`, elapsed about `00:17:05`.
- It is still in setup before `SETUP COMPLETE`; the current tail shows the already-passed `PARD2_OFFICIAL_PATCH_CHECKS` and completed native dependency builds including `flash-attn`, `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm`.
- The scanned driver tail still has no repeated `load_dataloader_state`, `mask_out_neg_inf_logprobs`, GDPO reward-key, official PARD-2 patch, import, traceback, runtime, or OOM failure.

19:02 CEST loadfix poll:

- Replacement job `2107786` remains `RUNNING` on `lyris0044`, elapsed about `00:21:54`.
- Raw driver tail confirms this is still environment setup rather than GRPO/model execution: `flash-attn`, `deep-ep`, `transformer-engine-torch`, `nv-grouped-gemm`, and `mamba-ssm` have built successfully; `causal-conv1d` is still the visible native build without a final `Built causal-conv1d` line yet.
- There is still no `SETUP COMPLETE`, no model load, no GRPO step, and no repeated staged-source import/config/runtime/OOM failure.

19:06 CEST dataplane retry:

- Job `2107786` failed after native dependency setup with `ModuleNotFoundError: No module named 'nemo_rl.data_plane'`.
- Root cause: the overlaid newer `grpo.py` imports `DataPlaneConfig` from a package that does not exist in the Lyris `RL-nightly-test` base checkout. The only use is the optional `MasterConfig.data_plane` type annotation.
- Patched `.tmp_remote_current_oci/nemo_rl/algorithms/grpo.py` to fall back to `DataPlaneConfig = dict[str, Any]` when `nemo_rl.data_plane.interfaces` is unavailable.
- Validation passed before resubmit: local `py_compile`, local `bash -n`, `git diff --check`, Lyris dry-run staging, vLLM draft refit target-projection tests, and `PARD2_OFFICIAL_PATCH_CHECKS`.
- Submitted replacement job `2108021` with run id `20260612_qwen8_pard2_official_online_lyris_dataplanefix`, account `coreai_dlalgo_llm`. Initial poll: `RUNNING` on `lyris0217`; the remote staged `grpo.py` contains the `DataPlaneConfig` fallback and the driver log is in env setup after passing PARD-2 patch checks.

19:19 CEST dataplane retry poll:

- Replacement job `2108021` remains `RUNNING` on `lyris0217`, elapsed about `00:12:51`.
- It is still in driver environment setup after passing `PARD2_OFFICIAL_PATCH_CHECKS`; `flash-attn`, `deep-ep`, and `transformer-engine-torch` have built.
- No repeated `data_plane` import failure, no new import/config/runtime/OOM failure, and no `SETUP COMPLETE` or GRPO step yet.

19:36 CEST hydrafix submission:

- Replacement job `2108021` failed after dependency setup with Hydra `ConfigAttributeError` for `policy.megatron_cfg.force_reconvert_from_hf`.
- Root cause: the Lyris `RL-nightly-test` base config does not define `force_reconvert_from_hf`; this is an optional compatibility knob, so it must be force-added rather than strictly overridden.
- Patched `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` to pass `++policy.megatron_cfg.force_reconvert_from_hf=${FORCE_RECONVERT_FROM_HF}`.
- Patched `experiments/eagle3_online/submit_lyris_qwen8_pard2_official_online_20260612.sh` to force-add the optional nested DDP overlap keys with `++policy.megatron_cfg.distributed_data_parallel_config.*`.
- Validation passed before submit: local `py_compile`, local `bash -n`, `git diff --check`, Lyris dry-run staging/preflight, vLLM draft-refit target-projection tests, `PARD2_OFFICIAL_PATCH_CHECKS`, and a targeted Hydra parse of the exact dry-run override vector against the staged Lyris config.
- Submitted replacement job `2108168` with run id `20260612_qwen8_pard2_official_online_lyris_hydrafix`, account `coreai_dlalgo_llm`. Follow-up poll: `RUNNING` on `lyris0267`, in driver env setup before `SETUP COMPLETE`.

19:42 CEST hydrafix poll / benchmark refresh:

- Job `2108168` remains `RUNNING` on `lyris0267`, elapsed about `00:05:40`.
- It is still in driver environment setup after `PARD2_OFFICIAL_PATCH_CHECKS`; the scanned log has no repeated Hydra/config/import/runtime failure.
- It has not reached `SETUP COMPLETE`, model load, or GRPO step yet.
- Refreshed benchmark artifacts and regenerated the expected-performance report. MATH500 now has `10` final rows after Qwen8 baseline BS2 completed; SWE-Bench long-OSL and Qwen235B remain at `65` and `46` final rows.

19:46 CEST hydrafix poll / MATH refresh:

- Job `2108168` remains `RUNNING` on `lyris0267`, elapsed about `00:09:39`.
- It is still in driver environment setup after `PARD2_OFFICIAL_PATCH_CHECKS`; `deep-ep` has built, but there is no `SETUP COMPLETE`, model load, or GRPO step yet.
- The scanned log has no repeated Hydra/config/import/runtime failure.
- MATH500 now has `12` final rows after Qwen30 PARD K3/K5 BS2 completed. The Qwen30 baseline is still running, so Qwen30 MATH speedups remain pending.

20:37 CEST Ray/MasterConfig retry:

- Online PARD-2 `2108168` failed after environment setup with a Ray version mismatch: the Ray head used `2.49.2` while the driver used `2.54.0`.
- Patched `experiments/eagle3_online/submit_lyris_qwen8_pard2_official_online_20260612.sh` to force a job-local `ray[default]==2.54.0` runtime by disabling existing-env reuse for Ray and using `/opt/ray_venvs`.
- Replacement `2108268` passed Ray connect and data setup, then failed in `grpo.setup` because the overlaid newer GRPO code expected `MasterConfig` attribute access while the Lyris runner passed a plain dict.
- Patched `.tmp_remote_current_oci/nemo_rl/algorithms/grpo.py` to convert dict configs with `MasterConfig(**master_config)` at setup entry.
- Validation passed before relying on the replacement: local `python3 -m py_compile`, local `bash -n`, `git diff --check`, Lyris dry-run/preflight, vLLM draft-refit target-projection tests, and `PARD2_OFFICIAL_PATCH_CHECKS`.
- Submitted replacement job `2108402` with run id `20260612_qwen8_pard2_official_online_lyris_masterconfigfix`, account `coreai_dlalgo_llm`. Current state: `RUNNING` on `lyris0049`; the Ray head log confirms creation of the local Ray 2.54.0 venv, and the driver remains in environment/native dependency setup after source checks with no new traceback visible.

20:43 CEST masterconfigfix poll:

- Replacement job `2108402` remains `RUNNING` on `lyris0049`, elapsed about `00:09:51`.
- The driver log is still in environment/native dependency setup after `PARD2_OFFICIAL_PATCH_CHECKS`; it has not reached `SETUP COMPLETE`, model load, or GRPO steps yet.
- Ray 2.54.0 is active on the Ray head, and no repeated Ray version mismatch, `MasterConfig` attribute error, Hydra/config error, import error, runtime error, CUDA/OOM, or traceback is visible in the scanned logs.

20:44 CEST masterconfigfix poll:

- Replacement job `2108402` remains `RUNNING` on `lyris0049`, elapsed about `00:12:33`.
- The driver has built `deep-ep` and `transformer-engine-torch`, but is still before `SETUP COMPLETE`, model load, or GRPO steps.
- No repeated Ray version mismatch, `MasterConfig` attribute error, Hydra/config error, import error, runtime/CUDA/OOM failure, or traceback is visible. Local `py_compile`, `bash -n`, and `git diff --check` passed again.

20:52 CEST masterconfigfix monitor:

- Replacement job `2108402` remains `RUNNING` on `lyris0049`, elapsed about `00:20:08`.
- A four-poll monitor still shows driver environment/native dependency setup, not GRPO execution. Built packages now include `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm`.
- There is still no `SETUP COMPLETE`, model load, GRPO step, repeated Ray version mismatch, `MasterConfig` attribute error, Hydra/config error, import error, runtime/CUDA/OOM failure, or traceback. `git diff --check` on the touched files passed.

21:00 CEST W&B fix submission:

- Replacement job `2108402` reached config/data setup and failed at `Logger(logger_config)` because `logger.wandb_enabled=true` caused `wandb.init` to require an API key on Lyris.
- Root cause: the shared online launcher hard-coded `logger.wandb_enabled=true`, which is not safe for non-interactive Lyris runs without W&B credentials.
- Patched `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` to expose `WANDB_ENABLED` and use it in the Hydra override, and patched `experiments/eagle3_online/submit_lyris_qwen8_pard2_official_online_20260612.sh` to pass `WANDB_ENABLED=false`.
- Validation passed before resubmission: local `bash -n`, local `py_compile`, local `git diff --check`, Lyris dry-run/preflight, vLLM draft-refit target-projection tests, and `PARD2_OFFICIAL_PATCH_CHECKS`. The exact dry-run command contained `logger.wandb_enabled=false`.
- Submitted replacement job `2108503` with run id `20260612_qwen8_pard2_official_online_lyris_wandbfix`, account `coreai_dlalgo_llm`. Initial state: `RUNNING` on `lyris0126`; Ray 2.54.0 started and the driver is in venv setup.

21:28 CEST TIS fix submission:

- Replacement job `2108503` got past W&B/logger setup, proving `logger.wandb_enabled=false` was applied, then failed when `ClippedPGLossFn` asserted on `loss_fn.truncated_importance_sampling_type='tis'` with `loss_fn.truncated_importance_sampling_ratio=None`.
- Root cause: the Lyris base config carried an inconsistent TIS setting for the overlaid newer loss implementation. I did not invent a ratio because that would change the experiment objective; the reviewed fix preserves the previous no-truncation behavior.
- Patched `experiments/eagle3_online/submit_lyris_qwen8_pard2_official_online_20260612.sh` to add `loss_fn.truncated_importance_sampling_type=null` and `loss_fn.truncated_importance_sampling_ratio=null` to `ONLINE_EXTRA_OVERRIDES`.
- Validation passed before resubmission: local `bash -n`, local `py_compile`, local `git diff --check`, Lyris dry-run/preflight, vLLM draft-refit target-projection tests, and `PARD2_OFFICIAL_PATCH_CHECKS`. The exact dry-run command contained both `logger.wandb_enabled=false` and the two TIS-null overrides.
- Submitted replacement job `2108658` with run id `20260612_qwen8_pard2_official_online_lyris_tisfix`, account `coreai_dlalgo_llm`. Initial poll: `RUNNING` on `lyris0213`, source checks passed, and the driver remains in dependency/environment setup before model setup.

21:39 CEST TIS retry poll:

- Replacement job `2108658` remains `RUNNING` on `lyris0213`, elapsed about `00:14:42`.
- It is still in dependency/native setup before `SETUP COMPLETE`; the latest scanned tail shows `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` built after `PARD2_OFFICIAL_PATCH_CHECKS`.
- No repeated W&B error, TIS assertion, import/config failure, CUDA/OOM, runtime error, or traceback is visible yet.

21:43 CEST TIS retry process check:

- Replacement job `2108658` remains `RUNNING` on `lyris0213`, elapsed about `00:18:31`.
- The launched driver command includes the reviewed overrides: `logger.wandb_enabled=false`, `loss_fn.truncated_importance_sampling_type=null`, and `loss_fn.truncated_importance_sampling_ratio=null`.
- A compute-node process scan shows active `nvcc`/`ptxas` compilation for `causal-conv1d` and `mamba-ssm`; this confirms the job is still making dependency-build progress rather than hanging in Nemo-RL setup.
- No new patch or resubmission was made because no actionable failure is present.

21:53 CEST RayVirtualCluster portfix submission:

- Replacement job `2108658` reached config/data/cluster setup and proved the W&B/TIS fixes were effective: the loaded config had `wandb_enabled=False`, `truncated_importance_sampling_type=None`, and `truncated_importance_sampling_ratio=None`.
- It then failed with `TypeError: RayVirtualCluster.__init__() got an unexpected keyword argument 'port_range_low'`.
- Root cause: the overlaid newer `grpo.py` passes `port_range_low` / `port_range_high`, while the Lyris `RayVirtualCluster` class has the older constructor signature.
- Patched `.tmp_remote_current_oci/nemo_rl/algorithms/grpo.py` with `_create_ray_virtual_cluster(**kwargs)`, which inspects `RayVirtualCluster` and drops `port_range_low` / `port_range_high` only when the runtime class does not support them.
- Validation passed before resubmission: local `py_compile`, Lyris dry-run/preflight, remote compile, vLLM draft-refit target-projection tests, and `PARD2_OFFICIAL_PATCH_CHECKS`.
- Submitted replacement job `2108818` with run id `20260612_qwen8_pard2_official_online_lyris_portfix`, account `coreai_dlalgo_llm`. Initial poll: `RUNNING` on `lyris0161`; staged `grpo.py` contains the helper and the three GRPO cluster call sites use it.

22:01 CEST portfix poll:

- Job `2108818` remains `RUNNING` on `lyris0161`, elapsed about `00:07:45` at the poll.
- The driver log has passed `PARD2_OFFICIAL_PATCH_CHECKS`, is reusing the official PARD-2 vLLM site, installed the driver environment, and is still in native dependency setup.
- No repeated W&B, TIS, Ray-version, `RayVirtualCluster` port-range, import/config, CUDA/OOM, runtime, model-load, or GRPO-step error is visible in the scanned tail.

22:05 CEST portfix poll / report refresh:

- Refreshed the MATH500, SWE-Bench long-OSL, and Qwen235B result artifacts, then regenerated the combined expected-performance PNG/Markdown/HTML/raw CSV bundle.
- Row counts are stable: `16` MATH500 final rows, `65` SWE-Bench long-OSL final rows, `46` Qwen235B final rows, and `187` combined raw data rows.
- Job `2108818` remains `RUNNING` on `lyris0161`, elapsed about `00:11:59`; the latest scanned driver log shows `deep-ep` and `transformer-engine-torch` built after `PARD2_OFFICIAL_PATCH_CHECKS`.
- It has not reached `SETUP COMPLETE`, model load, or GRPO steps yet, and no repeated W&B, TIS, Ray-version, `RayVirtualCluster`, import/config, CUDA/OOM, runtime, model-load, or GRPO-step error is visible.

22:08 CEST portfix monitor:

- A three-poll monitor showed `2108818` still `RUNNING` on `lyris0161` through `00:14:32` elapsed.
- The driver log remains in native dependency setup with `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` built.
- A compute-node process scan showed active `nvcc`/`cicc`/`ptxas` work for `mamba-ssm` and `causal-conv1d`, so the job is compile-bound rather than idle.
- It has not reached `SETUP COMPLETE`, model load, or GRPO steps yet, and no new failure signature is visible. No replacement job was submitted.

22:13 CEST portfix monitor:

- Job `2108818` remains `RUNNING` on `lyris0161` through `00:19:24` elapsed.
- The driver venv is `9.0G`; benchmark row counts are unchanged at `16` MATH500 final rows, `65` SWE-Bench long-OSL final rows, `46` Qwen235B final rows, and `187` combined raw data rows.
- A process scan still shows active CUDA compiler work for selective-scan / `mamba-ssm` plus `causal-conv1d`; the job remains compile-bound rather than idle.
- It has not reached `SETUP COMPLETE`, model load, or GRPO steps, and no repeated W&B, TIS, Ray-version, `RayVirtualCluster`, import/config, CUDA/OOM, runtime, model-load, or GRPO-step error is visible. No replacement job was submitted.

22:22 CEST vLLM env-propagation fix submission:

- Job `2108818` progressed beyond the port-range issue: it connected Ray, loaded the reviewed config, initialized data/compute clusters, created actor environments, and initialized vLLM workers.
- It then failed in `_patch_vllm_init_workers_ray()` with `RuntimeError: Could not patch vLLM ADDITIONAL_ENV_VARS for SpecDec runtime env propagation.`
- Root cause: the official PARD-2 vLLM site used on Lyris does not have the older `ADDITIONAL_ENV_VARS` constant; its Ray worker env allowlist is driven by `vllm/ray/ray_env.py` with `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` and `get_env_vars_to_copy`.
- Patched `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_worker.py` to merge the SpecDec runtime env names into `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` before vLLM worker startup. The previous `ADDITIONAL_ENV_VARS` patch path is still used when present, and the code still raises if a requested SpecDec run has neither supported propagation mechanism.
- Validation passed before resubmission: local `py_compile`, local `git diff --check`, Lyris dry-run/preflight, remote compile, vLLM draft-refit target-projection tests, and all `PARD2_OFFICIAL_PATCH_CHECKS`. The staged remote source contains the `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` fallback and retained strict failure path.
- Submitted replacement job `2108947` with run id `20260612_qwen8_pard2_official_online_lyris_envpropfix`, account `coreai_dlalgo_llm`. Initial poll: `RUNNING` on `lyris0018`, in driver environment setup after source checks.

22:27 CEST envpropfix poll:

- Job `2108947` remains `RUNNING` on `lyris0018`, elapsed about `00:03:52` at the status poll.
- It has passed `PARD2_OFFICIAL_PATCH_CHECKS`, the official PARD-2 vLLM source-site check, and the main Python package install.
- The compute node is actively compiling native extensions (`mamba-ssm`, `causal-conv1d`, `deep-ep`, `nv-grouped-gemm`, `transformer-engine`), and the job-local Ray 2.54.0 process is up.
- It has not reached `SETUP COMPLETE`, model load, vLLM initialization, or GRPO steps yet. No repeated W&B, TIS, Ray-version, `RayVirtualCluster`, import/config, CUDA/OOM, runtime, model-load, or env-propagation error is visible.

22:32 CEST envpropfix poll:

- Job `2108947` remains `RUNNING` on `lyris0018`, elapsed about `00:09:33`.
- The filtered driver log now shows `deep-ep` built after the official PARD-2 source checks.
- It remains before `SETUP COMPLETE`, model load, vLLM initialization, or GRPO steps. No new traceback, W&B/TIS/Ray/`RayVirtualCluster` regression, import/config failure, CUDA/OOM signal, runtime/model-load failure, or env-propagation error is visible.
- No replacement job was submitted because the active job is still making build progress and has no actionable failure.

22:38 CEST envpropfix process check:

- Job `2108947` remains `RUNNING` on `lyris0018`.
- The driver log now shows `deep-ep`, `transformer-engine-torch`, and `nv-grouped-gemm` built.
- A compute-node process scan shows active CUDA compiler work for `mamba-ssm` selective-scan kernels and `causal-conv1d`; the job-local Ray 2.54.0 head/raylet are up.
- The launched driver command contains the reviewed overrides: `logger.wandb_enabled=false`, TIS null overrides, `method=pard2`, `num_speculative_tokens=1`, and `parallel_drafting=true`.
- The job is still before `SETUP COMPLETE`, model load, vLLM initialization, and GRPO steps, with no new failure signature.

22:49 CEST ViT backend patch skip submission:

- Job `2108947` progressed through the previous env-propagation boundary: driver native builds completed, Ray connected, config/data/compute setup succeeded, the actor venv was created, and `vllm_policy` workers initialized `4/4`.
- The previous `ADDITIONAL_ENV_VARS` runtime-env propagation failure did not recur.
- It then failed in `_patch_vllm_vit_flash_attn_backend()` because the official PARD-2 vLLM source site does not contain the older `vllm/attention/layer.py` path.
- Root cause: the ViT backend patch is a narrow optional workaround for older vLLM vision attention backend selection. It should not be fatal for this Qwen text-only official PARD-2 vLLM layout.
- Patched `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_worker.py` so the ViT workaround logs and skips when `vllm/attention/layer.py` is absent, while preserving the existing patch behavior when the file exists.
- Validation passed before resubmission: local `py_compile`, local `bash -n`, local `git diff --check`, Lyris dry-run/preflight, vLLM draft-refit `target_proj` tests, and all `PARD2_OFFICIAL_PATCH_CHECKS`. The staged remote source contains the new skip log.
- Submitted replacement job `2109062` with run id `20260612_qwen8_pard2_official_online_lyris_vitpatchfix`, account `coreai_dlalgo_llm`. Initial poll: `RUNNING` on `lyris0149`.

22:56 CEST vitpatchfix poll:

- Job `2109062` remains `RUNNING` on `lyris0149`, elapsed about `00:06:59`.
- It has passed `PARD2_OFFICIAL_PATCH_CHECKS` and the official PARD-2 vLLM site check.
- It is still in driver/native dependency setup. A compute-node process scan shows active CUDA compiler work for `mamba-ssm`, `causal-conv1d`, `nv-grouped-gemm`, and `transformer-engine`.
- The job-local Ray process is up, and the launched command contains the reviewed W&B-disabled, TIS-null, and PARD-2 overrides.
- It has not reached Nemo config load, vLLM actor initialization, or GRPO steps yet. No new failure signature is visible, so no replacement job was submitted.
