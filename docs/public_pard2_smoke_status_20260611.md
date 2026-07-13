# Official PARD-2 Smoke Status - 2026-06-11

## 2026-06-12 Refresh

The 50-step Qwen8 public PARD-2 validation jobs are now complete:

- `3271789` static-equivalent initial-refit run completed with exit code `0:0`.
- `3271807` interval-5 online run completed with exit code `0:0`.

Matched steps `2-50` show interval-5 online PARD-2 stays close to the
static-equivalent run but is not yet an end-to-end speedup:

| Variant | Steps | Refits | Token acceptance | Step time | Generation time | Gen worker tok/s/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static-equivalent | 49 | 0 | `47.54%` | `32.68s` | `17.20s` | `140.70` |
| online interval-5 | 49 | 9 | `47.16%` | `32.93s` | `16.91s` | `142.77` |

Current conclusion: the PARD-2 online path is functional when
`policy.draft.initial_refit=true` and `pard_token=151670` are set. The older
`0%` Qwen8 canary reflects a bad no-initial-refit/dummy-draft path, not the
current working configuration. See
`docs/qwen8_pard2_online_initialrefit_diagnosis_20260612.md` for the compact
diagnosis.

## Source Boundary

AMD-AGI/PARD now publishes official PARD-2 assets. This is different from the
earlier local `PARD-2-style` / CAT-weighted PARD checkpoints used in prior
Qwen3-30B-A3B and Qwen3-235B experiments.

Public checkpoints under test:

- `amd/PARD2-Qwen3-8B`
- `amd/PARD2-Qwen3-14B`

Official PARD-2 format note:

- The AMD README marks PARD-2 weights/training/inference code as released on
  2026-06-10.
- The public PARD-2 checkpoints include target-aligned metadata. Their config
  uses fields such as `pard2`, `spd_type=pard2`, `pard2_target_layers`,
  `pard2_target_dim`, `pard2_scale`, and `pard2_proj_bias`.
- AMD's `pard2_infer.py` wraps the draft model with a target-feature projection
  (`target_proj`) loaded from `warp_model.bin`; AMD's `pard2_train.py` extracts
  target hidden states from selected target layers and feeds them as
  `target_feat`.
- However, the cached public HF snapshots currently contain only
  `config.json` and `model.safetensors`. The safetensors headers for both
  `amd/PARD2-Qwen3-8B` and `amd/PARD2-Qwen3-14B` have standard Qwen keys and
  no `target_proj`, `pard`, or `warp` weight keys. So the current vLLM/NeMo-RL
  static path is token-only PARD parallel drafting using `pard_token`, not AMD's
  reference target-feature wrapper.
- Therefore official PARD-2 is not equivalent to the earlier local
  `PARD-2-style` CAT/k-slot causal-LM path.

Initial compatibility target:

- `Qwen/Qwen3-8B`

Reason: the public checkpoint names are target-aligned for Qwen3-8B/14B. Do not
claim compatibility with Qwen3-30B-A3B, Qwen3-32B, or Qwen3-235B until load and
acceptance tests pass.

## vLLM Standalone Jobs

Prompt set:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/data/swebench_lite_prompts_64.jsonl`

Shared setup:

- target: `Qwen/Qwen3-8B`
- ISL/OSL: `4096/1024`
- batch sizes: `8 16`
- TP/PP: `1/1`
- allocation: `--gres=gpu:4` because `--gres=gpu:1` was rejected by QOSMinGRES

| Job | Variant | Draft | K | Status |
| --- | --- | --- | ---: | --- |
| `3267328` | baseline | none | 0 | completed JSON |
| `3267344` | official PARD-2 | `amd/PARD2-Qwen3-8B` | 5 | completed JSON |

Results:

| Batch | Baseline tok/s/GPU | PARD-2 tok/s/GPU | Speedup | PARD-2 acceptance | Accepted/draft |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 245.434 | 235.624 | 0.960 | 28.435% | 1.422 |
| 16 | 572.852 | 457.399 | 0.798 | 28.728% | 1.436 |

Interpretation: official PARD-2 loads and runs in vLLM v0.20.0 with
`method=draft_model` and `parallel_drafting=true`, but this short OSL1024
SWE-Bench Lite smoke is not enough to overcome draft/verification overhead.
Run an OSL16K small-slice test before drawing a performance conclusion.

Output JSONs:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen3_8b_swebench_lite_baseline_offset0_n64_isl4096_osl1024_bs8_16_20260611_publicpard2/breakdown.json`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen3_8b_swebench_lite_pard2_offset0_n64_isl4096_osl1024_bs8_16_k5_20260611_publicpard2/breakdown.json`

## vLLM Long-Output Follow-Up

Prompt set:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/data/swebench_verified_prompts_all.jsonl`

Shared setup:

- target: `Qwen/Qwen3-8B`
- ISL/OSL: `4096/16384`
- batch sizes: `1 2 4 8`
- TP/PP: `1/1`
- allocation: `--gres=gpu:4`

| Job | Variant | Draft | K | Status |
| --- | --- | --- | ---: | --- |
| `3267466` | baseline | none | 0 | running under `coreai_dlalgo_llm` |
| `3267467` | official PARD-2 | `amd/PARD2-Qwen3-8B` | 5 | running under `coreai_dlalgo_llm` |

Output JSONs:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen3_8b_swebench_verified_longosl16384_baseline_offset0_bs1_2_4_8_20260611_publicpard2/breakdown.json`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen3_8b_swebench_verified_longosl16384_pard2_offset0_bs1_2_4_8_k5_20260611_publicpard2/breakdown.json`

## vLLM Qwen3-14B Official PARD-2 Smoke

Prompt set:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/data/swebench_lite_prompts_64.jsonl`

Shared setup:

- target: `Qwen/Qwen3-14B`
- draft: `amd/PARD2-Qwen3-14B`
- ISL/OSL: `4096/1024`
- batch sizes: `4 8`
- TP/PP: `1/1`
- allocation: `--gres=gpu:4`

| Job | Variant | Draft | K | Status |
| --- | --- | --- | ---: | --- |
| `3268813` | baseline | none | 0 | completed JSON |
| `3268814` | official PARD-2 | `amd/PARD2-Qwen3-14B` | 5 | completed JSON |

Results:

| Batch | Baseline tok/s/GPU | PARD-2 tok/s/GPU | Speedup | PARD-2 acceptance | Mean accept length |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 137.757 | 108.761 | 0.790 | 18.022% | 1.901 |
| 8 | 238.292 | 187.807 | 0.788 | 18.468% | 1.923 |

Interpretation: official PARD-2 Qwen3-14B loads and runs, but this short
SWE-Bench Lite OSL1024 setup is slower than baseline.

Output JSONs:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen3_14b_swebench_lite_baseline_offset0_n64_isl4096_osl1024_bs4_8_20260611_publicpard2/breakdown.json`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen3_14b_swebench_lite_pard2_offset0_n64_isl4096_osl1024_bs4_8_k5_20260611_publicpard2/breakdown.json`

## NeMo-RL Jobs

Shared setup:

- target: `Qwen/Qwen3-8B`
- drafter: `amd/PARD2-Qwen3-8B`
- K: `5`
- max steps: `2`
- prompts/generations/GBS: `2/2/4`
- output length: `128`, min tokens `64`

| Job | Variant | policy.draft | Purpose | Status |
| --- | --- | --- | --- | --- |
| `3267356` | static official PARD-2 | disabled | first attempt | cancelled too early during normal Ray-head startup polling |
| `3267357` | online official PARD-2 | enabled, `type=pard2` | first attempt | cancelled too early during normal Ray-head startup polling |
| `3267402` | static official PARD-2 | disabled | test vLLM draft-model path inside NeMo-RL | completed |
| `3267403` | online official PARD-2 | enabled, `type=pard2` | test whether current Megatron PARD trainer can consume official HF PARD-2 format | failed at step 1 on missing train-time `pard_token` metadata |
| `3268034` | online official PARD-2 | enabled, `type=pard2` | retry after `pard_token` metadata/fallback fix | completed; trainer runs, but refit/semantics not validated |
| `3268826` | static official PARD-2 14B | disabled | test `amd/PARD2-Qwen3-14B` through NeMo-RL vLLM generation path | completed |
| `3269147` | online PARD-2 refit diagnostic | enabled, `type=pard2`, `allow_generic_pard2_fallback=true` | log exported draft HF keys and vLLM draft-load keys/results | completed; refit/load structurally works, acceptance still 0 |
| `3269327` | online PARD-2 static-before-refit gate | enabled, `type=pard2`, train/refit start step `2` | step 1 should use original HF drafter in vLLM, step 2 should refit Megatron-exported drafter | completed; acceptance already 0 before scheduled train/refit |
| `3269510` | online PARD-2 static-before-refit gate after initial-refit metadata fix | enabled, `type=pard2`, train/refit start step `2` | verify no initial draft export/refit occurs before step 1 | completed; no initial draft export, but acceptance still 0 because vLLM online engine dummy-loads the draft model |
| `3269702` | online PARD-2 roundtrip diagnostic | enabled, `type=pard2`, `initial_refit=true`, train/refit start step `99`, roundtrip logging enabled | compare Megatron-exported PARD2 tensors with original HF tensors before training | completed; representative tensors match HF exactly and vLLM draft-load succeeds |
| `3269985` | online PARD-2 static-equivalent initial-refit validation | enabled, `type=pard2`, `initial_refit=true`, train/refit start step `99` | prove online worker plus initial refit restores nonzero PARD-2 acceptance before any online training update | completed; acceptance restored to static level |
| `3270216` | online PARD-2 generic k-slot update after initial-refit | enabled, `type=pard2`, `initial_refit=true`, train/refit start step `1` | compare actual generic online drafter update against static-equivalent initial-refit result | completed |
| `3270522` | async-GRPO initial-refit ordering smoke | enabled, `type=pard2`, `initial_refit=true`, train/refit start step `99`, async GRPO enabled | verify async collector does not generate before initial draft refit | failed during actor venv rebuild, not model/runtime |
| `3270570` | 20-step long-output static-equivalent comparison | enabled, `type=pard2`, `initial_refit=true`, train/refit start step `99`, OSL1024, GBS8 | compare longer-horizon acceptance/perf without online drafter updates | failed during actor venv rebuild, not model/runtime |
| `3270571` | 20-step long-output online-update comparison | enabled, `type=pard2`, `initial_refit=true`, train/refit start step `1`, OSL1024, GBS8 | compare longer-horizon acceptance/perf with generic k-slot/CAT update every step | failed during actor venv rebuild, not model/runtime |
| `3270832` | async-GRPO initial-refit ordering smoke retry | same as `3270522`, using `/opt/ray_venvs` and rebuild-off settings from successful jobs | verify async collector does not generate before initial draft refit | failed on missing `vllm_cfg.async_engine=true` assertion |
| `3270833` | 20-step long-output static-equivalent comparison retry | same as `3270570`, using `/opt/ray_venvs` and rebuild-off settings from successful jobs | longer-horizon acceptance/perf without online drafter updates | completed; final 20-step comparison baseline |
| `3270834` | 20-step long-output online-update comparison retry | same as `3270571`, using `/opt/ray_venvs` and rebuild-off settings from successful jobs | longer-horizon acceptance/perf with generic k-slot/CAT update every step | completed; every-step online update did not beat static-equivalent |
| `3271041` | async-GRPO initial-refit ordering smoke retry2 | same as `3270832`, but with `policy.generation.vllm_cfg.async_engine=true` | determine whether AsyncGRPO plus draft-model PARD-2 can pass NeMo-RL async collector setup | failed on colocated async topology assertion |
| `3271410` | async-GRPO non-colocated smoke | non-colocated generation, `vllm_cfg.async_engine=true`, `initial_refit=true`, train/refit delayed to step 99 | verify PARD-2 can pass NeMo-RL async collector setup with non-colocated inference | completed; functional async compatibility gate passed, no acceptance metrics parsed |
| `3271494` | 20-step long-output online interval-5 retry | `initial_refit=true`, `pard_token=151670`, generic k-slot/CAT train/refit interval 5 | compare interval-based online update overhead and acceptance vs `3270833`/`3270834` | completed; interval-5 is the best point in this 20-step slice |
| `3271495` | 20-step long-output online interval-10 retry | `initial_refit=true`, `pard_token=151670`, generic k-slot/CAT train/refit interval 10 | compare lower-frequency online update overhead and acceptance vs `3270833`/`3270834` | completed; lower overhead but worse acceptance |
| `3271789` | 50-step long-output static-equivalent retry | `initial_refit=true`, train/refit delayed to step 99, OSL1024, GBS8 | validate the 20-step interval-5 signal against a longer static-equivalent baseline | completed with exit code `0:0`; final matched metrics through step 50 |
| `3271807` | 50-step long-output online interval-5 retry | `initial_refit=true`, `pard_token=151670`, generic k-slot/CAT train/refit interval 5, OSL1024, GBS8 | validate whether interval-5 acceptance/performance gain persists over 50 steps | completed with exit code `0:0`; final matched metrics through step 50 |

Static NeMo-RL evidence from `3267402`:

- reached step 2/2 and completed
- vLLM logged official PARD-2 spec-dec metrics
- representative avg draft acceptance rate: about 31-32% at K=5
- parsed weighted token acceptance: `31.92%`

Online failure and fix:

- `3267403` failed with `ValueError: PARD k-slot online draft training requires the draft model config to define pard_token or draft_pard_token.`
- The official cached HF config for `amd/PARD2-Qwen3-8B` does contain `pard_token: 151670`, so the issue was metadata propagation into the Megatron online drafter path, not vLLM inference format.
- Patched `nemo_rl/models/megatron/draft/utils.py` to read `policy.draft.pard_token` and `AutoConfig.from_pretrained(...).pard_token` before building the draft model.
- Patched `nemo_rl/models/megatron/draft/pard.py` to use the same `AutoConfig` fallback once and cache the resolved token id back onto draft metadata.
- Remote backup files were left as `utils.py.pre_public_pard2_tokenfix_20260611` and `pard.py.pre_public_pard2_tokenfix_20260611`.

Online retry evidence from `3268034`:

- reached step 2/2 and completed
- Step 1: `Draft Training Enabled=True`, `Draft Refit This Step=True`, `Draft Loss=2.2016`, total step time `33.18s`
- Step 2: `Draft Training Enabled=True`, `Draft Refit This Step=True`, `Draft Loss=2.1748`, total step time `19.25s`
- parsed weighted token acceptance after refit: `0.00%`
- vLLM metrics showed accepted `0` / drafted `2540` tokens for the visible post-refit metric window
- the driver log repeatedly printed `No mapping found for megatron_param: draft_model.module...`

Interpretation: the official PARD-2 online path now executes without crashing,
but it should not yet be treated as a valid online PARD-2 training result. The
current `type=pard2` path still routes through the generic PARD k-slot/CAT
trainer and does not explicitly consume the official PARD-2 target-aligned
config fields such as `pard2_target_dim` and `pard2_target_layers`.

Guardrail patch applied after this observation:

- `nemo_rl/models/megatron/draft/utils.py` now detects official PARD-2 HF
  configs and raises a clear `NotImplementedError` for online PARD training
  unless `++policy.draft.allow_generic_pard2_fallback=true` is set.
- `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` now records
  `++policy.draft.allow_generic_pard2_fallback=${POLICY_DRAFT_ALLOW_GENERIC_PARD2_FALLBACK}`
  in the submitted command.
- This keeps static vLLM/NeMo-RL PARD-2 evaluation enabled while preventing the
  generic online trainer from being misreported as official online PARD-2.
- Remote syntax checks passed for `nemo_rl/models/megatron/draft/utils.py`,
  `nemo_rl/models/megatron/draft/pard.py`, and the submit helper.

Refit diagnostic patch applied after confirming the public HF snapshots are
token-only:

- `nemo_rl/models/policy/workers/megatron_policy_worker.py` now temporarily
  hides the attached `draft_model` child module while building target-policy
  refit conversion tasks and while iterating target-policy HF exports. Draft
  weights are still exported separately under the `draft.` prefix. This removes
  the `No mapping found for megatron_param: draft_model.module...` traversal
  from the target export path.
- `nemo_rl/models/megatron/draft/utils.py` now logs PARD draft export count,
  first key names, shapes, and dtypes when `NRL_DEBUG_DRAFT_REFIT=true`.
- `nemo_rl/models/generation/vllm/vllm_backend.py` now logs the draft weights
  passed into the vLLM drafter and the return value from `load_weights()` when
  `NRL_DEBUG_DRAFT_REFIT=true`.
- `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` now forwards
  `NRL_DEBUG_DRAFT_REFIT` into the Ray driver command.
- Diagnostic smoke `3269147` uses the same Qwen3-8B/PARD2 setup as `3268034`,
  with `++policy.draft.allow_generic_pard2_fallback=true`, to determine whether
  the zero-acceptance online result is caused by draft export/load key mismatch
  or by the generic k-slot update changing the drafter into a bad distribution.

Refit diagnostic result from `3269147`:

- reached step 2/2 and completed with batch job exit code `0:0`
- target-policy export cleanup worked; the previous
  `No mapping found for megatron_param: draft_model.module...` warning no
  longer appears in the driver log
- PARD draft export produced `311` HF-style Qwen keys, including
  `model.embed_tokens.weight`, `lm_head.weight`, `model.norm.weight`, and
  normal layer projection weights
- vLLM draft load received the same `311` draft keys and
  `draft_model.load_weights()` returned loaded parameters, including vLLM's
  fused `qkv_proj` names
- visible vLLM post-refit metric window still had accepted/drafted tokens
  `0 / 2540`, weighted token acceptance `0.00%`, and mean acceptance length
  `1.00`
- draft loss was finite (`2.2016` then `2.1733`), so the trainer phase ran
  without crashing

Interpretation: the zero-acceptance online PARD-2 result is not explained by a
simple draft refit key-splitting or vLLM load failure. The next isolation step is
a round-trip tensor/acceptance check: compare the Megatron-imported/exported
drafter against the original HF PARD2 checkpoint before any online update, or
run a static-before-refit smoke so step 1 uses the original vLLM-loaded HF
drafter and step 2 refits the Megatron-exported drafter.

Static-before-refit gate result from `3269327`:

- completed with batch job exit code `0:0`
- command had `++policy.draft.train_start_step=2` and
  `++policy.draft.refit_start_step=2`
- step 1 summary showed `Draft Training Enabled=False` and
  `Draft Refit This Step=False`
- nevertheless, `[draft-refit] exported PARD draft weights count=311` appeared
  before and during step 1, so the online worker path still exports the
  Megatron-side drafter during initialization/prepare
- the first visible spec-dec metric window, attributed to the generation before
  step 2 training/refit, had accepted/drafted tokens `0 / 2540`, weighted token
  acceptance `0.00%`, and mean acceptance length `1.00`
- step 2 then trained/refit as requested and had finite draft loss `2.2729`, but
  acceptance remained `0.00%`

Interpretation: the official PARD-2 static path is valid, but enabling the
online drafter path does not preserve the original HF PARD2 drafter even before
the scheduled online training/refit step. The next fix should stop the initial
online-mode draft refit/export, or prove via tensor checksum that the
HF->Megatron->HF drafter round trip exactly preserves the official checkpoint.

Static-before-refit gate after initial metadata fix from `3269510`:

- completed with batch job exit code `0:0`
- command again had `++policy.draft.train_start_step=2` and
  `++policy.draft.refit_start_step=2`
- the setup-time `policy.prepare_refit_info()` path was patched so initial
  draft refit metadata is included only when an initial draft refit is actually
  pending
- step 1 summary showed `Draft Training Enabled=False` and
  `Draft Refit This Step=False`
- no `[draft-refit]` export appeared before step 1, so the initial draft export
  bug from `3269327` is fixed
- the first visible spec-dec metric window still had accepted/drafted tokens
  `0 / 2540`, weighted token acceptance `0.00%`, and mean acceptance length
  `1.00`
- step 2 trained/refit as requested with finite draft loss `2.2729`, but
  acceptance remained `0.00%`

Root cause narrowed by comparing vLLM engine logs:

- static NeMo-RL jobs `3267402` and `3268826` initialize vLLM with
  `load_format=auto`, and the official HF PARD-2 drafter gets nonzero
  acceptance
- online NeMo-RL job `3269510` initializes vLLM with `load_format=dummy`
- in vLLM v0.20.0, `gpu_model_runner.load_model(load_dummy_weights=True)`
  mutates the shared load config to `load_format="dummy"` before calling
  `self.drafter.load_model(self.model)`
- `DraftModelProposer._get_model()` builds the draft model through
  `get_model(vllm_config=draft_vllm_config, prefix="draft_model")` and does not
  pass a draft-specific `load_config`; therefore the draft model inherits the
  target engine's dummy-load mode in NeMo-RL online/refit runs

Interpretation: for official static PARD-2, vLLM/NeMo-RL works. For online
PARD-2, the immediate blocker is not the scheduled training interval anymore;
the vLLM worker starts with a dummy-loaded drafter unless NeMo-RL performs a
correct initial draft refit or vLLM is patched so `draft_model` loads from HF
even when the target policy uses dummy weights for later refit.

Roundtrip diagnostic result from `3269702`:

- completed with batch job exit code `0:0`
- command used `++policy.draft.initial_refit=true`,
  `++policy.draft.train_start_step=99`, and
  `++policy.draft.refit_start_step=99`, so no online drafter training ran
- PARD draft export produced `311` HF-style Qwen keys and vLLM draft-load
  accepted them
- representative HF-vs-export comparisons all matched exactly:
  `model.norm.weight`, `model.layers.0.input_layernorm.weight`,
  `model.layers.0.self_attn.q_proj.weight`, and
  `model.layers.0.mlp.gate_proj.weight` all had `max_abs_diff=0.0` and
  `allclose=True`
- the 1-step diagnostic completed too quickly for vLLM to emit a full
  `SpecDecoding metrics` line, so it validates weight roundtrip/load, not
  acceptance

Interpretation: the HF -> Megatron -> HF roundtrip is not corrupting the
official PARD-2 drafter, at least for representative tensors. Therefore the
online path should perform an initial draft refit for PARD/PARD-2 when vLLM
target loading is dummy-based. A follow-up 2-step validation job `3269985` was
submitted to confirm nonzero acceptance before enabling any online update.

Code update after `3269702`:

- `nemo_rl/algorithms/grpo.py` now defaults online draft initial refit to
  `true` for all online draft types. The explicit overrides
  `++policy.draft.initial_refit=false` and
  `++policy.draft.initial_draft_refit=false` still disable it for experiments.
- This is required for PARD/PARD-2 because NeMo-RL online vLLM workers use
  target dummy-load for later policy refit, and vLLM's `draft_model` proposer
  otherwise dummy-loads the drafter too.
- Remote checks passed:
  `python3 -m py_compile nemo_rl/algorithms/grpo.py` and
  `git diff --check -- nemo_rl/algorithms/grpo.py`.

Static-equivalent initial-refit validation from `3269985`:

- completed 2/2 steps with batch job exit code `0:0`
- step-level online drafter training and scheduled refit stayed off:
  `Draft Training Enabled=False` and `Draft Refit This Step=False` for both
  steps
- the only draft refit was the initial PARD-2 refit before rollout; vLLM still
  initialized with `load_format=dummy`, then accepted the 311 refit draft
  weights
- parsed weighted accepted/drafted tokens: `319 / 1035`
- parsed weighted token acceptance: `30.82%`
- parsed mean acceptance length: `2.5425`
- representative vLLM metric window:
  `Accepted: 80`, `Drafted: 260`, avg draft acceptance `30.8%`, per-position
  acceptance `0.692, 0.346, 0.212, 0.173, 0.115`

Interpretation: online PARD-2 plumbing is now functionally valid when initial
draft refit is enabled. The previous 0% acceptance in online-mode runs came
from a dummy-loaded drafter or from post-training generic-k-slot updates, not
from vLLM PARD-2 static inference or HF/Megatron weight roundtrip. Official
online PARD-2 training is still not claimed, because AMD's target-feature
PARD-2 path is not implemented in NeMo-RL; current online training remains the
generic PARD/PARD2 k-slot/CAT fallback behind
`allow_generic_pard2_fallback=true`.

Generic online k-slot/CAT update after initial refit from `3270216`:

- completed 2/2 steps with batch job exit code `0:0`
- online drafter training and scheduled draft refit were enabled on both steps:
  `Draft Training Enabled=True` and `Draft Refit This Step=True`
- parsed weighted accepted/drafted tokens: `316 / 990`
- parsed weighted token acceptance: `31.92%`
- parsed mean acceptance length: `2.595`
- parsed mean draft loss: `2.19835`
- step 2 visible vLLM metric examples showed avg draft acceptance around
  `31-32%` after the generic online update

Interpretation: for this very small 2-step Qwen3-8B smoke, generic PARD/PARD2
k-slot/CAT online update after initial refit does not collapse official PARD-2
acceptance. This is a functional validation only; it is not yet evidence that
generic online training improves acceptance over a meaningful RL horizon or
implements AMD's official target-feature PARD-2 objective.

Consolidated local comparison artifact:

- `docs/public_pard2_nemorl_comparison_20260611.csv`
- `docs/public_pard2_job_status_20260611.csv`
- `docs/public_pard2_nemorl_steps_20260611.csv`
- `docs/public_pard2_nemorl_summary_20260611.csv`
- `docs/public_pard2_vllm_standalone_20260611.csv`

This CSV compares the completed public PARD-2 NeMo-RL cases side by side:
initial-refit missing (`0%` acceptance), initial-refit static-equivalent
(`30.82%`), generic online update after initial-refit (`31.92%`), and Qwen3-14B
static (`15.52%`).

Refresh helper:

- `scripts/refresh_public_pard2_results.py`
- The helper records tracked job state, reruns the remote NeMo-RL step/summary
  parser, fetches those CSVs locally, and extracts available vLLM standalone
  `breakdown.json` files. Pending standalone JSONs are reported as warnings
  instead of failing the whole refresh.

Parsed remote CSVs:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_official_static_online_steps_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_official_static_online_summary_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen14_public_pard2_static_steps_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen14_public_pard2_static_summary_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_refitdiag_steps_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_refitdiag_summary_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_static_before_refit_steps_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_static_before_refit_summary_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_static_before_refit_gatefix_steps_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_static_before_refit_gatefix_summary_20260611.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_roundtrip_diag_20260611_steps.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_roundtrip_diag_20260611_summary.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_initialrefit_static_equiv_20260611_steps.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_initialrefit_static_equiv_20260611_summary.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_online_after_initialrefit_20260611_steps.csv`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/docs/qwen8_public_pard2_online_after_initialrefit_20260611_summary.csv`

Static NeMo-RL evidence from `3268826`:

- reached step 2/2 and completed with batch job exit code `0:0`
- target `Qwen/Qwen3-14B`, drafter `amd/PARD2-Qwen3-14B`, K=5
- weighted accepted/drafted tokens: `222 / 1430`
- parsed weighted token acceptance: `15.52%`
- parsed mean acceptance length: `1.775`
- generation worker throughput mean: `44.60 tokens/sec/GPU`
- step 2 visible vLLM metric examples:
  - avg draft acceptance rate: `13.4%` on one worker metric window
  - avg draft acceptance rate: `16.3%` on repeated worker metric windows
  - per-position acceptance around `0.409, 0.158, 0.092, 0.066, 0.053`

Remote job file:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen8_public_pard2_nemorl_20260611_jobs.txt`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen8_public_pard2_nemorl_retry1_20260611_jobs.txt`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen8_public_pard2_nemorl_retry2_tokenfix_20260611_jobs.txt`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen14_public_pard2_nemorl_static_20260611_jobs.txt`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen8_public_pard2_nemorl_refitdiag_20260611_jobs.txt`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen8_public_pard2_static_before_refit_20260611_jobs.txt`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen8_public_pard2_roundtrip_diag_20260611_jobs.txt`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen8_public_pard2_initialrefit_static_equiv_20260611_jobs.txt`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen8_public_pard2_online_after_initialrefit_20260611_jobs.txt`

Note: early `srun test` failures during Ray-head startup are not sufficient to
declare failure. The previous local Qwen8 canary showed the same startup polling
pattern before `ray-driver.log` attached.

## Expected Interpretation

If vLLM standalone fails before generation, first suspect runtime support for
the new PARD-2 config rather than the benchmark harness.

If static NeMo-RL passes but online NeMo-RL fails, the likely gap is the
trainer-side format or HF->Megatron->HF draft round trip: current online code was
built around PARD/PARD2-style k-slot causal-LM training and does not yet support
the official target-aligned PARD-2 path from AMD's reference `pard2_infer.py`.

Before any online PARD-2 result is compared against static PARD-2, require one
of these gates to pass:

- Megatron-import/export tensor checksum matches the original HF PARD2
  checkpoint for representative tensors before any training update. This passed
  in `3269702`.
- A static-before-refit NeMo-RL smoke shows nonzero acceptance before refit and
  identifies the exact refit/update point where acceptance collapses. Initial
  static-equivalent acceptance passed in `3269985`; the remaining comparison is
  static-equivalent initial refit vs actual generic online training updates over
  longer steps.

Additional async-GRPO guardrail:

- The sync GRPO path now has a validated ordering: initial PARD/PARD-2 draft
  refit happens before the first accepted static-equivalent metric window.
- The async GRPO collector path still needs a separate ordering smoke. Code
  review found that the background `AsyncTrajectoryCollector` is started before
  the visible initial stale-generation refit block in `grpo.py`; unless the
  collector internally blocks, an async run could enqueue dummy-drafter
  trajectories before the initial PARD/PARD-2 refit completes.
- Async ordering smoke `3270522` has been submitted to validate this ordering.
- `3270522` failed before model initialization because the actor venv rebuild
  path hit a package install race (`setuptools` dist-info missing). Retry
  `3270832` uses the same cached `/opt/ray_venvs` settings as the successful
  sync jobs.
- `3270832` passed venv/model setup and exported initial PARD-2 draft weights
  before async collector setup, then failed with NeMo-RL's
  `Async GRPO requires vLLM backend with vllm_cfg.async_engine=True` assertion.
  Retry `3271041` enables `policy.generation.vllm_cfg.async_engine=true` to test
  whether the async collector can run despite vLLM warning that draft-model
  speculative decoding disables internal async scheduling.

20-step long-output final result from `3270833`/`3270834`:

- Local partial CSVs:
  - `docs/qwen8_public_pard2_longosl1024_step20_retry1_partial_steps.csv`
  - `docs/qwen8_public_pard2_longosl1024_step20_retry1_partial_summary_skipstep1.csv`
- Local final CSVs:
  - `docs/qwen8_public_pard2_longosl1024_step20_retry1_steps.csv`
  - `docs/qwen8_public_pard2_longosl1024_step20_retry1_summary_skipstep1.csv`
- Step>=2 final summary:
  - `3270833` initial-refit-only static-equivalent: completed steps `19/19`,
    weighted accepted/drafted `157410 / 317880`, token acceptance `49.52%`,
    mean acceptance length `3.5311`, mean total step time `32.39s`,
    generation worker throughput `146.68 tok/s/GPU`
  - `3270834` online train/refit every step: completed steps `19/19`,
    weighted accepted/drafted `156456 / 318280`, token acceptance `49.16%`,
    mean acceptance length `3.4741`, mean draft loss `1.9823`, mean total
    step time `34.84s`, generation worker throughput `145.58 tok/s/GPU`
- Full 20-step all-step summary in `docs/public_pard2_nemorl_summary_20260611.csv`
  is directionally the same: initial-refit static-equivalent token acceptance
  `48.94%`, online every-step `48.32%`.
- Interpretation: OSL1024 increases PARD-2 acceptance substantially relative to
  the OSL128 functional smoke, but the generic online k-slot/CAT update every
  step does not improve acceptance over the static-equivalent initial-refit
  baseline. It also increases step time by about `2.46s` on step>=2, mainly from
  draft training/refit overhead. This argues for interval-based training/refit
  or true official PARD-2 target-feature training before claiming an online
  drafter-training advantage.

Async-GRPO retry2 result from `3271041`:

- The run enabled `policy.generation.vllm_cfg.async_engine=true`, initialized
  vLLM async workers, imported the policy, imported the PARD-2 drafter, and
  exported the initial PARD-2 draft weights before entering async GRPO.
- It failed before training with:
  `AssertionError: Colocated inference is not supported for async GRPO. Please use non-colocated inference.`
- Interpretation: this is a NeMo-RL async configuration limitation, not a
  PARD-2 checkpoint or draft-refit failure. The next async smoke must use
  non-colocated inference to test whether PARD-2 draft-model speculation can run
  through the async collector path.

Interval follow-up result from `3271494`/`3271495`:

- `3271403` / `3271404`: first Qwen3-8B public PARD-2 interval submissions
  for interval 5/10. These were cancelled because the wrapper overwrote
  `ONLINE_EXTRA_OVERRIDES`, so `++policy.draft.initial_refit=true` was missing
  from the actual Hydra overrides. Without that initial draft refit, the run can
  start from a dummy-loaded drafter and is not comparable to the final
  static-equivalent baseline.
- `3271494`: retry for Qwen3-8B public PARD-2 OSL1024/GBS8/K5, generic online
  k-slot/CAT training with `train_interval=5`, `refit_interval=5`,
  `++policy.draft.initial_refit=true`, and `++policy.draft.pard_token=151670`,
  completed with batch exit code `0:0`.
- `3271495`: same as `3271494`, but `train_interval=10` and
  `refit_interval=10`, completed with batch exit code `0:0`.
- Local interval CSVs:
  - `docs/qwen8_public_pard2_interval_longosl1024_step20_retry1_steps.csv`
  - `docs/qwen8_public_pard2_interval_longosl1024_step20_retry1_summary_skipstep1.csv`
- Step>=2 summary against the same baseline/every-step comparison:
  - `3271494` interval-5: completed steps `19/19`, draft train/refit steps
    `3/3`, weighted accepted/drafted `164055 / 325810`, token acceptance
    `50.35%`, mean acceptance length `3.5728`, mean draft loss `2.0045`,
    mean total step time `32.25s`, generation worker throughput
    `148.63 tok/s/GPU`
  - `3271495` interval-10: completed steps `19/19`, draft train/refit steps
    `1/1`, weighted accepted/drafted `154125 / 318980`, token acceptance
    `48.32%`, mean acceptance length `3.4252`, mean draft loss `1.7692`,
    mean total step time `33.21s`, generation worker throughput
    `144.44 tok/s/GPU`
- Interpretation: interval-5 is the best observed online-training point in this
  slice. It slightly beats the initial-refit-only static-equivalent baseline
  (`50.35%` vs `49.52%` token acceptance, `32.25s` vs `32.39s` mean step time)
  and clearly beats every-step online training (`50.35%` vs `49.16%`, `32.25s`
  vs `34.84s`). Interval-10 cuts update frequency further but loses too much
  acceptance in this short run.
- `3271410`: Qwen3-8B public PARD-2 async GRPO smoke with non-colocated
  generation (`cluster.num_nodes=2`, generation resources `1x4 GPUs`),
  `vllm_cfg.async_engine=true`, initial draft refit enabled, and scheduled
  online draft updates delayed to step 99. Purpose: isolate whether PARD-2
  draft-model speculation can pass the async collector setup once the colocated
  inference assertion from `3271041` is removed.

Async non-colocated result from `3271410`:

- Completed with batch exit code `0:0`.
- Actual overrides included `grpo.async_grpo.enabled=true`,
  `policy.generation.vllm_cfg.async_engine=true`,
  `policy.generation.colocated.enabled=false`,
  generation resources `1x4 GPUs`, `++policy.draft.initial_refit=true`, and
  `++policy.draft.pard_token=151670`.
- The log reached `SETUP COMPLETE`, `Running async GRPO training`, and
  completed `Step 1/2` and `Step 2/2`.
- This proves the previous `3271041` blocker was the colocated async topology,
  not PARD-2 draft-model loading/refit. The vLLM warning still appears:
  draft-model speculative decoding disables vLLM internal async scheduling, but
  NeMo-RL async GRPO can run with non-colocated workers.
- The short async smoke did not emit parsed vLLM spec-dec acceptance metrics, so
  it is a functional async compatibility gate, not a performance result.

The interval result changes the online-training conclusion: every-step generic
online PARD/PARD-2 fallback is not useful here, but interval-5 shows a small
positive acceptance/performance signal that is worth validating on longer RL
runs and longer-output SWE-style prompts.

50-step validation follow-up:

- `3271789`: Qwen3-8B public PARD-2 static-equivalent, OSL1024/GBS8/K5, 50
  steps, initial refit enabled and online train/refit delayed to step 99. This
  job was submitted during the first 50-step attempt even though the wrapping
  `sbatch` call timed out before the script could record the job id.
- `3271807`: matching Qwen3-8B public PARD-2 interval-5 online run,
  OSL1024/GBS8/K5, 50 steps, `train_interval=5`, `refit_interval=5`,
  `initial_refit=true`, and `pard_token=151670`.
- Remote job record:
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/latest_qwen8_public_pard2_interval5_longosl1024_step50_20260611_jobs.txt`
- Final queue state after the 2026-06-12 refresh:
  `3271789` and `3271807` both completed with batch exit code `0:0`.
- Matched steps 2-37 comparison:
  - `3271789` static-equivalent: token acceptance `48.05%`, mean acceptance
    length `3.442`, generation worker throughput `143.40 tok/s/GPU`, mean
    total step time `32.52s`.
  - `3271807` interval-5 online: token acceptance `48.30%`, mean acceptance
    length `3.462`, generation worker throughput `145.35 tok/s/GPU`, mean
    total step time `32.78s`, and seven train/refit updates with mean draft
    loss `1.77714`.
- Final matched steps 2-50 comparison:
  - `3271789` static-equivalent: token acceptance `47.54%`, mean acceptance
    length `3.404`, generation worker throughput `140.70 tok/s/GPU`, mean
    generation time `17.20s`, and mean total step time `32.68s`.
  - `3271807` interval-5 online: token acceptance `47.16%`, mean acceptance
    length `3.400`, generation worker throughput `142.77 tok/s/GPU`, mean
    generation time `16.91s`, mean total step time `32.93s`, and nine
    train/refit updates with mean draft loss `1.78618`.
- Current read: the earlier steps 2-11 negative slice was too early, and the
  matched steps 2-37 slice showed a small generation-throughput advantage for
  interval-5. Across the full matched steps 2-50, interval-5 still improves
  generation time and worker throughput slightly, but it does not beat the
  static-equivalent run end-to-end because train/refit overhead offsets the
  generation gain.
- Pending jobs `3267466`, `3267467`, `3268813`, and `3268814` were moved
  in-place from `coreai_dlalgo_nemorl` to `coreai_dlalgo_llm`; after the move,
  all four started running. The two Qwen3-14B Lite jobs later completed
  successfully.
