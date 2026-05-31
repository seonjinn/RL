# Qwen3 Eagle3 Remote Execution Inputs

Last updated: 2026-05-22 15:27 PDT

This file records the concrete cluster-side inputs found while preparing the
Qwen3-235B Thinking Eagle3 draft-model pilot. It is intentionally limited to
values that were observed on `oci-hsg-cs-001-vscode-02` or in the local
Qwen3-235B SWE launch scripts.

## Slurm

Recommended account for this workstream:

```text
coreai_dlalgo_nemorl
```

Evidence:

- `run_grpo_qwen3_235b_swe.sh` uses `SBATCH_ACCOUNT=coreai_dlalgo_nemorl`.
- `sacctmgr` shows the current user has access to `coreai_dlalgo_nemorl`.
- The same local launch script uses `SBATCH_PARTITION=batch`.
- `sinfo` on the remote host shows `batch` GPU nodes, and the dry-run cluster
  probe can see `sbatch`, `srun`, `squeue`, and `sinfo`.

Other visible accounts include:

```text
coreai_dlalgo_llm
llmservice_modelalignment_ppo
llmservice_nemotron_ultra
nemotron_agents_dev
nemotron_n3_post
```

The Eagle3 pilot scripts still default `SBATCH_ACCOUNT=dummy` for safety. Pass
the real account explicitly when moving beyond local dry-runs.

Latest rollout job snapshot:

```text
2855291|qwen3-235b-swe-rollout-vllm0102-raypatch-swegym-example-smoke1step|FAILED|00:03:57|2026-05-22T08:05:33|2026-05-22T08:09:30|exit 1:0
```

Runtime update: job `2855291` started and failed in vLLM actor initialization,
not in the Eagle3 data path. The failure was:

```text
ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda9SetDeviceEab
```

The target container uses `/opt/venv/bin/python`, aarch64, CUDA 12.9, and
`torch 2.8.0a0+5228986c39.nv25.05`. A follow-up native ABI probe confirmed
that the shared vLLM `0.10.2`, `0.11.2`, and `0.13.0` wheel target sites all
fail `import vllm._C`. The source-built path has now passed:

```text
2855535|q235b-vllm-build|FAILED after wheel build; missing pybase64 at probe
2856310|q235b-vllm-finalize|COMPLETED|installed pybase64 into tmp site and wrote source-build PASS
2856339|q235b-vllm-abi|COMPLETED|source-built vLLM ABI PASS
2856410|qwen3-235b-swe-rollout-vllm0102src-swegym-example-smoke1step|FAILED|missing cpuinfo during AsyncLLM import
2856499|q235b-vllm-finalize|COMPLETED|installed py-cpuinfo and verified AsyncLLM import
2856536|qwen3-235b-swe-rollout-vllm0102src-cpuinfo-swegym-example-smoke1step|FAILED|missing frozendict during Qwen3MoeForCausalLM inspection
2856588|q235b-vllm-finalize|COMPLETED|installed frozendict and verified Qwen3MoeForCausalLM import
2856645|q235b-vllm-runtime|PASS|AsyncLLM, Qwen3MoeForCausalLM, and AsyncEngineArgs.create_engine_config
2856680|q235b-vllm-runtime|FAILED strict pip check after runtime PASS|missing pure/runtime deps plus expected NeMo-stack version mismatches
2856741|q235b-vllm-finalize|FAILED probe after installing low-risk missing deps|pydantic-extra-types required pycountry
2856752|q235b-vllm-finalize|COMPLETED|installed pycountry and verified imports
2856767|q235b-vllm-runtime|FAILED strict pip check after runtime PASS|remaining deferred NeMo-stack mismatches
2856596|qwen3-235b-swe-rollout-vllm0102src-frozendict-swegym-example-smoke1step|FAILED/CANCELLED|model load reached, then torch._inductor.standalone_compile missing in vLLM compile path
2857291|qwen3-235b-swe-rollout-vllm0102src-eager-swegym-example-smoke1step|FAILED|Hydra append syntax missing for compilation_config.level
2857334|qwen3-235b-swe-rollout-vllm0102src-eager-compact16n4g-smoke1step|FAILED|same Hydra append syntax issue
2857503|qwen3-235b-swe-rollout-vllm0102src-eagerappend-compact16n4g-smoke1step|FAILED|16-node compact compile-off retry reached OpenAI serving setup, then hit model_config API drift
2857581|qwen3-235b-swe-rollout-vllm0102src-eagerappend-swegym-example-smoke1step|FAILED|32-node official-shape retry hit the same model_config API drift
2858232|qwen3-235b-swe-rollout-vllm0102src-modelconfigfix-compact16n4g-smoke1step|FAILED|reached vLLM server startup, then failed on missing Megatron `pg_utils`
2858693|qwen3-235b-swe-rollout-vllm0102src-pgcollectionfix-compact16n4g-smoke1step|FAILED|bypassed `pg_utils`, then failed on missing `ProcessGroupCollection`
2858922|run_megatron_compat_probe.sh|COMPLETED|1-node container import probe PASS for patched Megatron compatibility set; reusable replay script: `submit_megatron_compat_probe.sh`
2858959|qwen3-235b-swe-rollout-vllm0102src-megatroncompat-compact16n4g-smoke1step|FAILED|started on 16 nodes, then worker srun creation failed with `Memory required by task is not available`
2859878|qwen3-235b-swe-rollout-vllm0102src-megatroncompat-resourcefix-compact16n4g-smoke1step|CANCELLED|16-node compact retry rejected by topology preflight before running
2857812|q235b-vllm-build|FAILED|vLLM 0.13.0 source-build fallback built wheel, then failed native import probe on tokenizers dependency conflict
```

Current rollout smoke:

```text
active official job id: 2860014
active official nodes: 32
active official state at 2026-05-22 13:58 PDT: PENDING (Resources)
active official watcher: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2860014_vllm0102src_megatroncompat_resourcefix_official32n4g_swegym_smoke.log
active non-canonical output: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_vllm0102src_megatroncompat_resourcefix_official32n4g_smoke.jsonl
active balanced fallback job id: 2860150
active balanced fallback watcher: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2860150_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_swegym_smoke.log
active balanced fallback output: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g.jsonl
cancelled compact job id: 2859878
cancelled compact reason: train_world_size=(16-8)*4=32 is not divisible by required expert_tensor_model_pipeline_parallel=1*16*4=64
previous compact job id: 2858959
previous compact state: FAILED at 2026-05-22 13:33:51 PDT
previous compact watcher: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2858959_vllm0102src_megatroncompat_swegym_smoke.log
```

Current rollout queue snapshot:

```text
2860803|qwen3-235b-swe-rollout-capture-experimental20n4g-gen4-pluginfix|PENDING|20 nodes|(Resources)|start 2026-05-22T15:27:15
```

Current higher-version fallback queue snapshot:

```text
2857812|q235b-vllm-build|FAILED|built wheel, then failed native import probe on tokenizers 0.21.4 vs transformers requirement
watcher: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_vllm_source_build_2857812_0_13_0_then_abi.log
```

`watch_vllm_source_build_then_rollout.sh` had already exited before the
finalize recovery wrote PASS, so a replacement watcher was started and submitted
ABI probe `2856339`; after ABI PASS it submitted rollout smoke `2856410`.
That smoke failed on `ModuleNotFoundError: No module named 'cpuinfo'`.
The next smoke `2856536` failed on `ModuleNotFoundError: No module named
'frozendict'` through `compressed_tensors` while inspecting `Qwen3MoeForCausalLM`.
The lightweight runtime probe `2856645` then passed the same model-resolution
path up through `AsyncEngineArgs.create_engine_config()`.
Strict runtime probe `2856680` also passed imports and engine-config creation,
then failed `pip check`. The response is not to upgrade the whole NeMo stack.
Patch job `2856741` added `opencv-python-headless`, `astor`, `interegular`,
`pydantic-extra-types`, and `levenshtein`; its probe then exposed missing
`pycountry`. Patch job `2856752` added `pycountry` and verified imports.
Post-patch runtime probe `2856767` passed imports and engine-config creation
again, then failed only at strict `pip check`. Torch, Ray, TorchVision,
setuptools, Triton, `numba`, and `torchaudio` mismatches remain deferred unless
a concrete runtime failure requires changing them. Job `2856596` then proved
that Qwen3-MoE model loading starts under the source-built site, but failed in
vLLM V1's profile run because the active vLLM compile path requires
`torch._inductor.standalone_compile`, which is absent from the target NVIDIA
Torch build. Retries `2857291` and `2857334` set `enforce_eager=True` but
failed before runtime because the new `compilation_config` keys needed Hydra
append syntax. Jobs `2857503` and `2857581` fixed the syntax and reached vLLM
OpenAI serving setup, then failed because `OpenAIServingChat` and
`OpenAIServingTokenization` require `model_config` in this source-built vLLM
layout. Retries `2858232` and `2858693` then exposed Megatron-Bridge API drift.
Retry `2858959` included the full constructor-signature plus Megatron
compatibility patch set and reached Ray startup, but failed while creating later
worker sruns because Slurm reported `Memory required by task is not available`.
The cancelled compact retry `2859878` added the resource fix but was rejected by
the new topology preflight before it could run. The active retry `2860014` uses
the official 32x4GPU shape, keeps Ray worker CPUs at `NUM_GPU * 16` (64 here),
passes sbatch `--mem=0`, and uses the `ray.sub` GRES parser fix for
`gpu:4(S:0-1)` strings.

`vLLM 0.10.2` is the first source-build recovery path, not the final version
decision. The higher-version candidate path is now staged with versioned reports:

```text
experiments/eagle3_qwen3_235b/VLLM_VERSION_STRATEGY.md
experiments/eagle3_qwen3_235b/submit_vllm_native_source_build_0_13_0.sh
experiments/eagle3_qwen3_235b/submit_vllm_native_abi_probe_0_13_0.sh
experiments/eagle3_qwen3_235b/submit_source_vllm_rollout_smoke_0_13_0.sh
experiments/eagle3_qwen3_235b/watch_vllm_source_build_0_13_0_then_rollout.sh
experiments/eagle3_qwen3_235b/watch_vllm_source_build_fallback_0_13_0.sh
```

The first automatic `0.13.0` fallback attempt failed before scheduling because
`SBATCH_TIME=06:00:00` exceeded the partition limit. The build wrapper now
defaults to `04:00:00`, and job `2857812` ran as the ABI-only fallback. Its
wheel build completed, but native import validation failed on the target site's
`tokenizers==0.21.4` vs `transformers` requirement conflict. The `0.13.0` path
should be used for rollout only if that dependency conflict is fixed and native
ABI/import probes pass, or if the operator explicitly wants to spend a second
runtime slot. `vLLM 0.13.0` declares `torch==2.9.0`; the current public PyPI
line is newer still, while the target container has NVIDIA torch
`2.8.0a0+5228986c39.nv25.05`, so direct wheel replacement is not a safe
substitute for source build plus ABI validation.

Source-built output site:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_10_2_cu129_torch28nv_source_py312
```

No canonical `qwen3_235b_swe_rollout_conversations.jsonl` corpus has been
materialized yet, so the hidden-state/train/export pipeline is intentionally not
submitted. Job `2860014` is only a SWE-Gym example smoke; if it produces data,
that output remains non-canonical until inspected. Previous jobs `2857503` and
`2857581` produced no usable corpus because they failed at the vLLM OpenAI
serving `model_config` API boundary, and compact job `2859878` was cancelled
before running because its train-world topology was invalid.
`watch_rollout_capture_materialize.sh` now supports
`RUN_PENDING_STATE_REFRESH=true` and `PENDING_STATE_REFRESH_POLLS=5` so future
watcher runs refresh pending rollout state reports while waiting for Slurm
terminal state.
The old materialize watchers can be ignored for training data because no
`train_data_step*.jsonl` was produced by the failed runtime.

Queue-wait summary:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_queue_wait_summary.json
```

The report is generated by `summarize_rollout_queue_wait.py` and is now part of
`refresh_eagle3_operator_state.py`. The report should be refreshed after the
next rollout smoke is submitted with the source-built vLLM site.
`ensure_rollout_watchers.py` is also part of operator refresh and writes:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_watcher_ensure.json
```

The current ensure report is PASS with `restart_needed_count=0`. If a required
watcher is missing or dead, the report emits the exact background restart
commands; with `--execute --allow-background` it can start only the missing
watchers and write their PID files.

Balanced fallback:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_fallback_decision.json
```

The fixed-container/source-built-vLLM `balanced_24n4g_smoke` profile is now
prevalidated and submitted as job `2860150`. The old compact
`compact_16n4g_smoke` profile is rejected because its train-world topology is
invalid.

Pipeline gated submit evidence:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_pipeline_gated_submit.json
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_pipeline_gated_submit_contract.json
```

`submit_eagle3_pipeline_if_ready.py` now requires the submitted pipeline job
file to contain `dump_job`, `train_job`, and `export_job`. After execution it
copies `latest_eagle3_pipeline_jobs.txt` to
`reports/eagle3_pipeline_jobs.env` so the completion audit has stable job-id
evidence. While the rollout corpus is missing, the gated report is expected to
be `overall_status=fail`, `expected_not_ready=true`, and `executed=false`; with
`--exit-zero-if-not-ready`, operator refresh treats that as a successful
no-submit readiness check. `validate_pipeline_gated_submit_contract.py` is part
of operator refresh and currently PASSes the not-ready-without-flag,
not-ready-with-flag, ready-no-execute, and bad-command scenarios.

Watcher-health summary:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_watcher_health.json
```

`summarize_rollout_watcher_health.py` is also part of
`refresh_eagle3_operator_state.py`; the current report is PASS with both
materialization watchers, both pending-state watchers, and the rollout operator
follow-up watcher alive while both rollout jobs are still active. The report is
queue-context aware: pending-state watchers are required while the corresponding
Slurm job is active, but a normal exit after terminal state is not treated as a
stale failure. The pipeline watcher is expected to be missing before the
hidden-state/train/export pipeline is submitted.

## Container

The local Qwen3-235B SWE script defaults to:

```text
/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/nvcr.io/nvidian/nemo-rl:7684dc2-45115915.squashfs
```

That exact path was not visible from `oci-hsg-cs-001-vscode-02`.

Visible candidate containers:

```text
/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo-rl.sqsh
/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.04.02.sqsh
/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh
```

These candidate images are only path-verified. Before `SUBMIT=true`, run the
Slurm preflight inside the selected container with `PREFLIGHT_REQUIRE_MODELOPT_IMPORT=true`
or run `validate_modelopt_recipe_overrides.py --require-modelopt-import` inside
the same environment.

Dry-run the preflight-only wrapper first:

```bash
SUBMIT=false \
ARTIFACT_ROOT=/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
SBATCH_ACCOUNT=coreai_dlalgo_nemorl \
SBATCH_PARTITION=batch \
CONTAINER=/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh \
MOUNTS=/lustre:/lustre \
bash experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh

python3 experiments/eagle3_qwen3_235b/analyze_container_preflight.py \
  --job-file latest_eagle3_container_preflight_job.txt \
  --logs-dir logs \
  --cluster-probe-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/container_preflight_cluster_probe.json \
  --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
  --container /lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh \
  --sbatch-account coreai_dlalgo_nemorl \
  --sbatch-partition batch \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/container_preflight_analysis.md \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/container_preflight_analysis.json
```

## Existing RL References

Main remote RL repo:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL
```

Observed git state:

```text
branch: main
head: c40dba37789c
```

Useful files:

```text
examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml
examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n8g-async-1off.yaml
tests/test_suites/llm/performance/grpo-qwen3-235b-16n8g.sh
tests/test_suites/llm/performance/grpo-qwen3-235b-32n8g-async-1off.sh
3rdparty/Gym-workspace/Gym/responses_api_models/local_vllm_model/configs/qwen3_235b_a22b_instruct_2507.yaml
3rdparty/Gym-workspace/Gym/responses_api_models/local_vllm_model/scripts/launch_vllm_server_qwen3235ba22b_8nodes.sh
ray.sub
```

The SpecDec-RL Qwen3-235B performance recipes configure the real async GRPO
shape, but no Qwen3-235B Eagle3 training recipe was found under
`examples/configs`. The practical path is therefore:

1. Train or export the Eagle3 draft through the ModelOpt pipeline in this
   directory.
2. Convert/export it to the vLLM-compatible layout.
3. Wire that draft into the RL generation side and run baseline versus trained
   draft smoke/sweep jobs.

The integration validator confirms the RL generation config and SpecDec-RL
source hooks:

```bash
python3 experiments/eagle3_qwen3_235b/validate_nemo_rl_specdec_integration.py \
  --config grpo_qwen3_235b_swe.yaml \
  --draft-model nvidia/Qwen3-235B-A22B-Eagle3 \
  --specdec-rl-dir /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/nemo_rl_specdec_integration.md \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/nemo_rl_specdec_integration.json \
  --env-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/nemo_rl_specdec_overrides.env
```

On `oci-hsg-cs-001-vscode-02`, this currently passes for the public draft model
reference. For a trained local draft, replace `--draft-model` with the exported
`VLLM_DRAFT_DIR` and add `--require-draft-files` after export.

## Current Staged Inputs

Remote artifact root:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3
```

Staged verifier metadata:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/verifier_config/config.json
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/verifier_config/tokenizer_config.json
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/verifier_config/generation_config.json
```

Staged architecture/template:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/architecture/eagle3_architecture.json
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/templates/qwen3_generation_template.jinja2
```

Pilot data for mechanical smoke tests:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/pilot_existing_chat_content_64.jsonl
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/pilot_existing_chat_reasoning_32.jsonl
```

`pilot_existing_chat_content_64.jsonl` validates cleanly at `MAX_SEQ_LEN=16384`
and is the preferred short hidden-state dump input. It is not final training
data because it was not generated by `Qwen/Qwen3-235B-A22B-Thinking-2507`.

`pilot_existing_chat_reasoning_32.jsonl` is useful for formatting checks with
`reasoning_content` merged into `<think>...</think>`, but several rows exceed a
32k token estimate.

Final Qwen3-235B SWE/RL Eagle3 training data should come from the rollout
capture, not from DAPO/OpenMath math data:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_compact16n4g.jsonl
```

The planned training scale is staged:

```text
pilot: 8 conversations, MAX_STEPS=20, dump/train/export limits 2h/2h/1h
calibration: about 5k examples, max 2k training steps
production candidate: about 100k examples, one epoch, about 25k steps at effective batch 4
```

The concise data/duration decision is also tracked in:

```text
experiments/eagle3_qwen3_235b/TRAINING_DATA_DURATION_PLAN.md
```

The next-action planner has moved past the source-build gate. The
source-built vLLM native ABI proof is now present, and the first ready action is
to poll active rollout jobs `2860014` and `2860150` until one produces a
parseable SWE-Gym smoke corpus or exposes the next concrete runtime/API
failure.

The pipeline-submit preflight now also reads
`$ARTIFACT_ROOT/reports/eagle3_resource_profile.env` when the operator has not
explicitly set GPU/TP overrides. Direct preflight runs therefore use the same
4-GPU-node profile as the operator refresh path instead of falling back to the
older `8 GPU/node, TP=8` defaults.

The completion and goal-evidence audits still remain incomplete overall because
the canonical rollout corpus, Eagle3 hidden-state dump, draft training
checkpoint, export artifact, and RL smoke/sweep evidence do not exist yet.

## Missing For Real Training

The remaining blocking inputs before a real Qwen3-235B Thinking draft model can
be produced are:

- current consolidated action report:

  ```text
  /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.md
  ```

  It currently reports `ready_for_operator_submit` with `rollout_poll` as the
  ready action because the rollout jobs are still queued. The
  hidden-state/train/export pipeline remains `submit_ready=false`. After the
  container report is PASS and rollout state reaches `pipeline_dry_run`, the
  next ready action should become `run_pipeline_submit_preflight`; that action
  is no-submit and no-heavy-GPU.

  Inspect or print the selected action with:

  ```bash
  python3 experiments/eagle3_qwen3_235b/create_eagle3_operator_sheet.py \
    --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
    --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
    --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet.json \
    --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet.md
  ```

  This writes a no-submit operator sheet that shows action order, required
  `--execute`/allow flags, execution-record paths, and follow-up analyzer
  commands.

  Validate the operator sheet contract before copying any execute command:

  ```bash
  python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_sheet.py \
    --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
    --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
    --operator-sheet-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet.json \
    --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet_validation.json \
    --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet_validation.md
  ```

  After executing an action through the operator helper, validate the operator
  record:

  ```bash
  python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_execution.py \
    --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
    --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
    --operator-sheet-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet.json \
    --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_execution.json \
    --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_execution.md
  ```

  This validates only the local execution record and shell return codes; the
  real gate status still comes from the stage-specific analyzers.

  To see the whole objective as a proof matrix, run:

  ```bash
  python3 experiments/eagle3_qwen3_235b/audit_eagle3_goal_evidence.py \
    --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
    --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_goal_evidence.json \
    --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_goal_evidence.md
  ```

  This report is expected to remain `INCOMPLETE` until the actual rollout
  corpus, hidden-state dump, trained checkpoint, export, and RL sweep exist.

  ```bash
  python3 experiments/eagle3_qwen3_235b/run_eagle3_next_action.py \
    --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
    --list
  ```

  Execution is guarded. The container preflight requires `--execute
  --allow-slurm`; the rollout-capture smoke additionally requires
  `--allow-heavy-gpu`.

  Validate the plan structure before handing it to an operator:

  ```bash
  python3 experiments/eagle3_qwen3_235b/validate_eagle3_next_action_plan.py \
    --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
    --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions_validation.json \
    --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions_validation.md
  ```

- final target-domain conversation JSONL generated by Thinking-2507 or extracted
  from the actual RL rollout loop
- rollout capture validation report. Use
  `validate_rollout_capture_config.py` before the capture run; if SpecDec-RL
  only logs flat `content`, run
  `apply_specdec_rl_rollout_role_logging_patch.sh` and apply
  `specdec_rl_rollout_role_logging.patch` so `train_data_step*.jsonl` keeps
  `role` arrays for lossless normalization.
- rollout submit preflight from `preflight_rollout_capture_submit.py`; the
  current remote report already has `submit_ready=true`, so the remaining
  action is explicit submission of the 1-step capture smoke.
- rollout capture smoke plan from `run_rollout_capture_smoke.sh`, then
  `materialize_rollout_capture_corpus.sh` after the capture job has produced
  `train_data_step*.jsonl`.
- rollout artifact analysis from `analyze_rollout_capture.py`; this reports
  `missing_capture`, `needs_materialize`, `pass`, or `fail` and writes the
  next command into Markdown/JSON.
- post-submit rollout job analysis from `analyze_rollout_capture_job.py`; this
  reads `latest_235b_swe_job_id.txt`, Slurm/Ray logs, `train_data_step*.jsonl`,
  and the materialized corpus status.
- post-submit rollout state driver from `advance_rollout_capture_state.py`;
  run it after the capture job to select submit, poll, materialize, or pipeline
  dry-run as the next safe action.
- corpus strategy report from `analyze_corpus_strategy.py`; for the current
  Qwen3 SWE/RL target, actual RL rollout responses are primary while
  DAPO/OpenMathInstruct-style math data is supplemental unless the target
  rollout itself is math.
- selected, preflighted container image; the current candidate command uses
  `/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh`,
  but the analyzer must report PASS before hidden-state dump or Eagle3 training
- Slurm submit approval and final account/partition selection
- GPU pilot run that creates hidden states, a ModelOpt checkpoint, HF export,
  vLLM draft directory, and RL smoke/sweep evidence
- post-export artifact reports:

  ```text
  /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_training_checkpoint.json
  /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_export_artifacts.json
  ```

  These are required after the pilot hidden-state/train/export pipeline and
  before the trained-draft spec-token sweep. If the pipeline logs pass but
  either artifact report is missing, `plan_eagle3_next_actions.py` should
  promote `run_post_export_artifact_validations`, which is no-submit and
  no-heavy-GPU.
- gated pipeline submit report:

  ```text
  /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_pipeline_gated_submit.json
  ```

  Use `submit_eagle3_pipeline_if_ready.py` after
  `eagle3_pipeline_submit_preflight.json` reports `submit_ready=true`. The
  helper checks that the rollout corpus exists and that the pilot submit command
  still targets `submit_eagle3_pipeline.sh` before allowing
  `--execute --allow-heavy-gpu`. For no-submit operator refresh, pass
  `--exit-zero-if-not-ready` so an expected missing-corpus state records
  `expected_not_ready=true` without making the refresh fail. The completion audit
  and goal-evidence matrix now require this gated report separately from the
  no-submit preflight report, so a final PASS proves the heavy pipeline was
  submitted through the guard.

H100 is viable for the workflow, especially the offline hidden-state path, but
the 235B verifier forward pass dominates cost. GB200 is more practical for the
online base-model training path where the verifier remains in the training loop.
