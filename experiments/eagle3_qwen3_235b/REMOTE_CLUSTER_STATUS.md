# Remote Cluster Status

Last updated: 2026-05-23 03:08 PDT

## Current Local Probe Update

As of `2026-05-23 03:08 PDT`, the current workstation cannot reach any of the
configured OCI HSG aliases with the no-submit remote probe:

```text
overall_status: unreachable
reachable: 0 / requested: 4
hosts: oci-hsg-cs-001-vscode-02, oci-hsg-cs-001-vscode-01, oci-hsg-cs-001-login-01.nvidia.com, oci-hsg
```

This supersedes the older reachability notes below for current execution from
this workstation. The remote artifact root
`/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3` is also not
writable from the current local mount, so current operator reports should be
regenerated on the remote host once SSH/DNS recovers, or written to a local
temporary artifact root only for no-submit validation.

The current next-action planner state from the local workspace is:

```text
ready action 1: probe_remote_hosts
ready action 2: poll_megatron_compat_probe with PROBE_JOB_ID=2867766
blocked before rollout: remote host/Hayate path evidence, Megatron compatibility probe PASS report, canonical rollout corpus
```

`preflight_eagle3_operator_ready_submit.py` now validates `probe_remote_hosts`
as a first-class non-Slurm action: `ssh` must be visible, host aliases must be
explicit, output report parents must be writable, and the probe must remain
non-strict so unreachable hosts still produce structured evidence.

When SSH/DNS recovers, use the staged repo's no-submit resume entrypoint before
submitting more GPU work:

```bash
PRINT_ONLY=true \
SYNC_EXPERIMENTS=true \
SYNC_PROBE_JOB_FILE=true \
REMOTE_HOST=oci-hsg-cs-001-vscode-02 \
REMOTE_WORKDIR=/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap \
REMOTE_ARTIFACT_ROOT=/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
REMOTE_ENTRYPOINT=experiments/eagle3_qwen3_235b/resume_eagle3_operator_state.sh \
PROBE_JOB_ID=2867766 \
bash experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh
```

After reviewing the printed command, switch `PRINT_ONLY=false`. To execute only
the current safe gates on the remote host, add
`EXECUTE_SAFE_ACTIONS=true` and
`SAFE_ACTION_IDS="probe_remote_hosts poll_megatron_compat_probe"`. This still
does not submit rollout; the Megatron follow-up requires the separate
`SUBMIT_ROLLOUT=true ALLOW_HEAVY_GPU=true` pair before spending heavy GPUs.
Add `RUN_FULL_REFRESH=true` when the remote host should also regenerate the
broader no-submit evidence matrix: ModelOpt loss-mask patch validation, recipe
override validation, Hayate/SpecForge analyses, draft inventory, goal evidence,
and completion audit.

## Host Reachability

The short alias `oci-hsg` did not resolve from this workstation. The following
OCI HSG aliases were reachable with non-interactive SSH:

- `oci-hsg-cs-001-vscode-02`
- `oci-hsg-cs-001-vscode-01`
- `oci-hsg-cs-001-login-01.nvidia.com`

The current working host for dry-runs is:

```text
oci-hsg-cs-001-vscode-02
```

## Remote Workspace

The experiment scripts and a patched official ModelOpt checkout were staged at:

```text
/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap
```

Artifact root:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3
```

Remote ModelOpt:

```text
/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap/Model-Optimizer
```

This checkout is official `NVIDIA/Model-Optimizer@b02e8885509c...` with the
local TRT-LLM hidden-state `loss_mask` patch applied. Its expected dirty state is
one file:

```text
M examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py
```

## Slurm And Container Inputs

The current local Qwen3-235B SWE launch script uses:

```text
SBATCH_ACCOUNT=coreai_dlalgo_nemorl
SBATCH_PARTITION=batch
```

`sacctmgr` on `oci-hsg-cs-001-vscode-02` shows `sna` has access to
`coreai_dlalgo_nemorl`, and `sinfo` shows `batch` GPU nodes. The pilot scripts
still default to `SBATCH_ACCOUNT=dummy` so that accidental submits fail until a
real account is explicitly provided.

The local SWE script's default container path was not visible on
`oci-hsg-cs-001-vscode-02`:

```text
/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/nvcr.io/nvidian/nemo-rl:7684dc2-45115915.squashfs
```

Visible but not yet container-preflighted candidates:

```text
/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo-rl.sqsh
/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.04.02.sqsh
/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh
```

Before `SUBMIT=true`, choose one container and prove ModelOpt/Transformers/TRT-LLM
imports inside that exact Slurm environment.

The preflight-only wrapper for that gate is:

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

## Hayate/Hiso Visibility

Accessible Hayate/Hiso ModelOpt reference:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer
```

Observed state:

- branch/head: `main@4eacb0da723a`
- untracked experiment files under `examples/speculative_decoding`
- incompatible with the current Qwen3 wrapper API because it predates the newer
  `modelopt_recipes/general/speculative_decoding/eagle3.yaml` layout

User-mentioned paths:

- `.../TensorRT-Model-Optimizer-worktrees/eagle3`: not present
- `.../nemo-rl-internal-worktrees/feat-eagle3-online-specdec/models`:
  permission denied from `sna`

The inventory/provenance tools now record these as warnings instead of aborting.

## Dry-Run Result

The `SUBMIT=false RUN_PILOT=true PREP_DRY_RUN=true` remote cluster pilot ran on
`oci-hsg-cs-001-vscode-02` and completed through handoff bundle creation.

Reports were written under:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/handoff/RUNBOOK.md
```

Confirmed by dry-run:

- Slurm commands are visible: `sbatch`, `srun`, `squeue`, `sinfo`
- artifact root is writable
- remote artifact root has hundreds of TiB free
- patched remote ModelOpt is visible to preflight
- staged Qwen3-235B Thinking verifier metadata is visible
- ModelOpt offline/online wrapper overrides validate against the Qwen3 reference
- Slurm pipeline plan prints preflight -> dump -> validate hidden states -> train
  -> export
- the post-export planner gate now requires both
  `eagle3_training_checkpoint.json` and `eagle3_export_artifacts.json` to pass
  before the trained-draft sweep can become ready
- readiness audit reaches `INCOMPLETE`, not crash/fail, because required real
  inputs are still missing

Expected warnings/failures in the dry-run:

- `SBATCH_ACCOUNT=dummy`
- final Qwen3-235B-generated training conversations are not staged yet
- no hidden states/checkpoint/export artifacts exist yet
- GPU visibility on the login/vscode host is not expected
- Hayate draft model path is not readable by `sna`

## Active Runtime Gate

As of 2026-05-22 11:40 PDT, the source-built vLLM native ABI gate has passed and
the current active jobs are dependency-patched source-built rollout smoke
retries. Source-build job
`2855535` built a `vllm-0.10.2+cu129` aarch64 wheel inside the target NeMo
container, then failed its final import probe because `pybase64` was missing.
Finalize job `2856310` reused the tmp site, installed `pybase64`, reran native
imports, and wrote PASS for:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_10_2_cu129_torch28nv_source_py312
```

ABI probe job `2856339` then passed on this source-built site. The first
source-built rollout smoke `2856410` then moved past ABI and failed during
`AsyncLLM` import because `cpuinfo` was missing. Patch job `2856499` installed
`py-cpuinfo` into the same site and verified `AsyncLLM` import. The next retry
`2856536` reached Qwen3-MoE model inspection and failed because
`compressed_tensors` needed `frozendict`. Patch job `2856588` installed
`frozendict` and verified `Qwen3MoeForCausalLM` import. Runtime probe `2856645`
passed `AsyncEngineArgs.create_engine_config()` for Qwen3-235B Thinking. A
stricter probe `2856680` passed the same runtime path and then failed `pip
check`; it exposed missing cleanup dependencies plus expected NeMo-stack version
mismatches. Patch job `2856741` installed the low-risk missing packages but its
probe exposed missing `pycountry`. Follow-up patch `2856752` installed
`pycountry` and verified imports. Post-patch probe `2856767` passed the runtime
path and failed only on strict `pip check`.

Installed cleanup packages:

```text
opencv-python-headless
astor
interegular
pydantic-extra-types
levenshtein
pycountry
```

Do not replace the container's NVIDIA Torch, Ray, TorchVision, setuptools, or
Triton baseline unless a concrete runtime failure requires it. The same applies
to deferred `numba`, `torchaudio`, and Triton metadata issues from `pip check`.
The source-built rollout smoke `2856596` started on 32 nodes and got past Ray
startup, vLLM worker initialization, Qwen3-MoE resolution, and model loading. It
failed during vLLM V1 profile/KV-cache setup when the compile path tried to
import `torch._inductor.standalone_compile`, which is absent from the target
NVIDIA Torch build. The following retries disabled that compile path and then
exposed the next vLLM OpenAI-serving API drift:

```text
2857291|qwen3-235b-swe-rollout-vllm0102src-eager-swegym-example-smoke1step|FAILED|32 nodes|Hydra append syntax missing for compilation_config.level
2857334|qwen3-235b-swe-rollout-vllm0102src-eager-compact16n4g-smoke1step|FAILED|16 nodes|same Hydra append syntax issue
2857503|qwen3-235b-swe-rollout-vllm0102src-eagerappend-compact16n4g-smoke1step|FAILED|16 nodes|reached OpenAI serving setup; missing OpenAIServingChat/OpenAIServingTokenization model_config
2857581|qwen3-235b-swe-rollout-vllm0102src-eagerappend-swegym-example-smoke1step|FAILED|32 nodes|same model_config API drift
2858232|qwen3-235b-swe-rollout-vllm0102src-modelconfigfix-compact16n4g-smoke1step|FAILED|16 nodes|model_config fix worked far enough to expose missing Megatron `pg_utils`
2858693|qwen3-235b-swe-rollout-vllm0102src-pgcollectionfix-compact16n4g-smoke1step|FAILED|16 nodes|bypassed `pg_utils`, then failed on missing `ProcessGroupCollection`
2858959|qwen3-235b-swe-rollout-vllm0102src-megatroncompat-compact16n4g-smoke1step|FAILED|16 nodes|reached Ray startup, then worker srun creation failed with `Memory required by task is not available`
2859878|qwen3-235b-swe-rollout-vllm0102src-megatroncompat-resourcefix-compact16n4g-smoke1step|CANCELLED|16 nodes|resource-fix retry rejected by Megatron train-world topology preflight
2860014|qwen3-235b-swe-rollout-vllm0102src-megatroncompat-resourcefix-official32n4g-smoke1step|PENDING|32 nodes|topology-valid official retry; Ray worker CPUs=64 and sbatch `--mem=0`
```

Watchers:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2857291_vllm0102src_swegym_smoke.log
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2857334_vllm0102src_swegym_smoke.log
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2857503_vllm0102src_swegym_smoke.log
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2857581_vllm0102src_swegym_smoke.log
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2858232_vllm0102src_modelconfigfix_swegym_smoke.log
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2859878_vllm0102src_megatroncompat_resourcefix_swegym_smoke.log
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2860014_vllm0102src_megatroncompat_resourcefix_official32n4g_swegym_smoke.log
```

The active patched retry output, if produced, is intentionally non-canonical
smoke data:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_vllm0102src_megatroncompat_resourcefix_official32n4g_smoke.jsonl
```

Lightweight runtime probe `2856645` passed:

```text
vllm._C PASS
AsyncLLM PASS
Qwen3MoeForCausalLM PASS
AsyncEngineArgs.create_engine_config PASS for Qwen/Qwen3-235B-A22B-Thinking-2507
```

`eagle3_next_actions.md` now reports `vllm_source_build=pass` and
`vllm_abi_probe=pass`; the current ready action is to poll the rollout smoke.

`eagle3_completion_audit.md` and `eagle3_goal_evidence.md` now also require
`source-built vLLM native ABI PASS` before the path can be considered complete.
That runtime proof is now satisfied, but the overall goal remains incomplete
because no canonical rollout corpus, hidden-state dump, Eagle3 checkpoint, export
artifact, or trained-draft RL validation exists yet.

## vLLM Version Strategy

The active `0.10.2` source-built site is not intended as a claim that `0.10.2`
is the best final vLLM version. It is the first canonical source-build recovery
path for the already patched NeMo-RL integration. Wheel ABI probes for `0.10.2`,
`0.11.2`, and `0.13.0` all failed against the target container with the same
`vllm._C` unresolved symbol, so the immediate gate is source-building against
the target container's `torch 2.8.0a0+5228986c39.nv25.05` rather than selecting a
public wheel.

The current runtime uses the fixed
`/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh`
container. It is not a nightly vLLM Docker image. The visible llmservice
container directory currently has `nemo_25.04.02.sqsh`, `nemo_25.07.01.sqsh`,
and `nemo-rl.sqsh`; no `nemo_25.09`, `nemo_25.11`, or standalone vLLM container
was visible in that directory at the latest check.

A higher-version `vLLM 0.13.0` source-build wrapper and companion ABI/rollout
watcher wrappers have been prepared with distinct output/report/job-file paths
so they will not overwrite the canonical `2855535` watcher state:

```text
experiments/eagle3_qwen3_235b/submit_vllm_native_source_build_0_13_0.sh
experiments/eagle3_qwen3_235b/submit_vllm_native_abi_probe_0_13_0.sh
experiments/eagle3_qwen3_235b/submit_source_vllm_rollout_smoke_0_13_0.sh
experiments/eagle3_qwen3_235b/watch_vllm_source_build_0_13_0_then_rollout.sh
experiments/eagle3_qwen3_235b/watch_vllm_source_build_fallback_0_13_0.sh
```

The first automatic `0.13.0` fallback submit attempt failed before scheduling
because `SBATCH_TIME=06:00:00` exceeded the oci-hsg `batch` partition limit.
The wrapper now defaults to `04:00:00`. Source-build job `2857812` built the
wheel but failed the native import probe before writing a PASS report because
`transformers` required `tokenizers>=0.22.0,<=0.23.0` while the target runtime
resolved `tokenizers==0.21.4`. Its watcher is:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_vllm_source_build_2857812_0_13_0_then_abi.log
```

The watcher had `SUBMIT_ROLLOUT=false`, so it did not add another rollout smoke
automatically. `0.13.0` remains a failed fallback until this dependency conflict
is fixed and native ABI/import probes pass.

The `0.13.0` wrapper uses `TRANSFORMERS_SPEC=transformers>=4.56.0,<5`, but it
does not install another torch into the target site. This is intentional:
`vLLM 0.13.0` declares `torch==2.9.0`, while the target NeMo 25.07.01 container
has `torch 2.8.0a0+5228986c39.nv25.05`. `vLLM 0.21.0` is an even larger jump and
declares `torch==2.11.0`, so it should be treated as a separate track unless a
newer matching NeMo/vLLM container becomes available. Because source-built
`0.10.2` now passes native ABI, `0.13.0` is not the immediate blocker; it is the
next candidate if rollout smoke fails due vLLM runtime/API drift or if speed/
acceptance measurements demand a newer baseline.

The detailed decision is tracked in:

```text
experiments/eagle3_qwen3_235b/VLLM_VERSION_STRATEGY.md
```

## Current Next-Action Report

Important runtime update: rollout job `2855291` failed because the wheel-based
vLLM native extension could not resolve `_ZN3c104cuda9SetDeviceEab` against the
target container's `torch 2.8.0a0+5228986c39.nv25.05` build. The shared wheel
sites for `0.10.2`, `0.11.2`, and `0.13.0` all failed the same way. The
source-built `0.10.2` site now passes source-build and ABI checks:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_10_2_cu129_torch28nv_source_py312
```

The active rollout job is experimental 20n4g retry `2860803`. Previous
compile-off retries `2857503` and `2857581` reached vLLM OpenAI serving setup
and exposed the `model_config` constructor drift; later retries exposed
Megatron-Bridge API drift and a Slurm worker-step memory/accounting issue. The
current Megatron gate is fixed by Qwen3MoE plugin probe `2860778`, which passed
`Qwen3MoeForCausalLM` registration and synthetic provider creation while keeping
the container's `/opt/Megatron-Bridge` and `/opt/megatron-lm`. Do not submit the
hidden-state/train/export Eagle3 pipeline until the patched smoke is inspected
and a canonical rollout corpus is materialized. Compact retry `2859878` was
cancelled before running after topology preflight showed
`train_world_size=(16-8)*4=32` is not divisible by the required
`expert_tensor_model_pipeline_parallel=1*16*4=64`.

Current active rollout capture jobs:

```text
2860803|qwen3-235b-swe-rollout-capture-experimental20n4g-gen4-pluginfix|PENDING|20 nodes|(Resources)|start 2026-05-22T15:27:15
```

The current no-submit decision report is:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.md
```

As of the latest refresh on `oci-hsg-cs-001-vscode-02`, it reports:

```text
overall_status: ready_for_operator_submit
container_preflight: pass
rollout_submit_preflight: pass, submit_ready=true
rollout_state: running, next_step=poll
rollout_queue_wait: waiting, pending=1
rollout_watcher_health: pass
rollout_watcher_ensure: pass, restart_needed_count=0
rollout_watcher_ensure_validation: pass
pipeline_submit_preflight: incomplete, submit_ready=false
pipeline_gated_submit: fail, expected_not_ready=true, executed=false
pipeline_gated_submit_contract: pass
pipeline_analysis: missing
training_checkpoint: missing
export_artifacts: missing
trained_draft_sweep: missing
training_scale: incomplete
modelopt_loss_mask: pass
nemo_rl_drift: warn
completion_audit: incomplete
```

Latest rollout capture job:

```text
2855291|qwen3-235b-swe-rollout-vllm0102-raypatch-swegym-example-smoke1step|FAILED|00:03:57|2026-05-22T08:05:33|2026-05-22T08:09:30|exit 1:0
```

The current operator action is no longer simply `rollout_poll`; the next
effective action is to finish the vLLM source build and then resubmit rollout
capture with the source-built runtime. No materialized
`qwen3_235b_swe_rollout_conversations*.jsonl` corpus exists yet. The operator
refresh regenerates the no-submit pipeline submit preflight and gated readiness
check after queue/watcher reports, before replanning. The latest pipeline
submit preflight was regenerated at `2026-05-22 07:59 PDT` with
`DUMP_GPUS_PER_NODE=4`, `TRAIN_GPUS_PER_NODE=4`, `TP=4`, and
`CONTAINER=/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh`; it is expectedly
INCOMPLETE because the rollout corpus is missing. The gated report is expectedly
FAIL/`expected_not_ready=true`/`executed=false` until `submit_ready=true`, but
it records the required post-submit job keys: `dump_job`, `train_job`, and
`export_job`. The operator refresh treats this expected not-ready state as a
successful no-submit refresh and currently reports PASS. The synthetic gated
submit contract also PASSes the not-ready-without-flag, not-ready-with-flag,
ready-no-execute, and bad-command scenarios.

Balanced 24n4g fallback is prevalidated with the same fixed container,
source-built vLLM site, and Megatron topology check. Compact 16n4g is rejected
because its training world size is not divisible by `ETP*EP*PP`.
`ensure_rollout_watchers.py` is also in the operator refresh path; the current
report is PASS and would emit restart commands if a required watcher died. It
can now also emit lock-waiting extension watcher commands if timeout risk
appears before the rollout job reaches terminal state.
`validate_rollout_watcher_ensure.py` is now part of the operator refresh path
and PASSes the synthetic alive/restart/timeout-extension scenarios.
The gated pipeline submit helper also verifies that `dump_job`, `train_job`,
and `export_job` are recorded after submit, then copies the pipeline job file to
`reports/eagle3_pipeline_jobs.env` for stable audit evidence.

Do not submit the hidden-state/train/export pipeline until the container
preflight reports PASS and the rollout capture state reaches
`next_step=pipeline_dry_run`.

When both of those gates are true, the next-action planner should switch from
rollout/container actions to `run_pipeline_submit_preflight`. That command is
still no-submit and no-heavy-GPU; it proves the hidden-state dump, hidden
validation, offline train, export, and Slurm dependency chain before
`submit_eagle3_pilot_pipeline` can become ready.

After the pilot pipeline logs report PASS, the planner should not immediately
promote the trained-draft sweep. It first requires the two post-export artifact
contracts to pass:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_training_checkpoint.json
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_export_artifacts.json
```

If either report is missing or not PASS, the next ready action becomes
`run_post_export_artifact_validations`. This is a no-submit/no-heavy-GPU action
that validates the trained ModelOpt checkpoint, compares HF/vLLM configs, and
checks the exported HF/vLLM draft artifacts before any RL sweep is submitted.

For a guarded print-or-execute interface, use:

```bash
python3 experiments/eagle3_qwen3_235b/create_eagle3_operator_sheet.py \
  --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
  --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet.json \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet.md
```

The generated operator sheet is still no-submit. It lists the ready actions in
execution order, the print-only command, the explicit execute command with
required allow flags, the execution-record path, and the analyzer/refresh
commands to run afterward.

Validate the sheet itself before copying any execute command from it:

```bash
python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_sheet.py \
  --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
  --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
  --operator-sheet-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet.json \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet_validation.json \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet_validation.md
```

After any `run_eagle3_next_action.py --execute` run, validate the operator-side
record before interpreting stage analyzers:

```bash
python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_execution.py \
  --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
  --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
  --operator-sheet-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_sheet.json \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_execution.json \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_operator_execution.md
```

This only validates the operator command record and return codes. Slurm job
success still comes from `analyze_container_preflight.py`,
`advance_rollout_capture_state.py`, and `analyze_eagle3_pipeline.py`.

For the user-facing requirement matrix, use:

```bash
python3 experiments/eagle3_qwen3_235b/audit_eagle3_goal_evidence.py \
  --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_goal_evidence.json \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_goal_evidence.md
```

This report should stay `INCOMPLETE` until a real rollout corpus, hidden-state
dump, trained checkpoint, HF/vLLM export, and RL trained-draft sweep are all
present and validated.

```bash
python3 experiments/eagle3_qwen3_235b/run_eagle3_next_action.py \
  --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
  --list
```

The helper is print-only by default. It requires `--execute --allow-slurm` for
the container preflight and `--execute --allow-slurm --allow-heavy-gpu` for the
rollout-capture smoke.

Validate the report semantics with:

```bash
python3 experiments/eagle3_qwen3_235b/validate_eagle3_next_action_plan.py \
  --plan-json /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions.json \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions_validation.json \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_actions_validation.md
```

Validate the planner state machine, including the post-export artifact gate,
with:

```bash
python3 experiments/eagle3_qwen3_235b/validate_eagle3_next_action_transitions.py \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_action_transitions.json \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/eagle3_next_action_transitions.md
```

After the first dry-run, lightweight verifier metadata was staged from
`Qwen/Qwen3-235B-A22B-Thinking-2507`:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/verifier_config/config.json
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/verifier_config/tokenizer_config.json
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/verifier_config/generation_config.json
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/architecture/eagle3_architecture.json
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/templates/qwen3_generation_template.jinja2
```

The downloaded Thinking-2507 verifier config gives `rope_theta=5000000` and
`rms_norm_eps=1e-6`. This differs from the older non-thinking public draft
reference and is now the architecture source for this workstream.

## Training Data Candidates

A usable small pilot conversation file was materialized to prove the data path:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/pilot_existing_chat_content_64.jsonl
```

Validation result:

- 64 valid conversations
- no validation warnings or failures at `MAX_SEQ_LEN=16384`
- estimated tokens: p50 1509, p95 3230, max 3773

This is a mechanical smoke-test input only. It came from an existing
`messages` JSONL owned by another llmservice workflow, not from
`Qwen/Qwen3-235B-A22B-Thinking-2507`, so do not use it as the final draft-model
training corpus.

A second pilot file includes separate assistant `reasoning_content` merged into
`<think>...</think>` blocks:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/pilot_existing_chat_reasoning_32.jsonl
```

That file validates structurally, but several rows exceed a 32k token estimate.
It is useful for testing Qwen3 Thinking formatting, not for the first short
hidden-state dump.

The normalizer now supports `INCLUDE_REASONING_CONTENT=true` for rollout logs
that store assistant reasoning separately from final assistant content.

## SpecDec-RL References

A current remote RL repo was found at:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL
```

Observed state:

```text
branch: main
head: c40dba37789c
```

Useful Qwen3-235B references:

```text
examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml
examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n8g-async-1off.yaml
tests/test_suites/llm/performance/grpo-qwen3-235b-16n8g.sh
tests/test_suites/llm/performance/grpo-qwen3-235b-32n8g-async-1off.sh
3rdparty/Gym-workspace/Gym/responses_api_models/local_vllm_model/configs/qwen3_235b_a22b_instruct_2507.yaml
3rdparty/Gym-workspace/Gym/responses_api_models/local_vllm_model/scripts/launch_vllm_server_qwen3235ba22b_8nodes.sh
```

No Qwen3-235B Eagle3 recipe was found under SpecDec-RL `examples/configs`.
Treat SpecDec-RL as the RL validation reference and this directory's ModelOpt
pipeline as the draft-model training/export path.

## Next Real Inputs

Before `SUBMIT=true`, provide:

- real `SBATCH_ACCOUNT` such as `coreai_dlalgo_nemorl`
- correct `SBATCH_PARTITION`, currently expected to be `batch`
- `VERIFIER_CONFIG_DIR` can use the staged metadata directory above
- `TOKENIZER_CONFIG` can use the staged tokenizer config above
- final `INPUT_DATA` generated from Thinking-2507 or rollout roots that can
  produce target-domain ModelOpt conversation JSONL
- rollout capture gate from `validate_rollout_capture_config.py`; current
  SpecDec-RL logs compact `train_data_step*.jsonl` with
  `env.should_log_nemo_gym_responses=false`, but production corpus capture
  should run `apply_specdec_rl_rollout_role_logging_patch.sh` and then apply
  `specdec_rl_rollout_role_logging.patch` so the flat train-data logs include
  `role` beside `content`
- rollout submit preflight from `preflight_rollout_capture_submit.py`; the
  current remote report is PASS and writes the exact 1-step capture submit
  command into `eagle3_next_actions.md`
- short rollout capture plan from `run_rollout_capture_smoke.sh`, followed by
  `materialize_rollout_capture_corpus.sh` once `train_data_step*.jsonl` exists
- rollout artifact analysis from `analyze_rollout_capture.py`; this reports
  `missing_capture`, `needs_materialize`, `pass`, or `fail` and writes the
  next command into Markdown/JSON
- post-submit rollout job analysis from `analyze_rollout_capture_job.py`; this
  reads `latest_235b_swe_job_id.txt`, Slurm/Ray logs, `train_data_step*.jsonl`,
  and the materialized corpus status
- post-submit rollout state driver from `advance_rollout_capture_state.py`; this
  refreshes rollout reports and chooses submit, poll, materialize, or pipeline
  dry-run as the next safe action
- corpus strategy report from `analyze_corpus_strategy.py`; for the current
  Qwen3 SWE/RL target, actual RL rollout responses are primary while
  DAPO/OpenMathInstruct-style math data is supplemental unless the target
  rollout itself is math
- selected container image and mounts, with container-side ModelOpt preflight

Then run from this local machine:

```bash
PRINT_ONLY=false \
REMOTE_HOST=oci-hsg-cs-001-vscode-02 \
REMOTE_WORKDIR=/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap \
REMOTE_ARTIFACT_ROOT=/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
MODELOPT_DIR=/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap/Model-Optimizer \
SBATCH_ACCOUNT=<real_account> \
SBATCH_PARTITION=<real_partition> \
VERIFIER_CONFIG_DIR=/path/to/Qwen3-235B-A22B-Thinking-2507 \
TOKENIZER_CONFIG=/path/to/Qwen3-235B-A22B-Thinking-2507/tokenizer_config.json \
MODE=rollout \
INPUT_PATHS="/path/to/rollout_or_conversation_sources" \
PREP_DRY_RUN=false \
SUBMIT=false \
bash experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh
```

Only after that readiness report has no unexpected failures, change
`SUBMIT=true RUN_PILOT=true`.
