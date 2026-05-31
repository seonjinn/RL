# Eagle3 Draft Model Plan for Qwen3-235B RL

This note records the current state of the Qwen3-235B Eagle3 workstream:
what Hayate changed, what is reusable, and the recommended path for building
a draft model for `Qwen/Qwen3-235B-A22B-Thinking-2507` in the NeMo-RL SWE loop.
For the reusable, non-Qwen-specific procedure, see
`EAGLE3_DRAFT_MODEL_PLAYBOOK.md` in this directory.

## Current Target

- RL workload: async GRPO SWE, non-colocated vLLM generation.
- Current launch model: `Qwen/Qwen3-235B-A22B-Thinking-2507` from
  `run_grpo_qwen3_235b_swe.sh`.
- YAML default still says `Qwen/Qwen3-235B-A22B`; the launch script overrides it.
- Sequence length: `16384`.
- vLLM generation TP: `8`.
- Training parallelism: TP=4, PP=8, EP=16.
- Bottleneck: `exposed_generation` is the dominant step-time component in
  `STATUS_REPORT.md`.

Speculative decoding is therefore a good target, but the draft model must be
validated inside the RL rollout loop. Standalone vLLM throughput is only a
partial signal.

## Current Gate

Update `2026-05-23 03:08 PDT`: from the current local workstation, the
configured OCI HSG aliases are temporarily unreachable by the no-submit remote
probe (`reachable=0/requested=4`), and the remote artifact root is not writable
through the current local mount. Treat older `rollout_poll` state below as
historical unless the reports are regenerated on the remote host. The current
local next-action plan promotes two safe operator actions before any heavy
rollout: `probe_remote_hosts`, then `poll_megatron_compat_probe` for
`PROBE_JOB_ID=2867766`. The Eagle3 hidden-state/train/export path remains
gated on a canonical rollout corpus and a PASS Megatron compatibility report.

The reusable path is ready at the wrapper/preflight level: discovery,
provenance capture, chat-template/data prep, hidden-state dump, offline/online
ModelOpt training, export, smoke/sweep submission, pipeline analysis, and
completion audit all have dry-run or validation entrypoints. No Qwen3-235B
Thinking Eagle3 checkpoint has been proven yet. Completion still requires a real
rollout corpus, hidden-state dump, ModelOpt checkpoint, HF/vLLM export, and
baseline-vs-trained-draft validation inside the NeMo-RL generation stack.

As of 2026-05-22 17:33 PDT, the immediate execution gate is the Qwen3-235B
SWE/RL rollout-capture smoke, not the Eagle3 training code. The active job is:

```text
2861605|qwen3-235b-swe-rollout-vllm0102src-swegym-fixed-instancedict-smoke1step|PENDING (Priority)|16 nodes|start estimate 2026-05-22T19:39:58
```

This smoke uses the source-built vLLM `0.10.2` site that passed native ABI
checks in the target NeMo container, the SpecDec-RL compatibility patches for
vLLM OpenAI-serving/Megatron-Bridge drift, and a repaired five-row SWE-Gym
NemoGym input with `metadata.instance_dict`. It is expected to produce only
non-canonical SWE-Gym example output; the output must be inspected before any
real target-domain rollout corpus is promoted.

The full SWE-Gym train split is now materialized and submit-preflighted:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_train_nemogym_hf_full.jsonl
rows: 2438
full rollout submit preflight: PASS, submit_ready=true
```

The old full SWE/R2E path is still missing, and a direct R2E-Gym materialization
probe timed out with an empty output file. For the current target, full
SWE-Gym is the ready data source once the smoke proves runtime/capture.
The after-smoke full-rollout gate remains no-submit by default and writes
`$ARTIFACT_ROOT/reports/full_swegym_after_smoke_gate.{json,md}`. If a watcher
should submit the full SWE-Gym rollout automatically after smoke PASS, it must
be launched with both `AUTO_SUBMIT_FULL_ROLLOUT=true` and
`ALLOW_FULL_ROLLOUT_HEAVY_GPU=true`; otherwise the gate only records the ready
command or refuses execution with `rerun_with_allow_heavy_gpu`.
When auto-submit is enabled, `START_FULL_ROLLOUT_WATCHER=true` and
`ALLOW_FULL_ROLLOUT_BACKGROUND=true` are the defaults, so the submitted full
rollout gets its own `watch_rollout_capture_materialize.sh` process. That
watcher promotes the full rollout corpus to canonical reports and runs the
no-submit pipeline preflight after materialization.

The hidden-state/train/export pipeline remains intentionally gated because the
canonical corpus does not exist yet:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl
```

The current no-submit decision report is
`$ARTIFACT_ROOT/reports/eagle3_next_actions.md`. On the current remote artifact
root it reports `ready_for_operator_submit` with the single ready action
`rollout_poll`, while
`preflight_eagle3_pipeline_submit.py` remains `submit_ready=false` until the
canonical corpus is materialized and validated.
The `2026-05-22 17:33 PDT` remote operator refresh is PASS. The full rollout
gate is `waiting` / `poll_smoke`; watcher health and watcher ensure are PASS
after starting lock-waiting materialization watcher extension PID `3562350`.
The full-rollout gate validator includes active job-name parser coverage, so a
pending/running full rollout is detected before another full rollout can be
submitted.

The current mounted SpecDec-RL checkout is treated as fixed-draft first:
generation-only Eagle3 support is validated separately from online draft
training support, and online draft training remains a later gate unless
`check_nemo_rl_eagle3_drift.py` can prove the required `policy.draft` source
markers.

A higher-version vLLM `0.13.0` source-build fallback ran as ABI-only job
`2857812`, but its native import probe failed on a `tokenizers==0.21.4`
dependency conflict. Newer vLLM and `vllm-project/speculators` are now tracked
as a separate backend probe, because the Speculators online EAGLE3 docs assume
`speculators>=0.5.0` and a separate `vllm>=0.18` serving environment. Do not
replace the current rollout runtime with that stack until a dedicated environment
probe, data adapter, hidden extraction smoke, and RL serving smoke pass.

Earlier in this path, the 1-step Qwen3-235B SWE rollout got through Ray startup
and worker initialization, then failed when the vLLM actor imported the native
extension:

```text
ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda9SetDeviceEab
```

The target container is aarch64 with `torch 2.8.0a0+5228986c39.nv25.05` and
CUDA 12.9. Shared pip target sites for vLLM `0.10.2`, `0.11.2`, and `0.13.0`
all failed the strengthened native probe (`import vllm._C` and
`from vllm.config import CompilationConfig`). Source-building `0.10.2` inside
the same NeMo container fixed the native ABI problem. Subsequent rollout retries
then exposed missing pure Python deps, vLLM Inductor/compile incompatibility
with the NVIDIA Torch build, Hydra append syntax for compile-off overrides, and
finally the vLLM OpenAI-serving `model_config` API drift. Job `2858232` is the
first retry after that compatibility patch.

The training data decision is also fixed for the SWE/RL target. DAPO and
OpenMathInstruct-style math corpora remain useful references or supplemental
data for math workloads, but they are not the primary corpus for this run. The
canonical Eagle3 input should be Qwen3-235B Thinking responses captured from
the actual NeMo-RL SWE rollout loop and materialized as:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl
```

The first GPU training submission should still be a tiny pilot after that
corpus exists: about 8 conversations, 20 training steps, and Slurm limits of
roughly 2h hidden dump, 2h train, and 1h export. The first useful calibration
run should use about 5k-10k target-domain conversations and is expected to take
roughly half a day to one day including rollout and queue/debug time. A larger
50k-100k+ production-candidate corpus should only follow after pilot acceptance
rate, generation speedup, and reward-regression gates pass; that scale is more
likely a 1-3 day effort excluding queue delays.

The concise operator-facing version of this data and duration decision is
tracked in `TRAINING_DATA_DURATION_PLAN.md`.

## Hayate / Hiso Findings

On `oci-hsg-cs-001-vscode-02`, the originally mentioned
`TensorRT-Model-Optimizer-worktrees/eagle3` path is not present. The accessible
checkout is:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer
```

That checkout has untracked EAGLE3 workflow files under
`examples/speculative_decoding`: Qwen3 8B/30B/32B config JSONs,
`add_dapo17k.py`, `generate_responses.py`, and Slurm wrappers. The workflow is
DAPO-Math prompt preparation, target-model response generation, legacy
`main.py --mode eagle3` training, acceptance-rate validation, and HF export.
It does not contain a Qwen3-235B config or an RL-rollout corpus path.

Therefore the currently visible Hayate/Hiso ModelOpt work is useful as a
math/bootstrap reference, not as a drop-in Qwen3-235B SWE/RL recipe. For this
workstream, the reusable idea is the sequence of response generation -> Eagle3
training -> export -> acceptance validation; the data source must be replaced
with NeMo-RL rollout conversations and the training launcher must use the
current ModelOpt recipe interface.

The local `Model-Optimizer` checkout in this workspace uses the newer recipe
interface:

```bash
./launch_train.sh --config ../../modelopt_recipes/general/speculative_decoding/eagle3.yaml \
  model.model_name_or_path=... \
  data.offline_data_path=... \
  training.output_dir=...
```

Hayate's older scripts use legacy flags such as `--model`, `--data`, and
`--mode eagle3`; do not copy them directly into this checkout.

## SpecForge/SGLang Reference

SpecForge is explicitly described by its docs and repository README as an
SGLang-team ecosystem project for training speculative decoding models and
porting them to SGLang serving:

- https://github.com/sgl-project/SpecForge
- https://sgl-project.github.io/SpecForge/
- https://sgl-project.github.io/SpecForge/basic_usage/data_preparation.html#option-1-conversation-format

The referenced example,
`examples/run_qwen3_235b_a22b_eagle3.sh`, is useful as an SGLang/SpecForge
shape reference because it launches `scripts/train_eagle3.py` with
`--target-model-backend sglang`, `--chat-template qwen`, `--tp-size`, and a
Qwen-family Eagle3 config:

- https://github.com/sgl-project/SpecForge/blob/main/examples/run_qwen3_235b_a22b_eagle3.sh

Do not treat it as a drop-in replacement for this ModelOpt/NeMo-RL path. The
current upstream file name mentions Qwen3-235B-A22B, but its visible command
targets `Qwen3-Next-80B-A3B-Instruct-FP8` and
`configs/qwen3-next-80b-a3b-eagle3.json`. It also trains for SGLang serving,
whereas this workspace is building a ModelOpt Eagle3 draft and then validating
it inside the current vLLM/NeMo-RL RL generation path.

SpecForge's data-preparation guide defines two useful schemas:

- conversation format: `{"id": "...", "conversations": [{"role": "...",
  "content": "..."}]}`.
- pre-formatted text: `{"id": "...", "text": "..."}` with
  `--is-preformatted`; SpecForge still needs the matching chat template so it
  can find assistant spans and build the loss mask.

The main Qwen3-235B ModelOpt path keeps `{"conversation_id": "...",
"messages": [...]}` because the hidden-state dump and readiness gates are wired
around ModelOpt wrappers. For SGLang/SpecForge comparison runs, the rollout
normalizer and OpenAI-compatible generator can now emit the SpecForge
`id/conversations` schema via `--output-schema specforge` or
`OUTPUT_SCHEMA=specforge`.

Generate the machine-readable comparison report with:

```bash
python3 experiments/eagle3_qwen3_235b/analyze_specforge_reference.py \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/specforge_reference.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/specforge_reference.json
```

## Recent Public Evidence

As of 2026-05-22, public evidence supports speculative decoding in RL rollouts
but shows only partial public coverage for the exact SWE/RL combination:

- NeMo RL v0.6 documents Eagle3 speculative decoding for RL rollout generation,
  including generation-only fixed drafts and online draft training.
- NVIDIA's April 2026 RL speculative-decoding report evaluates GRPO reasoning
  workloads and projects the benefit at Qwen3-235B scale; it does not present a
  completed SWE-specific Eagle3 RL run.
- NeMo RL v0.6 separately publishes a long-context multi-step SWE RL benchmark,
  so the framework now has both SWE RL and RL speculative-decoding support.
- Red Hat's April 2026 vLLM/Eagle3 post reports SWE-bench serving gains, but
  that is inference benchmarking, not RL training.

That means the Qwen3-235B SWE/RL path here should treat actual RL rollout
corpus capture and in-loop vLLM/NeMo-RL smoke tests as required evidence rather
than relying on standalone serving benchmarks.

## Local Artifacts Added Here

This directory now contains the concrete wrappers for the first implementation
path:

- `run_baseline_smoke.sh`
  submits a short NeMo-RL SWE baseline job without speculative decoding.
- `run_rollout_capture_smoke.sh`
  plans or submits a short NeMo-RL SWE run whose purpose is to create
  role-aware `train_data_step*.jsonl` under `ROLLOUT_LOG_DIR` for Eagle3 corpus
  materialization. It defaults to `DRY_RUN=true`.
- `submit_source_vllm_rollout_smoke.sh`
  is the next retry wrapper after the vLLM source build passes. It points the
  rollout at the source-built shared site, uses the SWE-Gym example JSONL as a
  one-step runtime smoke, refuses real submit until
  `vllm_native_source_build.json` reports PASS, and starts a non-canonical
  materialization watcher for the smoke output.
- `watch_vllm_source_build_then_rollout.sh`
  polls the source-build Slurm job, submits a native ABI probe against the
  source-built site after PASS, and submits the source-built vLLM rollout smoke
  only if that ABI probe also PASSes.
- `watch_vllm_source_build_retry_on_timeout.sh`
  is a safety watchdog for the current source-build gate. If the active build
  times out before writing a PASS report, it submits a longer source-build retry
  and starts the normal source-build -> ABI -> rollout watcher for the retry.
- `VLLM_VERSION_STRATEGY.md`
  records why the active canonical build is `0.10.2`, why that is not a final
  version commitment, and how the `0.13.0` source-build candidate should be
  promoted only after source-build, ABI, and rollout gates pass.
- `submit_vllm_native_source_build_0_13_0.sh`,
  `submit_vllm_native_abi_probe_0_13_0.sh`,
  `submit_source_vllm_rollout_smoke_0_13_0.sh`, and
  `watch_vllm_source_build_0_13_0_then_rollout.sh`
  are the higher-version candidate path. They use versioned report/job files so
  they do not overwrite the current `2855535`/`0.10.2` watcher state.
- `watch_vllm_source_build_fallback_0_13_0.sh`
  watches the canonical source-build job and can submit the `0.13.0` candidate
  only after a non-timeout terminal failure. Timeout and cancellation remain
  owned by the longer `0.10.2` retry watchdog.
- `analyze_vllm_source_build_job.py`
  summarizes the source-build Slurm state, PASS/FAIL report visibility, tmp
  site size, and log tails so a long vLLM native build can be triaged without
  manually opening Slurm output files.
- `submit_megatron_compat_probe.sh`
  reruns the one-node container import/API probe used after the `pg_utils`,
  `ProcessGroupCollection`, `vocab_utils`, custom-FSDP, and Torch DTensor import
  compatibility patches. Use it before spending another 16-node rollout attempt
  when Megatron-side imports change. It records the submitted job both in
  `latest_megatron_compat_probe_job.txt` and
  `$ARTIFACT_ROOT/reports/megatron_compat_probe_job.env` so the next-action
  planner can poll the existing probe instead of submitting a duplicate.
- `followup_megatron_probe_to_rollout.sh`
  polls the recorded Megatron compatibility probe and validates the grouped
  expert checks in `megatron_compat_probe.json`. If the probe is PASS it prints
  the exact balanced 24n4g rollout retry command by default. It submits only
  when both `SUBMIT_ROLLOUT=true` and `ALLOW_HEAVY_GPU=true` are set.
- `validate_megatron_probe_followup.py`
  validates that the follow-up helper fails closed for missing or bad probe
  reports, prints the rollout command for a PASS report, and refuses heavy
  rollout submission unless `ALLOW_HEAVY_GPU=true` is explicit. Operator
  refresh writes this contract to
  `$ARTIFACT_ROOT/reports/megatron_probe_followup_validation.json/md`.
- `validate_eagle3_preflight_robustness.py`
  validates that the remote-host probe and lightweight-host preflights fail
  with structured JSON/Markdown instead of tracebacks, honor temporary artifact
  roots, and do not leak token environment values into dry-run reports.
- `preflight_rollout_capture_submit.py`
  is the no-submit gate for that capture job. It checks the SpecDec-RL checkout,
  Qwen3 config, staged chat template, wrapper dry-run, and final
  `run_grpo_qwen3_235b_swe.sh` dry-run before any Slurm submission.
- `preflight_eagle3_pipeline_submit.py`
  is the no-submit gate for the expensive hidden-state dump, hidden validation,
  offline train, and export Slurm chain. It should report `submit_ready=true`
  only after the rollout corpus and container preflight are both proven.
- `plan_eagle3_next_actions.py`
  reads the container, Megatron compatibility, rollout, pipeline, loss-mask,
  NeMo-RL drift, and training-scale reports and writes one ordered no-submit
  decision report. It
  is the place to check what should be done next and whether that next command
  submits Slurm or spends heavy GPU time. A missing or pending
  `megatron_compat_probe.json` now blocks rollout submission and promotes
  `submit_megatron_compat_probe.sh` or the guarded
  `followup_megatron_probe_to_rollout.sh` poll command first, so grouped-expert
  Bridge shims are proven before another multi-node Qwen3 rollout attempt.
  Once the container gate is PASS and rollout state reaches `pipeline_dry_run`,
  it promotes
  `preflight_eagle3_pipeline_submit.py` as the next no-submit action before the
  pilot Slurm chain can be submitted. After the hidden-state/train/export
  pipeline analysis reaches PASS, it now still requires
  `eagle3_training_checkpoint.json` and `eagle3_export_artifacts.json` to pass.
  If either artifact contract is missing, it promotes a no-submit
  post-export validation action instead of the trained-draft sweep. Only after
  those contracts pass does it promote
  `submit_trained_draft_spec_tokens_sweep.sh` before completion can pass.
- `run_eagle3_next_action.py`
  reads `eagle3_next_actions.json` and prints or executes one selected action.
  It defaults to print-only; Slurm actions require `--execute --allow-slurm`,
  and rollout/pipeline GPU actions also require `--allow-heavy-gpu`.
  Actual `--execute` runs write an execution record under
  `$ARTIFACT_ROOT/reports/operator_execution/<action>.json` when `--json-out`
  is not provided.
- `create_eagle3_operator_sheet.py`
  turns `eagle3_next_actions.json` into
  `$ARTIFACT_ROOT/reports/eagle3_operator_sheet.md/json`. The sheet preserves
  the planner order, shows the print-only wrapper, the explicit execute flags,
  the execution-record path, and the recorded analyzer/refresh commands for
  each ready action. It does not submit jobs.
- `validate_eagle3_operator_sheet.py`
  validates that sheet before any command is copied from it. It checks the
  no-submit print commands, required `--execute`/allow flags, ordered
  execution-record paths, refresh commands, and linkage back to the current
  next-action plan.
- `validate_eagle3_operator_execution.py`
  validates execution records under
  `$ARTIFACT_ROOT/reports/operator_execution`. It checks JSON shape, action id
  linkage to the current plan/operator sheet, action return code, and recorded
  follow-up analyzer return codes. It does not infer Slurm job success; stage
  analyzers remain authoritative for that.
- `validate_eagle3_next_action_plan.py`
  checks the semantic shape of `eagle3_next_actions.json`: ready actions must
  have commands, known action ids must carry the expected Slurm/heavy-GPU
  flags, and submit actions must include follow-up analyzer commands.
- `validate_eagle3_next_action_transitions.py`
  creates synthetic no-submit report states and proves that the planner moves
  through the intended operator sequence: container/rollout, pipeline preflight,
  pilot pipeline, post-export artifact validation when needed, trained-draft
  sweep, then no further ready action after sweep PASS.
- `audit_eagle3_goal_evidence.py`
  writes `$ARTIFACT_ROOT/reports/eagle3_goal_evidence.md/json`, a
  requirement-by-requirement proof matrix for the whole user-facing objective.
  It explicitly separates proven setup work, including Qwen3 static input
  materialization and dry-run pipeline manifest validation, from missing final
  evidence such as rollout corpus, hidden states, trained checkpoint, export
  artifacts, and RL/vLLM trained-draft sweep.
- `validate_eagle3_export_artifacts.py`
  validates the HF export and vLLM one-checkpoint draft directories after
  ModelOpt conversion. It checks config readability, non-empty weights, Eagle3
  contract fields, safetensors headers/offsets/tensor metadata, and HF/vLLM
  config comparison status before RL smoke tests.
- `validate_eagle3_training_checkpoint.py`
  validates the trained ModelOpt checkpoint before export. It checks the HF
  checkpoint config, non-empty weights, `trainer_state.json`,
  `modelopt_state.pth`, and, inside the cluster export job, requires the
  ModelOpt state to load and contain the Qwen3 Eagle3 `eagle` mode.
- `advance_rollout_capture_state.py`
  is the post-capture driver. It refreshes job/artifact/corpus reports and
  tells the operator whether the next step is submit, poll, materialize, or
  hidden-state pipeline dry-run. With `--materialize`, it converts visible
  `train_data_step*.jsonl` into the rollout conversation JSONL.
- `EAGLE3_DRAFT_MODEL_PLAYBOOK.md`
  is the generic runbook for deriving an Eagle3 draft architecture from an
  arbitrary verifier config, preparing RL-domain data, choosing offline versus
  ModelOpt online training, exporting, and validating in the RL generation path.
- `REMOTE_CLUSTER_STATUS.md`
  records the current `oci-hsg-cs-001-vscode-02` dry-run result, remote
  workspace, patched ModelOpt path, Hayate path visibility, and the concrete
  inputs still required before `SUBMIT=true`.
- `REMOTE_EXECUTION_INPUTS.md`
  records the discovered Slurm account/partition, visible container candidates,
  SpecDec-RL Qwen3-235B recipe references, staged verifier metadata, pilot data,
  and remaining real-training gates.
- `run_static_specdec_smoke.sh`
  submits a short NeMo-RL SWE job with static Eagle3 enabled.
- `submit_static_specdec_smoke_pair.sh`
  dry-runs or submits the baseline/static-Eagle3 smoke pair with distinct run
  names and optional `afterok` dependency.
- `submit_trained_draft_smoke_pair.sh`
  dry-runs or submits the baseline vs exported/trained Eagle3 draft smoke pair
  using `VLLM_DRAFT_DIR`.
- `submit_trained_draft_spec_tokens_sweep.sh`
  submits one baseline plus trained-draft Eagle3 smoke runs for a
  `num_speculative_tokens` sweep, defaulting to `2 3 4`. Its job file records
  the tested `VLLM_DRAFT_DIR` plus RL execution context so the completion audit
  can prove which config, env file, chat template, and SpecDec-RL checkout were
  used.
- `nemo_rl_specdec_overlay.yaml`
  is the minimal static generation-side overlay.
- `nemo_rl_eagle3_online_draft_overlay.yaml`
  is the NeMo-RL online draft-training overlay. It enables `policy.draft`,
  keeps Megatron on, disables DTensor and sequence packing, and should be used
  only after the fixed-draft path has a measured rollout win.
- `modelopt_qwen3_235b_dump_hidden_states.sh`
  wraps ModelOpt hidden-state dumping for HF or TRT-LLM backends.
- `generate_training_conversations_openai.py`
  turns prompt JSONL into full prompt+assistant training conversations through
  a vLLM/OpenAI-compatible endpoint, or normalizes existing rollout logs. It
  can also emit SpecForge `id/conversations` rows with
  `--output-schema specforge` for SGLang comparison runs.
- `discover_rollout_conversation_sources.py`
  scans NeMo-RL/RL output directories for JSONL files that actually contain
  extractable assistant responses, ranks them, and can normalize the top
  candidates into ModelOpt conversation JSONL.
- `discover_eagle3_run_inputs.py`
  scans mounted model/data roots for Qwen3 verifier configs, tokenizer configs,
  rollout/conversation JSONL files, and existing Eagle3 draft configs. It can
  write a bootstrap env file so cluster-side paths are captured in one place.
- `run_eagle3_cluster_pilot.sh`
  is the one-command cluster entrypoint. It runs input discovery, Hayate/draft
  inventories, upstream-drift/provenance capture, bootstrap dry-run or pilot
  submission, and handoff bundle creation from the same `ARTIFACT_ROOT`.
- `run_eagle3_remote_cluster_pilot.sh`
  is the SSH wrapper for running the cluster entrypoint on `oci-hsg` or another
  remote host. It prints or executes the remote dry-run command and can
  optionally rsync only this experiment directory. It forwards the Qwen3 static
  input materialization env vars so the same verifier config/template controls
  can be used through the SSH path.
- `resume_eagle3_operator_state.sh`
  rebuilds the no-submit operator plan, sheet, packet, ready-submit preflight,
  and queue on the current host. It is the preferred remote entrypoint after
  SSH/DNS recovers because it refreshes the reports that gate
  `probe_remote_hosts`, Megatron probe polling, rollout capture, and the later
  hidden-state/train/export pipeline. It executes nothing by default; setting
  `EXECUTE_SAFE_ACTIONS=true` only permits non-Slurm, non-heavy actions from
  `SAFE_ACTION_IDS`. On the cluster host, after ready-submit preflight PASS,
  `EXECUTE_SLURM_ACTIONS=true` plus a narrow `SLURM_ACTION_IDS` list permits
  only explicitly allowlisted Slurm submit actions such as source-vLLM build,
  source-vLLM ABI probe, Megatron compatibility probe, or container preflight;
  heavy-GPU actions still require `ALLOW_HEAVY_GPU_ACTIONS=true`. Set
  `RUN_FULL_REFRESH=true`
  after actions when the remote host should also refresh the broader no-submit
  evidence matrix
  including ModelOpt loss-mask validation, recipe overrides, Hayate/SpecForge
  analyses, draft inventory, goal evidence, and completion audit.
- `submit_eagle3_container_preflight.sh`
  dry-runs or submits only the `slurm_preflight.sbatch` container gate. Use it
  to prove a selected sqsh image can import ModelOpt, validate the Eagle3 recipe
  overrides, and validate the Qwen3 chat-template assistant mask before spending
  GPU time on hidden-state dump or training.
- `analyze_container_preflight.py`
  reads `latest_eagle3_container_preflight_job.txt` plus Slurm logs and reports
  whether the selected container preflight is planned, missing, failed, or
  passed. Its JSON is consumed by the readiness/handoff flow.
- `probe_cluster_environment.py`
  checks Slurm command visibility, account/partition settings, container/mount
  paths, artifact disk space, GPU visibility, and basic Python package presence
  without submitting jobs.
- `probe_eagle3_remote_host.py`
  checks SSH reachability for `oci-hsg` aliases, Slurm visibility, Lustre
  mounts, Hayate/Hiso ModelOpt and draft-model paths, and remote git heads
  without submitting jobs.
- `collect_eagle3_provenance.py`
  records host/git state, critical file hashes, visible artifact paths, and
  Hayate/Hiso draft config summaries into JSON/Markdown for reproducibility.
- `check_modelopt_upstream_drift.py`
  records whether the local ModelOpt checkout is dirty, whether official
  NVIDIA `main` has moved past the visible local ref, and whether key
  speculative-decoding files match Hayate's mounted checkout.
- `check_nemo_rl_eagle3_drift.py`
  records whether the target SpecDec-RL/NeMo-RL checkout proves fixed Eagle3
  rollout support, whether online draft-training source markers are present,
  and whether the checkout differs from official NeMo-RL upstream.
- `export_modelopt_eagle3_patch_bundle.py`
  exports the local ModelOpt TRT-LLM hidden-state dumper patch as a reusable
  `git apply` bundle with dependency checks and file snapshots.
- `create_eagle3_handoff_bundle.py`
  packages the current runbook, dashboard, optional discovery/readiness/pipeline
  reports, and executable command sheet into a handoff directory for teammates.
- `normalize_rl_rollouts_to_conversations.py`
  converts existing RL rollout/trajectory JSONL with assistant text into
  ModelOpt conversation JSONL without calling a generation endpoint. It can
  optionally merge separate assistant `reasoning_content` fields into
  `<think>...</think>` blocks for Qwen3 Thinking-style data, and can emit
  SpecForge `id/conversations` rows with `--output-schema specforge`.
- `prepare_training_conversations.sh`
  is the main data-prep wrapper. It supports rollout discovery, known rollout
  normalization, existing assistant-response conversion, and OpenAI-compatible
  generation, then always validates the produced conversation JSONL. Keep
  `OUTPUT_SCHEMA=modelopt` for the Qwen3-235B ModelOpt pipeline; use
  `OUTPUT_SCHEMA=specforge` only for SGLang/SpecForge comparison.
- `materialize_rollout_capture_corpus.sh`
  converts a completed capture run's `train_data_step*.jsonl` files into the
  final ModelOpt conversation JSONL and validates the result.
- `analyze_rollout_capture_job.py`
  reads the Qwen3 SWE rollout-capture job id, Slurm/Ray logs, captured
  `train_data_step*.jsonl`, and materialized corpus status, then recommends
  waiting, materializing, or continuing to the Eagle3 pipeline.
- `analyze_corpus_strategy.py`
  records whether the current target context should use actual RL rollouts,
  math instruction/rollout data, or only bootstrap conversations before Eagle3
  hidden-state dumping.
- `estimate_eagle3_training_scale.py`
  converts the visible corpus size and ModelOpt training defaults into pilot,
  calibration, and production-candidate step/storage estimates. The default
  Qwen3-235B fixed-draft path is 8-example/20-step pilot, then a
  hundreds-to-thousands calibration, then at least tens of thousands of
  target-domain SWE/RL assistant responses before treating a draft as a serious
  candidate.
- `validate_training_conversations.py`
  checks Eagle3 conversation JSONL before hidden-state dump, including schema,
  duplicate conversation ids, assistant-token presence, length estimates, and
  hidden-state storage estimates. It accepts ModelOpt
  `conversation_id/messages` and SpecForge `id/conversations` conversation
  format.
- `prepare_qwen3_generation_template.py`
  extracts a Qwen3 chat template from a local template, local
  `tokenizer_config.json`, or Hugging Face tokenizer config and adds
  Transformers `{% generation %}` tags for answer-only loss masking.
- `materialize_qwen3_static_inputs.py`
  stages verifier `config.json`/`tokenizer_config.json`, derives the Eagle3
  architecture JSON/env/dotlist, prepares the generation-tagged Qwen3 chat
  template, and records a static-input report without launching GPU work.
- `prepare_qwen3_chat_template.sh`
  wraps chat-template extraction/patching and assistant-mask validation into
  one command.
- `validate_chat_template_loss_mask.py`
  loads the tokenizer and template with Transformers and verifies that
  `return_assistant_tokens_mask=True` produces a positive assistant-token mask.
- `validate_hidden_state_dump.py`
  checks dumped `.pt` files for the keys/shapes expected by ModelOpt offline
  training and can optionally load them through ModelOpt's actual offline
  dataset/collator.
- `validate_modelopt_loss_mask_patch.py`
  statically verifies that the staged ModelOpt TRT-LLM dumper imports the
  answer-only helpers, computes `loss_mask`, writes it into TRT-LLM `.pt`
  hidden-state dumps, and that the Qwen3 dump wrapper passes
  `--answer-only-loss` and `--chat-template` in dry-run mode.
- `qwen3_235b_thinking_eagle3_architecture.json`
  records the draft architecture override used for Thinking-2507.
- `derive_eagle3_architecture.py`
  derives ModelOpt Eagle3 architecture JSON, shell env overrides, and OmegaConf
  dotlist overrides from an arbitrary verifier `config.json`. It mirrors
  ModelOpt's default EAGLE-3 aux layer rule.
- `compare_eagle3_configs.py`
  compares exported HF draft configs and converted vLLM one-checkpoint configs
  against the verifier and reference architecture.
- `inventory_eagle3_draft_configs.py`
  scans existing Eagle3 draft checkpoint/config directories, including
  Hayate/Hiso model artifacts, and reports whether their architecture fields
  match the Qwen3-235B Thinking reference. Its JSON includes `overall_status`
  and preserves permission/access warnings so inaccessible Hayate artifacts are
  explicit reference evidence rather than silent gaps.
- `audit_eagle3_readiness.py`
  summarizes the whole workstream state across local tooling, conversation
  data, offline/online ModelOpt recipe overrides, hidden-state dumps, ModelOpt
  training output, export output, smoke results, and Slurm dry-run dependencies.
- `audit_eagle3_completion.py`
  is the final evidence gate after cluster execution. It aggregates pipeline
  analysis, Megatron probe follow-up validation, hidden-state validation,
  trained/exported draft artifacts, config compare JSONs, and trained-draft
  sweep results into a pass/incomplete/fail completion report.
- `bootstrap_eagle3_path.sh`
  ties template prep, training-conversation prep, local preflight, Slurm
  pipeline planning, and readiness audit into one dry-run-first entry point.
  Use this when moving from the written plan to a concrete artifact directory.
- `preflight_eagle3_pipeline.py`
  checks local files, Qwen3-235B architecture defaults, conversation JSONL
  schema samples, env propagation, and wrapper dry-runs before expensive jobs
  are submitted.
- `validate_modelopt_recipe_overrides.py`
  checks that the offline and online training wrappers' OmegaConf dotlist
  overrides contain the required Qwen3-235B Eagle3 fields. In the training
  container it can also require a real `modelopt.recipe.load_recipe`
  validation.
- `validate_eagle3_operator_state_refresh.py`
  runs the broader no-submit operator-state refresh in a temporary artifact root
  and verifies that it preserves ModelOpt loss-mask validation, recipe override
  validation, and the corresponding goal-evidence requirements. It is separate
  from `validate_eagle3_preflight_robustness.py` to avoid refresh recursion.
- `validate_nemo_rl_specdec_integration.py`
  checks that the NeMo-RL Qwen3-235B config uses vLLM generation, that Eagle3
  Hydra overrides land under `policy.generation.vllm_kwargs.speculative_config`,
  and that the SpecDec-RL checkout contains the load-format/metric hooks needed
  for RL speculative-decoding validation. With
  `--integration-mode online-draft-training`, it also checks the NeMo-RL online
  constraints: Megatron enabled, DTensor disabled, sequence packing disabled,
  and visible `policy.draft`/draft-loss source support.
- `validate_rollout_capture_config.py`
  checks that the RL run will leave `train_data_step*.jsonl` under
  `logger.log_dir`, that the local normalizer can read SpecDec-RL flat
  `content`/`role` arrays, and whether the optional role-logging patch should
  be applied before collecting final Qwen3 rollout corpus.
- `preflight_rollout_capture_submit.py`
  combines the rollout-capture config gate with the actual capture wrapper and
  `run_grpo_qwen3_235b_swe.sh` dry-runs. A `submit_ready=true` report means the
  remaining missing item is the explicit GPU rollout-capture submission itself.
- `preflight_eagle3_pipeline_submit.py`
  combines the materialized rollout corpus, verifier config, chat template,
  ModelOpt wrapper dry-runs, container preflight report, and Slurm dependency
  dry-run into a single `submit_ready` gate for the hidden-state pipeline.
- `advance_rollout_capture_state.py`
  is the single command to run after submitting or completing rollout capture.
  It reads the Slurm/job report, optionally materializes the rollout corpus, and
  emits the next safe command for poll, materialize, or pipeline dry-run.
- `specdec_rl_rollout_role_logging.patch`
  is an idempotent-style `git apply` patch for SpecDec-RL `grpo.py` that adds
  `role` arrays beside the existing flat `content` arrays in train-data JSONL.
  Use it before production corpus capture when the checkout only logs content.
- `apply_specdec_rl_rollout_role_logging_patch.sh`
  checks whether that SpecDec-RL patch is already applied, whether it applies
  cleanly, and applies it only when `APPLY=true` is set.
- `analyze_specdec_smoke.py`
  parses NeMo-RL/vLLM smoke logs for step timings, throughput, and
  speculative decoding acceptance/draft metrics, with an optional baseline
  comparison gate.
- `analyze_eagle3_pipeline.py`
  reads `latest_eagle3_pipeline_jobs.txt` plus `logs/%x_%j.{out,err}` and
  summarizes the preflight -> dump -> validate -> train -> export pipeline.
  Use it immediately after a pilot submission to identify the first failed or
  incomplete stage.
- `analyze_spec_tokens_sweep.py`
  compares the trained-draft `num_speculative_tokens` sweep against its
  baseline and recommends the fastest passing setting.
- `analyze_static_specdec_smoke_pair.py`
  reads `latest_static_specdec_smoke_jobs.txt`, resolves `<jobid>-logs`, and
  runs the standard baseline-vs-specdec smoke analysis.
- `analyze_specforge_reference.py`
  records the SGLang/SpecForge Eagle3 reference script and explicitly marks it
  as reference-only for this ModelOpt/vLLM/NeMo-RL path.
- `analyze_hayate_modelopt_workflow.py`
  classifies the accessible Hayate/Hiso ModelOpt checkout. The currently
  visible workflow is DAPO-Math response generation plus legacy Qwen3
  8B/30B/32B Eagle3 training, so it is a math/bootstrap reference rather than a
  drop-in Qwen3-235B SWE/RL path.
- `inventory_hayate_eagle3_artifacts.sh`
  inventories Hayate/Hiso ModelOpt, SpecForge, and NeMo-RL paths from a host
  where the relevant Lustre paths are mounted.
- `modelopt_qwen3_235b_offline_train.sh`
  launches offline Eagle3 training with the current ModelOpt recipe API. It
  defaults `model.use_fake_base_for_offline=true` so ModelOpt loads its
  lightweight fake-base path instead of the full 235B verifier during
  draft-head training.
- `modelopt_qwen3_235b_online_train.sh`
  launches ModelOpt online Eagle3 training with `data.data_path` instead of
  precomputed hidden states. This is useful for matching Hayate's later
  workflow, but it keeps the 235B verifier in the training loop.
- `modelopt_qwen3_235b_export_vllm.sh`
  exports the trained ModelOpt checkpoint and converts it to vLLM format.
- `slurm_preflight.sbatch`
  Slurm template for the fast container-side preflight gate before expensive
  235B jobs.
- `slurm_dump_hidden_states.sbatch`
  Slurm template for distributed hidden-state dumping.
- `slurm_validate_hidden_states.sbatch`
  Slurm template for the cheap post-dump `.pt` validation gate before draft
  training starts.
- `slurm_offline_train.sbatch`
  Slurm template for ModelOpt offline draft training.
- `slurm_online_train.sbatch`
  Slurm template for optional ModelOpt online draft training with
  `data.data_path`.
- `slurm_export_vllm.sbatch`
  Slurm template for export and vLLM conversion.
- `submit_eagle3_pipeline.sh`
  dry-run/submission wrapper that chains preflight -> dump -> validate hiddens
  -> train -> export with Slurm `afterok` dependencies.
- `specdec_progress.html`
  is the human-readable status dashboard.
- `update_specdec_status_snapshot.py`
  updates the volatile queue/preflight sections in `specdec_progress.html`,
  `REMOTE_CLUSTER_STATUS.md`, and `REMOTE_EXECUTION_INPUTS.md` from the latest
  `reports/rollout_queue_wait_summary.json` and related operator reports. Run
  `refresh_eagle3_operator_state.py` first, then run this helper on the same
  host/artifact root.
- `SPECDEC_RL_REMOTE_PATCHES.md`
  records the current remote NeMo-RL patches needed for SWE/RL rollout capture:
  role-preserving GRPO logs, optional logger imports, and per-backend actor
  Python environment selection.

The top-level `run_grpo_qwen3_235b_swe.sh` now also supports optional
`EXTRA_HYDRA_OVERRIDES`, so smoke tests can append speculative decoding
overrides without editing the production defaults. It also accepts
`EXP_SUFFIX_OVERRIDE`, `WANDB_NAME`, `CHECKPOINT_SUBDIR`, and
`SBATCH_DEPENDENCY` so baseline/specdec smoke runs can use distinct job names,
checkpoint directories, and Slurm dependencies.

For the current OCI HSG container path, rollout capture also avoids full
`uv sync` and uses `/opt/venv/bin/python` directly. vLLM/MCore/NemoGym actor
env selection is controlled independently; the active shared-vLLM smoke uses
system actor envs for vLLM, MCore, and NemoGym, plus a no-deps precompiled vLLM
wheel install into a shared Lustre `python_site` target that is prepended to
`PYTHONPATH` before Ray starts actors. This avoids the failed uv source-build
path for `deep_ep` while keeping the container's torch/native library stack
intact and making `import vllm` visible on worker nodes. OCI HSG nodes are
`aarch64`, and the active NeMo 25.07.01 container probes as
`torch 2.8.0a0+nv25.05` / CUDA 12.9. A shallow pure-Python import made the
`vllm==0.10.2` target look usable at first, but the strengthened native probe
showed that vLLM `0.10.2`, `0.11.2`, and `0.13.0` wheel targets all fail
`import vllm._C` against this container with the same native
`c10::cuda::SetDevice` undefined symbol. The current shared-vLLM path must
therefore use a vLLM source build produced inside the target container, not a
pip wheel target, before the rollout smoke can be retried.

One local ModelOpt source patch was made for RL usefulness:

```text
Model-Optimizer/examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py
```

The TRT-LLM hidden-state dumper now supports the same `--answer-only-loss` and
`--chat-template` behavior as the HF dumper, and writes `loss_mask` into each
`.pt` output. This matters for SWE/RL conversations because the draft should
learn assistant tokens, not prompt/tool/context tokens indiscriminately.

All shell wrappers accept `DRY_RUN=true` to print the final command without
launching the expensive action.

The Slurm templates accept optional `CONTAINER` and `MOUNTS` env vars and pass
them to `srun --container-image/--container-mounts`, matching the style Hayate
used in the accessible ModelOpt scripts.

The ModelOpt training wrapper also now explicitly sets Eagle3 architecture
fields. This is necessary because simply passing aux layer ids is not enough:
`use_aux_hidden_state` must be true, and the draft attention/head shape should
match Qwen3-235B. The reference file in this directory uses:

```text
num_attention_heads=64
num_key_value_heads=4
intermediate_size=12288
head_dim=128
rms_norm_eps=1e-6
eagle_aux_hidden_state_layer_ids=[1,46,90]
use_aux_hidden_state=true
rope_theta=5000000
```

The public NVIDIA draft config confirms the Qwen3-235B Eagle3 shape
(`hidden_size=4096`, `num_attention_heads=64`, `num_key_value_heads=4`,
`intermediate_size=12288`, aux layers `[1,46,90]`), but the Thinking-2507
verifier config has `rope_theta=5000000`, so the Thinking-specific draft should
not blindly copy the public non-thinking draft rope settings:

- https://huggingface.co/nvidia/Qwen3-235B-A22B-Eagle3/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507/blob/main/config.json

For another verifier model, derive the architecture overrides from its
`config.json` instead of copying the Qwen3-235B constants:

```bash
python3 experiments/eagle3_qwen3_235b/derive_eagle3_architecture.py \
  --verifier-config /path/to/verifier/config.json \
  --json-out /path/to/eagle3_architecture.json \
  --env-out /path/to/eagle3_architecture.env \
  --dotlist-out /path/to/eagle3_architecture.dotlist

ARCH_ENV_FILE=/path/to/eagle3_architecture.env \
HIDDEN_STATES_DIR=/path/to/hidden_states \
OUTPUT_DIR=/path/to/modelopt_ckpt \
bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh
```

The aux layers are 0-based transformer-layer ids using ModelOpt's default
EAGLE-3 rule:

```text
sorted({1, max(0, num_hidden_layers // 2 - 1), max(0, num_hidden_layers - 4)})
```

For Qwen3-235B's 94 verifier layers, this gives `[1,46,90]`.

The same general workflow is captured in
`experiments/eagle3_qwen3_235b/EAGLE3_DRAFT_MODEL_PLAYBOOK.md`: collect the
verifier/tokenizer/data inputs, derive architecture, prepare answer-only
conversation data, dump and validate hidden states, train offline or with
ModelOpt online training, export, compare configs, and finally validate through
the RL generation stack.

## Public NVIDIA Draft Checkpoint

`nvidia/Qwen3-235B-A22B-Eagle3` exists on Hugging Face, but it targets
`Qwen/Qwen3-235B-A22B`, not `Qwen/Qwen3-235B-A22B-Thinking-2507`.

It is useful for a quick vLLM compatibility and acceptance-rate smoke test, but
it should not be treated as the final RL draft. The policy model, thinking
template, and SWE rollout distribution can all change acceptance rate.

## H100 vs GB200

Eagle3 draft training is not GB200-only. The draft model is small; the expensive
piece is running the 235B verifier to either generate training hidden states or
perform online training.

Practical implications:

- Static inference with an existing draft is feasible on H100 if vLLM supports
  the draft format and the generation workers have enough memory headroom.
- Offline training on H100 is possible if the verifier can run with enough
  tensor/pipeline parallelism and there is enough storage for hidden states.
- Online training for 235B is much more expensive because the verifier stays in
  the training loop. GB200-class systems are preferred; H100 may still work with
  enough nodes, but expect slower iteration and tighter memory scheduling.

Approximate hidden-state storage cost for Qwen3-235B with hidden size 4096:

```text
last hidden + 3 aux hidden states = 4 * 4096 bf16 values/token
                                 ~= 32 KiB/token
16k tokens/sample                ~= 512 MiB/sample before metadata/compression
32k tokens/sample                ~= 1 GiB/sample before metadata/compression
```

This is why Hayate experimented with aggregation and then moved training scripts
toward online training.

## RL Context Differences

In normal vLLM/TRT-LLM serving, the verifier is fixed. In RL, the verifier is
the policy and changes after updates. That creates three integration choices:

1. Static draft:
   train a draft against the initial policy/checkpoint and use it for rollout
   generation via `vllm_kwargs.speculative_config`.
   This is the lowest-risk first step.

2. Periodically refreshed draft:
   retrain or fine-tune the draft from recent rollout traces or policy
   checkpoints when acceptance falls.

3. Online/co-trained draft:
   train draft weights during RL from policy hidden states/logits and sync those
   weights to generation workers. This needs NeMo-RL integration beyond a vLLM
   config snippet.

Hayate's accessible NeMo-RL checkout shows static vLLM speculative config
examples. Source for a fuller Megatron-side specdec integration was not present
as checked-in `.py` files in the accessible tree; only stale `__pycache__`
artifacts hinted at hidden-state capture and Eagle3 Megatron modules. Treat that
as incomplete unless the real `oci-hsg` worktree can be accessed.

## Recommended Execution Path

### Phase -1: Bootstrap the Artifact Layout

On the cluster host where Lustre paths are mounted, first discover concrete
input candidates and write a sourceable env file:

```bash
ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
SBATCH_ACCOUNT=<account> \
bash experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh
```

The short host alias `oci-hsg` may not resolve from every workstation, and VPN
or DNS state can make all aliases temporarily unreachable. First run the remote
host probe to record the exact reachable alias and visible Hayate paths for the
current environment:

```bash
python3 experiments/eagle3_qwen3_235b/probe_eagle3_remote_host.py \
  --hosts oci-hsg-cs-001-vscode-02 oci-hsg-cs-001-vscode-01 oci-hsg-cs-001-login-01.nvidia.com oci-hsg \
  --remote-workdir /lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap \
  --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
  --json-out /path/to/eagle3_remote_host_probe.json \
  --markdown-out /path/to/eagle3_remote_host_probe.md
```

Use `--strict` when this is a required pre-submit gate; without it, the probe
still writes a structured `overall_status=unreachable` report that can be
attached to the handoff while waiting for DNS/VPN recovery. A final completion
audit requires `overall_status=pass`, at least one reachable host, visible
`git` and `python3`, and readable remote workdir, artifact-root,
TensorRT-Model-Optimizer, and Hayate draft/SpecForge paths. If the repo is
already present on the selected `oci-hsg` alias, launch the same safe dry-run
over SSH from this machine:

```bash
PRINT_ONLY=false \
REMOTE_HOST=oci-hsg-cs-001-vscode-02 \
REMOTE_WORKDIR=/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap \
REMOTE_ARTIFACT_ROOT=/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
SBATCH_ACCOUNT=<account> \
bash experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh
```

Keep `PRINT_ONLY=true` to inspect the SSH command without executing it. Set
`SYNC_EXPERIMENTS=true` only when the remote repo already exists but needs the
current `experiments/eagle3_qwen3_235b` scripts refreshed. Set
`SYNC_PROBE_JOB_FILE=true` to also copy
`latest_megatron_compat_probe_job.txt`, which is useful when resuming a
submitted Megatron compatibility probe such as `2867766`. The wrapper can run a
different remote script through `REMOTE_ENTRYPOINT`; for example, after DNS
recovers this print-only command refreshes scripts, syncs the probe job id, and
rebuilds the remote operator reports without executing any action:

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

To let that remote entrypoint run only the current non-Slurm/non-heavy gates,
keep `SUBMIT_ROLLOUT=false` and add:

```bash
EXECUTE_SAFE_ACTIONS=true \
SAFE_ACTION_IDS="probe_remote_hosts poll_megatron_compat_probe"
```

To submit only the current non-heavy runtime/container gates from the remote
cluster host after `eagle3_operator_ready_submit_preflight.json` is PASS, add a
narrow allowlist:

```bash
EXECUTE_SLURM_ACTIONS=true \
SLURM_ACTION_IDS="submit_vllm_source_build submit_source_vllm_abi_probe submit_container_preflight" \
RUN_AFTER_SLURM_ACTIONS=false \
ALLOW_HEAVY_GPU_ACTIONS=false
```

Add `RUN_FULL_REFRESH=true` after the safe actions when the remote host should
also regenerate ModelOpt loss-mask/recipe evidence, Hayate reference analyses,
draft inventory, goal evidence, and completion audit reports. On a workstation
without Slurm, set `FULL_REFRESH_SKIP_REMOTE_HOST_PROBE=true` for local dry-run
debugging only; remote completion evidence still requires a PASS remote-host
probe.

The Megatron follow-up still only prints the rollout retry unless the separate
heavy-GPU gate is set with both `SUBMIT_ROLLOUT=true` and
`ALLOW_HEAVY_GPU=true`.

Set
`SSH_PROXY_JUMP=<jump-host>` or `SSH_EXTRA_OPTS="-o Key=Value"` if the selected
host is reachable only through a jump configuration.

The current staged remote workspace is:

```text
REMOTE_HOST=oci-hsg-cs-001-vscode-02
REMOTE_WORKDIR=/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap
REMOTE_ARTIFACT_ROOT=/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3
MODELOPT_DIR=/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap/Model-Optimizer
```

That `MODELOPT_DIR` is official `NVIDIA/Model-Optimizer@b02e8885...` with the
local TRT-LLM `loss_mask` patch applied. A safe remote dry-run completed through
handoff creation. The readiness audit is still `INCOMPLETE` because real
`SBATCH_ACCOUNT`, hidden states, checkpoint, and export artifacts do not exist
yet. Verifier metadata, a Qwen3 generation-template file, and short pilot
conversation JSONLs are staged, but the final training data still needs
Thinking-2507-generated or RL-rollout assistant responses.

That entrypoint keeps `SUBMIT=false` by default. It discovers inputs, probes
the cluster substrate, captures Hayate/provenance/upstream-drift reports,
exports the local ModelOpt patch bundle, runs the normal bootstrap dry-run, and writes
`ARTIFACT_ROOT/handoff/RUNBOOK.md`.

The lower-level discovery command is:

```bash
python3 experiments/eagle3_qwen3_235b/discover_eagle3_run_inputs.py \
  /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso \
  /lustre/fsw/portfolios/coreai/users/sna \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --env-out /path/to/qwen3_235b_eagle3/eagle3_inputs.env \
  --markdown-out /path/to/qwen3_235b_eagle3/eagle3_input_discovery.md \
  --json-out /path/to/qwen3_235b_eagle3/eagle3_input_discovery.json

source /path/to/qwen3_235b_eagle3/eagle3_inputs.env
```

Start with a no-GPU dry-run that resolves the artifact paths and prints the
data/template/pipeline commands without submitting Slurm jobs:

```bash
ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
SBATCH_ACCOUNT=<account> \
VERIFIER_CONFIG_DIR=/path/to/Qwen3-235B-A22B-Thinking-2507 \
bash experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh
```

To materialize only the static Qwen3 inputs on a lightweight host, run:

```bash
ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
ALLOW_MISSING_TRANSFORMERS=true \
python3 experiments/eagle3_qwen3_235b/materialize_qwen3_static_inputs.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --allow-missing-transformers
```

This writes `verifier_config/config.json`, `verifier_config/tokenizer_config.json`,
`architecture/eagle3_architecture.{json,env,dotlist}`, a generation-tagged chat
template, and `reports/qwen3_static_inputs.{json,md}`. A lightweight host may
report `WARN` if Transformers is unavailable; repeat the mask validation in the
target container before the hidden-state dump.

If `VERIFIER_CONFIG_DIR/config.json` is visible, the bootstrap also derives
`ARTIFACT_ROOT/architecture/eagle3_architecture.{json,env,dotlist}` and passes
the env file through the Slurm pipeline. For Qwen3 this reproduces the checked
reference; for another LLaMA-like verifier it prevents copying stale Qwen3
constants into the draft architecture.
When `PREP_DRY_RUN=false` and `VERIFIER_CONFIG_DIR/config.json` is not visible,
`bootstrap_eagle3_path.sh` also runs the static-input materializer by default.
Use `RUN_STATIC_INPUT_PREP=false` to skip that network step, or
`RUN_STATIC_INPUT_PREP=true STATIC_INPUT_SOURCE_DIR=/path/to/verifier_snapshot`
to force materialization from a local snapshot.

When rollout data or prompts are available, materialize the local inputs first:

```bash
PREP_DRY_RUN=false \
MODE=rollout \
INPUT_PATHS="/path/to/rollouts.jsonl /path/to/trajectory_dir" \
TOKENIZER_CONFIG=/path/to/Qwen3-235B-A22B-Thinking-2507/tokenizer_config.json \
ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
SBATCH_ACCOUNT=<account> \
VERIFIER_CONFIG_DIR=/path/to/Qwen3-235B-A22B-Thinking-2507 \
bash experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh
```

Keep `SUBMIT=false` until the readiness audit reports only expected heavy
artifacts as missing. Then submit the chained pipeline:

```bash
SUBMIT=true \
PREP_DRY_RUN=false \
RUN_PILOT=true \
RUN_TRAINED_DRAFT_SMOKE=true \
RUN_TRAINED_DRAFT_SWEEP=true \
ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
SBATCH_ACCOUNT=<account> \
VERIFIER_CONFIG_DIR=/path/to/Qwen3-235B-A22B-Thinking-2507 \
bash experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh
```

Use `RUN_PILOT=true` for the first GPU attempt. It keeps the same pipeline but
defaults to a small hidden-state dump and short training run:

```text
DEBUG_MAX_NUM_CONVERSATIONS=8
DATA_SAMPLE_SIZE=8
MAX_STEPS=20
SAVE_STEPS=20
DUMP_TIME=02:00:00
TRAIN_TIME=02:00:00
EXPORT_TIME=01:00:00
```

Override any of those env vars if the pilot needs to be larger. Only remove
`RUN_PILOT=true` after the pilot proves chat-template masking, aux-layer hidden
state shapes, ModelOpt offline loader compatibility, export, and vLLM config
comparison.

For the Qwen3-235B SWE/RL path, the first real training input is the normalized
rollout corpus
`/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl`
or the compact fallback file with the same prefix. DAPO/OpenMath-style math data
is useful as a reference or supplemental corpus only when the target domain is
math; it is not the primary corpus for this SWE/RL draft. After the 8-conversation
pilot, use the 2,438-row SWE-Gym rollout as the first calibration, then collect
10k-50k target-domain SWE/RL rollout conversations if acceptance improves. The
first serious draft candidate should target 50k-100k+ SWE/RL conversations.
Only move to 300k-500k if the goal changes from this RL loop to a broad reusable
Qwen3-235B draft. This is also consistent with TAPS-style task-aware draft
results: math corpora such as MathInstruct/DAPO/OpenMathInstruct help math
acceptance, but they are not evidence that a SWE/RL rollout draft will accept
well unless the SWE/RL distribution is represented.

After a trained draft exists, sweep `num_speculative_tokens` before using it for
longer RL runs:

```bash
SUBMIT=false \
ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL \
SWE_REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL \
CONFIG_FILE=/path/to/Nemo-RL_Qwen3_Roadmap/grpo_qwen3_235b_swe.yaml \
ENV_FILE=/path/to/Nemo-RL_Qwen3_Roadmap/env.sh \
CHAT_TEMPLATE=/path/to/qwen3_generation_template.jinja2 \
VLLM_DRAFT_DIR=/path/to/qwen3_235b_eagle3_vllm \
SPEC_TOKENS_LIST="2 3 4" \
MAX_NUM_STEPS=2 \
EAGLE3_DRAFT_TP=1 \
bash experiments/eagle3_qwen3_235b/submit_trained_draft_spec_tokens_sweep.sh

python3 experiments/eagle3_qwen3_235b/analyze_spec_tokens_sweep.py \
  --job-file latest_trained_draft_spec_tokens_sweep_jobs.txt \
  --repo-root /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL \
  --markdown-out /path/to/trained_draft_spec_tokens_sweep.md \
  --json-out /path/to/trained_draft_spec_tokens_sweep.json \
  --fail-on-missing-spec-metrics
```

### Phase 0: Static Draft Smoke Test

Before training, try the public draft only to establish whether the local vLLM
and NeMo-RL stack accepts Eagle3 config:

```yaml
policy:
  generation:
    vllm_kwargs:
      speculative_config:
        method: eagle3
        model: nvidia/Qwen3-235B-A22B-Eagle3
        num_speculative_tokens: 3
        draft_tensor_parallel_size: 1
```

Wrapper:

```bash
WANDB_NAME=qwen3-235b-swe-baseline-smoke \
MAX_NUM_STEPS=1 \
bash experiments/eagle3_qwen3_235b/run_baseline_smoke.sh

EAGLE3_DRAFT_MODEL=nvidia/Qwen3-235B-A22B-Eagle3 \
EAGLE3_NUM_SPEC_TOKENS=3 \
EAGLE3_DRAFT_TP=1 \
MAX_NUM_STEPS=1 \
bash experiments/eagle3_qwen3_235b/run_static_specdec_smoke.sh
```

The pair wrapper prints both commands by default and, with `SUBMIT=true`, runs
the static Eagle3 smoke after the baseline job succeeds:

```bash
SUBMIT=false \
MAX_NUM_STEPS=1 \
BASELINE_WANDB_NAME=qwen3-235b-swe-baseline-smoke \
SPECDEC_WANDB_NAME=qwen3-235b-swe-eagle3-public-smoke \
bash experiments/eagle3_qwen3_235b/submit_static_specdec_smoke_pair.sh
```

Measure:

- acceptance rate if exposed by vLLM metrics,
- `exposed_generation`,
- total RL step time,
- invalid/malformed thinking rate,
- reward/regression risk.

After the job finishes, parse either the full `ray-driver.log`, Slurm output,
or an output directory:

```bash
python3 experiments/eagle3_qwen3_235b/analyze_static_specdec_smoke_pair.py \
  --repo-root /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL \
  --markdown-out /path/to/specdec_smoke_summary.md \
  --json-out /path/to/specdec_smoke_summary.json
```

Or pass explicit log paths:

```bash
python3 experiments/eagle3_qwen3_235b/analyze_specdec_smoke.py \
  /path/to/specdec-smoke-logs \
  --baseline /path/to/baseline-logs \
  --gen-outlier-threshold-s 800 \
  --min-generation-speedup-pct 10 \
  --min-acceptance-rate 0.45 \
  --markdown-out /path/to/specdec_smoke_summary.md \
  --json-out /path/to/specdec_smoke_summary.json
```

For multi-step smoke runs, add `--drop-first-step` to remove cold-start noise.

The acceptance metric may be absent even when speculative decoding is enabled,
depending on the vLLM build and where metrics are emitted. Treat missing
acceptance as a logging gap first, then verify via vLLM metrics endpoint or
worker logs before calling the smoke test a failure.

### Phase 1: Build Target-Domain Training Conversations

Use the same model, chat template, and rollout distribution as the target RL
run. Good first dataset choices:

- SWE prompts plus assistant outputs generated by
  `Qwen/Qwen3-235B-A22B-Thinking-2507`,
- existing successful/failed RL rollout trajectories if they are logged,
- a DAPO/Math mix only as supplemental data, not as the only source.

For a math-target speculative draft, DAPO/OpenMathInstruct-style data can be
the primary corpus. For the current Qwen3 SWE/RL target, it is not enough by
itself because draft acceptance depends strongly on the generation
distribution. Capture this decision in a report:

```bash
python3 experiments/eagle3_qwen3_235b/analyze_corpus_strategy.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --target-context swe_rl \
  --input-data /path/to/qwen3_235b_swe_conversations.jsonl \
  --rollout-capture-analysis-json /path/to/qwen3_235b_eagle3/reports/rollout_capture_analysis.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/corpus_strategy.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/corpus_strategy.json
```

Use OpenAI-style records with `messages` or `conversations` and a stable
`conversation_id`. For RL, prefer answer-only loss if the template exposes
assistant generation tags.

Use the wrapper first unless you need to debug a specific parser:

```bash
MODE=discover \
ROLLOUT_ROOTS="/path/to/nemo_rl_run /path/to/another_run" \
OUTPUT_DATA=/path/to/qwen3_235b_swe_conversations.jsonl \
bash experiments/eagle3_qwen3_235b/prepare_training_conversations.sh
```

If you already know the rollout files:

```bash
MODE=rollout \
INPUT_PATHS="/path/to/rollouts.jsonl /path/to/trajectories_dir" \
OUTPUT_DATA=/path/to/qwen3_235b_swe_conversations.jsonl \
bash experiments/eagle3_qwen3_235b/prepare_training_conversations.sh
```

If the rollout schema stores assistant reasoning separately, preserve that
distribution for Qwen3 Thinking data:

```bash
MODE=rollout \
INPUT_PATHS="/path/to/rollouts_with_reasoning_content.jsonl" \
OUTPUT_DATA=/path/to/qwen3_235b_swe_conversations_with_think.jsonl \
INCLUDE_REASONING_CONTENT=true \
bash experiments/eagle3_qwen3_235b/prepare_training_conversations.sh
```

If you only have prompts and need fresh Thinking-2507 outputs from a vLLM or
OpenAI-compatible endpoint:

```bash
MODE=generate \
PROMPT_DATA=/path/to/swe_prompts.jsonl \
OPENAI_BASE_URL=http://localhost:8000/v1 \
MODEL_PATH=Qwen/Qwen3-235B-A22B-Thinking-2507 \
OUTPUT_DATA=/path/to/qwen3_235b_swe_conversations.jsonl \
NUM_RESPONSES=1 \
bash experiments/eagle3_qwen3_235b/prepare_training_conversations.sh
```

The wrapper writes `${OUTPUT_DATA%.jsonl}.validation.json` with row counts,
length estimates, and hidden-state storage estimates. The hidden-state dump
phase should not start until this validation passes.

Generation wrapper:

```bash
python3 experiments/eagle3_qwen3_235b/generate_training_conversations_openai.py \
  --input /path/to/swe_prompts.jsonl \
  --output /tmp/unused.jsonl \
  --inspect-only
```

For raw SWE/NemoGym rows with `problem_statement`/`instance_id`/`repo`:

```bash
python3 experiments/eagle3_qwen3_235b/generate_training_conversations_openai.py \
  --input /path/to/swe_prompts.jsonl \
  --output /path/to/qwen3_235b_swe_conversations.jsonl \
  --api-base http://localhost:8000/v1 \
  --model Qwen/Qwen3-235B-A22B-Thinking-2507 \
  --num-responses 4 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-tokens 16384 \
  --append
```

If the source is already a rollout log with assistant messages, first inspect
how many conversations can be extracted:

```bash
python3 experiments/eagle3_qwen3_235b/validate_rollout_capture_config.py \
  --config grpo_qwen3_235b_swe.yaml \
  --specdec-rl-dir /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --chat-template /path/to/qwen3_235b_eagle3/templates/qwen3_generation_template.jinja2 \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_validation.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_validation.json \
  --env-out /path/to/qwen3_235b_eagle3/reports/rollout_capture.env

APPLY=false \
SPECDEC_RL_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL \
bash experiments/eagle3_qwen3_235b/apply_specdec_rl_rollout_role_logging_patch.sh

python3 experiments/eagle3_qwen3_235b/discover_rollout_conversation_sources.py \
  /path/to/nemo_rl_outputs_or_logs \
  --include-all-jsonl \
  --markdown-out /path/to/rollout_source_discovery.md \
  --json-out /path/to/rollout_source_discovery.json

python3 experiments/eagle3_qwen3_235b/discover_rollout_conversation_sources.py \
  /path/to/nemo_rl_outputs_or_logs \
  --include-all-jsonl \
  --top-k 4 \
  --prepare-output /path/to/qwen3_235b_swe_conversations.jsonl \
  --include-metadata

python3 experiments/eagle3_qwen3_235b/normalize_rl_rollouts_to_conversations.py \
  --input /path/to/train_data_step*.jsonl \
  --output /tmp/unused.jsonl \
  --inspect-only
```

The rollout normalizer also understands OpenAI Responses API style rows with
`responses_create_params.input` plus `response.output[*].content[*].text`, and
the common extracted response fields `model_output` and
`extracted_model_output`. This is useful for existing SWE/code rollout logs such
as `swerl_gen` and `code_gen`: they can be used as schema/reference material,
but they are not a substitute for Qwen3-235B SWE/RL rollout data unless their
model/provenance matches the target policy.

For Codex/SWE traces that include long tool histories, add
`--compact-current-turn` or set `COMPACT_CURRENT_TURN=true` through
`prepare_training_conversations.sh`. That keeps system/developer instructions
plus the final user turn and assistant response, while dropping prior assistant
and tool/function turns that would otherwise bloat a 16k Eagle3 training row.

Current SpecDec-RL writes compact `train_data_step*.jsonl` when
`env.should_log_nemo_gym_responses=false`. The Qwen3 SWE config already uses
that setting. For Eagle3 corpus quality, prefer applying
`specdec_rl_rollout_role_logging.patch` to SpecDec-RL before the capture run so
the logs contain both `content` and `role`; otherwise the normalizer can fall
back to `--infer-flat-content-roles`, which treats the final content item as the
assistant response and is intentionally lossy.

Plan the short capture run without submitting:

```bash
python3 experiments/eagle3_qwen3_235b/preflight_rollout_capture_submit.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --repo-root /path/to/SpecDec-RL \
  --config grpo_qwen3_235b_swe.yaml \
  --chat-template /path/to/qwen3_235b_eagle3/templates/qwen3_generation_template.jinja2 \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_submit_preflight.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_submit_preflight.json

DRY_RUN=true \
ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
ROLLOUT_LOG_DIR=/path/to/qwen3_235b_eagle3/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke \
OUTPUT_CONVERSATIONS=/path/to/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl \
bash experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh
```

After the job is explicitly submitted and completed, materialize the corpus:

```bash
python3 experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --repo-root /path/to/SpecDec-RL \
  --rollout-log-dir /path/to/qwen3_235b_eagle3/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke \
  --output-data /path/to/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl \
  --materialize \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_state_advance.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_state_advance.json

ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
ROLLOUT_LOG_DIR=/path/to/qwen3_235b_eagle3/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke \
OUTPUT_DATA=/path/to/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl \
bash experiments/eagle3_qwen3_235b/materialize_rollout_capture_corpus.sh
```

The rollout materialization watcher is non-submitting while the capture job is
pending. New watcher runs also refresh the rollout state report periodically
with `RUN_PENDING_STATE_REFRESH=true` and `PENDING_STATE_REFRESH_POLLS=5`, so
long Slurm queue waits do not leave `rollout_capture_state_advance.json` stale.
For already-running materialization watchers that were launched before that
logic existed, use `watch_rollout_pending_state_refresh.sh` as a separate
non-submitting heartbeat. It exits when the job leaves `squeue`; terminal
materialization remains owned by `watch_rollout_capture_materialize.sh`.
`summarize_rollout_queue_wait.py` reads both watcher logs and current `squeue`
state into `reports/rollout_queue_wait_summary.{json,md}`; it is included in
`refresh_eagle3_operator_state.py` so the operator sheet can distinguish a
normal long queue wait from missing rollout evidence. It also estimates watcher
timeout risk from watcher start time, `POLL_SECONDS`, and `MAX_POLLS`; if Slurm
start estimates drift beyond the watcher window, the queue report turns WARN so
the watcher can be restarted before terminal rollout handling is missed.
`summarize_rollout_watcher_health.py` writes
`reports/rollout_watcher_health.{json,md}` from the materialization,
pending-state, and operator follow-up PID files. It uses the queue-wait summary
and rollout state reports to decide which watchers are required now, so
pending-state helper processes are required while their Slurm jobs are active
but may exit normally after terminal state. A missing pipeline watcher is not a
problem before the hidden-state/train/export pipeline has been submitted. Active
generic rollout state reports are also freshness-checked; if a generic watcher
is alive but stops refreshing its dynamic `*_state_advance.json` for more than
15 minutes, health turns WARN instead of masking the stale poll state. If a
lock-waiting current-code extension watcher is started for an already-running
generic rollout, its launcher PID is tracked as an optional health row so a
stale pre-update watcher cannot silently become the only terminal handler.
`ensure_rollout_watchers.py` writes
`reports/rollout_watcher_ensure.{json,md}` and is also part of
`refresh_eagle3_operator_state.py`. It is no-submit by default: when required
watchers are missing or dead, it emits the exact background restart commands.
`validate_full_rollout_gate.py` includes a synthetic start-watcher scenario, but
that synthetic watcher uses `watcher-max-polls=0` and cleans its temp root so
the validator cannot recursively spawn temp operator refresh processes.
`preflight_eagle3_pipeline_submit.py` also refuses `RUN_PILOT=true` submit
readiness unless the input corpus has at least `MIN_PIPELINE_PILOT_ROWS`
valid conversations, defaulting to `DATA_SAMPLE_SIZE`/8. This keeps the
five-row rollout smoke in the runtime/capture proof lane and prevents it from
being mistaken for an Eagle3 pilot-training corpus.
`plan_eagle3_next_actions.py` reads `full_swegym_after_smoke_gate.json`; after
the smoke passes, if the full SWE-Gym preflight is ready, the next ready action
becomes `submit_full_swegym_rollout`. The transition validator has an explicit
`full_rollout_ready` scenario so this path does not regress back to a 5-row
pipeline submit.
`submit_full_rollout_after_smoke_if_ready.py` also uses wide `squeue` job-name
output and a validated parser to detect an already-active full rollout before
emitting another heavy submission path.
With `--execute --allow-background`, it starts only the missing watcher
processes and writes their PID files.

Before submitting hidden-state dump/train/export, run the pipeline submit
preflight. This is expected to be incomplete until the materialized rollout
corpus and container preflight are both present. After those two gates pass,
`plan_eagle3_next_actions.py` emits this as `run_pipeline_submit_preflight`
with `submits_slurm=false` and `heavy_gpu=false`:

```bash
python3 experiments/eagle3_qwen3_235b/validate_eagle3_next_action_transitions.py \
  --json-out /path/to/qwen3_235b_eagle3/reports/eagle3_next_action_transitions.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/eagle3_next_action_transitions.md

python3 experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline_submit.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --input-data /path/to/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl \
  --verifier-config-dir /path/to/qwen3_235b_eagle3/verifier_config \
  --chat-template /path/to/qwen3_235b_eagle3/templates/qwen3_generation_template.jinja2 \
  --modelopt-dir /path/to/Model-Optimizer \
  --container-preflight-json /path/to/qwen3_235b_eagle3/reports/container_preflight_analysis.json \
  --corpus-strategy-json /path/to/qwen3_235b_eagle3/reports/corpus_strategy.json \
  --rollout-state-json /path/to/qwen3_235b_eagle3/reports/rollout_capture_state_advance.json \
  --sbatch-account coreai_dlalgo_nemorl \
  --sbatch-partition batch \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/eagle3_pipeline_submit_preflight.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/eagle3_pipeline_submit_preflight.json
```

When wrapping an actual `SUBMIT=true` pipeline launch, add
`--fail-if-not-ready`; `run_eagle3_cluster_pilot.sh` and
`bootstrap_eagle3_path.sh` do this automatically when `SUBMIT=true` is set.

At any point, summarize the rollout artifact state and next command:

```bash
python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --rollout-log-dir /path/to/qwen3_235b_eagle3/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke \
  --output-data /path/to/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_analysis.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_analysis.json
```

After the capture job has been submitted, inspect the job/log/corpus state:

```bash
python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --repo-root /path/to/SpecDec-RL \
  --rollout-log-dir /path/to/qwen3_235b_eagle3/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke \
  --output-data /path/to/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_job_analysis.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/rollout_capture_job_analysis.json
```

Then normalize it directly:

```bash
python3 experiments/eagle3_qwen3_235b/normalize_rl_rollouts_to_conversations.py \
  --input /path/to/train_data_step*.jsonl \
  --output /path/to/qwen3_235b_swe_conversations.jsonl \
  --include-metadata
```

Validate the generated conversation file before any 235B hidden-state dump:

```bash
python3 experiments/eagle3_qwen3_235b/validate_training_conversations.py \
  /path/to/qwen3_235b_swe_conversations.jsonl \
  --max-seq-len 16384 \
  --json-out /path/to/qwen3_235b_swe_conversations.validation.json
```

The validator fails on malformed JSONL, missing/duplicate `conversation_id`,
missing `messages`, unsupported roles, and missing assistant content. It warns
on overlength samples by default; add `--fail-on-overlength` if the dump should
hard-stop on samples above the target sequence length. Passing `--tokenizer`
enables exact chat-template token counting when the target container has
Transformers available; otherwise it uses a conservative character-based
estimate.

For simple flat rollout logs, the OpenAI generator can also reuse an explicit
assistant response field:

```bash
python3 experiments/eagle3_qwen3_235b/generate_training_conversations_openai.py \
  --input /path/to/rollout_logs.jsonl \
  --output /path/to/qwen3_235b_swe_conversations.jsonl \
  --use-existing-assistant \
  --response-key response \
  --append
```

### Phase 2: Dump Hidden States

For 235B, prefer TensorRT-LLM or another distributed backend for hidden-state
dumping. The local HF dumper is simple but likely too slow/heavy for full
Qwen3-235B.

Run `validate_training_conversations.py` first. This is intentionally cheap and
catches data-shape failures before occupying verifier GPUs.

If `ANSWER_ONLY_LOSS=true`, prepare a chat template with Transformers
generation tags. The existing NeMo-RL YAML points at a
`qwen3_fixed_template.jinja2`, but that path may be host-specific. You can
create a local equivalent from a downloaded tokenizer config or directly from
Hugging Face:

```bash
TOKENIZER_CONFIG=/path/to/Qwen3-235B-A22B-Thinking-2507/tokenizer_config.json \
OUTPUT_TEMPLATE=/path/to/qwen3_generation_template.jinja2 \
bash experiments/eagle3_qwen3_235b/prepare_qwen3_chat_template.sh

BASE_MODEL=Qwen/Qwen3-235B-A22B-Thinking-2507 \
OUTPUT_TEMPLATE=/path/to/qwen3_generation_template.jinja2 \
bash experiments/eagle3_qwen3_235b/prepare_qwen3_chat_template.sh

python3 experiments/eagle3_qwen3_235b/prepare_qwen3_generation_template.py \
  --tokenizer-config /path/to/Qwen3-235B-A22B-Thinking-2507/tokenizer_config.json \
  --output /path/to/qwen3_generation_template.jinja2

python3 experiments/eagle3_qwen3_235b/prepare_qwen3_generation_template.py \
  --model Qwen/Qwen3-235B-A22B-Thinking-2507 \
  --output /path/to/qwen3_generation_template.jinja2
```

Set `CHAT_TEMPLATE=/path/to/qwen3_generation_template.jinja2` for hidden-state
dumping. The helper wraps Qwen-style assistant branches with generation tags so
`return_assistant_tokens_mask=True` can produce `loss_mask`.
The Qwen3 dump wrapper now defaults `ANSWER_ONLY_LOSS=true` and fails before
launching the verifier if `CHAT_TEMPLATE` is missing or lacks
`generation/endgeneration` tags. Set `ANSWER_ONLY_LOSS=false` only for an
intentional full-token-loss experiment.

Validate the template in the target container before hidden-state dump:

```bash
python3 experiments/eagle3_qwen3_235b/validate_chat_template_loss_mask.py \
  --model-or-tokenizer Qwen/Qwen3-235B-A22B-Thinking-2507 \
  --chat-template /path/to/qwen3_generation_template.jinja2
```

The Slurm preflight requires this check by default through
`PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK=true`, because the dump job otherwise
discovers template mistakes only after allocating verifier GPUs.
The shell wrapper uses strict validation by default; set
`ALLOW_MISSING_TRANSFORMERS=true` only on lightweight hosts where Transformers
is intentionally unavailable and repeat the validation in the target container.

Auxiliary layer ids should match the draft config. For a 94-layer Qwen3-235B
verifier, Eagle3 convention gives:

```text
[1, 46, 90]
```

The current local HF hidden-state script uses 0-based transformer-layer ids and
then indexes `outputs.hidden_states[lid + 1]`.

Wrapper:

```bash
INPUT_DATA=/path/to/qwen3_235b_swe_conversations.jsonl \
HIDDEN_STATES_DIR=/path/to/qwen3_235b_eagle3_hidden_states \
BACKEND=trtllm \
TP=8 \
DP_WORLD_SIZE=8 \
DP_RANK=${SLURM_PROCID:-0} \
ANSWER_ONLY_LOSS=true \
CHAT_TEMPLATE=/path/to/qwen3_generation_template.jinja2 \
bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_dump_hidden_states.sh
```

Slurm template:

```bash
sbatch --account=<account> \
  --export=ALL,INPUT_DATA=/path/to/qwen3_235b_swe_conversations.jsonl,HIDDEN_STATES_DIR=/path/to/qwen3_235b_eagle3_hidden_states,CHAT_TEMPLATE=/path/to/qwen3_fixed_template.jinja2 \
  experiments/eagle3_qwen3_235b/slurm_dump_hidden_states.sbatch
```

Use `BACKEND=hf DEBUG_MAX_NUM_CONVERSATIONS=2` only for small local dry-runs.
For full Qwen3-235B, prefer TRT-LLM or a similarly distributed backend.

Validate a hidden-state shard before launching training:

```bash
python3 experiments/eagle3_qwen3_235b/validate_hidden_state_dump.py \
  /path/to/qwen3_235b_eagle3_hidden_states \
  --require-loss-mask \
  --require-positive-loss-mask \
  --expected-hidden-size 4096 \
  --expected-aux-count 3 \
  --max-seq-len 16384 \
  --validate-modelopt-loader \
  --modelopt-dir /path/to/Model-Optimizer
```

The Slurm pipeline runs the same check automatically between dump and train by
default. It also sets `VALIDATE_MODELOPT_LOADER=true`, so the validation imports
ModelOpt's `OfflineSupervisedDataset` and `EagleOfflineDataCollator` in the same
container environment that will launch training:

```bash
sbatch --account=<account> \
  --export=ALL,HIDDEN_STATES_DIR=/path/to/qwen3_235b_eagle3_hidden_states \
  experiments/eagle3_qwen3_235b/slurm_validate_hidden_states.sbatch
```

### Phase 3: Train Offline Draft with Current ModelOpt

Use `modelopt_qwen3_235b_offline_train.sh` in this directory as the starting
point. It uses the current recipe API in this workspace instead of Hayate's old
legacy CLI.

Start with:

- `training.training_seq_len=16384`
- `eagle.eagle_ttt_steps=3`
- `training.per_device_train_batch_size=1`
- `training.learning_rate=1e-4`
- `training.num_train_epochs=1`
- `eagle.eagle_freeze_base_model=true`

Then sweep `num_speculative_tokens` in vLLM at 2, 3, and 4. Longer speculative
chains only help if acceptance stays high.

Wrapper:

```bash
HIDDEN_STATES_DIR=/path/to/qwen3_235b_eagle3_hidden_states \
OUTPUT_DIR=/path/to/qwen3_235b_eagle3_modelopt \
ANSWER_ONLY_LOSS=true \
USE_FAKE_BASE_FOR_OFFLINE=true \
bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh
```

Slurm template:

```bash
sbatch --account=<account> \
  --export=ALL,HIDDEN_STATES_DIR=/path/to/qwen3_235b_eagle3_hidden_states,OUTPUT_DIR=/path/to/qwen3_235b_eagle3_modelopt,ANSWER_ONLY_LOSS=true,USE_FAKE_BASE_FOR_OFFLINE=true \
  experiments/eagle3_qwen3_235b/slurm_offline_train.sbatch
```

The wrapper defaults to full-vocab draft training. If a 32k draft vocabulary is
desired later, calibrate it with ModelOpt's `scripts/calibrate_draft_vocab.py`
and set both `data.draft_vocab_cache` and
`eagle.eagle_architecture_config.draft_vocab_size`.

Optional ModelOpt online training wrapper:

```bash
INPUT_DATA=/path/to/qwen3_235b_swe_conversations.jsonl \
OUTPUT_DIR=/path/to/qwen3_235b_eagle3_modelopt_online \
ANSWER_ONLY_LOSS=true \
CHAT_TEMPLATE=/path/to/qwen3_generation_template.jinja2 \
bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_online_train.sh
```

Slurm template:

```bash
sbatch --account=<account> --nodes=<nodes> \
  --export=ALL,INPUT_DATA=/path/to/qwen3_235b_swe_conversations.jsonl,OUTPUT_DIR=/path/to/qwen3_235b_eagle3_modelopt_online,ANSWER_ONLY_LOSS=true,CHAT_TEMPLATE=/path/to/qwen3_generation_template.jinja2 \
  experiments/eagle3_qwen3_235b/slurm_online_train.sbatch
```

This uses `data.data_path` and lets ModelOpt run the verifier forward during
draft training. It matches the direction of Hayate's later scripts more closely
than the offline dump/train path, but it is much heavier for 235B. Use it on
GB200-class capacity or large H100 allocations. For a first Qwen3-235B SWE
draft, the offline path remains the recommended default because it separates
hidden-state generation, validation, and draft training into restartable stages.
When `ANSWER_ONLY_LOSS=true`, pass a Qwen3 chat template with
`{% generation %}` tags through `CHAT_TEMPLATE`; the online collator needs it to
build assistant-token labels. The one-command Slurm pipeline below intentionally
uses the offline path; submit `slurm_online_train.sbatch` separately when the
online verifier-in-loop path is the selected experiment.

### Phase 4: Export and Convert

After training with ModelOpt:

```bash
python3 experiments/eagle3_qwen3_235b/validate_eagle3_training_checkpoint.py \
  --checkpoint-dir "$TRAINED_CKPT" \
  --modelopt-dir "$MODELOPT_DIR" \
  --reference-arch /path/to/eagle3_architecture.json \
  --json-out /path/to/qwen3_235b_eagle3/reports/eagle3_training_checkpoint.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/eagle3_training_checkpoint.md \
  --require-modelopt-state-load \
  --fail-on-error

cd Model-Optimizer/examples/speculative_decoding
python scripts/export_hf_checkpoint.py \
  --model_path "$TRAINED_CKPT" \
  --export_path "$EXPORT_DIR"

python scripts/convert_to_vllm_ckpt.py \
  --input "$EXPORT_DIR" \
  --verifier "$BASE_MODEL_OR_LOCAL_VERIFIER_CONFIG" \
  --output "$VLLM_DRAFT_DIR"

python3 experiments/eagle3_qwen3_235b/validate_eagle3_export_artifacts.py \
  --export-dir "$EXPORT_DIR" \
  --vllm-draft-dir "$VLLM_DRAFT_DIR" \
  --verifier-config-dir "$VERIFIER_CONFIG_DIR" \
  --reference-arch /path/to/eagle3_architecture.json \
  --export-config-compare-json "$EXPORT_DIR/config_compare.json" \
  --vllm-config-compare-json "$VLLM_DRAFT_DIR/config_compare.json" \
  --json-out /path/to/qwen3_235b_eagle3/reports/eagle3_export_artifacts.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/eagle3_export_artifacts.md \
  --fail-on-error
```

The converter reads `config.json` from the verifier path, so a bare HF model id
may not be enough if the script expects a local directory. Use a local snapshot
or patch the converter to fetch config via Transformers if needed.

Wrapper:

```bash
TRAINED_CKPT=/path/to/qwen3_235b_eagle3_modelopt \
EXPORT_DIR=/path/to/qwen3_235b_eagle3_exported \
VLLM_DRAFT_DIR=/path/to/qwen3_235b_eagle3_vllm \
VERIFIER_CONFIG_DIR=/path/to/local/Qwen3-235B-A22B-Thinking-2507 \
EXPORT_CONFIG_COMPARE_JSON=/path/to/exported_config_compare.json \
VLLM_CONFIG_COMPARE_JSON=/path/to/vllm_config_compare.json \
TRAINING_CKPT_VALIDATION_JSON=/path/to/qwen3_235b_eagle3/reports/eagle3_training_checkpoint.json \
bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_export_vllm.sh
```

The wrapper runs `validate_eagle3_training_checkpoint.py` before export, then
`compare_eagle3_configs.py` after both export and vLLM conversion. Set
`RUN_TRAINING_CKPT_VALIDATION=false` or `RUN_CONFIG_COMPARE=false` only for
debugging a broken export. The compare script understands both the flat HF
draft config and the vLLM one-checkpoint `transformer_layer_config` layout.

Slurm template:

```bash
sbatch --account=<account> \
  --export=ALL,TRAINED_CKPT=/path/to/qwen3_235b_eagle3_modelopt,EXPORT_DIR=/path/to/qwen3_235b_eagle3_exported,VLLM_DRAFT_DIR=/path/to/qwen3_235b_eagle3_vllm,VERIFIER_CONFIG_DIR=/path/to/local/Qwen3-235B-A22B-Thinking-2507 \
  experiments/eagle3_qwen3_235b/slurm_export_vllm.sbatch
```

After export, compare the draft config:

```bash
python3 experiments/eagle3_qwen3_235b/compare_eagle3_configs.py \
  --draft-config /path/to/qwen3_235b_eagle3_exported \
  --verifier-config /path/to/local/Qwen3-235B-A22B-Thinking-2507 \
  --reference-arch experiments/eagle3_qwen3_235b/qwen3_235b_thinking_eagle3_architecture.json
```

To refresh the Hayate/Hiso reference inventory from a machine that can see the
Lustre paths:

```bash
REMOTE_HOST=cw-dfw-cs-001-vscode-01 \
bash experiments/eagle3_qwen3_235b/inventory_hayate_eagle3_artifacts.sh
```

For a structured Hayate workflow report:

```bash
python3 experiments/eagle3_qwen3_235b/analyze_hayate_modelopt_workflow.py \
  --hayate-modelopt-dir /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer \
  --json-out /path/to/qwen3_235b_eagle3/reports/hayate_modelopt_workflow.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/hayate_modelopt_workflow.md
```

Capture machine-readable provenance before submitting the pilot. The bootstrap
wrapper runs this by default with `RUN_PROVENANCE=true`, but it can also be run
directly:

```bash
python3 experiments/eagle3_qwen3_235b/collect_eagle3_provenance.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --modelopt-dir /path/to/Model-Optimizer \
  --verifier-config-dir /path/to/local/Qwen3-235B-A22B-Thinking-2507 \
  --input-data /path/to/qwen3_235b_swe_conversations.jsonl \
  --hidden-states-dir /path/to/qwen3_235b_eagle3_hidden_states \
  --output-dir /path/to/qwen3_235b_eagle3_modelopt \
  --export-dir /path/to/qwen3_235b_eagle3_exported \
  --vllm-draft-dir /path/to/qwen3_235b_eagle3_vllm \
  --json-out /path/to/qwen3_235b_eagle3/reports/eagle3_provenance.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/eagle3_provenance.md
```

Probe the cluster substrate before the first GPU submission:

```bash
python3 experiments/eagle3_qwen3_235b/probe_cluster_environment.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --modelopt-dir /path/to/Model-Optimizer \
  --verifier-config-dir /path/to/local/Qwen3-235B-A22B-Thinking-2507 \
  --input-data /path/to/qwen3_235b_swe_conversations.jsonl \
  --container /path/to/container.sqsh \
  --mounts /lustre:/lustre,/path/to/repo:/path/to/repo \
  --sbatch-account <account> \
  --sbatch-partition batch \
  --json-out /path/to/qwen3_235b_eagle3/reports/cluster_environment_probe.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/cluster_environment_probe.md
```

Then run the container-only preflight dry-run. This prints the exact preflight
`sbatch` command and writes a small probe report, but does not submit unless
`SUBMIT=true` is set:

```bash
SUBMIT=false \
ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
SBATCH_ACCOUNT=coreai_dlalgo_nemorl \
SBATCH_PARTITION=batch \
CONTAINER=/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh \
MOUNTS=/lustre:/lustre,$PWD:$PWD \
VERIFIER_CONFIG_DIR=/path/to/Qwen3-235B-A22B-Thinking-2507 \
INPUT_DATA=/path/to/qwen3_235b_swe_conversations.jsonl \
CHAT_TEMPLATE=/path/to/qwen3_generation_template.jinja2 \
bash experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh

python3 experiments/eagle3_qwen3_235b/analyze_container_preflight.py \
  --job-file latest_eagle3_container_preflight_job.txt \
  --logs-dir logs \
  --cluster-probe-json /path/to/qwen3_235b_eagle3/reports/container_preflight_cluster_probe.json \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --container /lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh \
  --sbatch-account coreai_dlalgo_nemorl \
  --sbatch-partition batch \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/container_preflight_analysis.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/container_preflight_analysis.json
```

After the dry-run command and probe report look correct, set `SUBMIT=true` for
this preflight-only job. Keep the full pipeline submit disabled until the
container preflight log proves ModelOpt import, recipe validation, and
assistant-mask validation inside the selected image. The analyzer should report
`Overall: PASS` before starting hidden-state dump.

Also capture ModelOpt drift against official NVIDIA `main` and the mounted
Hayate checkout. This command does not fetch or mutate refs; it only probes
upstream with `git ls-remote` when network is available:

```bash
python3 experiments/eagle3_qwen3_235b/check_modelopt_upstream_drift.py \
  --modelopt-dir /path/to/Model-Optimizer \
  --hayate-modelopt-dir /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer \
  --json-out /path/to/qwen3_235b_eagle3/reports/modelopt_upstream_drift.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/modelopt_upstream_drift.md
```

On 2026-05-21 from this workspace, the official upstream probe reported
`b02e8885509c...` for `NVIDIA/Model-Optimizer@main`, while the local checkout
and local `origin/main` were still `c9098b63fb5e...`. The local worktree also
has the intended TRT-LLM hidden-state dumper patch for answer-only `loss_mask`
storage.

Validate the live checkout before hidden-state dump:

```bash
python3 experiments/eagle3_qwen3_235b/validate_modelopt_loss_mask_patch.py \
  --modelopt-dir /path/to/Model-Optimizer \
  --json-out /path/to/qwen3_235b_eagle3/reports/modelopt_loss_mask_patch.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/modelopt_loss_mask_patch.md
```

Export that local ModelOpt patch before updating ModelOpt or moving to a
cluster checkout:

```bash
python3 experiments/eagle3_qwen3_235b/export_modelopt_eagle3_patch_bundle.py \
  --modelopt-dir /path/to/Model-Optimizer \
  --out-dir /path/to/qwen3_235b_eagle3/patches/modelopt_eagle3_qwen3
```

The bundle writes `modelopt_eagle3_qwen3.patch`, `manifest.json`,
`patch_report.md`, and snapshots of the patched TRT-LLM dumper plus its shared
`common.py` helpers. Add `--compat-modelopt-dir /path/to/clean-or-latest-Model-Optimizer`
to prove whether the patch can be applied to another checkout without modifying
that checkout. On 2026-05-21, this compatibility check passed against a shallow
clone of official `NVIDIA/Model-Optimizer@b02e8885509c...`.

To structurally compare existing Hayate/Hiso draft model configs against the
Qwen3-235B Thinking Eagle3 reference:

```bash
python3 experiments/eagle3_qwen3_235b/inventory_eagle3_draft_configs.py \
  /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/nemo-rl-internal-worktrees/feat-eagle3-online-specdec/models \
  /path/to/qwen3_235b_eagle3/vllm_draft \
  /path/to/qwen3_235b_eagle3/exported_hf \
  --reference-arch experiments/eagle3_qwen3_235b/qwen3_235b_thinking_eagle3_architecture.json \
  --markdown-out /path/to/eagle3_draft_config_inventory.md \
  --json-out /path/to/eagle3_draft_config_inventory.json
```

If the Hayate model directory is not readable from the current account, the
report should be kept: it records each requested root in `root_statuses` and
emits warnings for missing or inaccessible roots. The completion and goal
audits only treat the inventory as inspected when it scanned at least one
config or recorded concrete access-limit warnings.

### One-Command Slurm Pipeline

After conversations exist, this validates the visible `INPUT_DATA` sample and
prints the full Slurm plan without submitting:

```bash
python3 experiments/eagle3_qwen3_235b/estimate_eagle3_training_scale.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --input-data /path/to/qwen3_235b_swe_conversations.jsonl \
  --corpus-strategy-json /path/to/qwen3_235b_eagle3/reports/corpus_strategy.json \
  --pipeline-submit-preflight-json /path/to/qwen3_235b_eagle3/reports/eagle3_pipeline_submit_preflight.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/eagle3_training_scale.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/eagle3_training_scale.json
```

```bash
python3 experiments/eagle3_qwen3_235b/audit_eagle3_readiness.py \
  --input-data /path/to/qwen3_235b_swe_conversations.jsonl \
  --hidden-states-dir /path/to/qwen3_235b_eagle3_hidden_states \
  --output-dir /path/to/qwen3_235b_eagle3_modelopt \
  --export-dir /path/to/qwen3_235b_eagle3_exported \
  --vllm-draft-dir /path/to/qwen3_235b_eagle3_vllm \
  --verifier-config-dir /path/to/local/Qwen3-235B-A22B-Thinking-2507 \
  --reference-arch /path/to/eagle3_architecture.json \
  --arch-env-file /path/to/eagle3_architecture.env \
  --nemo-rl-drift-json /path/to/qwen3_235b_eagle3/reports/nemo_rl_eagle3_drift.json \
  --training-scale-json /path/to/qwen3_235b_eagle3/reports/eagle3_training_scale.json \
  --markdown-out /path/to/eagle3_readiness.md \
  --json-out /path/to/eagle3_readiness.json
```

Use the audit as a status report: `PASS` means that step has concrete evidence,
`MISSING` means the artifact has not been produced or is not visible on the
current host, and `WARN` means the path is plausible but not fully proven in the
current environment. The audit now also proves that `REFERENCE_ARCH` is
reproducible from `VERIFIER_CONFIG_DIR/config.json`, that `ARCH_ENV_FILE`
matches the architecture reference, and that both full and `RUN_PILOT=true`
Slurm dependency plans contain the expected gates. Add `--strict` when missing
runtime artifacts should return a nonzero exit code.

After a submitted pilot or full pipeline starts writing Slurm logs, summarize
stage status with:

```bash
python3 experiments/eagle3_qwen3_235b/analyze_eagle3_pipeline.py \
  --job-file latest_eagle3_pipeline_jobs.txt \
  --logs-dir logs \
  --input-data /path/to/qwen3_235b_swe_conversations.jsonl \
  --hidden-states-dir /path/to/qwen3_235b_eagle3_hidden_states \
  --hidden-validation-json /path/to/qwen3_235b_eagle3_hidden_states/validation_summary.json \
  --training-checkpoint-json /path/to/qwen3_235b_eagle3/reports/eagle3_training_checkpoint.json \
  --output-dir /path/to/qwen3_235b_eagle3_modelopt \
  --export-dir /path/to/qwen3_235b_eagle3_exported \
  --vllm-draft-dir /path/to/qwen3_235b_eagle3_vllm \
  --verifier-config-dir /path/to/local/Qwen3-235B-A22B-Thinking-2507 \
  --sbatch-account <account> \
  --run-pilot true \
  --markdown-out /path/to/eagle3_pipeline_analysis.md \
  --json-out /path/to/eagle3_pipeline_analysis.json
```

This does not call Slurm APIs. It only uses the job id file, local logs, and
artifact paths, so it works equally well after logs are copied back from the
cluster. The JSON and Markdown reports include `next_action` guidance with a
safe dry-run resume command. It disables pass-level stages before the first
open stage and reruns the remaining tail of the pipeline; switch `SUBMIT=true`
only after reviewing that printed plan on the cluster.

After export and the trained-draft token sweep, run the completion audit. A
`PASS` here means the Qwen3-235B Thinking architecture reference, remote
ModelOpt/Hayate path probe, Hayate ModelOpt/SpecForge reference reports,
ModelOpt recipe override validation, container preflight, actual SWE/RL rollout
corpus, pipeline submit preflight, gated pipeline submission, Megatron probe
follow-up guard, hidden-state dump, ModelOpt checkpoint, HF/vLLM export, config
comparisons, training-checkpoint contract, and RL smoke sweep all have concrete
evidence:

```bash
python3 experiments/eagle3_qwen3_235b/audit_eagle3_completion.py \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --provenance-json /path/to/qwen3_235b_eagle3/reports/eagle3_provenance.json \
  --cluster-probe-json /path/to/qwen3_235b_eagle3/reports/cluster_environment_probe.json \
  --remote-host-probe-json /path/to/qwen3_235b_eagle3/reports/eagle3_remote_host_probe.json \
  --hayate-workflow-json /path/to/qwen3_235b_eagle3/reports/hayate_modelopt_workflow.json \
  --hayate-specforge-reference-json /path/to/qwen3_235b_eagle3/reports/hayate_specforge_reference.json \
  --upstream-drift-json /path/to/qwen3_235b_eagle3/reports/modelopt_upstream_drift.json \
  --modelopt-recipe-overrides-json /path/to/qwen3_235b_eagle3/reports/modelopt_recipe_overrides_current.json \
  --modelopt-patch-manifest /path/to/qwen3_235b_eagle3/patches/modelopt_eagle3_qwen3/manifest.json \
  --next-action-plan-validation-json /path/to/qwen3_235b_eagle3/reports/eagle3_next_actions_validation.json \
  --operator-followup-validation-json /path/to/qwen3_235b_eagle3/reports/eagle3_operator_followups_validation.json \
  --megatron-probe-followup-validation-json /path/to/qwen3_235b_eagle3/reports/megatron_probe_followup_validation.json \
  --preflight-robustness-validation-json /path/to/qwen3_235b_eagle3/reports/eagle3_preflight_robustness_validation.json \
  --completion-contract-json /path/to/qwen3_235b_eagle3/reports/eagle3_completion_contract.json \
  --container-preflight-json /path/to/qwen3_235b_eagle3/reports/container_preflight_analysis.json \
  --rollout-state-json /path/to/qwen3_235b_eagle3/reports/rollout_capture_state_advance.json \
  --corpus-strategy-json /path/to/qwen3_235b_eagle3/reports/corpus_strategy.json \
  --pipeline-submit-preflight-json /path/to/qwen3_235b_eagle3/reports/eagle3_pipeline_submit_preflight.json \
  --pipeline-gated-submit-json /path/to/qwen3_235b_eagle3/reports/eagle3_pipeline_gated_submit.json \
  --pipeline-analysis-json /path/to/qwen3_235b_eagle3/reports/eagle3_pipeline_analysis.json \
  --hidden-validation-json /path/to/qwen3_235b_eagle3_hidden_states/validation_summary.json \
  --output-dir /path/to/qwen3_235b_eagle3_modelopt \
  --training-checkpoint-json /path/to/qwen3_235b_eagle3/reports/eagle3_training_checkpoint.json \
  --export-dir /path/to/qwen3_235b_eagle3_exported \
  --vllm-draft-dir /path/to/qwen3_235b_eagle3_vllm \
  --export-artifacts-json /path/to/qwen3_235b_eagle3/reports/eagle3_export_artifacts.json \
  --export-config-compare-json /path/to/qwen3_235b_eagle3_exported/config_compare.json \
  --vllm-config-compare-json /path/to/qwen3_235b_eagle3_vllm/config_compare.json \
  --sweep-json /path/to/qwen3_235b_eagle3/reports/trained_draft_spec_tokens_sweep.json \
  --draft-inventory-json /path/to/qwen3_235b_eagle3/reports/eagle3_draft_config_inventory.json \
  --hayate-inventory /path/to/qwen3_235b_eagle3/reports/hayate_inventory.txt \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/eagle3_completion_audit.md \
  --json-out /path/to/qwen3_235b_eagle3/reports/eagle3_completion_audit.json
```

The Slurm pipeline now defaults those config-compare JSONs to
`$EXPORT_DIR/config_compare.json` and `$VLLM_DRAFT_DIR/config_compare.json`, so
the audit can consume export evidence without scraping Slurm logs. Local vLLM
draft checks accept either a single `model.safetensors` file or sharded
`*.safetensors` weights, and the trained-draft sweep report must record the
same `VLLM_DRAFT_DIR` that the completion audit is validating. It must also
carry enough RL execution context: `artifact_root`, `config_file`, `env_file`,
`chat_template`, and either `repo_root` or `swe_repo_root`.
The training checkpoint report must also pass, proving the ModelOpt checkpoint
contains HF weights, a positive trainer step, `modelopt_state.pth`, and the
expected Qwen3 Eagle3 `eagle` mode before export is trusted.
The export artifact report must also pass, proving the HF export and vLLM
one-checkpoint draft directories contain readable configs, non-empty weights,
valid safetensors containers, and Eagle3 contract fields.

To hand off the current state to another teammate or attach it to a progress
thread:

```bash
python3 experiments/eagle3_qwen3_235b/create_eagle3_handoff_bundle.py \
  --out-dir /path/to/qwen3_235b_eagle3_handoff \
  --artifact-root /path/to/qwen3_235b_eagle3 \
  --sbatch-account <account> \
  --provenance-json /path/to/qwen3_235b_eagle3/reports/eagle3_provenance.json \
  --input-discovery-json /path/to/qwen3_235b_eagle3/eagle3_input_discovery.json \
  --cluster-probe-json /path/to/qwen3_235b_eagle3/reports/cluster_environment_probe.json \
  --remote-host-probe-json /path/to/qwen3_235b_eagle3/reports/eagle3_remote_host_probe.json \
  --hayate-workflow-json /path/to/qwen3_235b_eagle3/reports/hayate_modelopt_workflow.json \
  --hayate-specforge-reference-json /path/to/qwen3_235b_eagle3/reports/hayate_specforge_reference.json \
  --upstream-drift-json /path/to/qwen3_235b_eagle3/reports/modelopt_upstream_drift.json \
  --modelopt-recipe-overrides-json /path/to/qwen3_235b_eagle3/reports/modelopt_recipe_overrides_current.json \
  --modelopt-patch-manifest /path/to/qwen3_235b_eagle3/patches/modelopt_eagle3_qwen3/manifest.json \
  --readiness-json /path/to/qwen3_235b_eagle3/reports/eagle3_readiness.json \
  --nemo-rl-drift-json /path/to/qwen3_235b_eagle3/reports/nemo_rl_eagle3_drift.json \
  --training-scale-json /path/to/qwen3_235b_eagle3/reports/eagle3_training_scale.json \
  --next-action-transitions-json /path/to/qwen3_235b_eagle3/reports/eagle3_next_action_transitions.json \
  --pipeline-analysis-json /path/to/qwen3_235b_eagle3/reports/eagle3_pipeline_analysis.json \
  --training-checkpoint-json /path/to/qwen3_235b_eagle3/reports/eagle3_training_checkpoint.json \
  --export-artifacts-json /path/to/qwen3_235b_eagle3/reports/eagle3_export_artifacts.json \
  --sweep-json /path/to/qwen3_235b_eagle3/reports/trained_draft_spec_tokens_sweep.json \
  --completion-json /path/to/qwen3_235b_eagle3/reports/eagle3_completion_audit.json \
  --hayate-inventory /path/to/qwen3_235b_eagle3/reports/hayate_inventory.txt \
  --draft-inventory-json /path/to/qwen3_235b_eagle3/reports/eagle3_draft_config_inventory.json
```

The bundle contains `RUNBOOK.md`, `commands.sh`, `manifest.json`, the dashboard,
the Qwen3 architecture reference, and any supplied reports. Missing optional
reports are recorded in `manifest.json` instead of failing bundle creation. If
a report argument is omitted, the bundle now looks for the standard report under
`ARTIFACT_ROOT` first, so a normal refresh directory can be handed off without
spelling out every JSON path. The generated `commands.sh` records
`LOCAL_ARTIFACT_ROOT` for the copied evidence but defaults execution
`ARTIFACT_ROOT` to the remote Lustre artifact root; override `ARTIFACT_ROOT`
only for local dry-runs. Re-running the bundle generator cleans stale managed
files from previous bundle layouts by default; use `--no-clean-stale` only when
you intentionally want to keep extra handoff files in the same directory.

```bash
python3 experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline.py \
  --input-data /path/to/qwen3_235b_swe_conversations.jsonl \
  --hidden-states-dir /path/to/qwen3_235b_eagle3_hidden_states \
  --output-dir /path/to/qwen3_235b_eagle3_modelopt \
  --export-dir /path/to/qwen3_235b_eagle3_exported \
  --vllm-draft-dir /path/to/qwen3_235b_eagle3_vllm \
  --verifier-config-dir /path/to/local/Qwen3-235B-A22B-Thinking-2507 \
  --sbatch-account <account>
```

Inside the actual training container, add `--require-modelopt-import` to prove
that ModelOpt's Pydantic recipe loader accepts the overrides before submitting:

```bash
python3 experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline.py \
  --input-data /path/to/qwen3_235b_swe_conversations.jsonl \
  --hidden-states-dir /path/to/qwen3_235b_eagle3_hidden_states \
  --output-dir /path/to/qwen3_235b_eagle3_modelopt \
  --export-dir /path/to/qwen3_235b_eagle3_exported \
  --vllm-draft-dir /path/to/qwen3_235b_eagle3_vllm \
  --verifier-config-dir /path/to/local/Qwen3-235B-A22B-Thinking-2507 \
  --sbatch-account <account> \
  --require-modelopt-import
```

To validate only one training wrapper:

```bash
python3 experiments/eagle3_qwen3_235b/validate_modelopt_recipe_overrides.py \
  --wrapper experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh \
  --training-mode offline \
  --modelopt-dir /path/to/Model-Optimizer \
  --json-out /path/to/qwen3_235b_eagle3/reports/modelopt_recipe_overrides_current.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/modelopt_recipe_overrides_current.md

python3 experiments/eagle3_qwen3_235b/validate_modelopt_recipe_overrides.py \
  --wrapper experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_online_train.sh \
  --training-mode online \
  --modelopt-dir /path/to/Model-Optimizer \
  --json-out /path/to/qwen3_235b_eagle3/reports/modelopt_recipe_overrides_online.json \
  --markdown-out /path/to/qwen3_235b_eagle3/reports/modelopt_recipe_overrides_online.md
```

Then print the Slurm plan:

```bash
SUBMIT=false \
SBATCH_ACCOUNT=<account> \
INPUT_DATA=/path/to/qwen3_235b_swe_conversations.jsonl \
HIDDEN_STATES_DIR=/path/to/qwen3_235b_eagle3_hidden_states \
OUTPUT_DIR=/path/to/qwen3_235b_eagle3_modelopt \
EXPORT_DIR=/path/to/qwen3_235b_eagle3_exported \
VLLM_DRAFT_DIR=/path/to/qwen3_235b_eagle3_vllm \
VERIFIER_CONFIG_DIR=/path/to/local/Qwen3-235B-A22B-Thinking-2507 \
TRAINED_CKPT=/path/to/qwen3_235b_eagle3_modelopt \
CHAT_TEMPLATE=/path/to/qwen3_fixed_template.jinja2 \
CONTAINER=/path/to/container.squashfs \
MOUNTS=/lustre:/lustre,$PWD:$PWD \
bash experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh
```

To also chain the trained-draft smoke pair after export, add:

```bash
RUN_TRAINED_DRAFT_SMOKE=true \
SMOKE_EAGLE3_NUM_SPEC_TOKENS=3 \
SMOKE_MAX_NUM_STEPS=1 \
bash experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh
```

When submitted, the smoke baseline is held on `afterok:<export_job>`, and the
specdec smoke is held on the baseline job. This lets the smoke submitter accept
the future `VLLM_DRAFT_DIR` path before the export job has created it.

By default the submitter runs `slurm_preflight.sbatch` first with
`PREFLIGHT_REQUIRE_MODELOPT_IMPORT=true`, so the actual container must be able
to import ModelOpt, validate the recipe overrides, and prove that `CHAT_TEMPLATE`
produces an assistant-token mask before dump, hidden-state validation, train,
and export jobs can start. It also runs
`slurm_validate_hidden_states.sbatch`
between dump and train with `RUN_VALIDATE_HIDDENS=true`; set
`RUN_VALIDATE_HIDDENS=false` only when the hidden-state directory has already
passed `validate_hidden_state_dump.py` in the target environment. Set
`VALIDATE_MODELOPT_LOADER=false` only if the direct ModelOpt dataset/collator
check was already run separately in the target container. Set
`PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK=false` only when answer-only loss is
intentionally disabled or the same mask check already passed. Set
`RUN_PREFLIGHT=false` only when the same preflight check has already passed.

When the printed commands look correct, submit through the gated helper. The
operator planner's `submit_eagle3_pilot_pipeline` action uses this helper; do
not bypass it with a manual `SUBMIT=true` command:

```bash
python3 experiments/eagle3_qwen3_235b/submit_eagle3_pipeline_if_ready.py \
  --artifact-root /path/qwen3_235b_eagle3 \
  --preflight-json /path/qwen3_235b_eagle3/reports/eagle3_pipeline_submit_preflight.json \
  --json-out /path/qwen3_235b_eagle3/reports/eagle3_pipeline_gated_submit.json \
  --markdown-out /path/qwen3_235b_eagle3/reports/eagle3_pipeline_gated_submit.md \
  --execute --allow-heavy-gpu
```

The helper refuses to submit unless the preflight JSON reports
`submit_ready=true`, the input rollout corpus exists, the command targets
`submit_eagle3_pipeline.sh` with the expected environment, and the command env
matches the preflight report for corpus/config/template/artifact/resource
fields. The pipeline submitter writes submitted job ids to
`latest_eagle3_pipeline_jobs.txt`; the gated helper also requires `dump_job`,
`train_job`, and `export_job` to be present, then copies that job file to
`reports/eagle3_pipeline_jobs.env` for stable audit evidence.
`audit_eagle3_completion.py` treats the resulting
`eagle3_pipeline_gated_submit.json` as required evidence, separate from the
no-submit preflight report.

### Phase 5: NeMo-RL Static Integration

Use `nemo_rl_specdec_overlay.yaml` as the minimal overlay. It only changes the
generation side and does not train draft weights online.

The generation init path in Hayate/upstream NeMo-RL sets `load_format="auto"`
when `vllm_kwargs.speculative_config` is present. That is important because a
static draft checkpoint must be loaded as real weights, not through the usual
dummy-weight path.

The acceptance/performance gate for this phase is:

- vLLM workers start cleanly with `speculative_config.method=eagle3`.
- Median `exposed_generation` improves versus the non-specdec baseline after
  dropping the cold step and obvious generation outliers.
- Acceptance rate is available and healthy, or the logging gap is explicitly
  tracked via worker/metrics logs.
- Reward, malformed thinking, and SWE environment errors do not regress.

After `modelopt_qwen3_235b_export_vllm.sh` has produced a local vLLM draft
directory, run the trained-draft smoke pair:

```bash
SUBMIT=false \
VLLM_DRAFT_DIR=/path/to/qwen3_235b_eagle3_vllm \
EAGLE3_NUM_SPEC_TOKENS=3 \
bash experiments/eagle3_qwen3_235b/submit_trained_draft_smoke_pair.sh
```

The wrapper writes job ids to `latest_trained_draft_specdec_smoke_jobs.txt` and
prints the matching analyzer command. Set `SUBMIT=true` only after the dry-run
shows the expected draft path and run names. For final tuning, repeat this with
`EAGLE3_NUM_SPEC_TOKENS=2`, `3`, and `4`; longer chains only help when
acceptance stays healthy.

### Phase 6: Online RL Draft Training

Do not confuse this with ModelOpt online training above. The ModelOpt online
wrapper trains a draft from a fixed conversation dataset while the verifier is
available in the training job. NeMo-RL online draft training means the draft
evolves inside the RL loop while the policy itself changes.

NeMo-RL's public Eagle3 guide documents this as a supported integration mode,
but it currently requires Megatron, DTensor disabled, and sequence packing
disabled. The current Qwen3-235B SWE config already uses Megatron and disables
DTensor, but it enables sequence packing for policy training; the online overlay
therefore explicitly sets `policy.sequence_packing.enabled=false`.

Validate the target checkout before planning online RL draft training:

```bash
python3 experiments/eagle3_qwen3_235b/check_nemo_rl_eagle3_drift.py \
  --nemo-rl-dir /path/to/SpecDec-RL \
  --markdown-out /path/to/nemo_rl_eagle3_drift.md \
  --json-out /path/to/nemo_rl_eagle3_drift.json

python3 experiments/eagle3_qwen3_235b/validate_nemo_rl_specdec_integration.py \
  --config grpo_qwen3_235b_swe.yaml \
  --draft-model /path/to/qwen3_235b_eagle3_vllm \
  --integration-mode online-draft-training \
  --specdec-rl-dir /path/to/SpecDec-RL \
  --markdown-out /path/to/nemo_rl_eagle3_online_draft_integration.md \
  --json-out /path/to/nemo_rl_eagle3_online_draft_integration.json \
  --env-out /path/to/nemo_rl_eagle3_online_draft_overrides.env
```

If the checker cannot prove `policy.draft`/draft-loss support in the checkout,
stay on the fixed-draft generation path or update NeMo-RL before attempting
online draft training. Only do this after the static draft has a measured
generation win. Required pieces are:

- policy forward hidden-state capture for Eagle3 aux layers and embeddings,
- draft loss against the current policy logits/probabilities,
- Megatron/FSDP handling for draft parameters,
- checkpoint/export for the draft separately from the verifier,
- generation-worker draft weight sync after policy updates,
- monitoring acceptance-rate decay across RL steps.

This is a NeMo-RL feature project, not a ModelOpt-only change.
