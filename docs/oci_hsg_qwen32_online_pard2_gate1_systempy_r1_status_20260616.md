# Qwen32 MathRL Online PARD-2 Gate r1 - 2026-06-16

Snapshot time: `2026-06-16 01:43 PDT`.

## Launch

| Job | Method | State | Account | Shape | Max OSL | Notes |
|---:|---|---|---|---|---:|---|
| `3337769` | online PARD-2 K3, actor venv build | `FAILED` after `00:26:25` | `nemotron_n3_post` | 4 nodes x 4 GPUs, GBS512 | 1024 | Reached `SETUP COMPLETE` and `Step 1/1`, then failed during vLLM weight refit/update because a draft-side `fc` parameter was routed into `Qwen3ForCausalLM`. |
| `3337950` | online PARD-2 K3, system actor Python fallback | `FAILED` after `00:03:16` | `nemotron_n3_post` | 4 nodes x 4 GPUs, GBS512 | 1024 | Failed during `VllmGenerationWorker.__init__`; `/opt/nemo_rl_venv` path missed a PARD-2 vLLM overlay dependency. |

Tracker:

- `latest_oci_hsg_mathrl_qwen32_online_pard2_gate1_systempy_r1_20260616_jobs.csv`
- `latest_oci_hsg_mathrl_qwen32_online_pard2_gate1_systemactors_r2_20260616_jobs.csv`

Remote log root:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_multimodel_specdec_logs/20260616_mathrl_qwen32_online_pard2_gate1_systempy_r1/qwen32_online_pard2`

## Contract Checked Before Submission

Dry-run validation passed before submission and showed the expected online PARD-2 training contract:

- `policy.draft.enabled=true`
- `PARD_ONLINE_TRAINING=true`
- `policy.draft.type=pard2`
- `policy.draft.loss=pard2`
- `policy.draft.training_mode=k_slot`
- `policy.draft.max_training_sequence_length=256`
- `policy.draft.train_interval=1`
- `policy.draft.refit_interval=1`
- `policy.sequence_packing.enabled=false`
- `policy.megatron_cfg.context_parallel_size=1`
- `PYTHON_RUNNER_OVERRIDE=/opt/nemo_rl_venv/bin/python`
- PARD-2 vLLM overlay: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat_pyoverlay_nostable_nofp4out_basefa_nofp4fusion_20260614`

## Live Driver Progress

At `2026-06-16 01:17 PDT`, `ray-driver.log` showed:

- Ray connected and all `16/16` worker units came online.
- Final NeMo-RL config includes `draft.enabled=True`, `draft.type='pard2'`, `draft.loss='pard2'`, `draft.training_mode='k_slot'`, `draft.max_training_sequence_length=256`, `draft.train_interval=1`, and `draft.refit_interval=1`.
- Data loading completed for OpenMathInstruct-2 and the policy Ray cluster initialized on 4 nodes.
- The run was building/installing the VLLM generation actor environment; no `SETUP COMPLETE`, `Performance Metrics`, Step 1 result, traceback, or CUDA OOM had appeared yet.

At `2026-06-16 01:25 PDT`, r1 was still running and had no traceback/OOM, but it remained in the first actor venv build after downloading `nvidia-cutlass-dsl-libs-*` and `mlflow` packages. The r2 fallback was dry-run validated with `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` and submitted so actor workers use the system `/opt/nemo_rl_venv` path instead of building a fresh per-actor venv.

At `2026-06-16 01:30 PDT`, r1 had moved past the actor venv build and entered vLLM/PARD-2 engine initialization. The log showed `speculative_config=SpeculativeConfig(method='pard2', ... num_spec_tokens=3)` and the NeMo-RL vLLM worker extension was injected. No Step 1 metric or OOM had appeared yet.

At `2026-06-16 01:36 PDT`, r1 was still `RUNNING` at `00:22:26` elapsed. It had finished both actor venv installs, initialized the PARD-2 vLLM side, loaded PARD-2 target layers `(64, 57, 49, 41)`, and started `MegatronPolicyWorker` initialization with the cached actor Python:

```text
Initializing lm_policy workers: 25%|██▌       | 4/16
```

There was still no `SETUP COMPLETE`, Step 1 metric, traceback, CUDA OOM, or draft-training/refit metric. This was progress beyond the earlier r1 snapshot, but not yet proof that online PARD-2 passes Step 1.

At `2026-06-16 01:39 PDT`, r1 failed after `00:26:25` elapsed. It reached `SETUP COMPLETE` and entered `Step 1/1`, so the actor-venv path now passes the dependency/setup phase. The failure happened during `refit_policy_generation()` while `update_weights_via_ipc_zmq` was loading weights into vLLM:

```text
ValueError: There is no module or parameter named 'fc' in Qwen3ForCausalLM.
```

The available vLLM parameters include normal Qwen3 verifier weights plus `target_proj.weight`, but no `fc`. This points to a target/draft weight-routing or filtering issue: a PARD-2 draft-side parameter named `fc` is being sent to the verifier `Qwen3ForCausalLM` during refit. This is no longer an environment, scheduler, CUDA OOM, or startup problem; the next fix should be in the online PARD-2 refit weight mapping/filter.

The r2 fallback `3337950` failed quickly. It initialized 16 vLLM workers with `py_executable=/opt/nemo_rl_venv/bin/python`, but actor creation failed while importing the PARD-2 vLLM overlay:

```text
ModuleNotFoundError: No module named 'cbor2'
```

NeMo-RL then surfaced it as:

```text
ImportError: vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker covers the vllm dependency.
```

This means the system actor Python shortcut is incomplete for the PARD-2 overlay; the r1 actor-venv path is slower but includes the missing dependency set.

Current local refresh artifacts:

- `docs/oci_hsg_mathrl_active_refresh_summary_20260616.csv`
- `docs/oci_hsg_mathrl_active_refresh_steps_20260616.csv`

At the `2026-06-16 01:38 PDT` metric refresh, Qwen32 baseline job `3334219` had `16` completed steps and Qwen32 static PARD2-14B retry `3334113` had `10` completed steps. Static PARD2-14B remained much slower than baseline with about `1.70%` token acceptance and `~53.0` generation tok/s/GPU versus baseline `~76.1` generation tok/s/GPU.

## Why This Gate Exists

The Qwen235B online PARD-2 gate `3337599` is still pending on 32 nodes. This smaller Qwen32 gate uses the same online PARD-2 policy-draft path but only needs 4 nodes, so it should provide faster evidence on whether online PARD-2 can pass Step 1 under the current NeMo-RL patch set.

## Related Long-OSL Status

The long-output NeMo-RL refresh is tracked separately in:

- `docs/nemorl_long_osl_16k_32k_status_20260616.md`

Current long-OSL conclusion: 16K MathRL/SpecDec has completed evidence for Qwen3-32B static PARD and partial Qwen3-30B Step 1 evidence; true 32K NeMo-RL evidence exists for SWE-RL scaleout, but no completed 32K MathRL/PARD/Eagle/Suffix NeMo-RL training result was found.
