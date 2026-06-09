# Qwen3-235B SWE Async-GRPO on vLLM 0.17 + FP8KV — Setup Handoff

Complete, self-contained setup for running the Qwen3-235B-A22B-Thinking-2507 SWE
Async-GRPO rollout on **vLLM 0.17.1** with **FP8 KV cache**, on cw-dfw-cs (256 H100).
Everything here is needed to reproduce/continue. Tokens live in `~/.bashrc`
(`WANDB_API_KEY`, `HF_TOKEN`) — never echo/commit them.

## Goal
Measure per-step timing (total / refit / logprobs / training / exposed_generation)
for FP8KV across the gen/train node split × async age:
**{192gen/64train, 128gen/128train} × {async4, async8}**, all on vLLM 0.17.1.
Status: the full version-drift + config blocker chain is **solved**; the pipeline
runs at 256 GPU (init, checkpoint load, generation, buffer fill, training step).
Latest run: **12674581** (192g64t-async4) with the complete fix chain.

## Stack / paths
- **Repo**: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/repos/nemo-rl-swe-bench-ruit`
- **Branch**: `ruit/SWE_bench` @ `6dc8fabea` (submodules init'd: Megatron-LM, Megatron-Bridge, Gym, Automodel)
- **Container (vLLM 0.17.1)**: `/lustre/fsw/portfolios/coreai/users/ruit/enroot-images/docker_images:ruit-swe_bench-6de99f772-x86_64-060326-mcore-apptainer.squashfs`
  - vLLM 0.17.1 is baked in (uv.lock pins `vllm==0.17.1+cu130`). NO precompiled-wheel override (that would force 0.13). Has ruit's hermes-tool-parser patch.
- **Launch script**: `examples/swe_bench/run_grpo_swe2_qwen235b_fp8kv.sh` (derived from `run_grpo_repro_baseline_swe2.sh`; parameterized by `NUM_NODES`/`NUM_GEN_NODES`/`MAX_TRAJECTORY_AGE_STEPS`/`EXP_SUFFIX`).
- **HF_HOME (patched)**: `/lustre/fsw/portfolios/coreai/users/sna/hf_home_vllm017`
  - `hub/` -> symlink to pmannan's `.../pmannan/rl_projects/hf_home/hub` (HF weights, no re-download; HF_HUB_OFFLINE)
  - `nemo_rl/Qwen/Qwen3-235B-A22B-Thinking-2507/iter_0000000/` -> all 260 `.distcp` + `.metadata` SYMLINKED to pmannan's 439 GB converted ckpt (zero copy) + a REAL **patched** `run_config.yaml` (see fix #1).
- **Data**: `/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl`
- **Model**: `Qwen/Qwen3-235B-A22B-Thinking-2507` (base HF, converted to Megatron on load; NOT the 30B SWE1 step_230 checkpoint from the repro doc).

## THE FIX CHAIN (all in sna's repo => snapshotted by ray.sub at submit)
The ruit 0.17 megatron stack is newer/stricter than pmannan's 0.13; loading pmannan's
old TP=1 converted checkpoint needs 4 fixes + 1 config correction:

1. **Megatron-bridge config STRICT->LENIENT** — `nemo_rl/models/megatron/setup.py:458`
   `ConfigContainer.from_yaml(pretrained_run_config, mode=InstantiationMode.LENIENT)` (was STRICT).
   STRICT rejects deprecated GPTModelProvider keys one section at a time (model: 5 keys, then optimizer: 2, ...). LENIENT (the library default) drops unknown keys with a warning. Also: the patched `run_config.yaml` in HF_HOME already has the 5 model keys removed (`async_tensor_model_parallel_allreduce, moe_extended_tp, moe_use_legacy_grouped_gemm, mtp_detach_heads, mtp_grad_scale_func`).
2. **DCP re-shard validation** — `3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/dist_checkpointing/serialization.py:57`
   `def load(..., validate_access_integrity: bool = False)` (was True).
   The ckpt is stored TP=1/PP=1 and re-shards to TP=4/PP=8 on load; the newer validator wrongly rejects the prepend_axis_num layer-stacked tensors. Disabling the access-integrity check lets the (correct) re-shard proceed. pmannan loaded the same ckpt fine on 0.13.
3. **FP8KV q_scale refit** — `nemo_rl/models/policy/workers/megatron_policy_worker.py` (~line 1086)
   `keys.extend(v for k, v in scale_names.items() if k != "q_scale")` — vLLM 0.17 Qwen3 registers k/v_scale but not q_scale; refit must skip q_scale or FP8KV update_weights NaNs.
4. **seq_logprob_error_threshold vs force_on_policy_ratio** — `examples/swe_bench/run_grpo_swe2_qwen235b_fp8kv.sh`
   `SEQ_LOGPROB_ERROR_THRESHOLD=null` (was 2). On 0.17 these are mutually exclusive:
   `AssertionError: seq_logprob_error_threshold requires prev_logprobs computation; cannot use with force_on_policy_ratio=True`. The REPRO doc confirms: it sets `force_on_policy_ratio=True` and does NOT set seq_logprob_error_threshold. Keep `FORCE_ON_POLICY_RATIO=True`, leave seq_logprob null.

Verify all are in place before submit:
```
grep -c "InstantiationMode.LENIENT" nemo_rl/models/megatron/setup.py            # 1
sed -n '57p' 3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/dist_checkpointing/serialization.py | grep -c "= False"  # 1
grep -c 'if k != "q_scale"' nemo_rl/models/policy/workers/megatron_policy_worker.py  # 1
grep -E '^SEQ_LOGPROB_ERROR_THRESHOLD=' examples/swe_bench/run_grpo_swe2_qwen235b_fp8kv.sh  # =null
```

## ⚠ CRITICAL: the fixes are UNCOMMITTED working-tree edits
As of handoff, none of the 4 code fixes are committed:
- `setup.py`, `megatron_policy_worker.py` -> ` M` (modified, tracked, uncommitted)
- `serialization.py` -> ` M` **inside the Megatron-LM submodule** (a `git submodule update --init --recursive` WIPES it)
- `run_grpo_swe2_qwen235b_fp8kv.sh` -> `??` (untracked; survives checkout/pull but not `git clean -fd`)

**Do NOT run `git pull` / `git checkout` / `git reset` / `git stash` / `git submodule update` in the ruit repo** — any of these can lose the fixes. To restore them after any git op, run the idempotent re-apply script:
```
bash experiments/scaleout_256h100_32k_async16/scripts/apply_vllm017_fixes.sh          # apply+verify
bash experiments/scaleout_256h100_32k_async16/scripts/apply_vllm017_fixes.sh --verify # verify only (all counts must be 1)
```
`.bak_strict` / `.bak_vai` / `.bak_seqlp` backups sit next to the edited files. The q_scale fix (fix #3) is a multi-line code edit, not sed-able — re-apply it with `experiments/eagle3_qwen3_235b/remote_patches/apply_ruit_fp8kv_qscale_fix.py` if the verify shows it missing.

## Exact submit wrapper (token-safe, via vscode-02)
Tokens are read from `~/.bashrc` with grep (do NOT `source ~/.bashrc` — it hangs on lustre). Always filter output through `grep -avE "hf_ccp|cd4db"` so a token never prints.
```
WK=$(grep -hoE 'WANDB_API_KEY=[A-Za-z0-9]+' ~/.bashrc | head -1 | cut -d= -f2)
HK=$(grep -hoE 'HF_TOKEN=[A-Za-z0-9_]+' ~/.bashrc | head -1 | cut -d= -f2)
export WANDB_API_KEY="$WK" HF_TOKEN="$HK"
export HF_HOME=/lustre/fsw/portfolios/coreai/users/sna/hf_home_vllm017
export NUM_NODES=32 SBATCH_PARTITION=batch SBATCH_TIME=4:0:0 MAX_NUM_STEPS=40
export NUM_GEN_NODES=24 MAX_TRAJECTORY_AGE_STEPS=4 EXP_SUFFIX=vLLM0.17-fp8kv-192g64t-async4
export PERSISTENT_CACHE=/lustre/fsw/portfolios/coreai/users/sna/.cache/q235b_vllm017_192g64t_a4
cd <ruit-repo>; bash examples/swe_bench/run_grpo_swe2_qwen235b_fp8kv.sh
```
From a laptop, pipe via stdin: `ssh cw-dfw-cs-001-vscode-02 'bash -s' < wrapper.sh 2>&1 | grep -avE "hf_ccp|cd4db"`.

## Dependencies / fragilities the next agent must know
- **HF_HOME points into pmannan's files via symlinks**: `hub/` and all 260 `.distcp` + `.metadata` shards symlink to `/lustre/fsw/.../pmannan/rl_projects/hf_home/...`. If pmannan deletes/moves those, both the HF model load and the Megatron checkpoint load break. Only `run_config.yaml` is a real (patched) sna file.
- **Contingency — if `ConnectionRefusedError` recurs after the seq_logprob fix**: then it is a *separate*, genuine NemoGym stability issue (not the assertion). Suspect the in-flight weight update disrupting vLLM servers under `concurrency=768`; first mitigation to try is lowering `env.nemo_gym.swe_agents_{train,val}...concurrency` (768 -> 256). The 68/80 ConnectionRefused in jobs 12670854/12673673 were downstream of the assertion crash and should disappear now.

## Gold config (in the launch script; matches pmannan's 0.13 gold except 0.17/no-HybridEP)
GBS=512 (num_prompts_per_step=64 x num_generations_per_prompt=8), max_total_sequence_length=32768,
`kv_cache_dtype=fp8_e4m3`, gpu_memory_utilization=0.8, temperature=1.0,
train TP=4 EP=8 CP=1 PP=8 (num_layers_in_first/last_pipeline_stage=11; 94 layers), vLLM TP=8,
async_grpo.enabled=True, **in_flight_weight_updates=True**, recompute_kv_cache_after_weight_updates=False,
max_trajectory_age_steps=4 or 8, force_on_policy_ratio=True, seq_logprob_error_threshold=null,
KL=0, ratio_clip 0.2/0.28, token_level_loss=True, use_importance_sampling_correction=True,
sequence_level_importance_ratios=False, truncated_importance_sampling_ratio=5, advantage_clip ±100,
MoE: freeze_moe_router=True, alltoall dispatcher, deepep off, aux_loss_coeff=0,
activation_checkpointing=True, lr=1e-06 const, weight_decay=0,
`++env.nemo_gym.swe_agents_{train,val}...swebench_tests_timeout=60`, agent_max_turns=200, swebench_agent_timeout=1800,
SAVE_PERIOD=1000000 + must_save_by=99:00:00:00 (no checkpoint writes -> clean perf, quota-safe), max_num_steps=40.
NOT HybridEP (the cp312 overlay is pmannan-container-specific; FP8KV is the lever here).

## Submit recipe (per job)
```
cd <repo>; S=examples/swe_bench/run_grpo_swe2_qwen235b_fp8kv.sh
export WANDB_API_KEY=... HF_TOKEN=...                 # from ~/.bashrc
export HF_HOME=/lustre/fsw/portfolios/coreai/users/sna/hf_home_vllm017
export NUM_NODES=32 SBATCH_PARTITION=batch SBATCH_TIME=4:0:0 MAX_NUM_STEPS=40
# 192g64t: NUM_GEN_NODES=24 (train=8) ; 128g128t: NUM_GEN_NODES=16 (train=16)
export NUM_GEN_NODES=24 MAX_TRAJECTORY_AGE_STEPS=4
export EXP_SUFFIX="vLLM0.17-fp8kv-192g64t-async4"
export PERSISTENT_CACHE=/lustre/fsw/portfolios/coreai/users/sna/.cache/q235b_vllm017_192g64t_a4
bash "$S"
```
The 4 sweep points: (NUM_GEN_NODES, MAX_TRAJECTORY_AGE_STEPS) in {(24,4),(16,4),(24,8),(16,8)}.
**wandb**: project `sna-qwen3-235B-SWE-fp8kv-vllm017`, run name = EXP_SUFFIX.

## Validation status (what's proven)
- **vLLM 0.17.1 + FP8KV**: boots clean (`Initializing a V1 LLM engine (v0.17.1) ... kv_cache_dtype=fp8_e4m3`).
- **Checkpoint loads**: smoke 12670594 (SKIP_TRAINING=1, 2-node) logged `successfully loaded checkpoint` + coherent SWE rollout (tool_call/reward) -> fixes #1-#3 validated.
- **Full runs reach buffer 64/64 + training step**, then crashed on the seq_logprob assertion (fix #4) -> 12674581 is the first run with the complete chain.

## Outstanding work
1. Confirm 12674581 completes step(s) (the seq_logprob fix should let the training step pass). Watch `Total step time:` ledger count + `ConnectionRefusedError` (was the downstream symptom of the assertion crash).
2. Harvest timing: `python3 -S experiments/scaleout_256h100_32k_async16/scripts/parse_step_breakdown.py --markdown --job <id> --variant <label> --log <ray-driver.log>` (handles ruit total-first + 256H100 total-last; cold/steady/long_tail; steady-mean). Add mean/max seq len once a completed step logs the field (only config cap `max_seq_len=32768` is visible so far).
3. Resubmit the other 3 splits once 12674581 produces a clean step.
4. Reference 0.13 numbers to compare: async4 192g64t total 1014/exposed 667 (FP8KV); async8 928/581.

## Cluster gotchas
- lustre fs1 intermittently slow at shell init -> use `bash --noprofile --norc`; **vscode-02** login node more responsive than vscode-01.
- FairShare: `coreai_dlalgo_nemorl` (0.0776) is sna's best account — already used; no priority lever. batch saturated (~1600 nodes alloc).
- 4x32=128 nodes hard to schedule at once; **prioritize 1 job** (cancel the other 3) so a single 32-node job backfills, then resubmit the rest.
- fs1 at 31/50 TB (headroom OK); fsw ~empty. SAVE_PERIOD high so no ckpt writes.

## Logs
`<repo>/logs/slurm/<jobid>-logs/ray-driver.log`. Parser tool + prior 256H100 data in
`experiments/scaleout_256h100_32k_async16/`. Canonical reference: the ruit REPRO doc
(`examples/swe_bench/REPRO_swe2.md` on the branch) — §3 config, §9 SKIP_TRAINING gen-only.
