# SpecDec + RL Handoff (Qwen3-30B-A3B / 32B / 235B)

Single-file handoff for resuming the speculative-decoding-under-RL work in a fresh session
(including via Codex). It covers: what was done, the validated numbers, exactly how to
reproduce each result, every landmine hit, and the open items to pick up next.

Everything here targets **Lyris GB200** unless stated. CW is blocked for vLLM >= 0.20
(driver 535.216.03; needs >= 555 / CUDA 13).

Last updated: 2026-07-23. Author context lives in `~/.claude/.../memory/MEMORY.md`.

---

## 0. TL;DR — the one thing to know

The recurring "SpecDec is net-loss on Qwen3 MoE at RL concurrency" verdict was a
**cudagraph capture-cliff artifact, not a structural cost**. Spec-verify batches are
`max_num_seqs x (K+1)` tokens; if cudagraph captures don't cover that product with
`(K+1)`-multiples, verify falls back to eager and SpecDec becomes a net loss.

With dense `(K+1)`-multiple captures on the vLLM 0.25 / V2-runner standalone stack, all
three models win on math:

| Model | TP | max_num_seqs | Best config | Standalone gen speedup |
|-------|----|--------------|-------------|------------------------|
| Qwen3-30B-A3B | 1 | 128 | eagle3 K3 | **2.19x** (3-seed 2.192±0.007) |
| Qwen3-32B | 2 | 256 | eagle3 K3 | **1.27x** (was 0.92x pre-capfix) |
| Qwen3-235B-A22B | 4 | 64 | eagle3 **K5** | **1.71x** (K3 1.66x) |

The old base-235B "eagle3 net-loss 0.31–0.44x" verdict is **retracted** — it used recipe
captures `[1..64]` while K3 verify needs `256`, on the 0.24-era stack.

The catch: these gains are **engine-level (standalone)**. E2E NeMo-RL gains beyond 30B
require a **vLLM >= 0.24 container rebake** — the production E2E container is vLLM 0.20,
where FULL-graph spec verify (the mechanism, PR #45953) does not exist and capture
overrides are inert.

---

## 1. Environments (exact paths)

### Cluster / SLURM
- Cluster: **Lyris** (GB200, aarch64 compute / **x86 login node** — venv python only runs under `srun -p gb200`), MFA required.
- SSH alias: `login-lyris`. Account: `coreai_dlalgo_llm`. Partition: `gb200` (5h) / `batch_long` (7d).
- Driver 580.173. Compute nodes have direct internet; login node does not run aarch64 binaries.
- aarch64 `uv`: `/lustre/fsw/coreai_dlalgo_llm/users/sna/bin_aarch64/uv`

### vLLM venvs (standalone benchmarking)
- **vllm025** (primary): `/lustre/fsw/coreai_dlalgo_llm/users/sna/venvs/vllm025` — vLLM 0.25.0, torch 2.11 cu130, flashinfer 0.6.13. **Locally patched** (zerodiv guard + Qwen3MoE V2 auto-enable; `.orig` kept). `VLLM_USE_V2_MODEL_RUNNER=1` REQUIRED for Qwen3MoE (silent PIECEWISE downgrade otherwise).
- vllm024 (legacy): `/lustre/fsw/coreai_dlalgo_llm/users/sna/venvs/vllm024` — vLLM 0.24.0. Carries the per-K FULL-graph handicap; superseded by 0.25.

### NeMo-RL E2E stacks
- **E2E production container**: `nemo_rl_nightly_20260715.sqsh` = vLLM **0.25.1** (nightly worktree stack). Driver `/opt/nemo_rl_venv` + prebuilt lustre venvs `nrl_venvs_dynsd025` (pass `NEMO_RL_VENV_DIR` to avoid the baked `/opt/ray_venvs` vLLM 0.20).
- Legacy E2E container: vLLM **0.20** (the `submit_lyris_nemorl_perfcfg_specdec_matrix_*` default). Capture overrides are inert here.
- NeMo-RL Lyris clone (perfcfg path): `/project/coreai_dlalgo_llm/users/sna/RL-latest-main-canary-20260618/`
- NemoGym worktree: `RL-vllm0251-eagle3-fullcg-final-20260715` @f868c977 + PR#3243 cherry-picks (branch `nemogym-dynsd`).

### Shared data / cache
- `HF_HOME`: `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home` (Qwen3 0.6B/8B/30B/32B/235B + all eagle3/dflash drafts cached). Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for many-worker jobs (avoids HF 429).
- Prompts: `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/data/{openmath2_prompts_2048.jsonl, math_500_data_prompts_20260612.jsonl, swebench_verified_prompts_all.jsonl, dapo_*}`
- Remote run outputs: `vllm-benchmark/dynamic_sd_runs/<tag>/` on Lyris.

---

## 2. Models & drafters

Targets = **NeMo-RL recipe base models** (the verifier). Drafters = **Thinking speculators**
(RedHatAI / our trained ones). A Thinking-2507 drafter is compatible with the base target.

| Target (verifier) | TP | eagle3 drafter | dflash drafter (ours) |
|-------------------|----|-----------------|-----------------------|
| Qwen/Qwen3-30B-A3B | 1 | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` | `drafters/dflash_30bthink_{kl10k,mixswe}` |
| Qwen/Qwen3-32B | 2 | `RedHatAI/Qwen3-32B-Thinking-speculator.eagle3` | — |
| Qwen/Qwen3-235B-A22B(-Thinking-2507) | 4 | `RedHatAI/...235B...speculator.eagle3` | `/lustre/.../users/sna/drafters/dflash_235bthink_v1` (600K generic x3ep, val EAL 1.967) |

- Our 235B dflash **v1** rollout verdict: Thinking baseline 87 tok/s → v1@K5 116 tok/s = **1.33x** (unseen astropy).
- 235B dflash **v2** (850K mix incl. own-pool + rebench-openhands, seq 16384) was training as of 2026-07-21; verify checkpoint survival before use.
- **DFlash landmine**: drafter MUST run `attention_backend=FLASH_ATTN` (drafter-level), `draft_tensor_parallel_size=1`, `kernel_config.enable_flashinfer_autotune=false`. FLASHINFER drafter = CUDA illegal memory access. DFlash trained-block ≈ 5 → K plateau K2–K5, collapse at K>=7.
- **eagle3** has no trained-block cliff (AL grows with K) → optimum is cost-driven (235B K5).

---

## 3. Repo layout — key files

### `experiments/dynamic_sd_sync_rollout/` (standalone harness — the source of truth)
- `sync_rollout_dynamic_sd.py` — main harness (`--mode profile|rollout|replay`). Sync GRPO rollout: N prompts x G gens, barrier per step, temp/top_p 1.0, seed 42.
- `submit_lyris_dynamic_sd.sh` — job submitter. `spec_json_fixed()` builds the spec-config JSON. Knobs: `SPEC_METHOD` (eagle3/mtp/dflash), `FIXED_K`, `CUDAGRAPH_SIZES`, `DISABLE_FLASHINFER_AUTOTUNE`, `ATTENTION_BACKEND`, always `--disable-custom-all-reduce`.
- `submit_matrix_lyris.sh` — per-model presets (`qwen3_30ba3b`, `qwen3_30ba3b_40k`, `qwen3_32b`, `qwen3_235b`) x benches (`math500 openmath dapo swe_verified swe_full`).
- `derive_dynamic_k_table.py` — builds BS→K table; `--max-capture-tokens 512` caps K analytically to avoid the cliff.
- `configs/*.yaml` — **4 validated reproducible configs** (the canonical recipes; each has a `reproduce:` block):
  - `qwen3_30ba3b_eagle3_k3_math.yaml` (2.19x)
  - `qwen3_32b_eagle3_k3_math.yaml` (1.27x, captures to 1024)
  - `qwen3_235b_eagle3_k5_math.yaml` (1.71x, NEW BEST)
  - `qwen3_235b_dflash_v1_k5_swe.yaml` (1.39x SWE / 1.55x math)
- `PATCH_LEDGER.md` — all vLLM venv patches + perf impact (rendered on results page).
- `patches/`, `upstream/` — vLLM patches and 3 upstream PR drafts (zerodiv, mamba assert, Qwen3MoE V2 auto-enable).
- `summarize_results.py` → `plot_results.py` → `scripts/build_dynamic_sd_results_page.py` → `build_pages_index.py` — results-page pipeline.
- `latest_lyris_*_jobs.txt` — job-ID ledgers per (model, bench, method, K).

### `experiments/eagle3_online/` (E2E NeMo-RL launchers)
- `submit_lyris_nemorl_perfcfg_specdec_matrix_20260617.sh` — **the E2E launcher**. Env knobs: `MODELS` (qwen30ba3b/qwen32b/qwen235b), `METHODS` (baseline eagle3), `MAX_STEPS`, `SUBMIT`, `EAGLE3_SPEC_TOKENS` (K), `QWEN30_EAGLE3_DRAFT_MODEL`, `EXTRA_OVERRIDES`. wandb key MUST come from Lyris `~/.netrc` (see landmines).
- Many `submit_lyris_qwen*`/`submit_qwen*` — historical PARD/eagle3/suffix E2E attempts (mostly superseded; keep for provenance).

### NemoGym + SWERL (live in worktrees, not main)
- NemoGym launcher: `experiments/nemogym_swe1_specdec/submit_lyris.sh` (envs `VARIANT/METRICS/TAG/EXTRA_OVERRIDES/CONFIG/DATA`).
- SWERL 235B full-GRPO smoke: `swerl_fullgrpo_launchers/20260721_swerl_235b_dflash_v1_smoke/run_swerl_235b_dflash_smoke.sh`.

---

## 4. Validated scoreboard (all measured)

### Standalone, vLLM 0.25, dense captures (engine-level truth)
- **30B-A3B math** eagle3: K3 **2.19x**. E2E (0.20) gen 1.62x → step 1.07x.
- **32B math** eagle3 K3: **1.27x** with captures to 1024 (was 0.92–0.96x with default 512).
- **235B math** eagle3: K3 1.66x / **K5 1.71x (best)** / K7 1.58x (AL grows 2.55→3.0, no cliff).
- **235B math** dflash-v1: K1 1.37 / **K2 1.59 (dflash optimum)** / K3 1.58 / K5 1.55 / K7 1.03 / K9 0.99 / K16 0.76.
- **235B SWE-verified (1-turn)** dflash-v1: K5 **1.39x** (AL 2.25) > eagle3 K3 1.27x (generic-drafter gap).
- **Nemotron3 MTP** (in-ckpt): Super 120B K3 1.47–1.50x; **Ultra 550B K3 1.56–1.75x** (flips "MoE-scale net-loss" — it was external-drafter-specific).

### E2E NeMo-RL
- **30B-A3B 4n4g eagle3-K3** (0.20 container): gen 69.3→42.8s (1.62x), E2E step 217.5→203.1s (**1.07x**), train/logprob untouched (Amdahl-consistent). Replicated twice.
- **32B 4n4g eagle3-K3** (0.20): gen 0.93x (slower), E2E 1.00x — capture override **inert** on 0.20 (mechanism is 0.24+/V2). Standalone 1.27x needs the 0.25 stack.
- **235B 32n4g E2E**: structurally walled on 0.20 (see landmines).

### NemoGym real-env (vLLM 0.25.1)
- SWE1: vLLM 0.20 → 0.25.1 upgrade alone = **1.88x** (base 524s → 264–279s). But SpecDec itself loses on SWE1 (prefill-bound 22:1, BS~64/engine): K3-fixed 0.87x, DynSD 0.79x — baseline wins.
- SWE2 multi-turn (30B): baseline 209 tok/s < eagle3-K1 218 (1.04x) < **dflash-K9 250–310 (1.20–1.48x, best)**; eagle3-K3 189 (0.90x, long-ctx acceptance collapse >8K); suffix/ngram all lose (proposal cost scales with context).
- **Domain-data lever**: 30B mix-own drafter K5 = 313 tok/s (1.50x, beats public-800K). Instance-pool familiarity is the dominant axis; clean public SWE plateaus ~275–280.

---

## 5. Reproduce recipes (copy-paste)

All standalone runs go through `submit_matrix_lyris.sh` (presets) or `submit_lyris_dynamic_sd.sh`
(raw envs). Each `configs/*.yaml` has a self-contained `reproduce:` block — prefer those.

### 5a. 30B-A3B eagle3-K3 math (2.19x) — standalone
```bash
ssh login-lyris 'cd /project/coreai_dlalgo_llm/users/sna/RL-latest-main-canary-20260618 && git pull --ff-only && \
  MODE=rollout ROLLOUT_VARIANTS=fixed FIXED_K=3 \
  CUDAGRAPH_SIZES="1 2 4 8 12 16 24 32 48 64 96 128 192 256 384 512" \
  PYTHON_BIN=/lustre/fsw/coreai_dlalgo_llm/users/sna/venvs/vllm025/bin/python VLLM_USE_V2_MODEL_RUNNER=1 \
  bash experiments/dynamic_sd_sync_rollout/submit_matrix_lyris.sh qwen3_30ba3b math'
```

### 5b. 32B eagle3-K3 math (1.27x, captures to 1024) — standalone
```bash
MODEL="Qwen/Qwen3-32B" MODEL_LABEL="qwen3_32b_capfix" TP=2 BENCH=openmath \
PROMPT_JSONL=<remote>/data/openmath2_prompts_2048.jsonl ISL_CAP=1024 \
MAX_MODEL_LEN=8192 MAX_NUM_SEQS=256 NUM_PROMPTS_PER_STEP=8 NUM_GENERATIONS_PER_PROMPT=32 NUM_STEPS=3 MAX_TOKENS=4096 \
PYTHON_BIN=<venv>/bin/python VLLM_USE_V2_MODEL_RUNNER=1 MODE=rollout SPEC_METHOD=eagle3 \
DRAFT_MODEL=RedHatAI/Qwen3-32B-Thinking-speculator.eagle3 DISABLE_FLASHINFER_AUTOTUNE=true \
ROLLOUT_VARIANTS=fixed FIXED_K=3 CUDAGRAPH_SIZES="1 2 4 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024" \
bash experiments/dynamic_sd_sync_rollout/submit_lyris_dynamic_sd.sh
```

### 5c. 235B eagle3-K5 math (1.71x, best) — standalone
Use `configs/qwen3_235b_eagle3_k5_math.yaml`. TP4, max_num_seqs 64, captures `[1,2,4,6,12,24,48,96,192,384]` (covers 64x(5+1)=384). Preset: `submit_matrix_lyris.sh qwen3_235b math` with `FIXED_K=5`.

### 5d. 235B dflash-v1 K5 SWE (1.39x) — standalone
Use `configs/qwen3_235b_dflash_v1_k5_swe.yaml` (has full reproduce block). Note the 3 dflash landmines.

### 5e. E2E NeMo-RL 30B eagle3-K3 (20-step wandb)
```bash
# On Lyris, in the perfcfg clone. wandb key must resolve from ~/.netrc.
MODELS=qwen30ba3b METHODS="baseline eagle3" EAGLE3_SPEC_TOKENS=3 MAX_STEPS=20 SUBMIT=true \
  bash experiments/eagle3_online/submit_lyris_nemorl_perfcfg_specdec_matrix_20260617.sh
# wandb project: nvidia/nemo-rl-perfcfg-specdec-lyris
```

### 5f. NemoGym SWE1/SWE2 (vLLM 0.25.1)
Requires the `nemogym-dynsd` worktree + 2 fixes baked in (openai clamp, dense captures).
Launcher `experiments/nemogym_swe1_specdec/submit_lyris.sh`; for SWE2 pass
`CONFIG=grpo_qwen3_30ba3b_thinking_swe2_smoke.yaml DATA=data/swe2/val-mini3.jsonl NUM_PROMPTS=3 NUM_GENS=1`
and dense `EXTRA_OVERRIDES=++...compilation_config.cudagraph_capture_sizes=[1..512]` (unescaped brackets).

---

## 6. Landmines / trial-and-error (do not rediscover these)

### Capture cliff (THE central lesson)
- Spec-verify batch = `max_num_seqs x (K+1)` tokens. Captures must be `(K+1)`-multiples up to that product, else eager fallback → net loss. Default cap is 512; 32B (256 seqs, K3) needs 1024, so it looked "structurally weak" until fixed.
- `derive_dynamic_k_table.py --max-capture-tokens 512` caps K analytically. Profiled grid points alone cannot see between-point cliffs.
- vLLM 0.25.1 bug: setting `max_cudagraph_capture_size=512` directly dies with `TypeError: cannot pickle pydantic_core.ArgsKwargs` → use explicit `cudagraph_capture_sizes` list instead.

### vLLM venv patches (in `PATCH_LEDGER.md`, `.orig` kept)
1. DynamicSD + eagle3 engine-init `ZeroDivisionError` in `cudagraph_utils.py _init_candidates` (drafter reuses target formula → negative → round_up(x,0)). Guard `num_new>0`.
2. NemotronH + DynamicSD per-K capture: `mamba_attn.py:183` strict `assert max_query_len == 1+maxK` → patch to `<=`.
3. Qwen3MoE not in V2 auto-list → silent PIECEWISE downgrade → must set `VLLM_USE_V2_MODEL_RUNNER=1`.

### Attention / all-reduce
- `TP>=2 + eagle3 + full cudagraph` crashes custom all-reduce (`custom_all_reduce.cuh:455`) → ALWAYS `--disable-custom-all-reduce`.
- DFlash drafter must be `FLASH_ATTN` (FLASHINFER → CUDA illegal memory access), `draft_tp=1`, autotune off.
- `VLLM_ATTENTION_BACKEND` env is dead in 0.24+; use the `attention_backend` engine arg.
- GB200 has no nvcc → need `flashinfer-jit-cache==*+cu130` prebuilt.

### E2E NeMo-RL integration
- `use_system_env=false` + repo uv.lock does NOT swap the generation-worker vLLM — the container-baked `/opt/ray_venvs` (vLLM 0.20) wins. Only the training venv rebuilds. Full E2E DynamicSD needs `NRL_FORCE_REBUILD_VENVS` or a **container rebake**.
- Shared-lustre `NEMO_RL_VENV_DIR` write race between concurrent jobs poisons the venv (`ModuleNotFoundError: ray` crash loop). Pin per-job venv dirs.
- `SpeculativeConfig` on the 0.25.1 GRPO-train build rejects `attention_backend` kwarg → drop it from overrides in the GRPO-train path (env `VLLM_ATTENTION_BACKEND` covers it).
- wandb: new-format keys are rejected by the old client ("user is not logged in"); missing key silently kills runs. Extract from Lyris `~/.netrc`: `awk '/machine api.wandb.ai/{f=1} f && /password/{print $2; exit}'`.
- E2E capture-size override is **inert on the 0.20 container** (FULL-graph spec verify is a 0.24+/V2 mechanism, PR #45953). Confirmed by job 2463893 (32B, no effect).

### 235B E2E wall (fully mapped — do not retry blindly)
- Layer 1: hybridep dispatcher CU error at init → `moe_token_dispatcher_type=alltoall`.
- Layer 2: SHARP "Exceeded reservation resources limit" → drop `--network=sharp` (`SBATCH_EXTRA_ARGS`) + `NCCL_COLLNET_ENABLE=0`.
- Layer 3 (the real one): vLLM **TP8 cross-node ray executor** → TP worker `ActorDiedError` → `EngineDeadError`×199 crash loop.
- TP4 workaround → **host-RAM OOM** (470GB weight staging on one ~480GB LPDDR node; TP8 spreads it across 2 nodes — that's why the recipe uses TP8). Also TP4 eagle3 hits `SpeculativeConfig` ValidationError (script hardcodes draft_tp=8).
- **Conclusion**: 0.20-container 235B E2E is walled both ways. Needs the rebake.

### NemoGym / SWERL
- Gym `global_config.py` injects parent `openai==2.44.0` into subprocess venvs vs nemo-gym `openai<=2.7.2` → clamp to `2.7.2` (patch `gym_openai_clamp.patch`). Still required on latest Gym main (upstream-worthy).
- SWE2 session hang root cause: `setup_scripts/openhands.sh:55` hardcodes `jq-linux-amd64` on arm64 → Exec format error → tmux pane death → all commands pid=-1 timeout. Fix: `jq-linux-arm64`.
- `/dev/fuse:/dev/fuse` must be in MOUNTS for apptainer/squashfuse inside enroot.
- SWERL 235B full-GRPO smoke ladder (2458513→2464183): reached **megatron→vLLM refit SUCCESS with dflash engines** + real rollouts, then died on AsyncTrajectoryCollector robustness (empty-input `IndexError` in error path at `nemo_gym.py:444`; Ray worker-lease failure at 24-node concurrency). Bounded-wait patch draft exists at `experiments/scaleout_256h100_32k_async16/patches/`.
- Async collector `wait_for_pending_generations` (async_utils.py) is an unbounded while-True — root cause of hang-family failures.

### Publish pipeline
- `docs/specdec_current_nemorl_early_speedup.png` (5.3MB) gets copied to `public/` on every `build_pages_index` run and GitLab rejects >5MiB pushes. Before commit: `git checkout origin/main -- public/reports/specdec_current_nemorl_early_speedup.png`.

---

## 7. What's done vs open

### Done (validated)
- [x] Standalone sync-rollout harness (profile/rollout/replay modes), 3-seed stats.
- [x] Capture-cliff diagnosis + fix; all 3 Qwen3 models flipped to positive on math.
- [x] eagle3 K-curves (235B optimum K5 1.71x); dflash K-curves (optimum K2–K5).
- [x] E2E 30B eagle3-K3 (1.07x, Amdahl-consistent) on 0.20 container.
- [x] NemoGym SWE1/SWE2 full validation on 0.25.1 (dflash 1.20–1.48x best on SWE2).
- [x] Custom dflash drafter training pipeline (speculators, kl_div loss, domain-data lever); 235B v1 = 1.33x.
- [x] Nemotron3 MTP (Super/Ultra) — MoE-scale net-loss verdict reversed.
- [x] 4 reproducible config YAMLs + PATCH_LEDGER + upstream PR drafts.

### Open / next (pick up here)
1. **Container rebake — the unblocker for E2E >30B.** Bake a NeMo-RL container with vLLM >= 0.24 + the 3 ledger patches, so E2E 32B/235B can realize the standalone 1.27x/1.71x. This is the single highest-leverage item.
2. **235B E2E full-GRPO** — resume SWERL smoke from the AsyncTrajectoryCollector robustness fix (bounded wait + empty-input guard), lower gen-node concurrency to avoid Ray worker-lease storm.
3. **235B dflash v2** — verify checkpoint survival (50T cleanup risk), then rollout-eval vs v1 (target 1.5x via instance-pool adaptation).
4. **Results page correction + publish** (task #7) — the capture-cliff reversal and new K-curves make the current published page wrong. Regenerate via the summarize→plot→build_pages pipeline, mind the 5MiB PNG gotcha.
5. Optional: 30B/32B eagle3 K5 probes; DFlash config yaml for 30B agentic SWE; depth×BS-aware DynamicSD scheduler (needs dispatch-aware upstream impl — analytic model preferred).

---

## 8. Codex kickoff (fresh session)

To resume with Codex:
1. Read this file + `experiments/dynamic_sd_sync_rollout/README.md` + `configs/*.yaml` (`reproduce:` blocks) + `PATCH_LEDGER.md`.
2. Confirm SSH: `ssh login-lyris 'echo ok; sacct -u $USER --starttime today -X -o JobID,JobName%40,State | tail'`.
3. Sanity-repro the cheapest validated result first: **5a (30B eagle3-K3)**, expect ~2.19x gen. If that lands, the stack is healthy.
4. Then choose an open item from §7. Item 1 (rebake) unblocks the most.
5. Job-ID ledgers are in `experiments/dynamic_sd_sync_rollout/latest_lyris_*_jobs.txt` and `experiments/eagle3_online/latest_lyris_*_jobs.csv` — cross-reference for provenance.

Follow the repo rules: commit + push before submitting, `git pull` on remote, monitor 5 min
after RUNNING, results/logs stay in the experiment dir, never commit files > 25MB.
