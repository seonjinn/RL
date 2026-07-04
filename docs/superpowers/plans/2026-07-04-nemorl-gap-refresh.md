# NeMo-RL Gap Refresh and Safe Relaunch Plan

**Goal:** Make the latest NeMo-RL report reflect July 2-3 CUDA-graph-on performance-recipe results, then launch only the genuinely missing long-context configurations through validated baseline smokes.

**Scope:** Lyris GB200, temperature/top-p `1.0/1.0`, `enforce_eager=false`, performance recipe configs, `max_steps=20` for final runs. Existing CG-off, OCI OSL1024, and forced-sync Pretyche rows remain separate evidence.

### Task 1: Ingest July Results With Strict Baseline Keys

**Files:**
- Modify: `scripts/build_latest_specdec_html_pages.py`
- Modify: `tests/test_build_latest_specdec_html_pages.py`

1. Add the July 2 normalized result CSVs for Qwen3-30B-A3B, Qwen3-32B, and Qwen3-235B to a dedicated NeMo-RL source list.
2. Normalize their columns into the combined report schema, including CUDA graph state, resource shape, attention/MoE backend, segment, W&B URL, reward, and KL.
3. Match baselines on model, mode, OSL, temperature, top-p, CUDA graph state, cluster, resource shape, attention backend, and MoE backend.
4. Keep completed, partial, failed, held, and unmatched-baseline states distinct.
5. Rebuild and test the latest NeMo-RL CSV/HTML without changing historical dated pages.

### Task 2: Add Safe 32K Async Smoke Launchers

**Files:**
- Add: `experiments/eagle3_online/submit_lyris_qwen30_qwen32_async1off_osl32k_smoke_20260704.sh`
- Add/modify focused shell tests under `tests/`

1. Inherit the exact Qwen30 and Qwen32 async-1off performance recipes.
2. Preserve CUDA graphs, Triton attention/MoE backend, native prompt/generation counts, and recipe topology.
3. Set only the long-context safety overrides documented in `docs/qwen32_osl32k_eagle_long_context_diagnosis_20260702.md`.
4. Submit baseline smoke jobs first with `max_steps=2`; do not force `min_tokens=32768` or disable EOS.
5. Emit a reproducible manifest with repo SHA, config, nodes, segment, OSL, and W&B name.

### Task 3: Verify, Publish, and Submit

1. Run focused tests, Pyright, shell syntax checks, and `git diff --check`.
2. Rebuild the latest NeMo-RL report and visually verify model/mode sections.
3. Commit and push before submission.
4. Run launcher `--test-only`, check Lyris FairShare, submit baseline smokes, and monitor for five minutes.
5. Promote only successful baseline smokes to step20 and SpecDec K sweeps; record blocked Qwen235B NCCL failures rather than blind retries.
