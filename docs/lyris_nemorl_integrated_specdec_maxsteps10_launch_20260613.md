# Lyris NeMo-RL Integrated SpecDec Max-Step-10 Launch - 2026-06-13

Submitted through:

- `experiments/eagle3_online/submit_lyris_nemorl_integrated_specdec_matrix_20260613.sh`

Trackers:

- `latest_lyris_nemorl_integrated_specdec_maxsteps10_20260613_jobs.csv`
- `latest_lyris_nemorl_integrated_specdec_maxsteps10_20260613_dryrun.csv`
- `latest_lyris_nemorl_integrated_specdec_maxsteps10_20260613_raymatch_jobs.csv`

Refresh helper:

- `scripts/refresh_lyris_nemorl_integrated_specdec_results.sh`
- `scripts/fetch_lyris_nemorl_integrated_logs.sh`
- `scripts/summarize_lyris_nemorl_integrated_specdec.py`

Latest local launcher update:

- The integrated matrix default methods are now `baseline suffix pard eagle3`.
- `baseline` uses `DRAFT_FORMAT=auto`, `POLICY_DRAFT_ENABLED=false`, and `ENABLE_VLLM_SPECDEC=false`.
- Local contract validation now includes `baseline-no-spec` and verifies that no `speculative_config` override is emitted for the baseline path.
- The Qwen8 official PARD-2 comparison launcher now submits matched `baseline`, `static_pard2`, and `online_pard2` cells for online-training impact measurement.
- Qwen8 comparison validation now checks the launcher contract and tracker CSV schema locally; `docs/qwen8_pard2_official_comparison_contract_validation_20260613.md` is PASS, and the tracker row emits all 14 columns including `base_log_dir`.
- Metrics output is written to:
  - `docs/lyris_nemorl_integrated_specdec_maxsteps10_metrics_20260613.csv`
  - `docs/lyris_nemorl_integrated_specdec_maxsteps10_metrics_20260613.md`
- Log fetch output is written under `tmp/lyris_nemorl_integrated_logs/<run_id>/<model>_<method>/`, with manifest `docs/lyris_nemorl_integrated_specdec_maxsteps10_log_fetch_manifest_20260613.csv`.

Remote setup:

- Remote repo: `/lustre/fsw/coreai_dlalgo_llm/users/sna/SpecDec-RL-pard2-official-online-lyris-20260612`
- Account / partition: `coreai_dlalgo_llm` / `gb200`
- Container: `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo-rl-nightly-ultra.sqsh`
- HF cache: `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home`
- Suffix site: `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/.container_cache/arctic-inference-0.1.1-py313`

Pre-submit checks:

- Local `bash -n` passed for the new matrix wrapper and the generic submit helper.
- Remote preflight verified `ray.sub`, `examples/run_grpo.py`, Qwen30/Qwen32/Qwen235 recipe YAMLs, the container, the arctic suffix module, and required SpecDec patch markers.
- Remote dry-run passed for all 9 cells before submission.
- Retry dry-run verified that the GRPO driver now uses `uv run --python /opt/nemo_rl_venv/bin/python3` instead of forcing `/opt/nemo_rl_venv/bin/python`.

Common run controls:

- `grpo.max_num_steps=10`
- `policy.generation.max_new_tokens=1024`
- `NRL_VLLM_GENERATION_MIN_TOKENS=512`
- `policy.generation.temperature=1.0`, `top_p=1.0`, `top_k=-1`
- `checkpointing.enabled=false`
- `grpo.async_grpo.enabled=false`
- `policy.generation.vllm_cfg.async_engine=false`
- W&B disabled for this submission batch

Initial matrix, superseded:

| Job | Model | Method | Target | Drafter | K | Shape | State at submit check |
| ---: | --- | --- | --- | --- | ---: | --- | --- |
| `2109933` | Qwen3-30B-A3B | suffix | `Qwen/Qwen3-30B-A3B` | model-free suffix | 32 | 4x4, GBS512 | `PENDING (Priority)` |
| `2109934` | Qwen3-30B-A3B | PARD | `Qwen/Qwen3-30B-A3B` | `amd/PARD-Qwen3-0.6B` | 5 | 4x4, GBS512 | `PENDING (Priority)` |
| `2109935` | Qwen3-30B-A3B | Eagle-3 | `Qwen/Qwen3-30B-A3B` | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` | 3 | 4x4, GBS512 | `PENDING (Priority)` |
| `2109936` | Qwen3-32B | suffix | `Qwen/Qwen3-32B` | model-free suffix | 32 | 4x4, GBS512 | `PENDING (Priority)` |
| `2109937` | Qwen3-32B | PARD | `Qwen/Qwen3-32B` | `amd/PARD-Qwen3-0.6B` | 5 | 4x4, GBS512 | `PENDING (Priority)` |
| `2109938` | Qwen3-32B | Eagle-3 | `Qwen/Qwen3-32B` | `RedHatAI/Qwen3-32B-speculator.eagle3` | 3 | 4x4, GBS512 | `PENDING (Priority)` |
| `2109939` | Qwen3-235B-A22B | suffix | `Qwen/Qwen3-235B-A22B` | model-free suffix | 32 | 32x4, GBS256 | `PENDING (Priority)` |
| `2109940` | Qwen3-235B-A22B | PARD | `Qwen/Qwen3-235B-A22B` | `amd/PARD-Qwen3-0.6B` | 5 | 32x4, GBS256 | `PENDING (Priority)` |
| `2109942` | Qwen3-235B-A22B | Eagle-3 | `Qwen/Qwen3-235B-A22B` | `nvidia/Qwen3-235B-A22B-Eagle3` | 3 | 32x4, GBS256 | `PENDING (Priority)` |

The first Qwen30 cells (`2109933`, `2109934`, `2109935`) failed before GRPO because the Ray cluster started with Ray `2.54.0` while the driver was forced to `/opt/nemo_rl_venv/bin/python` with Ray `2.49.2`. The pending first-batch Qwen32/Qwen235B jobs were cancelled before useful work because they had the same launch setting.

Active retry matrix:

| Job | Model | Method | Target | Drafter | K | Shape | State at retry submit check |
| ---: | --- | --- | --- | --- | ---: | --- | --- |
| `2109990` | Qwen3-30B-A3B | suffix | `Qwen/Qwen3-30B-A3B` | model-free suffix | 32 | 4x4, GBS512 | `RUNNING` |
| `2109991` | Qwen3-30B-A3B | PARD | `Qwen/Qwen3-30B-A3B` | `amd/PARD-Qwen3-0.6B` | 5 | 4x4, GBS512 | `RUNNING` |
| `2109992` | Qwen3-30B-A3B | Eagle-3 | `Qwen/Qwen3-30B-A3B` | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` | 3 | 4x4, GBS512 | `RUNNING` |
| `2109993` | Qwen3-32B | suffix | `Qwen/Qwen3-32B` | model-free suffix | 32 | 4x4, GBS512 | `RUNNING` |
| `2109994` | Qwen3-32B | PARD | `Qwen/Qwen3-32B` | `amd/PARD-Qwen3-0.6B` | 5 | 4x4, GBS512 | `PENDING (Priority)` |
| `2109995` | Qwen3-32B | Eagle-3 | `Qwen/Qwen3-32B` | `RedHatAI/Qwen3-32B-speculator.eagle3` | 3 | 4x4, GBS512 | `PENDING (Priority)` |
| `2110001` | Qwen3-235B-A22B | suffix | `Qwen/Qwen3-235B-A22B` | model-free suffix | 32 | 32x4, GBS256 | `PENDING (MaxNodeRunMinsPerUser)` |
| `2110002` | Qwen3-235B-A22B | PARD | `Qwen/Qwen3-235B-A22B` | `amd/PARD-Qwen3-0.6B` | 5 | 32x4, GBS256 | `PENDING (MaxNodeRunMinsPerUser)` |
| `2110003` | Qwen3-235B-A22B | Eagle-3 | `Qwen/Qwen3-235B-A22B` | `nvidia/Qwen3-235B-A22B-Eagle3` | 3 | 32x4, GBS256 | `PENDING (MaxNodeRunMinsPerUser)` |

Latest retry log check: Qwen30 retry driver logs show `Using CPython 3.12.12 interpreter at: /opt/nemo_rl_venv/bin/python3`, fresh `.driver_venvs/...raymatch` creation, and Ray package download/build. The previous Ray version mismatch has not reappeared at this stage.

## Log-Safe Retry

The first retry logs were lost when a later shared-stage PARD-2 submission restaged the repo with `rsync --delete-excluded`. The launchers are now patched so new Slurm output goes to a run-specific `LOG_ROOT`, and PARD-2 official staging no longer deletes excluded logs.

Local validation passed on 2026-06-13 at `02:26+02:00`:

```bash
bash -n experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh \
  experiments/eagle3_online/submit_lyris_nemorl_integrated_specdec_matrix_20260613.sh \
  experiments/eagle3_online/submit_lyris_qwen8_pard2_official_comparison_20260613.sh \
  scripts/fetch_lyris_nemorl_integrated_logs.sh
python3 scripts/validate_nemorl_online_specdec_contract.py
python3 scripts/validate_nemorl_pard_source_bundle.py
python3 scripts/validate_qwen8_pard2_comparison_contract.py
```

Submit Qwen30/Qwen32 log-safe retries after Lyris MFA is active:

```bash
RUN_ID=20260613_nemorl_integrated_specdec_step10_logsafe \
LOG_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl-integrated-specdec-logs/20260613_nemorl_integrated_specdec_step10_logsafe \
MODELS='qwen30ba3b qwen32' \
METHODS='baseline suffix pard eagle3' \
SUBMIT=true \
OUT=latest_lyris_nemorl_integrated_specdec_maxsteps10_20260613_logsafe_jobs.csv \
bash experiments/eagle3_online/submit_lyris_nemorl_integrated_specdec_matrix_20260613.sh
```

The preferred path is the guarded next-pass wrapper, which validates locally, refreshes existing trackers first, and refuses `SUBMIT=true` unless Lyris SSH is verified:

```bash
SUBMIT=true RUN_SWE_SUFFIX=false RUN_SWE_DRAFTER=false RUN_MATH500=false \
bash scripts/run_lyris_specdec_next_pass_20260613.sh
```

To run only the Qwen8 official PARD-2 online impact comparison:

```bash
SUBMIT=true RUN_SWE_SUFFIX=false RUN_SWE_DRAFTER=false RUN_MATH500=false \
RUN_NEMORL_INTEGRATED=false RUN_QWEN8_PARD2_COMPARISON=true \
bash scripts/run_lyris_specdec_next_pass_20260613.sh
```

After jobs start or finish, the same wrapper refreshes status and fetches lightweight integrated logs before rebuilding the metrics table. To fetch only logs and regenerate the integrated metrics table:

```bash
bash scripts/fetch_lyris_nemorl_integrated_logs.sh
```

Do not duplicate the Qwen235B integrated cells until Slurm state is refreshed for `2110001`, `2110002`, and `2110003`. If those jobs are gone or failed, submit only Qwen235B with a distinct run id:

```bash
RUN_ID=20260613_nemorl_integrated_specdec_step10_qwen235b_logsafe \
LOG_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl-integrated-specdec-logs/20260613_nemorl_integrated_specdec_step10_qwen235b_logsafe \
MODELS='qwen235b' \
METHODS='baseline suffix pard eagle3' \
SUBMIT=true \
OUT=latest_lyris_nemorl_integrated_specdec_maxsteps10_20260613_qwen235b_logsafe_jobs.csv \
bash experiments/eagle3_online/submit_lyris_nemorl_integrated_specdec_matrix_20260613.sh
```

Notes:

- The Qwen30 Eagle-3 cell uses the cached Thinking-2507 Eagle-3 speculator because no exact non-Thinking Qwen30 Eagle-3 drafter was cached on this Lyris account.
- Slurm accepted every job and emitted only job-name-format warnings.
- These are NeMo-RL integrated Full-GRPO jobs with vLLM speculative decoding enabled, not standalone vLLM benchmark jobs.
- The old submitted trackers contain only speculative cells, so integrated speedup versus no-spec cannot be measured from those trackers alone. The next log-safe run should include the baseline cells above.
- The integrated matrix does not default to PARD-2 for Qwen30/Qwen32/Qwen235B because the official PARD-2 online path currently has validated Qwen8 target-feature plumbing, while Qwen235B PARD-2 SWE/Math evidence is standalone vLLM alias/native-format evidence.
