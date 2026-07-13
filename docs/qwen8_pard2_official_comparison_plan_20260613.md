# Qwen8 Official PARD-2 Online Comparison Plan - 2026-06-13

Purpose: measure the performance impact of online PARD-2 drafter training under matched NeMo-RL settings.

Launcher:

- `experiments/eagle3_online/submit_lyris_qwen8_pard2_official_comparison_20260613.sh`
- The launcher now supports `CLUSTER_PROFILE=lyris|oci-hsg`. OCI-HSG defaults use `oci-hsg-cs-001-vscode-02`, `coreai_dlalgo_llm`, `batch`, and `04:00:00`.

Default variants:

| Variant | vLLM SpecDec | Online draft training | Drafter | K | Purpose |
| --- | --- | --- | --- | ---: | --- |
| `baseline` | disabled | disabled | none | 0 | no-spec speed baseline |
| `static_pard2` | `method=pard2` | disabled | `amd/PARD2-Qwen3-8B` | 1 | static official PARD-2 speed/acceptance |
| `online_pard2` | `method=pard2` | enabled | `amd/PARD2-Qwen3-8B` | 1 | online PARD-2 training/refit impact |

Common controls:

- Target: `Qwen/Qwen3-8B`
- `grpo.max_num_steps=10`
- `max_new_tokens=256`, `min_tokens=128`
- `num_prompts=4`, `num_generations=4`, `train_global_batch_size=16`
- `policy.generation.temperature=1.0`, `top_p=1.0`, `top_k=-1`
- Official PARD-2 target-feature vLLM site: `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/vllm-runs/pard2_official_target_feat_smoke_r3_20260612/patched_vllm_site`

Local validation:

- `scripts/validate_qwen8_pard2_comparison_contract.py`
- Latest output: `docs/qwen8_pard2_official_comparison_contract_validation_20260613.md`
- Status at `2026-06-13T02:26+02:00`: PASS.
- The validator checks the `baseline`, `static_pard2`, and `online_pard2` launcher contracts, official PARD-2 staging/preflight hooks, and tracker CSV schema.
- A tracker formatting bug was fixed before submission: the row format now emits all 14 header columns, including `base_log_dir`.
- OCI-HSG profile validation now checks the host, partition, walltime, Python/Ray settings, verified official PARD-2 vLLM site, and remote patch bundle sync path.

OCI-HSG submission:

- Launch note: `docs/oci_hsg_qwen8_pard2_official_comparison_launch_20260613.md`
- Tracker: `latest_oci_hsg_qwen8_pard2_official_comparison_20260613_jobs.csv`
- Jobs: `3286953` baseline, `3286955` static PARD-2, `3286956` online PARD-2.
- Current state at `2026-06-13T03:12+02:00`: all three are `PENDING (Priority)`.

Plan-only command:

```bash
SUBMIT=false CHECK_SSH=false RUN_LOCAL_VALIDATION=false RUN_REFRESH=false \
RUN_SWE_SUFFIX=false RUN_SWE_DRAFTER=false RUN_MATH500=false \
RUN_NEMORL_INTEGRATED=false RUN_QWEN8_PARD2_COMPARISON=true \
bash scripts/run_lyris_specdec_next_pass_20260613.sh
```

Submit after Lyris MFA/ControlMaster is active:

```bash
SUBMIT=true RUN_SWE_SUFFIX=false RUN_SWE_DRAFTER=false RUN_MATH500=false \
RUN_NEMORL_INTEGRATED=false RUN_QWEN8_PARD2_COMPARISON=true \
bash scripts/run_lyris_specdec_next_pass_20260613.sh
```

Fetch logs and summarize after jobs run:

```bash
REMOTE_HOST=oci-hsg-cs-001-vscode-02 \
TRACKER_FILES=latest_oci_hsg_qwen8_pard2_official_comparison_20260613_jobs.csv \
bash scripts/fetch_lyris_nemorl_integrated_logs.sh

python3 scripts/build_qwen8_pard2_official_comparison_report.py
```

Expected evidence:

- matched no-spec baseline row for speedup
- static PARD-2 acceptance and generation throughput
- online PARD-2 acceptance, refit steps, draft loss, and generation throughput
- speedups computed from step-2+ aggregates to avoid cold-start bias

Report outputs:

- `docs/qwen8_pard2_official_comparison_metrics_20260613.csv`
- `docs/qwen8_pard2_official_comparison_metrics_20260613.md`
- `docs/qwen8_pard2_official_online_impact_20260613.csv`
- `docs/qwen8_pard2_official_online_impact_20260613.md`

Current limitation:

- This is Qwen8 official PARD-2 target-feature validation. Qwen235B PARD-2 SWE/Math coverage remains standalone vLLM alias/native-format evidence until a compatible large-target official PARD-2 online path is established.
