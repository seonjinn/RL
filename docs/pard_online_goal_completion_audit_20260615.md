# PARD/PARD-2 Online Goal Completion Audit - 2026-06-15

Objective: make PARD/PARD-2 online drafter training in NeMo-RL, evaluate the
performance impact of online drafter training, and apply PARD/PARD-2
speculative decoding to Math and SWEBench.

## Current Verdict

The goal is not complete. Current evidence proves the launcher contract,
Qwen3-8B and Qwen3-30B online PARD-2 runtime/performance comparisons, and 235B
SWE-RL baseline plus non-PARD-2 speculative short runs. It does not yet prove
235B SWE-RL PARD-2, 235B MathRL, or a 235B online-training performance result.

## Requirement Audit

| Requirement | Status | Evidence | Remaining proof needed |
| --- | --- | --- | --- |
| NeMo-RL launcher supports baseline, suffix, PARD static, PARD online, and PARD-2 online modes | Proven at launcher-contract level | `python3 scripts/validate_nemorl_online_specdec_contract.py` reports PASS for `baseline-no-spec`, `suffix-static`, `pard-static`, `pard-online`, `pard2-online`, and the k-slot guard. | Runtime proof for PARD online on the target 235B workflows. |
| Online PARD-2 drafter refit runs end to end | Proven on Qwen3-8B and Qwen3-30B | Jobs `3288181`, `3288182`, `3288183` completed for Qwen8; Qwen30 online comparison jobs `3265387`, `3265388`, and `3274811` completed against static `3265386`. | 235B PARD-2 runtime proof on SWE-RL and/or MathRL. |
| Online drafter training performance impact is evaluated | Partial | Qwen3-8B online/static comparison shows online PARD-2 was `0.9696x` gen-worker TPS and `0.8087x` E2E TPS vs static, despite higher acceptance. Qwen30 long-output online/static comparison shows roughly flat gen-worker TPS (`1.0015x`, `0.9932x`, `0.9919x`) with lower acceptance than static. | 235B online/static performance comparison after PARD-2 gates produce usable metrics. |
| 235B SWE-RL baseline short run works | Proven | Job `3299487` completed; step>=2 summary: 8 parsed steps, `103.75` E2E tok/s/GPU, `213.06` gen-worker tok/s/GPU. | None for baseline short-run evidence. |
| 235B SWE-RL PARD short run works | Proven for static/non-online PARD K5 | Job `3299489` completed; step>=2 summary: 5 parsed steps, `27.38` E2E tok/s/GPU, `57.07` gen-worker tok/s/GPU. | PARD online-training runtime proof, if claiming online PARD rather than static PARD. |
| 235B SWE-RL PARD-2 short run works | Missing | Earlier job `3299490` failed on staged PARD-2 vLLM `_C.abi3.so` Torch/C10 ABI mismatch. Current proof job `3308774` is pending. | `3308774` or an equivalent retry must start, reach steps, and produce parsed metrics. |
| 235B MathRL baseline/PARD short runs work | Missing | Current jobs `3315380`, `3315381`, `3315382` are pending with missing stdout. | Jobs must start, reach steps, and produce parsed metrics. |
| Lyris SWE-RL runtime blocker is cleared | Partial | r28 baseline `2126895` cleared the TransformerEngine source-build blocker, then failed before step metrics on a Ray/Python mismatch. r29 `2128989` fixed the Ray/Python mismatch but stalled in the user persistent-cache TE build path under inode quota pressure. r30 `2129203` is running with node-local `/tmp` cache and active TE build processes. | r30 baseline must reach a parsed step and then release dependent PARD/PARD-2 jobs. |

## Latest Active Gate Snapshot

As of `2026-06-15 08:29 PDT`, the r28/r29 Lyris baselines are superseded by the
r30 tmpcache raymatch chain. The Lyris baseline is running, and the OCI-HSG gates are
still pending:

| Job | Scope | State | Start | Priority | Runtime log |
| ---: | --- | --- | --- | ---: | --- |
| `2129203` | Lyris SWE-RL baseline step-1 retry | `RUNNING/None` | `2026-06-15T07:57:46` | `77298` | present; TE build active and advanced to fresh `ninja`/`c++`/`cc1plus` work; no parsed step metrics yet |
| `2129271` | Lyris SWE-RL PARD step-1 retry | `PENDING/Dependency` | `N/A` | `77284` | missing; batch script verified without `bash -x` |
| `2129272` | Lyris SWE-RL PARD-2 step-1 retry | `PENDING/Dependency` | `N/A` | `77284` | missing; batch script verified without `bash -x` |
| `3308774` | OCI-HSG SWE-RL PARD-2 step-1 | `PENDING/Priority` | volatile; see active snapshot | `133709` | missing |
| `3315380` | OCI-HSG MathRL baseline 10-step | `PENDING/Priority` | volatile; see active snapshot | `133657` | missing |
| `3315381` | OCI-HSG MathRL PARD K3 10-step | `PENDING/Priority` | volatile; see active snapshot | `133657` | missing |
| `3315382` | OCI-HSG MathRL PARD K5 10-step | `PENDING/Priority` | volatile; see active snapshot | `133657` | missing |

## Next Evidence Gates

1. Watch `3308774` for first stdout and parsed step evidence; this is the
   current 235B SWE-RL PARD-2 proof under `nemotron_n3_post`.
2. Watch `3315380`/`3315381`/`3315382` for MathRL baseline/PARD step evidence.
3. Watch Lyris r30 baseline `2129203`, then dependent PARD/PARD-2
   `2129271`/`2129272`.
4. Once any proof gate emits runtime logs, parse step metrics with
   `scripts/extract_nemorl_fullgrpo_step_metrics.py` before making performance
   claims.

Supporting artifacts:

- `docs/nemorl_235b_active_gates_latest_20260615.md`
- `docs/nemorl_235b_gate_runtime_report_latest_20260615.md`
- `docs/nemorl_10_step_repro_runbook_20260615.md`
- `docs/qwen8_pard2_official_online_impact_20260613.md`
- `docs/qwen30ba3b_pard2_online_long_output_win2048_comparison_20260611.md`
- `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_20260615.md`
- `scripts/fetch_and_parse_nemorl_235b_ready_gate_metrics.py`
