# Lyris PARD-2 Native Format Status - 2026-06-12

## Current Status

- Important caveat from the 14:11 CEST pre-submit review: "native PARD-2" in this file currently means `method=pard2` is accepted as a vLLM draft-model alias. It does not yet mean official PARD-2 target-feature conditioning. The local vLLM path does not pass target hidden states into the drafter, does not read `pard2_target_layers`, and does not load/inject `target_proj` / `warp_model.bin`.
- The stock Lyris vLLM container rejects `method=pard2` without a patch.
- The local `sitecustomize.py` hook now enables `method=pard2` as a PARD-2 draft-model alias at import time and keeps `pard2` on the draft-model proposer return-shape path.
- A reusable source patch now exists for stock vLLM builds: `experiments/eagle3_qwen3_235b/patches/vllm_pard2_method_alias.patch`.
- The native vLLM source-build launcher now applies the same proposer return-shape fix, so `method=pard2` no longer falls into the tuple-return EAGLE/MTP path.
- Extended probe `2101651` completed successfully in the Lyris v0.20.2 container.
- Qwen3-235B native PARD-2 smoke retry `2101655` completed and wrote `breakdown.json`.
- Follow-up native PARD-2 jobs:
  - OSL1K K3, n=16, BS1/2: `2101759` completed
  - OSL32K K1 pilot, n=2, BS1: `2101818` completed
  - OSL32K K2 pilot, n=2, BS1: `2101819` completed
  - OSL32K K3 pilot, n=2, BS1: `2101761` completed
- Latest rows are already in `docs/lyris_specdec_expected_performance_20260612.html`; `2101759` has written completed BS1 and BS2 rows, Eagle-3 K1/K3 OSL32K have written final rows, and native PARD-2 OSL32K K1/K2/K3 all have final rows.

Supported PARD-2 native vLLM config form:

```json
{
  "method": "pard2",
  "model": "amd/PARD2-Qwen3-8B",
  "num_speculative_tokens": 3,
  "draft_tensor_parallel_size": 4,
  "parallel_drafting": true
}
```

This requires either the runtime alias hook (`VLLM_PARD2_METHOD_ALIAS=1`) or a source build with `ENABLE_PARD2_METHOD_ALIAS=true`. The SWE-Bench Lyris launcher enables the alias automatically when `PARD2_CONFIG_METHOD=pard2`; without that patch/hook, stock vLLM still rejects `method=pard2`.

## Evidence

Probe `2101651` log:

```text
PARD2_METHOD_ALIAS_PROBE=PASS
checks {'pard2_literal': True, 'uses_draft_model_runtime': True, 'post_init_accepts_pard2_branch': True, 'proposer_treats_pard2_as_tensor_return': True, 'proposer_return_shape_branch_contains_pard2': True}
```

First smoke `2101640` passed config validation but failed in the draft proposer:

```text
ValueError: too many values to unpack (expected 2)
```

That showed `method=pard2` was being classified like tuple-return EAGLE/MTP paths in `SpecDecodeBaseProposer.model_returns_tuple()`.

Smoke retry `2101655` speculative config:

```json
{"method": "pard2", "model": "amd/PARD2-Qwen3-8B", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 4, "parallel_drafting": true}
```

Smoke retry `2101655` completed:

```text
bs=1 batches=1 latency=19.395s out/gpu=3.30 coverage=0.0% acceptance=34.13% drafted=378 accepted=129
wrote /lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/vllm-runs/qwen235b_a22b_swev_osl256_nativepard2_smoke_retry1_lyris_tp4_fp8kv_pard2_offset0_n1_isl4096_osl256_bs1_k3_20260611/breakdown.json
```

Completed rows from `2101759`:

```text
Qwen3-235B / SWE-Bench Verified / ISL4096 OSL1024 / native PARD-2 K3
BS1: output_tok_s_per_gpu=3.6812452473962347, speedup=1.8062559718806677, acceptance=38.85510312004231, mean_acceptance_length=2.1656530936012692
BS2: output_tok_s_per_gpu=6.923049603394577, speedup=1.6985194373642405, acceptance=38.949922788440325, mean_acceptance_length=2.16849768365321
```

Latest completed rows from the refreshed report:

```text
2101759 OSL1K native PARD-2 K3: completed, BS1 speedup 1.806x, BS2 speedup 1.699x
2101581 OSL32K Eagle-3 K1: completed, output_tok_s_per_gpu=3.753865585927842, acceptance=86.92775036367266%, mean_acceptance_length=1.8692775036367266
2101582 OSL32K Eagle-3 K3: completed, output_tok_s_per_gpu=5.188925339723579, acceptance=54.27796048234005%, mean_acceptance_length=2.6283388144702013
2101818 OSL32K native PARD-2 K1: completed, output_tok_s_per_gpu=1.876236078645655, speedup=0.8994612558660255, acceptance=11.531850439932606%, mean_acceptance_length=1.1153185043993261
2101819 OSL32K native PARD-2 K2: completed, output_tok_s_per_gpu=1.8957229137016633, speedup=0.9088031789490651, acceptance=6.650991511211771%, mean_acceptance_length=1.1330198302242354
2101761 OSL32K native PARD-2 K3: completed, output_tok_s_per_gpu=1.9126647339586869, speedup=0.9169250305105374, acceptance=4.641460234680574%, mean_acceptance_length=1.1392438070404172
2102005/2102006/2102008/2102009 OSL32K suffix K1/K2/K4/K8: completed; final speedups 1.812x/2.460x/3.360x/5.586x and final acceptances 82.10%/77.92%/75.43%/86.35%
```

OSL32K read as of this refresh: native PARD-2 K1 has the highest native PARD-2 acceptance at `11.53%`, but throughput is still below baseline at `0.899x`. K2 is `0.909x` with `6.65%` acceptance, and K3 is the fastest native PARD-2 point at `0.917x` with `4.64%` acceptance. This means lowering K helps acceptance but does not fix throughput for Qwen3-235B OSL32K. In the same final OSL32K table, suffix K32 reaches `6.042x`, suffix K8 reaches `5.586x`, Eagle-3 K3 reaches `2.488x`, and PARD K5 reaches `1.529x`.

Last checked at `2026-06-12 08:12 CEST`:

```text
2101759 OSL1K native PARD-2 K3: COMPLETED on lyris0214, elapsed 00:33:36, BS1/BS2 breakdown rows present.
2101581 OSL32K Eagle-3 K1: breakdown row present.
2101582 OSL32K Eagle-3 K3: breakdown row present.
2101818 OSL32K native PARD-2 K1: COMPLETED, elapsed 03:40:10, breakdown row present.
2101819 OSL32K native PARD-2 K2: COMPLETED, elapsed 03:37:39, breakdown row present.
2101761 OSL32K native PARD-2 K3: COMPLETED, breakdown row present.
```

Refresh at `2026-06-12 08:12 CEST` reached Lyris and regenerated the expected-performance HTML/PNG/MD/CSV. The raw expected-performance table now has `109` rows, `docs/lyris_qwen235b_suffix_metrics_20260612.csv` has `42` final metric rows, and final `breakdown.json` rows remain authoritative over live-log snapshots.

## Implementation

- Runtime hook: `experiments/eagle3_qwen3_235b/specdec_breakdown_instrumentation/sitecustomize.py`
- Source patch: `experiments/eagle3_qwen3_235b/patches/vllm_pard2_method_alias.patch`
- Native source-build wiring: `experiments/eagle3_qwen3_235b/slurm_build_vllm_native_site.sbatch`
- Probe launcher: `experiments/eagle3_qwen3_235b/submit_lyris_pard2_alias_probe.sh`
- SWE-Bench launcher wiring: `experiments/eagle3_qwen3_235b/submit_lyris_swebench32k_standalone_specdec.sh`

## Next Gate

Do not expand native PARD-2 K on Qwen3-235B OSL32K just to chase speedup: K1/K2/K3 are all below baseline. The useful follow-up is to debug why the PARD-2 checkpoint/domain has such low acceptance on Qwen3-235B SWE-Bench, while keeping suffix and Eagle-3 as the stronger current baselines for large-model SWE-Bench.
