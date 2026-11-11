# SpecDec on latest main (2026-11-11)

This directory holds the latest-main revalidation of the drafter PR workspace. Open `index.html` for the stage tables. `data/` holds the per-step W&B history for each run and `metadata.json` lists commits and job IDs.

| Workload | DFlash | DSpark | Basis |
|---|---|---|---|
| Qwen3-30B-A3B GRPO | 1.19x E2E, 1.72x gen (K7) | 1.17x E2E, 1.64x gen (K8) | step time |
| Nemotron 3.5 Nano GRPO | 1.05x E2E, 1.14x gen (K7) | 1.07x E2E, 1.14x gen (K8) | step time |
| Qwen3-30B-A3B SWE RL (V2) | 2.03x model call, 1.23x E2E (K7) | 2.01x model call, 1.25x E2E (K8) | model-call tok/s |
| Qwen3.5-35B-A3B GRPO | 0.89x E2E, 0.61x gen (K7) | 0.94x E2E, 0.84x gen (K8) | step time |

Qwen3.5-35B-A3B is slower at block-size K because vLLM holds 1+K Gated DeltaNet state blocks per request. Small-K arms and Qwen3-235B are still running. Each arm has one run, so the numbers have no confidence intervals.