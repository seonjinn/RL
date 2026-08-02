# NeMo-RL vLLM 0.25.1 Safe Adaptive Canary Design

## Objective

Verify that NeMo-RL main can load the corrected vLLM 0.25.1 MXFP8 Safe
Adaptive implementation without adding tuning work to the rollout request path.
Compare it with the CuTeDSL baseline under the same model, prompts, CUDA Graph
configuration, topology, and allocation.

## Experiment boundary

The first gate is generation-only. It uses NeMo-RL's `VllmGeneration` actor
path, not a standalone `vllm serve` process. Both arms use the same custom vLLM
source commit so the only functional difference is the selected dense MXFP8
backend and Safe Adaptive configuration.

The second gate runs only after the generation canary passes. It adds one refit
and measures the time spent rebuilding backend-specific prepared weights. This
keeps model-load overhead, steady rollout performance, and refit overhead as
three separate quantities.

## Arms

- `baseline`: `linear_backend=flashinfer_cutedsl`; no tactic table is loaded.
- `adaptive`: `linear_backend=flashinfer_trtllm`; corrected prepared weights,
  adaptive 8x4/128x4 layout, exact tactic table, and layer allowlist are enabled.

An exact tactic miss is not an error. It uses the TRTLLM runner default tactic.
A layer allowlist miss uses the CuTeDSL fallback implemented by the custom vLLM
branch.

## Runtime contract

The tactic table is generated offline and is immutable during a run. Each worker
loads and validates it once before CUDA Graph capture. Runtime dispatch performs
an exact dictionary lookup by the physical execution signature. No shmoo runs
inside generation.

The canary records:

- NeMo-RL and custom vLLM commits;
- vLLM, FlashInfer, CUDA, GPU, model, topology, and container identity;
- tactic table path and SHA256;
- model initialization wall time;
- CUDA Graph enabled state;
- end-to-end evaluation wall time and emitted token count;
- exact arm environment after secret filtering.

## Promotion gates

1. Every rank imports vLLM from the requested source tree.
2. The adaptive table SHA256 matches the submitted value.
3. CUDA Graph is enabled in both arms.
4. Both arms complete with valid output tokens.
5. The adaptive arm has no correctness or engine errors.
6. Refit testing is not promoted until the generation-only gate passes.

## NeMo-RL integration model

No NeMo-RL production source patch is required for the canary. The existing
`generation.vllm_cfg.env_vars` path forwards the adaptive environment into the
outer generation actor. `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` forwards the custom
source path to internal TP workers. A production deployment should use a custom
wheel or immutable container rather than a mutable source overlay.

