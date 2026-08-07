# MXFP8 MoE Workload-Replayed Tactic Audit

This experiment audits workload-replayed FlashInfer TRTLLM MXFP8 MoE FC1/FC2
tactic pairs for Qwen3-30B-A3B. It is an opt-in, reproducible audit; it does
not change the production request path until every gate passes.

## Provenance Contract

- Runtime: vLLM 0.25.1 at commit
  `a76062edee3a3ac23d47a93c7ce466f06a19111f`, branch
  `sna/mxfp8-moe-tactic-audit-v0251`.
- FlashInfer: `FlashInfer 0.6.13`.
- Target: `Ptyche GB200`, using the current four-node NeMo-RL MXFP8
  performance recipe.
- Keep `moe_backend=flashinfer_trtllm`, the dense linear backend, model
  revision, quantization scope, topology, generation settings, container, and
  node count identical between baseline and candidate arms.

## Trace Privacy

Trace artifacts contain execution metadata only. Never write prompts, token IDs, hidden values, or model outputs, credentials, or Hugging Face/W&B tokens to artifacts. Preserve every observed routing signature in the raw trace, but select representative signatures covering at least 95% of observed MoE GPU time.

## Replay and Qualification

Profile every legal FC1/FC2 tactic pair with three warmups, at least ten timed
repetitions, CUDA Graph replay, and cold-L2 inputs. Keep CUDA Graphs enabled for
shmoo replay, vLLM validation, and NeMo-RL performance measurements; use eager
mode only for the dedicated routing-trace collection run.

Promote a tactic only when all candidate thresholds hold: weighted-median
improvement is at least 2%, coefficient of variation is at most 3%, and no
high-weight profile regresses by more than 1%. Missing cache entries and
metadata mismatches fall back to stock FlashInfer behavior; cache misses are
not errors.

## Validation Gates

Promotion requires passing micro-correctness, CUDA Graph replay, deterministic
vLLM generation, the matched 1,319-example GSM8K gate, and NeMo-RL finite-metric
gates. The GSM8K comparison uses the immutable matched `1,319-example GSM8K`
evaluation set.

## Entry Points

```text
submit_trace_ptyche.sh -> select_profiles.py -> shmoo_moe_tactics.py
-> qualify_cache.py -> submit_validation_ptyche.sh -> build_report.py
```

Every later launcher must consume this contract and record the exact NeMo-RL,
FlashInfer, CUDA, driver, container, model-revision, topology, and source
fingerprints in its output metadata.
