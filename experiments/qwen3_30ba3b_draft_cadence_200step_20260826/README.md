# Qwen3-30B-A3B fixed-interval drafter study

This experiment compares DFlash and DSpark online drafter updates every 5, 10,
or 20 Math GRPO policy steps. All six arms inherit the official
`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` recipe.

## Matrix

| Drafter | Update interval | Variant |
|---|---:|---|
| DFlash K5 | 5 steps | `dflash-fixed5` |
| DFlash K5 | 10 steps | `dflash-fixed10` |
| DFlash K5 | 20 steps | `dflash-fixed20` |
| DSpark K5 | 5 steps | `dspark-fixed5` |
| DSpark K5 | 10 steps | `dspark-fixed10` |
| DSpark K5 | 20 steps | `dspark-fixed20` |

Two matched Fixed-10 CUDA Graph arms retain the same workload, drafter, and
online-update cadence while extending the capture ladder from the vLLM default
512-token ceiling through 2048 tokens:

| Drafter | Default reference | Extended-capture variant |
|---|---|---|
| DFlash K5 | `dflash-fixed10` | `dflash-fixed10-cg2048` |
| DSpark K5 | `dspark-fixed10` | `dspark-fixed10-cg2048` |

The explicit ladder preserves every default bucket through 512, adds
576/640/704/768 to cover the high-concurrency K5 verification range, and then
adds 832/896/960/1024/1280/1536/1792/2048 for mixed prefill-decode coverage.
DFlash also includes 2046 because its MRV1 FULL-decode path rounds capture
sizes to multiples of `K + 1 = 6`; without that entry, a nominal 2048 ceiling
would round above the limit and silently fall back to 1794. The 768 bucket is
the important one for the current 128-request generation worker
(`128 * (K + 1) = 768`). Buckets above 768 are deliberately retained as an
experimental request, but they are not expected to improve pure K5 decode at
the current concurrency and may increase graph memory and startup time.

Here, fixed interval N means that the drafter is trained and refit every N
policy steps. It is different from the separate `fixed` arms in the
fixed-versus-always study, where the drafter remains frozen and receives no
online update.

## Preserved performance recipe

The inherited workload remains 4 nodes × 4 GPUs, 64 prompts × 32 generations,
global batch 2048, a 4096-token policy/generation limit, TP1/EP16/PP1/CP1,
sequence packing with fused loss, validation every 10 steps, shuffled
OpenMathInstruct-2 with a 5% validation split, disabled checkpoint writes, and
the Triton MoE backend. The overlays add only:

- 200 total steps;
- local target and tokenizer paths;
- W&B cadence metrics without durable checkpoint evidence;
- one K5 DFlash or DSpark drafter, optimizer, and fixed update schedule;
- `policy.offload_optimizer_for_refit=false`, which prevents the optimizer CPU
  copy from overlapping vLLM's sleep-weight backup on the GB200 nodes.

The jobs use OCI-HSG `batch_long` with an 18-hour limit. This keeps
checkpointing disabled so save time is not mixed into the performance samples.
The CUDA Graph and first two step gates retain 45-minute diagnostic deadlines;
the first scheduled refit gate waits while the training process is alive and
therefore cannot misclassify a slow fixed-20 run as a refit hang.

The launcher does not override `data_plane.enabled`, vLLM `max_num_seqs`, the
compilation backend, CUDA Graph mode, or capture sizes. Runtime defaults decide
the CUDA Graph coverage, matching the official performance flow.

DSpark jobs additionally copy the container's installed vLLM package to
node-local scratch and apply the ten runtime-file changes from vLLM #48167,
followed by the two stale-causality replacements from commit `bf372f9bb5`.
Code tracing showed that its attention guard alone is insufficient on v0.25.1:
the stock model runner creates the CUDA Graph manager before draft attention is
initialized and the stock DSpark speculator does not consult the corrected
support classification. The PR's final patch also removed `dflash_causal`
initialization while leaving two CUDA Graph call sites that still read it. The
follow-up uses the group-level causality initialized by `set_attn`. The helper
pins both patch digests, requires a clean forward or reverse application,
writes per-file digests, and fails closed on source drift. DFlash remains on
the unmodified package.

## Immutable inputs

- Product source: `/home/sna/nemorl-q30-cadence-product-20260826`
- Product SHA: `d5c8bfa987025949699f7cfff188b349480bb8b5`
- Target: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B`
- DFlash: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dflash/exported-checkpoint-25391`
- DSpark: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dspark/exported-checkpoint-25391`
- W&B project: `sna-specdec`
- W&B group: `q30ba3b-draft-cadence-200step-20260826`

## Validation and submission

Each variant must pass the exact state-dict check, composed-config verifier,
W&B authentication, scheduler dry-run, CUDA Graph gate, Step 1 gate, Step 2
gate, and its first requested drafter-refit gate. `dspark-fixed5` is the
correctness canary for the vLLM runtime patch; `dflash-fixed20` independently
validates the corrected long-run gate on the unchanged DFlash runtime.

```bash
uv run --no-project --with pytest --with pydantic python -m pytest -q \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests
uv run --no-project --with ruff ruff check \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826
bash -n \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh
```
