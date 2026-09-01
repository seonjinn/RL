# Task 5 report: safe Q30 Lyris renderer

## Outcome

Implemented a side-effect-free, typed renderer for the Q30 vLLM 0.28 Lyris
experiment. It renders six independent one-GPU canaries, all 108 independent
calibration cells, and the five-arm 4n4g/16-worker barrier, with the DSpark
adaptive arm available only as a sixth explicit opt-in barrier job.

Implementation commits:

- `c43b88ae3bbe662a63956babc54c46495e627ed8`
- Review hardening: `eee28d84558d9c3b084b87e96aaad7ba74a13deb`

## Files

- `experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml`
- `experiments/vllm_028_q30_sync_dynamicsd/submit.py`
- `experiments/vllm_028_q30_sync_dynamicsd/live_runner.py`
- `tests/test_vllm028_q30_sync_dynamicsd.py`

No v0.25.1 or v0.27.1 experiment path was modified.

## Runtime and safety contract

- Pins account `coreai_dlalgo_llm`, partition `gb200`, the authenticated v0.28
  image path, base commit, patchset digest, and image artifact digest.
- Requires exact HEAD and an empty
  `git status --porcelain --untracked-files=all` before model loading.
- Uses a Task-6 preflight receipt containing the exact container SHA256, size,
  and nanosecond mtime. Jobs verify the small metadata/receipt plus `stat`; they
  never rehash the full sqsh.
- Uses `/home` for source, `/raid/scratch` for caches and node-local adaptive
  checkpoint overlays, and `/lustre` for prompts and durable results.
- Hashes the target config, drafter config, and complete real-prompt JSONL at
  runtime and writes an atomic structured job-provenance receipt.
- Seals exactly the first 64 usable real prompts. Task-3 request construction
  gives each of the 16 barrier workers four consecutive prompts with 32 seeded
  generations each.
- Uses one explicit `SamplingParams(n=1, ...)` object per request, natural EOS,
  OSL 1024, temperature/top-p 1.0, TP1/DP1, `FULL_AND_PIECEWISE`, and
  `flashinfer_trtllm` MoE. It also pins BF16, GPU memory utilization 0.9,
  max batched tokens 32768, prefix caching off, chunked prefill on, and max
  model length 4096.
- Renders method-aware DFlash, DSpark, DynamicSD, and adaptive configs. The
  adaptive arm copies DSpark once per allocated node and changes only the two
  required top-level confidence-head keys in the node-local copy. Each worker
  hashes and records the completed overlay config it actually loads, together
  with its path and source-config hash.
- K0 is a distinct controller: DFlash configures physical K7 and DSpark K8,
  while both select verifier K0 through
  `num_speculative_tokens_per_batch_size=[[1,128,0]]`.
- Calibration receipts record their actual batch-size request count; barrier
  and barrier-derived canaries record 128 requests per worker.
- Refuses to render over a non-empty output directory, refuses result-run and
  worker-result overwrite, and exposes scheduler calls only through explicit
  `test-only` or `submit` dispatch modes with an injectable subprocess runner.
  Accepted submission IDs are appended and `fsync`ed to JSONL before callback
  delivery. A later failure raises `SubmissionDispatchError` carrying every
  prior ID and the durable receipt path. Rendering itself has no scheduler or
  SSH side effects.
- Job keys and result subdirectories reject traversal/metacharacters. All
  rendered path scalars are visibly POSIX-quoted.

## Evidence gate

Stock vLLM 0.28 offline output exposes per-request `last_token_ts` and aggregate
speculative counters, but not the exact DynamicSD selected-K histogram or the
full-span physical drafter widths required by Task 3. Task 5 therefore does not
invent trace evidence or weaken Task-3 validation.

Every live arm runs model generation first and atomically preserves request
outputs, generation-only elapsed time, finish times, and before/after exposed
counters. Baseline and positive fixed-K write `complete_raw.json` and exit zero.
Baseline is labeled `baseline_no_speculation`; fixed K records either validated
aggregate-only evidence or an explicit counter-unavailable reason. Neither
claims physical-trace or CUDA-graph validation, and both retain
`promotion_allowed=false`.

K0, DynamicSD, and DSpark adaptive write `unvalidated_raw.json`, then an
explicit `unsupported-receipt.json`, and exit 2 because exact selected-K and
physical-width evidence is unavailable. They retain useful raw performance
without making K0 absence/execution claims. No profiler overhead is included in
raw generation throughput. Strict Task-3 promotion remains blocked until a
full-span, content-addressed trace integration closes the evidence gap.

## TDD evidence

Initial RED:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py \
  -k 'renderer or submit_modes or live_adapter or real_prompt or adaptive_overlay'

ModuleNotFoundError: No module named
'experiments.vllm_028_q30_sync_dynamicsd.live_runner'
```

Final GREEN and compatibility checks:

- Review RED: 12 expected failures covering all eight findings.
- Review GREEN: 12 passed, 100 deselected.
- Full Q30 suite: 112 passed.
- v0.28 foundation suite: 88 passed.
- Ruff: all checks passed.
- Pyright: 0 errors, 0 warnings.
- Rendered 120 scripts total: 6 canary + 108 calibration + 6 barrier with the
  adaptive opt-in arm. `bash -n` passed on every rendered script.
- `git diff --check`: clean.

The Pytest runs emitted only the pre-existing macOS temporary-directory cleanup
warnings recorded by earlier tasks; all commands exited zero.

## Task 6 handoff

Task 6 must render again using the post-commit source SHA, push/pull that exact
commit, leave the `/home` checkout completely clean, produce the one-time
container verification receipt, run remote path and `sbatch --test-only`
preflight, and submit only the approved canaries. Submission must use the durable
JSONL receipt/callback path. `complete_raw` means successful execution, not
strict Task-3 promotion. Raw or unsupported receipts must not be reported as
validated results. A truthful profiler/postprocessor integration is still
required before K0, DynamicSD, or adaptive evidence can pass Task-3 promotion.
