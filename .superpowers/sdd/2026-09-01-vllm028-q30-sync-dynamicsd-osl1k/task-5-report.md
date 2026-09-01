# Task 5 report: safe Q30 Lyris renderer

## Outcome

Implemented a side-effect-free, typed renderer for the Q30 vLLM 0.28 Lyris
experiment. It renders six independent one-GPU canaries, all 108 independent
calibration cells, and the five-arm 4n4g/16-worker barrier, with the DSpark
adaptive arm available only as a sixth explicit opt-in barrier job.

Implementation commit: `c43b88ae3bbe662a63956babc54c46495e627ed8`.

## Files

- `experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml`
- `experiments/vllm_028_q30_sync_dynamicsd/submit.py`
- `experiments/vllm_028_q30_sync_dynamicsd/live_runner.py`
- `tests/test_vllm028_q30_sync_dynamicsd.py`

No v0.25.1 or v0.27.1 experiment path was modified.

## Runtime and safety contract

- Pins account `coreai_dlalgo_llm`, partition `gb200`, the authenticated v0.28
  image path, base commit, patchset digest, and image artifact digest.
- Uses `/home` for source, `/raid/scratch` for caches and node-local adaptive
  checkpoint overlays, and `/lustre` for prompts and durable results.
- Hashes the target config, drafter config, and complete real-prompt JSONL at
  runtime and writes an atomic structured job-provenance receipt.
- Seals exactly the first 64 usable real prompts. Task-3 request construction
  gives each of the 16 barrier workers four consecutive prompts with 32 seeded
  generations each.
- Uses one explicit `SamplingParams(n=1, ...)` object per request, natural EOS,
  OSL 1024, temperature/top-p 1.0, TP1/DP1, `FULL_AND_PIECEWISE`, and
  `flashinfer_trtllm` MoE.
- Renders method-aware DFlash, DSpark, DynamicSD, and adaptive configs. The
  adaptive arm copies DSpark once per allocated node and changes only the two
  required top-level confidence-head keys in the node-local copy.
- Refuses to render over a non-empty output directory, refuses result-run and
  worker-result overwrite, and exposes scheduler calls only through explicit
  `test-only` or `submit` dispatch modes with an injectable subprocess runner.
  Rendering itself has no scheduler or SSH side effects.

## Evidence gate

Stock vLLM 0.28 offline output exposes per-request `last_token_ts` and aggregate
speculative counters, but not the exact DynamicSD selected-K histogram or the
full-span physical drafter widths required by Task 3. Task 5 therefore does not
invent trace evidence or weaken Task-3 validation.

Every live arm runs model generation first and atomically preserves
`unvalidated_raw.json`, including request outputs, generation-only elapsed time,
finish times, and before/after exposed counters. It then writes an explicit
`unsupported-receipt.json` with `promotion_allowed=false` and exits 2. K0 and
DynamicSD can therefore retain useful raw performance without making K0
absence/execution claims. No profiler overhead is included in raw generation
throughput. Promotion remains blocked until a later full-span,
content-addressed trace integration closes the evidence gap.

## TDD evidence

Initial RED:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py \
  -k 'renderer or submit_modes or live_adapter or real_prompt or adaptive_overlay'

ModuleNotFoundError: No module named
'experiments.vllm_028_q30_sync_dynamicsd.live_runner'
```

Final GREEN and compatibility checks:

- Focused renderer/adapter selection: 9 passed, 91 deselected.
- Full Q30 suite: 100 passed.
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
commit, run remote path and `sbatch --test-only` preflight, and submit only the
approved canaries. Raw or unsupported receipts must not be promoted, calibrated,
or reported as validated results. A truthful profiler/postprocessor integration
is still required before K0 or DynamicSD evidence can pass Task-3 promotion.
