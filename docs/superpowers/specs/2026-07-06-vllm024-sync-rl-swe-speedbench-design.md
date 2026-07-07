# vLLM 0.24 Sync-RL SWE and SPEED-Bench Design

## Objective

Extend the vLLM 0.24 speculative-decoding benchmark to model synchronous RL
rollouts on SWE prompts, including 32K and 64K generation tails, and add a
pinned SPEED-Bench track. Cover Qwen3-30B-A3B, Qwen3-32B,
Qwen3-235B-A22B, Nemotron 3 Super, and Nemotron 3 Ultra without reporting
cross-runtime or checkpoint-incompatible comparisons as equivalent results.

## Result Cohorts

### Forced-Length Barrier Replay

This cohort measures work-equivalent rollout makespan. Requests use a
versioned length plan, `ignore_eos=true`, and `min_tokens=max_tokens` so every
matched method produces exactly the same planned token count. The primary
length distribution is `8:4:3:1` across short, medium, long, and tail requests:

| Profile | Short | Medium | Long | Tail |
|---|---:|---:|---:|---:|
| 32K | 4K | 8K | 16K | 32K |
| 64K | 4K | 8K | 16K | 64K |

Each rollout batch is a synchronization barrier. The next batch starts only
after every request in the current batch finishes. Primary runs use 16 prompts,
4 samples per prompt, and 3 barriers. A 16 by 16 by 3 confirmation is required
for the 32K profile after the smaller matrix establishes runtime and capacity.
The full 64K confirmation is opt-in after wall-time extrapolation.

### Natural-EOS Accuracy Sentinel

This cohort uses the same prompts, caps, and seeds but allows normal EOS and
saves complete response JSONL files. It reports patch extraction, output-length
distribution, finish reasons, reward when available, and SWE-bench pass rates.
Forced-length results must never be labeled as accuracy results.

## Model and Method Compatibility

| Model | Primary topology | Supported first methods | Deferred or unsupported |
|---|---|---|---|
| Qwen3-30B-A3B | TP1, one GB200 node | baseline, Eagle-3 static, DynamicSD, PARD | PARD-2 needs a target-dimension checkpoint; DFlash and DFlare checkpoints are Qwen3-8B-specific |
| Qwen3-32B | TP2, one node | baseline, Eagle-3 static, DynamicSD, PARD | PARD-2 needs a target-dimension checkpoint; DFlash and DFlare are incompatible |
| Qwen3-235B-A22B | TP8, two nodes, Ray | baseline, Eagle-3 static, single-range DynamicSD, PARD diagnostic | multi-range DynamicSD currently hangs; PARD-2 has no target-trained checkpoint; DFlash and DFlare are incompatible |
| Nemotron 3 Super BF16 | TP2, one node | baseline, native MTP, DynamicMTP after static validation | Qwen external drafters are incompatible |
| Nemotron 3 Ultra BF16 | TP8, two nodes, Ray | baseline, native MTP, DynamicMTP after static validation | Qwen external drafters are incompatible |

Every checkpoint is pinned by repository revision. PARD-2 runs use their
patched vLLM overlay and a baseline from that exact overlay. DFlare runs use an
AngelSlim baseline from the same runtime. Neither may use the stock vLLM
baseline as the denominator.

## Context Handling

The pinned Qwen configurations have a native 40,960-token context. A 4K prompt
plus 32K output fits natively. A 4K prompt plus 64K output requires matched YaRN
views for both target and Eagle drafter. Context extension metadata and hashes
are part of the comparison key.

Nemotron 3 Super and Ultra support both profiles natively. The first 64K runs
are BS1 capacity canaries with FP8 KV cache. Concurrency increases only after
the canary completes without OOM or context overflow.

## Dynamic Schedule Calibration

DynamicSD and DynamicMTP schedules are calibrated against active engine
concurrency, not queued request count. Each schedule must exercise at least two
K tiers during the measured run. For each model, context profile, and method:

1. Sweep fixed K over representative active concurrency values.
2. Use three measured repeats.
3. Select the smallest K within 2 percent of the best median throughput.
4. Fit a monotone non-increasing K as concurrency rises.
5. Recalibrate for 32K and 64K instead of copying the 4K schedule.

DynamicSD is treated as validated only for Eagle and Eagle-3 until upstream or
local evidence establishes another method. Nemotron native MTP scheduling is
reported as DynamicMTP, not Eagle DynamicSD.

## SWE Workload

The performance track uses pinned SWE-bench Verified issue prompts without the
gold patch. Prompt truncation is rejected by default. Each result records
instance ID, prompt hash, rendered token length, output cap, seed, and request
plan hash.

The performance replay measures generation only. The accuracy sentinel uses
the official SWE-bench harness on saved natural-EOS responses. Reports clearly
separate throughput evidence from repository-level correctness.

## SPEED-Bench Tracks

The official track pins:

- SPEED-Bench dataset revision `487aa718444e816458d1a0a52bfce7a454285cf4`.
- Model Optimizer measurement framework revision
  `43fee0cd70fa9e5f85782d52a4bd8ad9c8b88446`.

Official qualitative evaluation keeps multi-turn structure and runs through the
upstream asynchronous runner. Official throughput evaluation preserves the
provided 1K, 2K, 8K, 16K, and 32K ISL buckets and reports nominal ISL plus the
actual target-tokenizer ISL. It does not claim to be a 64K-output benchmark.

A separate Sync-RL overlay selects 48 throughput rows per ISL, balanced across
low, mixed, and high entropy, expands each prompt with sampled completions, and
uses an explicit barrier. The overlay may use 32K or 64K output plans but is
labeled non-official.

## Metrics and Validity Gates

Matched reports include:

- Rollout makespan and reduction versus baseline.
- Output tokens per second per GPU and speedup.
- Requests per second.
- Acceptance rate and mean accepted length.
- Output-token ratio, cap-hit rate, and finish-reason counts.
- Output-length p50, p90, p99, and maximum.
- Barrier tail as maximum completion time minus median completion time.
- Acceptance by output windows: 0-4K, 4-8K, 8-16K, 16-32K, and 32-64K.

A forced-length result is valid only when request count, planned output tokens,
prompt hashes, request-plan hash, runtime image, graph mode, topology, dataset
revision, and all three barriers match. It must also have no prompt truncation,
context overflow, fallback, OOM, or missing speculative counters.

## Artifacts and Reporting

Each run writes `result.json`, `resolved_request_plan.json`, `benchmark.log`,
`submit.sbatch`, and `jobs.tsv`. Natural-EOS runs additionally write compressed
response JSONL files. Summaries are emitted as CSV and JSON.

The HTML report groups results by model, workload, context profile, runtime,
and method family. Unsupported cells explain the checkpoint or runtime reason
instead of appearing as missing data. Official SPEED-Bench, Sync-RL overlay,
and full NeMo-RL results remain distinct sections.

## Staged Execution

1. Implement and test the request-plan core and provenance gates.
2. Run Qwen3-32B 4 by 4 by 1 canaries for both context profiles.
3. Run the primary Qwen matrix at 16 by 4 by 3.
4. Run the full 32K Qwen confirmation at 16 by 16 by 3.
5. Validate Nemotron static MTP, then calibrate DynamicMTP.
6. Stage and run official SPEED-Bench before the Sync-RL overlay.
7. Integrate PARD starting with Qwen3-30B-A3B TP1.
8. Add PARD-2, DFlash, or DFlare rows only when exact compatible checkpoints
   and matched runtime baselines exist.
9. Run natural-EOS SWE accuracy sentinels on the best configurations.
