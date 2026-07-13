# Qwen3-235B Adaptive-K Runtime Gate Plan

Date: 2026-06-08

## Why This Exists

Static PARD K is not stable across regimes:

| Regime | Best current static signal |
| --- | --- |
| OpenMath standalone batch 32 | K9 is slightly fastest, but only `29.98%` acceptance |
| OpenMath standalone batch 64/128 | K3 is fastest so far: K3 `1.308x/1.244x`, K5 `1.259x/1.170x`, K7 `1.196x/1.114x`, K8 `1.156x/1.041x`, K9 `1.186x/1.039x` |
| NeMo-RL Qwen3-235B GBS512 fixed256 | K5 is the static winner: K5 total `1.810x`, generation `2.285x`, E2E `1.815x`; K3 completed but is slower at total `1.597x`, generation `1.934x`, E2E `1.599x` |
| NeMo-RL fixed256 Step2-5 | K9 is slightly faster, K5 has much better acceptance: `42.19%` vs `28.60%` |
| Long-output GRPO-style tails | Static high K is risky because later draft positions have weak acceptance |

OpenMath high-batch K gaps were filled with public PARD K3/K8 standalone jobs:

| K | Job | Batch 64 | Batch 128 | Acceptance |
| ---: | ---: | ---: | ---: | ---: |
| 3 | `3212856` | `1.308x` | `1.244x` | `57.82%/58.30%` |
| 8 | `3212858` | `1.156x` | `1.041x` | `32.33%/31.44%` |

The completed sweep is tracked in
`docs/qwen3_235b_public_pard_openmath_k_sweep_20260608.csv`. Current completed
rows show K3 as the best high-batch static point: K3 `1.308x/1.244x`,
K5 `1.259x/1.170x`, K7 `1.196x/1.114x`, K8 `1.156x/1.041x`, and K9
`1.186x/1.039x` for batch `64/128`.

The matched NeMo-RL GBS512 K3 gate did not preserve that ordering. Job
`3212919` completed 5/5 with Step2-5 total `96.28s`, generation `59.76s`,
E2E `13.34`, generation worker `192.73`, and visible acceptance `56.64%`.
It is still a strong win over baseline `3212012`, but static K5 `3212209`
remains faster: K5 is about `1.133x` faster in total step and `1.182x` faster
in generation time than K3.

So the next systems lever should be dynamic: keep speculation aggressive only
where the scheduler shape and acceptance make it worthwhile.

## Current Runtime Patch State

Remote checkout:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606`

Deployed and verified on 2026-06-08:

| Control | Current remote status |
| --- | --- |
| `VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=true` | Deployed |
| `VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD` | Deployed |
| `VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD` | Deployed |
| `VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL` | Deployed |
| `VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS=true` | Deployed in repo overlay; actor venv smoke pending |
| `VLLM_SPECDEC_ADAPTIVE_GATE_MODE` | Not implemented in the deployed minimal patch |

Patch overlay update after the GBS512 K3/K5 comparison:

- `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/specdec_runtime_gate_patch.py`
  now contains a batch-global dynamic-K draft cap for non-async PARD
  parallel-drafting paths.
- The patch changes the actual drafter proposer cap, not only the
  scheduler list length: it temporarily lowers the proposer
  `num_speculative_tokens`, adjusts the parallel-drafting slot counts for the
  drafter forward, and truncates the CPU draft-token list to the selected K.
- This has passed local apply/idempotency/`py_compile` checks against the vLLM
  0.20 source copy and has been synced into the remote repo overlay. It still
  must be proven inside the runtime actor venv before using dynamic-K perf
  numbers.
- A first live smoke attempt, job `3213293`, was intentionally cancelled after
  marker checking showed the actor venv still had unpatched vLLM files. Root
  cause: the Qwen3-235B submit wrapper had hardcoded
  `NRL_FORCE_REBUILD_ACTOR_VENVS=false`, so the running actor reused a stale
  cached venv that did not contain the updated runtime patch module.
- The wrapper now allows `NRL_FORCE_REBUILD_ACTOR_VENVS` to be overridden.
  Dynamic-K smoke/perf jobs must set `NRL_FORCE_REBUILD_ACTOR_VENVS=true` until
  the actor venv cache is known to contain these markers.

Future-submit readiness:

- `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` now passes
  `VLLM_SPECDEC_DYNAMIC_DRAFT_*` env vars through the driver command.
- `experiments/eagle3_online/submit_qwen235b_public_pard_noncolocated_tp4_fixed256_step5.sh`
  now exposes `K_DYNAMIC_*` wrapper variables for future dynamic-K jobs.
- The same Qwen3-235B wrapper now defaults to the performance shape
  `NUM_PROMPTS=16`, `NUM_GENERATIONS=32`, and
  `TRAIN_GLOBAL_BATCH_SIZE=512`. Treat the earlier GBS256 20-step pass as a
  functional/stability check, not the Qwen3-235B performance gate.
- The wrapper also now allows actor venv rebuild to be forced through
  `NRL_FORCE_REBUILD_ACTOR_VENVS=true`, which is required for the first
  dynamic-K smoke after patch updates.
- This means dynamic-K can be submitted for smoke, but not yet counted as a
  performance result. The runtime vLLM package inside the actor venv must still
  be marker-checked, and the logs must show selected-K counters moving.

Important: do not submit dynamic-K or adaptive-threshold jobs until the
runtime patch file contains the dynamic drafter-cap markers and a smoke run
proves the selected-K counters move. The remote repo overlay now contains the
markers; the remaining gate is actor-venv/runtime proof.

Activation evidence:

- Job `3212586` was submitted as a GBS512 K5 runtime-gate smoke.
- vLLM package files inside the actor venv contained
  `NRL_SPECDEC_BATCH_GATE_PATCH_V9`,
  `NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V6`, and
  `NRL_SPECDEC_SCHEDULER_OUTPUT_SCRUB_ON_GATE_V2`.
- `ray-driver.log` showed the gate reading `request_threshold=16` and
  disabling speculation at `requests=32`.
- Job `3212586` was cancelled before timing metrics because
  `GATE_LOG_INTERVAL=1` created heavy logging overhead. It is activation proof
  only, not a performance result.
- Performance-valid gated K5 job `3212643` was submitted with
  `GATE_LOG_INTERVAL=0`, but failed before driver metrics because Ray workers
  did not start cleanly on that allocation.
- Retry2 `3212702` produced Step2-5 metrics with no fatal/OOM pattern. It is
  baseline-like, not a SpecDec win: total `151.19s`, generation `115.99s`,
  E2E `8.49`, generation worker `99.29`. Against matched GBS512 baseline
  `3212012`, that is only `1.017x` total-step and `0.996x` generation-time.
  Against static K5 `3212209`, gate16 is much worse; static K5 is `1.779x`
  faster in total-step and `2.293x` faster in generation-time. No acceptance
  buckets emitted, consistent with gate16 disabling speculation in the dense
  decode phase.
- Dynamic-K smoke attempt `3213293` was cancelled after actor-venv marker checks
  showed a stale vLLM package. Root cause was the wrapper forcing
  `NRL_FORCE_REBUILD_ACTOR_VENVS=false`; the wrapper now permits override.
- Dynamic-K smoke attempt `3213449` was cancelled after Ray head failed during
  container filesystem setup (`pyxis_3213449_ray-head/proc` already existed).
  Treat it as a cluster/container startup failure, not a dynamic-K result.
- Retry `3213497` was submitted with the same GBS512 shape and
  `NRL_FORCE_REBUILD_ACTOR_VENVS=true`.
- Job `3213497` completed the one-step GBS512 dynamic-K smoke. Actor venv marker
  check passed inside the `ray-head` container:
  `static_gate=present`, `dynamic_gate=required_present`.
- Runtime logs proved true dynamic drafter-cap behavior, not only scheduler
  list trimming: `NRL SpecDec dynamic draft cap: checked=1 selected=3
  tier=large requests=32 tokens=256 max=5`, while the batch gate stayed enabled
  at `request_threshold=128`.
- `3213497` Step1 timing: total step `309.83s`, generation `65.78s`, E2E
  `4.10` tokens/sec/GPU, generation worker `173.62` tokens/sec/GPU. This is a
  smoke/runtime proof, not a matched perf result, because Step1 includes cold
  setup effects and there is no Step2-5 window.
- Do not attach to `--container-name=ray-head` during startup; first verify Ray
  head readiness from logs, then attach to the running container with
  `--jobid=<job_id>`. A host filesystem read of the pyxis root is useful for
  discovery but may hit permissions on package files.
- Dynamic medium16 5-step perf job `3213606` was submitted after `3213497`
  passed. It uses the same GBS512 shape and forces dense `requests=32` to the
  large tier (`selected=3`) for a real dynamic-K-vs-static-K3/K5 comparison.
- A marker check on the `ray-head` venv for `3213606` looked stale, but that
  was a false negative: the head container did not host an active
  `VllmGenerationWorker`. Checking an actual generation container
  (`ray-worker-1` on `nvl72068-T02`, found from the IP/node mapping in
  `ray-driver.log`) passed with `static_gate=present` and
  `dynamic_gate=required_present`. Future marker checks should target a
  container that actually hosts a generation actor.
- Job `3213606` completed 5/5 at GBS512 with no OOM. Dynamic logs showed
  `selected=3 tier=large requests=32` through the dense decode phase and
  `selected=5 tier=small` only on small tail batches.
- `3213606` Step2-5 averages: total `98.2425s`, generation `60.6025s`,
  policy training `14.3850s`, E2E `13.1450` tokens/sec/GPU, generation worker
  `190.0850` tokens/sec/GPU, weighted logged acceptance `56.83%`.
- Against matched baseline `3212012`, `3213606` is positive: total-step
  `1.565x`, generation-time `1.907x`, E2E `1.575x`, generation-worker
  throughput `1.908x`.
- Against static K5 `3212209`, `3213606` is worse: K5 is `1.156x` faster in
  total time and `1.198x` faster in generation time. Against static K3
  `3212919`, `3213606` is almost identical but slightly slower: static K3 is
  about `1.020x` faster in total time and `1.014x` faster in generation time.
  Conclusion: medium16 dynamic-K is valid runtime proof, but static K5 remains
  the validated GBS512 NeMo-RL default.

## Planned Patch Surface

The fuller dynamic/adaptive patch should expose these controls:

| Control | Meaning |
| --- | --- |
| `VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=true` | Required guard; fails fast if thresholds are set but the runtime patch is not active |
| `VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD` | Disable speculative drafting above this active-request count |
| `VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD` | Disable speculative drafting above this scheduled-token count |
| `VLLM_SPECDEC_ADAPTIVE_GATE_MODE` | Enables adaptive adjustment of the request/token gate thresholds |
| `VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO` | Target fraction of scheduler iterations with speculation enabled |
| `VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS=true` | Enables tiered dynamic draft-token caps |
| `VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS` | Draft tokens for small scheduler shape |
| `VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS` | Draft tokens for medium scheduler shape |
| `VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS` | Draft tokens for large scheduler shape |

The patch must validate that gate/dynamic settings cannot silently run as
global SpecDec. The current deployed minimal static gate already fails fast for
static request/token thresholds when the runtime patch guard is not enabled.

## Candidate Policies

### Policy A: Static K5 Control

Use this as the current reference after job `3211706`:

```text
NUM_SPECULATIVE_TOKENS=5
VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=false
```

This is the baseline for the submitted 20-step stability job `3211900`.

### Policy B: K7/K5/K3 Dynamic Cap

This is a future lower-priority exploratory candidate, not ready to submit on
the current remote checkout. Standalone K7 job `3211982` completed, but K7 did
not beat K5 at batch 64 or 128.

```text
NUM_SPECULATIVE_TOKENS=7
VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=true
VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS=true
VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD=128
VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD=16
VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD=64
VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS=7
VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS=5
VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS=3
VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL=64
```

Interpretation: allow K7 only in small scheduler shapes, use K5 in moderate
batch pressure, and drop to K3 under high pressure. Because K7 was slower than
K5 at high batch, this should not be the default production gate.

### Policy C: K5/K3/Off Conservative Gate

Use this as the preferred future adaptive-K candidate after the current
results: K3 is the best high-batch standalone point, but static K5 is the
validated Qwen3-235B GBS512 NeMo-RL winner after `3212919`. Use K3 only as a
defensive cap for overload/long-tail shapes, not as the default replacement for
K5. Do not submit this exact policy until the deployed runtime patch includes
dynamic draft-token cap support.

```text
NUM_SPECULATIVE_TOKENS=5
VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=true
VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS=true
VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD=128
VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD=16
VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD=64
VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS=5
VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS=5
VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS=3
VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL=64
```

This avoids K9 entirely, preserves K5 for the validated GBS512 dense region,
and only drops to K3 under heavier scheduler pressure.

## Next Gates

1. Treat `3212702` as the negative control for static on/off request gating:
   request-threshold gating is not useful for `max_num_seqs=32` because it
   disables speculation during the dense part of the batch.
2. Treat `3212919` as the matched static-K3 GBS512 result. It completed 5/5
   and confirms K5 remains the NeMo-RL GBS512 static baseline despite K3 being
   best in standalone OpenMath batch64/128.
3. Only after the dynamic drafter-cap patch is observed inside the actor venv
   and smoke verified, run Policy C as a true K5/K3/Off candidate.
4. Keep Policy B only as an exploratory small-batch/K7 cap, not as the default
   high-batch setting.
5. Treat `3213606` as the first completed 5-step dynamic medium16 performance
   job. It confirms the runtime cap works at GBS512 and does not OOM, but it is
   slower than static K5 and essentially static-K3-like because
   `medium_request_threshold=16` selects K3 during dense `max_num_seqs=32`
   decode. Do not replace the current GBS512 default with this policy.

## Dynamic-K Implementation Notes

Static request gating is a blunt instrument for the current Qwen3-235B GBS512
shape:

- vLLM runs with `max_num_seqs=32`.
- Activation smoke `3212586` showed steady scheduler batches at `requests=32`
  and `tokens=32`.
- With `VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD=16`, speculation is disabled
  for the dense decode phase instead of reduced from K5 to K3/K1.
- With threshold `32`, the gate almost never disables because the current
  predicate is strict `requests > threshold`.

So a useful Nightjar-style path is not pure on/off gating. It needs dynamic
lookahead caps:

```text
NUM_SPECULATIVE_TOKENS=5
VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=true
VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS=true
VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD=16
VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD=64
VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS=5
VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS=5
VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS=3
```

This keeps K5 for the normal `requests=32` dense region and drops to K3 only
under higher scheduler pressure or future decode-heavy shapes. The patch overlay
now implements the safer first version as a batch-global cap for non-async PARD
parallel drafting. It intentionally does not attempt per-request heterogeneous
K, because that requires deeper scheduler/kernel changes.

The current remote repo overlay contains dynamic request/token gating plus the
batch-global drafter cap. Do not treat a dynamic job as valid until these
markers appear inside the actor venv:

```text
NRL_SPECDEC_DYNAMIC_DRAFT_RESET_PATCH_V1
NRL_SPECDEC_DYNAMIC_DRAFT_CAP_PATCH_V1
NRL_SPECDEC_DYNAMIC_DRAFT_PROPOSER_CAP_PATCH_V1
NRL_SPECDEC_DYNAMIC_DRAFT_CPU_COPY_PATCH_V1
NRL_SPECDEC_DYNAMIC_DRAFT_CPU_TRUNCATE_PATCH_V1
```

Use the local checker before any future dynamic-K smoke:

```bash
python3 experiments/eagle3_online/check_runtime_gate_markers.py \
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/lib/python3.13/site-packages/vllm \
  --require-dynamic
```

For the current minimal static gate, omit `--require-dynamic`.

Implementation caveat: for PARD `parallel_drafting=true`, a scheduler-only cap
that trims `scheduled_spec_decode_tokens` or `request.spec_token_ids` may reduce
verification/KV pressure but not necessarily drafter compute. The PARD drafter
path uses vLLM `DraftModelProposer`/`SpecDecodeBaseProposer`, where parallel
drafting can generate the configured K slots in one pass. A performance-valid
dynamic-K patch must reduce the actual drafter input/output depth, or the
controller may look correct in acceptance counters while still paying K5/K8
draft cost.

## Reporting Rule

Do not claim adaptive-K benefit from acceptance alone. Each gate must report:

- total step time
- generation time
- E2E tokens/sec/GPU
- generation worker tokens/sec/GPU
- acceptance rate
- enabled/disabled or dynamic-tier counts from the runtime gate logs
