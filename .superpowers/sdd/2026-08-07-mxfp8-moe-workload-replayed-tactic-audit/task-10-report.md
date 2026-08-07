# Task 10 Report: MXFP8 MoE Tactic-Audit Review Fixes

## Delivered

- Collector now parses the GRPO producer labels `Loss`, `Generation KL Error`,
  `Avg Reward`, and `Mean Generation Length` together with the established
  timing and throughput fields. It requires exactly measured steps 3-8,
  explicit positive per-step realized token evidence, exit success, and
  successful `refit`, `rollout`, `logprob`, and `train` phases.
- Validation submission now writes `run_evidence.json` with arm, exit, phase,
  metadata, and observed Training Results evidence. It intentionally records
  unavailable exact token counts as `null`; the collector fails closed instead
  of inferring tokens from mean generation length.
- Qualification decisions, stock/candidate cache selections, trace summary,
  profile coverage, and hashes bind exact successful shmoo rows. Failed or
  zero-timing rows cannot select a tactic. NSys FC1/GEMM1 and FC2/GEMM2 rows
  are keyed by signature, cache key, arm, component, and selected tactic.
- Executed failed correctness evidence renders `REJECT` with raw data;
  insufficient/malformed evidence renders `INCOMPLETE` or `NOT YET EXECUTED`.
  Promotion requires at least two comparable runs per arm and measured
  run-to-run variation. Within-run steps are displayed separately.
- HTML is structured with tables and embedded figures. Executed reports carry
  raw steps 3-8 reward/loss/KL tables, four figure links, cache/trace/decision
  hashes, source fingerprints, and an explicit KEEP/REJECT conclusion.
- Regenerated the committed no-artifacts template. It is explicitly `NOT YET
  EXECUTED`, contains no performance claims, and embeds four unavailable-data
  figures for Task 11 to replace with measured artifacts.

## Verification

```text
PYTHONPATH=. .venv/bin/pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py

24 passed, 16 macOS pytest temporary-directory cleanup warnings in 12.69s

PYTHONPATH=. .venv/bin/ruff check \
  experiments/mxfp8_moe_tactic_audit/collect_results.py \
  experiments/mxfp8_moe_tactic_audit/plot_results.py \
  experiments/mxfp8_moe_tactic_audit/build_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py

All checks passed.

PYTHONPATH=. .venv/bin/pyright \
  experiments/mxfp8_moe_tactic_audit/collect_results.py \
  experiments/mxfp8_moe_tactic_audit/plot_results.py \
  experiments/mxfp8_moe_tactic_audit/build_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py

0 errors; 1 seaborn source-resolution warning.
```

All four template PNGs are 4140x1473 at 600 DPI. Their PDF counterparts are
single-page vector PDF 1.4 files. Visual inspection confirms the template
plots clearly state `NOT YET EXECUTED` and contain no fabricated values.
The regenerated HTML has four `<figure>/<img>` entries and no `<pre>` block;
the Markdown and HTML include steps 3-8, phase status, realized-token,
finite-metric, 95% coverage, FC1/GEMM1, FC2/GEMM2, cache hit/fallback, GSM8K,
and trace/qualification provenance requirements.

## Commit

Review-fix implementation: `8f02405429af48879426c81d1a2e62cce00a6beb`

## Fix Round 2

- The synchronous GRPO producer emits explicit successful `refit`, `rollout`,
  `logprob`, and `train` markers for each completed step. The validation
  launcher derives `run_evidence.json` from those markers and the producer's
  `train_data_step*.jsonl` `token_loss_mask` fields. It records only positive,
  directly measured generated-token totals for steps 3-8.
- `qualify_cache.py` now emits authoritative `qualification_decisions.json`
  alongside its real cache manifest. It includes selected and stock tactics,
  promotion decisions, signature bindings, and cache/trace/profile/shmoo
  fingerprints. The report checks these bindings and excludes failed/zero
  shmoo rows before any tactic metric is accepted.
- Trace-set provenance now uses the same sorted member-digest JSON algorithm
  as `CacheProvenance._sha256_file_set`. Report source hashes list each raw
  trace artifact. Run evidence must carry cache-manifest runtime fingerprints.
- Paired GSM8K output now records `matched_examples=1319`; collection requires
  provenance matching, all paired count fields summing to 1319, accuracy,
  McNemar, and CI data. Executed reports show replay coverage and paired GSM8K
  comparison values.
- Existing execution artifacts that are malformed, mismatched, or missing
  component evidence render `INCOMPLETE`; only a complete failed gate renders
  `REJECT`. Duplicate paths/run IDs, wrong arm labels, and non-comparable
  repetition manifests cannot satisfy the stability gate. Per-step plots now
  show every repetition, labelled by arm/run/step.

## Fix Round 2 Verification

```text
PYTHONPATH=. .venv/bin/pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py \
  tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py

60 passed, 16 macOS pytest temporary-directory cleanup warnings

PYTHONPATH=. .venv/bin/ruff check <changed audit modules and tests>
All checks passed.

PYTHONPATH=. .venv/bin/pyright <changed audit modules>
0 errors; 1 pre-existing seaborn source-resolution warning.
```

Regenerated template report: all four PNGs are 4140x1473 at 600 DPI; HTML is
structured and has four embedded figure elements. The template is explicitly
`NOT YET EXECUTED` and makes no measured-performance claim.

## Fix Round 3

- The shmoo launcher now runs the real `nsys stats nvtxppsum` conversion after
  capture. `nsys_to_component_csv.py` accepts the tagged NSys summary and
  emits per-profile signature/cache/arm/component/tactic/cache-event/call-
  weight/call-count/timing rows. Real shmoo runs load the stock
  `autotune_configs.json`, force those exact stock tactics, and tag FC1/GEMM1
  and FC2/GEMM2 independently; synthetic smoke remains isolated.
- The collector validates the canonical `CacheManifest` schema, all
  qualification decision fingerprints, exact arm-to-cache hashes, and the
  independently observed runtime values. Validation evidence computes the
  cache hash at runtime and records actual NeMo-RL/vLLM checkout SHAs,
  FlashInfer/CUDA/GPU values, topology settings, and CUDA-graph mode.
- Repetitions allow a shared logical comparison ID across stock and candidate
  while requiring unique IDs within each arm, matching run settings, and one
  invariant cache identity per arm. Correctness and GSM8K payloads must bind
  the exact stock/candidate run-manifest hashes and comparison IDs.
- GSM8K acceptance recomputes stock/candidate accuracy and delta from the four
  paired cells, verifies all 1319 examples, validates statistical/pass-field
  consistency, and rejects inconsistent producer payloads. Fractional NSys
  call counts or call weights fail closed as `INCOMPLETE`.
- Component charts receive the per-profile FC1/GEMM1 and FC2/GEMM2
  distributions rather than component means. Captions enumerate every stock
  and candidate arm/run/batch/topology represented. Raw trace files are named
  only as sanitized member labels in report provenance.

## Fix Round 3 Verification

```text
PYTHONPATH=. .venv/bin/pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py \
  tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py

69 passed, 16 macOS pytest temporary-directory cleanup warnings in 28.42s

.venv/bin/ruff check <changed audit modules and tests>
All checks passed.

.venv/bin/pyright <changed audit modules and tests>
0 errors.

bash -n experiments/mxfp8_moe_tactic_audit/submit_shmoo_ptyche.sh \
  experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh
0 errors.
```

Regenerated and visually inspected the explicit `NOT YET EXECUTED` template:
all four PNGs are 4140x1473 at 600 DPI, every PDF is a single-page PDF 1.4,
and the HTML has four `<figure>/<img>` entries with no escaped Markdown or
`<pre>` block. The template carries no fabricated performance values.

## Fix Round 4

- The shmoo launcher now consumes the actual `nsys-selected.nsys-rep` output
  and keeps `selected_profiles.json` as an `audit_write_manifest` argument.
  Stock and candidate FC1/GEMM1 and FC2/GEMM2 paths each perform graph setup,
  warmups, and capture before entering ten equivalent measured NVTX ranges.
  Correctness checks remain outside those ranges.
- NVTX labels carry the exact comparison tactic and a cache hit/fallback event
  observed from the active FlashInfer autotuner state. The converter emits the
  aggregate statistic as `mean_us`. The report pairs each stock range to its
  candidate comparison and weights speedups only by the selected profile's
  trace `call_weight`, independent of NSys range instance count.
- `observe_runtime.py` independently reads runtime checkouts, package/CUDA/GPU
  identity, execution topology, CUDA Graph mode, model snapshot revision, the
  exact cache file hash, and the container hash. File hashing uses bounded
  1-MiB reads. Runtime evidence and cache provenance require
  `container_sha256`; the report binds the observed container and cache hashes
  to each exact run manifest.
- Deterministic generation and GSM8K producers now bind their output to exact
  stock/candidate manifest hashes, explicit arm IDs, and sorted logical
  comparison IDs. Compare mode supports multiple repetition IDs and writes a
  distinctly named deterministic-generation artifact rather than presenting
  one generation gate as the complete correctness summary.
- Every stock/candidate repetition must agree on batch, topology, run kind,
  generation settings, and every non-cache run-manifest field. Cache identity
  is invariant within each arm and distinct across arms.
- GSM8K output records the ordered paired outcomes, their SHA256, and the
  deterministic bootstrap seed/sample contract. Report collection recomputes
  the exact two-sided McNemar p-value and bootstrap CI, validates probability
  and CI ranges, requires the observed delta inside the interval, and verifies
  the producer's `passed` value from the recomputed gates.
- Regressions exercise the real launcher with fake `sbatch`, actual shmoo NVTX
  range production through the converter/report, runtime cache-event evidence,
  bounded container hashing, exact correctness/GSM8K producer bindings,
  profile-only weighting, repetition mismatches, runtime hash mismatches, and
  malformed or unreproducible GSM8K statistics.

## Fix Round 4 Verification

```text
PYTHONPATH=. .venv/bin/pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py \
  tests/experiments/test_mxfp8_moe_tactic_correctness.py \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py

141 passed, 16 macOS pytest temporary-directory cleanup warnings in 39.19s

ruff check <changed audit modules and tests>
All checks passed.

pyright <changed audit production modules>
0 errors, 0 warnings, 0 informations.

bash -n experiments/mxfp8_moe_tactic_audit/submit_shmoo_ptyche.sh \
  experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh \
  experiments/mxfp8_moe_tactic_audit/provenance.sh
0 errors.

ruff format --check <changed Python modules and tests>
All files formatted.

git diff --check
0 errors.
```

No GPU or live NSys workload was executed locally. GPU behavior is covered by
the existing mocked CUDA contracts and the new producer-to-converter tests;
the corrected launcher path still requires execution on the target GB200
cluster to produce measured audit artifacts.

## Fix Round 5

- The shmoo launcher redirects `nsys stats` CSV stdout to the exact
  `nsys-nvtx.csv` path consumed by the converter, independent of NSys report
  suffix conventions. A shell/mock regression executes the rendered launcher
  command with an NSys mock that reproduces the old `_nvtxppsum.csv` suffix.
- Measured NVTX ranges now identify the FC1 cumulative endpoint and the
  FC1+FC2 cumulative endpoint. Device synchronization occurs after each range.
  The converter rejects legacy direct-component labels, missing or duplicate
  cumulative stages, unequal call counts, and non-positive subtraction before
  emitting normalized FC1/GEMM1 and derived FC2/GEMM2 rows.
- Producer-derived timing tests track the active range and captured backend
  mode. They prove that setup, warmup, graph capture, and correctness work are
  outside measured ranges, that each range contains only its intended graph
  replay, and that synchronization is outside every range.
- Deterministic generation and GSM8K producers derive bindings only from the
  stock/candidate artifacts they actually evaluate. Their CLIs no longer
  accept arbitrary extra run roots, compare mode requires exactly one run ID,
  and the report accepts that exact evaluated pair while repeated E2E runs
  remain the independent variation evidence.

## Fix Round 5 Verification

```text
PYTHONPATH=. .venv/bin/pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py \
  tests/experiments/test_mxfp8_moe_tactic_correctness.py \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py

145 passed, 16 macOS pytest temporary-directory cleanup warnings in 40.69s

.venv/bin/ruff check <changed Python modules and tests>
All checks passed.

.venv/bin/ruff format --check <changed Python modules and tests>
9 files already formatted.

.venv/bin/pyright <changed production modules and type-clean changed tests>
0 errors, 0 warnings, 0 informations.

bash -n experiments/mxfp8_moe_tactic_audit/submit_shmoo_ptyche.sh \
  experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh \
  experiments/mxfp8_moe_tactic_audit/provenance.sh
0 errors.
```

The changed correctness test module retains 43 pre-existing Pyright errors in
unrelated fixtures; the new binding regression is outside those diagnostics.
`uv run` cannot currently parse the repository workspace because `nemo-gym`
is configured as a workspace source but is not a workspace member, so focused
verification used the existing `.venv` directly. No GPU or live NSys workload
was executed locally; the shell/mock and CUDA producer contracts cover the
scoped regression paths pending target GB200 execution.
