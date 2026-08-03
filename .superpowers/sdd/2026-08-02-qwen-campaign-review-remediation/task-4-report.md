# Task 4 final runbook and verification report

Status: local verification complete; remote setup and GPU submission remain
pending. No scheduler, W&B network, or remote-cluster mutation occurred during
this task.

Implementation:

- The runbook distinguishes the historical Nano evidence from the unsubmitted
  Qwen3-30B-A3B and Qwen3-235B-A22B campaign. It documents immutable source,
  nested-gitlink, runtime, container, profile, Router Replay, and promotion
  evidence requirements.
- TensorBoard and W&B examples use the strict 12-field launch identity. The W&B
  path scans unfiltered sparse history and requires an explicit optimizer-step
  key. Neither exporter fills missing metrics with zero.
- Final reporting audit found a real contract gap: graph-arm export required
  cache misses, but the worker did not emit them. Commit `f31d46874` adds an
  exact source counter. `warming` and `captured` are lookup misses, `hit` is
  not, and no miss is synthesized from capture count. Runtime and aggregation
  enforce `capture_count <= cache_miss_count` and
  `eviction_count <= capture_count`.
- Qwen smoke and performance commands are model-isolated. A Qwen30 promotion
  gate cannot promote Qwen235, and every performance arm requires its own
  model-matched five-step smoke-promotion evidence.
- Final documentation audit found that the former Qwen235 C/E R3 envelope was
  content-addressed but self-attested: it did not bind raw diagnostic bytes,
  exact argv, exit status, or the successful Slurm attempt. R3 validation now
  rejects every such gate until a committed content-bound producer exists, so
  Qwen235 C/E are dependency-blocked while A/B remain runnable.
- The same audit found that legacy generic Qwen performance/accuracy wrappers
  could bypass or deterministically fail the campaign contract. They now reject
  Qwen selectors before launch output and route operators to the dedicated
  campaign submitter.
- The fail-closed boundary remains explicit: Router Replay plus
  `moe_router`/`moe_preprocess` graph reuse is not submitted because route IDs
  are not graph replay inputs. Arm E graphs only attention while the router
  remains eager.

Verification evidence:

- CUDA Graph lifecycle: 89 passed.
- Policy worker plus packing: 70 passed.
- Algorithm telemetry: 59 passed.
- Independent targeted cache/export/reporting review: 292 passed.
- Campaign matrix: 45 passed; launcher suite: 96 passed (141 combined).
- Exporter rerun: 59 passed.
- Shell syntax, Python compilation, Ruff, and `git diff --check`: passed.
- Offline `TEST_ONLY=1` smoke render passed for Qwen30 A/B/C/E and Qwen235
  A/B, with no run-directory creation or scheduler contact.
- Invalid `TEST_ONLY=2`, mutually enabled dry-run controls, and direct Qwen235
  C without R3 evidence each exited 2 and printed no `SBATCH:` line.
- Final pre-audit and post-audit independent reviews both reported
  `ADDRESSED`, with 0 critical findings, 0 warnings, and 0 nits. The post-audit
  review reran the 141 campaign tests plus shell syntax, compilation, Ruff, and
  diff checks.

Local environment limitation:

- `tests/unit/algorithms/test_grpo.py` and
  `tests/unit/single_controller/test_sc_utils_helpers.py` cannot collect on the
  arm64 macOS lock because Linux-only `torchdata` and `tensordict` are absent.
  Their changed expectations are schema-only. Run them in the attested OCI
  nightly environment before promoting the first smoke.

Remaining operational gates:

1. Push the reviewed source and create a clean recursive-submodule checkout on
   OCI-HSG.
2. Generate a new source snapshot and four-GPU runtime attestation for the
   exact outer commit, Bridge/MCore gitlinks, lockfile, nightly container
   digest, Transformer Engine revision, managed Python, and uv executable.
3. Recheck FairShare and run `sbatch --test-only` from the attested checkout.
4. Submit only the Qwen30 five-step smoke, monitor it for at least five minutes,
   and export all planned correctness and graph metrics before promotion.
