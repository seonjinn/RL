# Task 3 identity-safe metric export report

Status: local-only verification; no W&B network call, scheduler contact, push,
or remote-cluster mutation.

Implementation:

- `export_tensorboard.py` now parses `run-metadata.env` as strict untrusted
  UTF-8 data from one `O_NOFOLLOW` regular-file descriptor. It rejects empty,
  duplicate, malformed, quoted, continued, shell-active, CR/NUL, and invalid
  UTF-8 input without evaluating any value.
- TensorBoard and W&B share one canonical launch identity, provenance/parity
  validation, tag aliases, per-step completeness and finiteness checks, record
  builder, and atomic JSONL writer. Metadata/CLI disagreements fail before
  metric reads or output replacement.
- Router Replay is a mandatory `off|on` identity field. The two existing
  reporting call sites now pass it explicitly.
- TensorBoard is imported only in its event reader. The new W&B exporter
  imports `wandb` only in CLI code and accepts an injected `scan_history()`
  run protocol for pure unit tests.
- W&B history is scanned unfiltered so sparse metric rows are retained, then
  coalesced by configured optimizer-step keys. Bool, fractional, non-finite,
  missing, conflicting, or out-of-window step identities fail closed; `_step`
  has no implicit optimizer-step meaning.
- Baselines may omit all graph telemetry. Every non-baseline optimizer step
  must contain every graph and correctness metric, including
  `cuda_graph/cache_misses`; missing or non-finite values preserve the prior
  output unchanged.

TDD and verification evidence:

- RED: TensorBoard and W&B regressions proved that a non-baseline missing
  `cuda_graph/cache_misses` was incorrectly accepted.
- GREEN: the shared optional-tag exception was removed; both regressions pass,
  and the stale reporting expectation was corrected.
- Exporter/reporting suite: `74 passed`.
- Matrix/gate suite: `47 passed`.
- Launcher suite: `86 passed`.
- Python compilation, Ruff checks, and `git diff --check` passed.
- Final independent re-review: `ADDRESSED`, with 0 critical findings,
  0 warnings, and 0 nits.

The aggregate related verification is 207 passing tests. No optional exporter
dependency was required at module import, and unit tests made no W&B network
calls.
