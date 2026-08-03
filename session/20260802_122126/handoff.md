# Handoff

## Resume From Here

The Qwen3-30B-A3B and Qwen3-235B-A22B implementation is committed through
`75ddbef3d`, with final documentation and session notes still in the working
tree. The safe matrix is A = eager/R3-off, B = `moe_router`/R3-off,
C = eager/R3-on, and E = `attn`/R3-on. Do not launch R3 plus `moe_router` or
`moe_preprocess` graph reuse: route IDs are not graph replay inputs and the
launcher rejects this before Slurm. No campaign GPU job has been submitted
from this branch. Qwen235 C/E are also dependency-blocked because the former
R3 preflight envelope was not bound to raw diagnostic execution; only A/B may
run until a content-bound Slurm producer is implemented.

## Next Actions

- Finish the focused tests/re-review for the legacy-launcher and self-attested
  R3 fail-closed remediation.
- Commit the final gate/runbook/session ledger, then push the reviewed
  experiment branch. Exact cache-miss telemetry is committed as `f31d46874`
  and unattested-route remediation as `75ddbef3d`.
- Create a clean remote OCI campaign checkout, then run a new four-GPU runtime
  attestation with the exact source, nested revisions, lockfile, nightly image
  digest, and mounted managed Python/uv paths.
- Put the successful attestation path and preflight job ID into the OCI profile.
- Recheck OCI FairShare and `sbatch --test-only`, then submit only the Qwen30
  five-step smoke and monitor it for at least five minutes.
- Gate 20-step Qwen30 and Qwen235 comparisons on R3 trace/validation, router
  and expert parity, finite gradients, and no logprob/KL correctness outlier.

## Watch Outs

- Use exactly three successful optimizer warmups and disable checkpoints.
- Use `batch`, not `backfill`, and monitor every submitted job for at least five minutes.
- Do not compare independent stochastic rollouts as proof of CUDA Graph correctness; use fixed-input route/output/gradient parity first.
- Do not store credentials in session or experiment files.
- Keep A/B/C/E paired on model, phase, dispatcher, cluster/runtime profile,
  repeat, and Router Replay state when interpreting throughput; do not compare
  R3-on and R3-off arms as a pure CUDA Graph speedup.
- A graph arm is incomplete unless every optimizer step exports exact
  `cache_miss_count`; do not substitute capture count because warmup misses do
  not capture.
- Do not revive Qwen235 C/E by hand-authoring a gate JSON. Implement and review
  the content-bound Slurm diagnostic producer first.
