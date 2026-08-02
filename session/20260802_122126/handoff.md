# Handoff

## Resume From Here

The Qwen3-30B-A3B and Qwen3-235B-A22B design is committed as `d94ccc8d8` and
the implementation now has the safe A/B/C/E matrix. OCI-HSG already completed
a separate non-CG Qwen235 16n4g 20-step run, so the model/cache/topology are
ready. Do not launch R3 plus `moe_router` or `moe_preprocess` CG reuse: route
IDs are not graph replay inputs and the launcher rejects this before Slurm.

## Next Actions

- Complete the dedicated final source review and push the reviewed experiment
  branch.
- Create a clean remote OCI campaign checkout, then run a new four-GPU runtime
  attestation with the exact source, nested revisions, lockfile, nightly image
  digest, and mounted managed Python/uv paths.
- Put the successful attestation path and preflight job ID into the OCI profile.
- Run `TEST_ONLY=1` for the Qwen30 and Qwen235 A/B/C/E smoke matrices, submit
  only the Qwen30 five-step smoke, and monitor it for five minutes.
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
