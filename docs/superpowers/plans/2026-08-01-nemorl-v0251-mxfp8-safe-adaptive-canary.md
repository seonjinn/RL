# NeMo-RL vLLM 0.25.1 Safe Adaptive Canary Plan

1. Add a pure arm-contract helper that validates paths and constructs the exact
   baseline or adaptive environment.
2. Add a log summarizer that records completion, elapsed time, output tokens,
   and relevant provenance without accepting partial runs as final.
3. Add a NeMo-RL eval configuration derived from the current Ultra generation
   settings with TP8/EP8 and CUDA Graph enabled.
4. Add a Ptyche launcher that runs both arms sequentially in one two-node
   allocation and preserves separate logs and manifests.
5. Run unit, shell syntax, lint, and diff checks locally.
6. Commit and push the exact source state, pull it on Ptyche, run scheduling and
   import preflights, submit, and monitor the first five minutes.
7. If both generation arms pass, add a one-refit canary on the refit branch and
   report prepared-weight rebuild time separately.

