# ModelOpt Upstream Drift Report

Overall: **WARN**
Generated: `2026-05-22 06:51:22 PDT`
Official example: https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/speculative_decoding

## Summary

- Local ModelOpt: `/Users/sna/Nemo-RL_Qwen3_Roadmap/Model-Optimizer`
- Local branch/head: `main` / `c9098b63fb5e`
- Dirty files: `1`
- Upstream probe: `ok` / `3ff15ccef3f0`
- Training source decision: `warn`
- Training source summary: local ModelOpt HEAD is not at official upstream main
- Hayate ModelOpt visible: `False`
- Hayate branch/head: `None` / `None`
- Hayate dirty/untracked files: `0`

## Training Source Decision

- Status: `warn`
- Upstream HEAD matches local: `False`
- Allowed focus diffs: `examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py`
- Disallowed focus diffs: `-`
- Unrelated dirty files: `0`
- Hayate reference only: `True`
- Recommendation: Use the local/remote ModelOpt checkout as the training source. Treat Hayate's checkout as workflow reference only; port only intentional workflow ideas, not the older checkout wholesale.

## Notes
- local ModelOpt worktree has uncommitted changes
- local ModelOpt HEAD does not match probed official upstream main
- Hayate ModelOpt path is not visible on this host

## Local Dirty Files

- `M examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py`

## Local Focus Diff Stat

```text
.../compute_hidden_states_trtllm.py                | 49 ++++++++++++++++++----
 1 file changed, 40 insertions(+), 9 deletions(-)
```
