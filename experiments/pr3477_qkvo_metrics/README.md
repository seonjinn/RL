# PR 3477 QKVO MXFP8 metric ranges

This experiment compares routed-expert-only MXFP8 with routed experts plus
QKVO MXFP8 for Qwen3-30B-A3B and Nemotron3 Nano. Both arms use BF16 training,
NCCL Reshard refit, FlashInfer TRTLLM MoE, CUDA Graphs, the same seed, and 20
steps on GB200.

`audit_scope.py` runs before model startup. It requires QKVO and routed experts
to match the selected arm while keeping routers, `lm_head`, embeddings, Mamba
projections, shared experts, and speculative heads outside the added scope.

Final reporting uses the observed minimum and maximum over completed W&B steps
3 through 20. It reports no mean as the primary result. The short run checks
functional stability and metric ranges; it does not establish long-run
convergence parity.

Generate the final range-only report with:

```bash
uv run --with wandb python experiments/pr3477_qkvo_metrics/report_ranges.py \
  <wandb-run-url> [<wandb-run-url> ...]
```
