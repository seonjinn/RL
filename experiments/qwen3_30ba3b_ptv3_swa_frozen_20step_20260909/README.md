# Qwen3-30B-A3B PTV3-SWA frozen 44K evaluation

This experiment evaluates the PTV3-SWE SWA drafter exports at checkpoint
44,000 against a matched no-SpecDec baseline.

- Math uses the official Qwen3-30B-A3B 4-node × 4-GPU performance recipe for
  20 optimizer steps.
- All stable arms are frozen: policy draft training and draft refit are off.
- DFlash and DSpark run at K3, K5, and K7 with FULL_AND_PIECEWISE CUDA Graphs.
- The Qwen3-30B-A3B Thinking DSpark checkpoint is excluded from Math because
  target and drafter variants must match.
- DFlash2 K7 remains recorded but blocked until the NeMo-RL vLLM >=0.28 runtime
  cohort is available. It will require its own matched baseline.

The stable matrix is `baseline`, `dflash_k3`, `dflash_k5`, `dflash_k7`,
`dspark_k3`, `dspark_k5`, and `dspark_k7`.
