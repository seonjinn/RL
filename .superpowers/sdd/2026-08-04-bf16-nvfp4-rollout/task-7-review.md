# Task 7 Independent Review

- **nemo_rl/models/generation/vllm/vllm_backend.py:1445** - warning - Real-NVFP4 NCCL refit performs two consecutive device-wide completion fences: `VllmQuantInternalWorkerExtension` already synchronizes in its finalizer at `vllm_quant_backend.py:818`, then the shared NCCL caller synchronizes again here without scheduling work between them. This violates the requested single-finalize/single-fence lifecycle and adds an avoidable full-device stall to every NVFP4 refit. Centralize collective completion fencing so the real-quant path has exactly one owner while identity/MXFP8 retain their required fence.

- **tests/unit/models/generation/test_vllm_generation.py:185** - nit - The new handshake tests cover duplicate unioning and target-format preservation, but none exercises the conflict branch in `merge_refit_transform_requests()`. Add a regression where the same parameter is returned as legacy MXFP8 and typed NVFP4 (and/or W4A16 versus W4A4) and assert that the outer generation boundary raises, since rejecting such disagreement is the safety property that prevents source/destination format corruption.

Validation performed:

- `test_refit_transforms.py` and `test_nccl_reshard_utils.py`: 93 passed.
- Ruff check on all changed production Python files: passed.
- `git diff --check` for the reviewed commit range: passed.
- Real vLLM/ModelOpt/CUDA/NCCL execution was unavailable on this macOS host.

Summary: 0 critical, 1 warning, 1 nit.
