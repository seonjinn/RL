# Timeline

## 2026-08-05 03:09 PDT

- User requested the cumulative PR 3477/3478 step-time effect and asked whether
  BF16 training with NVFP4 rollout had been tested.
- Verified that the implementation and smoke matrix exist but no GPU job has
  been submitted for this exact precision combination.
- Selected GCP-NRT B200 for the runtime validation.

## 2026-08-05 03:13 PDT

- `bash -n` and both smoke scripts' static dry-run contracts passed.
- Local pytest stopped before collection because the Gym workspace submodule is
  not initialized in the local worktree. Deferred the focused pytest to the
  recursive-submodule GCP clone and target container.

## 2026-08-05 03:19 PDT

- Cloned commit `441dc40df` recursively to GCP-NRT.
- Added and syntax-checked `submit_gcp_nrt.sh` with unique code snapshots,
  commit-versioned caches, `--gpus-per-node=8`, and `sbatch --test-only`.
- The submitted command runs `tests/test_nvfp4_rollout_recipes.py` in the target
  container before starting the GPU smoke.

## 2026-08-05 04:00 PDT

- Submitted W4A16 legacy job `497394` and NCCL-Reshard job `497395` on GCP-NRT.
- Both target-container recipe gates passed (`4 passed`) and both jobs failed
  during vLLM actor initialization before step 1.
- Root cause: the Ray actor could not resolve the repo-relative
  `examples/modelopt/quant_configs/nvfp4_experts_weightonly.yaml` path.
- Added project-relative quant-config normalization before real-NVFP4 mode
  resolution, with focused tests for Ray-like non-repository working directories.

## 2026-08-05 04:06 PDT

- W4A16 retry jobs `497435` and `497436` passed all six container gates and
  resolved the custom quantization recipe correctly.
- Both then failed vLLM backend validation because the BF16 base recipe's
  `moe_backend: triton` was inherited by the real-NVFP4 overlay.
- Pinned W4A16 to Marlin and W4A4 to FlashInfer TRT-LLM, and extended the
  recipe contract to reject accidental backend inheritance.

## 2026-08-05 04:16 PDT

- The first backend-pinned retry stopped at the expanded recipe contract: its
  parametrized tuple omitted the newly added backend value.
- Corrected the test tuple before model initialization; no training result was
  produced by this retry.
