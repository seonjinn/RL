# MXFP8 refit fix — branch deployment

Branch: `sna/mxfp8-twin-port-3294-20260904`
Base: PR #3294 head (grouped MoE MXFP8)
Fork: `https://github.com/seonjinn/RL.git`

## What this branch fixes

Refit into vLLM MXFP8 layers dropped the checkpoint scales when vLLM's
`initialize_model` layerwise reload snapshot fired between `create_weights`
and `process_weights_after_loading`. On the grouped MoE path this dropped
both linear and expert scale twins after the first refit.

The fix moves the `*_from_checkpoint` twin allocation for both the linear
and MoE methods into `create_weights` so vLLM's snapshot captures them, and
routes every kernel-layout writeback through `install_processed_tensor` so
scratch-buffer aliasing cannot leak into the Parameter storage. Collapses
the `first_load` branch on both PWAL paths onto a single `seed_checkpoint_scales`.

## Fix commits (on top of PR #3294 head)

- `546d3460` test(mxfp8): use current refit loader contract
- `db15416d` test(mxfp8): update remaining loader fixtures
- `2be9d25f` mxfp8: allocate refit twins in create_weights (#3294 port)

## Deploy

```bash
git fetch https://github.com/seonjinn/RL.git sna/mxfp8-twin-port-3294-20260904
git checkout FETCH_HEAD -b mxfp8-refit-3294
```

## Test

```bash
uv run pytest tests/unit/models/generation/test_vllm_fp8_quantization.py -q
```

Regression tests to look for:
- `test_mxfp8_moe_checkpoint_scales_survive_layerwise_reload`
- `test_mxfp8_linear_checkpoint_scale_survives_layerwise_reload`
- `test_apply_fp8_patches_registers_modelopt_patches_only_for_mxfp8`

## End-to-end (cluster)

Uses the nightly Docker image. Requires `policy.refit_timeout_s: 300.0` in
the recipe so a hung refit fails fast instead of stalling for hours.
