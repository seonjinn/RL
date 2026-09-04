# MXFP8 refit fix — branch deployment

Branch: `sna/mxfp8-twin-port-3659-20260904`
Base: PR #3659 head (mixed BF16 + MXFP8 rollout scope)
Fork: `https://github.com/seonjinn/RL.git`

## What this branch fixes

Refit into vLLM MXFP8 layers dropped the checkpoint scales when vLLM's
`initialize_model` layerwise reload snapshot fired between `create_weights`
and `process_weights_after_loading`. Refit ran without the pristine scales,
so generation silently diverged.

The fix moves the `*_from_checkpoint` twin allocation into `create_weights`
(where vLLM's snapshot captures it) and uses `install_processed_tensor` for
kernel-layout writeback so scratch-buffer aliasing cannot leak into the
Parameter storage.

Covers the "first N + last M BF16, middle MXFP8" mixed scope requested by
sharonyu-115.

## Fix commits (on top of PR #3659 head)

- `16efbdf9` Fix MXFP8 refit scales dropped by vLLM layerwise reload
- `164883a4` Add regression tests for MXFP8 scale twins across layerwise reload
- `09b36bf4` Let the processed-tensor writeback create a missing parameter
- `ceea5533` mxfp8: clone processed tensor before wrapping as Parameter

## Deploy

```bash
git fetch https://github.com/seonjinn/RL.git sna/mxfp8-twin-port-3659-20260904
git checkout FETCH_HEAD -b mxfp8-refit-3659
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
