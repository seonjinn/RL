# MXFP8 refit fix — branch deployment

Branch: `sna/mxfp8-twin-port-3630-20260904`
Base: PR #3630 head (MoE padding path)
Fork: `https://github.com/seonjinn/RL.git`

## What this branch fixes

Refit into vLLM MXFP8 layers dropped the checkpoint scales when vLLM's
`initialize_model` layerwise reload snapshot fired between `create_weights`
and `process_weights_after_loading`. On the padded MoE path this meant the
first refit worked and every subsequent refit ran with stale scales.

The fix moves the `*_from_checkpoint` twin allocation into `create_weights`
so vLLM's snapshot captures the twin, then rewrites the padded writeback
through `install_processed_tensor` so scratch-buffer aliasing cannot leak
into the Parameter storage.

## Fix commits (on top of PR #3630 head)

- `cb659af0` Fix MXFP8 refit scales dropped by vLLM layerwise reload
- `80ba56c5` test(vllm): initialize TP state for MXFP8 padding refit
- `99267337` mxfp8: clone processed tensor before wrapping as Parameter

## Deploy

```bash
git fetch https://github.com/seonjinn/RL.git sna/mxfp8-twin-port-3630-20260904
git checkout FETCH_HEAD -b mxfp8-refit-3630
```

## Test

```bash
uv run pytest tests/unit/models/generation/test_vllm_fp8_quantization.py -q
```

## End-to-end (cluster)

Uses the nightly Docker image. Requires `policy.refit_timeout_s: 300.0` in
the recipe so a hung refit fails fast instead of stalling for hours.
