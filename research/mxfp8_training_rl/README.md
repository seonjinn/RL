# MXFP8 training in RL

This experiment validates MXFP8 training compute with BF16 parameter storage
(`fp8_param: false`) and MXFP8 vLLM rollout. It covers Qwen3-30B-A3B and
Nemotron-3 Nano on GB200.

Run a two-step smoke test first:

```bash
MODEL=qwen30 MAX_STEPS=2 ACTION=test-only ./submit_oci_hsg.sh
MODEL=qwen30 MAX_STEPS=2 ACTION=submit ./submit_oci_hsg.sh

MODEL=nano MAX_STEPS=2 ACTION=test-only ./submit_oci_hsg.sh
MODEL=nano MAX_STEPS=2 ACTION=submit ./submit_oci_hsg.sh
```

After both jobs complete, use `MAX_STEPS=20` for performance measurements. The
steady-state summary should average steps 2 through 19.

The launcher requires `REPO`, `CONTAINER`, `HF_HOME`, `WANDB_HOME`,
`RESULT_ROOT`, and `SLURM_ACCOUNT`. It stores source in `/home`, worker virtual
environments and JIT caches in `/raid/scratch`, and durable logs in `/lustre`.

## Per-module Transformer Engine precision

NeMo-RL loads a Megatron per-module recipe from
`policy.megatron_cfg.te_precision_config_file`. Megatron matches the full module
path against the YAML matchers in order; the first match wins.

For the common case where training uses MXFP8 by default and only selected
layers remain in BF16, keep `fp8_cfg.enabled: true` with
`fp8_recipe: mxfp8`, then point `te_precision_config_file` at a recipe that only
lists the BF16 exceptions:

```yaml
configs:
  bf16:
    transformer_engine_config_type: TEQuantizationParams
    training_recipe: {}

matchers:
  first_two_layers:
    config: bf16
    type: glob
    pattern: "*layers.[01].*"
    enabled: true
  final_layer:
    config: bf16
    type: glob
    pattern: "*layers.47.*"
    enabled: true
```

Matched TE GEMMs open `fp8_autocast(enabled=False)` and therefore execute in
BF16. Unmatched modules inherit the global MXFP8 context.

For the inverse case, BF16 by default with only selected TE GEMMs in MXFP8,
disable global `fp8_cfg` and explicitly enable quantization from the recipe:

```yaml
configs:
  mxfp8:
    transformer_engine_config_type: TEQuantizationParams
    training_recipe:
      fp8_quantization_recipe: mxfp8
      override_nonquantized_autocast: true

matchers:
  routed_expert_fc1:
    config: mxfp8
    type: glob
    pattern: "*.mlp.experts.linear_fc1"
    enabled: true
  routed_expert_fc2:
    config: mxfp8
    type: glob
    pattern: "*.mlp.experts.linear_fc2"
    enabled: true
```

This recipe controls TE compute precision. It does not enable MXFP8 parameter
storage. `fp8_param: true` changes parameter and all-gather storage and requires
a separate refit path for native MXFP8 data and E8M0 scales.
