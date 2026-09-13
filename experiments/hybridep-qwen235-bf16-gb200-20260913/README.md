# Qwen3-235B-A22B BF16 dispatch comparison

## Plan

Compare original GB200 performance recipes against AlltoAll-only overrides.
This extends the previous BF16 benchmark on the same source/image baseline;
it is not an exact-current-main validation. No production files are modified.

- [x] Inspect Sync16n4g, Sync32n4g, Async-1off32n4g recipes.
- [ ] Verify resolved configs differ only in dispatcher/backend.
- [ ] Commit and push; prepare isolated remote worktree and recursive submodules.
- [ ] Verify cached model and image, then scheduler test-only.
- [ ] Submit each AlltoAll baseline; HybridEP depends on its baseline success.
- [ ] Monitor first five minutes after each start; diagnose failures before retry.
- [ ] Require20completedsteps and final validation; report actualsteps2–20.

The HybridEP arm uses the corresponding original performance YAML directly.
Preserve train/gen batch, sequence length8192, BF16 precision, PP4, activation
checkpointing, generationTP8, recipe memory fraction, and all topology settings.
Do not enable PP1-only prepadding. Keep existing inactive HybridEP environment
settings in the AlltoAll arm so only dispatcher and backend differ.

| Recipe | Nodes×GPUs | Train TP/PP/CP/EP | Generation |
|---|---|---|---|
| Sync16n4g |16×4|2/4/2/16|TP8;memory0.4|
| Sync32n4g |32×4|2/4/2/16|TP8;memory0.6|
| Async-1off32n4g |32×4|4/4/1/16|16policy+16generationnodes;TP8;memory0.88|

Reuse ../hybridep-super-gb200-20260912/submit_ptyche.sh with explicit cluster,
account, image, output directory, node count, segment16 and four-hour walltime.
Common operational overrides only: maxsteps20, checkpointdisabled, unique logs
and W&B names. No MXFP8, memory tuning or source patches. AlltoAll/HybridEP arms
must share the exact immutable source and image.

Record E2E, policy training and logprob seconds and logged tokens/sec/GPU,
reward, gen_kl_error, loss, gradnorm, ratios and validation accuracy. Track each
metric's count, source/image identifiers, resolved configs and W&B links. Label
OOM runs unavailable rather than reporting startup timings as performance.
