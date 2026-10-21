# vLLM 0.29 GDN metadata reuse backport

This directory backports vLLM #58762 (v0.31.0), #58763 (v0.32.0) and #57102 (v0.30.0) to vLLM 0.29. vLLM 0.29's V2 model runner builds the Gated DeltaNet metadata for each KV cache group on every step. With the backport, it builds the metadata for the first group and copies only the state indices for the other groups with the same spec. Qwen3.5-35B-A3B with a DFlash drafter has 30 GDN groups out of 46, so the build costs about 14 ms of CPU on each step (see [kv_group_isolation](../kv_group_isolation/README.md)).

`gdn_metadata_reuse.py` changes the source of two vLLM modules when they are imported. The installed vLLM files do not change. Each patch must match exactly once, or the import fails. On other vLLM versions the hook does nothing.

Usage: put this directory first on the `PYTHONPATH` of the process that imports vLLM. Then `sitecustomize.py` installs the hook when the interpreter starts. NeMo-RL sends the driver's `PYTHONPATH` to all Ray actors. Each process that loads a patched module prints `[gdn_metadata_reuse] patched <module>` to stderr.

The backport is narrower than upstream. It reuses metadata only for GDN builders on Model Runner V2, and only without prefix caching (`mamba_cache_mode` is not `align`). Mamba1 and Mamba2 builders and align mode keep the per-group build of 0.29. On any vLLM other than 0.29.x, the hook prints `[gdn_metadata_reuse] skipped` and does nothing. Enable it only on runs where you measure it (`GDN_METADATA_REUSE=1 ../submit.sh ...`). Remove this directory when NeMo-RL moves to vLLM 0.32 or later.