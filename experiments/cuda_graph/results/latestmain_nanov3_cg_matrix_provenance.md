# Latest-main NanoV3 CUDA Graph matrix provenance

## Ptyche nightly staging

| Field | Value |
| --- | --- |
| Cluster | Ptyche |
| Staging job | `2456107` (`COMPLETED`, exit `0:0`, 26m31s) |
| Source image index digest | `sha256:c97efad5a565da596ee206072f49aa35fc632a4fca453ddff7049451a641b423` |
| ARM64 manifest digest | `sha256:67ad116cb0a969ad2644869a4d0e2e3c5d7a859588dd1789dc25732ef3700dba` |
| Imported image reference | `nvcr.io/nvidian/nemo-rl@sha256:67ad116cb0a969ad2644869a4d0e2e3c5d7a859588dd1789dc25732ef3700dba` |
| Experiment source SHA | `24996e441d7095638f4bd8addf8f6d72aeb3c434` |
| Immutable artifact | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260727_2456107.sqsh` |
| Stable link | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly.sqsh` |
| Artifact SHA256 | `5781f94d8d7224957bb1c5d0d5a230c04042ac40545f086c9297e12d8c77b64b` |
| Metadata | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260727_2456107.sqsh.metadata.txt` |
| Staging log | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/stage_nemo_rl_nightly_2456107.log` |
| Integrity check | Metadata SHA256 equals an independently recomputed SHA256 |
| Runtime probe job | `2456229` (`FAILED`, exit `1:0`, 1m02s) |
| Runtime probe log | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/RL-latestmain-nanov3-stage-20260727-8663093c/exp_logs/cuda_graph_runtime_probe/ptyche_2456229.out` |
| Runtime gate result | Python `3.13.13` and four GPUs passed; `transformer_engine` import failed with `ModuleNotFoundError` |

The required Transformer Engine gate failed before `megatron.core`, `mamba_ssm`,
the current config parse, or any model job could run. No NanoV3 CUDA Graph
model smoke was submitted. OCI-HSG staging and all performance/accuracy work
remain held because the same ARM64 image digest cannot satisfy this gate.

## Forced uv runtime overlay

| Field | Value |
| --- | --- |
| Probe job | `2456255` (`FAILED`, exit `1:0`, 2m23s) |
| Fresh source checkout | `0d37b777f4d9188ceaa5164a3244976292fe566a` |
| Probe log | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/RL-latestmain-nanov3-uv-overlay-20260727-0d37b777/exp_logs/cuda_graph_uv_overlay_probe/ptyche_2456255.out` |
| Tracked lock check | Failed: `nemo_gym` references a workspace but the fresh checkout had not initialized the Gym submodule |
| Frozen source integrity | `uv.lock` remained `30a35a07db7a646a7e0fb4e458daf264cf6c805a`, identical to `HEAD:uv.lock`; no source `.venv` was created |
| Frozen overlay import | Failed: `ModuleNotFoundError: No module named 'transformer_engine'` |

`transformer-engine`, `megatron-core`, and `mamba-ssm` are declared by the
optional `mcore` extra. The tested `NRL_FORCE_REBUILD_VENVS=true uv run`
command did not select that extra, so rebuilding its default environment could
not supply Transformer Engine. This confirms the environment gate without
rewriting or pushing the tracked lock.

The 2026-07-15 image and the Megatron-LM CI image are not valid baselines for
this matrix and are not used here.
