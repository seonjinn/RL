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

## Exact frozen MCore runtime overlay

| Field | Value |
| --- | --- |
| Scheduler preflight | `2456308` accepted one exclusive Ptyche GB200 node (test-only) |
| Probe job | `2456309` (`FAILED`, exit `1:0`, 6m41s) |
| Fresh source checkout | `69de6bd55b86b57173e4db21a9586be869a1e642` |
| Fresh source path | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/RL-latestmain-nanov3-uv-overlay-mcore-20260727-69de6bd55-r1` |
| Recursive workspaces | Gym present; Bridge `8cf2311e75e103861101238ea091af03d17efc03`; nested MCore `53f5161ce000b5320bc16cb260949c2e6808da83` |
| Probe log | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/RL-latestmain-nanov3-uv-overlay-mcore-20260727-69de6bd55-r1/exp_logs/cuda_graph_uv_overlay_mcore_probe/ptyche_2456309.out` |
| Exact launcher | `NRL_FORCE_REBUILD_VENVS=true uv run --extra mcore --frozen python -c ...` |
| Lock integrity | `uv lock --check` reported that the lock needs an update; nevertheless `uv.lock` stayed clean and equal to `HEAD:uv.lock` (`30a35a07db7a646a7e0fb4e458daf264cf6c805a`, SHA256 `58535d4aab618394a6926d6ad93da4b3ca01d95bee8d36b1a88635aa9ea36e66`); no source `.venv` was created |
| MCore locked Transformers | `5.8.1`, selected by the MCore marker `>=5.8.1,<5.9.0` |
| Failure | Editable Megatron-Bridge build rejected its own dependency cache before Transformer Engine/MCore imports: Bridge `pyproject.toml` declares `transformers>=5.8,<=5.12.1`, but the NeMo-RL workspace wrapper cache declares `transformers>=5.8.1,<5.9.0` |

The two declarations have independent provenance. The Bridge `pyproject.toml`
range was introduced by Bridge commit `0e7ffb79c8c36b16cc246d9e262c2e2d06b76105`
(`build(deps): upgrade Transformers to 5.12.1`, 2026-07-20), which is an
ancestor of the pinned Bridge checkout. The wrapper cache range was introduced
by NeMo-RL commit `db8a0ef4304e449e8629201602cf4f8fdc531582`
(`build: Update deps for Megatron Inference`, 2026-06-04). This is a source
packaging/lock coherence failure, not a CUDA Graph or model-runtime result.
No OCI staging or model job was submitted.

## Corrected Bridge exact MCore overlay

| Field | Value |
| --- | --- |
| Scheduler preflight | `2456757` accepted one exclusive Ptyche GB200 node (test-only) |
| Probe job | `2456759` (`TIMEOUT`, scheduler limit, 20m13s) |
| Fresh source checkout | `51727413636105f0b1a3ff8a6178b68b34b0dd02`, a probe-contract-only child of reviewed NeMo source `e4880ea4f7e6b2b805644fccc1f7434107b5beaf` |
| Fresh source path | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/RL-latestmain-nanov3-uv-overlay-mcore-20260727-51727413` |
| Recursive workspaces | Gym present; corrected Bridge `59c163cce9cb8cc209dcd0424b2b9de9d1be5027`; nested MCore `53f5161ce000b5320bc16cb260949c2e6808da83` |
| Probe log | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/RL-latestmain-nanov3-uv-overlay-mcore-20260727-51727413/exp_logs/cuda_graph_uv_overlay_mcore_probe/ptyche_2456759.out` |
| Exact launcher | `NRL_FORCE_REBUILD_VENVS=true uv run --extra mcore --frozen python -c ...` importing Transformer Engine, MCore, Mamba, and NeMo-RL |
| Successful stages | The corrected editable Bridge build, `megatron-core`, `nemo-rl`, DeepEP, Mamba, causal-conv1d, and the other non-TE native dependencies built successfully |
| Terminal blocker | Transformer Engine `42b840051647eef89761a16dfdff87e82bb253ab` was still building when Slurm cancelled the job at the script's 20-minute limit; imports never ran and `uv_overlay_imports=passed` was not emitted |
| Lock integrity | `uv lock --check` still reported that the tracked lock needs an update, while frozen `uv.lock` remained clean and equal to `HEAD:uv.lock` (`30a35a07db7a646a7e0fb4e458daf264cf6c805a`, SHA256 `58535d4aab618394a6926d6ad93da4b3ca01d95bee8d36b1a88635aa9ea36e66`); no source `.venv` was created |

This timeout supersedes the earlier Bridge metadata failure as the current
environment gate. It establishes that the Bridge correction resolves editable
package construction, but does not establish successful Transformer Engine or
Mamba import. No OCI staging, model smoke, CUDA Graph matrix, performance, or
accuracy job was submitted after this terminal result.

## Verified offline NanoV3 snapshot and baseline

| Field | Value |
| --- | --- |
| HF snapshot root | `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` |
| Revision | `97ab8012882a655dc38df4fee47422aca9caca07` |
| Snapshot size | 59 GiB |
| Integrity | 13 non-empty safetensors shards; zero broken symlinks; zero incomplete markers |
| Required metadata | `config.json`, tokenizer files, and `model.safetensors.index.json` present |
| Initial actual baseline | `2457754` (`FAILED` before model creation because W&B had no API key) |
| Offline W&B retry | `2457775` (`FAILED` at GRPO step 0 because vLLM requested Hub metadata and received HTTP 429) |
| Local-path/offline source | `f7addaa2be16e88c0f45ad7de08b4f2aba688f1c` |
| Fresh source path | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/RL-latestmain-nanov3-offline-snapshot-20260727-220a05a67` |
| Recursive pins | Bridge `59c163cce9cb8cc209dcd0424b2b9de9d1be5027`; MCore `53f5161ce000b5320bc16cb260949c2e6808da83` |
| Scheduler preflight | `2458269` accepted four Ptyche nodes in `backfill` |
| Current 20-step baseline | `2458270` (`PENDING`, reason `Priority`, scheduler estimate 2026-07-28 03:55 PDT) |
| Checkpoint policy | Disabled |

The current launcher passes the verified snapshot directory as both
`policy.model_name` and `policy.tokenizer.name`, and propagates
`HF_HUB_OFFLINE=1` plus `TRANSFORMERS_OFFLINE=1` to the Slurm/Ray job. This
removes the Hub metadata request that stopped `2457775`; performance and
accuracy fields remain empty until `2458270` reaches GRPO steps.
