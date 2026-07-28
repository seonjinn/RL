# NeMo-RL MXFP8 Adaptive vLLM Rollout Design

**Status:** Approved on 2026-07-28

## Objective

Connect the validated vLLM 0.20.2 adaptive dense MXFP8 linear path to the
latest NeMo-RL main branch and measure whether it improves real rollout
performance over the original vLLM MXFP8 dense linear path.

The primary performance metric is rollout generation output tokens per second
per GPU. The secondary metrics are generation phase duration, total step
duration, TTFT, TPOT, output-token count, and numerical/correctness signals.

## Fixed Version Contract

- NeMo-RL base: `NVIDIA-NeMo/RL` commit
  `80555d3a0595ce3cf76f2ca1b2bf123339064556`.
- Custom vLLM fork: `https://github.com/puririshi98/vllm.git`.
- Custom vLLM base ref: `nemo-speed-v0.20.2`.
- Custom vLLM base commit:
  `5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1`.
- Custom vLLM integration branch:
  `sna/mxfp8-adaptive-v0.20.2-nemorl`.
- FlashInfer: `0.6.8.post1`, including its private TRTLLM runner API.
- Initial hardware target: four GB200 GPUs with vLLM TP4.
- Initial model target: Nemotron 3 Ultra MXFP8.

Both experimental arms use the same NeMo-RL commit, custom vLLM commit,
precompiled native vLLM wheel, FlashInfer build, container, model checkpoint,
topology, scheduler configuration, prompt set, generation limits, and random
seeds.

## Custom vLLM Source Mapping

The adaptive implementation is copied from benchmark commit
`4bb11d11b2fdef33cd84b5430d4403428c07a2e1`. The following source files are
mapped into the clean custom vLLM branch:

- `vllm/model_executor/kernels/linear/mxfp8/flashinfer.py`
- `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`
- `vllm/utils/flashinfer.py`

Their expected SHA256 values before JSON-loader changes are:

- `b1de017ff41c3714712a56a7575b0f8fdbda9a05ce33b100828a4b76ed1bbd9a`
- `476defbfc9943138b06fdf920c2120ea8ebce3e327f2e42bbeb04f4c992c4015`
- `50e4b527876303c2e3b830745a6b6c1c712b1c6a54752c8509f2417adadedfdb`

The dirty native CUDA experiments in
`src_inspect/vllm-pr24-nemo-speed-v0202` are excluded. The adaptive path uses
FlashInfer's existing quantizer and direct TRTLLM runner and does not require
those experimental native symbols.

## Dense MXFP8 Runtime Policy

The optimized arm uses:

- direct FlashInfer TRTLLM dense MXFP8 GEMM;
- 8x4 activation scale-factor layout for logical `M <= 256`;
- 128x4 activation scale-factor layout for larger logical `M`;
- independent runner, workspace, and exact-physical-shape tactic maps for the
  two layouts;
- runner default tactic `-1` when a physical shape is absent from the
  qualified table;
- pre-capture runner and workspace preparation;
- fail-closed validation if runtime configuration changes after preparation.

The original arm uses the same custom vLLM build with adaptive mode disabled
and follows the original vLLM 0.20.2 MXFP8 dense linear selection.

The MoE backend is held fixed. This experiment changes only the dense linear
MXFP8 execution path.

## JSON Configuration Contract

The custom vLLM branch adds one primary runtime variable:

```text
VLLM_MXFP8_DENSE_CONFIG_FILE
```

The value names a JSON file. Absolute paths are accepted. Relative paths are
resolved only inside the custom vLLM package's fixed
`mxfp8/tactic_configs/` directory, so worker current-directory differences
cannot select another file.

The optimized manifest has this schema:

```json
{
  "schema_version": 1,
  "mode": "adaptive",
  "compatibility": {
    "vllm_version": "0.20.2",
    "vllm_base_commit": "5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1",
    "flashinfer_version": "0.6.8.post1",
    "compute_capability": "10.0",
    "gpu_family": "GB200",
    "model": "Nemotron 3 Ultra MXFP8",
    "tensor_parallel_size": 4
  },
  "policy": {
    "gemm_backend": "trtllm",
    "layout": "adaptive",
    "switch_m": 256,
    "direct_trtllm": true,
    "require_direct_trtllm": true,
    "quant_backend": "cuda",
    "require_8x4_quant": true,
    "pad_to_128": true,
    "default_tactic": -1
  },
  "tactics": {
    "8x4": [
      {"m": 1, "n": 2048, "k": 8192, "tactic": 66}
    ],
    "128x4": [
      {"m": 1000, "n": 2048, "k": 8192, "tactic": 70}
    ]
  },
  "provenance": {
    "source_manifest_sha256": "sha256-hex",
    "source_hint_sha256": "sha256-hex",
    "container_sha256": "sha256-hex",
    "qualification_repeat_count": 3,
    "minimum_cosine_similarity": 0.999,
    "minimum_speedup_vs_default": 1.02
  }
}
```

The checked-in manifest replaces each illustrative tactic entry and
`sha256-hex` value with the complete qualified data.

Validation is fail-closed for:

- missing or unsupported schema version;
- malformed or non-integer shape/tactic values;
- non-positive shape dimensions;
- duplicate `(m, n, k)` entries within a layout;
- the same physical shape appearing in both layouts;
- a non-positive or non-128-aligned `switch_m`;
- vLLM, FlashInfer, or GPU compute-capability mismatch;
- an unreadable file or changed file content after configuration freeze;
- simultaneous file-based and legacy inline tactic configuration.

Model, TP size, source hashes, and container hash remain mandatory provenance.
The launcher verifies them before submission; custom vLLM logs them during
worker startup.

Legacy inline tactic variables remain available only for compatibility and
short debugging runs:

- `VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS`
- `VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4`

The production NeMo-RL recipes use only the JSON file contract.

## NeMo-RL Dependency Integration

`tools/build-custom-vllm.sh` is updated for the vLLM 0.20 stack:

- no stale vLLM 0.16, torch 2.10/cu129, or xformers defaults;
- Git URL, exact ref, and matching precompiled wheel are explicit inputs;
- all existing `vllm` requirement forms are removed, including
  `vllm==...`, bare `vllm`, and `vllm @ ...`;
- `[tool.uv.sources].vllm` contains exactly one editable
  `3rdparty/vllm` source;
- `pyproject.toml` and `uv.lock` are regenerated together;
- `nemo-rl.env` records the exact Git ref, wheel location, and source commit.

The rollout actors continue to use NeMo-RL's dedicated
`python-VllmGenerationWorker` environment. A driver-environment `pip install`
or a `PYTHONPATH` overlay is not an accepted production path.

`vllm_cfg.env_vars` already reaches the outer NeMo-RL generation actors.
NeMo-RL is additionally changed so every configured variable name is merged
into vLLM 0.20's internal Ray-worker `ADDITIONAL_ENV_VARS` list. The merge is
deterministic, monotonic, idempotent, and protected by the existing patch
file lock. A later actor with a different environment-variable set must
extend the whitelist rather than silently retain the first actor's list.
Missing or changed vLLM 0.20 patch anchors fail loudly.

Docker custom-build arguments are quoted independently so an omitted optional
argument cannot shift the wheel URL into the Git-ref position.

## A/B Recipes

Two minimal recipe overlays inherit the same latest-main Nemotron workload:

- `original`: the JSON config variable is absent and adaptive mode is
  explicitly disabled;
- `adaptive`: `VLLM_MXFP8_DENSE_CONFIG_FILE` selects the qualified TP4
  manifest and shape tracing is enabled for the initial smoke run.

The initial integration run is rollout-only. It is followed by a short GRPO
run only after the worker-level and rollout-level gates pass.

## Validation Gates

### Gate 0: Source and Environment

Every rollout rank records:

- `sys.executable`;
- `vllm.__version__` and `vllm.__file__`;
- custom vLLM Git commit;
- PyTorch, CUDA, and FlashInfer versions;
- JSON manifest path and SHA256;
- loaded 8x4 and 128x4 entry counts.

All ranks must report the same values.

### Gate 1: CPU and Import Tests

Run the JSON parser/configuration tests and the ported adaptive contracts:

- adaptive layout policy;
- weight/B-scale shuffle contract;
- exact-shape tactic lookup;
- invalid and conflicting configuration rejection;
- configuration freeze;
- NeMo-RL custom-vLLM dependency rewrite;
- repeatable union of configured names into vLLM internal Ray-worker
  environment forwarding;
- NeMo-RL vLLM refit-loader compatibility.

### Gate 2: GPU Kernel and CUDA Graph

On GB200:

- validate B/B-scale numerical correctness, including non-128-aligned
  physical N;
- validate fixed 8x4 and 128x4 outputs;
- capture and replay mixed-shape CUDA Graph ranges;
- verify unseen shapes use runner default `-1`;
- verify all emitted rollout tokens are valid.

### Gate 3: NeMo-RL Rollout A/B

Run matched original and adaptive arms with warmup excluded and at least three
measured repetitions. Compare:

- output tokens per second per GPU;
- generation duration normalized by output-token count;
- TTFT and TPOT when available;
- prompt and output token counts;
- reward and log-probability health;
- shape coverage, layout selection, tactic hit rate, and runner-default rate.

The standalone tactic table is not accepted as final evidence until actual
NeMo-RL rollout shapes are traced. Missing or new shapes are re-shmooed three
times and qualified using the manifest thresholds.

### Gate 4: Short End-to-End GRPO

Run a short matched GRPO A/B and compare total step duration in addition to
the rollout metrics. Refit, log-probability, and training settings remain
identical. Any numerical instability or EngineDead/CUDA Graph failure blocks
the optimized configuration.

## Experiment Artifacts

All plans, launch commands, immutable provenance, job IDs, logs, raw metrics,
shape traces, shmoo inputs, qualified tables, and reports live under one
NeMo-RL experiment directory rather than the repository root. No performance
claim is based only on terminal output or an unversioned remote container.

## Non-Goals

- Upstreaming the private FlashInfer runner integration to public vLLM.
- Changing the MXFP8 MoE backend.
- Comparing BF16 against MXFP8.
- Reusing TP8, another model's, or another FlashInfer build's tactic IDs
  without requalification.
- Running a production-length RL campaign before the short A/B gates pass.
