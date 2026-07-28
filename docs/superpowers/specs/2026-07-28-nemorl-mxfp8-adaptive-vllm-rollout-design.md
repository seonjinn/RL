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
- Primary hardware target: four OCI-HSG GB200 nodes with four GPUs per node.
  OCI-HSG is reachable from the current development session and accepts the
  4n4g scheduling request. Pre-Tyche is the same-hardware reproduction target
  after its Kerberos credential is refreshed.
- Primary workload: the latest-main Qwen3-30B-A3B 4n4g GRPO recipe, with
  MXFP8 rollout enabled and vLLM tensor parallel size 1.
- Secondary efficacy workload: Nemotron 3 Ultra MXFP8 with vLLM tensor
  parallel size 4, used only if the Qwen trace has no eligible dense MXFP8
  calls or after the Qwen integration gate passes.

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
    "model": "Qwen/Qwen3-30B-A3B",
    "tensor_parallel_size": 1
  },
  "policy": {
    "gemm_backend": "trtllm",
    "layout": "adaptive",
    "switch_m": 256,
    "direct_trtllm": true,
    "require_direct_trtllm": true,
    "quant_backend": "cuda",
    "require_8x4_quant": true,
    "pad_to_128": false,
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
`sha256-hex` value with Qwen TP1 data qualified from the actual 4n4g rollout
trace. A Qwen table is never initialized from the Nemotron Ultra TP4 seed.

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

The custom vLLM fork owns a separate offline qualification CLI. Runtime only
reads an immutable JSON manifest; it never benchmarks, promotes, or rewrites
tactics. The offline path:

1. aggregates exact physical `(layout, M, N, K)` records from a real rollout;
2. aborts if the trace contains zero eligible dense MXFP8 calls;
3. shmoos runner default `-1` and valid tactics on the same GPU, container,
   model topology, and layout;
4. requires BF16/reference correctness and at least three repeats;
5. promotes only tactics with at least 1.02 median speedup over default; and
6. deterministically regenerates and validates the runtime JSON.

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
Qwen rollout TP1 does not create vLLM internal Ray workers. For a later TP>1
run, vLLM 0.20.2's native `ray_env.get_env_vars_to_copy()` copies every
`VLLM_*` variable before internal worker initialization, so
`VLLM_MXFP8_DENSE_CONFIG_FILE` requires no NeMo source rewrite. The obsolete
`ADDITIONAL_ENV_VARS` assignment patch remains optional and must not become a
required vLLM 0.20.2 anchor. Arbitrary non-`VLLM_*` variables, if ever needed,
use vLLM's supported `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` contract.

Docker custom-build arguments are quoted independently so an omitted optional
argument cannot shift the wheel URL into the Git-ref position.

## Qwen3-30B-A3B A/B Recipe

Two minimal recipe overlays inherit the latest-main
`grpo-qwen3-30ba3b-4n4g.yaml` workload and the rollout precision/ignored-layer
settings from its MXFP8 sibling:

- `original`: the JSON config variable is absent and adaptive mode is
  explicitly disabled;
- `adaptive`: `VLLM_MXFP8_DENSE_CONFIG_FILE` selects the qualified Qwen TP1
  manifest and shape tracing is enabled for the initial smoke run.

Both arms remain MXFP8; this is not a BF16-versus-MXFP8 comparison. The initial
integration run is rollout-only. It is followed by a short GRPO run only after
the worker-level and rollout-level gates pass.

Qwen3-30B-A3B is a MoE model, and the current MXFP8 recipe ignores the q/k/v/o
dense projections. Its trace may therefore contain zero eligible dense MXFP8
linear calls. That result is a valid integration finding but not a kernel
performance result: the pipeline reports `not-applicable`, does not create an
empty optimized table, and moves the efficacy measurement to the Ultra TP4
workload.

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
- exact JSON-key forwarding into every NeMo generation actor and vLLM 0.20's
  native `VLLM_*` internal-worker copy contract;
- NeMo-RL vLLM refit-loader compatibility.

### Gate 2: Qwen Trace Applicability

Run the exact Qwen 4n4g MXFP8 rollout on OCI-HSG with adaptive shape tracing
and default tactic `-1`.

- If at least one eligible dense shape is observed, offline-shmoo every unique
  physical shape and build a Qwen TP1 manifest.
- If zero eligible shapes are observed, fail the promotion stage clearly,
  record the Qwen result as `not-applicable`, and continue with the Ultra TP4
  efficacy workload.

### Gate 3: GPU Kernel and CUDA Graph

On GB200:

- validate B/B-scale numerical correctness, including non-128-aligned
  physical N;
- validate fixed 8x4 and 128x4 outputs;
- capture and replay mixed-shape CUDA Graph ranges;
- verify unseen shapes use runner default `-1`;
- verify all emitted rollout tokens are valid.

### Gate 4: NeMo-RL Rollout A/B

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

### Gate 5: Short End-to-End GRPO

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
- Reusing TP4/TP8, another model's, or another FlashInfer build's tactic IDs
  without requalification.
- Running a production-length RL campaign before the short A/B gates pass.
