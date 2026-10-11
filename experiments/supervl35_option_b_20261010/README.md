# SuperVL 3.5 Option B precision comparison

## Status

Source integration and HSG input inspection are complete. Three paired
performance jobs are submitted with success dependencies; measurements have
not started. Runtime build job `7862269` compiles pinned MCore helpers and
prepares the eight Gym service environments before the first performance job.
The official MARS client was built from its unmodified `broker-remote-v0.9.0`
source using the documented build workflow and connected to OCI-HSG.
OCI-AGA is not selected: the inputs are on OCI-HSG.

| Inspection job | Purpose | Outcome |
|---|---|---|
| 7861725 | CPU input inspection | Cancelled before allocation due to long queue |
| 7861750 | Model, dataset and supplied recipe | Completed, exit 0 |
| 7861761 | Recipe inheritance chain | Completed, exit 0 |
| 7861794 | TE recipe and existing image location | Completed, exit 0 |
| 7861985 | CUDA, versions and media accessibility | Completed, exit 0; 1065/1065 sampled media readable |
| 7862146 | Worker imports and service inventory | Diagnostics completed; missing helpers_cpp and eight service venvs |
| 7862269 | Build helpers and pinned-source Gym environments | Submitted; running |

| Performance job | Arms, 20 steps each | Dependency |
|---|---|---|
| 7862317 | BF16 training: BF16 / MXFP8 rollout | afterok:7862269 |
| 7862319 | MXFP8 training, BF16 parameters: Option B OFF / ON | afterok:7862317 |
| 7862323 | MXFP8 training, FP8 parameters: Option B OFF / ON | afterok:7862319 |

All performance jobs request 16 nodes x 4 GB200 GPUs, four hours, account
`nemotron_sw_post`, partition `batch`, QoS `normal`, segment 8. Each pair runs
sequentially within one allocation. The broker rejected the 32-node request
with `invalid_resource: nodes must be between 1 and 16`; it created no job.
This is a **64-GPU cohort**, not a reproduction of 128-GPU absolute throughput.
GBS 2048, microbatch 1, context lengths, precision scope and all sampling
settings are retained. Policy and generation receive eight nodes each.

Experiment source: `13d905d68fbd03d45ca17eddebffc200108cef7c`.
Launcher: `d1e3fe6193`, staged at `launcher16-d1e3fe6193` in broker workspace
`supervl35-optionb-20261010`. Image:
`/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/images/vllm029-20261006/nemo_rl_main_aligned_20261007_7776525.sqsh`.
The runtime build publishes an immutable archive plus SHA256 under the user's
`nemo-rl/supervl35-optionb-20261010/runtime` directory on Lustre. Source, Gym
venvs, C++ build, caches and temporary logs are used from node-local scratch;
durable per-node archives are copied out on exit.

The ordinary broker submission API does not expose `--test-only`. Account,
QoS and resource limits were queried before submission; the broker performs
its own admission checks. No raw Slurm fallback or split-allocation workaround
was used.

The integration branch starts from `sna/mxfp8-perf` at
`87cdfe118235cd0091bd73187190efe4369d524b` and merges
`rohit/unified-teacher-supervl3p5` at
`3fc5ea315d3f11bee73fe5b7d41cda74905443b7`.
Merge commit: `93201eed3f`. Both parent lineages are retained.
Gym changes to `c004bce8068eaa35690be8705d3e263be10581e8` on the shared
SuperVL branch. Megatron Bridge remains
`ec835530efeff71b55ec015f36bcc0c33b8b52b7`.

## Experiment matrix

| Arm | Training computation | Parameter storage | Rollout | Option B |
|---|---|---|---|---|
| bf16-bf16 | BF16 | BF16 | BF16 | OFF |
| bf16-mxfp8 | BF16 | BF16 | MXFP8 | OFF |
| mxfp8-bf16params-off | MXFP8 scoped modules | BF16 | MXFP8 | OFF |
| mxfp8-bf16params-on | MXFP8 scoped modules | BF16 | MXFP8 | ON |
| mxfp8-fp8params-off | MXFP8 scoped modules | FP8 where supported | MXFP8 | OFF |
| mxfp8-fp8params-on | MXFP8 scoped modules | FP8 where supported | MXFP8 | ON |

`arm_overrides.yaml` records precision deltas. `configs/` contains six composed
recipes from the inspected HSG inheritance chain; `configs/provenance.json`
pins the input hashes. `compose_configs.py` reproduces the composition from
those input snapshots. The remote base recipe matches the merged branch's
starter byte for byte. Runtime preparation gates the submitted performance jobs.

All arms retain R3 token capture and BF16 LM heads. MXFP8 scope is routed
experts only: decoder layer 0, layers 80-87, dense projections, shared experts,
MTP and multimodal modules remain BF16. The TE matchers explicitly inherit
global parameter-storage precision, so `fp8_param` distinguishes the two
storage pairs. BF16 training arms clear the TE precision recipe; BF16 rollout
also clears MXFP8 ignore patterns. The control token is supplied at runtime.

Option B is the existing training optimization: grouped-tensor MoE,
Transformer Engine op fuser, `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`, and
`CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL=1`. Its OFF counterpart explicitly resets
these flags. Both MXFP8 sides retain grouped GEMM and fused weighted squared
ReLU. `NRL_MXFP8_DIRECT_SCALE_REFIT` is a separate refit experiment and is not
part of this matrix.

## Workload and measurement

Start from
`configs/examples/recipes/vlm/vlm_grpo-supervl3p5-unified-teachers-ready-first-32n4g.yaml`
and the supplied HSG precision profile. Preserve media, task routing, rewards,
sampling and topology across the matrix. The starter uses:

- 32 nodes x 4 GPUs, 16 policy nodes and 16 generation nodes;
- policy TP2 / EP16 / CP2, generation TP4 / EP4;
- 128 prompts x 16 responses = GBS 2048, seed 42;
- 65,536 total context, 32,768 response limit, 64 video frames;
- SingleController, ready-first sampler, maximum staleness 2.

Use the same merged source, container, Gym/Bridge/MCore revisions, model and
dataset manifest across all arms. Disable checkpoints and validation for the
20-step performance trials, and use distinct output/cache destinations.
Report steps 2-20 with actual valid sample/token counts, output lengths,
rewards, raw/filtered mismatch metrics, rejection counts and finite gradients.
These short runs do not establish convergence equivalence.

Generation reporting must account for ready-first and heterogeneous Gym tasks.
SingleController already provides
`rollout/throughput/generation_output_tokens_per_second` and
`rollout/throughput/committed_output_tokens_per_second`; use their elapsed
intervals and reject counter-discontinuity intervals. Report group rollout
latency separately from training `timing/train/total_step_time`, policy,
logprob and refit time. Do not reuse the legacy full-batch Async duration
formula without checking its semantics for this controller.

## Remaining execution steps

1. Verify runtime dependencies, media references and quantization exclusions
   on workers through the official broker. No raw-SSH fallback is used.
2. Qualify the existing `nemo_rl_main_aligned_20261007_7776525.sqsh` image
   against the merged SuperVL source, including Lens, Gym, MCore helpers and
   video dependencies. Its previous text-model qualification does not
   establish SuperVL compatibility.
3. Confirm runtime preparation exits successfully, then monitor the first
   performance allocation for at least five minutes after startup. Full-model
   memory fit, quantization scope and repeated refit remain execution checks.
4. Collect all six completed 20-step arms and compare Option B OFF/ON
   separately for each parameter-storage setting. Never report queued jobs
   or successful imports as measured performance.

## Integration checks

- Merge was automatic with no conflicts; 20 merged Python files parsed.
- Fixed BF16 boundary-name corruption for
  `language_model.backbone.layers.*`: preserve the HF prefix so the existing
  vLLM mapper can convert it to `language_model.model.layers.*`.
- New CPU name-conversion regression failed before the fix and all six cases
  passed afterward. This does not replace an installed-vLLM scope check.
- Text-only Super ignore globs need the explicit VL prefix; do not copy
  `model.layers.*` wildcard exclusions unchanged into SuperVL.
- Ruff checks and formatting checks pass for the boundary-name fix and test.
- YAML validation confirms six arms and exactly four Option B differences
  within each parameter-storage pair.
- The existing `test_multimodal_image_encoding.py` could not collect in the
  local macOS environment. With the repository's `transformers==5.12.1` pin
  and Lens dependency, collection stopped at missing `zmq`. No multimedia
  test passed in that attempt. Run it in the pinned experiment container;
  the merged runtime is not yet qualified.
