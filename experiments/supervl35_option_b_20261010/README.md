# SuperVL 3.5 Option B precision comparison

## Status

Source integration is complete; GPU jobs have **not** been submitted.
The official MARS client is absent from this workstation. HSG paths supplied
by the user still need remote validation. OCI-AGA is not selected: the user
confirmed that the inputs are on OCI-HSG.

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

`arm_overrides.yaml` records precision deltas. It is not a launch-ready
recipe: the exact supplied `mxfp8-last8bf16-r3-bf16lmhead` profile must first
be read, copied read-only into this experiment, checked and pinned. Do not
infer its R3 or exclusion settings solely from its filename. The starter
ready-first recipe currently has R3 disabled and FP32 LM heads, which differ
from the separate profile's name.

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

1. Establish official MARS client access to OCI-HSG; verify account, partition,
   QoS and resource eligibility. No raw-SSH fallback is configured here.
2. Inspect the supplied model/config/dataset read-only. Check media references
   from bounded samples and confirm they resolve on every worker.
3. Read the exact shared precision/R3 profile and compose six resolved recipes.
   Verify quantization exclusions against actual HF and vLLM module names.
4. Pin a compatible container. The shared guide names a September 30 image;
   the previous text-model experiment used October 7/vLLM 0.29. Neither image
   has yet been validated for this merged SuperVL job. Inspect software,
   source patches, Lens, Gym, MCore helpers and video dependencies first.
5. Validate imports, quantization scope, repeated refit and a short full-shape
   smoke. Then submit 20-step arms, with a scheduling preview and five-minute
   startup monitoring. Share exact resources and commands before submission.
6. Preserve paired node allocations where practical and compare Option B
   OFF/ON separately for each parameter-storage setting.

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
