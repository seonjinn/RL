# SuperVL 3.5 Option B precision comparison

## vLLM-only rollout A/B

`build_rollout_ab_configs.py` derives six BF16-training rollout recipes from
`configs/bf16-bf16.yaml` and `configs/bf16-mxfp8.yaml`: BF16, routed MoE-only
MXFP8, MoE plus attention QKV/O, MoE plus Mamba in/out, MoE plus both, and an
exploratory broad decoder-linear MXFP8 scope. The last arm also quantizes the first/last
decoder layers and is not the production mixed-precision recipe. Vision,
sound, MTP, LM head and MoE router remain unquantized in every MXFP8 arm.
Mamba conv/SSM kernels are not MXFP8 linear projections.

`moe_attention` and `moe_mamba` isolate the contributions of the two
projection families. They are follow-up arms; the original four-arm result
remains a separate comparison.

All six arms set `moe_backend=flashinfer_trtllm` and
`kernel_config.linear_backend=flashinfer_cutedsl` on vLLM 0.29. The latter
selects linear GEMMs; vLLM 0.29 does not offer `flashinfer_cutedsl` for MXFP8
MoE. The benchmark runs one TP4/EP4 replica on four GB200 GPUs and replays
identical synthetic text token IDs and fixed output lengths. It excludes model
initialization, training, Gym and refit time. It uses `load_format=auto` from
the supplied HF checkpoint rather than the RL recipe's `load_format=dummy`
plus NCCL refit. Consequently these results isolate a vLLM text-generation
path, not the complete video-teacher RL rollout. Confirm loaded weight dtypes,
kernel selection and numerical sanity before treating performance as valid;
actual video-request and post-refit results require a separate matched trace.

Generate the six configs from this worktree with:

```bash
python3 experiments/supervl35_option_b_20261010/build_rollout_ab_configs.py \
  --bf16 experiments/supervl35_option_b_20261010/configs/bf16-bf16.yaml \
  --moe experiments/supervl35_option_b_20261010/configs/bf16-mxfp8.yaml \
  --output-dir /path/to/vllm-ab-stage/configs
```

Stage `bench_vllm_generation.py` and `run_vllm_ab.sh` next to that `configs/`
directory. The launcher reuses the prepared source runtime for commit
`15045cb9425d`, stages it once on node-local scratch and writes one compact
JSON per arm under the experiment's Lustre `vllm_ab/<job-id>/` directory. It
aborts if any output has fewer tokens than requested. Use the same image,
model checkpoint, GPU topology and prompt hash for every arm; do not mix these
standalone numbers with the 32-node ready-first `exposed_generation` timer.
The pinned image keeps vLLM in the async generation worker venv under
`/opt/ray_venvs`, not the base `/opt/nemo_rl_venv`; the launcher checks the
worker interpreter before loading the model.

Set `BENCHMARK_KIND=concurrency` to run the separate one-node request sweep
instead. It keeps the RL generation replica's TP4/PP1/EP4 and
`max_num_seqs=128`, uses 512-token prompts and 128-token outputs, and submits
1, 2, 4, 8, 16, 32, 128, 256, 512, and 1024 requests in each trial. Results
go to `vllm_concurrency/<job-id>/`. Values above 128 represent outstanding
client requests, not 256-1024 simultaneously active GPU sequences; vLLM queues
them behind the recipe's `max_num_seqs` cap. The single replica does not
replicate the RL recipe's 16 generation nodes, so report its throughput per
four-GPU replica rather than total RL rollout throughput.

For the requested length matrix, use `BENCHMARK_KIND=length_sweep` and one of
`BENCH_ISL/BENCH_OSL=1024/10240`, `10240/1024`, or `1024/1024`. All three use
the same 1-1024 outstanding-request sweep and keep `max_num_seqs=128`. Values
up to 128 have two trials; 256-1024 have one trial to bound the 10K-output
case. `BENCH_ARMS` optionally selects comma-separated arms; split the
1024/10240 matrix into one job per arm under the four-hour batch limit. The
results are in `vllm_length_sweep/isl<tokens>_osl<tokens>/<job-id>/`.

For a bounded worker-side CPU/CUDA trace, set `BENCHMARK_KIND=profile`, one
`BENCH_ARMS` value (`bf16` or `moe`), `BENCH_ISL/BENCH_OSL` to `1024/10240`
or `10240/1024`, and `BENCH_CONCURRENCY` to `16` or `128`. The profiler begins
after model warm-up and captures at most 32 worker iterations; it starts
immediately for the prefill-heavy case and after 64 iterations for the
decode-heavy case. Override those limits with `BENCH_PROFILE_DELAY_ITERATIONS`
and `BENCH_PROFILE_MAX_ITERATIONS`. The full requested output length is still
generated and checked. Each job writes a single `trace.tar`, its SHA-256,
`driver.log`, provenance, and result JSON under
`vllm_profile/isl<ISL>_osl<OSL>/c<concurrency>/<job-id>/`. Profiled elapsed
time is diagnostic; use unprofiled length-sweep JSONs for throughput claims.

For a four-GPU TP2/DP2 preflight, stage this experiment directory under
`<workspace>/experiments/supervl35_option_b_20261010` and set
`BENCHMARK_KIND=dp2_preflight BENCH_ARMS=bf16,moe` when running
`run_vllm_ab.sh`. The two offline vLLM DP ranks each use TP2 and participate
in a joint EP4 group. The first pass submits 2, 4, and 8 identical 1K-input,
128-output requests, with one trial each. The benchmark requires both ranks
to load and finish with exact output lengths before writing an aggregate
result under `vllm_dp2_preflight/<job-id>/`. It uses 64 active sequences per
rank, preserving the TP4 reference's 128-sequence total cap across four GPUs;
per-rank batched-token cap is also halved from 65536 to 32768. This is a
compatibility/memory check, not a throughput conclusion. The full length sweep
must be run separately after BF16 and MXFP8 both pass. A single outstanding
request is omitted because offline MoE DP needs work on both ranks.
Use `BENCHMARK_KIND=tp4_preflight` with the same arms for a matching TP4/DP1
reference; it writes to `vllm_tp4_preflight/<job-id>/`. Do not compare the
TP2 preflight with the earlier 1K/1K sweep, which used 1024 output tokens.

For the TP2/DP2 length matrix, use `BENCHMARK_KIND=dp2_length_sweep` with
`BENCH_ISL/BENCH_OSL=1024/10240`, `10240/1024`, or `1024/1024`. Each arm uses
the same four GPUs, 64 active sequences per DP rank, and the TP4 length
sweep's request counts and trial policy except for one outstanding request:
offline DP2 requires at least one request per rank. Results are stored under
`vllm_dp2_length_sweep/isl<tokens>_osl<tokens>/<job-id>/`. Per-rank case
timings are also written to `driver.log`, so completed cases remain inspectable
if a later case fails or reaches the walltime.

## Status

The direct Slurm launcher requests the supplied 32-node / 128-GPU topology,
with 16 nodes each for policy and generation. The broker's 16-node client
limit does not apply to direct Slurm submission by the user on the login node.

| 32-node job | Arm | Outcome |
|---|---|---|
| 7867884 | BF16 / BF16 | Cancelled before driver startup by Slurm node failure on `nvl72017-T08`; no baseline metrics. |
| 7867896 | BF16 / MXFP8 | Completed with exit 0; validate 20-step metrics separately. |
| 7867900, 7867902, 7867906 | MXFP8 training | Failed on full CPU-tensor GPU staging during NCCL batch broadcast. |
| 7867904 | MXFP8 training, FP8 params, Option B off | Completed with exit 0; validate 20-step metrics separately. |
| 7878414 | BF16 / BF16 retry | Failed during policy logprob: attempted a 51.76 GiB CUDA allocation with only 50.71 GiB free in the same CPU-tensor broadcast path. |

The NCCL staging fix is at `af93c00e9f` and its GB200 broadcast suite passed
11/11 tests. Job `7878414` used the older `c8734274ac58` runtime and therefore
did not contain the fix. All new arms, including both BF16 comparisons, now
default to the patched source. Prepare that runtime archive before submitting;
full 32-node, 20-step retests on it are still pending.

This is a single-controller asynchronous GRPO `ready_first` recipe with
`max_staleness_versions: 2`, not a strict Async 1-off recipe. That knob bounds
rollout admission lookahead; previously admitted late groups remain selectable.
Keep this supplied recipe unchanged for like-for-like comparisons. A strict
1-off experiment needs a separately named, matched recipe for every arm.

For users with direct Slurm access on an OCI-HSG login node, pull this branch
into a `/home` worktree and prepare the runtime for its current commit first:

```bash
cd /home/sna/RL-supervl35-32n
bash experiments/supervl35_option_b_20261010/submit_runtime_sbatch.sh
```

The one-node build writes `runtime-build-<jobid>/build.log` under
`~/supervl35-optionb-control` and a commit-scoped archive under the experiment's
Lustre `runtime/` directory. Only after the build has completed with exit 0
and the `.tar.gz.sha256` file exists, submit the 32-node arm with the same
source commit:

```bash
NRL_EXPERIMENT_SOURCE_COMMIT=$(git rev-parse HEAD) \
SLURM_ACCOUNT=coreai_dlalgo_nemorl \
bash experiments/supervl35_option_b_20261010/submit_arm_sbatch.sh bf16-bf16
```

The BF16/BF16 baseline was submitted as job `7883612` using runtime source
`15045cb9425da21622220c0fa3046169a6be017c`. To submit the other five arms
independently, without waiting for the baseline or each other, run:

```bash
NRL_EXPERIMENT_SOURCE_COMMIT=15045cb9425da21622220c0fa3046169a6be017c \
SLURM_ACCOUNT=coreai_dlalgo_nemorl \
bash experiments/supervl35_option_b_20261010/submit_remaining_sbatch.sh 7883612
```

The submitted arms are BF16/MXFP8, MXFP8 training with FP8 parameters
(Option B ON and OFF), and MXFP8 training with BF16 parameters (Option B ON
and OFF). Each 32-node job has no Slurm dependency; actual start times depend
on cluster capacity and scheduler priority. Submitted IDs are recorded in
`~/supervl35-optionb-control/matrix-parallel-7883612/submissions.tsv`; inspect
that file instead of rerunning the launcher, which refuses duplicate submissions.
The launcher also refuses to submit if an earlier dependent matrix record
exists for the baseline; check those job IDs before retrying.
The jobs above were submitted with TensorBoard enabled and W&B disabled.
Changing the launcher does not enable W&B retroactively for running jobs.
After pulling this branch on the OCI-HSG login node, capture a running job's
rank-0 log into its existing Lustre results directory without starting a new
allocation:

```bash
bash experiments/supervl35_option_b_20261010/capture_running_driver_log.sh 7883612 bf16-bf16
```

While a job runs, the command attaches one small task to each node of the
existing allocation, selects the Ray head by its recorded IP, and appends
only new bytes to
`/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010/runs/<job-id>/<arm>/driver.log`.
After the job exits, the same command restores `node.log` from its rank-0
archive if that archive exists; an incomplete archive does not replace an
existing driver log. It can be repeated manually for the five other job/arm pairs in
`matrix-parallel-7883612/submissions.tsv`; do not poll it in a loop. This
captures text timing and step metrics, not TensorBoard event files. Those are
archived on normal job exit.

New submissions enable both TensorBoard and W&B. Export `WANDB_API_KEY` on the
login node before calling `submit_arm_sbatch.sh`, or submission fails before
requesting GPUs. The default W&B destination is
`nvidia/nemo-rl-supervl35-mxfp8`; override the project with the exported
`MM_TRAINER_WANDB_PROJECT` variable. Each run is named `<arm>-<job-id>` and
uses `supervl35-<job-id>-<arm>` as its W&B ID. The credential is inherited by
Slurm and is never included in a command argument or log message.

For new jobs, only rank 0 mirrors `node.log` to the per-arm `driver.log` above
every 60 seconds, appending new bytes rather than recopying the full file.
Cleanup performs one final sync. Other ranks keep their logs on node-local
scratch until their exit archives are created. W&B's local spool is kept under
node-local `results/wandb` and included in the normal results archive.

To compare completed training steps in the captured logs, run this from the
same checkout on the OCI-HSG login node:

```bash
python3 experiments/supervl35_option_b_20261010/summarize_driver_logs.py \
  --run bf16-bf16:7883612 \
  --run bf16-mxfp8:7884656 \
  --run mxfp8-fp8params-on:7884658 \
  --run mxfp8-fp8params-off:7884660 \
  --run mxfp8-bf16params-on:7884662 \
  --run mxfp8-bf16params-off:7884664
```

The default window is steps 2-20; incomplete runs fail instead of silently
reporting a short average. Use `--allow-partial` for provisional comparisons:
it restricts every arm to the same completed step IDs. The table reports
E2E, policy training, policy/reference logprob, weight sync, exposed generation
wait, valid-token throughput per GPU, and ratios to BF16/BF16. `weight_sync`
includes more than the tensor transfer; `exposed_generation` measures trainer
wait for generated samples, **not** total backend generation time. For actual
generation output throughput and numerical metrics, use the logged
`rollout/throughput/*` and `train/*` TensorBoard/W&B series. The existing six
jobs have TensorBoard enabled but W&B disabled, and their TensorBoard files
become durable in the normal per-node exit archives. Compare those series over
the same step window; do not add overlapping Async stage times to estimate E2E.

Both launchers run `sbatch --test-only` before submitting the real job. The direct
submission path was exercised by the user for the jobs above; the agent's
approved MARS client cannot submit a 32-node job. The deployed Slurm account
and partition limits still determine admission.

Source integration and HSG input inspection are complete. Runtime builds
`7862269` and `7864934` completed successfully. The earlier 16-node
performance sequence began with job `7862631`; its outcomes are retained in
the local experiment reports. Completed 32-node arms still require their
20-step metric and numerical-correctness checks.
The first three performance attempts were cancelled when the broker
did not enforce the requested success dependencies. A subsequent attempt,
`7862517`, failed during image extraction after an overlapping diagnostic
`job exec` caused a driver-file collision in the same named container.
The launcher now publishes per-node phase files; startup monitoring does not
attach to containers while they are being created. Failed attempts are not
performance measurements.
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
| 7862269 | Build helpers and pinned-source Gym environments | Completed; all five interpreter checks and eight service environments passed |

The historical 16-node sequence started with `7862631`, configured to run
`bf16-bf16` then `bf16-mxfp8`. `submit_sequence.py` was configured to submit
`mxfp8-bf16params-off/on`, then `mxfp8-fp8params-off/on`. It polls accounting
every 120 seconds, stops on a
failed predecessor or an ambiguous submission, and never retries a mutation.
Local state: `reports/supervl35-optionb-20261010/sequence-state-v3.json` in the
experiment workspace. Inspect that file for subsequently allocated job IDs.
The monitor is a local process; interruption requires reconciling state before
restarting, and a successful job exit still requires metric/step-count review.

Runtime archive SHA256:
`98d7070c2ac6faed7f39bcb3d75deb40f1be6130f1f30a3748aa90450f82c696`.
Source imports, the MCore helper functions and RADIO source-patch compatibility
passed. This is not full-model training/refit qualification.

| Cancelled performance attempt | Arms, 20 steps each | Requested dependency (not enforced) |
|---|---|---|
| 7862317 | BF16 training: BF16 / MXFP8 rollout | afterok:7862269 |
| 7862319 | MXFP8 training, BF16 parameters: Option B OFF / ON | afterok:7862317 |
| 7862323 | MXFP8 training, FP8 parameters: Option B OFF / ON | afterok:7862319 |

The historical performance attempts requested 16 nodes x 4 GB200 GPUs,
four hours, account `nemotron_sw_post`, partition `batch`, QoS `normal`,
segment 8. These were a **64-GPU cohort**, not a reproduction of 128-GPU
absolute throughput. The current launcher restores 32 nodes x 4 GPUs,
retains GBS 2048, microbatch 1, context lengths, precision scope and
sampling settings, and assigns 16 nodes each to policy and generation.

The initial 16-node attempt used source `13d905d68fbd03d45ca17eddebffc200108cef7c`
and launcher `80409e8756`, staged at `launcher16-80409e8756` in broker workspace
`supervl35-optionb-20261010`. Historical jobs used source
`c8734274ac5883cab9c942b88a37820805a6aeae`; all new retests default to
patched source `af93c00e9f8daff2f446d2b00882b0a9b4619da5` and require
its own prepared runtime archive. Image:
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
starter byte for byte. Runtime preparation must finish before performance resubmission.

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
3. Confirm runtime preparation exits successfully, resubmit without relying
   on the unsupported dependency behavior, then monitor the first
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

## Offline SuperVL3.5 MXFP8 checkpoint

`offline_convert_supervl35_mxfp8.py` converts the per-expert, rank-2 routed
`up_proj`/`down_proj` weights in the middle MoE layers of the BF16 SuperVL3.5
checkpoint. Its default scope keeps the first 2 and last 6 decoder layers in
BF16. Non-expert weights, MTP, vision, routers, and shared experts remain BF16.
This differs from the Nemotron 4 converter, which expects grouped rank-3
expert tensors. The converter reuses the source shard names, writes one output
shard at a time, validates tensor names/shapes/dtypes, and finalizes the index
and config only after all shards succeed. Rerunning resumes validated shards.

Run on a CUDA node with PyTorch and safetensors:

```bash
python experiments/supervl35_option_b_20261010/offline_convert_supervl35_mxfp8.py \
  /lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/models/super-vl-35-rlvr-v43-falcon-r3-20260905/hf \
  /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/checkpoints/supervl35-middle-experts-mxfp8-first2-last6 \
  --first-bf16 2 --last-bf16 6
```

The OCI-HSG conversion job was 7900539. It completed with 64 output shards,
36,864 MXFP8 expert weights, and 1,724 ignored module prefixes. The expected
tensor payload decreased from 248.91 GB to 150.61 GB (excluding metadata and
copied auxiliary files). Job 7900763 checked one excluded BF16 expert weight
bit-for-bit and found 2.66% relative RMSE for sample MXFP8 up/down weights.
This is a sample quantization check, not full-model numerical parity.

For a standalone vLLM TP2/DP2 load/generation check, use
`BENCHMARK_KIND=serialized_dp2_preflight`, `BENCH_ARMS=moe`, and
`SERIALIZED_CHECKPOINT=<output path>` with `run_vllm_ab.sh`. The benchmark
loads the serialized `quantization_config` from the checkpoint instead of
injecting NeMo-RL's dynamic FP8 override. The vLLM 0.29 modular FlashInfer
path also needs the ReLU2 activation-forwarding fix before TP2/DP2 MXFP8
generation. `apply_vllm_029_modular_activation.sh <vLLM worker Python>`
applies `vllm_029_modular_activation.patch` to the vLLM 0.29.0 environment;
run it inside an ephemeral container before the preflight. NeMo-RL's current
`load_format: dummy` plus per-step refit does not
automatically load this offline checkpoint; using it in RL requires a separate
load/refit integration and correctness check.
