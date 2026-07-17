# NeMo-RL vLLM 0.25.1 Drafter Matrix

This experiment measures the applicable speculative proposers shipped by
vLLM 0.25.1 against matched NeMo-RL performance-recipe baselines. Final
comparisons average steps 2 through 20; step 1 is initialization warmup.

## Controlled Recipes

| Model | Target revision | Recipe | Topology | Max OSL | Sampling |
|---|---|---|---:|---:|---|
| Qwen3-30B-A3B | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | `grpo-qwen3-30ba3b-4n4g.yaml` | 4 nodes x 4 GPUs | 4096 | temperature 1.0, top-p 1.0 |
| Qwen3-32B | `9216db5781bf21249d130ec9da846c4624c16137` | `grpo-qwen3-32b-4n4g.yaml` | 4 nodes x 4 GPUs | 4096 | temperature 1.0, top-p 1.0 |
| Qwen3-235B-A22B | `8efa61729e24bd65b1d152b5ab5409052aa80e65` | `grpo-qwen3-235b-16n4g.yaml` | 16 nodes x 4 GPUs | 8192 | temperature 1.0, top-p 1.0 |

The recipe remains authoritative for model, dataset, batching, placement,
parallelism, MoE backend, and sampling. The matrix changes only step count,
output/log paths, checkpoint saving, CUDA Graph mode, and SpecDec settings.
Every run uses `enforce_eager=false`, `FULL_AND_PIECEWISE`, native recipe/vLLM
capture sizing, Triton MoE from the recipes, and `checkpointing.enabled=false`.
Qwen3-235B additionally sets `NRL_DISABLE_VLLM_PORT_OVERRIDE=1` so its TP=8
multi-node engines use vLLM 0.25.1 rendezvous allocation instead of colliding
on NeMo-RL's deterministic `7000 + n*100` override. The smaller recipes retain
the default NeMo-RL port path that passed their smoke gates.

## Matrix

| Variant | Runner | Qwen30 | Qwen32 | Qwen235 | Setting/checkpoint |
|---|---|---:|---:|---:|---|
| `baseline` | MRv2 | yes | yes | yes | no speculative config |
| `baseline_mrv1` | MRv1 | yes | yes | yes | matched control for MRv1 methods |
| `eagle3_k1/k3/k5` | MRv2 | yes | yes | yes | exact target-specific EAGLE3 head |
| `eagle3_thinking_k1/k2/k3/k4/k5` | MRv2 | alias | yes | yes | reasoning-distribution EAGLE3 head |
| `eagle3_thinking_dynamic_k123` | MRv2 | no | yes | no | historical K0-K3 smoke schedule only |
| `eagle3_thinking_dynamic_k5` | MRv2 | no | yes | no | calibrated K0-K5 DynamicSD schedule artifact |
| `dflash_k3/k5` | MRv2 | yes | yes | no | exact DFlash head, draft `FLASH_ATTN` |
| `draft_k1/k5` | MRv1 | yes | yes | yes | sequential `amd/PARD-Qwen3-0.6B` |
| `pard_k5/k16` | MRv1 | yes | yes | yes | parallel `amd/PARD-Qwen3-0.6B` |
| `suffix_k32` | MRv1 | yes | yes | yes | suffix tree depth 32 |
| `ngram_k5` | MRv1 | yes | yes | yes | prompt lookup min=max=5 |
| `ngram_gpu_k5` | MRv1 | yes | yes | yes | GPU prompt lookup min=max=5 |

Exact model-based drafter identities:

| Target | Method | Repository | Revision |
|---|---|---|---|
| Qwen30 | EAGLE3 | `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` | `6afc5aa2477b923467fb9a8d906782b984a9a6ba` |
| Qwen32 | EAGLE3 | `RedHatAI/Qwen3-32B-speculator.eagle3` | `dc84fe7ff1db31efa824776f49c141fc8195eb47` |
| Qwen235 | EAGLE3 | `nvidia/Qwen3-235B-A22B-Eagle3` | `33f3c01ce807376d1171301b9a148b1b28f239ba` |
| Qwen32 | EAGLE3 Thinking | `RedHatAI/Qwen3-32B-Thinking-speculator.eagle3` | `a1403e07b73a66fc9ef561463631c31864616933` |
| Qwen235 | EAGLE3 Thinking | `RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3` | `3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87` |
| Qwen30 | DFlash | `RedHatAI/Qwen3-30B-A3B-speculator.dflash` | `edcff83783141eb9383e2bd6c33610d9a3104288` |
| Qwen32 | DFlash | `AICP-Labs/qwen3-32b-dflash-en-zh` | `68ccc7fd27b104271321b179a2959c759dce5eef` |
| all | draft/PARD | `amd/PARD-Qwen3-0.6B` | `f9f650fbab180c26498817718f0db5cae8f25136` |

Qwen30 does not get a duplicate Thinking row. The selected base repository at
revision `6afc5aa2477b923467fb9a8d906782b984a9a6ba` and
`RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` at revision
`a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf` resolve to the same config blob
`4e11c4dbb9b0bd911748a6f567d41f57c3dcdbe3` and model LFS SHA-256
`d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa`.
The selected checkpoint was trained with reasoning enabled. Qwen32 and
Qwen235 have distinct Thinking checkpoints, so their K1/K2/K3/K5 rows are
controlled A/B comparisons against the base EAGLE3 rows.

Qwen235 DFlash has no exact public checkpoint and is rejected before
submission. Native MTP requires target-embedded heads absent from these Qwen3
checkpoints. DSpark and Medusa lack exact target-specific checkpoints;
`mlp_speculator` has an MRv1 runtime gap; hidden-state extraction and custom
classes are not acceleration proposers. PARD-2 and DFlare require non-upstream
patches and stay in separate experiments.

## Run Workflow

Run these commands from the clean, pushed cluster checkout. Results default to
`/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/vllm0251_drafter_matrix`,
outside the Git worktree.

Stage missing immutable drafters through a bounded compute job. The control
plane uses login-node `python3`; the worker switches to the container's
`/opt/nemo_rl_venv/bin/python`.

```bash
STAGE=experiments/vllm_0251_drafter_matrix/submit_stage_drafters.sh
STAGE_OUT=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/vllm0251_drafter_matrix/staging

bash "$STAGE" show --output-dir "$STAGE_OUT"
bash "$STAGE" test-only --output-dir "$STAGE_OUT"
bash "$STAGE" submit --output-dir "$STAGE_OUT"
```

The staging manifest deduplicates the shared PARD checkpoint and records exact
repository, revision, cache path, state, and job ID. The submitter snapshots
the worker source into a content-addressed Lustre directory, submits it held,
writes `queued`, then releases it. The CPU-only staging job is non-exclusive
and forwards `HF_HOME` plus an explicit authentication/proxy allowlist. It
never inherits the ambient environment. Existing snapshots are verified
through the same immutable path. The `reconcile` command converts a stale
queued manifest to a terminal failure after pre-wrapper scheduler failures.

```bash
SCRIPT=experiments/vllm_0251_drafter_matrix/submit_matrix.sh

bash "$SCRIPT" show \
  --model qwen30 --variant eagle3_k3 --phase smoke2 --cluster lyris

bash "$SCRIPT" test-only \
  --model qwen30 --variant eagle3_k3 --phase smoke2 --cluster lyris

bash "$SCRIPT" submit \
  --model qwen30 --variant eagle3_k3 --phase smoke2 --cluster lyris
```

Promotion is `smoke2` (load/config) to `smoke5` (short performance) to
`final20` (reportable). The smoke5 gate may be satisfied in place when a
final20 allocation reaches step 5 with the identical immutable config; this
avoids duplicating a scarce multi-node allocation. A run advances only after
its exact baseline and candidate complete without config fallback, missing
metrics, or early exit.
Lyris jobs use account `coreai_dlalgo_llm`, partition `gb200`, four GPUs per
node, `--segment=<nodes>`, no `--gres`, and no dependency/singleton constraint.

DynamicSD requires an explicit versioned schedule artifact. The checked-in
`calibration/qwen32_thinking_k123_seed.json` uses the historical vLLM 0.24
profile only as a smoke seed: K3 for scheduler batch sizes 1-127 and K1 for
128-256. It may run only `smoke2` or `smoke5`; `final20` rejects it until a
matched vLLM 0.25.1 calibration replaces the seed status. Dynamic runs apply
the source-guarded EAGLE3 CUDA Graph and variable-width drafting patch through
a run-scoped post-sync hook; fixed-K runs do not load that patch. The checked-in
K0-K5 calibration passed GPU smoke job `2412001`: selected, requested, and
returned widths matched for K0, K1, K2, K3, and K5 with CUDA Graphs enabled.
Final20 additionally requires the exact reviewed schedule artifact SHA-256 in
the final allowlist, so editing metadata cannot bypass that runtime gate.

Reportable K0-K5 DynamicSD uses a matched offline profile before NeMo-RL is
submitted. The profiler follows the goodput method from vLLM PR #32374:
`accepted_length(K) / median_ITL(batch_size,K)`. It measures K0-K5 at scheduler
batch-size points `1,4,16,32,64,128,192,256`, with twenty batches per point,
then linearly interpolates ITL between adjacent measured batch sizes and
selects the maximizing K. The runtime key is the number of requests actually
scheduled in one engine step, so serving concurrency is a controlled proxy;
final promotion also requires selected-K and verified-draft telemetry from the
NeMo-RL smoke run.

The profile is matched to Qwen3-32B TP2, Thinking drafter TP1, temperature 1.0,
top-p 1.0, max model length 4096, max batched tokens 16384, chunked prefill,
disabled prefix caching, and `FULL_AND_PIECEWISE` CUDA Graphs. K0 is measured
as a true no-drafter server. K5 separately records position-level acceptance
on deterministic OpenMathInstruct-2 prompts rendered with `cot.txt`. Every
completed result is written immediately, so a timeout preserves completed
cells, but an incomplete 48-cell grid cannot produce a calibrated schedule.
The dataset snapshot is pinned to revision
`469216e3f46f4dacf476b382e192485ea51a143e`.

Show the exact six-job plan, run all scheduler preflights, and submit only
after the checkout has been committed, pushed, and pulled on Lyris:

```bash
PROFILE_LAUNCHER=experiments/vllm_0251_drafter_matrix/profile_dynamic_sd.py
PROFILE_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/vllm0251_dynamic_profile/qwen32-thinking
COMMON_ARGS=(
  --repo-dir "$PWD"
  --output-dir "$PROFILE_ROOT"
  --hf-home /lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home
  --container /lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh
  --mounts /lustre:/lustre
)

python3 "$PROFILE_LAUNCHER" show "${COMMON_ARGS[@]}"
python3 "$PROFILE_LAUNCHER" test-only "${COMMON_ARGS[@]}"
python3 "$PROFILE_LAUNCHER" submit "${COMMON_ARGS[@]}"
```

The launcher explicitly passes `--dependency=` because the shared `ray.sub`
contains a historical singleton directive. Each K uses an independent
one-node, `--segment=1`, five-hour allocation without GRES. A per-job locked
vLLM 0.25.1 environment is materialized under `/tmp`; the base container
environment is not replaced.

After all profile jobs complete, assemble and derive the immutable artifacts:

```bash
PROFILE_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/vllm0251_dynamic_profile/qwen32-thinking
WORKER=experiments/vllm_0251_drafter_matrix/dynamic_profile_worker.py
CALIBRATOR=experiments/vllm_0251_drafter_matrix/calibrate_dynamic_sd.py

/opt/nemo_rl_venv/bin/python "$WORKER" assemble \
  --root "$PROFILE_ROOT" \
  --target-revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --drafter-revision a1403e07b73a66fc9ef561463631c31864616933 \
  --output "$PROFILE_ROOT/profile.json"

/opt/nemo_rl_venv/bin/python "$CALIBRATOR" \
  "$PROFILE_ROOT/profile.json" \
  --output "$PROFILE_ROOT/schedule.json"
```

The completed Qwen3-32B Thinking calibration is checked in as
`calibration/qwen32_thinking_k5_vllm0251_profile.json` and
`calibration/qwen32_thinking_k5_vllm0251_schedule.json`. Its zero-margin
schedule is `[[1,34,5],[35,75,3],[76,85,2],[86,256,1]]`. This schedule is
derived from all 48 fixed-K profile cells and its corrected five-step runtime
smoke completed successfully before the schedule SHA-256 was allowlisted for
final20.

Schema-v2 schedules declare global K5 even when profiling selects only lower
Ks. This prevents vLLM from silently clamping a selected K4/K5 to a K3 global
maximum. Final20 rejects schema-v1 schedules, non-vLLM-0.25.1 profiles, and
schedule artifacts whose reviewed SHA-256 is not allowlisted.

```bash
bash "$SCRIPT" submit \
  --model qwen32 --variant eagle3_thinking_dynamic_k123 \
  --phase smoke2 --cluster lyris \
  --dynamic-schedule \
  experiments/vllm_0251_drafter_matrix/calibration/qwen32_thinking_k123_seed.json
```

Submission requires a clean checkout whose exact HEAD is present on the same
branch under remote `fork`, plus recursively initialized submodules. The CLI
validates the container, target `refs/main`, exact drafter snapshot, and writes
atomic `provenance.json` and `provenance.txt` before invoking SLURM. W&B uses
project `nemo-rl-vllm0251-drafter-matrix`; credentials come only from the
environment.

## Collect Results

Each input is a JSONL file with one validated metric record per step. Strict
collection requires the complete 2-20 window; use partial output only for
timeout diagnosis.

```bash
/opt/nemo_rl_venv/bin/python \
  experiments/vllm_0251_drafter_matrix/collect_results.py \
  /lustre/path/to/steps-*.jsonl \
  --csv /lustre/path/to/drafter-matrix.csv \
  --markdown /lustre/path/to/drafter-matrix.md
```

The report keeps E2E/generation time and throughput, policy and logprob time,
generation ratio, acceptance rate, mean accepted length, runner, CUDA Graph
resolution/coverage, job/log/W&B links, and explicit failed/unsupported states.
