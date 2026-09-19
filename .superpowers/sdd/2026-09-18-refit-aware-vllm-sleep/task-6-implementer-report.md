# Task 6 Implementer Report

## Status

Implementation delivered; GB200 acceptance is NOT complete. No GPU job was
submitted, no commit was pushed, and Task 7 was not implemented. The ledger
was read but left unchanged for the controller to own acceptance status.

Started from verified clean HEAD
`80ce89281aef0906503e191c114c1245a2a74a33` in the requested isolated worktree.
No prior production changes or unrelated files were reverted.

Implementation and fix commits, all SSH-signed and carrying `Signed-off-by`:

1. `d15ffd4a876cb24cad5d24089d53194d1bb51422`: CPU numerical oracles,
   process deadline, and real-ZMQ production-loop regression.
2. `ede4e2c97c052e558c898c2a157c43813dd4af39`: functional harness, test worker,
   recipes, runtime provenance, wrappers, and nightly registration.
3. `f007a6aa1b8f7ccddcc62d4d967450761b6ab76c`: preserve the inherited generation
   budget and add a guard against runtime recipe-limit overrides.
4. `7850aed87d220a1f2d164be91b4f699e0ca6cb3e`: include runtime environment
   overrides (including NRL and uv environment selection) in provenance.
5. `cf73fcf7abcccd08e4e32ea47bfcc9dafa13fe83`: preserve the initialized
   refit manifest and share one owned Ray driver connection across selected
   functional nodes.
6. `b8479e3e5518dd0204b80775339e48fd7a06ab2e`: consume only the three
   mandatory `tools/launch` provenance arguments and continue rejecting
   arbitrary recipe or pytest overrides.

`git log --format='%h %G? %s'` reported `G` for all six signatures.
This report is committed separately. The eventual GB200 run must record the
actual final HEAD, not assume the last implementation SHA above is its HEAD.

## Fix Round 1

All three independent-review warnings are addressed in the Task 6 harness and
wrapper boundary without changing `nemo_rl/`, either original recipe, or Task 7.

- Initial IPC metadata is now explicitly captured while the new policy is
  resident and the generation engine is asleep, before the preserving A refit.
  The failure node reuses that shape/dtype manifest after A offloads policy
  storage; it no longer calls policy `prepare_refit_info()` after offload.
- A session-scoped pytest fixture owns one Ray driver connection for either a
  one-node-ID or two-node-ID wrapper selection. Function-scoped model fixtures
  still shut down models and placement groups. Fresh B/C engine construction
  remains inside the same live driver session. The fixture disconnects at
  session teardown only when it established the connection itself.
- `refit_sleep.env` accepts, validates, records, and consumes only
  `logger.wandb.name=...`, `++git_meta=...`, and `++container=...`. Empty,
  duplicate, and all other arguments fail closed. These values are exported
  under `NRL_REFIT_SLEEP_LAUNCH_*` and therefore included by sanitized runtime
  provenance; they are not forwarded as recipe overrides.

The detailed fix evidence is in `task-6-fix-round-1-report.md`.

## Implemented Gate

- Dedicated functional pytest, not a 16-GPU unit test. The Moonlight test and
  existing generation/policy parity helper supplied the lifecycle and masking
  patterns; there was no existing Qwen3-30B-A3B integration node to extend.
- Inherits the exact Qwen3 MXFP8 performance recipe. Physical placement remains
  4 nodes x 4 GB200 GPUs, segment size 4, two colocated worker groups,
  Megatron EP16, and FlashInfer TRTLLM. Importantly, EP16 is the **policy**
  setting: the unchanged recipe inherits vLLM EP1/TP1, with 16 rollout owners.
  The harness does not silently change rollout EP to 16.
- Exact ignore patterns remain `model.layers.*.self_attn.*`,
  `model.layers.*.mlp.gate`, and `lm_head`. These retain mixed BF16/MXFP8
  execution; no first/last-layer override was added.
- Fixed inputs are four literal prompts repeated four times to cover all
  rollout owners. The seed and generation/context budgets come from the recipe.
  Greedy generation and `raw_logprobs` are observation controls. Normal
  `configure_generation_config` setup is retained, and Megatron receives the
  recipe's `max_num_steps` as its required scheduler horizon.
- Initial preserving A refit must establish real runtime coverage. B scales
  actual BF16/FP32 policy parameters by 1.0625; C scales B by 0.875. The test-only
  worker uses normal policy preparation, materialization, and optimizer
  `reload_model_params()` to keep master shards consistent. No learning rate,
  batch size, parallelism, precision, or kernel setting is changed. These are
  deterministic weight updates, not optimizer-training/performance evidence.
- The same candidate engine experiences both destructive boundaries, followed
  by real IPC refit, capability checks, and token/logprob measurement. Sleep
  precedes the colocated policy update, matching the training/refit lifecycle.
- Exact B/C states are saved through `Policy.save_checkpoint()` plus
  `finalize_async_save()`. Candidate workers are shut down before fresh policy
  and generation pairs load each checkpoint and perform preserving refits.
  This avoids two live receivers sharing the per-device IPC socket and retains
  the original two-worker-group topology. No second engine steals IPC traffic.
- Fresh B/C comparisons require exact generated token sequences and compare
  raw selected-token logprobs, not unavailable full logits or packed weights.
  Megatron parity excludes prompt/pad tokens and reports multiplicative
  probability error and sampled k3 KL(P_generation || P_policy).
- A != B and B != C are required. Once a tolerance is supplied, unchanged
  tokens require a logprob difference greater than twice the fresh-engine
  tolerance, so numerical noise alone cannot satisfy the stale-weight guard.

## Failure Path

No production failure-injection hook was added. The existing protocol already
provides a stronger safe path than an artificial receiver hold. The real sender
manifest is captured during initial communicator setup, while policy storage is
resident and generation is asleep. After actual discard, the failure node reuses
that manifest to install one extra unsent receiver entry, without exporting from
offloaded policy storage. All real sender tensors still stream normally.
COMPLETE fails manifest validation, but the receiver's existing `finally` sends
its ACK.

The GPU node requires a real manifest/update failure, all 16 sender futures
successfully complete, every receiver future finishes, no pending refs remain,
KV cache is never woken, and stale/discarded state stays set. A timeout is not
an accepted failure result. An outer test-only watchdog covers discarded-state
work and model shutdown; deadline expiry exits the test process with code 124.
The default configured deadline is 600 seconds, including shutdown in the
test's elapsed-time assertion. No one-sided sleep or sender hold is injected.

CPU evidence uses real REQ/REP sockets and the production manifest/receive-loop
source isolated by AST to avoid importing Linux-only vLLM. CUDA rebuilding,
loading, and finalization are substituted; both ACKs and coverage invalidation
are observed. This is explicitly not CUDA IPC or GB200 evidence.

## BF16 Control

The separate 6x4 wrapper runs only the new BF16 pytest node, which delegates
to the existing Qwen3.5 BF16 FlashInfer TRTLLM non-colocated NCCL Reshard shell
recipe. Its topology/settings and existing KL/reward checks are unchanged.
The node additionally requires all 20 finite loss/KL steps and explicit
non-colocated preserving/reset-only decisions, with no discard decision.
The existing mean-KL bound of 0.002 is reused, not claimed as a new observation.
This control is not labeled or counted as level 2.

`common.env` has one opt-in change: `NRL_TEST_RUN_DIR` redirects output away
from the repository. Existing callers retain their old default. The recipe
naming check recognizes these two exact functional filenames as GRPO-based;
it does not broadly permit arbitrary `vllm-*` training recipes.

## Node IDs

```text
tests/functional/test_vllm_refit_sleep.py::test_qwen3_mxfp8_destructive_refit_abc
tests/functional/test_vllm_refit_sleep.py::test_qwen3_mxfp8_missing_manifest_after_discard
tests/functional/test_vllm_refit_sleep.py::test_qwen35_bf16_nccl_reshard_preserving_control
```

The 4x4 wrapper invokes exactly the first two; the 6x4 wrapper invokes exactly
the third. Only the mandatory `tools/launch` provenance arguments are consumed;
arbitrary shell arguments/pytest overrides are rejected. Both declare 240
minutes and one run. The outer process limit is 230 minutes, with 220-minute
per-test timeouts and time reserved for log collection. Runtime is unmeasured;
these are budget ceilings, not performance claims.

## Tolerance Calibration

No MXFP8 tolerance was guessed or copied from another model. Without
`NRL_REFIT_SLEEP_TOLERANCES`, the ABC test records its observations then fails
explicitly. The wrapper overrides pytest's default `-x`, so an uncalibrated
ABC result does not suppress the independent failure node.

After reviewing a real GB200 observation artifact, supply a JSON record with:

- `hardware`: `GB200`.
- `source_sha`: the full observed source commit.
- `image_digest`: the observed `sha256:` identity.
- `observation_artifact`: the durable observation location.
- Numeric `fresh_logprob_atol`, `token_mult_prob_error_max`, and
  `gen_kl_error_max`, chosen only after inspecting those observations.

All bounds must be finite/nonnegative; multiplicative error must be at least
one. This implementation validates provenance fields but cannot certify that
an operator's calibration artifact is scientifically sufficient. The controller
must review/pin it and rerun both gates before accepting Task 6.

## Local RED/GREEN Evidence

All commands ran on macOS in the supplied worktree with its existing local
environment, using `uv run --no-sync`. No Linux dependency resolution or GPU
execution is implied.

| Check | RED evidence | GREEN/result |
| --- | --- | --- |
| Core oracle | 10 failures for missing observation/parity/update/manifest helpers | Implemented helpers pass |
| Recipe/wrappers | 5 failures for missing recipes, wrappers and functional nodes | Contracts pass |
| Node-local output | `common.env` returned repository path instead of requested local path | Opt-in redirection passes |
| Deadline | Two child-process failures because deadline helper did not exist | Exit 124 on expiry; exit 0 after cancellation |
| Noise/stale guard | `logprob_atol` unsupported | Noise-sized state change rejected |
| Provenance | Credential leaked; subsequent broad filter dropped HybridEP token setting | Secrets excluded; runtime setting retained |
| Environment overrides | Missing `NRL_FORCE_REBUILD_VENVS` and uv project environment | Overrides captured without credentials |
| BF16 completion | Missing completion validator | Incomplete/nonfinite 20-step control rejected |
| Recipe naming | Existing test rejected the functional `vllm-*` name | Exact-name mapping passes |
| Frozen limits | Harness assigned `max_new_tokens=32` | Override removed; inherited limits guarded |
| New CPU suites | Combined fix-round final run | 27 passed |
| Existing synchronizer suite | Regression run | 91 passed |
| Combined CPU total | Prior 114 plus four fix-round regressions | 118 passed |
| Suite registration/naming | Five selected repository-wide checks | 5 passed, 11 deselected |
| Functional collection | Exact new module | 3 nodes collected |
| Functional CPU execution | No GB200 opt-in | 3 skipped, not GPU passes |
| Ruff check/format | All nine changed Python files | Passed |
| compileall | All nine changed Python files | Passed |
| Shell syntax | Common env, gate env and both wrappers | Passed |
| Wrapper dry-run | `TEST_DRYRUN=1` for each wrapper | Both passed |
| Exact launcher arguments | Three mandatory forms plus arbitrary override | 2 passed, 7 deselected |
| Manifest lifecycle | Missing resident-only initializer | Cached manifest reused after offload |
| Ray lifecycle | Missing session owner | One init/two model fixtures/one shutdown |
| Diff check | `git diff --check` and staged checks | Passed |

Reproduce the main CPU runs:

```bash
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_refit_sleep_oracle.py \
  tests/unit/models/generation/test_refit_sleep_recipe.py \
  tests/unit/models/generation/test_refit_sleep_runtime.py \
  tests/unit/models/generation/test_refit_sleep_ipc_protocol.py -q -o addopts=''
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit/weight_sync \
  tests/unit/weight_sync/test_weight_synchronizer.py -q -o addopts=''
PYTHONPATH=. uv run --no-sync pytest --noconftest \
  tests/unit/test_recipes_and_test_suites.py \
  -k 'start_with_algo or accounted_for or match_gpus or no_overlap' -q -o addopts=''
PYTHONPATH=. uv run --no-sync pytest --collect-only \
  tests/functional/test_vllm_refit_sleep.py -q -o addopts=''
```

Two environment limitations were observed and not disguised as RED evidence:

- Direct execution of existing
  `test_vllm_backend.py::test_update_weights_via_ipc_acks_manifest_error_and_returns_false`
  fails importing vLLM (`ModuleNotFoundError: No module named 'vllm'`). The
  new AST-isolated CPU socket test passes without that Linux-only import.
- `DRYRUN=1 ... ./tools/launch <both-new-wrappers>` fails in the existing
  `extract_config` GNU-sed expression on macOS BSD sed. GNU sed is not installed.
  This happens before launcher argument construction. No snapshot/submission
  occurred. `bash -n`, both wrapper dry-runs, and executable tests of the exact
  three generated argument forms pass. The launcher was not modified.

The first helper test attempt also omitted `PYTHONPATH=.` and failed collection;
the corrected command above produced the actual RED and GREEN test results.

## Required GB200 Run

No image, partition, account, job ID, or numerical tolerance was fabricated.
Use a reviewed nightly image and the existing cluster configuration. After
the controller authorizes/pushes the branch, pull it under `/home` on the
cluster and verify the desired clean HEAD. This turn intentionally did neither
push nor submission. Example on the cluster, with caller-supplied variables:

```bash
export NRL_REFIT_SLEEP_EXPECTED_SHA=$(git rev-parse HEAD)
export NRL_REFIT_SLEEP_IMAGE="$NIGHTLY_SQSH"
export NRL_REFIT_SLEEP_IMAGE_DIGEST="sha256:$(sha256sum "$NIGHTLY_SQSH" | cut -d' ' -f1)"
export CONTAINER="$NRL_REFIT_SLEEP_IMAGE"
export MOUNTS="/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch"
export NRL_REFIT_SLEEP_ARTIFACT_DIR="/lustre/$USER/experiments/refit-sleep/$RUN_ID/qwen3"
export NRL_REFIT_SLEEP_CACHE_ROOT="/raid/scratch/$USER/refit-sleep-$RUN_ID"
export RAY_TMPDIR="$NRL_REFIT_SLEEP_CACHE_ROOT/ray"
export GPUS_PER_NODE=4
export BASE_LOG_DIR="/home/$USER/task6-launch/$RUN_ID"
export COMMAND="bash tests/test_suites/llm/vllm-destructive-refit-qwen3-30ba3b-4n4g.sh"
mkdir -p "$BASE_LOG_DIR" "$NRL_REFIT_SLEEP_ARTIFACT_DIR"
sbatch --test-only --account="$ACCOUNT" --partition="$PARTITION" \
  --nodes=4 --ntasks-per-node=1 --exclusive --segment=4 --time=04:00:00 \
  --job-name=refit-sleep-qwen3 --export=ALL \
  --output="$NRL_REFIT_SLEEP_ARTIFACT_DIR/slurm-%j.out" ray.sub
```

After scheduling review, repeat that `sbatch` command without `--test-only`.
For the BF16 control use its new wrapper as `COMMAND`, a distinct artifact/cache
run directory, `--nodes=6`, and `--segment=2`. Add site-specific required GPU/QoS
flags from the cluster configuration without changing GPU topology. Both runs
require the nightly driver environment with pytest/pytest-timeout and the normal
separate mcore/vLLM actor environments. Preserve the image's CUDA/runtime settings.
Set `NRL_REFIT_SLEEP_TOLERANCES` only after the first observed GB200 calibration.
Poll one filtered `squeue -j JOBID` per minute, monitoring at least five minutes.

Source/config stay under `/home`; HF, compiler, Torch, vLLM, uv, Ray and temporary
caches use `/raid/scratch`. Set `RAY_TMPDIR` before Ray starts, not just inside
the wrapper. Stage shared model inputs once per node into those caches.
Only durable B/C checkpoints and final artifacts go to `/lustre`. The wrapper
records exact source SHA, declared immutable image identity (file bytes are
hashed when accessible), driver packages/interpreter, sanitized runtime env,
submodule SHAs, GPU UUIDs/driver versions, resolved Qwen3 configs, fixed token
inputs, B/C updates, and numerical/failure observations. Final driver and
per-node Ray logs are archived on success or failure. Log-archive failures
also fail the wrapper. The controller must ensure the declared image is the
actual launched image; an in-container process cannot independently prove a
registry runtime digest merely from an environment variable.

## Exact Changed Files

```text
examples/configs/recipes/llm/vllm-destructive-refit-qwen3-30ba3b-4n4g.yaml
examples/configs/recipes/llm/vllm-preserving-refit-qwen3.5-35ba3b-6n4g-bf16-trtllm.yaml
tests/functional/refit_sleep_policy_worker.py
tests/functional/refit_sleep_runtime.py
tests/functional/refit_sleep_utils.py
tests/functional/test_vllm_refit_sleep.py
tests/test_suites/llm/common.env
tests/test_suites/llm/refit_sleep.env
tests/test_suites/llm/vllm-destructive-refit-qwen3-30ba3b-4n4g.sh
tests/test_suites/llm/vllm-preserving-refit-qwen3.5-35ba3b-6n4g-bf16-trtllm.sh
tests/test_suites/nightly_gb200.txt
tests/unit/models/generation/test_refit_sleep_ipc_protocol.py
tests/unit/models/generation/test_refit_sleep_oracle.py
tests/unit/models/generation/test_refit_sleep_recipe.py
tests/unit/models/generation/test_refit_sleep_runtime.py
tests/unit/test_recipes_and_test_suites.py
.superpowers/sdd/2026-09-18-refit-aware-vllm-sleep/task-6-implementer-report.md
.superpowers/sdd/2026-09-18-refit-aware-vllm-sleep/task-6-fix-round-1-report.md
```

No `nemo_rl/` production file, original performance recipe, or original BF16
recipe/wrapper changed; `git diff --exit-code` against the requested base
verified that boundary.

## Residual Risks / Acceptance Work

- Neither 4x4 MXFP8 nor 6x4 BF16 ran here. Actual coverage eligibility, packed
  layout reconstruction, checkpoint reloading, numerical ranges, CUDA IPC ACK
  completion, and deadline/shutdown latency remain GB200 acceptance work.
- Full inherited generation budgets, four model-pair initializations in the
  MXFP8 wrapper, and B/C checkpoint I/O may exceed the provisional time budget.
  Reserve roughly two full BF16 Qwen3 checkpoints plus logs in durable storage;
  runtime/storage must be measured without shrinking recipe settings.
- Exact greedy token parity over long generations is intentionally strict;
  any divergence must be investigated rather than silently relaxed locally.
- The fresh baseline uses real Megatron checkpoint restoration and preserving
  IPC refit, not exposed full-vocabulary logits or packed-weight inspection.
- The failure case proves manifest rejection after complete draining, not an
  arbitrary CUDA loader crash or a wedged kernel. It deliberately adds no
  production injection hook and no one-sided timeout stimulus.
- Deadline expiry is a failing process exit, never successful recovery.
  No in-process restart/recovery behavior is added.
- Review was scoped self-review plus local tests/source inspection. No
  independent reviewer/subagent tool was available in this session. Controller
  review and final Linux/GB200 acceptance remain required before Task 7.
