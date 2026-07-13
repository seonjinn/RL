# NeMo-RL 10-Step Status Refresh - 2026-06-15

Refreshed from Lyris and OCI-HSG SLURM at `2026-06-15 01:32:46 PDT`.

## Current Live State

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B MathRL latest-main | Lyris | `2113812`, `2113813`, `2113814` | `FAILED` | No step proof. Baseline failed in policy worker setup on `nvidia_resiliency_ext.__version__`; PARD K3/K5 failed in policy worker setup on missing `transformers.models.ernie4_5_vl_moe`. |
| 235B MathRL latest-main | OCI-HSG | `3290316`, `3290317`, `3290318` | `PENDING (Priority)` | Not started. Latest start estimate was `2026-06-15T15:30:00`. |
| 235B SWE-RL r27 staged-SIF | Lyris | `2124206`, `2124207`, `2124208` | `PENDING (Priority)` | Not started. Latest start estimates were `2026-06-15T05:14:00`, `05:19:00`, and `05:23:00`. |
| 235B SWE-RL PARD-2 r16 | OCI-HSG | `3308774` | `PENDING (Priority)` | Not started. Latest start estimate was `2026-06-15T07:30:00`. |

## Lyris MathRL Failure Details

| Job | Variant | Elapsed | Primary error |
| --- | --- | ---: | --- |
| `2113812` | baseline | `00:04:52` | `AttributeError: module 'nvidia_resiliency_ext' has no attribute '__version__'` from `IsolatedWorkerInitializer.create_worker()`. |
| `2113813` | PARD K3 | `00:07:34` | `ModuleNotFoundError: No module named 'transformers.models.ernie4_5_vl_moe'` from `IsolatedWorkerInitializer.create_worker()`. |
| `2113814` | PARD K5 | `00:07:58` | Same missing `transformers.models.ernie4_5_vl_moe` import. |

The Ray `Check failed: !core_worker_process` lines happen after the first Python exception and look secondary. PARD K3/K5 had already loaded Qwen3-235B checkpoint shards through vLLM before the policy-side import failure, so this is an actor environment/package compatibility blocker rather than a speculative decoding acceptance/result issue.

## Known Usable NeMo-RL 10-Step Examples

| Scope | Cluster | Jobs | Result | Usefulness |
| --- | --- | --- | --- | --- |
| Qwen8 official PARD-2 comparison | OCI-HSG | `3288181`, `3288182`, `3288183` | baseline, static PARD-2, and online PARD-2 all `COMPLETED` for 10 steps | Cleanest functional online PARD-2 comparison. |
| 235B SWE-RL after-prewarm W&B retry | OCI-HSG | `3299487`, `3299489`, `3299491` | baseline, PARD K5, and Eagle-3 K3 all `COMPLETED` for 10 steps | Best 235B SWE-RL 10-step evidence so far, but PARD-2 and suffix failed in the same matrix. |
| Qwen8 official PARD-2 online 20-step | OCI-HSG | `3279589` | online PARD-2 `COMPLETED` for 20 steps | Good correctness proof for online draft train/refit behavior, not a 235B result. |

## Not Usable As 10-Step Examples

| Scope | Reason |
| --- | --- |
| 235B MathRL latest-main Lyris retry3 | All three variants failed before step 1 on policy actor environment/import errors. |
| 235B MathRL latest-main OCI-HSG | Still pending; no runtime evidence yet. |
| Lyris integrated NeMo-RL SpecDec maxsteps10 matrix | Terminal batch was `FAILED=12`, `CANCELLED=6`; no usable speed/result proof. |
| 235B SWE-RL PARD-2 | No clean 235B SWE-RL PARD-2 step proof yet; current OCI and Lyris retries are pending. |

## Short Answer

MathRL and SWE-RL are not both uniformly broken. The safest 10-step examples right now are:

1. Qwen8 official PARD-2 baseline/static/online comparison on OCI-HSG.
2. 235B SWE-RL OCI baseline/PARD/Eagle-3 completed 10-step run.

For the exact current question, 235B MathRL has no successful 10-step proof yet, and the newest 235B SWE-RL retries have not started. The next useful action is to wait for the pending OCI-HSG MathRL and SWE-RL jobs; if MathRL OCI fails with the same actor import path, patch the actor environment for `nvidia_resiliency_ext.__version__` and the missing `transformers.models.ernie4_5_vl_moe` import before another Lyris retry.

## Continuation Update - 2026-06-15 01:38 PDT

Live SLURM refresh:

| Track | Cluster | Jobs | State | Latest estimate |
| --- | --- | --- | --- | --- |
| 235B SWE-RL r27 staged-SIF | Lyris | `2124206`, `2124207`, `2124208` | `PENDING (Priority)` | `2026-06-15T03:16:00`, `03:57:00`, `05:14:00` |
| 235B MathRL latest-main | OCI-HSG | `3290316`, `3290317`, `3290318` | `PENDING (Priority)` | `2026-06-15T15:20:00`, `16:10:00`, `16:10:00` |
| 235B SWE-RL PARD-2 r16 | OCI-HSG | `3308774` | `PENDING (Priority)` | `2026-06-15T10:30:00` |

Remote source pre-fix applied before the pending OCI-HSG MathRL jobs start:

| Cluster | Repo | Guard |
| --- | --- | --- |
| Lyris | `/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-main-mathrl-20260613` | `nvrx.__version__` was already guarded; added optional ERNIE-VL import guard in `megatron/bridge/models/__init__.py`. |
| OCI-HSG | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-mathrl-20260613` | Added `getattr(nvrx, "__version__", "0.0.0")` guard in Megatron-LM `nvrx.py` and optional ERNIE-VL import guard in Megatron-Bridge `models/__init__.py`. |

Both remote patches passed `python3 -m py_compile` on the patched files. This does not prove MathRL will complete 10 steps, but it removes the two policy-worker import blockers already observed on Lyris before the pending OCI-HSG MathRL attempt starts.

## Continuation Update - 2026-06-15 01:54 PDT

The old OCI-HSG MathRL jobs `3290316`, `3290317`, and `3290318` were still pending under account `coreai_dlalgo_llm`, despite the requested `nemotron_n3_post` account. They were cancelled before start.

Launcher/source updates:

| File or repo | Change |
| --- | --- |
| `experiments/eagle3_online/submit_oci_hsg_qwen235b_mathrl_latest_main_20260613.sh` | Default account changed to `nemotron_n3_post`; default `PYTHON_RUNNER_OVERRIDE` changed to `/opt/nemo_rl_venv/bin/python3` for the OCI container. |
| OCI-HSG remote MathRL repo | Already patched with the NVRx version guard and optional ERNIE-VL import guard. |

New OCI-HSG MathRL 10-step jobs submitted from a passing dry-run/preflight:

| Job | Method | Account | Run ID | State at submit |
| --- | --- | --- | --- | --- |
| `3315267` | baseline | `nemotron_n3_post` | `20260615_qwen235b_mathrl_latest_main_guards_n3post_py3` | `PENDING` |
| `3315268` | PARD K3 | `nemotron_n3_post` | `20260615_qwen235b_mathrl_latest_main_guards_n3post_py3` | `PENDING` |
| `3315269` | PARD K5 | `nemotron_n3_post` | `20260615_qwen235b_mathrl_latest_main_guards_n3post_py3` | `PENDING` |

Import-smoke notes:

| Job | Cluster | Result |
| --- | --- | --- |
| `2126217` | Lyris | Base-env smoke proved the NVRx `__version__` crash is gone (`nvrx_min_version=False` rather than an AttributeError), then failed because the base env lacked `modelopt`; this smoke did not use the actual `uv --extra mcore` dependency path. |
| `2126230` | Lyris | `uv --extra mcore` smoke used CPython 3.12 and failed against the lockfile's Python `>=3.13` environment constraint. |
| `3315213` | OCI-HSG | Corrected `uv --extra mcore` smoke using `/opt/nemo_rl_venv/bin/python3`; still pending at `2026-06-15 01:54 PDT`. |

## Continuation Update - 2026-06-15 02:00 PDT

Live SLURM refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL r27 staged-SIF | Lyris | `2124206`, `2124207` | `FAILED 1:0` | Both reached "All workers connected!" and then failed before training because `uv run --frozen --extra mcore` tried to build `transformer-engine` from source and CMake failed with `CUDA_ARCHITECTURES is empty for target "transformer_engine"`. |
| 235B SWE-RL r27 staged-SIF | Lyris | `2124208` | `CANCELLED` | Cancelled after baseline/PARD failed in the identical setup path, to avoid repeating the same pre-step failure. |
| 235B MathRL latest-main import smoke | OCI-HSG | `3315213` | `FAILED 2:0` | Smoke failed before import: `uv` reported no interpreter at `/opt/nemo_rl_venv/bin/python3` inside the submitted runtime. |
| 235B MathRL latest-main | OCI-HSG | `3315267`, `3315268`, `3315269` | `PENDING (Priority)` | Resubmitted under `nemotron_n3_post`; latest start estimate is `2026-06-15T10:20:00` for all three. |
| 235B SWE-RL PARD-2 r16 | OCI-HSG | `3308774` | `PENDING (Priority)` | Submitted under `nemotron_n3_post`; latest start estimate is `2026-06-15T09:30:00`. |

Log hygiene:

| Scope | Result |
| --- | --- |
| Lyris SWE-RL r27 log root | Exact W&B key was redacted from 34 files; exact-key rescan returned clean. |

Updated short answer:

MathRL and SWE-RL are not both fundamentally unusable, but the latest retry attempts are not all clean:

1. 235B SWE-RL has a usable 10-step example on OCI-HSG: `3299487` baseline, `3299489` PARD K5, and `3299491` Eagle-3 K3 all completed 10 steps on `2026-06-14`.
2. 235B SWE-RL PARD-2 still does not have a clean 235B 10-step proof; current OCI job `3308774` is pending and the latest Lyris r27 attempt was stopped after setup failure.
3. 235B MathRL still has no successful latest-main 10-step proof. The patched OCI jobs are pending, but the separate smoke shows the Python runner path still needs correction in that runtime.
4. The cleanest online PARD-2 functional proof remains the Qwen8 OCI-HSG official comparison: `3288181`, `3288182`, and `3288183` completed 10 steps.

## Continuation Update - 2026-06-15 02:03 PDT

The first patched OCI-HSG MathRL set `3315267`, `3315268`, and `3315269` was still pending, but the separate import smoke proved its absolute Python runner override was invalid in the runtime. Those three jobs were cancelled before start.

Launcher correction:

| File | Change |
| --- | --- |
| `experiments/eagle3_online/submit_oci_hsg_qwen235b_mathrl_latest_main_20260613.sh` | `PYTHON_RUNNER_OVERRIDE` default changed from `/opt/nemo_rl_venv/bin/python3` to empty, so `USE_SYSTEM_ENV=true` selects `python` in `submit_nemorl_online_draft_specdec.sh`. |

Replacement OCI-HSG MathRL 10-step jobs:

| Job | Method | Account | Run ID | State |
| --- | --- | --- | --- | --- |
| `3315380` | baseline | `nemotron_n3_post` | `20260615_qwen235b_mathrl_latest_main_guards_n3post_systempy` | `PENDING (Priority)` |
| `3315381` | PARD K3 | `nemotron_n3_post` | `20260615_qwen235b_mathrl_latest_main_guards_n3post_systempy` | `PENDING (Priority)` |
| `3315382` | PARD K5 | `nemotron_n3_post` | `20260615_qwen235b_mathrl_latest_main_guards_n3post_systempy` | `PENDING (Priority)` |

Dry-run for the replacement set shows the driver command now invokes `python .../examples/run_grpo.py` rather than the missing `/opt/nemo_rl_venv/bin/python3` path.

Final poll at `2026-06-15 02:04 PDT`: OCI-HSG SWE-RL PARD-2 `3308774` is still `PENDING (Priority)` with start estimate `2026-06-15T10:00:00`; MathRL `3315380`, `3315381`, and `3315382` are `PENDING (Priority)` with start estimate still `N/A`.

## Continuation Update - 2026-06-15 02:24 PDT

OCI-HSG NeMo-RL state is unchanged from the last poll:

| Track | Jobs | State |
| --- | --- | --- |
| 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` |
| 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, latest start estimate `N/A` |

Lyris standalone temp1/top_p1 refresh completed:

| Scope | Result |
| --- | --- |
| SWE standalone | `90/90` completed. Under RL sampling, suffix decoding is still the strongest standalone method; EAGLE3 generally beats baseline; PARD/PARD2 learned drafters are weak because acceptance is low. |
| Math500 standalone | `3/5` completed. Suffix K32, PARD K5, and EAGLE3 K3 produced `breakdown.json`; baseline `2124147` and official PARD2 `2124150` timed out at 5 hours without final breakdown rows. |

Updated standalone artifacts:

| Artifact | Notes |
| --- | --- |
| `docs/lyris_qwen235b_standalone_temp1rl_20260614.md` | Refreshed queue, live telemetry, and completed breakdown rows. |
| `docs/lyris_qwen235b_standalone_temp1rl_20260614_metrics.csv` | Refreshed metrics from remote `breakdown.json` files. |
| `docs/lyris_qwen235b_standalone_temp1rl_swe_perf_summary_20260615.md` | Updated summary to reflect final `93/95` completion and Math500 timeouts. |

## Continuation Update - 2026-06-15 02:29 PDT

Live server refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | All are under `nemotron_n3_post`. Start estimates: baseline `2026-06-15T08:40:00`, PARD K3 `2026-06-15T10:30:00`, PARD K5 `2026-06-15T11:10:00`. |
| 235B SWE-RL PARD-2 r16 | OCI-HSG | `3308774` | `PENDING (Priority)` | Under `nemotron_n3_post`; start estimate `2026-06-15T06:22:28`. No runtime evidence yet. |
| 235B SWE-RL r27 staged-SIF | Lyris | `2124206`, `2124207`, `2124208` | `FAILED`, `FAILED`, `CANCELLED` | Baseline and PARD both reached Ray worker connection, then failed before GRPO training because `transformer-engine` source build hit `CUDA_ARCHITECTURES is empty for target "transformer_engine"`. PARD-2 was cancelled to avoid repeating the same pre-step setup failure. |

Best 10-step candidates at this point:

| Candidate | Jobs | Result | Notes |
| --- | --- | --- | --- |
| Qwen8 official PARD-2 comparison on OCI-HSG | `3288181`, `3288182`, `3288183` | All `COMPLETED 0:0` | Cleanest small functional example: baseline, static PARD-2, online PARD-2. Parsed 9 post-step metric rows out of the 10-step run. |
| 235B SWE-RL after-prewarm on OCI-HSG | `3299487`, `3299489`, `3299491` | All `COMPLETED 0:0` | Best 235B SWE-RL proof so far: baseline, PARD K5, and EAGLE3 K3 completed. The same matrix's suffix and PARD-2 cells failed early, so this is not yet a clean PARD-2 example. |
| 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | Still pending | No successful 235B MathRL 10-step proof yet. These are the current guarded retries. |

Short answer for the current question: NeMo-RL is not uniformly broken. For a 10-step example, use the OCI-HSG Qwen8 PARD-2 comparison if the goal is a clean online-drafter functional proof, or OCI-HSG 235B SWE-RL baseline/PARD/EAGLE3 if the goal is a 235B SWE-RL run that actually finishes. Avoid using Lyris r27 as an example until the TransformerEngine build/runtime environment is fixed.

## Continuation Update - 2026-06-15 02:35 PDT

Live server refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | All are under `nemotron_n3_post`. Start estimates: baseline `2026-06-15T09:30:00`, PARD K3 `2026-06-15T09:30:00`, PARD K5 `2026-06-15T12:10:00`. |
| 235B SWE-RL PARD-2 r16 | OCI-HSG | `3308774` | `PENDING (Priority)` | Under `nemotron_n3_post`; start estimate `2026-06-15T08:10:00`. No runtime evidence yet. |
| 235B MathRL latest-main | Lyris | `2113812`, `2113813`, `2113814` | `FAILED` | Baseline failed on `nvidia_resiliency_ext.__version__`; PARD K3/K5 failed on missing `transformers.models.ernie4_5_vl_moe`. These were pre-step policy-worker import failures. |
| 235B SWE-RL r27 staged-SIF | Lyris | `2124206`, `2124207`, `2124208` | `FAILED`, `FAILED`, `CANCELLED` | Baseline and PARD failed before step 1 because the `transformer-engine` source build hit `CUDA_ARCHITECTURES is empty for target "transformer_engine"`. PARD-2 was cancelled to avoid a duplicate setup failure. |

Verified 10-step examples from OCI-HSG `sacct`:

| Candidate | Jobs | Result | Notes |
| --- | --- | --- | --- |
| Qwen8 official PARD-2 comparison | `3288181`, `3288182`, `3288183` | All `COMPLETED 0:0` | Best small online-drafter functional proof: baseline, static PARD-2, online PARD-2. |
| 235B SWE-RL after-prewarm | `3299487`, `3299489`, `3299491` | All `COMPLETED 0:0` | Best 235B SWE-RL proof: baseline, PARD K5, and Eagle-3 K3 completed 10 steps. Same matrix's suffix `3299488` and PARD-2 `3299490` failed early. |

Current recommendation: if the goal is a clean 10-step online PARD-2 example, use the Qwen8 OCI-HSG set. If the goal is a 235B NeMo-RL run that demonstrably completes, use the OCI-HSG 235B SWE-RL baseline/PARD/Eagle-3 set. There is still no successful 235B MathRL latest-main 10-step proof, and no successful 235B SWE-RL PARD-2 10-step proof yet.

## Continuation Update - 2026-06-15 02:43 PDT

Lyris SWE-RL r27 TransformerEngine failure was narrowed to the source-build architecture list:

| Evidence | Read |
| --- | --- |
| r27 logs | `uv run --frozen --extra mcore` built TransformerEngine commit `366798ef...` with `-DCMAKE_CUDA_ARCHITECTURES=100`, then failed during CMake generation with `CUDA_ARCHITECTURES is empty for target "transformer_engine"`. |
| TE CMakeLists | The source removes `100` from `CMAKE_CUDA_ARCHITECTURES` and moves it into TE-specific generic/specific arch lists. If `100` is the only input arch, the target-level CUDA architecture list becomes empty. |

Launcher fix:

| File | Change |
| --- | --- |
| `experiments/eagle3_online/submit_lyris_swerl_qwen235b_fullgrpo_specdec_matrix_20260613.sh` | Changed default `NVTE_CUDA_ARCHS` from `100` to `90;100`, and shell-escaped runtime env prefix values with `printf %q` so the semicolon is not interpreted as a command separator. |

Dry-run validation:

| Check | Result |
| --- | --- |
| Local `bash -n` | Passed. |
| Baseline 1-step dry-run | Passed with staged Lyris SIF and staged SWE dataset. The dry-run command now shows `NVTE_CUDA_ARCHS=90\;100`. |
| Remote generated launcher | Contains `export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90;100}"` and the `printf %q` runtime env escaping. |

New minimal Lyris proof run:

| Job | Track | Method | Steps | State | Current read |
| --- | --- | --- | ---: | --- | --- |
| `2126895` | 235B SWE-RL TE-arch fix r28 | baseline | 1 | `PENDING (Resources)` | Submitted on Lyris using the staged SIF, staged SWE dataset, and W&B key parts. Estimated start `2026-06-15T06:34:00`. This is intended to prove the r27 pre-step TE build blocker is fixed before expanding to PARD/PARD-2. |

Follow-up poll at `2026-06-15 02:45 PDT`: `2126895` is now `PENDING (Priority)` with estimated start `2026-06-15T05:56:00`. OCI-HSG pending jobs `3308774`, `3315380`, `3315381`, and `3315382` remain `PENDING (Priority)` with start estimate `N/A`.

## Continuation Update - 2026-06-15 02:48 PDT

Lyris launcher preflight was tightened so method-specific retries do not fail on unrelated assets:

| File | Change |
| --- | --- |
| `experiments/eagle3_online/submit_lyris_swerl_qwen235b_fullgrpo_specdec_matrix_20260613.sh` | Added `NEEDS_SUFFIX_SITE` and `NEEDS_PARD2_SITE`; Arctic suffix checks now run only when `METHODS` includes `suffix`, and PARD-2 overlay checks now run only when `METHODS` includes `pard2`. |

Dry-run checks:

| Dry-run | Result |
| --- | --- |
| baseline-only staged Lyris SIF | Passed with default `PARD2_REJECT_COMPILED_C=true`; command retained `NVTE_CUDA_ARCHS=90\;100`. |
| PARD-2-only staged Lyris SIF | Passed and generated the PARD-2 speculative-config overrides with `NVTE_CUDA_ARCHS=90\;100`. |

Submitted dependent PARD/PARD-2 proof jobs:

| Job | Method | Steps | Dependency | State |
| --- | --- | ---: | --- | --- |
| `2126914` | PARD K5 | 1 | `afterok:2126895` | `PENDING (Dependency)` |
| `2126915` | PARD-2 K1 | 1 | `afterok:2126895` | `PENDING (Dependency)` |

This keeps the PARD/PARD-2 checks queued but prevents them from running if the baseline TE-arch proof `2126895` fails.

Latest poll at `2026-06-15 02:48 PDT`:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T05:56:00` |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `2126895` |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T06:27:13` |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)`, estimates `08:40`, `10:20`, and `12:40` PDT |

Follow-up poll at `2026-06-15 02:49 PDT`: `2126895` remains `PENDING (Priority)`, with estimate updated to `2026-06-15T06:44:00`; `2126914` and `2126915` remain `PENDING (Dependency)` on `afterok:2126895`.

## Continuation Update - 2026-06-15 02:50 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T06:47:00`; no log files created yet. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; no log files created yet. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T08:20:00`; no log files created yet. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)`, estimates `08:40`, `12:40`, and `14:30` PDT; no log files created yet. |

Qwen8 online PARD-2 impact artifact refreshed:

| Artifact | Current read |
| --- | --- |
| `docs/qwen8_pard2_official_online_impact_20260613.md` | Online PARD-2 refit ran for 9 post-step rows and improved acceptance from `1.836` to `2.553`, but generation-worker TPS was `0.9696x` of static PARD-2 and E2E TPS was `0.8087x` of static. Versus baseline, static PARD-2 was `0.6071x` generation-worker TPS and online PARD-2 was `0.5887x`. This validates online refit mechanics but does not show a throughput win in the Qwen8 10-step proof. |

Follow-up poll at `2026-06-15 02:51 PDT`: Lyris `2126895` remains `PENDING (Priority)` with estimate `2026-06-15T05:56:00`; `2126914` and `2126915` remain `PENDING (Dependency)`. OCI-HSG `3308774`, `3315380`, and `3315381` are `PENDING (Priority)` with estimate `2026-06-15T10:20:00`; `3315382` is `PENDING (Priority)` with estimate `2026-06-15T13:40:00`.

## Continuation Update - 2026-06-15 02:53 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T05:56:00`; no log files created yet. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; no log files created yet. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, start estimate `N/A`; no log files created yet. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)`, start estimate `N/A`; no log files created yet. |

Added `docs/online_pard2_impact_current_read_20260615.md` as a concise current-read artifact. It separates the completed Qwen8 online PARD-2 functional evidence from the pending 235B gates. Current conclusion remains: Qwen8 online refit improves acceptance but does not show throughput gain; 235B Math/SWE claims need one of the pending gates to complete with parsed metrics.

## Continuation Update - 2026-06-15 02:57 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T05:56:00`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T08:10:00`; no runtime log evidence yet. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; estimates `10:30`, `11:10`, and `11:10` PDT; no runtime log evidence yet. |

Current answer to the 10-step example question:

| Candidate | Jobs | Result | Use |
| --- | --- | --- | --- |
| Qwen8 official PARD-2 comparison on OCI-HSG | `3288181`, `3288182`, `3288183` | all `COMPLETED 0:0` for `max_steps=10` | Cleanest online-drafter functional example: baseline, static PARD-2, and online PARD-2. |
| 235B SWE-RL after-prewarm on OCI-HSG | `3299487`, `3299489`, `3299491` | baseline, PARD K5, and Eagle-3 K3 all `COMPLETED 0:0` for `max_steps=10` | Best 235B NeMo-RL SWE example that actually finishes. Do not use the same matrix's suffix `3299488` or PARD-2 `3299490`; both failed early. |
| 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | still pending | No successful 235B MathRL 10-step proof yet. |

Short read: NeMo-RL is not uniformly broken. SWE-RL has a usable 235B 10-step path for baseline/PARD/Eagle-3 on OCI-HSG, but 235B PARD-2 and MathRL are still waiting on the current gated retries. If the goal is simply a compact online-drafter example, the Qwen8 official PARD-2 comparison is the safest example to reuse.

## Continuation Update - 2026-06-15 02:59 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T05:56:00`. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T08:17:55`. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; estimates `10:30`, `11:10`, and `14:30` PDT. |

Because none of the 235B gates have started, there are still no new runtime logs or step markers to inspect. I used the queue wait to revalidate the local contract surface:

| Validation | Result | Artifact |
| --- | --- | --- |
| Shared NeMo-RL online SpecDec launcher contract | `PASS` | `docs/nemorl_online_specdec_contract_validation_20260615.md` |
| PARD/PARD-2 online source bundle contract | `PASS` | `docs/nemorl_pard_source_bundle_validation_20260615.md` |
| Qwen235B SWE/Math standalone launcher contract | `PASS` | `docs/qwen235b_swe_math_launcher_contract_validation_20260615.md` |
| SWE-RL Full-GRPO SpecDec launcher contract | `PASS` | `docs/swerl_fullgrpo_specdec_launcher_contract_validation_20260615.md` |
| PARD-2 target-feature alignment test syntax | `PASS` via `py_compile` | Runtime execution was not possible in the local Python because `torch` is not installed. |

This strengthens the pre-submit/static evidence but does not replace the pending runtime proof. The remaining decisive gates are still completed steps and parsed metrics from `2126895`/`2126914`/`2126915`, `3308774`, and `3315380`-`3315382`.

## Continuation Update - 2026-06-15 03:02 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T05:56:00`; no runtime logs beyond job-id markers. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimate moved later to `2026-06-15T12:20:00`; no runtime logs beyond job-id marker. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; all currently estimated `2026-06-15T14:30:00`; no runtime logs yet. |

Comparable completed 235B SWE-RL evidence was regenerated from the already fetched OCI-HSG logs:

| Artifact | Current read |
| --- | --- |
| `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_20260615.md` | Baseline, PARD K5, and Eagle-3 K3 completed 10-step jobs. Step>=2 throughput shows no speculative decoding win in this completed set: PARD K5 is `0.2639x` E2E / `0.2679x` generation-worker vs baseline, and Eagle-3 K3 is `0.4827x` E2E / `0.4857x` generation-worker vs baseline. |
| `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_step_metrics_20260615.csv` | Step-level parsed metrics for `3299487`, `3299489`, and `3299491`. |
| `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_stepge2_20260615.csv` | Step>=2 aggregate summary for the same completed jobs. |

This completed 235B SWE-RL set is useful as a runnable 10-step example and comparison baseline, but not as a PARD-2 performance answer. The PARD-2 performance gate still depends on the pending PARD-2 jobs.

## Continuation Update - 2026-06-15 03:06 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T05:46:00`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T14:20:00`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; all currently estimated `2026-06-15T14:20:00`; runtime log directories are present but empty. |

No pending gate has started since the previous update, so there is no new NeMo-RL failure evidence from these jobs. The latest completed 235B SWE-RL evidence is still the OCI-HSG 10-step set `3299487` baseline, `3299489` PARD K5, and `3299491` Eagle-3 K3; the decisive 235B PARD-2 and MathRL gates remain queued.

## Continuation Update - 2026-06-15 03:08 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start moved to `2026-06-15T06:15:00`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, current start estimate `N/A`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`, current start estimate `N/A`; runtime log directories are still empty. |

No new runtime issue is visible from these gates. The Lyris TE-arch fix and the OCI-HSG 235B PARD-2/MathRL retries are still waiting on scheduler allocation.

## Continuation Update - 2026-06-15 03:10 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T06:01:00`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T15:50:00`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; all currently estimated `2026-06-15T15:50:00`; runtime log directories are still empty. |

There is still no new runtime failure or success evidence from the pending 235B gates. The next decisive signal remains whether Lyris `2126895` clears the TransformerEngine architecture build issue and whether the OCI-HSG PARD-2/MathRL retries produce step metrics once allocated.

## Continuation Update - 2026-06-15 03:12 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start still `2026-06-15T06:01:00`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, current start estimate `N/A`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`, current start estimate `N/A`; runtime log directories are still empty. |

No pending 235B gate has started or produced logs since the previous poll. The latest completed runnable 235B evidence remains the OCI-HSG SWE-RL 10-step baseline/PARD/Eagle-3 set, while 235B PARD-2 and MathRL still need scheduler allocation before they can produce decisive metrics.

## Continuation Update - 2026-06-15 03:14 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T05:56:00`; `scontrol` reports `Reason=Priority`, `Priority=76788`, `Account=coreai_dlalgo_llm`, `QOS=user-restrictions`, `NumNodes=16-16`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T15:20:00`; `scontrol` reports `Reason=Priority`, `Priority=127646`, `Account=nemotron_n3_post`, `QOS=normal`, `NumNodes=16`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; all currently estimated `2026-06-15T15:20:00`; `sprio` reports priority `127593` for each; runtime log directories are still empty. |

The pending gates are still scheduler-priority limited, not failing in runtime setup. No additional submit/fix action is justified until at least one gate allocates nodes and produces logs.

## Continuation Update - 2026-06-15 03:18 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start `2026-06-15T06:08:00`; `scontrol` reports `Reason=Priority`, `Priority=76792`, `Account=coreai_dlalgo_llm`, `QOS=user-restrictions`, `NumNodes=16-16`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T15:00:00`; `scontrol` reports `Reason=Priority`, `Priority=127646`, `Account=nemotron_n3_post`, `QOS=normal`, `NumNodes=16`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; all currently estimated `2026-06-15T15:00:00`; `sprio` reports priority `127594` for each; runtime log directories are still empty. |

OCI-HSG user queue sample also shows `3308725` pending under `nemotron_n3_post` with 16 nodes, so the current gate remains scheduler allocation for the account rather than a new NeMo-RL runtime failure.

## Continuation Update - 2026-06-15 03:20 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start still `2026-06-15T06:08:00`; `scontrol` reports `Reason=Priority`, `Priority=76792`, `Account=coreai_dlalgo_llm`, `QOS=user-restrictions`, `NumNodes=16-16`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start moved earlier to `2026-06-15T09:20:00`; `scontrol` reports `Reason=Priority`, `Priority=127646`, `Account=nemotron_n3_post`, `QOS=normal`, `NumNodes=16`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; current estimates are `2026-06-15T12:50:00`, `2026-06-15T13:20:00`, and `2026-06-15T14:20:00`; `sprio` reports priority `127594` for each; runtime log directories are still empty. |

No pending 235B gate has allocated nodes yet, but the OCI-HSG estimates improved materially. The next likely decisive runtime signal is `3308774` around `09:20 PDT`, followed by the MathRL latest-main retries later in the day.

## Continuation Update - 2026-06-15 03:22 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start still `2026-06-15T06:08:00`; `scontrol` reports `Reason=Priority`, `Priority=76792`, `Account=coreai_dlalgo_llm`, `QOS=user-restrictions`, `NumNodes=16-16`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, current start estimate returned to `N/A`; `scontrol` reports `Reason=Priority`, `Priority=127647`, `Account=nemotron_n3_post`, `QOS=normal`, `NumNodes=16`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`, current start estimates `N/A`; `sprio` reports priority `127594` for each; runtime log directories are still empty. |

No new runtime logs have appeared. The current evidence still points to scheduler priority wait rather than NeMo-RL runtime failure.

## Continuation Update - 2026-06-15 03:24 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start still `2026-06-15T06:08:00`; `scontrol` reports `Reason=Priority`, `Priority=76795`, `Account=coreai_dlalgo_llm`, `QOS=user-restrictions`, `NumNodes=16-16`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T11:30:00`; `scontrol` reports `Reason=Priority`, `Priority=127647`, `Account=nemotron_n3_post`, `QOS=normal`, `NumNodes=16`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; current estimates are `2026-06-15T11:30:00`, `2026-06-15T12:20:00`, and `2026-06-15T12:20:00`; `sprio` reports priority `127594` for each; runtime log directories are still empty. |

The scheduler estimates are fluctuating, but no 235B gate has reached runtime yet. The next actionable event remains the first allocation and creation of driver/Ray logs.

## Continuation Update - 2026-06-15 03:26 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start still `2026-06-15T06:08:00`; `scontrol` reports `Reason=Priority`, `Priority=76795`, `Account=coreai_dlalgo_llm`, `QOS=user-restrictions`, `NumNodes=16-16`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, current start estimate `N/A`; `scontrol` reports `Reason=Priority`, `Priority=127647`, `Account=nemotron_n3_post`, `QOS=normal`, `NumNodes=16`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; current start estimates `N/A`; `sprio` reports priority `127595` for each; runtime log directories are still empty. |

No runtime/Ray/driver logs have been created by the pending gates. The observed changes are scheduler-estimate churn only.

## Continuation Update - 2026-06-15 03:31 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start still `2026-06-15T06:08:00`; `scontrol` reports `Reason=Priority`, `Priority=76799`, `Account=coreai_dlalgo_llm`, `QOS=user-restrictions`, `NumNodes=16-16`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start `2026-06-15T14:00:00`; `scontrol` reports `Reason=Priority`, `Priority=127647`, `Account=nemotron_n3_post`, `QOS=normal`, `NumNodes=16`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; all currently estimated `2026-06-15T14:00:00`; runtime log directories are still empty. |

No pending gate has started; only the OCI-HSG scheduler estimate changed back from `N/A` to `14:00 PDT`.

## Continuation Update - 2026-06-15 03:33 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start moved to `2026-06-15T05:56:00`; `scontrol` reports `Reason=Priority`, `Priority=76802`, `Account=coreai_dlalgo_llm`, `QOS=user-restrictions`, `NumNodes=16-16`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start moved to `2026-06-15T16:40:00`; `scontrol` reports `Reason=Priority`, `Priority=127647`, `Account=nemotron_n3_post`, `QOS=normal`, `NumNodes=16`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; all currently estimated `2026-06-15T16:40:00`; runtime log directories are still empty. |

No pending gate has started; only scheduler estimates changed.

## Continuation Update - 2026-06-15 03:38 PDT

Live poll:

| Track | Jobs | State |
| --- | --- | --- |
| Lyris 235B SWE-RL TE-arch proof | `2126895` | `PENDING (Priority)`, estimated start still `2026-06-15T05:56:00`; `scontrol` reports `Reason=Priority`, `Priority=76806`, `Account=coreai_dlalgo_llm`, `QOS=user-restrictions`, `NumNodes=16-16`; only `latest_235b_scale_gen_job_id.txt` exists under the log root. |
| Lyris 235B SWE-RL dependent PARD/PARD-2 | `2126914`, `2126915` | `PENDING (Dependency)`, waiting on `afterok:2126895`; only job-id marker files exist under the log roots. |
| OCI-HSG 235B SWE-RL PARD-2 | `3308774` | `PENDING (Priority)`, estimated start moved to `2026-06-15T11:20:00`; `scontrol` reports `Reason=Priority`, `Priority=127648`, `Account=nemotron_n3_post`, `QOS=normal`, `NumNodes=16`; only the job-id marker exists under the log root. |
| OCI-HSG 235B MathRL latest-main | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` under `nemotron_n3_post`; all currently estimated `2026-06-15T11:30:00`; runtime log directories are still empty. |

The pending 235B gates are still scheduler-limited, not failing inside NeMo-RL. The confirmed 10-step choices remain the OCI-HSG Qwen8 official PARD-2 comparison and the OCI-HSG 235B SWE-RL baseline/PARD/Eagle-3 completed set.

## Continuation Update - 2026-06-15 04:43 PDT

Live gate refresh and sacct verification:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Under `nemotron_n3_post`; start estimate `2026-06-15T11:50:00`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Under `nemotron_n3_post`; start estimates are `12:40`, `13:20`, and `13:20`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate `2026-06-15T08:01:00`; dependent PARD/PARD-2 jobs remain blocked on `afterok:2126895`. |
| 235B MathRL latest-main retry3 | Lyris | `2113812`, `2113813`, `2113814` | `FAILED` | sacct confirms all three terminally failed before step evidence; these are not usable 10-step examples. |

Verified usable NeMo-RL short examples remain unchanged:

| Candidate | Jobs | sacct result | Use |
| --- | --- | --- | --- |
| Qwen8 official PARD-2 comparison | `3288181`, `3288182`, `3288183` | All `COMPLETED 0:0` | Best small online-drafter functional proof. |
| 235B SWE-RL after-prewarm | `3299487`, `3299489`, `3299491` | All `COMPLETED 0:0` | Best 235B SWE-RL baseline/PARD/Eagle-3 short-run proof. |

Important distinction: standalone MATH500/vLLM benchmark artifacts contain
successful Math decoding rows, but they are not NeMo-RL MathRL training runs.
Do not cite those standalone MATH500 rows as proof that 235B MathRL reaches
10 GRPO steps.

## Continuation Update - 2026-06-15 04:48 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Under `nemotron_n3_post`; start estimate moved to `2026-06-15T16:10:00`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Under `nemotron_n3_post`; all three now estimate `2026-06-15T16:40:00`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved to `2026-06-15T08:28:00`; dependent PARD/PARD-2 jobs remain blocked on `afterok:2126895`. |

`PARSE_READY_METRICS=true` was enabled for the poll, but `READY=0`: no watched
stdout or runtime logs exist yet, so no new step metrics were fetched.

## Continuation Update - 2026-06-15 04:50 PDT

Live gate refresh and queue context:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate improved to `2026-06-15T12:40:00`; stdout still missing; `sprio` priority `128514`. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate `2026-06-15T16:40:00`; stdout still missing; `sprio` priority `128461`. |
| 235B MathRL latest-main PARD | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved to `2026-06-15T18:00:00`; stdout still missing; `sprio` priority `128461`. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved to `2026-06-15T08:13:00`; stdout still missing; `squeue` priority `76854`. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; no stdout. |

`READY=0` again, so no watched gate has started and there are no new step
metrics or runtime failures to triage.

## Continuation Update - 2026-06-15 04:58 PDT

Live gate refresh with scheduler priority captured in the canonical snapshot:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Under `nemotron_n3_post`; start estimate `2026-06-15T18:00:00`; priority `128514`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Under `nemotron_n3_post`; start estimate `2026-06-15T18:00:00`; priority `128462`; stdout still missing. |
| 235B MathRL latest-main PARD | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Under `nemotron_n3_post`; start estimate `2026-06-15T18:00:00`; priorities `128461` and `128461`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate still `2026-06-15T08:13:00`; priority `76861`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; no stdout. |

`SYNC_CANONICAL_DOCS=true PARSE_READY_METRICS=true` returned `READY=0`, so
these are still scheduler waits rather than new NeMo-RL runtime failures. The
monitoring scripts now propagate SLURM priority into the active snapshot,
history, change report, and runtime report.

## Continuation Update - 2026-06-15 05:01 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved slightly earlier to `2026-06-15T17:50:00`; priority `128514`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimate remains `2026-06-15T18:00:00`; priorities `128462`, `128461`, and `128461`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T08:13:00`; priority `76861`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; no stdout. |

`READY=0`; no watched gate has allocated nodes or emitted runtime logs yet.

## Continuation Update - 2026-06-15 05:03 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate returned to `2026-06-15T18:00:00`; priority `128514`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimate remains `2026-06-15T18:00:00`; priorities `128462`, `128461`, and `128461`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T07:55:00`; priority `76865`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; no stdout. |

`READY=0`; there are still no new step metrics or runtime failures to triage.

## Continuation Update - 2026-06-15 05:06 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate changed to `N/A`; priority `128515`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimate changed to `N/A`; priority `128463` for all three; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:55:00`; priority `76865`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; no stdout. |

`READY=0`; the current evidence is still scheduler wait, not a NeMo-RL failure.

## Continuation Update - 2026-06-15 05:09 PDT

Live gate refresh after adding `sacct` fallback parsing to the monitor:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate returned to `2026-06-15T17:20:00`; priority `128514`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimate returned to `2026-06-15T17:20:00`; priorities `128462`, `128461`, and `128461`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:55:00`; priority `76868`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; no stdout. |

`READY=0`; no watched gate has produced runtime logs. The monitor now also uses
top-level `sacct` rows as fallback evidence if a gate leaves `squeue`.

## Continuation Update - 2026-06-15 05:11 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate changed back to `N/A`; priority `128516`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimate changed back to `N/A`; priority `128463` for all three; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:55:00`; priority `76868`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; no stdout. |

`READY=0`; still no runtime logs or step metrics to parse.

## Continuation Update - 2026-06-15 05:15 PDT

Live gate refresh and direct scheduler context:

| Track | Cluster | Jobs | State | Scheduler detail |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Account `nemotron_n3_post`, partition `batch`, QOS `normal`, 16 nodes, 4h limit, start estimate `2026-06-15T18:20:00`, priority `128516`; stdout missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Account `nemotron_n3_post`, partition `batch`, QOS `normal`, 32 nodes each, 4h limit, start estimate `2026-06-15T18:20:00`, priority `128464`; stdout missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Account `coreai_dlalgo_llm`, partition `gb200`, QOS `user-restrictions`, 16 nodes, 5h limit, start estimate `2026-06-15T07:55:00`, priority `76872`; stdout missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Both still wait on `afterok:2126895`, partition `gb200`, QOS `user-restrictions`, 16 nodes, 5h limit, priority `76754`; stdout missing. |

`SYNC_CANONICAL_DOCS=true PARSE_READY_METRICS=true MAX_POLLS=1` returned
`READY=0`: no watched gate has allocated nodes or produced runtime logs. The
current evidence is scheduler wait, not a fresh NeMo-RL runtime failure. No
walltime reduction was applied; the observed blocker is priority/dependency,
and shortening these already-short proof jobs risks timeout without proving the
runtime path.

## Continuation Update - 2026-06-15 05:19 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate improved from `2026-06-15T18:20:00` to `2026-06-15T12:40:00`; priority `128516`; stdout still missing. |
| 235B MathRL latest-main baseline/PARD K3 | OCI-HSG | `3315380`, `3315381` | `PENDING (Priority)` | Start estimate improved from `2026-06-15T18:20:00` to `2026-06-15T12:40:00`; priority `128464`; stdout still missing. |
| 235B MathRL latest-main PARD K5 | OCI-HSG | `3315382` | `PENDING (Priority)` | Start estimate improved from `2026-06-15T18:20:00` to `2026-06-15T13:30:00`; priority `128464`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:55:00`; priority increased to `76875`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0` again: no watched gate has produced stdout or runtime logs yet. The
important change is queue movement on OCI-HSG, not new runtime evidence.

## Continuation Update - 2026-06-15 05:21 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate returned from `2026-06-15T12:40:00` to `N/A`; priority increased to `128517`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates returned to `N/A`; priority remains `128464`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:55:00`; priority `76875`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The OCI-HSG
start estimate is fluctuating while the jobs remain `PENDING/Priority`; this is
still scheduler wait, not a Nemo-RL runtime failure.

## Continuation Update - 2026-06-15 05:23 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate returned from `N/A` to `2026-06-15T17:40:00`; priority `128517`; stdout still missing. |
| 235B MathRL latest-main baseline/PARD K3 | OCI-HSG | `3315380`, `3315381` | `PENDING (Priority)` | Start estimate returned from `N/A` to `2026-06-15T17:40:00`; priority `128464`; stdout still missing. |
| 235B MathRL latest-main PARD K5 | OCI-HSG | `3315382` | `PENDING (Priority)` | Start estimate returned from `N/A` to `2026-06-15T18:20:00`; priority `128464`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:55:00`; priority increased to `76879`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. This continues
to be scheduler wait rather than Nemo-RL runtime evidence.

## Continuation Update - 2026-06-15 05:26 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved from `2026-06-15T17:40:00` to `2026-06-15T18:20:00`; priority `128517`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimate moved to `2026-06-15T18:30:00`; priority increased to `128465`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:55:00`; priority `76879`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The only
material change is scheduler estimate movement and a small MathRL priority
increase.

## Continuation Update - 2026-06-15 05:28 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate remains `2026-06-15T18:20:00`; priority `128517`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimate moved from `2026-06-15T18:30:00` to `2026-06-15T18:40:00`; priority `128465`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved from `2026-06-15T07:55:00` to `2026-06-15T08:13:00`; priority increased to `76882`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. There is no
new Nemo-RL runtime failure or metric evidence yet.

## Continuation Update - 2026-06-15 05:31 PDT

Live gate refresh and artifact scan:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate returned from `2026-06-15T18:20:00` to `N/A`; priority increased to `128518`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates returned from `2026-06-15T18:40:00` to `N/A`; priority remains `128465`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T08:13:00`; priority `76882`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. A local
artifact scan for `MathRL`, `3315380`, `3315381`, and `3315382` found pending
history, dry-run artifacts, and launch metadata only; it did not find completed
235B MathRL step metrics.

## Continuation Update - 2026-06-15 05:35 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate remains `N/A`; priority `128518`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates remain `N/A`; priority increased to `128466` for all three; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T08:13:00`; priority increased to `76885`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The OCI-HSG
start estimates continue to fluctuate between concrete times and `N/A`, but the
observed blocker is still scheduler priority/dependency rather than a new
NeMo-RL runtime failure.

## Continuation Update - 2026-06-15 05:38 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate remains `N/A`; priority `128518`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates remain `N/A`; priority `128465` for all three; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T08:13:00`; priority increased to `76889`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The only
material movement since 05:35 PDT is scheduler priority drift; there is still no
new NeMo-RL runtime failure or completed-step evidence.

## Continuation Update - 2026-06-15 05:41 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate returned to `2026-06-15T14:10:00`; priority `128519`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate returned to `2026-06-15T14:10:00`; priority `128466`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates returned to `2026-06-15T18:10:00`; priority `128466`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T08:04:00`; priority `76889`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. A focused local
artifact scan also found completed Qwen30 PARD2 online/static comparison
evidence in `docs/qwen30ba3b_pard2_online_long_output_win2048_comparison_20260611.md`.
That strengthens the online-training performance read, but it does not replace
the missing 235B SWE-RL PARD-2 and 235B MathRL runtime proofs.

## Continuation Update - 2026-06-15 05:44 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate changed back to `N/A`; priority `128519`; stdout still missing. |
| 235B MathRL latest-main | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates changed back to `N/A`; priority `128466` for all three; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T07:48:00`; priority increased to `76892`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. This remains
scheduler wait, not a fresh NeMo-RL runtime failure.

## Continuation Update - 2026-06-15 05:47 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate returned to `2026-06-15T14:20:00`; priority `128519`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate returned to `2026-06-15T17:30:00`; priority `128466`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates returned to `2026-06-15T18:20:00`; priority `128466`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:48:00`; priority increased to `76896`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The OCI-HSG
start estimates continue to oscillate, but there is still no runtime evidence
or failure to triage.

## Continuation Update - 2026-06-15 05:50 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T13:30:00`; priority `128519`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T14:20:00`; priority `128466`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved to `2026-06-15T17:30:00` and `2026-06-15T18:00:00`; priority `128466`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:48:00`; priority `76896`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `76754`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The queue
estimates improved on OCI-HSG, but the remaining proof gap is unchanged.

## Continuation Update - 2026-06-15 05:54 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved later to `2026-06-15T18:00:00`; priority `128520`; stdout still missing. |
| 235B MathRL latest-main baseline/PARD K3/PARD K5 | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates are now all `2026-06-15T18:10:00`; priority `128467`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:28:00`; priority `77429`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. This is still
scheduler wait, not a new MathRL or SWE-RL runtime failure.

## Continuation Update - 2026-06-15 05:58 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate changed to `N/A`; priority `128520`; stdout still missing. |
| 235B MathRL latest-main baseline/PARD K3/PARD K5 | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates changed to `N/A`; priority `128468`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:28:00`; priority increased to `77433`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The OCI-HSG
estimates are oscillating again, but this is still scheduler wait with no new
runtime failure to triage.

## Continuation Update - 2026-06-15 06:01 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate returned to `2026-06-15T14:20:00`; priority `128521`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate returned to `2026-06-15T14:20:00`; priority `128468`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates returned to `2026-06-15T18:20:00` and `2026-06-15T18:30:00`; priority `128468`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:28:00`; priority `77433`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. Older MathRL
attempts were also audited: `3290316`-`3290318` and `3315267`-`3315269` were
cancelled before runtime, while Lyris retry3 `2113812`-`2113814` failed during
isolated policy worker creation on missing
`transformers.models.ernie4_5_vl_moe`.

## Continuation Update - 2026-06-15 06:08 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T10:40:00`; priority `128521`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate moved to `2026-06-15T14:50:00`; priority `128469`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved to `2026-06-15T18:50:00` and `2026-06-15T19:00:00`; priority `128469`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:28:00`; priority increased to `77440`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. This remains
scheduler wait, with no new NeMo-RL runtime failure to triage.

## Continuation Update - 2026-06-15 06:11 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved later to `2026-06-15T16:30:00`; priority `128522`; stdout still missing. |
| 235B MathRL latest-main baseline/PARD K3/PARD K5 | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates are now all `2026-06-15T18:20:00`; priority `128469`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved later to `2026-06-15T08:10:00`; priority `77440`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The latest
queue movement is estimate drift only; there is still no new runtime failure
and no step metric to parse.

## Continuation Update - 2026-06-15 06:14 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T14:40:00`; priority `128522`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate moved to `2026-06-15T18:40:00`; priority `128469`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved to `2026-06-15T18:50:00`; priority `128469`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T08:10:00`; priority increased to `77443`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. There is still
no new NeMo-RL runtime failure and no metric parsing target.

## Continuation Update - 2026-06-15 06:16 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate remains `2026-06-15T14:40:00`; priority `128522`; stdout still missing. |
| 235B MathRL latest-main baseline/PARD K3/PARD K5 | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved earlier to `2026-06-15T14:40:00`; priority increased to `128470`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved slightly later to `2026-06-15T08:16:00`; priority `77443`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. No runtime
triage or metric extraction is possible yet.

## Continuation Update - 2026-06-15 06:19 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved later to `2026-06-15T15:40:00`; priority `128522`; stdout still missing. |
| 235B MathRL latest-main baseline/PARD K3/PARD K5 | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved later to `2026-06-15T18:30:00`; priority `128470`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T08:16:00`; priority increased to `77447`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. This remains
scheduler wait only.

## Continuation Update - 2026-06-15 06:22 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T14:40:00`; priority increased to `128523`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate remains `2026-06-15T18:30:00`; priority `128470`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved slightly later to `2026-06-15T18:40:00`; priority `128470`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T08:16:00`; priority `77447`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. This is still
scheduler wait rather than a new MathRL or SWE-RL runtime failure.

## Continuation Update - 2026-06-15 06:25 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved later to `2026-06-15T18:40:00`; priority `128523`; stdout still missing. |
| 235B MathRL latest-main baseline/PARD K3/PARD K5 | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates are now all `2026-06-15T18:40:00`; priority `128470`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T07:55:00`; priority increased to `77450`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The only new
signal is scheduler estimate movement; there is still no NeMo-RL runtime
failure or step metric to triage.

## Continuation Update - 2026-06-15 06:28 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T14:40:00`; priority `128523`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T14:40:00`; priority increased to `128471`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates remain `2026-06-15T18:40:00`; priority increased to `128471`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T07:00:00`; priority increased to `77454`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. This remains
scheduler wait only; there is no new NeMo-RL runtime failure or parsed metric.

## Continuation Update - 2026-06-15 06:31 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate remains `2026-06-15T14:40:00`; priority increased to `128524`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate moved later to `2026-06-15T18:40:00`; priority `128471`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved later to `2026-06-15T18:50:00`; priority `128471`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T07:00:00`; priority `77454`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. This remains
scheduler wait only; no metric parsing target is available.

## Continuation Update - 2026-06-15 06:33 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T13:00:00`; priority `128524`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T13:00:00`; priority `128471`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved earlier to `2026-06-15T16:20:00`; priority `128471`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T06:40:00`; priority increased to `77457`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. The queue
estimates improved, but there is still no runtime failure or step metric.

## Continuation Update - 2026-06-15 06:36 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate remains `2026-06-15T13:00:00`; priority `128524`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate moved later to `2026-06-15T14:20:00`; priority increased to `128472`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates remain `2026-06-15T16:20:00`; priority increased to `128472`; stdout still missing. |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `PENDING (Priority)` | Start estimate remains `2026-06-15T06:40:00`; priority `77457`; stdout still missing. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |

`READY=0`: no watched gate has produced stdout or runtime logs. Lyris baseline
is still estimated near-term, but no log file has appeared yet.

## Continuation Update - 2026-06-15 06:44 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `RUNNING` | Started `2026-06-15T06:41:02`; stdout present; `19` runtime logs; `ray-driver.log` fetched and redacted; no parsed step metrics yet. Current driver tail is still in setup/TransformerEngine build. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved later to `2026-06-15T19:00:00`; priority increased to `128525`; stdout still missing. |
| 235B MathRL latest-main baseline/PARD K3/PARD K5 | OCI-HSG | `3315380`, `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved to `2026-06-15T19:00:00`; priorities `128472`; stdout still missing. |

`READY=1`: Lyris baseline has runtime logs, but there is still no completed
step or throughput metric. This proves the job allocated and entered setup, not
that NeMo-RL reached step 1.

## Continuation Update - 2026-06-15 06:47 PDT

Live gate refresh:

| Track | Cluster | Jobs | State | Current read |
| --- | --- | --- | --- | --- |
| 235B SWE-RL TE-arch proof | Lyris | `2126895` | `RUNNING` | Still running from `2026-06-15T06:41:02`; stdout present; `19` runtime logs. Fetched `ray-driver.log` still ends at `Building transformer-engine`; no step metrics yet. |
| 235B SWE-RL dependent PARD/PARD-2 | Lyris | `2126914`, `2126915` | `PENDING (Dependency)` | Still waiting on `afterok:2126895`; priority `77284`; stdout missing. |
| 235B SWE-RL PARD-2 proof | OCI-HSG | `3308774` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T14:40:00`; priority `128525`; stdout still missing. |
| 235B MathRL latest-main baseline | OCI-HSG | `3315380` | `PENDING (Priority)` | Start estimate moved earlier to `2026-06-15T18:40:00`; priority `128472`; stdout still missing. |
| 235B MathRL latest-main PARD K3/K5 | OCI-HSG | `3315381`, `3315382` | `PENDING (Priority)` | Start estimates moved to `2026-06-15T19:10:00`; priority `128472`; stdout still missing. |

`READY=1`: `2126895` remains the only runtime gate. The fetcher now handles
`STDOUT_STATUS=present size=...`, uses the `login-lyris` alias for Lyris
fetches, and treats no-step-metrics-yet as a live empty result instead of a
failed parser run.
