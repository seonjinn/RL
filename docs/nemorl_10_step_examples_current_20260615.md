# NeMo-RL 10-Step Examples Current Read - 2026-06-15

This note separates usable 10-step examples from failed/running 235B proof
gates.

## Verification Addendum - 2026-06-15 21:03 PDT

Short read:

- No new completed 10-step PARD-2 proof yet, but OCI MathRL Eagle-3 r2
  `3331856` has now passed `SETUP COMPLETE` and progressed through
  `Step 1/5`, `Step 2/5`, `Step 3/5`, and into `Step 4/5` generation with
  the requested generation-bound sampling shape (`1024` min/max tokens,
  `temperature=1.0`, `top_p=1.0`, `top_k=-1`).
- OCI MathRL r2 baseline `3331855` failed before Step because
  `policy.model_name` was still the HF repo id and Megatron Bridge could not
  find local `.safetensors`. r2 online PARD-2 `3331858` failed from HF Hub
  `429 Too Many Requests` during vLLM actor creation. I patched the launcher
  so both target model and tokenizer use the local Qwen3-235B snapshot and the
  Megatron checkpoint dir is method-specific.
- Replacement MathRL r3 jobs are queued under `nemotron_n3_post`: baseline
  `3332282` and online PARD-2 `3332283`, both `PENDING|Priority` at
  `21:00:04 PDT`. Static PARD-2 r2 `3331857` is still `RUNNING`; it has HF
  `429` warnings but no fatal exit yet.
- Lyris r39 `2133251` failed after Ray reached `64/64` actors because
  `research/template_project/pyproject.toml` still required Python
  `>=3.13.13`. The SWE-RL submit path now lowers that workspace member to
  `>=3.13.9`. r40 `2133287` then failed before Ray worker startup because
  pyxis could not mount node-local cache override path
  `/tmp/uv_cache_swerl_r40` on worker nodes. I submitted r41 `2133292`
  without node-local override mounts; it is `PENDING` at `21:03:28 PDT`.

## Verification Addendum - 2026-06-15 20:43 PDT

Short read:

- No new completed 10-step or completed PARD-2 proof yet. The newest useful
  evidence is that OCI MathRL generation-bound r2 is fully running:
  baseline `3331855`, Eagle-3 `3331856`, static PARD-2 `3331857`, and online
  PARD-2 `3331858`.
- r2 has the intended generation-bound/RL sampling shape
  (`max_new_tokens=min_tokens=1024`, `temperature=1.0`, `top_p=1.0`,
  `top_k=-1`) and the fix for the r1 training crash:
  `context_parallel_size=1`, `sequence_packing.enabled=false`. PARD-2 jobs now
  use the shared Lustre actor venv root instead of node-local `/opt`. The logs
  are still in setup/model-init, so there is no new Step-complete evidence yet.
- Lyris is authenticated and usable again. I patched the Lyris SWE-RL launcher
  so persistent uv/pip/torch cache envs reach the Ray driver command, fixed the
  Lyris default repo/container paths, and validated the contract. Baseline
  1-step smoke r37 `2133224` failed quickly because Python `3.13.13` is not
  available in the container-managed installs; r38 `2133226` is now submitted
  with Python `3.13.9` plus explicit Lustre cache env exports, but also failed
  quickly on uv extraction disk quota. r39 `2133251` is now running with
  Python `3.13.9` and node-local `/tmp` cache/venv paths; it has passed Ray
  head bootstrap (`Prepared/Installed 57 packages`, `ray start --head`).
- Lyris standalone temp=1/top_p=1 result status is unchanged: SWE standalone
  completed; Math500 baseline and PARD-2 timed out at 5 hours.

## Verification Addendum - 2026-06-15 20:30 PDT

Short read:

- No new completed 10-step PARD-2 proof yet. The generation-bound MathRL r1
  baseline `3330755` did reach `Step 1/5` and completed generation/logprob/
  advantage computation, but failed at policy training in packed loss input
  construction with tensor length `1312` vs `1168`.
- The r1 Eagle-3/PARD-2 jobs were cancelled and replaced by r2:
  `3331855` baseline, `3331856` Eagle-3 K3, `3331857` static PARD-2 K3, and
  `3331858` online PARD-2 K3. r2 keeps the requested generation-bound shape
  (`1024` generated tokens, `temperature=1.0`, `top_p=1.0`, `top_k=-1`) and
  forces `CP=1`, `sequence_packing=false` for all methods to avoid the r1
  packed-loss mismatch. At `20:31:17 PDT`, baseline, Eagle-3, and static
  PARD-2 are `RUNNING`; online PARD-2 is `PENDING|Priority` with estimated
  start `2026-06-15T20:43:08`.
- PARD-2 r2 also uses a shared Lustre actor venv root plus `READY_ENV_BUILDER`
  markers, fixing the r1 static PARD-2 issue where serialized venv creation
  under node-local `/opt/ray_venvs` made other nodes fail with missing
  `bin/python`.
- SWE-RL r28 `3331533` was cancelled before setup because Ray startup looped
  at `0/64` actors with missing
  `/tmp/nemo_rl_ray_3331533_3.12.13_2.54.0/bin/ray`. This is not a Step-1
  training failure; it is a Ray runtime venv materialization failure.

## Verification Addendum - 2026-06-15 20:01 PDT

Short read:

- SWE-RL PARD-2 still has no successful Step-1 training evidence. The r27 job
  `3327299` moved much farther than prior attempts: it reached
  `SETUP COMPLETE`, total setup `1744.9s`, and rollout collection up to
  `29/32`, but failed during the first policy-to-generation refit/broadcast on
  a Megatron tensor-model-parallel NCCL `ALLGATHER` watchdog timeout
  (`600000ms`).
- That failure class is now patched for retry: remote Math/SWE repos include
  `NRL_MEGATRON_PG_TIMEOUT_V1`, and the SWE submit wrapper now propagates
  `NRL_MEGATRON_NCCL_TIMEOUT_SECONDS` and
  `NRL_MEGATRON_PROCESS_GROUP_TIMEOUT_SECONDS` into the Ray driver/actor
  environment. Retry `3331533` is queued under `nemotron_n3_post` with
  `max_steps=1` and `3600s` Megatron PG timeout.
- MathRL generation-bound baseline `3330755` and Eagle-3 K3 `3330756` are
  still running and currently in actor/vLLM environment materialization after
  `All workers connected!`. Static PARD-2 K3 `3330757` and online PARD-2 K3
  `3330758` are still pending.
- Existing clean 10-step proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10. For 235B PARD-2, the latest evidence is Step-1 entry but
  not a completed training step.

## Verification Addendum - 2026-06-15 19:49 PDT

Short read:

- New Step-1 evidence exists for 235B MathRL online PARD-2, but it is not a
  completed training step. Run `3324571` reached `SETUP COMPLETE`, entered
  `Step 1/10`, and began `Generating responses for batch of size 64`; it then
  failed during the first policy-to-generation refit with
  `KeyError: 'draft.fc.weight'`.
- The failure mechanism is now patched in the Math and SWE remote repos:
  `Policy.prepare_refit_info()` merges metadata from all policy workers and
  logs `NRL_REFIT_INFO_MERGE_V1`, avoiding loss of `draft.*` metadata from
  non-first workers. This should apply to pending Math online PARD-2
  generation-bound job `3330758` when it starts.
- SWE-RL PARD-2 r27 `3327299` has moved past worker setup: it reached
  `64/64` actors and `All workers connected!`, and `ray-driver.log` shows
  rollout collection progress up to `29/32`. It still has no NeMo-RL Step
  marker.
- Generation-bound MathRL status: baseline `3330755` and Eagle-3 K3 `3330756`
  are `RUNNING` and both reached `All workers connected!`; static PARD-2 K3
  `3330757` and online PARD-2 K3 `3330758` are still `PENDING|Priority`.

## Verification Addendum - 2026-06-15 19:30 PDT

Short read:

- Existing Step-1-plus evidence is still unchanged: 235B MathRL baseline/PARD
  completed 10/10, 235B static PARD-2 has one prior run that reached
  `Step 1/10` before failing later, and SWE-RL PARD-2 still has no Step-1
  evidence.
- Current live Math online PARD-2 `3324571` and SWE-RL r27 `3327299` are both
  `RUNNING` since `19:05:01 PDT`, but neither has emitted `SETUP COMPLETE` or
  `Step` markers yet. Math static PARD-2 `3324570` failed before setup/Step
  from a `uv run --locked --extra mcore` actor-venv build error:
  missing `megatron/core/datasets/helpers_cpp.cpython-313-aarch64-linux-gnu.so`.
- Submitted the requested generation-bound MathRL comparison with longer
  decode: baseline `3330755`, Eagle-3 K3 `3330756`, static PARD-2 K3
  `3330757`, and online PARD-2 K3 `3330758`. All use
  `max_new_tokens=min_tokens=1024`, `temperature=1.0`, `top_p=1.0`,
  `top_k=-1`, and 5 steps. All four are currently `PENDING|Priority`; latest
  start estimates are `19:43:43`, `20:05:01`, `20:21:17`, and `21:39:44`
  PDT, respectively.

## Access Note - 2026-06-15 19:07 PDT

I attempted a fresh live poll, but this local shell currently resolves through
public DNS (`8.8.8.8`) and cannot resolve the internal SSH aliases
`oci-hsg-cs-001-vscode-02` or `login-lyris`. The latest successful remote poll
therefore remains `18:47 PDT`; this is an access/DNS limitation, not a new
runtime failure signal.

## Verification Addendum - 2026-06-15 18:47 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  were all still `PENDING|Priority`; their log roots had no runtime logs.
- Latest estimates were `18:52:03 PDT` for Math static `3324570`,
  `19:17:49 PDT` for Math online `3324571`, and `20:21:17 PDT` for SWE-RL
  r27 `3327299`. Priorities were `129934`, `129934`, and `129917`,
  respectively.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later, and active 235B SWE-RL PARD-2 has no training Step 1 evidence
  yet.

## Verification Addendum - 2026-06-15 18:43 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; their log roots have no runtime logs.
- Latest estimates are `18:52:03 PDT` for Math static `3324570`, and
  `19:13:02 PDT` for both Math online `3324571` and SWE-RL r27 `3327299`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:41 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; their log roots have no runtime logs.
- Latest estimates are `18:52:03 PDT` for Math static `3324570`,
  `19:13:02 PDT` for Math online `3324571`, and `20:00:00 PDT` for SWE-RL
  r27 `3327299`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:39 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; their log roots have no runtime logs.
- Latest estimates are `18:52:03 PDT` for Math static `3324570`,
  `19:13:02 PDT` for SWE-RL r27 `3327299`, and `19:17:49 PDT` for Math
  online `3324571`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:37 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates after five OCI-HSG
  polls from `18:32:43 PDT` through `18:37:17 PDT`. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299` are all
  still `PENDING|Priority`; their log roots have no runtime logs.
- Latest estimates are `18:52:03 PDT` for Math static `3324570`,
  `19:17:49 PDT` for Math online `3324571`, and `19:13:02 PDT` for SWE-RL
  r27 `3327299`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:31 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; their log roots have no runtime logs.
- Latest estimates are `18:52:00 PDT` for Math static `3324570`,
  `18:52:03 PDT` for Math online `3324571`, and `19:13:02 PDT` for SWE-RL
  r27 `3327299`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:29 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; their log roots have no runtime logs.
- Latest estimates shifted again: SWE-RL r27 `3327299` is now estimated at
  `20:25:06 PDT`, while both Math PARD-2 gates `3324570`/`3324571` moved to
  `20:52:39 PDT`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:26 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; their log roots have no runtime logs.
- Math priorities ticked up to `129932`, but the jobs still have no actual
  allocation. Latest estimates are `18:52:00 PDT` for Math static `3324570`,
  `18:52:03 PDT` for Math online `3324571`, and `20:52:39 PDT` for SWE-RL
  r27 `3327299`.
- I am keeping the current 32-node 10-step Math PARD-2 gates rather than
  submitting a smaller replacement, because the requested proof needs the same
  235B MathRL shape that can produce comparable 10-step metrics.

## Verification Addendum - 2026-06-15 18:24 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; their log roots have no runtime logs.
- The near-current `3324570` estimate did not turn into allocation. Latest
  estimates are `18:52:00 PDT` for Math static `3324570`, `18:52:03 PDT` for
  Math online `3324571`, and `20:52:39 PDT` for SWE-RL r27 `3327299`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:21 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; no runtime logs exist under their log
  roots.
- `scontrol` now shows scheduler candidate nodes for all three jobs, but no
  actual `NodeList` allocation yet. Latest estimates are `19:17:48 PDT` for
  Math static `3324570`, `20:21:17 PDT` for Math online `3324571`, and
  `20:52:39 PDT` for SWE-RL r27 `3327299`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:19 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; no runtime logs exist under their log
  roots.
- The short-lived `18:18:35 PDT` start estimate did not turn into allocation.
  Latest estimates drifted to `19:17:49 PDT` for Math static `3324570`,
  `20:21:17 PDT` for SWE-RL r27 `3327299`, and `20:52:39 PDT` for Math
  online `3324571`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:16 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`, and their log roots have no runtime logs.
- Latest start estimates remain `18:42:26 PDT` for Math static `3324570`,
  `19:17:48 PDT` for Math online `3324571`, and `20:21:17 PDT` for SWE-RL
  r27 `3327299`.
- Existing Step-1-plus proof remains unchanged: 235B MathRL baseline/PARD
  completed 10/10, and 235B MathRL static PARD-2 reached `Step 1/10` before
  failing later. Active 235B SWE-RL PARD-2 has not started yet.

## Verification Addendum - 2026-06-15 18:15 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`; their log roots have no runtime logs yet.
- Latest start estimates are `18:42:26 PDT` for Math static `3324570`,
  `19:17:48 PDT` for Math online `3324571`, and `20:21:17 PDT` for SWE-RL
  r27 `3327299`.
- The currently proven Step-1-plus examples remain unchanged: 235B MathRL
  baseline/PARD completed 10/10, and 235B MathRL static PARD-2 reached
  `Step 1/10` before failing later. The active 235B SWE-RL PARD-2 retry has
  not started yet.

## Verification Addendum - 2026-06-15 18:13 PDT

Short read:

- Still no new Step-1 evidence from the active proof gates. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27 `3327299`
  are all still `PENDING|Priority`.
- Latest start estimates moved to `18:52:00 PDT` for Math static `3324570`,
  `19:17:49 PDT` for Math online `3324571`, and `19:37:27 PDT` for SWE-RL
  r27 `3327299`.
- The answer to "Step 1 이상" is unchanged: yes for completed 235B MathRL
  baseline/PARD and partial 235B MathRL PARD-2, but not yet for the active
  235B SWE-RL PARD-2 retry.

## Verification Addendum - 2026-06-15 18:11 PDT

Short read:

- No new Step-1 evidence yet from the current proof gates because they remain
  queued. Math static PARD-2 `3324570`, Math online PARD-2 `3324571`, and
  SWE-RL r27 `3327299` are all `PENDING|Priority`.
- Latest estimates are `19:17:48 PDT` for Math static `3324570`,
  `19:37:27 PDT` for SWE-RL r27 `3327299`, and `20:16:46 PDT` for Math
  online `3324571`.
- Current Step-1-plus evidence remains unchanged: clean 235B MathRL baseline
  `3321180`, PARD K3 `3321423`, and PARD K5 `3321424` completed 10/10;
  Math static PARD-2 `3324200` reached `Step 1/10` then failed later; the
  latest 235B SWE-RL PARD-2 retry has not started yet.

## Verification Addendum - 2026-06-15 18:06 PDT

Short read:

- No new Step-1 evidence yet because all active proof gates remain queued.
  Math static PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27
  `3327299` are all `PENDING|Priority`.
- Latest estimates are `19:17:48 PDT` for `3324570`, `19:37:27 PDT` for
  `3324571`, and `20:16:46 PDT` for `3327299`.
- Current usable examples remain unchanged: clean 235B MathRL baseline
  `3321180`, PARD K3 `3321423`, and PARD K5 `3321424` completed 10/10;
  SWE-RL still has no verified training `Step 1`.

## Verification Addendum - 2026-06-15 18:05 PDT

Short read:

- Active proof gates are still queued: Math static PARD-2 `3324570`, Math
  online PARD-2 `3324571`, and SWE-RL r27 `3327299` are all
  `PENDING|Priority`.
- I reduced only the Math PARD-2 gates to `01:00:00` walltime to improve
  backfill. Existing completed Math 10-step runs are `00:30:20` to `00:33:37`,
  and the prior static PARD-2 attempt reached its failure after `00:25:11`, so
  1h still has margin for the Math proof. SWE r27 remains `02:00:00`.
- Latest estimates: `19:37:27 PDT` for Math static `3324570`, `20:21:17 PDT`
  for Math online `3324571`, and `20:21:17 PDT` for SWE r27 `3327299`.

## Verification Addendum - 2026-06-15 18:03 PDT

Short read:

- Still no new Step-1 evidence because all active proof gates remain queued:
  Math static PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL r27
  `3327299` are all `PENDING|Priority`.
- Latest estimates are `19:17:49 PDT` for `3324570`, `19:56:03 PDT` for
  `3324571`, and `21:11:03 PDT` for `3327299`.
- I added a guard so the SWE-RL submit wrapper preflights the remote Gym
  OpenHands `PYTHONPATH` and miniforge shebang-bind patches before future SWE
  submissions. The local contract validator and an OCI-HSG dry-run both pass.

## Verification Addendum - 2026-06-15 17:59 PDT

Short read:

- No new Step-1 proof appeared after r27 submission because the active gates
  are still queued. Math static PARD-2 `3324570`, Math online PARD-2
  `3324571`, and SWE-RL r27 `3327299` are all `PENDING|Priority`.
- The near-term allocation estimate drifted during polling: latest values are
  `19:17:49 PDT` for Math static `3324570`, `19:56:03 PDT` for Math online
  `3324571`, and `21:11:03 PDT` for SWE r27 `3327299`.
- The previous status is otherwise unchanged: clean 235B Step-10 examples are
  MathRL baseline `3321180`, PARD K3 `3321423`, and PARD K5 `3321424`;
  SWE-RL still has no verified training `Step 1`.

## Verification Addendum - 2026-06-15 17:53 PDT

Short read:

- For SWE-RL 235B, still no verified training `Step 1`. r25 `3324801`
  reached `SETUP COMPLETE` and `Collecting rollouts: 0/32`, but final marker
  counts before cancellation were `Step=0`, `bad interpreter/return code 126=124`,
  and `NeMo Gym returned a result with no generation data=462`.
- The r25 root cause was inner OpenHands/SWE-rebench `poetry`: its shebang
  pointed to `/opt/nemo-rl/.../swe_openhands_setup/miniforge3/bin/python3.12`,
  which was not mounted inside the SWE-rebench SIF.
- I patched OCI-HSG remote Gym `app.py` to bind the shebang's miniforge path,
  verified `py_compile`, submitted patched retry r27 `3327299`, and cancelled
  stale r25 `3324801` plus pre-patch fallback r26 `3325343`.
- r27 `3327299` is `PENDING|Priority` under `nemotron_n3_post` with no
  dependency; current scheduler estimate is `2026-06-15 19:17:48 PDT`.
- Existing Step-1-plus examples are unchanged: clean 235B MathRL baseline
  `3321180`, PARD K3 `3321423`, and PARD K5 `3321424` completed `Step 1/10`
  through `Step 10/10`; Math static PARD-2 `3324200` reached `Step 1/10`
  and failed later.

## Verification Addendum - 2026-06-15 17:42 PDT

Short read:

- Yes, there are NeMo-RL runs that operated beyond Step 1. The cleanest 235B
  examples are MathRL baseline `3321180`, PARD K3 `3321423`, and PARD K5
  `3321424`; all three completed `Step 1/10` through `Step 10/10`.
- There is also partial 235B PARD-2 evidence: Math static PARD-2 `3324200`
  reached `Step 1/10`, generated responses, computed logprobs, and then failed
  during policy training. That is Step-1-plus behavior, but not a clean
  completed example.
- The smaller Qwen3-8B official comparison remains the clean online PARD-2
  mechanics proof: `3288181` baseline, `3288182` static PARD-2, and `3288183`
  online PARD-2 completed the 10-step run; online refit changed acceptance but
  did not improve throughput.
- Current SWE-RL r25 `3324801` has now reached `SETUP COMPLETE` and all
  `100/100` Ray actors are alive. It has not emitted a `Step 1` marker yet as
  of `17:42 PDT`; it is currently building `NemoGym` actor environments after
  setup.

## Verification Addendum - 2026-06-15 17:36 PDT

Short read:

- If the question is about SWE-RL r25 `3324801`, it did not terminate
  immediately. It is still `RUNNING` on `nvl72092-T[01-16]` with
  `RunTime=00:26:58` at the latest poll.
- The immediately failing pieces are short startup/probe Slurm steps. They
  tried to execute `/tmp/nemo_rl_ray_3324801_3.13.13_2.54.0/bin/ray` before
  the Ray venv had been created inside the job container, then one `ray status`
  probe hit Ray's `AttributeError: 'NoneType' object has no attribute 'decode'`.
- This was not terminal: the same Slurm log later reached `64/64` actors and
  `All workers connected!`. The current issue is that PARD-2 vLLM setup has
  not progressed to `SETUP COMPLETE` or Step 1 yet.
- Math static r25 `3324570` and Math online r25 `3324571` are still
  `PENDING|Priority`; fallback SWE-RL r26 `3325343` is still
  `PENDING|Dependency`.
- Lyris still cannot refresh Slurm state from this session because Kerberos is
  absent and `/etc/slurm/slurm.conf` returns `Permission denied`.

## Verification Addendum - 2026-06-15 17:26 PDT

Short read:

- SWE-RL r25 `3324801` is running, not immediately terminated. At the latest
  poll it was `RUNNING` for `00:17:25` on `nvl72092-T[01-16]`.
- The scary early `ray: No such file or directory` lines were transient
  health-check `srun` probes before the Ray venv finished installing. The
  Slurm log later reached `64/64` actors and `All workers connected!`.
- r25 has progressed beyond the prior r24 Ray/Python mismatch and r23
  `nemo_gym` import failure: the driver loaded SWE datasets, initialized
  policy workers, started PARD-2 vLLM engines, and loaded Qwen3 checkpoint
  shards. It has not yet emitted `SETUP COMPLETE` or Step 1.
- Math static r25 `3324570` and Math online r25 `3324571` are still
  `PENDING|Priority`; fallback SWE-RL r26 `3325343` is still
  `PENDING|Dependency`.
- Lyris is reachable by SSH but not usable for Slurm yet. `klist -s` returns
  `1`, and `squeue`/`sacct` still fail on `/etc/slurm/slurm.conf: Permission
  denied`.

## Verification Addendum - 2026-06-15 17:07 PDT

Short read:

- No active 235B gate has started yet. Math static r25 `3324570`, Math online
  r25 `3324571`, and SWE-RL r25 `3324801` are still `PENDING|Priority`; r26
  `3325343` remains dependency-blocked.
- Scheduler estimates are back: Math static r25 at `21:30 PDT`, Math online
  r25 at `22:20 PDT`, and SWE-RL r25 at `22:30 PDT`.
- There are still no stdout/driver logs, so this remains queue wait rather
  than a new Nemo-RL launch failure.
- Lyris Slurm access remains blocked by expired Kerberos credentials and the
  direct login-node path requires MFA/publickey authentication.

## Verification Addendum - 2026-06-15 17:04 PDT

Short read:

- No active 235B gate has started yet. Math static r25 `3324570`, Math online
  r25 `3324571`, and SWE-RL r25 `3324801` are still `PENDING|Priority`; r26
  `3325343` remains dependency-blocked on `afternotok:3324801`.
- Current scheduler estimates are unavailable again: `squeue --start` returns
  `N/A`, and `scontrol` reports `StartTime=Unknown`.
- There are still no stdout/driver logs, so this remains queue wait rather
  than a new Nemo-RL launch failure.
- MathRL/SWE-RL validators and doc checks pass. Lyris Slurm access remains
  blocked by expired Kerberos credentials and needs a fresh MFA/Kerberos login.

## Verification Addendum - 2026-06-15 16:59 PDT

Short read:

- No active gate has started yet; no stdout or driver logs exist.
- Scheduler estimates are back: Math static r25 at `22:20 PDT`, Math online
  r25 and SWE-RL r25 at `22:30 PDT`; r26 remains dependency-blocked.
- MathRL and SWE-RL launcher validators still pass.
- Lyris Slurm access remains Kerberos-blocked.

## Verification Addendum - 2026-06-15 16:57 PDT

Short read:

- No active gate has started yet. Math static r25 `3324570`, Math online r25
  `3324571`, and SWE-RL r25 `3324801` are still `PENDING|Priority`; r26
  `3325343` remains dependency-blocked.
- Scheduler estimates are currently back to `N/A` / `StartTime=Unknown`.
- There are still no stdout/driver logs, so this is queue wait rather than a
  new launch failure.
- Lyris Slurm access remains blocked by expired Kerberos tickets.

## Verification Addendum - 2026-06-15 16:53 PDT

Short read:

- Active gates are still queue-waiting with no logs. Latest estimates are
  Math static r25 `20:50 PDT`, SWE-RL r25 `22:20 PDT`, and Math online r25
  `22:30 PDT`; r26 remains dependency-blocked.
- Extended the SWE-RL launcher contract validator to cover the Ray/Python
  explicit sbatch export path and NemoGym source path propagation. The
  regenerated validation report is `PASS`.
- Math PARD-2 CP=1 validator still passes. Lyris Slurm access remains blocked
  by expired Kerberos tickets.

## Verification Addendum - 2026-06-15 16:50 PDT

Short read:

- Active OCI-HSG gates are still queue-waiting with no logs: Math static r25
  estimate `22:12:30 PDT`; Math online r25 and SWE-RL r25 estimate
  `22:20 PDT`; fallback r26 remains dependency-blocked.
- Added a focused MathRL PARD-2 contract validator. It passes and confirms the
  pending r25 launcher path has the intended `sequence_packing=false` plus
  `context_parallel_size=1` combination, and that only `online_pard2_k3`
  enables online draft/refit.
- Lyris Slurm access remains blocked by expired Kerberos tickets.

## Verification Addendum - 2026-06-15 16:49 PDT

Short read:

- Still no new runtime evidence: no active r25/r26 gate has stdout or a driver
  log.
- OCI-HSG scheduler estimates are back: Math static r25 `3324570` at
  `22:12:30 PDT`, and Math online r25 `3324571` plus SWE-RL r25 `3324801` at
  `22:20 PDT`.
- r26 `3325343` remains dependency-blocked on `afternotok:3324801`.
- Lyris Slurm access is still blocked by expired Kerberos tickets.

## Verification Addendum - 2026-06-15 16:47 PDT

Short read:

- Still no new 235B runtime evidence. The three r25 gates remain
  `PENDING|Priority`, and r26 remains `PENDING|Dependency`.
- OCI-HSG scheduler start estimates are currently unavailable again:
  `squeue --start` reports `N/A`, and `scontrol` reports `StartTime=Unknown`
  for `3324570`, `3324571`, and `3324801`.
- No active gate has produced stdout or driver logs. This is queue wait, not a
  new launch failure.
- Lyris remains Kerberos-blocked for Slurm refresh.

## Verification Addendum - 2026-06-15 16:45 PDT

Short read:

- No new 235B runtime evidence yet. SWE-RL r25 `3324801`, Math static r25
  `3324570`, Math online r25 `3324571`, and fallback r26 `3325343` are still
  pending and have no stdout/driver logs.
- Current OCI-HSG estimate moved to `21:30 PDT` for all three r25 jobs.
  Fallback r26 remains dependency-blocked on `afternotok:3324801`.
- Lyris Slurm remains inaccessible from this session because expired Kerberos
  tickets still make `/etc/slurm/slurm.conf` unreadable.

## Verification Addendum - 2026-06-15 16:42 PDT

Short read:

- Still no new 235B runtime evidence. SWE-RL r25 `3324801`, Math static r25
  `3324570`, Math online r25 `3324571`, and fallback r26 `3325343` are
  pending with no stdout/driver logs.
- Current OCI-HSG estimates moved earlier: Math static/online r25 at
  `18:20 PDT`, SWE-RL r25 at `19:50 PDT`; fallback r26 remains dependency
  blocked on `afternotok:3324801`.
- Lyris Slurm remains inaccessible in this session because Kerberos tickets
  expired at `16:08:40 PDT`, making `/etc/slurm/slurm.conf` unreadable.

## Verification Addendum - 2026-06-15 16:36 PDT

Short read:

- Still no new 235B runtime evidence. SWE-RL r25 `3324801`, Math static r25
  `3324570`, Math online r25 `3324571`, and fallback r26 `3325343` have no
  stdout/driver logs yet.
- The current OCI-HSG estimate remains `22:10 PDT` for the three r25 jobs.
- Lyris is still inaccessible for Slurm because the Kerberos ticket backing the
  NFSv4 `/etc/slurm` mount is expired.

## Verification Addendum - 2026-06-15 16:35 PDT

Short read:

- No new runtime evidence yet. SWE-RL r25 `3324801`, Math static r25
  `3324570`, Math online r25 `3324571`, and fallback r26 `3325343` are all
  still pending with no stdout/driver logs.
- Latest OCI-HSG estimate is now `22:10 PDT` for all three r25 gates after the
  walltime reduction to `2h`.
- Lyris Slurm refresh remains blocked by expired Kerberos credentials; `squeue`
  still fails on `/etc/slurm/slurm.conf: Permission denied`.

## Verification Addendum - 2026-06-15 16:31 PDT

Short read:

- No new 235B runtime logs yet. r25/r26 are still pending.
- I lowered the active proof gate walltimes from `4h` to `2h` to improve
  backfill chances. Math 10-step completed examples are around 30-34 minutes,
  so this still leaves margin for the Math static/online PARD-2 gates.
- SLURM now reports `19:30 PDT` for SWE-RL r25 `3324801` and `22:20 PDT` for
  Math static/online r25 `3324570`/`3324571`. Fallback r26 `3325343` remains
  dependency-blocked on `afternotok:3324801`.

## Verification Addendum - 2026-06-15 16:26 PDT

Short read:

- OCI-HSG 235B replacements are still pending. SWE-RL r25 `3324801` now has
  the nearest estimate, `17:10 PDT`; Math static/online r25 `3324570`/`3324571`
  moved to `22:20 PDT`.
- No new r25/r26 stdout or driver logs exist yet.
- r25 does not show explicit `--export` in `scontrol SubmitLine`, so its Ray
  bootstrap depends on submit-time environment export. Fallback r26 `3325343`
  does show explicit Ray `2.54.0` / Python `3.13.13` export and remains
  dependency-blocked on `afternotok:3324801`.
- I hardened the shared SWE-RL submit wrapper for future retries. A dry-run now
  emits `--export=ALL,RAY_VERSION=2.54.0,RAY_PYTHON_VERSION=3.13.13,
  RAY_PYTHON_SPEC=3.13.13,RAY_USE_EXISTING_ENV=false,UV_PYTHON=3.13.13,
  UV_PYTHON_DOWNLOADS=auto` on the generated OCI-HSG sbatch line.

## Verification Addendum - 2026-06-15 16:25 PDT

Short read:

- Lyris queue refresh is blocked by expired Kerberos credentials, not by a new
  job failure. The active SSH ControlMaster still connects, but Lyris Kerberos
  tickets expired at `16:08:40 PDT`; `/home` and `/etc/slurm` are NFSv4
  `sec=krb5` mounts, so `/home/sna/.bashrc` and `/etc/slurm/slurm.conf` return
  `Permission denied`.
- `kinit -R` on Lyris failed with `Ticket expired while renewing credentials`.
  A fresh MFA/Kerberos login is required before `squeue`/`sacct` can refresh
  Lyris job state again.

## Verification Addendum - 2026-06-15 16:21 PDT

Short read:

- No new 235B runtime evidence yet. Math static r25 `3324570`, Math online r25
  `3324571`, SWE-RL r25 `3324801`, and fallback r26 `3325343` are still
  pending and have no stdout/driver logs.
- Latest OCI-HSG estimates improved to `20:20 PDT` for Math static and
  `20:40 PDT` for Math online plus SWE-RL r25. Fallback r26 stays blocked on
  `afternotok:3324801`.
- Lyris SSH is reachable, but `squeue` still fails because the expired
  Kerberos/NFS credential makes `/etc/slurm/slurm.conf` unreadable. This is not
  a job-level failure signal.
- Qwen3-8B online PARD-2 artifacts were regenerated. `3288183` remains the
  smaller 10/10 mechanics proof: online refit parsed on `9` post-step rows and
  acceptance increased from static `1.836%` to online `2.553%`. It still does
  not show throughput improvement; online generation-worker TPS is `0.9696x`
  static and `0.5887x` baseline.

## Verification Addendum - 2026-06-15 16:16 PDT

Short read:

- No new 235B runtime evidence yet. Math static r25 `3324570`, Math online r25
  `3324571`, SWE-RL r25 `3324801`, and fallback r26 `3325343` are still
  pending, with no stdout/driver logs.
- Latest OCI estimates are `20:30 PDT` for Math static and `22:10 PDT` for
  Math online/SWE-RL.
- Lyris SSH is reachable, but Slurm commands currently fail on
  `/etc/slurm/slurm.conf` permission/stale-handle errors, so Lyris queue state
  cannot be refreshed through `squeue` until that login-node config issue is
  cleared.
- Still-good examples remain unchanged: 235B MathRL baseline `3321180`, PARD
  K3 `3321423`, PARD K5 `3321424`, and smaller Qwen3-8B online PARD2
  `3288183` completed 10/10.

## Verification Addendum - 2026-06-15 16:14 PDT

Short read:

- r25 is still waiting for allocation: Math static `3324570` estimate
  `20:20 PDT`, Math online `3324571` estimate `22:10 PDT`, SWE-RL `3324801`
  estimate `22:10 PDT`.
- No r25 stdout or driver logs exist yet.
- Added SWE-RL fallback r26 `3325343` with dependency
  `afternotok:3324801`. It is `PENDING|Dependency` and has explicit sbatch
  Ray/Python exports to avoid repeating the r24 Ray `2.49.2` / Python
  `3.12.13` mismatch.
- Still-good examples remain unchanged: 235B MathRL baseline `3321180`, PARD
  K3 `3321423`, PARD K5 `3321424`, and smaller Qwen3-8B online PARD2
  `3288183` completed 10/10.

## Verification Addendum - 2026-06-15 16:08 PDT

Short read:

- Active 235B r25 gates are still pending. Current estimates are Math static
  `3324570` at `19:40 PDT`, Math online `3324571` at `19:40 PDT`, and SWE-RL
  `3324801` at `20:30 PDT`.
- No `slurm-*.out` or `ray-driver.log` exists yet for the three r25 jobs.
- The r25 poll helper now checks the correct Math log root
  `mathrl_latest_main_logs`, so future polls will catch the first stdout and
  driver logs for `3324570`/`3324571`.
- Still-good examples remain unchanged: 235B MathRL baseline `3321180`, PARD
  K3 `3321423`, and PARD K5 `3321424` completed 10/10; smaller Qwen3-8B
  online PARD2 `3288183` completed 10/10 with draft refit active.

## Verification Addendum - 2026-06-15 16:05 PDT

Short read:

- Active 235B r25 gates remain queued, not failed. Current SLURM estimates are
  Math static r25 `3324570` at `19:00 PDT`, SWE-RL r25 `3324801` at
  `19:40 PDT`, and Math online r25 `3324571` at `20:10 PDT`.
- There are still no r25 driver logs, so no new 235B Step 1 evidence exists.
- Remote SWE-RL repo has the NemoGym actor `PYTHONPATH` fix marker
  `NRL_NEMO_GYM_CREATE_ENV_PYTHONPATH_V1`; this addresses the r23
  `ModuleNotFoundError: No module named 'nemo_gym'` failure class.
- Smaller online PARD-2 proof is confirmed: Qwen3-8B online PARD2 job
  `3288183` completed 10/10 on OCI-HSG, and the driver log shows draft refit
  active through Step 10/10.
- Still-good 235B 10-step examples are unchanged: MathRL baseline `3321180`,
  PARD K3 `3321423`, and PARD K5 `3321424` completed 10/10.

## Verification Addendum - 2026-06-15 15:56 PDT

Short read:

- No runtime change from the 15:54 poll. SWE-RL r25 `3324801`, Math static
  r25 `3324570`, and Math online r25 `3324571` are still `PENDING|Priority`.
- `squeue --start` reports `N/A`; `scontrol` reports `StartTime=Unknown`.
- No driver logs exist yet, so there is still no Step 1 evidence from the r25
  replacements.

## Verification Addendum - 2026-06-15 15:54 PDT

Short read:

- Active OCI-HSG r25 jobs are still queued: SWE-RL r25 `3324801`, Math static
  r25 `3324570`, and Math online r25 `3324571` are all `PENDING|Priority`.
- Latest scheduler estimate is unavailable again: `squeue --start` shows `N/A`
  and `scontrol` shows `StartTime=Unknown` for all three.
- Added `scripts/poll_nemorl_r25_pard2_gates_20260615.sh` to monitor these
  exact r25 jobs and their driver logs without printing secrets.
- Still-good 10-step examples are unchanged: MathRL baseline `3321180`, PARD
  K3 `3321423`, and PARD K5 `3321424` completed 10/10.

## Verification Addendum - 2026-06-15 15:50 PDT

Short read:

- Lyris optimized mxfp8 standalone jobs are now complete through bs=32.
  bf16-KV job `2131234` completed `0:0` with bs32 output throughput
  `2417.10 tok/s`; fp8-KV job `2131235` completed `0:0` with bs32 output
  throughput `2427.55 tok/s`.
- The bf16 denominator remains invalid/cancelled, so these are valid optimized
  standalone rows but not a valid bf16-vs-mxfp8 denominator comparison.
- OCI-HSG active r25 jobs remain queued: SWE-RL r25 `3324801` estimate
  `19:50 PDT`; Math static/online r25 `3324570`/`3324571` estimate
  `20:00 PDT`.
- I added the NemoGym actor `PYTHONPATH` fix to the local patch bundle so
  future patched-source deployments do not depend on the one-off remote edit.
- Still-good 10-step examples are unchanged: MathRL baseline `3321180`, PARD
  K3 `3321423`, and PARD K5 `3321424` completed 10/10.

## Verification Addendum - 2026-06-15 15:38 PDT

Short read:

- The active OCI-HSG replacements are still queued, not failed: SWE-RL r25
  (`3324801`), Math static r25 (`3324570`), and Math online r25 (`3324571`) are
  all `PENDING|Priority` under `nemotron_n3_post`, with `StartTime=Unknown` in
  the latest `scontrol` read.
- The quick SWE-RL r24 (`3324728`) failure was Ray head/driver mismatch:
  cluster Ray `2.49.2` / Python `3.12.13` versus driver Ray `2.54.0` / Python
  `3.13.13`. r25 explicitly sets Ray `2.54.0` and Python `3.13.13`.
- The Lyris bf16 repair (`2131558`) ended without a valid result because bs=1
  generation stalled and hit an NCCL `_ALLGATHER_BASE` watchdog timeout in the
  logits all-gather path after 600 seconds. SLURM records it as
  `CANCELLED by 2001147693`; no raw bs=1 result JSON was written.
- Lyris optimized mxfp8 bf16-KV (`2131234`) and fp8-KV (`2131235`) jobs are
  still running and valid through bs=16. The bf16 denominator remains invalid.
- Still-good 10-step examples are unchanged: MathRL baseline `3321180`, PARD
  K3 `3321423`, and PARD K5 `3321424` completed 10/10.

## Verification Addendum - 2026-06-15 15:28 PDT

Short read:

- SWE-RL r24 (`3324728`) failed before model setup because the Ray head used
  Ray `2.49.2` / Python `3.12.13` while the driver used Ray `2.54.0` / Python
  `3.13.13`. This was a Ray bootstrap mismatch, not a NemoGym import result.
- I submitted SWE-RL r25 (`3324801`) with Ray `2.54.0` and Python `3.13.13`
  explicitly set, plus the NemoGym source path fix. It is currently
  `PENDING|Priority`.
- Follow-up at `15:29 PDT`: SWE-RL r25 (`3324801`), Math static r25
  (`3324570`), and Math online r25 (`3324571`) are all `PENDING|Priority`;
  latest SLURM estimate is `N/A` for all three.
- Lyris standalone mxfp8 optimized rows are now valid through bs=16 for both
  bf16-KV (`1661.07` output tok/s) and fp8-KV (`1663.80` output tok/s).
  The bf16 denominator remains invalid through bs=16.
- Follow-up at `15:31 PDT`: r25 (`3324801`) remains `PENDING|Priority`, with
  current estimate `20:50 PDT`. Math static/online r25 (`3324570`, `3324571`)
  remain `PENDING|Priority`, current estimate `20:00 PDT`.
- Still-good 10-step examples are unchanged: MathRL baseline `3321180`, PARD
  K3 `3321423`, and PARD K5 `3321424` completed 10/10.

## Verification Addendum - 2026-06-15 15:18 PDT

Short read:

- SWE-RL r23 (`3324381`) failed after `00:34:16`, but it was not an immediate
  launch failure and not a PARD-2/vLLM setup failure. It reached
  `SETUP COMPLETE`, async vLLM Uvicorn serving, PARD-2 drafter load, and CUDA
  graph capture.
- Root cause: SWE environment actor creation failed because `NemoGym.__init__()`
  could not import `nemo_gym`: `ModuleNotFoundError: No module named
  'nemo_gym'`. This happened before Step 1 generation/training markers.
- I patched the SWE submit wrapper so `NRL_ACTOR_VENV_CACHE_SUFFIX` is
  propagated into the Ray driver command, then submitted r24 (`3324728`) with a
  fresh actor venv namespace (`/opt/ray_venvs_swerl_ray254_r24` plus suffix
  `swe_nemo_gym_source_r24`). It is currently `PENDING|Priority`.
- Follow-up at `15:20 PDT`: SWE-RL r24 (`3324728`), Math static r25
  (`3324570`), and Math online r25 (`3324571`) are all still
  `PENDING|Priority`; current SLURM estimate for all three is `22:20 PDT`.
- Still-good 10-step examples are unchanged: MathRL baseline `3321180`, PARD
  K3 `3321423`, and PARD K5 `3321424` completed 10/10.

## Verification Addendum - 2026-06-15 15:09 PDT

Short read:

- SWE-RL r23 (`3324381`) made more progress. The driver log updated through
  15:09 and now shows PARD-2 drafter load, FlashInfer autotuning, CUDA graph
  capture, and at least one async vLLM worker serving on Uvicorn.
- It still has not reached `SETUP COMPLETE` or Step 1, but there is no
  traceback. Current evidence is “still setting up and progressing”, not a
  failed run.
- Math PARD-2 CP=1 replacements remain queued: static r25 (`3324570`) and
  online r25 (`3324571`) are both `PENDING|Priority`.

## Verification Addendum - 2026-06-15 15:05 PDT

Short read:

- Math PARD-2 CP=1 replacements are still queued: static r25 (`3324570`) and
  online r25 (`3324571`) are both `PENDING|Priority`.
- SWE-RL r23 (`3324381`) is still running. It has not reached `SETUP COMPLETE`
  or Step 1 yet, but there is still no traceback. Node diagnostics show live
  MegatronPolicyWorker processes with GPU memory allocated and CPU-side
  compile/setup activity.
- Lyris bf16 repair (`2131352`) completed at the SLURM level but is not a
  usable denominator: the result says zero output tokens, and the log shows an
  NCCL `_ALLGATHER_BASE` watchdog timeout followed by SIGABRT and connection
  refused retries.
- Still-good 10-step examples are unchanged: MathRL baseline `3321180`, PARD
  K3 `3321423`, and PARD K5 `3321424` completed 10/10.

## Verification Addendum - 2026-06-15 15:01 PDT

Short read:

- The latest Math online PARD-2 r23 (`3324365`) failed before `SETUP COMPLETE`
  and before Step 1, but not immediately at launch. It ran `00:18:30`, reached
  vLLM/PARD-2 drafter load and 128/128 policy workers, then failed while
  constructing Megatron policy workers.
- Root cause: `policy.sequence_packing.enabled=false` was combined with
  `policy.megatron_cfg.context_parallel_size=2`. MCore requires sequence
  packing when context parallelism is enabled, so the actor died with
  `AssertionError: Sequence Packing must be enabled to use Context Parallelism
  with MCore.`
- I changed the MathRL launcher so static/online PARD-2 sequence-packing-off
  runs use `context_parallel_size=1`. The stale static r24 job (`3324460`) was
  cancelled before start, and CP=1 replacements are queued: static r25
  (`3324570`) and online r25 (`3324571`).
- SWE-RL r23 (`3324381`) is still running. It passed W&B init, dataset load,
  vLLM startup, and 32/32 policy worker initialization; it has not produced
  Step 1 evidence yet.
- Still-good 10-step examples are unchanged: MathRL baseline `3321180`, PARD
  K3 `3321423`, and PARD K5 `3321424` completed 10/10.

## Verification Addendum - 2026-06-15 14:37 PDT

Short read:

- Replacement Math online r23 (`3324365`) and SWE-RL r23 (`3324381`) have
  moved from pending to `RUNNING|Prolog`. Their driver logs are not created
  yet, so there is no Step 1 evidence or new failure from these replacements.
- Math static r24 (`3324460`) is still `PENDING|Priority`.
- Secret scan for the W&B key fragments across the updated artifacts returned
  clean.

## Verification Addendum - 2026-06-15 14:35 PDT

Short read:

- The quick SWE-RL exit was W&B setup, not PARD-2: r22 (`3324276`) failed in
  `wandb.init` with `No API key configured`. Replacement r23 (`3324381`) is
  queued with the key supplied through the submit environment.
- Math static PARD-2 r22 (`3324200`) was not an immediate launch failure. It
  ran `00:25:11`, reached `Step 1/10`, generated responses, computed
  logprobs, and then failed during policy training.
- Root cause for `3324200`: sequence packing tried to pack a sequence length
  of `600` into an `input_ids` row with only `408` tokens. I changed
  `static_pard2_k3` to disable sequence packing and submitted replacement r24
  (`3324460`), currently `PENDING|Priority`.
- Still-good 10-step examples are unchanged: MathRL baseline `3321180`, PARD
  K3 `3321423`, and PARD K5 `3321424` completed 10/10.

## Verification Addendum - 2026-06-15 14:29 PDT

Short read:

- Lyris is reachable again through `login-lyris`; batch SSH and `squeue` work.
- OCI-HSG Math static PARD-2 r22 (`3324200`) remains `RUNNING` and is still
  progressing through setup/checkpoint load with no new traceback. Online
  Math r23 (`3324365`) and SWE-RL r23 (`3324381`) are still queued.
- Lyris standalone `mxfp8_fp8kv` job `2131235` has a valid bs=1 result so far:
  output throughput `119.45 tok/s`, request throughput `0.0119 req/s`, mean
  TTFT `3203 ms`, p99 TTFT `27645 ms`. The job is still running more batch
  sizes.
- Lyris standalone bf16 denominator job `2131233` is not a valid result:
  EngineCore died after an NCCL timeout and retries produced zero output
  tokens. Repair job `2131352` is running to regenerate the bf16 bs=1
  denominator.

## Verification Addendum - 2026-06-15 14:24 PDT

Short read:

- SWE-RL PARD-2 r22 (`3324276`) exited quickly because W&B had no API key
  configured. It had already reached the driver and loaded train/validation
  datasets; the failure was in `wandb.init`, not PARD-2/vLLM.
- I added a submit-time W&B-key guard and resubmitted SWE-RL r23 as `3324381`.
  It is currently `PENDING` on `Priority`.
- MathRL online PARD-2 r21 (`3323893`) failed later in setup because online
  drafter training and sequence packing were both enabled. The launcher now
  disables sequence packing for `online_pard2_k3`, and replacement r23
  `3324365` is `PENDING` on `Priority`.
- Static MathRL PARD-2 r22 (`3324200`) remains `RUNNING` and has advanced past
  vLLM and policy worker initialization. No repeat of the raw spec-counter
  crash has appeared so far.

## Verification Addendum - 2026-06-15 14:04 PDT

Short read:

- MathRL baseline/PARD remains the working 10-step path:
  `3321180`, `3321423`, and `3321424` completed.
- MathRL PARD-2 r21 static (`3323891`) got farther than all earlier PARD-2
  attempts: `SETUP COMPLETE`, `Step 1/10`, and generation start. It failed
  only after that on a new spec-counter API mismatch:
  `ActorHandle` had no `_get_raw_spec_counters`.
- I patched `BaseVllmGenerationWorker` to expose `_get_raw_spec_counters()`
  using the existing vLLM spec decode metrics reader, deployed it to both
  remote MathRL and SWE-RL checkouts, and verified `py_compile`.
- Static replacement `3324200` is queued as r22. Online `3323893` is still
  queued and should use the patched worker when it starts. SWE-RL `3322947`
  is running and has moved past the long TransformerEngine build into async
  vLLM worker environment setup.

## Verification Addendum - 2026-06-15 13:20 PDT

Short read:

- Runtime is still pending on SLURM, but the NeMo-RL patch application itself
  is verified in local and remote code.
- Local patch bundle, remote MathRL checkout, and remote SWE-RL checkout all
  have the outer actor runtime-env override markers and the nested vLLM dynamic
  runtime-env marker.
- Patched NeMo-RL files pass `python3 -m py_compile` locally and remotely.
- Remote MathRL/SWE-RL patched NeMo-RL files have stale r12/r14 path count `0`.
- Both active PARD-2 source overlays still have `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1`,
  stale active r12/r14 path count `0`, and pass `python3 -m py_compile`.

## Latest Poll Addendum - 2026-06-15 13:17 PDT

Short read:

- No r20 allocation yet and no job-specific log directories yet.
- `3322940`, `3322941`, and `3322947` remain `PENDING` on `Priority`.
- SLURM start estimates changed back to `N/A` for all three jobs.
- Priority remains `131209`.

## Latest Poll Addendum - 2026-06-15 13:15 PDT

Short read:

- No r20 allocation yet and no job-specific log directories yet.
- Latest estimates: `3322940` MathRL static PARD-2 K3 remains
  `2026-06-15T16:50:00`; `3322941` MathRL online PARD-2 K3 and `3322947`
  SWE-RL PARD-2 step-1 both moved to `2026-06-15T18:00:00`.
- Priority remains `131209` for all three jobs.

## Latest Poll Addendum - 2026-06-15 13:12 PDT

Short read:

- Added `scripts/poll_nemorl_r20_pard2_gates_20260615.sh` for repeatable
  sanitized polling of the three r20 gates. It also supports `--watch`.
- No r20 allocation yet and no job-specific log directories yet.
- Latest estimates: `3322940` MathRL static PARD-2 K3 at
  `2026-06-15T19:40:00`, and `3322941` MathRL online PARD-2 K3 plus `3322947`
  SWE-RL PARD-2 step-1 at `2026-06-15T20:10:00`.
- Priority is now `131209` for all three jobs.

## Latest Poll Addendum - 2026-06-15 13:10 PDT

Short read:

- No r20 allocation yet and no job-specific log directories yet.
- Latest estimates are visible again: `3322940` MathRL static PARD-2 K3 at
  `2026-06-15T16:50:00`, `3322947` SWE-RL PARD-2 step-1 at
  `2026-06-15T17:02:43`, and `3322941` MathRL online PARD-2 K3 at
  `2026-06-15T18:00:00`.
- Active PARD-2 source overlays remain patched: `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1`
  is present, stale r12/r14 path count is `0`, and both overlay files pass
  `python3 -m py_compile`.

## Latest Poll Addendum - 2026-06-15 13:08 PDT

Short read:

- No r20 allocation yet. MathRL PARD-2 r20 gates `3322940` and `3322941` and
  SWE-RL PARD-2 r20 gate `3322947` are all still `PENDING` on `Priority`.
- SLURM start estimates changed back to `N/A` for all three jobs.
- No `ray-driver.log` exists yet, and the job-specific log directories have not
  been created.
- Lyris is still not refreshable from this noninteractive session because
  `login-lyris` requires MFA and there is no active ControlMaster.

## Latest Poll Addendum - 2026-06-15 13:06 PDT

Short read:

- No r20 runtime logs yet. MathRL PARD-2 r20 gates `3322940` and `3322941`
  remain `PENDING` on `Priority`.
- Latest estimates: `3322940` static PARD-2 K3 at
  `2026-06-15T20:20:00`, `3322941` online PARD-2 K3 at
  `2026-06-15T20:30:00`.
- SWE-RL PARD-2 r20 gate `3322947` remains `PENDING` but its estimate improved
  to `2026-06-15T14:50:00`.
- I left the `04:00:00` walltime unchanged. r20 is validating fresh PARD-2
  actor/vLLM runtime-env behavior, and a timeout during setup would destroy the
  proof value.

## Patch Addendum - 2026-06-15 13:04 PDT

Short read:

- While the r20 jobs are still queued, I proactively patched the two active
  OCI-HSG PARD-2 vLLM source overlays used by MathRL and SWE-RL.
- The active `ray_executor.py` files no longer contain stale r12/r14 concrete
  `/opt/ray_venvs_swerl_ray254_*` nested worker paths. They now carry
  `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1` and derive nested Ray worker Python
  from the current actor process.
- Both patched overlay files pass `python3 -m py_compile`; backups of the old
  files were left in place. The queued r20 jobs use the same shared overlay
  paths, so this should apply without resubmission.

## Latest Poll Addendum - 2026-06-15 13:00 PDT

Short read:

- MathRL baseline/PARD remains the working 235B 10-step path:
  `3321180`, `3321423`, and `3321424` completed cleanly.
- MathRL PARD-2 r20 gates `3322940` and `3322941` remain `PENDING` on
  `Priority`; latest estimate moved to `2026-06-15T20:10:00`.
- SWE-RL PARD-2 r20 gate `3322947` also remains `PENDING` on `Priority`, with
  the same latest estimate `2026-06-15T20:10:00`.
- No r20 driver logs exist yet, so there is still no Step 1 proof, failure
  signal, or runtime validation of the dynamic nested vLLM runtime-env patch.
- I did not submit duplicate or smaller smoke jobs because the current queued
  jobs are the relevant proof shapes: 32-node 235B MathRL static/online PARD-2
  and 16-node SWE-RL PARD-2 step-1.

## Latest Poll Addendum - 2026-06-15 12:50 PDT

Short read:

- No runtime change yet. MathRL PARD-2 r20 gates `3322940` and `3322941` are
  still `PENDING` on `Priority`, estimated `2026-06-15T13:22:47`.
- SWE-RL PARD-2 r20 gate `3322947` is still `PENDING` on `Priority`, estimated
  `2026-06-15T13:49:31`.
- No r20 driver logs exist yet, so there is still no Step 1 proof or new
  failure signal from the dynamic nested vLLM runtime-env patch.
- Lyris automatic refresh remains unavailable in this session: `login-lyris`
  requires interactive MFA, and the `lyris` alias is not resolvable locally.

## Latest Poll Addendum - 2026-06-15 12:48 PDT

Short read:

- MathRL 10-step baseline/PARD remains good: `3321180`, `3321423`, and
  `3321424` are still the usable completed 235B MathRL examples.
- New MathRL PARD-2 r20 gates `3322940` and `3322941` are still
  `PENDING` on `Priority`, estimated `2026-06-15T13:22:47`; no driver logs
  exist yet.
- SWE-RL PARD-2 r20 gate `3322947` is still `PENDING` on `Priority`,
  estimated `2026-06-15T13:49:31`; no driver log exists yet.
- Lyris still cannot be refreshed automatically from this session because
  there is no active ControlMaster and batch-mode SSH is rejected by MFA.

## Latest Poll Addendum - 2026-06-15 12:45 PDT

Short read:

- r20 jobs remain queued. MathRL `3322940` and `3322941` are still
  `PENDING` on `Priority`, estimated `2026-06-15T13:22:47`.
- SWE-RL `3322947` is still `PENDING` on `Priority`; its estimated start moved
  later to `2026-06-15T13:49:31`.
- No r20 driver logs exist yet.

## Latest Poll Addendum - 2026-06-15 12:43 PDT

Short read:

- Still no allocation for r20. `3322940`, `3322941`, and `3322947` remain
  `PENDING` on `Priority`, estimated `2026-06-15T13:22:47`; no driver logs
  exist yet.

## Latest Poll Addendum - 2026-06-15 12:42 PDT

Short read:

- Still waiting for allocation. `3322940`, `3322941`, and `3322947` remain
  `PENDING` on `Priority`, estimated `2026-06-15T13:22:47`; no driver logs
  exist yet.

## Latest Poll Addendum - 2026-06-15 12:40 PDT

Short read:

- No runtime evidence yet for r20. `3322940`, `3322941`, and `3322947` are
  still `PENDING` on `Priority`, estimated `2026-06-15T13:22:47`; no driver
  logs exist yet.

## Latest Poll Addendum - 2026-06-15 12:39 PDT

Short read:

- r20 PARD-2 proof gates are still waiting for allocation. `3322940`,
  `3322941`, and `3322947` are all `PENDING` on `Priority` under
  `nemotron_n3_post`.
- The estimated start remains `2026-06-15T13:22:47` for all three.
- There are no r20 driver logs yet, so the new
  `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1` path is patched and submitted but not
  runtime-validated yet.

## Latest Poll Addendum - 2026-06-15 12:36 PDT

Short read:

- MathRL baseline/PARD remains good: `3321180`, `3321423`, and `3321424` are
  still the usable completed 235B MathRL examples.
- The r18/r19 PARD-2 path exposed one more stale-runtime-env layer: the outer
  NeMo-RL Ray actor used the fresh executable, but the shared PARD-2 vLLM
  overlay still had a concrete r14 `py_executable` written into
  `vllm/v1/executor/ray_executor.py`, so nested vLLM Ray workers tried the
  missing r14 path.
- Patched `vllm_worker.py` so future vLLM overlay patching writes
  `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1`: nested vLLM Ray workers now derive
  runtime env from the actor process `sys.executable` instead of a persisted
  `/opt/ray_venvs...` path. The patch was synced to both OCI-HSG remote
  checkouts and passed `py_compile`.
- Cancelled stuck/superseded jobs `3322475`, `3322611`, and `3322621`.
  Submitted r20 replacements: `3322940` MathRL static PARD-2 K3, `3322941`
  MathRL online PARD-2 K3, and `3322947` SWE-RL PARD-2 step-1. All are
  `PENDING` on `Priority` at `2026-06-15 12:36 PDT`; the 12:37 poll showed
  estimated start `2026-06-15T13:22:47`.

## Latest Poll Addendum - 2026-06-15 12:27 PDT

Short read:

- MathRL baseline/PARD remains good: `3321180`, `3321423`, and `3321424` are
  still the usable completed 235B MathRL examples with `Step 10/10` evidence.
- MathRL PARD-2 r19 static/online jobs are still waiting in the queue, not
  running and not failed. `3322611` and `3322621` are both `PENDING` on
  `Priority`, estimated start `2026-06-15T13:22:47`, and still have no driver
  logs.
- SWE-RL PARD-2 r18 (`3322475`) is still `RUNNING`; elapsed was `00:42:12` at
  the 12:25 poll. By 12:27 the driver log had grown to `124335` bytes. It is
  past TransformerEngine build and is installing actor environments under
  `/opt/ray_venvs_swerl_ray254_r18`, including the async vLLM worker path. No
  `SETUP COMPLETE` or `Step 1/1` marker yet.
- Lyris still cannot be refreshed from this noninteractive session: there is no
  active ControlMaster and batch-mode SSH is rejected by keyboard-interactive
  MFA.

## Latest Poll Addendum - 2026-06-15 12:22 PDT

Short read:

- MathRL 10-step remains healthy for baseline/PARD: `3321180`, `3321423`, and
  `3321424` are still the usable completed 235B MathRL examples.
- MathRL PARD-2 r19 static/online jobs have not started yet. `3322611` is
  `PENDING` on `Priority`, estimated `2026-06-15T13:22:47`; `3322621` is
  `PENDING` on `Priority`, estimated `2026-06-15T13:22:47`. There are no
  driver logs yet.
- SWE-RL PARD-2 r18 (`3322475`) is still `RUNNING`; elapsed was `00:39:04` at
  the poll. It has not reached `SETUP COMPLETE` or `Step 1/1`, but the
  TransformerEngine build completed and Ray is now creating actor venvs under
  `/opt/ray_venvs_swerl_ray254_r18/...VllmAsyncGenerationWorker`. vLLM and
  flash-attn build/cache artifacts were still being written through
  `2026-06-15 12:22:41 PDT`.
- Lyris could not be directly refreshed from this noninteractive session because
  `login-lyris` requires keyboard-interactive MFA and there is no active
  ControlMaster here. The Lyris retry statuses below are retained from earlier
  evidence rather than this poll.

## Latest Poll Addendum - 2026-06-15 11:59 PDT

Short read:

- MathRL baseline/PARD remains good: `3321180`, `3321423`, and `3321424`
  completed cleanly with `Step 10/10` and `sacct` exit `0:0`.
- MathRL PARD-2 r18 (`3322390`, `3322392`) was cancelled after it still tried
  to launch async vLLM workers from the stale
  `/opt/ray_venvs_swerl_ray254_r12/.../VllmAsyncGenerationWorker/bin/python`
  path.
- The actor-runtime hotfix is now applied to the local patch bundle and both
  remote OCI-HSG checkouts. It adds actor venv cache suffixing plus
  `NRL_ACTOR_RUNTIME_ENV_OVERRIDE_V1`/`NRL_ACTOR_PY_EXEC_V1` diagnostics so the
  per-call actor `py_executable` wins over stale Ray class defaults.
- MathRL PARD-2 r19 jobs are queued: `3322611` static PARD-2 K3 and `3322621`
  online PARD-2 K3. Both are `PENDING` on priority with estimated start
  `2026-06-15T13:22:47`.
- SWE-RL PARD-2 r18 (`3322475`) is `RUNNING`; Ray head is up and the driver log
  exists, but it is still at TransformerEngine build and has no `Step 1/1`
  marker yet.

## Latest Poll Addendum - 2026-06-15 11:07 PDT

Short read:

- Use OCI-HSG MathRL `3321180` as the current 235B MathRL 10-step baseline
  example. It completed successfully with `sacct` state `COMPLETED` and exit
  `0:0`.
- Use OCI-HSG SWE-RL `3299487` only as the closest completed 235B SWE-RL
  short-run baseline. The SLURM job completed with exit `0:0`, but the driver
  log only shows step markers through `Step 9/10` and no max-step guard line.
- Use OCI-HSG SWE-RL `3299489` only as evidence that PARD K5 can complete a
  short 235B NeMo-RL run; it is not a speedup result and the driver log only
  shows markers through `Step 6/10`.
- Do not use the current 235B SWE-RL PARD-2 proof `3308774` as a working
  example yet. It is still running, has not reached `SETUP COMPLETE` or a
  parsed GRPO step marker, and is repeatedly failing to launch the async vLLM
  worker from a stale/missing `r14` actor venv path.
- `3321423` and `3321424` are completed OCI-HSG 235B MathRL static PARD K3/K5
  follow-ups using the reduced shape from `3321180`. Both reached
  `Step 10/10`, trained policy on Step 10, logged the max-step guard, and
  completed with exit `0:0`.
- `3321785` and `3321786` are newly submitted OCI-HSG 235B MathRL PARD-2
  follow-ups for static PARD-2 and online PARD-2 drafter training. Both are
  pending at the latest poll.

Latest proof-gate updates:

| Job | Scope | Latest state | Read |
| ---: | --- | --- | --- |
| `3321180` | OCI-HSG 235B MathRL reduced-shape baseline, `temperature=1.0`, `top_p=1.0`, `fuse_loss=false`, `train_global_batch_size=64` | `COMPLETED` at `2026-06-15 10:27:57 PDT`, exit `0:0` | Submitted as the next 10-step candidate after `3321070` OOMed at Step 2. Uses `num_prompts=4`, `num_generations=16`, `generation_batch_size=16`, `max_num_seqs=16`, and `gpu_memory_utilization=0.80`. Driver reached all `Step 1/10` through `Step 10/10` markers and then logged `Max number of steps has been reached, stopping training early`. Worker TCPStore/NCCL warnings appear after shutdown, but the SLURM job completed cleanly. |
| `3321070` | OCI-HSG 235B MathRL baseline, `temperature=1.0`, `top_p=1.0`, `fuse_loss=false` | `FAILED` | Reached `SETUP COMPLETE`, finished Step 1 training, entered `Step 2/10`, then failed in Step 2 generation with vLLM `CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:139`. This is not a full 10-step example, but it is the first MathRL proof here that got past Step 1 training. |
| `3320856` | OCI-HSG 235B MathRL baseline, `temperature=0.0`, `fuse_loss=false` | `FAILED` | Reached `Step 1/10` and `Training policy`, then failed in Megatron DDP grad checking with local grad-norm `NaN`. This got past the earlier packed fused-loss copy mismatch but still produced no completed step. |
| `3321423` | OCI-HSG 235B MathRL reduced-shape static PARD K3, `temperature=1.0`, `top_p=1.0`, `fuse_loss=false`, `train_global_batch_size=64` | `COMPLETED` at `2026-06-15 11:06:42 PDT`, exit `0:0` | Started at `2026-06-15T10:33:05` on `nemotron_n3_post` and elapsed `00:33:37`. It passed Ray/vLLM setup, reached `SETUP COMPLETE`, hit all `Step 1/10` through `Step 10/10` markers, trained policy on Step 10, and logged `Max number of steps has been reached, stopping training early`. |
| `3321424` | OCI-HSG 235B MathRL reduced-shape static PARD K5, `temperature=1.0`, `top_p=1.0`, `fuse_loss=false`, `train_global_batch_size=64` | `COMPLETED` at `2026-06-15 11:06:21 PDT`, exit `0:0` | Started at `2026-06-15T10:33:05` on `nemotron_n3_post` and elapsed `00:33:16`. It passed Ray/vLLM setup, reached `SETUP COMPLETE`, hit all `Step 1/10` through `Step 10/10` markers, trained policy on Step 10, and logged `Max number of steps has been reached, stopping training early`. |
| `3321785` | OCI-HSG 235B MathRL reduced-shape static PARD-2 K3, `temperature=1.0`, `top_p=1.0`, `fuse_loss=false`, `train_global_batch_size=64` | `FAILED`, exit `1:0` | Failed before `SETUP COMPLETE`/Step 1 because Ray tried to launch async vLLM workers from the stale `/opt/ray_venvs_swerl_ray254_r12/.../VllmAsyncGenerationWorker/bin/python` path. Superseded by hotfix r19 job `3322611`. |
| `3321786` | OCI-HSG 235B MathRL reduced-shape online PARD-2 K3 drafter training, `temperature=1.0`, `top_p=1.0`, `fuse_loss=false`, `train_global_batch_size=64` | `FAILED`, exit `1:0` | Same stale async vLLM actor executable path failure before `SETUP COMPLETE`/Step 1. Superseded by hotfix r19 job `3322621`. |
| `3308774` | OCI-HSG 235B SWE-RL PARD-2 step-1 proof | `CANCELLED` | Cancelled after repeated async vLLM actor startup failure under `/opt/ray_venvs_swerl_ray254_r14/.../VllmAsyncGenerationWorker/bin/python` and no Step 1 marker. Superseded by fresh-venv r18 job `3322475`. |
| `2129715` | Lyris 235B SWE-RL baseline step-1 retry using existing container Ray/Python | `FAILED`, exit `2:0` | This proved the existing container Ray bootstrap path can start (`/opt/nemo_rl_venv/bin/ray`) and reached connected actors, but the driver failed because the repo still required Python `3.13.13` while the existing container env is Python `3.12.12`. |
| `2129746` | Lyris 235B SWE-RL baseline step-1 retry with Python `3.13.9` Ray venv | `FAILED`, exit `1:0` | `uv` found CPython `3.13.9` and created `/tmp/nemo_rl_ray_2129746_3.13.9_2.54.0`, but package extraction failed with an I/O error and the venv never got `bin/ray`; Ray head then failed repeatedly with `No such file or directory`. |
| `2129624` | Lyris 235B SWE-RL baseline step-1 retry after TE/setuptools and `.pth` patches | `FAILED` | Failed before driver startup with repeated `/tmp/nemo_rl_ray_2129624_3.13.13_2.54.0/bin/ray: No such file or directory`; did not reach the patched TE build or `nemo_gym` import path. |
| `2129593` | Lyris 235B SWE-RL baseline step-1 retry with shared uv cache | `FAILED` | Reached `64/64` workers, then failed while building `transformer-engine`: `ModuleNotFoundError: No module named 'setuptools.build_meta'`. This is after the prior quota and `nemo_gym` source-path issues. |
| `2129556` | Lyris 235B SWE-RL baseline step-1 retry after `.pth` patch | `FAILED` | Reached `64/64` workers, then failed on TE/CUTLASS checkout with `Disk quota exceeded`; replaced by `2129593`. |
| `2129203` | Lyris 235B SWE-RL baseline step-1 retry | `FAILED` | Failed in `NemoGym.__init__()` with `ModuleNotFoundError: No module named 'nemo_gym'`; patched by adding a source-root `.pth` into actor venvs. |

## Best 235B SWE-RL Example

Use the OCI-HSG after-prewarm N3Post W&B matrix as the closest completed 235B
SWE-RL short-run reference, not as a clean `Step 10/10` proof:

- Tracker: `latest_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1_20260614_jobs.csv`
- Remote repo: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613`
- Launcher: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613/test_assets/qwen-235B/run_grpo_qwen3_235b_swe_scale_gen.sh`
- Run id: `20260614_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1`

Completed cells:

| Job | Method | State | Parsed completed steps | Mean step time | E2E tok/s/GPU | Gen-worker tok/s/GPU |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| `3299487` | baseline | `COMPLETED` | 8 | `1243.29s` | `103.75` | `213.06` |
| `3299489` | PARD K5 | `COMPLETED` | 5 | `1952.39s` | `27.38` | `57.07` |
| `3299491` | Eagle-3 K3 | `COMPLETED` | 7 | `1389.83s` | `50.08` | `103.47` |

Direct driver-log step marker check at `2026-06-15 10:50 PDT`:

| Job | Method | `sacct` | Last driver step marker | Max-step guard logged |
| ---: | --- | --- | --- | --- |
| `3299487` | baseline | `COMPLETED/0:0` | `Step 9/10` | no |
| `3299489` | PARD K5 | `COMPLETED/0:0` | `Step 6/10` | no |
| `3299491` | Eagle-3 K3 | `COMPLETED/0:0` | `Step 8/10` | no |

Failed cells in the same matrix:

| Job | Method | Primary failure |
| ---: | --- | --- |
| `3299488` | suffix K32 | vLLM actor env missed `arctic_inference.suffix_decoding._C`. |
| `3299490` | PARD-2 K1 | Staged PARD-2 vLLM `_C.abi3.so` had a Torch/C10 ABI symbol mismatch. |

Use this matrix when the goal is to demonstrate that a 235B SWE-RL baseline and
non-PARD-2 speculative variants can finish a short NeMo-RL run. Do not use it
as PARD-2 success evidence.

Detailed artifacts:

- `docs/oci_hsg_swerl_fullgrpo_specdec_after_prewarm_n3post_wandb_r1_status_20260614.md`
- `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_20260615.md`
- `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_stepge2_20260615.csv`

Repro handles:

- Jobs and launch metadata: `latest_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1_20260614_jobs.csv`
- Local fetched logs: `tmp/oci_hsg_swerl_fullgrpo_logs_live_extract/20260614_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1/`
- Step metric parser: `scripts/extract_nemorl_fullgrpo_step_metrics.py`
- Recommended citation: use `3299487` for a 235B SWE-RL baseline short-run example; use `3299489` only to show PARD K5 can complete, not to claim a speedup.

## Best Small PARD-2 Online Example

Use the OCI-HSG Qwen8 official PARD-2 comparison when the goal is to show the
online-drafter path itself works end to end:

- Tracker: `latest_oci_hsg_qwen8_pard2_official_comparison_20260613_jobs.csv`
- Run id: `20260613_oci_hsg_qwen8_pard2_official_comparison_masterconfigfix_shortqos`
- Target model: `Qwen/Qwen3-8B`
- PARD-2 drafter: `amd/PARD2-Qwen3-8B`
- Shape: `max_steps=10`, `max_new_tokens=256`, `min_tokens=128`, `num_prompts=4`, `num_generations=4`, `train_global_batch_size=16`

Completed cells:

| Job | Method | State | Steps | Acceptance | Gen-worker speedup vs baseline | E2E speedup vs baseline |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| `3288181` | baseline | `COMPLETED` | 9/9 parsed | | `1.0000x` | `1.0000x` |
| `3288182` | static PARD-2 | `COMPLETED` | 9/9 parsed | `1.836` | `0.6071x` | `0.8291x` |
| `3288183` | online PARD-2 | `COMPLETED` | 9/9 parsed | `2.553` | `0.5887x` | `0.6705x` |

This proves online refit is wired into NeMo-RL and changes acceptance, but it
does not show a throughput win.

Detailed artifacts:

- `docs/oci_hsg_qwen8_pard2_official_comparison_status_20260613.md`
- `docs/qwen8_pard2_official_comparison_metrics_20260613.md`

Repro handles:

- Jobs and launch metadata: `latest_oci_hsg_qwen8_pard2_official_comparison_20260613_jobs.csv`
- Metrics: `docs/qwen8_pard2_official_online_impact_20260613.md`
- Recommended citation: use this matrix to show the online PARD-2 refit path changes acceptance; do not use it as a throughput-win claim.

## Failed/Running 235B Math And PARD-2 Gates

The following are proof gates and supporting failed runs. `3321180` is now a
working 235B MathRL 10-step example; the current 235B PARD-2 proof is still not
working.

| Job | Scope | State at `2026-06-15 11:07 PDT` | Account | Priority | Notes |
| ---: | --- | --- | --- | ---: | --- |
| `3321180` | 235B MathRL reduced-shape 10-step baseline | `COMPLETED/0:0` | `nemotron_n3_post` | `135000` | Started at `2026-06-15T09:57:37` and completed at `2026-06-15T10:27:57` after `00:30:20`. It reached `Step 1/10` through `Step 10/10`, trained policy on Step 10, and stopped via the max-step guard. |
| `2129624` | Lyris 235B SWE-RL baseline step-1 retry after TE/setuptools and `.pth` patches | `FAILED` | `coreai_dlalgo_llm` | | Failed at `2026-06-15T09:56:33` before driver startup with repeated missing `/tmp/nemo_rl_ray_2129624_3.13.13_2.54.0/bin/ray`; no Step 1 metric. |
| `2129715` | Lyris 235B SWE-RL baseline step-1 retry with existing container Ray/Python | `FAILED/2:0` | `coreai_dlalgo_llm` | | Started `2026-06-15T10:16:49` and failed at `2026-06-15T10:18:51`. Ray startup used existing `/opt/nemo_rl_venv`, but the driver failed on Python `3.13.13` not being available. |
| `2129746` | Lyris 235B SWE-RL baseline step-1 retry with Python `3.13.9` Ray venv | `FAILED/1:0` | `coreai_dlalgo_llm` | | Started `2026-06-15T10:25:00` and failed at `2026-06-15T10:26:40`. CPython `3.13.9` venv creation began, but package extraction failed and the Ray executable was absent. |
| `2129556` | Lyris 235B SWE-RL baseline step-1 retry after `nemo_gym` source-path patch | `FAILED` | `coreai_dlalgo_llm` | | Reached `64/64` workers, then failed on TE/CUTLASS checkout with `Disk quota exceeded`; superseded by `2129593` and `2129624`. |
| `2129203` | Lyris 235B SWE-RL baseline step-1 retry | `FAILED` | `coreai_dlalgo_llm` | | Failed at `2026-06-15T09:08:23` after setup. Ray/vLLM and policy workers came up, but `NemoGym.__init__()` failed in `nemo_rl/environments/nemo_gym.py` with `ModuleNotFoundError: No module named 'nemo_gym'`, before any GRPO Step 1 metric. |
| `2129271` | Lyris 235B SWE-RL PARD step-1 retry | `CANCELLED` | `coreai_dlalgo_llm` | | Cancelled because `afterok:2129203` was not satisfied. |
| `2129272` | Lyris 235B SWE-RL PARD-2 step-1 retry | `CANCELLED` | `coreai_dlalgo_llm` | | Cancelled because `afterok:2129203` was not satisfied. |
| `3308774` | 235B SWE-RL PARD-2 step-1 proof | `CANCELLED` | `nemotron_n3_post` | `135103` | Started `2026-06-15T08:46:16` but never reached setup-complete or a parsed GRPO step metric. It repeatedly failed to find the async vLLM worker Python executable under `/opt/ray_venvs_swerl_ray254_r14/...` and is superseded by fresh-venv r18 job `3322475`. |
| `3321070` | 235B MathRL baseline 10-step temp1/fused-loss-off retry | `FAILED` | `nemotron_n3_post` | | Reached `Step 2/10`, then failed during Step 2 generation with vLLM CUDA OOM at `cumem_allocator.cpp:139`. |
| `3320856` | 235B MathRL baseline 10-step fused-loss-off retry | `FAILED` | `nemotron_n3_post` | | Reached `Step 1/10` and `Training policy`, then failed with local grad-norm `NaN`. |
| `3315380` | 235B MathRL baseline 10-step | `FAILED` | `nemotron_n3_post` | | Reached `Step 1/10`, generated responses, computed logprobs/advantages, then failed during `Training policy` in packed loss input: `_pack_input_ids` tried to copy length `1216` from an `input_ids` row of width `408`. No completed step metric. |
| `3315381` | 235B MathRL PARD K3 10-step | `FAILED` | `nemotron_n3_post` | | Reached `Step 1/10`, generated responses, computed logprobs/advantages, then failed during `Training policy` with the same packed loss input mismatch, `1216` vs `408`. No completed step metric. |
| `3315382` | 235B MathRL PARD K5 10-step | `FAILED` | `nemotron_n3_post` | | Reached `Step 1/10`, generated responses, computed logprobs/advantages, then failed during `Training policy` with packed loss input mismatch, `560` vs `408`. No completed step metric. |
| `2126895` | Superseded Lyris 235B SWE-RL baseline step-1 gate | `FAILED` | `coreai_dlalgo_llm` | | Failed at `2026-06-15T07:16:07` with `sacct_exit=1:0`; TransformerEngine built successfully, then the driver hit Ray/Python version mismatch: cluster `Ray 2.49.2`/`Python 3.12.13` versus driver `Ray 2.54.0`/`Python 3.13.13`; no step metrics. |
| `2128989` | Superseded Lyris 235B SWE-RL baseline step-1 gate | `FAILED` | `coreai_dlalgo_llm` | | r29 fixed the Ray/Python mismatch and reached 64/64 connected actors, but then stalled at TransformerEngine source build in the user persistent cache. User inode quota was over soft quota and close to hard limit, so the chain was cancelled and replaced by r30 using node-local `/tmp` cache. |
| `2128992` | Superseded Lyris 235B SWE-RL PARD step-1 gate | `CANCELLED` | `coreai_dlalgo_llm` | | Cancelled because r29 baseline `2128989` was superseded; stdout missing. |
| `2128993` | Superseded Lyris 235B SWE-RL PARD-2 step-1 gate | `CANCELLED` | `coreai_dlalgo_llm` | | Cancelled because r29 baseline `2128989` was superseded; stdout missing. |
| `2126914` | Superseded Lyris 235B SWE-RL PARD step-1 gate | `CANCELLED` | `coreai_dlalgo_llm` | | Cancelled because `afterok:2126895` was not satisfied; stdout missing. |
| `2126915` | Superseded Lyris 235B SWE-RL PARD-2 step-1 gate | `CANCELLED` | `coreai_dlalgo_llm` | | Cancelled because `afterok:2126895` was not satisfied; stdout missing. |

First files to inspect when the jobs start, or failed logs to cite for terminal jobs:

| Job | Slurm stdout |
| ---: | --- |
| `2129556` | `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/../swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_raymatch_tmpcache_pth_r31/baseline_steps1/slurm-2129556.out` |
| `2129203` | `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/../swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_raymatch_tmpcache_r30/baseline_steps1/slurm-2129203.out` |
| `2129271` | `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/../swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_raymatch_tmpcache_r30_afterok/pard_steps1/slurm-2129271.out` |
| `2129272` | `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/../swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_raymatch_tmpcache_r30_afterok/pard2_steps1/slurm-2129272.out` |
| `3308774` | `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/../swerl_fullgrpo_logs/20260614_oci_hsg_swerl_qwen235b_fullgrpo_pard2_pyoverlay_eager_nolevel_r16/pard2_steps1/slurm-3308774.out` |
| `3320856` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_fuseloss_off_baseline/baseline/slurm-3320856.out` |
| `3315380` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_latest_main_guards_n3post_systempy/baseline/slurm-3315380.out` |
| `3315381` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_latest_main_guards_n3post_systempy/pard_k3/slurm-3315381.out` |
| `3315382` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_latest_main_guards_n3post_systempy/pard_k5/slurm-3315382.out` |

Current monitoring artifacts:

- `docs/nemorl_235b_active_gates_latest_20260615.md`
- `docs/nemorl_235b_gate_runtime_report_latest_20260615.md`
- `docs/nemorl_235b_active_gates_history_20260615.csv`
- `docs/nemorl_235b_active_gates_changes_latest_20260615.md`
- `docs/nemorl_235b_failed_step1_metrics_20260615.csv`
- `scripts/inspect_nemorl_235b_gate_first_logs_20260615.sh`
- `scripts/monitor_nemorl_235b_gates_until_runtime_20260615.sh`
- `scripts/fetch_and_parse_nemorl_235b_ready_gate_metrics.py`

The latest monitor at `2026-06-15 09:11 PDT` found `3315380`, `3315381`, and
`3315382` terminal failed. All three entered `Step 1/10`, but the parser records
them as `incomplete_step`; there are no completed 235B MathRL step rows from
this retry.

Use `SYNC_CANONICAL_DOCS=true` with the monitor wrapper when the per-poll
snapshot should also update the canonical `docs/nemorl_235b_active_gates_*`
files in the same pass.
The latest monitor already fetched the redacted `ray-driver.log` for Lyris
`2129203`; the ready-gate step metric outputs are empty because that run failed
during `NemoGym` actor creation before a parsed training step appeared.

## Math Results That Are Not MathRL Training

There are successful Math/MATH500 speculative decoding benchmarks, but these
are standalone vLLM generation runs rather than NeMo-RL training examples:

| Artifact | Scope | Read |
| --- | --- | --- |
| `docs/oci_qwen235b_math500_osl32k_status_20260613.md` | 235B OCI-HSG MATH500 standalone | PARD K5 and Eagle-3 completed; baseline and official PARD-2 had timeout/failed status, though metric rows exist for several standalone breakdowns. |
| `docs/lyris_math500_osl32k_status_20260612.md` | Qwen8/Qwen30 Lyris MATH500 standalone | Baseline, suffix, PARD, and Eagle-3 completed for smaller models; Qwen8 official PARD-2 timed out at the job level but emitted metrics. |

Use these for standalone decoding performance discussion only. They should not
be cited as proof that 235B MathRL NeMo-RL training reaches 10 steps.

## Non-Usable Integrated Math/SpecDec Attempts

Do not use the older Lyris integrated max-step-10 matrix as a working Math or
SpecDec training example:

- `docs/lyris_nemorl_integrated_specdec_maxsteps10_status_20260613.md`
- `docs/lyris_nemorl_integrated_specdec_maxsteps10_metrics_20260613.md`

That matrix is terminal with `FAILED=12` and `CANCELLED by 2001147693=6`;
the metric summary has `missing_log=18`. It is useful only as negative
historical evidence.

The older 235B MathRL latest-main attempts are also negative or superseded:

- OCI-HSG `3290316`, `3290317`, and `3290318` were cancelled before runtime.
- Lyris `2113550`/`2113551`, `2113662`/`2113663`, and
  `2113744`/`2113745` failed during early import/setup compatibility checks.
- Lyris retry3 `2113812`, `2113813`, and `2113814` progressed through the
  earlier Python/decord/soundfile/tensordict blockers, but failed during
  isolated policy worker creation with
  `ModuleNotFoundError: No module named 'transformers.models.ernie4_5_vl_moe'`.
- OCI-HSG py3 guard jobs `3315267`, `3315268`, and `3315269` were cancelled
  before runtime and are superseded by the later `3315380`/`3315381`/`3315382`
  failures and the completed reduced-shape MathRL run `3321180`.

## Practical Recommendation

For a 10-step example today:

1. Use `3321180` as the clean 235B MathRL reduced-shape `Step 10/10` example.
2. Use `3299487` only as the closest completed 235B SWE-RL baseline short-run;
   it is not a clean `Step 10/10` proof from the driver log.
3. Use `3299489` only as "PARD K5 can complete", not as a performance win or a
   clean `Step 10/10` proof.
4. Use `3288181`/`3288182`/`3288183` for the online PARD-2 mechanics example.
5. Do not use `3308774` as a 235B SWE-RL PARD-2 success claim; the current run
   is stuck before Step 1 with the missing async vLLM worker venv path.
6. Use `3321423`/`3321424` as 235B MathRL static PARD K3/K5 10-step examples;
   both completed cleanly with `Step 10/10`, Step 10 policy training, max-step
   guard, and SLURM exit `0:0`.
7. Watch `3321785`/`3321786` for MathRL static-vs-online PARD-2 impact; both
   are submitted but pending.
8. Do not use current Lyris SWE-RL retries as examples yet; `2129715` and
   `2129746` both failed before a GRPO step.
9. Do not use `3315380`/`3315381`/`3315382` as 235B MathRL examples; they only
   prove the current MathRL setup reaches Step 1 and then fails in packed-loss
   training.

## 2026-06-15 14:11 PDT PARD-2 Refresh

Superseded by the 14:35 PDT addendum above: `3323893` and `3324200` are now
terminal failed, with replacements `3324365` and `3324460` queued.

- SWE-RL r20 (`3322947`) is not a usable example. It reached async vLLM startup
  and then failed during PARD-2 drafter load because the configured local draft
  snapshot did not contain `warp_model.bin`; vLLM raised `HFValidationError`
  after treating the absolute snapshot path as a Hub repo id. I cancelled it.
- SWE-RL r21 (`3324256`) is the current replacement. It uses the corrected
  PARD-2 snapshot, fresh `/opt/ray_venvs_swerl_ray254_r21` actor venvs, and
  `nemotron_n3_post`; it had just started and had no driver log yet.
- MathRL online r21 (`3323893`) and static r22 (`3324200`) are both running,
  have initialized `128/128` vLLM workers, and are loading checkpoint shards.
  They have not yet proven Step 1 completion after the raw-counter patch.
- A 14:12 PDT follow-up showed both MathRL PARD-2 jobs entering PARD-2 drafter
  load and logging `Using PARD-2 target layers from draft config: (94, 87, 79,
  71)`. This proves the corrected warp snapshot path is being used; it still
  does not yet prove a completed Step 1.
- SWE-RL r21 (`3324256`) failed before driver startup because
  `UV_PYTHON_DOWNLOADS` was passed as an empty value. The submit script now
  defaults this to `auto`, and r22 (`3324276`) is the active replacement. At
  14:17 PDT r22 was still running at elapsed `00:02:04`, with no driver log yet.
