# NeMo-RL 235B Active Gate Snapshot

Manual verification addendum at `2026-06-15 21:03 PDT`:

- Lyris remains usable from this shell. SWE-RL baseline 1-step smoke r40
  `2133287` failed after `00:01:20` (`20:59:25` to `21:00:45`) because the
  worker containers could not mount node-local override cache paths:
  `enroot-mount: failed to mount: /tmp/uv_cache_swerl_r40 ... No such file or
  directory`. This is a pyxis host-path creation issue before Ray worker
  startup, not a NeMo-RL Step failure.
- I submitted r41 `2133292` with the same Python/template-project fix but
  without `UV_CACHE_DIR_OVERRIDE`/`PIP_CACHE_DIR_OVERRIDE`/
  `TORCH_EXTENSIONS_DIR_OVERRIDE`, matching the r39 path that reached
  `64/64` actors. At `21:03:28 PDT`, r41 is `PENDING` with
  `RUN_ID=20260615_lyris_swerl_qwen235b_fullgrpo_templatepy3139_nooverride_r41`.
- r39 `2133251` is now confirmed `FAILED 2:0`. It got past the previous Ray
  bootstrap/cache blocker and reached `64/64` actors, but the Ray driver
  exited because uv resolved Python `3.13.9` while workspace member
  `research/template_project/pyproject.toml` still required
  `>=3.13.13`. I patched the SWE-RL submit path to lower both top-level
  `pyproject.toml` and `research/template_project/pyproject.toml` to
  `>=3.13.9` before submit; r40 includes that fix.
- OCI MathRL r2 split into method-specific outcomes. Eagle-3 K3 `3331856`
  is still `RUNNING` and has passed `SETUP COMPLETE`, `Step 1/5`,
  `Step 2/5`, `Step 3/5`, and is in `Step 4/5` generation. Static PARD-2 K3
  `3331857` is still `RUNNING`; it has shown HF Hub `429` warnings during
  setup but no fatal exit yet.
- OCI MathRL r2 baseline `3331855` failed in `prepare_refit_info()` with
  `FileNotFoundError: No .safetensors files or index found in
  Qwen/Qwen3-235B-A22B`. The run used a local tokenizer path but left
  `policy.model_name` as the HF repo id, so Megatron Bridge tried to inspect
  the remote repo id as if it were a local snapshot.
- OCI MathRL r2 online PARD-2 `3331858` failed in vLLM actor creation from
  HF Hub API throttling: `429 Too Many Requests` on
  `https://huggingface.co/api/models/Qwen/Qwen3-235B-A22B/tree/...`.
- I patched the MathRL launcher so both `TARGET_MODEL` and `TOKENIZER_NAME`
  default to the local Qwen3-235B snapshot, and made
  `NRL_MEGATRON_CHECKPOINT_DIR` method-specific to avoid concurrent
  checkpoint races. Contract validation passes. Replacement r3 jobs are queued:
  baseline `3332282` and online PARD-2 `3332283`, both `PENDING|Priority` at
  `21:00:04 PDT`.

Manual verification addendum at `2026-06-15 20:43 PDT`:

- Lyris is usable again from this shell. Fresh check reached
  `login-lyris02.lyris.clusters.nvidia.com`, `klist_rc=0`, and both
  `squeue`/`sacct` work. Before the new retry there were no active Lyris jobs
  for `sna`.
- Lyris standalone temp=1/top_p=1 status is unchanged: SWE standalone jobs
  completed, while Math500 baseline `2124147` and PARD-2 `2124150` timed out
  at 5 hours. The earlier Lyris SWE-RL retries `2129614`, `2129624`,
  `2129715`, and `2129746` are all failed; r35 failed on missing Python
  `3.13.13`, and r36 failed because uv used `/root/.cache/uv` and hit disk
  quota before `bin/ray` existed.
- I patched the Lyris SWE-RL launcher so Ray sbatch/driver command propagation
  includes the persistent cache variables (`UV_CACHE_DIR`, `PIP_CACHE_DIR`,
  `TORCH_EXTENSIONS_DIR`, overrides, and uv timeouts). I also changed the
  Lyris defaults to the user-owned SWE repo and visible nightly container.
  Contract validation passes.
- Submitted Lyris SWE-RL baseline 1-step smoke r37 as job `2133224`:
  `RUN_ID=20260615_lyris_swerl_qwen235b_fullgrpo_cacheenv_py31313_r37`,
  Ray `2.54.0`, Python `3.13.13`, persistent uv Python/cache dirs, and final
  `logger.wandb_enabled=false`. It failed quickly (`20:42:18` to `20:43:48`)
  because the container could not find managed Python `3.13.13`; consequently
  `/tmp/nemo_rl_ray_2133224_3.13.13_2.54.0/bin/ray` did not exist.
- Submitted Lyris SWE-RL baseline 1-step smoke r38 as job `2133226` with
  Python `3.13.9` (the version that got farther in r36) and explicit
  sbatch-exported `UV_CACHE_DIR`, `PIP_CACHE_DIR`, and
  `TORCH_EXTENSIONS_DIR` on Lustre to avoid the r36 `/root/.cache/uv` quota
  failure. It also failed quickly (`20:45:25` to `20:46:59`) with uv
  extraction `Disk quota exceeded`, so the Lustre cache path itself is not a
  sufficient fix.
- Submitted Lyris SWE-RL baseline 1-step smoke r39 as job `2133251`, keeping
  Python `3.13.9` but moving Ray/uv/pip/torch caches and venv dirs to
  node-local `/tmp` paths. At `20:48:06 PDT`, it is `RUNNING` since
  `20:48:01`. At `20:49:51`, the Ray head bootstrap had passed the prior
  failure point: CPython `3.13.9`, `Prepared 57 packages`, `Installed 57
  packages`, `ray start --head`, and `Local node IP` are present in
  `ray-head.log`.
- OCI MathRL generation-bound r2 is now fully allocated. At `20:43:08 PDT`,
  `3331855` baseline, `3331856` Eagle-3 K3, `3331857` static PARD-2 K3, and
  `3331858` online PARD-2 K3 are all `RUNNING`. Driver logs confirm the final
  config has `context_parallel_size=1` and `sequence_packing.enabled=false`.
  PARD-2 static/online are creating actor venvs under the shared Lustre
  `.actor_venvs/mathrl_pard2_genbound1024_r2` path, not node-local `/opt`.
  No new fatal error or Step completion marker has appeared yet.

Manual verification addendum at `2026-06-15 20:30 PDT`:

- MathRL generation-bound r1 baseline `3330755` is `FAILED` after reaching
  `SETUP COMPLETE`, entering `Step 1/5`, generating the 32-response batch,
  computing logprobs/advantages, and starting policy training. The failure was
  in packed loss input construction:
  `RuntimeError: The expanded size of the tensor (1312) must match the existing
  size (1168)` at `nemo_rl/algorithms/loss/utils.py`, `_pack_input_ids`.
- I cancelled stale r1 jobs `3330756` Eagle-3, `3330757` static PARD-2, and
  `3330758` online PARD-2. Eagle-3 had the same baseline `CP=2` /
  `sequence_packing=true` training-risk path. Static PARD-2 was already
  repeating `bin/python: No such file or directory` because serialized actor
  venv creation wrote under node-local `/opt/ray_venvs`; online PARD-2 used
  the same risky actor-venv path.
- I patched the Math launcher so PARD-2 static/online use a shared Lustre actor
  venv root, and copied the robust remote `venvs.py` helper with
  `READY_ENV_BUILDER` markers into the Math repo. The local contract validator
  now checks that PARD-2 no longer serializes into `/opt`.
- Submitted generation-bound MathRL r2 under `nemotron_n3_post`:
  `3331855` baseline, `3331856` Eagle-3 K3, `3331857` static PARD-2 K3, and
  `3331858` online PARD-2 K3. All use `max_new_tokens=min_tokens=1024`,
  `temperature=1.0`, `top_p=1.0`, `top_k=-1`, `max_steps=5`,
  `num_prompts=4`, `num_generations=8`, and `train_global_batch_size=32`.
  r2 also forces `policy.megatron_cfg.context_parallel_size=1` and
  `policy.sequence_packing.enabled=false` for all four methods to avoid the r1
  packed-loss mismatch. At `20:31:17 PDT`, baseline `3331855`, Eagle-3
  `3331856`, and static PARD-2 `3331857` are `RUNNING`; online PARD-2
  `3331858` remains `PENDING|Priority` with estimated start
  `2026-06-15T20:43:08`.
- SWE-RL PARD-2 r28 `3331533` was cancelled after `00:05:26`; it never reached
  actor setup. The live log looped at `0/64` actors because the head container
  could not execute `/tmp/nemo_rl_ray_3331533_3.12.13_2.54.0/bin/ray`
  (`No such file or directory`). This is a Ray runtime venv materialization
  issue before the previous NCCL timeout patch is exercised.

Manual verification addendum at `2026-06-15 20:01 PDT`:

- SWE-RL PARD-2 r27 `3327299` is now `FAILED` after `47:00` elapsed
  (`2026-06-15T19:05:01` to `2026-06-15T19:52:01`). It reached
  `SETUP COMPLETE`, total setup `1744.9s`, and rollout collection up to
  `Collecting rollouts: 29/32`, but then died during policy-to-generation
  refit/broadcast from the Megatron tensor-model-parallel group NCCL watchdog:
  `WorkNCCL(SeqNum=197, OpType=ALLGATHER, Timeout(ms)=600000) ran for
  600087 milliseconds`. This is not the earlier `draft.fc.weight` failure and
  not an OOM signature.
- I patched both remote Math and SWE repos with `NRL_MEGATRON_PG_TIMEOUT_V1`
  so Megatron model-parallel process groups receive the requested timeout, and
  updated the local SWE submit wrapper to pass
  `NRL_MEGATRON_NCCL_TIMEOUT_SECONDS` and
  `NRL_MEGATRON_PROCESS_GROUP_TIMEOUT_SECONDS` into the Ray driver/actor
  command path. The contract validator now checks this.
- Submitted the SWE-RL PARD-2 retry under `nemotron_n3_post` as job `3331533`
  with `RUN_ID=20260615_oci_hsg_swerl_qwen235b_fullgrpo_pard2_pgtimeout_r28`,
  `max_steps=1`, new actor venv suffix
  `swe_nemo_gym_source_miniforgebind_len65k_r28_pgtimeout3600`, W&B disabled,
  and Megatron PG timeout `3600s`. It is currently `PENDING|Priority` with
  estimate `2026-06-15T20:21:17`.
- MathRL generation-bound baseline `3330755` and Eagle-3 K3 `3330756` remain
  `RUNNING`. Both reached `All workers connected!` and are now building the
  vLLM actor environments; neither has `SETUP COMPLETE` or Step markers yet.
  There are no fatal errors in the current driver tails. Static PARD-2 K3
  `3330757` and online PARD-2 K3 `3330758` remain `PENDING|Priority`, both
  currently estimated for `2026-06-15T20:21:17`.

Manual verification addendum at `2026-06-15 19:49 PDT`:

- OCI-HSG current live queue: SWE-RL PARD-2 r27 `3327299`, MathRL
  generation-bound baseline `3330755`, and MathRL generation-bound Eagle-3 K3
  `3330756` are all `RUNNING`. Static PARD-2 K3 `3330757` and online PARD-2
  K3 `3330758` remain `PENDING|Priority`.
- The generation-bound MathRL comparison is now partially allocated:
  `3330755` baseline has been running since `2026-06-15T19:34:34`, and
  `3330756` Eagle-3 K3 since `2026-06-15T19:43:22`. Both have emitted
  `All workers connected!`, but no `SETUP COMPLETE` or Step marker yet.
  Pending estimates are `2026-06-15T20:21:17` for static PARD-2 `3330757`
  and `2026-06-15T20:52:39` for online PARD-2 `3330758`.
- Math online PARD-2 fixed-256 `3324571` is now `FAILED`. It reached
  `SETUP COMPLETE`, entered `Step 1/10`, and began
  `Generating responses for batch of size 64`; setup timing was
  `vLLM init: 1212.9s`, `Policy init: 99.6s`, `Other setup: 316.6s`,
  `Total setup: 1629.2s`. It failed during the first policy-to-generation
  refit with `KeyError: 'draft.fc.weight'` in
  `VllmInternalWorkerExtension.update_weights_via_ipc_zmq`, followed by the
  policy streamer timing out after the generation worker died.
- I patched the root cause in both Math and SWE remote repos: policy
  `prepare_refit_info()` now merges metadata from all policy workers and logs
  `NRL_REFIT_INFO_MERGE_V1`, so draft keys such as `draft.fc.weight` are not
  dropped when only a non-first worker reports them. The local Math launcher
  also applies this hotfix before future submissions.
- SWE-RL PARD-2 r27 `3327299` is still `RUNNING` since
  `2026-06-15T19:05:01`. Its transient `ray status` traceback was non-fatal;
  the job then reached `64/64` actors and `All workers connected!`. The
  `ray-driver.log` shows active SWE rollout collection, latest observed at
  `Collecting rollouts: 29/32`, but no NeMo-RL Step marker yet.

Manual verification addendum at `2026-06-15 19:30 PDT`:

- OCI-HSG access is working again from this shell. Current live queue state:
  Math online PARD-2 fixed-256 run `3324571` is `RUNNING` since
  `2026-06-15T19:05:01`, and SWE-RL PARD-2 r27 `3327299` is also `RUNNING`
  since `2026-06-15T19:05:01`. Neither has emitted `SETUP COMPLETE` or
  `Step` markers yet.
- Math static PARD-2 fixed-256 run `3324570` failed at
  `2026-06-15T19:17:43` before `SETUP COMPLETE` or `Step 1`. The driver
  error was actor venv creation via `uv run --locked --extra mcore`, ending
  with missing Megatron-Core native helper
  `megatron/core/datasets/helpers_cpp.cpython-313-aarch64-linux-gnu.so`.
  I treated this as a shared editable-build / actor-venv materialization
  failure, not a PARD-2 algorithmic Step failure.
- Per request, I submitted a generation-bound MathRL batch with longer output
  length and RL sampling settings: `max_new_tokens=1024`, `min_tokens=1024`,
  `temperature=1.0`, `top_p=1.0`, `top_k=-1`, `max_steps=5`,
  `num_prompts=4`, `num_generations=8`, and `train_global_batch_size=32`.
  The four jobs are `3330755` baseline, `3330756` Eagle-3 K3, `3330757`
  static PARD-2 K3, and `3330758` online PARD-2 K3. All four are currently
  `PENDING|Priority`. Latest `squeue` estimates are `2026-06-15T19:43:43`
  for baseline `3330755`, `2026-06-15T20:05:01` for Eagle-3 `3330756`,
  `2026-06-15T20:21:17` for static PARD-2 `3330757`, and
  `2026-06-15T21:39:44` for online PARD-2 `3330758`.
- The longer-output PARD-2 jobs use `NRL_ACTOR_UV_LOCK_MODE=unlocked` and
  `NRL_SERIALIZE_ACTOR_VENV_CREATION=true` so the previous static PARD-2
  `helpers_cpp` venv-build race is not repeated if the jobs allocate.

Manual access note at `2026-06-15 19:07 PDT`:

- I attempted a fresh live poll from this local shell, but the local resolver is
  pointed at public DNS (`8.8.8.8`) and cannot resolve the internal SSH aliases
  `oci-hsg-cs-001-vscode-02` or `login-lyris` (`NXDOMAIN`). No newer SLURM
  state was collected in this attempt.
- The latest successful remote state remains the `18:47 PDT` OCI-HSG/Lyris
  poll below. This is an access/DNS limitation in the current shell, not a new
  job failure signal.

Manual verification addendum at `2026-06-15 18:47 PDT`:

- OCI-HSG active proof gates remained pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  were all still `PENDING|Priority`; `sacct` still reported `Start=Unknown`,
  `End=Unknown`, and no node assignment for all three.
- Latest estimates were `2026-06-15T18:52:03` for Math static `3324570`,
  `2026-06-15T19:17:49` for Math online `3324571`, and
  `2026-06-15T20:21:17` for SWE-RL r27 `3327299`.
- Priorities were `129934` for the two Math jobs and `129917` for SWE r27.
  `scontrol show job -dd` showed no actual `NodeList`; candidate placement was
  empty/not printed at this poll.
- All three active log roots still had no runtime evidence: no `slurm-*.out`,
  no `ray-driver.log`, no setup marker, no Step marker, and no actionable
  failure.
- Lyris was still SSH-reachable in the earlier poll path, but Slurm remained
  inaccessible: `/home/sna/.bashrc` and `/etc/slurm/slurm.conf` returned
  `Permission denied`, `klist -s` returned `1`, and `squeue`/`sacct` could not
  query jobs.

Manual verification addendum at `2026-06-15 18:43 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no actual node assignment for all three.
- Latest estimates are `2026-06-15T18:52:03` for Math static `3324570`, and
  `2026-06-15T19:13:02` for both Math online `3324571` and SWE-RL r27
  `3327299`.
- Priorities remain `129933` for the two Math jobs and `129916` for SWE r27.
  `scontrol` still shows candidate `SchedNodeList` placements, but actual
  `NodeList` is empty.
- All three active log roots are still empty of runtime evidence: no
  `slurm-*.out`, no `ray-driver.log`, no setup marker, no Step marker, and no
  actionable failure.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:41 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no actual node assignment for all three.
- Latest estimates are `2026-06-15T18:52:03` for Math static `3324570`,
  `2026-06-15T19:13:02` for Math online `3324571`, and
  `2026-06-15T20:00:00` for SWE-RL r27 `3327299`.
- Priorities remain `129933` for the two Math jobs and `129916` for SWE r27.
  `scontrol` still shows candidate `SchedNodeList` placements, but actual
  `NodeList` is empty.
- All three active log roots are still empty of runtime evidence: no
  `slurm-*.out`, no `ray-driver.log`, no setup marker, no Step marker, and no
  actionable failure.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:39 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no actual node assignment for all three.
- Latest estimates are `2026-06-15T18:52:03` for Math static `3324570`,
  `2026-06-15T19:13:02` for SWE-RL r27 `3327299`, and
  `2026-06-15T19:17:49` for Math online `3324571`.
- Priorities remain `129933` for the two Math jobs and `129916` for SWE r27.
  `scontrol` still shows candidate `SchedNodeList` placements, but actual
  `NodeList` is empty.
- All three active log roots are still empty of runtime evidence: no
  `slurm-*.out`, no `ray-driver.log`, no setup marker, no Step marker, and no
  actionable failure.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:37 PDT`:

- I polled OCI-HSG five times from `18:32:43 PDT` through `18:37:17 PDT`.
  The active proof gates remained pre-allocation throughout. Math static
  PARD-2 `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27
  `3327299` are all still `PENDING|Priority`; `sacct` still reports
  `Start=Unknown` and no actual node assignment for all three.
- During polling, Math priorities ticked up from `129932` to `129933`, and
  SWE r27 ticked up from `129915` to `129916`.
- Latest estimates at `18:37:17 PDT` are `2026-06-15T18:52:03` for Math
  static `3324570`, `2026-06-15T19:17:49` for Math online `3324571`, and
  `2026-06-15T19:13:02` for SWE-RL r27 `3327299`.
- All three active log roots are still empty of runtime evidence: no
  `slurm-*.out`, no `ray-driver.log`, no setup marker, no Step marker, and no
  actionable failure.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:31 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no actual node assignment for all three.
- Latest estimates are `2026-06-15T18:52:00` for Math static `3324570`,
  `2026-06-15T18:52:03` for Math online `3324571`, and
  `2026-06-15T19:13:02` for SWE-RL r27 `3327299`. The SWE estimate moved
  earlier again, but still has no allocation.
- Priorities remain `129932` for the two Math jobs and `129915` for SWE r27.
  `scontrol` still shows candidate `SchedNodeList` placements, but actual
  `NodeList` is empty.
- All three active log roots are still empty of runtime evidence: no
  `slurm-*.out`, no `ray-driver.log`, no setup marker, no Step marker, and no
  actionable failure.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:29 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no actual node assignment for all three.
- Latest estimates shifted again: SWE-RL r27 `3327299` moved earlier to
  `2026-06-15T20:25:06`, while Math static/online PARD-2 `3324570`/`3324571`
  both moved later to `2026-06-15T20:52:39`.
- Priorities are `129932` for the two Math jobs and `129915` for SWE r27.
  `scontrol` still shows candidate `SchedNodeList` placements, but actual
  `NodeList` is empty.
- The user's visible OCI-HSG queue also contains older/lower-priority 32-node
  jobs, but they are behind the active gates by priority. I did not cancel or
  replace them in this cycle.
- All three active log roots are still empty of runtime evidence: no
  `slurm-*.out`, no `ray-driver.log`, no setup marker, no Step marker, and no
  actionable failure.

Manual verification addendum at `2026-06-15 18:26 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no actual node assignment for all three.
- Math `3324570` and `3324571` priority ticked up to `129932`; SWE r27
  `3327299` is `129915`. Latest estimates are `2026-06-15T18:52:00` for
  Math static `3324570`, `2026-06-15T18:52:03` for Math online `3324571`,
  and `2026-06-15T20:52:39` for SWE-RL r27 `3327299`.
- `scontrol show job -dd` still reports candidate `SchedNodeList` placements
  for all three jobs, but the actual `NodeList` is empty. A nearby
  `nemotron_n3_post` queue snapshot shows several 1-node eval jobs running at
  nearby priorities, while these 32-node Math jobs are waiting for a large
  placement to open.
- All three log roots are still empty of runtime evidence: no `slurm-*.out`,
  no `ray-driver.log`, no setup marker, no Step marker, and no actionable
  failure.
- I did not submit a smaller replacement in this cycle. The current 32-node
  10-step Math PARD-2 gates preserve the target proof scope; changing to a
  smaller proof would be a separate, weaker signal rather than a replacement
  for these gates.

Manual verification addendum at `2026-06-15 18:24 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no actual node assignment for all three.
- The near-current `3324570` start estimate did not materialize. Latest
  estimates drifted to `2026-06-15T18:52:00` for Math static `3324570`,
  `2026-06-15T18:52:03` for Math online `3324571`, and
  `2026-06-15T20:52:39` for SWE-RL r27 `3327299`.
- All three log roots are still empty of runtime evidence: no `slurm-*.out`,
  no `ray-driver.log`, no setup marker, no Step marker, and no actionable
  failure.
- This is queue wait, not a Nemo-RL launch failure. No patch or resubmission
  was triggered.

Manual verification addendum at `2026-06-15 18:21 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no actual node assignment for all three.
- `scontrol show job -dd` now has scheduler candidate nodes for all three
  gates. `3324570` has priority `129931` and `SchedNodeList` spanning
  `nvl72004`/`nvl72150`; `3324571` has priority `129931` and a candidate
  `nvl72090`/`nvl72158` placement; `3327299` has priority `129914` and
  candidate `nvl72065-T[01-16]`.
- Latest start estimates are `2026-06-15T19:17:48` for Math static
  `3324570`, `2026-06-15T20:21:17` for Math online `3324571`, and
  `2026-06-15T20:52:39` for SWE-RL r27 `3327299`.
- All three log roots are still empty of runtime evidence: no `slurm-*.out`,
  no `ray-driver.log`, no setup marker, no Step marker, and no actionable
  failure.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:19 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no node assignment for all three.
- The temporary `18:18:35` start estimate did not materialize. Latest
  estimates drifted to `2026-06-15T19:17:49` for Math static `3324570`,
  `2026-06-15T20:21:17` for SWE-RL r27 `3327299`, and
  `2026-06-15T20:52:39` for Math online `3324571`.
- All three log roots are still empty of runtime evidence: no `slurm-*.out`,
  no `ray-driver.log`, no setup marker, no Step marker, and no actionable
  failure.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:16 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no node assignment for all three.
- Latest `squeue --start` estimates remain `2026-06-15T18:42:26` for Math
  static `3324570`, `2026-06-15T19:17:48` for Math online `3324571`, and
  `2026-06-15T20:21:17` for SWE-RL r27 `3327299`.
- All three log roots are still empty of runtime evidence: no `slurm-*.out`,
  no `ray-driver.log`, no setup marker, no Step marker, and no actionable
  failure.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:15 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no node assignment for all three.
- Latest `squeue --start` estimates are `2026-06-15T18:42:26` for Math static
  `3324570`, `2026-06-15T19:17:48` for Math online `3324571`, and
  `2026-06-15T20:21:17` for SWE-RL r27 `3327299`.
- All three log roots are still empty of runtime evidence: no `slurm-*.out`,
  no `ray-driver.log`, no setup marker, no Step marker, and no actionable
  failure.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:13 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all still `PENDING|Priority`; `sacct` still reports `Start=Unknown` and
  no node assignment for all three.
- Latest `squeue` start estimates moved again: `2026-06-15T18:52:00` for
  Math static `3324570`, `2026-06-15T19:17:49` for Math online `3324571`,
  and `2026-06-15T19:37:27` for SWE-RL r27 `3327299`.
- There are still no runtime logs or Step markers from the active gates, so no
  new success/failure evidence exists beyond the queue state.

Manual verification addendum at `2026-06-15 18:11 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all `PENDING|Priority`; `sacct` still reports `Start=Unknown` and no
  node assignment for all three.
- Latest `squeue --start` estimates are `2026-06-15T19:17:48` for
  Math static `3324570`, `2026-06-15T19:37:27` for SWE-RL r27 `3327299`,
  and `2026-06-15T20:16:46` for Math online `3324571`.
- Log roots exist but contain no new `slurm-*.out`, `ray-driver.log`, setup
  marker, Step marker, or actionable failure for these active gates.
- Lyris remains SSH-reachable but not Slurm-queryable: `klist -s` returns `1`,
  the ticket cache shows service tickets expired at `2026-06-15 16:08:40 PDT`,
  `kinit -R` fails with `Ticket expired while renewing credentials`,
  `/home/sna/.bashrc` is denied, and `squeue`/`sacct` fail on
  `/etc/slurm/slurm.conf: Permission denied` before querying any jobs.

Manual verification addendum at `2026-06-15 18:06 PDT`:

- OCI-HSG active proof gates remain pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all `PENDING|Priority`; `sacct` still reports `Start=Unknown` and no
  node assignment for all three.
- Latest `squeue --start` estimates are now `2026-06-15T19:17:48` for
  `3324570`, `2026-06-15T19:37:27` for `3324571`, and
  `2026-06-15T20:16:46` for `3327299`.
- No new `slurm-*.out`, `ray-driver.log`, setup marker, Step marker, or
  actionable failure exists yet for these active gates.
- Lyris remains SSH-reachable but not Slurm-queryable: `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue` fails on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 18:05 PDT`:

- OCI-HSG active proof gates are still pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all `PENDING|Priority`.
- I reduced only the two Math PARD-2 gates from `02:00:00` to `01:00:00` to
  improve backfill fit. This is based on prior 235B Math evidence:
  baseline `3321180` completed 10 steps in `00:30:20`, PARD K3 `3321423`
  in `00:33:37`, PARD K5 `3321424` in `00:33:16`, and the previous static
  PARD-2 attempt `3324200` reached its failure after `00:25:11`.
  SWE r27 stays at `02:00:00` because r25 needed about `27m` just for setup
  and SWE rollout has longer tail risk.
- Latest `squeue --start` estimates after the walltime change are
  `2026-06-15T19:37:27` for Math static `3324570`,
  `2026-06-15T20:21:17` for Math online `3324571`, and
  `2026-06-15T20:21:17` for SWE r27 `3327299`.
- There are still no new runtime logs or Step markers for these three jobs.

Manual verification addendum at `2026-06-15 18:03 PDT`:

- OCI-HSG active proof gates are still pre-allocation. Math static PARD-2
  `3324570`, Math online PARD-2 `3324571`, and SWE-RL patched r27 `3327299`
  are all `PENDING|Priority`; `sacct` has no start time or node assignment.
- Latest `squeue --start` estimates are `2026-06-15T19:17:49` for
  `3324570`, `2026-06-15T19:56:03` for `3324571`, and
  `2026-06-15T21:11:03` for `3327299`.
- I hardened the SWE-RL submit wrapper so future submissions fail preflight if
  the remote Gym `responses_api_agents/swe_agents/app.py` is missing either
  the OpenHands `PYTHONPATH=/openhands_setup/OpenHands` patch or the
  `poetry_shebang` / `shebang_python.parent.parent` miniforge-bind patch that
  fixed r25's rollout-layer failure.
- Validation passed: `bash -n` for the SWE submit wrapper, regenerated
  `docs/swerl_fullgrpo_launcher_contract_validation_20260615.md` with
  `Overall: PASS`, direct remote marker check
  `PYTHONPATH=/openhands_setup/OpenHands=1; poetry_shebang=1;
  shebang_python.parent.parent=1`, and an OCI-HSG dry-run through the new
  remote preflight.
- Lyris remains SSH-reachable but not Slurm-queryable. `klist -s` returns `1`,
  `/home/sna/.bashrc` is denied, and `squeue` still fails on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 17:59 PDT`:

- OCI-HSG r27 SWE-RL PARD-2 `3327299`, Math static PARD-2 `3324570`, and
  Math online PARD-2 `3324571` are still pre-allocation:
  all three are `PENDING|Priority` with no driver logs or stdout evidence.
- Scheduler estimates moved during polling. At `17:55 PDT`, Math static
  `3324570` was estimated at `18:04:10`, Math online `3324571` at
  `18:34:08`, and SWE r27 `3327299` at `18:34:08`. By `17:59 PDT`, estimates
  had drifted to `19:17:49` for `3324570`, `19:56:03` for `3324571`, and
  `21:11:03` for `3327299`.
- No new actionable failure appeared after the r27 miniforge-bind patch.
  The next proof point is allocation plus either `SETUP COMPLETE`/rollout
  markers for SWE r27, or `Step 1/10` for Math static/online PARD-2.
- Lyris remains blocked for Slurm status refresh. SSH through `login-lyris`
  reaches the host, but `klist -s` returns `1`, `/home/sna/.bashrc` is denied,
  and the latest `squeue` call fails before querying jobs with
  `DNS SRV lookup failed` / `Could not establish a configuration source`.

Manual verification addendum at `2026-06-15 17:53 PDT`:

- SWE-RL r25 `3324801` did not reach training `Step 1`. It reached
  `SETUP COMPLETE` and entered NemoGym rollout collection, then repeatedly
  returned empty generation data because the inner OpenHands/SWE-rebench
  `poetry` executable had a bad shebang.
- Final r25 evidence before cancellation: `Step` marker count `0`,
  `bad interpreter` / `return code 126` marker count `124`, and
  `NeMo Gym returned a result with no generation data` marker count `462`.
  The decisive path was
  `/openhands_setup/miniforge3/bin/poetry` pointing at an unmounted
  `/opt/nemo-rl/.../swe_openhands_setup/miniforge3/bin/python3.12`.
- I patched the OCI-HSG remote Gym
  `responses_api_agents/swe_agents/app.py` so `_build_apptainer_command()`
  reads the `poetry` shebang and adds a read-only bind mount from the actual
  Lustre `swe_openhands_setup/miniforge3` directory to the shebang's
  `/opt/nemo-rl/.../swe_openhands_setup/miniforge3` destination. Remote
  `python3 -m py_compile` passed, and the smoke check prints the expected
  `src=<lustre...>/miniforge3` to `dst=/opt/nemo-rl/.../miniforge3` mapping.
- Submitted patched SWE-RL PARD-2 retry r27 `3327299` under
  `nemotron_n3_post` with Ray `2.54.0`, Python `3.13.13`, fresh actor venv
  suffix `swe_nemo_gym_source_miniforgebind_len65k_r27`, one-step PARD-2,
  sequence/model length caps `65568`, `num_prompts_per_step=1`,
  `num_generations_per_prompt=32`, `train_global_batch_size=32`,
  NemoGym concurrency `128`, and final override `logger.wandb_enabled=False`.
  It is `PENDING|Priority` with no dependency; `scontrol` estimates
  `StartTime=2026-06-15T19:17:48`.
- Cancelled superseded r25 `3324801` and stale fallback r26 `3325343` after
  r27 was accepted. r25 was still `COMPLETING` immediately after cancellation;
  `sacct` records it as `CANCELLED by 150081` after `00:43:13`.
- Math static PARD-2 r25 `3324570` and Math online PARD-2 r25 `3324571`
  remain `PENDING|Priority`.
- Lyris SSH works through the `login-lyris` MFA ControlMaster, but Slurm
  refresh remains blocked: `/home/sna/.bashrc` and
  `/etc/slurm/slurm.conf` both return `Permission denied`, and `squeue`,
  `sacct`, and `sinfo` fail before querying jobs.

Manual verification addendum at `2026-06-15 17:42 PDT`:

- SWE-RL r25 `3324801` is still `RUNNING` on `nvl72092-T[01-16]`, elapsed
  `00:33:05` with `01:26:55` remaining at the latest poll.
- The earlier vLLM actor-creation lag has cleared. `ray list actors` now
  reports `100` total actors and `{'ALIVE': 100}` across the expected actor
  classes (`16` `VllmAsyncGenerationWorker`, `23` `MegatronPolicyWorker`,
  `21` `RayWorkerWrapper`, and `40` isolated worker initializers).
- The driver reached `SETUP COMPLETE` with `Total setup time: 1640.0s`.
  Worker initialization timing was `vLLM init: 1597.7s`, `Policy init: 104.2s`,
  and `Other setup: 17.3s`.
- This is the furthest current SWE-RL PARD-2 r25 has reached: Ray/Python,
  NemoGym path, PARD-2 vLLM initialization, CUDA graph capture, and actor
  creation gates have passed. It is now materializing `NemoGym` actor virtual
  environments.
- No `Step 1` marker has appeared yet for r25 as of `17:42 PDT`.
- Existing successful Step-1-plus NeMo-RL examples remain the MathRL jobs:
  `3321180` baseline, `3321423` PARD K3, and `3321424` PARD K5 completed
  `Step 1/10` through `Step 10/10`. Math static PARD-2 `3324200` also reached
  `Step 1/10` and failed later during policy training, so it is a partial
  PARD-2 Step-1 proof but not a clean 10-step run.
- Math static r25 `3324570` and Math online r25 `3324571` remain
  `PENDING|Priority`; fallback SWE-RL r26 `3325343` remains
  `PENDING|Dependency`.

Manual verification addendum at `2026-06-15 17:36 PDT`:

- SWE-RL r25 `3324801` is still `RUNNING`, not terminal. `scontrol` reports
  `JobState=RUNNING`, `RunTime=00:26:58`, `TimeLimit=02:00:00`,
  `Reason=None`, and `NodeList=nvl72092-T[01-16]`.
- The "immediate exit" symptom is from short startup/probe steps, not the
  whole job. `sacct` shows many early one- to four-second `enroot` steps
  failed while probing Ray before the per-job Ray venv existed, with
  `enroot-nsenter: failed to execute:
  /tmp/nemo_rl_ray_3324801_3.13.13_2.54.0/bin/ray: No such file or directory`.
  One later `ray status` probe also hit Ray's
  `AttributeError: 'NoneType' object has no attribute 'decode'`.
- Those probe failures recovered: the same `slurm-3324801.out` later reports
  `64/64` actors and `All workers connected!`. The active main step is
  `3324801.91|enroot|RUNNING`, started at `2026-06-15T17:13:50`.
- Current runtime blocker is still inside PARD-2 vLLM initialization, after
  SWE datasets, worker setup, PARD-2 engine startup, and checkpoint shard load.
  The driver log has not emitted `SETUP COMPLETE` or Step 1 yet.
- Math static r25 `3324570` and Math online r25 `3324571` remain
  `PENDING|Priority`; fallback SWE-RL r26 `3325343` remains
  `PENDING|Dependency`.
- Lyris remains unusable for Slurm refresh from this session: SSH reaches
  `login-lyris02`, but `klist -s` returns `1`, and `squeue`/`sacct` still fail
  because `/etc/slurm/slurm.conf` is unreadable.

Manual verification addendum at `2026-06-15 17:26 PDT`:

- SWE-RL r25 `3324801` is no longer queued; it is `RUNNING` on
  `nvl72092-T[01-16]` with elapsed `00:17:25` at the latest poll. `sacct` reports
  `RUNNING|0:0` and `End=Unknown`.
- The early `slurm-3324801.out` errors
  `enroot-nsenter: failed to execute: /tmp/nemo_rl_ray_3324801_3.13.13_2.54.0/bin/ray:
  No such file or directory` were from startup health-check `srun` probes
  racing ahead of Ray venv creation. They were not terminal: the same Slurm
  log later reached `64/64` actors and `All workers connected!`.
- The Ray/Python mismatch and NemoGym path gates are passed further than the
  prior failures. The r25 driver created a W&B run, loaded the SWE train and
  validation datasets, initialized `32/32` vLLM policy workers and `32/32`
  LM policy workers, started PARD-2 vLLM engines with
  `SpeculativeConfig(method='pard2', num_spec_tokens=1)`, and loaded Qwen3
  checkpoint shards through repeated `100% Completed | 118/118` markers.
- No `SETUP COMPLETE` or Step 1 marker has appeared yet. The driver log size
  was still `170326` bytes at `17:26 PDT` and had not grown since the previous
  poll, so the current state is "running after model/checkpoint setup, awaiting
  the next marker", not a completed Step 1 proof.
- Math static r25 `3324570` and Math online r25 `3324571` remain
  `PENDING|Priority`; fallback SWE-RL r26 `3325343` remains
  `PENDING|Dependency` on `afternotok:3324801`.
- Lyris SSH still reaches `login-lyris02`, but Slurm refresh remains blocked:
  `klist -s` returns `1`, and both `squeue` and `sacct` fail because
  `/etc/slurm/slurm.conf` is unreadable with `Permission denied`.

Manual verification addendum at `2026-06-15 17:07 PDT`:

- Active OCI-HSG gates are still pre-allocation and have produced no stdout or
  driver logs. Math static r25 `3324570`, Math online r25 `3324571`, and
  SWE-RL r25 `3324801` are all `PENDING|Priority`; fallback r26 `3325343`
  remains `PENDING|Dependency` on `afternotok:3324801`.
- Scheduler estimates are available again as of the `17:06 PDT` poll:
  `3324570` at `2026-06-15T21:30:00`, `3324571` at
  `2026-06-15T22:20:00`, and `3324801` at `2026-06-15T22:30:00`. r26 remains
  dependency-blocked and has no estimate.
- `scontrol` still reports `RunTime=00:00:00`, `StartTime=Unknown`, and
  `TimeLimit=02:00:00` for all four active gates. Math r25 log roots still
  have no files; SWE-RL r25/r26 roots still only contain job-id marker files.
- Lyris remains inaccessible for Slurm refresh from this session. SSH reaches
  `login-lyris02`, but `klist -s` fails, the active Kerberos ticket expired at
  `16:08:40 PDT`, and direct login-node access requires the documented MFA /
  publickey path.

Manual verification addendum at `2026-06-15 17:04 PDT`:

- Active OCI-HSG gates are still pre-allocation. Math static r25 `3324570`,
  Math online r25 `3324571`, and SWE-RL r25 `3324801` are all
  `PENDING|Priority`; fallback r26 `3325343` is still
  `PENDING|Dependency` on `afternotok:3324801`.
- `scontrol` reports `RunTime=00:00:00`, `StartTime=Unknown`, and
  `TimeLimit=02:00:00` for all four active gates. No `slurm-*.out`,
  `ray-driver.log`, or terminal accounting record exists yet.
- `squeue --start` now reports `N/A` for the three r25 jobs, so the earlier
  `22:20` / `22:30 PDT` estimates are no longer current.
- MathRL and SWE-RL launcher contract validators pass, Python validators
  compile, shell syntax checks pass, and the HTML/doc sanity checks pass.
- Lyris SSH still reaches `login-lyris02`, but Slurm refresh remains blocked:
  `klist -s` fails, the active Kerberos ticket expired at `16:08:40 PDT`,
  `kinit -R` reports `Ticket expired while renewing credentials`, and direct
  `login-lyris02.lyris.clusters.nvidia.com` access requires the MFA/publickey
  path.

Manual verification addendum at `2026-06-15 16:59 PDT`:

- Active OCI-HSG gates are still pre-allocation and have no stdout, driver log,
  or terminal accounting record.
- Scheduler estimates are available again: Math static r25 `3324570` is
  estimated at `2026-06-15T22:20:00`; Math online r25 `3324571` and SWE-RL
  r25 `3324801` are estimated at `2026-06-15T22:30:00`; fallback r26 `3325343`
  remains dependency-blocked.
- `scontrol` shows scheduled node lists for all three r25 jobs, but
  `JobState=PENDING` and `Reason=Priority` are unchanged.
- The sanitized poll helper still reports `NO_DRIVER_LOG_YET` and
  `NO_LOG_DIR_YET` for all active gates.
- MathRL and SWE-RL launcher contract validators remain `PASS`.
- Lyris Slurm remains blocked by expired Kerberos tickets.

Manual verification addendum at `2026-06-15 16:57 PDT`:

- Active OCI-HSG gates are still waiting in queue. All three r25 jobs are
  `PENDING|Priority`; fallback r26 is still `PENDING|Dependency`.
- Scheduler estimates are currently unavailable again: `squeue --start`
  reports `N/A` and `scontrol` reports `StartTime=Unknown` for Math static r25
  `3324570`, Math online r25 `3324571`, and SWE-RL r25 `3324801`.
- No active gate has stdout, driver log, or job log directory yet. The SWE-RL
  r25/r26 log roots still only contain their job-id marker files; Math r25 log
  roots still have no job files.
- There are no new terminal accounting records since the `16:53 PDT` poll.
- Lyris remains unchanged: SSH reaches the login node, but expired Kerberos
  tickets still make `/etc/slurm/slurm.conf` unreadable for Slurm commands.

Manual verification addendum at `2026-06-15 16:53 PDT`:

- Active OCI-HSG gates remain pending and still have no stdout, driver log, or
  terminal accounting record. Current scheduler estimates: Math static r25
  `3324570` moved earlier to `2026-06-15T20:50:00`; SWE-RL r25 `3324801`
  remains estimated at `2026-06-15T22:20:00`; Math online r25 `3324571` moved
  to `2026-06-15T22:30:00`; fallback r26 `3325343` remains dependency-blocked.
- Log roots remain pre-allocation only: Math r25 roots have no job files; SWE-RL
  r25/r26 roots only have the job-id marker files.
- Extended `scripts/validate_swerl_fullgrpo_launcher_contract.py` so it now
  checks the r24 Ray/Python export fix and r23 NemoGym path fix, including
  `SBATCH_EXPORT_RAY_ENV`, `append_sbatch_ray_export`, Ray/Python/UV export
  list, `NRL_NEMO_GYM_SOURCE_ROOT`, actor venv suffix pass-through, and
  command-level env propagation into the Ray driver. The regenerated
  `docs/swerl_fullgrpo_launcher_contract_validation_20260615.md` reports
  `Overall: PASS`.
- Math and SWE-RL launcher validators both pass; Python validators compile with
  `py_compile`, and shell syntax checks pass for the Math/SWE submit wrappers
  and r25 poll helper.
- Lyris remains Kerberos-blocked for Slurm refresh.

Manual verification addendum at `2026-06-15 16:50 PDT`:

- Active OCI-HSG gates remain pending with the same scheduler estimates as the
  prior poll: Math static r25 `3324570` at `2026-06-15T22:12:30`; Math online
  r25 `3324571` and SWE-RL r25 `3324801` at `2026-06-15T22:20:00`; fallback
  r26 `3325343` remains `PENDING|Dependency`.
- The sanitized poll helper confirms no `ray-driver.log`, `slurm-*.out`, or
  job log directory exists yet for any active r25/r26 gate.
- Added and ran
  `scripts/validate_qwen235b_mathrl_pard2_cp1_contract.py`. It validates that
  the MathRL launcher keeps both `static_pard2_k3` and `online_pard2_k3` on
  `sequence_packing_enabled=false` plus `context_parallel_size=1`, while only
  `online_pard2_k3` enables the online draft/refit controls. This directly
  guards against recurrence of the r22/r23 CP/sequence-packing failure class.
- `bash -n` also passes for the MathRL submit wrapper, the SWE-RL submit
  wrapper, and the r25 poll helper.
- Lyris is still Kerberos-blocked for Slurm refresh.

Manual verification addendum at `2026-06-15 16:49 PDT`:

- OCI-HSG active r25/r26 gates are still pending and have not produced stdout
  or driver logs. There are no new terminal accounting records since the last
  poll.
- Scheduler estimates are available again: Math static r25 `3324570` is
  estimated at `2026-06-15T22:12:30`; Math online r25 `3324571` and SWE-RL
  r25 `3324801` are estimated at `2026-06-15T22:20:00`.
- `scontrol` shows reserved scheduled node lists for all three r25 jobs, but
  `JobState=PENDING` and `Reason=Priority` remain unchanged.
- Log roots confirm no allocation has started: Math r25 roots have no files;
  SWE-RL r25/r26 roots only have their `latest_235b_scale_gen_job_id.txt`
  marker files and no `ray-driver.log` or `*-logs` directory.
- SWE-RL fallback r26 `3325343` remains dependency-blocked on
  `afternotok:3324801`.
- Lyris is unchanged: SSH reaches the login node, but the Kerberos tickets
  expired at `16:08:40 PDT`, so Slurm commands still fail on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 16:47 PDT`:

- OCI-HSG active r25/r26 gates are still pending and have not started.
  `squeue --start` is back to `N/A` and `scontrol` reports
  `StartTime=Unknown` for Math static r25 `3324570`, Math online r25
  `3324571`, and SWE-RL r25 `3324801`.
- The pending reason remains `Priority`, not failure. Current `sprio` priority
  components are roughly `Priority=131477` for the Math r25 jobs and
  `Priority=131474` for SWE-RL r25, with `FairShare=81467`, `Partition=10000`,
  and `QOS=40000`.
- No active gate has a `slurm-*.out` or `ray-driver.log`. Math r25 log roots
  are still empty; SWE-RL r25/r26 roots still contain only
  `latest_235b_scale_gen_job_id.txt`.
- SWE-RL fallback r26 `3325343` remains `PENDING|Dependency` with
  `Dependency=afternotok:3324801(unfulfilled)` and `EligibleTime=Unknown`.
- Lyris state is unchanged: Kerberos tickets expired at `16:08:40 PDT`, so
  `squeue`/`sacct` still fail on `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 16:45 PDT`:

- OCI-HSG active r25/r26 gates remain pending with no allocation and no new
  terminal accounting records since `16:30 PDT`.
- Latest scheduler estimate moved again: Math static r25 `3324570`, Math
  online r25 `3324571`, and SWE-RL r25 `3324801` are all now estimated at
  `2026-06-15T21:30:00`, with `TimeLimit=02:00:00`.
- No active gate has produced `slurm-*.out` or `ray-driver.log`. The Math r25
  log roots are still empty; the SWE-RL r25/r26 log roots contain only the
  `latest_235b_scale_gen_job_id.txt` marker files.
- SWE-RL fallback r26 `3325343` remains `PENDING|Dependency` on
  `afternotok:3324801` and has no allocation.
- Lyris remains blocked by expired Kerberos tickets. SSH reaches the login
  node, but `squeue`/`sacct` still fail because `/etc/slurm/slurm.conf` is
  unreadable after ticket expiry at `16:08:40 PDT`.

Manual verification addendum at `2026-06-15 16:42 PDT`:

- OCI-HSG active r25/r26 gates are still pending, not failed. No active gate
  has a `slurm-*.out` or `ray-driver.log` yet, so there is still no new 235B
  Step 1 evidence from these replacements.
- Latest OCI-HSG scheduler estimates improved after the walltime reduction:
  Math static r25 `3324570` and Math online r25 `3324571` are both estimated
  at `2026-06-15T18:20:00`; SWE-RL r25 `3324801` is estimated at
  `2026-06-15T19:50:00`.
- SWE-RL fallback r26 `3325343` remains `PENDING|Dependency` on
  `afternotok:3324801`; it has no allocation and no logs.
- OCI-HSG `sacct` from `2026-06-15T15:30:00` onward shows no new terminal
  state for these active gates.
- Lyris is still blocked by expired Kerberos tickets. SSH reaches the login
  node, but the tickets expired at `16:08:40 PDT`; `squeue` and `sacct` still
  fail because `/etc/slurm/slurm.conf` is unreadable.

Manual verification addendum at `2026-06-15 16:36 PDT`:

- No runtime change: all active r25/r26 gates are still pending and no
  `slurm-*.out` or `ray-driver.log` exists yet.
- Latest OCI-HSG scheduler estimate remains `2026-06-15T22:10:00` for SWE-RL
  r25 `3324801`, Math static r25 `3324570`, and Math online r25 `3324571`,
  all with `TimeLimit=02:00:00`.
- Fallback r26 `3325343` remains dependency-blocked on
  `afternotok:3324801`.
- Lyris remains blocked by expired Kerberos tickets; `squeue` still fails on
  `/etc/slurm/slurm.conf: Permission denied`.

Manual verification addendum at `2026-06-15 16:35 PDT`:

- Active r25/r26 gates are still pending and have no stdout or driver logs.
- Latest OCI-HSG poll after the `02:00:00` walltime change shows all three
  r25 gates estimated at `2026-06-15T22:10:00`: SWE-RL r25 `3324801`,
  Math static r25 `3324570`, and Math online r25 `3324571`.
- Fallback r26 `3325343` remains `PENDING|Dependency` on
  `afternotok:3324801`, also with `TimeLimit=02:00:00`.
- Lyris is unchanged: SSH reaches `login-lyris02`, but Kerberos tickets are
  still expired at `16:08:40 PDT`, so `/home/sna/.bashrc` and
  `/etc/slurm/slurm.conf` remain unreadable and `squeue` cannot run.

Manual verification addendum at `2026-06-15 16:31 PDT`:

- Active r25/r26 gates still have no stdout or driver logs. They remain
  pending, not failed.
- Scheduler estimates briefly went back to `N/A`. I reduced the active proof
  gate walltimes from `04:00:00` to `02:00:00` to improve backfill
  eligibility. This should still cover the known Math 10-step runtime
  (previous completed runs were roughly 30-34 minutes) and the SWE-RL 1-step
  proof gate.
- After the walltime update and follow-up poll, SLURM reported SWE-RL r25
  `3324801` at `2026-06-15T19:30:00`, while Math static r25 `3324570` and
  Math online r25 `3324571` remain estimated at `2026-06-15T22:20:00`. r26
  `3325343` remains `Dependency=afternotok:3324801(unfulfilled)`, now with
  `TimeLimit=02:00:00`.

Manual verification addendum at `2026-06-15 16:26 PDT`:

- OCI-HSG active 235B gates are still queued. Latest scheduler estimates are
  SWE-RL r25 `3324801` at `2026-06-15T17:10:00`, Math static r25 `3324570`
  at `2026-06-15T22:20:00`, and Math online r25 `3324571` at
  `2026-06-15T22:20:00`.
- No r25/r26 stdout or driver logs exist yet, so there is still no new 235B
  Step 1 evidence from these replacements.
- `scontrol` confirms r26 `3325343` has explicit `--export=ALL,RAY_VERSION=2.54.0,
  RAY_PYTHON_VERSION=3.13.13,RAY_PYTHON_SPEC=3.13.13,RAY_USE_EXISTING_ENV=false,
  UV_PYTHON=3.13.13,UV_PYTHON_DOWNLOADS=auto` and remains blocked on
  `afternotok:3324801`.
- r25 `3324801` has no explicit `--export` in `SubmitLine`, so it depends on
  submit-time environment export for Ray bootstrap values. I left it queued
  because it is scheduled soon and r26 already provides the explicit-export
  fallback if r25 repeats the Ray mismatch.
- Hardened the shared SWE-RL submit wrapper for future retries: when Ray/UV
  bootstrap variables are set and `SBATCH_EXTRA_ARGS` does not already contain
  `--export`, it now appends `--export=ALL,...` automatically. A dry-run with
  Ray `2.54.0` / Python `3.13.13` confirmed the generated OCI-HSG sbatch line
  includes the explicit export list.

Manual verification addendum at `2026-06-15 16:25 PDT`:

- Lyris Slurm refresh blocker is now identified: the active Lyris SSH
  ControlMaster is still connected, but its Kerberos tickets expired at
  `2026-06-15 16:08:40 PDT`. Both `/home` and `/etc/slurm` are NFSv4
  `sec=krb5` mounts, so the expired ticket makes `/home/sna/.bashrc` and
  `/etc/slurm/slurm.conf` unreadable. `kinit -R` failed with `Ticket expired
  while renewing credentials`, so Lyris queue/log refresh needs a fresh MFA /
  Kerberos login before `squeue` can work again.

Manual verification addendum at `2026-06-15 16:21 PDT`:

- OCI-HSG active 235B gates are still queued, not failed. Latest scheduler
  estimates moved to Math static r25 `3324570` at `2026-06-15T20:20:00`,
  Math online r25 `3324571` at `2026-06-15T20:40:00`, and SWE-RL r25
  `3324801` at `2026-06-15T20:40:00`, all under `nemotron_n3_post`.
- SWE-RL fallback r26 `3325343` remains `PENDING|Dependency` on
  `afternotok:3324801`; it still has no stdout/driver log and will only run if
  r25 exits non-zero.
- No r25/r26 `ray-driver.log` or `slurm-*.out` exists yet. Current evidence
  therefore still does not prove Step 1 for the 235B PARD-2 replacements.
- Lyris SSH still reaches `login-lyris02.lyris.clusters.nvidia.com`, but Slurm
  remains unavailable through the login node because the expired Kerberos/NFS
  credential makes `/etc/slurm/slurm.conf` unreadable. Queue state cannot be
  refreshed there until a fresh MFA/Kerberos login is established.
- Regenerated the Qwen3-8B official PARD-2 online impact artifacts from local
  logs and current status. Online PARD-2 job `3288183` completed 10/10 with
  `9` parsed refit steps; acceptance improved from static `1.836%` to
  online `2.553%`, but online generation-worker TPS was `0.9696x` of static
  and `0.5887x` of baseline. Use this as online drafter mechanics evidence,
  not as a throughput-win claim.

Manual verification addendum at `2026-06-15 16:16 PDT`:

- OCI-HSG r25/r26 state is unchanged operationally: all proof gates are still
  pending and no stdout/driver logs exist yet.
- Latest scheduler estimates: Math static r25 `3324570` at
  `2026-06-15T20:30:00`, Math online r25 `3324571` at
  `2026-06-15T22:10:00`, and SWE-RL r25 `3324801` at
  `2026-06-15T22:10:00`.
- SWE-RL fallback r26 `3325343` remains `PENDING|Dependency` on
  `afternotok:3324801`; it is not eligible unless r25 fails.
- Lyris SSH reaches `login-lyris02.lyris.clusters.nvidia.com`, but Slurm
  commands currently fail because `/etc/slurm/slurm.conf` cannot be read by the
  Slurm client (`Permission denied`, with `/etc/slurm` also reporting stale
  file handle on directory listing). This looks like a login-node Slurm config
  mount/permission issue, not a submitted job failure.

Manual verification addendum at `2026-06-15 16:14 PDT`:

- Active 235B gates remain queued. Latest OCI-HSG estimates are Math static
  r25 `3324570` at `2026-06-15T20:20:00`, Math online r25 `3324571` at
  `2026-06-15T22:10:00`, and SWE-RL r25 `3324801` at
  `2026-06-15T22:10:00`.
- No r25 stdout or driver logs exist yet.
- Submitted a SWE-RL PARD-2 fallback, r26 `3325343`, with
  `--dependency=afternotok:3324801`. It is `PENDING|Dependency`, so it will not
  consume allocation unless r25 exits non-zero.
- r26 has explicit sbatch export for the Ray bootstrap values:
  `RAY_VERSION=2.54.0`, `RAY_PYTHON_VERSION=3.13.13`,
  `RAY_PYTHON_SPEC=3.13.13`, `RAY_USE_EXISTING_ENV=false`,
  `UV_PYTHON=3.13.13`, and `UV_PYTHON_DOWNLOADS=auto`. W&B is disabled on this
  fallback via final Hydra override `logger.wandb_enabled=False`.
- `scripts/poll_nemorl_r25_pard2_gates_20260615.sh` now tracks r26 as well as
  the three r25 gates.

Manual verification addendum at `2026-06-15 16:08 PDT`:

- r25 jobs are still queued, not failed. Latest estimates moved to Math static
  r25 `3324570` at `2026-06-15T19:40:00`, Math online r25 `3324571` at
  `2026-06-15T19:40:00`, and SWE-RL r25 `3324801` at
  `2026-06-15T20:30:00`.
- `scontrol` now shows reserved scheduled node lists for all three pending
  jobs, but no allocation has started. No `slurm-*.out` or `ray-driver.log`
  exists yet for the three r25 jobs.
- Fixed `scripts/poll_nemorl_r25_pard2_gates_20260615.sh` to use the actual
  Math r25 log roots under `mathrl_latest_main_logs` and to print parent log
  directory marker files even before `*-logs/ray-driver.log` exists.
- SLURM does not expose the queued job environment in `scontrol -dd`, so the
  effective Ray/Python values for SWE-RL r25 must be confirmed from the first
  `slurm-3324801.out` Ray venv path once the job starts.

Manual verification addendum at `2026-06-15 16:05 PDT`:

- Active OCI-HSG r25 gates are still queued, not failed. Current scheduler
  estimates are now populated: Math static r25 `3324570` starts
  `2026-06-15T19:00:00`, SWE-RL r25 `3324801` starts
  `2026-06-15T19:40:00`, and Math online r25 `3324571` starts
  `2026-06-15T20:10:00`, all under `nemotron_n3_post`.
- No r25 driver logs exist yet. The SWE-RL r25 log root still contains only
  `latest_235b_scale_gen_job_id.txt`, so there is still no Step 1 or setup
  evidence from these replacements.
- The remote SWE repo contains the NemoGym actor env fix
  `NRL_NEMO_GYM_CREATE_ENV_PYTHONPATH_V1`: `create_env()` now injects
  `NRL_NEMO_GYM_SOURCE_ROOT` into the Ray actor runtime env before creating the
  NemoGym actor.
- The saved r25 batch script still has `ray.sub` defaults of Ray `2.49.2` /
  Python `3.12.13`, but the submit wrapper launches the patched launcher with
  Ray/Python variables in the environment and relies on SLURM's default env
  export. The decisive check will be the first r25 `slurm-3324801.out` Ray venv
  path when the job starts.
- Smaller online-drafter proof is already valid: OCI-HSG Qwen3-8B
  `3288183` completed `online_pard2` 10/10, and its driver log shows
  `Draft Refit This Step: True` plus `[draft-refit] exported PARD draft
  weights` through Step 10/10.

Manual verification addendum at `2026-06-15 15:56 PDT`:

- No allocation yet. SWE-RL r25 (`3324801`), Math static r25 (`3324570`), and
  Math online r25 (`3324571`) remain `PENDING|Priority`, with
  `squeue --start` showing `N/A` and `scontrol` showing `StartTime=Unknown`.
- Driver logs are still absent for all three. The SWE-RL r25 log root contains
  only `latest_235b_scale_gen_job_id.txt`; the `3324801-logs/ray-driver.log`
  directory has not been created yet.

Manual verification addendum at `2026-06-15 15:54 PDT`:

- Active OCI-HSG r25 jobs are still queued and have not produced driver logs:
  SWE-RL r25 (`3324801`), Math static r25 (`3324570`), and Math online r25
  (`3324571`) are all `PENDING|Priority` under `nemotron_n3_post`.
- The scheduler estimate moved back to unavailable: `squeue --start` reports
  `N/A`, and `scontrol` reports `StartTime=Unknown` for all three jobs.
- Added and validated `scripts/poll_nemorl_r25_pard2_gates_20260615.sh`, a
  sanitized r25-specific poll helper that checks queue/accounting state, first
  driver log presence, and Step 1/setup/error markers for the three active
  gates.

Manual verification addendum at `2026-06-15 15:50 PDT`:

- Lyris optimized standalone jobs completed successfully:
  `2131234` mxfp8 bf16-KV finished `COMPLETED|0:0` after `01:43:11`, and
  `2131235` mxfp8 fp8-KV finished `COMPLETED|0:0` after `01:42:09`.
- Valid bs=32 results are now available. bf16-KV: `320` requests,
  `3,200,000` output tokens, `2417.10` output tok/s, `2658.81` total tok/s,
  `0.2417` req/s, mean TTFT `2263.99 ms`. fp8-KV: `320` requests,
  `3,200,000` output tokens, `2427.55` output tok/s, `2670.31` total tok/s,
  `0.2428` req/s, mean TTFT `2283.87 ms`.
- OCI-HSG active r25 jobs are still queued: SWE-RL r25 (`3324801`) is
  `PENDING|Priority` with current estimate `2026-06-15T19:50:00`; Math static
  r25 (`3324570`) and Math online r25 (`3324571`) are `PENDING|Priority` with
  current estimate `2026-06-15T20:00:00`.
- I persisted the tested NemoGym actor `PYTHONPATH` fix into the local patch
  bundle at
  `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/environments/utils.py`.
  The SWE launcher contract validator now matches the current override-aware
  cache lines and passes. Default PARD/PARD-2 source bundle validation also
  passes.

Manual verification addendum at `2026-06-15 15:38 PDT`:

- The currently active OCI-HSG gates have not ended: SWE-RL r25 (`3324801`),
  Math static r25 (`3324570`), and Math online r25 (`3324571`) are all still
  `PENDING|Priority` under `nemotron_n3_post`. The latest `scontrol` read shows
  `StartTime=Unknown` for all three, so there is still no driver log or Step 1
  evidence from the replacements.
- If the question is about the "quickly ended" SWE-RL r24 (`3324728`), the
  cause remains Ray bootstrap mismatch: Ray head `2.49.2` / Python `3.12.13`
  versus driver Ray `2.54.0` / Python `3.13.13`. r25 was submitted with
  explicit Ray/Python settings to avoid that mismatch.
- If the question is about the Lyris bf16 repair job (`2131558`), it did not
  produce a usable result. The server became healthy and the bs=1 benchmark
  started, but generation stalled during the first request and rank 7 hit an
  NCCL `_ALLGATHER_BASE` watchdog timeout after 600 seconds in the logits
  all-gather path. No `raw_bench_serve_bs1.json` was written. SLURM records
  this job as `CANCELLED by 2001147693` after `00:15:44`.
- Lyris bf16 denominator job `2131233` is also cancelled and invalid through
  bs=16. The optimized mxfp8 bf16-KV (`2131234`) and fp8-KV (`2131235`) jobs
  are still `RUNNING` and remain valid through bs=16; no bs=32 result is
  available yet.

Manual verification addendum at `2026-06-15 15:28 PDT`:

- SWE-RL r24 (`3324728`) failed quickly after `00:03:55`, before model/vLLM
  setup. Root cause: the driver was Ray `2.54.0` / Python `3.13.13`, but it
  connected to a Ray head started with Ray `2.49.2` / Python `3.12.13`, causing
  Ray version mismatch during `init_ray()`.
- I submitted SWE-RL r25 (`3324801`) with explicit Ray head settings:
  `RAY_VERSION=2.54.0`, `RAY_PYTHON_VERSION=3.13.13`,
  `RAY_PYTHON_SPEC=3.13.13`, and `RAY_USE_EXISTING_ENV=false`. It is currently
  `PENDING|Priority` under `nemotron_n3_post`.
- The NemoGym import fix is now stronger than r24: the remote repo's
  `nemo_rl/environments/utils.py` adds the Gym source root to NemoGym actor
  `PYTHONPATH`, and the local SWE submit wrapper injects
  `${REPO_ROOT}/3rdparty/Gym-workspace/Gym` into launcher `PYTHONPATH` before
  submitting.
- Follow-up at `2026-06-15 15:29 PDT`: SWE-RL r25 (`3324801`), Math static
  r25 (`3324570`), and Math online r25 (`3324571`) are all
  `PENDING|Priority`; latest SLURM estimate is `N/A` for all three.
- Lyris standalone update: bf16 denominator is invalid through bs=16. The
  mxfp8 optimized bf16-KV and fp8-KV runs are valid through bs=16. At bs=16,
  bf16-KV reports `1661.07` output tok/s and fp8-KV reports `1663.80` output
  tok/s.
- Follow-up at `2026-06-15 15:31 PDT`: r25 generated launcher has the
  `export COMMAND="${runtime_env_prefix# } ${COMMAND}"` path intact, so Ray
  version envs are prepended to the driver command. SWE-RL r25 (`3324801`)
  remains `PENDING|Priority`, current estimate `2026-06-15T20:50:00` PDT.
  Math static/online r25 (`3324570`, `3324571`) remain `PENDING|Priority`,
  current estimate `2026-06-15T20:00:00` PDT.

Manual verification addendum at `2026-06-15 15:18 PDT`:

- SWE-RL r23 (`3324381`) has now ended `FAILED|1:0` after `00:34:16`.
  This was not a PARD-2/vLLM startup failure: it reached `SETUP COMPLETE`,
  initialized async vLLM workers, loaded the PARD-2 drafter, and completed CUDA
  graph capture before failing during SWE environment actor creation.
- Root cause for `3324381`: `NemoGym.__init__()` could not import the mounted
  Gym package: `ModuleNotFoundError: No module named 'nemo_gym'`. The failing
  path is `nemo_rl/environments/nemo_gym.py`, line 54, before any Step 1
  GRPO marker.
- I patched the SWE submit wrapper to propagate `NRL_ACTOR_VENV_CACHE_SUFFIX`
  into the Ray driver command, then submitted r24 (`3324728`) with a fresh
  actor venv namespace: `NEMO_RL_VENV_DIR=/opt/ray_venvs_swerl_ray254_r24`,
  `NRL_ACTOR_VENV_CACHE_SUFFIX=swe_nemo_gym_source_r24`, and
  `NRL_NEMO_GYM_SKIP_PACKAGE_BUILD=true`. It is currently `PENDING|Priority`
  under `nemotron_n3_post`; no driver log exists yet.
- Follow-up at `2026-06-15 15:20 PDT`: SWE-RL r24 (`3324728`), Math static
  r25 (`3324570`), and Math online r25 (`3324571`) are all still
  `PENDING|Priority` under `nemotron_n3_post`; current SLURM estimate for all
  three is `2026-06-15T22:20:00` PDT.
- Lyris standalone update: mxfp8 optimized bf16-KV and fp8-KV rows are valid
  through bs=8. The bf16 denominator remains invalid; the initial and repair
  runs fail with NCCL/Ray worker death or zero-output-token wrapper failures.

Manual verification addendum at `2026-06-15 15:09 PDT`:

- SWE-RL r23 (`3324381`) is still `RUNNING` at `00:32:22`, and the driver log
  is active again. It now shows PARD-2 drafter load, CUDA graph profiling and
  capture, FlashInfer autotuning, and at least one async vLLM worker reaching
  `Uvicorn running on http://0.0.0.0:50259`.
- There is still no `SETUP COMPLETE`, no Step 1 marker, and no traceback. This
  is a forward-progress signal in vLLM/PARD-2 setup, not a confirmed hang.
- Math CP=1 replacements remain queued: static r25 (`3324570`) is
  `PENDING|Priority` with estimate `2026-06-15T20:00:00`; online r25
  (`3324571`) is `PENDING|Priority` with estimate `2026-06-15T21:40:00`.

Manual verification addendum at `2026-06-15 15:05 PDT`:

- OCI-HSG Math CP=1 replacement jobs are still queued: static r25 (`3324570`)
  and online r25 (`3324571`) are both `PENDING|Priority` with no driver logs.
- SWE-RL r23 (`3324381`) is still `RUNNING` at `00:27:03` on
  `nvl72138-T[01-16]`. The driver log still has no traceback, no
  `SETUP COMPLETE`, and no Step 1 marker. Node-level overlap diagnostics show
  MegatronPolicyWorker processes alive, about 54 GiB GPU memory allocated per
  GPU on the sampled node, and 0% GPU util while CPU-side workers continue
  model/checkpoint setup or compile work.
- Lyris standalone bf16 repair job (`2131352`) completed with SLURM
  `COMPLETED|0:0`, but the benchmark result is invalid. The wrapper recorded
  `benchmark bs=1 produced zero output tokens; expected 100000`, and the log
  shows an NCCL watchdog timeout on `_ALLGATHER_BASE` followed by SIGABRT; all
  retries then failed with connection refused to the local vLLM server port.
- Lyris standalone jobs `2131233`, `2131234`, and `2131235` are still
  `RUNNING` at about 59 minutes elapsed. The valid mxfp8 optimized rows remain
  usable; bf16 denominator rows remain unusable.

Manual verification addendum at `2026-06-15 15:01 PDT`:

- Math online PARD-2 r23 (`3324365`) failed after `00:18:30`. It was not a
  launch-only failure: it reached a 32-node Ray allocation, 128/128 vLLM
  generation workers, PARD-2 drafter load, GPU KV-cache allocation, and
  128/128 policy worker initialization.
- Root cause for `3324365`: the launcher disabled sequence packing for online
  PARD-2, but still passed `policy.megatron_cfg.context_parallel_size=2`.
  MCore asserts that context parallelism requires sequence packing:
  `AssertionError: Sequence Packing must be enabled to use Context Parallelism
  with MCore.`
- I patched the MathRL launcher so static/online PARD-2 sequence-packing-off
  jobs use `context_parallel_size=1`. The old static r24 job (`3324460`) was
  cancelled before allocation because it had the same bad `sequence_packing=false`
  plus CP=2 combination.
- CP=1 replacements are queued under `nemotron_n3_post`: static r25
  (`3324570`) and online r25 (`3324571`) are both `PENDING|Priority`, with
  SLURM estimate `2026-06-15T22:10:00`.
- SWE-RL r23 (`3324381`) is still `RUNNING` at `00:24:13`. It has passed the
  W&B failure point, loaded datasets, initialized 32/32 policy workers, and is
  in policy checkpoint/model setup. It has not reached `SETUP COMPLETE` or
  Step 1 yet, and there is no traceback in the driver log.

Manual verification addendum at `2026-06-15 14:37 PDT`:

- Replacement jobs just started allocation: Math online r23 (`3324365`) and
  SWE-RL r23 (`3324381`) are `RUNNING|Prolog`. Driver logs do not exist yet,
  so there is no new runtime success/failure signal from these replacements.
- Math static r24 (`3324460`) remains `PENDING|Priority`, with SLURM showing
  estimated start `2026-06-15T15:39:10`.
- The secret scan for the W&B key fragments across docs, latest CSVs, scripts,
  and patch files returned clean.

Manual verification addendum at `2026-06-15 14:35 PDT`:

- The just-ended OCI-HSG Math static PARD-2 r22 (`3324200`) did not exit at
  launch. It ran for `00:25:11`, reached `SETUP COMPLETE`, `Step 1/10`,
  generation, advantage computation, and logprob computation, then failed in
  Step 1 policy training.
- Root cause for `3324200`: sequence-packing metadata reported an actual
  sequence length of `600` while the worker's flattened `input_ids` tensor had
  width `408`. The failing line was
  `nemo_rl/algorithms/loss/utils.py::_pack_input_ids`, with
  `RuntimeError: The expanded size of the tensor (600) must match the existing
  size (408)`.
- I changed the MathRL launcher so `static_pard2_k3` also disables sequence
  packing, matching the earlier online PARD-2 fix. Replacement static r24
  (`3324460`) is submitted and currently `PENDING|Priority` under
  `nemotron_n3_post`.
- Current OCI-HSG queued gates: Math online r23 (`3324365`), SWE-RL r23
  (`3324381`), and Math static r24 (`3324460`) are all `PENDING|Priority`.
- SWE-RL r22 (`3324276`) remains the true quick-exit case: it exited in
  `wandb.init` with `No API key configured`. Replacement r23 (`3324381`) was
  submitted with the key passed through environment only.

Manual verification addendum at `2026-06-15 14:29 PDT`:

- Lyris access is now working through `login-lyris`; the ControlMaster is
  active and batch SSH succeeds.
- Current OCI-HSG PARD-2 gates: Math static r22 (`3324200`) is still
  `RUNNING`; it has reached 128/128 vLLM workers, PARD-2 drafter load, and
  128/128 policy workers. Math online r23 (`3324365`) and SWE-RL r23
  (`3324381`) are still `PENDING` on `Priority`.
- Current Lyris standalone jobs: `2131234` and `2131235` are still `RUNNING`
  and producing requests. `2131235` already emitted a valid bs=1 JSON result:
  `output_token_throughput=119.45 tok/s`, `request_throughput=0.0119 req/s`,
  `mean_ttft=3203 ms`, `p99_ttft=27645 ms`.
- Lyris standalone bf16 denominator job `2131233` is not usable as-is. Its
  first bs=1 run hit an NCCL timeout, the vLLM engine died, and retries
  produced zero output tokens. Repair job `2131352` is now `RUNNING` for the
  bf16 bs=1 denominator.

Manual verification addendum at `2026-06-15 14:24 PDT`:

- SWE-RL PARD-2 r22 (`3324276`) did not fail in PARD-2/vLLM. It reached the
  driver, loaded datasets, then exited during W&B initialization with
  `UsageError: No API key configured`.
- I added a submit-time guard so SWE-RL submitted jobs fail before SLURM
  allocation if the W&B key is empty, then resubmitted r23 as `3324381` with
  the key passed through environment only. At the 14:23 poll, `3324381` was
  `PENDING` on `Priority`.
- MathRL online PARD-2 r21 (`3323893`) got through vLLM worker init and PARD-2
  drafter load, then failed before Step 1 because NeMo-RL online drafter
  training does not support sequence packing:
  `policy.draft.enabled=true does not support sequence packing yet`.
- I changed the MathRL submit script so `online_pard2_k3` automatically sets
  `policy.sequence_packing.enabled=false`, then resubmitted r23 as `3324365`.
  At the 14:23 poll, `3324365` was `PENDING` on `Priority`.
- Static MathRL PARD-2 r22 (`3324200`) is still `RUNNING`. It has progressed
  past 128/128 vLLM worker init, PARD-2 drafter load, and 128/128 policy worker
  init; it has not hit the old `_get_raw_spec_counters` failure yet.

Manual verification addendum at `2026-06-15 14:04 PDT`:

- The earlier r20 MathRL PARD-2 failures (`3322940`, `3322941`) were not the
  stale-venv issue anymore. Ray/vLLM workers reached model load, then failed
  because the configured PARD-2 draft snapshot did not contain
  `warp_model.bin`.
- I changed the MathRL launcher default PARD-2 snapshot to the local snapshot
  that does contain `warp_model.bin` and added a preflight
  `test -s "${PARD2_DRAFT_MODEL}/warp_model.bin"`.
- r21 static MathRL PARD-2 (`3323891`) proved that fix: it reached
  `SETUP COMPLETE`, `Step 1/10`, and `Generating responses for batch of size
  256`, with PARD-2 drafter load successful. It then failed on a new NeMo-RL
  API mismatch: `AttributeError: 'ActorHandle' object has no attribute
  '_get_raw_spec_counters'`.
- I patched `BaseVllmGenerationWorker` with `_get_raw_spec_counters()` so the
  generation-level spec counter snapshot API matches the worker API. The patch
  is in the local patch bundle and was deployed to both remote MathRL and
  SWE-RL checkouts; both remote files pass `python -m py_compile`.
- Replacement static MathRL PARD-2 job `3324200` is submitted as r22 and is
  `PENDING`. Online MathRL PARD-2 r21 job `3323893` is still `PENDING` and
  should pick up the patched worker file when it starts. SWE-RL r20 job
  `3322947` is `RUNNING`; it passed the long TransformerEngine build stage and
  is creating async vLLM worker environments.

Manual verification addendum at `2026-06-15 13:20 PDT`:

- Runtime status is still blocked on scheduling: `3322940`, `3322941`, and
  `3322947` are `PENDING` on `Priority`, with no driver logs or job-specific
  log directories yet. Latest estimates from the 13:19 poll are
  `2026-06-15T20:30:00` for `3322940` and `2026-06-15T21:20:00` for
  `3322941`/`3322947`; priority is `132484`.
- NeMo-RL patch application is verified separately from runtime scheduling:
  the local patch bundle, the remote MathRL checkout, and the remote SWE-RL
  checkout all contain `NRL_ACTOR_RUNTIME_ENV_OVERRIDE_V1`,
  `NRL_ACTOR_PY_EXEC_V1`, `_actor_local_venv_name`, and
  `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1`.
- Local and remote patched NeMo-RL files pass `python3 -m py_compile`.
- Remote MathRL and SWE-RL patched NeMo-RL files have stale r12/r14 path count
  `0`.
- Both active PARD-2 vLLM source overlays still carry
  `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1`, have stale active r12/r14 path count
  `0`, and pass `python3 -m py_compile`.

Manual poll addendum at `2026-06-15 13:17 PDT`:

- No allocation yet. `3322940`, `3322941`, and `3322947` remain `PENDING` on
  `Priority`, with `RunTime=00:00:00`, `Priority=131209`, and no driver logs or
  job-specific log directories yet.
- SLURM start estimates changed back to `N/A` for all three jobs.

Manual poll addendum at `2026-06-15 13:15 PDT`:

- The r20 jobs are still queued. `3322940`, `3322941`, and `3322947` remain
  `PENDING` on `Priority`, with `RunTime=00:00:00`, `Priority=131209`, and no
  driver logs or job-specific log directories yet.
- Latest estimates: `3322940` MathRL static PARD-2 K3 remains
  `2026-06-15T16:50:00`; `3322941` MathRL online PARD-2 K3 and `3322947`
  SWE-RL PARD-2 step-1 both show `2026-06-15T18:00:00`.

Manual poll addendum at `2026-06-15 13:12 PDT`:

- Added a reusable local poll helper:
  `scripts/poll_nemorl_r20_pard2_gates_20260615.sh`. It runs the same
  sanitized OCI-HSG `squeue`/`sacct`/driver-log check for r20 jobs and supports
  `--watch`.
- The helper verified the r20 jobs are still queued, with no driver logs and no
  job-specific log directories yet.
- Latest scheduler view: `3322940` MathRL static PARD-2 K3 is `PENDING` on
  `Priority` with estimated start `2026-06-15T19:40:00`; `3322941` MathRL
  online PARD-2 K3 and `3322947` SWE-RL PARD-2 step-1 are both `PENDING` on
  `Priority` with estimated start `2026-06-15T20:10:00`.
- Priority is now `131209` for all three jobs.

Manual poll addendum at `2026-06-15 13:10 PDT`:

- Still no allocation or logs. `3322940`, `3322941`, and `3322947` remain
  `PENDING` on `Priority`, with `RunTime=00:00:00`, `Priority=130949`, and no
  job-specific log directories yet.
- Start estimates are visible again: `3322940` MathRL static PARD-2 K3 at
  `2026-06-15T16:50:00`, `3322947` SWE-RL PARD-2 step-1 at
  `2026-06-15T17:02:43`, and `3322941` MathRL online PARD-2 K3 at
  `2026-06-15T18:00:00`.
- Rechecked both active PARD-2 source overlays. The
  `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1` marker is still present, active
  stale r12/r14 path count remains `0`, and both files still pass
  `python3 -m py_compile`.

Manual poll addendum at `2026-06-15 13:08 PDT`:

- No allocation yet. `3322940`, `3322941`, and `3322947` are still
  `PENDING` on `Priority` under `nemotron_n3_post`, with `RunTime=00:00:00`
  and `Priority=130949`.
- SLURM start estimates changed back to `N/A` for all three jobs. No
  `ray-driver.log` exists yet, and the job-specific log directories are not
  created yet.
- Lyris direct refresh remains unavailable from this noninteractive session:
  `login-lyris` has no active ControlMaster and batch-mode SSH is rejected by
  keyboard-interactive MFA.

Manual poll addendum at `2026-06-15 13:06 PDT`:

- OCI-HSG r20 PARD-2 jobs are still queued and have no driver logs yet.
  `3322940`, `3322941`, and `3322947` are all `PENDING` on `Priority` under
  `nemotron_n3_post`, now with `Priority=130944`.
- Latest start estimates split again: `3322947` SWE-RL PARD-2 step-1 moved
  earlier to `2026-06-15T14:50:00`; `3322940` MathRL static PARD-2 K3 is
  estimated `2026-06-15T20:20:00`; `3322941` MathRL online PARD-2 K3 is
  estimated `2026-06-15T20:30:00`.
- I checked `sprio`: the visible components are `AGE=3`, `FAIRSHARE=80941`,
  `JOBSIZE=0`, `PARTITION=10000`, and `QOS=40000` for all three jobs.
- I did not reduce the `04:00:00` walltime. The completed MathRL baseline/PARD
  runs took about 30-34 minutes, but r20 is the first live PARD-2 run after the
  nested-runtime-env fix and may spend substantial time rebuilding actor/vLLM
  environments. Preserving enough runtime is safer than risking a timeout before
  the proof reaches Step 1/10 or Step 1/1.

Manual patch addendum at `2026-06-15 13:04 PDT`:

- The r20 jobs are still queued, but I removed one remaining static risk before
  they start. Both active OCI-HSG PARD-2 vLLM source overlays still contained
  stale concrete nested Ray worker runtime envs:
  `vllm_pard2_official_target_feat` had an r12 path and
  `vllm_pard2_official_target_feat_pyoverlay_nostable_nofp4out_basefa_nofp4fusion_20260614`
  had an r14 path in `vllm/v1/executor/ray_executor.py`.
- I patched both active overlay `ray_executor.py` files in place to use
  `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1`, deriving nested worker
  `py_executable` from the current actor process `sys.executable`.
- Both overlay files now have `stale_active_count=0` for r12/r14 paths and pass
  `python3 -m py_compile`. Backups were left next to the patched files.
- No job resubmission is needed: the queued r20 MathRL and SWE-RL jobs refer to
  these same shared overlay paths and will see the patched active files when
  they start.

Manual poll addendum at `2026-06-15 13:00 PDT`:

- OCI-HSG r20 PARD-2 proof gates are still not running. `3322940`,
  `3322941`, and `3322947` are all `PENDING` on `Priority`, with
  `RunTime=00:00:00`, `Priority=130943`, and account `nemotron_n3_post`.
- The estimated start moved again and now shows `2026-06-15T20:10:00` for all
  three jobs. Earlier estimates around `13:22:47` were not stable.
- No r20 driver logs exist yet for MathRL static PARD-2, MathRL online PARD-2,
  or SWE-RL PARD-2, so there is still no Step 1 runtime evidence and no live
  validation of `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1`.
- I did not submit an additional smaller smoke job. The queued MathRL 235B
  shape is the one already proven for baseline/PARD, and changing node count
  would likely test a different configuration rather than accelerate the
  PARD-2 proof. SWE-RL already has the 16-node 1-step proof queued.

Manual poll addendum at `2026-06-15 12:50 PDT`:

- No state change from the 12:48 poll. OCI-HSG r20 PARD-2 proof gates
  `3322940`, `3322941`, and `3322947` are all still `PENDING` on `Priority`
  under `nemotron_n3_post`, with `RunTime=00:00:00`.
- Estimated starts remain `2026-06-15T13:22:47` for the two 32-node MathRL
  jobs and `2026-06-15T13:49:31` for the 16-node SWE-RL job.
- No r20 driver logs exist yet, so there is still no Step 1 runtime evidence
  and no live validation of `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1`.
- Lyris was also checked through both likely aliases. `login-lyris` still has
  no active ControlMaster and batch-mode SSH is rejected by MFA; `lyris` is not
  resolvable in the current local SSH configuration.

Manual poll addendum at `2026-06-15 12:48 PDT`:

- OCI-HSG r20 PARD-2 proof gates are still queued, not running:
  `3322940` MathRL static PARD-2 K3, `3322941` MathRL online PARD-2 K3, and
  `3322947` SWE-RL PARD-2 step-1 are all `PENDING` on `Priority` under
  `nemotron_n3_post`.
- SLURM shows estimated start `2026-06-15T13:22:47` for the two 32-node MathRL
  jobs and `2026-06-15T13:49:31` for the 16-node SWE-RL job.
- No r20 `ray-driver.log` files exist yet, so there is still no Step 1 runtime
  evidence and no new failure evidence for the dynamic nested vLLM runtime-env
  patch.
- Lyris refresh remains blocked in this noninteractive session:
  `ssh -O check login-lyris` has no ControlMaster socket and batch-mode SSH is
  rejected by keyboard-interactive MFA.

Manual poll addendum at `2026-06-15 12:45 PDT`:

- r20 jobs are still queued. `3322940` and `3322941` remain `PENDING` on
  `Priority` with estimated start `2026-06-15T13:22:47`.
- SWE-RL r20 job `3322947` remains `PENDING` on `Priority`, and its estimated
  start moved later to `2026-06-15T13:49:31`.
- No driver logs exist yet for any r20 job.

Manual poll addendum at `2026-06-15 12:43 PDT`:

- No state change from the 12:42 poll. `3322940`, `3322941`, and `3322947`
  remain `PENDING` on `Priority`; estimated start remains
  `2026-06-15T13:22:47`; no driver logs exist yet.

Manual poll addendum at `2026-06-15 12:42 PDT`:

- No state change from the 12:40 poll. `3322940`, `3322941`, and `3322947`
  remain `PENDING` on `Priority`; estimated start remains
  `2026-06-15T13:22:47`; no driver logs exist yet.

Manual poll addendum at `2026-06-15 12:40 PDT`:

- No state change from the 12:39 poll. `3322940`, `3322941`, and `3322947`
  remain `PENDING` on `Priority`; estimated start remains
  `2026-06-15T13:22:47`; no driver logs exist yet.

Manual poll addendum at `2026-06-15 12:39 PDT`:

- r20 replacements are still queued, not started:
  `3322940` MathRL static PARD-2 K3, `3322941` MathRL online PARD-2 K3, and
  `3322947` SWE-RL PARD-2 step-1 are all `PENDING` on `Priority` under
  `nemotron_n3_post`.
- SLURM still shows estimated start `2026-06-15T13:22:47` for all three.
- No driver logs exist yet for any r20 job, so the dynamic nested vLLM runtime
  env marker has not been observed in a live run yet.

Manual poll addendum at `2026-06-15 12:36 PDT`:

- Fixed the nested vLLM Ray runtime-env failure. The prior actor hotfix made
  the outer `VllmAsyncGenerationWorker` use the fresh r18 executable, but the
  shared vLLM Python overlay still had a hard-coded
  `/opt/ray_venvs_swerl_ray254_r14/.../VllmAsyncGenerationWorker/bin/python`
  inside `vllm/v1/executor/ray_executor.py`. The local patch bundle and both
  remote OCI-HSG checkouts now patch vLLM with
  `NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV_V1`, which computes the nested Ray worker
  runtime env from the current actor process `sys.executable` instead of
  writing a concrete venv path into the shared overlay.
- Cancelled the superseded/stuck jobs:
  `3322475` SWE-RL PARD-2 r18 was cancelled at `2026-06-15T12:34:41` after
  reaching outer worker init but repeatedly launching nested vLLM Ray workers
  from stale r14; pending MathRL r19 jobs `3322611` and `3322621` were also
  cancelled before they started.
- Submitted fresh r20 gates under `nemotron_n3_post`:
  `3322940` MathRL static PARD-2 K3, `3322941` MathRL online PARD-2 K3, and
  `3322947` SWE-RL PARD-2 step-1. At `2026-06-15 12:36 PDT` all three are
  `PENDING` on `Priority`; at the 12:37 poll SLURM showed estimated start
  `2026-06-15T13:22:47` for all three.

Manual poll addendum at `2026-06-15 12:27 PDT`:

- MathRL baseline/PARD remains healthy: `3321180`, `3321423`, and `3321424`
  are still the completed 235B MathRL 10-step examples.
- MathRL PARD-2 r19 hotfix replacements are still queued, not failing:
  `3322611` static PARD-2 K3 and `3322621` online PARD-2 K3 are both
  `PENDING` on `Priority`, with scheduled start shown as
  `2026-06-15T13:22:47`. No driver logs exist yet, so there is no Step 1
  evidence or failure evidence for these r19 jobs.
- SWE-RL PARD-2 r18 (`3322475`) is still `RUNNING`. At the 12:25 poll it had
  elapsed `00:42:12`; at 12:27 the driver log was still being written and had
  grown to `124335` bytes. It has moved past TransformerEngine build and is
  actively creating/installing Ray actor environments under
  `/opt/ray_venvs_swerl_ray254_r18`, including the async vLLM worker path.
  Several actor builders had prepared or installed roughly 290-311 packages.
  There is still no `SETUP COMPLETE` or `Step 1/1` marker.
- Lyris direct refresh is still blocked from this noninteractive session:
  `ssh -O check login-lyris` found no ControlMaster and batch-mode SSH is
  rejected with keyboard-interactive MFA.

Manual poll addendum at `2026-06-15 12:22 PDT`:

- MathRL baseline/PARD status is unchanged and good: `3321180`, `3321423`,
  and `3321424` remain `COMPLETED/0:0` with `Step 10/10` evidence.
- MathRL PARD-2 r19 hotfix replacements have not started yet:
  `3322611` static PARD-2 K3 is `PENDING` on `Priority` with estimated start
  `2026-06-15T13:22:47`; `3322621` online PARD-2 K3 is `PENDING` on
  `Priority` with estimated start `2026-06-15T13:22:47`. No driver logs exist
  yet for either job.
- SWE-RL PARD-2 r18 (`3322475`) is still `RUNNING` at elapsed `00:39:04`.
  The driver log advanced: TransformerEngine built successfully, and Ray is
  now creating actor virtual environments under the intended
  `/opt/ray_venvs_swerl_ray254_r18/...VllmAsyncGenerationWorker` path rather
  than the stale r14/r12 paths. Build/cache files for vLLM and flash-attn were
  still being written through `2026-06-15 12:22:41 PDT`. There is still no
  `SETUP COMPLETE` or `Step 1/1` marker.
- Direct Lyris refresh was attempted via `login-lyris`, but the host requires
  keyboard-interactive MFA and the current noninteractive session has no active
  ControlMaster. Existing Lyris states below remain from the prior server-side
  evidence and were not refreshed in this poll.

Manual poll addendum at `2026-06-15 11:59 PDT`:

- MathRL baseline/PARD remains the working 235B 10-step path:
  `3321180`, `3321423`, and `3321424` are `COMPLETED/0:0` and reached
  `Step 10/10`.
- MathRL PARD-2 r18 (`3322390`, `3322392`) was cancelled after reproducing the
  stale async worker executable path
  `/opt/ray_venvs_swerl_ray254_r12/.../VllmAsyncGenerationWorker/bin/python`.
- The actor-runtime hotfix is applied in both remote checkouts and in the local
  patch bundle. It forces per-call actor `runtime_env.py_executable` over stale
  class defaults and logs `NRL_ACTOR_RUNTIME_ENV_OVERRIDE_V1` /
  `NRL_ACTOR_PY_EXEC_V1` markers.
- MathRL PARD-2 r19 replacements are pending:
  `3322611` static PARD-2 K3 and `3322621` online PARD-2 K3, both queued under
  `nemotron_n3_post` with estimated start `2026-06-15T13:22:47`.
- SWE-RL PARD-2 r18 (`3322475`) is `RUNNING`; Ray head is up and the driver log
  exists, but it is still building TransformerEngine and has not reached
  `Step 1/1`.

Manual poll addendum at `2026-06-15 11:07 PDT`:

- `3321180` is now a completed OCI-HSG MathRL reduced-shape temp1/fuse-loss-off
  baseline retry. It uses `train_global_batch_size=64`, `num_prompts=4`,
  `num_generations=16`, `generation_batch_size=16`, `max_num_seqs=16`, and
  `gpu_memory_utilization=0.80` to avoid the Step 2 vLLM OOM seen in `3321070`.
  It completed with `sacct` state `COMPLETED`, exit `0:0`, elapsed `00:30:20`.
  The driver reached `Step 1/10` through `Step 10/10`, then logged `Max number
  of steps has been reached, stopping training early`.
- `3321070` is now `FAILED`; it reached `SETUP COMPLETE`, finished Step 1
  training, entered `Step 2/10`, then failed in Step 2 generation with vLLM
  `CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:139`.
- `3320856` is now `FAILED`; it reached `Step 1/10` and `Training policy`, then
  failed on Megatron DDP grad checking with a local grad-norm `NaN`.
- `3321423` and `3321424` are now completed OCI-HSG MathRL reduced-shape
  static PARD K3/K5 follow-ups using the same working shape as `3321180`.
  Both passed `SETUP COMPLETE`, reached `Step 10/10`, trained policy on Step 10,
  logged the max-step guard, and completed with exit `0:0`.
- `3321785` and `3321786` were the first OCI-HSG MathRL PARD-2 comparison
  gates. They later failed before `SETUP COMPLETE`/Step 1 because Ray launched
  async vLLM workers from the stale
  `/opt/ray_venvs_swerl_ray254_r12/.../VllmAsyncGenerationWorker/bin/python`
  path, and are superseded by hotfix r19 jobs `3322611`/`3322621`.
- `3308774` was cancelled after repeated async vLLM worker launch failures from
  the stale/missing `/opt/ray_venvs_swerl_ray254_r14/.../bin/python` actor venv
  path and no Step 1 marker. It is superseded by fresh-venv r18 job `3322475`.
- `2129715` is now `FAILED` with exit `2:0`. It used the existing container
  Python/Ray environment (`/opt/nemo_rl_venv`, Python `3.12.12`, Ray `2.49.2`)
  instead of trying to create a Python `3.13.13` Ray venv. That path brought up
  Ray, but the driver then failed because the repo requires Python `3.13.13`.
- `2129746` is now `FAILED` with exit `1:0`. It found downloadable CPython
  `3.13.9` and created the Ray venv path, but package extraction failed with an
  I/O error, leaving no `bin/ray` executable for the head node.
- `2129624` is now `FAILED`; Lyris failed before driver startup with repeated
  missing `/tmp/nemo_rl_ray_2129624_3.13.13_2.54.0/bin/ray`, so it did not
  validate the patched TE build path.
- `2129593` is now `FAILED`; Lyris reached `64/64` workers and then failed
  while building `transformer-engine` because `setuptools.build_meta` was
  missing in the build environment.
- `2129556` is superseded and failed on user inode/quota during the TE/CUTLASS
  checkout. `2129203` is superseded by the actor-venv `.pth` patch after the
  earlier `nemo_gym` import failure.

Summary: CANCELLED=2, COMPLETED=3, FAILED=10, PENDING=2, RUNNING=1; runtime_log_present=13

Manual poll addendum at `2026-06-15 14:11 PDT`:

- SWE-RL PARD-2 r20 (`3322947`) did not terminate because of SLURM or Ray
  scheduling. It reached async vLLM engine startup and then failed while loading
  the PARD-2 drafter. The configured local draft snapshot was the older snapshot
  without `warp_model.bin`, so vLLM fell back through `hf_hub_download` with the
  absolute snapshot path and raised `HFValidationError`. The job was cancelled
  after the fatal EngineCore init error.
- The SWE submit script now gives OCI-HSG its own PARD-2 draft snapshot default
  and preflights absolute PARD-2 draft paths for `warp_model.bin` and
  `config.json`. Lyris keeps the Hub repo default.
- SWE-RL replacement `3324256` was submitted with the corrected PARD-2 snapshot,
  `NEMO_RL_VENV_DIR=/opt/ray_venvs_swerl_ray254_r21`, fresh actor venvs, and
  `nemotron_n3_post`. It was `RUNNING` at elapsed `00:00:56`; no driver log had
  appeared yet.
- MathRL online PARD-2 r21 (`3323893`) and static PARD-2 r22 (`3324200`) are
  both `RUNNING`. Both created driver logs, initialized `128/128` vLLM policy
  workers, and were loading Qwen3 safetensors checkpoint shards. No
  `HFValidationError`, `_get_raw_spec_counters`, OOM, or traceback had appeared
  at the latest poll.
- Follow-up at `2026-06-15 14:12 PDT`: both MathRL PARD-2 jobs reached PARD-2
  drafter load and logged `Using PARD-2 target layers from draft config:
  (94, 87, 79, 71)`. Static r22 logged temporary HuggingFace API rate-limit
  errors while listing `Qwen/Qwen3-235B-A22B`, but vLLM fell back to individual
  weight patterns and continued checkpoint loading.
- Follow-up at `2026-06-15 14:15 PDT`: SWE-RL replacement `3324256` failed in
  `00:03:02` before real driver startup. Its `ray-driver.log` only contained
  `error: invalid value for UV_PYTHON_DOWNLOADS`, caused by the submit script
  writing an empty value into the launcher. I changed the SWE submit script
  default to `UV_PYTHON_DOWNLOADS=auto` and submitted r22 `3324276`, which was
  `RUNNING` in prolog under `nemotron_n3_post`.
- Follow-up at `2026-06-15 14:17 PDT`: SWE-RL r22 `3324276` was still
  `RUNNING` at elapsed `00:02:04`, so it passed the immediate
  `UV_PYTHON_DOWNLOADS` failure window. No driver log had appeared yet.

| Job | Cluster | State | Reason | Start | Nodes | Account | Priority | Non-marker logs | Latest log |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| 3321180 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | COMPLETED | sacct_exit=0:0 | 2026-06-15T09:57:37 | 32 | nemotron_n3_post | 135000 | 1 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_reduced64_baseline/baseline/3321180-logs/ray-driver.log` |
| 3308774 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | CANCELLED | stale async actor venv path | 2026-06-15T08:46:16 | 16 | nemotron_n3_post | 135103 | 19 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_fullgrpo_logs/20260614_oci_hsg_swerl_qwen235b_fullgrpo_pard2_pyoverlay_eager_nolevel_r16/pard2_steps1/3308774-logs/ray-driver.log` |
| 3321423 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | COMPLETED | sacct_exit=0:0 | 2026-06-15T10:33:05 | 32 | nemotron_n3_post | 135000 | 1 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_reduced64_pard_k3k5/pard_k3/3321423-logs/ray-driver.log` |
| 3321424 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | COMPLETED | sacct_exit=0:0 | 2026-06-15T10:33:05 | 32 | nemotron_n3_post | 135000 | 1 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_reduced64_pard_k3k5/pard_k5/3321424-logs/ray-driver.log` |
| 3321785 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | FAILED | stale async actor venv path | 2026-06-15T12:11:51 | 32 | nemotron_n3_post | 134998 | 0 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_reduced64_pard2_static_online/static_pard2_k3/3321785-logs/ray-driver.log` |
| 3321786 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | FAILED | stale async actor venv path | 2026-06-15T12:07:56 | 32 | nemotron_n3_post | 134998 | 0 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_reduced64_pard2_static_online/online_pard2_k3/3321786-logs/ray-driver.log` |
| 2129715 | Lyris SWE-RL Raymatch/PARD Proof Gates | FAILED | sacct_exit=2:0 | 2026-06-15T10:16:49 | 16 | coreai_dlalgo_llm |  | 1 | `/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_ray_existingenv_pth_r35/baseline_steps1/2129715-logs/ray-driver.log` |
| 2129746 | Lyris SWE-RL Raymatch/PARD Proof Gates | FAILED | sacct_exit=1:0 | 2026-06-15T10:25:00 | 16 | coreai_dlalgo_llm |  | 1 | `/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_py3139_r36/baseline_steps1/2129746-logs/ray-head.log` |
| 2129624 | Lyris SWE-RL Raymatch/PARD Proof Gates | FAILED | sacct_exit=1:0 | 2026-06-15T09:55:04 |  | coreai_dlalgo_llm |  | 1 | `/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_raymatch_tebuild_pth_r34/baseline_steps1/slurm-2129624.out` |
| 2129556 | Lyris SWE-RL Raymatch/PARD Proof Gates | FAILED | sacct_exit=1:0 | 2026-06-15T09:29:19 |  | coreai_dlalgo_llm |  | 1 | `/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_raymatch_tmpcache_pth_r31/baseline_steps1/slurm-2129556.out` |
| 2129203 | Lyris SWE-RL Raymatch/PARD Proof Gates | FAILED | sacct_exit=1:0 | 2026-06-15T07:57:46 |  | coreai_dlalgo_llm |  | 19 | `/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_raymatch_tmpcache_r30/baseline_steps1/slurm-2129203.out` |
| 2129271 | Lyris SWE-RL Raymatch/PARD Proof Gates | CANCELLED | sacct_exit=0:0 | None |  | coreai_dlalgo_llm |  | 0 | `/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_raymatch_tmpcache_r30_afterok/pard_steps1/latest_235b_scale_gen_job_id.txt` |
| 2129272 | Lyris SWE-RL Raymatch/PARD Proof Gates | CANCELLED | sacct_exit=0:0 | None |  | coreai_dlalgo_llm |  | 0 | `/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_logs/20260615_lyris_swerl_qwen235b_fullgrpo_raymatch_tmpcache_r30_afterok/pard2_steps1/latest_235b_scale_gen_job_id.txt` |
| 3321070 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | FAILED | sacct_exit=1:0 | 2026-06-15T09:37:25 |  | nemotron_n3_post |  | 1 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_fuseloss_off_temp1_baseline/baseline/3321070-logs/ray-driver.log` |
| 3320856 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | FAILED | sacct_exit=1:0 | 2026-06-15T09:17:45 |  | nemotron_n3_post |  | 1 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_fuseloss_off_baseline/baseline/3320856-logs/ray-driver.log` |
| 3315380 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | FAILED | sacct_exit=1:0 | 2026-06-15T08:46:16 |  | nemotron_n3_post |  | 20 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_latest_main_guards_n3post_systempy/baseline/slurm-3315380.out` |
| 3315381 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | FAILED | sacct_exit=1:0 | 2026-06-15T08:47:05 |  | nemotron_n3_post |  | 20 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_latest_main_guards_n3post_systempy/pard_k3/slurm-3315381.out` |
| 3315382 | OCI-HSG SWE-RL PARD-2 And MathRL Proof Gates | FAILED | sacct_exit=1:0 | 2026-06-15T08:47:05 |  | nemotron_n3_post |  | 20 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_latest_main_guards_n3post_systempy/pard_k5/slurm-3315382.out` |
