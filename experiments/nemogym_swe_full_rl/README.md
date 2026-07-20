# Qwen3-30B-A3B Full SWE GRPO

This experiment runs the complete NeMo-RL SWE training path through
`examples/nemo_gym/run_grpo_nemo_gym.py`: rollout, reward, policy logprob,
policy optimization, and generation-worker refit. It does not use the
rollout-only benchmark entrypoint.

The smoke matrix uses the Qwen3-30B-A3B Thinking SWE2 recipe with two prompts,
two generations per prompt, and two optimizer steps. Generation occupies one
four-GPU GB200 node. The Megatron policy occupies eight four-GPU nodes with
TP4, PP2, CP4, and EP8. Checkpoint saving is disabled.

The launcher preserves the SWE2 recipe's 131072-token total sequence and
generation limits. Its async collector also honors `grpo.max_num_epochs`, so a
finite smoke dataset can be reused across configured epochs just like the
synchronous GRPO path.

`env.nemo_gym.subprocess_openai_version=2.7.2` keeps Gym-created subprocesses
on the newest version allowed by the pinned Gym submodule. The parent NeMo-RL
environment remains unchanged.

NeMo Gym server environments are stored under the shared Lustre
`gym_venvs/<Gym commit>-py312-openai2.7.2` directory. Gym launches Ray workers
with the server environment's Python path, so a node-local `/opt/gym_venvs`
directory is invalid for this multi-node run.

The pinned Gym revision has three OpenHands runtime compatibility issues on
Lyris. First, it exports an incomplete `TMUX` client value before `libtmux`
creates its server, which points the client at a dead socket. Second, a shared
OpenHands setup can contain an x86-64 Miniforge `jq`, while the SWE image is
ARM64. Sourcing `instance_swe_entry.sh` then exits the tmux shell and appears as
a 600-second OpenHands timeout because the completion marker can no longer be
printed. Third, SWE tools are serialized with optional `None` fields. OpenAI
2.7.2 rejects `defer_loading: null`, but its function-tool schema also requires
the `strict` boolean. Omitting every null therefore changes the first error
into a missing-field error. Both failures cause the Gym `/run` endpoint to
return HTTP 500 while validating a completed rollout.

After validating a clean worktree, the launcher applies exact-match source
fixes in `gym_openhands_tmux.py`. `TMUX_TMPDIR` remains `/tmp`, `TMUX` is unset
so `libtmux.Server().new_session()` owns server creation, and only a `jq` that
passes an execution probe is exposed through `/tmp/nemorl-native-tools`. The
SWE image's native `/usr/bin/jq` is preferred over the shared Miniforge copy.
Completed-rollout tool definitions omit optional null fields and normalize a
function tool's optional runtime `strict=None` value to schema-valid
`strict=false`.
The patched source SHA is stored under `gym_openhands_runtime_fix` in
`provenance.json`. Use a fresh remote worktree for each submitted run because
Gym writes runtime artifacts below its source tree.

Variants:

- `baseline`: V2 model runner baseline for Eagle-3.
- `eagle3_k3`: Thinking-2507 Eagle-3 drafter, K=3, V2 model runner.
- `baseline_v1`: V1 model runner baseline for DFlash, using
  FULL_AND_PIECEWISE graphs and request capture sizes `[1,2,4,8,16]`.
- `dflash_k7`: DFlash K=7 with its 40960-token draft limit and V1 runner.
- `dflash_k9`: DFlash K=9 exploration with the same V1 runner.

Eagle-3 and DFlash are reported in separate matched lanes. DFlash uses the V1
model runner because vLLM 0.25.1 V1 stops drafting safely after the drafter's
40960-token context limit, while the target can continue to the recipe's
131072-token limit. Comparing it with `baseline_v1` avoids attributing model
runner differences to speculative decoding.
The baseline request shapes correspond to DFlash's K7 query-token shapes
`[8,16,32,64,128]`, so both lanes cover 1, 2, 4, 8, and 16 requests.
The launcher caps `max_num_seqs` at the rollout concurrency. It does not pass
the internal-only `max_num_scheduled_tokens` field through `AsyncEngineArgs`.
Instead, `max_num_batched_tokens` preserves a derived 2048-token target budget
and reserves `(K-1) * max_num_seqs` parallel-draft slots, as required by vLLM.
CUDA Graph diagnostics register vLLM's text logger when `cudagraph_metrics` is
enabled, so runtime FULL, PIECEWISE, and NONE counts are retained in the job log.

Validate OpenHands `libtmux` startup in the same Astropy SWE image before the
multi-node run:

```bash
REPO_DIR="$PWD" \
OPENHANDS_SETUP="$PWD/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swe_openhands_setup" \
sbatch --export=ALL,REPO_DIR,OPENHANDS_SETUP \
  experiments/nemogym_swe_full_rl/verify_openhands_libtmux_lyris.sh
```

Run scheduling validation before submission:

```bash
./experiments/nemogym_swe_full_rl/submit_lyris.sh \
  --mode test-only --variant baseline
```

Submit the two-step smoke:

```bash
./experiments/nemogym_swe_full_rl/submit_lyris.sh \
  --mode submit --variant baseline
```

After a smoke reaches rollout, reward, logprob, policy training, and refit,
submit its matched 20-step run with `--steps 20`.
