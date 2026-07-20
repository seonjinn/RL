# Qwen3-30B-A3B Full SWE GRPO

This experiment runs the complete NeMo-RL SWE training path through
`examples/nemo_gym/run_grpo_nemo_gym.py`: rollout, reward, policy logprob,
policy optimization, and generation-worker refit. It does not use the
rollout-only benchmark entrypoint.

The smoke matrix uses the Qwen3-30B-A3B Thinking SWE2 recipe with two prompts,
two generations per prompt, and two optimizer steps. Generation occupies one
four-GPU GB200 node. The Megatron policy occupies eight four-GPU nodes with
TP4, PP2, CP4, and EP8. Checkpoint saving is disabled.

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
2.7.2 rejects `defer_loading: null` while validating the completed rollout,
causing the Gym `/run` endpoint to return HTTP 500.

After validating a clean worktree, the launcher applies exact-match source
fixes in `gym_openhands_tmux.py`. `TMUX_TMPDIR` remains `/tmp`, `TMUX` is unset
so `libtmux.Server().new_session()` owns server creation, and only a `jq` that
passes an execution probe is exposed through `/tmp/nemorl-native-tools`. The
SWE image's native `/usr/bin/jq` is preferred over the shared Miniforge copy.
Completed-rollout tool definitions are serialized with `exclude_none=True`.
The patched source SHA is stored under `gym_openhands_runtime_fix` in
`provenance.json`. Use a fresh remote worktree for each submitted run because
Gym writes runtime artifacts below its source tree.

Variants:

- `baseline`
- `dflash_k7`
- `dflash_k9`

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
