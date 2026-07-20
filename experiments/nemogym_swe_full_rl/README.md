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

The pinned Gym revision exports an incomplete `TMUX` client value before
OpenHands lets `libtmux` create its server. That points `libtmux` at a dead
socket and stalls `instance_swe_entry.sh`. On submission, the launcher applies
the exact-match `gym_openhands_tmux.py` compatibility fix after validating a
clean worktree: `TMUX_TMPDIR` remains `/tmp`, while `TMUX` is unset so
`libtmux.Server().new_session()` owns server creation. The patched source SHA is
stored in `provenance.json`. Use a fresh remote worktree for each submitted run
because Gym writes runtime artifacts below its source tree.

Variants:

- `baseline`
- `dflash_k7`
- `dflash_k9`

Validate OpenHands `libtmux` startup in the same Astropy SWE image before the
multi-node run:

```bash
REPO_DIR="$PWD" sbatch \
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
