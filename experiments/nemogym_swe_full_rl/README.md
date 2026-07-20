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

Variants:

- `baseline`
- `dflash_k7`
- `dflash_k9`

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
