#!/bin/bash
set -uo pipefail
root="/raid/scratch/sna/supervl35-imports-${SLURM_JOB_ID}/source"
mkdir -p "$root"
tar --exclude=.git -cf - -C /home/sna/brr-workspaces/supervl35-optionb-20261010/source-13d905d68fbd03d45ca17eddebffc200108cef7c . | tar -xf - -C "$root"
output=/home/sna/brr-workspaces/supervl35-optionb-20261010/import-results-v2
mkdir -p "$output"
exec >"$output/check.log" 2>&1
export PYTHONPATH="$root:$root/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$root/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:$root/3rdparty/Gym-workspace/Gym:$root/examples/nemo_gym/supervl3p5"
export PYTHONPYCACHEPREFIX="${root%/source}/pycache"
export TMPDIR="${root%/source}/tmp"
mkdir -p "$TMPDIR"
export NEMO_GYM_EXTRA_ROOTS="$root/3rdparty/Gym-workspace/Gym:$root/examples/nemo_gym/supervl3p5"
for python in /opt/nemo_rl_venv/bin/python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python /opt/ray_venvs/nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor/bin/python; do
    timeout 120 "$python" /home/sna/brr-workspaces/supervl35-optionb-20261010/imports/check_imports.py "$output"
done
for service in responses_api_models/vllm_model responses_api_agents/simple_agent responses_api_agents/image_tools_agent resources_servers/gui_coordinate resources_servers/math_with_judge resources_servers/mcqa resources_servers/string_match resources_servers/sav_tracks; do
    if test -x "/opt/gym_venvs/$service/.venv/bin/python"; then echo "SERVICE_PRESENT $service"; else echo "SERVICE_MISSING $service"; fi
done
