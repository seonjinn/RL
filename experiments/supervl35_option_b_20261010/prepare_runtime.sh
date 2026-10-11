#!/bin/bash
set -euo pipefail
umask 077
commit=$1
runtime_name="supervl35-runtime-${commit:0:12}"
runtime="/raid/scratch/sna/$runtime_name"
workspace=$PWD
output="$workspace/runtime-build-$SLURM_JOB_ID"
durable=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010
mkdir -p "$output" "$runtime/source" "$runtime/tmp" "$durable/runtime"
exec >"$runtime/build.log" 2>&1
trap 'status=$?; cp "$runtime/build.log" "$output/build.log"; exit "$status"' EXIT
test "$(git -C "$workspace/source-$commit" rev-parse HEAD)" = "$commit"
tar -cf - -C "$workspace/source-$commit" . | tar -xf - -C "$runtime/source"
export TMPDIR="$runtime/tmp" UV_CACHE_DIR="$runtime/uv-cache"
export PYTHONPYCACHEPREFIX="$runtime/pycache"
root="$runtime/source"
gym="$root/3rdparty/Gym-workspace/Gym"
mcore="$root/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export PYTHONPATH="$root:$root/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$mcore:$gym:$root/examples/nemo_gym/supervl3p5"
export NEMO_GYM_EXTRA_ROOTS="$gym:$root/examples/nemo_gym/supervl3p5"
policy_python=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
gym_python=/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python
vllm_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
"$policy_python" -m pybind11 --includes >/dev/null || uv pip install --python "$policy_python" pybind11==3.1.0
make -C "$mcore/megatron/core/datasets" \
    CPPFLAGS="$("$policy_python" -m pybind11 --includes)" \
    LIBEXT="$("$policy_python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
"$policy_python" -c 'import megatron.core.datasets.helpers_cpp as h; assert callable(h.build_sample_idx_int32); assert callable(h.build_sample_idx_int64)'
"$vllm_python" - <<'PY'
import importlib.util
from pathlib import Path
from nemo_rl.models.generation.vllm.patches import _radio_final_layernorm_source
path=Path(importlib.util.find_spec('vllm').origin).parent/'model_executor/models/nano_nemotron_vl.py'
patched, changed=_radio_final_layernorm_source(path.read_text())
compile(patched,str(path),'exec')
print('RADIO source patch compatible:',changed)
PY
for service in responses_api_models/vllm_model responses_api_agents/simple_agent responses_api_agents/image_tools_agent resources_servers/gui_coordinate resources_servers/math_with_judge resources_servers/mcqa resources_servers/string_match resources_servers/sav_tracks; do
    component="$gym/$service"
    test -d "$component" || component="$root/examples/nemo_gym/supervl3p5/$service"
    venv="$runtime/gym_venvs/$service/.venv"
    uv venv --python "$gym_python" "$venv"
    if test -f "$component/pyproject.toml"; then
        uv pip install --python "$venv/bin/python" -e "$component"
    else
        requirements="$runtime/tmp/$(basename "$service").txt"
        sed '/^-e /d' "$component/requirements.txt" >"$requirements"
        uv pip install --python "$venv/bin/python" -e "$gym[dev]" -r "$requirements"
    fi
    "$venv/bin/python" -c 'import nemo_gym, fastapi; print(nemo_gym.__file__)'
    uv pip freeze --python "$venv/bin/python" >"$output/$(basename "$service")-freeze.txt"
done
mkdir -p "$output/imports"
for python in /opt/nemo_rl_venv/bin/python "$policy_python" "$vllm_python" "$gym_python" /opt/ray_venvs/nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor/bin/python; do
    "$python" "$workspace/launcher/check_imports.py" "$output/imports"
done
/opt/nemo_rl_venv/bin/python - "$output/imports" <<'PY'
import json,sys
from pathlib import Path
reports=list(Path(sys.argv[1]).glob('*.json'))
assert len(reports)==5
assert all(c['ok'] for p in reports for c in json.loads(p.read_text())['checks'].values())
PY
printf '%s\n' "$commit" >"$runtime/SOURCE_COMMIT"
tar -czf "$durable/runtime/$runtime_name.tar.gz.tmp-$SLURM_JOB_ID" \
    -C /raid/scratch/sna "$runtime_name/source" "$runtime_name/gym_venvs" "$runtime_name/SOURCE_COMMIT"
test ! -e "$durable/runtime/$runtime_name.tar.gz"
mv "$durable/runtime/$runtime_name.tar.gz.tmp-$SLURM_JOB_ID" "$durable/runtime/$runtime_name.tar.gz"
sha256sum "$durable/runtime/$runtime_name.tar.gz" >"$durable/runtime/$runtime_name.tar.gz.sha256"
cp "$durable/runtime/$runtime_name.tar.gz.sha256" "$output/runtime.sha256"
printf '%s\n' RUNTIME_PREPARATION_COMPLETE
