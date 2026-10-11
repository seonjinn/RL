import importlib
import json
import pathlib
import sys
import traceback

common = ['nemo_rl.telemetry.instrumentation']
checks = {
    'MegatronPolicyWorker': ['megatron.core.datasets.helpers', 'megatron.bridge.models.nemotron_vl.nemotron_vl_provider', 'transformer_engine.pytorch.ops', 'nemo_rl.models.policy.workers.megatron_policy_worker'],
    'VllmAsyncGenerationWorker': ['vllm.model_executor.models.nano_nemotron_vl', 'vllm.model_executor.models.radio', 'torchcodec', 'nemo_gym.token_id_capture.staging.capture', 'nemo_rl.models.generation.vllm.vllm_worker_async'],
    'NemoGym': ['nemo_gym.cli', 'nemo_gym.rollout_collection', 'nemo_gym.token_id_capture.lineage', 'nemo_gym.token_id_capture.staging.records'],
    'RolloutReassemblerActor': ['nemo_rl.experience.rollout_reassembler_actor'],
}
name = pathlib.Path(sys.executable).parents[1].name
report = {'python': sys.executable, 'checks': {}}
modules = common + next((v for k, v in checks.items() if name.endswith(k)), [])
for module in modules:
    try:
        obj = importlib.import_module(module)
        report['checks'][module] = {'ok': True, 'file': str(getattr(obj, '__file__', None))}
    except Exception:
        report['checks'][module] = {'ok': False, 'traceback': traceback.format_exc()}
pathlib.Path(sys.argv[1], name + '.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'python': sys.executable, 'passed': sum(x['ok'] for x in report['checks'].values()), 'total': len(modules)}), flush=True)
