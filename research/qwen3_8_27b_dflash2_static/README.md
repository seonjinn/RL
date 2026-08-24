# Qwen3.8-27B DFlash2 static rollout scaffold

This experiment holds the target, GSM8K workload, sampling, and vLLM engine
settings constant between `baseline.yaml` and `dflash2.yaml`. The DFlash2 arm
adds only the published static speculative configuration: vLLM method
`dflash`, seven speculative tokens, and the public DFlash2 checkpoint.

The YAML files are experiment manifests, not direct `vllm serve --config`
files. A launcher must translate `engine` into server arguments, `workload`
into benchmark-client arguments, and `speculative_config` into the vLLM JSON
argument. This separation prevents workload fields from being passed to the
server accidentally.

Run `preflight.py` inside the selected vLLM environment before starting either
arm. NeMo-RL's current vLLM 0.25.1 pin predates DFlash2 and is intentionally
rejected. The capable runtime must contain both the Qwen3 DFlash2 model and V2
speculator modules, and it must not force the V1 runner.

This milestone covers static rollout only. Target-only NeMo-RL refits are
allowed by the runtime boundary, but online DFlash2 training and live draft
refit remain unsupported and fail closed.
