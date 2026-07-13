# NeMo-RL 235B Ray Actor Probe - 2026-06-15

Probe time: `2026-06-15 08:48 PDT`

Scope: Lyris `2129203`, SWE-RL 235B baseline step-1 retry.

## Current Read

- SLURM state: `RUNNING`
- Driver log fetched: yes
- Parsed GRPO step metrics: none
- vLLM generation actors: `32` `VllmAsyncGenerationWorker` actors alive
- Ray worker wrappers: `32` `RayWorkerWrapper` actors alive
- Queue actor: `1` `_QueueActor` alive
- Runtime env builders: `15` `_env_builder` tasks still running; `17` finished
- Megatron policy worker actor: not observed in Ray actor summary
- GPU utilization snapshot: `0%` across sampled nodes
- vLLM model memory: present on 8 generation nodes, about `154341 MiB` per GPU
- Training/policy nodes: GPU memory mostly empty at the probe time

## Interpretation

`2129203` has cleared Ray cluster startup and vLLM worker initialization, but it
has not reached a GRPO training step. The current bottleneck is still policy-side
runtime environment construction: the policy workers have not appeared as
Megatron policy actors, and 15 `_env_builder` tasks remain active.

This is not yet evidence that the GRPO step fails. It is evidence that the run is
still in pre-step setup after vLLM initialization.
