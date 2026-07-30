# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from pathlib import Path

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

USE_SYSTEM_EXECUTABLE = os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1"
REQUIRE_SYSTEM_MCORE = os.environ.get("NEMO_RL_REQUIRE_SYSTEM_MCORE", "0") == "1"


def _require_pinned_mcore_system_interpreter() -> None:
    """Fail closed when an experiment pins MCore actors to the driver venv."""
    expected_python = os.environ.get("NEMO_RL_MCORE_SYSTEM_PYTHON")
    if not expected_python:
        raise RuntimeError(
            "NEMO_RL_REQUIRE_SYSTEM_MCORE=1 requires NEMO_RL_MCORE_SYSTEM_PYTHON"
        )

    expected_path = Path(expected_python)
    if not expected_path.is_file() or not expected_path.resolve().is_file():
        raise RuntimeError(
            "NEMO_RL_MCORE_SYSTEM_PYTHON must name an existing Python interpreter: "
            f"{expected_python}"
        )

    # Do not compare only resolved paths: uv venv Python launchers share a common
    # managed interpreter target.  The lexical venv path and sys.prefix identify
    # the locked environment actually selected for the Ray actor.
    if os.path.abspath(sys.executable) != os.path.abspath(expected_python):
        raise RuntimeError(
            "NEMO_RL_MCORE_SYSTEM_PYTHON must match sys.executable exactly; "
            f"expected {expected_python}, got {sys.executable}"
        )
    expected_prefix = expected_path.parent.parent
    if Path(sys.prefix) != expected_prefix:
        raise RuntimeError(
            "NEMO_RL_MCORE_SYSTEM_PYTHON venv root must match sys.prefix; "
            f"expected {expected_prefix}, got {sys.prefix}"
        )


VLLM_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.VLLM
)
SGLANG_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.SGLANG
)
MCORE_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM
    if USE_SYSTEM_EXECUTABLE or REQUIRE_SYSTEM_MCORE
    else PY_EXECUTABLES.MCORE
)
TRTLLM_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.TRTLLM
)

ACTOR_ENVIRONMENT_REGISTRY: dict[str, str] = {
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": VLLM_EXECUTABLE,
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": VLLM_EXECUTABLE,
    "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker": SGLANG_EXECUTABLE,
    "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": PY_EXECUTABLES.FSDP,
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": PY_EXECUTABLES.AUTOMODEL,
    "nemo_rl.models.value.workers.dtensor_value_worker_v2.DTensorValueWorkerV2": PY_EXECUTABLES.AUTOMODEL,
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": MCORE_EXECUTABLE,
    "nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker": MCORE_EXECUTABLE,
    "nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker": TRTLLM_EXECUTABLE,
    "nemo_rl.environments.math_environment.MathEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.math_environment.MathMultiRewardEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.vlm_environment.VLMEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.code_environment.CodeEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.reward_model_environment.RewardModelEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.code_jaccard_environment.CodeJaccardEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.games.sliding_puzzle.SlidingPuzzleEnv": PY_EXECUTABLES.SYSTEM,
    # AsyncTrajectoryCollector needs vLLM environment to handle exceptions from VllmGenerationWorker
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector": PY_EXECUTABLES.VLLM,
    # ReplayBuffer needs vLLM environment to handle trajectory data from VllmGenerationWorker
    "nemo_rl.algorithms.async_utils.ReplayBuffer": PY_EXECUTABLES.VLLM,
    # SyncRolloutActor doesn't import vllm directly — policy_generation is a
    # Ray actor handle. The VLLM env is needed because (1) transfer_queue is
    # bundled into the VLLM venv (and the policy training venvs), and the
    # actor writes flattened tensors to TQ via dp_client.put_samples;
    # (2) same-node colocation with VllmGenerationWorker avoids duplicate
    # venv caches.
    "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor": PY_EXECUTABLES.VLLM,
    "nemo_rl.environments.tools.retriever.RAGEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.nemo_gym.NemoGym": PY_EXECUTABLES.NEMO_GYM,
}

from nemo_rl.modelopt.registry import MODELOPT_ACTOR_REGISTRY

ACTOR_ENVIRONMENT_REGISTRY.update(MODELOPT_ACTOR_REGISTRY)


def get_actor_python_env(actor_class_fqn: str) -> str:
    if actor_class_fqn in ACTOR_ENVIRONMENT_REGISTRY:
        if REQUIRE_SYSTEM_MCORE and actor_class_fqn in {
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
            "nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker",
        }:
            _require_pinned_mcore_system_interpreter()
        return ACTOR_ENVIRONMENT_REGISTRY[actor_class_fqn]
    else:
        raise ValueError(
            f"No actor environment registered for {actor_class_fqn}. "
            f"You're attempting to create an actor ({actor_class_fqn}) "
            "without specifying a python environment for it. Please either"
            "specify a python environment in the registry "
            "(nemo_rl.distributed.ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY) "
            "or pass a py_executable to the RayWorkerBuilder. If you're unsure about which "
            "environment to use, a good default is PY_EXECUTABLES.SYSTEM for ray actors that "
            "don't have special dependencies. If you do have special dependencies (say, you're "
            "adding a new generation framework or training backend), you'll need to specify the "
            "appropriate environment. See uv.md for more details."
        )
