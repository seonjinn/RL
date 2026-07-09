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
import gc
import re
import traceback
from collections.abc import Iterable
from typing import Any, Protocol, cast

import torch
import zmq

from nemo_rl.models.generation.vllm.config import MTP_SPECULATIVE_METHODS
from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


class _DraftModel(Protocol):
    """vLLM draft-model operations used by the refit extension."""

    def load_weights(
        self,
        *,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> object: ...

    def named_modules(self) -> Iterable[tuple[str, torch.nn.Module]]: ...


def fix_gemma3_vision_weight_name(key: str) -> str:
    """Re-insert the `vision_model` segment into Gemma3 vision-tower weights.

    When performing refit, the vision-tower weight paths are flattened. This unflattens them.
    """
    return re.sub(
        r"vision_tower\.(?!vision_model\.)", "vision_tower.vision_model.", key
    )


def _validate_draft_weight_load_result(load_result: object) -> None:
    """Fail when a vLLM loader reports that required draft weights were not loaded."""
    if load_result is None:
        raise RuntimeError(
            "vLLM drafter loader returned no load receipt; draft weight "
            "completeness cannot be verified."
        )

    if isinstance(load_result, (set, frozenset)):
        if not load_result:
            raise RuntimeError("The vLLM drafter loader reported no loaded weights.")
        return

    if isinstance(load_result, bool):
        if load_result:
            return
        raise RuntimeError("The vLLM drafter loader returned failure.")

    if not hasattr(load_result, "missing_keys") or not hasattr(
        load_result, "unexpected_keys"
    ):
        raise RuntimeError(
            "The vLLM drafter loader returned an unsupported result type: "
            f"{type(load_result).__name__}."
        )

    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys

    if missing_keys:
        raise RuntimeError(
            f"The vLLM drafter loader reported missing weights: {sorted(missing_keys)}"
        )
    if unexpected_keys:
        raise RuntimeError(
            "The vLLM drafter loader reported unexpected weights: "
            f"{sorted(unexpected_keys)}"
        )


def _read_mtp_layer_weights_from_checkpoint(
    model_path: str, mtp_layer_indices: set[int]
) -> list[tuple[str, torch.Tensor]]:
    """Read only MTP draft weights from a local HF safetensors checkpoint.

    For sharded checkpoints, only shards containing MTP weights are opened.
    Explicit ``mtp`` and ``mtp_layers`` namespaces are model-owned draft
    modules; otherwise, only the resolved transformer layer indices are read.

    Args:
        model_path: Path to the HF checkpoint directory.
        mtp_layer_indices: Transformer layer indices belonging to the MTP module(s).

    Returns:
        A list of ``(weight_name, tensor)`` pairs for the requested layers, with
        tensors on CPU.
    """
    import json
    import os

    from safetensors import safe_open

    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    mtp_namespace_re = re.compile(r"(?:^|\.)mtp_layers\.\d+\.")

    def is_mtp_weight(name: str) -> bool:
        layer_match = layer_re.search(name)
        return (
            name.startswith("mtp.")
            or mtp_namespace_re.search(name) is not None
            or (
                layer_match is not None
                and int(layer_match.group(1)) in mtp_layer_indices
            )
        )

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = (
        model_path
        if os.path.isfile(model_path)
        else os.path.join(model_path, "model.safetensors")
    )
    if not os.path.exists(index_path):
        if not os.path.exists(single_path):
            raise FileNotFoundError(
                "Expected model.safetensors.index.json or model.safetensors "
                f"under local checkpoint {model_path}."
            )
        weights: list[tuple[str, torch.Tensor]] = []
        with safe_open(single_path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if is_mtp_weight(name):
                    weights.append((name, reader.get_tensor(name)))
        return weights

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if is_mtp_weight(name):
            shard_to_names.setdefault(shard, []).append(name)

    weights: list[tuple[str, torch.Tensor]] = []
    for shard, names in shard_to_names.items():
        with safe_open(
            os.path.join(model_path, shard), framework="pt", device="cpu"
        ) as reader:
            for name in names:
                weights.append((name, reader.get_tensor(name)))
    return weights


class VllmInternalWorkerExtension:
    state_dict_info: dict[str, Any]
    require_mtp_draft_weights: bool
    _pending_draft_weights: list[tuple[str, torch.Tensor]] | None
    _observed_update_weight_names: set[str] | None
    _draft_weights_updated: bool

    def bind_numa(self) -> bool:
        """Pin this TP worker to its GPU's NUMA-local CPUs/memory.

        Invoked via ``collective_rpc`` on each vLLM TP worker once the engine
        (and CUDA) is up, so the worker's physical GPU id is resolved from its
        local device index (see ``resolve_visible_gpu_id``).
        """
        import torch

        from nemo_rl.distributed.numa_utils import (
            bind_to_gpu_numa,
            resolve_visible_gpu_id,
        )

        gpu_id = resolve_visible_gpu_id(torch.cuda.current_device())
        if gpu_id is None:
            return False
        return bind_to_gpu_numa(gpu_id)

    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group.init_nccl_communicator(device=self.device)

    def report_device_id(self) -> str:
        """Retrieve the UUID of the current CUDA device."""
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(
        self,
        state_dict_info: dict[str, Any],
        require_mtp_draft_weights: bool = False,
    ) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}
        """
        if not state_dict_info:
            raise ValueError(
                "The vLLM refit weight manifest is empty; refusing to run an "
                "update that cannot prove target-weight coverage."
            )
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
        self.require_mtp_draft_weights = require_mtp_draft_weights

    def _begin_weight_update(self) -> None:
        if not getattr(self, "state_dict_info", None):
            raise RuntimeError(
                "The vLLM refit weight manifest is empty; refusing to begin the "
                "weight update."
            )
        if getattr(self, "_pending_draft_weights", None) is not None:
            raise RuntimeError("A vLLM weight update is already in progress.")
        self._pending_draft_weights: list[tuple[str, torch.Tensor]] | None = []
        self._observed_update_weight_names: set[str] | None = set()
        self._draft_weights_updated = False

    def _abort_weight_update(self) -> None:
        self._pending_draft_weights = None
        self._observed_update_weight_names = None
        self._draft_weights_updated = False

    def _draft_update_requires_atomic_load(self) -> bool:
        vllm_config = getattr(self.model_runner, "vllm_config", None)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        return getattr(speculative_config, "method", None) in MTP_SPECULATIVE_METHODS

    def _get_draft_model(self) -> _DraftModel | None:
        """Resolve the draft model from either vLLM model-runner generation."""
        draft_owner = getattr(self.model_runner, "drafter", None)
        if draft_owner is None:
            draft_owner = getattr(self.model_runner, "speculator", None)
        if draft_owner is None:
            return None
        return cast(_DraftModel | None, getattr(draft_owner, "model", None))

    def _process_weights_after_update(self, *, draft_weights_updated: bool) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model, self.model_config, self.device
            )
            if draft_weights_updated:
                draft_model = self._get_draft_model()
                if draft_model is None:
                    raise RuntimeError(
                        "Draft weights were updated but the vLLM drafter is unavailable."
                    )
                speculative_config = self.model_runner.vllm_config.speculative_config
                process_weights_after_loading(
                    draft_model,
                    speculative_config.draft_model_config,
                    self.device,
                )

    def _finish_weight_update(self) -> None:
        pending_draft_weights = getattr(self, "_pending_draft_weights", None)
        observed_names = getattr(self, "_observed_update_weight_names", None)
        if pending_draft_weights is None or observed_names is None:
            raise RuntimeError("No vLLM weight update is in progress.")

        try:
            expected_names = set(self.state_dict_info)
            missing_names = expected_names - observed_names
            unexpected_names = observed_names - expected_names
            if missing_names:
                raise RuntimeError(
                    "The vLLM refit transport completed with missing weights: "
                    f"{sorted(missing_names)}"
                )
            if unexpected_names:
                raise RuntimeError(
                    "The vLLM refit transport received unexpected weights: "
                    f"{sorted(unexpected_names)}"
                )

            draft_weights_updated = bool(pending_draft_weights) or bool(
                self._draft_weights_updated
            )
            if (
                getattr(self, "require_mtp_draft_weights", False)
                and self._draft_update_requires_atomic_load()
            ):
                from vllm.distributed.parallel_state import get_pp_group

                if get_pp_group().is_last_rank and not draft_weights_updated:
                    raise RuntimeError(
                        "MTP refit completed without draft weights on the drafter-owner "
                        "pipeline rank. Check the trainer export names and MTP routing."
                    )
            self._load_draft_weights(pending_draft_weights)
            self._process_weights_after_update(
                draft_weights_updated=draft_weights_updated
            )
        finally:
            self._abort_weight_update()

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache (static scales)."""
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
            return

        # FP8 KV cache: process KV scales after weight loading
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model,
                self.model_runner.model_config,
                target_device,
            )

    def _get_mtp_layer_indices(self) -> set[int]:
        draft_model = self._get_draft_model()
        predictor = getattr(draft_model, "model", draft_model)
        mtp_start_layer_idx = getattr(predictor, "mtp_start_layer_idx", None)
        num_mtp_layers = getattr(predictor, "num_mtp_layers", None)
        if not isinstance(mtp_start_layer_idx, int) or not isinstance(
            num_mtp_layers, int
        ):
            vllm_config = getattr(self.model_runner, "vllm_config", None)
            speculative_config = getattr(vllm_config, "speculative_config", None)
            draft_model_config = getattr(speculative_config, "draft_model_config", None)
            draft_hf_config = getattr(draft_model_config, "hf_config", None)
            mtp_start_layer_idx = getattr(
                draft_hf_config, "num_hidden_layers", mtp_start_layer_idx
            )
            for field in (
                "n_predict",
                "num_nextn_predict_layers",
                "mtp_num_hidden_layers",
            ):
                value = getattr(draft_hf_config, field, None)
                if isinstance(value, int):
                    num_mtp_layers = value
                    break

        if (
            not isinstance(mtp_start_layer_idx, int)
            or not isinstance(num_mtp_layers, int)
            or num_mtp_layers < 1
        ):
            return set()
        return set(range(mtp_start_layer_idx, mtp_start_layer_idx + num_mtp_layers))

    def _split_policy_and_draft_weights(
        self,
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Split trainer-owned draft weights from policy weights.

        Eagle exports draft parameters under a ``draft.`` prefix. MTP weights
        retain their HF layer names, so route layers owned by the vLLM MTP
        predictor, including persistent buffers, to the drafter as well.
        """
        uses_mtp_specdec = self._draft_update_requires_atomic_load()
        mtp_layer_indices = self._get_mtp_layer_indices() if uses_mtp_specdec else set()

        policy_weights = []
        draft_weights = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
                continue

            if uses_mtp_specdec and key.startswith("mtp."):
                draft_weights.append((key, tensor))
                continue

            if uses_mtp_specdec and re.search(r"(?:^|\.)mtp_layers\.\d+\.", key):
                draft_weights.append((key, tensor))
                continue

            layer_match = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
            if (
                layer_match is not None
                and int(layer_match.group(1)) in mtp_layer_indices
            ):
                draft_weights.append((key, tensor))
                continue

            policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    @staticmethod
    def _trim_vocab_padding(
        draft_model: _DraftModel,
        draft_weights: list[tuple[str, torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Trim padded vocab dimensions from draft weights.

        Megatron pads vocab to a multiple, but vLLM 0.20's autoloader
        strictly asserts loaded_weight.shape[0] == org_vocab_size on
        VocabParallelEmbedding layers. Each such layer may have a
        different org_vocab_size (e.g. embed_tokens uses vocab_size
        while lm_head uses draft_vocab_size), so we match each weight
        to its target module by name.
        """
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        vocab_sizes: dict[str, int] = {}
        for name, module in draft_model.named_modules():
            if isinstance(module, VocabParallelEmbedding):
                vocab_sizes[name] = module.org_vocab_size

        if not vocab_sizes:
            return draft_weights

        trimmed = []
        for key, tensor in draft_weights:
            for mod_name, org_vocab_size in vocab_sizes.items():
                leaf = mod_name.rsplit(".", 1)[-1]
                if leaf in key and tensor.shape[0] > org_vocab_size:
                    tensor = tensor[:org_vocab_size]
                    break
            trimmed.append((key, tensor))
        return trimmed

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        draft_model = self._get_draft_model()

        if draft_model is None:
            raise RuntimeError(
                "Received draft weights but the vLLM drafter is unavailable."
            )
        draft_weights = self._trim_vocab_padding(draft_model, draft_weights)
        load_result = draft_model.load_weights(weights=draft_weights)
        _validate_draft_weight_load_result(load_result)

    def load_mtp_weights_from_disk(self, model_path: str) -> bool | None:
        """Load only the MTP (multi-token-prediction) draft weights from disk.

        Used when an MTP speculative-decoding policy runs with
        ``load_format="dummy"``: the main model receives real weights via refit,
        but the MTP draft layer is not covered by refit (the trainer runs with
        ``mtp_num_layers=0``), so its weights must come from the checkpoint. Only
        the MTP layer(s) are read, avoiding a full base-model load (~1.3 TB for
        DeepSeek-V3) on every inference replica.

        Args:
            model_path: Path to the HF checkpoint directory.

        Returns:
            ``True`` on drafter-owner ranks after loading, and ``None`` on
            non-owner pipeline ranks.
        """
        draft_model = self._get_draft_model()
        if draft_model is None:
            from vllm.distributed.parallel_state import get_pp_group

            if not get_pp_group().is_last_rank:
                return None
            raise RuntimeError(
                "Cannot load MTP weights because the vLLM drafter is unavailable "
                "on the last pipeline rank."
            )

        mtp_layer_indices = self._get_mtp_layer_indices()
        weights = _read_mtp_layer_weights_from_checkpoint(model_path, mtp_layer_indices)
        if not weights:
            raise ValueError(
                "No MTP draft weights "
                f"for resolved layers {sorted(mtp_layer_indices)} "
                f"found in checkpoint at {model_path}. The checkpoint must "
                "include an mtp.* namespace or the resolved MTP layer indices."
            )

        self._load_draft_weights(weights)

        # The MTP block contains MoE experts whose weights need post-load
        # processing (e.g. grouped-GEMM layout), matching the main-model path.
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        draft_model_config = (
            self.model_runner.vllm_config.speculative_config.draft_model_config
        )
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(draft_model, draft_model_config, self.device)
        print(
            f"[mtp] Loaded MTP draft weights for layers "
            f"{sorted(mtp_layer_indices)} from {model_path}"
        )
        return True

    def _load_weights(self, weights):
        """Load weights with Gemma3 vision-tower weight name fix, FP8, and draft-weight support.

        Applies Gemma3 vision-tower weight name fix if needed, splits policy/draft
        weights, applies FP8 conversion if needed, and loads draft weights
        into the drafter model.
        """
        from nemo_rl.models.generation.vllm.quantization import fp8

        source_weight_names = {key for key, _ in weights}
        if (
            "Gemma3ForConditionalGeneration"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weights[idx] = (fix_gemma3_vision_weight_name(key), weight)

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        if policy_weights:
            if fp8.is_fp8_model(self.model_runner.vllm_config):
                fp8.load_weights(policy_weights, self.model_runner)
            else:
                self.model_runner.model.load_weights(weights=policy_weights)

        if draft_weights:
            draft_model = self._get_draft_model()
            if draft_model is None:
                from vllm.distributed.parallel_state import get_pp_group

                if get_pp_group().is_last_rank:
                    raise RuntimeError(
                        "Received draft weights but the vLLM drafter is unavailable "
                        "on the last pipeline rank."
                    )
                draft_weights = []

        observed_names = getattr(self, "_observed_update_weight_names", None)
        pending_draft_weights = getattr(self, "_pending_draft_weights", None)
        if observed_names is not None and pending_draft_weights is not None:
            observed_names.update(source_weight_names)
            if draft_weights and self._draft_update_requires_atomic_load():
                pending_draft_weights.extend(
                    (
                        key,
                        tensor.detach().to(device="cpu", copy=True),
                    )
                    for key, tensor in draft_weights
                )
            elif draft_weights:
                self._load_draft_weights(draft_weights)
                self._draft_weights_updated = True
        else:
            self._load_draft_weights(draft_weights)

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weights = None
        reply_pending = False

        try:
            self.maybe_init_zmq()
            self._begin_weight_update()
            while True:
                # Blocking receive with timeout (this is the main operation)
                payload = self.zmq_socket.recv_pyobj()
                reply_pending = True

                if payload == IPCProtocol.COMPLETE:
                    self._finish_weight_update()
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    reply_pending = False
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

                weight = None
                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]  # pyrefly
                    if isinstance(shape, list):
                        shape = torch.Size(shape)

                    # Get the weight from the buffer
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weight = (
                        buffer[offset : offset + size_in_bytes]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    weights.append((key, weight))

                    # Move offset to the next weight
                    aligned_size = calculate_aligned_size(size_in_bytes)
                    offset += aligned_size

                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
                )

                # Load weights into the model
                self._load_weights(weights)

                torch.cuda.current_stream().synchronize()

                # CRITICAL: Delete views before ACK to prevent corruption.
                # 'weights' contains views into IPC shared memory. Even though load_weights()
                # copied the data, Python may not garbage collect these view objects immediately.
                # If sender reuses the buffer before GC runs, old views would read corrupted data.
                # Explicit del ensures immediate cleanup before sending ACK.
                del weight, weights, buffer
                weight = None
                weights = None
                buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                reply_pending = False

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            self._abort_weight_update()
            if reply_pending:
                try:
                    error_message = f"{IPCProtocol.ERROR.value}:{type(e).__name__}: {e}"
                    self.zmq_socket.send(error_message.encode()[:4096])
                except zmq.ZMQError:
                    pass
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n"
                f"{traceback.format_exc()}"
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        load_model_weight_func = self._load_weights

        try:
            self._begin_weight_update()
            packed_broadcast_consumer(
                iterator=iter(self.state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=load_model_weight_func,
            )

            torch.cuda.synchronize(self.device)
            self._finish_weight_update()

        except Exception as e:
            self._abort_weight_update()
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        gc.collect()
        torch.cuda.empty_cache()
        return True

    def cleanup(self) -> None:
        """Shutdown and cleanup resources."""
        # Close ZMQ socket and context if they exist
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
