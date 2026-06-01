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

import copy
import fcntl
import gc
import json
import os
import re
import sys
from importlib.util import find_spec
from typing import Any, Optional, cast

import ray
import torch
from transformers import AutoConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.utils import format_prompt_for_vllm_generation
from nemo_rl.models.huggingface.common import ModelFlag
from nemo_rl.models.policy.utils import is_vllm_v1_engine_enabled
from nemo_rl.utils.nsys import wrap_with_nvtx_name


# Use a base class to share some functions to avoid code duplication.
class BaseVllmGenerationWorker:
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        return f"{self.__class__.__name__}"

    def _maybe_register_eagle3_model_in_process(
        self, llm_kwargs: dict[str, Any]
    ) -> None:
        """Avoid vLLM registry subprocess crashes for Eagle3 draft inspection."""
        speculative_config = llm_kwargs.get("speculative_config")
        if not isinstance(speculative_config, dict):
            return
        if speculative_config.get("method") != "eagle3":
            return
        if os.environ.get("NRL_VLLM_REGISTER_EAGLE3_IN_PROCESS", "1").lower() in {
            "0",
            "false",
            "no",
        }:
            return

        speculative_model = speculative_config.get("model")
        draft_architectures = []
        if isinstance(speculative_model, str):
            config_path = os.path.join(speculative_model, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        draft_config = json.load(f)
                    draft_architectures = list(
                        draft_config.get("architectures", []) or []
                    )
                except Exception as e:
                    print(
                        "Warning: failed to inspect local Eagle3 draft config for "
                        f"registry workaround: {e}",
                        flush=True,
                    )

        llama_compatible_arches = {
            "Eagle3DraftModel",
            "FSDPEagle3DraftModel",
            "Eagle3LlamaForCausalLM",
            "LlamaForCausalLMEagle3",
            "PEagleDraftModel",
            "PeagleLlamaForCausalLM",
        }
        if draft_architectures:
            arch_names = [
                arch for arch in draft_architectures if arch in llama_compatible_arches
            ]
            if not arch_names:
                print(
                    "Warning: skipping Eagle3 registry workaround for unsupported "
                    f"draft architectures: {draft_architectures}",
                    flush=True,
                )
                return
        else:
            arch_names = ["Eagle3LlamaForCausalLM", "LlamaForCausalLMEagle3"]

        try:
            from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
            from vllm.model_executor.models.registry import ModelRegistry
        except Exception as e:
            print(
                "Warning: failed to import vLLM Eagle3 model registry workaround: "
                f"{e}",
                flush=True,
            )
            return

        for arch in arch_names:
            ModelRegistry.register_model(arch, Eagle3LlamaForCausalLM)

    @staticmethod
    def _accumulate_vllm_spec_decode_metric(
        metrics: dict[str, Any], metric: Any
    ) -> bool:
        name = getattr(metric, "name", "")
        if name == "vllm:spec_decode_num_drafts":
            metrics["num_drafts"] += int(getattr(metric, "value", 0))
            return True
        if name == "vllm:spec_decode_num_draft_tokens":
            metrics["num_draft_tokens"] += int(getattr(metric, "value", 0))
            return True
        if name == "vllm:spec_decode_num_accepted_tokens":
            metrics["num_accepted_tokens"] += int(getattr(metric, "value", 0))
            return True
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values = list(getattr(metric, "values", []) or [])
            if not values and hasattr(metric, "value"):
                values = [getattr(metric, "value", 0)]
            current = metrics["num_accepted_tokens_per_pos"]
            if len(current) < len(values):
                current.extend([0] * (len(values) - len(current)))
            for idx, value in enumerate(values):
                current[idx] += int(value)
            return True
        return False

    def _get_vllm_metrics_snapshot(self):
        llm = getattr(self, "llm", None)
        if llm is not None and hasattr(llm, "get_metrics"):
            try:
                metrics = llm.get_metrics()
                if metrics is not None:
                    return metrics
            except Exception:
                pass

        try:
            from vllm.v1.metrics.reader import get_metrics_snapshot
        except Exception:
            return []
        return get_metrics_snapshot()

    def _read_vllm_spec_decode_metrics(self) -> dict[str, Any]:
        """Read in-process vLLM SpecDec counters, if available."""
        metrics = {
            "metrics_available": False,
            "num_drafts": 0,
            "num_draft_tokens": 0,
            "num_accepted_tokens": 0,
            "num_accepted_tokens_per_pos": [],
        }

        saw_spec_decode_metric = False
        for metric in self._get_vllm_metrics_snapshot():
            saw_spec_decode_metric = (
                self._accumulate_vllm_spec_decode_metric(metrics, metric)
                or saw_spec_decode_metric
            )
        if not saw_spec_decode_metric:
            return {}
        metrics["metrics_available"] = True
        metrics["active"] = metrics["num_draft_tokens"] > 0
        return metrics

    @staticmethod
    def _diff_vllm_spec_decode_metrics(
        current: dict[str, Any], baseline: dict[str, Any]
    ) -> dict[str, Any]:
        if not current:
            return {}
        diff = {
            "metrics_available": bool(current.get("metrics_available", True)),
            "active": False,
            "num_drafts": max(
                0, int(current.get("num_drafts", 0)) - int(baseline.get("num_drafts", 0))
            ),
            "num_draft_tokens": max(
                0,
                int(current.get("num_draft_tokens", 0))
                - int(baseline.get("num_draft_tokens", 0)),
            ),
            "num_accepted_tokens": max(
                0,
                int(current.get("num_accepted_tokens", 0))
                - int(baseline.get("num_accepted_tokens", 0)),
            ),
            "num_accepted_tokens_per_pos": [],
        }
        current_pos = list(current.get("num_accepted_tokens_per_pos", []) or [])
        baseline_pos = list(baseline.get("num_accepted_tokens_per_pos", []) or [])
        max_len = max(len(current_pos), len(baseline_pos))
        for idx in range(max_len):
            cur = int(current_pos[idx]) if idx < len(current_pos) else 0
            base = int(baseline_pos[idx]) if idx < len(baseline_pos) else 0
            diff["num_accepted_tokens_per_pos"].append(max(0, cur - base))
        diff["active"] = diff["num_draft_tokens"] > 0
        return diff

    @staticmethod
    def _gate_number(value: Any) -> int | float | bool | str | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float | str):
            return value
        return None

    @staticmethod
    def _add_gate_sum(bucket: dict[str, Any], key: str, value: Any) -> None:
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int | float):
            bucket[key] = bucket.get(key, 0) + value

    @staticmethod
    def _add_gate_observed_value(bucket: dict[str, Any], key: str, value: Any) -> None:
        value = BaseVllmGenerationWorker._gate_number(value)
        if value is None:
            return
        observed_key = f"{key}_observed"
        values = bucket.setdefault(observed_key, [])
        if value not in values:
            values.append(value)
        bucket[key] = value

    @staticmethod
    def _diff_vllm_specdec_gate_metrics(
        current: dict[str, Any], baseline: dict[str, Any]
    ) -> dict[str, Any]:
        if not current:
            return {}
        diff = copy.deepcopy(current)
        for group in ("runner", "scheduler"):
            current_group = current.get(group)
            if not isinstance(current_group, dict):
                continue
            baseline_group = baseline.get(group, {}) if isinstance(baseline, dict) else {}
            diff_group = diff.setdefault(group, {})
            # NRL_SPECDEC_DYNAMIC_STORE_COUNTERS_DIFF_V1
            gate_sum_keys = (
                "checked",
                "enabled",
                "disabled",
                "dynamic_small_selected_count",
                "dynamic_medium_selected_count",
                "dynamic_large_selected_count",
                "dynamic_small_selected_token_count",
                "dynamic_medium_selected_token_count",
                "dynamic_large_selected_token_count",
            ) + tuple(
                f"dynamic_pos{pos_idx}_selected_count" for pos_idx in range(1, 9)
            )
            for key in gate_sum_keys:
                diff_group[key] = max(
                    0,
                    int(current_group.get(key, 0))
                    - int((baseline_group or {}).get(key, 0)),
                )
            checked = int(diff_group.get("checked", 0))
            if checked > 0:
                diff_group["enabled_ratio"] = (
                    float(diff_group.get("enabled", 0)) / checked
                )
        return diff

    def _iter_vllm_gate_metric_objects(self):
        """Best-effort walk over local vLLM objects that may own gate counters."""
        llm = getattr(self, "llm", None)
        queue: list[Any] = [obj for obj in (self, llm) if obj is not None]
        seen: set[int] = set()
        max_objects = 512

        while queue and len(seen) < max_objects:
            obj = queue.pop(0)
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            if isinstance(obj, dict):
                queue.extend(obj.values())
                continue
            if isinstance(obj, list | tuple | set):
                queue.extend(obj)
                continue
            try:
                attrs = vars(obj)
            except TypeError:
                continue
            if any(name.startswith("_nrl_specdec") for name in attrs):
                yield obj

            module = type(obj).__module__
            if obj is not self and not module.startswith(("vllm", "nemo_rl")):
                continue
            for value in attrs.values():
                if isinstance(value, int | float | str | bytes | bool | type(None)):
                    continue
                queue.append(value)

    def _read_vllm_specdec_gate_metrics(self) -> dict[str, Any]:
        """Read runtime SpecDec long-tail gate counters, if the patch installed them."""
        metrics: dict[str, Any] = {
            "metrics_available": False,
            "runner": {"num_reporting_objects": 0},
            "scheduler": {"num_reporting_objects": 0},
        }
        runner_sums = {
            "_nrl_specdec_batch_gate_checked_count": "checked",
            "_nrl_specdec_batch_gate_enabled_count": "enabled",
            "_nrl_specdec_batch_gate_disabled_count": "disabled",
        }
        runner_values = {
            "_nrl_specdec_batch_gate_threshold": "request_threshold",
            "_nrl_specdec_batch_gate_token_threshold": "token_threshold",
            "_nrl_specdec_batch_gate_last_num_requests": "last_num_requests",
            "_nrl_specdec_batch_gate_last_num_tokens": "last_num_tokens",
            "_nrl_specdec_batch_gate_last_disabled": "last_disabled",
            "_nrl_specdec_adaptive_gate_mode": "adaptive_mode",
            "_nrl_specdec_adaptive_request_threshold": "adaptive_request_threshold",
            "_nrl_specdec_adaptive_token_threshold": "adaptive_token_threshold",
            "_nrl_specdec_adaptive_target_enabled_ratio": "adaptive_target_enabled_ratio",
            "_nrl_specdec_adaptive_last_enabled_ratio": "adaptive_last_enabled_ratio",
            "_nrl_specdec_adaptive_window_checked": "adaptive_window_checked",
            "_nrl_specdec_adaptive_window_enabled": "adaptive_window_enabled",
        }
        scheduler_sums = {
            "_nrl_specdec_scheduler_gate_checked_count": "checked",
            "_nrl_specdec_scheduler_gate_enabled_count": "enabled",
            "_nrl_specdec_scheduler_gate_disabled_count": "disabled",
            "_nrl_specdec_scheduler_dynamic_small_selected_count": "dynamic_small_selected_count",
            "_nrl_specdec_scheduler_dynamic_medium_selected_count": "dynamic_medium_selected_count",
            "_nrl_specdec_scheduler_dynamic_large_selected_count": "dynamic_large_selected_count",
            "_nrl_specdec_scheduler_dynamic_small_selected_token_count": "dynamic_small_selected_token_count",
            "_nrl_specdec_scheduler_dynamic_medium_selected_token_count": "dynamic_medium_selected_token_count",
            "_nrl_specdec_scheduler_dynamic_large_selected_token_count": "dynamic_large_selected_token_count",
            "_nrl_specdec_scheduler_dynamic_pos1_selected_count": "dynamic_pos1_selected_count",
            "_nrl_specdec_scheduler_dynamic_pos2_selected_count": "dynamic_pos2_selected_count",
            "_nrl_specdec_scheduler_dynamic_pos3_selected_count": "dynamic_pos3_selected_count",
            "_nrl_specdec_scheduler_dynamic_pos4_selected_count": "dynamic_pos4_selected_count",
            "_nrl_specdec_scheduler_dynamic_pos5_selected_count": "dynamic_pos5_selected_count",
            "_nrl_specdec_scheduler_dynamic_pos6_selected_count": "dynamic_pos6_selected_count",
            "_nrl_specdec_scheduler_dynamic_pos7_selected_count": "dynamic_pos7_selected_count",
            "_nrl_specdec_scheduler_dynamic_pos8_selected_count": "dynamic_pos8_selected_count",
        }
        scheduler_values = {
            "_nrl_specdec_scheduler_gate_threshold": "request_threshold",
            "_nrl_specdec_scheduler_gate_token_threshold": "token_threshold",
            "_nrl_specdec_scheduler_gate_last_num_requests": "last_num_requests",
            "_nrl_specdec_scheduler_gate_last_active_requests": "last_active_requests",
            "_nrl_specdec_scheduler_gate_last_num_tokens": "last_num_tokens",
            "_nrl_specdec_scheduler_gate_last_disabled": "last_disabled",
            "_nrl_specdec_scheduler_gate_effective_lookahead_tokens": "effective_lookahead_tokens",
            "_nrl_specdec_scheduler_dynamic_draft_tokens_enabled": "dynamic_draft_tokens_enabled",
            "_nrl_specdec_scheduler_dynamic_last_selected_tokens": "dynamic_last_selected_tokens",
            "_nrl_specdec_scheduler_dynamic_last_selected_tier": "dynamic_last_selected_tier",
            "_nrl_specdec_scheduler_dynamic_last_stored_tokens": "dynamic_last_stored_tokens",
            "_nrl_specdec_scheduler_dynamic_small_request_threshold": "dynamic_small_request_threshold",
            "_nrl_specdec_scheduler_dynamic_medium_request_threshold": "dynamic_medium_request_threshold",
            "_nrl_specdec_scheduler_dynamic_small_token_threshold": "dynamic_small_token_threshold",
            "_nrl_specdec_scheduler_dynamic_medium_token_threshold": "dynamic_medium_token_threshold",
            "_nrl_specdec_scheduler_dynamic_small_tokens": "dynamic_small_tokens",
            "_nrl_specdec_scheduler_dynamic_medium_tokens": "dynamic_medium_tokens",
            "_nrl_specdec_scheduler_dynamic_large_tokens": "dynamic_large_tokens",
            "_nrl_specdec_scheduler_adaptive_mode": "adaptive_mode",
            "_nrl_specdec_scheduler_adaptive_request_threshold": "adaptive_request_threshold",
            "_nrl_specdec_scheduler_adaptive_token_threshold": "adaptive_token_threshold",
            "_nrl_specdec_scheduler_adaptive_target_enabled_ratio": "adaptive_target_enabled_ratio",
            "_nrl_specdec_scheduler_adaptive_last_enabled_ratio": "adaptive_last_enabled_ratio",
            "_nrl_specdec_scheduler_adaptive_window_checked": "adaptive_window_checked",
            "_nrl_specdec_scheduler_adaptive_window_enabled": "adaptive_window_enabled",
        }

        for obj in self._iter_vllm_gate_metric_objects():
            attrs = vars(obj)
            saw_runner = False
            for attr, key in runner_sums.items():
                if attr in attrs:
                    self._add_gate_sum(metrics["runner"], key, attrs[attr])
                    saw_runner = True
            for attr, key in runner_values.items():
                if attr in attrs:
                    self._add_gate_observed_value(metrics["runner"], key, attrs[attr])
                    saw_runner = True
            if saw_runner:
                metrics["runner"]["num_reporting_objects"] += 1

            saw_scheduler = False
            for attr, key in scheduler_sums.items():
                if attr in attrs:
                    self._add_gate_sum(metrics["scheduler"], key, attrs[attr])
                    saw_scheduler = True
            for attr, key in scheduler_values.items():
                if attr in attrs:
                    self._add_gate_observed_value(metrics["scheduler"], key, attrs[attr])
                    saw_scheduler = True
            if saw_scheduler:
                metrics["scheduler"]["num_reporting_objects"] += 1

        saw_gate = any(
            metrics[group].get("num_reporting_objects", 0) > 0
            for group in ("runner", "scheduler")
        )
        if not saw_gate:
            return {}
        metrics["metrics_available"] = True
        for group in ("runner", "scheduler"):
            bucket = metrics[group]
            checked = int(bucket.get("checked", 0))
            if checked > 0:
                bucket["enabled_ratio"] = float(bucket.get("enabled", 0)) / checked
        return metrics

    def clear_vllm_logger_metrics(self) -> None:
        self._vllm_spec_decode_metrics_baseline = (
            self._read_vllm_spec_decode_metrics()
        )
        self._vllm_specdec_gate_metrics_baseline = (
            self._read_vllm_specdec_gate_metrics()
        )

    def get_vllm_logger_metrics(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        gate_current = self._read_vllm_specdec_gate_metrics()
        if gate_current:
            gate_baseline = getattr(self, "_vllm_specdec_gate_metrics_baseline", None)
            result["spec_decode_gate"] = (
                self._diff_vllm_specdec_gate_metrics(gate_current, gate_baseline)
                if gate_baseline is not None
                else gate_current
            )
        current = self._read_vllm_spec_decode_metrics()
        if not current:
            result["spec_decode"] = {"metrics_available": False}
            return result
        baseline = getattr(self, "_vllm_spec_decode_metrics_baseline", None)
        spec_decode = (
            self._diff_vllm_spec_decode_metrics(current, baseline)
            if baseline is not None
            else current
        )
        if not spec_decode:
            result["spec_decode"] = {"metrics_available": False}
            return result
        result["spec_decode"] = spec_decode
        return result

    def set_specdec_runtime_controls(self, controls: dict[str, Any]) -> dict[str, Any]:
        """Best-effort runtime update for adaptive SpecDec gate/depth controls."""
        attr_map = {
            "scheduler_dynamic_small_tokens": "_nrl_specdec_scheduler_dynamic_small_tokens",
            "scheduler_dynamic_medium_tokens": "_nrl_specdec_scheduler_dynamic_medium_tokens",
            "scheduler_dynamic_large_tokens": "_nrl_specdec_scheduler_dynamic_large_tokens",
            "scheduler_dynamic_small_request_threshold": "_nrl_specdec_scheduler_dynamic_small_request_threshold",
            "scheduler_dynamic_medium_request_threshold": "_nrl_specdec_scheduler_dynamic_medium_request_threshold",
            "scheduler_dynamic_small_token_threshold": "_nrl_specdec_scheduler_dynamic_small_token_threshold",
            "scheduler_dynamic_medium_token_threshold": "_nrl_specdec_scheduler_dynamic_medium_token_threshold",
            "scheduler_adaptive_request_threshold": "_nrl_specdec_scheduler_adaptive_request_threshold",
            "scheduler_adaptive_token_threshold": "_nrl_specdec_scheduler_adaptive_token_threshold",
            "scheduler_adaptive_target_enabled_ratio": "_nrl_specdec_scheduler_adaptive_target_enabled_ratio",
            "runner_adaptive_request_threshold": "_nrl_specdec_adaptive_request_threshold",
            "runner_adaptive_token_threshold": "_nrl_specdec_adaptive_token_threshold",
            "runner_adaptive_target_enabled_ratio": "_nrl_specdec_adaptive_target_enabled_ratio",
        }
        applied: dict[str, Any] = {}
        updated_objects = 0
        for obj in self._iter_vllm_gate_metric_objects():
            attrs = vars(obj)
            updated_this_object = False
            for key, attr in attr_map.items():
                if key not in controls or attr not in attrs:
                    continue
                value = controls[key]
                if attr.endswith("_enabled_ratio"):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                else:
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        continue
                    if value < 0:
                        continue
                setattr(obj, attr, value)
                applied[key] = value
                updated_this_object = True
            if updated_this_object:
                updated_objects += 1
        return {"updated_objects": updated_objects, "applied": applied}

    @staticmethod
    def configure_worker(
        num_gpus: int | float, bundle_indices: Optional[tuple[int, list[int]]] = None
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """Provides complete worker configuration for vLLM tensor and pipeline parallelism.

        This method configures the worker based on its role in tensor and pipeline parallelism,
        which is determined directly from the bundle_indices parameter.

        Args:
            num_gpus: Original GPU allocation for this worker based on the placement group
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for parallelism (if applicable)

        Returns:
            tuple with complete worker configuration:
              - 'resources': Resource allocation (e.g., num_gpus)
              - 'env_vars': Environment variables for this worker
              - 'init_kwargs': Parameters to pass to __init__ of the worker
        """
        # Initialize configuration
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        local_bundle_indices = None
        if bundle_indices is not None:
            node_idx = bundle_indices[0]
            local_bundle_indices = bundle_indices[1]
            init_kwargs["bundle_indices"] = local_bundle_indices

            """
            compute a unique seed from the node_idx and bundle_indices:
            node_idx = 0, bundle_indices = [0, 1, 2, 3] -> seed = 0*1024 + 0
            node_idx = 0, bundle_indices = [4, 5, 6, 7] -> seed = 0*1024 + 1
            node_idx = 1, bundle_indices = [0, 1, 2, 3] -> seed = 1*1024 + 0
            node_idx = 1, bundle_indices = [4, 5, 6, 7] -> seed = 1*1024 + 1
            """
            # For single worker groups, use a simpler seed calculation
            if len(local_bundle_indices) == 1:
                seed = node_idx * 1024 + local_bundle_indices[0]
            else:
                # For parallel groups, use the original calculation
                bundle_id = local_bundle_indices[0] // len(local_bundle_indices)
                seed = node_idx * 1024 + bundle_id

            init_kwargs["seed"] = seed
            # Need to give each DP group its own vllm cache to address:
            # https://github.com/vllm-project/vllm/issues/18851
            env_vars["VLLM_CACHE_ROOT"] = os.path.expanduser(f"~/.cache/vllm_{seed}")

        # Check if this worker is part of a parallel group (TP or TP+PP).
        # A worker is part of a parallel group if it's a secondary member (local_bundle_indices is None)
        # or if it's a primary member of a group with multiple workers.
        is_part_of_parallel_workers = (
            local_bundle_indices is not None and len(local_bundle_indices) > 1
        ) or local_bundle_indices is None

        if is_part_of_parallel_workers:
            # Ray + vllm likes to manage GPU assignment internally for parallel groups
            resources["num_gpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            init_kwargs["fraction_of_gpus"] = num_gpus

        env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = os.environ.get(
            "NRL_VLLM_ENABLE_V1_MULTIPROCESSING", "0"
        )
        # Skip vllm P2P check and rely on driver to report peer to peer capability.
        env_vars["VLLM_SKIP_P2P_CHECK"] = "1"

        return resources, env_vars, init_kwargs

    def __init__(
        self,
        config: VllmConfig,
        bundle_indices: Optional[list[int]] = None,
        fraction_of_gpus: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize a vLLM worker for distributed inference.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices within a node for parallelism.
                          Only needed for the first worker in each tied worker group.
            fraction_of_gpus: Fraction of GPUs to use for this worker
            seed: Random seed for initialization
        """
        self.cfg = config
        self.model_name = self.cfg["model_name"]
        self.tensor_parallel_size = self.cfg["vllm_cfg"]["tensor_parallel_size"]
        self.pipeline_parallel_size = self.cfg["vllm_cfg"]["pipeline_parallel_size"]
        self.expert_parallel_size = self.cfg["vllm_cfg"]["expert_parallel_size"]
        self.enable_expert_parallel = self.expert_parallel_size > 1
        self.gpu_memory_utilization = self.cfg["vllm_cfg"]["gpu_memory_utilization"]
        self.precision = self.cfg["vllm_cfg"]["precision"]
        self.fraction_of_gpus = fraction_of_gpus
        self.is_model_owner = bundle_indices is not None

        # Store the Python executable being used by this worker
        self.py_executable = sys.executable

        # Skip model loading if we're not the model owner
        if not self.is_model_owner:
            self.llm = None
            self.tokenizer = None
            self.rank = 0
            self.world_size = 1
            return

        # In Ray+vLLM setup, each worker process considers itself rank 0
        # vLLM handles the parallelism internally through Ray
        self.rank = 0
        self.world_size = 1

        # Monkey patches for vLLM behavior. We avoid importing vllm modules
        # here to prevent side effects during initialization and instead
        # locate the files via importlib metadata.

        from vllm.logger import init_logger

        logger = init_logger("vllm_patch")

        def _get_vllm_file(relative_path: str) -> str:
            """Return absolute path to a vLLM file or raise if it cannot be found.

            The relative_path should be a POSIX-style path under the vllm
            package root, e.g. "v1/executor/ray_executor.py" or
            "attention/layer.py".
            """
            spec = find_spec("vllm")
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError(
                    "vLLM package not found while attempting to patch "
                    f"'{relative_path}'. Ensure vLLM is installed and "
                    "available in this environment."
                )

            base_dir = next(iter(spec.submodule_search_locations))
            file_path = os.path.join(base_dir, *relative_path.split("/"))

            if not os.path.exists(file_path):
                raise RuntimeError(
                    "Failed to locate expected vLLM file to patch. "
                    f"Looked for '{relative_path}' at '{file_path}'. "
                    "This likely indicates an unexpected vLLM installation "
                    "layout or version mismatch."
                )

            return file_path

        def _patch_vllm_init_workers_ray():
            """Patch the vLLM ray_distributed_executor.py file.

            1. Pass custom runtime_env in _init_workers_ray call.
                - This allows passing custom py_executable to worker initialization.
            2. Add NCCL_CUMEM_ENABLE and NCCL_NVLS_ENABLE to vLLM ADDITIONAL_ENV_VARS.
                - This is a workaround to fix async vllm in some scenarios.
                - See https://github.com/NVIDIA-NeMo/RL/pull/898 for more details.
            """
            file_to_patch = None
            for candidate in (
                "v1/executor/ray_executor.py",
                "v1/executor/ray_distributed_executor.py",
                "executor/ray_distributed_executor.py",
            ):
                try:
                    file_to_patch = _get_vllm_file(candidate)
                    break
                except RuntimeError:
                    continue
            if file_to_patch is None:
                logger.warning(
                    "Could not find a vLLM Ray executor file to patch; "
                    "continuing without _init_workers_ray patch."
                )
                return

            with open(file_to_patch, "r") as f:
                content = f.read()

            init_workers_old = "self._init_workers_ray(placement_group)"
            init_workers_new = (
                'self._init_workers_ray(placement_group, '
                f'runtime_env={{"py_executable": "{self.py_executable}"}})'
            )

            specdec_runtime_env_vars = (
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "NRL_ALLOW_SPECDEC_REQUEST_LOGPROBS",
                "NRL_VLLM_OMIT_GENERATION_LOGPROBS",
                "NCCL_CUMEM_ENABLE",
                "NCCL_NVLS_ENABLE",
                "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
                "VLLM_ATTENTION_BACKEND",
                "VLLM_CACHE_ROOT",
                "VLLM_COMPILATION_LEVEL",
                "VLLM_CUDAGRAPH_CAPTURE_SIZES",
                "VLLM_CUDAGRAPH_MODE",
                "VLLM_DEEP_GEMM_WARMUP",
                "VLLM_DISABLE_USAGE_STATS",
                "VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH",
                "VLLM_SKIP_P2P_CHECK",
                "VLLM_SPECDEC_ADAPTIVE_ADJUST_INTERVAL",
                "VLLM_SPECDEC_ADAPTIVE_GATE_MODE",
                "VLLM_SPECDEC_ADAPTIVE_HYSTERESIS",
                "VLLM_SPECDEC_ADAPTIVE_INITIAL_REQUEST_THRESHOLD",
                "VLLM_SPECDEC_ADAPTIVE_INITIAL_TOKEN_THRESHOLD",
                "VLLM_SPECDEC_ADAPTIVE_MAX_REQUEST_THRESHOLD",
                "VLLM_SPECDEC_ADAPTIVE_MAX_TOKEN_THRESHOLD",
                "VLLM_SPECDEC_ADAPTIVE_MIN_REQUEST_THRESHOLD",
                "VLLM_SPECDEC_ADAPTIVE_MIN_TOKEN_THRESHOLD",
                "VLLM_SPECDEC_ADAPTIVE_REQUEST_STEP",
                "VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO",
                "VLLM_SPECDEC_ADAPTIVE_TOKEN_STEP",
                "VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL",
                "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD",
                "VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD",
                "VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS",
                "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD",
                "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD",
                "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS",
                "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD",
                "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD",
                "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS",
                "VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS",
                "VLLM_USE_RAY_COMPILED_DAG",
                "VLLM_USE_RAY_SPMD_WORKER",
                "VLLM_USE_RAY_WRAPPED_PP_COMM",
            )
            additional_env_vars_new = (
                "ADDITIONAL_ENV_VARS = {"
                + ", ".join(f'"{name}"' for name in specdec_runtime_env_vars)
                + "}"
            )
            additional_env_vars_old = (
                'ADDITIONAL_ENV_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}'
            )
            additional_env_vars_previous_patch = (
                'ADDITIONAL_ENV_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", '
                '"NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE", '
                '"RAY_ENABLE_UV_RUN_RUNTIME_ENV"}'
            )

            need_replace = False
            if init_workers_new not in content and init_workers_old in content:
                content = content.replace(init_workers_old, init_workers_new)
                need_replace = True

            if additional_env_vars_new not in content:
                for old_line in (
                    additional_env_vars_previous_patch,
                    additional_env_vars_old,
                ):
                    if old_line in content:
                        content = content.replace(old_line, additional_env_vars_new)
                        need_replace = True
                        break
                else:
                    match = re.search(r"ADDITIONAL_ENV_VARS\s*=\s*\{([^}]*)\}", content)
                    if match:
                        existing = {
                            item[0] or item[1]
                            for item in re.findall(
                                r'"([^"]+)"|\'([^\']+)\'', match.group(1)
                            )
                        }
                        merged = []
                        for name in specdec_runtime_env_vars:
                            if name not in merged:
                                merged.append(name)
                        for name in sorted(existing):
                            if name and name not in merged:
                                merged.append(name)
                        replacement = (
                            "ADDITIONAL_ENV_VARS = {"
                            + ", ".join(f'"{name}"' for name in merged)
                            + "}"
                        )
                        content = (
                            content[: match.start()]
                            + replacement
                            + content[match.end() :]
                        )
                        need_replace = True
                    elif spec_decode_requested:
                        raise RuntimeError(
                            "Could not patch vLLM ADDITIONAL_ENV_VARS for "
                            "SpecDec runtime env propagation."
                        )

            if not need_replace:
                return

            # Write back the patched content
            with open(file_to_patch, "w") as f:
                f.write(content)

        def _patch_vllm_vit_flash_attn_backend():
            """Patch vLLM vision attention backend selection logic.

            Modify the CUDA branch of maybe_get_vit_flash_attn_backend in
            vllm.attention.layer to avoid overriding the backend when it
            is already set to XFORMERS. This avoids flash attention related
            errors when the ViT head dimension is not a multiple of 32.

            Related issues:
            - https://github.com/vllm-project/vllm/issues/27562
            - https://github.com/vllm-project/vllm/issues/26989

            This is properly fixed in https://github.com/vllm-project/vllm/pull/28763. We can remove this patch once we upgrade to a version of vllm that contains this fix.
            """
            file_to_patch = _get_vllm_file("attention/layer.py")
            with open(file_to_patch, "r") as f:
                content = f.read()

            old_snippet = (
                "    elif current_platform.is_cuda():\n"
                "        if (\n"
                "            attn_backend != AttentionBackendEnum.FLASH_ATTN\n"
                "            and check_upstream_fa_availability(torch.get_default_dtype())\n"
                "        ):\n"
                "            attn_backend = AttentionBackendEnum.FLASH_ATTN\n"
                "            use_upstream_fa = True\n"
            )

            new_snippet = (
                "    elif current_platform.is_cuda():\n"
                "        if (\n"
                "            attn_backend != AttentionBackendEnum.FLASH_ATTN\n"
                "            and attn_backend != AttentionBackendEnum.XFORMERS\n"
                "            and check_upstream_fa_availability(torch.get_default_dtype())\n"
                "        ):\n"
                "            attn_backend = AttentionBackendEnum.FLASH_ATTN\n"
                "            use_upstream_fa = True\n"
            )

            # Only patch if the file still has the old snippet and
            # hasn't been patched already.
            if new_snippet in content or old_snippet not in content:
                return

            content = content.replace(old_snippet, new_snippet)

            with open(file_to_patch, "w") as f:
                f.write(content)

        def _patch_vllm_speculative_decoding_post_step(required: bool) -> int:
            """Patch vLLM in-process client to publish draft ids after each step.

            Some vLLM versions return `(outputs, model_executed)` from
            EngineCore.step() but the in-process client discards the flag and
            never calls `post_step()`. With SpecDec this can leave the scheduler
            with zero draft tokens even though the drafter ran.
            """

            file_to_patch = _get_vllm_file("v1/engine/core_client.py")
            with open(file_to_patch) as f:
                content = f.read()

            if "post_step(model_executed=model_executed)" in content:
                return 0

            replacements = [
                (
                    "    def get_output(self) -> EngineCoreOutputs:\n"
                    "        outputs, _ = self.engine_core.step()\n"
                    "        return outputs.get(0) or EngineCoreOutputs()\n",
                    "    def get_output(self) -> EngineCoreOutputs:\n"
                    "        outputs, model_executed = self.engine_core.step()\n"
                    "        self.engine_core.post_step(model_executed=model_executed)\n"
                    "        return outputs.get(0) or EngineCoreOutputs()\n",
                ),
                (
                    "    def get_output(self) -> EngineCoreOutputs:\n"
                    "        outputs, _ = self.engine_core.step_fn()\n"
                    "        return outputs and outputs.get(0) or EngineCoreOutputs()",
                    "    def get_output(self) -> EngineCoreOutputs:\n"
                    "        outputs, model_executed = self.engine_core.step_fn()\n"
                    "        self.engine_core.post_step(model_executed=model_executed)\n"
                    "        return outputs and outputs.get(0) or EngineCoreOutputs()",
                ),
            ]

            for old, new in replacements:
                if old in content:
                    with open(file_to_patch, "w") as f:
                        f.write(content.replace(old, new, 1))
                    return 1

            inproc_get_output_pattern = re.compile(
                r"(class InprocClient\(EngineCoreClient\):.*?"
                r"    def get_output\(self\) -> EngineCoreOutputs:\n)"
                r"(?:        .*\n)+?"
                r"(?=\n    def get_supported_tasks)",
                flags=re.DOTALL,
            )

            def _generic_inproc_replacement(match: re.Match[str]) -> str:
                return (
                    match.group(1)
                    + "        step_fn = getattr(self.engine_core, \"step_fn\", None)\n"
                    + "        if step_fn is None:\n"
                    + "            step_fn = self.engine_core.step\n"
                    + "        step_result = step_fn()\n"
                    + "        if isinstance(step_result, tuple) and len(step_result) == 2:\n"
                    + "            outputs, model_executed = step_result\n"
                    + "            post_step = getattr(self.engine_core, \"post_step\", None)\n"
                    + "            if post_step is not None:\n"
                    + "                post_step(model_executed=model_executed)\n"
                    + "        else:\n"
                    + "            outputs = step_result\n"
                    + "        return outputs and outputs.get(0) or EngineCoreOutputs()\n"
                )

            patched_content, count = inproc_get_output_pattern.subn(
                _generic_inproc_replacement, content, count=1
            )
            if count:
                with open(file_to_patch, "w") as f:
                    f.write(patched_content)
                return 1

            message = (
                "Could not patch vLLM speculative decoding post_step in "
                f"{file_to_patch}: missing expected InprocClient.get_output snippet"
            )
            if required:
                raise RuntimeError(message)
            logger.warning(message)
            return 0

        def _patch_vllm_batch_gated_speculative_decoding():
            """Patch vLLM to avoid SpecDec work when the active batch is large.

            vLLM's public `disable_by_batch_size` is request-admission scoped in
            some releases, which does not catch the long-tail phase of a large
            GRPO batch. Patch both the scheduler and the model runner: the
            scheduler avoids allocating EAGLE lookahead KV slots for large active
            batches, while the model runner skips proposing the next draft.
            Already-proposed draft tokens can still be verified, preserving
            correctness across gate transitions.
            """

            def _patch_file(path, replacement_groups):
                with open(path) as f:
                    content = f.read()

                required_markers = [
                    "NRL_SPECDEC_BATCH_GATE_PATCH_V4",
                    "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD",
                    "VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD",
                    "_nrl_specdec_batch_gate_threshold",
                    "_nrl_specdec_batch_gate_token_threshold",
                    "specdec_batch_gate_num_requests",
                    "specdec_batch_gate_num_tokens",
                    "specdec_batch_gate_threshold_disabled",
                    "specdec_batch_gate_disabled",
                    "specdec_batch_gate_disabled = False",
                    "not specdec_batch_gate_disabled",
                    "specdec_batch_gate_checked_count",
                    "specdec_batch_gate_log_interval",
                    "NRL SpecDec batch gate:",
                ]
                if (
                    "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD" in content
                    or "specdec_batch_gate_disabled" in content
                ):
                    scheduled_token_deadlock_block = (
                        "            specdec_scheduled_tokens = getattr(\n"
                        "                scheduler_output, \"scheduled_spec_decode_tokens\", None\n"
                        "            )\n"
                        "            if specdec_scheduled_tokens is not None and hasattr(\n"
                        "                specdec_scheduled_tokens, \"__len__\"\n"
                        "            ):\n"
                        "                specdec_batch_gate_disabled = (\n"
                        "                    specdec_batch_gate_threshold_disabled\n"
                        "                    or (\n"
                        "                        len(specdec_scheduled_tokens) == 0\n"
                        "                        and specdec_batch_gate_num_requests > 0\n"
                        "                    )\n"
                        "                )\n"
                        "            else:\n"
                        "                specdec_batch_gate_disabled = specdec_batch_gate_threshold_disabled\n"
                    )
                    threshold_only_block = (
                        "            specdec_batch_gate_disabled = specdec_batch_gate_threshold_disabled\n"
                    )
                    if scheduled_token_deadlock_block in content:
                        upgraded_content = content.replace(
                            scheduled_token_deadlock_block,
                            threshold_only_block,
                        )
                        with open(path, "w") as f:
                            f.write(upgraded_content)
                        content = upgraded_content
                    if "specdec_batch_gate_threshold_disabled" not in content:
                        upgraded_content = content.replace(
                            "            specdec_scheduled_tokens = getattr(\n"
                            "                scheduler_output, \"scheduled_spec_decode_tokens\", None\n"
                            "            )\n"
                            "            if specdec_scheduled_tokens is not None and hasattr(\n"
                            "                specdec_scheduled_tokens, \"__len__\"\n"
                            "            ):\n"
                            "                specdec_batch_gate_disabled = (\n"
                            "                    len(specdec_scheduled_tokens) == 0\n"
                            "                    and specdec_batch_gate_num_requests > 0\n"
                            "                )\n"
                            "            else:\n"
                            "                specdec_batch_gate_disabled = (\n"
                            "                    (\n"
                            "                        specdec_batch_gate_threshold > 0\n"
                            "                        and specdec_batch_gate_num_requests > specdec_batch_gate_threshold\n"
                            "                    )\n"
                            "                    or (\n"
                            "                        specdec_batch_gate_token_threshold > 0\n"
                            "                        and specdec_batch_gate_num_tokens\n"
                            "                        > specdec_batch_gate_token_threshold\n"
                            "                    )\n"
                            "                )\n",
                            "            specdec_batch_gate_threshold_disabled = (\n"
                            "                (\n"
                            "                    specdec_batch_gate_threshold > 0\n"
                            "                    and specdec_batch_gate_num_requests > specdec_batch_gate_threshold\n"
                            "                )\n"
                            "                or (\n"
                            "                    specdec_batch_gate_token_threshold > 0\n"
                            "                    and specdec_batch_gate_num_tokens\n"
                            "                    > specdec_batch_gate_token_threshold\n"
                            "                )\n"
                            "            )\n"
                            "            specdec_batch_gate_disabled = specdec_batch_gate_threshold_disabled\n",
                        )
                        if upgraded_content != content:
                            with open(path, "w") as f:
                                f.write(upgraded_content)
                            content = upgraded_content
                    missing_markers = [
                        marker for marker in required_markers if marker not in content
                    ]
                    if missing_markers:
                        raise RuntimeError(
                            "Found a partial or outdated vLLM SpecDec batch-gate "
                            f"patch in {path}; missing markers: "
                            + ", ".join(missing_markers)
                        )
                    return 0

                failures = []
                for group_idx, replacements in enumerate(replacement_groups):
                    patched_content = content
                    applied = 0
                    missing = []
                    for idx, (old, new) in enumerate(replacements):
                        if old not in patched_content:
                            missing.append((idx, old))
                            continue
                        patched_content = patched_content.replace(old, new, 1)
                        applied += 1

                    if not missing and applied:
                        missing_markers = [
                            marker
                            for marker in required_markers
                            if marker not in patched_content
                        ]
                        if missing_markers:
                            raise RuntimeError(
                                "Applied vLLM SpecDec batch-gate patch to "
                                f"{path}, but the patched file is missing markers: "
                                + ", ".join(missing_markers)
                            )
                        with open(path, "w") as f:
                            f.write(patched_content)
                        return applied

                    failures.append(
                        f"group {group_idx}: "
                        + ", ".join(
                            f"missing snippet #{idx}: {old[:160]!r}"
                            for idx, old in missing[:3]
                        )
                    )

                raise RuntimeError(
                    "Could not patch vLLM batch-gated SpecDec in "
                    f"{path}. Tried {len(replacement_groups)} known layouts. "
                    + " | ".join(failures)
                )

            def _patch_scheduler_file(path, replacement_groups):
                with open(path) as f:
                    content = f.read()

                def _has_scheduler_lookahead_gate_target(text):
                    return (
                        "num_lookahead_tokens=(" in text
                        or "num_lookahead_tokens=_nrl_specdec_scheduler_lookahead_tokens("
                        in text
                        or "else _nrl_specdec_scheduler_lookahead_tokens(" in text
                    )

                static_active_batch_marker = (
                    "max(len(self.running), len(num_scheduled_tokens) + 1)"
                )
                adaptive_active_batch_marker = (
                    "active_requests = max(num_requests, len(self.running))"
                )
                required_markers = [
                    "NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V5",
                    "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD",
                    "VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD",
                    "nrl_specdec_scheduler_gate_threshold",
                    "nrl_specdec_scheduler_gate_token_threshold",
                ]
                if "NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V5" in content:
                    if (
                        static_active_batch_marker not in content
                        and adaptive_active_batch_marker not in content
                    ):
                        upgraded_content = content.replace(
                            "(len(num_scheduled_tokens) + 1)",
                            static_active_batch_marker,
                        )
                        if upgraded_content != content:
                            with open(path, "w") as f:
                                f.write(upgraded_content)
                            content = upgraded_content
                    missing_markers = [
                        marker for marker in required_markers if marker not in content
                    ]
                    if (
                        static_active_batch_marker not in content
                        and adaptive_active_batch_marker not in content
                    ):
                        missing_markers.append("active-batch request pressure marker")
                    if not _has_scheduler_lookahead_gate_target(content):
                        missing_markers.append("scheduler lookahead gate target")
                    if missing_markers:
                        raise RuntimeError(
                            "Found a partial vLLM SpecDec scheduler-gate patch "
                            f"in {path}; missing markers: "
                            + ", ".join(missing_markers)
                        )
                    return 0
                if (
                    "NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V" in content
                    or "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD" in content
                ):
                    raise RuntimeError(
                        "Found an outdated vLLM SpecDec scheduler-gate patch in "
                        f"{path}. Rebuild the Ray vLLM environment before running "
                        "the scheduler lookahead gate."
                    )

                failures = []
                for group_idx, replacements in enumerate(replacement_groups):
                    patched_content = content
                    applied = 0
                    missing = []
                    for idx, (old, new) in enumerate(replacements):
                        if old not in patched_content:
                            missing.append((idx, old))
                            continue
                        patched_content = patched_content.replace(old, new, 1)
                        applied += 1

                    if not missing and applied:
                        missing_markers = [
                            marker
                            for marker in required_markers
                            if marker not in patched_content
                        ]
                        if not _has_scheduler_lookahead_gate_target(patched_content):
                            missing_markers.append("scheduler lookahead gate target")
                        if missing_markers:
                            raise RuntimeError(
                                "Applied vLLM SpecDec scheduler-gate patch to "
                                f"{path}, but the patched file is missing markers: "
                                + ", ".join(missing_markers)
                            )
                        with open(path, "w") as f:
                            f.write(patched_content)
                        return applied

                    failures.append(
                        f"group {group_idx}: "
                        + ", ".join(
                            f"missing snippet #{idx}: {old[:160]!r}"
                            for idx, old in missing[:3]
                        )
                    )

                raise RuntimeError(
                    "Could not patch vLLM scheduler-gated SpecDec lookahead in "
                    f"{path}. Tried {len(replacement_groups)} known layouts. "
                    + " | ".join(failures)
                )

            gpu_model_runner = _get_vllm_file("v1/worker/gpu_model_runner.py")
            applied = _patch_file(
                gpu_model_runner,
                [
                    [
                        (
                            "        if self.speculative_config:\n"
                            "            assert spec_decode_common_attn_metadata is not None\n"
                            "            with record_function_or_nullcontext(\"Draft\"):\n"
                            "                self._draft_token_ids = self.propose_draft_token_ids(\n"
                            "                    scheduler_output,\n"
                            "                    valid_sampled_token_ids,\n"
                            "                    self.input_batch.sampling_metadata,\n"
                            "                    hidden_states,\n"
                            "                    sample_hidden_states,\n"
                            "                    aux_hidden_states,\n"
                            "                    spec_decode_metadata,\n"
                            "                    spec_decode_common_attn_metadata,\n"
                            "                )\n",
                            "        # NRL_SPECDEC_BATCH_GATE_PATCH_V4\n"
                            "        specdec_batch_gate_disabled = False\n"
                            "        if self.speculative_config:\n"
                            "            specdec_batch_gate_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_batch_gate_threshold\", None\n"
                            "            )\n"
                            "            if specdec_batch_gate_threshold is None:\n"
                            "                specdec_batch_gate_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                specdec_batch_gate_threshold = (\n"
                            "                    int(specdec_batch_gate_threshold_str)\n"
                            "                    if specdec_batch_gate_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_batch_gate_threshold = specdec_batch_gate_threshold\n"
                            "            specdec_batch_gate_token_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_batch_gate_token_threshold\", None\n"
                            "            )\n"
                            "            if specdec_batch_gate_token_threshold is None:\n"
                            "                specdec_batch_gate_token_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                specdec_batch_gate_token_threshold = (\n"
                            "                    int(specdec_batch_gate_token_threshold_str)\n"
                            "                    if specdec_batch_gate_token_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_batch_gate_token_threshold = specdec_batch_gate_token_threshold\n"
                            "            specdec_batch_gate_num_requests = (\n"
                            "                len(scheduler_output.num_scheduled_tokens)\n"
                            "                if hasattr(scheduler_output.num_scheduled_tokens, \"__len__\")\n"
                            "                else (1 if scheduler_output.num_scheduled_tokens else 0)\n"
                            "            )\n"
                            "            if hasattr(scheduler_output.num_scheduled_tokens, \"values\"):\n"
                            "                specdec_batch_gate_num_tokens = sum(\n"
                            "                    scheduler_output.num_scheduled_tokens.values()\n"
                            "                )\n"
                            "            elif isinstance(scheduler_output.num_scheduled_tokens, int):\n"
                            "                specdec_batch_gate_num_tokens = scheduler_output.num_scheduled_tokens\n"
                            "            else:\n"
                            "                specdec_batch_gate_num_tokens = sum(\n"
                            "                    scheduler_output.num_scheduled_tokens\n"
                            "                )\n"
                            "            specdec_batch_gate_threshold_disabled = (\n"
                            "                (\n"
                            "                    specdec_batch_gate_threshold > 0\n"
                            "                    and specdec_batch_gate_num_requests > specdec_batch_gate_threshold\n"
                            "                )\n"
                            "                or (\n"
                            "                    specdec_batch_gate_token_threshold > 0\n"
                            "                    and specdec_batch_gate_num_tokens\n"
                            "                    > specdec_batch_gate_token_threshold\n"
                            "                )\n"
                            "            )\n"
                            "            specdec_batch_gate_disabled = specdec_batch_gate_threshold_disabled\n"
                            "            self._nrl_specdec_batch_gate_last_num_requests = specdec_batch_gate_num_requests\n"
                            "            self._nrl_specdec_batch_gate_last_num_tokens = specdec_batch_gate_num_tokens\n"
                            "            self._nrl_specdec_batch_gate_last_disabled = specdec_batch_gate_disabled\n"
                            "            specdec_batch_gate_checked_count = getattr(\n"
                            "                self, \"_nrl_specdec_batch_gate_checked_count\", 0\n"
                            "            ) + 1\n"
                            "            self._nrl_specdec_batch_gate_checked_count = specdec_batch_gate_checked_count\n"
                            "            if specdec_batch_gate_disabled:\n"
                            "                self._nrl_specdec_batch_gate_disabled_count = getattr(\n"
                            "                    self, \"_nrl_specdec_batch_gate_disabled_count\", 0\n"
                            "                ) + 1\n"
                            "            else:\n"
                            "                self._nrl_specdec_batch_gate_enabled_count = getattr(\n"
                            "                    self, \"_nrl_specdec_batch_gate_enabled_count\", 0\n"
                            "                ) + 1\n"
                            "            specdec_batch_gate_log_interval = getattr(\n"
                            "                self, \"_nrl_specdec_batch_gate_log_interval\", None\n"
                            "            )\n"
                            "            if specdec_batch_gate_log_interval is None:\n"
                            "                specdec_batch_gate_log_interval_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL\", \"256\"\n"
                            "                )\n"
                            "                specdec_batch_gate_log_interval = (\n"
                            "                    int(specdec_batch_gate_log_interval_str)\n"
                            "                    if specdec_batch_gate_log_interval_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_batch_gate_log_interval = specdec_batch_gate_log_interval\n"
                            "            if specdec_batch_gate_log_interval > 0 and (\n"
                            "                specdec_batch_gate_checked_count == 1\n"
                            "                or specdec_batch_gate_checked_count % specdec_batch_gate_log_interval == 0\n"
                            "            ):\n"
                            "                try:\n"
                            "                    logger.info(\n"
                            "                        \"NRL SpecDec batch gate: checked=%s disabled=%s enabled=%s last_disabled=%s requests=%s tokens=%s request_threshold=%s token_threshold=%s\",\n"
                            "                        specdec_batch_gate_checked_count,\n"
                            "                        getattr(self, \"_nrl_specdec_batch_gate_disabled_count\", 0),\n"
                            "                        getattr(self, \"_nrl_specdec_batch_gate_enabled_count\", 0),\n"
                            "                        specdec_batch_gate_disabled,\n"
                            "                        specdec_batch_gate_num_requests,\n"
                            "                        specdec_batch_gate_num_tokens,\n"
                            "                        specdec_batch_gate_threshold,\n"
                            "                        specdec_batch_gate_token_threshold,\n"
                            "                    )\n"
                            "                except Exception:\n"
                            "                    pass\n"
                            "        if specdec_batch_gate_disabled:\n"
                            "            self._draft_token_ids = None\n"
                            "        if self.speculative_config and not specdec_batch_gate_disabled:\n"
                            "            assert spec_decode_common_attn_metadata is not None\n"
                            "            with record_function_or_nullcontext(\"Draft\"):\n"
                            "                self._draft_token_ids = self.propose_draft_token_ids(\n"
                            "                    scheduler_output,\n"
                            "                    valid_sampled_token_ids,\n"
                            "                    self.input_batch.sampling_metadata,\n"
                            "                    hidden_states,\n"
                            "                    sample_hidden_states,\n"
                            "                    aux_hidden_states,\n"
                            "                    spec_decode_metadata,\n"
                            "                    spec_decode_common_attn_metadata,\n"
                            "                )\n",
                        ),
                    ],
                    [
                        (
                            "        input_fits_in_drafter = spec_decode_common_attn_metadata and (\n"
                            "            spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens\n"
                            "            <= effective_drafter_max_model_len\n"
                            "        )\n",
                            "        input_fits_in_drafter = spec_decode_common_attn_metadata and (\n"
                            "            spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens\n"
                            "            <= effective_drafter_max_model_len\n"
                            "        )\n"
                            "        # NRL_SPECDEC_BATCH_GATE_PATCH_V4\n"
                            "        specdec_batch_gate_disabled = False\n"
                            "        if self.speculative_config:\n"
                            "            specdec_batch_gate_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_batch_gate_threshold\", None\n"
                            "            )\n"
                            "            if specdec_batch_gate_threshold is None:\n"
                            "                specdec_batch_gate_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                specdec_batch_gate_threshold = (\n"
                            "                    int(specdec_batch_gate_threshold_str)\n"
                            "                    if specdec_batch_gate_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_batch_gate_threshold = specdec_batch_gate_threshold\n"
                            "            specdec_batch_gate_token_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_batch_gate_token_threshold\", None\n"
                            "            )\n"
                            "            if specdec_batch_gate_token_threshold is None:\n"
                            "                specdec_batch_gate_token_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                specdec_batch_gate_token_threshold = (\n"
                            "                    int(specdec_batch_gate_token_threshold_str)\n"
                            "                    if specdec_batch_gate_token_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_batch_gate_token_threshold = specdec_batch_gate_token_threshold\n"
                            "            specdec_batch_gate_num_requests = (\n"
                            "                len(scheduler_output.num_scheduled_tokens)\n"
                            "                if hasattr(scheduler_output.num_scheduled_tokens, \"__len__\")\n"
                            "                else (1 if scheduler_output.num_scheduled_tokens else 0)\n"
                            "            )\n"
                            "            if hasattr(scheduler_output.num_scheduled_tokens, \"values\"):\n"
                            "                specdec_batch_gate_num_tokens = sum(\n"
                            "                    scheduler_output.num_scheduled_tokens.values()\n"
                            "                )\n"
                            "            elif isinstance(scheduler_output.num_scheduled_tokens, int):\n"
                            "                specdec_batch_gate_num_tokens = scheduler_output.num_scheduled_tokens\n"
                            "            else:\n"
                            "                specdec_batch_gate_num_tokens = sum(\n"
                            "                    scheduler_output.num_scheduled_tokens\n"
                            "                )\n"
                            "            specdec_batch_gate_threshold_disabled = (\n"
                            "                (\n"
                            "                    specdec_batch_gate_threshold > 0\n"
                            "                    and specdec_batch_gate_num_requests > specdec_batch_gate_threshold\n"
                            "                )\n"
                            "                or (\n"
                            "                    specdec_batch_gate_token_threshold > 0\n"
                            "                    and specdec_batch_gate_num_tokens\n"
                            "                    > specdec_batch_gate_token_threshold\n"
                            "                )\n"
                            "            )\n"
                            "            specdec_batch_gate_disabled = specdec_batch_gate_threshold_disabled\n"
                            "            self._nrl_specdec_batch_gate_last_num_requests = specdec_batch_gate_num_requests\n"
                            "            self._nrl_specdec_batch_gate_last_num_tokens = specdec_batch_gate_num_tokens\n"
                            "            self._nrl_specdec_batch_gate_last_disabled = specdec_batch_gate_disabled\n"
                            "            specdec_batch_gate_checked_count = getattr(\n"
                            "                self, \"_nrl_specdec_batch_gate_checked_count\", 0\n"
                            "            ) + 1\n"
                            "            self._nrl_specdec_batch_gate_checked_count = specdec_batch_gate_checked_count\n"
                            "            if specdec_batch_gate_disabled:\n"
                            "                self._nrl_specdec_batch_gate_disabled_count = getattr(\n"
                            "                    self, \"_nrl_specdec_batch_gate_disabled_count\", 0\n"
                            "                ) + 1\n"
                            "            else:\n"
                            "                self._nrl_specdec_batch_gate_enabled_count = getattr(\n"
                            "                    self, \"_nrl_specdec_batch_gate_enabled_count\", 0\n"
                            "                ) + 1\n"
                            "            specdec_batch_gate_log_interval = getattr(\n"
                            "                self, \"_nrl_specdec_batch_gate_log_interval\", None\n"
                            "            )\n"
                            "            if specdec_batch_gate_log_interval is None:\n"
                            "                specdec_batch_gate_log_interval_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL\", \"256\"\n"
                            "                )\n"
                            "                specdec_batch_gate_log_interval = (\n"
                            "                    int(specdec_batch_gate_log_interval_str)\n"
                            "                    if specdec_batch_gate_log_interval_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_batch_gate_log_interval = specdec_batch_gate_log_interval\n"
                            "            if specdec_batch_gate_log_interval > 0 and (\n"
                            "                specdec_batch_gate_checked_count == 1\n"
                            "                or specdec_batch_gate_checked_count % specdec_batch_gate_log_interval == 0\n"
                            "            ):\n"
                            "                try:\n"
                            "                    logger.info(\n"
                            "                        \"NRL SpecDec batch gate: checked=%s disabled=%s enabled=%s last_disabled=%s requests=%s tokens=%s request_threshold=%s token_threshold=%s\",\n"
                            "                        specdec_batch_gate_checked_count,\n"
                            "                        getattr(self, \"_nrl_specdec_batch_gate_disabled_count\", 0),\n"
                            "                        getattr(self, \"_nrl_specdec_batch_gate_enabled_count\", 0),\n"
                            "                        specdec_batch_gate_disabled,\n"
                            "                        specdec_batch_gate_num_requests,\n"
                            "                        specdec_batch_gate_num_tokens,\n"
                            "                        specdec_batch_gate_threshold,\n"
                            "                        specdec_batch_gate_token_threshold,\n"
                            "                    )\n"
                            "                except Exception:\n"
                            "                    pass\n"
                            "        if specdec_batch_gate_disabled:\n"
                            "            self._draft_token_ids = None\n",
                        ),
                        (
                            "            if input_fits_in_drafter:\n"
                            "                # EAGLE speculative decoding can use the GPU sampled tokens\n"
                            "                # as inputs, and does not need to wait for bookkeeping to finish.\n"
                            "                propose_draft_token_ids(sampled_token_ids)\n"
                            "            elif self.valid_sampled_token_count_event is not None:\n",
                            "            if input_fits_in_drafter:\n"
                            "                if not specdec_batch_gate_disabled:\n"
                            "                    # EAGLE speculative decoding can use the GPU sampled tokens\n"
                            "                    # as inputs, and does not need to wait for bookkeeping to finish.\n"
                            "                    propose_draft_token_ids(sampled_token_ids)\n"
                            "            elif self.valid_sampled_token_count_event is not None:\n",
                        ),
                        (
                            "        if (\n"
                            "            self.speculative_config\n"
                            "            and not use_padded_batch_for_eagle\n"
                            "            and input_fits_in_drafter\n"
                            "        ):\n",
                            "        if (\n"
                            "            self.speculative_config\n"
                            "            and not use_padded_batch_for_eagle\n"
                            "            and input_fits_in_drafter\n"
                            "            and not specdec_batch_gate_disabled\n"
                            "        ):\n",
                        ),
                    ],
                ],
            )

            scheduler = _get_vllm_file("v1/core/sched/scheduler.py")
            applied += _patch_scheduler_file(
                scheduler,
                [
                    [
                        (
                            "        # Spec decode-related.\n"
                            "        scheduled_spec_decode_tokens: dict[str, list[int]] = {}\n",
                            "        # Spec decode-related.\n"
                            "        scheduled_spec_decode_tokens: dict[str, list[int]] = {}\n"
                            "        # NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V5\n"
                            "        if self.num_lookahead_tokens > 0:\n"
                            "            nrl_specdec_scheduler_gate_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_scheduler_gate_threshold\", None\n"
                            "            )\n"
                            "            if nrl_specdec_scheduler_gate_threshold is None:\n"
                            "                nrl_specdec_scheduler_gate_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                nrl_specdec_scheduler_gate_threshold = (\n"
                            "                    int(nrl_specdec_scheduler_gate_threshold_str)\n"
                            "                    if nrl_specdec_scheduler_gate_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_scheduler_gate_threshold = nrl_specdec_scheduler_gate_threshold\n"
                            "            nrl_specdec_scheduler_gate_token_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_scheduler_gate_token_threshold\", None\n"
                            "            )\n"
                            "            if nrl_specdec_scheduler_gate_token_threshold is None:\n"
                            "                nrl_specdec_scheduler_gate_token_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                nrl_specdec_scheduler_gate_token_threshold = (\n"
                            "                    int(nrl_specdec_scheduler_gate_token_threshold_str)\n"
                            "                    if nrl_specdec_scheduler_gate_token_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_scheduler_gate_token_threshold = nrl_specdec_scheduler_gate_token_threshold\n"
                            "        else:\n"
                            "            nrl_specdec_scheduler_gate_threshold = 0\n"
                            "            nrl_specdec_scheduler_gate_token_threshold = 0\n",
                        ),
                        (
                            "                new_blocks = self.kv_cache_manager.allocate_slots(\n"
                            "                    request,\n"
                            "                    num_new_tokens,\n"
                            "                    num_lookahead_tokens=self.num_lookahead_tokens)\n",
                            "                new_blocks = self.kv_cache_manager.allocate_slots(\n"
                            "                    request,\n"
                            "                    num_new_tokens,\n"
                            "                    num_lookahead_tokens=(\n"
                            "                        0\n"
                            "                        if nrl_specdec_scheduler_gate_threshold > 0\n"
                            "                        and max(len(self.running), len(num_scheduled_tokens) + 1)\n"
                            "                        > nrl_specdec_scheduler_gate_threshold\n"
                            "                        or nrl_specdec_scheduler_gate_token_threshold > 0\n"
                            "                        and sum(num_scheduled_tokens.values()) + num_new_tokens\n"
                            "                        > nrl_specdec_scheduler_gate_token_threshold\n"
                            "                        else self.num_lookahead_tokens))\n",
                        ),
                        (
                            "                effective_lookahead_tokens = (0 if request.num_computed_tokens\n"
                            "                                              == 0 else\n"
                            "                                              self.num_lookahead_tokens)\n",
                            "                effective_lookahead_tokens = (\n"
                            "                    0\n"
                            "                    if request.num_computed_tokens == 0\n"
                            "                    or (\n"
                            "                        nrl_specdec_scheduler_gate_threshold > 0\n"
                            "                        and max(len(self.running), len(num_scheduled_tokens) + 1)\n"
                            "                        > nrl_specdec_scheduler_gate_threshold\n"
                            "                    )\n"
                            "                    or (\n"
                            "                        nrl_specdec_scheduler_gate_token_threshold > 0\n"
                            "                        and sum(num_scheduled_tokens.values()) + num_new_tokens\n"
                            "                        > nrl_specdec_scheduler_gate_token_threshold\n"
                            "                    )\n"
                            "                    else self.num_lookahead_tokens\n"
                            "                )\n",
                        ),
                    ],
                    [
                        (
                            "        # Spec decode-related.\n"
                            "        scheduled_spec_decode_tokens: dict[str, list[int]] = {}\n",
                            "        # Spec decode-related.\n"
                            "        scheduled_spec_decode_tokens: dict[str, list[int]] = {}\n"
                            "        # NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V5\n"
                            "        if self.num_lookahead_tokens > 0:\n"
                            "            nrl_specdec_scheduler_gate_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_scheduler_gate_threshold\", None\n"
                            "            )\n"
                            "            if nrl_specdec_scheduler_gate_threshold is None:\n"
                            "                nrl_specdec_scheduler_gate_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                nrl_specdec_scheduler_gate_threshold = (\n"
                            "                    int(nrl_specdec_scheduler_gate_threshold_str)\n"
                            "                    if nrl_specdec_scheduler_gate_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_scheduler_gate_threshold = nrl_specdec_scheduler_gate_threshold\n"
                            "            nrl_specdec_scheduler_gate_token_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_scheduler_gate_token_threshold\", None\n"
                            "            )\n"
                            "            if nrl_specdec_scheduler_gate_token_threshold is None:\n"
                            "                nrl_specdec_scheduler_gate_token_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                nrl_specdec_scheduler_gate_token_threshold = (\n"
                            "                    int(nrl_specdec_scheduler_gate_token_threshold_str)\n"
                            "                    if nrl_specdec_scheduler_gate_token_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_scheduler_gate_token_threshold = nrl_specdec_scheduler_gate_token_threshold\n"
                            "        else:\n"
                            "            nrl_specdec_scheduler_gate_threshold = 0\n"
                            "            nrl_specdec_scheduler_gate_token_threshold = 0\n",
                        ),
                        (
                            "                    new_blocks = self.kv_cache_manager.allocate_slots(\n"
                            "                        request,\n"
                            "                        num_new_tokens,\n"
                            "                        num_lookahead_tokens=self.num_lookahead_tokens,\n"
                            "                    )\n",
                            "                    new_blocks = self.kv_cache_manager.allocate_slots(\n"
                            "                        request,\n"
                            "                        num_new_tokens,\n"
                            "                        num_lookahead_tokens=(\n"
                            "                            0\n"
                            "                            if nrl_specdec_scheduler_gate_threshold > 0\n"
                            "                            and max(len(self.running), len(num_scheduled_tokens) + 1)\n"
                            "                            > nrl_specdec_scheduler_gate_threshold\n"
                            "                            or nrl_specdec_scheduler_gate_token_threshold > 0\n"
                            "                            and sum(num_scheduled_tokens.values()) + num_new_tokens\n"
                            "                            > nrl_specdec_scheduler_gate_token_threshold\n"
                            "                            else self.num_lookahead_tokens\n"
                            "                        ),\n"
                            "                    )\n",
                        ),
                        (
                            "                effective_lookahead_tokens = (\n"
                            "                    0 if limit_lookahead_tokens else self.num_lookahead_tokens\n"
                            "                )\n",
                            "                effective_lookahead_tokens = (\n"
                            "                    0\n"
                            "                    if limit_lookahead_tokens\n"
                            "                    or (\n"
                            "                        nrl_specdec_scheduler_gate_threshold > 0\n"
                            "                        and max(len(self.running), len(num_scheduled_tokens) + 1)\n"
                            "                        > nrl_specdec_scheduler_gate_threshold\n"
                            "                    )\n"
                            "                    or (\n"
                            "                        nrl_specdec_scheduler_gate_token_threshold > 0\n"
                            "                        and sum(num_scheduled_tokens.values()) + num_new_tokens\n"
                            "                        > nrl_specdec_scheduler_gate_token_threshold\n"
                            "                    )\n"
                            "                    else self.num_lookahead_tokens\n"
                            "                )\n",
                        ),
                    ],
                    [
                        (
                            "        # Spec decode-related.\n"
                            "        scheduled_spec_decode_tokens: dict[str, list[int]] = {}\n",
                            "        # Spec decode-related.\n"
                            "        scheduled_spec_decode_tokens: dict[str, list[int]] = {}\n"
                            "        # NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V5\n"
                            "        if self.num_lookahead_tokens > 0:\n"
                            "            nrl_specdec_scheduler_gate_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_scheduler_gate_threshold\", None\n"
                            "            )\n"
                            "            if nrl_specdec_scheduler_gate_threshold is None:\n"
                            "                nrl_specdec_scheduler_gate_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                nrl_specdec_scheduler_gate_threshold = (\n"
                            "                    int(nrl_specdec_scheduler_gate_threshold_str)\n"
                            "                    if nrl_specdec_scheduler_gate_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_scheduler_gate_threshold = nrl_specdec_scheduler_gate_threshold\n"
                            "            nrl_specdec_scheduler_gate_token_threshold = getattr(\n"
                            "                self, \"_nrl_specdec_scheduler_gate_token_threshold\", None\n"
                            "            )\n"
                            "            if nrl_specdec_scheduler_gate_token_threshold is None:\n"
                            "                nrl_specdec_scheduler_gate_token_threshold_str = __import__(\"os\").environ.get(\n"
                            "                    \"VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD\", \"0\"\n"
                            "                )\n"
                            "                nrl_specdec_scheduler_gate_token_threshold = (\n"
                            "                    int(nrl_specdec_scheduler_gate_token_threshold_str)\n"
                            "                    if nrl_specdec_scheduler_gate_token_threshold_str.isdigit()\n"
                            "                    else 0\n"
                            "                )\n"
                            "                self._nrl_specdec_scheduler_gate_token_threshold = nrl_specdec_scheduler_gate_token_threshold\n"
                            "        else:\n"
                            "            nrl_specdec_scheduler_gate_threshold = 0\n"
                            "            nrl_specdec_scheduler_gate_token_threshold = 0\n",
                        ),
                        (
                            "                    new_blocks = self.kv_cache_manager.allocate_slots(\n"
                            "                        request,\n"
                            "                        num_new_tokens,\n"
                            "                        num_lookahead_tokens=self.num_lookahead_tokens,\n"
                            "                    )\n",
                            "                    new_blocks = self.kv_cache_manager.allocate_slots(\n"
                            "                        request,\n"
                            "                        num_new_tokens,\n"
                            "                        num_lookahead_tokens=(\n"
                            "                            0\n"
                            "                            if nrl_specdec_scheduler_gate_threshold > 0\n"
                            "                            and max(len(self.running), len(num_scheduled_tokens) + 1)\n"
                            "                            > nrl_specdec_scheduler_gate_threshold\n"
                            "                            or nrl_specdec_scheduler_gate_token_threshold > 0\n"
                            "                            and sum(num_scheduled_tokens.values()) + num_new_tokens\n"
                            "                            > nrl_specdec_scheduler_gate_token_threshold\n"
                            "                            else self.num_lookahead_tokens\n"
                            "                        ),\n"
                            "                    )\n",
                        ),
                    ],
                ],
            )

            return applied

        def _patch_vllm_adaptive_specdec_gate():
            """Add an adaptive controller to the already-installed batch gate.

            The V4 gate is a static long-tail guard. This patch keeps the same
            correctness boundary, but lets the scheduler lookahead gate and the
            model runner proposal gate tune thresholds toward a target
            enabled-ratio. The controllers are deliberately local to each vLLM
            worker process: if the adaptive controller is disabled, the static
            V4/V5 behavior is unchanged.
            """

            mode = os.environ.get("VLLM_SPECDEC_ADAPTIVE_GATE_MODE", "off").lower()
            dynamic_mode = os.environ.get(
                "VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS", "0"
            ).lower()
            dynamic_requested = dynamic_mode in {"1", "true", "yes", "y", "on"}
            if mode in {"", "0", "off", "false", "no"} and not dynamic_requested:
                return 0

            applied = 0
            gpu_model_runner = _get_vllm_file("v1/worker/gpu_model_runner.py")
            with open(gpu_model_runner) as f:
                content = f.read()

            runner_required_markers = [
                "NRL_SPECDEC_ADAPTIVE_GATE_PATCH_V1",
                "VLLM_SPECDEC_ADAPTIVE_GATE_MODE",
                "VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO",
                "_nrl_specdec_adaptive_request_threshold",
                "_nrl_specdec_adaptive_token_threshold",
                "_nrl_specdec_adaptive_window_checked",
                "_nrl_specdec_adaptive_window_enabled",
                "NRL SpecDec adaptive gate:",
            ]
            if "NRL_SPECDEC_ADAPTIVE_GATE_PATCH_V1" not in content:
                if "NRL_SPECDEC_BATCH_GATE_PATCH_V4" not in content:
                    raise RuntimeError(
                        "Adaptive SpecDec gate requires the V4 batch gate to be "
                        f"installed first in {gpu_model_runner}."
                    )

                init_anchor = "            specdec_batch_gate_threshold_disabled = (\n"
                init_block = (
                    "            # NRL_SPECDEC_ADAPTIVE_GATE_PATCH_V1\n"
                    "            specdec_adaptive_gate_mode = getattr(\n"
                    "                self, \"_nrl_specdec_adaptive_gate_mode\", None\n"
                    "            )\n"
                    "            if specdec_adaptive_gate_mode is None:\n"
                    "                _nrl_os = __import__(\"os\")\n"
                    "                specdec_adaptive_gate_mode = _nrl_os.environ.get(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_GATE_MODE\", \"off\"\n"
                    "                ).lower()\n"
                    "                self._nrl_specdec_adaptive_gate_mode = specdec_adaptive_gate_mode\n"
                    "                def _nrl_adaptive_int(name, default):\n"
                    "                    value = _nrl_os.environ.get(name, \"\")\n"
                    "                    return int(value) if value.isdigit() else default\n"
                    "                def _nrl_adaptive_float(name, default):\n"
                    "                    try:\n"
                    "                        return float(_nrl_os.environ.get(name, str(default)))\n"
                    "                    except ValueError:\n"
                    "                        return default\n"
                    "                self._nrl_specdec_adaptive_target_enabled_ratio = _nrl_adaptive_float(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO\", 0.35\n"
                    "                )\n"
                    "                self._nrl_specdec_adaptive_hysteresis = _nrl_adaptive_float(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_HYSTERESIS\", 0.05\n"
                    "                )\n"
                    "                self._nrl_specdec_adaptive_interval = max(\n"
                    "                    1,\n"
                    "                    _nrl_adaptive_int(\"VLLM_SPECDEC_ADAPTIVE_ADJUST_INTERVAL\", 512),\n"
                    "                )\n"
                    "                self._nrl_specdec_adaptive_request_step = max(\n"
                    "                    1,\n"
                    "                    _nrl_adaptive_int(\"VLLM_SPECDEC_ADAPTIVE_REQUEST_STEP\", 4),\n"
                    "                )\n"
                    "                self._nrl_specdec_adaptive_token_step = max(\n"
                    "                    1,\n"
                    "                    _nrl_adaptive_int(\"VLLM_SPECDEC_ADAPTIVE_TOKEN_STEP\", 256),\n"
                    "                )\n"
                    "                min_request = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_MIN_REQUEST_THRESHOLD\", 1\n"
                    "                )\n"
                    "                max_request = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_MAX_REQUEST_THRESHOLD\", 128\n"
                    "                )\n"
                    "                min_token = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_MIN_TOKEN_THRESHOLD\", 256\n"
                    "                )\n"
                    "                max_token = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_MAX_TOKEN_THRESHOLD\", 8192\n"
                    "                )\n"
                    "                self._nrl_specdec_adaptive_min_request_threshold = min_request\n"
                    "                self._nrl_specdec_adaptive_max_request_threshold = max_request\n"
                    "                self._nrl_specdec_adaptive_min_token_threshold = min_token\n"
                    "                self._nrl_specdec_adaptive_max_token_threshold = max_token\n"
                    "                initial_request = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_INITIAL_REQUEST_THRESHOLD\",\n"
                    "                    specdec_batch_gate_threshold,\n"
                    "                )\n"
                    "                initial_token = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_INITIAL_TOKEN_THRESHOLD\",\n"
                    "                    specdec_batch_gate_token_threshold\n"
                    "                    if specdec_batch_gate_token_threshold > 0\n"
                    "                    else 0,\n"
                    "                )\n"
                    "                if initial_request > 0:\n"
                    "                    initial_request = min(max(initial_request, min_request), max_request)\n"
                    "                if initial_token > 0:\n"
                    "                    initial_token = min(max(initial_token, min_token), max_token)\n"
                    "                self._nrl_specdec_adaptive_request_threshold = initial_request\n"
                    "                self._nrl_specdec_adaptive_token_threshold = initial_token\n"
                    "                self._nrl_specdec_adaptive_window_checked = 0\n"
                    "                self._nrl_specdec_adaptive_window_enabled = 0\n"
                    "            if specdec_adaptive_gate_mode not in {\"\", \"0\", \"off\", \"false\", \"no\"}:\n"
                    "                specdec_batch_gate_threshold = getattr(\n"
                    "                    self,\n"
                    "                    \"_nrl_specdec_adaptive_request_threshold\",\n"
                    "                    specdec_batch_gate_threshold,\n"
                    "                )\n"
                    "                specdec_batch_gate_token_threshold = getattr(\n"
                    "                    self,\n"
                    "                    \"_nrl_specdec_adaptive_token_threshold\",\n"
                    "                    specdec_batch_gate_token_threshold,\n"
                    "                )\n"
                )

                adapt_anchor = "            specdec_batch_gate_log_interval = getattr(\n"
                adapt_block = (
                    "            if specdec_adaptive_gate_mode not in {\"\", \"0\", \"off\", \"false\", \"no\"}:\n"
                    "                adaptive_window_checked = getattr(\n"
                    "                    self, \"_nrl_specdec_adaptive_window_checked\", 0\n"
                    "                ) + 1\n"
                    "                adaptive_window_enabled = getattr(\n"
                    "                    self, \"_nrl_specdec_adaptive_window_enabled\", 0\n"
                    "                ) + (0 if specdec_batch_gate_disabled else 1)\n"
                    "                adaptive_interval = getattr(\n"
                    "                    self, \"_nrl_specdec_adaptive_interval\", 512\n"
                    "                )\n"
                    "                if adaptive_window_checked >= adaptive_interval:\n"
                    "                    adaptive_enabled_ratio = adaptive_window_enabled / max(\n"
                    "                        1, adaptive_window_checked\n"
                    "                    )\n"
                    "                    adaptive_target = getattr(\n"
                    "                        self, \"_nrl_specdec_adaptive_target_enabled_ratio\", 0.35\n"
                    "                    )\n"
                    "                    adaptive_hysteresis = getattr(\n"
                    "                        self, \"_nrl_specdec_adaptive_hysteresis\", 0.05\n"
                    "                    )\n"
                    "                    request_threshold = getattr(\n"
                    "                        self, \"_nrl_specdec_adaptive_request_threshold\", 0\n"
                    "                    )\n"
                    "                    token_threshold = getattr(\n"
                    "                        self, \"_nrl_specdec_adaptive_token_threshold\", 0\n"
                    "                    )\n"
                    "                    if adaptive_enabled_ratio > adaptive_target + adaptive_hysteresis:\n"
                    "                        if request_threshold > 0:\n"
                    "                            request_threshold = max(\n"
                    "                                getattr(self, \"_nrl_specdec_adaptive_min_request_threshold\", 1),\n"
                    "                                request_threshold\n"
                    "                                - getattr(self, \"_nrl_specdec_adaptive_request_step\", 4),\n"
                    "                            )\n"
                    "                        if token_threshold > 0:\n"
                    "                            token_threshold = max(\n"
                    "                                getattr(self, \"_nrl_specdec_adaptive_min_token_threshold\", 256),\n"
                    "                                token_threshold\n"
                    "                                - getattr(self, \"_nrl_specdec_adaptive_token_step\", 256),\n"
                    "                            )\n"
                    "                    elif adaptive_enabled_ratio < adaptive_target - adaptive_hysteresis:\n"
                    "                        if request_threshold > 0:\n"
                    "                            request_threshold = min(\n"
                    "                                getattr(self, \"_nrl_specdec_adaptive_max_request_threshold\", 128),\n"
                    "                                request_threshold\n"
                    "                                + getattr(self, \"_nrl_specdec_adaptive_request_step\", 4),\n"
                    "                            )\n"
                    "                        if token_threshold > 0:\n"
                    "                            token_threshold = min(\n"
                    "                                getattr(self, \"_nrl_specdec_adaptive_max_token_threshold\", 8192),\n"
                    "                                token_threshold\n"
                    "                                + getattr(self, \"_nrl_specdec_adaptive_token_step\", 256),\n"
                    "                            )\n"
                    "                    self._nrl_specdec_adaptive_request_threshold = request_threshold\n"
                    "                    self._nrl_specdec_adaptive_token_threshold = token_threshold\n"
                    "                    self._nrl_specdec_adaptive_last_enabled_ratio = adaptive_enabled_ratio\n"
                    "                    self._nrl_specdec_adaptive_window_checked = 0\n"
                    "                    self._nrl_specdec_adaptive_window_enabled = 0\n"
                    "                    try:\n"
                    "                        logger.info(\n"
                    "                            \"NRL SpecDec adaptive gate: mode=%s enabled_ratio=%.4f target=%.4f request_threshold=%s token_threshold=%s\",\n"
                    "                            specdec_adaptive_gate_mode,\n"
                    "                            adaptive_enabled_ratio,\n"
                    "                            adaptive_target,\n"
                    "                            request_threshold,\n"
                    "                            token_threshold,\n"
                    "                        )\n"
                    "                    except Exception:\n"
                    "                        pass\n"
                    "                else:\n"
                    "                    self._nrl_specdec_adaptive_window_checked = adaptive_window_checked\n"
                    "                    self._nrl_specdec_adaptive_window_enabled = adaptive_window_enabled\n"
                )

                patched = content.replace(init_anchor, init_block + init_anchor, 1)
                patched = patched.replace(adapt_anchor, adapt_block + adapt_anchor, 1)
                if patched == content:
                    raise RuntimeError(
                        "Could not install adaptive SpecDec gate in "
                        f"{gpu_model_runner}; expected V4 gate anchors were missing."
                    )
                if "NRL_SPECDEC_ADAPTIVE_GATE_PATCH_V1" not in patched:
                    raise RuntimeError(
                        "Adaptive SpecDec gate patch did not leave the required marker "
                        f"in {gpu_model_runner}."
                    )
                missing_markers = [
                    marker for marker in runner_required_markers if marker not in patched
                ]
                if missing_markers:
                    raise RuntimeError(
                        "Adaptive SpecDec gate runner patch installed only partially "
                        f"in {gpu_model_runner}; missing markers: "
                        + ", ".join(missing_markers)
                    )
                with open(gpu_model_runner, "w") as f:
                    f.write(patched)
                applied += 1
            else:
                missing_markers = [
                    marker for marker in runner_required_markers if marker not in content
                ]
                if missing_markers:
                    raise RuntimeError(
                        "Found a partial vLLM adaptive SpecDec runner-gate patch "
                        f"in {gpu_model_runner}; missing markers: "
                        + ", ".join(missing_markers)
                    )
            __import__("py_compile").compile(gpu_model_runner, doraise=True)

            scheduler = _get_vllm_file("v1/core/sched/scheduler.py")
            with open(scheduler) as f:
                scheduler_content = f.read()

            def _has_scheduler_adaptive_lookahead_call(text):
                return (
                    "num_lookahead_tokens=_nrl_specdec_scheduler_lookahead_tokens("
                    in text
                    or "else _nrl_specdec_scheduler_lookahead_tokens(" in text
                )

            def _assert_scheduler_dynamic_request_id_arity(text, path):
                # NRL_SPECDEC_SCHEDULER_REQUEST_ID_ARITY_GUARD_V1
                if "NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_BY_REQUEST_V1" not in text:
                    return
                expected_signature = (
                    "def _nrl_specdec_scheduler_lookahead_tokens("
                    "request_id, num_requests, num_tokens):"
                )
                if expected_signature not in text:
                    raise RuntimeError(
                        "Per-request dynamic SpecDec scheduler patch in "
                        f"{path} is missing the 3-argument lookahead helper "
                        "signature with request_id."
                    )
                bad_requestless_call = re.search(
                    r"_nrl_specdec_scheduler_lookahead_tokens\(\s*\n"
                    r"\s*len\(num_scheduled_tokens\)\s*\+\s*1\s*,",
                    text,
                )
                if bad_requestless_call:
                    raise RuntimeError(
                        "Per-request dynamic SpecDec scheduler patch in "
                        f"{path} still contains a request-less lookahead call."
                    )
                good_request_call = re.search(
                    r"_nrl_specdec_scheduler_lookahead_tokens\(\s*\n"
                    r"\s*request\.request_id\s*,\s*\n"
                    r"\s*len\(num_scheduled_tokens\)\s*\+\s*1\s*,",
                    text,
                )
                if not good_request_call:
                    raise RuntimeError(
                        "Per-request dynamic SpecDec scheduler patch in "
                        f"{path} has no verified request_id lookahead call."
                    )

            dynamic_scheduler_lookahead_block = (
                "            # NRL_SPECDEC_SCHEDULER_DYNAMIC_DRAFT_CAP_PATCH_V1\n"
                "            effective_lookahead_tokens = 0 if disabled else self.num_lookahead_tokens\n"
                "            dynamic_enabled = getattr(\n"
                "                self, \"_nrl_specdec_scheduler_dynamic_draft_tokens_enabled\", None\n"
                "            )\n"
                "            if dynamic_enabled is None:\n"
                "                _nrl_os = __import__(\"os\")\n"
                "                dynamic_enabled = _nrl_os.environ.get(\n"
                "                    \"VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS\", \"0\"\n"
                "                ).lower() in {\"1\", \"true\", \"yes\", \"y\", \"on\"}\n"
                "                self._nrl_specdec_scheduler_dynamic_draft_tokens_enabled = dynamic_enabled\n"
                "                def _nrl_dynamic_int(name, default):\n"
                "                    value = _nrl_os.environ.get(name, \"\")\n"
                "                    return int(value) if value.isdigit() else default\n"
                "                self._nrl_specdec_scheduler_dynamic_small_request_threshold = _nrl_dynamic_int(\n"
                "                    \"VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD\", 4\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_medium_request_threshold = _nrl_dynamic_int(\n"
                "                    \"VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD\", 16\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_small_token_threshold = _nrl_dynamic_int(\n"
                "                    \"VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD\", 0\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_medium_token_threshold = _nrl_dynamic_int(\n"
                "                    \"VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD\", 0\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_small_tokens = _nrl_dynamic_int(\n"
                "                    \"VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS\", self.num_lookahead_tokens\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_medium_tokens = _nrl_dynamic_int(\n"
                "                    \"VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS\", min(self.num_lookahead_tokens, 2)\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_large_tokens = _nrl_dynamic_int(\n"
                "                    \"VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS\", 1\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_small_selected_count = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_small_selected_count\", 0\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_medium_selected_count = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_medium_selected_count\", 0\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_large_selected_count = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_large_selected_count\", 0\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_small_selected_token_count = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_small_selected_token_count\", 0\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_medium_selected_token_count = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_medium_selected_token_count\", 0\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_large_selected_token_count = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_large_selected_token_count\", 0\n"
                "                )\n"
                "                for _nrl_pos_idx in range(1, 9):\n"
                "                    _nrl_pos_name = f\"_nrl_specdec_scheduler_dynamic_pos{_nrl_pos_idx}_selected_count\"\n"
                "                    setattr(self, _nrl_pos_name, getattr(self, _nrl_pos_name, 0))\n"
                "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_last_selected_tier\", \"\"\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_selected_by_request = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_selected_by_request\", {}\n"
                "                )\n"
                "            # NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_BY_REQUEST_V1\n"
                "            selected_by_request = getattr(\n"
                "                self, \"_nrl_specdec_scheduler_dynamic_selected_by_request\", {}\n"
                "            )\n"
                "            if not isinstance(selected_by_request, dict):\n"
                "                selected_by_request = {}\n"
                "            self._nrl_specdec_scheduler_dynamic_selected_by_request = selected_by_request\n"
                "            request_key = str(request_id)\n"
                "            if dynamic_enabled and effective_lookahead_tokens > 0:\n"
                "                small_request_threshold = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_small_request_threshold\", 4\n"
                "                )\n"
                "                medium_request_threshold = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_medium_request_threshold\", 16\n"
                "                )\n"
                "                small_token_threshold = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_small_token_threshold\", 0\n"
                "                )\n"
                "                medium_token_threshold = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_medium_token_threshold\", 0\n"
                "                )\n"
                "                selected_tokens = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_large_tokens\", 1\n"
                "                )\n"
                "                selected_tier = \"large\"\n"
                "                if active_requests <= small_request_threshold and (\n"
                "                    small_token_threshold <= 0 or num_tokens <= small_token_threshold\n"
                "                ):\n"
                "                    selected_tokens = getattr(\n"
                "                        self,\n"
                "                        \"_nrl_specdec_scheduler_dynamic_small_tokens\",\n"
                "                        self.num_lookahead_tokens,\n"
                "                    )\n"
                "                    selected_tier = \"small\"\n"
                "                elif active_requests <= medium_request_threshold and (\n"
                "                    medium_token_threshold <= 0 or num_tokens <= medium_token_threshold\n"
                "                ):\n"
                "                    selected_tokens = getattr(\n"
                "                        self,\n"
                "                        \"_nrl_specdec_scheduler_dynamic_medium_tokens\",\n"
                "                        min(self.num_lookahead_tokens, 2),\n"
                "                    )\n"
                "                    selected_tier = \"medium\"\n"
                "                effective_lookahead_tokens = min(\n"
                "                    self.num_lookahead_tokens,\n"
                "                    max(1, int(selected_tokens)),\n"
                "                )\n"
                "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = selected_tier\n"
                "                selected_by_request[request_key] = (\n"
                "                    effective_lookahead_tokens,\n"
                "                    selected_tier,\n"
                "                )\n"
                "            else:\n"
                "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = \"\"\n"
                "                selected_by_request.pop(request_key, None)\n"
                "            self._nrl_specdec_scheduler_dynamic_last_selected_tokens = effective_lookahead_tokens\n"
            )

            scheduler_required_markers = [
                "NRL_SPECDEC_SCHEDULER_ADAPTIVE_GATE_PATCH_V1",
                "NRL_SPECDEC_SCHEDULER_DYNAMIC_DRAFT_CAP_PATCH_V1",
                "NRL_SPECDEC_SCHEDULER_DYNAMIC_UPDATE_DRAFT_CAP_PATCH_V1",
                "def _nrl_specdec_scheduler_lookahead_tokens",
                "VLLM_SPECDEC_ADAPTIVE_GATE_MODE",
                "VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO",
                "VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS",
                "_nrl_specdec_scheduler_adaptive_request_threshold",
                "_nrl_specdec_scheduler_adaptive_token_threshold",
                "_nrl_specdec_scheduler_adaptive_window_checked",
                "_nrl_specdec_scheduler_gate_last_active_requests",
                "active_requests = max(num_requests, len(self.running))",
                "_nrl_specdec_scheduler_gate_effective_lookahead_tokens",
                "_nrl_specdec_scheduler_dynamic_last_selected_tokens",
                "_nrl_specdec_scheduler_dynamic_last_selected_tier",
                "_nrl_specdec_scheduler_dynamic_selected_by_request",
                "_nrl_specdec_scheduler_dynamic_last_stored_tokens",
                "_nrl_specdec_scheduler_dynamic_small_selected_count",
                "_nrl_specdec_scheduler_dynamic_small_selected_token_count",
                "_nrl_specdec_scheduler_dynamic_pos1_selected_count",
                "NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_COUNTERS_ON_STORE_V1",
                "NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_BY_REQUEST_V1",
                "NRL SpecDec scheduler adaptive gate:",
                "return effective_lookahead_tokens",
            ]
            update_draft_cap_anchors = [
                "            # Add newly generated spec token ids to the request.\n"
                "            if self.structured_output_manager.should_advance(request):\n"
                "                metadata = request.structured_output_request\n"
                "                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]\n"
                "                    spec_token_ids\n"
                "                )\n"
                "            else:\n"
                "                request.spec_token_ids = spec_token_ids\n",
                "            # Add newly generated spec token ids to the request.\n"
                "            if self.structured_output_manager.should_advance(request):\n"
                "                metadata = request.structured_output_request\n"
                "                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]\n"
                "            request.spec_token_ids = spec_token_ids\n",
                "            # Add newly generated spec token ids to the request.\n"
                "            if self.structured_output_manager.should_advance(request):\n"
                "                metadata = request.structured_output_request\n"
                "                assert metadata is not None and metadata.grammar is not None\n"
                "                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)\n"
                "            request.spec_token_ids = spec_token_ids\n",
                "            # Add newly generated spec token ids to the request.\n"
                "            if self.structured_output_manager.should_advance(request):\n"
                "                metadata = request.structured_output_request\n"
                "                assert metadata is not None\n"
                "                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]\n"
                "            request.spec_token_ids = spec_token_ids\n",
            ]
            old_update_draft_cap_block = (
                "            # Add newly generated spec token ids to the request.\n"
                "            # NRL_SPECDEC_SCHEDULER_DYNAMIC_UPDATE_DRAFT_CAP_PATCH_V1\n"
                "            dynamic_draft_cap = getattr(\n"
                "                self,\n"
                "                \"_nrl_specdec_scheduler_dynamic_last_selected_tokens\",\n"
                "                self.num_spec_tokens,\n"
                "            )\n"
                "            try:\n"
                "                dynamic_draft_cap = int(dynamic_draft_cap)\n"
                "            except (TypeError, ValueError):\n"
                "                dynamic_draft_cap = self.num_spec_tokens\n"
                "            dynamic_draft_cap = max(0, min(self.num_spec_tokens, dynamic_draft_cap))\n"
                "            if dynamic_draft_cap == 0:\n"
                "                spec_token_ids = []\n"
                "            elif len(spec_token_ids) > dynamic_draft_cap:\n"
                "                spec_token_ids = list(spec_token_ids[:dynamic_draft_cap])\n"
                "            else:\n"
                "                spec_token_ids = list(spec_token_ids)\n"
                "            self._nrl_specdec_scheduler_dynamic_last_stored_tokens = len(spec_token_ids)\n"
                "            if self.structured_output_manager.should_advance(request):\n"
                "                metadata = request.structured_output_request\n"
                "                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]\n"
                "            request.spec_token_ids = spec_token_ids\n"
            )
            update_draft_cap_block = (
                "            # Add newly generated spec token ids to the request.\n"
                "            # NRL_SPECDEC_SCHEDULER_DYNAMIC_UPDATE_DRAFT_CAP_PATCH_V1\n"
                "            selected_by_request = getattr(\n"
                "                self, \"_nrl_specdec_scheduler_dynamic_selected_by_request\", {}\n"
                "            )\n"
                "            if not isinstance(selected_by_request, dict):\n"
                "                selected_by_request = {}\n"
                "            request_key = str(getattr(request, \"request_id\", \"\"))\n"
                "            dynamic_selection = None\n"
                "            if request_key:\n"
                "                dynamic_selection = selected_by_request.pop(request_key, None)\n"
                "            if (\n"
                "                isinstance(dynamic_selection, (tuple, list))\n"
                "                and len(dynamic_selection) >= 2\n"
                "            ):\n"
                "                dynamic_draft_cap = dynamic_selection[0]\n"
                "                dynamic_tier = dynamic_selection[1]\n"
                "            else:\n"
                "                dynamic_draft_cap = getattr(\n"
                "                    self,\n"
                "                    \"_nrl_specdec_scheduler_dynamic_last_selected_tokens\",\n"
                "                    self.num_spec_tokens,\n"
                "                )\n"
                "                dynamic_tier = getattr(\n"
                "                    self, \"_nrl_specdec_scheduler_dynamic_last_selected_tier\", \"\"\n"
                "                )\n"
                "            self._nrl_specdec_scheduler_dynamic_selected_by_request = selected_by_request\n"
                "            try:\n"
                "                dynamic_draft_cap = int(dynamic_draft_cap)\n"
                "            except (TypeError, ValueError):\n"
                "                dynamic_draft_cap = self.num_spec_tokens\n"
                "            dynamic_draft_cap = max(0, min(self.num_spec_tokens, dynamic_draft_cap))\n"
                "            if dynamic_draft_cap == 0:\n"
                "                spec_token_ids = []\n"
                "            elif len(spec_token_ids) > dynamic_draft_cap:\n"
                "                spec_token_ids = list(spec_token_ids[:dynamic_draft_cap])\n"
                "            else:\n"
                "                spec_token_ids = list(spec_token_ids)\n"
                "            if self.structured_output_manager.should_advance(request):\n"
                "                metadata = request.structured_output_request\n"
                "                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]\n"
                "            request.spec_token_ids = spec_token_ids\n"
                "            # NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_COUNTERS_ON_STORE_V1\n"
                "            self._nrl_specdec_scheduler_dynamic_last_stored_tokens = len(spec_token_ids)\n"
                "            if dynamic_tier in {\"small\", \"medium\", \"large\"} and len(spec_token_ids) > 0:\n"
                "                count_name = f\"_nrl_specdec_scheduler_dynamic_{dynamic_tier}_selected_count\"\n"
                "                setattr(self, count_name, getattr(self, count_name, 0) + 1)\n"
                "                token_count_name = f\"_nrl_specdec_scheduler_dynamic_{dynamic_tier}_selected_token_count\"\n"
                "                setattr(\n"
                "                    self,\n"
                "                    token_count_name,\n"
                "                    getattr(self, token_count_name, 0) + len(spec_token_ids),\n"
                "                )\n"
                "                for _nrl_pos_idx in range(min(len(spec_token_ids), self.num_spec_tokens)):\n"
                "                    pos_count_name = f\"_nrl_specdec_scheduler_dynamic_pos{_nrl_pos_idx + 1}_selected_count\"\n"
                "                    setattr(self, pos_count_name, getattr(self, pos_count_name, 0) + 1)\n"
            )
            if "NRL_SPECDEC_SCHEDULER_ADAPTIVE_GATE_PATCH_V1" not in scheduler_content:
                if "NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V5" not in scheduler_content:
                    raise RuntimeError(
                        "Adaptive SpecDec scheduler gate requires the V5 "
                        f"lookahead gate to be installed first in {scheduler}."
                    )

                helper_anchor = (
                    "        else:\n"
                    "            nrl_specdec_scheduler_gate_threshold = 0\n"
                    "            nrl_specdec_scheduler_gate_token_threshold = 0\n"
                )
                helper_block = (
                    "        # NRL_SPECDEC_SCHEDULER_ADAPTIVE_GATE_PATCH_V1\n"
                    "        def _nrl_specdec_scheduler_lookahead_tokens(request_id, num_requests, num_tokens):\n"
                    "            active_requests = max(num_requests, len(self.running))\n"
                    "            request_threshold = nrl_specdec_scheduler_gate_threshold\n"
                    "            token_threshold = nrl_specdec_scheduler_gate_token_threshold\n"
                    "            adaptive_mode = getattr(\n"
                    "                self, \"_nrl_specdec_scheduler_adaptive_mode\", None\n"
                    "            )\n"
                    "            if adaptive_mode is None:\n"
                    "                _nrl_os = __import__(\"os\")\n"
                    "                adaptive_mode = _nrl_os.environ.get(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_GATE_MODE\", \"off\"\n"
                    "                ).lower()\n"
                    "                self._nrl_specdec_scheduler_adaptive_mode = adaptive_mode\n"
                    "                def _nrl_adaptive_int(name, default):\n"
                    "                    value = _nrl_os.environ.get(name, \"\")\n"
                    "                    return int(value) if value.isdigit() else default\n"
                    "                def _nrl_adaptive_float(name, default):\n"
                    "                    try:\n"
                    "                        return float(_nrl_os.environ.get(name, str(default)))\n"
                    "                    except ValueError:\n"
                    "                        return default\n"
                    "                self._nrl_specdec_scheduler_adaptive_target_enabled_ratio = _nrl_adaptive_float(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO\", 0.35\n"
                    "                )\n"
                    "                self._nrl_specdec_scheduler_adaptive_hysteresis = _nrl_adaptive_float(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_HYSTERESIS\", 0.05\n"
                    "                )\n"
                    "                self._nrl_specdec_scheduler_adaptive_interval = max(\n"
                    "                    1,\n"
                    "                    _nrl_adaptive_int(\"VLLM_SPECDEC_ADAPTIVE_ADJUST_INTERVAL\", 512),\n"
                    "                )\n"
                    "                self._nrl_specdec_scheduler_adaptive_request_step = max(\n"
                    "                    1,\n"
                    "                    _nrl_adaptive_int(\"VLLM_SPECDEC_ADAPTIVE_REQUEST_STEP\", 4),\n"
                    "                )\n"
                    "                self._nrl_specdec_scheduler_adaptive_token_step = max(\n"
                    "                    1,\n"
                    "                    _nrl_adaptive_int(\"VLLM_SPECDEC_ADAPTIVE_TOKEN_STEP\", 256),\n"
                    "                )\n"
                    "                self._nrl_specdec_scheduler_adaptive_min_request_threshold = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_MIN_REQUEST_THRESHOLD\", 1\n"
                    "                )\n"
                    "                self._nrl_specdec_scheduler_adaptive_max_request_threshold = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_MAX_REQUEST_THRESHOLD\", 128\n"
                    "                )\n"
                    "                self._nrl_specdec_scheduler_adaptive_min_token_threshold = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_MIN_TOKEN_THRESHOLD\", 256\n"
                    "                )\n"
                    "                self._nrl_specdec_scheduler_adaptive_max_token_threshold = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_MAX_TOKEN_THRESHOLD\", 8192\n"
                    "                )\n"
                    "                initial_request = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_INITIAL_REQUEST_THRESHOLD\",\n"
                    "                    request_threshold,\n"
                    "                )\n"
                    "                initial_token = _nrl_adaptive_int(\n"
                    "                    \"VLLM_SPECDEC_ADAPTIVE_INITIAL_TOKEN_THRESHOLD\",\n"
                    "                    token_threshold,\n"
                    "                )\n"
                    "                if initial_request > 0:\n"
                    "                    initial_request = min(\n"
                    "                        max(\n"
                    "                            initial_request,\n"
                    "                            self._nrl_specdec_scheduler_adaptive_min_request_threshold,\n"
                    "                        ),\n"
                    "                        self._nrl_specdec_scheduler_adaptive_max_request_threshold,\n"
                    "                    )\n"
                    "                if initial_token > 0:\n"
                    "                    initial_token = min(\n"
                    "                        max(\n"
                    "                            initial_token,\n"
                    "                            self._nrl_specdec_scheduler_adaptive_min_token_threshold,\n"
                    "                        ),\n"
                    "                        self._nrl_specdec_scheduler_adaptive_max_token_threshold,\n"
                    "                    )\n"
                    "                self._nrl_specdec_scheduler_adaptive_request_threshold = initial_request\n"
                    "                self._nrl_specdec_scheduler_adaptive_token_threshold = initial_token\n"
                    "                self._nrl_specdec_scheduler_adaptive_window_checked = 0\n"
                    "                self._nrl_specdec_scheduler_adaptive_window_enabled = 0\n"
                    "            if adaptive_mode not in {\"\", \"0\", \"off\", \"false\", \"no\"}:\n"
                    "                request_threshold = getattr(\n"
                    "                    self,\n"
                    "                    \"_nrl_specdec_scheduler_adaptive_request_threshold\",\n"
                    "                    request_threshold,\n"
                    "                )\n"
                    "                token_threshold = getattr(\n"
                    "                    self,\n"
                    "                    \"_nrl_specdec_scheduler_adaptive_token_threshold\",\n"
                    "                    token_threshold,\n"
                    "                )\n"
                    "            disabled = (\n"
                    "                (request_threshold > 0 and active_requests > request_threshold)\n"
                    "                or (token_threshold > 0 and num_tokens > token_threshold)\n"
                    "            )\n"
                    + dynamic_scheduler_lookahead_block
                    + "            self._nrl_specdec_scheduler_gate_last_num_requests = num_requests\n"
                    "            self._nrl_specdec_scheduler_gate_last_active_requests = active_requests\n"
                    "            self._nrl_specdec_scheduler_gate_last_num_tokens = num_tokens\n"
                    "            self._nrl_specdec_scheduler_gate_last_disabled = disabled\n"
                    "            self._nrl_specdec_scheduler_gate_effective_lookahead_tokens = effective_lookahead_tokens\n"
                    "            self._nrl_specdec_scheduler_gate_checked_count = getattr(\n"
                    "                self, \"_nrl_specdec_scheduler_gate_checked_count\", 0\n"
                    "            ) + 1\n"
                    "            if disabled:\n"
                    "                self._nrl_specdec_scheduler_gate_disabled_count = getattr(\n"
                    "                    self, \"_nrl_specdec_scheduler_gate_disabled_count\", 0\n"
                    "                ) + 1\n"
                    "            else:\n"
                    "                self._nrl_specdec_scheduler_gate_enabled_count = getattr(\n"
                    "                    self, \"_nrl_specdec_scheduler_gate_enabled_count\", 0\n"
                    "                ) + 1\n"
                    "            if adaptive_mode not in {\"\", \"0\", \"off\", \"false\", \"no\"}:\n"
                    "                adaptive_window_checked = getattr(\n"
                    "                    self, \"_nrl_specdec_scheduler_adaptive_window_checked\", 0\n"
                    "                ) + 1\n"
                    "                adaptive_window_enabled = getattr(\n"
                    "                    self, \"_nrl_specdec_scheduler_adaptive_window_enabled\", 0\n"
                    "                ) + (0 if disabled else 1)\n"
                    "                adaptive_interval = getattr(\n"
                    "                    self, \"_nrl_specdec_scheduler_adaptive_interval\", 512\n"
                    "                )\n"
                    "                if adaptive_window_checked >= adaptive_interval:\n"
                    "                    adaptive_enabled_ratio = adaptive_window_enabled / max(\n"
                    "                        1, adaptive_window_checked\n"
                    "                    )\n"
                    "                    adaptive_target = getattr(\n"
                    "                        self,\n"
                    "                        \"_nrl_specdec_scheduler_adaptive_target_enabled_ratio\",\n"
                    "                        0.35,\n"
                    "                    )\n"
                    "                    adaptive_hysteresis = getattr(\n"
                    "                        self,\n"
                    "                        \"_nrl_specdec_scheduler_adaptive_hysteresis\",\n"
                    "                        0.05,\n"
                    "                    )\n"
                    "                    if adaptive_enabled_ratio > adaptive_target + adaptive_hysteresis:\n"
                    "                        if request_threshold > 0:\n"
                    "                            request_threshold = max(\n"
                    "                                getattr(\n"
                    "                                    self,\n"
                    "                                    \"_nrl_specdec_scheduler_adaptive_min_request_threshold\",\n"
                    "                                    1,\n"
                    "                                ),\n"
                    "                                request_threshold\n"
                    "                                - getattr(\n"
                    "                                    self,\n"
                    "                                    \"_nrl_specdec_scheduler_adaptive_request_step\",\n"
                    "                                    4,\n"
                    "                                ),\n"
                    "                            )\n"
                    "                        if token_threshold > 0:\n"
                    "                            token_threshold = max(\n"
                    "                                getattr(\n"
                    "                                    self,\n"
                    "                                    \"_nrl_specdec_scheduler_adaptive_min_token_threshold\",\n"
                    "                                    256,\n"
                    "                                ),\n"
                    "                                token_threshold\n"
                    "                                - getattr(\n"
                    "                                    self,\n"
                    "                                    \"_nrl_specdec_scheduler_adaptive_token_step\",\n"
                    "                                    256,\n"
                    "                                ),\n"
                    "                            )\n"
                    "                    elif adaptive_enabled_ratio < adaptive_target - adaptive_hysteresis:\n"
                    "                        if request_threshold > 0:\n"
                    "                            request_threshold = min(\n"
                    "                                getattr(\n"
                    "                                    self,\n"
                    "                                    \"_nrl_specdec_scheduler_adaptive_max_request_threshold\",\n"
                    "                                    128,\n"
                    "                                ),\n"
                    "                                request_threshold\n"
                    "                                + getattr(\n"
                    "                                    self,\n"
                    "                                    \"_nrl_specdec_scheduler_adaptive_request_step\",\n"
                    "                                    4,\n"
                    "                                ),\n"
                    "                            )\n"
                    "                        if token_threshold > 0:\n"
                    "                            token_threshold = min(\n"
                    "                                getattr(\n"
                    "                                    self,\n"
                    "                                    \"_nrl_specdec_scheduler_adaptive_max_token_threshold\",\n"
                    "                                    8192,\n"
                    "                                ),\n"
                    "                                token_threshold\n"
                    "                                + getattr(\n"
                    "                                    self,\n"
                    "                                    \"_nrl_specdec_scheduler_adaptive_token_step\",\n"
                    "                                    256,\n"
                    "                                ),\n"
                    "                            )\n"
                    "                    self._nrl_specdec_scheduler_adaptive_request_threshold = request_threshold\n"
                    "                    self._nrl_specdec_scheduler_adaptive_token_threshold = token_threshold\n"
                    "                    self._nrl_specdec_scheduler_adaptive_last_enabled_ratio = adaptive_enabled_ratio\n"
                    "                    self._nrl_specdec_scheduler_adaptive_window_checked = 0\n"
                    "                    self._nrl_specdec_scheduler_adaptive_window_enabled = 0\n"
                    "                    try:\n"
                    "                        logger.info(\n"
                    "                            \"NRL SpecDec scheduler adaptive gate: mode=%s enabled_ratio=%.4f target=%.4f request_threshold=%s token_threshold=%s active_requests=%s scheduled_requests=%s\",\n"
                    "                            adaptive_mode,\n"
                    "                            adaptive_enabled_ratio,\n"
                    "                            adaptive_target,\n"
                    "                            request_threshold,\n"
                    "                            token_threshold,\n"
                    "                            active_requests,\n"
                    "                            num_requests,\n"
                    "                        )\n"
                    "                    except Exception:\n"
                    "                        pass\n"
                    "                else:\n"
                    "                    self._nrl_specdec_scheduler_adaptive_window_checked = adaptive_window_checked\n"
                    "                    self._nrl_specdec_scheduler_adaptive_window_enabled = adaptive_window_enabled\n"
                    "            return effective_lookahead_tokens\n"
                )

                scheduler_content = scheduler_content.replace(
                    helper_anchor, helper_anchor + helper_block, 1
                )
                lookahead_replacements = [
                    (
                        "                    num_lookahead_tokens=(\n"
                        "                        0\n"
                        "                        if nrl_specdec_scheduler_gate_threshold > 0\n"
                        "                        and max(len(self.running), len(num_scheduled_tokens) + 1)\n"
                        "                        > nrl_specdec_scheduler_gate_threshold\n"
                        "                        or nrl_specdec_scheduler_gate_token_threshold > 0\n"
                        "                        and sum(num_scheduled_tokens.values()) + num_new_tokens\n"
                        "                        > nrl_specdec_scheduler_gate_token_threshold\n"
                        "                        else self.num_lookahead_tokens))\n",
                        "                    num_lookahead_tokens=_nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                        request.request_id,\n"
                        "                        len(num_scheduled_tokens) + 1,\n"
                        "                        sum(num_scheduled_tokens.values()) + num_new_tokens,\n"
                        "                    ))\n",
                    ),
                    (
                        "                        num_lookahead_tokens=(\n"
                        "                            0\n"
                        "                            if nrl_specdec_scheduler_gate_threshold > 0\n"
                        "                            and max(len(self.running), len(num_scheduled_tokens) + 1)\n"
                        "                            > nrl_specdec_scheduler_gate_threshold\n"
                        "                            or nrl_specdec_scheduler_gate_token_threshold > 0\n"
                        "                            and sum(num_scheduled_tokens.values()) + num_new_tokens\n"
                        "                            > nrl_specdec_scheduler_gate_token_threshold\n"
                        "                            else self.num_lookahead_tokens\n"
                        "                        ),\n",
                        "                        num_lookahead_tokens=_nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                            request.request_id,\n"
                        "                            len(num_scheduled_tokens) + 1,\n"
                        "                            sum(num_scheduled_tokens.values()) + num_new_tokens,\n"
                        "                        ),\n",
                    ),
                    (
                        "                effective_lookahead_tokens = (\n"
                        "                    0\n"
                        "                    if request.num_computed_tokens == 0\n"
                        "                    or (\n"
                        "                        nrl_specdec_scheduler_gate_threshold > 0\n"
                        "                        and max(len(self.running), len(num_scheduled_tokens) + 1)\n"
                        "                        > nrl_specdec_scheduler_gate_threshold\n"
                        "                    )\n"
                        "                    or (\n"
                        "                        nrl_specdec_scheduler_gate_token_threshold > 0\n"
                        "                        and sum(num_scheduled_tokens.values()) + num_new_tokens\n"
                        "                        > nrl_specdec_scheduler_gate_token_threshold\n"
                        "                    )\n"
                        "                    else self.num_lookahead_tokens\n"
                        "                )\n",
                        "                effective_lookahead_tokens = (\n"
                        "                    0\n"
                        "                    if request.num_computed_tokens == 0\n"
                        "                    else _nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                        request.request_id,\n"
                        "                        len(num_scheduled_tokens) + 1,\n"
                        "                        sum(num_scheduled_tokens.values()) + num_new_tokens,\n"
                        "                    )\n"
                        "                )\n",
                    ),
                    (
                        "                effective_lookahead_tokens = (\n"
                        "                    0\n"
                        "                    if limit_lookahead_tokens\n"
                        "                    or (\n"
                        "                        nrl_specdec_scheduler_gate_threshold > 0\n"
                        "                        and max(len(self.running), len(num_scheduled_tokens) + 1)\n"
                        "                        > nrl_specdec_scheduler_gate_threshold\n"
                        "                    )\n"
                        "                    or (\n"
                        "                        nrl_specdec_scheduler_gate_token_threshold > 0\n"
                        "                        and sum(num_scheduled_tokens.values()) + num_new_tokens\n"
                        "                        > nrl_specdec_scheduler_gate_token_threshold\n"
                        "                    )\n"
                        "                    else self.num_lookahead_tokens\n"
                        "                )\n",
                        "                effective_lookahead_tokens = (\n"
                        "                    0\n"
                        "                    if limit_lookahead_tokens\n"
                        "                    else _nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                        request.request_id,\n"
                        "                        len(num_scheduled_tokens) + 1,\n"
                        "                        sum(num_scheduled_tokens.values()) + num_new_tokens,\n"
                        "                    )\n"
                        "                )\n",
                    ),
                ]
                replacement_count = 0
                for old, new in lookahead_replacements:
                    if old in scheduler_content:
                        scheduler_content = scheduler_content.replace(old, new)
                        replacement_count += 1
                if (
                    replacement_count == 0
                    or "NRL_SPECDEC_SCHEDULER_ADAPTIVE_GATE_PATCH_V1"
                    not in scheduler_content
                    or "_nrl_specdec_scheduler_lookahead_tokens"
                    not in scheduler_content
                ):
                    raise RuntimeError(
                        "Could not install adaptive SpecDec scheduler gate in "
                        f"{scheduler}; expected V5 gate anchors were missing."
                    )
                update_draft_cap_anchor = next(
                    (
                        anchor
                        for anchor in update_draft_cap_anchors
                        if anchor in scheduler_content
                    ),
                    None,
                )
                if update_draft_cap_anchor is None:
                    raise RuntimeError(
                        "Could not install dynamic SpecDec draft update cap in "
                        f"{scheduler}; expected update_draft_token_ids anchor "
                        "was missing."
                    )
                scheduler_content = scheduler_content.replace(
                    update_draft_cap_anchor,
                    update_draft_cap_block,
                    1,
                )
                with open(scheduler, "w") as f:
                    f.write(scheduler_content)
                applied += 1
            else:
                if (
                    "active_requests = max(num_requests, len(self.running))"
                    not in scheduler_content
                ):
                    upgraded_content = scheduler_content
                    upgraded_content = upgraded_content.replace(
                        "        def _nrl_specdec_scheduler_lookahead_tokens(num_requests, num_tokens):\n"
                        "            request_threshold = nrl_specdec_scheduler_gate_threshold\n",
                        "        def _nrl_specdec_scheduler_lookahead_tokens(num_requests, num_tokens):\n"
                        "            active_requests = max(num_requests, len(self.running))\n"
                        "            request_threshold = nrl_specdec_scheduler_gate_threshold\n",
                    )
                    upgraded_content = upgraded_content.replace(
                        "(request_threshold > 0 and num_requests > request_threshold)",
                        "(request_threshold > 0 and active_requests > request_threshold)",
                    )
                    upgraded_content = upgraded_content.replace(
                        "            self._nrl_specdec_scheduler_gate_last_num_requests = num_requests\n"
                        "            self._nrl_specdec_scheduler_gate_last_num_tokens = num_tokens\n",
                        "            self._nrl_specdec_scheduler_gate_last_num_requests = num_requests\n"
                        "            self._nrl_specdec_scheduler_gate_last_active_requests = active_requests\n"
                        "            self._nrl_specdec_scheduler_gate_last_num_tokens = num_tokens\n",
                    )
                    upgraded_content = upgraded_content.replace(
                        "request_threshold=%s token_threshold=%s num_requests=%s\",\n",
                        "request_threshold=%s token_threshold=%s active_requests=%s scheduled_requests=%s\",\n",
                    )
                    upgraded_content = upgraded_content.replace(
                        "                            num_requests,\n"
                        "                        )\n"
                        "                    except Exception:\n",
                        "                            active_requests,\n"
                        "                            num_requests,\n"
                        "                        )\n"
                        "                    except Exception:\n",
                    )
                    if upgraded_content != scheduler_content:
                        with open(scheduler, "w") as f:
                            f.write(upgraded_content)
                        scheduler_content = upgraded_content
                if "NRL_SPECDEC_SCHEDULER_DYNAMIC_DRAFT_CAP_PATCH_V1" not in scheduler_content:
                    static_effective_block = (
                        "            self._nrl_specdec_scheduler_gate_last_num_requests = num_requests\n"
                        "            self._nrl_specdec_scheduler_gate_last_active_requests = active_requests\n"
                        "            self._nrl_specdec_scheduler_gate_last_num_tokens = num_tokens\n"
                        "            self._nrl_specdec_scheduler_gate_last_disabled = disabled\n"
                        "            self._nrl_specdec_scheduler_gate_effective_lookahead_tokens = (\n"
                        "                0 if disabled else self.num_lookahead_tokens\n"
                        "            )\n"
                    )
                    dynamic_effective_block = (
                        dynamic_scheduler_lookahead_block
                        + "            self._nrl_specdec_scheduler_gate_last_num_requests = num_requests\n"
                        "            self._nrl_specdec_scheduler_gate_last_active_requests = active_requests\n"
                        "            self._nrl_specdec_scheduler_gate_last_num_tokens = num_tokens\n"
                        "            self._nrl_specdec_scheduler_gate_last_disabled = disabled\n"
                        "            self._nrl_specdec_scheduler_gate_effective_lookahead_tokens = effective_lookahead_tokens\n"
                    )
                    static_return = "            return 0 if disabled else self.num_lookahead_tokens\n"
                    if static_effective_block not in scheduler_content or static_return not in scheduler_content:
                        raise RuntimeError(
                            "Could not upgrade existing adaptive SpecDec scheduler "
                            f"gate with dynamic draft-depth cap in {scheduler}; "
                            "expected V1 scheduler anchors were missing."
                        )
                    scheduler_content = scheduler_content.replace(
                        static_effective_block, dynamic_effective_block, 1
                    )
                    scheduler_content = scheduler_content.replace(
                        static_return, "            return effective_lookahead_tokens\n", 1
                    )
                    with open(scheduler, "w") as f:
                        f.write(scheduler_content)
                if (
                    "NRL_SPECDEC_SCHEDULER_DYNAMIC_UPDATE_DRAFT_CAP_PATCH_V1"
                    in scheduler_content
                    and "NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_COUNTERS_ON_STORE_V1"
                    not in scheduler_content
                ):
                    # NRL_SPECDEC_SCHEDULER_STORE_COUNTERS_BEFORE_PER_REQUEST_V1
                    # Upgrade the update_draft_token_ids block before mutating
                    # scalar dynamic-K state into per-request state. Otherwise
                    # the old scalar anchor disappears and this upgrade can get
                    # stuck on a partially patched scheduler.
                    if old_update_draft_cap_block not in scheduler_content:
                        raise RuntimeError(
                            "Could not upgrade existing adaptive SpecDec scheduler "
                            f"with store-time dynamic counters in {scheduler}; "
                            "expected old update_draft_token_ids block was missing."
                        )
                    scheduler_content = scheduler_content.replace(
                        old_update_draft_cap_block,
                        update_draft_cap_block,
                        1,
                    )
                    with open(scheduler, "w") as f:
                        f.write(scheduler_content)
                if (
                    "NRL_SPECDEC_SCHEDULER_DYNAMIC_DRAFT_CAP_PATCH_V1"
                    in scheduler_content
                    and "_nrl_specdec_scheduler_dynamic_small_selected_count"
                    not in scheduler_content
                ):
                    dynamic_selected_count_anchor = (
                        "                self._nrl_specdec_scheduler_dynamic_large_tokens = _nrl_dynamic_int(\n"
                        "                    \"VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS\", 1\n"
                        "                )\n"
                    )
                    dynamic_selected_count_block = (
                        dynamic_selected_count_anchor
                        + "                self._nrl_specdec_scheduler_dynamic_small_selected_count = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_small_selected_count\", 0\n"
                        "                )\n"
                        "                self._nrl_specdec_scheduler_dynamic_medium_selected_count = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_medium_selected_count\", 0\n"
                        "                )\n"
                        "                self._nrl_specdec_scheduler_dynamic_large_selected_count = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_large_selected_count\", 0\n"
                        "                )\n"
                    )
                    if dynamic_selected_count_anchor not in scheduler_content:
                        raise RuntimeError(
                            "Could not upgrade existing adaptive SpecDec scheduler "
                            f"gate dynamic counters in {scheduler}; expected dynamic "
                            "config anchor was missing."
                        )
                    scheduler_content = scheduler_content.replace(
                        dynamic_selected_count_anchor,
                        dynamic_selected_count_block,
                        1,
                    )
                    with open(scheduler, "w") as f:
                        f.write(scheduler_content)
                if (
                    "NRL_SPECDEC_SCHEDULER_DYNAMIC_DRAFT_CAP_PATCH_V1"
                    in scheduler_content
                    and (
                        "_nrl_specdec_scheduler_dynamic_small_selected_token_count"
                        not in scheduler_content
                        or "_nrl_specdec_scheduler_dynamic_pos1_selected_count"
                        not in scheduler_content
                    )
                ):
                    # NRL_SPECDEC_SCHEDULER_DYNAMIC_POS_COUNTERS_PARTIAL_UPGRADE_V1
                    # Some reused Ray/vLLM venvs contain the token-denominator
                    # upgrade but not the per-position selected counters. Repair
                    # that state explicitly instead of treating token counters as
                    # proof that the whole denominator upgrade is present.
                    dynamic_token_count_anchor = (
                        "                self._nrl_specdec_scheduler_dynamic_large_selected_count = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_large_selected_count\", 0\n"
                        "                )\n"
                    )
                    dynamic_token_count_block = (
                        dynamic_token_count_anchor
                        + "                self._nrl_specdec_scheduler_dynamic_small_selected_token_count = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_small_selected_token_count\", 0\n"
                        "                )\n"
                        "                self._nrl_specdec_scheduler_dynamic_medium_selected_token_count = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_medium_selected_token_count\", 0\n"
                        "                )\n"
                        "                self._nrl_specdec_scheduler_dynamic_large_selected_token_count = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_large_selected_token_count\", 0\n"
                        "                )\n"
                        "                for _nrl_pos_idx in range(1, 9):\n"
                        "                    _nrl_pos_name = f\"_nrl_specdec_scheduler_dynamic_pos{_nrl_pos_idx}_selected_count\"\n"
                        "                    setattr(self, _nrl_pos_name, getattr(self, _nrl_pos_name, 0))\n"
                        "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_last_selected_tier\", \"\"\n"
                        "                )\n"
                    )
                    dynamic_pos_count_anchor = (
                        "                self._nrl_specdec_scheduler_dynamic_large_selected_token_count = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_large_selected_token_count\", 0\n"
                        "                )\n"
                    )
                    dynamic_pos_count_block = (
                        dynamic_pos_count_anchor
                        + "                for _nrl_pos_idx in range(1, 9):\n"
                        "                    _nrl_pos_name = f\"_nrl_specdec_scheduler_dynamic_pos{_nrl_pos_idx}_selected_count\"\n"
                        "                    setattr(self, _nrl_pos_name, getattr(self, _nrl_pos_name, 0))\n"
                    )
                    if (
                        "_nrl_specdec_scheduler_dynamic_small_selected_token_count"
                        not in scheduler_content
                    ):
                        if dynamic_token_count_anchor not in scheduler_content:
                            raise RuntimeError(
                                "Could not upgrade existing adaptive SpecDec scheduler "
                                f"gate dynamic token counters in {scheduler}; expected "
                                "selected-count anchor was missing."
                            )
                        scheduler_content = scheduler_content.replace(
                            dynamic_token_count_anchor,
                            dynamic_token_count_block,
                            1,
                        )
                    elif (
                        "_nrl_specdec_scheduler_dynamic_pos1_selected_count"
                        not in scheduler_content
                    ):
                        if dynamic_pos_count_anchor not in scheduler_content:
                            raise RuntimeError(
                                "Could not repair existing adaptive SpecDec scheduler "
                                f"gate dynamic position counters in {scheduler}; "
                                "expected token-count anchor was missing."
                            )
                        scheduler_content = scheduler_content.replace(
                            dynamic_pos_count_anchor,
                            dynamic_pos_count_block,
                            1,
                        )
                    with open(scheduler, "w") as f:
                        f.write(scheduler_content)
                if (
                    "NRL_SPECDEC_SCHEDULER_DYNAMIC_DRAFT_CAP_PATCH_V1"
                    in scheduler_content
                    and "_nrl_specdec_scheduler_dynamic_last_selected_tier"
                    in scheduler_content
                    and "count_name = f\"_nrl_specdec_scheduler_dynamic_{selected_tier}_selected_count\""
                    in scheduler_content
                ):
                    old_lookahead_counter_block = (
                        "                count_name = f\"_nrl_specdec_scheduler_dynamic_{selected_tier}_selected_count\"\n"
                        "                setattr(self, count_name, getattr(self, count_name, 0) + 1)\n"
                        "            self._nrl_specdec_scheduler_dynamic_last_selected_tokens = effective_lookahead_tokens\n"
                    )
                    new_lookahead_counter_block = (
                        "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = selected_tier\n"
                        "            else:\n"
                        "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = \"\"\n"
                        "            self._nrl_specdec_scheduler_dynamic_last_selected_tokens = effective_lookahead_tokens\n"
                    )
                    if old_lookahead_counter_block not in scheduler_content:
                        raise RuntimeError(
                            "Could not upgrade existing adaptive SpecDec scheduler "
                            f"lookahead counters in {scheduler}; expected old "
                            "lookahead-counter block was missing."
                        )
                    scheduler_content = scheduler_content.replace(
                        old_lookahead_counter_block,
                        new_lookahead_counter_block,
                        1,
                    )
                    with open(scheduler, "w") as f:
                        f.write(scheduler_content)
                if (
                    "NRL_SPECDEC_SCHEDULER_DYNAMIC_DRAFT_CAP_PATCH_V1"
                    in scheduler_content
                    and "NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_BY_REQUEST_V1"
                    not in scheduler_content
                ):
                    scheduler_content = scheduler_content.replace(
                        "        def _nrl_specdec_scheduler_lookahead_tokens(num_requests, num_tokens):\n",
                        "        def _nrl_specdec_scheduler_lookahead_tokens(request_id, num_requests, num_tokens):\n",
                    )
                    scheduler_content = scheduler_content.replace(
                        "num_lookahead_tokens=_nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                        len(num_scheduled_tokens) + 1,\n",
                        "num_lookahead_tokens=_nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                        request.request_id,\n"
                        "                        len(num_scheduled_tokens) + 1,\n",
                    )
                    scheduler_content = scheduler_content.replace(
                        "num_lookahead_tokens=_nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                            len(num_scheduled_tokens) + 1,\n",
                        "num_lookahead_tokens=_nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                            request.request_id,\n"
                        "                            len(num_scheduled_tokens) + 1,\n",
                    )
                    scheduler_content = scheduler_content.replace(
                        "else _nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                        len(num_scheduled_tokens) + 1,\n",
                        "else _nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                        request.request_id,\n"
                        "                        len(num_scheduled_tokens) + 1,\n",
                    )
                    scheduler_content = scheduler_content.replace(
                        "else _nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                            len(num_scheduled_tokens) + 1,\n",
                        "else _nrl_specdec_scheduler_lookahead_tokens(\n"
                        "                            request.request_id,\n"
                        "                            len(num_scheduled_tokens) + 1,\n",
                    )
                    dynamic_request_state_anchor = (
                        "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_last_selected_tier\", \"\"\n"
                        "                )\n"
                        "            if dynamic_enabled and effective_lookahead_tokens > 0:\n"
                    )
                    dynamic_request_state_block = (
                        "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_last_selected_tier\", \"\"\n"
                        "                )\n"
                        "                self._nrl_specdec_scheduler_dynamic_selected_by_request = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_selected_by_request\", {}\n"
                        "                )\n"
                        "            # NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_BY_REQUEST_V1\n"
                        "            selected_by_request = getattr(\n"
                        "                self, \"_nrl_specdec_scheduler_dynamic_selected_by_request\", {}\n"
                        "            )\n"
                        "            if not isinstance(selected_by_request, dict):\n"
                        "                selected_by_request = {}\n"
                        "            self._nrl_specdec_scheduler_dynamic_selected_by_request = selected_by_request\n"
                        "            request_key = str(request_id)\n"
                        "            if dynamic_enabled and effective_lookahead_tokens > 0:\n"
                    )
                    if dynamic_request_state_anchor not in scheduler_content:
                        raise RuntimeError(
                            "Could not upgrade existing adaptive SpecDec scheduler "
                            f"with per-request dynamic K state in {scheduler}; "
                            "expected dynamic tier state anchor was missing."
                        )
                    scheduler_content = scheduler_content.replace(
                        dynamic_request_state_anchor,
                        dynamic_request_state_block,
                        1,
                    )
                    scalar_selected_tier_block = (
                        "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = selected_tier\n"
                        "            else:\n"
                        "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = \"\"\n"
                        "            self._nrl_specdec_scheduler_dynamic_last_selected_tokens = effective_lookahead_tokens\n"
                    )
                    request_selected_tier_block = (
                        "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = selected_tier\n"
                        "                selected_by_request[request_key] = (\n"
                        "                    effective_lookahead_tokens,\n"
                        "                    selected_tier,\n"
                        "                )\n"
                        "            else:\n"
                        "                self._nrl_specdec_scheduler_dynamic_last_selected_tier = \"\"\n"
                        "                selected_by_request.pop(request_key, None)\n"
                        "            self._nrl_specdec_scheduler_dynamic_last_selected_tokens = effective_lookahead_tokens\n"
                    )
                    if scalar_selected_tier_block not in scheduler_content:
                        raise RuntimeError(
                            "Could not upgrade existing adaptive SpecDec scheduler "
                            f"with per-request dynamic selected tier in {scheduler}; "
                            "expected scalar selected-tier block was missing."
                        )
                    scheduler_content = scheduler_content.replace(
                        scalar_selected_tier_block,
                        request_selected_tier_block,
                        1,
                    )
                    scalar_dynamic_cap_getter = (
                        "            dynamic_draft_cap = getattr(\n"
                        "                self,\n"
                        "                \"_nrl_specdec_scheduler_dynamic_last_selected_tokens\",\n"
                        "                self.num_spec_tokens,\n"
                        "            )\n"
                    )
                    request_dynamic_cap_getter = (
                        "            selected_by_request = getattr(\n"
                        "                self, \"_nrl_specdec_scheduler_dynamic_selected_by_request\", {}\n"
                        "            )\n"
                        "            if not isinstance(selected_by_request, dict):\n"
                        "                selected_by_request = {}\n"
                        "            request_key = str(getattr(request, \"request_id\", \"\"))\n"
                        "            dynamic_selection = None\n"
                        "            if request_key:\n"
                        "                dynamic_selection = selected_by_request.pop(request_key, None)\n"
                        "            if (\n"
                        "                isinstance(dynamic_selection, (tuple, list))\n"
                        "                and len(dynamic_selection) >= 2\n"
                        "            ):\n"
                        "                dynamic_draft_cap = dynamic_selection[0]\n"
                        "                dynamic_tier = dynamic_selection[1]\n"
                        "            else:\n"
                        "                dynamic_draft_cap = getattr(\n"
                        "                    self,\n"
                        "                    \"_nrl_specdec_scheduler_dynamic_last_selected_tokens\",\n"
                        "                    self.num_spec_tokens,\n"
                        "                )\n"
                        "                dynamic_tier = getattr(\n"
                        "                    self, \"_nrl_specdec_scheduler_dynamic_last_selected_tier\", \"\"\n"
                        "                )\n"
                        "            self._nrl_specdec_scheduler_dynamic_selected_by_request = selected_by_request\n"
                    )
                    if scalar_dynamic_cap_getter in scheduler_content:
                        scheduler_content = scheduler_content.replace(
                            scalar_dynamic_cap_getter,
                            request_dynamic_cap_getter,
                            1,
                        )
                    after_store_dynamic_tier_getter = (
                        "            # NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_COUNTERS_ON_STORE_V1\n"
                        "            self._nrl_specdec_scheduler_dynamic_last_stored_tokens = len(spec_token_ids)\n"
                        "            dynamic_tier = getattr(\n"
                        "                self, \"_nrl_specdec_scheduler_dynamic_last_selected_tier\", \"\"\n"
                        "            )\n"
                    )
                    after_store_no_tier_getter = (
                        "            # NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_COUNTERS_ON_STORE_V1\n"
                        "            self._nrl_specdec_scheduler_dynamic_last_stored_tokens = len(spec_token_ids)\n"
                    )
                    if after_store_dynamic_tier_getter in scheduler_content:
                        scheduler_content = scheduler_content.replace(
                            after_store_dynamic_tier_getter,
                            after_store_no_tier_getter,
                            1,
                        )
                    with open(scheduler, "w") as f:
                        f.write(scheduler_content)
                if (
                    "NRL_SPECDEC_SCHEDULER_DYNAMIC_UPDATE_DRAFT_CAP_PATCH_V1"
                    in scheduler_content
                    and "NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_COUNTERS_ON_STORE_V1"
                    not in scheduler_content
                ):
                    if old_update_draft_cap_block not in scheduler_content:
                        raise RuntimeError(
                            "Could not upgrade existing adaptive SpecDec scheduler "
                            f"with store-time dynamic counters in {scheduler}; "
                            "expected old update_draft_token_ids block was missing."
                        )
                    scheduler_content = scheduler_content.replace(
                        old_update_draft_cap_block,
                        update_draft_cap_block,
                        1,
                    )
                    with open(scheduler, "w") as f:
                        f.write(scheduler_content)
                if (
                    "NRL_SPECDEC_SCHEDULER_DYNAMIC_UPDATE_DRAFT_CAP_PATCH_V1"
                    not in scheduler_content
                ):
                    update_draft_cap_anchor = next(
                        (
                            anchor
                            for anchor in update_draft_cap_anchors
                            if anchor in scheduler_content
                        ),
                        None,
                    )
                    if update_draft_cap_anchor is None:
                        raise RuntimeError(
                            "Could not upgrade existing adaptive SpecDec scheduler "
                            f"with dynamic draft update cap in {scheduler}; "
                            "expected update_draft_token_ids anchor was missing."
                        )
                    scheduler_content = scheduler_content.replace(
                        update_draft_cap_anchor,
                        update_draft_cap_block,
                        1,
                    )
                    with open(scheduler, "w") as f:
                        f.write(scheduler_content)
                missing_markers = [
                    marker
                    for marker in scheduler_required_markers
                    if marker not in scheduler_content
                ]
                if not _has_scheduler_adaptive_lookahead_call(scheduler_content):
                    missing_markers.append("adaptive scheduler lookahead call")
                if missing_markers:
                    raise RuntimeError(
                        "Found a partial vLLM adaptive SpecDec scheduler-gate patch "
                        f"in {scheduler}; missing markers: "
                        + ", ".join(missing_markers)
                    )
                _assert_scheduler_dynamic_request_id_arity(
                    scheduler_content,
                    scheduler,
                )
            __import__("py_compile").compile(scheduler, doraise=True)

            return applied

        spec_decode_requested = isinstance(
            self.cfg.get("vllm_kwargs", {}).get("speculative_config"), dict
        )
        require_post_step_patch = os.environ.get(
            "NRL_REQUIRE_VLLM_SPECDEC_POST_STEP_PATCH", "1"
        ).lower() in {"1", "true", "yes", "y", "on"}

        def _nrl_env_nonnegative_int(name: str, default: int = 0) -> int:
            value = os.environ.get(name)
            if value is None or str(value).strip() == "":
                return default
            try:
                parsed = int(str(value).strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"{name} must be a non-negative integer, got {value!r}."
                ) from exc
            if parsed < 0:
                raise RuntimeError(
                    f"{name} must be a non-negative integer, got {value!r}."
                )
            return parsed

        def _nrl_env_optional_positive_int(name: str) -> int | None:
            value = os.environ.get(name)
            if value is None or str(value).strip() == "":
                return None
            parsed = _nrl_env_nonnegative_int(name)
            if parsed < 1:
                raise RuntimeError(f"{name} must be >= 1, got {value!r}.")
            return parsed

        def _nrl_env_bool(name: str, default: bool = False) -> bool:
            value = os.environ.get(name)
            if value is None or str(value).strip() == "":
                return default
            normalized = str(value).strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
            raise RuntimeError(
                f"{name} must be a boolean value, got {value!r}."
            )

        def _nrl_env_float(
            name: str,
            default: float,
            *,
            min_value: float | None = None,
            max_value: float | None = None,
        ) -> float:
            value = os.environ.get(name)
            if value is None or str(value).strip() == "":
                return default
            try:
                parsed = float(str(value).strip())
            except ValueError as exc:
                raise RuntimeError(f"{name} must be a float, got {value!r}.") from exc
            if parsed != parsed:
                raise RuntimeError(f"{name} must not be NaN, got {value!r}.")
            if parsed in {float("inf"), float("-inf")}:
                raise RuntimeError(f"{name} must be finite, got {value!r}.")
            if min_value is not None and parsed < min_value:
                raise RuntimeError(
                    f"{name} must be >= {min_value}, got {value!r}."
                )
            if max_value is not None and parsed > max_value:
                raise RuntimeError(
                    f"{name} must be <= {max_value}, got {value!r}."
                )
            return parsed

        specdec_batch_gate_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD"
        )
        specdec_batch_gate_token_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD"
        )
        _nrl_env_nonnegative_int("VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL", 256)
        specdec_adaptive_initial_request_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_ADAPTIVE_INITIAL_REQUEST_THRESHOLD"
        )
        specdec_adaptive_initial_token_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_ADAPTIVE_INITIAL_TOKEN_THRESHOLD"
        )
        for _nrl_adaptive_int_name, _nrl_adaptive_int_default in (
            ("VLLM_SPECDEC_ADAPTIVE_ADJUST_INTERVAL", 512),
            ("VLLM_SPECDEC_ADAPTIVE_REQUEST_STEP", 4),
            ("VLLM_SPECDEC_ADAPTIVE_TOKEN_STEP", 256),
        ):
            _nrl_env_nonnegative_int(
                _nrl_adaptive_int_name, _nrl_adaptive_int_default
            )
        specdec_adaptive_min_request_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_ADAPTIVE_MIN_REQUEST_THRESHOLD", 1
        )
        specdec_adaptive_max_request_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_ADAPTIVE_MAX_REQUEST_THRESHOLD", 128
        )
        specdec_adaptive_min_token_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_ADAPTIVE_MIN_TOKEN_THRESHOLD", 256
        )
        specdec_adaptive_max_token_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_ADAPTIVE_MAX_TOKEN_THRESHOLD", 8192
        )
        if specdec_adaptive_min_request_threshold > specdec_adaptive_max_request_threshold:
            raise RuntimeError(
                "VLLM_SPECDEC_ADAPTIVE_MIN_REQUEST_THRESHOLD must be <= "
                "VLLM_SPECDEC_ADAPTIVE_MAX_REQUEST_THRESHOLD."
            )
        if specdec_adaptive_min_token_threshold > specdec_adaptive_max_token_threshold:
            raise RuntimeError(
                "VLLM_SPECDEC_ADAPTIVE_MIN_TOKEN_THRESHOLD must be <= "
                "VLLM_SPECDEC_ADAPTIVE_MAX_TOKEN_THRESHOLD."
            )
        _nrl_env_float(
            "VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO",
            0.35,
            min_value=0.0,
            max_value=1.0,
        )
        _nrl_env_float(
            "VLLM_SPECDEC_ADAPTIVE_HYSTERESIS",
            0.05,
            min_value=0.0,
            max_value=1.0,
        )
        specdec_dynamic_small_request_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD", 4
        )
        specdec_dynamic_medium_request_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD", 16
        )
        specdec_dynamic_small_token_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD", 0
        )
        specdec_dynamic_medium_token_threshold = _nrl_env_nonnegative_int(
            "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD", 0
        )
        if (
            specdec_dynamic_small_request_threshold
            > specdec_dynamic_medium_request_threshold
        ):
            raise RuntimeError(
                "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD must be <= "
                "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD."
            )
        if (
            specdec_dynamic_small_token_threshold > 0
            and specdec_dynamic_medium_token_threshold > 0
            and specdec_dynamic_small_token_threshold
            > specdec_dynamic_medium_token_threshold
        ):
            raise RuntimeError(
                "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD must be <= "
                "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD when both "
                "are positive."
            )
        dynamic_small_tokens = _nrl_env_optional_positive_int(
            "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS"
        )
        dynamic_medium_tokens = _nrl_env_optional_positive_int(
            "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS"
        )
        dynamic_large_tokens = _nrl_env_optional_positive_int(
            "VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS"
        )
        if (
            dynamic_small_tokens is not None
            and dynamic_medium_tokens is not None
            and dynamic_small_tokens < dynamic_medium_tokens
        ):
            raise RuntimeError(
                "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS must be >= "
                "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS."
            )
        if (
            dynamic_medium_tokens is not None
            and dynamic_large_tokens is not None
            and dynamic_medium_tokens < dynamic_large_tokens
        ):
            raise RuntimeError(
                "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS must be >= "
                "VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS."
            )
        specdec_adaptive_gate_mode = os.environ.get(
            "VLLM_SPECDEC_ADAPTIVE_GATE_MODE", "off"
        ).lower()
        specdec_adaptive_gate_requested = specdec_adaptive_gate_mode not in {
            "",
            "0",
            "off",
            "false",
            "no",
        }
        specdec_dynamic_draft_tokens_requested = _nrl_env_bool(
            "VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS", False
        )
        enable_runtime_specdec_gate_patch = os.environ.get(
            "VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH", "0"
        ).lower() in {"1", "true", "yes", "y", "on"}
        specdec_gate_threshold_requested = (
            specdec_batch_gate_threshold > 0
            or specdec_batch_gate_token_threshold > 0
            or specdec_adaptive_gate_requested
            or specdec_dynamic_draft_tokens_requested
        )
        if specdec_adaptive_gate_requested and not (
            specdec_batch_gate_threshold > 0
            or specdec_batch_gate_token_threshold > 0
            or specdec_adaptive_initial_request_threshold > 0
            or specdec_adaptive_initial_token_threshold > 0
        ):
            raise RuntimeError(
                "VLLM_SPECDEC_ADAPTIVE_GATE_MODE is set, but no positive "
                "static or initial request/token threshold is configured. "
                "Set VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD, "
                "VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD, "
                "VLLM_SPECDEC_ADAPTIVE_INITIAL_REQUEST_THRESHOLD, or "
                "VLLM_SPECDEC_ADAPTIVE_INITIAL_TOKEN_THRESHOLD so adaptive "
                "SpecDec cannot silently become global SpecDec."
            )
        if specdec_gate_threshold_requested and not enable_runtime_specdec_gate_patch:
            raise RuntimeError(
                "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD or "
                "VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD or "
                "VLLM_SPECDEC_ADAPTIVE_GATE_MODE or "
                "VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS is set, but "
                "VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH is not true. "
                "This would run with global SpecDec instead of the intended "
                "runtime scheduler gate."
            )

        patch_lock_path = os.path.join(
            os.path.dirname(_get_vllm_file("__init__.py")),
            ".nemo_rl_patch.lock",
        )
        with open(patch_lock_path, "a") as patch_lock:
            fcntl.flock(patch_lock.fileno(), fcntl.LOCK_EX)
            _patch_vllm_init_workers_ray()
            logger.info("Successfully patched vllm _init_workers_ray.")

            _patch_vllm_vit_flash_attn_backend()
            logger.info("Successfully patched vllm vit flash attention backend.")

            if spec_decode_requested:
                post_step_patch_status = _patch_vllm_speculative_decoding_post_step(
                    required=require_post_step_patch
                )
                post_step_patch_message = (
                    "applied"
                    if post_step_patch_status
                    else "already-present-or-not-needed"
                )
            else:
                post_step_patch_status = 0
                post_step_patch_message = "skipped-no-specdec"
            logger.info(
                "vLLM speculative decoding post_step patch status: %s.",
                post_step_patch_message,
            )

            if (
                enable_runtime_specdec_gate_patch
                and specdec_gate_threshold_requested
            ):
                batch_gate_patch_status = _patch_vllm_batch_gated_speculative_decoding()
                adaptive_gate_patch_status = (
                    _patch_vllm_adaptive_specdec_gate()
                    if specdec_adaptive_gate_requested
                    or specdec_dynamic_draft_tokens_requested
                    else 0
                )
            else:
                batch_gate_patch_status = 0
                adaptive_gate_patch_status = 0
            fcntl.flock(patch_lock.fileno(), fcntl.LOCK_UN)

        if (
            enable_runtime_specdec_gate_patch
            and specdec_gate_threshold_requested
        ):
            logger.info(
                "vLLM batch-gated speculative decoding patch status: %s "
                "(threshold=%s active requests, token_threshold=%s scheduled tokens, "
                "adaptive_mode=%s, dynamic_draft_tokens=%s, adaptive_status=%s).",
                "applied" if batch_gate_patch_status else "already-present",
                specdec_batch_gate_threshold,
                specdec_batch_gate_token_threshold,
                specdec_adaptive_gate_mode,
                specdec_dynamic_draft_tokens_requested,
                "applied" if adaptive_gate_patch_status else "already-present-or-off",
            )

        try:
            import vllm

            self.SamplingParams = vllm.SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
                "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
                "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
                "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
            )
        vllm_kwargs: dict[str, Any] = copy.deepcopy(self.cfg.get("vllm_kwargs", {}))

        # Calculate total parallel size (TP * PP)
        model_parallel_size = self.tensor_parallel_size * self.pipeline_parallel_size

        # Special handling for parallel case (either TP or PP or both)
        if model_parallel_size > 1:
            # Configure vLLM for tensor/pipeline parallelism within Ray
            # Reset CUDA_VISIBLE_DEVICES to allow vLLM to manage GPU assignment
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(
                self.fraction_of_gpus / model_parallel_size
            )

            # Set bundle indices for parallel workers
            bundle_indices_str = ",".join(map(str, bundle_indices))
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = bundle_indices_str
            print(
                f"VLLM_RAY_BUNDLE_INDICES environment variable set to: {os.environ.get('VLLM_RAY_BUNDLE_INDICES')}"
            )

            # Use Ray for distributed execution in parallel mode
            vllm_kwargs["distributed_executor_backend"] = "ray"
        else:
            # For non-parallel mode, explicitly set executor to None to avoid Ray issues
            vllm_kwargs["distributed_executor_backend"] = None

        os.environ["VLLM_USE_V1"] = "1" if is_vllm_v1_engine_enabled() else "0"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # We should use vLLM DP if ep_size > tp_size since EP_SIZE = DP_SIZE * TP_SIZE in vLLM.
        # See details in https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/data_parallel.py
        if self.expert_parallel_size > self.tensor_parallel_size:
            # set vLLM DP rank
            world_size = int(os.environ["VLLM_DP_SIZE"]) * model_parallel_size
            rank = int(os.environ["RANK"]) % world_size
            os.environ["VLLM_DP_RANK"] = str(rank // model_parallel_size)
            os.environ["VLLM_DP_RANK_LOCAL"] = str((rank % 8) // model_parallel_size)
            # set vLLM DP address and port
            leader_rank = int(os.environ["RANK"]) // world_size * world_size
            addr_list = eval(os.environ["AVAILABLE_ADDR_LIST"])
            port_list = eval(os.environ["AVAILABLE_PORT_LIST"])
            os.environ["VLLM_DP_MASTER_IP"] = addr_list[leader_rank]
            os.environ["VLLM_DP_MASTER_PORT"] = str(port_list[leader_rank])

        load_format = self.cfg["vllm_cfg"]["load_format"]
        if ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(self.model_name):
            load_format = "auto"

        if (
            len(get_nsight_config_if_pattern_matches("vllm_generation_worker")) > 0
            and vllm_kwargs["distributed_executor_backend"] == "ray"
        ):
            logger.warning(
                "Nsight profiling is enabled for vllm generation worker through the vllm ray distributed executor. "
                "The nsight command-line args and output file names are automatically picked by the ray distributed "
                "executor. Refer to https://github.com/vllm-project/vllm/blob/7e3a8dc90670fd312ce1e0d4eba9bf11c571e3ad/vllm/executor/ray_distributed_executor.py#L136 "
                "for more information."
            )
            vllm_kwargs["ray_workers_use_nsight"] = True

        # Call init_fp8 when precision is fp8
        # (kv_cache_dtype can be fp8/fp8_e4m3 or auto, validated in init_fp8)
        if self.cfg["vllm_cfg"]["precision"] == "fp8":
            from nemo_rl.models.generation.fp8 import init_fp8

            fp8_kwargs = init_fp8(
                self.cfg["vllm_cfg"], self.model_name, model_parallel_size
            )

            vllm_kwargs.update(fp8_kwargs)
            # overriden by quant config, however vllm complains if this not passed
            self.precision = "bfloat16"

        if not isinstance(vllm_kwargs.get("hf_overrides"), dict):
            vllm_kwargs["hf_overrides"] = {}
        vllm_kwargs["hf_overrides"].update(
            self.cfg["vllm_cfg"].get("hf_overrides", {}) or {}
        )

        # Override HF config for gpt-oss models to ensure compatibility with megatron
        # The megatron --> hf export is done in bf16, so we disable quantization
        hf_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        if "GptOssForCausalLM" in getattr(hf_config, "architectures", []):
            if "quantization_config" in hf_config:
                assert load_format == "dummy", (
                    "Loading quantized GPT-OSS models is currently only supported with load_format='dummy'."
                )
                # disable quantization
                vllm_kwargs["hf_overrides"]["quantization_config"] = {}

        default_disable_log_stats = "true"

        llm_kwargs = dict(
            model=self.model_name,
            served_model_name=self.model_name,
            load_format=load_format,
            # Set in nemo_rl.models.generation.configure_generation_config
            skip_tokenizer_init=self.cfg["vllm_cfg"]["skip_tokenizer_init"],
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            enable_expert_parallel=self.enable_expert_parallel,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=torch.cuda.get_device_capability()[0] >= 8,
            dtype=self.precision,
            seed=seed,
            enforce_eager=self.cfg["vllm_cfg"]["enforce_eager"],
            max_model_len=self.cfg["vllm_cfg"]["max_model_len"],
            trust_remote_code=True,
            worker_extension_cls="nemo_rl.models.generation.vllm.vllm_backend.VllmInternalWorkerExtension",
            enable_sleep_mode=True,
            disable_log_stats=os.environ.get(
                "NRL_VLLM_DISABLE_LOG_STATS", default_disable_log_stats
            ).lower()
            not in {"0", "false", "no"},
            logprobs_mode="processed_logprobs",
            **vllm_kwargs,
        )

        speculative_config = llm_kwargs.get("speculative_config")
        if isinstance(speculative_config, dict):
            specdec_request_logprobs = os.environ.get(
                "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS", "0"
            ).lower() in {"1", "true", "yes", "y", "on"}
            logger.info(
                "NeMo-RL SpecDec effective config: method=%s, "
                "num_speculative_tokens=%s, draft_tensor_parallel_size=%s, "
                "disable_by_batch_size=%s, runtime_scheduler_gate_enabled=%s, "
                "runtime_scheduler_gate_threshold=%s, "
                "runtime_scheduler_gate_token_threshold=%s, "
                "runtime_scheduler_adaptive_mode=%s, disable_log_stats=%s, "
                "request_logprobs=%s",
                speculative_config.get("method"),
                speculative_config.get("num_speculative_tokens"),
                speculative_config.get("draft_tensor_parallel_size"),
                speculative_config.get("disable_by_batch_size"),
                enable_runtime_specdec_gate_patch,
                specdec_batch_gate_threshold,
                specdec_batch_gate_token_threshold,
                specdec_adaptive_gate_mode,
                llm_kwargs.get("disable_log_stats"),
                specdec_request_logprobs,
            )
            if (
                speculative_config.get("disable_by_batch_size") is not None
                and os.environ.get("NRL_ALLOW_SPECDEC_DISABLE_BY_BATCH_SIZE", "0").lower()
                not in {"1", "true", "yes", "y", "on"}
            ):
                raise ValueError(
                    "SpecDec disable_by_batch_size is configured, but this is "
                    "not the NeMo-RL long-tail gate. It is request-admission "
                    "scoped in some vLLM releases and can produce misleading "
                    "all-zero draft intervals. Use "
                    "VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=true with "
                    "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD or "
                    "VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD instead, or set "
                    "NRL_ALLOW_SPECDEC_DISABLE_BY_BATCH_SIZE=1 to keep the "
                    "vLLM-native behavior intentionally."
                )

        self._maybe_register_eagle3_model_in_process(llm_kwargs)
        self._create_engine(llm_kwargs)

        # will be initialized in post_init
        # used in update_weights_from_ipc_handles
        self.vllm_device_ids = None

    def llm(self):
        return self.llm

    def is_alive(self):
        """Check if the worker is alive."""
        return True

    def _merge_stop_strings(self, batch_stop_strings):
        stop_set: set[str] = set()

        if self.cfg.get("stop_strings"):
            stop_set.update(self.cfg["stop_strings"])

        if batch_stop_strings is not None:
            for sample_ss in batch_stop_strings:
                if sample_ss:
                    stop_set.update(sample_ss)

        return list(stop_set) if stop_set else None

    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        stop_strings,
        max_new_tokens: Optional[int] = None,
    ):
        top_k_cfg = self.cfg["top_k"]
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)

        temperature = 0.0 if greedy else self.cfg["temperature"]

        max_tokens = (
            max_new_tokens if max_new_tokens is not None else self.cfg["max_new_tokens"]
        )
        spec_decode_requested = isinstance(
            self.cfg.get("vllm_kwargs", {}).get("speculative_config"), dict
        )
        force_logprobs = os.environ.get(
            "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS", "0"
        ).lower() in {"1", "true", "yes", "y", "on"}
        omit_generation_logprobs = os.environ.get(
            "NRL_VLLM_OMIT_GENERATION_LOGPROBS", "0"
        ).lower() in {"1", "true", "yes", "y", "on"}
        if spec_decode_requested and force_logprobs:
            allow_specdec_logprobs = os.environ.get(
                "NRL_ALLOW_SPECDEC_REQUEST_LOGPROBS", "0"
            ).lower() in {"1", "true", "yes", "y", "on"}
            if not allow_specdec_logprobs:
                raise RuntimeError(
                    "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS=1 requests per-token "
                    "logprobs from vLLM. vLLM V1 disables speculative decoding "
                    "for logprob requests, which makes acceptance appear as 0 "
                    "even when the drafter is loaded. Leave this unset for "
                    "SpecDec throughput/acceptance runs, or set "
                    "NRL_ALLOW_SPECDEC_REQUEST_LOGPROBS=1 intentionally."
                )
        if omit_generation_logprobs and force_logprobs:
            raise RuntimeError(
                "NRL_VLLM_OMIT_GENERATION_LOGPROBS=true conflicts with "
                "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS=true."
            )
        request_logprobs = (not spec_decode_requested) or force_logprobs
        if omit_generation_logprobs:
            request_logprobs = False

        return self.SamplingParams(
            temperature=temperature,
            top_p=self.cfg["top_p"],
            top_k=top_k_val,
            max_tokens=max_tokens,
            logprobs=0 if request_logprobs else None,
            stop_token_ids=self.cfg["stop_token_ids"],
            stop=stop_strings,
            include_stop_str_in_output=True,
        )

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()
        if self.llm is not None:
            self.llm.collective_rpc("start_gpu_profiling", args=tuple())

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
        if self.llm is not None:
            self.llm.collective_rpc("stop_gpu_profiling", args=tuple())


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)  # pragma: no cover
class VllmGenerationWorker(BaseVllmGenerationWorker):
    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        import vllm

        self.llm = vllm.LLM(**llm_kwargs)

    def post_init(self):
        self.vllm_device_ids = self.report_device_id()

    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        self.llm.collective_rpc(
            "init_collective",
            args=(
                rank_prefix,
                ip,
                port,
                world_size,
                train_world_size,
            ),
        )

    @wrap_with_nvtx_name("vllm_genertion_worker/generate")
    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using vLLM generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs with proper padding
                - logprobs: Log probabilities for tokens
                - generation_lengths: Lengths of each response
                - unpadded_sequence_lengths: Lengths of each input + generated sequence
        """
        # Handle empty input case
        if len(data["input_ids"]) == 0:
            # Return empty BatchedDataDict with all required fields
            return BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": torch.zeros((0, 0), dtype=torch.long),
                    "logprobs": torch.zeros((0, 0), dtype=torch.float),
                    "generation_lengths": torch.zeros(0, dtype=torch.long),
                    "unpadded_sequence_lengths": torch.zeros(0, dtype=torch.long),
                    "truncated": torch.zeros(0, dtype=torch.bool),
                }
            )

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_stop_strings: list[list[str]] = data.get("stop_strings") or []

        # verify inputs have correct padding
        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        # Original input length with padding
        padded_input_length = input_ids.size(1)
        max_model_len = self.cfg["vllm_cfg"]["max_model_len"]
        max_new_tokens = self.cfg["max_new_tokens"]
        remaining_ctx_per_sample = torch.clamp(
            max_model_len - input_lengths, min=0, max=max_new_tokens
        )
        allowed_new_tokens_per_sample = [
            int(tokens) for tokens in remaining_ctx_per_sample.detach().cpu().tolist()
        ]
        generation_indices = [
            idx
            for idx, allowed_new_tokens in enumerate(allowed_new_tokens_per_sample)
            if allowed_new_tokens > 0
        ]

        # Generate outputs
        assert self.llm is not None, (
            "Attempting to generate with either an uninitialized vLLM or non-model-owner"
        )
        outputs_by_index = {}
        if generation_indices:
            prompts = [
                format_prompt_for_vllm_generation(data, idx)
                for idx in generation_indices
            ]
            sampling_params = [
                self._build_sampling_params(
                    greedy=greedy,
                    stop_strings=self._merge_stop_strings(
                        [batch_stop_strings[idx]]
                        if idx < len(batch_stop_strings)
                        else None
                    ),
                    max_new_tokens=allowed_new_tokens_per_sample[idx],
                )
                for idx in generation_indices
            ]
            outputs = self.llm.generate(prompts, sampling_params)
            outputs_by_index = dict(zip(generation_indices, outputs))

        # Process the outputs - but preserve the original input padding structure
        output_ids_list = []
        logprobs_list = []
        generation_lengths = []
        unpadded_sequence_lengths = []
        truncated = []
        max_length = 0
        for output in outputs_by_index.values():
            max_length = max(max_length, len(output.outputs[0].token_ids))

        for i in range(input_ids.size(0)):
            # Extract generated tokens
            sequence_length = input_lengths[i]
            output = outputs_by_index.get(i)
            generation = output.outputs[0] if output is not None else None
            generated_tokens = list(generation.token_ids) if generation else []

            # Calculate total sequence length (original input length + generated tokens)
            total_length = padded_input_length + max_length

            # Create a new tensor with the right size and fill with padding token
            full_output = torch.full(
                (total_length,), self.cfg["_pad_token_id"], dtype=input_ids.dtype
            )

            # Copy original input (with padding) into the beginning
            full_output[:sequence_length] = input_ids[i][:sequence_length]

            # Add generated tokens after the original input
            full_output[sequence_length : sequence_length + len(generated_tokens)] = (
                torch.tensor(generated_tokens)
            )

            output_ids_list.append(full_output)
            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            if (
                generation is not None
                and hasattr(generation, "logprobs")
                and generation.logprobs
            ):
                try:
                    for idx, logprob_dict in enumerate(generation.logprobs):
                        if logprob_dict and idx < len(generated_tokens):
                            token_id = generated_tokens[idx]
                            logprob_entry = logprob_dict.get(token_id)
                            if logprob_entry is None:
                                logprob_entry = logprob_dict.get(str(token_id))
                            if logprob_entry is not None:
                                position = sequence_length + idx
                                full_logprobs[position] = logprob_entry.logprob
                except Exception:
                    import traceback

                    traceback.print_exc()

            logprobs_list.append(full_logprobs)

            response_length = sequence_length + len(generated_tokens)
            generation_lengths.append(len(generated_tokens))
            unpadded_sequence_lengths.append(response_length)
            truncated.append(
                allowed_new_tokens_per_sample[i] == 0
                or getattr(generation, "finish_reason", None) == "length"
            )
            assert response_length <= self.llm.llm_engine.model_config.max_model_len, (
                f"response_length={response_length} > max_model_len={self.llm.llm_engine.model_config.max_model_len}, which should not happen. Please check this behavior in isolation by running `uv run --extra vllm tools/model_diagnostics/1.max_model_len_respected.py {self.llm.llm_engine.model_config.model}` and raise this issue with the vllm team."
            )

        # Create return data conforming to GenerationOutputSpec
        output_ids = torch.stack(output_ids_list)
        logprobs = torch.stack(logprobs_list)

        return_data = BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids,
                "logprobs": logprobs,
                "generation_lengths": torch.tensor(
                    generation_lengths, dtype=torch.long
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    unpadded_sequence_lengths, dtype=torch.long
                ),
                "truncated": torch.tensor(truncated, dtype=torch.bool),
            }
        )

        return return_data

    @wrap_with_nvtx_name("vllm_genertion_worker/generate_text")
    def generate_text(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate text responses using vLLM generation.

        Args:
            data: BatchedDataDict containing prompts with text strings
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict containing:
                - texts: List of generated text responses
        """
        # Check if async engine is enabled
        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_text cannot be used with async_engine=True. Use generate_text_async instead."
            )

        # Extract stop_strings if provided, else use default from config
        batch_stop_strings: list[list[str] | None] = data.get(
            "stop_strings", [self.cfg.get("stop_strings")] * len(data["prompts"])
        )

        # This function requires all generations have the same stop strings, so we collect all here
        stop_strings: set[str] = set()
        for sample_stop_strings in batch_stop_strings:
            if sample_stop_strings:
                stop_strings.update(sample_stop_strings)

        # Add default stop strings from config
        if self.cfg.get("stop_strings", None):
            stop_strings.update(self.cfg["stop_strings"])

        stop_strings = list(stop_strings) if len(stop_strings) > 0 else None

        # Read generation parameters from config
        top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
        sampling_params = self.SamplingParams(
            temperature=self.cfg["temperature"] if not greedy else 0,
            top_p=self.cfg["top_p"],
            top_k=top_k if not greedy else 1,
            max_tokens=self.cfg["max_new_tokens"],
            stop_token_ids=self.cfg["stop_token_ids"],
            stop=stop_strings,
            include_stop_str_in_output=True,  # returning stop strings like hf
        )

        # Generate outputs
        assert self.llm is not None, (
            "Attempting to generate with either an uninitialized vLLM or non-model-owner"
        )
        outputs = self.llm.generate(data["prompts"], sampling_params)
        texts = [output.outputs[0].text for output in outputs]

        # Convert to BatchedDataDict
        return_data: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict(
            {"texts": texts}
        )
        return return_data

    def report_device_id(self) -> list[str]:
        """Report device ID from the vLLM worker."""
        assert self.llm is not None, (
            "Attempting to report device id with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "report_device_id cannot be used with async_engine=True. Use report_device_id_async instead."
            )

        list_of_worker_results = self.llm.collective_rpc(
            "report_device_id", args=tuple()
        )
        return cast(list[str], list_of_worker_results)

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare the info for refit."""
        self.llm.collective_rpc("prepare_refit_info", args=(state_dict_info,))

    @wrap_with_nvtx_name("vllm_genertion_worker/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Update weights from IPC handles via ZMQ socket."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_via_ipc_zmq cannot be used with async_engine=True. Use update_weights_via_ipc_zmq_async instead."
                )

            result_or_coro = self.llm.collective_rpc(
                "update_weights_via_ipc_zmq",
                args=tuple(),
            )
            worker_result = result_or_coro[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    @wrap_with_nvtx_name("vllm_genertion_worker/update_weights_from_collective")
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_collective can only be used with async_engine=False. Use update_weights_from_collective_async instead."
                )

            result_or_coro = self.llm.collective_rpc(
                "update_weights_from_collective", args=tuple()
            )
            worker_result = result_or_coro[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    def reset_prefix_cache(self):
        """Reset the prefix cache of vLLM engine."""
        assert self.llm is not None, (
            "Attempting to reset prefix cache with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "reset_prefix_cache can only be used with async_engine=False. Use reset_prefix_cache_async instead."
            )

        self.llm.llm_engine.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def sleep(self):
        """Put the vLLM engine to sleep."""
        assert self.llm is not None, (
            "Attempting to sleep with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "sleep cannot be used with async_engine=True. Use sleep_async instead."
            )

        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        self.llm.llm_engine.reset_prefix_cache()
        self.llm.sleep(level=1)

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, **kwargs):
        """Wake up the vLLM engine."""
        assert self.llm is not None, (
            "Attempting to wake up with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "wake_up cannot be used with async_engine=True. Use wake_up_async instead."
            )

        tags = kwargs.get("tags")

        wake_up_args = {}
        if tags is not None:
            wake_up_args["tags"] = tags

        self.llm.wake_up(**wake_up_args)

    def shutdown(self) -> bool:
        """Clean up vLLM resources."""
        try:
            if self.llm is not None:
                # Clean up extension resources (e.g., ZMQ sockets)
                self.llm.collective_rpc("cleanup", args=tuple())

                # Explicitly delete the engine. This may trigger its __del__ method.
                del self.llm

            self.llm = None
            self.tokenizer = None

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False
