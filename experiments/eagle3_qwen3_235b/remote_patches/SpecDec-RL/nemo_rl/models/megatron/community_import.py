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
from inspect import getsource, signature
from typing import Any, Optional

from megatron.bridge import AutoBridge

from nemo_rl.models.policy import MegatronConfig


def _call_accepts_kwarg(func, name: str) -> bool:
    try:
        params = signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(param.kind == param.VAR_KEYWORD for param in params.values())


def _provider_forwards_expert_tensor_parallel_size(model_provider) -> bool:
    """Return whether Bridge already forwards ETP to Megatron-Core."""
    try:
        source = getsource(model_provider.initialize_model_parallel)
    except (OSError, TypeError):
        return False
    return "expert_tensor_parallel_size" in source


def _hf_tokenizer_kwargs(bridge) -> Optional[dict[str, Any]]:
    model_bridge = getattr(bridge, "_model_bridge", None)
    getter = getattr(model_bridge, "get_hf_tokenizer_kwargs", None)
    if callable(getter):
        return getter()
    return None


def _patch_distcp_writer_for_ray_import() -> None:
    """Avoid nested forked checkpoint writer processes during HF import.

    The older Megatron-Core writer used in the target container forks one async
    writer process and then forks per-file writer children. Under Ray actors this
    can leave Qwen3-235B imports stuck with partial ``.__*.distcp.tmp`` files.
    Keeping the per-rank file writes inline trades some import speed for a much
    more predictable one-time conversion.
    """
    if os.environ.get("NRL_MEGATRON_IMPORT_INLINE_WRITER", "1").lower() in {"0", "false", "no"}:
        return

    try:
        from megatron.core.dist_checkpointing.strategies.filesystem_async import (
            FileSystemWriterAsync,
        )
    except ImportError:
        return

    if getattr(FileSystemWriterAsync, "_nrl_import_inline_writer", False):
        return

    original_write = FileSystemWriterAsync.write_preloaded_data

    def write_preloaded_data_inline(transform_list, use_msc, rank, write_buckets, global_results_queue):
        import queue

        write_results_or_exc: dict[int, Any] | Exception = {}
        local_results_queue: queue.Queue = queue.Queue()
        count_queue: queue.Queue = queue.Queue()

        for local_proc_idx, write_bucket in enumerate(write_buckets):
            count_queue.put(local_proc_idx)
            original_write(
                transform_list,
                local_proc_idx,
                write_bucket,
                local_results_queue,
                count_queue,
                True,
                use_msc=use_msc,
            )

            result_idx, local_result = local_results_queue.get()
            if isinstance(local_result, Exception):
                write_results_or_exc = local_result
                break
            write_results_or_exc[result_idx] = local_result

        global_results_queue.put(write_results_or_exc)

    FileSystemWriterAsync.write_preloaded_data_multiproc = staticmethod(write_preloaded_data_inline)
    FileSystemWriterAsync._nrl_import_inline_writer = True


def _save_megatron_model_with_checkpointing(
    megatron_model,
    output_path: str,
    hf_model_name: str,
    model_provider,
) -> None:
    from megatron.bridge.training.checkpointing import (
        init_checkpointing_context,
        save_checkpoint,
    )
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        LoggerConfig,
        MockGPTDatasetConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )
    from megatron.bridge.training.state import GlobalState, TrainState

    _patch_distcp_writer_for_ray_import()

    checkpoint_config = CheckpointConfig(
        save=output_path,
        save_interval=1,
        save_optim=False,
        save_rng=False,
        load=None,
        load_optim=False,
        load_rng=False,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
        async_save=False,
    )
    state = GlobalState()
    state.cfg = ConfigContainer(
        train=TrainingConfig(micro_batch_size=1, global_batch_size=1, train_iters=0),
        model=model_provider,
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=0.0,
            min_lr=0.0,
            bf16=True,
        ),
        scheduler=SchedulerConfig(lr_decay_style="constant", lr_decay_iters=0),
        dataset=MockGPTDatasetConfig(
            random_seed=0,
            sequence_length=1,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        ),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        ),
        checkpoint=checkpoint_config,
    )
    state.train_state = TrainState(step=0)
    checkpointing_context = init_checkpointing_context(checkpoint_config)
    save_checkpoint(
        state,
        megatron_model,
        None,
        None,
        0,
        checkpointing_context=checkpointing_context,
    )


def _save_megatron_model(bridge, megatron_model, output_path: str, hf_model_name: str, model_provider) -> None:
    """Save through the Bridge method when available, otherwise use the training helper.

    Megatron-Bridge save APIs moved across releases. The oci-hsg container keeps
    only the older checkpointing helper, while newer Bridge builds expose a
    direct ``save_megatron_model`` method or ``model_load_save`` helper.
    """
    save = getattr(bridge, "save_megatron_model", None)
    if not callable(save):
        try:
            from megatron.bridge.training.model_load_save import save_megatron_model as save
        except ImportError as exc:
            try:
                _save_megatron_model_with_checkpointing(
                    megatron_model,
                    output_path,
                    hf_model_name,
                    model_provider,
                )
                return
            except ImportError:
                raise ImportError(
                    "No compatible Megatron-Bridge native checkpoint save API is available."
                ) from exc

    kwargs: dict[str, Any] = {}
    if _call_accepts_kwarg(save, "hf_tokenizer_path"):
        kwargs["hf_tokenizer_path"] = hf_model_name
    tokenizer_kwargs = _hf_tokenizer_kwargs(bridge)
    if tokenizer_kwargs is not None and _call_accepts_kwarg(save, "hf_tokenizer_kwargs"):
        kwargs["hf_tokenizer_kwargs"] = tokenizer_kwargs
    save(megatron_model, output_path, **kwargs)


def import_model_from_hf_name(
    hf_model_name: str,
    output_path: str,
    megatron_config: Optional[MegatronConfig] = None,
    **config_overrides: Any,
):
    """Import a Hugging Face model into Megatron checkpoint format and save the Megatron checkpoint to the output path.

    Args:
        hf_model_name: Hugging Face model ID or local path (e.g., 'meta-llama/Llama-3.1-8B-Instruct').
        output_path: Directory to write the Megatron checkpoint (e.g., /tmp/megatron_ckpt).
        megatron_config: Optional megatron config with paralellism settings for distributed megatron model import.
    """
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **config_overrides
    )

    model_provider = bridge.to_megatron_provider(load_weights=True)

    # Keep track of defaults so can restore them to the config after loading the model
    orig_tensor_model_parallel_size = model_provider.tensor_model_parallel_size
    orig_pipeline_model_parallel_size = model_provider.pipeline_model_parallel_size
    orig_context_parallel_size = model_provider.context_parallel_size
    orig_expert_model_parallel_size = model_provider.expert_model_parallel_size
    orig_expert_tensor_parallel_size = model_provider.expert_tensor_parallel_size
    orig_num_layers_in_first_pipeline_stage = (
        model_provider.num_layers_in_first_pipeline_stage
    )
    orig_num_layers_in_last_pipeline_stage = (
        model_provider.num_layers_in_last_pipeline_stage
    )
    orig_pipeline_dtype = model_provider.pipeline_dtype

    if megatron_config is not None:
        model_provider.tensor_model_parallel_size = megatron_config[
            "tensor_model_parallel_size"
        ]
        model_provider.pipeline_model_parallel_size = megatron_config[
            "pipeline_model_parallel_size"
        ]
        model_provider.context_parallel_size = megatron_config["context_parallel_size"]
        model_provider.expert_model_parallel_size = megatron_config[
            "expert_model_parallel_size"
        ]
        model_provider.expert_tensor_parallel_size = megatron_config[
            "expert_tensor_parallel_size"
        ]
        model_provider.num_layers_in_first_pipeline_stage = megatron_config[
            "num_layers_in_first_pipeline_stage"
        ]
        model_provider.num_layers_in_last_pipeline_stage = megatron_config[
            "num_layers_in_last_pipeline_stage"
        ]
        model_provider.pipeline_dtype = megatron_config["pipeline_dtype"]
        model_provider.sequence_parallel = megatron_config["sequence_parallel"]
    model_provider.finalize()
    model_parallel_kwargs = {}
    if megatron_config is not None:
        # Older Bridge finalize can recouple MoE ETP to TP; restore the runtime
        # import parallelism. Some Bridge versions already forward ETP from the
        # provider attribute, while older ones require the mixin's **kwargs path.
        model_provider.expert_tensor_parallel_size = megatron_config[
            "expert_tensor_parallel_size"
        ]
        if not _provider_forwards_expert_tensor_parallel_size(model_provider):
            model_parallel_kwargs["expert_tensor_parallel_size"] = megatron_config[
                "expert_tensor_parallel_size"
            ]
    model_provider.initialize_model_parallel(seed=0, **model_parallel_kwargs)
    provide_distributed_model = getattr(
        model_provider, "provide_distributed_model", None
    )
    if callable(provide_distributed_model):
        megatron_model = provide_distributed_model(wrap_with_ddp=False)
    else:
        megatron_model = model_provider.provide_models(wrap_with_ddp=False)

    # The above parallelism settings are used to load the model in a distributed manner.
    # However, we do not want to save the parallelism settings to the checkpoint config
    # because they may result in validation errors when loading the checkpoint.
    config = megatron_model[0].config
    config.tensor_model_parallel_size = orig_tensor_model_parallel_size
    config.pipeline_model_parallel_size = orig_pipeline_model_parallel_size
    config.context_parallel_size = orig_context_parallel_size
    config.expert_model_parallel_size = orig_expert_model_parallel_size
    config.expert_tensor_parallel_size = orig_expert_tensor_parallel_size
    config.num_layers_in_first_pipeline_stage = orig_num_layers_in_first_pipeline_stage
    config.num_layers_in_last_pipeline_stage = orig_num_layers_in_last_pipeline_stage
    config.pipeline_dtype = orig_pipeline_dtype

    _save_megatron_model(bridge, megatron_model, output_path, hf_model_name, model_provider)

    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()


def export_model_from_megatron(
    hf_model_name: str,
    input_path: str,
    output_path: str,
    hf_tokenizer_path: str,
    overwrite: bool = False,
    hf_overrides: Optional[dict[str, Any]] = {},
):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {output_path}. Delete it to run or set overwrite=True."
        )

    try:
        from megatron.bridge.training.model_load_save import (
            temporary_distributed_context,
        )
    except ImportError:
        raise ImportError("megatron.bridge.training is not available.")

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **hf_overrides
    )

    # Export performs on CPU with proper distributed context
    with temporary_distributed_context(backend="gloo"):
        # Need to set model parallel cuda manual seed for mamba mixer
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)

        # Load the Megatron model
        megatron_model = bridge.load_megatron_model(
            input_path, skip_temp_dist_context=True
        )

        # Save in HuggingFace format
        bridge.save_hf_pretrained(megatron_model, output_path)

    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()
