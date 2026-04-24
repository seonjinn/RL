# coding=utf-8
# Copyright 2024 AI21 Labs Ltd. and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""NemotronH model configuration"""

import re

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NemotronHConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NemotronHModel`]. It is used to instantiate a
    NemotronH model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the NemotronH-v0.1 model.
    """

    model_type = "nemotron_h"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        tie_word_embeddings=False,
        hidden_size=4096,
        intermediate_size=21504,
        num_hidden_layers=52,
        hybrid_override_pattern="M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
        num_attention_heads=32,
        head_dim=128,
        num_key_value_heads=8,
        mlp_hidden_act="relu2",
        attention_bias=False,
        mlp_bias=False,
        use_bias=False,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        residual_in_fp32=False,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=None,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_mamba_kernels=True,
        ssm_state_size=128,
        mamba_num_heads=128,
        mamba_n_groups=8,
        mamba_head_dim=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_hidden_act="silu",
        mamba_dt_min=0.001,
        mamba_dt_max=0.1,
        mamba_dt_limit=(0.0, float("inf")),
        mamba_dt_init_floor=1e-4,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_chunk_size=256,
        rescale_prenorm_residual=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hybrid_override_pattern = hybrid_override_pattern
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        assert len(self.hybrid_override_pattern) == self.num_hidden_layers, (
            "hybrid_override_pattern must have the same length as num_hidden_layers"
        )
        assert re.match(r"^[*-M]+$", self.hybrid_override_pattern), (
            "hybrid_override_pattern must only contain characters 'M', '*', or '-'"
        )

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        self.use_mamba_kernels = use_mamba_kernels
        self.n_groups = mamba_n_groups
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.conv_kernel = mamba_d_conv
        self.expand = mamba_expand
        self.mamba_hidden_act = mamba_hidden_act
        self.time_step_min = mamba_dt_min
        self.time_step_max = mamba_dt_max
        self.time_step_limit = mamba_dt_limit
        self.time_step_floor = mamba_dt_init_floor
        self.use_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.chunk_size = mamba_chunk_size
        self.rescale_prenorm_residual = rescale_prenorm_residual

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return [
            "mamba"
            if self.hybrid_override_pattern[i] == "M"
            else "attention"
            if self.hybrid_override_pattern[i] == "*"
            else "mlp"
            for i in range(self.num_hidden_layers)
        ]
