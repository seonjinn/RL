# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real provider construction, update, and checkpoint reload with SWA."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from nemo_rl.models.megatron.draft.block_plan import build_dflash_batch_plan
from nemo_rl.models.megatron.draft.training import DFlashSpeculator, DSparkSpeculator
from nemo_rl.models.policy.draft_config import DFlashDraftConfig, DSparkDraftConfig

pytestmark = pytest.mark.mcore


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_provider_inherits_checkpoint_window_and_reloads_updated_weights(
    method: str,
    tmp_path: Path,
) -> None:
    dims = SimpleNamespace(
        num_layers=4,
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
        fp16=False,
        bf16=False,
        params_dtype=torch.float32,
        hidden_size=8,
        ffn_hidden_size=12,
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=4,
        rotary_base=10000.0,
        layernorm_epsilon=1e-6,
        init_method_std=0.02,
        vocab_size=16,
    )
    options = dict(
        enabled=True,
        anchors_per_sample=1,
        mask_token_id=3,
        target_hidden_state_layer_ids=[1, 2],
        num_layers=1,
        sliding_window=3,
    )
    config = (
        DFlashDraftConfig(gamma=1, **options)
        if method == "dflash"
        else DSparkDraftConfig(block_size=2, markov_rank=2, **options)
    )
    provider_type = DFlashSpeculator if method == "dflash" else DSparkSpeculator
    provider = provider_type(config)
    build_args = dict(
        model_provider=dims,
        pg_collection=SimpleNamespace(tp=None),
        policy_model_chunk=torch.nn.Identity(),
    )
    draft = provider.build_model(**build_args)
    assert draft is not None
    body = draft if method == "dflash" else draft.body
    assert body.config.sliding_window == 3
    plan = build_dflash_batch_plan(
        torch.ones((1, 8), dtype=torch.bool),
        torch.tensor([7]),
        anchors_per_sample=1,
        gamma=1,
        optimizer_step=0,
        seed=7,
    )
    taps = torch.randn(1, 8, 2, 8)
    embeddings = torch.randn(1, 2, 8)
    before = body(target_taps=taps, block_embeddings=embeddings, plan=plan)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=0.01)
    (before * torch.randn_like(before)).sum().backward()
    optimizer.step()
    after = body(target_taps=taps, block_embeddings=embeddings, plan=plan)
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)

    exported = {
        name.removeprefix("draft."): tensor.detach().contiguous()
        for name, tensor in provider.export_weights(draft)
    }
    save_file(exported, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "sliding_window": 3,
                "dflash_config": {
                    "use_swa": True,
                    "swa_window_size": 3,
                    "causal": False,
                },
            }
        )
    )
    restored = provider_type(
        config.model_copy(
            update={
                "model_name": str(tmp_path),
                "sliding_window": None,
            }
        )
    ).build_model(**build_args)
    restored_body = restored if method == "dflash" else restored.body
    assert restored_body.config.sliding_window == 3
    actual = restored_body(target_taps=taps, block_embeddings=embeddings, plan=plan)
    torch.testing.assert_close(actual, after)
