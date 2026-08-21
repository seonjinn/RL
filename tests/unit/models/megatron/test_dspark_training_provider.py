from __future__ import annotations

from types import SimpleNamespace

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.draft.training import (
    DSparkForwardOutput,
    DSparkSpeculator,
    resolve_draft_speculator,
)
from nemo_rl.models.policy.draft_config import DSparkDraftConfig


def _config() -> DSparkDraftConfig:
    return DSparkDraftConfig(
        enabled=True,
        model_name=(
            "deepseek-ai/dspark_qwen3_8b_block7"
            "@03326e5043815da1f81b109078b2889737c26017"
        ),
        block_size=7,
        anchors_per_sample=2,
        mask_token_id=151669,
        target_hidden_state_layer_ids=[1, 9, 17, 25, 33],
        num_layers=5,
        markov_rank=256,
        confidence_enabled=True,
        confidence_with_markov=True,
        seed=42,
    )


def test_dspark_registry_resolves_public_training_provider() -> None:
    provider = resolve_draft_speculator(_config())

    assert isinstance(provider, DSparkSpeculator)
    assert provider.capture_layer_ids() == (1, 9, 17, 25, 33)


def test_dspark_k7_plan_trains_anchor_and_six_masks_on_seven_future_tokens() -> None:
    provider = DSparkSpeculator(_config())
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(24).reshape(2, 12),
            "input_lengths": torch.tensor([12, 11]),
            "draft_sample_ids": torch.tensor([101, 303], dtype=torch.int64),
        }
    )

    plan = provider.prepare_batch(data, optimizer_step=9)

    assert plan.block_size == 7
    assert plan.query_positions.shape == (4, 7)
    torch.testing.assert_close(plan.label_positions, plan.query_positions + 1)
    assert torch.equal(plan.loss_mask, plan.slot_valid)
    assert bool(plan.loss_mask[:, 0].all())
    assert int(plan.label_positions[plan.loss_mask].max()) < 12


def test_dspark_loss_uses_sampled_rollout_tokens_as_hard_labels() -> None:
    provider = DSparkSpeculator(_config())
    input_ids = torch.arange(24).reshape(2, 12)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([12, 11]),
            "draft_sample_ids": torch.tensor([101, 303], dtype=torch.int64),
        }
    )
    plan = provider.prepare_batch(data, optimizer_step=9)
    captured: dict[str, torch.Tensor] = {}

    class _Adapter:
        def objective_stats(
            self,
            *,
            hard_labels: torch.Tensor,
            **_: object,
        ) -> SimpleNamespace:
            captured["hard_labels"] = hard_labels
            bins = torch.ones(plan.block_size)
            return SimpleNamespace(
                combined=SimpleNamespace(
                    numerators=bins,
                    counts=bins,
                    weights=bins,
                )
            )

    data["dspark_output"] = DSparkForwardOutput(
        hidden=torch.zeros((*plan.query_positions.shape, 4)),
        plan=plan,
        output_weight=torch.zeros(8, 4),
        previous_token_ids=input_ids[plan.sample_rows[:, None], plan.query_positions],
        adapter=_Adapter(),
    )
    target_logits = torch.zeros(2, 12, 8)
    target_logits[..., 0] = 1

    provider.loss_stats(
        target_logits=target_logits,
        data=data,
        prepare_fn=lambda **_: None,
        vocab_parallel_rank=0,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )

    expected = input_ids[plan.sample_rows[:, None], plan.label_positions]
    torch.testing.assert_close(captured["hard_labels"], expected)
    assert not torch.equal(
        captured["hard_labels"],
        target_logits.argmax(dim=-1)[plan.sample_rows[:, None], plan.query_positions],
    )


def test_dspark_export_uses_runtime_names_without_target_owned_weights(
    monkeypatch,
) -> None:
    provider = DSparkSpeculator(_config())
    body_weight = torch.ones(2, 2)
    markov_w1 = torch.ones(3, 2)
    markov_w2 = torch.ones(3, 2)
    confidence_weight = torch.ones(1, 4)
    confidence_bias = torch.ones(1)
    model = SimpleNamespace(
        body=object(),
        markov_head=SimpleNamespace(
            markov_w1=SimpleNamespace(weight=markov_w1),
            markov_w2=SimpleNamespace(weight=markov_w2),
            target_vocab_size=3,
            draft_vocab_size=3,
        ),
        confidence_head=SimpleNamespace(
            proj=SimpleNamespace(weight=confidence_weight, bias=confidence_bias)
        ),
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.draft.training.export_dflash_weights_to_hf",
        lambda body: [("fc.weight", body_weight)],
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.draft.training.export_dspark_heads_to_hf",
        lambda adapter: [
            ("markov_head.markov_w1.weight", markov_w1),
            ("markov_head.markov_w2.weight", markov_w2),
            ("confidence_head.proj.weight", confidence_weight),
            ("confidence_head.proj.bias", confidence_bias),
        ],
    )

    exported = provider.export_weights(model)  # type: ignore[arg-type]

    assert [name for name, _ in exported] == [
        "fc.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    ]
    assert not any(
        name.startswith(("embed_tokens.", "lm_head.")) for name, _ in exported
    )
