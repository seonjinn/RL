import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import torch


DRAFT_LOSS_PATH = Path(__file__).parents[3] / "nemo_rl/algorithms/loss/draft.py"


def _draft_loss_module():
    spec = importlib.util.spec_from_file_location("draft_loss", DRAFT_LOSS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dflash_adapter_validates_tp_metadata_once() -> None:
    draft_loss = _draft_loss_module()
    generator = torch.Generator().manual_seed(24601)
    draft_hidden = torch.randn(2, 3, 4, generator=generator).requires_grad_(True)
    output_weight = torch.randn(7, 4, generator=generator).requires_grad_(True)
    teacher_logits = torch.randn(2, 4, 7, generator=generator).requires_grad_(True)
    sample_rows = torch.tensor([1, 0])
    label_positions = torch.tensor([[-1, 2, 3], [-1, 1, 2]])
    loss_mask = torch.tensor([[False, True, True], [False, True, False]])

    with patch.object(
        draft_loss,
        "_tp_assert_projected_metadata_agreement",
        wraps=draft_loss._tp_assert_projected_metadata_agreement,
    ) as agreement:
        draft_loss.dflash_projected_vocab_parallel_soft_ce(
            draft_hidden=draft_hidden,
            output_weight=output_weight,
            teacher_logits=teacher_logits,
            sample_rows=sample_rows,
            label_positions=label_positions,
            loss_mask=loss_mask,
            position_decay=0.5,
            token_chunk_size=2,
            tp_group=None,
        )

    assert agreement.call_count == 1
