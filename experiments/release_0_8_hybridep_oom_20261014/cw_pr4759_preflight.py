"""Validate PR 4759 MiniMax CP8 topology in the pinned release image."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate
from nemo_automodel.components.models.minimax_m2.model import MiniMaxM2ForCausalLM

sys.path.insert(0, "/opt/nemo-rl")
from nemo_rl.utils.config import load_config  # noqa: E402


recipe = (
    Path(__file__).parents[2]
    / "examples/configs/recipes/llm/grpo-minimax-m27-dapo-8n8g-automodel-ep64.yaml"
)
cfg = load_config(recipe)
assert cfg.policy.make_sequence_length_divisible_by == 512
assert cfg.policy.dynamic_batching.sequence_length_round == 512
assert cfg.policy.optimizer.kwargs.foreach is False
assert cfg.policy.automodel_cfg.context_parallel_size == 8
assert cfg.policy.automodel_cfg.expert_parallel_size == 64
assert cfg.policy.automodel_cfg.tensor_parallel_size == 1

model = MiniMaxM2ForCausalLM.__new__(MiniMaxM2ForCausalLM)
torch.nn.Module.__init__(model)
model.backend = SimpleNamespace(attn="te")
mesh = SimpleNamespace(tp_size=1, cp_size=8, pp_size=1, ep_size=64)
attach_capabilities_and_validate(model, mesh)

for length in range(1, 10241):
    padded_length = ((length + 511) // 512) * 512
    assert padded_length // 8 % 64 == 0

print("PR 4759 MiniMax CP8/EP64 recipe and capability validation passed", flush=True)
