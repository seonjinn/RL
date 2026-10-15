from types import SimpleNamespace

import torch
from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate
from nemo_automodel.components.models.minimax_m2.model import MiniMaxM2ForCausalLM


model = MiniMaxM2ForCausalLM.__new__(MiniMaxM2ForCausalLM)
torch.nn.Module.__init__(model)
mesh = SimpleNamespace(tp_size=8, cp_size=1, pp_size=1, ep_size=64)

print(f"MiniMax-M2 declared TP support: {MiniMaxM2ForCausalLM.ModelCapabilities.supports_tp}")
print("Validating the TP8 EP64 model mesh in the pinned release container")
attach_capabilities_and_validate(model, mesh)
print("TP8 EP64 model capability validation passed")
