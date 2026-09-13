"""Inspect actual TE BF16 state aliases before implementing CPU snapshot reuse."""

import inspect
import json
import os

import torch
from transformer_engine.pytorch.module import GroupedLinear


def payload(tensor: torch.Tensor) -> torch.Tensor:
    rowwise = getattr(tensor, "rowwise_data", None)
    return rowwise if rowwise is not None else tensor


def probe(single: bool) -> dict[str, object]:
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1" if single else "0"
    module = GroupedLinear(
        num_gemms=2,
        in_features=128,
        out_features=256,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        use_grouped_tensor=single,
    )
    owners = dict(module.named_parameters())
    with torch.no_grad():
        for param in owners.values():
            payload(param).fill_(2)
        state = module.state_dict()
        records = []
        for name, value in state.items():
            if "extra_state" in name or not isinstance(value, torch.Tensor):
                continue
            owner = owners.get(name)
            if owner is None:
                raise AssertionError(f"State key has no exact parameter owner: {name}")
            live = payload(owner)
            saved = payload(value)
            same_storage = live.untyped_storage().data_ptr() == saved.untyped_storage().data_ptr()
            snapshot = live.detach().to("cpu", copy=True)
            live.fill_(7)
            sees_mutation = bool(torch.all(saved == 7).item())
            live.copy_(snapshot)
            torch.testing.assert_close(live, torch.full_like(live, 2), rtol=0, atol=0)
            records.append({
                "name": name,
                "owner_type": type(owner).__name__,
                "state_type": type(value).__name__,
                "owner_shape": list(owner.shape),
                "payload_shape": list(live.shape),
                "state_shape": list(value.shape),
                "same_storage": same_storage,
                "state_sees_live_mutation": sees_mutation,
                "restored_exactly": True,
            })
        if not records:
            raise AssertionError("No weight entries inspected")
    return {"single": single, "records": records}


if __name__ == "__main__":
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "signature": str(inspect.signature(GroupedLinear))}))
    for single in (False, True):
        print(json.dumps(probe(single)), flush=True)
