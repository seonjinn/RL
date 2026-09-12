"""Isolated layout diagnostic; does not modify the production implementation."""

import json
import os
import types

import torch
import torch.nn.functional as F
from flashinfer import (
    block_scale_interleave,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)

from nemo_rl.models.generation.vllm.quantization import fp8


def pad_rows(scale: torch.Tensor) -> torch.Tensor:
    return F.pad(scale, (0, 0, 0, (-scale.shape[-2]) % 128))


def run_case(gated: bool, intermediate: int, hidden: int) -> dict:
    experts = 4
    rows = intermediate * (2 if gated else 1)

    def random_bytes(*shape: int) -> torch.Tensor:
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    w13 = random_bytes(experts, rows, hidden).view(torch.float8_e4m3fn)
    w2 = random_bytes(experts, hidden, intermediate).view(torch.float8_e4m3fn)
    s13 = random_bytes(experts, rows, hidden // 32)
    s2 = random_bytes(experts, hidden, intermediate // 32)
    rejected = False
    try:
        fp8._shuffle_mxfp8_moe_batched(
            types.SimpleNamespace(), w13, w2, s13, s2, gated, 128
        )
    except AssertionError:
        rejected = True
    assert rejected == (rows % 128 != 0 or hidden % 128 != 0)

    p13, p2 = fp8._mxfp8_moe_row_permutations(
        types.SimpleNamespace(), w13, w2, gated, 128
    )
    weights = [w13.view(torch.uint8), w2.view(torch.uint8)]
    scales = [s13, s2]
    actual = []
    for weight, scale, perm in zip(weights, scales, (p13, p2)):
        actual.append(torch.index_select(weight, 1, perm))
        shuffled = torch.index_select(fp8.pad_flashinfer_scale_k(scale), 1, perm)
        actual.append(block_scale_interleave(pad_rows(shuffled)).view(experts, -1))

    expected = [[] for _ in range(4)]
    for expert in range(experts):
        first_weight, first_scale = weights[0][expert], s13[expert]
        if gated:
            first_weight = reorder_rows_for_gated_act_gemm(first_weight.clone())
            first_scale = reorder_rows_for_gated_act_gemm(first_scale.clone())
        for slot, weight, scale in (
            (0, first_weight, first_scale),
            (2, weights[1][expert], s2[expert]),
        ):
            expected[slot].append(shuffle_matrix_a(weight, 128))
            padded = pad_rows(fp8.pad_flashinfer_scale_k(scale))
            expected[slot + 1].append(shuffle_matrix_sf_a(padded, 128).flatten())
    for produced, reference_rows in zip(actual, expected):
        reference = torch.stack(reference_rows)
        assert produced.shape == reference.shape, (produced.shape, reference.shape)
        assert torch.equal(produced, reference)
    torch.cuda.synchronize()
    return {
        "gated": gated,
        "intermediate": intermediate,
        "hidden": hidden,
        "existing_path_rejected": rejected,
        "candidate_byte_parity": True,
        "shapes": [list(t.shape) for t in actual],
    }


if __name__ == "__main__":
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    cases = [(True, 128, 256), (False, 672, 1024), (False, 256, 672), (True, 672, 1024)]
    result = run_case(*cases[rank % len(cases)])
    print(json.dumps({"rank": rank, **result}), flush=True)
