#!/usr/bin/env python3
"""Probe whether dumped Eagle3 hidden states align with the verifier lm_head.

This is intentionally cheaper than a full drafter validation: it loads only the
verifier lm_head/embed weights through ModelOpt FakeBaseModel, projects sampled
hidden states, and checks next-token CE/top-k under several token shifts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-states-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--modelopt-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--sample-files", type=int, default=8)
    parser.add_argument("--max-positions-per-file", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--shifts",
        default="-1,0,1,2",
        help="Comma-separated label_pos=context_pos+shift candidates.",
    )
    return parser.parse_args()


def tensor_summary(x: torch.Tensor) -> dict[str, Any]:
    xf = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(xf.mean().item()),
        "std": float(xf.std().item()) if x.numel() > 1 else 0.0,
        "norm_mean": float(torch.linalg.vector_norm(xf, dim=-1).mean().item())
        if x.ndim >= 2
        else None,
    }


def add_counts(dst: dict[str, float], *, ce: float, top1: int, top5: int, total: int) -> None:
    dst["ce_sum"] += ce
    dst["top1"] += top1
    dst["top5"] += top5
    dst["tokens"] += total


def finalize_counts(src: dict[str, float]) -> dict[str, Any]:
    tokens = int(src["tokens"])
    if tokens == 0:
        return {"tokens": 0, "cross_entropy": None, "perplexity": None, "top1": None, "top5": None}
    ce = src["ce_sum"] / tokens
    ppl = math.exp(ce) if ce < 80 else float("inf")
    return {
        "tokens": tokens,
        "cross_entropy": ce,
        "perplexity": ppl,
        "top1": src["top1"] / tokens,
        "top5": src["top5"] / tokens,
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.modelopt_dir))

    from modelopt.torch.speculative.plugins.modeling_fakebase import FakeBaseModel

    shifts = [int(x.strip()) for x in args.shifts.split(",") if x.strip()]
    files = sorted(args.hidden_states_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files found under {args.hidden_states_dir}")
    rng = random.Random(args.seed)
    if args.sample_files > 0 and len(files) > args.sample_files:
        files = rng.sample(files, args.sample_files)
    files = sorted(files)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    base = FakeBaseModel.from_source(args.base_model, trust_remote_code=args.trust_remote_code)
    lm_head = base.lm_head.to(device).eval()
    vocab_size = int(base.config.vocab_size)

    counts_by_shift_and_mask: dict[str, dict[str, float]] = defaultdict(
        lambda: {"ce_sum": 0.0, "top1": 0.0, "top5": 0.0, "tokens": 0.0}
    )
    file_summaries: list[dict[str, Any]] = []

    for path in files:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        input_ids = obj["input_ids"].long()
        hiddens = obj["hidden_states"]
        loss_mask = obj.get("loss_mask")
        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids)
        else:
            loss_mask = loss_mask.long()

        seq_len = int(input_ids.shape[0])
        file_summary: dict[str, Any] = {
            "path": str(path),
            "seq_len": seq_len,
            "loss_mask_tokens": int(loss_mask.sum().item()),
            "hidden_states": tensor_summary(hiddens),
        }
        if "aux_hidden_states" in obj:
            file_summary["aux_hidden_states"] = tensor_summary(obj["aux_hidden_states"])

        for shift in shifts:
            all_context_pos = torch.arange(seq_len)
            label_pos = all_context_pos + shift
            valid = (label_pos >= 0) & (label_pos < seq_len)
            context_pos = all_context_pos[valid]
            label_pos = label_pos[valid]
            if context_pos.numel() == 0:
                continue

            mask_variants = {
                "all": torch.ones_like(context_pos, dtype=torch.bool),
                "loss_mask_at_context": loss_mask[context_pos].bool(),
                "loss_mask_at_label": loss_mask[label_pos].bool(),
            }
            for mask_name, mask in mask_variants.items():
                pos = context_pos[mask]
                lab_pos = label_pos[mask]
                if pos.numel() == 0:
                    continue
                if args.max_positions_per_file > 0 and pos.numel() > args.max_positions_per_file:
                    keep = torch.tensor(
                        rng.sample(range(pos.numel()), args.max_positions_per_file), dtype=torch.long
                    )
                    pos = pos[keep]
                    lab_pos = lab_pos[keep]

                key = f"shift_{shift}:{mask_name}"
                for start in range(0, pos.numel(), args.chunk_size):
                    p = pos[start : start + args.chunk_size]
                    lp = lab_pos[start : start + args.chunk_size]
                    labels = input_ids[lp].to(device)
                    logits = lm_head(hiddens[p].to(device))
                    ce = F.cross_entropy(logits.float(), labels, reduction="sum")
                    topk = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
                    pred = topk[:, 0]
                    add_counts(
                        counts_by_shift_and_mask[key],
                        ce=float(ce.item()),
                        top1=int((pred == labels).sum().item()),
                        top5=int((topk == labels[:, None]).any(dim=-1).sum().item()),
                        total=int(labels.numel()),
                    )

        file_summaries.append(file_summary)

    results = {
        "status": "pass",
        "base_model": args.base_model,
        "hidden_states_dir": str(args.hidden_states_dir),
        "sampled_files": len(files),
        "sample_files": [str(p) for p in files],
        "device": str(device),
        "vocab_size": vocab_size,
        "metrics": {
            key: finalize_counts(value) for key, value in sorted(counts_by_shift_and_mask.items())
        },
        "files": file_summaries,
        "interpretation": {
            "expected_good_shift": "shift_1 with loss_mask_at_label/context should usually be the best next-token sanity check.",
            "zero_acceptance_hint": "If every shift has near-zero top-k and very high CE, the hidden states/lm_head/token alignment is suspect; retraining longer is unlikely to fix acceptance.",
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"status": "pass", "json_out": str(args.json_out), "metrics": results["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
