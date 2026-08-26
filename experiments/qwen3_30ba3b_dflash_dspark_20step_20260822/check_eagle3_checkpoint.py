"""Validate the pinned EAGLE-3 checkpoint against its intended verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--target-model", required=True)
parser.add_argument("--num-speculative-tokens", type=int, required=True)
args = parser.parse_args()

config_path = args.checkpoint / "config.json"
weights_path = args.checkpoint / "model.safetensors"
if not config_path.is_file() or not weights_path.is_file():
    raise SystemExit("EAGLE-3 checkpoint must contain config.json and model.safetensors")

config = json.loads(config_path.read_text())
if config.get("architectures") != ["Eagle3DraftModel"]:
    raise SystemExit("EAGLE-3 architecture mismatch")
speculators = config.get("speculators_config")
if not isinstance(speculators, dict) or speculators.get("algorithm") != "eagle3":
    raise SystemExit("EAGLE-3 algorithm mismatch")
verifier = speculators.get("verifier")
actual_target = verifier.get("name_or_path") if isinstance(verifier, dict) else None
if actual_target != args.target_model:
    raise SystemExit(
        f"verifier target mismatch: {actual_target!r} expected={args.target_model!r}"
    )
proposal_methods = speculators.get("proposal_methods")
tokens = {
    proposal.get("speculative_tokens")
    for proposal in proposal_methods or []
    if isinstance(proposal, dict)
}
if args.num_speculative_tokens not in tokens:
    raise SystemExit(
        "EAGLE-3 checkpoint does not declare the requested speculative token count"
    )
print(
    "EAGLE3_CHECKPOINT_GATE "
    f"target={actual_target} num_speculative_tokens={args.num_speculative_tokens}"
)
