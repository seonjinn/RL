# Qwen3 8B online drafter efficiency

`base_contract.json` freezes the terminal-GREEN base shared by the online drafter
efficiency tasks. It records the product commit, immutable container digest, and
two durable PASS receipts: the full gate and packed end-to-end gate.

To regenerate the receipt, provide JSON for both gate environment variables and
the container digest:

```bash
uv run research/qwen3_8b_online_drafter_efficiency/base_contract.py record \
  --full-gate-receipt "${NRL_EFFICIENCY_FULL_GATE_RECEIPT:?required}" \
  --packed-e2e-receipt "${NRL_EFFICIENCY_PACKED_E2E_RECEIPT:?required}" \
  --container-sha256 "${NRL_EFFICIENCY_CONTAINER_SHA256:?required}" \
  --output research/qwen3_8b_online_drafter_efficiency/base_contract.json
```

Validate the recorded contract against the checkout before consuming it:

```bash
uv run research/qwen3_8b_online_drafter_efficiency/base_contract.py validate \
  --contract research/qwen3_8b_online_drafter_efficiency/base_contract.json \
  --current-head "$(git rev-parse HEAD)"
```
