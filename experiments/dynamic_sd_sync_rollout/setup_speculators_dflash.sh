#!/usr/bin/env bash
# Speculators (DFlash drafter training) setup for AWS-DFW GB200.
# Run on the login node. Assumes uv available or installs it to ~/.local/bin.
set -euo pipefail

BASE="${BASE:?set BASE=/lustre/...path.../users/$USER/dflash_training}"
HF="${HF_HOME:-$BASE/hf_home}"
mkdir -p "$BASE" "$HF"
cd "$BASE"

# 0. uv
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 1. Two venvs (official recipe keeps them separate)
[ -d speculators_venv ] || uv venv speculators_venv --python 3.12
[ -d vllm_venv ] || uv venv vllm_venv --python 3.12
uv pip install --python speculators_venv/bin/python "speculators>=0.5.0" datasets trackio
uv pip install --python vllm_venv/bin/python "vllm>=0.25" flashinfer-python

# 2. Speculators repo (training/prep scripts)
[ -d speculators ] || git clone --depth 1 https://github.com/vllm-project/speculators.git

# 3. Stage target models (sanity: 30B Thinking; main: 235B Thinking)
export HF_HOME="$HF"
speculators_venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for m in [
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    # "Qwen/Qwen3-235B-A22B-Thinking-2507",   # enable for phase 1 (470 GB)
]:
    print("downloading", m)
    snapshot_download(m)
PY

echo "SETUP DONE at $BASE"
echo "Next: bash speculators/examples/train/dflash_qwen3_8b_sharegpt_online_5k.sh (adapted)"
