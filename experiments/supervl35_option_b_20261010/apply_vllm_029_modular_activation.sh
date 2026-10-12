#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?Pass the vLLM worker Python executable}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
site_packages=$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
target="$site_packages/vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py"
test -f "$target"

if grep -Fq 'activation_type=activation_to_flashinfer_int(activation),' "$target"; then
    echo "vLLM modular activation fix already installed: $target"
    exit 0
fi

patch --batch -p1 -d "$site_packages" < "$script_dir/vllm_029_modular_activation.patch"
grep -Fq 'activation_type=activation_to_flashinfer_int(activation),' "$target"
echo "Installed vLLM modular activation fix: $target"
