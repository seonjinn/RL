#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?Pass the vLLM 0.31 Python executable}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
version=$("$python_bin" -c 'import vllm; print(vllm.__version__)')
test "$version" = 0.31.0 || { echo "Expected vLLM 0.31.0, got $version" >&2; exit 1; }
site_packages=$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
target="$site_packages/vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py"
test -f "$target"

if grep -Fq 'activation_type=activation_to_flashinfer_int(activation),' "$target"; then
    echo "vLLM 0.31 modular activation fix already installed: $target"
    exit 0
fi

patch --batch -p1 -d "$site_packages" < "$script_dir/vllm_031_modular_activation.patch"
grep -Fq 'activation_type=activation_to_flashinfer_int(activation),' "$target"
echo "Installed vLLM 0.31 modular activation fix: $target"
