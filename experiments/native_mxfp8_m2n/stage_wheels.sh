#!/usr/bin/env bash
set -euo pipefail

: "${M2N_WHEELHOUSE:?Set a durable directory for the two pinned wheel artifacts}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${M2N_WHEELHOUSE}"
cd "${M2N_WHEELHOUSE}"
urls=(
  https://files.pythonhosted.org/packages/22/27/568725ca5b767342b4e52795e081d01e2adb31ad647a796260b8a6c8a994/nccl4py-0.5.0-cp313-cp313-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl
  https://files.pythonhosted.org/packages/90/8b/0d98ca96c8625af88d834230b34115fac032ed69eb84c8bde7c38b956d16/nccl_extensions-0.1.0-cp313-cp313-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl
)
for url in "${urls[@]}"; do
  wheel=${url##*/}
  if [[ ! -f "${wheel}" ]]; then
    curl --fail --location --retry 2 --max-time 60 --output "${wheel}.partial" "${url}"
    mv "${wheel}.partial" "${wheel}"
  fi
done
sha256sum --check "${SCRIPT_DIR}/wheels.sha256"
