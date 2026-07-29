#!/usr/bin/env bash

set -euo pipefail

echo "python=$(python3 --version 2>&1)"
echo "ray_venvs_root=/opt/ray_venvs"

if [[ ! -d /opt/ray_venvs ]]; then
  echo "missing /opt/ray_venvs" >&2
  exit 1
fi

echo "== environments =="
find /opt/ray_venvs -mindepth 2 -maxdepth 2 -type f -name pyvenv.cfg \
  -printf '%h\n' | sort

echo "== python executables =="
find /opt/ray_venvs -mindepth 3 -maxdepth 3 \
  \( -type f -o -type l \) -path '*/bin/python*' \
  -printf '%p -> %l\n' | sort

echo "== sizes =="
du -sh /opt/ray_venvs/* | sort -h

echo "== targeted imports =="
while IFS= read -r executable; do
  environment=${executable%/bin/python}
  case "${environment}" in
    *VllmGenerationWorker*|*VllmQuantGenerationWorker*)
      imports="import modelopt, ray, requests, urllib3, vllm"
      ;;
    *MegatronPolicyWorker*|*MegatronQuantPolicyWorker*)
      imports="import megatron.core, modelopt, ray, requests, torch, urllib3"
      ;;
    *)
      continue
      ;;
  esac

  printf '%s: ' "${environment}"
  if "${executable}" -c "${imports}; print('ok')" 2>&1; then
    :
  else
    echo "failed"
  fi
done < <(
  find /opt/ray_venvs -mindepth 3 -maxdepth 3 \
    \( -type f -o -type l \) -path '*/bin/python' | sort
)
