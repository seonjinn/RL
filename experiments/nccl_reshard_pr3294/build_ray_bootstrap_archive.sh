#!/usr/bin/env bash

set -euo pipefail

PYTHON_BASE=${PYTHON_BASE:-$(python -c 'import sys; print(sys.base_prefix)')}
SITE_PACKAGES_SOURCE=${SITE_PACKAGES_SOURCE:?SITE_PACKAGES_SOURCE is required}
RAY_ENTRYPOINT=${RAY_ENTRYPOINT:?RAY_ENTRYPOINT is required}
UV_BINARY=${UV_BINARY:?UV_BINARY is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
ARCHIVE_TAG=${ARCHIVE_TAG:-$(date +%Y%m%d-%H%M%S)}

test -x "${PYTHON_BASE}/bin/python3.13"
test -d "${SITE_PACKAGES_SOURCE}"
test -x "${RAY_ENTRYPOINT}"
test -x "${UV_BINARY}"

BUILD_ROOT="${OUTPUT_DIR}/archive-build-${ARCHIVE_TAG}-$$"
ARCHIVE="${OUTPUT_DIR}/ray-bootstrap-${ARCHIVE_TAG}.tar.gz"
ROOT="${BUILD_ROOT}/ray-bootstrap"
VERIFY_ROOT="/tmp/ray-bootstrap-verify-$$"
trap 'rm -rf "${VERIFY_ROOT}"' EXIT

mkdir -p "${ROOT}/lib/python3.13/site-packages"
cp -a "${PYTHON_BASE}/." "${ROOT}/"
cp -a "${SITE_PACKAGES_SOURCE}/." "${ROOT}/lib/python3.13/site-packages/"
cp -a "${RAY_ENTRYPOINT}" "${ROOT}/bin/ray"
cp -a "${UV_BINARY}" "${ROOT}/bin/uv"

# Keep Ray's Python entrypoint relocatable through PATH after node-local extraction.
sed -i \
  "2s#'[^']*/python'#'/usr/bin/env' 'python3.13'#" \
  "${ROOT}/bin/ray"

mkdir -p "${OUTPUT_DIR}"
tar -C "${BUILD_ROOT}" -I "gzip -1" -cf "${ARCHIVE}" ray-bootstrap
sha256sum "${ARCHIVE}" >"${ARCHIVE}.sha256"

mkdir -p "${VERIFY_ROOT}"
tar -xzf "${ARCHIVE}" -C "${VERIFY_ROOT}" --strip-components=1
PATH="${VERIFY_ROOT}/bin:/usr/bin:/bin" \
  "${VERIFY_ROOT}/bin/python3.13" -c \
  'import ray, requests, urllib3; print(ray.__version__, requests.__version__, urllib3.__version__)'
PATH="${VERIFY_ROOT}/bin:/usr/bin:/bin" \
  "${VERIFY_ROOT}/bin/ray" --version

printf 'archive=%s\n' "${ARCHIVE}"
stat -c 'size=%s bytes' "${ARCHIVE}"
cat "${ARCHIVE}.sha256"
