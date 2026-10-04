#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?Set the committed diagnostic checkout}"
: "${BASE_ARCHIVE:?Set the frozen complete source archive}"
: "${BASE_ARCHIVE_SHA256:?Pin the complete source archive hash}"
: "${OUTPUT_ARCHIVE:?Set a new immutable output archive filename}"
BASE_SHA=d9d1c98014903edaca1b1d1d50f0efadf3cba07c
head=$(git -C "${REPO}" rev-parse HEAD)
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=all)" ]]
git -C "${REPO}" merge-base --is-ancestor "${BASE_SHA}" "${head}"
[[ -z "$(git -C "${REPO}" diff --name-only --diff-filter=D "${BASE_SHA}" "${head}")" ]]
[[ ! -e "${OUTPUT_ARCHIVE}" && ! -e "${OUTPUT_ARCHIVE}.provenance" ]]
[[ "$(sha256sum "${BASE_ARCHIVE}" | cut -d ' ' -f 1)" == "${BASE_ARCHIVE_SHA256}" ]]

# Preserve every pinned submodule from the original archive. Append only the
# committed diagnostic overlay; normal tar extraction selects the last entry.
temporary=$(mktemp "${TMPDIR:-/tmp}/storage-source.XXXXXX.tar")
manifest=$(mktemp "${TMPDIR:-/tmp}/storage-overlay.XXXXXX")
trap 'rm -f "${temporary}" "${manifest}"' EXIT
git -C "${REPO}" diff --name-only -z "${BASE_SHA}" "${head}" > "${manifest}"
cp "${BASE_ARCHIVE}" "${temporary}"
tar --append --file="${temporary}" --null -C "${REPO}" -T "${manifest}"
mkdir -p "$(dirname "${OUTPUT_ARCHIVE}")"
mv "${temporary}" "${OUTPUT_ARCHIVE}"
digest=$(sha256sum "${OUTPUT_ARCHIVE}" | cut -d ' ' -f 1)
printf 'base_sha=%s\nbase_archive=%s\nbase_sha256=%s\noverlay_sha=%s\narchive_sha256=%s\n' \
  "${BASE_SHA}" "${BASE_ARCHIVE}" "${BASE_ARCHIVE_SHA256}" "${head}" "${digest}" \
  > "${OUTPUT_ARCHIVE}.provenance"
printf 'source_payload_sha=%s\nsource_archive=%s\nsource_archive_sha256=%s\n' \
  "${head}" "${OUTPUT_ARCHIVE}" "${digest}"
