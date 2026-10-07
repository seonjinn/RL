#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --time=00:30:00
#SBATCH --mem=0
set -euo pipefail
: "${REPO:?Set the existing baseline checkout}"
: "${BASE_COMMIT:?Set the archived baseline source commit}"
: "${SOURCE_COMMIT:?Set the pushed optimization source commit}"
: "${BASE_ARCHIVE:?Set the immutable baseline source archive}"
: "${SOURCE_ARCHIVE:?Set a new optimization source archive path}"
test -f "$BASE_ARCHIVE"
test ! -e "$SOURCE_ARCHIVE"

git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
test "$(git -C "$REPO" rev-parse HEAD)" = "$BASE_COMMIT"
git -C "$REPO" fetch origin refs/heads/codex/pr3294-lightning-gbs512-20261007
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$REPO" diff --name-only "$BASE_COMMIT" "$SOURCE_COMMIT" -- \
  pyproject.toml uv.lock nemo_rl/distributed/actor_environments.py 3rdparty)"

local_root="/raid/scratch/${USER}/nr-opt-prepare-${SLURM_JOB_ID}"
source_root="${local_root}/source"
patch_file="${local_root}/source.patch"
mkdir -p "$source_root" "$(dirname "$SOURCE_ARCHIVE")"
git -C "$REPO" diff --binary "$BASE_COMMIT" "$SOURCE_COMMIT" > "$patch_file"
git -C "$REPO" diff --check "$BASE_COMMIT" "$SOURCE_COMMIT"
tar -xf "$BASE_ARCHIVE" -C "$source_root"
git -C "$source_root" apply --check "$patch_file"
git -C "$source_root" apply "$patch_file"
tar -cf "${SOURCE_ARCHIVE}.partial" -C "$source_root" .
mv "${SOURCE_ARCHIVE}.partial" "$SOURCE_ARCHIVE"
cp "${BASE_ARCHIVE}.fingerprint.json" "${SOURCE_ARCHIVE}.fingerprint.json"
{
  printf 'base_commit=%s\nsource_commit=%s\n' "$BASE_COMMIT" "$SOURCE_COMMIT"
  sha256sum "$BASE_ARCHIVE" "$patch_file" "$SOURCE_ARCHIVE"
} > "${SOURCE_ARCHIVE}.metadata.txt"
