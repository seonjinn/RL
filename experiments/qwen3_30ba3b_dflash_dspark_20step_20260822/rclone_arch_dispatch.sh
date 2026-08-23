#!/usr/bin/env bash
set -euo pipefail

readonly architecture="${RCLONE_ARCH_OVERRIDE:-$(uname -m)}"
case "${architecture}" in
  x86_64|amd64)
    exec "${RCLONE_AMD64_BIN:-/home/sna/.local/lib/rclone/rclone-amd64}" "$@"
    ;;
  aarch64|arm64)
    exec "${RCLONE_ARM64_BIN:-/home/sna/.local/lib/rclone/rclone-arm64}" "$@"
    ;;
  *)
    echo "unsupported rclone architecture: ${architecture}" >&2
    exit 1
    ;;
esac
