#!/bin/sh
set -eu

arch="${RCLONE_ARCH:-$(uname -m)}"
root="${RCLONE_ROOT:-${HOME}/.local/lib/rclone}"

case "${arch}" in
  x86_64|amd64)
    binary="${root}/rclone-amd64"
    ;;
  aarch64|arm64)
    binary="${root}/rclone-arm64"
    ;;
  *)
    echo "unsupported architecture: ${arch}" >&2
    exit 1
    ;;
esac

exec "${binary}" "$@"
