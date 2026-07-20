#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


GYM_APP_RELATIVE_PATH = Path(
    "3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py"
)
LEGACY_TMUX_BLOCK = '''            "export TMUX_TMPDIR=/tmp && "
            "export TMUX=/tmp/tmux-$uid/default && "
            "mkdir -p /tmp/tmux-$uid && "
            "chown $uid:$uid /tmp/tmux-$uid || true && "
            "chmod 700 /tmp/tmux-$uid && "
            "tmux -S /tmp/tmux-$uid/default start-server || true && "'''
FIXED_TMUX_BLOCK = '''            "export TMUX_TMPDIR=/tmp && "
            "unset TMUX && "'''
LEGACY_JQ_BLOCK = '''            "cp /openhands_setup/miniforge3/bin/jq /usr/local/bin/jq 2>/dev/null || true && "'''
FIXED_JQ_BLOCK = '''            "mkdir -p /tmp/nemorl-native-tools && "
            "if /usr/bin/jq --version >/dev/null 2>&1; then "
            "ln -sf /usr/bin/jq /tmp/nemorl-native-tools/jq; "
            "elif /openhands_setup/miniforge3/bin/jq --version >/dev/null 2>&1; then "
            "ln -sf /openhands_setup/miniforge3/bin/jq /tmp/nemorl-native-tools/jq; "
            "else echo 'No runnable jq found for OpenHands SWE setup' >&2; exit 126; fi && "
            "export PATH=/tmp/nemorl-native-tools:$PATH && "'''


def patch_gym_openhands_tmux_source(source: str) -> str:
    """Remove a stale tmux client socket that prevents libtmux startup."""
    legacy_count = source.count(LEGACY_TMUX_BLOCK)
    fixed_count = source.count(FIXED_TMUX_BLOCK)
    if legacy_count == 1 and fixed_count == 0:
        return source.replace(LEGACY_TMUX_BLOCK, FIXED_TMUX_BLOCK)
    if legacy_count == 0 and fixed_count == 1:
        return source
    raise ValueError(
        "expected Gym tmux setup block exactly once; "
        f"found legacy={legacy_count}, fixed={fixed_count}"
    )


def patch_gym_openhands_jq_source(source: str) -> str:
    """Select a runnable jq instead of trusting a cached host architecture."""
    legacy_count = source.count(LEGACY_JQ_BLOCK)
    fixed_count = source.count(FIXED_JQ_BLOCK)
    if legacy_count == 1 and fixed_count == 0:
        return source.replace(LEGACY_JQ_BLOCK, FIXED_JQ_BLOCK)
    if legacy_count == 0 and fixed_count == 1:
        return source
    raise ValueError(
        "expected Gym jq setup block exactly once; "
        f"found legacy={legacy_count}, fixed={fixed_count}"
    )


def patch_gym_openhands_source(source: str) -> str:
    patched = patch_gym_openhands_tmux_source(source)
    return patch_gym_openhands_jq_source(patched)


def apply_gym_openhands_runtime_fix(repo_dir: Path) -> tuple[Path, bool, str]:
    app_path = repo_dir / GYM_APP_RELATIVE_PATH
    source = app_path.read_text()
    patched = patch_gym_openhands_source(source)
    changed = patched != source
    if changed:
        app_path.write_text(patched)
    digest = hashlib.sha256(patched.encode()).hexdigest()
    return app_path, changed, digest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply NeMo Gym OpenHands runtime compatibility fixes"
    )
    parser.add_argument("--repo-dir", type=Path, required=True)
    args = parser.parse_args()
    app_path, changed, digest = apply_gym_openhands_runtime_fix(args.repo_dir)
    print(f"path={app_path}")
    print(f"changed={str(changed).lower()}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
