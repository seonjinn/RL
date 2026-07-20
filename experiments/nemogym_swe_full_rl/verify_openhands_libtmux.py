#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import libtmux


def main() -> int:
    socket_root = Path(tempfile.mkdtemp(prefix="nemorl-openhands-tmux-"))
    os.environ["TMUX_TMPDIR"] = str(socket_root)
    os.environ.pop("TMUX", None)
    session_name = f"nemorl-smoke-{uuid.uuid4().hex[:12]}"
    server = libtmux.Server()
    session = server.new_session(
        session_name=session_name,
        attach=False,
        kill_session=True,
    )
    session.cmd("display-message", "-p", "#{session_name}")
    observed_names = {item.name for item in server.sessions}
    if session_name not in observed_names:
        raise RuntimeError(
            f"libtmux session {session_name!r} missing from {observed_names!r}"
        )
    session.kill()
    print(f"libtmux_session={session_name}")
    print(f"tmux_tmpdir={socket_root}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
