"""Post-sync hook: install arctic-inference (suffix decoding plugin) into the
vLLM worker venv. Runs as {venv}/bin/python via NRL_VENV_POST_SYNC_SCRIPT."""
import subprocess
import sys

subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable, "arctic-inference==0.2.0"],
    check=True,
)
print("[arctic-post-sync] installed arctic-inference into", sys.executable)
