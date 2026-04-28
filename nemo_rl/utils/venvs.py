# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import shlex
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import ray
from ray.util import placement_group

dir_path = os.path.dirname(os.path.abspath(__file__))
git_root = os.path.abspath(os.path.join(dir_path, "../.."))
DEFAULT_VENV_DIR = os.path.join(git_root, "venvs")

logger = logging.getLogger(__name__)
_UV_RUN_PROJECT_FLAGS_WITH_VALUES = {
    "--extra",
    "--group",
    "--directory",
    "--project",
    "--config-file",
}
_VLLM_BUILD_OVERRIDE_ENV_VARS_TO_STRIP = {
    "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM",
    "VLLM_PRECOMPILED_WHEEL_COMMIT",
    "VLLM_PRECOMPILED_WHEEL_VARIANT",
    "VLLM_USE_PRECOMPILED",
}


def _prepare_uv_environment_commands(
    py_executable: str, python_path: str
) -> tuple[list[str], list[str], list[str]]:
    """Derive scoped bootstrap/install commands from a ``uv run`` template."""
    exec_cmd = shlex.split(py_executable)
    if exec_cmd[:2] != ["uv", "run"]:
        return [], [], exec_cmd

    options = exec_cmd[2:]
    project_path = git_root
    extras: list[str] = []
    groups: list[str] = []
    config_file: str | None = None

    i = 0
    while i < len(options):
        arg = options[i]
        if arg in _UV_RUN_PROJECT_FLAGS_WITH_VALUES and i + 1 < len(options):
            value = options[i + 1]
            if arg in {"--directory", "--project"}:
                project_path = value
            elif arg == "--extra":
                extras.append(value)
            elif arg == "--group":
                groups.append(value)
            elif arg == "--config-file":
                config_file = value
            i += 2
            continue
        i += 1

    build_cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        python_path,
        "--project",
        project_path,
        "--group",
        "build",
    ]
    install_cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        python_path,
        "--project",
        project_path,
        "--editable",
    ]

    if config_file is not None:
        build_cmd.extend(["--config-file", config_file])
        install_cmd.extend(["--config-file", config_file])
    editable_target = project_path
    if extras:
        editable_target = f"{project_path}[{','.join(extras)}]"
    install_cmd.append(editable_target)
    for group in groups:
        install_cmd.extend(["--group", group])

    exec_cmd = exec_cmd.copy()
    if "--no-sync" not in exec_cmd:
        exec_cmd.append("--no-sync")

    return build_cmd, install_cmd, exec_cmd


def _py_executable_requests_extra(py_executable: str, extra_name: str) -> bool:
    """Return whether a ``uv run`` template requests a given extra."""
    exec_cmd = shlex.split(py_executable)
    if exec_cmd[:2] != ["uv", "run"]:
        return False

    options = exec_cmd[2:]
    i = 0
    while i < len(options):
        arg = options[i]
        if arg == "--extra" and i + 1 < len(options):
            if options[i + 1] == extra_name:
                return True
            i += 2
            continue
        if arg in _UV_RUN_PROJECT_FLAGS_WITH_VALUES and i + 1 < len(options):
            i += 2
            continue
        i += 1
    return False


def _prepare_uv_install_env(
    base_env: dict[str, str], py_executable: str
) -> dict[str, str]:
    """Prepare the environment for editable uv installs.

    For vLLM actor envs, preserve the container-pinned wheel location but strip
    stale version/commit overrides that can steer the editable install toward an
    incompatible native extension.
    """
    env = base_env.copy()
    if _py_executable_requests_extra(py_executable, "vllm"):
        for key in _VLLM_BUILD_OVERRIDE_ENV_VARS_TO_STRIP:
            env.pop(key, None)
    return env


def _prepare_uv_bootstrap_packages(py_executable: str) -> list[str]:
    """Prepare seed packages for a fresh uv worker environment."""
    packages = ["setuptools", "setuptools_scm", "torch==2.10.0"]
    if _py_executable_requests_extra(py_executable, "vllm"):
        packages.extend(["cmake>=3.26.1", "ninja"])
    return packages


@lru_cache(maxsize=None)
def create_local_venv(
    py_executable: str, venv_name: str, force_rebuild: bool = False
) -> str:
    """Create a virtual environment using uv and execute a command within it.

    The output can be used as a py_executable for a Ray worker assuming the worker
    nodes also have access to the same file system as the head node.

    This function is cached to avoid multiple calls to uv to create the same venv,
    which avoids duplicate logging.

    Args:
        py_executable (str): Command to run with the virtual environment (e.g., "uv.sh run --locked")
        venv_name (str): Name of the virtual environment (e.g., "foobar.Worker")
        force_rebuild (bool): If True, force rebuild the venv even if it already exists

    Returns:
        str: Path to the python executable in the created virtual environment
    """
    # This directory is where virtual environments will be installed
    # It is local to the driver process but should be visible to all worker nodes
    # If this directory is not accessible from worker nodes (e.g., on a distributed
    # cluster with non-shared filesystems), you may encounter errors when workers
    # try to access the virtual environments
    #
    # You can override this location by setting the NEMO_RL_VENV_DIR environment variable

    NEMO_RL_VENV_DIR = os.path.normpath(
        os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    )
    logger.info(f"NEMO_RL_VENV_DIR is set to {NEMO_RL_VENV_DIR}.")

    # Create the venv directory if it doesn't exist
    os.makedirs(NEMO_RL_VENV_DIR, exist_ok=True)

    # Full path to the virtual environment
    venv_path = os.path.join(NEMO_RL_VENV_DIR, venv_name)

    # Force rebuild if requested
    if force_rebuild and os.path.exists(venv_path):
        logger.info(f"Force rebuilding venv at {venv_path}")
        shutil.rmtree(venv_path)

    # Optional fast path: when ``NRL_VENVS_TRUST_EXISTING=1`` and the venv
    # directory + python interpreter already exist (e.g. the container
    # ships pre-built venvs at ``/opt/ray_venvs/<actor_class>``), skip the
    # ``uv venv`` + ``uv pip install --editable .[extras]`` re-resolution
    # entirely and trust what's on disk. This avoids re-hitting private
    # package indexes (e.g. ``flashinfer-jit-cache==0.6.5+cu129`` from the
    # internal NVIDIA pypi) on every actor spawn when the container
    # already has the right wheels installed.
    trust_existing = os.environ.get("NRL_VENVS_TRUST_EXISTING", "0") == "1"
    if trust_existing and not force_rebuild:
        candidate_python = os.path.join(venv_path, "bin", "python")
        if os.path.exists(candidate_python):
            logger.info(
                f"NRL_VENVS_TRUST_EXISTING=1: reusing existing venv at "
                f"{venv_path} without re-running uv install"
            )
            return candidate_python

    logger.info(f"Creating new venv at {venv_path}")

    # Create the virtual environment
    uv_venv_cmd = ["uv", "venv", "--allow-existing", venv_path]
    subprocess.run(uv_venv_cmd, check=True)

    # Execute the command with the virtual environment
    env = os.environ.copy()
    # NOTE: UV_PROJECT_ENVIRONMENT is appropriate here only b/c there should only be
    #  one call to this in the driver. It is not safe to use this in a multi-process
    #  context.
    #  https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
    env["UV_PROJECT_ENVIRONMENT"] = venv_path

    python_path = os.path.join(venv_path, "bin", "python")
    build_cmd, install_cmd, exec_cmd = _prepare_uv_environment_commands(
        py_executable, python_path
    )
    uv_env = _prepare_uv_install_env(env, py_executable)
    # Command doesn't matter; we only use it as a final `uv run --no-sync` sanity check.
    exec_cmd.extend(["echo", f"Finished creating venv {venv_path}"])

    # TODO: After Gate 6 passes on the ultra-rl-v0.17 container, re-check whether
    # this local bootstrap is still needed or whether we can rely entirely on the
    # container-baked torch/build toolchain for fresh Ray worker envs.
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--torch-backend=cu129",
            *_prepare_uv_bootstrap_packages(py_executable),
        ],
        env=uv_env | {"VIRTUAL_ENV": venv_path},
        check=True,
    )

    # Workaround for transformer-engine + uv build sandbox.
    #
    # uv cannot build ``transformer-engine-torch`` from source (its
    # no-build-isolation mode does not set ``sys.path`` correctly for the
    # ``build_tools`` package, and the build sandbox is missing CUDA
    # headers). For that reason ``pyproject.toml`` pins bare
    # ``transformer-engine==2.12.0`` (no ``[pytorch]`` extra) in the
    # ``mcore`` / ``automodel`` extras and ``override-dependencies``, so
    # the workspace ``uv sync`` below does not try to pull
    # ``transformer-engine-torch`` through the broken uv build path.
    #
    # Here -- only for actor venvs that actually need TE (i.e. those
    # built against the ``mcore`` or ``automodel`` extras) -- we manually
    # install the three TE packages with ``--no-deps`` so the venv ends
    # up with ``libtransformer_engine.so`` (cu12 backend) and
    # ``libtransformer_engine_torch.so`` (PyTorch C++ extension), which
    # Megatron / TE-PyTorch import at runtime. Without this step the
    # venv only contains the bare ``transformer_engine`` Python wheel
    # plus ``wheel_lib/libtransformer_engine.so`` and Megatron actors
    # die with ``Could not find shared object file for Transformer
    # Engine torch lib``.
    #
    # Mirrors the Omni branch's
    # ``nemo-rl-omni/nemo_rl/utils/venvs.py:create_local_venv`` (which
    # uses the 2.9.0 line; we are pinned to 2.12.0).
    needs_te = _py_executable_requests_extra(
        py_executable, "mcore"
    ) or _py_executable_requests_extra(py_executable, "automodel")
    venv_python = os.path.join(venv_path, "bin", "python")

    if needs_te:
        # Pre-install the bare ``transformer-engine`` Python wheel BEFORE
        # the workspace ``uv sync`` runs, so that uv sees TE already
        # satisfied and does not attempt to re-resolve / re-build it.
        subprocess.run(
            ["uv", "pip", "install", "transformer-engine==2.12.0", "--no-deps"],
            env=uv_env | {"VIRTUAL_ENV": venv_path},
            check=True,
        )

    if build_cmd:
        subprocess.run(build_cmd, env=uv_env, check=True)
    if install_cmd:
        subprocess.run(install_cmd, env=uv_env, check=True)

    if needs_te:
        # Install the cu12 backend + PyTorch C++ extension wheels AFTER
        # the workspace install. ``pip install --no-deps
        # --no-build-isolation`` pulls down the prebuilt wheels (no
        # source build) and drops the required ``.so`` files into the
        # venv site-packages. Use ``venv_python -m pip`` (not ``uv pip``)
        # to ensure the installs land in this venv specifically.
        subprocess.run(
            [
                venv_python, "-m", "pip", "install",
                "transformer-engine-cu12==2.12.0",
                "--no-deps",
            ],
            env=uv_env,
            check=True,
        )
        subprocess.run(
            [
                venv_python, "-m", "pip", "install",
                "transformer-engine-torch==2.12.0",
                "--no-build-isolation",
                "--no-deps",
            ],
            env=uv_env,
            check=True,
        )

    subprocess.run(exec_cmd, env=uv_env, check=True)

    # Return the path to the python executable in the virtual environment
    return python_path


# Ray-based helper to create a virtual environment on each Ray node
@ray.remote(num_cpus=1)  # pragma: no cover
def _env_builder(
    py_executable: str, venv_name: str, node_idx: int, force_rebuild: bool = False
):
    # Check if another node is already building
    NEMO_RL_VENV_DIR = os.path.normpath(
        os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    )
    venv_path = Path(NEMO_RL_VENV_DIR) / venv_name
    python_path = venv_path / "bin" / "python"
    started_file = venv_path / "STARTED_ENV_BUILDER"

    # Skip early return if force_rebuild is True
    if not force_rebuild and python_path.exists():
        logger.info(f"Using existing venv at {venv_path}")
        return str(python_path)

    # Sleep to stagger node startup
    time.sleep(1 * node_idx)

    if started_file.exists():
        # Another node is already building, wait for completion
        logger.info(
            f"Node {node_idx}: Another node is building {venv_name}, skipping..."
        )
        # Wait for the venv to be ready (check for python executable)
        python_path = venv_path / "bin" / "python"
        while not python_path.exists():
            time.sleep(1)
        return str(python_path)

    # Create the venv directory if needed
    venv_path.mkdir(parents=True, exist_ok=True)

    # Touch the started file to signal we're building
    started_file.touch()
    try:
        # Create the virtual environment on this node
        return create_local_venv(py_executable, venv_name, force_rebuild=force_rebuild)
    finally:
        # Clean up the started file
        if started_file.exists():
            started_file.unlink()


def create_local_venv_on_each_node(py_executable: str, venv_name: str):
    """Create a virtual environment on each Ray node.

    Args:
        py_executable (str): Command to run with the virtual environment
        venv_name (str): Name of the virtual environment

    Returns:
        str: Path to the python executable in the created virtual environment
    """
    # Determine the number of alive Ray nodes
    nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    num_nodes = len(nodes)
    # Reserve one CPU on each node using a STRICT_SPREAD placement group
    bundles = [{"CPU": 1} for _ in range(num_nodes)]
    pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    force_rebuild = os.environ.get("NRL_FORCE_REBUILD_VENVS", "false").lower() == "true"
    # Launch one actor per node
    actors = [
        _env_builder.options(placement_group=pg).remote(
            py_executable, venv_name, i, force_rebuild
        )
        for i, _ in enumerate(nodes)
    ]
    # ensure setup runs on each node
    paths = ray.get([actor for actor in actors])
    # Normalize paths to handle double slashes and other path inconsistencies
    normalized_paths = [os.path.normpath(p) for p in paths]
    assert len(set(normalized_paths)) == 1, (
        f"All nodes should have the same venv, but got: {set(normalized_paths)}"
    )

    # Clean up the placement group
    ray.util.remove_placement_group(pg)
    # Return mapping from node IP to venv python path
    return paths[0]
