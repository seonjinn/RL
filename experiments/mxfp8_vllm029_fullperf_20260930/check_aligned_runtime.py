"""Audit the driver and every actor used by this performance matrix."""

import argparse
import hashlib
import json
import re
import runpy
import subprocess
import tomllib
from pathlib import Path

MATRIX_ACTORS = (
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
    "nemo_rl.algorithms.async_utils.ReplayBuffer",
    "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
)
UV_TRAMPOLINE = (
    "#!/bin/sh",
    '\'\'\'exec\' "$(dirname -- "$(realpath -- "$0")")"/\'python3\' "$0" "$@"',
    "' '''",
)
PROBE = r"""
import importlib.metadata as md
import json
import pathlib
import platform
import sys
from urllib.parse import unquote, urlparse

import ray

packages = {}
editables = {}
for name in ("ray", "torch", "vllm", "flashinfer-python", "transformer-engine",
             "megatron-core", "megatron-bridge", "nemo-gym", "nvidia-nccl-cu13"):
    try:
        distribution = md.distribution(name)
    except md.PackageNotFoundError:
        packages[name] = None
        continue
    packages[name] = distribution.version
    raw = distribution.read_text("direct_url.json")
    if raw:
        direct = json.loads(raw)
        if direct.get("dir_info", {}).get("editable"):
            path = pathlib.Path(unquote(urlparse(direct["url"]).path))
            editables[name] = {"path": str(path), "exists": path.is_dir()}

print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "python": platform.python_version(),
    "machine": platform.machine(),
    "ray_import_version": ray.__version__,
    "ray_import_file": ray.__file__,
    "python_target": str(pathlib.Path(sys.executable).resolve()),
    "packages": packages,
    "editables": editables,
}))
"""


def environments(root: Path) -> list[tuple[str, Path, list[str]]]:
    registry = runpy.run_path(str(root / "nemo_rl/distributed/actor_environments.py"))[
        "ACTOR_ENVIRONMENTS"
    ]
    return [("driver", Path("/opt/nemo_rl_venv"), [])] + [
        (actor, Path("/opt/ray_venvs") / actor, registry[actor])
        for actor in MATRIX_ACTORS
    ]


def ray_cli_python(ray_cli: Path, *, environment: Path) -> Path:
    lines = ray_cli.read_text().splitlines()
    bin_dir = environment / "bin"
    if tuple(lines[:3]) == UV_TRAMPOLINE:
        interpreter = ray_cli.resolve().parent / "python3"
    elif lines and lines[0].startswith("#!"):
        interpreter = Path(lines[0][2:])
    else:
        raise ValueError("Ray daemon CLI has an unsupported entry point")
    if interpreter.parent != bin_dir or not re.fullmatch(
        r"python(?:\d+(?:\.\d+)?)?", interpreter.name
    ):
        raise ValueError("Ray daemon CLI does not use the driver environment")
    return interpreter


def audit(root: Path, output: Path, inventory_only: bool) -> None:
    lock_bytes = (root / "uv.lock").read_bytes()
    locked_ray = {
        entry["version"]
        for entry in tomllib.loads(lock_bytes.decode())["package"]
        if entry["name"] == "ray"
    }
    assert locked_ray == {"2.58.0"}, locked_ray
    rows = []
    errors = []
    for role, environment, extras in environments(root):
        result = subprocess.run(
            [str(environment / "bin/python"), "-c", PROBE],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        row = json.loads(result.stdout)
        row.update(role=role, extras=extras)
        rows.append(row)
        if row["packages"]["ray"] != "2.58.0" or row["ray_import_version"] != "2.58.0":
            errors.append(f"{role}: Ray does not match uv.lock")
        if row["python"] != "3.13.14" or row["machine"] != "aarch64":
            errors.append(f"{role}: wrong Python/architecture")
        if row["prefix"] != str(environment):
            errors.append(f"{role}: wrong interpreter prefix")
        if "vllm" in extras and row["packages"]["vllm"] != "0.29.0":
            errors.append(f"{role}: wrong vLLM version")
        for name, editable in row["editables"].items():
            if not editable["exists"] or not editable["path"].startswith(
                "/opt/nemo-rl/"
            ):
                errors.append(f"{role}: unstable editable path for {name}")
    ray_cli = Path("/opt/nemo_rl_venv/bin/ray")
    shebang = ray_cli.read_text().splitlines()[0]
    cli_row = {}
    try:
        cli_python = ray_cli_python(ray_cli, environment=Path("/opt/nemo_rl_venv"))
        result = subprocess.run(
            [str(cli_python), "-c", PROBE],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        cli_row = json.loads(result.stdout)
        cli_row["interpreter"] = str(cli_python)
        if (
            cli_row["prefix"] != rows[0]["prefix"]
            or cli_row["python_target"] != rows[0]["python_target"]
            or cli_row["ray_import_file"] != rows[0]["ray_import_file"]
        ):
            errors.append("Ray daemon CLI and driver resolve different runtimes")
        result = subprocess.run(
            [str(ray_cli), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        cli_row["version_output"] = result.stdout.strip()
        if cli_row["version_output"] != "ray, version 2.58.0":
            errors.append("Ray daemon CLI version does not match uv.lock")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        errors.append(f"Ray daemon CLI check failed: {error}")
    if not list(
        Path("/opt/nemo_rl_venv/lib64").glob(
            "python*/site-packages/ray/_private/runtime_env/nsight.py"
        )
    ):
        errors.append("ray.sub Nsight patch target is missing")
    report = {
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "scope": (
            "driver and six performance-matrix actors; other backends not certified"
        ),
        "ray_cli_shebang": shebang,
        "ray_cli": cli_row,
        "environments": rows,
        "errors": errors,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if errors and not inventory_only:
        raise SystemExit(f"Runtime alignment failed: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/opt/nemo-rl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    audit(args.root, args.output, args.inventory_only)


if __name__ == "__main__":
    main()
