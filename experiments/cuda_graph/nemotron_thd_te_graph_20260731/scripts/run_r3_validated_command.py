#!/usr/bin/env python3
"""Execute one digest-bound Router Replay driver and atomically attest it."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import stat
import subprocess
import sys
from typing import NoReturn


def fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def open_dir(path: str) -> int:
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        fail(f"unsafe R3 directory: {error}")


def remove_tree(parent_fd: int, name: str) -> None:
    try:
        item = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(item.st_mode):
        fail("R3 trace directory must not be a symlink")
    if not stat.S_ISDIR(item.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        return
    child_fd = os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
    )
    try:
        for child in os.listdir(child_fd):
            remove_tree(child_fd, child)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)


def atomic_write(directory_fd: int, name: str, value: bytes) -> None:
    temporary = f".{name}.{secrets.token_hex(12)}"
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            offset = 0
            while offset < len(value):
                offset += os.write(fd, value[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    except OSError:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    os.fsync(directory_fd)


def read_bound_file(path: str, expected_sha256: str, label: str) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        fail(f"R3 {label} rejected: {error}")
    try:
        details = os.fstat(fd)
        if not stat.S_ISREG(details.st_mode):
            fail(f"R3 {label} must be regular")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
    finally:
        os.close(fd)
    content = b"".join(chunks)
    if not hmac.compare_digest(hashlib.sha256(content).hexdigest(), expected_sha256):
        fail(f"R3 {label} digest mismatch")
    if not content:
        fail(f"R3 {label} is empty")
    return content


def normalized_return_code(code: int) -> int:
    return 128 + -code if code < 0 else code


def valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def runtime_python_from_uv(uv: str) -> str:
    suffix = os.path.join("uv", "uv")
    if not uv.endswith(os.sep + suffix):
        fail("R3 uv executable must use the staged-runtimes/<sha256>/uv/uv layout")
    stage_root = uv[: -len(suffix)].rstrip(os.sep)
    stage_key = os.path.basename(stage_root)
    if os.path.basename(
        os.path.dirname(stage_root)
    ) != "staged-runtimes" or not valid_sha256(stage_key):
        fail("R3 uv executable must use the staged-runtimes/<sha256>/uv/uv layout")
    runtime_python = os.path.join(stage_root, "environment", "bin", "python")
    if not os.path.isfile(runtime_python) or not os.access(runtime_python, os.X_OK):
        fail("R3 attested runtime Python is missing or not executable")
    return runtime_python


def encode(value: str) -> str:
    return base64.b64encode(value.encode()).decode("ascii")


def main() -> int:
    if len(sys.argv) != 7:
        fail(
            "usage: run_r3_validated_command.py RUN_LOG_DIR REPO_ROOT UV DRIVER_FILE DRIVER_SHA256 CHECKER_SHA256"
        )
    base, repo, uv, driver_file, driver_sha, checker_sha = sys.argv[1:]
    job_id = os.environ.get("NRL_SLURM_JOB_ID", "")
    restart = os.environ.get("NRL_SLURM_RESTART_COUNT", "0")
    absolute_paths = (base, repo, uv, driver_file)
    valid_job_id = job_id.isascii() and job_id.isdecimal() and int(job_id) > 0
    valid_restart = restart.isascii() and restart.isdecimal()
    if not (
        all(os.path.isabs(path) for path in absolute_paths)
        and valid_job_id
        and valid_restart
    ):
        fail("invalid R3 paths or Slurm identity")
    if not valid_sha256(driver_sha) or not valid_sha256(checker_sha):
        fail("R3 driver and checker digests must be lowercase SHA256")
    runtime_python = runtime_python_from_uv(uv)
    driver_bytes = read_bound_file(driver_file, driver_sha, "driver command file")
    try:
        driver = driver_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"R3 driver command is not UTF-8: {error}")
    checker = os.path.join(repo, "tools", "check_r3_trace.py")
    checker_bytes = read_bound_file(checker, checker_sha, "checker")
    base_fd = open_dir(base)
    attempt_name = f"r3-validation-job-{job_id}-restart-{restart}"
    trace_name = f"trace-job-{job_id}-restart-{restart}"
    try:
        try:
            os.mkdir(attempt_name, 0o700, dir_fd=base_fd)
        except FileExistsError:
            pass
        attempt_fd = os.open(
            attempt_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=base_fd
        )
        try:
            remove_tree(attempt_fd, trace_name)
            os.mkdir(trace_name, 0o700, dir_fd=attempt_fd)
            trace = f"{base}/{attempt_name}/{trace_name}"
            checker_command = [
                runtime_python,
                "-I",
                "-B",
                "-S",
                "-",
                trace,
                "--require-forward-verify",
                "--require-cp-identity",
            ]

            def record(
                status: str,
                driver_rc: int,
                checker_rc: int | None,
                driver_raw_rc: int | None,
                checker_raw_rc: int | None,
            ) -> None:
                data = {
                    "schema_version": 1,
                    "status": status,
                    "trace_dir": trace,
                    "driver_command_file": driver_file,
                    "driver_command_sha256": driver_sha,
                    "driver_command": driver,
                    "checker_source_path": checker,
                    "checker_expected_sha256": checker_sha,
                    "checker_actual_sha256": hashlib.sha256(checker_bytes).hexdigest(),
                    "checker_command": checker_command,
                    "driver_exit_code": driver_rc,
                    "checker_exit_code": checker_rc,
                    "driver_raw_return_code": driver_raw_rc,
                    "checker_raw_return_code": checker_raw_rc,
                    "slurm_job_id": int(job_id),
                    "slurm_restart_count": int(restart),
                }
                env = "\n".join(
                    (
                        "schema_version=1",
                        f"status={status}",
                        f"trace_dir_base64={encode(trace)}",
                        f"driver_command_file_base64={encode(driver_file)}",
                        f"driver_command_base64={encode(driver)}",
                        f"driver_command_sha256={driver_sha}",
                        f"checker_source_path_base64={encode(checker)}",
                        f"checker_command_json_base64={encode(json.dumps(checker_command))}",
                        f"checker_sha256={checker_sha}",
                        f"driver_exit_code={driver_rc}",
                        f"checker_exit_code={'' if checker_rc is None else checker_rc}",
                        f"driver_raw_return_code={'' if driver_raw_rc is None else driver_raw_rc}",
                        f"checker_raw_return_code={'' if checker_raw_rc is None else checker_raw_rc}",
                        f"slurm_job_id={job_id}",
                        f"slurm_restart_count={restart}",
                        "",
                    )
                )
                atomic_write(attempt_fd, "r3-validation.env", env.encode())
                atomic_write(
                    attempt_fd,
                    "r3-validation.json",
                    (json.dumps(data, sort_keys=True) + "\n").encode(),
                )

            record("pending", -1, None, None, None)
            driver_env = {
                key: value
                for key, value in os.environ.items()
                if key not in {"BASH_ENV", "ENV"}
            }
            driver_env.update(
                {
                    "NRL_R3_TRACE_DIR": trace,
                    "NRL_R3_TRACE_STEPS": "5",
                    "NRL_R3_TRACE_SAMPLES": "2",
                    "NRL_R3_TRACE_MICROBATCHES": "2",
                    "NRL_R3_TRACE": "1",
                    "NRL_R3_TRACE_VERIFY_FORWARD": "1",
                    "NRL_ROUTER_REPLAY_VALIDATE": "1",
                }
            )
            result = subprocess.run(
                ["/bin/bash"], input=driver_bytes, env=driver_env, check=False
            )
            if result.returncode:
                code = normalized_return_code(result.returncode)
                record("not_run_driver_failed", code, None, result.returncode, None)
                return code
            checked = subprocess.run(
                checker_command,
                input=checker_bytes,
                env=driver_env,
                check=False,
            )
            if checked.returncode:
                code = normalized_return_code(checked.returncode)
                record("failed", 0, code, 0, checked.returncode)
                return code
            record("passed", 0, 0, 0, 0)
            return 0
        finally:
            os.close(attempt_fd)
    finally:
        os.close(base_fd)


if __name__ == "__main__":
    raise SystemExit(main())
