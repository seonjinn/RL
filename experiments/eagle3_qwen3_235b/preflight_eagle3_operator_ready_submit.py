#!/usr/bin/env python3
"""No-submit preflight for the currently ready Eagle3 operator submit actions.

Operator sheet validation proves command shape. This preflight proves the
runtime inputs referenced by those commands are visible from the current host:
Slurm client binaries, container/image paths, config files, artifact paths, and
job-file state. It does not submit jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_CONTAINER = Path("/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh")
JOB_ID_RE = re.compile(r"\b\d{4,}\b")


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--operator-sheet-json", type=Path)
    parser.add_argument("--operator-submit-packet-validation-json", type=Path)
    parser.add_argument("--rollout-submit-preflight-json", type=Path)
    parser.add_argument(
        "--action-ids",
        nargs="*",
        default=None,
        help="Optional ready action ids to preflight. Defaults to every ready action in the operator sheet.",
    )
    parser.add_argument(
        "--allow-missing-action-ids",
        action="store_true",
        help="Treat requested action ids that are not currently ready as warnings instead of failures.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--require-slurm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser.parse_args()


def with_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root = args.artifact_root
    defaults = {
        "operator_sheet_json": Path(os.environ.get("OPERATOR_SHEET_JSON", root / "reports/eagle3_operator_sheet.json")),
        "operator_submit_packet_validation_json": Path(
            os.environ.get(
                "OPERATOR_SUBMIT_PACKET_VALIDATION_JSON",
                root / "reports/eagle3_operator_submit_packet_validation.json",
            )
        ),
        "rollout_submit_preflight_json": Path(
            os.environ.get("ROLLOUT_SUBMIT_PREFLIGHT_JSON", root / "reports/rollout_capture_submit_preflight.json")
        ),
        "json_out": Path(
            os.environ.get("OPERATOR_READY_SUBMIT_PREFLIGHT_JSON", root / "reports/eagle3_operator_ready_submit_preflight.json")
        ),
        "markdown_out": Path(
            os.environ.get("OPERATOR_READY_SUBMIT_PREFLIGHT_MARKDOWN", root / "reports/eagle3_operator_ready_submit_preflight.md")
        ),
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def load_json(path: Path | None) -> tuple[Any | None, str | None]:
    if path is None:
        return None, "not provided"
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid JSON: {exc}"


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def report_status(payload: Any, error: str | None) -> str:
    if error:
        return "missing"
    data = as_dict(payload)
    decision = as_dict(data.get("decision"))
    return str(data.get("overall_status") or data.get("status") or decision.get("overall_status") or "unknown")


def split_env_command(command: str) -> tuple[dict[str, str], list[str]]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return {}, []
    env: dict[str, str] = {}
    idx = 0
    for idx, token in enumerate(tokens):
        if "=" not in token or token.startswith("-"):
            break
        key, value = token.split("=", 1)
        if not key.replace("_", "").isalnum() or not key[:1].isalpha():
            break
        env[key] = value
    else:
        idx = len(tokens)
    return env, tokens[idx:]


def resolve_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def check_path(
    checks: list[dict[str, Any]],
    action_id: str,
    label: str,
    value: Any,
    *,
    required: bool = True,
    file: bool = False,
    dir: bool = False,
    nonempty: bool = False,
) -> Path | None:
    path = resolve_path(value)
    name = f"{action_id} {label}"
    if path is None:
        add(checks, "path", name, "fail" if required else "warn", "path is not set")
        return None
    exists = path.exists()
    kind_ok = exists and (not file or path.is_file()) and (not dir or path.is_dir())
    size_ok = kind_ok and (not nonempty or path.stat().st_size > 0)
    if size_ok:
        add(checks, "path", name, "pass", f"visible: {path}", path=str(path))
    else:
        status = "fail" if required else "warn"
        if not exists:
            detail = f"not visible: {path}"
        elif file and not path.is_file():
            detail = f"not a file: {path}"
        elif dir and not path.is_dir():
            detail = f"not a directory: {path}"
        else:
            detail = f"empty file: {path}"
        add(checks, "path", name, status, detail, path=str(path), exists=exists)
    return path


def check_writable_parent(checks: list[dict[str, Any]], action_id: str, label: str, value: Any) -> None:
    path = resolve_path(value)
    name = f"{action_id} {label}"
    if path is None:
        add(checks, "write", name, "fail", "path is not set")
        return
    parent = path.parent
    if parent.exists() and os.access(parent, os.W_OK):
        add(checks, "write", name, "pass", f"parent is writable: {parent}", path=str(path))
    elif parent.exists():
        add(checks, "write", name, "fail", f"parent is not writable: {parent}", path=str(path))
    else:
        ancestor = parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if ancestor.exists() and os.access(ancestor, os.W_OK):
            add(
                checks,
                "write",
                name,
                "pass",
                f"parent will be created under writable ancestor: {ancestor}",
                path=str(path),
                missing_parent=str(parent),
            )
        else:
            add(checks, "write", name, "fail", f"no writable ancestor found for missing parent: {parent}", path=str(path))


def check_slurm(checks: list[dict[str, Any]], require_slurm: bool, any_slurm_action: bool) -> None:
    if not any_slurm_action:
        return
    for binary in ("sbatch", "squeue", "sacct"):
        found = shutil.which(binary)
        if found:
            add(checks, "slurm", binary, "pass", f"visible: {found}", path=found)
        else:
            add(checks, "slurm", binary, "fail" if require_slurm else "warn", f"{binary} is not on PATH")


def check_job_file(checks: list[dict[str, Any]], action_id: str, path: Path, *, expect_placeholder_ok: bool = True) -> None:
    if not path.exists():
        add(checks, "job_file", action_id, "pass", f"job file is absent and will be created: {path}", path=str(path))
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    has_job_id = bool(JOB_ID_RE.search(text))
    has_placeholder = "PREFLIGHT_JOB_ID" in text or "JOB_ID" in text
    if has_job_id:
        add(checks, "job_file", action_id, "warn", "job file already contains a numeric job id; avoid duplicate submit without reviewing it", path=str(path), text=text[:500])
    elif has_placeholder and expect_placeholder_ok:
        add(checks, "job_file", action_id, "pass", "job file contains only a dry-run placeholder and will be overwritten on submit", path=str(path))
    else:
        add(checks, "job_file", action_id, "warn", "job file exists but no numeric job id was found", path=str(path), text=text[:500])


def option_value(argv: list[str], option: str) -> str | None:
    if option not in argv:
        return None
    idx = argv.index(option)
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def option_values(argv: list[str], option: str) -> list[str]:
    if option not in argv:
        return []
    values: list[str] = []
    for token in argv[argv.index(option) + 1 :]:
        if token.startswith("-"):
            break
        values.append(token)
    return values


def concrete_remote_path(value: str | None) -> bool:
    if not value:
        return False
    if value.startswith("$") or value.startswith("${") or "<" in value or ">" in value:
        return False
    return Path(value).is_absolute()


def true_env(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes"}


def concrete_account(value: str | None) -> bool:
    return bool(value) and value not in {"dummy", "<account>"}


def positive_int(value: str | None, default: int) -> tuple[bool, int | str | None]:
    raw = value if value not in {None, ""} else str(default)
    try:
        parsed = int(str(raw))
    except ValueError:
        return False, raw
    return parsed > 0, parsed


def path_env(env: dict[str, str], key: str, default: Path) -> Path:
    return Path(env.get(key) or default)


def report_dir(env: dict[str, str], artifact_root: Path) -> Path:
    return path_env(env, "REPORT_DIR", artifact_root / "reports")


def default_mounts(artifact_root: Path) -> str:
    return f"/lustre:/lustre,{ROOT}:{ROOT},{artifact_root}:{artifact_root}"


def check_submit_and_account(
    checks: list[dict[str, Any]],
    area: str,
    action_id: str,
    env: dict[str, str],
) -> None:
    if true_env(env.get("SUBMIT")):
        add(checks, area, f"{action_id} SUBMIT", "pass", "command is an actual submit command")
    else:
        add(checks, area, f"{action_id} SUBMIT", "fail", "command must set SUBMIT=true")
    account = env.get("SBATCH_ACCOUNT", "")
    if concrete_account(account):
        add(checks, area, f"{action_id} SBATCH_ACCOUNT", "pass", "Slurm account is concrete", sbatch_account=account)
    else:
        add(checks, area, f"{action_id} SBATCH_ACCOUNT", "fail", "Slurm account is missing or dummy", sbatch_account=account)


def check_gpu_request(
    checks: list[dict[str, Any]],
    area: str,
    action_id: str,
    env: dict[str, str],
    *,
    key: str = "GPUS_PER_NODE",
    default: int = 4,
) -> None:
    ok, value = positive_int(env.get(key), default)
    if ok:
        add(checks, area, f"{action_id} {key}", "pass", "Slurm command requests at least one GPU", value=value)
    else:
        add(checks, area, f"{action_id} {key}", "fail", "Slurm command must request at least one GPU", value=value)


def check_common_container_runtime(
    checks: list[dict[str, Any]],
    area: str,
    action_id: str,
    env: dict[str, str],
    artifact_root: Path,
) -> None:
    check_path(checks, action_id, "container", env.get("CONTAINER") or DEFAULT_CONTAINER, file=True)
    python_bin = env.get("PYTHON_BIN") or "/opt/venv/bin/python"
    if Path(python_bin).is_absolute():
        add(checks, area, f"{action_id} PYTHON_BIN", "pass", "container Python path is absolute", python_bin=python_bin)
    else:
        add(checks, area, f"{action_id} PYTHON_BIN", "fail", "container Python path must be absolute", python_bin=python_bin)
    mounts = env.get("MOUNTS") or default_mounts(artifact_root)
    if "/lustre:/lustre" in mounts and str(artifact_root) in mounts:
        add(checks, area, f"{action_id} mounts", "pass", "mounts include /lustre and the artifact root", mounts=mounts)
    else:
        add(checks, area, f"{action_id} mounts", "fail", "mounts must include /lustre and the artifact root", mounts=mounts)


def check_url_or_file(checks: list[dict[str, Any]], action_id: str, label: str, value: str) -> None:
    if value.startswith(("https://", "http://")):
        add(checks, "path", f"{action_id} {label}", "pass", "source is a URL", source=value)
    else:
        check_path(checks, action_id, label, value, file=True, nonempty=True)


def validate_common_action(checks: list[dict[str, Any]], item: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    action_id = str(item.get("id") or "")
    command = str(item.get("raw_command") or item.get("command") or "")
    env, argv = split_env_command(command)
    if item.get("status") == "ready_for_operator":
        add(checks, "action", action_id, "pass", "action is ready_for_operator")
    else:
        add(checks, "action", action_id, "fail", "action is not ready_for_operator", status=item.get("status"))
    if command and argv:
        add(checks, "command", action_id, "pass", "raw command shell-splits successfully", argv=argv[:6])
    else:
        add(checks, "command", action_id, "fail", "raw command is missing or cannot be shell-split")
    if "bash" in argv:
        script = argv[argv.index("bash") + 1] if argv.index("bash") + 1 < len(argv) else None
        check_path(checks, action_id, "script", script, file=True, nonempty=True)
    elif argv and Path(argv[0]).name.startswith("python"):
        script = next((part for part in argv[1:] if not part.startswith("-")), None)
        check_path(checks, action_id, "python script", script, file=True, nonempty=True)
    elif item.get("submits_slurm"):
        add(checks, "command", action_id, "fail", "Slurm submit action should invoke a bash wrapper", argv=argv)
    elif argv:
        executable = shutil.which(argv[0])
        if executable:
            add(checks, "command", action_id, "pass", "non-Slurm command executable is visible", executable=executable)
        else:
            target = resolve_path(argv[0])
            if target and target.exists():
                add(checks, "command", action_id, "pass", "non-Slurm command target is visible", path=str(target))
            else:
                add(checks, "command", action_id, "warn", "non-Slurm command target was not resolved", argv=argv)
    else:
        add(checks, "command", action_id, "fail", "raw command does not identify an executable", argv=argv)
    check_writable_parent(checks, action_id, "execution_record", item.get("execution_record"))
    if item.get("submits_slurm"):
        check_writable_parent(checks, action_id, "followup_record", item.get("followup_record"))
    return env, argv


def validate_container_action(checks: list[dict[str, Any]], item: dict[str, Any], env: dict[str, str]) -> None:
    action_id = str(item.get("id") or "")
    check_submit_and_account(checks, "container", action_id, env)
    preflight_gpus = env.get("PREFLIGHT_GPUS_PER_NODE")
    try:
        preflight_gpus_int = int(preflight_gpus or "")
    except ValueError:
        preflight_gpus_int = 0
    if preflight_gpus_int > 0:
        add(
            checks,
            "container",
            f"{action_id} PREFLIGHT_GPUS_PER_NODE",
            "pass",
            "container preflight command requests a GPU for GPU-only partitions",
            preflight_gpus_per_node=preflight_gpus_int,
        )
    else:
        add(
            checks,
            "container",
            f"{action_id} PREFLIGHT_GPUS_PER_NODE",
            "fail",
            "container preflight command must set PREFLIGHT_GPUS_PER_NODE because the batch partition rejects non-GPU jobs",
            preflight_gpus_per_node=preflight_gpus,
        )
    check_path(checks, action_id, "container", env.get("CONTAINER"), file=True)
    modelopt = check_path(checks, action_id, "ModelOpt dir", env.get("MODELOPT_DIR"), dir=True)
    if modelopt:
        check_path(checks, action_id, "ModelOpt launch_train.sh", modelopt / "examples/speculative_decoding/launch_train.sh", file=True, nonempty=True)
    verifier = check_path(checks, action_id, "verifier config dir", env.get("VERIFIER_CONFIG_DIR"), dir=True)
    if verifier:
        check_path(checks, action_id, "verifier config.json", verifier / "config.json", file=True, nonempty=True)
    # The container-only preflight proves runtime/ModelOpt/template viability before
    # rollout materialization. Missing target rollout JSONL is tracked by the
    # rollout/corpus gates, so keep it visible as a warning here.
    check_path(checks, action_id, "input data", env.get("INPUT_DATA"), required=False, file=True, nonempty=True)
    check_path(checks, action_id, "chat template", env.get("CHAT_TEMPLATE"), file=True, nonempty=True)
    if env.get("PREFLIGHT_JSON"):
        check_writable_parent(checks, action_id, "structured preflight json", env.get("PREFLIGHT_JSON"))
    if env.get("PREFLIGHT_MARKDOWN"):
        check_writable_parent(checks, action_id, "structured preflight markdown", env.get("PREFLIGHT_MARKDOWN"))
    mounts = env.get("MOUNTS", "")
    if "/lustre:/lustre" in mounts:
        add(checks, "container", f"{action_id} mounts", "pass", "mounts include /lustre:/lustre", mounts=mounts)
    else:
        add(checks, "container", f"{action_id} mounts", "warn", "mounts do not explicitly include /lustre:/lustre", mounts=mounts)
    check_job_file(checks, action_id, ROOT / "latest_eagle3_container_preflight_job.txt")


def validate_vllm_source_build_action(
    checks: list[dict[str, Any]],
    item: dict[str, Any],
    env: dict[str, str],
    artifact_root: Path,
) -> None:
    action_id = str(item.get("id") or "")
    reports = report_dir(env, artifact_root)
    output_site = path_env(env, "OUTPUT_SITE", artifact_root / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312")
    json_out = path_env(env, "JSON_OUT", reports / "vllm_native_source_build.json")
    markdown_out = path_env(env, "MARKDOWN_OUT", reports / "vllm_native_source_build.md")
    job_file = path_env(env, "JOB_FILE", ROOT / "latest_vllm_native_source_build_job.txt")
    source_spec = env.get("VLLM_SOURCE_SPEC") or "https://files.pythonhosted.org/packages/7d/0a/278d7bbf454f7de5322a5007427eed3e8b34ed6c2802491b56bbdfd7bbb4/vllm-0.10.2.tar.gz"

    check_submit_and_account(checks, "vllm_source", action_id, env)
    check_gpu_request(checks, "vllm_source", action_id, env)
    check_common_container_runtime(checks, "vllm_source", action_id, env, artifact_root)
    check_url_or_file(checks, action_id, "VLLM_SOURCE_SPEC", source_spec)
    check_path(checks, action_id, "source-build sbatch", ROOT / "experiments/eagle3_qwen3_235b/slurm_build_vllm_native_site.sbatch", file=True, nonempty=True)
    check_writable_parent(checks, action_id, "output site", output_site / ".write_probe")
    check_writable_parent(checks, action_id, "source-build json", json_out)
    check_writable_parent(checks, action_id, "source-build markdown", markdown_out)
    check_job_file(checks, action_id, job_file)


def validate_vllm_abi_probe_action(
    checks: list[dict[str, Any]],
    item: dict[str, Any],
    env: dict[str, str],
    artifact_root: Path,
) -> None:
    action_id = str(item.get("id") or "")
    reports = report_dir(env, artifact_root)
    json_out = path_env(env, "JSON_OUT", reports / "vllm_native_abi_probe.json")
    markdown_out = path_env(env, "MARKDOWN_OUT", reports / "vllm_native_abi_probe.md")
    job_file = path_env(env, "JOB_FILE", ROOT / "latest_vllm_native_abi_probe_job.txt")
    candidates = [part for part in shlex.split(env.get("VLLM_SITE_CANDIDATES") or "") if part]

    check_submit_and_account(checks, "vllm_abi", action_id, env)
    check_gpu_request(checks, "vllm_abi", action_id, env)
    check_common_container_runtime(checks, "vllm_abi", action_id, env, artifact_root)
    check_path(checks, action_id, "ABI probe sbatch", ROOT / "experiments/eagle3_qwen3_235b/slurm_vllm_native_abi_probe.sbatch", file=True, nonempty=True)
    if not candidates:
        add(checks, "vllm_abi", f"{action_id} VLLM_SITE_CANDIDATES", "fail", "ABI probe command must set VLLM_SITE_CANDIDATES")
    for idx, candidate in enumerate(candidates, start=1):
        check_path(checks, action_id, f"VLLM_SITE_CANDIDATES[{idx}]", candidate, dir=True)
    check_writable_parent(checks, action_id, "ABI probe json", json_out)
    check_writable_parent(checks, action_id, "ABI probe markdown", markdown_out)
    check_job_file(checks, action_id, job_file)


def validate_remote_probe_action(checks: list[dict[str, Any]], item: dict[str, Any], argv: list[str]) -> None:
    action_id = str(item.get("id") or "")
    if item.get("submits_slurm") is False:
        add(checks, "remote_probe", f"{action_id} submits_slurm", "pass", "remote host probe does not submit Slurm")
    else:
        add(checks, "remote_probe", f"{action_id} submits_slurm", "fail", "remote host probe must not submit Slurm")
    if item.get("heavy_gpu") is False:
        add(checks, "remote_probe", f"{action_id} heavy_gpu", "pass", "remote host probe does not require heavy GPU")
    else:
        add(checks, "remote_probe", f"{action_id} heavy_gpu", "fail", "remote host probe must not require heavy GPU")

    ssh = shutil.which("ssh")
    if ssh:
        add(checks, "remote_probe", f"{action_id} ssh", "pass", f"ssh is visible: {ssh}", path=ssh)
    else:
        add(checks, "remote_probe", f"{action_id} ssh", "fail", "ssh is not on PATH")

    hosts = option_values(argv, "--hosts")
    if hosts:
        add(checks, "remote_probe", f"{action_id} hosts", "pass", "remote host aliases are explicit", hosts=hosts)
    else:
        add(checks, "remote_probe", f"{action_id} hosts", "fail", "remote host probe command has no --hosts values")

    remote_workdir = option_value(argv, "--remote-workdir")
    if concrete_remote_path(remote_workdir):
        add(checks, "remote_probe", f"{action_id} remote workdir", "pass", "remote workdir is a concrete absolute path", path=remote_workdir)
    else:
        add(checks, "remote_probe", f"{action_id} remote workdir", "fail", "remote workdir must be a concrete absolute path", path=remote_workdir)

    remote_artifact_root = option_value(argv, "--artifact-root")
    if concrete_remote_path(remote_artifact_root):
        add(
            checks,
            "remote_probe",
            f"{action_id} remote artifact root",
            "pass",
            "remote artifact root is a concrete absolute path",
            path=remote_artifact_root,
        )
    else:
        add(
            checks,
            "remote_probe",
            f"{action_id} remote artifact root",
            "fail",
            "remote artifact root must be a concrete absolute path",
            path=remote_artifact_root,
        )

    json_out = option_value(argv, "--json-out")
    markdown_out = option_value(argv, "--markdown-out")
    check_writable_parent(checks, action_id, "remote probe json", json_out)
    check_writable_parent(checks, action_id, "remote probe markdown", markdown_out)
    if "--strict" in argv:
        add(
            checks,
            "remote_probe",
            f"{action_id} strict",
            "fail",
            "operator remote probe must be non-strict so unreachable hosts still produce structured evidence",
        )
    else:
        add(
            checks,
            "remote_probe",
            f"{action_id} strict",
            "pass",
            "remote probe is non-strict and can record unreachable status without aborting the runner",
        )


def validate_rollout_action(
    checks: list[dict[str, Any]],
    item: dict[str, Any],
    env: dict[str, str],
    rollout_submit: dict[str, Any],
    rollout_submit_error: str | None,
) -> None:
    action_id = str(item.get("id") or "")
    if env.get("DRY_RUN") == "false":
        add(checks, "rollout", f"{action_id} DRY_RUN", "pass", "rollout capture command is an actual submit command")
    else:
        add(checks, "rollout", f"{action_id} DRY_RUN", "fail", "rollout capture command must set DRY_RUN=false")
    if env.get("MAX_NUM_STEPS") == "1":
        add(checks, "rollout", f"{action_id} MAX_NUM_STEPS", "pass", "rollout capture is limited to one step")
    else:
        add(checks, "rollout", f"{action_id} MAX_NUM_STEPS", "warn", "rollout capture is not the expected one-step smoke", max_num_steps=env.get("MAX_NUM_STEPS"))
    account = env.get("SBATCH_ACCOUNT", "")
    if account and account not in {"dummy", "<account>"}:
        add(checks, "rollout", f"{action_id} SBATCH_ACCOUNT", "pass", "Slurm account is concrete", sbatch_account=account)
    else:
        add(checks, "rollout", f"{action_id} SBATCH_ACCOUNT", "fail", "Slurm account is missing or dummy", sbatch_account=account)
    repo = check_path(checks, action_id, "SpecDec-RL repo", env.get("SWE_REPO_ROOT") or env.get("REPO_ROOT"), dir=True)
    if repo:
        check_path(checks, action_id, "ray.sub", repo / "ray.sub", file=True, nonempty=True)
        check_path(checks, action_id, "NeMo-Gym entrypoint", repo / "examples/nemo_gym/run_grpo_nemo_gym.py", file=True, nonempty=True)
        check_job_file(checks, action_id, repo / "latest_235b_swe_job_id.txt")
    check_path(checks, action_id, "Qwen3 SWE config", env.get("CONFIG_FILE"), file=True, nonempty=True)
    check_path(checks, action_id, "env file", env.get("ENV_FILE"), required=False, file=True, nonempty=True)
    check_path(checks, action_id, "chat template", env.get("CHAT_TEMPLATE"), file=True, nonempty=True)
    check_writable_parent(checks, action_id, "rollout log dir", Path(str(env.get("ROLLOUT_LOG_DIR", ""))) / ".write_probe")
    check_writable_parent(checks, action_id, "output conversations", env.get("OUTPUT_CONVERSATIONS"))
    rollout_status = report_status(rollout_submit, rollout_submit_error)
    submit_ready = bool(as_dict(rollout_submit).get("submit_ready"))
    if rollout_status == "pass" and submit_ready:
        add(checks, "rollout", "rollout submit preflight", "pass", "rollout submit preflight is PASS and submit_ready=true")
    else:
        add(checks, "rollout", "rollout submit preflight", "fail", "rollout submit preflight is not ready", status=rollout_status, submit_ready=submit_ready)


def validate_pipeline_action(checks: list[dict[str, Any]], item: dict[str, Any], argv: list[str]) -> None:
    action_id = str(item.get("id") or "")
    if item.get("submits_slurm") is True:
        add(checks, "pipeline", f"{action_id} submits_slurm", "pass", "pipeline action is marked as a Slurm submit")
    else:
        add(checks, "pipeline", f"{action_id} submits_slurm", "fail", "pipeline action must be marked as a Slurm submit")
    if item.get("heavy_gpu") is True:
        add(checks, "pipeline", f"{action_id} heavy_gpu", "pass", "pipeline action is marked as heavy GPU")
    else:
        add(checks, "pipeline", f"{action_id} heavy_gpu", "fail", "pipeline action must be marked as heavy GPU")
    if "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline_if_ready.py" in argv:
        add(checks, "pipeline", f"{action_id} gated helper", "pass", "pipeline submit goes through the gated helper")
    else:
        add(
            checks,
            "pipeline",
            f"{action_id} gated helper",
            "fail",
            "pipeline submit must use submit_eagle3_pipeline_if_ready.py instead of direct submit_eagle3_pipeline.sh",
            argv=argv,
        )
    if "--execute" in argv:
        add(checks, "pipeline", f"{action_id} execute flag", "pass", "gated helper is in execute mode")
    else:
        add(checks, "pipeline", f"{action_id} execute flag", "fail", "pipeline gated submit command must include --execute")
    if "--allow-heavy-gpu" in argv:
        add(checks, "pipeline", f"{action_id} heavy allow flag", "pass", "gated helper has the heavy GPU allow flag")
    else:
        add(checks, "pipeline", f"{action_id} heavy allow flag", "fail", "pipeline gated submit command must include --allow-heavy-gpu")

    preflight_json = option_value(argv, "--preflight-json")
    gated_json = option_value(argv, "--json-out")
    gated_markdown = option_value(argv, "--markdown-out")
    preflight_path = check_path(checks, action_id, "pipeline submit preflight json", preflight_json, file=True, nonempty=True)
    check_writable_parent(checks, action_id, "pipeline gated submit json", gated_json)
    check_writable_parent(checks, action_id, "pipeline gated submit markdown", gated_markdown)
    if preflight_path:
        payload, error = load_json(preflight_path)
        data = as_dict(payload)
        if error:
            add(checks, "pipeline", f"{action_id} preflight status", "fail", error, path=str(preflight_path))
        elif data.get("overall_status") == "pass" and data.get("submit_ready") is True:
            add(checks, "pipeline", f"{action_id} preflight status", "pass", "pipeline submit preflight is PASS and submit_ready=true")
        else:
            add(
                checks,
                "pipeline",
                f"{action_id} preflight status",
                "fail",
                "pipeline submit preflight is not ready",
                overall_status=data.get("overall_status"),
                submit_ready=data.get("submit_ready"),
            )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    sheet_raw, sheet_error = load_json(args.operator_sheet_json)
    packet_validation_raw, packet_validation_error = load_json(args.operator_submit_packet_validation_json)
    rollout_submit_raw, rollout_submit_error = load_json(args.rollout_submit_preflight_json)
    sheet = as_dict(sheet_raw)
    ready_actions = [item for item in as_list(sheet.get("ready_actions")) if isinstance(item, dict)]
    requested_action_ids = set(args.action_ids or [])
    original_ready_action_ids = {str(item.get("id") or "") for item in ready_actions}
    if requested_action_ids:
        ready_actions = [item for item in ready_actions if str(item.get("id") or "") in requested_action_ids]
    checks: list[dict[str, Any]] = []

    if sheet_error:
        add(checks, "input", "operator sheet", "fail", sheet_error, path=str(args.operator_sheet_json))
    else:
        add(checks, "input", "operator sheet", "pass", "operator sheet is readable", ready_actions=len(ready_actions))
    if requested_action_ids:
        matched = {str(item.get("id") or "") for item in ready_actions}
        missing = sorted(requested_action_ids - matched)
        if missing:
            status = "warn" if args.allow_missing_action_ids else "fail"
            add(
                checks,
                "input",
                "action filter",
                status,
                "requested action ids are not present in the operator sheet ready actions",
                requested=sorted(requested_action_ids),
                available=sorted(original_ready_action_ids),
                missing=missing,
            )
        else:
            add(
                checks,
                "input",
                "action filter",
                "pass",
                "preflight is scoped to the requested ready actions",
                requested=sorted(requested_action_ids),
            )
    packet_status = report_status(packet_validation_raw, packet_validation_error)
    if packet_status == "pass":
        add(checks, "input", "operator submit packet validation", "pass", "submit packet validation is PASS")
    else:
        add(
            checks,
            "input",
            "operator submit packet validation",
            "warn",
            "submit packet validation is not PASS",
            report_status=packet_status,
        )

    command_actions = [item for item in ready_actions if item.get("command") or item.get("raw_command")]
    if not command_actions and not sheet_error:
        add(checks, "action", "ready actions", "warn", "operator sheet has no ready submit actions")
    check_slurm(checks, args.require_slurm, any(bool(item.get("submits_slurm")) for item in command_actions))

    action_summaries: list[dict[str, Any]] = []
    for item in command_actions:
        action_id = str(item.get("id") or "")
        env, argv = validate_common_action(checks, item)
        if action_id == "submit_container_preflight":
            validate_container_action(checks, item, env)
        elif action_id == "submit_vllm_source_build":
            validate_vllm_source_build_action(checks, item, env, args.artifact_root)
        elif action_id == "submit_source_vllm_abi_probe":
            validate_vllm_abi_probe_action(checks, item, env, args.artifact_root)
        elif action_id == "probe_remote_hosts":
            validate_remote_probe_action(checks, item, argv)
        elif action_id in {"submit_rollout_capture", "submit_rollout_fallback"}:
            validate_rollout_action(checks, item, env, as_dict(rollout_submit_raw), rollout_submit_error)
        elif action_id == "submit_eagle3_pilot_pipeline":
            validate_pipeline_action(checks, item, argv)
        else:
            add(checks, "action", action_id, "pass", "no action-specific submit preflight is required")
        action_summaries.append(
            {
                "id": action_id,
                "submits_slurm": bool(item.get("submits_slurm")),
                "heavy_gpu": bool(item.get("heavy_gpu")),
                "env_keys": sorted(env),
            }
        )

    counts: dict[str, int] = {}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    if any(check["status"] == "fail" for check in checks):
        status = "fail"
    elif any(check["status"] == "warn" for check in checks):
        status = "warn"
    else:
        status = "pass"
    submit_ready = status == "pass"
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": status,
        "submit_ready": submit_ready,
        "artifact_root": str(args.artifact_root),
        "operator_sheet_json": str(args.operator_sheet_json),
        "action_filter": sorted(requested_action_ids),
        "ready_actions": action_summaries,
        "counts": counts,
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator Ready-Submit Preflight",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Submit ready: **{str(payload['submit_ready']).lower()}**",
        f"Operator sheet: `{payload['operator_sheet_json']}`",
        "",
        "| area | check | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        detail = str(check["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {check['area']} | {check['name']} | {check['status'].upper()} | {detail} |")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = with_defaults(parse_args())
    payload = build_payload(args)
    markdown = render_markdown(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    if payload["overall_status"] == "fail":
        return 1
    if args.fail_on_warn and payload["overall_status"] == "warn":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
