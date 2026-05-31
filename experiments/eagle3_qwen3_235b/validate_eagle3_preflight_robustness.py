#!/usr/bin/env python3
"""Validate that local preflight helpers fail cleanly on lightweight hosts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_ARTIFACT_ROOT = "/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3"
SECRET_SENTINELS = {
    "WANDB_API_KEY": "EAGLE3_SENTINEL_WANDB_SHOULD_NOT_LEAK",
    "HUGGINGFACE_TOKEN": "EAGLE3_SENTINEL_HF_SHOULD_NOT_LEAK",
    "GITHUB_TOKEN": "EAGLE3_SENTINEL_GITHUB_SHOULD_NOT_LEAK",
    "GITLAB_TOKEN": "EAGLE3_SENTINEL_GITLAB_SHOULD_NOT_LEAK",
}
CANONICAL_ROLLOUT_INPUT = "qwen3_235b_swe_rollout_conversations.jsonl"
LEGACY_BOOTSTRAP_INPUT = "qwen3_235b_swe_conversations.jsonl"
EXPECTED_PLAYBOOK_ARTIFACT_FLOW = [
    "rollout_conversation_corpus",
    "verifier_hidden_states",
    "modelopt_checkpoint",
    "hf_eagle3_export",
    "vllm_eagle3_draft",
    "rl_vllm_draft_validation",
]
EXPECTED_PLAYBOOK_READY_ACTIONS = [
    "probe_remote_hosts",
    "submit_vllm_source_build",
    "poll_megatron_compat_probe",
    "submit_container_preflight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(SECRET_SENTINELS)
    if env:
        merged.update(env)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid json: {exc}"


def file_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def add(checks: list[dict[str, Any]], name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"name": name, "status": status, "detail": detail, "evidence": evidence})


def contains_any(text: str, needles: list[str]) -> list[str]:
    return [needle for needle in needles if needle and needle in text]


def status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_fake_ssh(bin_dir: Path) -> None:
    fake = bin_dir / "ssh"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(
        r'''#!/usr/bin/env bash
host=""
skip_next=false
for arg in "$@"; do
  if [[ "$skip_next" == "true" ]]; then
    skip_next=false
    continue
  fi
  case "$arg" in
    -S|-o)
      skip_next=true
      continue
      ;;
    -*)
      continue
      ;;
  esac
  host="$arg"
  break
done

if [[ "$host" == "badhost" ]]; then
  echo "ssh: Could not resolve hostname badhost: nodename nor servname provided, or not known" >&2
  exit 255
fi

printf 'FIELD\tprobe_time\t2026-05-23T00:00:00+00:00\n'
printf 'FIELD\thostname\tfake-remote\n'
printf 'FIELD\tpwd\t/home/sna\n'
printf 'FIELD\tnvidia_smi\tGPU 0: Fake GPU\n'
printf 'FIELD\tartifact_df\tfakefs 1T 1G 999G 1%% /lustre\n'
printf 'CMD\tsbatch\t/usr/bin/sbatch\n'
printf 'CMD\tsrun\t/usr/bin/srun\n'
printf 'CMD\tsqueue\t/usr/bin/squeue\n'
printf 'CMD\tsinfo\t/usr/bin/sinfo\n'
printf 'CMD\tsacct\t/usr/bin/sacct\n'
printf 'CMD\tgit\t/usr/bin/git\n'
printf 'CMD\tpython3\t/usr/bin/python3\n'
printf 'CMD\tnvidia-smi\t/usr/bin/nvidia-smi\n'
printf 'PATH\t/lustre\ttrue\ttrue\ttrue\tdrwxr-xr-x fake /lustre\tmain\tabcdef1234567890\n'
exit 0
''',
        encoding="utf-8",
    )
    fake.chmod(0o755)


def validate_remote_host_probe_contract(root: Path, checks: list[dict[str, Any]]) -> None:
    bin_dir = root / "fake_bin"
    write_fake_ssh(bin_dir)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "REQUIRE_SOURCE_VLLM_RUNTIME": "false",
    }
    report_dir = root / "remote_probe"
    unreachable_json = report_dir / "unreachable.json"
    unreachable_md = report_dir / "unreachable.md"
    strict_json = report_dir / "strict.json"
    pass_json = report_dir / "pass.json"
    pass_md = report_dir / "pass.md"

    unreachable = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/probe_eagle3_remote_host.py",
            "--hosts",
            "badhost",
            "--json-out",
            str(unreachable_json),
            "--markdown-out",
            str(unreachable_md),
        ],
        env=env,
    )
    strict = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/probe_eagle3_remote_host.py",
            "--hosts",
            "badhost",
            "--json-out",
            str(strict_json),
            "--strict",
        ],
        env=env,
    )
    reachable = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/probe_eagle3_remote_host.py",
            "--hosts",
            "goodhost",
            "--path",
            "/lustre",
            "--json-out",
            str(pass_json),
            "--markdown-out",
            str(pass_md),
        ],
        env=env,
    )

    unreachable_payload, unreachable_error = read_json(unreachable_json)
    strict_payload, strict_error = read_json(strict_json)
    pass_payload, pass_error = read_json(pass_json)
    pass_markdown = file_text(pass_md)
    problems = []
    if unreachable.returncode != 0 or unreachable_error:
        problems.append("unreachable probe did not write JSON with zero returncode")
    if (unreachable_payload or {}).get("overall_status") != "unreachable":
        problems.append("unreachable probe did not report overall_status=unreachable")
    if ((unreachable_payload or {}).get("counts") or {}).get("unreachable") != 1:
        problems.append("unreachable probe counts are wrong")
    unreachable_host = ((unreachable_payload or {}).get("hosts") or [{}])[0]
    if not isinstance(unreachable_host.get("local_resolution"), dict):
        problems.append("unreachable probe did not record local resolution diagnostics")
    if not isinstance(unreachable_host.get("ssh_config"), dict):
        problems.append("unreachable probe did not record effective SSH config diagnostics")
    if ((unreachable_payload or {}).get("host_discovery") or {}).get("include_ssh_config_hosts") is not False:
        problems.append("unreachable probe should not include ssh-config hosts unless explicitly requested")
    if strict.returncode != 2 or strict_error:
        problems.append("strict unreachable probe did not return 2 while writing JSON")
    if (strict_payload or {}).get("overall_status") != "unreachable":
        problems.append("strict unreachable probe did not write unreachable status")
    if reachable.returncode != 0 or pass_error:
        problems.append("reachable probe did not write JSON with zero returncode")
    if (pass_payload or {}).get("overall_status") != "pass":
        problems.append("reachable probe did not report overall_status=pass")
    if (pass_payload or {}).get("reachable_hosts") != ["goodhost"]:
        problems.append("reachable probe did not preserve reachable_hosts")
    reachable_host = ((pass_payload or {}).get("hosts") or [{}])[0]
    if not isinstance(reachable_host.get("local_resolution"), dict):
        problems.append("reachable probe did not record local resolution diagnostics")
    if "Overall: **PASS**" not in pass_markdown:
        problems.append("reachable markdown does not include PASS summary")
    if problems:
        add(
            checks,
            "remote host probe contract is structured",
            "fail",
            "probe_eagle3_remote_host.py contract regressed",
            problems=problems,
            unreachable_returncode=unreachable.returncode,
            strict_returncode=strict.returncode,
            reachable_returncode=reachable.returncode,
            unreachable_stdout=unreachable.stdout[-2000:],
            strict_stdout=strict.stdout[-2000:],
            reachable_stdout=reachable.stdout[-2000:],
        )
        return
    add(
        checks,
        "remote host probe contract is structured",
        "pass",
        "remote host probe reports pass/unreachable status, counts, strict failure, and markdown summary",
        unreachable_counts=(unreachable_payload or {}).get("counts"),
        strict_returncode=strict.returncode,
        reachable_hosts=(pass_payload or {}).get("reachable_hosts"),
    )


def validate_remote_access_diagnostics_contract(root: Path, checks: list[dict[str, Any]]) -> None:
    report_dir = root / "remote_access_diagnostics"
    probe_json = report_dir / "probe.json"
    diag_json = report_dir / "diagnostics.json"
    diag_md = report_dir / "diagnostics.md"
    probe_json.parent.mkdir(parents=True, exist_ok=True)
    probe_json.write_text(
        json.dumps(
            {
                "overall_status": "unreachable",
                "host_discovery": {
                    "include_ssh_config_hosts": True,
                    "ssh_config_hosts": ["oci-hsg-cs-001-vscode-03"],
                },
                "hosts": [
                    {
                        "host": "oci-hsg-cs-001-vscode-03",
                        "reachable": False,
                        "returncode": 255,
                        "stderr": "ssh: Could not resolve hostname oci-hsg-cs-001-vsode-03",
                        "ssh_config": {"hostname": "oci-hsg-cs-001-vsode-03", "port": "22"},
                        "local_resolution": {
                            "resolved": False,
                            "query": "oci-hsg-cs-001-vsode-03",
                            "error": "synthetic DNS failure",
                        },
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/diagnose_eagle3_remote_access.py",
            "--remote-host-probe-json",
            str(probe_json),
            "--json-out",
            str(diag_json),
            "--markdown-out",
            str(diag_md),
        ]
    )
    payload, error = read_json(diag_json)
    markdown = file_text(diag_md)
    findings = (payload or {}).get("configuration_findings") or []
    problems = []
    if result.returncode != 0 or error:
        problems.append("remote diagnostics did not write JSON with zero returncode")
    if (payload or {}).get("overall_status") != "blocked_local_dns":
        problems.append("remote diagnostics did not preserve blocked_local_dns status")
    if ((payload or {}).get("counts") or {}).get("ssh_config_hostname_warnings") != 1:
        problems.append("remote diagnostics did not count the SSH HostName warning")
    if not findings or findings[0].get("finding") != "possible_ssh_config_hostname_typo":
        problems.append("remote diagnostics did not flag the likely SSH config HostName typo")
    if "Configuration Findings" not in markdown:
        problems.append("remote diagnostics markdown does not include configuration findings")
    if problems:
        add(
            checks,
            "remote access diagnostics flags SSH config typos",
            "fail",
            "diagnose_eagle3_remote_access.py did not preserve the operator-facing SSH config finding",
            problems=problems,
            payload=payload,
            output_tail=result.stdout[-2000:],
        )
        return
    add(
        checks,
        "remote access diagnostics flags SSH config typos",
        "pass",
        "diagnose_eagle3_remote_access.py records probable HostName typos without closing the remote gate",
        findings=findings,
    )


def validate_remote_probe_runner_action(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "remote_probe_runner_action"
    report_dir = artifact / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    remote_probe_json = report_dir / "eagle3_remote_host_probe.json"
    remote_probe_md = report_dir / "eagle3_remote_host_probe.md"
    plan_json = report_dir / "eagle3_next_actions.json"
    plan_md = report_dir / "eagle3_next_actions.md"
    sheet_json = report_dir / "eagle3_operator_sheet.json"
    sheet_md = report_dir / "eagle3_operator_sheet.md"
    packet_validation_json = report_dir / "eagle3_operator_submit_packet_validation.json"
    ready_preflight_json = report_dir / "eagle3_operator_ready_submit_preflight.json"
    ready_preflight_md = report_dir / "eagle3_operator_ready_submit_preflight.md"
    record_json = report_dir / "operator_execution/01_probe_remote_hosts.json"
    remote_probe_json.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-23 00:00:00 UTC",
                "overall_status": "unreachable",
                "hosts_requested": ["badhost"],
                "reachable_hosts": [],
                "counts": {"reachable": 0, "unreachable": 1, "requested": 1},
                "hosts": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for report_name in ["container_preflight_analysis.json", "megatron_compat_probe.json"]:
        (report_dir / report_name).write_text(
            json.dumps({"overall_status": "pass"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    bin_dir = root / "fake_bin_runner"
    write_fake_ssh(bin_dir)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "REQUIRE_SOURCE_VLLM_RUNTIME": "false",
    }
    plan_result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/plan_eagle3_next_actions.py",
            "--artifact-root",
            str(artifact),
            "--remote-host-probe-json",
            str(remote_probe_json),
            "--json-out",
            str(plan_json),
            "--markdown-out",
            str(plan_md),
        ],
        env=env,
    )
    sheet_result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/create_eagle3_operator_sheet.py",
            "--artifact-root",
            str(artifact),
            "--plan-json",
            str(plan_json),
            "--json-out",
            str(sheet_json),
            "--markdown-out",
            str(sheet_md),
        ],
        env=env,
    )
    packet_validation_json.write_text(
        json.dumps({"overall_status": "pass", "counts": {"pass": 1}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ready_preflight_result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/preflight_eagle3_operator_ready_submit.py",
            "--artifact-root",
            str(artifact),
            "--operator-sheet-json",
            str(sheet_json),
            "--operator-submit-packet-validation-json",
            str(packet_validation_json),
            "--json-out",
            str(ready_preflight_json),
            "--markdown-out",
            str(ready_preflight_md),
            "--no-require-slurm",
        ],
        env=env,
    )
    runner_result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/run_eagle3_next_action.py",
            "--artifact-root",
            str(artifact),
            "--plan-json",
            str(plan_json),
            "--action-id",
            "probe_remote_hosts",
            "--execute",
            "--json-out",
            str(record_json),
        ],
        env=env,
    )
    list_result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/run_eagle3_next_action.py",
            "--artifact-root",
            str(artifact),
            "--list",
        ],
        env=env,
    )

    plan_payload, plan_error = read_json(plan_json)
    sheet_payload, sheet_error = read_json(sheet_json)
    ready_preflight_payload, ready_preflight_error = read_json(ready_preflight_json)
    record_payload, record_error = read_json(record_json)
    probe_payload, probe_error = read_json(remote_probe_json)
    actions = {
        str(item.get("id")): item
        for item in (plan_payload or {}).get("next_actions") or []
        if isinstance(item, dict)
    }
    action = actions.get("probe_remote_hosts") or {}
    combined = "\n".join(
        [
            plan_result.stdout,
            sheet_result.stdout,
            ready_preflight_result.stdout,
            runner_result.stdout,
            list_result.stdout,
            file_text(plan_json),
            file_text(sheet_json),
            file_text(ready_preflight_json),
            file_text(record_json),
            file_text(remote_probe_json),
            file_text(remote_probe_md),
        ]
    )
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    after_rows = (record_payload or {}).get("after_returncodes")
    problems = []
    if plan_result.returncode != 0 or plan_error:
        problems.append("next-action planner did not write JSON with zero returncode")
    if sheet_result.returncode != 0 or sheet_error:
        problems.append("operator sheet did not write JSON with zero returncode")
    if ready_preflight_result.returncode != 0 or ready_preflight_error:
        problems.append("operator ready-submit preflight did not write JSON with zero returncode")
    if (ready_preflight_payload or {}).get("overall_status") != "pass":
        problems.append("operator ready-submit preflight did not pass for probe_remote_hosts")
    if not any(
        isinstance(item, dict) and item.get("id") == "probe_remote_hosts"
        for item in (ready_preflight_payload or {}).get("ready_actions") or []
    ):
        problems.append("operator ready-submit preflight did not summarize probe_remote_hosts")
    if action.get("status") != "ready_for_operator":
        problems.append("probe_remote_hosts was not ready_for_operator")
    if action.get("submits_slurm") is not False:
        problems.append("probe_remote_hosts is not marked submits_slurm=false")
    if action.get("heavy_gpu") is not False:
        problems.append("probe_remote_hosts is not marked heavy_gpu=false")
    if runner_result.returncode != 0 or record_error:
        problems.append("runner did not execute probe_remote_hosts with zero returncode and a record")
    if list_result.returncode != 0:
        problems.append("runner --list did not use --artifact-root to locate the default plan JSON")
    if f"artifact_root={artifact}" not in list_result.stdout:
        problems.append("runner --list output did not report the requested artifact root")
    if (record_payload or {}).get("mode") != "execute":
        problems.append("runner record mode is not execute")
    if (record_payload or {}).get("returncode") != 0:
        problems.append("runner record returncode is not zero")
    if ((record_payload or {}).get("action") or {}).get("id") != "probe_remote_hosts":
        problems.append("runner record action id is not probe_remote_hosts")
    if (record_payload or {}).get("after_policy") != "after_command_success":
        problems.append("runner record after_policy is not after_command_success")
    if after_rows != []:
        problems.append("runner should not run after_commands without --run-after")
    if probe_error:
        problems.append("runner command did not write remote host probe JSON")
    if (probe_payload or {}).get("overall_status") != "pass":
        problems.append("fake-ssh runner probe did not produce overall_status=pass")
    if ((probe_payload or {}).get("counts") or {}).get("reachable", 0) < 1:
        problems.append("fake-ssh runner probe did not record a reachable host")
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if traces:
        problems.append("traceback leaked into runner action output")

    if problems:
        add(
            checks,
            "remote probe ready action executes through runner",
            "fail",
            "run_eagle3_next_action.py did not preserve the non-Slurm remote-probe execution contract",
            problems=problems,
            plan_returncode=plan_result.returncode,
            sheet_returncode=sheet_result.returncode,
            ready_preflight_returncode=ready_preflight_result.returncode,
            runner_returncode=runner_result.returncode,
            list_returncode=list_result.returncode,
            plan_error=plan_error,
            sheet_error=sheet_error,
            ready_preflight_error=ready_preflight_error,
            record_error=record_error,
            probe_error=probe_error,
            action=action,
            sheet_status=(sheet_payload or {}).get("overall_status"),
            ready_preflight_status=(ready_preflight_payload or {}).get("overall_status"),
            record=record_payload,
            probe_status=(probe_payload or {}).get("overall_status"),
            list_output=list_result.stdout[-2000:],
            output_tail=combined[-4000:],
        )
        return
    add(
        checks,
        "remote probe ready action executes through runner",
        "pass",
        "probe_remote_hosts can be planned and executed through the operator runner without Slurm or heavy GPU flags",
        runner_returncode=runner_result.returncode,
        list_returncode=list_result.returncode,
        ready_preflight_status=(ready_preflight_payload or {}).get("overall_status"),
        probe_status=(probe_payload or {}).get("overall_status"),
        reachable_hosts=(probe_payload or {}).get("reachable_hosts"),
        after_command_count=len(action.get("after_commands") or []),
    )


def validate_operator_resume_entrypoint(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "operator_resume_entrypoint"
    bin_dir = root / "fake_bin_resume"
    write_fake_ssh(bin_dir)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "ARTIFACT_ROOT": str(artifact),
        "REQUIRE_SLURM": "false",
        "REQUIRE_SOURCE_VLLM_RUNTIME": "false",
        "EXECUTE_SAFE_ACTIONS": "true",
        "SAFE_ACTION_IDS": "probe_remote_hosts",
        "EXECUTE_SLURM_ACTIONS": "false",
        "RUN_AFTER_SAFE_ACTIONS": "false",
        "RUN_FULL_REFRESH": "false",
    }
    result = run(
        [
            "bash",
            "experiments/eagle3_qwen3_235b/resume_eagle3_operator_state.sh",
        ],
        env=env,
    )
    report_dir = artifact / "reports"
    ready_preflight_json = report_dir / "eagle3_operator_ready_submit_preflight.json"
    queue_json = report_dir / "eagle3_operator_queue.json"
    record_json = report_dir / "operator_execution/auto_probe_remote_hosts.json"
    remote_probe_json = report_dir / "eagle3_remote_host_probe.json"
    ready_preflight_payload, ready_preflight_error = read_json(ready_preflight_json)
    queue_payload, queue_error = read_json(queue_json)
    record_payload, record_error = read_json(record_json)
    probe_payload, probe_error = read_json(remote_probe_json)
    combined = "\n".join(
        [
            result.stdout,
            file_text(ready_preflight_json),
            file_text(queue_json),
            file_text(record_json),
            file_text(remote_probe_json),
        ]
    )
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    problems = []
    if result.returncode != 0:
        problems.append("resume_eagle3_operator_state.sh returned nonzero")
    if ready_preflight_error or (ready_preflight_payload or {}).get("overall_status") != "pass":
        problems.append("resume entrypoint did not write PASS ready-submit preflight")
    if set((ready_preflight_payload or {}).get("action_filter") or []) != {"probe_remote_hosts"}:
        problems.append("resume entrypoint did not scope ready-submit preflight to the safe action allowlist")
    if {
        str(item.get("id") or "")
        for item in (ready_preflight_payload or {}).get("ready_actions") or []
        if isinstance(item, dict)
    } != {"probe_remote_hosts"}:
        problems.append("resume entrypoint ready-submit preflight included actions outside the safe allowlist")
    if queue_error or (queue_payload or {}).get("overall_status") not in {"ready_for_operator_submit", "blocked"}:
        problems.append("resume entrypoint did not write a valid operator queue summary")
    if record_error or (record_payload or {}).get("mode") != "execute":
        problems.append("resume entrypoint did not write an execute-mode record for probe_remote_hosts")
    if (record_payload or {}).get("returncode") != 0:
        problems.append("resume entrypoint probe_remote_hosts record returncode is not zero")
    if ((record_payload or {}).get("action") or {}).get("id") != "probe_remote_hosts":
        problems.append("resume entrypoint execution record action id is not probe_remote_hosts")
    if probe_error or (probe_payload or {}).get("overall_status") != "pass":
        problems.append("resume entrypoint did not refresh remote probe with fake-ssh PASS")
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if traces:
        problems.append("traceback leaked into resume entrypoint output")
    if problems:
        add(
            checks,
            "operator resume entrypoint refreshes and executes safe gates",
            "fail",
            "resume_eagle3_operator_state.sh did not preserve the remote-resume no-submit/safe-action contract",
            problems=problems,
            returncode=result.returncode,
            ready_preflight_status=(ready_preflight_payload or {}).get("overall_status"),
            queue_status=(queue_payload or {}).get("overall_status"),
            record=record_payload,
            probe_status=(probe_payload or {}).get("overall_status"),
            output_tail=combined[-4000:],
        )
        return
    add(
        checks,
        "operator resume entrypoint refreshes and executes safe gates",
        "pass",
        "resume_eagle3_operator_state.sh rebuilds the operator reports and can execute the allowlisted non-Slurm remote probe",
        ready_preflight_status=(ready_preflight_payload or {}).get("overall_status"),
        action_filter=(ready_preflight_payload or {}).get("action_filter"),
        queue_status=(queue_payload or {}).get("overall_status"),
        record_returncode=(record_payload or {}).get("returncode"),
        probe_status=(probe_payload or {}).get("overall_status"),
    )


def validate_ready_submit_action_filter(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "ready_submit_action_filter"
    report_dir = artifact / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    sheet_json = report_dir / "eagle3_operator_sheet.json"
    packet_validation_json = report_dir / "eagle3_operator_submit_packet_validation.json"
    json_out = report_dir / "eagle3_operator_ready_submit_preflight.json"
    markdown_out = report_dir / "eagle3_operator_ready_submit_preflight.md"
    missing_container = artifact / "missing_container.sqsh"
    sheet_json.write_text(
        json.dumps(
            {
                "overall_status": "ready_for_operator",
                "ready_actions": [
                    {
                        "id": "probe_remote_hosts",
                        "status": "ready_for_operator",
                        "submits_slurm": False,
                        "heavy_gpu": False,
                        "execution_record": str(report_dir / "operator_execution/probe_remote_hosts.json"),
                        "raw_command": (
                            "python3 experiments/eagle3_qwen3_235b/probe_eagle3_remote_host.py "
                            "--hosts badhost "
                            "--remote-workdir /lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap "
                            f"--artifact-root {artifact} "
                            f"--json-out {report_dir / 'eagle3_remote_host_probe.json'} "
                            f"--markdown-out {report_dir / 'eagle3_remote_host_probe.md'}"
                        ),
                    },
                    {
                        "id": "submit_container_preflight",
                        "status": "ready_for_operator",
                        "submits_slurm": True,
                        "heavy_gpu": False,
                        "execution_record": str(report_dir / "operator_execution/submit_container_preflight.json"),
                        "followup_record": str(report_dir / "operator_followups/submit_container_preflight.json"),
                        "raw_command": (
                            "SUBMIT=true "
                            "SBATCH_ACCOUNT=coreai_dlalgo_nemorl "
                            "PREFLIGHT_GPUS_PER_NODE=1 "
                            f"ARTIFACT_ROOT={artifact} "
                            f"CONTAINER={missing_container} "
                            "MODELOPT_DIR=/missing/modelopt "
                            f"VERIFIER_CONFIG_DIR={artifact / 'verifier_config'} "
                            f"INPUT_DATA={artifact / 'data/qwen3_235b_swe_rollout_conversations.jsonl'} "
                            f"CHAT_TEMPLATE={artifact / 'templates/qwen3_generation_template.jinja2'} "
                            "bash experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh"
                        ),
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    packet_validation_json.write_text(
        json.dumps({"overall_status": "pass", "counts": {"pass": 1}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/preflight_eagle3_operator_ready_submit.py",
            "--artifact-root",
            str(artifact),
            "--operator-sheet-json",
            str(sheet_json),
            "--operator-submit-packet-validation-json",
            str(packet_validation_json),
            "--action-ids",
            "probe_remote_hosts",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--no-require-slurm",
        ]
    )
    payload, payload_error = read_json(json_out)
    combined = "\n".join([result.stdout, file_text(json_out), file_text(markdown_out)])
    problems = []
    if result.returncode != 0 or payload_error:
        problems.append("filtered ready-submit preflight did not return PASS JSON")
    if (payload or {}).get("overall_status") != "pass":
        problems.append("filtered ready-submit preflight did not pass")
    if set((payload or {}).get("action_filter") or []) != {"probe_remote_hosts"}:
        problems.append("filtered ready-submit preflight did not record the requested action_filter")
    ready_ids = {
        str(item.get("id") or "")
        for item in (payload or {}).get("ready_actions") or []
        if isinstance(item, dict)
    }
    if ready_ids != {"probe_remote_hosts"}:
        problems.append("filtered ready-submit preflight included the unrequested Slurm action")
    if "submit_container_preflight" in combined:
        problems.append("filtered ready-submit preflight inspected the unrequested Slurm action")
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if "Traceback (most recent call last)" in combined:
        problems.append("traceback leaked into filtered ready-submit preflight output")
    if problems:
        add(
            checks,
            "ready-submit preflight supports action filtering",
            "fail",
            "safe-action preflight must not be blocked by unrequested Slurm/container gates",
            problems=problems,
            returncode=result.returncode,
            payload=payload,
            output_tail=combined[-4000:],
        )
        return
    add(
        checks,
        "ready-submit preflight supports action filtering",
        "pass",
        "ready-submit preflight can validate only the requested safe action while leaving Slurm/container gates untouched",
        action_filter=(payload or {}).get("action_filter"),
        ready_actions=(payload or {}).get("ready_actions"),
    )


def validate_remote_resume_slurm_action_contract(root: Path, checks: list[dict[str, Any]]) -> None:
    result = run(
        [
            "bash",
            "experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh",
        ],
        env={
            "PRINT_ONLY": "true",
            "REMOTE_HOST": "oci-hsg-cs-001-vscode-02",
            "REMOTE_WORKDIR": "/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap",
            "REMOTE_ARTIFACT_ROOT": "/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3",
            "REMOTE_ENTRYPOINT": "experiments/eagle3_qwen3_235b/resume_eagle3_operator_state.sh",
            "RUN_STATIC_INPUT_PREP": "auto",
            "STATIC_INPUT_SOURCE_DIR": "/lustre/fsw/portfolios/coreai/users/sna/qwen3_static_snapshot",
            "STATIC_INPUT_SKIP_TEMPLATE_VALIDATION": "true",
            "REVISION": "main",
            "EXECUTE_SAFE_ACTIONS": "true",
            "SAFE_ACTION_IDS": "probe_remote_hosts poll_megatron_compat_probe",
            "EXECUTE_SLURM_ACTIONS": "true",
            "SLURM_ACTION_IDS": "submit_vllm_source_build submit_source_vllm_abi_probe submit_container_preflight",
            "RUN_AFTER_SLURM_ACTIONS": "false",
            "ALLOW_HEAVY_GPU_ACTIONS": "false",
        },
    )
    output = (result.stdout or "") + (result.stderr or "")
    resume_text = file_text(ROOT / "experiments/eagle3_qwen3_235b/resume_eagle3_operator_state.sh")
    wrapper_text = file_text(ROOT / "experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh")
    required_output = [
        "resume_eagle3_operator_state.sh",
        "RUN_STATIC_INPUT_PREP=auto",
        "STATIC_INPUT_SOURCE_DIR=/lustre/fsw/portfolios/coreai/users/sna/qwen3_static_snapshot",
        "STATIC_INPUT_SKIP_TEMPLATE_VALIDATION=true",
        "REVISION=main",
        "EXECUTE_SLURM_ACTIONS=true",
        "SLURM_ACTION_IDS=submit_vllm_source_build",
        "submit_source_vllm_abi_probe",
        "submit_container_preflight",
        "RUN_AFTER_SLURM_ACTIONS=false",
        "ALLOW_HEAVY_GPU_ACTIONS=false",
    ]
    required_resume = [
        "EXECUTE_SLURM_ACTIONS",
        "SLURM_ACTION_IDS",
        "ALLOW_HEAVY_GPU_ACTIONS",
        "RUN_AFTER_SLURM_ACTIONS",
        "--allow-slurm",
        "submits_slurm",
        "heavy_gpu",
    ]
    required_wrapper = [
        "RUN_STATIC_INPUT_PREP",
        "STATIC_INPUT_SOURCE_DIR",
        "STATIC_INPUT_SKIP_TEMPLATE_VALIDATION",
        "STATIC_INPUT_MODEL_OR_TOKENIZER",
        "EXECUTE_SLURM_ACTIONS",
        "SLURM_ACTION_IDS",
        "ALLOW_HEAVY_GPU_ACTIONS",
    ]
    problems = []
    if result.returncode != 0:
        problems.append(f"remote wrapper print-only returned {result.returncode}")
    for snippet in required_output:
        if snippet not in output:
            problems.append(f"remote wrapper output missing {snippet}")
    for snippet in required_resume:
        if snippet not in resume_text:
            problems.append(f"resume script missing {snippet}")
    for snippet in required_wrapper:
        if snippet not in wrapper_text:
            problems.append(f"remote wrapper missing {snippet}")
    for forbidden in ["ALLOW_HEAVY_GPU_ACTIONS=true", "SUBMIT_ROLLOUT=true"]:
        if forbidden in output:
            problems.append(f"remote wrapper output contains forbidden snippet {forbidden}")
    leaked = contains_any(output, list(SECRET_SENTINELS.values()))
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if problems:
        add(
            checks,
            "remote resume can target non-heavy Slurm gates explicitly",
            "fail",
            "remote resume Slurm-action contract regressed",
            problems=problems,
            returncode=result.returncode,
            output_tail=output[-4000:],
        )
        return
    add(
        checks,
        "remote resume can target non-heavy Slurm gates explicitly",
        "pass",
        "remote wrapper forwards static-input and explicit non-heavy Slurm action controls without enabling heavy GPU submits",
    )


def validate_rollout_resource_profiles(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "rollout_resource_profiles"
    report = artifact / "reports/rollout_resource_profiles_preflight.json"
    markdown = artifact / "reports/rollout_resource_profiles_preflight.md"
    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/preflight_rollout_resource_profiles.py",
            "--artifact-root",
            str(artifact),
            "--repo-root",
            str(artifact / "missing-SpecDec-RL"),
            "--json-out",
            str(report),
            "--markdown-out",
            str(markdown),
        ]
    )
    payload, error = read_json(report)
    combined = "\n".join([result.stdout, file_text(report), file_text(markdown)])
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    profiles = (payload or {}).get("profiles") or []
    slurm_codes = [profile.get("slurm_test_returncode") for profile in profiles if isinstance(profile, dict)]
    if (
        result.returncode == 1
        and not error
        and (payload or {}).get("overall_status") == "fail"
        and 127 in slurm_codes
        and not leaked
        and not traces
    ):
        add(
            checks,
            "rollout resource profile preflight fails cleanly",
            "pass",
            "missing remote repo/Slurm state produces structured FAIL evidence without secret leakage",
            returncode=result.returncode,
            slurm_test_returncodes=slurm_codes,
        )
        return
    add(
        checks,
        "rollout resource profile preflight fails cleanly",
        "fail",
        "rollout resource profile preflight did not preserve the lightweight-host contract",
        returncode=result.returncode,
        json_error=error,
        overall_status=(payload or {}).get("overall_status"),
        slurm_test_returncodes=slurm_codes,
        leaked_sentinels=leaked,
        tracebacks=traces,
        output_tail=result.stdout[-4000:],
    )


def validate_unproven_slurm_profile_preserves_pipeline_env(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "unproven_slurm_resource_profile"
    report_dir = artifact / "reports"
    capacity_json = report_dir / "eagle3_slurm_capacity.json"
    capacity_md = report_dir / "eagle3_slurm_capacity.md"
    profile_env = report_dir / "eagle3_resource_profile.env"
    application_json = report_dir / "eagle3_resource_profile_application.json"
    application_md = report_dir / "eagle3_resource_profile_application.md"
    bin_dir = root / "fake_bin_no_sinfo"
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_sinfo = bin_dir / "sinfo"
    fake_sinfo.write_text("#!/usr/bin/env bash\necho 'sinfo unavailable on lightweight host' >&2\nexit 127\n", encoding="utf-8")
    fake_sinfo.chmod(0o755)
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}

    capacity = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/probe_eagle3_slurm_capacity.py",
            "--artifact-root",
            str(artifact),
            "--dump-gpus-per-node",
            "4",
            "--train-gpus-per-node",
            "4",
            "--export-gpus-per-node",
            "1",
            "--tp",
            "4",
            "--json-out",
            str(capacity_json),
            "--markdown-out",
            str(capacity_md),
            "--env-out",
            str(profile_env),
        ],
        env=env,
    )
    application = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/validate_eagle3_resource_profile_application.py",
            "--artifact-root",
            str(artifact),
            "--resource-profile-env",
            str(profile_env),
            "--json-out",
            str(application_json),
            "--markdown-out",
            str(application_md),
        ],
        env=env,
    )

    capacity_payload, capacity_error = read_json(capacity_json)
    application_payload, application_error = read_json(application_json)
    profile_text = file_text(profile_env)
    combined = "\n".join(
        [
            capacity.stdout,
            application.stdout,
            file_text(capacity_json),
            file_text(capacity_md),
            profile_text,
            file_text(application_json),
            file_text(application_md),
        ]
    )
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    required_env = {
        "DUMP_GPUS_PER_NODE": "4",
        "TRAIN_GPUS_PER_NODE": "4",
        "EXPORT_GPUS_PER_NODE": "1",
        "TP": "4",
        "EAGLE3_RESOURCE_PROFILE_STATUS": "slurm_shape_unproven",
    }
    missing_env = [
        f"{key}={value}"
        for key, value in required_env.items()
        if f"export {key}={value}" not in profile_text
    ]
    profile = (capacity_payload or {}).get("resource_profile") if isinstance((capacity_payload or {}).get("resource_profile"), dict) else {}
    problems = []
    if capacity.returncode != 0 or capacity_error:
        problems.append("capacity probe did not write JSON with zero returncode")
    if (capacity_payload or {}).get("overall_status") != "warn":
        problems.append("capacity probe should report warn when Slurm shape is unproven")
    if profile.get("status") != "requested_unverified":
        problems.append("capacity probe did not mark the profile requested_unverified")
    if missing_env:
        problems.append(f"resource profile env is missing requested values: {missing_env}")
    if application.returncode != 0 or application_error:
        problems.append("resource profile application validator did not pass")
    if (application_payload or {}).get("overall_status") != "pass":
        problems.append("resource profile application validator did not report pass")
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if traces:
        problems.append("traceback leaked into resource profile output")
    if problems:
        add(
            checks,
            "unproven Slurm profile preserves pipeline env",
            "fail",
            "resource profile probe/application contract regressed on lightweight hosts",
            problems=problems,
            capacity_returncode=capacity.returncode,
            application_returncode=application.returncode,
            capacity_status=(capacity_payload or {}).get("overall_status"),
            profile=profile,
            application_status=(application_payload or {}).get("overall_status"),
            output_tail=combined[-4000:],
        )
        return
    add(
        checks,
        "unproven Slurm profile preserves pipeline env",
        "pass",
        "missing sinfo keeps capacity unproven while preserving requested GPU/TP values through the pipeline dry-run",
        capacity_status=(capacity_payload or {}).get("overall_status"),
        profile_status=profile.get("status"),
        application_status=(application_payload or {}).get("overall_status"),
    )


def validate_pipeline_submit_preflight(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "pipeline_submit"
    report = artifact / "reports/eagle3_pipeline_submit_preflight.json"
    markdown = artifact / "reports/eagle3_pipeline_submit_preflight.md"
    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline_submit.py",
            "--artifact-root",
            str(artifact),
            "--json-out",
            str(report),
            "--markdown-out",
            str(markdown),
        ]
    )
    payload, error = read_json(report)
    combined = "\n".join([result.stdout, file_text(report), file_text(markdown)])
    leaked_defaults = contains_any(combined, [DEFAULT_REMOTE_ARTIFACT_ROOT])
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    rooted_paths = [
        str((payload or {}).get("input_data") or ""),
        str((payload or {}).get("hidden_states_dir") or ""),
        str((payload or {}).get("slurm_capacity_json") or ""),
        str((payload or {}).get("slurm_capacity_env") or ""),
    ]
    wrong_paths = [path for path in rooted_paths if path and not path.startswith(str(artifact))]
    if result.returncode == 0 and not error and (payload or {}).get("artifact_root") == str(artifact) and not leaked_defaults and not traces and not wrong_paths:
        add(
            checks,
            "pipeline submit preflight honors artifact root",
            "pass",
            "temporary artifact roots propagate to nested preflight reports without leaking the remote default root",
            returncode=result.returncode,
            overall_status=(payload or {}).get("overall_status"),
            submit_ready=(payload or {}).get("submit_ready"),
        )
        return
    add(
        checks,
        "pipeline submit preflight honors artifact root",
        "fail",
        "pipeline submit preflight leaked default remote paths or failed to write structured evidence",
        returncode=result.returncode,
        json_error=error,
        artifact_root=(payload or {}).get("artifact_root"),
        leaked_defaults=leaked_defaults,
        tracebacks=traces,
        wrong_paths=wrong_paths,
        output_tail=result.stdout[-4000:],
    )


def write_conversation_jsonl(path: Path, rows: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for idx in range(rows):
            fh.write(
                json.dumps(
                    {
                        "conversation_id": f"rollout-{idx}",
                        "messages": [
                            {"role": "user", "content": f"fix issue {idx}"},
                            {"role": "assistant", "content": f"patch result {idx}"},
                        ],
                    }
                )
                + "\n"
            )


def validate_corpus_strategy_rollout_provenance(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "corpus_strategy_contract"
    reports = artifact / "reports"
    data_dir = artifact / "data"
    reports.mkdir(parents=True, exist_ok=True)
    good_input = data_dir / "qwen3_235b_swe_rollout_conversations.jsonl"
    wrong_input = data_dir / "wrong_valid_but_not_rollout_output.jsonl"
    write_conversation_jsonl(good_input, rows=2)
    write_conversation_jsonl(wrong_input, rows=2)
    rollout_json = reports / "rollout_capture_analysis.json"
    rollout_json.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "output_data": {"path": str(good_input), "status": "pass"},
                "train_data": {"file_count": 1, "extractable_conversations": 2, "invalid_json": 0},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    good_report = reports / "corpus_strategy_good.json"
    mismatch_report = reports / "corpus_strategy_mismatch.json"
    good = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/analyze_corpus_strategy.py",
            "--artifact-root",
            str(artifact),
            "--input-data",
            str(good_input),
            "--rollout-capture-analysis-json",
            str(rollout_json),
            "--json-out",
            str(good_report),
        ]
    )
    mismatch = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/analyze_corpus_strategy.py",
            "--artifact-root",
            str(artifact),
            "--input-data",
            str(wrong_input),
            "--rollout-capture-analysis-json",
            str(rollout_json),
            "--json-out",
            str(mismatch_report),
        ]
    )
    good_payload, good_error = read_json(good_report)
    mismatch_payload, mismatch_error = read_json(mismatch_report)
    good_provenance = (good_payload or {}).get("rollout_alignment") or {}
    mismatch_provenance = (mismatch_payload or {}).get("rollout_alignment") or {}
    combined = "\n".join([good.stdout, mismatch.stdout, file_text(good_report), file_text(mismatch_report)])
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    problems = []
    if good.returncode != 0 or good_error:
        problems.append("valid rollout corpus strategy did not return zero and write JSON")
    if (good_payload or {}).get("overall_status") != "pass":
        problems.append("valid rollout corpus strategy did not report pass")
    if good_provenance.get("proves_actual_rollout_corpus") is not True:
        problems.append("valid rollout corpus strategy did not prove actual rollout provenance")
    if mismatch.returncode == 0 or mismatch_error:
        problems.append("mismatched rollout corpus strategy should return nonzero and write JSON")
    if (mismatch_payload or {}).get("overall_status") != "fail":
        problems.append("mismatched rollout corpus strategy did not report fail")
    if mismatch_provenance.get("output_matches_input") is not False:
        problems.append("mismatched rollout corpus strategy did not record output/input mismatch")
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if traces:
        problems.append("traceback leaked into corpus strategy output")
    if problems:
        add(
            checks,
            "corpus strategy requires rollout output provenance",
            "fail",
            "corpus strategy accepted weak or mismatched rollout evidence",
            problems=problems,
            good_returncode=good.returncode,
            mismatch_returncode=mismatch.returncode,
            good_status=(good_payload or {}).get("overall_status"),
            mismatch_status=(mismatch_payload or {}).get("overall_status"),
            output_tail=combined[-4000:],
        )
        return
    add(
        checks,
        "corpus strategy requires rollout output provenance",
        "pass",
        "SWE/RL corpus PASS requires validated INPUT_DATA to match the rollout analysis output path",
        good_status=(good_payload or {}).get("overall_status"),
        mismatch_status=(mismatch_payload or {}).get("overall_status"),
    )


def validate_hayate_reference_analyzers(root: Path, checks: list[dict[str, Any]]) -> None:
    report_dir = root / "hayate_references"
    workflow_json = report_dir / "hayate_modelopt_workflow.json"
    workflow_md = report_dir / "hayate_modelopt_workflow.md"
    workflow_fallback_json = report_dir / "hayate_modelopt_workflow_fallback.json"
    workflow_fallback_md = report_dir / "hayate_modelopt_workflow_fallback.md"
    specforge_json = report_dir / "hayate_specforge_reference.json"
    specforge_md = report_dir / "hayate_specforge_reference.md"
    specforge_fallback_json = report_dir / "hayate_specforge_reference_fallback.json"
    specforge_fallback_md = report_dir / "hayate_specforge_reference_fallback.md"
    workflow = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/analyze_hayate_modelopt_workflow.py",
            "--hayate-modelopt-dir",
            str(root / "missing-hayate-modelopt"),
            "--disable-bundled-fallback",
            "--json-out",
            str(workflow_json),
            "--markdown-out",
            str(workflow_md),
        ]
    )
    workflow_fallback = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/analyze_hayate_modelopt_workflow.py",
            "--hayate-modelopt-dir",
            str(root / "missing-hayate-modelopt"),
            "--json-out",
            str(workflow_fallback_json),
            "--markdown-out",
            str(workflow_fallback_md),
        ]
    )
    specforge = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/analyze_hayate_specforge_reference.py",
            "--specforge-dir",
            str(root / "missing-specforge"),
            "--artifact-root",
            str(root / "missing-artifact-root"),
            "--disable-bundled-fallback",
            "--json-out",
            str(specforge_json),
            "--markdown-out",
            str(specforge_md),
        ]
    )
    specforge_fallback = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/analyze_hayate_specforge_reference.py",
            "--specforge-dir",
            str(root / "missing-specforge"),
            "--artifact-root",
            str(root / "missing-artifact-root"),
            "--json-out",
            str(specforge_fallback_json),
            "--markdown-out",
            str(specforge_fallback_md),
        ]
    )
    workflow_payload, workflow_error = read_json(workflow_json)
    workflow_fallback_payload, workflow_fallback_error = read_json(workflow_fallback_json)
    specforge_payload, specforge_error = read_json(specforge_json)
    specforge_fallback_payload, specforge_fallback_error = read_json(specforge_fallback_json)
    combined = "\n".join(
        [
            workflow.stdout,
            workflow_fallback.stdout,
            specforge.stdout,
            specforge_fallback.stdout,
            file_text(workflow_json),
            file_text(workflow_md),
            file_text(workflow_fallback_json),
            file_text(workflow_fallback_md),
            file_text(specforge_json),
            file_text(specforge_md),
            file_text(specforge_fallback_json),
            file_text(specforge_fallback_md),
        ]
    )
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    problems = []
    if workflow.returncode != 0 or workflow_error:
        problems.append("Hayate ModelOpt workflow analyzer did not write JSON with zero returncode")
    if (workflow_payload or {}).get("overall_status") != "missing_reference":
        problems.append("Hayate ModelOpt workflow analyzer did not report missing_reference")
    if not (workflow_payload or {}).get("inspected_paths"):
        problems.append("Hayate ModelOpt workflow analyzer did not record inspected_paths")
    if workflow_fallback.returncode != 0 or workflow_fallback_error:
        problems.append("Hayate ModelOpt workflow bundled fallback did not write JSON with zero returncode")
    if (workflow_fallback_payload or {}).get("overall_status") != "reference_only":
        problems.append("Hayate ModelOpt workflow bundled fallback did not report reference_only")
    if (workflow_fallback_payload or {}).get("source") != "bundled_remote_drift_snapshot":
        problems.append("Hayate ModelOpt workflow bundled fallback did not record bundled_remote_drift_snapshot source")
    if (workflow_fallback_payload or {}).get("live_hayate_visible") is not False:
        problems.append("Hayate ModelOpt workflow bundled fallback did not record live_hayate_visible=false")
    if not (workflow_fallback_payload or {}).get("qwen_configs"):
        problems.append("Hayate ModelOpt workflow bundled fallback did not preserve visible Qwen config evidence")
    if specforge.returncode != 0 or specforge_error:
        problems.append("Hayate SpecForge analyzer did not write JSON with zero returncode")
    if (specforge_payload or {}).get("overall_status") != "missing_reference":
        problems.append("Hayate SpecForge analyzer did not report missing_reference")
    comparison = (specforge_fallback_payload or {}).get("qwen3_235b_comparison") or {}
    fallback_rows = [row for row in comparison.get("rows") or [] if isinstance(row, dict)]
    fallback_mismatches = [row for row in fallback_rows if row.get("match") is False]
    if specforge_fallback.returncode != 0 or specforge_fallback_error:
        problems.append("Hayate SpecForge bundled fallback did not write JSON with zero returncode")
    if (specforge_fallback_payload or {}).get("overall_status") != "reference_only":
        problems.append("Hayate SpecForge bundled fallback did not report reference_only")
    if (specforge_fallback_payload or {}).get("source") != "bundled_reference_snapshot":
        problems.append("Hayate SpecForge bundled fallback did not record bundled_reference_snapshot source")
    if (specforge_fallback_payload or {}).get("live_specforge_visible") is not False:
        problems.append("Hayate SpecForge bundled fallback did not record live_specforge_visible=false")
    if comparison.get("status") != "reference_only" or not fallback_mismatches:
        problems.append("Hayate SpecForge bundled fallback did not preserve the Qwen3-235B comparison mismatches")
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if traces:
        problems.append("traceback leaked into analyzer output")
    if problems:
        add(
            checks,
            "Hayate reference analyzers record missing references cleanly",
            "fail",
            "Hayate reference analyzer missing-reference contract regressed",
            problems=problems,
            workflow_returncode=workflow.returncode,
            workflow_fallback_returncode=workflow_fallback.returncode,
            specforge_returncode=specforge.returncode,
            specforge_fallback_returncode=specforge_fallback.returncode,
            workflow_stdout=workflow.stdout[-2000:],
            workflow_fallback_stdout=workflow_fallback.stdout[-2000:],
            specforge_stdout=specforge.stdout[-2000:],
            specforge_fallback_stdout=specforge_fallback.stdout[-2000:],
        )
        return
    add(
        checks,
        "Hayate reference analyzers record missing references cleanly",
        "pass",
        "missing Hayate ModelOpt/SpecForge paths produce structured missing reports, and bundled snapshots can be used without claiming live path access",
        workflow_status=(workflow_payload or {}).get("overall_status"),
        workflow_fallback_status=(workflow_fallback_payload or {}).get("overall_status"),
        workflow_fallback_source=(workflow_fallback_payload or {}).get("source"),
        specforge_status=(specforge_payload or {}).get("overall_status"),
        specforge_fallback_status=(specforge_fallback_payload or {}).get("overall_status"),
        specforge_fallback_source=(specforge_fallback_payload or {}).get("source"),
        inspected_path_count=len((workflow_payload or {}).get("inspected_paths") or []),
    )


def validate_pipeline_followup_job_file_inference(root: Path, checks: list[dict[str, Any]]) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_eagle3_slurm_followups_under_test",
        ROOT / "experiments/eagle3_qwen3_235b/run_eagle3_slurm_followups.py",
    )
    if spec is None or spec.loader is None:
        add(
            checks,
            "pipeline follow-up job-file inference includes gated copy",
            "fail",
            "could not import run_eagle3_slurm_followups.py",
        )
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact = root / "pipeline_followup_artifacts"
    command = (
        f"python3 experiments/eagle3_qwen3_235b/submit_eagle3_pipeline_if_ready.py "
        f"--artifact-root {artifact} "
        f"--preflight-json {artifact / 'reports/eagle3_pipeline_submit_preflight.json'} "
        f"--json-out {artifact / 'reports/eagle3_pipeline_gated_submit.json'} "
        f"--markdown-out {artifact / 'reports/eagle3_pipeline_gated_submit.md'} "
        "--execute --allow-heavy-gpu"
    )
    after_commands = [
        "python3 experiments/eagle3_qwen3_235b/analyze_eagle3_pipeline.py "
        "--job-file latest_eagle3_pipeline_jobs.txt --logs-dir logs"
    ]
    paths = module.infer_job_files("submit_eagle3_pilot_pipeline", command, after_commands)
    observed = {str(path) for path in paths}
    expected = {
        str(ROOT / "latest_eagle3_pipeline_jobs.txt"),
        str(artifact / "reports/eagle3_pipeline_jobs.env"),
    }
    missing = sorted(expected - observed)
    if missing:
        add(
            checks,
            "pipeline follow-up job-file inference includes gated copy",
            "fail",
            "run_eagle3_slurm_followups.py does not inspect both live and artifact-copied pipeline job files",
            missing=missing,
            observed=sorted(observed),
        )
        return
    add(
        checks,
        "pipeline follow-up job-file inference includes gated copy",
        "pass",
        "pipeline Slurm follow-up guard inspects latest_eagle3_pipeline_jobs.txt and reports/eagle3_pipeline_jobs.env",
        observed=sorted(observed),
    )


def validate_container_preflight_missing_input_is_warning(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "container_preflight_missing_input"
    reports = artifact / "reports"
    verifier = artifact / "verifier_config"
    template = artifact / "templates/qwen3_generation_template.jinja2"
    container = artifact / "containers/nemo.sqsh"
    sheet_json = reports / "eagle3_operator_sheet.json"
    packet_validation_json = reports / "eagle3_operator_submit_packet_validation.json"
    ready_json = reports / "eagle3_operator_ready_submit_preflight.json"
    ready_md = reports / "eagle3_operator_ready_submit_preflight.md"
    missing_input = artifact / "data/missing_rollout_conversations.jsonl"

    verifier.mkdir(parents=True, exist_ok=True)
    template.parent.mkdir(parents=True, exist_ok=True)
    container.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (verifier / "config.json").write_text('{"model_type":"qwen3_moe"}\n', encoding="utf-8")
    template.write_text("{% generation %}{{ messages[-1]['content'] }}{% endgeneration %}\n", encoding="utf-8")
    container.write_text("synthetic container placeholder\n", encoding="utf-8")

    command = (
        f"SUBMIT=true ARTIFACT_ROOT={artifact} SBATCH_ACCOUNT=coreai_dlalgo_nemorl "
        f"SBATCH_PARTITION=batch PREFLIGHT_GPUS_PER_NODE=1 MODELOPT_DIR={ROOT / 'Model-Optimizer'} "
        f"VERIFIER_CONFIG_DIR={verifier} INPUT_DATA={missing_input} CHAT_TEMPLATE={template} "
        f"CONTAINER={container} MOUNTS=/lustre:/lustre "
        f"PREFLIGHT_JSON={reports / 'container_preflight_pipeline.json'} "
        f"PREFLIGHT_MARKDOWN={reports / 'container_preflight_pipeline.md'} "
        "bash experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh"
    )
    sheet_json.write_text(
        json.dumps(
            {
                "overall_status": "ready_for_operator",
                "ready_actions": [
                    {
                        "id": "submit_container_preflight",
                        "status": "ready_for_operator",
                        "command": command,
                        "submits_slurm": True,
                        "heavy_gpu": False,
                        "execution_record": str(reports / "operator_execution/submit_container_preflight.json"),
                        "followup_record": str(reports / "operator_followups/submit_container_preflight.json"),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    packet_validation_json.write_text(
        json.dumps({"overall_status": "pass", "counts": {"pass": 1}, "packet_status": "ready_for_operator_submit"}, indent=2) + "\n",
        encoding="utf-8",
    )
    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/preflight_eagle3_operator_ready_submit.py",
            "--artifact-root",
            str(artifact),
            "--operator-sheet-json",
            str(sheet_json),
            "--operator-submit-packet-validation-json",
            str(packet_validation_json),
            "--json-out",
            str(ready_json),
            "--markdown-out",
            str(ready_md),
            "--no-require-slurm",
        ]
    )
    payload, error = read_json(ready_json)
    rows = (payload or {}).get("checks") or []
    fail_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "fail"]
    input_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("name") == "submit_container_preflight input data"
    ]
    problems: list[str] = []
    if result.returncode != 0 or error:
        problems.append("operator ready-submit preflight did not write JSON with zero returncode")
    if fail_rows:
        problems.append(f"unexpected fail rows: {[row.get('name') for row in fail_rows]}")
    if not input_rows or input_rows[0].get("status") != "warn":
        problems.append("missing container-preflight INPUT_DATA was not downgraded to WARN")
    if (payload or {}).get("overall_status") != "warn":
        problems.append(f"overall_status is {(payload or {}).get('overall_status')!r}, expected warn")
    combined = "\n".join([result.stdout, file_text(ready_json), file_text(ready_md)])
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if traces:
        problems.append("traceback leaked into operator ready-submit preflight output")
    if problems:
        add(
            checks,
            "container preflight can precede rollout data",
            "fail",
            "operator ready-submit preflight over-required INPUT_DATA for the container-only runtime gate",
            problems=problems,
            returncode=result.returncode,
            overall_status=(payload or {}).get("overall_status"),
            input_rows=input_rows,
            fail_rows=fail_rows,
            output_tail=combined[-4000:],
        )
        return
    add(
        checks,
        "container preflight can precede rollout data",
        "pass",
        "missing rollout INPUT_DATA is a warning for submit_container_preflight, while runtime/container inputs remain checked",
        overall_status=(payload or {}).get("overall_status"),
        input_status=input_rows[0].get("status"),
        fail_count=len(fail_rows),
    )


def validate_vllm_runtime_ready_submit_checks(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "vllm_runtime_ready_submit"
    reports = artifact / "reports"
    python_site = artifact / "python_site"
    source_site = python_site / "vllm_0_10_2_cu129_torch28nv_source_py312"
    container = artifact / "containers/nemo.sqsh"
    source_job = artifact / "latest_vllm_native_source_build_job.txt"
    abi_job = artifact / "latest_vllm_native_abi_probe_job.txt"
    sheet_json = reports / "eagle3_operator_sheet.json"
    packet_validation_json = reports / "eagle3_operator_submit_packet_validation.json"
    ready_json = reports / "eagle3_operator_ready_submit_preflight.json"
    ready_md = reports / "eagle3_operator_ready_submit_preflight.md"
    bin_dir = root / "fake_bin_vllm_runtime"

    source_site.mkdir(parents=True, exist_ok=True)
    container.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    container.write_text("synthetic container placeholder\n", encoding="utf-8")
    for binary in ["sbatch", "squeue", "sacct"]:
        path = bin_dir / binary
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    source_command = (
        f"ARTIFACT_ROOT={artifact} SBATCH_ACCOUNT=coreai_dlalgo_nemorl SBATCH_PARTITION=batch "
        f"CONTAINER={container} GPUS_PER_NODE=1 OUTPUT_SITE={source_site} "
        f"JOB_FILE={source_job} SUBMIT=true "
        "bash experiments/eagle3_qwen3_235b/submit_vllm_native_source_build.sh"
    )
    abi_command = (
        f"ARTIFACT_ROOT={artifact} SBATCH_ACCOUNT=coreai_dlalgo_nemorl SBATCH_PARTITION=batch "
        f"CONTAINER={container} GPUS_PER_NODE=1 VLLM_SITE_CANDIDATES={source_site} "
        f"JOB_FILE={abi_job} SUBMIT=true "
        "bash experiments/eagle3_qwen3_235b/submit_vllm_native_abi_probe.sh"
    )
    sheet_json.write_text(
        json.dumps(
            {
                "overall_status": "ready_for_operator",
                "ready_actions": [
                    {
                        "id": "submit_vllm_source_build",
                        "status": "ready_for_operator",
                        "command": source_command,
                        "submits_slurm": True,
                        "heavy_gpu": False,
                        "execution_record": str(reports / "operator_execution/submit_vllm_source_build.json"),
                        "followup_record": str(reports / "operator_followups/submit_vllm_source_build.json"),
                    },
                    {
                        "id": "submit_source_vllm_abi_probe",
                        "status": "ready_for_operator",
                        "command": abi_command,
                        "submits_slurm": True,
                        "heavy_gpu": False,
                        "execution_record": str(reports / "operator_execution/submit_source_vllm_abi_probe.json"),
                        "followup_record": str(reports / "operator_followups/submit_source_vllm_abi_probe.json"),
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    packet_validation_json.write_text(
        json.dumps({"overall_status": "pass", "counts": {"pass": 1}, "packet_status": "ready_for_operator_submit"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/preflight_eagle3_operator_ready_submit.py",
            "--artifact-root",
            str(artifact),
            "--operator-sheet-json",
            str(sheet_json),
            "--operator-submit-packet-validation-json",
            str(packet_validation_json),
            "--json-out",
            str(ready_json),
            "--markdown-out",
            str(ready_md),
        ],
        env={"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"},
    )
    payload, error = read_json(ready_json)
    rows = [row for row in (payload or {}).get("checks") or [] if isinstance(row, dict)]
    row_status = {str(row.get("name")): str(row.get("status")) for row in rows}
    required_pass_rows = [
        "submit_vllm_source_build SUBMIT",
        "submit_vllm_source_build SBATCH_ACCOUNT",
        "submit_vllm_source_build container",
        "submit_vllm_source_build VLLM_SOURCE_SPEC",
        "submit_vllm_source_build source-build sbatch",
        "submit_source_vllm_abi_probe SUBMIT",
        "submit_source_vllm_abi_probe SBATCH_ACCOUNT",
        "submit_source_vllm_abi_probe container",
        "submit_source_vllm_abi_probe VLLM_SITE_CANDIDATES[1]",
        "submit_source_vllm_abi_probe ABI probe sbatch",
    ]
    problems = []
    if result.returncode != 0 or error:
        problems.append("operator ready-submit preflight did not return zero and write JSON")
    if (payload or {}).get("overall_status") != "pass":
        problems.append(f"overall_status is {(payload or {}).get('overall_status')!r}, expected pass")
    missing_or_bad = [name for name in required_pass_rows if row_status.get(name) != "pass"]
    if missing_or_bad:
        problems.append(f"vLLM source/ABI action checks did not pass: {missing_or_bad}")
    combined = "\n".join([result.stdout, file_text(ready_json), file_text(ready_md)])
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if traces:
        problems.append("traceback leaked into operator ready-submit preflight output")
    if problems:
        add(
            checks,
            "source vLLM Slurm gates have action-specific preflight",
            "fail",
            "operator ready-submit preflight did not validate source-build and source-ABI submit inputs",
            problems=problems,
            returncode=result.returncode,
            overall_status=(payload or {}).get("overall_status"),
            observed={name: row_status.get(name) for name in required_pass_rows},
            output_tail=combined[-4000:],
        )
        return
    add(
        checks,
        "source vLLM Slurm gates have action-specific preflight",
        "pass",
        "source-build and source-ABI actions validate submit mode, Slurm account, container, candidate/output paths, sbatch files, and job files",
        overall_status=(payload or {}).get("overall_status"),
        checked_rows=len(required_pass_rows),
    )


def validate_canonical_rollout_input_defaults(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "canonical_rollout_input_defaults"
    env_out = artifact / "eagle3_inputs.env"
    json_out = artifact / "eagle3_input_discovery.json"
    markdown_out = artifact / "eagle3_input_discovery.md"
    scan_root = artifact / "empty_scan_root"
    scan_root.mkdir(parents=True, exist_ok=True)
    sample_jsonl = scan_root / "sample_rollout_conversations.jsonl"
    sample_jsonl.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Fix the failing test."},
                    {"role": "assistant", "content": "The failing assertion should compare normalized paths."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/discover_eagle3_run_inputs.py",
            str(scan_root),
            "--artifact-root",
            str(artifact),
            "--env-out",
            str(env_out),
            "--markdown-out",
            str(markdown_out),
            "--json-out",
            str(json_out),
        ]
    )
    env_text = file_text(env_out)
    payload, payload_error = read_json(json_out)
    candidates = (payload or {}).get("conversation_candidates") or []
    sample_candidate = next(
        (item for item in candidates if item.get("path") == str(sample_jsonl)),
        None,
    )
    stale_files: dict[str, list[int]] = {}
    runtime_files = [
        ROOT / "experiments/eagle3_qwen3_235b/discover_eagle3_run_inputs.py",
        ROOT / "experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh",
        ROOT / "experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh",
        ROOT / "experiments/eagle3_qwen3_235b/analyze_corpus_strategy.py",
        ROOT / "experiments/eagle3_qwen3_235b/prepare_training_conversations.sh",
    ]
    for path in runtime_files:
        line_numbers = [
            idx
            for idx, line in enumerate(file_text(path).splitlines(), start=1)
            if LEGACY_BOOTSTRAP_INPUT in line and not line.lstrip().startswith("#")
        ]
        if line_numbers:
            stale_files[str(path.relative_to(ROOT))] = line_numbers

    problems = []
    if result.returncode not in {0, 2}:
        problems.append("input discovery did not write env defaults with an expected returncode")
    if not env_out.exists():
        problems.append("input discovery env file is missing")
    if CANONICAL_ROLLOUT_INPUT not in env_text:
        problems.append("input discovery env does not default INPUT_DATA to canonical rollout conversations")
    if LEGACY_BOOTSTRAP_INPUT in env_text:
        problems.append("input discovery env still references legacy bootstrap conversation filename")
    if payload_error:
        problems.append(f"input discovery JSON is not readable: {payload_error}")
    if not sample_candidate:
        problems.append("input discovery did not rank the synthetic rollout JSONL")
    elif sample_candidate.get("sample_error"):
        problems.append(f"input discovery recorded sample_error: {sample_candidate.get('sample_error')}")
    elif int(sample_candidate.get("extracted_rollout_conversations") or 0) < 1:
        problems.append("input discovery did not extract conversations from a messages JSONL row")
    if stale_files:
        problems.append("runtime entrypoints still contain non-comment legacy INPUT_DATA defaults")

    if problems:
        add(
            checks,
            "entrypoints default to canonical rollout corpus",
            "fail",
            "Qwen3-235B Eagle3 entrypoints may still default to a non-rollout corpus",
            problems=problems,
            stale_files=stale_files,
            sample_candidate=sample_candidate,
            env_tail=env_text[-2000:],
            returncode=result.returncode,
        )
        return
    add(
        checks,
        "entrypoints default to canonical rollout corpus",
        "pass",
        "input discovery parses messages JSONL rows and entrypoints default to qwen3_235b_swe_rollout_conversations.jsonl",
        env_file=str(env_out),
        extracted_rollout_conversations=sample_candidate.get("extracted_rollout_conversations") if sample_candidate else None,
    )


def validate_playbook_artifact_flow_contract(root: Path, checks: list[dict[str, Any]]) -> None:
    del root
    playbook_path = ROOT / "experiments/eagle3_qwen3_235b/EAGLE3_DRAFT_MODEL_PLAYBOOK.md"
    text = file_text(playbook_path)
    problems = []

    if "## Canonical Artifact Flow" not in text:
        problems.append("playbook does not contain a canonical artifact-flow section")
    missing_artifacts = [artifact_id for artifact_id in EXPECTED_PLAYBOOK_ARTIFACT_FLOW if artifact_id not in text]
    if missing_artifacts:
        problems.append(f"playbook is missing artifact-flow ids: {missing_artifacts}")
    missing_ready_actions = [action_id for action_id in EXPECTED_PLAYBOOK_READY_ACTIONS if action_id not in text]
    if missing_ready_actions:
        problems.append(f"playbook is missing current ready actions: {missing_ready_actions}")
    required_phrases = [
        "artifact_flow_complete",
        "proof_status=pass",
        "actual Qwen3 SWE/RL rollout corpus",
        "source-built vLLM ABI probe",
        "Megatron compatibility probe",
        "container preflight",
        "Hayate artifacts remain reference-only",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
    if missing_phrases:
        problems.append(f"playbook is missing guard phrases: {missing_phrases}")
    if problems:
        add(
            checks,
            "playbook records artifact-flow completion guard",
            "fail",
            "EAGLE3_DRAFT_MODEL_PLAYBOOK.md may allow operators to skip current gate order or final artifact proof",
            problems=problems,
            playbook=str(playbook_path),
        )
        return
    add(
        checks,
        "playbook records artifact-flow completion guard",
        "pass",
        "playbook documents the canonical artifact flow, ready runtime actions, and completion guard",
        playbook=str(playbook_path),
        artifact_flow=EXPECTED_PLAYBOOK_ARTIFACT_FLOW,
    )


def validate_handoff_bundle_uses_concrete_sbatch_account(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "handoff_concrete_sbatch"
    reports = artifact / "reports"
    out_dir = artifact / "handoff"
    cluster_probe = reports / "cluster_environment_probe.json"
    training_path_manifest = reports / "eagle3_training_path_manifest.json"
    operator_ready_submit = reports / "eagle3_operator_ready_submit_preflight.json"
    operator_safe_actions = reports / "eagle3_operator_safe_actions_preflight.json"
    reports.mkdir(parents=True, exist_ok=True)
    cluster_probe.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "environment": {
                    "sbatch_account": "coreai_dlalgo_nemorl",
                    "sbatch_partition": "batch",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    training_path_manifest.write_text(
        json.dumps(
            {
                "overall_status": "defined",
                "open_gates": ["runtime_container"],
                "gate_closure_contracts": [
                    {
                        "id": "runtime_container",
                        "closed": False,
                        "closure_evidence_missing": ["report:vllm_source_build"],
                        "candidate_next_action_ids": ["submit_vllm_source_build"],
                    }
                ],
                "operator_gate_action_matrix": [
                    {
                        "gate_id": "runtime_container",
                        "status": "ready_action_available",
                        "current_ready_action_ids": ["submit_vllm_source_build"],
                        "future_candidate_action_ids": ["submit_source_vllm_abi_probe"],
                        "missing_evidence": ["report:vllm_source_build"],
                    }
                ],
                "artifact_flow_complete": False,
                "artifact_flow": [
                    {
                        "id": "rollout_conversation_corpus",
                        "proof_status": "open",
                        "producer_gate": "target_rollout_corpus",
                        "consumer_gate": "hidden_train_export_submit",
                        "required_reports": ["rollout_state", "corpus_strategy"],
                        "required_invariants": ["primary_source=actual_rl_rollout"],
                        "closure_action_ids": ["submit_rollout_capture", "rollout_materialize_and_refresh"],
                        "current_closure_action_ids": [],
                        "future_closure_action_ids": ["submit_rollout_capture", "rollout_materialize_and_refresh"],
                        "report_statuses": {"rollout_state": "missing", "corpus_strategy": "missing_capture"},
                        "path_visible": False,
                        "path": str(artifact / "data/qwen3_235b_swe_rollout_conversations.jsonl"),
                    },
                    {
                        "id": "vllm_eagle3_draft",
                        "proof_status": "open",
                        "producer_gate": "hidden_train_export_submit",
                        "consumer_gate": "trained_artifact_contracts",
                        "required_reports": ["pipeline_analysis", "export_artifacts"],
                        "required_invariants": ["vllm_config_exists", "draft_weights_present"],
                        "closure_action_ids": [
                            "run_pipeline_submit_preflight",
                            "submit_eagle3_pilot_pipeline",
                            "run_post_export_artifact_validations",
                        ],
                        "current_closure_action_ids": [],
                        "future_closure_action_ids": [
                            "run_pipeline_submit_preflight",
                            "submit_eagle3_pilot_pipeline",
                            "run_post_export_artifact_validations",
                        ],
                        "report_statuses": {"pipeline_analysis": "incomplete", "export_artifacts": "missing"},
                        "path_visible": False,
                        "path": str(artifact / "vllm_draft"),
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    operator_ready_submit.write_text(
        json.dumps(
            {
                "overall_status": "fail",
                "counts": {"fail": 2, "pass": 4, "warn": 1},
                "checks": [
                    {"area": "slurm", "name": "sbatch", "status": "fail", "detail": "sbatch is not on PATH"},
                    {
                        "area": "path",
                        "name": "submit_container_preflight container",
                        "status": "fail",
                        "detail": "not visible: /cluster/container.sqsh",
                    },
                    {
                        "area": "path",
                        "name": "submit_container_preflight input data",
                        "status": "warn",
                        "detail": "not visible: rollout corpus",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    operator_safe_actions.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "counts": {"pass": 3},
                "ready_actions": [
                    {"id": "probe_remote_hosts", "submits_slurm": False, "heavy_gpu": False},
                ],
                "checks": [
                    {"area": "action", "name": "probe_remote_hosts", "status": "pass", "detail": "safe"},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/create_eagle3_handoff_bundle.py",
            "--artifact-root",
            str(artifact),
            "--out-dir",
            str(out_dir),
        ]
    )
    commands = file_text(out_dir / "commands.sh")
    runbook = file_text(out_dir / "RUNBOOK.md")
    manifest, manifest_error = read_json(out_dir / "manifest.json")
    noargs = run(["bash", str(out_dir / "commands.sh")])
    submit_denied = run(["bash", str(out_dir / "commands.sh"), "3_submit_pilot"])
    repo_denied = run(
        ["bash", str(out_dir / "commands.sh"), "0_collect_provenance"],
        env={"EAGLE3_REPO_ROOT": str(root / "missing_repo")},
    )
    run_all_selected = run(
        ["bash", str(out_dir / "commands.sh")],
        env={"EAGLE3_RUN_ALL_SECTIONS": "true", "EAGLE3_PRINT_SELECTED_SECTIONS": "true"},
    )
    problems = []
    if result.returncode != 0:
        problems.append("handoff bundle generation returned nonzero")
    if noargs.returncode != 0:
        problems.append("commands.sh no-argument mode returned nonzero")
    if "Default behavior is print-only" not in noargs.stdout:
        problems.append("commands.sh no-argument mode does not advertise print-only behavior")
    if "0_restore_materialized_static_inputs" not in noargs.stdout:
        problems.append("commands.sh no-argument mode does not list selectable sections")
    if "# === 0_restore_materialized_static_inputs ===" in noargs.stdout:
        problems.append("commands.sh no-argument mode executed a section")
    if submit_denied.returncode != 3:
        problems.append("commands.sh did not reject submit section without explicit submit allowance")
    if "refusing submit section 3_submit_pilot" not in submit_denied.stdout:
        problems.append("commands.sh submit-section rejection did not explain the required allowance")
    if "# === 3_submit_pilot ===" in submit_denied.stdout:
        problems.append("commands.sh printed or entered the submit section before checking the guard")
    if repo_denied.returncode != 4:
        problems.append("commands.sh did not fail closed when a command section used a bad EAGLE3_REPO_ROOT")
    if "EAGLE3_REPO_ROOT must point" not in repo_denied.stdout:
        problems.append("commands.sh bad-repo-root rejection did not explain how to fix EAGLE3_REPO_ROOT")
    if run_all_selected.returncode != 0:
        problems.append("commands.sh RUN_ALL selected-section introspection returned nonzero")
    selected_sections = {line.strip() for line in run_all_selected.stdout.splitlines() if line.strip()}
    submit_sections = {"3_submit_pilot", "5_sweep_trained_draft"}
    if selected_sections & submit_sections:
        problems.append("commands.sh RUN_ALL selected submit sections")
    if "0_restore_materialized_static_inputs" not in selected_sections:
        problems.append("commands.sh RUN_ALL selected-section introspection missed safe sections")
    if "export SBATCH_ACCOUNT=coreai_dlalgo_nemorl" not in commands:
        problems.append("commands.sh does not default SBATCH_ACCOUNT to the inferred concrete account")
    if 'MODELOPT_DIR="${MODELOPT_DIR:-$EAGLE3_REPO_ROOT/Model-Optimizer}"' not in commands:
        problems.append("commands.sh does not anchor MODELOPT_DIR to EAGLE3_REPO_ROOT")
    if "export SBATCH_ACCOUNT='<account>'" in commands or "export SBATCH_ACCOUNT=<account>" in commands:
        problems.append("commands.sh still contains an SBATCH_ACCOUNT placeholder")
    if manifest_error:
        problems.append(f"handoff manifest is not readable: {manifest_error}")
    if (manifest or {}).get("command_sbatch_account") != "coreai_dlalgo_nemorl":
        problems.append("handoff manifest does not record the concrete command_sbatch_account")
    training_summary = ((manifest or {}).get("summaries") or {}).get("training_path_manifest") or {}
    matrix = training_summary.get("operator_gate_action_matrix") if isinstance(training_summary, dict) else []
    if not matrix or matrix[0].get("current_ready_action_ids") != ["submit_vllm_source_build"]:
        problems.append("handoff manifest does not preserve operator gate/action matrix current ready actions")
    flow = training_summary.get("artifact_flow") if isinstance(training_summary, dict) else []
    if not flow or flow[0].get("id") != "rollout_conversation_corpus":
        problems.append("handoff manifest does not preserve artifact_flow summary")
    elif flow[0].get("report_statuses", {}).get("rollout_state") != "missing":
        problems.append("handoff manifest does not preserve artifact_flow report statuses")
    if "Command Slurm account default: `coreai_dlalgo_nemorl`" not in runbook:
        problems.append("runbook does not summarize the concrete Slurm account default")
    if "Current Ready Action Mapping" not in runbook or "submit_vllm_source_build" not in runbook:
        problems.append("runbook does not render the operator gate/action matrix")
    if "Artifact Flow" not in runbook or "rollout_conversation_corpus" not in runbook:
        problems.append("runbook does not render the artifact flow")
    if "rollout_state=missing" not in runbook or "primary_source=actual_rl_rollout" not in runbook:
        problems.append("runbook does not render artifact-flow proof requirements and current report statuses")
    if "submit_rollout_capture" not in runbook or "submit_eagle3_pilot_pipeline" not in runbook:
        problems.append("runbook does not render artifact-flow closure actions")
    if "Ready-submit blocker" not in runbook or "slurm/sbatch" not in runbook:
        problems.append("runbook does not render ready-submit failed checks")
    if "Safe-action preflight" not in runbook:
        problems.append("runbook does not render safe-action preflight status")
    combined = "\n".join([result.stdout, commands, runbook])
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if traces:
        problems.append("traceback leaked into handoff bundle output")
    if problems:
        add(
            checks,
            "handoff bundle carries concrete Slurm account",
            "fail",
            "generated handoff may require manual account placeholder edits before runtime/container gates",
            problems=problems,
            returncode=result.returncode,
            noargs_returncode=noargs.returncode,
            submit_denied_returncode=submit_denied.returncode,
            repo_denied_returncode=repo_denied.returncode,
            run_all_selected_returncode=run_all_selected.returncode,
            commands_head=commands[:1000],
            runbook_head=runbook[:1000],
            output_tail="\n".join(
                [result.stdout, noargs.stdout, submit_denied.stdout, repo_denied.stdout, run_all_selected.stdout]
            )[-2000:],
        )
        return
    add(
        checks,
        "handoff bundle carries concrete Slurm account",
        "pass",
        "commands.sh is section-selectable, repo-root anchored, excludes submit sections from RUN_ALL, guards submit sections, and uses the inferred concrete Slurm account instead of an account placeholder",
        command_sbatch_account=(manifest or {}).get("command_sbatch_account"),
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Preflight Robustness Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['name']} | {check['status'].upper()} | {check['detail']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="eagle3_preflight_robustness_"))
    checks: list[dict[str, Any]] = []
    try:
        validate_remote_host_probe_contract(temp_root, checks)
        validate_remote_access_diagnostics_contract(temp_root, checks)
        validate_remote_probe_runner_action(temp_root, checks)
        validate_operator_resume_entrypoint(temp_root, checks)
        validate_ready_submit_action_filter(temp_root, checks)
        validate_remote_resume_slurm_action_contract(temp_root, checks)
        validate_rollout_resource_profiles(temp_root, checks)
        validate_unproven_slurm_profile_preserves_pipeline_env(temp_root, checks)
        validate_pipeline_submit_preflight(temp_root, checks)
        validate_corpus_strategy_rollout_provenance(temp_root, checks)
        validate_hayate_reference_analyzers(temp_root, checks)
        validate_pipeline_followup_job_file_inference(temp_root, checks)
        validate_container_preflight_missing_input_is_warning(temp_root, checks)
        validate_vllm_runtime_ready_submit_checks(temp_root, checks)
        validate_canonical_rollout_input_defaults(temp_root, checks)
        validate_playbook_artifact_flow_contract(temp_root, checks)
        validate_handoff_bundle_uses_concrete_sbatch_account(temp_root, checks)
    finally:
        if args.keep_temp:
            checks.append(
                {
                    "name": "temporary artifacts",
                    "status": "info",
                    "detail": str(temp_root),
                    "evidence": {},
                }
            )
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    overall = "pass" if checks and all(check["status"] in {"pass", "info"} for check in checks) else "fail"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "counts": status_counts(checks),
        "check_status_counts": status_counts(checks),
        "checks": checks,
    }
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
