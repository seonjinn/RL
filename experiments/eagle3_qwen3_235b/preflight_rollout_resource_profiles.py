#!/usr/bin/env python3
"""No-submit validation for Qwen3-235B rollout-capture resource profiles."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_SPECDEC_RL_DIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
)
DEFAULT_ROLLOUT_CONTAINER = Path(
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"
)
DEFAULT_SOURCE_VLLM_SITE = (
    DEFAULT_ARTIFACT_ROOT / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312"
)
DEFAULT_SWEGYM_EXAMPLE_DATA = Path(
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/ultra/"
    "tk-nemo-gym/responses_api_agents/swe_agents/data/example.jsonl"
)


def default_train_data_path(artifact_root: Path) -> Path:
    fixed_example = artifact_root / "data/swegym_example_for_sweagent_with_instance_dict.jsonl"
    if fixed_example.exists():
        return fixed_example
    return DEFAULT_SWEGYM_EXAMPLE_DATA


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("REPO_ROOT") or os.environ.get("SWE_REPO_ROOT") or DEFAULT_SPECDEC_RL_DIR),
    )
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("CONFIG_FILE", ROOT / "grpo_qwen3_235b_swe.yaml")))
    parser.add_argument("--env-file", type=Path, default=Path(os.environ.get("ENV_FILE", ROOT / "env.sh")))
    parser.add_argument("--chat-template", type=Path, default=artifact_root / "templates/qwen3_generation_template.jinja2")
    parser.add_argument("--sbatch-account", default=os.environ.get("SBATCH_ACCOUNT", "coreai_dlalgo_nemorl"))
    parser.add_argument("--sbatch-partition", default=os.environ.get("SBATCH_PARTITION", "batch"))
    parser.add_argument("--container", type=Path, default=Path(os.environ.get("CONTAINER", DEFAULT_ROLLOUT_CONTAINER)))
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=Path(os.environ["TRAIN_DATA_PATH"]) if os.environ.get("TRAIN_DATA_PATH") else default_train_data_path(artifact_root),
        help="Effective NemoGym/SWE JSONL used for dry-run validation.",
    )
    parser.add_argument(
        "--val-data-path",
        type=Path,
        default=Path(os.environ["VAL_DATA_PATH"]) if os.environ.get("VAL_DATA_PATH") else None,
        help="Effective validation NemoGym/SWE JSONL used for dry-run validation.",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Also test leaner queue shapes. These are report-only and are not part of the default fallback selector.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=merged,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if exc.stderr:
            output = f"{output}{exc.stderr}"
        output = f"{output}\nTimed out after {timeout}s\n"
        return {
            "command": cmd,
            "returncode": 124,
            "output": output,
            "output_tail": output[-8000:],
        }
    except OSError as exc:
        output = f"{exc.__class__.__name__}: {exc}\n"
        return {
            "command": cmd,
            "returncode": 127,
            "output": output,
            "output_tail": output[-8000:],
        }
    return {
        "command": cmd,
        "returncode": result.returncode,
        "output": result.stdout,
        "output_tail": result.stdout[-8000:],
    }


def shell_join(env: dict[str, str], command: list[str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    body = " ".join(shlex.quote(part) for part in command)
    return f"{prefix} {body}".strip()


def int_env(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def resolved_parallelism(profile: dict[str, Any]) -> dict[str, int]:
    env = profile["env"]
    gpus_per_node = int_env(env, "NUM_GPU", int(profile["gpus_per_node"]))
    pp_default = 4 if gpus_per_node == 4 else 8
    return {
        "tensor_model_parallel_size": int_env(env, "TP", 4),
        "expert_tensor_parallel_size": int_env(env, "ETP", 1),
        "expert_model_parallel_size": int_env(env, "EP", 16),
        "pipeline_model_parallel_size": int_env(env, "PP", pp_default),
    }


def topology_checks(profile: dict[str, Any]) -> list[dict[str, Any]]:
    env = profile["env"]
    nodes = int_env(env, "NUM_NODES", int(profile["nodes"]))
    generation_nodes = int_env(env, "NUM_GEN_NODES", 0)
    gpus_per_node = int_env(env, "NUM_GPU", int(profile["gpus_per_node"]))
    train_nodes = nodes - generation_nodes
    train_world_size = train_nodes * gpus_per_node
    parallelism = resolved_parallelism(profile)
    expert_tensor_model_pipeline_size = (
        parallelism["expert_tensor_parallel_size"]
        * parallelism["expert_model_parallel_size"]
        * parallelism["pipeline_model_parallel_size"]
    )
    divisible = (
        train_nodes > 0
        and expert_tensor_model_pipeline_size > 0
        and train_world_size % expert_tensor_model_pipeline_size == 0
    )
    detail = (
        f"train_world_size=({nodes}-{generation_nodes})*{gpus_per_node}={train_world_size}; "
        "expert_tensor_model_pipeline_parallel="
        f"{parallelism['expert_tensor_parallel_size']}*{parallelism['expert_model_parallel_size']}"
        f"*{parallelism['pipeline_model_parallel_size']}={expert_tensor_model_pipeline_size}"
    )
    return [
        {
            "name": "megatron_train_world_size",
            "status": "pass" if divisible else "fail",
            "detail": detail,
            "train_nodes": train_nodes,
            "train_world_size": train_world_size,
            "expert_tensor_model_pipeline_parallel_size": expert_tensor_model_pipeline_size,
            **parallelism,
        }
    ]


def profile_defs(include_experimental: bool = False) -> list[dict[str, Any]]:
    profiles = [
        {
            "id": "official_32n4g_async",
            "detail": "SpecDec-RL official 32n4g async shape; same size class as the submitted rollout job.",
            "rollout_log_name": "qwen3_235b_swe_capture_smoke",
            "output_name": "qwen3_235b_swe_rollout_conversations.jsonl",
            "wandb_name": "qwen3-235b-swe-rollout-capture-smoke",
            "env": {
                "NUM_GPU": "4",
                "NUM_NODES": "32",
                "NUM_GEN_NODES": "16",
            },
            "nodes": 32,
            "gpus_per_node": 4,
        },
        {
            "id": "compact_16n4g_smoke",
            "detail": "Smaller 16n4g queue-shape candidate for 1-step rollout capture if the 32-node job stalls.",
            "rollout_log_name": "qwen3_235b_swe_capture_compact16n4g",
            "output_name": "qwen3_235b_swe_rollout_conversations_compact16n4g.jsonl",
            "wandb_name": "qwen3-235b-swe-rollout-capture-compact16n4g",
            "env": {
                "NUM_GPU": "4",
                "NUM_NODES": "16",
                "NUM_GEN_NODES": "8",
                "PPS": "4",
                "GPP": "2",
                "GBS": "8",
            },
            "nodes": 16,
            "gpus_per_node": 4,
        },
        {
            "id": "balanced_24n4g_smoke",
            "detail": "Topology-valid 24n4g fallback: 16 train nodes plus 8 generation nodes.",
            "rollout_log_name": "qwen3_235b_swe_capture_balanced24n4g",
            "output_name": "qwen3_235b_swe_rollout_conversations_balanced24n4g.jsonl",
            "wandb_name": "qwen3-235b-swe-rollout-capture-balanced24n4g",
            "env": {
                "NUM_GPU": "4",
                "NUM_NODES": "24",
                "NUM_GEN_NODES": "8",
            },
            "nodes": 24,
            "gpus_per_node": 4,
        },
    ]
    if include_experimental:
        profiles.extend(
            [
                {
                    "id": "experimental_20n4g_gen4_smoke",
                    "detail": (
                        "Report-only lean candidate: 16 train nodes plus 4 generation nodes. "
                        "This is not selected by the default fallback logic unless explicitly promoted."
                    ),
                    "rollout_log_name": "qwen3_235b_swe_capture_experimental20n4g_gen4",
                    "output_name": "qwen3_235b_swe_rollout_conversations_experimental20n4g_gen4.jsonl",
                    "wandb_name": "qwen3-235b-swe-rollout-capture-experimental20n4g-gen4",
                    "env": {
                        "NUM_GPU": "4",
                        "NUM_NODES": "20",
                        "NUM_GEN_NODES": "4",
                    },
                    "nodes": 20,
                    "gpus_per_node": 4,
                    "experimental": True,
                },
                {
                    "id": "experimental_18n4g_gen2_smoke",
                    "detail": (
                        "Report-only minimal topology candidate: 16 train nodes plus 2 generation nodes. "
                        "This is likely generation-memory risky and must not be auto-selected."
                    ),
                    "rollout_log_name": "qwen3_235b_swe_capture_experimental18n4g_gen2",
                    "output_name": "qwen3_235b_swe_rollout_conversations_experimental18n4g_gen2.jsonl",
                    "wandb_name": "qwen3-235b-swe-rollout-capture-experimental18n4g-gen2",
                    "env": {
                        "NUM_GPU": "4",
                        "NUM_NODES": "18",
                        "NUM_GEN_NODES": "2",
                    },
                    "nodes": 18,
                    "gpus_per_node": 4,
                    "experimental": True,
                },
            ]
        )
    return profiles


def submit_env(args: argparse.Namespace, profile: dict[str, Any]) -> dict[str, str]:
    rollout_log_dir = args.artifact_root / "rl_rollout_capture_logs" / str(profile["rollout_log_name"])
    output_conversations = args.artifact_root / "data" / str(profile["output_name"])
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path or train_data_path
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "SWE_REPO_ROOT": str(args.repo_root),
        "REPO_ROOT": str(args.repo_root),
        "SOURCE_VLLM_SITE": str(args.artifact_root / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312"),
        "SHARED_VLLM_SITE": str(args.artifact_root / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312"),
        "CONFIG_FILE": str(args.config),
        "ENV_FILE": str(args.env_file),
        "CHAT_TEMPLATE": str(args.chat_template),
        "RESOURCE_PROFILE_ENV": str(args.artifact_root / "reports/eagle3_resource_profile.env"),
        "ROLLOUT_LOG_DIR": str(rollout_log_dir),
        "OUTPUT_CONVERSATIONS": str(output_conversations),
        "DRY_RUN": "false",
        "REQUIRE_SOURCE_BUILD_PASS": "true",
        "START_WATCHER": "true",
        "MAX_NUM_STEPS": "1",
        "WANDB_NAME": str(profile["wandb_name"]),
        "EXP_SUFFIX_OVERRIDE": str(profile["wandb_name"]),
        "CHECKPOINT_SUBDIR": str(profile["wandb_name"]),
        "VLLM_ENFORCE_EAGER": "True",
        "VLLM_COMPILATION_LEVEL": "0",
        "VLLM_USE_INDUCTOR": "False",
        "SWEGYM_EXAMPLE_DATA": str(train_data_path),
        "TRAIN_DATA_PATH": str(train_data_path),
        "VAL_DATA_PATH": str(val_data_path),
        "ROLLOUT_REPORT_PREFIX_TAG": f"vllm0102src_megatroncompat_resourcefix_{profile['id'].removesuffix('_smoke')}",
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
        "CONTAINER": str(args.container),
        **profile["env"],
    }
    return env


def check_markers(profile: dict[str, Any], output: str, container: Path) -> list[dict[str, Any]]:
    env = profile["env"]
    nodes = env["NUM_NODES"]
    gen_nodes = env["NUM_GEN_NODES"]
    markers = [
        f"Nodes: {nodes}, GPUs/node: 4",
        "Ray worker CPUs: 64",
        f"Container: {container}",
        "Parallelism: TP=4, EP=16, CP=1, PP=4, vLLM_TP=8",
        "PP stages: first=23, last=23",
        f"cluster.num_nodes={nodes}",
        "cluster.gpus_per_node=4",
        "policy.megatron_cfg.expert_model_parallel_size=16",
        "policy.megatron_cfg.pipeline_model_parallel_size=4",
        "policy.megatron_cfg.num_layers_in_first_pipeline_stage=23",
        "policy.megatron_cfg.num_layers_in_last_pipeline_stage=23",
        f"policy.generation.colocated.resources.num_nodes={gen_nodes}",
        "policy.generation.colocated.resources.gpus_per_node=4",
        "--gres=gpu:4",
        "--mem=0",
    ]
    if profile["id"] == "compact_16n4g_smoke":
        markers.extend(
            [
                "grpo.num_prompts_per_step=4",
                "grpo.num_generations_per_prompt=2",
                "policy.train_global_batch_size=8",
            ]
        )
    return [
        {
            "marker": marker,
            "status": "pass" if marker in output else "fail",
        }
        for marker in markers
    ]


def validate_profile(args: argparse.Namespace, profile: dict[str, Any]) -> dict[str, Any]:
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path or train_data_path
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "REPO_ROOT": str(args.repo_root),
        "CONFIG_FILE": str(args.config),
        "ENV_FILE": str(args.env_file),
        "CHAT_TEMPLATE": str(args.chat_template),
        "DRY_RUN": "true",
        "SHARED_VLLM_SITE": str(args.artifact_root / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312"),
        "MAX_NUM_STEPS": "1",
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
        "CONTAINER": str(args.container),
        "VLLM_ENFORCE_EAGER": "True",
        "VLLM_COMPILATION_LEVEL": "0",
        "VLLM_USE_INDUCTOR": "False",
        "TRAIN_DATA_PATH": str(train_data_path),
        "VAL_DATA_PATH": str(val_data_path),
        "WANDB_API_KEY": "redacted",
        "HUGGINGFACE_TOKEN": "redacted",
        "GITHUB_TOKEN": "redacted",
        "GITLAB_TOKEN": "redacted",
        "HF_HOME": os.environ.get("HF_HOME", "/tmp/hf_home"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", "/tmp/hf_datasets"),
        **profile["env"],
    }
    dry_run = run(["bash", "run_grpo_qwen3_235b_swe.sh"], cwd=ROOT, env=env)
    markers = check_markers(profile, str(dry_run["output"]), args.container)
    topology = topology_checks(profile)

    test_cmd = [
        "sbatch",
        "--test-only",
        f"--nodes={profile['nodes']}",
        f"--account={args.sbatch_account}",
        f"--job-name=q235b-{profile['id']}",
        f"--partition={args.sbatch_partition}",
        "--time=4:0:0",
        f"--gres=gpu:{profile['gpus_per_node']}",
        "--exclusive",
        "--mem=0",
        "--dependency=singleton",
        "ray.sub",
    ]
    slurm_test = run(test_cmd, cwd=args.repo_root)

    marker_fail = [item for item in markers if item["status"] != "pass"]
    topology_fail = [item for item in topology if item["status"] != "pass"]
    status = (
        "pass"
        if dry_run["returncode"] == 0
        and slurm_test["returncode"] == 0
        and not marker_fail
        and not topology_fail
        else "fail"
    )
    container_status = "pass" if args.container.exists() and args.container.is_file() else "fail"
    if container_status != "pass":
        status = "fail"
    return {
        "id": profile["id"],
        "detail": profile["detail"],
        "experimental": bool(profile.get("experimental")),
        "status": status,
        "env": profile["env"],
        "submit_env": submit_env(args, profile),
        "submit_command": shell_join(
            submit_env(args, profile),
            ["bash", "experiments/eagle3_qwen3_235b/submit_source_vllm_rollout_smoke.sh"],
        ),
        "container": str(args.container),
        "container_status": container_status,
        "train_data_path": str(train_data_path),
        "val_data_path": str(val_data_path),
        "dry_run_returncode": dry_run["returncode"],
        "slurm_test_returncode": slurm_test["returncode"],
        "markers": markers,
        "topology_checks": topology,
        "dry_run_output_tail": dry_run["output_tail"],
        "slurm_test_output": slurm_test["output"],
    }


def profile_topology_status(profile: dict[str, Any]) -> str:
    checks = profile.get("topology_checks") or []
    if not checks:
        return "unknown"
    if all(isinstance(item, dict) and item.get("status") == "pass" for item in checks):
        return "pass"
    return "fail"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Rollout Capture Resource Profile Preflight",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Includes experimental profiles: **{str(payload.get('include_experimental', False)).lower()}**",
        "",
        "| profile | experimental | status | topology | dry-run | slurm test-only | detail |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for profile in payload["profiles"]:
        lines.append(
            f"| `{profile['id']}` | `{str(profile.get('experimental', False)).lower()}` | `{profile['status']}` | "
            f"`{profile_topology_status(profile)}` | {profile['dry_run_returncode']} | "
            f"{profile['slurm_test_returncode']} | {profile['detail']} |"
        )
    lines += ["", "## Topology Checks", ""]
    for profile in payload["profiles"]:
        lines += [f"### {profile['id']}", ""]
        for check in profile.get("topology_checks") or []:
            lines.append(f"- `{check.get('name')}`: `{check.get('status')}` - {check.get('detail')}")
        lines.append("")
    lines += ["", "## Submit Commands", ""]
    for profile in payload["profiles"]:
        lines += [f"### {profile['id']}", "", "```bash", str(profile["submit_command"]), "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    profiles = [validate_profile(args, profile) for profile in profile_defs(args.include_experimental)]
    passing_profiles = [profile for profile in profiles if profile["status"] == "pass"]
    if len(passing_profiles) == len(profiles):
        overall = "pass"
    elif passing_profiles:
        overall = "warn"
    else:
        overall = "fail"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "artifact_root": str(args.artifact_root),
        "repo_root": str(args.repo_root),
        "include_experimental": bool(args.include_experimental),
        "profiles": profiles,
    }
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0 if overall in {"pass", "warn"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
