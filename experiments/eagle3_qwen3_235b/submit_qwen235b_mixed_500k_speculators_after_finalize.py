#!/usr/bin/env python3
"""Submit the Qwen3-235B 500K Speculators EAGLE3 train chain.

The final target-response corpus is produced by
``finalize_qwen235b_mixed_500k_corpus.sh``.  This submitter starts after that
finalizer succeeds: prepare Speculators Arrow data, dump verifier hidden states
with Ray-backed vLLM shards, then train the Eagle3 drafter once the hidden-state
cache is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]

REPO_ROOT = Path(
    os.environ.get(
        "REPO_ROOT",
        str(LOCAL_REPO_ROOT),
    )
)
ARTIFACT_ROOT = Path(
    os.environ.get(
        "ARTIFACT_ROOT",
        "/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3",
    )
)
PIPELINE_SBATCH = REPO_ROOT / "experiments/eagle3_qwen3_235b/slurm_speculators_offline_pipeline.sbatch"
RAY_SUB = Path(
    os.environ.get(
        "RAY_SUB",
        str(REPO_ROOT / "experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/ray.sub"),
    )
)
FALLBACK_RAY_SUB = REPO_ROOT / "scripts/share/ray.sub"
PIPELINE_SCRIPT = REPO_ROOT / "experiments/eagle3_qwen3_235b/speculators_qwen3_235b_offline_pipeline.sh"
RAY_SUB_NO_SINGLETON = REPO_ROOT / "logs/ray_no_singleton_for_speculators.sub"

ACCOUNT = os.environ.get("ACCOUNT", "coreai_dlalgo_nemorl")
PARTITION = os.environ.get("PARTITION", "batch")
CONTAINER = os.environ.get(
    "CONTAINER",
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh",
)
HF_HOME = os.environ.get(
    "HF_HOME",
    "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home",
)

MODEL = "Qwen/Qwen3-235B-A22B-Thinking-2507"
SEQ_LENGTH = "8192"
TARGET_LAYERS = "1 46 90"
LAYERS_TAG = "layers94_mlen8193"


def run(cmd: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> str:
    print("+", " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return "Submitted batch job DRYRUN"
    try:
        out = subprocess.check_output(
            cmd,
            cwd=REPO_ROOT,
            env={**os.environ, **(env or {})},
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        if exc.output:
            print(exc.output, flush=True)
        raise
    print(out.strip(), flush=True)
    return out


def parse_job_id(sbatch_output: str) -> str:
    if "DRYRUN" in sbatch_output:
        return "DRYRUN"
    for line in sbatch_output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[:3] == ["Submitted", "batch", "job"]:
            return parts[3]
    raise RuntimeError(f"could not parse sbatch job id from: {sbatch_output!r}")


def line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def paths() -> dict[str, Path]:
    root = ARTIFACT_ROOT / "speculators/eagle3_qwen3_235b_mixed_math_nonopenmath_500k_parallel"
    return {
        "final_conversations": ARTIFACT_ROOT
        / "data/mixed_math_nonopenmath_qwen3_235b_conversations_500k_unique.jsonl",
        "speculators_jsonl": ARTIFACT_ROOT
        / "data/mixed_math_nonopenmath_qwen3_235b_conversations_500k_unique_speculators.jsonl",
        "output_dir": root,
        "hidden_states_dir": root / f"hidden_states_{LAYERS_TAG}_500k_global",
        "hidden_state_manifest": root / f"hidden_states_{LAYERS_TAG}_500k_global/nrl_hidden_state_manifest.json",
        "checkpoint_dir": root / f"checkpoints_train_500k_{LAYERS_TAG}",
        "vllm_tmp_hidden_states": root / f"vllm_tmp_hidden_states_{LAYERS_TAG}_500k",
        "prepare_manifest": root / "nrl_prepare_manifest.json",
        "report": ARTIFACT_ROOT / "reports/qwen3_235b_500k_speculators_submit_summary.json",
        "denylist": ARTIFACT_ROOT / "data/openmath_reasoning_cot_conversations_50k.jsonl",
    }


def validate_inputs(p: dict[str, Path], *, allow_pending_finalizer: bool) -> None:
    if allow_pending_finalizer:
        return
    missing = [item for item in (p["final_conversations"], p["speculators_jsonl"]) if not item.exists()]
    if missing:
        raise FileNotFoundError("missing finalized corpus files:\n" + "\n".join(str(item) for item in missing))
    bad = [(item, line_count(item)) for item in (p["final_conversations"], p["speculators_jsonl"]) if line_count(item) != 500_000]
    if bad:
        raise RuntimeError("finalized corpus files must have exactly 500000 rows:\n" + "\n".join(f"{item}: {rows}" for item, rows in bad))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_prepare_manifest(p: dict[str, Path], *, include_sha256: bool) -> dict[str, object]:
    speculators_jsonl = p["speculators_jsonl"]
    source_conversations = p["final_conversations"]
    manifest: dict[str, object] = {
        "model": MODEL,
        "seq_length": int(SEQ_LENGTH),
        "target_layer_ids": [int(item) for item in TARGET_LAYERS.split()],
        "speculators_jsonl": str(speculators_jsonl),
        "speculators_jsonl_size": speculators_jsonl.stat().st_size,
        "speculators_jsonl_rows": line_count(speculators_jsonl),
        "source_conversations": str(source_conversations),
        "source_conversations_size": source_conversations.stat().st_size,
        "source_conversations_rows": line_count(source_conversations),
    }
    if include_sha256:
        manifest["speculators_jsonl_sha256"] = sha256_file(speculators_jsonl)
    return manifest


def validate_manifest_matches(actual: dict, expected: dict[str, object]) -> None:
    mismatches = []
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            mismatches.append(f"{key}: actual={actual.get(key)!r} expected={expected_value!r}")
    if mismatches:
        raise RuntimeError(
            "cannot reuse prepare job; prepare manifest does not match current inputs:\n"
            + "\n".join(mismatches)
        )


def validate_prepared_outputs(p: dict[str, Path]) -> None:
    required = [
        p["output_dir"] / "state.json",
        p["output_dir"] / "dataset_info.json",
        p["output_dir"] / "token_freq.pt",
    ]
    missing = [item for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError("cannot reuse prepare job; prepared outputs are missing:\n" + "\n".join(str(item) for item in missing))
    dataset_info = load_json(p["output_dir"] / "dataset_info.json")
    split = (dataset_info.get("splits") or {}).get("train") or {}
    if split.get("num_examples") != 500_000:
        raise RuntimeError(f"cannot reuse prepare job; dataset_info train.num_examples={split.get('num_examples')!r} != 500000")
    checksums = dataset_info.get("download_checksums") or {}
    spec_info = checksums.get(str(p["speculators_jsonl"]))
    if not isinstance(spec_info, dict):
        raise RuntimeError(
            "cannot reuse prepare job; dataset_info does not reference the current "
            f"Speculators JSONL: {p['speculators_jsonl']}"
        )
    actual_spec_size = p["speculators_jsonl"].stat().st_size
    if spec_info.get("num_bytes") != actual_spec_size:
        raise RuntimeError(
            "cannot reuse prepare job; dataset_info Speculators JSONL size mismatch: "
            f"actual={spec_info.get('num_bytes')!r} expected={actual_spec_size}"
        )
    state = load_json(p["output_dir"] / "state.json")
    data_files = [item.get("filename") for item in state.get("_data_files", []) if isinstance(item, dict)]
    if not data_files:
        raise RuntimeError("cannot reuse prepare job; state.json has no Arrow data files")
    missing_data = [p["output_dir"] / name for name in data_files if not (p["output_dir"] / name).exists()]
    if missing_data:
        raise FileNotFoundError("cannot reuse prepare job; Arrow shards from state.json are missing:\n" + "\n".join(str(item) for item in missing_data[:20]))
    strict_hash = os.environ.get("STRICT_PREPARE_REUSE_HASH", "true").lower() not in {"0", "false", "no"}
    expected = expected_prepare_manifest(p, include_sha256=strict_hash)
    manifest_path = p["prepare_manifest"]
    if manifest_path.exists():
        validate_manifest_matches(load_json(manifest_path), expected)
    else:
        if strict_hash:
            raise RuntimeError(
                "cannot reuse prepare job in strict mode because prepare manifest is missing: "
                f"{manifest_path}. Re-run prepare or set STRICT_PREPARE_REUSE_HASH=false "
                "only after manually verifying the prepared Arrow data."
            )
        bootstrap = {
            **expected,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "created_by": "submit_qwen235b_mixed_500k_speculators_after_finalize.py",
            "note": "bootstrap manifest for pre-existing prepared outputs after dataset_info/state validation",
        }
        manifest_path.write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"+ wrote bootstrap prepare manifest {manifest_path}", flush=True)


def expected_hidden_state_manifest(p: dict[str, Path]) -> dict[str, object]:
    prepare_manifest = load_json(p["prepare_manifest"])
    return {
        "model": MODEL,
        "seq_length": int(SEQ_LENGTH),
        "target_layer_ids": [int(item) for item in TARGET_LAYERS.split()],
        "expected_hidden_states": 500_000,
        "hidden_states_dir": str(p["hidden_states_dir"]),
        "prepare_manifest": str(p["prepare_manifest"]),
        "prepare_manifest_sha256": sha256_file(p["prepare_manifest"]),
        "speculators_jsonl": prepare_manifest.get("speculators_jsonl"),
        "speculators_jsonl_sha256": prepare_manifest.get("speculators_jsonl_sha256"),
    }


def validate_hidden_state_manifest_reuse_safety(p: dict[str, Path]) -> None:
    manifest_path = p["hidden_state_manifest"]
    expected = expected_hidden_state_manifest(p)
    hidden_files = list(p["hidden_states_dir"].glob("hs_*.safetensors"))
    if manifest_path.exists():
        if not hidden_files:
            print(
                "+ hidden-state manifest exists but no hidden-state files exist; "
                "datagen will write a fresh manifest",
                flush=True,
            )
            return
        actual = load_json(manifest_path)
        mismatches = [
            key
            for key, value in expected.items()
            if actual.get(key) != value
        ]
        if not mismatches:
            return
        if hidden_files:
            raise RuntimeError(
                "hidden-state manifest does not match current prepared data, "
                f"but {len(hidden_files)} hidden-state files already exist: {mismatches}"
            )
        print(
            "+ hidden-state manifest mismatch but no hidden-state files exist; "
            "datagen will write a fresh manifest",
            flush=True,
        )
        return
    if hidden_files:
        raise RuntimeError(
            "hidden-state files already exist but hidden-state manifest is missing; "
            f"refusing to bootstrap stale outputs in {p['hidden_states_dir']}"
        )
    print(
        "+ hidden-state manifest is absent and no hidden-state files exist; "
        "datagen will write it after successful generation",
        flush=True,
    )


def common_env(p: dict[str, Path]) -> dict[str, str]:
    return {
        "REPO_ROOT": str(REPO_ROOT),
        "ARTIFACT_ROOT": str(ARTIFACT_ROOT),
        "MODEL": MODEL,
        "SOURCE_CONVERSATIONS": str(p["final_conversations"]),
        "SPECULATORS_JSONL": str(p["speculators_jsonl"]),
        "OUTPUT_DIR": str(p["output_dir"]),
        "HIDDEN_STATES_DIR": str(p["hidden_states_dir"]),
        "HIDDEN_STATE_MANIFEST": str(p["hidden_state_manifest"]),
        "CHECKPOINT_DIR": str(p["checkpoint_dir"]),
        "VLLM_TMP_HIDDEN_STATES": str(p["vllm_tmp_hidden_states"]),
        "SEQ_LENGTH": SEQ_LENGTH,
        "VLLM_MAX_MODEL_LEN": SEQ_LENGTH,
        "MAX_SAMPLES": "0",
        "MIN_HIDDEN_STATES": "500000",
        "MINIMUM_VALID_TOKENS": "1",
        "PREPARE_MANIFEST": str(p["prepare_manifest"]),
        "TARGET_LAYER_IDS": TARGET_LAYERS,
        "DRAFT_VOCAB_SIZE": "32000",
        "SPECULATOR_TYPE": "eagle3",
        "EPOCHS": "1",
        "LR": "5e-5",
        "TTT_STEPS": "3",
        "TTT_STEP_LOSS_DECAY": "1.0",
        "BLOCK_SIZE": "8",
        "MAX_ANCHORS": "256",
        "NUM_LAYERS": "1",
        "DRAFT_ARCH": "llama",
        "NUM_TRAIN_GPUS": "4",
        "VLLM_TP": "8",
        "VLLM_DP": "1",
        "VLLM_GPU_UTIL": "0.82",
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
        "CONCURRENCY": "16",
        "REQUEST_TIMEOUT": "300",
        "MAX_RETRIES": "3",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "VLLM_EXTRA_ARGS": "--distributed-executor-backend ray --attention-backend TRITON_ATTN --max-num-seqs 1 --max-cudagraph-capture-size 1 --disable-custom-all-reduce",
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": "PYTHONPATH",
        "RUN_CLONE": "false",
        "VALIDATE_OUTPUTS": "true",
        "VALIDATE_SOURCE_CONVERSATIONS": "true",
        "FAIL_ON_DUPLICATE_PROMPTS": "true",
        "DENYLIST_PROMPTS_FROM": str(p["denylist"]),
        "INSTALL_SPECULATORS": "true",
        "APPLY_COMPAT_PATCHES": "false",
        "SPECULATORS_DISABLE_TORCH_COMPILE": "false",
        "SPECULATORS_FSDP_WRAP_LAYERS": "true",
    }


def submit_sbatch_pipeline(
    *,
    name: str,
    dependency: str,
    time_limit: str,
    env: dict[str, str],
    dry_run: bool,
) -> str:
    cmd = [
        "sbatch",
        "--nodes=1",
        f"--account={ACCOUNT}",
        f"--partition={PARTITION}",
        "--gres=gpu:4",
        f"--time={time_limit}",
        "--mem=0",
        f"--job-name={name}",
        "--output=logs/%x_%j.out",
        "--error=logs/%x_%j.err",
    ]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(PIPELINE_SBATCH))
    return parse_job_id(run(cmd, env=env, dry_run=dry_run))


def setup_command(vllm_site: Path) -> str:
    setup_pythonpath = (
        f"{shlex.quote(str(vllm_site))}:{shlex.quote(str(REPO_ROOT))}:${{PYTHONPATH:-}}"
    )
    return f"""PYTHONPATH={setup_pythonpath} python - <<'PY'
import sys
import vllm
print("setup vllm", vllm.__version__, vllm.__file__)
print("setup sys.path head", sys.path[:5])
PY"""


def submit_ray_datagen(
    *,
    shard: int,
    start: int,
    end: int,
    dependency: str,
    env: dict[str, str],
    datagen_time: str,
    num_nodes: int,
    vllm_site: Path,
    ray_sub: Path,
    dry_run: bool,
) -> str:
    shard_env = {
        **env,
        "RUN_CONVERT": "false",
        "RUN_PREPARE": "false",
        "RUN_DATAGEN": "true",
        "RUN_TRAIN": "false",
        "DATAGEN_START_INDEX": str(start),
        "DATAGEN_END_INDEX": str(end),
        "VLLM_SITE": str(vllm_site),
        "VLLM_TMP_HIDDEN_STATES": str(env_path(env["OUTPUT_DIR"]) / f"vllm_tmp_hidden_states_{LAYERS_TAG}_500k_shard{shard:03d}"),
    }
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(shard_env.items()))
    command = f"set -euo pipefail; cd {shlex.quote(str(REPO_ROOT))}; {assignments} bash {shlex.quote(str(PIPELINE_SCRIPT))}"
    mounts = f"/lustre:/lustre,{REPO_ROOT}:{REPO_ROOT},{ARTIFACT_ROOT}:{ARTIFACT_ROOT}"
    submit_env = {
        "COMMAND": command,
        "CONTAINER": CONTAINER,
        "MOUNTS": mounts,
        "GPUS_PER_NODE": "4",
        "SETUP_COMMAND": setup_command(vllm_site),
        "HF_HOME": HF_HOME,
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", f"{HF_HOME}/datasets"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", f"{HF_HOME}/hub"),
        "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", str(ARTIFACT_ROOT / "vllm_cache")),
        "VLLM_DISABLE_USAGE_STATS": "1",
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
        "VLLM_ALLREDUCE_USE_SYMM_MEM": os.environ.get("VLLM_ALLREDUCE_USE_SYMM_MEM", "0"),
        "VLLM_USE_RAY_COMPILED_DAG": os.environ.get("VLLM_USE_RAY_COMPILED_DAG", "0"),
        "VLLM_USE_RAY_SPMD_WORKER": os.environ.get("VLLM_USE_RAY_SPMD_WORKER", "0"),
        "VLLM_USE_RAY_WRAPPED_PP_COMM": os.environ.get("VLLM_USE_RAY_WRAPPED_PP_COMM", "0"),
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": "PYTHONPATH",
        "RAY_INCLUDE_DASHBOARD": "False",
        "RAY_USE_EXISTING_ENV": "true",
        "RAY_CLI": "ray",
        "RAY_DEDUP_LOGS": "0",
        "RAY_raylet_start_wait_time_s": "300",
    }
    cmd = [
        "sbatch",
        f"--nodes={num_nodes}",
        f"--account={ACCOUNT}",
        f"--partition={PARTITION}",
        "--gres=gpu:4",
        "--mem=0",
        f"--time={datagen_time}",
        f"--job-name=qwen3_235b-speculators-500k-hs-{shard:03d}",
        str(ray_sub),
    ]
    if dependency:
        cmd.insert(-1, f"--dependency={dependency}")
    return parse_job_id(run(cmd, env=submit_env, dry_run=dry_run))


def export_statements(env: dict[str, str]) -> str:
    return "; ".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items()))


def submit_ray_datagen_array(
    *,
    shards: int,
    dependency: str,
    env: dict[str, str],
    datagen_time: str,
    num_nodes: int,
    vllm_site: Path,
    ray_sub: Path,
    max_concurrent: int,
    dry_run: bool,
) -> str:
    shard_env = {
        **env,
        "RUN_CONVERT": "false",
        "RUN_PREPARE": "false",
        "RUN_DATAGEN": "true",
        "RUN_TRAIN": "false",
        "VLLM_SITE": str(vllm_site),
    }
    command = f"""set -euo pipefail
cd {shlex.quote(str(REPO_ROOT))}
{export_statements(shard_env)}
shard="${{SLURM_ARRAY_TASK_ID}}"
shard_size=$((500000 / {shards}))
start=$((shard * shard_size))
if [ "$shard" -eq $(({shards} - 1)) ]; then
  end=500000
else
  end=$(((shard + 1) * shard_size))
fi
printf -v shard_tag "%03d" "$shard"
export DATAGEN_START_INDEX="$start"
export DATAGEN_END_INDEX="$end"
export VLLM_TMP_HIDDEN_STATES="${{OUTPUT_DIR}}/vllm_tmp_hidden_states_{LAYERS_TAG}_500k_shard${{shard_tag}}"
bash {shlex.quote(str(PIPELINE_SCRIPT))}"""
    mounts = f"/lustre:/lustre,{REPO_ROOT}:{REPO_ROOT},{ARTIFACT_ROOT}:{ARTIFACT_ROOT}"
    submit_env = {
        "COMMAND": command,
        "CONTAINER": CONTAINER,
        "MOUNTS": mounts,
        "GPUS_PER_NODE": "4",
        "SETUP_COMMAND": setup_command(vllm_site),
        "HF_HOME": HF_HOME,
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", f"{HF_HOME}/datasets"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", f"{HF_HOME}/hub"),
        "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", str(ARTIFACT_ROOT / "vllm_cache")),
        "VLLM_DISABLE_USAGE_STATS": "1",
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
        "VLLM_ALLREDUCE_USE_SYMM_MEM": os.environ.get("VLLM_ALLREDUCE_USE_SYMM_MEM", "0"),
        "VLLM_USE_RAY_COMPILED_DAG": os.environ.get("VLLM_USE_RAY_COMPILED_DAG", "0"),
        "VLLM_USE_RAY_SPMD_WORKER": os.environ.get("VLLM_USE_RAY_SPMD_WORKER", "0"),
        "VLLM_USE_RAY_WRAPPED_PP_COMM": os.environ.get("VLLM_USE_RAY_WRAPPED_PP_COMM", "0"),
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": "PYTHONPATH",
        "RAY_INCLUDE_DASHBOARD": "False",
        "RAY_USE_EXISTING_ENV": "true",
        "RAY_CLI": "ray",
        "RAY_DEDUP_LOGS": "0",
        "RAY_raylet_start_wait_time_s": "300",
    }
    array_spec = f"0-{shards - 1}"
    if max_concurrent > 0:
        array_spec += f"%{max_concurrent}"
    cmd = [
        "sbatch",
        f"--nodes={num_nodes}",
        f"--account={ACCOUNT}",
        f"--partition={PARTITION}",
        "--gres=gpu:4",
        "--mem=0",
        f"--time={datagen_time}",
        "--job-name=qwen3_235b-speculators-500k-hs",
        f"--array={array_spec}",
        str(ray_sub),
    ]
    if dependency:
        cmd.insert(-1, f"--dependency={dependency}")
    return parse_job_id(run(cmd, env=submit_env, dry_run=dry_run))


def env_path(value: str) -> Path:
    return Path(value)


def materialize_ray_submit_script(*, dry_run: bool) -> Path:
    """Use ray.sub without singleton so shard jobs can depend only on prepare."""
    if dry_run:
        return RAY_SUB_NO_SINGLETON
    source = RAY_SUB if RAY_SUB.exists() else FALLBACK_RAY_SUB
    text = source.read_text(encoding="utf-8")
    filtered = "\n".join(
        line for line in text.splitlines() if line.strip() != "#SBATCH --dependency=singleton"
    )
    filtered = filtered.replace(
        "--include-dashboard=True",
        "--include-dashboard=False",
    )
    filtered = filtered.replace(
        "--include-dashboard=${RAY_INCLUDE_DASHBOARD}",
        "--include-dashboard=False",
    )
    filtered = filtered.replace(
        "--include-dashboard=${RAY_INCLUDE_DASHBOARD:-True}",
        "--include-dashboard=False",
    )
    filtered = filtered.replace(
        "--include-dashboard=${RAY_INCLUDE_DASHBOARD:-False}",
        "--include-dashboard=False",
    )
    RAY_SUB_NO_SINGLETON.write_text(filtered + "\n", encoding="utf-8")
    RAY_SUB_NO_SINGLETON.chmod(0o755)
    return RAY_SUB_NO_SINGLETON


def submit(args: argparse.Namespace) -> dict[str, object]:
    p = paths()
    validate_inputs(p, allow_pending_finalizer=args.allow_pending_finalizer)
    if args.existing_prepare_job_id and not args.dry_run:
        validate_prepared_outputs(p)
        validate_hidden_state_manifest_reuse_safety(p)
    if not args.dry_run:
        (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)
        p["report"].parent.mkdir(parents=True, exist_ok=True)
    ray_sub = materialize_ray_submit_script(dry_run=args.dry_run)

    base = common_env(p)
    suffix = args.job_suffix
    finalizer_dependency = f"afterok:{args.finalizer_job_id}" if args.finalizer_job_id else args.dependency

    prep_env = {
        **base,
        "RUN_CONVERT": "false",
        "RUN_PREPARE": "true",
        "RUN_DATAGEN": "false",
        "RUN_TRAIN": "false",
    }
    prepare_job_reused = bool(args.existing_prepare_job_id)
    if prepare_job_reused:
        prep_id = args.existing_prepare_job_id
        print(f"+ reusing existing prepare job {prep_id}; prepared outputs already validated", flush=True)
        hidden_state_prepare_dependency = ""
    else:
        prep_id = submit_sbatch_pipeline(
            name=f"qwen3_235b-speculators-mixed-500k-prep{suffix}",
            dependency=finalizer_dependency,
            time_limit=args.prep_time,
            env=prep_env,
            dry_run=args.dry_run,
        )
        hidden_state_prepare_dependency = f"afterok:{prep_id}"

    if args.individual_shard_jobs:
        shard_ids: list[str] = []
        shard_size = 500_000 // args.shards
        for shard in range(args.shards):
            start = shard * shard_size
            end = 500_000 if shard == args.shards - 1 else (shard + 1) * shard_size
            shard_ids.append(
                submit_ray_datagen(
                    shard=shard,
                    start=start,
                    end=end,
                    dependency=hidden_state_prepare_dependency,
                    env=base,
                    datagen_time=args.datagen_time,
                    num_nodes=args.datagen_nodes,
                    vllm_site=Path(args.vllm_site),
                    ray_sub=ray_sub,
                    dry_run=args.dry_run,
                )
            )
        hidden_state_dependency = "afterok:" + ":".join(shard_ids)
        hidden_state_jobs = shard_ids
        hidden_state_array_job = ""
    else:
        array_id = submit_ray_datagen_array(
            shards=args.shards,
            dependency=hidden_state_prepare_dependency,
            env=base,
            datagen_time=args.datagen_time,
            num_nodes=args.datagen_nodes,
            vllm_site=Path(args.vllm_site),
            ray_sub=ray_sub,
            max_concurrent=args.max_concurrent_shards,
            dry_run=args.dry_run,
        )
        hidden_state_dependency = f"afterok:{array_id}"
        hidden_state_jobs = [f"{array_id}_[0-{args.shards - 1}]"]
        hidden_state_array_job = array_id

    train_env = {
        **base,
        "RUN_CONVERT": "false",
        "RUN_PREPARE": "false",
        "RUN_DATAGEN": "false",
        "RUN_TRAIN": "true",
        "VALIDATE_SOURCE_CONVERSATIONS": "false",
    }
    if args.from_pretrained:
        train_env["FROM_PRETRAINED"] = args.from_pretrained
    train_id = submit_sbatch_pipeline(
        name=f"qwen3_235b-speculators-mixed-500k-train{suffix}",
        dependency=hidden_state_dependency,
        time_limit=args.train_time,
        env=train_env,
        dry_run=args.dry_run,
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "mode": "qwen3_235b_parallel_500k_after_finalizer",
        "model": MODEL,
        "finalizer_job_id": args.finalizer_job_id,
        "dependency": finalizer_dependency,
        "shards": args.shards,
        "datagen_nodes": args.datagen_nodes,
        "max_concurrent_shards": args.max_concurrent_shards,
        "individual_shard_jobs": args.individual_shard_jobs,
        "final_conversations": str(p["final_conversations"]),
        "speculators_jsonl": str(p["speculators_jsonl"]),
        "output_dir": str(p["output_dir"]),
        "hidden_states_dir": str(p["hidden_states_dir"]),
        "checkpoint_dir": str(p["checkpoint_dir"]),
        "prepare_job": prep_id,
        "prepare_job_reused": prepare_job_reused,
        "hidden_state_prepare_dependency": hidden_state_prepare_dependency,
        "hidden_state_jobs": hidden_state_jobs,
        "hidden_state_array_job": hidden_state_array_job,
        "train_job": train_id,
        "ray_sub": str(ray_sub),
        "ray_sub_source": str(RAY_SUB if RAY_SUB.exists() else FALLBACK_RAY_SUB),
        "ray_sub_singleton_removed": True,
        "ray_include_dashboard": "False",
        "ray_use_existing_env": "true",
        "from_pretrained": args.from_pretrained or "",
        "allow_pending_finalizer": args.allow_pending_finalizer,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        p["report"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalizer-job-id", default="")
    parser.add_argument("--dependency", default="")
    parser.add_argument("--existing-prepare-job-id", default="")
    parser.add_argument("--allow-pending-finalizer", action="store_true")
    parser.add_argument("--shards", type=int, default=100)
    parser.add_argument("--datagen-nodes", type=int, default=2)
    parser.add_argument("--max-concurrent-shards", type=int, default=0)
    parser.add_argument("--individual-shard-jobs", action="store_true")
    parser.add_argument("--prep-time", default="02:00:00")
    parser.add_argument("--datagen-time", default="04:00:00")
    parser.add_argument("--train-time", default="04:00:00")
    parser.add_argument("--from-pretrained", default="")
    parser.add_argument("--job-suffix", default="")
    parser.add_argument("--vllm-site", default=str(ARTIFACT_ROOT / "python_site/vllm_0_17_0_extract_py312"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shards <= 0 or 500_000 % args.shards != 0:
        raise ValueError("--shards must be a positive divisor of 500000")
    if not args.finalizer_job_id and not args.dependency and args.allow_pending_finalizer:
        raise ValueError("--allow-pending-finalizer requires --finalizer-job-id or --dependency")
    submit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
