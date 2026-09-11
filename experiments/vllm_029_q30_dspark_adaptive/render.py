#!/usr/bin/env python3
"""Render independent OCI-HSG SLURM jobs for the matched benchmark matrix."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from .contract import Arm, ExperimentContract, build_arms


def render_baseline_smoke_sbatch(
    contract: ExperimentContract,
    *,
    source_root: str,
    source_commit: str,
    container_image: str,
    result_dir: str,
    arm: Arm | None = None,
) -> str:
    """Render a one-worker canary with production worker settings."""
    if len(source_commit) != 40:
        raise ValueError("source_commit must be a full 40-character Git SHA")
    if arm is None:
        arm = next(item for item in build_arms(contract) if item.key == "baseline")
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=q30v029.{arm.key}-smoke
#SBATCH --account={contract.account}
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output={result_dir}/slurm-%j.out

set -euo pipefail

readonly SOURCE_ROOT={source_root}
readonly EXPECTED_SOURCE_COMMIT={source_commit}
readonly CONTAINER_IMAGE={container_image}
readonly RESULT_DIR={result_dir}
readonly TARGET_SOURCE={contract.target_path}
readonly DRAFTER_SOURCE={contract.drafter_path}
readonly NODE_LOCAL_ROOT=/raid/scratch/${{USER}}/q30-vllm029-smoke-${{SLURM_JOB_ID}}
readonly NODE_TARGET=${{NODE_LOCAL_ROOT}}/target
readonly NODE_DRAFTER=${{NODE_LOCAL_ROOT}}/dspark
readonly NODE_PROMPTS=${{NODE_LOCAL_ROOT}}/math500-prompts.jsonl

[[ "$(git -C "${{SOURCE_ROOT}}" rev-parse HEAD)" == "${{EXPECTED_SOURCE_COMMIT}}" ]] || {{
  echo "source commit mismatch" >&2
  exit 2
}}
[[ -f "${{CONTAINER_IMAGE}}" ]] || {{ echo "missing container: ${{CONTAINER_IMAGE}}" >&2; exit 2; }}
mkdir -p "${{RESULT_DIR}}" "${{NODE_TARGET}}" "${{NODE_DRAFTER}}"
trap 'rm -rf "${{NODE_LOCAL_ROOT}}"' EXIT
cp -aL "${{TARGET_SOURCE}}/." "${{NODE_TARGET}}/"
cp -aL "${{DRAFTER_SOURCE}}/." "${{NODE_DRAFTER}}/"
cp "${{SOURCE_ROOT}}/experiments/dynamic_sd_sync_rollout/data/math500_prompts.jsonl" "${{NODE_PROMPTS}}"
cd "${{SOURCE_ROOT}}"
python3 -m experiments.vllm_029_q30_dspark_adaptive.prepare_dspark_overlay "${{NODE_DRAFTER}}"

readonly CONTAINER_MOUNTS=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch
export SOURCE_ROOT RESULT_DIR NODE_TARGET NODE_DRAFTER NODE_PROMPTS
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpu-bind=none \
  --container-image="${{CONTAINER_IMAGE}}" \
  --container-mounts="${{CONTAINER_MOUNTS}}" \
  bash -lc '
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_HOME=/usr/local/cuda-13.0
    export CUDA_PATH="${{CUDA_HOME}}"
    export PATH="${{CUDA_HOME}}/bin:${{PATH}}"
    test -x "${{CUDA_HOME}}/bin/ptxas"
    cd "${{SOURCE_ROOT}}"
    python3 -m experiments.vllm_029_q30_dspark_adaptive.runtime \
      --arm {arm.key} \
      --worker-index 0 \
      --target-path "${{NODE_TARGET}}" \
      --drafter-path "${{NODE_DRAFTER}}" \
      --prompt-jsonl "${{NODE_PROMPTS}}" \
      --output "${{RESULT_DIR}}/worker-00.json"
  '
trap - EXIT
rm -rf "${{NODE_LOCAL_ROOT}}"
'''


def render_arm_sbatch(
    contract: ExperimentContract,
    arm: Arm,
    *,
    source_root: str,
    source_commit: str,
    container_image: str,
    result_dir: str,
) -> str:
    if len(source_commit) != 40:
        raise ValueError("source_commit must be a full 40-character Git SHA")
    version_probe_code = (
        "import torch,vllm; "
        "assert torch.cuda.is_available(); "
        f"assert vllm.__version__ == '{contract.vllm_version}'; "
        "print(torch.cuda.get_device_name(0), vllm.__version__)"
    )
    version_probe = shlex.quote(f"python3 -c {shlex.quote(version_probe_code)}")
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=q30v029.{arm.key}
#SBATCH --account={contract.account}
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --segment=4
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output={result_dir}/slurm-%j.out

set -euo pipefail

readonly SOURCE_ROOT={source_root}
readonly EXPECTED_SOURCE_COMMIT={source_commit}
readonly CONTAINER_IMAGE={container_image}
readonly RESULT_DIR={result_dir}
readonly TARGET_SOURCE={contract.target_path}
readonly DRAFTER_SOURCE={contract.drafter_path}
readonly NODE_LOCAL_ROOT=/raid/scratch/${{USER}}/q30-vllm029-${{SLURM_JOB_ID}}
readonly NODE_TARGET=${{NODE_LOCAL_ROOT}}/target
readonly NODE_DRAFTER=${{NODE_LOCAL_ROOT}}/dspark
readonly NODE_PROMPTS=${{NODE_LOCAL_ROOT}}/math500-prompts.jsonl

[[ "$(git -C "${{SOURCE_ROOT}}" rev-parse HEAD)" == "${{EXPECTED_SOURCE_COMMIT}}" ]] || {{
  echo "source commit mismatch" >&2
  exit 2
}}
[[ -f "${{CONTAINER_IMAGE}}" ]] || {{ echo "missing container: ${{CONTAINER_IMAGE}}" >&2; exit 2; }}
mkdir -p "${{RESULT_DIR}}/workers"

cleanup() {{
  srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    bash -lc 'rm -rf "${{NODE_LOCAL_ROOT}}"' || true
}}
trap cleanup EXIT

# Stage each shared input once per node. Four local engines then reuse it.
export NODE_LOCAL_ROOT NODE_TARGET NODE_DRAFTER NODE_PROMPTS SOURCE_ROOT TARGET_SOURCE DRAFTER_SOURCE
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  mkdir -p "${{NODE_TARGET}}" "${{NODE_DRAFTER}}"
  cp -aL "${{TARGET_SOURCE}}/." "${{NODE_TARGET}}/"
  cp -aL "${{DRAFTER_SOURCE}}/." "${{NODE_DRAFTER}}/"
  cp "${{SOURCE_ROOT}}/experiments/dynamic_sd_sync_rollout/data/math500_prompts.jsonl" "${{NODE_PROMPTS}}"
  cd "${{SOURCE_ROOT}}"
  python3 -m experiments.vllm_029_q30_dspark_adaptive.prepare_dspark_overlay "${{NODE_DRAFTER}}"
'

readonly CONTAINER_MOUNTS=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --container-image="${{CONTAINER_IMAGE}}" \
  --container-mounts="${{CONTAINER_MOUNTS}}" \
  bash -lc {version_probe}

export RESULT_DIR
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpu-bind=none \
  --container-image="${{CONTAINER_IMAGE}}" \
  --container-mounts="${{CONTAINER_MOUNTS}}" \
  bash -lc '
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${{SLURM_LOCALID}}"
    export CUDA_HOME=/usr/local/cuda-13.0
    export CUDA_PATH="${{CUDA_HOME}}"
    export PATH="${{CUDA_HOME}}/bin:${{PATH}}"
    test -x "${{CUDA_HOME}}/bin/ptxas"
    worker=$(printf "%02d" "${{SLURM_PROCID}}")
    runtime_root="${{NODE_LOCAL_ROOT}}/runtime-${{worker}}"
    mkdir -p "${{runtime_root}}/tmp" "${{runtime_root}}/xdg" \
      "${{runtime_root}}/vllm" "${{runtime_root}}/torchinductor" \
      "${{runtime_root}}/triton" "${{runtime_root}}/cuda"
    export TMPDIR="${{runtime_root}}/tmp"
    export XDG_CACHE_HOME="${{runtime_root}}/xdg"
    export VLLM_CACHE_ROOT="${{runtime_root}}/vllm"
    export TORCHINDUCTOR_CACHE_DIR="${{runtime_root}}/torchinductor"
    export TRITON_CACHE_DIR="${{runtime_root}}/triton"
    export CUDA_CACHE_PATH="${{runtime_root}}/cuda"
    cd "${{SOURCE_ROOT}}"
    python3 -m experiments.vllm_029_q30_dspark_adaptive.runtime \
      --arm {arm.key} \
      --worker-index "${{SLURM_PROCID}}" \
      --target-path "${{NODE_TARGET}}" \
      --drafter-path "${{NODE_DRAFTER}}" \
      --prompt-jsonl "${{NODE_PROMPTS}}" \
      --output "${{RESULT_DIR}}/workers/worker-${{worker}}.json"
  '

cd "${{SOURCE_ROOT}}"
python3 -m experiments.vllm_029_q30_dspark_adaptive.aggregate \
  --workers-dir "${{RESULT_DIR}}/workers" \
  --arm {arm.key} \
  --output "${{RESULT_DIR}}/summary.json"
trap - EXIT
cleanup
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parsed = parser.parse_args()
    if parsed.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {parsed.output_dir}")
    parsed.output_dir.mkdir(parents=True)
    contract = ExperimentContract()
    smoke_path = parsed.output_dir / "baseline_smoke.sbatch"
    smoke_path.write_text(
        render_baseline_smoke_sbatch(
            contract,
            source_root=parsed.source_root,
            source_commit=parsed.source_commit,
            container_image=parsed.container_image,
            result_dir=f"{parsed.result_root}/baseline_smoke",
        ),
        encoding="utf-8",
    )
    print(smoke_path)
    adaptive_arm = next(
        arm for arm in build_arms(contract) if arm.key == "dspark_adaptive_k7"
    )
    adaptive_smoke_path = parsed.output_dir / "adaptive_smoke.sbatch"
    adaptive_smoke_path.write_text(
        render_baseline_smoke_sbatch(
            contract,
            source_root=parsed.source_root,
            source_commit=parsed.source_commit,
            container_image=parsed.container_image,
            result_dir=f"{parsed.result_root}/adaptive_smoke",
            arm=adaptive_arm,
        ),
        encoding="utf-8",
    )
    print(adaptive_smoke_path)
    for arm in build_arms(contract):
        text = render_arm_sbatch(
            contract,
            arm,
            source_root=parsed.source_root,
            source_commit=parsed.source_commit,
            container_image=parsed.container_image,
            result_dir=f"{parsed.result_root}/{arm.key}",
        )
        path = parsed.output_dir / f"{arm.key}.sbatch"
        path.write_text(text, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
