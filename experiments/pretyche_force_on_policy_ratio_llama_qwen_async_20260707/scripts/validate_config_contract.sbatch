#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=36x2-a01r
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --time=00:20:00
#SBATCH --comment=metrics
#SBATCH --job-name=coreai_dlalgo_llm-nemorl.force-ratio-async-retry-config
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/pretyche_force_on_policy_ratio_async_retry_20260707/slurm/config-smoke-%j.out

set -euo pipefail

BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna
REPO=$BASE/nemo-rl-main-pr3030-q235-20260701
EXP_ROOT=$BASE/pretyche_force_on_policy_ratio_async_retry_20260707
CONTAINER=$BASE/containers/nemo_rl_nightly_20260630_0215.sqsh
EXPECTED_REPO_SHA=d4cfecf90db41cdf142629963b54b67ab479ab02
EXPECTED_CONTAINER_SHA=bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_REPO_SHA"
test "$(git -C "$REPO" rev-parse '@{u}')" = "$EXPECTED_REPO_SHA"
test -z "$(git -C "$REPO" status --porcelain --ignore-submodules=untracked)"
! git -C "$REPO" submodule status --recursive | grep -Eq '^[-+U]'
test -s "$CONTAINER"
test "$(sha256sum "$CONTAINER" | awk '{print $1}')" = "$EXPECTED_CONTAINER_SHA"
mkdir -p "$EXP_ROOT/slurm" "$BASE/uv_cache"

srun \
    --ntasks=1 \
    --container-image="$CONTAINER" \
    --container-mounts=/lustre:/lustre,/project:/project \
    --container-workdir="$REPO" \
    bash -lc "
        set -euo pipefail
        cleanup_unit_results() {
            rm -rf \
              '$REPO/tests/unit/unit_results.json' \
              '$REPO/tests/unit/unit_results'
        }
        trap cleanup_unit_results EXIT
        test \"\$(git rev-parse HEAD)\" = '$EXPECTED_REPO_SHA'
        NRL_FORCE_REBUILD_VENVS=false \
        UV_CACHE_DIR='$BASE/uv_cache' \
        uv run --locked pytest \
          tests/unit/algorithms/test_grpo.py::test_grpo_train_skips_prev_logprobs_when_force_on_policy_ratio \
          tests/unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_force_on_policy_ratio \
          -q
        NRL_FORCE_REBUILD_VENVS=false \
        UV_CACHE_DIR='$BASE/uv_cache' \
        uv run --locked '$EXP_ROOT/scripts/validate_config_contract.py'
    "
