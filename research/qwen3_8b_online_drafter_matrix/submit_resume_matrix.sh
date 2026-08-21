#!/bin/bash

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed resume harness SHA}"
: "${RESULT_ROOT:?Set existing matched-matrix /lustre result root}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SBATCH_ACCOUNT:?Set best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly experiment="${REMOTE_REPO}/research/qwen3_8b_online_drafter_matrix"
readonly runner="${experiment}/run_oci_hsg.sbatch"
readonly contract="${experiment}/runtime_contract.py"
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44

# arm source-directory checkpoint-source-SHA existing-W&B-ID
readonly arms=$(cat <<'EOF'
baseline 6b041659 6b041659ff58fe2f00793077bcd99c56b41d68db 92cc3c9e
dflash-k5 6b041659 6b041659ff58fe2f00793077bcd99c56b41d68db 410d15fd
dflash-k7 6b041659 6b041659ff58fe2f00793077bcd99c56b41d68db 96f3d9cf
dflash-fixed-k5 242ead65 242ead657783fc6df0676f76ae8ad3d625c55d0c 6acd29cc
dflash-fixed-k7 242ead65 242ead657783fc6df0676f76ae8ad3d625c55d0c fcc39683
EOF
)
readonly milestones="400 700 1000"

test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"

adopt_all() {
  while read -r arm source_dir source_sha wandb_id; do
    local final_dir="${RESULT_ROOT}/${source_dir}/${arm}"
    local checkpoint_dir="${final_dir}/checkpoints"
    local manifest="${final_dir}/resume-manifest.json"
    if [[ -e "${manifest}" ]]; then
      python3 "${contract}" --arm "${arm}" --checkpoint-dir "${checkpoint_dir}" \
        --manifest "${manifest}" --runtime-git-sha "${EXPECTED_HEAD}" \
        --checkpoint-source-sha "${source_sha}" --wandb-run-id "${wandb_id}" \
        --target-revision "${target_revision}" --drafter-revision "${drafter_revision}" \
        --container-sha256 "${container_sha}" --validate-resume-manifest
    else
      python3 "${contract}" --arm "${arm}" --checkpoint-dir "${checkpoint_dir}" \
        --manifest "${manifest}" --runtime-git-sha "${EXPECTED_HEAD}" \
        --checkpoint-source-sha "${source_sha}" --wandb-run-id "${wandb_id}" \
        --target-revision "${target_revision}" --drafter-revision "${drafter_revision}" \
        --container-sha256 "${container_sha}" --adopt-existing
    fi
  done <<<"${arms}"
}

options_for() {
  local arm=$1 source_dir=$2 source_sha=$3 wandb_id=$4 milestone=$5
  local final_dir="${RESULT_ROOT}/${source_dir}/${arm}"
  local exports="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${final_dir},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},MATRIX_ARM=${arm},WANDB_RUN_ID=${wandb_id},CHECKPOINT_SOURCE_SHA=${source_sha},STAGE_MIN_STEP=${milestone},STAGE_DEADLINE=00:03:30:00,IS_GATE=0,WANDB_RESUME=must"
  printf '%s\n' --account="${SBATCH_ACCOUNT}" --time=04:00:00 \
    --output="/raid/scratch/matrix-%j.out" \
    --job-name="q8-${arm}-to${milestone}" --export="${exports}"
}

preflight_all() {
  while read -r arm source_dir source_sha wandb_id; do
    for milestone in 400 700 1000; do
      mapfile -t options < <(options_for \
        "${arm}" "${source_dir}" "${source_sha}" "${wandb_id}" "${milestone}")
      sbatch --test-only "${options[@]}" "${runner}"
    done
  done <<<"${arms}"
}

submit_all() {
  while read -r arm source_dir source_sha wandb_id; do
    local previous=""
    for milestone in ${milestones}; do
      mapfile -t options < <(options_for \
        "${arm}" "${source_dir}" "${source_sha}" "${wandb_id}" "${milestone}")
      if [[ -n "${previous}" ]]; then
        options+=(--dependency="afterok:${previous}")
      fi
      previous="$(sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1)"
      echo "CHAIN ${arm} to${milestone}=${previous}"
    done
    echo "WANDB_URL=https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/${wandb_id}"
  done <<<"${arms}"
}

adopt_all
preflight_all
submit_all
