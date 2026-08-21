#!/bin/bash

set -euo pipefail

: "${HARNESS_REPO:?Set the clean pushed /home harness checkout}"
: "${EXPECTED_HARNESS_HEAD:?Set the pushed harness commit SHA}"
: "${SBATCH_ACCOUNT:?Set the best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly online_repo="${ONLINE_REPO:-/home/sna/RL-online-dspark-matrix-21fed912}"
readonly fixed_repo="${FIXED_REPO:-/home/sna/RL-fixed-dspark-matrix-6026ee11}"
readonly online_sha=21fed91219cad821f1e7cdbaf3fa2edc9f188939
readonly fixed_sha=2c3e0e064f98bf9a1eb1fac16ed6764ec4d8927b
readonly user_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna
readonly result_root="${user_root}/experiments/online-drafter"
readonly container="${user_root}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh"
readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=03326e5043815da1f81b109078b2889737c26017
readonly target_snapshot="${user_root}/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/${target_revision}"
readonly drafter_snapshot="${user_root}/hf_home/hub/models--deepseek-ai--dspark_qwen3_8b_block7/snapshots/${drafter_revision}"
readonly resume_step=400
readonly final_step=1000
readonly walltime=07:00:00
readonly checkpoint_deadline=00:06:30:00
readonly gpus_per_node=4

readonly -a arms=(
  "online|dspark-k5|5|${online_repo}|${online_sha}|${result_root}/qwen3-8b-dspark-online-matrix-21fed912/dspark-k5|r4a81508a"
  "online|dspark-k7|7|${online_repo}|${online_sha}|${result_root}/qwen3-8b-dspark-online-matrix-21fed912/dspark-k7|rcf9fa648"
  "fixed|dspark-k5|5|${fixed_repo}|${fixed_sha}|${result_root}/qwen3-8b-dspark-fixed-control-matrix-2c3e0e06/dspark-k5|rb53842d0"
  "fixed|dspark-k7|7|${fixed_repo}|${fixed_sha}|${result_root}/qwen3-8b-dspark-fixed-control-matrix-2c3e0e06/dspark-k7|r765298fc"
)

test "$(git -C "${HARNESS_REPO}" rev-parse HEAD)" = "${EXPECTED_HARNESS_HEAD}"
git -C "${HARNESS_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${HARNESS_REPO}" ls-files --others --exclude-standard)"
grep -Fqx "sha256=${container_sha}" "${container}.metadata.txt"
test "$(basename "${target_snapshot}")" = "${target_revision}"
test "$(basename "${drafter_snapshot}")" = "${drafter_revision}"

parse_arm() {
  local record=$1
  IFS='|' read -r family arm num_speculative_tokens repo source_sha final_dir wandb_id <<<"${record}"
  [[ "${family}" == online || "${family}" == fixed ]]
  [[ "${arm}" == "dspark-k${num_speculative_tokens}" ]]
  [[ "${num_speculative_tokens}" == 5 || "${num_speculative_tokens}" == 7 ]]
  [[ "${repo}" == /home/* ]]
  [[ "${final_dir}" == /lustre/* ]]
}

preflight_arm() {
  local record=$1
  local family arm num_speculative_tokens repo source_sha final_dir wandb_id
  local experiment checkpoint_dir manifest contract manifest_wandb_id
  parse_arm "${record}"
  experiment="${repo}/research/qwen3_8b_dspark_${family}_cp1"
  checkpoint_dir="${final_dir}/checkpoints"
  manifest="${final_dir}/gate-manifest.json"
  contract="${experiment}/resume_contract.py"

  test "$(git -C "${repo}" rev-parse HEAD)" = "${source_sha}"
  git -C "${repo}" diff-index --quiet --ignore-submodules=all HEAD --
  test -z "$(git -C "${repo}" ls-files --others --exclude-standard)"
  python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" \
    --expected-step "${resume_step}"

  local -a identity=(
    --checkpoint-dir "${checkpoint_dir}" --manifest "${manifest}"
    --git-sha "${source_sha}" --target-revision "${target_revision}"
    --drafter-revision "${drafter_revision}" --container-sha256 "${container_sha}"
    --num-speculative-tokens "${num_speculative_tokens}"
  )
  if [[ -e "${manifest}" ]]; then
    python3 "${contract}" "${identity[@]}" --validate-manifest
  else
    python3 "${contract}" "${identity[@]}" \
      --wandb-run-id "${wandb_id}" --write-manifest
  fi
  manifest_wandb_id="$(python3 "${contract}" "${identity[@]}" \
    --print-manifest-wandb-id)"
  test "${manifest_wandb_id}" = "${wandb_id}"
  echo "PREFLIGHT family=${family} arm=${arm} checkpoint=step_${resume_step} wandb=${wandb_id}"
}

run_arm() {
  local action=$1 record=$2
  local family arm num_speculative_tokens repo source_sha final_dir wandb_id
  local runner output_prefix job_id
  parse_arm "${record}"
  runner="${HARNESS_REPO}/research/qwen3_8b_dspark_${family}_cp1/run_segment_oci_hsg.sbatch"
  output_prefix="nrl-dspark-${family}"
  local common="ALL,REMOTE_REPO=${repo},EXPECTED_HEAD=${source_sha},FINAL_DIR=${final_dir},CONTAINER=${container},TARGET_SNAPSHOT=${target_snapshot},DRAFTER_SNAPSHOT=${drafter_snapshot},ARM_NAME=${arm},NUM_SPECULATIVE_TOKENS=${num_speculative_tokens},WANDB_RUN_ID=${wandb_id},WANDB_RESUME=must,STAGE_MODE=resume,STAGE_MIN_STEP=${final_step},STAGE_DEADLINE=${checkpoint_deadline}"
  local -a options=(
    --account="${SBATCH_ACCOUNT}"
    --partition=batch_long
    --nodes=1
    --gres="gpu:${gpus_per_node}"
    --time="${walltime}"
    --output="/raid/scratch/${output_prefix}-%j.out"
    --job-name="q8-${family}-${arm}-r1000"
    --export="${common}"
  )

  if [[ "${action}" == test-only ]]; then
    sbatch --test-only "${options[@]}" "${runner}"
    echo "TEST_ONLY family=${family} arm=${arm} topology=TP2/DP2/CP1,packing=false,SP=false"
  elif [[ "${action}" == submit ]]; then
    job_id="$(sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1)"
    test -n "${job_id}"
    echo "SUBMITTED family=${family} arm=${arm} job=${job_id} wandb=${wandb_id}"
  else
    echo "unknown action: ${action}" >&2
    return 2
  fi
}

run_all() {
  local action=$1 record
  for record in "${arms[@]}"; do
    run_arm "${action}" "${record}"
  done
}

echo "MATRIX common science contract=seed42,GBS32,PPS8,GPS4"
for record in "${arms[@]}"; do
  preflight_arm "${record}"
done
run_all test-only
run_all submit
