#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly LAUNCHER="${SCRIPT_DIR}/submit_math_gate.sh"
readonly MAX_STEPS="${Q30_K_SWEEP_MAX_STEPS:-5}"

arms=(baseline)
for cohort in ptv2 legacy; do
  for method in dflash dspark; do
    for k in 1 2 3 5 7; do
      arms+=("${cohort}_${method}_k${k}")
    done
  done
done

usage() {
  echo "usage: $0 --list|--test-only|--submit" >&2
  exit 2
}

mode="${1:-}"
case "${mode}" in
  --list)
    printf '%s\n' "${arms[@]}"
    exit 0
    ;;
  --test-only)
    for arm in "${arms[@]}"; do
      Q30_PTV2_MAX_STEPS="${MAX_STEPS}" bash "${LAUNCHER}" --test-only "${arm}"
    done
    exit 0
    ;;
  --submit) ;;
  *) usage ;;
esac

baseline_output="$(
  Q30_PTV2_MAX_STEPS="${MAX_STEPS}" bash "${LAUNCHER}" --submit baseline
)"
printf '%s\n' "${baseline_output}"
baseline_job="$(
  printf '%s\n' "${baseline_output}" | awk '/Submitted batch job/ {print $4}' | tail -1
)"
test -n "${baseline_job}"
printf 'arm=baseline job_id=%s dependency=none steps=%s\n' "${baseline_job}" "${MAX_STEPS}"

for arm in "${arms[@]:1}"; do
  output="$(
    Q30_PTV2_MAX_STEPS="${MAX_STEPS}" \
      SBATCH_DEPENDENCY="afterok:${baseline_job}" \
      bash "${LAUNCHER}" --submit "${arm}"
  )"
  printf '%s\n' "${output}"
  job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job/ {print $4}' | tail -1)"
  test -n "${job_id}"
  printf 'arm=%s job_id=%s dependency=afterok:%s steps=%s\n' \
    "${arm}" "${job_id}" "${baseline_job}" "${MAX_STEPS}"
done
