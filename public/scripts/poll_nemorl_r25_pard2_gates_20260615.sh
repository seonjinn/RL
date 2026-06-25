#!/usr/bin/env bash
# Poll current OCI-HSG NeMo-RL r25/r26 PARD-2 proof gates without printing secrets.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-oci-hsg-cs-001-vscode-02}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
MAX_POLLS="${MAX_POLLS:-0}"
WATCH=false

usage() {
  cat <<'USAGE'
Usage: scripts/poll_nemorl_r25_pard2_gates_20260615.sh [--watch]

Environment:
  REMOTE_HOST       SSH host, default oci-hsg-cs-001-vscode-02
  INTERVAL_SECONDS  Watch interval, default 60
  MAX_POLLS         Watch poll cap; 0 means no cap
USAGE
}

while (($#)); do
  case "$1" in
    --watch)
      WATCH=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

poll_once() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" 'bash -s' <<'REMOTE_EOF'
set +e
sanitize() {
  perl -pe 's/\e\[[0-9;]*[mK]//g' \
    | grep -avE 'COMMAND|WANDB_API_KEY|[[:xdigit:]]{40}' \
    | cut -c1-2200
}

printf 'DATE='; date '+%Y-%m-%d %H:%M:%S %Z'

printf '\nSQUEUE\n'
squeue -j 3324801,3324570,3324571,3325343 \
  -o '%i|%j|%T|%r|%S|%M|%l|%D|%a|%Q' 2>&1 | sanitize

printf '\nSTARTS\n'
squeue --start -j 3324801,3324570,3324571,3325343 \
  -o '%i|%j|%S|%D|%R' 2>&1 | sanitize

printf '\nSACCT\n'
sacct -j 3324801,3324570,3324571,3325343 \
  --format=JobID,JobName%96,State,ExitCode,Elapsed,Start,End,Timelimit,Account%25 \
  -P -n 2>/dev/null \
  | egrep '^(3324801|3324570|3324571|3325343)(\||\.)' \
  | tail -n 200 \
  | sanitize

printf '\nSCONTROL\n'
for jid in 3324801 3324570 3324571 3325343; do
  echo "--- ${jid} ---"
  scontrol show job "${jid}" 2>/dev/null \
    | egrep 'JobId=|JobState=|Reason=|StartTime=|Priority=|Account=|Dependency=|NumNodes=|SchedNodeList=' \
    | sanitize
done

printf '\nLOGS_AND_DIRS\n'
for pair in \
  '3324801|swerl_pard2|/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_fullgrpo_logs/20260615_oci_hsg_swerl_qwen235b_fullgrpo_pard2_ray254_nemogympath_r25/pard2_steps1/3324801-logs/ray-driver.log' \
  '3325343|swerl_pard2_export_fallback|/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_fullgrpo_logs/20260615_oci_hsg_swerl_qwen235b_fullgrpo_pard2_ray254_export_r26/pard2_steps1/3325343-logs/ray-driver.log' \
  '3324570|math_static_pard2_k3|/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_reduced64_pard2_seqpackoff_cp1_r25/static_pard2_k3/3324570-logs/ray-driver.log' \
  '3324571|math_online_pard2_k3|/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_reduced64_pard2_seqpackoff_cp1_r25/online_pard2_k3/3324571-logs/ray-driver.log'; do
  IFS='|' read -r jid method f <<<"${pair}"
  d=$(dirname "${f}")
  parent=$(dirname "${d}")
  echo "--- ${jid} ${method} ---"
  stat -c 'PARENT|%y|%n' "${parent}" 2>/dev/null || echo 'NO_PARENT_LOG_DIR_YET'
  find "${parent}" -maxdepth 1 -type f -printf 'PARENT_FILE|%TY-%Tm-%Td %TH:%TM|%s bytes|%p\n' 2>/dev/null \
    | sort \
    | tail -n 20 \
    | sanitize
  stat -c 'DRIVER|%y|%s bytes|%n' "${f}" 2>/dev/null || echo 'NO_DRIVER_LOG_YET'
  stat -c 'DIR|%y|%n' "${d}" 2>/dev/null || echo 'NO_LOG_DIR_YET'
  if [ -f "${f}" ]; then
    grep -anE 'SETUP COMPLETE|Step [0-9]+/(10|1)|Generating responses|Computing advantages|Computing logprobs|Training policy|Max number of steps|Traceback|RuntimeError|AssertionError|ModuleNotFoundError|CUDA Error: out of memory|OOM|No such file or directory|ActorDiedError|worker_pool|Draft Training Enabled|Draft Refit This Step|Draft Loss|acceptance|NRL_ACTOR_RUNTIME_ENV|NRL_ACTOR_PY_EXEC|NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV|NRL_NEMO_GYM_CREATE_ENV|ray254_r1[024]|Creating virtual environment|Finished creating venv|VllmAsyncGenerationWorker|VllmGenerationWorker|Building transformer-engine|Built transformer-engine|Successfully built|Version mismatch|ERROR|Error|Exception|Worker died|failed|Failed' "${f}" 2>/dev/null \
      | tail -n 320 \
      | sanitize
    echo 'TAIL:'
    tail -n 100 "${f}" 2>/dev/null | sanitize
  fi
done
REMOTE_EOF
}

if ! "${WATCH}"; then
  poll_once
  exit 0
fi

poll_count=0
while :; do
  poll_count=$((poll_count + 1))
  output="$(poll_once)"
  printf '%s\n' "${output}"
  if printf '%s\n' "${output}" | grep -qE '\|RUNNING\||\|COMPLETED\||\|FAILED\||\|CANCELLED\||^DRIVER\|'; then
    exit 0
  fi
  if [[ "${MAX_POLLS}" != "0" && "${poll_count}" -ge "${MAX_POLLS}" ]]; then
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
