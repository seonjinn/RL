#!/usr/bin/env bash
# Poll OCI-HSG NeMo-RL r20 PARD-2 proof gates without printing secrets.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-oci-hsg-cs-001-vscode-02}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
MAX_POLLS="${MAX_POLLS:-0}"
WATCH=false

usage() {
  cat <<'USAGE'
Usage: scripts/poll_nemorl_r20_pard2_gates_20260615.sh [--watch]

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
  ssh -o ConnectTimeout=10 "${REMOTE_HOST}" 'bash -s' <<'REMOTE_EOF'
set +e
sanitize() {
  perl -pe 's/\e\[[0-9;]*[mK]//g' \
    | grep -avE 'COMMAND|WANDB_API_KEY|[[:xdigit:]]{40}' \
    | cut -c1-2200
}

printf 'DATE='; date '+%Y-%m-%d %H:%M:%S %Z'

printf '\nSQUEUE\n'
squeue -j 3322940,3322941,3322947 \
  -o '%i|%j|%T|%r|%S|%M|%l|%D|%a|%Q' 2>&1 | sanitize

printf '\nSACCT\n'
sacct -j 3322940,3322941,3322947 \
  --format=JobID,JobName%96,State,ExitCode,Elapsed,Start,End,Timelimit,Account%25 \
  -P -n 2>/dev/null \
  | egrep '^(3322940|3322941|3322947)(\||\.)' \
  | tail -n 160 \
  | sanitize

printf '\nLOGS_AND_DIRS\n'
for pair in \
  '3322940 static_pard2_k3 /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_reduced64_pard2_dynamicvenv_r20/static_pard2_k3/3322940-logs/ray-driver.log' \
  '3322941 online_pard2_k3 /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_reduced64_pard2_dynamicvenv_r20/online_pard2_k3/3322941-logs/ray-driver.log' \
  '3322947 swerl_pard2 /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_fullgrpo_logs/20260615_oci_hsg_swerl_qwen235b_fullgrpo_pard2_dynamicvenv_r20/pard2_steps1/3322947-logs/ray-driver.log'; do
  set -- $pair
  jid=$1; method=$2; f=$3; d=$(dirname "$f")
  echo "--- $jid $method ---"
  stat -c 'DRIVER|%y|%s bytes|%n' "$f" 2>/dev/null || echo 'NO_DRIVER_LOG_YET'
  stat -c 'DIR|%y|%n' "$d" 2>/dev/null || echo 'NO_LOG_DIR_YET'
  if [ -f "$f" ]; then
    grep -anE 'SETUP COMPLETE|Step [0-9]+/(10|1)|Generating responses|Computing advantages|Computing logprobs|Training policy|Max number of steps|Traceback|RuntimeError|CUDA Error: out of memory|OOM|No such file or directory|ActorDiedError|worker_pool|Draft Training Enabled|Draft Refit This Step|Draft Loss|acceptance|NRL_ACTOR_RUNTIME_ENV|NRL_ACTOR_PY_EXEC|NRL_VLLM_DYNAMIC_RAY_RUNTIME_ENV|ray254_r1[024]|Creating virtual environment|Finished creating venv|VllmAsyncGenerationWorker|VllmGenerationWorker|Building transformer-engine|Built transformer-engine|Successfully built|ERROR|Error|Exception|Worker died|failed|Failed' "$f" 2>/dev/null \
      | tail -n 260 \
      | sanitize
    echo 'TAIL:'
    tail -n 80 "$f" 2>/dev/null | sanitize
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
