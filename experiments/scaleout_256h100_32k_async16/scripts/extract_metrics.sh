#!/bin/bash
# Extract per-step Performance Metrics from ray-driver.log for 256 H100 jobs.
# Usage: ./extract_metrics.sh <jobid>     (run on CW)
#        Or pull log first: scp cw:.../<jobid>-logs/ray-driver.log . && ./extract_metrics.sh <jobid>
#
# Output: CSV to stdout — step,train_step,logprob,gen_exposed,total_e2e,prefill_avg,decode_avg

set -euo pipefail
JOBID=${1:?jobid required}
REPO_DIR=${REPO_DIR:-/lustre/fsw/portfolios/coreai/users/sna/repos/nemo-rl-qwen-swe}
LOG="${REPO_DIR}/${JOBID}-logs/ray-driver.log"

if [ ! -f "$LOG" ]; then
  echo "log not found: $LOG" >&2
  exit 1
fi

echo "step,train_step,logprob,gen_exposed,total_e2e,prefill_avg,decode_avg"

# Each step block looks roughly like:
#   ════ Performance Metrics ════
#   train_step:            60.31s
#   logprob_compute:       12.05s
#   generation_compute:   240.40s
#   total_step:           411.40s
#   vllm prefill_time:      4.23s avg
#   vllm decode_time:     210.18s avg

awk '
  /Step [0-9]+ / { step = $2; }
  /train_step:/        { gsub("s",""); ts = $2 }
  /logprob_compute:/   { gsub("s",""); lp = $2 }
  /generation_compute:/{ gsub("s",""); gn = $2 }
  /total_step:/        { gsub("s",""); tt = $2 }
  /vllm prefill_time:/ { gsub("s",""); pf = $3 }
  /vllm decode_time:/  { gsub("s",""); dc = $3;
                          if (step && ts && tt) {
                            printf "%s,%s,%s,%s,%s,%s,%s\n", step, ts, lp, gn, tt, pf, dc
                            step=""; ts=""; lp=""; gn=""; tt=""; pf=""; dc=""
                          }
                        }
' "$LOG"
