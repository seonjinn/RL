#!/bin/bash
# Times the OpenHands startup steps that run before the first LLM call. Runs inside the agent container.
# Prints one CSV row per step: step,start_epoch,end_epoch,exit_code.
cd /openhands_setup/OpenHands || exit 1
export PATH=/openhands_setup/miniforge3/bin:$PATH
export VIRTUAL_ENV=/openhands_setup/OpenHands/.venv
export POETRY_VIRTUALENVS_PATH=/openhands_setup/OpenHands
export POETRY_CACHE_DIR=/tmp/poetry-cache

step() {
    local name=$1 t0 rc
    shift
    t0=$(date +%s.%N)
    "$@" >/dev/null 2>&1
    rc=$?
    echo "$name,$t0,$(date +%s.%N),$rc"
}

step git_check git diff --quiet HEAD HEAD
step poetry_version poetry run python -c "from openhands import get_version; print(get_version())"
step poetry_noop poetry run python -c pass
step venv_noop .venv/bin/python -c pass
step infer_imports .venv/bin/python -c "import pandas, toml, datasets, jinja2, openhands.agenthub, evaluation.utils.shared; from openhands.core.main import create_runtime, run_controller"
step server_import .venv/bin/python -c "import openhands.runtime.action_execution_server"