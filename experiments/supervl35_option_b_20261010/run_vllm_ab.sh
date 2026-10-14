#!/bin/bash
set -euo pipefail
umask 077

source_commit=15045cb9425da21622220c0fa3046169a6be017c
root=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010
model=/lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/models/super-vl-35-rlvr-v43-falcon-r3-20260905/hf
job_id=${SLURM_JOB_ID:?}
stage=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
local_root=/raid/scratch/sna/supervl35-vllm-ab-$job_id
case "${BENCHMARK_KIND:-ab}" in
    ab)
        result_dir=$root/vllm_ab/$job_id
        benchmark_args=(--prompt-tokens 10500 --trials 2)
        ;;
    concurrency)
        result_dir=$root/vllm_concurrency/$job_id
        benchmark_args=(--prompt-tokens 512 --trials 2 \
            --concurrencies 1,2,4,8,16,32,128,256,512,1024 --output-tokens 128)
        ;;
    length_sweep)
        : "${BENCH_ISL:?Set BENCH_ISL to 1024 or 10240}"
        : "${BENCH_OSL:?Set BENCH_OSL to 1024 or 10240}"
        case "$BENCH_ISL/$BENCH_OSL" in
            1024/10240|10240/1024|1024/1024) ;;
            *) echo "Unsupported ISL/OSL: $BENCH_ISL/$BENCH_OSL" >&2; exit 2 ;;
        esac
        result_dir=$root/vllm_length_sweep/isl${BENCH_ISL}_osl${BENCH_OSL}/$job_id
        benchmark_args=(--prompt-tokens "$BENCH_ISL" --trials "${BENCH_TRIALS:-2}" \
            --high-concurrency-trials "${BENCH_HIGH_CONCURRENCY_TRIALS:-1}" \
            --concurrencies "${BENCH_CONCURRENCIES:-1,2,4,8,16,32,128,256,512,1024}" \
            --output-tokens "$BENCH_OSL")
        ;;
    profile)
        : "${BENCH_ISL:?Set BENCH_ISL to 1024 or 10240}"
        : "${BENCH_OSL:?Set BENCH_OSL to 1024 or 10240}"
        : "${BENCH_CONCURRENCY:?Set BENCH_CONCURRENCY to 16 or 128}"
        case "$BENCH_ISL/$BENCH_OSL/$BENCH_CONCURRENCY" in
            1024/10240/16|1024/10240/128|10240/1024/16|10240/1024/128) ;;
            *) echo "Unsupported profile case: $BENCH_ISL/$BENCH_OSL/$BENCH_CONCURRENCY" >&2; exit 2 ;;
        esac
        result_dir=$root/vllm_profile/isl${BENCH_ISL}_osl${BENCH_OSL}/c${BENCH_CONCURRENCY}/$job_id
        profile_dir=$local_root/trace
        if [[ "$BENCH_ISL" == 10240 ]]; then
            default_profile_delay=0
        else
            default_profile_delay=64
        fi
        benchmark_args=(--prompt-tokens "$BENCH_ISL" --output-tokens "$BENCH_OSL" \
            --concurrencies "$BENCH_CONCURRENCY" --trials 1 --high-concurrency-trials 1 \
            --profile-dir "$profile_dir" \
            --profile-delay-iterations "${BENCH_PROFILE_DELAY_ITERATIONS:-$default_profile_delay}" \
            --profile-max-iterations "${BENCH_PROFILE_MAX_ITERATIONS:-32}")
        ;;
    ntrace)
        BENCH_ISL=${BENCH_ISL:-1024}
        BENCH_OSL=${BENCH_OSL:-10240}
        BENCH_CONCURRENCY=${BENCH_CONCURRENCY:-16}
        [[ "$BENCH_ISL/$BENCH_OSL/$BENCH_CONCURRENCY" == 1024/10240/16 ]] || {
            echo "ntrace is qualified only for the late 1K/10K c16 comparison" >&2
            exit 2
        }
        result_dir=$root/vllm_ntrace/isl${BENCH_ISL}_osl${BENCH_OSL}/c${BENCH_CONCURRENCY}/$job_id
        profile_dir=$local_root/trace
        benchmark_args=(--prompt-tokens "$BENCH_ISL" --output-tokens "$BENCH_OSL" \
            --concurrencies "$BENCH_CONCURRENCY" --trials 1 --high-concurrency-trials 1 \
            --profile-dir "$profile_dir" --profile-kind ntrace \
            --profile-delay-iterations "${BENCH_PROFILE_DELAY_ITERATIONS:-8192}" \
            --profile-max-iterations "${BENCH_PROFILE_MAX_ITERATIONS:-32}")
        ;;
    dp2_preflight)
        result_dir=$root/vllm_dp2_preflight/$job_id
        benchmark_args=(--prompt-tokens 1024 --output-tokens 128 \
            --concurrencies 2,4,8 --trials 1 --timeout-seconds 1800)
        ;;
    serialized_dp2_preflight)
        : "${SERIALIZED_CHECKPOINT:?Set SERIALIZED_CHECKPOINT to a completed checkpoint}"
        test -f "$SERIALIZED_CHECKPOINT/config.json"
        test -f "$SERIALIZED_CHECKPOINT/model.safetensors.index.json"
        result_dir=$root/vllm_serialized_dp2_preflight/$job_id
        benchmark_args=(--prompt-tokens 1024 --output-tokens 128 \
            --concurrencies 2,4,8 --trials 1 --timeout-seconds 1800 \
            --serialized-checkpoint "$SERIALIZED_CHECKPOINT")
        ;;
    dp2_length_sweep)
        : "${BENCH_ISL:?Set BENCH_ISL to 1024 or 10240}"
        : "${BENCH_OSL:?Set BENCH_OSL to 1024 or 10240}"
        case "$BENCH_ISL/$BENCH_OSL" in
            1024/10240|10240/1024|1024/1024) ;;
            *) echo "Unsupported ISL/OSL: $BENCH_ISL/$BENCH_OSL" >&2; exit 2 ;;
        esac
        result_dir=$root/vllm_dp2_length_sweep/isl${BENCH_ISL}_osl${BENCH_OSL}/$job_id
        benchmark_args=(--prompt-tokens "$BENCH_ISL" --output-tokens "$BENCH_OSL" \
            --concurrencies "${BENCH_CONCURRENCIES:-2,4,8,16,32,128,256,512,1024}" \
            --trials "${BENCH_TRIALS:-2}" \
            --high-concurrency-trials "${BENCH_HIGH_CONCURRENCY_TRIALS:-1}" \
            --max-num-seqs-per-rank "${BENCH_MAX_NUM_SEQS_PER_RANK:-64}" \
            --timeout-seconds 13800)
        ;;
    tp4_preflight)
        result_dir=$root/vllm_tp4_preflight/$job_id
        benchmark_args=(--prompt-tokens 1024 --output-tokens 128 \
            --concurrencies 2,4,8 --trials 1)
        ;;
    *)
        echo "Unsupported BENCHMARK_KIND=${BENCHMARK_KIND}" >&2
        exit 2
        ;;
esac
if [[ ${BENCHMARK_KIND:-ab} == length_sweep || ${BENCHMARK_KIND:-ab} == dp2_length_sweep ]]; then
    benchmark_args+=(--request-multiplier "${BENCH_REQUEST_MULTIPLIER:-1}")
    if [[ ${BENCH_ENABLE_PREFIX_CACHING:-0} == 1 ]]; then
        benchmark_args+=(--enable-prefix-caching)
    fi
    if [[ ${BENCHMARK_KIND:-ab} == length_sweep && -n ${BENCH_MAX_NUM_SEQS:-} ]]; then
        benchmark_args+=(--max-num-seqs "$BENCH_MAX_NUM_SEQS")
    fi
fi
IFS=, read -r -a arms <<< "${BENCH_ARMS:-bf16,moe,qkvo_mamba,all_decoder}"
for arm in "${arms[@]}"; do
    if [[ ! -f "$stage/configs/$arm.yaml" ]]; then
        echo "Missing standalone vLLM config: $stage/configs/$arm.yaml" >&2
        exit 2
    fi
done
attention_args=()
if [[ -n ${BENCH_ATTENTION_BACKEND:-} ]]; then
    case "$BENCH_ATTENTION_BACKEND" in
        FLASHINFER|TRITON_ATTN) attention_args=(--attention-backend "$BENCH_ATTENTION_BACKEND") ;;
        *) echo "Unsupported BENCH_ATTENTION_BACKEND=$BENCH_ATTENTION_BACKEND" >&2; exit 2 ;;
    esac
    case "${BENCHMARK_KIND:-ab}" in
        ab|concurrency|length_sweep|tp4_preflight|profile|ntrace) ;;
        *) echo "Attention override is unsupported for ${BENCHMARK_KIND:-ab}" >&2; exit 2 ;;
    esac
fi
mkdir -p "$local_root" "$result_dir"
cleanup() {
    status=$?
    cp "$local_root/driver.log" "$result_dir/driver.log" 2>/dev/null || true
    if [[ ${BENCHMARK_KIND:-ab} == profile || ${BENCHMARK_KIND:-ab} == ntrace ]]; then
        if compgen -G "$profile_dir/*" >/dev/null; then
            if tar -C "$local_root" -cf "$result_dir/trace.tar" trace; then
                sha256sum "$result_dir/trace.tar" >"$result_dir/trace.tar.sha256" || status=1
            else
                status=1
            fi
        else
            echo "Profiler produced no trace files" >&2
            status=1
        fi
    fi
    printf '%s\n' "$status" >"$result_dir/exit-code"
    exit "$status"
}
trap cleanup EXIT
exec >"$local_root/driver.log" 2>&1

shopt -s nullglob
archives=("$root"/runtime/supervl35-runtime-${source_commit:0:12}-ray*.tar.gz)
if (( ${#archives[@]} != 1 )); then
    echo "Expected one prepared runtime for $source_commit, found ${#archives[@]}" >&2
    exit 1
fi
archive=${archives[0]}
expected_sha=$(awk '{print $1}' "$archive.sha256")
runtime=/raid/scratch/sna/$(basename "${archive%.tar.gz}")
(
    flock -x 9
    if [[ ! -f "$runtime/.verified-sha256" ]] ||
        [[ $(cat "$runtime/.verified-sha256") != "$expected_sha" ]]; then
        staged_archive=$local_root/runtime.tar.gz
        cp "$archive" "$staged_archive"
        printf '%s  %s\n' "$expected_sha" "$staged_archive" | sha256sum -c -
        tar -xzf "$staged_archive" -C /raid/scratch/sna
        printf '%s\n' "$expected_sha" >"$runtime/.verified-sha256"
        rm -f "$staged_archive"
    fi
) 9>"$runtime.lock"
test "$(cat "$runtime/SOURCE_COMMIT")" = "$source_commit"

export NRL_EXPERIMENT_SOURCE=$runtime/source
export MM_TRAINER_MODEL_PATH=$model
export HF_HOME=$local_root/hf
export HF_MODULES_CACHE=$HF_HOME/modules
export TMPDIR=$local_root/tmp
export PYTHONPYCACHEPREFIX=$local_root/pycache
export XDG_CACHE_HOME=$local_root/cache
export TRITON_CACHE_DIR=$local_root/triton
export TORCHINDUCTOR_CACHE_DIR=$local_root/inductor
export CUDA_CACHE_PATH=$local_root/cuda
export VLLM_CACHE_ROOT=$local_root/vllm-cache
export NRL_VIDEO_BACKEND=torchcodec
export VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
export VLLM_TRITON_FORCE_FIRST_CONFIG=1
export PYTHONPATH="$NRL_EXPERIMENT_SOURCE:$NRL_EXPERIMENT_SOURCE/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$NRL_EXPERIMENT_SOURCE/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:$NRL_EXPERIMENT_SOURCE/3rdparty/Gym-workspace/Gym:$NRL_EXPERIMENT_SOURCE/examples/nemo_gym/supervl3p5"
export PYTHONPATH="$HF_MODULES_CACHE:$PYTHONPATH"
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT"

/opt/nemo_rl_venv/bin/python "$NRL_EXPERIMENT_SOURCE/experiments/supervl35_option_b_20261010/prewarm_hf_modules.py" "$model"
vllm_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "$vllm_python"
"$vllm_python" -c 'import vllm; print("vLLM worker version:", vllm.__version__)'
if [[ ${BENCHMARK_KIND:-ab} == ntrace ]]; then
    ntrace_archive=$stage/ntrace-source.tar.gz
    test -f "$ntrace_archive" && test -f "$ntrace_archive.sha256"
    (cd "$stage" && sha256sum -c ntrace-source.tar.gz.sha256)
    ntrace_source=$local_root/ntrace-source
    mkdir -p "$ntrace_source"
    tar -xzf "$ntrace_archive" -C "$ntrace_source"
    test -f "$ntrace_source/pyproject.toml"
    export UV_CACHE_DIR=$local_root/uv-cache
    NTRACE_BUILD_CUPTI_CPP=ON NTRACE_REQUIRE_CUXXFILT=1 \
        uv pip install --python "$vllm_python" --target "$local_root/ntrace-target" \
        --no-deps "$ntrace_source"
    export PYTHONPATH="$local_root/ntrace-target:$stage:$PYTHONPATH"
    export NTRACE_DECODE_DELAY_ITERATIONS=${BENCH_PROFILE_DELAY_ITERATIONS:-8192}
    export NTRACE_DECODE_NUM_ITERS=${BENCH_PROFILE_MAX_ITERATIONS:-32}
    export NTRACE_ROLLOUT_RANKS=0-3
    export NTRACE_ROLLOUT_CAPTURE_ITER=0
    export NTRACE_ROLLOUT_NUM_ITERS=$NTRACE_DECODE_NUM_ITERS
    export NTRACE_ROLLOUT_GRAPH_CAPTURE=early
    export NTRACE_ROLLOUT_OUTPUT_DIR=$profile_dir
    export NTRACE_STACK_CAPTURE_SCOPE=first
    export NTRACE_SAVE_CPU_NVTX=1
    "$vllm_python" -c 'import ntrace, ntrace._cupti_cpp, pyarrow; from ntrace.backends import get_backend; print("ntrace", ntrace.__file__, "native", ntrace._cupti_cpp.__file__, "pyarrow", pyarrow.__version__, "backend", type(get_backend()).__name__)'
    native_so=$("$vllm_python" -c 'import ntrace._cupti_cpp; print(ntrace._cupti_cpp.__file__)')
    sha256sum "$native_so" >"$result_dir/ntrace-native.sha256"
    cp "$stage/ntrace-commit.txt" "$result_dir/ntrace-commit.txt"
fi
printf 'source_commit=%s\nbench_code_commit=%s\nruntime_sha256=%s\nimage=%s\nbenchmark_kind=%s\nattention_backend=%s\nrequest_multiplier=%s\nprefix_caching_override=%s\n' \
    "$source_commit" "${BENCH_CODE_COMMIT:-unknown}" "$expected_sha" "${NRL_BENCH_IMAGE:-unknown}" \
    "${BENCHMARK_KIND:-ab}" "${BENCH_ATTENTION_BACKEND:-recipe}" \
    "${BENCH_REQUEST_MULTIPLIER:-1}" "${BENCH_ENABLE_PREFIX_CACHING:-0}" >"$result_dir/provenance.txt"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

if [[ ( ${BENCHMARK_KIND:-ab} == profile || ${BENCHMARK_KIND:-ab} == ntrace ) && ${#arms[@]} -ne 1 ]]; then
    echo "Profiling requires exactly one BENCH_ARMS value" >&2
    exit 2
fi
for arm in "${arms[@]}"; do
    case "$arm" in
        bf16|moe|moe_attention|moe_mamba|qkvo_mamba|all_decoder) ;;
        *) echo "Unsupported arm: $arm" >&2; exit 2 ;;
    esac
    printf '%s Starting %s\n' "$(date -u +%FT%TZ)" "$arm"
    if [[ ${BENCHMARK_KIND:-ab} == dp2_preflight || ${BENCHMARK_KIND:-ab} == dp2_length_sweep || ${BENCHMARK_KIND:-ab} == serialized_dp2_preflight ]]; then
        PYTHONPATH="$(dirname "$(dirname "$stage")"):$PYTHONPATH" \
            "$vllm_python" "$stage/bench_vllm_dp.py" \
            --config "$stage/configs/$arm.yaml" \
            --output "$result_dir/$arm.json" \
            "${benchmark_args[@]}"
    else
        "$vllm_python" "$stage/bench_vllm_generation.py" \
            --config "$stage/configs/$arm.yaml" \
            --output "$result_dir/$arm.json" \
            "${benchmark_args[@]}" "${attention_args[@]}"
    fi
    printf '%s Completed %s\n' "$(date -u +%FT%TZ)" "$arm"
done
