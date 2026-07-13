# SWE-RL Qwen30 SpecDec Manifest Submit Status

Manifest: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_launch_manifest_20260616.csv`
Summary: not_selected=1, submitted=5

| action | method | K | online | host | preflight | validation | operation | job | stdout | stderr |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| poll_existing | `suffix` | 32 | false | `oci-hsg-cs-001-vscode-02` | ok_controlmaster | ok | not_selected | 3351394 | `` | `` |
| launch_static | `eagle3` | 3 | false | `oci-hsg-cs-001-vscode-02` | ok_controlmaster | ok; stages patched non-suffix launcher | submitted | 3365630 | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P1_launch_static_eagle3_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17.stdout.log` | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P1_launch_static_eagle3_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17.stderr.log` |
| launch_static | `pard` | 5 | false | `oci-hsg-cs-001-vscode-02` | ok_controlmaster | ok; stages patched non-suffix launcher | submitted | 3365631 | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P2_launch_static_pard_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18.stdout.log` | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P2_launch_static_pard_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18.stderr.log` |
| launch_static | `pard2` | 3 | false | `oci-hsg-cs-001-vscode-02` | ok_controlmaster | ok; stages patched non-suffix launcher | submitted | 3365632 | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P2_launch_static_pard2_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19.stdout.log` | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P2_launch_static_pard2_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19.stderr.log` |
| launch_online | `pard` | 5 | true | `oci-hsg-cs-001-vscode-02` | ok_controlmaster | ok; stages patched non-suffix launcher | submitted | 3365633 | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P3_launch_online_pard_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20.stdout.log` | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P3_launch_online_pard_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20.stderr.log` |
| launch_online | `pard2` | 3 | true | `oci-hsg-cs-001-vscode-02` | ok_controlmaster | ok; stages patched non-suffix launcher | submitted | 3365634 | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P3_launch_online_pard2_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21.stdout.log` | `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/2026-06-17T143229-0700_P3_launch_online_pard2_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21.stderr.log` |

## Commands

### suffix_step1_smoke_ctx40k_vllm40k_specdec_k32_arcticpy313_ray254_py313_r16

```bash
set -o pipefail; echo '[squeue]'; squeue -j 3351394 -h -o '%i|%T|%R|%M|%L|%S|%N' || true; echo '[sacct]'; sacct -X -j 3351394 --format=JobIDRaw,JobName%80,State,ExitCode,Elapsed,Start,End -P -n || true; echo '[tail]'; if [ -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_ctx40k_vllm40k_specdec_k32_arcticpy313_ray254_py313_r16/suffix_step1 ] && [ -d /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_ctx40k_vllm40k_specdec_k32_arcticpy313_ray254_py313_r16/suffix_step1 ]; then logs=$(timeout 20s find /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_ctx40k_vllm40k_specdec_k32_arcticpy313_ray254_py313_r16/suffix_step1 -maxdepth 5 -type f \( -name 'ray-driver*.log' -o -name 'slurm-*.out' \) 2>/dev/null | sort | tail -n 3) || true; if [ -n "$logs" ]; then printf '%s
' "$logs" | while IFS= read -r log; do if [ -n "$log" ]; then echo "[log] $log"; timeout 20s tail -n 80 "$log" | awk -v chunk=20 'BEGIN { n = 0 } { lines[++n] = $0 } END { if (n == 0) { print "[tail-chunk] empty" } for (i = 1; i <= n; i += chunk) { end = i + chunk - 1; if (end > n) end = n; print "[tail-chunk] lines " i "-" end " of " n; for (j = i; j <= end; j++) print lines[j]; } }'; fi; done; else echo '[log] no log found'; fi; else echo '[log] logs_dir missing'; fi
```

### eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 && test -s /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && mkdir -p /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17/eagle3_step1 && python3 - /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17/eagle3_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh <<'PY_PATCH_SWERL_SPECDEC_LAUNCHER'
from pathlib import Path
import os
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding='utf-8')
old = '''    ngram)
      ;;
    *)
      echo "ERROR: SWERL launcher currently supports SPECDEC_METHOD=suffix or ngram, got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
new = '''    ngram|draft_model|pard2|eagle3)
      ;;
    *)
      echo "ERROR: SWERL launcher supports SPECDEC_METHOD=suffix, ngram, draft_model, pard2, or eagle3; got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
if old not in text:
    if 'draft_model|pard2|eagle3' not in text:
        print('[patched-launcher] no SPECDEC_METHOD gate found; copying launcher unchanged', flush=True)
    else:
        print('[patched-launcher] launcher gate already supports non-suffix methods', flush=True)
else:
    text = text.replace(old, new, 1)
dst.write_text(text, encoding='utf-8')
os.chmod(dst, 0o755)
print(f'[patched-launcher] {dst}', flush=True)
PY_PATCH_SWERL_SPECDEC_LAUNCHER
bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17/eagle3_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && env ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=eagle3 RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17/eagle3_step1 DRAFT_FORMAT=eagle3 ENABLE_VLLM_SPECDEC=true POLICY_DRAFT_ENABLED=false SPECDEC_METHOD=eagle3 NUM_SPECULATIVE_TOKENS=3 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=false DRY_RUN=false REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 MODEL_PATH=/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-30B-A3B-Thinking-2507 EXP_SUFFIX=20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17 SBATCH_ACCOUNT=nemotron_n3_post SBATCH_PARTITION=batch MAX_NUM_STEPS=1 'EXTRA_ARGS=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 policy.draft.enabled=false ++policy.generation.vllm_kwargs.speculative_config.method=eagle3 ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1' 'EXTRA_HYDRA_OVERRIDES=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 policy.draft.enabled=false ++policy.generation.vllm_kwargs.speculative_config.method=eagle3 ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1' bash /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17/eagle3_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh
```

### pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 && test -s /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && mkdir -p /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18/pard_step1 && python3 - /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18/pard_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh <<'PY_PATCH_SWERL_SPECDEC_LAUNCHER'
from pathlib import Path
import os
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding='utf-8')
old = '''    ngram)
      ;;
    *)
      echo "ERROR: SWERL launcher currently supports SPECDEC_METHOD=suffix or ngram, got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
new = '''    ngram|draft_model|pard2|eagle3)
      ;;
    *)
      echo "ERROR: SWERL launcher supports SPECDEC_METHOD=suffix, ngram, draft_model, pard2, or eagle3; got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
if old not in text:
    if 'draft_model|pard2|eagle3' not in text:
        print('[patched-launcher] no SPECDEC_METHOD gate found; copying launcher unchanged', flush=True)
    else:
        print('[patched-launcher] launcher gate already supports non-suffix methods', flush=True)
else:
    text = text.replace(old, new, 1)
dst.write_text(text, encoding='utf-8')
os.chmod(dst, 0o755)
print(f'[patched-launcher] {dst}', flush=True)
PY_PATCH_SWERL_SPECDEC_LAUNCHER
bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18/pard_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && env ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=pard RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18/pard_step1 DRAFT_FORMAT=pard ENABLE_VLLM_SPECDEC=true SPECDEC_METHOD=draft_model NUM_SPECULATIVE_TOKENS=5 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=true DRY_RUN=false REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 MODEL_PATH=/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-30B-A3B-Thinking-2507 EXP_SUFFIX=20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18 SBATCH_ACCOUNT=nemotron_n3_post SBATCH_PARTITION=batch MAX_NUM_STEPS=1 'EXTRA_ARGS=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 policy.draft.enabled=false ++policy.generation.vllm_kwargs.speculative_config.method=draft_model ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true' 'EXTRA_HYDRA_OVERRIDES=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 policy.draft.enabled=false ++policy.generation.vllm_kwargs.speculative_config.method=draft_model ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true' bash /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18/pard_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh
```

### pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 && test -s /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && mkdir -p /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19/pard2_step1 && python3 - /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19/pard2_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh <<'PY_PATCH_SWERL_SPECDEC_LAUNCHER'
from pathlib import Path
import os
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding='utf-8')
old = '''    ngram)
      ;;
    *)
      echo "ERROR: SWERL launcher currently supports SPECDEC_METHOD=suffix or ngram, got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
new = '''    ngram|draft_model|pard2|eagle3)
      ;;
    *)
      echo "ERROR: SWERL launcher supports SPECDEC_METHOD=suffix, ngram, draft_model, pard2, or eagle3; got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
if old not in text:
    if 'draft_model|pard2|eagle3' not in text:
        print('[patched-launcher] no SPECDEC_METHOD gate found; copying launcher unchanged', flush=True)
    else:
        print('[patched-launcher] launcher gate already supports non-suffix methods', flush=True)
else:
    text = text.replace(old, new, 1)
dst.write_text(text, encoding='utf-8')
os.chmod(dst, 0o755)
print(f'[patched-launcher] {dst}', flush=True)
PY_PATCH_SWERL_SPECDEC_LAUNCHER
bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19/pard2_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && env ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=pard2 RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19/pard2_step1 DRAFT_FORMAT=pard2 ENABLE_VLLM_SPECDEC=true SPECDEC_METHOD=pard2 NUM_SPECULATIVE_TOKENS=3 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=true SOURCE_VLLM_SITE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat_pyoverlay_nostable_nofp4out_basefa_nofp4fusion_20260614 DRY_RUN=false REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 MODEL_PATH=/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-30B-A3B-Thinking-2507 EXP_SUFFIX=20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19 ARCTIC_SITE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat_pyoverlay_nostable_nofp4out_basefa_nofp4fusion_20260614 SBATCH_ACCOUNT=nemotron_n3_post SBATCH_PARTITION=batch MAX_NUM_STEPS=1 'EXTRA_ARGS=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 policy.draft.enabled=false ++policy.generation.vllm_kwargs.speculative_config.method=pard2 ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true' 'EXTRA_HYDRA_OVERRIDES=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 policy.draft.enabled=false ++policy.generation.vllm_kwargs.speculative_config.method=pard2 ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true' bash /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19/pard2_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh
```

### pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 && test -s /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && mkdir -p /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20/pard_step1 && python3 - /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20/pard_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh <<'PY_PATCH_SWERL_SPECDEC_LAUNCHER'
from pathlib import Path
import os
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding='utf-8')
old = '''    ngram)
      ;;
    *)
      echo "ERROR: SWERL launcher currently supports SPECDEC_METHOD=suffix or ngram, got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
new = '''    ngram|draft_model|pard2|eagle3)
      ;;
    *)
      echo "ERROR: SWERL launcher supports SPECDEC_METHOD=suffix, ngram, draft_model, pard2, or eagle3; got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
if old not in text:
    if 'draft_model|pard2|eagle3' not in text:
        print('[patched-launcher] no SPECDEC_METHOD gate found; copying launcher unchanged', flush=True)
    else:
        print('[patched-launcher] launcher gate already supports non-suffix methods', flush=True)
else:
    text = text.replace(old, new, 1)
dst.write_text(text, encoding='utf-8')
os.chmod(dst, 0o755)
print(f'[patched-launcher] {dst}', flush=True)
PY_PATCH_SWERL_SPECDEC_LAUNCHER
bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20/pard_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && env ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=pard RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20/pard_step1 DRAFT_FORMAT=pard ENABLE_VLLM_SPECDEC=true SPECDEC_METHOD=draft_model NUM_SPECULATIVE_TOKENS=5 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=true PARD_ONLINE_TRAINING=true POLICY_DRAFT_ENABLED=true POLICY_DRAFT_TYPE=pard POLICY_DRAFT_LOSS=hard_ce PARD_TRAINING_MODE=k_slot POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH=128 POLICY_DRAFT_TRAIN_INTERVAL=1 POLICY_DRAFT_REFIT_INTERVAL=1 POLICY_DRAFT_CAT_WEIGHTING=false DRY_RUN=false REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 MODEL_PATH=/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-30B-A3B-Thinking-2507 EXP_SUFFIX=20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20 SBATCH_ACCOUNT=nemotron_n3_post SBATCH_PARTITION=batch MAX_NUM_STEPS=1 'EXTRA_ARGS=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 ++policy.generation.vllm_kwargs.speculative_config.method=draft_model ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true policy.draft.enabled=true ++policy.draft.type=pard ++policy.draft.loss=hard_ce ++policy.draft.training_mode=k_slot ++policy.draft.max_training_sequence_length=128 ++policy.draft.train_interval=1 ++policy.draft.refit_interval=1 ++policy.draft.cat_weighting=false' 'EXTRA_HYDRA_OVERRIDES=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 ++policy.generation.vllm_kwargs.speculative_config.method=draft_model ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true policy.draft.enabled=true ++policy.draft.type=pard ++policy.draft.loss=hard_ce ++policy.draft.training_mode=k_slot ++policy.draft.max_training_sequence_length=128 ++policy.draft.train_interval=1 ++policy.draft.refit_interval=1 ++policy.draft.cat_weighting=false' bash /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20/pard_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh
```

### pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 && test -s /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && mkdir -p /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21/pard2_step1 && python3 - /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21/pard2_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh <<'PY_PATCH_SWERL_SPECDEC_LAUNCHER'
from pathlib import Path
import os
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding='utf-8')
old = '''    ngram)
      ;;
    *)
      echo "ERROR: SWERL launcher currently supports SPECDEC_METHOD=suffix or ngram, got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
new = '''    ngram|draft_model|pard2|eagle3)
      ;;
    *)
      echo "ERROR: SWERL launcher supports SPECDEC_METHOD=suffix, ngram, draft_model, pard2, or eagle3; got ${SPECDEC_METHOD}" >&2
      exit 1
      ;;
'''
if old not in text:
    if 'draft_model|pard2|eagle3' not in text:
        print('[patched-launcher] no SPECDEC_METHOD gate found; copying launcher unchanged', flush=True)
    else:
        print('[patched-launcher] launcher gate already supports non-suffix methods', flush=True)
else:
    text = text.replace(old, new, 1)
dst.write_text(text, encoding='utf-8')
os.chmod(dst, 0o755)
print(f'[patched-launcher] {dst}', flush=True)
PY_PATCH_SWERL_SPECDEC_LAUNCHER
bash -n /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21/pard2_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh && env ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=pard2 RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21/pard2_step1 DRAFT_FORMAT=pard2 ENABLE_VLLM_SPECDEC=true SPECDEC_METHOD=pard2 NUM_SPECULATIVE_TOKENS=3 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=true SOURCE_VLLM_SITE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat_pyoverlay_nostable_nofp4out_basefa_nofp4fusion_20260614 PARD_ONLINE_TRAINING=true POLICY_DRAFT_ENABLED=true POLICY_DRAFT_TYPE=pard2 POLICY_DRAFT_LOSS=pard2 PARD_TRAINING_MODE=k_slot POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH=128 POLICY_DRAFT_TRAIN_INTERVAL=1 POLICY_DRAFT_REFIT_INTERVAL=1 POLICY_DRAFT_CAT_WEIGHTING=true POLICY_DRAFT_ALLOW_GENERIC_PARD2_FALLBACK=false DRY_RUN=false REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613 MODEL_PATH=/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-30B-A3B-Thinking-2507 EXP_SUFFIX=20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21 ARCTIC_SITE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat_pyoverlay_nostable_nofp4out_basefa_nofp4fusion_20260614 SBATCH_ACCOUNT=nemotron_n3_post SBATCH_PARTITION=batch MAX_NUM_STEPS=1 'EXTRA_ARGS=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 ++policy.generation.vllm_kwargs.speculative_config.method=pard2 ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true policy.draft.enabled=true ++policy.draft.type=pard2 ++policy.draft.loss=pard2 ++policy.draft.training_mode=k_slot ++policy.draft.max_training_sequence_length=128 ++policy.draft.train_interval=1 ++policy.draft.refit_interval=1 ++policy.draft.cat_weighting=true ++policy.draft.allow_generic_pard2_fallback=false' 'EXTRA_HYDRA_OVERRIDES=grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 ++policy.generation.vllm_kwargs.speculative_config.method=pard2 ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true policy.draft.enabled=true ++policy.draft.type=pard2 ++policy.draft.loss=pard2 ++policy.draft.training_mode=k_slot ++policy.draft.max_training_sequence_length=128 ++policy.draft.train_interval=1 ++policy.draft.refit_interval=1 ++policy.draft.cat_weighting=true ++policy.draft.allow_generic_pard2_fallback=false' bash /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21/pard2_step1/patched_run_qwen30ba3b_swerl_gb200_suffix_smoke.sh
```

