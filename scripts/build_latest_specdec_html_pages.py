#!/usr/bin/env python3
# pyright: reportCallIssue=false, reportArgumentType=false, reportAssignmentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportGeneralTypeIssues=false, reportOptionalMemberAccess=false
"""Build latest SpecDec benchmark HTML pages from refreshed CSV artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import importlib.util
import json
import math
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pandas as pd

from vllm024_dflare_report import (
    load_completed_dflare_results,
    match_dflare_baselines,
    relativize_sources,
    render_dflare_section,
    render_dflare_status_section,
    target_profile_rows,
)
from vllm024_profile_report import render_profile_section


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PUBLIC_DATA = ROOT / "public/data"
DFLARE_RESULT_ROOT = ROOT / "experiments/vllm_024_dynamicsd/report"
SYNC_RL_EXPERIMENT_ROOT = ROOT / "experiments/vllm_024_dynamicsd"
DFLARE_COMPLETED_OUT = DFLARE_RESULT_ROOT / "dflare_completed_latest.csv"
DFLARE_STATUS_CSV = DFLARE_RESULT_ROOT / "dflare_job_status_latest.csv"
VLLM024_PROFILE_CSV = DFLARE_RESULT_ROOT / "vllm024_profiles_latest.csv"
SPEEDBENCH_STAGE_SCRIPT = SYNC_RL_EXPERIMENT_ROOT / "stage_speedbench.sh"
SPEEDBENCH_RUNNER = SYNC_RL_EXPERIMENT_ROOT / "benchmark_speedbench_sync_rollout.py"
SYNC_RL_MODEL_MATRIX = SYNC_RL_EXPERIMENT_ROOT / "model_method_matrix.json"
SYNC_RL_SUMMARY_FILES = {
    "DAPO-Math-17k": DFLARE_RESULT_ROOT / "results/dapo_sync_full/summary.csv",
    "OpenMathInstruct-2": DFLARE_RESULT_ROOT / "results/openmath_sync_full/summary.csv",
}
PERFCFG_DYNAMIC_REPLAY_CSV = (
    DFLARE_RESULT_ROOT
    / "results/perfcfg_dynamic_replay_20260706"
    / "vllm024_perfcfg_dynamic_replay_20260706.csv"
)
NEMOTRON_MTP_LEGACY_SMOKE_ROOT = (
    DFLARE_RESULT_ROOT / "results/nemotron_mtp_smoke_20260704"
)
NEMOTRON_MTP_LEGACY_SMOKE_RESULTS = (
    (
        "super",
        "baseline",
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2326451",
    ),
    (
        "super",
        "mtp_static",
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2326452",
    ),
    (
        "super",
        "mtp_dynamic",
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2326453",
    ),
    (
        "ultra",
        "baseline",
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04",
        "2326448",
    ),
    (
        "ultra",
        "mtp_static",
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04",
        "2326449",
    ),
    (
        "ultra",
        "mtp_dynamic",
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04",
        "2326450",
    ),
)
NEMOTRON_MTP_LEGACY_SMOKE_SHARED_METADATA = (
    (("status",), "complete"),
    (("runtime", "vllm_version"), "0.24.0"),
    (("config", "cudagraph_mode"), "PIECEWISE"),
    (("config", "compilation_config", "cudagraph_mode"), "PIECEWISE"),
    (("config", "max_new_tokens"), 128),
    (("config", "temperature"), 1.0),
    (("config", "top_p"), 0.95),
)
NEMOTRON_MTP_K_SWEEP_ROOT = (
    DFLARE_RESULT_ROOT / "results/nemotron_mtp_k_sweep_osl4k_20260706"
)
NEMOTRON_MTP_K_SWEEP_RUNTIME_IMAGE_SHA256 = (
    "4abe89d00ef3581710958ad86b7f2063df753914165441287cca64563f56ab6d"
)
NEMOTRON_MTP_K_SWEEP_DRAFTER_IDENTITY_HASH = (
    "2d1fbc9813736d303849fe0abc46927b8e9d6a998981d253fcac2c0a1bd68f6d"
)
NEMOTRON_MTP_K_SWEEP_PROMPT_SET_HASH = (
    "e340aa9c1183cd8650b3fc288320d75f061f981c63b22da5e358f48668441f89"
)
NEMOTRON_MTP_K_SWEEP_PROMPT_BATCH_HASHES = [
    "cc038f48ccfa59808525ca6f9874ad8c56e2f8b9b21b994411366627a4b72ccb",
    "d3c2455a28c358105c0a665b441f83eb9979be72c51747733f0f5601ba975efc",
    "e073b87ad663ed00aee96bf1b6fb0b15d0355a355e8a0015b4b7372cadd22caf",
]
NEMOTRON_MTP_K_SWEEP_REQUEST_PROVENANCE_HASHES = (
    "c9682cbe32280f9c7367aa7f0ad85e4bb178ef3842c5b75929e555fb80005001",
    "ae57be6315bc695c3905cd807b2ada44506505d44796b9089793ab2435e4b4ba",
    "c19d6f1d69c497901658f8ffff9cda09fb96fa82db3e3cd0ea4a444b89a1b181",
)
NEMOTRON_MTP_K_SWEEP_SEED = 1234
NEMOTRON_MTP_K_SWEEP_RUNTIME_GPU_COUNT = 4
NEMOTRON_MTP_K_SWEEP_CONTEXT_PROFILE = "builtin_smoke_or_pinned_math_dataset"
NEMOTRON_MTP_K_SWEEP_ROPE_CONFIG_HASH = (
    "c5a448d4ee1c1c3c1acff52a016d20f1466d01f81cf2ba48207a7de52578a206"
)
NEMOTRON_MTP_K_SWEEP_MODEL_IDENTITIES = {
    "super": {
        "model_config_hash": (
            "699f34f0fc645d29ebffa5767fb59e6ae6ec98e3a4605485eb9913256d0df7e6"
        ),
        "model_checkpoint_hash": (
            "e734fe7158a6a698869f4354c1a645304a081c163a23a575e558f1c3da0a1f98"
        ),
        "model_view_marker_hash": (
            "d0571d9d675cd617819590be521daf3c423e25c47ef15051a1e732da2afe5d24"
        ),
        "distributed_executor_backend": "mp",
    },
    "ultra": {
        "model_config_hash": (
            "8f92735a43afae0d94b73fb9e658910ed548818a188eb2fc51513e88c9e689cd"
        ),
        "model_checkpoint_hash": (
            "36c8d39c827c8fe26f02070cc812ecd2e105a37ea13af92ce4e14b82af503ddd"
        ),
        "model_view_marker_hash": (
            "09a782c09cd3f0f1446b790e463e81f5eeb3ada2e7f3ce5f43b962efd542ac7d"
        ),
        "distributed_executor_backend": "ray",
    },
}
NEMOTRON_MTP_K_SWEEP_RESULTS = (
    (
        "super",
        "baseline",
        0,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2335027",
        2,
        1,
    ),
    (
        "super",
        "k1",
        1,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2335049",
        2,
        1,
    ),
    (
        "super",
        "k3",
        3,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2335028",
        2,
        1,
    ),
    (
        "super",
        "k5",
        5,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2335033",
        2,
        1,
    ),
    (
        "ultra",
        "baseline",
        0,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04",
        "2335029",
        8,
        2,
    ),
    (
        "ultra",
        "k1",
        1,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04",
        "2335051",
        8,
        2,
    ),
    (
        "ultra",
        "k3",
        3,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04",
        "2335053",
        8,
        2,
    ),
    (
        "ultra",
        "k5",
        5,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04",
        "2335030",
        8,
        2,
    ),
)
NEMOTRON_MTP_K_SWEEP_SHARED_METADATA = (
    (("status",), "complete"),
    (("runtime", "vllm_version"), "0.24.0"),
    (
        ("config", "runtime_image_sha256"),
        NEMOTRON_MTP_K_SWEEP_RUNTIME_IMAGE_SHA256,
    ),
    (("config", "cudagraph_mode"), "PIECEWISE"),
    (("config", "compilation_config", "cudagraph_mode"), "PIECEWISE"),
    (("config", "temperature"), 1.0),
    (("config", "top_p"), 1.0),
    (("config", "max_new_tokens"), 4096),
    (("config", "seed"), NEMOTRON_MTP_K_SWEEP_SEED),
    (("config", "num_prompts"), 8),
    (("config", "samples_per_prompt"), 4),
    (("config", "rollout_batches"), 3),
    (("config", "scenario"), "synchronous_rl_rollout"),
    (("config", "sync_barrier"), "LLM.generate_return"),
    (("config", "source_recipe"), "sync-rl-math-rollout"),
    (
        ("config", "drafter_config_hash"),
        NEMOTRON_MTP_K_SWEEP_DRAFTER_IDENTITY_HASH,
    ),
    (
        ("config", "drafter_checkpoint_hash"),
        NEMOTRON_MTP_K_SWEEP_DRAFTER_IDENTITY_HASH,
    ),
    (
        ("config", "drafter_view_marker_hash"),
        NEMOTRON_MTP_K_SWEEP_DRAFTER_IDENTITY_HASH,
    ),
    (("config", "prompt_set_hash"), NEMOTRON_MTP_K_SWEEP_PROMPT_SET_HASH),
    (
        ("config", "prompt_batch_hashes"),
        NEMOTRON_MTP_K_SWEEP_PROMPT_BATCH_HASHES,
    ),
    (("config", "pipeline_parallel_size"), 1),
    (("runtime", "gpu_count"), NEMOTRON_MTP_K_SWEEP_RUNTIME_GPU_COUNT),
    (("config", "context_profile"), NEMOTRON_MTP_K_SWEEP_CONTEXT_PROFILE),
    (("config", "rope_config_hash"), NEMOTRON_MTP_K_SWEEP_ROPE_CONFIG_HASH),
)
NEMOTRON_MTP_OSL16K_FULL_ROOT = (
    DFLARE_RESULT_ROOT / "results/nemotron_mtp_osl16k_20260706"
)
NEMOTRON_MTP_OSL16K_PROMPT_JSONL = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/datasets/"
    "openmathinstruct2_469216e3f46f_prompts_1024_offset0.jsonl"
)
NEMOTRON_MTP_OSL16K_PROMPT_SET_HASH = (
    "fd03df32bffccd0fc627eb86db08979047408a73e668703b16c360f3f4bbf08c"
)
NEMOTRON_MTP_OSL16K_PROMPT_BATCH_HASHES = [
    "e013833a4e66f9bc7f05e005e209da3e192d3273ab6f42c59e82144e186ca4de",
    "594583a8182a207bd5a5b610a00a1d4399dee7c5505765148d334ffa936dd643",
    "aef0606b42d855273cbe00fcaecc68d67842bc3e8835961c5b8cbd16b186ed0b",
]
NEMOTRON_MTP_OSL16K_REQUEST_PROVENANCE_HASHES = (
    "1f6c94cf1fd52469695a7d508183e0e573562af5aa4cccbfe92a703a905efe18",
    "362a1e51144fe7e778d6ee8d6514a261d6b116f84dc57e0aa09e3ec45148271d",
    "c68b0cdd1799ae7f42cefceb54e9939554e9338ae0cd601e246909d495bddc1a",
)
NEMOTRON_MTP_OSL16K_FULL_RESULTS = (
    (
        "super",
        "baseline",
        0,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2335018",
        2,
        1,
    ),
    (
        "super",
        "k3",
        3,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2335019",
        2,
        1,
    ),
    (
        "super",
        "k5",
        5,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e",
        "2335035",
        2,
        1,
    ),
    (
        "ultra",
        "baseline",
        0,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04",
        "2335020",
        8,
        2,
    ),
    (
        "ultra",
        "k5",
        5,
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04",
        "2335021",
        8,
        2,
    ),
)
NEMOTRON_MTP_OSL16K_FULL_SHARED_METADATA = (
    (("status",), "complete"),
    (("runtime", "vllm_version"), "0.24.0"),
    (("runtime", "gpu_count"), NEMOTRON_MTP_K_SWEEP_RUNTIME_GPU_COUNT),
    (
        ("config", "runtime_image_sha256"),
        NEMOTRON_MTP_K_SWEEP_RUNTIME_IMAGE_SHA256,
    ),
    (("config", "cudagraph_mode"), "PIECEWISE"),
    (("config", "compilation_config", "cudagraph_mode"), "PIECEWISE"),
    (("config", "temperature"), 1.0),
    (("config", "top_p"), 1.0),
    (("config", "max_new_tokens"), 16384),
    (("config", "seed"), NEMOTRON_MTP_K_SWEEP_SEED),
    (("config", "num_prompts"), 16),
    (("config", "samples_per_prompt"), 4),
    (("config", "requests_per_rollout_batch"), 64),
    (("config", "global_requests_per_rollout_batch"), 64),
    (("config", "rollout_batches"), 3),
    (("config", "scenario"), "synchronous_rl_rollout"),
    (("config", "sync_barrier"), "LLM.generate_return"),
    (("config", "source_recipe"), "sync-rl-math-rollout"),
    (("config", "prompt_jsonl"), NEMOTRON_MTP_OSL16K_PROMPT_JSONL),
    (
        ("config", "drafter_config_hash"),
        NEMOTRON_MTP_K_SWEEP_DRAFTER_IDENTITY_HASH,
    ),
    (
        ("config", "drafter_checkpoint_hash"),
        NEMOTRON_MTP_K_SWEEP_DRAFTER_IDENTITY_HASH,
    ),
    (
        ("config", "drafter_view_marker_hash"),
        NEMOTRON_MTP_K_SWEEP_DRAFTER_IDENTITY_HASH,
    ),
    (("config", "prompt_set_hash"), NEMOTRON_MTP_OSL16K_PROMPT_SET_HASH),
    (
        ("config", "prompt_batch_hashes"),
        NEMOTRON_MTP_OSL16K_PROMPT_BATCH_HASHES,
    ),
    (("config", "pipeline_parallel_size"), 1),
    (("config", "context_profile"), NEMOTRON_MTP_K_SWEEP_CONTEXT_PROFILE),
    (("config", "rope_config_hash"), NEMOTRON_MTP_K_SWEEP_ROPE_CONFIG_HASH),
)
DFLARE_COMPLETED_DIRS = [
    DFLARE_RESULT_ROOT / "20260703_dflare_completed",
    DFLARE_RESULT_ROOT / "20260704_dflare_completed",
]


def resolve_data_source(name: str) -> Path:
    docs_path = DOCS / name
    return docs_path if docs_path.exists() else PUBLIC_DATA / name


def publish_public_data(src: Path, dst_dir: Path = PUBLIC_DATA) -> Path | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    if dst.suffix in {".csv", ".html", ".json", ".txt"}:
        raw = dst.read_bytes()
        dst.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return dst


MAIN_VLLM = resolve_data_source("vllm_standalone_all_batches_combined_20260619.csv")
VLLM_LIVE_SOURCES = [
    (
        DOCS / "oci_qmath_extra_k_live_log_metrics_20260620.csv",
        "OCI Math extra-K sweep, refreshed 2026-06-21",
        20,
    ),
    (
        DOCS / "oci_qmath_pard2_k_sweep_live_log_metrics_20260620.csv",
        "OCI Math PARD-2 K sweep, refreshed 2026-06-21",
        10,
    ),
    (
        DOCS / "oci_qmath_pard_pard2_k16_focus_live_log_metrics_20260620.csv",
        "OCI Math Qwen32 PARD/PARD-2 K16 retry, refreshed 2026-06-21",
        30,
    ),
    (
        DOCS / "lyris_qwen235b_swe_pard2_k_sweep_live_log_metrics_20260620.csv",
        "Lyris SWE Qwen235B PARD-2 K sweep",
        15,
    ),
]
DFLASH = DOCS / "qwen3_235b_dflash_retry28_openmath_metrics.csv"
VLLM_LEGACY_NORMALIZED = DOCS / "vllm_standalone_qwen30_qwen8_legacy_breakdowns_20260625.csv"
VLLM_TEMP_TRENDS = DOCS / "vllm_standalone_temp0_temp1_trends_20260616.csv"
VLLM_ADDED_OUT = DOCS / "vllm_standalone_added_results_latest.csv"
VLLM_ADDED_INPUT = PUBLIC_DATA / "vllm_standalone_added_results_latest.csv"
VLLM_HTML_LATEST = DOCS / "vllm_standalone_results_latest.html"

NEMORL_MANIFESTS = (
    sorted(ROOT.glob("latest_lyris_nemorl_qwen235b_*20260621_jobs.csv"))
    + sorted(DOCS.glob("latest_lyris_nemorl_perfcfg_*wandb_20260622_jobs.csv"))
    + sorted(DOCS.glob("latest_lyris_nemorl_*20260623_jobs.csv"))
    + sorted(DOCS.glob("latest_lyris_nemorl_*20260624_jobs.csv"))
    + sorted(DOCS.glob("latest_lyris_nemorl_*20260625_jobs.csv"))
)
NEMORL_SUMMARY = DOCS / "lyris_qwen235b_pr2879_live_summary_skip_step1_20260621.csv"
NEMORL_ADDITIONAL_SUMMARIES = [
    DOCS / "lyris_20260623_current_plus_eagerfalse_summary_skip_step1.csv",
    DOCS / "qwen32_pardk1_20260624_summary_skip1_latest.csv",
]
NEMORL_COMPARISON_SUMMARIES = [
    DOCS / "qwen32_pard_eagerfalse_compare_20260624.csv",
    DOCS / "nemorl_specdec_slowdown_watchlist_20260624.csv",
]
NEMORL_PREJULY_CANONICAL = (
    DOCS / "lyris_nemorl_perfcfg_specdec_combined_prejuly_20260701.csv"
)
NEMORL_JULY_SOURCES = [
    {
        "path": DOCS / "lyris_nemorl_v020_best_math_live_metrics_20260704.csv",
        "source_group": "Lyris NeMo-RL v0.20 best-Math live 2026-07-04",
        "cluster": "lyris",
    },
    {
        "path": DOCS / "lyris_qwen30_sync_pard_strict_matched_metrics_20260702.csv",
        "source_group": "Lyris Qwen30 sync PerfCfg CG-on matched 2026-07-02",
        "cluster": "lyris",
        "num_nodes": 4,
        "gpus_per_node": 4,
        "segment": 4,
        "config_segment_size": 4,
        "target_tensor_parallel_size": 1,
        "draft_tensor_parallel_size": 1,
        "attention_backend": "TRITON_ATTN",
        "moe_backend": "triton",
        "cohort": "standard",
        "fuse_allreduce_rms": True,
    },
    {
        "path": DOCS / "lyris_qwen30_async1off_strict_matched_live_metrics_20260702.csv",
        "source_group": "Lyris Qwen30 async-1off PerfCfg CG-on matched 2026-07-02",
        "cluster": "lyris",
        "target_tensor_parallel_size": 1,
        "draft_tensor_parallel_size": 1,
        "cohort": "standard",
        "fuse_allreduce_rms": True,
    },
    {
        "path": DOCS / "lyris_qwen32_sync_eagle3_matched_live_metrics_20260702.csv",
        "source_group": "Lyris Qwen32 sync Eagle-3 PerfCfg CG-on matched 2026-07-02",
        "cluster": "lyris",
        "num_nodes": 4,
        "gpus_per_node": 4,
        "segment": 4,
        "config_segment_size": 4,
        "target_tensor_parallel_size": 2,
        "draft_tensor_parallel_size": 1,
        "attention_backend": "TRITON_ATTN",
        "moe_backend": "triton",
        "cohort": "standard",
        "fuse_allreduce_rms": True,
    },
    {
        "path": DOCS / "lyris_qwen32_sync_pard_tp2_noarrms_matched_live_metrics_20260702.csv",
        "source_group": "Lyris Qwen32 sync PARD TP2 no-AR-RMS CG-on matched 2026-07-02",
        "cluster": "lyris",
        "num_nodes": 4,
        "gpus_per_node": 4,
        "segment": 4,
        "config_segment_size": 4,
        "attention_backend": "TRITON_ATTN",
        "moe_backend": "triton",
        "max_num_seqs": 64,
        "max_num_batched_tokens": 32768,
        "fuse_allreduce_rms": False,
    },
    {
        "path": DOCS / "lyris_qwen32_async1off_eagle3_matched_live_metrics_20260702.csv",
        "source_group": "Lyris Qwen32 async-1off PerfCfg CG-on matched 2026-07-02",
        "cluster": "lyris",
        "target_tensor_parallel_size": 1,
        "draft_tensor_parallel_size": 1,
        "cohort": "standard",
        "fuse_allreduce_rms": True,
    },
    {
        "path": DOCS / "lyris_qwen235b_sync_eagle3_absolute_metrics_20260702.csv",
        "source_group": "Lyris Qwen235B sync Eagle-3 PerfCfg CG-on absolute 2026-07-02",
        "cluster": "lyris",
        "model": "Qwen3-235B-A22B",
        "mode": "sync",
        "num_nodes": 32,
        "gpus_per_node": 4,
        "segment": 16,
        "config_segment_size": 16,
        "target_tensor_parallel_size": 8,
        "draft_tensor_parallel_size": 1,
        "attention_backend": "TRITON_ATTN",
        "moe_backend": "triton",
        "cohort": "standard",
        "fuse_allreduce_rms": True,
    },
    {
        "path": DOCS / "pretyche_qwen32_sync_osl32k_matched_live_metrics_20260702.csv",
        "source_group": "Pretyche Qwen32 sync OSL32768 PerfCfg CG-on matched 2026-07-02",
        "cluster": "pretyche",
        "num_nodes": 4,
        "gpus_per_node": 4,
        "segment": 4,
        "config_segment_size": 4,
        "target_tensor_parallel_size": 2,
        "draft_tensor_parallel_size_by_method": {"eagle3": 1, "pard": 2},
        "attention_backend": "TRITON_ATTN",
        "moe_backend": "triton",
    },
]
NEMORL_SACCT = DOCS / "lyris_qwen235b_pr2879_sacct_20260621.psv"
NEMORL_OUT = DOCS / "lyris_qwen235b_pr2879_live_enriched_20260621.csv"
NEMORL_LYRIS_HISTORICAL_SOURCES = [
    (
        DOCS / "lyris_nemorl_qwen30_qwen32_pr2879_step20_speedups_20260622.csv",
        "Lyris Qwen30/Qwen32 PerfCfg OSL4096 latest-main+PR2879 2026-06-22",
        "performance recipe default plus latest-main+PR2879 topology-aware fix, enforce_eager=true (CUDA graph disabled), temp=1.0/top_p=1.0, step>=2 summary",
        1,
    ),
    (
        DOCS / "lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv",
        "Lyris Qwen30/Qwen32 PerfCfg OSL4096 2026-06-18",
        "performance recipe default, enforce_eager=true (CUDA graph disabled), temp=1.0/top_p=1.0, step>=2 live summary",
        2,
    ),
]
NEMORL_OCI_HISTORICAL = DOCS / "nemorl_integrated_specdec_results_clean_20260617.csv"
NEMORL_LIVE_K_SWEEP_SUMMARY = DOCS / "lyris_nemorl_qwen30_qwen32_eagle3_k_sweep_live_summary_20260622.csv"
NEMORL_LIVE_K_SWEEP_SOURCE_GROUP = "Lyris Qwen30/Qwen32 PerfCfg OSL4096 CUDA-graph-disabled K sweep 2026-06-22"
NEMORL_LIVE_K_SWEEP_CHECKED_AT = "2026-06-22 21:31 PDT"
NEMORL_COMBINED_OUT = DOCS / "lyris_nemorl_perfcfg_specdec_combined_latest.csv"
NEMORL_HTML = DOCS / "lyris_nemorl_perfcfg_specdec_live_status_latest.html"
NEMORL_HTML_DATED = DOCS / "lyris_nemorl_perfcfg_specdec_live_status_20260622.html"
WANDB_ENTITY = "nvidia"

NEMORL_WANDB_URL_BY_JOB = {
    "2182145": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/5cpkkvty",
    "2182146": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/5yg0y4re",
    "2182147": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/lseuvql7",
    "2182148": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/egbzz2wt",
    "2182149": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/yx9yziip",
    "2182151": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/p2i13z4l",
    "2188681": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/iek8hu2z",
    "2188682": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/4ig7pz6k",
    "2191503": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/z7osk7c9",
    "2191504": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/shd6n6iz",
    "2191506": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/wnxmitja",
    "2191507": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/wd0prcvj",
    "2191509": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/8qese1bd",
    "2191510": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/fraoa396",
    "2191511": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/sb7xqyyz",
    "2191513": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/h0a8bwb1",
    "2191514": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/5gbqmceg",
    "2191801": "https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/pnq3aguc",
}

NEMORL_CONFIRMED_WANDB_DISABLED_JOBS = {
    "2152193",
    "2152194",
    "2152195",
    "2152196",
    "2175019",
    "3333528",
    "3333533",
}

NEMORL_LIVE_K_SWEEP_META = [
    {
        "job_id": "2177867",
        "model": "qwen30ba3b",
        "mode": "sync",
        "k": 5,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "FAILED",
        "elapsed": "00:08:00",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "failed_before_completed_step",
        "notes": "Old pre-fix K5 sync attempt; CUBLAS GEMM failure at step 1.",
        "error": "RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k5_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_sync_eagle3/2177867-logs/ray-driver.log",
    },
    {
        "job_id": "2177868",
        "model": "qwen30ba3b",
        "mode": "async-1off",
        "k": 5,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:19",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore error; not clean performance data.",
        "error": "RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasSetStream",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k5_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_async1off_eagle3/2177868-logs/ray-driver.log",
    },
    {
        "job_id": "2177869",
        "model": "qwen32",
        "mode": "sync",
        "k": 5,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "02:21:17",
        "metric_state": "parsed_completed",
        "notes": "Completed 20/20; log confirms enforce_eager=true.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k5_contextclamp_step20_recipe_osl_temp1/logs/qwen32_sync_eagle3/2177869-logs/ray-driver.log",
    },
    {
        "job_id": "2177870",
        "model": "qwen32",
        "mode": "async-1off",
        "k": 5,
        "nodes_x_gpus": "8x4",
        "segment": 8,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:01",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore error; not clean performance data.",
        "error": "RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasGemmEx",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k5_contextclamp_step20_recipe_osl_temp1/logs/qwen32_async1off_eagle3/2177870-logs/ray-driver.log",
    },
    {
        "job_id": "2177871",
        "model": "qwen30ba3b",
        "mode": "sync",
        "k": 7,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "FAILED",
        "elapsed": "00:07:46",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "failed_before_completed_step",
        "notes": "Old pre-fix K7 sync attempt; Triton device-side assert at step 1.",
        "error": "RuntimeError: Triton Error [CUDA]: device-side assert triggered",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k7_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_sync_eagle3/2177871-logs/ray-driver.log",
    },
    {
        "job_id": "2177872",
        "model": "qwen30ba3b",
        "mode": "async-1off",
        "k": 7,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:12",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore error; not clean performance data.",
        "error": "RuntimeError: Triton Error [CUDA]: device-side assert triggered",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k7_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_async1off_eagle3/2177872-logs/ray-driver.log",
    },
    {
        "job_id": "2177873",
        "model": "qwen32",
        "mode": "sync",
        "k": 7,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "02:26:16",
        "metric_state": "parsed_completed",
        "notes": "Completed 20/20; log confirms enforce_eager=true.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k7_contextclamp_step20_recipe_osl_temp1/logs/qwen32_sync_eagle3/2177873-logs/ray-driver.log",
    },
    {
        "job_id": "2177874",
        "model": "qwen32",
        "mode": "async-1off",
        "k": 7,
        "nodes_x_gpus": "8x4",
        "segment": 8,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:01",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore traceback; not clean performance data.",
        "error": "EngineCore traceback at step 1",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k7_contextclamp_step20_recipe_osl_temp1/logs/qwen32_async1off_eagle3/2177874-logs/ray-driver.log",
    },
    {
        "job_id": "2177875",
        "model": "qwen30ba3b",
        "mode": "sync",
        "k": 9,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "01:45:29",
        "metric_state": "parsed_completed",
        "notes": "Completed 20/20; log confirms enforce_eager=true.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k9_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_sync_eagle3/2177875-logs/ray-driver.log",
    },
    {
        "job_id": "2177876",
        "model": "qwen30ba3b",
        "mode": "async-1off",
        "k": 9,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "01:59:05",
        "metric_state": "parsed_completed_with_shutdown_warning",
        "notes": "Completed 20/20 with enforce_eager=true; async log lacks generation-time breakdown, so throughput rows are cleaner than time-speedup rows.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k9_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_async1off_eagle3/2177876-logs/ray-driver.log",
    },
    {
        "job_id": "2177877",
        "model": "qwen32",
        "mode": "sync",
        "k": 9,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "02:39:13",
        "metric_state": "parsed_completed",
        "notes": "Completed 20/20; log confirms enforce_eager=true.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k9_contextclamp_step20_recipe_osl_temp1/logs/qwen32_sync_eagle3/2177877-logs/ray-driver.log",
    },
    {
        "job_id": "2177878",
        "model": "qwen32",
        "mode": "async-1off",
        "k": 9,
        "nodes_x_gpus": "8x4",
        "segment": 8,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:12",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore error; not clean performance data.",
        "error": "RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasSetStream",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k9_contextclamp_step20_recipe_osl_temp1/logs/qwen32_async1off_eagle3/2177878-logs/ray-driver.log",
    },
]


MODEL_MAP = {
    "qwen235b": "Qwen3-235B-A22B",
    "qwen30ba3b": "Qwen3-30B-A3B",
    "qwen30": "Qwen3-30B-A3B",
    "qwen32": "Qwen3-32B",
    "qwen8": "Qwen3-8B",
}

PALETTE = {
    "baseline": "#6b7280",
    "baseline_fuselossfalse": "#b8b8b8",
    "eagle3_k3": "#1f78b4",
    "eagle3_k5": "#a6cee3",
    "eagle3_k7": "#6a3d9a",
    "eagle3_k9": "#cab2d6",
    "eagle3_k8": "#2563eb",
    "pard_k1_tp1": "#ff7f00",
    "pard_k1_tp2": "#fdbf6f",
    "pard_k5": "#e31a1c",
    "pard_k8": "#fb9a99",
    "pard_k12": "#b15928",
    "pard_k16": "#8b1a1a",
    "pard2": "#33a02c",
    "pard2_8b": "#b2df8a",
    "pard2_14b": "#ffff99",
    "pard2_k16": "#1b9e77",
    "pard2_k11": "#66a61e",
    "pard2_k9": "#d95f02",
    "pard2_k5": "#7570b3",
    "pard2_k3": "#e7298a",
    "pard2_k1": "#a6761d",
    "suffix_k32": "#17a398",
    "temp0": "#2563eb",
    "temp1": "#dc2626",
}
NEMOTRON_MODEL_SERIES_COLORS = {
    "Super": "#2563eb",
    "Ultra": "#dc2626",
}

METRIC_PALETTE = {
    "Generation throughput": "#1f78b4",
    "E2E throughput": "#33a02c",
    "Generation time": "#fb9a99",
    "E2E step time": "#e31a1c",
}


WANDB_URL_RE = re.compile(r"https?://wandb\.ai/[^\s\x1b\"'<>]+")


def short_model(value: object) -> str:
    text = str(value)
    replacements = {
        "Qwen3-235B-A22B": "235B",
        "Qwen3-30B-A3B": "30B-A3B",
        "Qwen3-32B": "32B",
        "Qwen3-8B": "8B",
    }
    return replacements.get(text, text.replace("Qwen3-", ""))


def method_label(value: object) -> str:
    text = str(value)
    match = re.fullmatch(r"([a-z0-9]+)_k(\d+)", text)
    if not match:
        return text.replace("_", " ")
    base, k = match.groups()
    names = {
        "eagle3": "Eagle-3",
        "pard": "PARD",
        "pard2": "PARD-2",
        "suffix": "Suffix",
        "dflash": "DFlash",
    }
    return f"{names.get(base, base)} K{k}"


def nemorl_method_label(value: object) -> str:
    text = str(value)
    if text == "baseline":
        return "Baseline"
    if text == "baseline_fuselossfalse":
        return "Baseline fuse_loss=false"
    if text == "pard2":
        return "PARD-2"
    if text == "pard2_8b":
        return "PARD-2 8B"
    if text == "pard2_14b":
        return "PARD-2 14B"
    if text == "pard_k1_tp1":
        return "PARD K1 TP1"
    if text == "pard_k1_tp2":
        return "PARD K1 TP2"
    return method_label(text)


def chart_value(value: object, metric: str) -> str:
    if metric == "speedup":
        return fmt(value, 2, "x")
    if metric == "acceptance_pct":
        return fmt(value, 0, "%")
    return fmt(value, 2)


def chart_tick(value: float, metric: str) -> str:
    if metric == "speedup":
        return f"{value:.1f}x"
    if metric == "acceptance_pct":
        return f"{value:.0f}%"
    if metric == "mean_accept_len":
        return f"{value:.1f}"
    return f"{value:.1f}"


def chart_y_max(max_value: float, metric: str) -> float:
    if metric == "speedup":
        return max(1.1, max_value * 1.22)
    if metric == "acceptance_pct":
        return max(10.0, max_value * 1.22)
    if metric == "mean_accept_len":
        return max(1.0, max_value * 1.22)
    return max(1.0, max_value * 1.22)


def legend_svg(
    methods: list[str],
    x: float,
    y: float,
    gap: float = 116,
    series_colors: Mapping[str, str] = PALETTE,
) -> str:
    width = max(0, (len(methods) - 1) * gap)
    start = x - width / 2
    chunks = []
    for idx, method in enumerate(methods):
        lx = start + idx * gap
        color = series_colors.get(method, "#4b5563")
        chunks.append(
            f'<rect x="{lx:.1f}" y="{y - 8:.1f}" width="14" height="14" rx="2" fill="{color}"/>'
            f'<text x="{lx + 20:.1f}" y="{y + 3:.1f}" font-size="13" fill="#374151">{esc(method_label(method))}</text>'
        )
    return "".join(chunks)


def grouped_bar_svg(rows: pd.DataFrame, title: str, metric: str, methods: list[str]) -> str:
    if rows.empty:
        return ""
    models = [m for m in ["Qwen3-235B-A22B", "Qwen3-30B-A3B", "Qwen3-32B", "Qwen3-8B"] if m in set(rows["model"])]
    if not models:
        models = sorted(rows["model"].dropna().astype(str).unique())
    rows = rows[rows["method"].isin(methods)].copy()
    rows = rows.groupby(["model", "method"], as_index=False)[metric].mean()
    max_value = clean_float(rows[metric].max())
    if math.isnan(max_value) or max_value <= 0:
        return ""
    y_max = chart_y_max(max_value, metric)
    width, height = 760, 330
    left, right, top, bottom = 58, 22, 66, 48
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_for(group_idx: int, method_idx: int) -> float:
        group_w = plot_w / max(1, len(models))
        bar_gap = 4
        inner = min(104, group_w * 0.72)
        bar_w = (inner - bar_gap * (len(methods) - 1)) / len(methods)
        return left + group_idx * group_w + (group_w - inner) / 2 + method_idx * (bar_w + bar_gap)

    def y_for(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    group_w = plot_w / max(1, len(models))
    inner = min(104, group_w * 0.72)
    bar_gap = 4
    bar_w = (inner - bar_gap * (len(methods) - 1)) / len(methods)
    lookup = {(str(row["model"]), str(row["method"])): clean_float(row[metric]) for _, row in rows.iterrows()}
    baseline_line = ""
    if metric == "speedup" and y_max > 1:
        y = y_for(1)
        baseline_line = (
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" '
            'stroke="#94a3b8" stroke-dasharray="5 5"/>'
            f'<text x="{width - right - 72}" y="{y - 6:.1f}" font-size="12" fill="#64748b">1.0x baseline</text>'
        )
    grid = []
    for frac in [0, 0.5, 1.0]:
        value = y_max * frac
        y = y_for(value)
        label = chart_tick(value, metric)
        grid.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="13" fill="#64748b">{label}</text>'
        )
    bars = []
    for gi, model in enumerate(models):
        gx = left + gi * group_w + group_w / 2
        bars.append(f'<text x="{gx:.1f}" y="{height - 17}" text-anchor="middle" font-size="14" fill="#111827">{esc(short_model(model))}</text>')
        for mi, method in enumerate(methods):
            value = lookup.get((model, method), math.nan)
            if math.isnan(value):
                continue
            x = x_for(gi, mi)
            y = y_for(value)
            color = PALETTE.get(method, "#4b5563")
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" rx="3" fill="{color}"/>'
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="12" fill="#111827">{chart_value(value, metric)}</text>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{esc(title)}</text>'
        f'{legend_svg(methods, width / 2, 48)}'
        f'{"".join(grid)}{baseline_line}'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#cbd5e1"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#cbd5e1"/>'
        f'{"".join(bars)}</svg>'
    )


def line_svg(
    rows: pd.DataFrame,
    title: str,
    metric: str,
    x_key: str,
    series_key: str,
    series_colors: Mapping[str, str] = PALETTE,
) -> str:
    if rows.empty:
        return ""
    rows = rows.dropna(subset=[metric, x_key, series_key]).copy()
    if rows.empty:
        return ""
    rows[x_key] = pd.to_numeric(rows[x_key], errors="coerce")
    rows = rows.dropna(subset=[x_key])
    series = sorted(rows[series_key].dropna().astype(str).unique())
    x_values = sorted(rows[x_key].dropna().unique())
    if not series or not x_values:
        return ""
    max_value = clean_float(rows[metric].max())
    if math.isnan(max_value) or max_value <= 0:
        return ""
    y_max = chart_y_max(max_value, metric)
    width, height = 760, 330
    left, right, top, bottom = 58, 24, 66, 48
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_for(value: float) -> float:
        if len(x_values) == 1:
            return left + plot_w / 2
        return left + (list(x_values).index(value) / (len(x_values) - 1)) * plot_w

    def y_for(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    grid = []
    for frac in [0, 0.5, 1.0]:
        value = y_max * frac
        y = y_for(value)
        label = chart_tick(value, metric)
        grid.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="13" fill="#64748b">{label}</text>'
        )
    baseline_line = ""
    if metric == "speedup" and y_max > 1:
        y = y_for(1)
        baseline_line = (
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" '
            'stroke="#94a3b8" stroke-dasharray="5 5"/>'
            f'<text x="{width - right - 72}" y="{y - 6:.1f}" font-size="12" '
            'fill="#64748b">1.0x baseline</text>'
        )
    axis_labels = [
        f'<text x="{x_for(v):.1f}" y="{height - 17}" text-anchor="middle" font-size="14" fill="#111827">{int(v)}</text>'
        for v in x_values
    ]
    lines = []
    for idx, item in enumerate(series):
        color = series_colors.get(
            item,
            ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"][idx % 5],
        )
        sub = rows[rows[series_key].astype(str) == item].sort_values(x_key)
        points = []
        for _, row in sub.iterrows():
            value = clean_float(row[metric])
            if math.isnan(value):
                continue
            points.append((x_for(row[x_key]), y_for(value), value))
        if not points:
            continue
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        lines.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y, value in points:
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
                f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12" fill="#111827">{chart_value(value, metric)}</text>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{esc(title)}</text>'
        f'{legend_svg(series, width / 2, 48, gap=122, series_colors=series_colors)}'
        f'{"".join(grid)}{baseline_line}'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#cbd5e1"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#cbd5e1"/>'
        f'{"".join(axis_labels)}{"".join(lines)}</svg>'
    )


def nemorl_grouped_metric_svg(
    rows: pd.DataFrame,
    title: str,
    y_label: str,
    series: list[tuple[str, str, str]],
    *,
    reference_line: bool = False,
    lower_is_better: bool = False,
) -> str:
    if rows.empty:
        return ""
    rows = rows.copy()
    rows["method_display"] = rows["method_k"].map(nemorl_method_label)
    method_order = [
        "Baseline",
        "Eagle-3 K3",
        "Eagle-3 K5",
        "Suffix K32",
        "PARD K5",
        "PARD K16",
        "PARD-2 K5",
        "PARD-2 K16",
    ]
    methods = [method for method in method_order if method in set(rows["method_display"])]
    if not methods:
        methods = rows["method_display"].dropna().astype(str).tolist()
    plotted_series = []
    max_value = 0.0
    for column, label, color in series:
        values = []
        for method in methods:
            sub = rows[rows["method_display"] == method]
            value = clean_float(sub[column].iloc[0]) if not sub.empty else math.nan
            values.append(value)
            if not math.isnan(value):
                max_value = max(max_value, value)
        if any(not math.isnan(value) for value in values):
            plotted_series.append((column, label, color, values))
    if not plotted_series or max_value <= 0:
        return ""

    width, height = 920, 390
    left, right, top, bottom = 76, 28, 78, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = max(1.15 if reference_line else 0.1, max_value * 1.22)
    group_w = plot_w / max(1, len(methods))
    inner = min(132, group_w * 0.78)
    bar_gap = 4
    bar_w = (inner - bar_gap * (len(plotted_series) - 1)) / len(plotted_series)

    def y_for(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    def fmt_metric(value: float, column: str) -> str:
        if "speedup" in column:
            return f"{value:.2f}x"
        if "time" in column:
            return f"{value:.0f}s"
        return f"{value:.1f}"

    grid = []
    for frac in [0, 0.5, 1.0]:
        value = y_max * frac
        y = y_for(value)
        label = f"{value:.1f}x" if reference_line else f"{value:.0f}" if max_value > 20 else f"{value:.1f}"
        grid.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d1d5db" stroke-dasharray="6 6"/>'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="14" fill="#4b5563">{label}</text>'
        )

    baseline = ""
    if reference_line and y_max > 1:
        y = y_for(1.0)
        baseline = (
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#111827" stroke-dasharray="5 5" stroke-width="1.4"/>'
            f'<text x="{width - right - 76}" y="{y - 8:.1f}" font-size="13" fill="#111827">1.0x baseline</text>'
        )

    legend_parts = []
    legend_gap = 176
    legend_start = width / 2 - ((len(plotted_series) - 1) * legend_gap) / 2
    for idx, (_, label, color, _) in enumerate(plotted_series):
        x = legend_start + idx * legend_gap
        legend_parts.append(
            f'<rect x="{x:.1f}" y="37" width="15" height="15" rx="2" fill="{color}" stroke="#192133" stroke-width="1.8"/>'
            f'<text x="{x + 22:.1f}" y="50" font-size="14" fill="#111827">{esc(label)}</text>'
        )

    bars = []
    for group_idx, method in enumerate(methods):
        gx = left + group_idx * group_w + group_w / 2
        bars.append(f'<text x="{gx:.1f}" y="{height - 24}" text-anchor="middle" font-size="14" fill="#111827">{esc(method)}</text>')
        for series_idx, (column, _, color, values) in enumerate(plotted_series):
            value = values[group_idx]
            if math.isnan(value):
                continue
            x = left + group_idx * group_w + (group_w - inner) / 2 + series_idx * (bar_w + bar_gap)
            y = y_for(value)
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" rx="3" fill="{color}" stroke="#192133" stroke-width="1.8"/>'
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="11" fill="#111827">{fmt_metric(value, column)}</text>'
            )

    direction = "lower is better" if lower_is_better else "higher is better"
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">{esc(title)}</text>'
        f'{"".join(legend_parts)}'
        f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="15" fill="#111827">{esc(y_label)}</text>'
        f'{"".join(grid)}{baseline}'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>'
        f'{"".join(bars)}'
        f'<text x="{width - right}" y="{height - 6}" text-anchor="end" font-size="12" fill="#64748b">{direction}</text>'
        '</svg>'
    )


def nemorl_chart_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    current = rows[pd.to_numeric(rows.get("max_steps"), errors="coerce") == 20].copy()
    current = current[current.apply(nemorl_has_complete_step20_window, axis=1)]
    metric_cols = [
        "gen_tps_speedup",
        "e2e_tps_speedup",
        "generation_time_speedup",
        "e2e_step_time_speedup",
        "generation_worker_tokens_per_sec_per_gpu_mean",
        "e2e_tokens_per_sec_per_gpu_mean",
        "generation_time_s_mean",
        "total_step_time_s_mean",
    ]
    for col in metric_cols:
        current[col] = pd.to_numeric(current.get(col), errors="coerce")
    current = current.dropna(subset=["generation_worker_tokens_per_sec_per_gpu_mean"])
    if current.empty:
        return current
    current["has_gen_speedup"] = current["gen_tps_speedup"].notna()
    current = current.sort_values(
        ["has_gen_speedup", "completed_steps", "gen_tps_speedup"],
        ascending=[False, False, False],
    )
    return current.drop_duplicates(
        subset=["source_group", "model_name", "mode", "max_new_tokens", "method_k"],
        keep="first",
    ).drop(columns=["has_gen_speedup"], errors="ignore")


def nemorl_metric_window(row: pd.Series) -> str:
    completed = clean_float(row.get("completed_steps"))
    last = clean_float(row.get("last_step"))
    max_steps = clean_float(row.get("max_steps"))
    span = first_text(row, "completed_step_span")
    step_filter = first_text(row, "step_filter").lower()

    if not span and not math.isnan(last) and not math.isnan(completed):
        if "step>=2" in step_filter or (
            not math.isnan(max_steps)
            and int(completed) == int(max_steps) - 1
            and int(last) == int(max_steps)
        ):
            span = f"2-{int(last)}"
        elif int(completed) == int(last):
            span = f"1-{int(last)}"

    if span and not math.isnan(completed):
        return f"steps {span} ({int(completed)} metrics)"
    if not math.isnan(completed) and not math.isnan(max_steps):
        last_text = f", last step {int(last)}" if not math.isnan(last) else ""
        return f"partial: {int(completed)}/{int(max_steps)} metrics{last_text}"
    return "not parsed"


def nemorl_has_complete_step20_window(row: pd.Series) -> bool:
    max_steps = clean_float(row.get("max_steps"))
    completed = clean_float(row.get("completed_steps"))
    last = clean_float(row.get("last_step"))
    if math.isnan(max_steps) or int(max_steps) != 20 or math.isnan(completed):
        return False
    if completed >= max_steps:
        return True
    return not math.isnan(last) and last >= max_steps and completed >= max_steps - 1


def nemorl_cuda_graph_label(row: pd.Series) -> str:
    explicit = text_value(row.get("enforce_eager", "")).lower()
    evidence = " ".join(
        [
            explicit,
            text_value(row.get("source_group", "")).lower(),
            text_value(row.get("config_basis", "")).lower(),
            text_value(row.get("run_id", "")).lower(),
        ]
    )
    if explicit in {"true", "1", "yes"} or any(
        token in evidence for token in ["enforce_eager=true", "cuda-graph-disabled", "cudagraphoff", "eagertrue"]
    ):
        return "CG-off"
    if explicit in {"false", "0", "no"} or "enforce_eager=false" in evidence or "eagerfalse" in evidence:
        return "CG-on"
    return "CG-unknown"


def nemorl_setup_label(row: pd.Series) -> str:
    cluster = text_value(row.get("cluster", "")).lower()
    cluster_label = {"lyris": "LYR", "oci-hsg": "OCI"}.get(cluster, cluster.upper() or "cluster?")
    source = text_value(row.get("source_group", ""))
    date_match = re.search(r"2026-(\d{2})-(\d{2})", source)
    date_label = f"{date_match.group(1)}-{date_match.group(2)}" if date_match else "date?"
    source_lower = source.lower()
    if "pard diagnostics" in source_lower:
        source_tag = "PARD"
    elif "k sweep" in source_lower:
        source_tag = "K-sweep"
    elif "pr2879" in source_lower:
        source_tag = "PR2879"
    elif "w&b matrix" in source_lower:
        source_tag = "W&B"
    elif "math-rl" in source_lower:
        source_tag = "Math"
    else:
        source_tag = ""
    return " ".join(
        part for part in [cluster_label, date_label, nemorl_cuda_graph_label(row), source_tag] if part
    )


def nemorl_chart_model_order(models: list[str]) -> list[str]:
    preferred = ["Qwen3-235B-A22B", "Qwen3-30B-A3B", "Qwen3-32B", "Qwen3-8B"]
    present = list(dict.fromkeys(str(model) for model in models if str(model) and str(model) != "nan"))
    ordered = [model for model in preferred if model in present]
    ordered.extend(model for model in present if model not in ordered)
    return ordered


def nemorl_verified_eagle_k3_rows(rows: pd.DataFrame) -> pd.DataFrame:
    verified = nemorl_chart_rows(rows)
    if verified.empty:
        return verified
    verified = verified[verified["method_k"].astype(str).eq("eagle3_k3")].copy()
    if verified.empty:
        return verified
    return verified.sort_values(
        ["model_name", "mode", "max_new_tokens", "source_group", "job_id"],
        na_position="last",
    )


def nemorl_charts_section(rows: pd.DataFrame) -> str:
    chart_rows = nemorl_chart_rows(rows)
    if chart_rows.empty:
        return '<section><h2>Completed 20-Step Baseline-Relative Charts</h2><p class="note">No completed step20 timing windows are available yet for charting.</p></section>'
    metric_specs = [
        ("Generation Throughput Speedup", "gen_tps_speedup", "Speedup vs baseline"),
        ("E2E Throughput Speedup", "e2e_tps_speedup", "Speedup vs baseline"),
        ("Generation Step-Time Speedup", "generation_time_speedup", "Baseline time / run time"),
        ("E2E Step-Time Speedup", "e2e_step_time_speedup", "Baseline time / run time"),
    ]
    model_sections = []
    for model in nemorl_chart_model_order(chart_rows["model_name"].astype(str).tolist()):
        sub = chart_rows[chart_rows["model_name"].astype(str) == model].copy()
        if sub.empty:
            continue
        cards = [
            nemorl_multigroup_metric_svg(
                sub,
                title,
                metric,
                y_label,
                include_model_in_group=False,
                max_groups=6,
            )
            for title, metric, y_label in metric_specs
        ]
        rendered = "".join(f'<div class="chart-card">{card}</div>' for card in cards if card)
        if not rendered:
            continue
        model_sections.append(
            f'<h3>{esc(model)}</h3>'
            '<p class="note">Within this model, x-axis groups are matched setup slices: mode, max OSL, and cluster/source. Method colors compare against the matched baseline inside each slice.</p>'
            f'<div class="model-charts">{rendered}</div>'
        )
    return (
        '<section><h2>Completed 20-Step Baseline-Relative Charts</h2>'
        '<p class="note">Charts include only completed 20-step jobs: either all 20 metrics or the steady-state steps 2-20 after excluding cold-start step 1. Baselines are matched by model, mode, max OSL, temperature/top_p, CUDA Graph state, and source setup. Partial jobs remain in the tables but are excluded here.</p>'
        + "".join(model_sections)
        + '</section>'
    )


def nemorl_group_label(row: pd.Series, *, include_model: bool = True) -> str:
    model = short_model(row.get("model_name", row.get("model", ""))) if include_model else ""
    mode = str(row.get("mode", "") or "sync")
    osl = clean_float(row.get("max_new_tokens"))
    osl_label = f"OSL{int(osl)}" if not math.isnan(osl) else "OSL?"
    return "\n".join(part for part in [model, mode, osl_label, nemorl_setup_label(row)] if part)


def svg_multiline_text(x: float, y: float, lines: list[str], *, size: int = 12, anchor: str = "middle") -> str:
    tspans = []
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else size + 2
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="#111827">' + "".join(tspans) + "</text>"


def nemorl_method_order(methods: list[str]) -> list[str]:
    preferred = [
        "baseline",
        "baseline_fuselossfalse",
        "eagle3_k3",
        "eagle3_k5",
        "eagle3_k7",
        "eagle3_k9",
        "suffix_k32",
        "pard_k1_tp1",
        "pard_k1_tp2",
        "pard_k5",
        "pard_k8",
        "pard_k12",
        "pard_k16",
        "pard2",
        "pard2_k5",
        "pard2_k16",
        "pard2_8b",
        "pard2_14b",
    ]
    present = list(dict.fromkeys(str(method) for method in methods if str(method) and str(method) != "nan"))
    ordered = [method for method in preferred if method in present]
    ordered.extend(method for method in present if method not in ordered)
    return ordered


def nemorl_multigroup_metric_svg(
    rows: pd.DataFrame,
    title: str,
    metric: str,
    y_label: str,
    *,
    reference_line: bool = True,
    max_groups: int = 10,
    include_model_in_group: bool = True,
    max_methods: int = 8,
) -> str:
    if rows.empty:
        return ""
    rows = rows.copy()
    rows[metric] = pd.to_numeric(rows.get(metric), errors="coerce")
    rows = rows.dropna(subset=[metric])
    rows = rows[pd.to_numeric(rows.get("max_steps"), errors="coerce") == 20]
    rows = rows[rows.apply(nemorl_has_complete_step20_window, axis=1)]
    if rows.empty:
        return ""
    rows["group_label"] = rows.apply(lambda row: nemorl_group_label(row, include_model=include_model_in_group), axis=1)
    rows["source_rank"] = rows["source_group"].astype(str).map(
        lambda value: 0 if "Qwen235B" in value else 1 if "Lyris" in value else 2
    )
    rows = rows.sort_values(["source_rank", "model_name", "mode", "max_new_tokens", "completed_steps"])
    group_labels = list(dict.fromkeys(rows["group_label"].astype(str).tolist()))[:max_groups]
    rows = rows[rows["group_label"].isin(group_labels)]
    methods = nemorl_method_order(rows["method_k"].astype(str).tolist())
    if not group_labels or not methods:
        return ""
    if len(methods) > max_methods:
        always = [method for method in ["baseline", "baseline_fuselossfalse"] if method in methods]
        metric_rank = (
            rows.groupby("method_k")[metric]
            .mean()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        selected = list(dict.fromkeys(always + metric_rank))[:max_methods]
        methods = [method for method in methods if method in selected]
        rows = rows[rows["method_k"].astype(str).isin(methods)]

    max_value = clean_float(rows[metric].max())
    if math.isnan(max_value) or max_value <= 0:
        return ""
    y_max = max(1.15 if reference_line else 0.1, max_value * 1.18)
    legend_cols = min(4, len(methods))
    legend_rows = math.ceil(len(methods) / legend_cols)
    width = max(820, 110 + 108 * len(group_labels))
    height = 338 + max(0, legend_rows - 1) * 22
    left, right, top, bottom = 62, 22, 68 + max(0, legend_rows - 1) * 22, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / max(1, len(group_labels))
    inner = min(98, group_w * 0.84)
    bar_gap = 2.5
    bar_w = max(6, (inner - bar_gap * (len(methods) - 1)) / len(methods))

    def y_for(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    lookup: dict[tuple[str, str], float] = {}
    for _, row in rows.iterrows():
        lookup[(str(row["group_label"]), str(row["method_k"]))] = clean_float(row.get(metric))

    grid = []
    for frac in [0, 0.5, 1.0]:
        value = y_max * frac
        y = y_for(value)
        label = f"{value:.1f}x" if reference_line else f"{value:.1f}"
        grid.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d1d5db" stroke-dasharray="6 6"/>'
            f'<text x="{left - 8}" y="{y + 5:.1f}" text-anchor="end" font-size="13" fill="#4b5563">{label}</text>'
        )

    baseline = ""
    if reference_line and y_max > 1:
        y = y_for(1.0)
        baseline = (
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#111827" stroke-dasharray="5 5" stroke-width="1.3"/>'
            f'<text x="{width - right - 82}" y="{y - 8:.1f}" font-size="12" fill="#111827">1.0x baseline</text>'
        )

    legend_cell_w = 150
    legend_total_w = legend_cols * legend_cell_w
    legend_start = max(left, (width - legend_total_w) / 2)
    legend_parts = []
    for idx, method in enumerate(methods):
        x = legend_start + (idx % legend_cols) * legend_cell_w
        y = 35 + (idx // legend_cols) * 22
        color = PALETTE.get(method, "#4b5563")
        legend_parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="13" height="13" rx="2" fill="{color}" stroke="#192133" stroke-width="1.3"/>'
            f'<text x="{x + 19:.1f}" y="{y + 11.5:.1f}" font-size="12.5" fill="#111827">{esc(nemorl_method_label(method))}</text>'
        )

    bars = []
    show_bar_labels = bar_w >= 11 and len(group_labels) * len(methods) <= 42
    for group_idx, label in enumerate(group_labels):
        gx = left + group_idx * group_w + group_w / 2
        bars.append(svg_multiline_text(gx, height - 58, label.split("\n"), size=12))
        for method_idx, method in enumerate(methods):
            value = lookup.get((label, method), math.nan)
            if math.isnan(value):
                continue
            x = left + group_idx * group_w + (group_w - inner) / 2 + method_idx * (bar_w + bar_gap)
            y = y_for(value)
            color = PALETTE.get(method, "#4b5563")
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" rx="2.5" fill="{color}" stroke="#192133" stroke-width="1.2">'
                f'<title>{esc(nemorl_method_label(method))}: {value:.2f}x</title></rect>'
                + (
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="10.5" fill="#111827">{value:.2f}x</text>'
                    if show_bar_labels
                    else ""
                )
            )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
        f'<text x="{width / 2}" y="23" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{esc(title)}</text>'
        f'{"".join(legend_parts)}'
        f'<text x="17" y="{top + plot_h / 2:.1f}" transform="rotate(-90 17 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13" fill="#111827">{esc(y_label)}</text>'
        f'{"".join(grid)}{baseline}'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>'
        f'{"".join(bars)}</svg>'
    )


def charts_section(added: pd.DataFrame) -> str:
    if added.empty:
        return ""
    valid = added[added["valid_result"]].copy()
    if valid.empty:
        return ""
    focus_methods = ["eagle3_k8", "pard_k16", "pard2_k16"]
    cards = []
    for temp in [0.0, 1.0]:
        sub = valid[(valid["domain"] == "Math") & (valid["temperature"] == temp) & valid["method"].isin(focus_methods)]
        cards.append(grouped_bar_svg(sub, f"Math Temp {temp:.1f} Speedup", "speedup", focus_methods))
        cards.append(grouped_bar_svg(sub, f"Math Temp {temp:.1f} Acceptance", "acceptance_pct", focus_methods))
        cards.append(grouped_bar_svg(sub, f"Math Temp {temp:.1f} Mean Accepted Length", "mean_accept_len", focus_methods))
    for temp in [0.0, 1.0]:
        sub = valid[
            (valid["domain"] == "Math")
            & (valid["temperature"] == temp)
            & (valid["model"] == "Qwen3-235B-A22B")
            & valid["method"].isin(focus_methods)
        ].copy()
        cards.append(line_svg(sub, f"Qwen3-235B Math Temp {temp:.1f}: Speedup vs Batch", "speedup", "batch_size", "method"))
        cards.append(line_svg(sub, f"Qwen3-235B Math Temp {temp:.1f}: Mean Accepted Length vs Batch", "mean_accept_len", "batch_size", "method"))
    pard2 = valid[
        (valid["model"] == "Qwen3-235B-A22B")
        & (valid["method"].astype(str).str.startswith("pard2_k"))
    ].copy()
    if not pard2.empty:
        pard2["k"] = pard2["method"].astype(str).str.extract(r"k(\d+)").astype(float)
        for domain in ["Math", "SWE"]:
            sub = pard2[pard2["domain"] == domain].copy()
            if sub.empty:
                continue
            sub["series"] = sub["temperature"].map(lambda v: f"temp{float(v):.0f}")
            summary = sub.groupby(["k", "series"], as_index=False).agg(
                speedup=("speedup", "mean"),
                acceptance_pct=("acceptance_pct", "mean"),
                mean_accept_len=("mean_accept_len", "mean"),
            )
            cards.append(line_svg(summary, f"Qwen3-235B {domain} PARD-2 K Sweep Speedup", "speedup", "k", "series"))
            cards.append(line_svg(summary, f"Qwen3-235B {domain} PARD-2 K Sweep Acceptance", "acceptance_pct", "k", "series"))
            cards.append(line_svg(summary, f"Qwen3-235B {domain} PARD-2 K Sweep Mean Accepted Length", "mean_accept_len", "k", "series"))
    cards = [card for card in cards if card]
    if not cards:
        return ""
    return (
        '<section class="section"><h2>Visual Summary</h2>'
        '<p class="note">Charts use matched-baseline speedups and average repeated batch rows where needed. Legends are centered; tables below keep exact row provenance.</p>'
        '<div class="charts">'
        + "".join(f'<div class="chart-card">{card}</div>' for card in cards)
        + "</div></section>"
    )


def esc(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return html.escape(str(value), quote=True)


def text_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def first_text(row: pd.Series, *keys: str) -> str:
    for key in keys:
        value = text_value(row.get(key, ""))
        if value:
            return value
    return ""


def normalize_wandb_url(value: object) -> str:
    text = text_value(value)
    if not text:
        return ""
    match = WANDB_URL_RE.search(text)
    if match:
        return match.group(0).rstrip(".,)")
    return text if text.startswith(("http://", "https://")) else ""


def link_html(value: object, label: str = "W&B") -> str:
    url = normalize_wandb_url(value)
    if not url:
        return ""
    return f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(label)}</a>'


def published_data_html(value: object) -> str:
    text = text_value(value)
    if not text:
        return ""
    name = Path(text).name
    if name and (ROOT / "public" / "data" / name).exists():
        return f'<a href="../data/{esc(name)}"><code>{esc(name)}</code></a>'
    if text.startswith("docs/") and name:
        return f"<code>{esc(name)}</code>"
    return esc(text)


def local_result_link_html(value: object, source: object) -> str:
    label = text_value(value)
    path = text_value(source)
    if not label or not path:
        return esc(label)
    href = path if path.startswith(("./", "../")) else f"../{path}"
    return f'<a href="{esc(href)}"><code>{esc(label)}</code></a>'


def wandb_link_html(row: pd.Series) -> str:
    direct_url = normalize_wandb_url(row.get("wandb_url", ""))
    if direct_url:
        return link_html(direct_url, "run")
    job_id = text_value(row.get("job_id", ""))
    enabled = text_value(row.get("wandb_enabled", "")).lower()
    if job_id in NEMORL_CONFIRMED_WANDB_DISABLED_JOBS or enabled in {"false", "0", "no"}:
        return '<span class="not-logged" title="logger.wandb_enabled=false">not logged</span>'
    project = first_text(row, "wandb_project")
    if not project:
        return '<span class="not-logged">not available</span>'
    return link_html(f"https://wandb.ai/{WANDB_ENTITY}/{project}", "project")


def clean_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return math.nan
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    value = clean_float(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def fmt_x(value: object) -> str:
    return fmt(value, 2, "x")


def fmt_pct(value: object) -> str:
    return fmt(value, 1, "%")


def fmt_pct_2dp(value: object) -> str:
    return fmt(value, 2, "%")


def fmt_ratio_pct(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return fmt(float(value) * 100.0, 2, "%")


def model_name(value: object) -> str:
    text = str(value)
    lower = text.lower()
    for key, name in MODEL_MAP.items():
        if key in lower:
            return name
    if "235b" in lower:
        return "Qwen3-235B-A22B"
    if "30b" in lower:
        return "Qwen3-30B-A3B"
    if "32b" in lower:
        return "Qwen3-32B"
    if "8b" in lower:
        return "Qwen3-8B"
    return text


def method_with_k(method: object, k: object) -> str:
    method_text = str(method).strip()
    if not method_text or method_text == "baseline":
        return method_text or "baseline"
    k_value = clean_float(k)
    if math.isnan(k_value):
        return method_text
    return f"{method_text}_k{int(k_value)}"


def refine_nemorl_method_from_run(method_k: object, run_id: object) -> str:
    method = str(method_k)
    run = str(run_id).lower()
    if method == "pard_k1":
        if "drafttp1_targettp1" in run:
            return "pard_k1_tp1"
        if "pardk1" in run:
            return "pard_k1_tp2"
    return method


def parse_completed_last(value: object) -> tuple[float, float]:
    match = re.search(r"(\d+)\s*/\s*(\d+)", str(value))
    if not match:
        return math.nan, math.nan
    return float(match.group(1)), float(match.group(2))


def normalize_nemorl_method(method: object, label: object = "", k: object = math.nan) -> str:
    method_text = str(method).strip()
    label_text = str(label).strip()
    lower = f"{method_text} {label_text}".lower()
    k_value = clean_float(k)
    if "baseline" in lower:
        if "fuse_loss=false" in lower or "fuselossfalse" in lower:
            return "baseline_fuselossfalse"
        return "baseline"
    if "eagle" in lower:
        return f"eagle3_k{int(k_value)}" if not math.isnan(k_value) and k_value > 0 else "eagle3_k3"
    if "suffix" in lower:
        return f"suffix_k{int(k_value)}" if not math.isnan(k_value) and k_value > 0 else "suffix_k32"
    if "pard-2" in lower or "pard2" in lower:
        if not math.isnan(k_value) and k_value > 0:
            return f"pard2_k{int(k_value)}"
        if "14b" in lower:
            return "pard2_14b"
        if "8b" in lower:
            return "pard2_8b"
        return "pard2"
    if "pard" in lower:
        if not math.isnan(k_value) and k_value > 0:
            return f"pard_k{int(k_value)}"
        match = re.search(r"k[=_-]?(\d+)", lower)
        return f"pard_k{match.group(1)}" if match else "pard"
    return method_text.lower().replace(" ", "_")


def normalize_nemorl_diagnostic_method(method: object) -> str:
    text = str(method).strip()
    lower = text.lower()
    base = normalize_nemorl_method(text)
    if base == "pard_k1":
        if "tp1" in lower:
            return "pard_k1_tp1"
        if "tp2" in lower:
            return "pard_k1_tp2"
    return base


def effective_metric(row: pd.Series, final_col: str, live_col: str) -> float:
    final = clean_float(row.get(final_col))
    if not math.isnan(final):
        return final
    return clean_float(row.get(live_col))


def baseline_lookup(main: pd.DataFrame) -> dict[tuple[object, ...], float]:
    lookup: dict[tuple[object, ...], float] = {}
    baselines = main[main["method"].astype(str) == "baseline"]
    for _, row in baselines.iterrows():
        key = (
            str(row["domain"]),
            str(row["model"]),
            float(row["temperature"]),
            int(row["batch_size"]),
            int(row["isl"]),
            int(row["osl"]),
        )
        lookup[key] = float(row["tok_s_gpu"])
    return lookup


def vllm_baseline_key(row: pd.Series) -> tuple[object, ...] | None:
    try:
        return (
            str(row["domain"]),
            str(row["model"]),
            float(row["temperature"]),
            int(clean_float(row["batch_size"])),
            int(clean_float(row["isl"])),
            int(clean_float(row["osl"])),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def fill_vllm_added_speedups(main: pd.DataFrame, added: pd.DataFrame) -> pd.DataFrame:
    if added.empty:
        return added
    added = added.copy()
    baselines = baseline_lookup(main)
    valid_baselines = added[(added["method"].astype(str) == "baseline") & (added["valid_result"])]
    for _, row in valid_baselines.iterrows():
        key = vllm_baseline_key(row)
        tok = clean_float(row.get("tok_s_gpu"))
        if key is not None and not math.isnan(tok):
            baselines.setdefault(key, tok)
    for idx, row in added.iterrows():
        key = vllm_baseline_key(row)
        if key is None:
            continue
        baseline = baselines.get(key, math.nan)
        tok = clean_float(row.get("tok_s_gpu"))
        if str(row.get("method")) == "baseline" and not math.isnan(tok):
            added.at[idx, "baseline_tok_s_gpu"] = tok
            added.at[idx, "speedup"] = 1.0
            continue
        if math.isnan(clean_float(row.get("baseline_tok_s_gpu"))) and not math.isnan(baseline):
            added.at[idx, "baseline_tok_s_gpu"] = baseline
        if math.isnan(clean_float(row.get("speedup"))) and not math.isnan(tok) and not math.isnan(baseline) and baseline:
            added.at[idx, "speedup"] = tok / baseline
    return added


def load_vllm_added(main: pd.DataFrame) -> pd.DataFrame:
    baselines = baseline_lookup(main)
    parts: list[pd.DataFrame] = []
    for path, source_label, priority in VLLM_LIVE_SOURCES:
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        rows = []
        for _, row in raw.iterrows():
            domain = "Math" if str(row.get("domain", "")).lower() == "math" else "SWE"
            model = model_name(row.get("model_group", ""))
            temperature = clean_float(row.get("temperature"))
            batch = int(clean_float(row.get("batch_size")))
            isl = int(clean_float(row.get("isl")))
            osl = int(clean_float(row.get("osl")))
            method = method_with_k(row.get("method"), row.get("k"))
            tok = effective_metric(row, "final_tok_s_gpu", "live_tok_s_gpu_approx")
            acceptance = effective_metric(row, "final_acceptance_pct", "live_acceptance_pct")
            mean_len = effective_metric(row, "final_mean_accept_len", "live_mean_accept_len")
            baseline = baselines.get((domain, model, temperature, batch, isl, osl), math.nan)
            speedup = tok / baseline if not math.isnan(tok) and not math.isnan(baseline) and baseline else math.nan
            rows.append(
                {
                    "domain": domain,
                    "model": model,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "batch_size": batch,
                    "isl": isl,
                    "osl": osl,
                    "method": method,
                    "job_id": str(row.get("job_id", "")),
                    "state": str(row.get("state", "")),
                    "tok_s_gpu": tok,
                    "baseline_tok_s_gpu": baseline,
                    "speedup": speedup,
                    "acceptance_pct": acceptance,
                    "mean_accept_len": mean_len,
                    "basis": "final breakdown" if str(row.get("breakdown_valid", "")) == "1" else "live log",
                    "source": str(path.relative_to(ROOT)),
                    "source_label": source_label,
                    "source_priority": priority,
                    "logs_dir": str(row.get("logs_dir", "")),
                    "valid_result": str(row.get("state", "")) == "COMPLETED" and not math.isnan(tok),
                }
            )
        parts.append(pd.DataFrame(rows))
    if VLLM_LEGACY_NORMALIZED.exists():
        legacy = pd.read_csv(VLLM_LEGACY_NORMALIZED)
        parts.append(legacy)
    if DFLASH.exists():
        dflash = pd.read_csv(DFLASH)
        rows = []
        for _, row in dflash.iterrows():
            rows.append(
                {
                    "domain": "Math",
                    "model": "Qwen3-235B-A22B",
                    "temperature": math.nan,
                    "top_p": math.nan,
                    "batch_size": int(clean_float(row.get("batch_size"))),
                    "isl": math.nan,
                    "osl": math.nan,
                    "method": str(row.get("method", "")).lower().replace(" ", "_"),
                    "job_id": "",
                    "state": "COMPLETED",
                    "tok_s_gpu": clean_float(row.get("output_tok_s_per_gpu")),
                    "baseline_tok_s_gpu": math.nan,
                    "speedup": math.nan,
                    "acceptance_pct": clean_float(row.get("acceptance_pct")),
                    "mean_accept_len": clean_float(row.get("mean_acceptance_length")),
                    "basis": "legacy DFlash OpenMath; no matched OSL32K baseline",
                    "source": str(DFLASH.relative_to(ROOT)),
                    "source_label": "DFlash OpenMath retry28",
                    "source_priority": 5,
                    "logs_dir": "",
                    "valid_result": True,
                }
            )
        parts.append(pd.DataFrame(rows))
    if not parts and VLLM_ADDED_INPUT.exists():
        parts.append(pd.read_csv(VLLM_ADDED_INPUT))
    if not parts:
        return pd.DataFrame()
    added = pd.concat(parts, ignore_index=True)
    added["valid_result"] = added["valid_result"].astype(str).str.lower().isin({"1", "true", "yes"})
    added = fill_vllm_added_speedups(main, added)
    added = added.sort_values(
        [
            "domain",
            "model",
            "temperature",
            "batch_size",
            "method",
            "source_priority",
            "valid_result",
        ],
        na_position="last",
    )
    key = ["domain", "model", "temperature", "batch_size", "isl", "osl", "method"]
    added = added.groupby(key, dropna=False, as_index=False).tail(1).copy()
    added = fill_vllm_added_speedups(main, added)
    added = added.sort_values(["domain", "temperature", "model", "method", "batch_size"], na_position="last")
    return added


def aggregate_added(added: pd.DataFrame) -> pd.DataFrame:
    if added.empty:
        return added
    valid = added[added["valid_result"]].copy()
    if valid.empty:
        return valid
    grouped = (
        valid.groupby(["domain", "temperature", "model", "method", "source_label"], dropna=False)
        .agg(
            rows=("job_id", "count"),
            batches=("batch_size", lambda s: "/".join(str(int(v)) for v in sorted(pd.to_numeric(s, errors="coerce").dropna()))),
            isl=("isl", "first"),
            osl=("osl", "first"),
            tok_s_gpu=("tok_s_gpu", "mean"),
            speedup=("speedup", "mean"),
            acceptance_pct=("acceptance_pct", "mean"),
            mean_accept_len=("mean_accept_len", "mean"),
            basis=("basis", "first"),
            source=("source", "first"),
        )
        .reset_index()
    )
    return grouped.sort_values(["domain", "temperature", "model", "method"], na_position="last")


def matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, temp, model, method), group in df.groupby(["domain", "temperature", "model", "method"], dropna=False):
        row = {
            "domain": domain,
            "temperature": temp,
            "model": model,
            "method": method,
        }
        for batch in [1, 2, 4, 8, 16, 32]:
            values = group[group["batch_size"] == batch]["speedup"].dropna()
            row[f"b{batch}_speedup"] = float(values.iloc[-1]) if len(values) else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["domain", "temperature", "model", "method"], na_position="last")


def temp_trends_section() -> str:
    if not VLLM_TEMP_TRENDS.exists():
        return ""
    rows = pd.read_csv(VLLM_TEMP_TRENDS)
    if rows.empty:
        return ""
    rows = rows.copy()
    rows = rows.rename(
        columns={
            "mean_speedup_vs_baseline": "mean_speedup",
            "mean_tok_s_per_gpu": "mean_tok_s_gpu",
            "mean_acceptance_pct": "mean_acceptance",
            "mean_acceptance_length": "mean_accept_len",
        }
    )
    rows = rows.sort_values(["domain", "model", "temperature", "method"], na_position="last")
    return (
        '<section class="section"><h2>Historical Temp0/Temp1 Trend Summary</h2>'
        '<p class="note">This preserves the older extensive Math/SWE temperature analysis page. It is an aggregate view; exact batch-level rows are reflected in the detailed sections below when the underlying CSV or breakdown JSON exists.</p>'
        '<div class="table-wrap">'
        + table(
            rows,
            [
                ("domain", "Domain", "text"),
                ("dataset", "Dataset", "text"),
                ("model", "Model", "text"),
                ("temperature", "Temp", "text"),
                ("method", "Method", "text"),
                ("rows", "Rows", "int"),
                ("mean_speedup", "Mean speedup", "x"),
                ("min_speedup", "Min", "x"),
                ("max_speedup", "Max", "x"),
                ("mean_tok_s_gpu", "tok/s/GPU", "num"),
                ("mean_acceptance", "Acceptance", "pct"),
                ("mean_accept_len", "Mean len", "num"),
                ("basis", "Basis", "text"),
                ("source", "Source", "text"),
            ],
        )
        + "</div></section>"
    )


def related_vllm_reports_section() -> str:
    reports = [
        (
            "Temp0/Temp1 Trend Page",
            "Math/SWE temperature 0 vs 1 aggregate trends and key interpretation.",
            "vllm_standalone_temp0_temp1_trends_20260616.html",
        ),
        (
            "Broad SpecDec Dashboard",
            "Older wide dashboard with vLLM standalone, SWE/Math snapshots, and status fragments.",
            "specdec_benchmark_metrics_dashboard_20260616.html",
        ),
        (
            "Clean Primary Results",
            "Curated 2026-06-17 vLLM standalone primary/supplemental split.",
            "vllm_standalone_clean_results_20260617.html",
        ),
        (
            "6/19 Batch Matrix",
            "Earlier all-batch report before the latest legacy-source refresh.",
            "vllm_standalone_results_20260619.html",
        ),
        (
            "Qwen235B SWE Batch Sweep",
            "Dedicated Qwen3-235B SWE OSL32K batch-sweep speedup page.",
            "lyris_qwen235b_swebench_osl32k_batch_sweep_speedups_20260612.html",
        ),
        (
            "Qwen235B Diagnostics",
            "Live diagnostic page from the older Qwen3-235B standalone runs.",
            "lyris_qwen235b_standalone_live_diagnostics_20260613.html",
        ),
    ]
    items = []
    for title, desc, href in reports:
        if not (DOCS / href).exists():
            continue
        items.append(
            '<div class="card">'
            f'<b><a href="{esc(href)}">{esc(title)}</a></b>'
            f'<span>{esc(desc)}</span>'
            f'<code>{esc(href)}</code>'
            '</div>'
        )
    if not items:
        return ""
    return (
        '<section class="section"><h2>Related Broader Reports</h2>'
        '<p class="note">This latest page is intentionally focused on matched ISL4096/OSL32768 comparisons. Use these archive pages for broader historical, long-OSL, partial, or aggregate views that are not all directly comparable in one speedup matrix.</p>'
        '<div class="cards">'
        + "".join(items)
        + "</div></section>"
    )


def table(rows: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    if rows.empty:
        return '<p class="note">No rows.</p>'
    head = "".join(f"<th>{esc(label)}</th>" for _, label, _ in columns)
    body = []
    text_classes = {
        "source_group": "source-col",
        "source_label": "source-col",
        "source": "source-col",
        "source_file": "source-col",
        "basis": "note-col",
        "manifest": "manifest-col",
        "wandb_name": "name-col",
        "latest_error": "note-col error-col",
        "notes": "note-col",
        "logs_dir": "path-col",
    }
    for _, row in rows.iterrows():
        cells = []
        for key, _, kind in columns:
            value = row.get(key, "")
            cls = "num" if kind in {"num", "x", "pct", "pct_2dp", "ratio_pct", "int", "temp"} else text_classes.get(key, "")
            if key == "slurm_state":
                cls = str(value).strip()
            if kind == "num":
                text = fmt(value, 2)
            elif kind == "int":
                text = "n/a" if pd.isna(value) else str(int(float(value)))
            elif kind == "x":
                text = fmt_x(value)
            elif kind == "pct":
                text = fmt_pct(value)
            elif kind == "pct_2dp":
                text = fmt_pct_2dp(value)
            elif kind == "ratio_pct":
                text = fmt_ratio_pct(value)
            elif kind == "temp":
                text = "n/a" if pd.isna(value) else f"{float(value):.1f}"
            elif kind == "link":
                text = wandb_link_html(row) if key == "wandb_url" else link_html(value)
            elif kind == "result_link":
                text = local_result_link_html(
                    value,
                    row.get("result_href", row.get("source", "")),
                )
            else:
                text = published_data_html(value) if key in {"source", "source_file", "manifest"} else esc(value)
            title = esc(value) if key in text_classes and text else ""
            attr = f' class="{cls}"' if cls else ""
            attr += f' title="{title}"' if title else ""
            cells.append(f"<td{attr}>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table>"


def shell_assignment(path: Path, name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}=\"([^\"]+)\"",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else "unknown"


def _method_variant_label(method: str, variant: str) -> str:
    labels = {
        ("baseline", "baseline"): "Baseline",
        ("eagle3", "static"): "Eagle-3 static",
        ("eagle3", "dynamic"): "DynamicSD",
        ("mtp_static", "mtp_static"): "native MTP static",
        ("mtp_dynamic", "mtp_dynamic"): "native MTP dynamic",
    }
    return labels.get((method, variant), variant.replace("_", " "))


def _profile_summary(profile: dict[str, object]) -> str:
    key = str(profile.get("key", "")).upper()
    policy = str(profile.get("context_policy", ""))
    if policy == "native_32k":
        label = f"{key} native"
    elif policy == "yarn4_64k":
        label = f"{key} YaRN-4"
    else:
        label = key
    max_new_tokens = profile.get("max_new_tokens")
    return f"{label} (OSL {max_new_tokens})" if max_new_tokens else label


def load_speedbench_runner_capabilities() -> dict[str, tuple[str, ...]]:
    module_name = "_vllm024_speedbench_report_capabilities"
    spec = importlib.util.spec_from_file_location(module_name, SPEEDBENCH_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SPEED-Bench runner: {SPEEDBENCH_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    experiment_path = str(SYNC_RL_EXPERIMENT_ROOT)
    inserted_path = experiment_path not in sys.path
    if inserted_path:
        sys.path.insert(0, experiment_path)
    try:
        spec.loader.exec_module(module)
        raw_capabilities = module.speedbench_runner_capabilities()
    finally:
        sys.modules.pop(module_name, None)
        if inserted_path:
            sys.path.remove(experiment_path)
    if not isinstance(raw_capabilities, dict):
        raise TypeError("SPEED-Bench runner capabilities must be a mapping")
    capabilities: dict[str, tuple[str, ...]] = {}
    for cohort in ("official", "overlay"):
        modes = raw_capabilities.get(cohort)
        if not isinstance(modes, tuple) or not modes:
            raise TypeError(f"SPEED-Bench {cohort} capabilities must be a tuple")
        capabilities[cohort] = tuple(str(mode) for mode in modes)
    return capabilities


def load_sync_speedbench_support() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SYNC_RL_MODEL_MATRIX.exists():
        return pd.DataFrame(), pd.DataFrame()
    matrix = json.loads(SYNC_RL_MODEL_MATRIX.read_text(encoding="utf-8"))
    runner_capabilities = load_speedbench_runner_capabilities()
    official_modes = set(runner_capabilities["official"])
    overlay_modes = set(runner_capabilities["overlay"])
    official_launcher_modes = {"baseline", "static"}
    qwen_rows: list[dict[str, str]] = []
    nemotron_rows: list[dict[str, str]] = []
    for model in matrix.get("models", []):
        methods = model.get("methods", {})
        supported: list[str] = []
        integration: list[str] = []
        unsupported: list[str] = []
        supported_variants: list[tuple[str, str]] = []
        for method in matrix.get("method_order", []):
            meta = methods.get(method)
            if not isinstance(meta, dict):
                continue
            status = meta.get("status")
            variants = meta.get("variants") or []
            if status == "supported":
                if variants:
                    supported_variants.extend(
                        (str(method), str(variant)) for variant in variants
                    )
                    supported.extend(
                        _method_variant_label(str(method), str(variant))
                        for variant in variants
                    )
                else:
                    supported.append(str(method))
            elif status == "integration":
                integration.append(str(method).upper().replace("_", "-"))
            elif status == "unsupported":
                unsupported.append(str(method).upper().replace("_", "-"))
        row = {
            "model": str(model.get("label", "")),
            "profiles": ", ".join(
                _profile_summary(profile)
                for profile in model.get("profiles", [])
                if isinstance(profile, dict)
            ),
            "supported": ", ".join(supported),
            "integration": ", ".join(integration) if integration else "none",
            "unsupported": ", ".join(unsupported) if unsupported else "none",
        }
        launcher = model.get("launcher")
        if launcher == "swe_sync_rollout":
            qwen_rows.append(row)
        elif launcher == "nemotron_sync_rl_mtp":
            row["profiles"] = "official/overlay SPEED-Bench only"
            row["official_support"] = ", ".join(
                _method_variant_label(method, variant)
                for method, variant in supported_variants
                if variant in official_modes and variant in official_launcher_modes
            )
            row["overlay_support"] = ", ".join(
                _method_variant_label(method, variant)
                for method, variant in supported_variants
                if variant in overlay_modes
            )
            official_limitations = []
            for method, variant in supported_variants:
                label = _method_variant_label(method, variant)
                if variant in official_modes and variant not in official_launcher_modes:
                    official_limitations.append(
                        f"{label}: low-level runner capability only; "
                        "no official Nemotron MTP launcher"
                    )
                elif variant not in official_modes:
                    official_limitations.append(f"{label} unsupported")
            row["official_limitations"] = (
                ", ".join(official_limitations) if official_limitations else "none"
            )
            dynamic_overlay_labels = [
                _method_variant_label(method, variant)
                for method, variant in supported_variants
                if variant in {"dynamic", "mtp_dynamic"} and variant in overlay_modes
            ]
            row["overlay_gates"] = (
                ", ".join(dynamic_overlay_labels)
                + ": signed model/profile calibration artifact required; excluded from smoke"
                if dynamic_overlay_labels
                else "none"
            )
            nemotron_rows.append(row)
    return pd.DataFrame(qwen_rows), pd.DataFrame(nemotron_rows)


def load_completed_qwen32_math_dynamic_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, path in SYNC_RL_SUMMARY_FILES.items():
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        dynamic = summary[summary["variant"] == "dynamic"]
        if dynamic.empty:
            continue
        row = dynamic.iloc[0]
        rows.append(
            {
                "result_scope": "Qwen3-32B Math DynamicSD",
                "dataset": dataset,
                "sampling": (
                    f"temp {float(row['temperature']):.1f} / top_p {float(row['top_p']):.1f}"
                ),
                "tok_s_gpu": float(row["output_tok_s_per_gpu"]),
                "speedup_vs_baseline": float(row["throughput_speedup_vs_baseline"]),
                "speedup_vs_static": float(row["throughput_speedup_vs_static"]),
                "time_reduction_vs_baseline_pct": float(
                    row["rollout_time_reduction_vs_baseline_pct"]
                ),
                "acceptance_rate": float(row["acceptance_rate"]),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def load_perfcfg_dynamic_replay_rows() -> pd.DataFrame:
    if not PERFCFG_DYNAMIC_REPLAY_CSV.exists():
        return pd.DataFrame()
    rows = pd.read_csv(PERFCFG_DYNAMIC_REPLAY_CSV)
    rows["baseline_job"] = rows["baseline_job"].astype(str)
    rows["dynamic_job"] = rows["dynamic_job"].astype(str)
    return rows


def _nemotron_smoke_model_label(model_path: object) -> str:
    model = text_value(model_path)
    if "NVIDIA-Nemotron-3-Super-120B-A12B-BF16" in model:
        return "Nemotron-3-Super-120B-A12B-BF16"
    if "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" in model:
        return "Nemotron-3-Ultra-550B-A55B-BF16"
    return Path(model).name


def _nemotron_smoke_metadata_value(
    payload: dict[str, object],
    field_path: tuple[str, ...],
) -> tuple[bool, object]:
    value: object = payload
    for key in field_path:
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _validate_nemotron_smoke_payload(
    payload: dict[str, object],
    relative_path: Path,
    *,
    expected_mode: str,
    expected_model: str,
    expected_job_id: str,
) -> None:
    expected_metadata = list(NEMOTRON_MTP_LEGACY_SMOKE_SHARED_METADATA)
    expected_metadata.extend(
        [
            (("config", "mode"), expected_mode),
            (("config", "model"), expected_model),
            (("runtime", "environment", "SLURM_JOB_ID"), expected_job_id),
        ]
    )
    mismatches: list[str] = []
    for field_path, expected in expected_metadata:
        found, actual = _nemotron_smoke_metadata_value(payload, field_path)
        field_name = ".".join(field_path)
        if not found or actual != expected:
            actual_display = repr(actual) if found else "<missing>"
            mismatches.append(
                f"{field_name}: expected {expected!r}, got {actual_display}"
            )
    sha_found, runtime_image_sha256 = _nemotron_smoke_metadata_value(
        payload,
        ("config", "runtime_image_sha256"),
    )
    if sha_found and runtime_image_sha256 not in (None, ""):
        mismatches.append(
            "config.runtime_image_sha256: expected missing or empty, "
            f"got {runtime_image_sha256!r}"
        )
    if mismatches:
        raise ValueError(
            f"Nemotron MTP legacy smoke payload {relative_path.as_posix()} "
            f"does not match the expected cohort: {'; '.join(mismatches)}"
        )


def _nemotron_smoke_schedule(config: dict[str, object]) -> str:
    mode = text_value(config.get("mode"))
    speculative = config.get("speculative_config")
    if not isinstance(speculative, dict):
        return "n/a"
    static_k = int(speculative["num_speculative_tokens"])
    if mode == "mtp_static":
        return f"K={static_k}"
    raw_schedule = speculative.get("num_speculative_tokens_per_batch_size", [])
    schedule = [
        f"batch {int(start)}-{int(end)}: K={int(k)}"
        for start, end, k in raw_schedule
    ]
    return "; ".join(schedule) + " (uncalibrated)"


def load_nemotron_mtp_legacy_smoke_rows(
    result_root: Path = NEMOTRON_MTP_LEGACY_SMOKE_ROOT,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    method_labels = {
        "baseline": "Baseline",
        "mtp_static": "Native MTP static",
        "mtp_dynamic": "Native MTP dynamic",
    }
    expected_results = {
        (model_key, mode): (model, job_id)
        for model_key, mode, model, job_id in NEMOTRON_MTP_LEGACY_SMOKE_RESULTS
    }
    expected_paths = {
        Path(model_key) / mode / "result.json"
        for model_key, mode, _model, _job_id in NEMOTRON_MTP_LEGACY_SMOKE_RESULTS
    }
    discovered_paths = {
        path.relative_to(result_root)
        for path in result_root.glob("**/result.json")
        if path.is_file()
    }
    missing_paths = sorted(expected_paths - discovered_paths)
    unexpected_paths = sorted(discovered_paths - expected_paths)
    if missing_paths or unexpected_paths:
        details = []
        if missing_paths:
            missing = ", ".join(path.as_posix() for path in missing_paths)
            details.append(f"missing expected payloads: {missing}")
        if unexpected_paths:
            unexpected = ", ".join(path.as_posix() for path in unexpected_paths)
            details.append(f"unexpected payloads: {unexpected}")
        raise ValueError(
            "Nemotron MTP legacy smoke cohort must contain exactly the six expected "
            f"result.json payloads; {'; '.join(details)}"
        )
    for model_key in ("super", "ultra"):
        payloads: dict[str, tuple[dict[str, object], Path]] = {}
        for mode in method_labels:
            path = result_root / model_key / mode / "result.json"
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_payload, dict):
                relative_path = path.relative_to(result_root)
                raise ValueError(
                    "Nemotron MTP legacy smoke payload "
                    f"{relative_path.as_posix()} must be a JSON object"
                )
            payload = cast(dict[str, object], raw_payload)
            expected_model, expected_job_id = expected_results[(model_key, mode)]
            _validate_nemotron_smoke_payload(
                payload,
                path.relative_to(result_root),
                expected_mode=mode,
                expected_model=expected_model,
                expected_job_id=expected_job_id,
            )
            payloads[mode] = (payload, path)
        baseline_summary = cast(
            dict[str, object], payloads["baseline"][0]["summary"]
        )
        baseline_tokens = int(baseline_summary["total_output_tokens"])
        baseline_tok_s_gpu = float(baseline_summary["output_tok_s_per_gpu"])
        baseline_time_s = float(baseline_summary["total_rollout_time_s"])
        for mode, method in method_labels.items():
            payload, path = payloads[mode]
            config = cast(dict[str, object], payload["config"])
            runtime = cast(dict[str, object], payload["runtime"])
            summary = cast(dict[str, object], payload["summary"])
            output_tokens = int(summary["total_output_tokens"])
            output_tok_s_gpu = float(summary["output_tok_s_per_gpu"])
            rollout_time_s = float(summary["total_rollout_time_s"])
            output_token_ratio = output_tokens / baseline_tokens
            work_matches = _output_work_within_one_percent(
                output_tokens,
                baseline_tokens,
            )
            throughput_speedup = output_tok_s_gpu / baseline_tok_s_gpu
            rollout_speedup = baseline_time_s / rollout_time_s
            spec_metrics = cast(
                dict[str, object], summary.get("spec_decode_metrics", {})
            )
            environment = cast(dict[str, object], runtime.get("environment", {}))
            if mode == "baseline":
                throughput_display = "1.00x (reference)"
                rollout_display = "1.00x (reference)"
                validity = (
                    "legacy capability smoke; reference only; one measured realization; "
                    "runtime_image_sha256 missing/empty"
                )
            else:
                throughput_display = f"{throughput_speedup:.2f}x (directional only)"
                if work_matches:
                    rollout_display = f"{rollout_speedup:.2f}x (directional only)"
                else:
                    rollout_display = "n/a (invalid: output-token ratio outside 1%)"
                natural_eos = "; natural EOS" if model_key == "super" else ""
                validity = (
                    "legacy capability smoke; directional only"
                    f"{natural_eos}; one measured realization; "
                    "runtime_image_sha256 missing/empty"
                )
                if not work_matches:
                    validity += "; direct rollout-time comparison invalid"
            rows.append(
                {
                    "model": _nemotron_smoke_model_label(config.get("model")),
                    "method": method,
                    "job_id": text_value(environment.get("SLURM_JOB_ID")),
                    "output_tok_s_gpu": output_tok_s_gpu,
                    "throughput_speedup": throughput_display,
                    "rollout_speedup": rollout_display,
                    "output_token_ratio": (
                        f"{output_tokens}/{baseline_tokens} = "
                        f"{output_token_ratio:.4f}x"
                    ),
                    "acceptance_rate": (
                        f"{float(spec_metrics['acceptance_rate']) * 100.0:.2f}%"
                        if spec_metrics
                        else "n/a"
                    ),
                    "mean_acceptance_length": (
                        f"{float(spec_metrics['mean_acceptance_length']):.2f}"
                        if spec_metrics
                        else "n/a"
                    ),
                    "schedule": _nemotron_smoke_schedule(config),
                    "validity": validity,
                    "source": str(path.relative_to(ROOT)),
                    "result_relative_path": str(path.relative_to(result_root)),
                }
            )
    return pd.DataFrame(rows)


def publish_nemotron_mtp_legacy_smoke_evidence(
    *,
    result_root: Path = NEMOTRON_MTP_LEGACY_SMOKE_ROOT,
    public_data_dir: Path = PUBLIC_DATA,
) -> tuple[Path, ...]:
    load_nemotron_mtp_legacy_smoke_rows(result_root)
    destination_root = public_data_dir / result_root.name
    if destination_root.exists():
        shutil.rmtree(destination_root)
    published = []
    for model_key, mode, _model, _job_id in NEMOTRON_MTP_LEGACY_SMOKE_RESULTS:
        relative_path = Path(model_key) / mode / "result.json"
        destination = publish_public_data(
            result_root / relative_path,
            destination_root / relative_path.parent,
        )
        if destination is None:
            raise ValueError(
                "Nemotron MTP legacy smoke evidence disappeared after validation: "
                f"{relative_path.as_posix()}"
            )
        published.append(destination)
    return tuple(published)


def render_nemotron_mtp_legacy_smoke_section(
    evidence_href_root: str | None = None,
) -> str:
    rows = load_nemotron_mtp_legacy_smoke_rows()
    if evidence_href_root is not None:
        href_root = evidence_href_root.rstrip("/")
        rows["result_href"] = rows["result_relative_path"].map(
            lambda relative_path: f"{href_root}/{relative_path}"
        )
    return "".join(
        [
            '<section class="section"><h2>Nemotron Native MTP Legacy Smoke</h2>',
            '<p class="note">Every row is legacy capability smoke from vLLM 0.24.0 '
            "with CUDA Graph PIECEWISE, OSL/max_new_tokens 128, temperature 1.0, "
            "top_p 0.95, one measured realization, and runtime_image_sha256 "
            "missing/empty. "
            "Dynamic rows use uncalibrated dynamic schedules.</p>",
            '<p class="note">These rows are excluded from calibrated '
            "DynamicSD/DynamicMTP claims and existing validated matrices. "
            "Baseline-relative ratios are directional only because there is one "
            "measured realization; Super rows can also differ through natural EOS. "
            "Rollout-time speedup is shown only when aggregate output-token work is "
            "within 1% of the model baseline.</p>",
            '<div class="table-wrap">',
            table(
                rows,
                [
                    ("model", "Model", "text"),
                    ("method", "Method", "text"),
                    ("job_id", "Job ID", "result_link"),
                    ("output_tok_s_gpu", "Output tok/s/GPU", "num"),
                    (
                        "throughput_speedup",
                        "Baseline-relative throughput speedup",
                        "text",
                    ),
                    ("rollout_speedup", "Rollout-time speedup", "text"),
                    ("output_token_ratio", "Output-token ratio", "text"),
                    ("acceptance_rate", "Acceptance rate", "text"),
                    (
                        "mean_acceptance_length",
                        "Mean acceptance length",
                        "text",
                    ),
                    ("schedule", "Static K / dynamic schedule", "text"),
                    ("validity", "Validity", "text"),
                ],
            ),
            "</div></section>",
        ]
    )


def _validate_nemotron_mtp_k_sweep_payload(
    payload: dict[str, object],
    relative_path: Path,
    *,
    expected_model_key: str,
    expected_mode: str,
    expected_k: int,
    expected_model: str,
    expected_job_id: str,
    expected_tp: int,
    expected_nodes: int,
    expected_shared_metadata: tuple[
        tuple[tuple[str, ...], object], ...
    ] = NEMOTRON_MTP_K_SWEEP_SHARED_METADATA,
    expected_model_identities: Mapping[
        str, Mapping[str, object]
    ] = NEMOTRON_MTP_K_SWEEP_MODEL_IDENTITIES,
    expected_seed: int = NEMOTRON_MTP_K_SWEEP_SEED,
    expected_requests_per_batch: int = 32,
    expected_samples_per_prompt: int = 4,
    expected_max_tokens: int = 4096,
    cohort_label: str = "Nemotron MTP OSL 4K K-sweep",
) -> None:
    model_identity = expected_model_identities[expected_model_key]
    expected_backend = model_identity["distributed_executor_backend"]
    expected_metadata = list(expected_shared_metadata)
    expected_metadata.extend(
        [
            (("config", "mode"), expected_mode),
            (("config", "model"), expected_model),
            (("runtime", "environment", "SLURM_JOB_ID"), expected_job_id),
            (("config", "topology", "tensor_parallel_size"), expected_tp),
            (("config", "topology", "nodes"), expected_nodes),
            (("config", "tensor_parallel_size"), expected_tp),
            (("config", "node_count"), expected_nodes),
            (("config", "total_gpus"), expected_tp),
            (
                ("config", "distributed_executor_backend"),
                expected_backend,
            ),
            (("config", "topology", "pipeline_parallel_size"), 1),
            (
                ("config", "topology", "distributed_executor_backend"),
                expected_backend,
            ),
            (
                ("config", "model_config_hash"),
                model_identity["model_config_hash"],
            ),
            (
                ("config", "model_checkpoint_hash"),
                model_identity["model_checkpoint_hash"],
            ),
            (
                ("config", "model_view_marker_hash"),
                model_identity["model_view_marker_hash"],
            ),
        ]
    )
    if expected_mode == "baseline":
        expected_metadata.append((("config", "speculative_config"), None))
    else:
        expected_metadata.extend(
            [
                (("config", "speculative_config", "method"), "mtp"),
                (
                    ("config", "speculative_config", "num_speculative_tokens"),
                    expected_k,
                ),
            ]
        )

    mismatches: list[str] = []
    for field_path, expected in expected_metadata:
        found, actual = _nemotron_smoke_metadata_value(payload, field_path)
        field_name = ".".join(field_path)
        if not found or actual != expected:
            actual_display = repr(actual) if found else "<missing>"
            mismatches.append(
                f"{field_name}: expected {expected!r}, got {actual_display}"
            )

    topology_values = {
        field: _nemotron_smoke_metadata_value(payload, path)[1]
        for field, path in (
            ("tensor_parallel_size", ("config", "tensor_parallel_size")),
            ("pipeline_parallel_size", ("config", "pipeline_parallel_size")),
            ("total_gpus", ("config", "total_gpus")),
            ("node_count", ("config", "node_count")),
            ("runtime_gpu_count", ("runtime", "gpu_count")),
        )
    }
    if all(
        not isinstance(value, bool) and isinstance(value, int)
        for value in topology_values.values()
    ):
        tensor_parallel_size = cast(int, topology_values["tensor_parallel_size"])
        pipeline_parallel_size = cast(
            int,
            topology_values["pipeline_parallel_size"],
        )
        total_gpus = cast(int, topology_values["total_gpus"])
        node_count = cast(int, topology_values["node_count"])
        runtime_gpu_count = cast(int, topology_values["runtime_gpu_count"])
        active_gpus = tensor_parallel_size * pipeline_parallel_size
        if total_gpus != active_gpus:
            mismatches.append(
                "config.total_gpus: expected tensor_parallel_size * "
                f"pipeline_parallel_size = {active_gpus}, got "
                f"{total_gpus!r}"
            )
        visible_gpu_capacity = runtime_gpu_count * node_count
        if active_gpus > visible_gpu_capacity:
            mismatches.append(
                "runtime.gpu_count: local GPU inventory across config.node_count "
                f"provides {visible_gpu_capacity} GPUs for {active_gpus} active GPUs"
            )

    prompt_found, prompt_jsonl = _nemotron_smoke_metadata_value(
        payload,
        ("config", "prompt_jsonl"),
    )
    if not prompt_found or "openmath" not in str(prompt_jsonl).lower():
        actual_display = repr(prompt_jsonl) if prompt_found else "<missing>"
        mismatches.append(
            "config.prompt_jsonl: expected an OpenMath prompt source, "
            f"got {actual_display}"
        )

    rollout_batches = payload.get("rollout_batches")
    if not isinstance(rollout_batches, list) or len(rollout_batches) != 3:
        actual_count = len(rollout_batches) if isinstance(rollout_batches, list) else None
        mismatches.append(
            f"rollout_batches: expected 3 rollout barriers, got {actual_count!r}"
        )
    else:
        for batch_index, raw_batch in enumerate(rollout_batches):
            if not isinstance(raw_batch, dict):
                mismatches.append(
                    f"rollout_batches[{batch_index}]: expected an object, "
                    f"got {type(raw_batch).__name__}"
                )
                continue
            request_count = raw_batch.get("request_count")
            if raw_batch.get("batch_index") != batch_index:
                mismatches.append(
                    f"rollout_batches[{batch_index}].batch_index: expected "
                    f"{batch_index}, got {raw_batch.get('batch_index')!r}"
                )
            if request_count != expected_requests_per_batch:
                mismatches.append(
                    f"rollout_batches[{batch_index}].request_count: expected "
                    f"{expected_requests_per_batch}, got {request_count!r}"
                )
            requests = raw_batch.get("requests")
            request_length = len(requests) if isinstance(requests, list) else None
            if (
                not isinstance(requests, list)
                or len(requests) != expected_requests_per_batch
            ):
                mismatches.append(
                    f"rollout_batches[{batch_index}].requests: expected "
                    f"{expected_requests_per_batch}, got {request_length!r}"
                )
            for vector_field in (
                "actual_output_tokens",
                "planned_output_tokens",
                "forced_output_mask",
                "output_token_hashes",
            ):
                vector = raw_batch.get(vector_field)
                vector_length = len(vector) if isinstance(vector, list) else None
                if (
                    not isinstance(vector, list)
                    or vector_length != request_count
                    or vector_length != request_length
                ):
                    mismatches.append(
                        f"rollout_batches[{batch_index}].{vector_field}: expected "
                        "one entry per request with "
                        f"request_count={request_count!r} and "
                        f"len(requests)={request_length!r}, got {vector_length!r}"
                    )
            if (
                not isinstance(requests, list)
                or len(requests) != expected_requests_per_batch
            ):
                continue
            for request_index, raw_request in enumerate(requests):
                if not isinstance(raw_request, dict):
                    mismatches.append(
                        f"rollout_batches[{batch_index}].requests[{request_index}]: "
                        f"expected an object, got {type(raw_request).__name__}"
                    )
                    continue
                expected_request_protocol = {
                    "sample_index": request_index % expected_samples_per_prompt,
                    "seed": expected_seed
                    + batch_index * expected_requests_per_batch
                    + request_index,
                    "max_tokens": expected_max_tokens,
                    "min_tokens": 0,
                    "ignore_eos": False,
                }
                for field, expected in expected_request_protocol.items():
                    actual = raw_request.get(field)
                    if actual != expected or type(actual) is not type(expected):
                        mismatches.append(
                            f"rollout_batches[{batch_index}].requests[{request_index}]"
                            f".{field}: expected {expected!r}, got {actual!r}"
                        )

    if mismatches:
        raise ValueError(
            f"{cohort_label} payload {relative_path.as_posix()} "
            f"does not match the expected cohort: {'; '.join(mismatches)}"
        )


def _nemotron_k_sweep_csv_value_matches(
    actual: object,
    expected: object,
    *,
    exact_float: bool = False,
) -> bool:
    if expected is None:
        return bool(pd.isna(actual))
    if isinstance(expected, bool):
        return str(actual).strip().lower() == str(expected).lower()
    if isinstance(expected, int):
        try:
            numeric = float(actual)
            return (
                math.isfinite(numeric)
                and numeric.is_integer()
                and int(numeric) == expected
            )
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            numeric = float(actual)
            if exact_float:
                return math.isfinite(numeric) and numeric == expected
            return math.isclose(
                numeric,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _nemotron_k_sweep_json_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _nemotron_k_sweep_json_value_matches(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    if isinstance(expected, int):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isfinite(float(actual))
            and float(actual).is_integer()
            and int(actual) == expected
        )
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    return actual == expected


def _raise_nemotron_k_sweep_payload_mismatch(
    relative_path: Path,
    field: str,
    expected: object,
    actual: object,
) -> None:
    raise ValueError(
        f"Nemotron MTP OSL 4K K-sweep payload {relative_path.as_posix()} "
        f"has inconsistent raw evidence at {field}: expected {expected!r}, "
        f"got {actual!r}"
    )


def _validate_nemotron_k_sweep_json_value(
    relative_path: Path,
    field: str,
    actual: object,
    expected: object,
) -> None:
    if not _nemotron_k_sweep_json_value_matches(actual, expected):
        _raise_nemotron_k_sweep_payload_mismatch(
            relative_path,
            field,
            expected,
            actual,
        )


def _nemotron_k_sweep_exact_int(
    value: object,
    relative_path: Path,
    field: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or int(value) != value
        or value < 0
    ):
        _raise_nemotron_k_sweep_payload_mismatch(
            relative_path,
            field,
            "a non-negative integer",
            value,
        )
    return int(value)


def _nemotron_k_sweep_positive_float(
    value: object,
    relative_path: Path,
    field: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        _raise_nemotron_k_sweep_payload_mismatch(
            relative_path,
            field,
            "a positive finite number",
            value,
        )
    return float(value)


def _nemotron_k_sweep_spec_metrics_from_counters(
    *,
    k: int,
    num_drafts: int,
    num_draft_tokens: int,
    num_accepted_tokens: int,
    num_accepted_tokens_per_pos: list[int],
) -> dict[str, object]:
    acceptance_rate = num_accepted_tokens / num_draft_tokens
    accepted_tokens_per_draft = num_accepted_tokens / num_drafts
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "num_accepted_tokens_per_pos": num_accepted_tokens_per_pos,
        "active": True,
        "acceptance_rate": acceptance_rate,
        "mean_acceptance_length": 1.0 + accepted_tokens_per_draft,
        "accepted_tokens_per_draft": accepted_tokens_per_draft,
        "metrics_available": True,
        "acceptance_rate_per_pos": [
            accepted / num_drafts for accepted in num_accepted_tokens_per_pos
        ],
    }


def _validate_and_derive_nemotron_k_sweep_spec_metrics(
    raw_metrics: object,
    relative_path: Path,
    field: str,
    k: int,
) -> dict[str, object]:
    if k == 0:
        _validate_nemotron_k_sweep_json_value(
            relative_path,
            field,
            raw_metrics,
            {},
        )
        return {}
    if not isinstance(raw_metrics, dict):
        _raise_nemotron_k_sweep_payload_mismatch(
            relative_path,
            field,
            "a speculative-metrics object",
            raw_metrics,
        )

    metrics = cast(dict[str, object], raw_metrics)
    num_drafts = _nemotron_k_sweep_exact_int(
        metrics.get("num_drafts"),
        relative_path,
        f"{field}.num_drafts",
    )
    num_draft_tokens = _nemotron_k_sweep_exact_int(
        metrics.get("num_draft_tokens"),
        relative_path,
        f"{field}.num_draft_tokens",
    )
    num_accepted_tokens = _nemotron_k_sweep_exact_int(
        metrics.get("num_accepted_tokens"),
        relative_path,
        f"{field}.num_accepted_tokens",
    )
    raw_per_position = metrics.get("num_accepted_tokens_per_pos")
    if not isinstance(raw_per_position, list) or len(raw_per_position) != k:
        _raise_nemotron_k_sweep_payload_mismatch(
            relative_path,
            f"{field}.num_accepted_tokens_per_pos",
            f"a {k}-element list",
            raw_per_position,
        )
    num_accepted_tokens_per_pos = [
        _nemotron_k_sweep_exact_int(
            value,
            relative_path,
            f"{field}.num_accepted_tokens_per_pos[{index}]",
        )
        for index, value in enumerate(raw_per_position)
    ]
    _validate_nemotron_k_sweep_json_value(
        relative_path,
        f"{field}.num_draft_tokens",
        num_draft_tokens,
        num_drafts * k,
    )
    _validate_nemotron_k_sweep_json_value(
        relative_path,
        f"{field}.num_accepted_tokens",
        num_accepted_tokens,
        sum(num_accepted_tokens_per_pos),
    )
    if num_drafts == 0 or num_draft_tokens == 0:
        _raise_nemotron_k_sweep_payload_mismatch(
            relative_path,
            field,
            "positive speculative counters",
            metrics,
        )

    expected = _nemotron_k_sweep_spec_metrics_from_counters(
        k=k,
        num_drafts=num_drafts,
        num_draft_tokens=num_draft_tokens,
        num_accepted_tokens=num_accepted_tokens,
        num_accepted_tokens_per_pos=num_accepted_tokens_per_pos,
    )
    _validate_nemotron_k_sweep_json_value(
        relative_path,
        f"{field}.keys",
        sorted(metrics),
        sorted(expected),
    )
    for metric, expected_value in expected.items():
        _validate_nemotron_k_sweep_json_value(
            relative_path,
            f"{field}.{metric}",
            metrics.get(metric),
            expected_value,
        )
    return expected


def _derive_nemotron_k_sweep_raw_metrics(
    payload: dict[str, object],
    relative_path: Path,
    *,
    k: int,
    total_gpus: int,
) -> dict[str, object]:
    rollout_batches = cast(list[object], payload["rollout_batches"])
    total_output_tokens = 0
    rollout_times: list[float] = []
    total_num_drafts = 0
    total_num_draft_tokens = 0
    total_num_accepted_tokens = 0
    total_num_accepted_tokens_per_pos = [0] * k

    for batch_index, raw_batch in enumerate(rollout_batches):
        batch = cast(dict[str, object], raw_batch)
        field = f"rollout_batches[{batch_index}]"
        actual_output_tokens = batch.get("actual_output_tokens")
        if not isinstance(actual_output_tokens, list):
            _raise_nemotron_k_sweep_payload_mismatch(
                relative_path,
                f"{field}.actual_output_tokens",
                "a request-level token-count list",
                actual_output_tokens,
            )
        request_output_tokens = [
            _nemotron_k_sweep_exact_int(
                value,
                relative_path,
                f"{field}.actual_output_tokens[{index}]",
            )
            for index, value in enumerate(actual_output_tokens)
        ]
        batch_output_tokens = _nemotron_k_sweep_exact_int(
            batch.get("output_tokens"),
            relative_path,
            f"{field}.output_tokens",
        )
        _validate_nemotron_k_sweep_json_value(
            relative_path,
            f"{field}.actual_output_tokens",
            sum(request_output_tokens),
            batch_output_tokens,
        )
        rollout_time_s = _nemotron_k_sweep_positive_float(
            batch.get("rollout_time_s"),
            relative_path,
            f"{field}.rollout_time_s",
        )
        batch_output_tok_s = batch_output_tokens / rollout_time_s
        _validate_nemotron_k_sweep_json_value(
            relative_path,
            f"{field}.output_tok_s",
            batch.get("output_tok_s"),
            batch_output_tok_s,
        )
        _validate_nemotron_k_sweep_json_value(
            relative_path,
            f"{field}.output_tok_s_per_gpu",
            batch.get("output_tok_s_per_gpu"),
            batch_output_tok_s / total_gpus,
        )

        spec_metrics = _validate_and_derive_nemotron_k_sweep_spec_metrics(
            batch.get("spec_decode_metrics"),
            relative_path,
            f"{field}.spec_decode_metrics",
            k,
        )
        if spec_metrics:
            total_num_drafts += int(spec_metrics["num_drafts"])
            total_num_draft_tokens += int(spec_metrics["num_draft_tokens"])
            total_num_accepted_tokens += int(spec_metrics["num_accepted_tokens"])
            for position, accepted in enumerate(
                cast(list[int], spec_metrics["num_accepted_tokens_per_pos"])
            ):
                total_num_accepted_tokens_per_pos[position] += accepted

        total_output_tokens += batch_output_tokens
        rollout_times.append(rollout_time_s)

    total_rollout_time_s = math.fsum(rollout_times)
    output_tok_s = total_output_tokens / total_rollout_time_s
    aggregate_spec_metrics = (
        _nemotron_k_sweep_spec_metrics_from_counters(
            k=k,
            num_drafts=total_num_drafts,
            num_draft_tokens=total_num_draft_tokens,
            num_accepted_tokens=total_num_accepted_tokens,
            num_accepted_tokens_per_pos=total_num_accepted_tokens_per_pos,
        )
        if k > 0
        else {}
    )
    expected_summary = {
        "total_output_tokens": total_output_tokens,
        "total_rollout_time_s": total_rollout_time_s,
        "output_tok_s": output_tok_s,
        "output_tok_s_per_gpu": output_tok_s / total_gpus,
        "spec_decode_metrics": aggregate_spec_metrics,
    }
    raw_summary = payload.get("summary")
    if not isinstance(raw_summary, dict):
        _raise_nemotron_k_sweep_payload_mismatch(
            relative_path,
            "summary",
            "an aggregate summary object",
            raw_summary,
        )
    summary = cast(dict[str, object], raw_summary)
    for metric, expected_value in expected_summary.items():
        if metric == "spec_decode_metrics":
            raw_spec_metrics = summary.get(metric)
            if not isinstance(raw_spec_metrics, dict):
                _raise_nemotron_k_sweep_payload_mismatch(
                    relative_path,
                    f"summary.{metric}",
                    expected_value,
                    raw_spec_metrics,
                )
            _validate_nemotron_k_sweep_json_value(
                relative_path,
                f"summary.{metric}.keys",
                sorted(raw_spec_metrics),
                sorted(cast(dict[str, object], expected_value)),
            )
            for spec_metric, spec_expected in cast(
                dict[str, object], expected_value
            ).items():
                _validate_nemotron_k_sweep_json_value(
                    relative_path,
                    f"summary.{metric}.{spec_metric}",
                    raw_spec_metrics.get(spec_metric),
                    spec_expected,
                )
            continue
        _validate_nemotron_k_sweep_json_value(
            relative_path,
            f"summary.{metric}",
            summary.get(metric),
            expected_value,
        )
    return expected_summary


def _nemotron_k_sweep_request_provenance_hash(
    requests: list[dict[str, object]],
) -> str:
    canonical_requests = json.dumps(
        requests,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical_requests).hexdigest()


def _validate_nemotron_k_sweep_request_provenance(
    payload: dict[str, object],
    relative_path: Path,
    expected_hashes: tuple[
        str, ...
    ] = NEMOTRON_MTP_K_SWEEP_REQUEST_PROVENANCE_HASHES,
) -> None:
    rollout_batches = cast(list[dict[str, object]], payload["rollout_batches"])
    for batch_index, (batch, expected_hash) in enumerate(
        zip(
            rollout_batches,
            expected_hashes,
            strict=True,
        )
    ):
        requests = cast(list[dict[str, object]], batch["requests"])
        _validate_nemotron_k_sweep_json_value(
            relative_path,
            f"rollout_batches[{batch_index}].request_provenance_hash "
            "(prompt_id, prompt_sha256, source_prompt_sha256, sample_index, "
            "seed, prompt_tokens, max_tokens, min_tokens, ignore_eos)",
            _nemotron_k_sweep_request_provenance_hash(requests),
            expected_hash,
        )


def _output_work_within_one_percent(
    output_tokens: int,
    baseline_tokens: int,
) -> bool:
    return (
        output_tokens >= 0
        and baseline_tokens > 0
        and 99 * baseline_tokens <= 100 * output_tokens <= 101 * baseline_tokens
    )


def _validate_nemotron_k_sweep_summary_row(
    csv_row: pd.Series,
    expected_values: dict[str, object],
    relative_path: Path,
    cohort_label: str = "Nemotron MTP OSL 4K K-sweep",
    *,
    exact_floats: bool = False,
) -> None:
    mismatches = []
    for field, expected in expected_values.items():
        actual = csv_row.get(field)
        if not _nemotron_k_sweep_csv_value_matches(
            actual,
            expected,
            exact_float=exact_floats,
        ):
            mismatches.append(
                f"{field}: expected {expected!r}, got {actual!r}"
            )
    if mismatches:
        raise ValueError(
            f"{cohort_label} summary.csv row "
            f"{relative_path.as_posix()} does not match result.json: "
            f"{'; '.join(mismatches)}"
        )


def _load_nemotron_mtp_fixed_k_rows(
    result_root: Path,
    *,
    expected_results: tuple[tuple[str, str, int, str, str, int, int], ...],
    expected_shared_metadata: tuple[tuple[tuple[str, ...], object], ...],
    expected_request_provenance_hashes: tuple[str, ...],
    expected_seed: int,
    expected_requests_per_batch: int,
    expected_samples_per_prompt: int,
    expected_max_tokens: int,
    expected_num_prompts: int,
    runtime_image_sha256: str,
    cohort_label: str,
    exact_summary_csv_floats: bool = False,
) -> pd.DataFrame:
    expected_paths = {
        Path(model_key) / method_key / "result.json"
        for model_key, method_key, _k, _model, _job_id, _tp, _nodes in (
            expected_results
        )
    }
    discovered_paths = {
        path.relative_to(result_root)
        for path in result_root.glob("**/result.json")
        if path.is_file()
    }
    missing_paths = sorted(expected_paths - discovered_paths)
    unexpected_paths = sorted(discovered_paths - expected_paths)
    if missing_paths or unexpected_paths:
        details = []
        if missing_paths:
            missing = ", ".join(path.as_posix() for path in missing_paths)
            details.append(f"missing expected payloads: {missing}")
        if unexpected_paths:
            unexpected = ", ".join(path.as_posix() for path in unexpected_paths)
            details.append(f"unexpected payloads: {unexpected}")
        raise ValueError(
            f"{cohort_label} cohort must contain exactly the "
            f"{len(expected_results)} expected result.json payloads; "
            f"{'; '.join(details)}"
        )

    summary_csv = result_root / "summary.csv"
    if not summary_csv.is_file():
        raise ValueError(f"{cohort_label} cohort requires summary.csv")
    summary_rows = pd.read_csv(
        summary_csv,
        dtype={"job_id": str, "result_path": str},
        float_precision="round_trip",
    )
    required_summary_columns = {
        "model",
        "method",
        "k",
        "job_id",
        "status",
        "temperature",
        "top_p",
        "max_new_tokens",
        "num_prompts",
        "samples_per_prompt",
        "rollout_batches",
        "tp",
        "nodes",
        "cudagraph_mode",
        "runtime_image_sha256",
        "total_output_tokens",
        "total_rollout_time_s",
        "output_tok_s_per_gpu",
        "throughput_speedup",
        "time_speedup",
        "output_token_ratio",
        "time_speedup_valid",
        "acceptance_rate",
        "mean_accept_len",
        "result_path",
    }
    missing_columns = sorted(required_summary_columns - set(summary_rows.columns))
    if missing_columns:
        raise ValueError(
            f"{cohort_label} summary.csv missing columns: "
            f"{', '.join(missing_columns)}"
        )
    summary_paths = summary_rows["result_path"].astype(str).tolist()
    expected_path_strings = {path.as_posix() for path in expected_paths}
    if (
        len(summary_rows) != len(expected_results)
        or len(summary_paths) != len(set(summary_paths))
        or set(summary_paths) != expected_path_strings
    ):
        raise ValueError(
            f"{cohort_label} summary.csv must contain exactly one row "
            "for each expected result.json payload"
        )
    summary_by_path = summary_rows.set_index("result_path", drop=False)

    payloads: dict[tuple[str, str], tuple[dict[str, object], Path]] = {}
    for model_key, method_key, k, model, job_id, tp, nodes in (
        expected_results
    ):
        path = result_root / model_key / method_key / "result.json"
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise ValueError(
                f"{cohort_label} payload "
                f"{path.relative_to(result_root).as_posix()} must be a JSON object"
            )
        payload = cast(dict[str, object], raw_payload)
        _validate_nemotron_mtp_k_sweep_payload(
            payload,
            path.relative_to(result_root),
            expected_model_key=model_key,
            expected_mode="baseline" if method_key == "baseline" else "mtp_static",
            expected_k=k,
            expected_model=model,
            expected_job_id=job_id,
            expected_tp=tp,
            expected_nodes=nodes,
            expected_shared_metadata=expected_shared_metadata,
            expected_model_identities=NEMOTRON_MTP_K_SWEEP_MODEL_IDENTITIES,
            expected_seed=expected_seed,
            expected_requests_per_batch=expected_requests_per_batch,
            expected_samples_per_prompt=expected_samples_per_prompt,
            expected_max_tokens=expected_max_tokens,
            cohort_label=cohort_label,
        )
        payloads[(model_key, method_key)] = (payload, path)

    for payload, path in payloads.values():
        _validate_nemotron_k_sweep_request_provenance(
            payload,
            path.relative_to(result_root),
            expected_request_provenance_hashes,
        )

    raw_metrics = {
        (model_key, method_key): _derive_nemotron_k_sweep_raw_metrics(
            payloads[(model_key, method_key)][0],
            payloads[(model_key, method_key)][1].relative_to(result_root),
            k=k,
            total_gpus=tp,
        )
        for model_key, method_key, k, _model, _job_id, tp, _nodes in (
            expected_results
        )
    }

    rows: list[dict[str, object]] = []
    for model_key, method_key, k, _model, job_id, tp, nodes in (
        expected_results
    ):
        payload, path = payloads[(model_key, method_key)]
        config = cast(dict[str, object], payload["config"])
        metrics = raw_metrics[(model_key, method_key)]
        baseline_metrics = raw_metrics[(model_key, "baseline")]
        output_tokens = int(metrics["total_output_tokens"])
        baseline_tokens = int(baseline_metrics["total_output_tokens"])
        output_tok_s_gpu = float(metrics["output_tok_s_per_gpu"])
        baseline_tok_s_gpu = float(baseline_metrics["output_tok_s_per_gpu"])
        rollout_time_s = float(metrics["total_rollout_time_s"])
        baseline_time_s = float(baseline_metrics["total_rollout_time_s"])
        output_token_ratio = output_tokens / baseline_tokens
        throughput_speedup = output_tok_s_gpu / baseline_tok_s_gpu
        raw_rollout_time_speedup = baseline_time_s / rollout_time_s
        time_speedup_valid = _output_work_within_one_percent(
            output_tokens,
            baseline_tokens,
        )
        spec_metrics = cast(
            dict[str, object], metrics["spec_decode_metrics"]
        )
        acceptance_rate = (
            float(spec_metrics["acceptance_rate"]) if spec_metrics else None
        )
        mean_acceptance_length = (
            float(spec_metrics["mean_acceptance_length"]) if spec_metrics else None
        )
        relative_path = path.relative_to(result_root)
        csv_expected = {
            "model": model_key,
            "method": "baseline" if method_key == "baseline" else "mtp_static",
            "k": None if method_key == "baseline" else k,
            "job_id": job_id,
            "status": "complete",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": expected_max_tokens,
            "num_prompts": expected_num_prompts,
            "samples_per_prompt": expected_samples_per_prompt,
            "rollout_batches": 3,
            "tp": tp,
            "nodes": nodes,
            "cudagraph_mode": "PIECEWISE",
            "runtime_image_sha256": runtime_image_sha256,
            "total_output_tokens": output_tokens,
            "total_rollout_time_s": rollout_time_s,
            "output_tok_s_per_gpu": output_tok_s_gpu,
            "throughput_speedup": throughput_speedup,
            "time_speedup": raw_rollout_time_speedup,
            "output_token_ratio": output_token_ratio,
            "time_speedup_valid": time_speedup_valid,
            "acceptance_rate": acceptance_rate,
            "mean_accept_len": mean_acceptance_length,
            "result_path": relative_path.as_posix(),
        }
        _validate_nemotron_k_sweep_summary_row(
            summary_by_path.loc[relative_path.as_posix()],
            csv_expected,
            relative_path,
            cohort_label,
            exact_floats=exact_summary_csv_floats,
        )
        rows.append(
            {
                "model_key": model_key,
                "model": _nemotron_smoke_model_label(config.get("model")),
                "model_display": model_key.title(),
                "method_key": method_key,
                "method": "Baseline" if method_key == "baseline" else f"K{k}",
                "k": k,
                "job_id": job_id,
                "output_tok_s_gpu": output_tok_s_gpu,
                "throughput_speedup": throughput_speedup,
                "rollout_time_speedup": (
                    raw_rollout_time_speedup if time_speedup_valid else math.nan
                ),
                "output_token_ratio": output_token_ratio,
                "acceptance_rate": acceptance_rate,
                "mean_acceptance_length": mean_acceptance_length,
                "time_speedup_valid": time_speedup_valid,
                "topology": f"TP{tp} / {nodes} {'node' if nodes == 1 else 'nodes'}",
                "source": str(path.relative_to(ROOT)),
                "result_relative_path": relative_path.as_posix(),
            }
        )

    frame = pd.DataFrame(rows)
    frame["absolute_best"] = False
    frame["selected_by_policy"] = False
    for model_key in frame["model_key"].drop_duplicates():
        candidates = frame[
            (frame["model_key"] == model_key) & (frame["method_key"] != "baseline")
        ]
        best_index = candidates["throughput_speedup"].idxmax()
        best_speedup = float(frame.loc[best_index, "throughput_speedup"])
        eligible = candidates[
            candidates["throughput_speedup"] >= best_speedup * 0.98
        ].sort_values("k")
        selected_index = eligible.index[0]
        frame.loc[best_index, "absolute_best"] = True
        frame.loc[selected_index, "selected_by_policy"] = True

    def validity(row: pd.Series) -> str:
        parts = ["validated fixed-K evidence"]
        if row["method_key"] == "baseline":
            parts.append("reference")
        elif bool(row["selected_by_policy"]):
            parts.append("selected by smallest-K-within-2% policy")
        if bool(row["time_speedup_valid"]):
            parts.append("rollout-time speedup comparable within 1% output work")
        else:
            parts.append("rollout-time speedup suppressed: output-token ratio outside 1%")
        return "; ".join(parts)

    frame["validity"] = frame.apply(validity, axis=1)
    return frame


def load_nemotron_mtp_k_sweep_rows(
    result_root: Path = NEMOTRON_MTP_K_SWEEP_ROOT,
) -> pd.DataFrame:
    return _load_nemotron_mtp_fixed_k_rows(
        result_root,
        expected_results=NEMOTRON_MTP_K_SWEEP_RESULTS,
        expected_shared_metadata=NEMOTRON_MTP_K_SWEEP_SHARED_METADATA,
        expected_request_provenance_hashes=(
            NEMOTRON_MTP_K_SWEEP_REQUEST_PROVENANCE_HASHES
        ),
        expected_seed=NEMOTRON_MTP_K_SWEEP_SEED,
        expected_requests_per_batch=32,
        expected_samples_per_prompt=4,
        expected_max_tokens=4096,
        expected_num_prompts=8,
        runtime_image_sha256=NEMOTRON_MTP_K_SWEEP_RUNTIME_IMAGE_SHA256,
        cohort_label="Nemotron MTP OSL 4K K-sweep",
    )


def load_nemotron_mtp_osl16k_full_rows(
    result_root: Path = NEMOTRON_MTP_OSL16K_FULL_ROOT,
) -> pd.DataFrame:
    rows = _load_nemotron_mtp_fixed_k_rows(
        result_root,
        expected_results=NEMOTRON_MTP_OSL16K_FULL_RESULTS,
        expected_shared_metadata=NEMOTRON_MTP_OSL16K_FULL_SHARED_METADATA,
        expected_request_provenance_hashes=(
            NEMOTRON_MTP_OSL16K_REQUEST_PROVENANCE_HASHES
        ),
        expected_seed=NEMOTRON_MTP_K_SWEEP_SEED,
        expected_requests_per_batch=64,
        expected_samples_per_prompt=4,
        expected_max_tokens=16384,
        expected_num_prompts=16,
        runtime_image_sha256=NEMOTRON_MTP_K_SWEEP_RUNTIME_IMAGE_SHA256,
        cohort_label="Nemotron MTP OSL 16K full",
        exact_summary_csv_floats=True,
    )

    def validity(row: pd.Series) -> str:
        parts = ["strictly validated fixed-K full-run evidence"]
        if row["method_key"] == "baseline":
            parts.append("reference")
        if bool(row["time_speedup_valid"]):
            parts.append("rollout-time speedup comparable within 1% output work")
        else:
            parts.append(
                "rollout-time speedup suppressed: output-token ratio outside 1%"
            )
        return "; ".join(parts)

    rows["validity"] = rows.apply(validity, axis=1)
    return rows


def publish_nemotron_mtp_k_sweep_evidence(
    *,
    result_root: Path = NEMOTRON_MTP_K_SWEEP_ROOT,
    public_data_dir: Path = PUBLIC_DATA,
) -> tuple[Path, ...]:
    load_nemotron_mtp_k_sweep_rows(result_root)
    destination_root = public_data_dir / result_root.name
    if destination_root.exists():
        shutil.rmtree(destination_root)
    published = []
    for model_key, method_key, _k, _model, _job_id, _tp, _nodes in (
        NEMOTRON_MTP_K_SWEEP_RESULTS
    ):
        relative_path = Path(model_key) / method_key / "result.json"
        destination = publish_public_data(
            result_root / relative_path,
            destination_root / relative_path.parent,
        )
        if destination is None:
            raise ValueError(
                "Nemotron MTP OSL 4K K-sweep evidence disappeared after "
                f"validation: {relative_path.as_posix()}"
            )
        published.append(destination)
    return tuple(published)


def render_nemotron_mtp_k_sweep_section(
    result_root: Path = NEMOTRON_MTP_K_SWEEP_ROOT,
    evidence_href_root: str | None = None,
) -> str:
    rows = load_nemotron_mtp_k_sweep_rows(result_root)
    display = rows.copy()
    if evidence_href_root is not None:
        href_root = evidence_href_root.rstrip("/")
        display["result_href"] = display["result_relative_path"].map(
            lambda relative_path: f"{href_root}/{relative_path}"
        )
    display["throughput_speedup_display"] = display["throughput_speedup"].map(
        lambda value: f"{float(value):.3f}x"
    )
    display["rollout_time_speedup_display"] = display.apply(
        lambda row: (
            f"{float(row['rollout_time_speedup']):.3f}x"
            if bool(row["time_speedup_valid"])
            else "n/a (output-token ratio outside 1%)"
        ),
        axis=1,
    )
    display["output_token_ratio_display"] = display["output_token_ratio"].map(
        lambda value: f"{float(value):.4f}x"
    )
    display["acceptance_rate_display"] = display["acceptance_rate"].map(
        lambda value: "n/a" if pd.isna(value) else f"{float(value) * 100.0:.2f}%"
    )
    display["mean_acceptance_length_display"] = display[
        "mean_acceptance_length"
    ].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.3f}")

    super_best = rows[(rows["model_key"] == "super") & rows["absolute_best"]].iloc[0]
    ultra_best = rows[(rows["model_key"] == "ultra") & rows["absolute_best"]].iloc[0]
    ultra_selected = rows[
        (rows["model_key"] == "ultra") & rows["selected_by_policy"]
    ].iloc[0]
    chart_rows = rows[rows["method_key"] != "baseline"].rename(
        columns={"throughput_speedup": "speedup"}
    )
    chart = line_svg(
        chart_rows,
        "Nemotron Native MTP OSL 4K throughput speedup by fixed K "
        "(baseline = 1.0x)",
        "speedup",
        "k",
        "model_display",
        NEMOTRON_MODEL_SERIES_COLORS,
    )
    return "".join(
        [
            '<section class="section"><h2>Nemotron Native MTP OSL 4K K Sweep</h2>',
            '<p class="note">This is validated fixed-K evidence, not DynamicMTP, '
            "and remains separate from calibrated dynamic-method results. All eight "
            "jobs completed on vLLM 0.24.0 with CUDA Graph PIECEWISE, temperature "
            "1.0 / top_p 1.0, max_new_tokens 4096, runtime image SHA "
            f"<code>{NEMOTRON_MTP_K_SWEEP_RUNTIME_IMAGE_SHA256}</code>, Super TP2 / "
            "1 node, and Ultra TP8 / 2 nodes.</p>",
            '<p class="note">Methodology: OpenMath natural-EOS Sync-RL-style '
            "rollout, so output work can differ. Each job uses 8 prompts, 4 samples "
            "per prompt, and 3 rollout barriers. Rollout-time speedup is shown only "
            "when the aggregate output-token ratio is within 1% of the matched model "
            "baseline.</p>",
            '<p class="callout"><strong>Key finding.</strong> '
            f"Super best K{int(super_best['k'])} at "
            f"{float(super_best['throughput_speedup']):.3f}x; Ultra absolute best "
            f"K{int(ultra_best['k'])} at "
            f"{float(ultra_best['throughput_speedup']):.3f}x, while "
            f"K{int(ultra_selected['k'])} is within 2% of best and is selected by "
            "the smallest-K-within-2% policy.</p>",
            f'<div class="chart-card k-sweep-chart">{chart}</div>',
            '<div class="table-wrap">',
            table(
                display,
                [
                    ("model", "Model", "text"),
                    ("method", "Method", "text"),
                    ("job_id", "Job ID", "result_link"),
                    ("output_tok_s_gpu", "tok/s/GPU", "num"),
                    (
                        "throughput_speedup_display",
                        "Throughput speedup",
                        "text",
                    ),
                    (
                        "rollout_time_speedup_display",
                        "Rollout-time speedup",
                        "text",
                    ),
                    ("output_token_ratio_display", "Output ratio", "text"),
                    ("acceptance_rate_display", "Acceptance rate", "text"),
                    (
                        "mean_acceptance_length_display",
                        "Mean acceptance length",
                        "text",
                    ),
                    ("validity", "Validity", "text"),
                ],
            ),
            "</div></section>",
        ]
    )


def publish_nemotron_mtp_osl16k_full_evidence(
    *,
    result_root: Path = NEMOTRON_MTP_OSL16K_FULL_ROOT,
    public_data_dir: Path = PUBLIC_DATA,
) -> tuple[Path, ...]:
    load_nemotron_mtp_osl16k_full_rows(result_root)
    destination_root = public_data_dir / result_root.name
    if destination_root.exists():
        shutil.rmtree(destination_root)
    published = []
    for model_key, method_key, _k, _model, _job_id, _tp, _nodes in (
        NEMOTRON_MTP_OSL16K_FULL_RESULTS
    ):
        relative_path = Path(model_key) / method_key / "result.json"
        destination = publish_public_data(
            result_root / relative_path,
            destination_root / relative_path.parent,
        )
        if destination is None:
            raise ValueError(
                "Nemotron MTP OSL 16K full evidence disappeared after "
                f"validation: {relative_path.as_posix()}"
            )
        published.append(destination)
    return tuple(published)


def render_nemotron_mtp_osl16k_full_section(
    result_root: Path = NEMOTRON_MTP_OSL16K_FULL_ROOT,
    evidence_href_root: str | None = None,
) -> str:
    rows = load_nemotron_mtp_osl16k_full_rows(result_root)
    display = rows.copy()
    if evidence_href_root is not None:
        href_root = evidence_href_root.rstrip("/")
        display["result_href"] = display["result_relative_path"].map(
            lambda relative_path: f"{href_root}/{relative_path}"
        )
    display["throughput_speedup_display"] = display["throughput_speedup"].map(
        lambda value: f"{float(value):.3f}x"
    )
    display["rollout_time_speedup_display"] = display.apply(
        lambda row: (
            f"{float(row['rollout_time_speedup']):.3f}x"
            if bool(row["time_speedup_valid"])
            else "n/a (output-token ratio outside 1%)"
        ),
        axis=1,
    )
    display["output_token_ratio_display"] = display["output_token_ratio"].map(
        lambda value: f"{float(value) * 100.0:.2f}%"
    )
    display["acceptance_rate_display"] = display["acceptance_rate"].map(
        lambda value: "n/a" if pd.isna(value) else f"{float(value) * 100.0:.2f}%"
    )
    display["mean_acceptance_length_display"] = display[
        "mean_acceptance_length"
    ].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.2f}")

    indexed = rows.set_index(["model_key", "method_key"])
    super_k3 = indexed.loc[("super", "k3")]
    super_k5 = indexed.loc[("super", "k5")]
    ultra_k5 = indexed.loc[("ultra", "k5")]
    chart_rows = rows[rows["method_key"] != "baseline"].rename(
        columns={"throughput_speedup": "speedup"}
    )
    chart = line_svg(
        chart_rows,
        "Nemotron Native MTP OSL 16K fixed-K throughput speedup "
        "by model / K (baseline = 1.0x)",
        "speedup",
        "k",
        "model_display",
        NEMOTRON_MODEL_SERIES_COLORS,
    )
    return "".join(
        [
            '<section class="section"><h2>Nemotron Native MTP OSL 16K Full</h2>',
            '<p class="note">This is validated fixed-K evidence, not DynamicMTP, '
            "and remains separate from calibrated dynamic-method claims. All five "
            "jobs completed on vLLM 0.24.0 with CUDA Graph PIECEWISE, temperature "
            "1.0 / top_p 1.0, max_new_tokens 16384, runtime image SHA "
            f"<code>{NEMOTRON_MTP_K_SWEEP_RUNTIME_IMAGE_SHA256}</code>, Super TP2 / "
            "1 node, and Ultra TP8 / 2 nodes.</p>",
            '<p class="note">Methodology: OpenMath natural-EOS Sync-RL-style '
            "rollout with an OSL cap 16K, so output work can differ. Each job uses "
            "16 prompts, 4 samples per prompt, and 3 rollout barriers. Rollout-time "
            "speedup is shown only when the exact aggregate output-token ratio is "
            "within the inclusive &plusmn;1% bound of the matched model baseline.</p>",
            '<p class="callout"><strong>Key findings.</strong> '
            f"Super K3 {float(super_k3['throughput_speedup']):.3f}x and K5 "
            f"{float(super_k5['throughput_speedup']):.3f}x throughput; both Super "
            "rollout-time comparisons are invalid because their work ratios are "
            f"{float(super_k3['output_token_ratio']) * 100.0:.2f}% and "
            f"{float(super_k5['output_token_ratio']) * 100.0:.2f}%. Ultra K5 "
            f"{float(ultra_k5['throughput_speedup']):.3f}x throughput and "
            f"{float(ultra_k5['rollout_time_speedup']):.3f}x rollout-time speedup "
            f"with a {float(ultra_k5['output_token_ratio']) * 100.0:.2f}% work "
            f"ratio, {float(ultra_k5['acceptance_rate']) * 100.0:.2f}% acceptance, "
            "and mean accepted length "
            f"{float(ultra_k5['mean_acceptance_length']):.2f}.</p>",
            f'<div class="chart-card k-sweep-chart">{chart}</div>',
            '<div class="table-wrap">',
            table(
                display,
                [
                    ("model", "Model", "text"),
                    ("method", "Method / K", "text"),
                    ("job_id", "Job ID", "result_link"),
                    ("output_tok_s_gpu", "tok/s/GPU", "num"),
                    (
                        "throughput_speedup_display",
                        "Throughput speedup",
                        "text",
                    ),
                    (
                        "rollout_time_speedup_display",
                        "Rollout-time speedup",
                        "text",
                    ),
                    (
                        "output_token_ratio_display",
                        "Output-token ratio",
                        "text",
                    ),
                    ("acceptance_rate_display", "Acceptance", "text"),
                    (
                        "mean_acceptance_length_display",
                        "Mean accept length",
                        "text",
                    ),
                    ("validity", "Validity / evidence", "text"),
                ],
            ),
            "</div></section>",
        ]
    )


def count_speedbench_result_artifacts() -> dict[str, int]:
    counts = {"official": 0, "overlay": 0}
    for path in sorted(DFLARE_RESULT_ROOT.glob("**/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "complete":
            continue
        cohort = payload.get("config", {}).get("cohort")
        if cohort in counts:
            counts[cohort] += 1
    return counts


def render_sync_speedbench_status_section() -> str:
    speedbench_dataset_revision = shell_assignment(
        SPEEDBENCH_STAGE_SCRIPT,
        "SPEED_DATASET_REVISION",
    )
    modelopt_revision = shell_assignment(SPEEDBENCH_STAGE_SCRIPT, "MODELOPT_REVISION")
    qwen_support, nemotron_support = load_sync_speedbench_support()
    completed = load_completed_qwen32_math_dynamic_rows()
    perfcfg_replay = load_perfcfg_dynamic_replay_rows()
    speedbench_counts = count_speedbench_result_artifacts()
    cohort_rows = pd.DataFrame(
        [
            {
                "cohort": "Official SPEED-Bench",
                "protocol": "official-modelopt",
                "sampling": "captured from instrumented official config",
                "revisions": (
                    f"SPEED-Bench {speedbench_dataset_revision}; "
                    f"ModelOpt {modelopt_revision}"
                ),
                "local_status": (
                    f"{speedbench_counts['official']} completed local result.json artifact(s)"
                ),
            },
            {
                "cohort": "Sync-RL overlay",
                "protocol": "sync-rl-overlay-user",
                "sampling": "temperature 1.0 / top_p 1.0",
                "revisions": (
                    f"SPEED-Bench {speedbench_dataset_revision}; "
                    f"ModelOpt {modelopt_revision}"
                ),
                "local_status": (
                    f"{speedbench_counts['overlay']} completed local result.json artifact(s)"
                ),
            },
        ]
    )
    parts = [
        (
            "<section class=\"section\"><h2>Sync-RL SWE and SPEED-Bench Status</h2>"
            "<p class=\"note\">This Task 6 snapshot is local-only and keeps completed "
            "evidence separate from launch support. Official SPEED-Bench and Sync-RL "
            "overlay remain separate cohorts, and pending remote jobs are not scored "
            "or summarized here.</p>"
        ),
        "<div class=\"table-wrap\">",
        table(
            cohort_rows,
            [
                ("cohort", "Cohort", "text"),
                ("protocol", "Protocol", "text"),
                ("sampling", "Sampling", "text"),
                ("revisions", "Pinned revisions", "text"),
                ("local_status", "Local status", "text"),
            ],
        ),
        "</div>",
    ]
    if speedbench_counts["official"] == 0 and speedbench_counts["overlay"] == 0:
        parts.append(
            "<p class=\"note\">No completed SPEED-Bench official or overlay "
            "result.json artifacts are present in this checkout.</p>"
        )
    parts.extend(
        [
            "<h3>Completed local Qwen3-32B Math DynamicSD results</h3>",
            "<p class=\"note\">These completed rows are legacy Math Sync-RL summaries "
            "with temp 1.0 / top_p 0.9. They are intentionally reported separately "
            "from the pending SWE 32K/64K and SPEED-Bench cohorts.</p>",
            "<div class=\"table-wrap\">",
            table(
                completed,
                [
                    ("result_scope", "Result scope", "text"),
                    ("dataset", "Dataset", "text"),
                    ("sampling", "Sampling", "text"),
                    ("tok_s_gpu", "tok/s/GPU", "num"),
                    ("speedup_vs_baseline", "Speedup vs baseline", "x"),
                    ("speedup_vs_static", "Speedup vs static", "x"),
                    (
                        "time_reduction_vs_baseline_pct",
                        "Time reduction vs baseline",
                        "pct_2dp",
                    ),
                    ("acceptance_rate", "Acceptance", "ratio_pct"),
                    ("source", "Source", "source"),
                ],
            ),
            "</div>",
            "<h3>Performance-Recipe DynamicSD Replay</h3>",
            "<p class=\"note\">These temp 1.0 / top_p 1.0 rows replay the "
            "historical schedule with CUDA Graph PIECEWISE. They are a historical "
            "schedule replay and are excluded from calibrated claims until a signed "
            "K-calibration artifact is available.</p>",
            "<div class=\"table-wrap\">",
            table(
                perfcfg_replay,
                [
                    ("model", "Model", "text"),
                    ("scope", "Scope", "text"),
                    ("baseline_job", "Baseline job", "text"),
                    ("dynamic_job", "Dynamic job", "text"),
                    ("max_new_tokens", "Max OSL", "int"),
                    ("requests_per_rollout_batch", "Requests/barrier", "int"),
                    ("baseline_time_s", "Baseline time (s)", "num"),
                    ("dynamic_time_s", "Dynamic time (s)", "num"),
                    ("baseline_tok_s_gpu", "Baseline tok/s/GPU", "num"),
                    ("dynamic_tok_s_gpu", "Dynamic tok/s/GPU", "num"),
                    ("throughput_speedup", "Throughput speedup", "x"),
                    ("time_speedup", "Time speedup", "x"),
                    ("time_reduction_pct", "Time reduction", "pct_2dp"),
                    ("acceptance_rate", "Acceptance", "ratio_pct"),
                    ("mean_accept_len", "Mean accept len", "num"),
                    ("work_match", "Work match", "text"),
                    ("calibration_status", "Calibration status", "text"),
                ],
            ),
            "</div>",
            "<h3>Qwen SWE Sync-RL support</h3>",
            "<div class=\"table-wrap\">",
            table(
                qwen_support,
                [
                    ("model", "Model", "text"),
                    ("profiles", "Profiles", "text"),
                    ("supported", "Supported", "text"),
                    ("integration", "Integration only", "text"),
                    ("unsupported", "Unsupported", "text"),
                ],
            ),
            "</div>",
            "<h3>Nemotron SPEED-Bench support</h3>",
            "<div class=\"table-wrap\">",
            table(
                nemotron_support,
                [
                    ("model", "Model", "text"),
                    ("profiles", "Profiles", "text"),
                    ("official_support", "Official launcher support", "text"),
                    ("overlay_support", "Sync-RL overlay support", "text"),
                    ("official_limitations", "Official limitations", "text"),
                    ("overlay_gates", "Overlay gates", "text"),
                    ("unsupported", "Unsupported", "text"),
                ],
            ),
            "</div>",
            "</section>",
        ]
    )
    return "".join(parts)


def build_vllm_html(
    main: pd.DataFrame,
    added: pd.DataFrame,
    native_rows: pd.DataFrame,
    dflare_rows: pd.DataFrame,
    dflare_status: pd.DataFrame,
    nemotron_evidence_href_root: str | None = None,
    nemotron_k_sweep_evidence_href_root: str | None = None,
    nemotron_osl16k_evidence_href_root: str | None = None,
) -> str:
    updated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added_summary = aggregate_added(added)
    matrix_path = DOCS / "vllm_standalone_all_batches_combined_matrix_20260619.csv"
    main_matrix = pd.read_csv(matrix_path) if matrix_path.exists() else matrix(main)
    added_matrix = matrix(added[added["valid_result"]]) if not added.empty else pd.DataFrame()
    valid_added = added[added["valid_result"]].copy() if not added.empty else pd.DataFrame()
    unmatched = valid_added[pd.to_numeric(valid_added.get("speedup"), errors="coerce").isna()].copy() if not valid_added.empty else pd.DataFrame()
    focus = added[
        added["method"].isin(["pard_k16", "pard2_k16"])
        & added["model"].isin(["Qwen3-32B", "Qwen3-235B-A22B", "Qwen3-30B-A3B", "Qwen3-8B"])
    ].copy()
    failed = added[~added["valid_result"]].copy() if not added.empty else pd.DataFrame()
    eagle8 = added[
        (added["model"] == "Qwen3-235B-A22B")
        & (added["method"] == "eagle3_k8")
        & (added["domain"] == "Math")
        & added["valid_result"]
    ]
    eagle_lines = []
    for temp, label in [(0.0, "temp0"), (1.0, "temp1")]:
        sub = eagle8[eagle8["temperature"] == temp]
        if not sub.empty:
            eagle_lines.append(
                f"Qwen3-235B Eagle-3 K8 Math {label}: mean speedup {sub['speedup'].mean():.2f}x, "
                f"acceptance {sub['acceptance_pct'].mean():.1f}%."
            )
    q32_pard16 = added[
        (added["model"] == "Qwen3-32B")
        & (added["method"] == "pard_k16")
        & (added["temperature"] == 1.0)
        & added["valid_result"]
    ]
    if not q32_pard16.empty:
        eagle_lines.append(
            f"Qwen3-32B PARD K16 temp1 retry completed {len(q32_pard16)} rows, "
            f"mean speedup {q32_pard16['speedup'].mean():.2f}x."
        )
    key_finding = " ".join(eagle_lines) if eagle_lines else "Latest CSV refresh completed; no new valid rows found."
    css = """
:root{--text:#111827;--muted:#6b7280;--line:#d8dee8;--bg:#f7f8fb;--panel:#fff;--blue:#1f5fbf;--good:#e8f3ff;--bad:#fff0f0;--warn:#fff7df}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;font-size:15px;line-height:1.42}main{max-width:1500px;margin:0 auto;padding:24px}h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:28px 0 10px}h3{font-size:16px;margin:18px 0 8px}.topbar{margin-bottom:12px}.topbar a{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:8px;background:#fff;padding:6px 10px;text-decoration:none;font-weight:700;color:var(--blue)}.sub,.note{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:18px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}.card b{display:block;font-size:22px}.pill{display:inline-block;border:1px solid var(--line);background:#fff;border-radius:999px;padding:4px 9px;margin:2px 4px 2px 0;color:#374151}.section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;margin:14px 0}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:12px}.chart-card{border:1px solid var(--line);border-radius:8px;background:#fff;padding:10px;min-width:0}.chart-card svg{width:100%;height:auto;display:block}.k-sweep-chart{max-width:760px;margin:12px auto 14px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 14px}th,td{border:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}th{background:#eef2f7;font-size:13px}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.source-col,.manifest-col,.path-col,.note-col,.name-col{max-width:260px;white-space:normal;overflow-wrap:anywhere}.manifest-col,.path-col{font-size:12px}.error-col{max-width:360px}.good{background:var(--good)}.bad{background:var(--bad)}.warn{background:var(--warn)}code{background:#f3f4f6;padding:1px 4px;border-radius:4px}a code{color:var(--blue)}.native-profile-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.native-profile-matrix{min-width:0}.native-profile-matrix h4{font-size:14px;margin:8px 0 4px}.native-speedup-matrix{font-size:14px}.native-speedup-matrix th,.native-speedup-matrix td{text-align:center;vertical-align:middle}.native-speedup-matrix th:first-child,.native-speedup-matrix th:nth-child(2),.native-speedup-matrix td:first-child,.native-speedup-matrix td:nth-child(2){text-align:left}.native-speedup-matrix .speed-cell{font-weight:750;font-variant-numeric:tabular-nums;white-space:nowrap}.native-speedup-matrix .speed-cell.slowdown{background:var(--matrix-red);color:var(--matrix-text,#8f1d16)}.native-speedup-matrix .speed-cell.neutral{background:#edf1f5;color:#374151}.native-speedup-matrix .speed-cell.speedup{background:var(--matrix-blue);color:var(--matrix-text,#17406d)}.native-speedup-matrix .speed-cell.partial{outline:2px solid #d89b22;outline-offset:-2px}.native-speedup-matrix .speed-cell.empty{background:#f8fafc;color:#94a3b8}.native-profile-details{margin-top:14px}.native-profile-details summary{cursor:pointer;font-weight:750;color:#374151}details.archive-table{margin-top:12px}details.archive-table summary{cursor:pointer;font-weight:750;color:#374151}@media(max-width:1200px){.cards{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:1000px){.native-profile-grid{grid-template-columns:1fr}.charts{grid-template-columns:1fr}}@media(max-width:900px){main{padding:16px}.cards{grid-template-columns:1fr 1fr}table{font-size:13px}}"""
    parts = [
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        f"<title>vLLM Standalone SpecDec Results</title><style>{css}</style></head><body><main>",
        '<div class="topbar"><a href="../index.html">Back to report hub</a></div>',
        "<h1>vLLM Standalone SpecDec Results</h1>",
        f"<p class=\"sub\">Updated {esc(updated)}. Data refresh from the 6/19 batch matrix, 6/20 extra-K/PARD sweeps, 6/16 temp0/temp1 trend analysis, and refreshed Lyris legacy breakdown JSONs for Qwen3-30B-A3B and Qwen3-8B.</p>",
        "<div><span class=\"pill\">ISL 4096</span><span class=\"pill\">OSL 32768</span><span class=\"pill\">batch 1/2/4/8/16/32</span><span class=\"pill\">temperature 0.0 and 1.0</span><span class=\"pill\">top_p 1.0 where available</span></div>",
        "<div class=\"cards\">",
        f"<div class=\"card\"><b>{len(main)}</b><span>existing 6/19 rows</span></div>",
        f"<div class=\"card\"><b>{int(added['valid_result'].sum()) if not added.empty else 0}</b><span>valid added rows</span></div>",
        f"<div class=\"card\"><b>{len(unmatched)}</b><span>valid rows waiting baseline</span></div>",
        f"<div class=\"card\"><b>{len(failed)}</b><span>failed or invalid added rows</span></div>",
        f"<div class=\"card\"><b>{len(added_summary)}</b><span>added summary groups</span></div>",
        "</div>",
        "<section class=\"section\"><h2>Scope</h2><p>This page is the matched-comparison view for <b>ISL 4096 / OSL 32768</b>. It keeps speedup cells blank when the exact baseline is missing for the same domain, model, temperature, batch size, ISL, and OSL.</p></section>",
        render_nemotron_mtp_legacy_smoke_section(nemotron_evidence_href_root),
        render_nemotron_mtp_k_sweep_section(
            evidence_href_root=nemotron_k_sweep_evidence_href_root
        ),
        render_nemotron_mtp_osl16k_full_section(
            evidence_href_root=nemotron_osl16k_evidence_href_root
        ),
        render_sync_speedbench_status_section(),
        related_vllm_reports_section(),
        "<section class=\"section\"><h2>Key Findings</h2><p>" + esc(key_finding) + "</p><p class=\"note\">Speedups are computed only when a matched baseline exists with the same domain, model, temperature, batch size, ISL and OSL.</p></section>",
        render_profile_section(native_rows),
        render_dflare_section(dflare_rows),
        render_dflare_status_section(dflare_status),
        "<section class=\"section\"><h2>Task 5 Data Artifacts</h2><p class=\"note\"><code>vllm024_profiles_latest.csv</code>, <code>dflare_completed_latest.csv</code>, and <code>dflare_job_status_latest.csv</code> are published under <code>public/data</code> and linked from the report index.</p></section>",
        charts_section(added),
        temp_trends_section(),
        "<section class=\"section\"><h2>PARD / PARD-2 K=16 Focus</h2><div class=\"table-wrap\">",
        table(
            focus,
            [
                ("domain", "Domain", "text"),
                ("model", "Model", "text"),
                ("temperature", "Temp", "temp"),
                ("batch_size", "Batch", "int"),
                ("method", "Method", "text"),
                ("state", "State", "text"),
                ("tok_s_gpu", "tok/s/GPU", "num"),
                ("speedup", "Speedup", "x"),
                ("acceptance_pct", "Acceptance", "pct"),
                ("mean_accept_len", "Mean len", "num"),
                ("basis", "Basis", "text"),
            ],
        ),
        "</div></section>",
        "<section class=\"section\"><h2>Main 6/19 Batch-Speedup Matrix</h2><p class=\"note\">This is the existing baseline/reference matrix kept for continuity.</p>",
        "<div class=\"table-wrap\">",
        table(
            main_matrix,
            [
                ("domain", "Domain", "text"),
                ("temperature", "Temp", "temp"),
                ("model", "Model", "text"),
                ("method", "Method", "text"),
                ("batch_1_speedup", "B1", "x"),
                ("batch_2_speedup", "B2", "x"),
                ("batch_4_speedup", "B4", "x"),
                ("batch_8_speedup", "B8", "x"),
                ("batch_16_speedup", "B16", "x"),
                ("batch_32_speedup", "B32", "x"),
            ],
        ),
        "</div></section>",
        "<section class=\"section\"><h2>Added And Legacy Results Summary</h2><div class=\"table-wrap\">",
        table(
            added_summary,
            [
                ("domain", "Domain", "text"),
                ("temperature", "Temp", "temp"),
                ("model", "Model", "text"),
                ("method", "Method", "text"),
                ("rows", "Rows", "int"),
                ("batches", "Batches", "text"),
                ("tok_s_gpu", "tok/s/GPU", "num"),
                ("speedup", "Speedup", "x"),
                ("acceptance_pct", "Acceptance", "pct"),
                ("mean_accept_len", "Mean len", "num"),
                ("source_label", "Source", "text"),
            ],
        ),
        "</div></section>",
        "<section class=\"section\"><h2>Added And Legacy Speedup Matrix</h2><div class=\"table-wrap\">",
        table(
            added_matrix,
            [
                ("domain", "Domain", "text"),
                ("temperature", "Temp", "temp"),
                ("model", "Model", "text"),
                ("method", "Method", "text"),
                ("b1_speedup", "B1", "x"),
                ("b2_speedup", "B2", "x"),
                ("b4_speedup", "B4", "x"),
                ("b8_speedup", "B8", "x"),
                ("b16_speedup", "B16", "x"),
                ("b32_speedup", "B32", "x"),
            ],
        ),
        "</div></section>",
        '<section class="section"><details class="archive-table"><summary>Rows Waiting For Matched Baseline</summary><p class="note">These rows have valid throughput/acceptance measurements but no exact baseline with the same domain, model, temperature, batch size, ISL, and OSL. They are excluded from speedup-focused interpretation until a baseline exists.</p><div class="table-wrap">',
        table(
            unmatched,
            [
                ("domain", "Domain", "text"),
                ("model", "Model", "text"),
                ("temperature", "Temp", "temp"),
                ("batch_size", "Batch", "int"),
                ("isl", "ISL", "int"),
                ("osl", "OSL", "int"),
                ("method", "Method", "text"),
                ("state", "State", "text"),
                ("tok_s_gpu", "tok/s/GPU", "num"),
                ("acceptance_pct", "Acceptance", "pct"),
                ("mean_accept_len", "Mean len", "num"),
                ("basis", "Basis", "text"),
                ("source", "Source", "text"),
            ],
        ),
        "</div></details></section>",
        '<section class="section"><details class="archive-table"><summary>Failed Or Invalid Added Rows</summary><div class="table-wrap">',
        table(
            failed,
            [
                ("domain", "Domain", "text"),
                ("model", "Model", "text"),
                ("temperature", "Temp", "temp"),
                ("batch_size", "Batch", "int"),
                ("method", "Method", "text"),
                ("job_id", "Job", "text"),
                ("state", "State", "text"),
                ("basis", "Basis", "text"),
                ("source_label", "Source", "text"),
            ],
        ),
        "</div></details></section>",
        "<section class=\"section\"><h2>Sources</h2><p class=\"note\"><code>docs/vllm_standalone_all_batches_combined_20260619.csv</code>, <code>docs/vllm_standalone_all_batches_combined_matrix_20260619.csv</code>, <code>docs/vllm_standalone_temp0_temp1_trends_20260616.csv</code>, <code>docs/vllm_standalone_qwen30_qwen8_legacy_breakdowns_20260625.csv</code>, <code>docs/oci_qmath_extra_k_live_log_metrics_20260620.csv</code>, <code>docs/oci_qmath_pard2_k_sweep_live_log_metrics_20260620.csv</code>, <code>docs/oci_qmath_pard_pard2_k16_focus_live_log_metrics_20260620.csv</code>, <code>docs/lyris_qwen235b_swe_pard2_k_sweep_live_log_metrics_20260620.csv</code>, and <code>docs/qwen3_235b_dflash_retry28_openmath_metrics.csv</code>.</p></section>",
        "</main></body></html>",
    ]
    return "\n".join(parts)


def load_dflare_status_rows(status_csv: Path = DFLARE_STATUS_CSV) -> pd.DataFrame:
    if not status_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(status_csv)


def build_dflare_completed_rows(
    *,
    completed_dirs: list[Path] | None = None,
    repository_root: Path = ROOT,
) -> pd.DataFrame:
    source_dirs = completed_dirs if completed_dirs is not None else DFLARE_COMPLETED_DIRS
    result_paths = (
        path
        for completed_dir in source_dirs
        for path in completed_dir.glob("**/result.json")
    )
    return match_dflare_baselines(
        relativize_sources(
            target_profile_rows(load_completed_dflare_results(result_paths)),
            repository_root,
        )
    )


def build_latest_vllm_outputs(
    *,
    output_html: Path = VLLM_HTML_LATEST,
    added_csv_out: Path = VLLM_ADDED_OUT,
    completed_csv_out: Path = DFLARE_COMPLETED_OUT,
    public_data_dir: Path = PUBLIC_DATA,
    status_csv: Path = DFLARE_STATUS_CSV,
    profile_csv: Path = VLLM024_PROFILE_CSV,
    nemotron_evidence_href_root: str | None = None,
    nemotron_k_sweep_evidence_href_root: str | None = None,
    nemotron_osl16k_evidence_href_root: str | None = None,
) -> Path:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    added_csv_out.parent.mkdir(parents=True, exist_ok=True)
    completed_csv_out.parent.mkdir(parents=True, exist_ok=True)
    main_vllm = pd.read_csv(MAIN_VLLM)
    added = load_vllm_added(main_vllm)
    added.to_csv(added_csv_out, index=False)
    native_rows = pd.read_csv(profile_csv) if profile_csv.exists() else pd.DataFrame()
    dflare_status = load_dflare_status_rows(status_csv)
    dflare_rows = build_dflare_completed_rows()
    dflare_rows.to_csv(completed_csv_out, index=False)
    publish_public_data(profile_csv, public_data_dir)
    publish_public_data(completed_csv_out, public_data_dir)
    publish_public_data(status_csv, public_data_dir)
    publish_public_data(PERFCFG_DYNAMIC_REPLAY_CSV, public_data_dir)
    publish_nemotron_mtp_legacy_smoke_evidence(public_data_dir=public_data_dir)
    publish_nemotron_mtp_k_sweep_evidence(public_data_dir=public_data_dir)
    publish_nemotron_mtp_osl16k_full_evidence(public_data_dir=public_data_dir)
    vllm_html = build_vllm_html(
        main_vllm,
        added,
        native_rows,
        dflare_rows,
        dflare_status,
        nemotron_evidence_href_root,
        nemotron_k_sweep_evidence_href_root,
        nemotron_osl16k_evidence_href_root,
    )
    output_html.write_text(vllm_html, encoding="utf-8")
    return output_html


def load_sacct() -> pd.DataFrame:
    if not NEMORL_SACCT.exists():
        return pd.DataFrame(columns=["job_id", "job_name", "slurm_state", "exit_code", "elapsed", "start", "end"])
    rows = []
    for line in NEMORL_SACCT.read_text().splitlines():
        parts = line.split("|")
        if len(parts) < 7 or "." in parts[0]:
            continue
        rows.append(
            {
                "job_id": parts[0],
                "job_name": parts[1],
                "slurm_state": parts[2],
                "exit_code": parts[3],
                "elapsed": parts[4],
                "start": parts[5],
                "end": parts[6],
            }
        )
    return pd.DataFrame(rows)


def load_nemorl_manifest() -> pd.DataFrame:
    parts = []
    for path in NEMORL_MANIFESTS:
        raw = pd.read_csv(path)
        raw["manifest"] = str(path.relative_to(ROOT))
        parts.append(raw)
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True)
    rows["job_id"] = rows["job_id"].astype(str)
    rows = rows.drop_duplicates(subset=["job_id"], keep="last")
    return rows


def load_nemorl_summary() -> pd.DataFrame:
    parts = []
    for path in [NEMORL_SUMMARY, *NEMORL_ADDITIONAL_SUMMARIES]:
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw["summary_source"] = str(path.relative_to(ROOT))
        parts.append(raw)
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True, sort=False)
    rows["job_id"] = rows["job_id"].astype(str)
    return rows.drop_duplicates(subset=["job_id"], keep="last")


def nemorl_source_group_from_run_id(run_id: object) -> str:
    text = str(run_id).lower()
    if "20260624" in text:
        return "Lyris PerfCfg enforce_eager=false PARD diagnostics 2026-06-24"
    if "cudagraphoff" in text or "eagertrue" in text:
        return "Lyris PerfCfg CUDA-graph-disabled triton W&B matrix 2026-06-23"
    if "eagerfalse" in text:
        date = "2026-06-22" if "20260622" in text else "2026-06-23"
        return f"Lyris PerfCfg enforce_eager=false triton W&B matrix {date}"
    if "20260623" in text:
        return "Lyris PerfCfg triton W&B matrix 2026-06-23 (CUDA Graph state unknown)"
    return "Lyris Qwen235B PR2879 OSL8192 2026-06-21"


def nemorl_config_basis_from_run_id(run_id: object) -> str:
    text = str(run_id).lower()
    if "20260624" in text:
        return (
            "performance recipe default plus latest-main+PR2879 topology-aware fix; "
            "enforce_eager=false, MoE backend=triton, max_num_seqs=64, max_num_batched_tokens=32760/32768; PARD diagnostic sweep"
        )
    if "cudagraphoff" in text or "eagertrue" in text:
        return (
            "performance recipe default plus latest-main+PR2879 topology-aware fix; "
            "enforce_eager=true (CUDA graph disabled), MoE backend=triton, W&B enabled"
        )
    if "eagerfalse" in text:
        return (
            "performance recipe default plus latest-main+PR2879 topology-aware fix; "
            "enforce_eager=false, MoE backend=triton, max_num_seqs=64, max_num_batched_tokens=32768, W&B enabled"
        )
    return "performance recipe default, latest main plus PR2879 topology-aware fix"


def nemorl_enforce_eager(explicit: object, run_id: object) -> bool | str:
    explicit_text = text_value(explicit).lower()
    if explicit_text in {"true", "1", "yes"}:
        return True
    if explicit_text in {"false", "0", "no"}:
        return False
    run_text = str(run_id).lower()
    if "cudagraphoff" in run_text or "eagertrue" in run_text:
        return True
    if "eagerfalse" in run_text:
        return False
    return ""


def enrich_nemorl() -> pd.DataFrame:
    manifest = load_nemorl_manifest()
    summary = load_nemorl_summary()
    sacct = load_sacct()
    if manifest.empty:
        return pd.DataFrame()
    if not summary.empty:
        summary["job_id"] = summary["job_id"].astype(str)
    rows = manifest.merge(summary, on="job_id", how="left", suffixes=("", "_metric"))
    if not sacct.empty:
        rows = rows.merge(sacct, on="job_id", how="left")
    for col in ["wandb_enabled", "wandb_project", "wandb_name", "wandb_url"]:
        metric_col = f"{col}_metric"
        if col not in rows:
            rows[col] = ""
        if metric_col in rows:
            rows[col] = rows[col].where(rows[col].map(text_value).ne(""), rows[metric_col])
    rows["wandb_url"] = rows["wandb_url"].map(normalize_wandb_url)
    rows["method_k"] = rows.apply(lambda r: method_with_k(r.get("method"), r.get("num_speculative_tokens")), axis=1)
    rows["method_k"] = rows.apply(lambda r: refine_nemorl_method_from_run(r.get("method_k"), r.get("run_id")), axis=1)
    rows["model_name"] = rows["model"].map(model_name)
    rows["cluster"] = "lyris"
    rows["source_group"] = rows["run_id"].map(nemorl_source_group_from_run_id)
    rows["config_basis"] = rows["run_id"].map(nemorl_config_basis_from_run_id)
    rows["enforce_eager"] = rows.apply(
        lambda row: nemorl_enforce_eager(row.get("enforce_eager", ""), row.get("run_id", "")),
        axis=1,
    )
    rows["source_priority"] = 0
    if "slurm_state" not in rows:
        rows["slurm_state"] = ""
    rows["slurm_state"] = rows["slurm_state"].where(rows["slurm_state"].map(text_value).ne(""), "SUBMITTED")
    rows["completed_last_step"] = rows.apply(
        lambda r: (
            f"{int(clean_float(r.get('completed_steps')))}/{int(clean_float(r.get('last_step')))}"
            if not math.isnan(clean_float(r.get("completed_steps"))) and not math.isnan(clean_float(r.get("last_step")))
            else "0/0"
        ),
        axis=1,
    )
    for col in [
        "generation_worker_tokens_per_sec_per_gpu_mean",
        "e2e_tokens_per_sec_per_gpu_mean",
        "generation_time_s_mean",
        "total_step_time_s_mean",
    ]:
        rows[col] = pd.to_numeric(rows.get(col), errors="coerce")
    for col in [
        "gen_tps_speedup",
        "e2e_tps_speedup",
        "generation_time_speedup",
        "e2e_step_time_speedup",
    ]:
        rows[col] = math.nan
    group_cols = ["model", "mode", "max_steps", "max_new_tokens", "temperature", "top_p"]
    for _, idx in rows.groupby(group_cols, dropna=False).groups.items():
        sub = rows.loc[list(idx)]
        base = sub[sub["method"].astype(str) == "baseline"]
        if base.empty:
            continue
        base = base.iloc[0]
        base_gen = clean_float(base.get("generation_worker_tokens_per_sec_per_gpu_mean"))
        base_e2e = clean_float(base.get("e2e_tokens_per_sec_per_gpu_mean"))
        base_gen_time = clean_float(base.get("generation_time_s_mean"))
        base_step_time = clean_float(base.get("total_step_time_s_mean"))
        for row_idx in idx:
            gen = clean_float(rows.at[row_idx, "generation_worker_tokens_per_sec_per_gpu_mean"])
            e2e = clean_float(rows.at[row_idx, "e2e_tokens_per_sec_per_gpu_mean"])
            gen_time = clean_float(rows.at[row_idx, "generation_time_s_mean"])
            step_time = clean_float(rows.at[row_idx, "total_step_time_s_mean"])
            if not math.isnan(base_gen) and not math.isnan(gen) and base_gen:
                rows.at[row_idx, "gen_tps_speedup"] = gen / base_gen
            if not math.isnan(base_e2e) and not math.isnan(e2e) and base_e2e:
                rows.at[row_idx, "e2e_tps_speedup"] = e2e / base_e2e
            if not math.isnan(base_gen_time) and not math.isnan(gen_time) and gen_time:
                rows.at[row_idx, "generation_time_speedup"] = base_gen_time / gen_time
            if not math.isnan(base_step_time) and not math.isnan(step_time) and step_time:
                rows.at[row_idx, "e2e_step_time_speedup"] = base_step_time / step_time
    rows = rows.sort_values(["max_steps", "method", "job_id"], ascending=[False, True, True])
    return rows


def load_lyris_historical_nemorl() -> pd.DataFrame:
    rows = []
    for path, source_group, config_basis, source_priority in NEMORL_LYRIS_HISTORICAL_SOURCES:
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw = raw[raw["model"].astype(str).isin(["Qwen3-30B-A3B", "Qwen3-32B"])].copy()
        for _, row in raw.iterrows():
            completed, last = parse_completed_last(row.get("completed_last_step"))
            rows.append(
                {
                    "job_id": str(row.get("job_id", "")),
                    "model": str(row.get("model", "")),
                    "model_name": str(row.get("model", "")),
                    "mode": str(row.get("mode", "")),
                    "method": str(row.get("method", "")),
                    "method_k": normalize_nemorl_method(row.get("method"), row.get("label")),
                    "max_steps": 20,
                    "max_new_tokens": clean_float(row.get("max_osl")),
                    "temperature": clean_float(row.get("temperature")),
                    "top_p": clean_float(row.get("top_p")),
                    "enforce_eager": True,
                    "isl": row.get("isl", ""),
                    "cluster": "lyris",
                    "source_group": source_group,
                    "config_basis": config_basis,
                    "source_priority": source_priority,
                    "slurm_state": str(row.get("slurm_state", "")),
                    "exit_code": "",
                    "completed_steps": completed,
                    "last_step": last,
                    "completed_last_step": str(row.get("completed_last_step", "")),
                    "metric_state": str(row.get("metric_state", "")),
                    "total_step_time_s_mean": clean_float(row.get("e2e_step_time_s")),
                    "generation_time_s_mean": clean_float(row.get("generation_time_s")),
                    "e2e_tokens_per_sec_per_gpu_mean": clean_float(row.get("e2e_throughput_tok_s_gpu")),
                    "generation_worker_tokens_per_sec_per_gpu_mean": clean_float(row.get("generation_throughput_tok_s_gpu")),
                    "e2e_step_time_speedup": clean_float(row.get("e2e_step_time_speedup")),
                    "e2e_tps_speedup": clean_float(row.get("e2e_throughput_speedup")),
                    "generation_time_speedup": clean_float(row.get("generation_time_speedup")),
                    "gen_tps_speedup": clean_float(row.get("generation_throughput_speedup")),
                    "vllm_token_acceptance_pct": clean_float(row.get("acceptance_pct")),
                    "vllm_acceptance_length_mean_weighted_mean": clean_float(row.get("mean_accept_len")),
                    "manifest": str(path.relative_to(ROOT)),
                    "wandb_enabled": str(row.get("wandb_enabled", "")),
                    "wandb_project": str(row.get("wandb_project", "")),
                    "wandb_name": str(row.get("wandb_name", "")),
                    "wandb_url": normalize_wandb_url(row.get("wandb_url", "")),
                    "notes": str(row.get("notes", "")).strip(),
                    "log_path": str(row.get("source_log", "")),
                }
            )
    return pd.DataFrame(rows)


def load_oci_historical_nemorl() -> pd.DataFrame:
    if not NEMORL_OCI_HISTORICAL.exists():
        return pd.DataFrame()
    raw = pd.read_csv(NEMORL_OCI_HISTORICAL)
    raw = raw[
        raw["domain"].astype(str).eq("Math-RL")
        & raw["run_group"].astype(str).str.contains("step20 temp1 OSL1024", na=False)
        & raw["model"].astype(str).isin(["Qwen3-30B-A3B", "Qwen3-32B"])
    ].copy()
    rows = []
    for _, row in raw.iterrows():
        completed = clean_float(row.get("completed_steps"))
        max_steps = clean_float(row.get("max_steps"))
        rows.append(
            {
                "job_id": str(row.get("job_id", "")),
                "model": str(row.get("model", "")),
                "model_name": str(row.get("model", "")),
                "mode": "sync",
                "method": str(row.get("method", "")),
                "method_k": normalize_nemorl_method(row.get("method"), k=row.get("k")),
                "max_steps": max_steps,
                "max_new_tokens": clean_float(row.get("max_new_tokens")),
                "temperature": 1.0,
                "top_p": 1.0,
                "enforce_eager": True,
                "isl": "",
                "cluster": "oci-hsg",
                "source_group": "OCI-HSG Qwen30/Qwen32 Math-RL OSL1024 2026-06-16",
                "config_basis": str(row.get("config_basis", "")),
                "source_priority": 2,
                "slurm_state": str(row.get("state", "")),
                "exit_code": str(row.get("exit_code", "")),
                "completed_steps": completed,
                "last_step": clean_float(row.get("parsed_steps")),
                "completed_last_step": (
                    f"{int(completed)}/{int(max_steps)}"
                    if not math.isnan(completed) and not math.isnan(max_steps)
                    else ""
                ),
                "metric_state": str(row.get("metric_status", "")),
                "total_step_time_s_mean": clean_float(row.get("e2e_step_time_s")),
                "generation_time_s_mean": clean_float(row.get("generation_time_s")),
                "e2e_tokens_per_sec_per_gpu_mean": clean_float(row.get("e2e_tokens_per_sec_per_gpu")),
                "generation_worker_tokens_per_sec_per_gpu_mean": clean_float(row.get("generation_worker_tokens_per_sec_per_gpu")),
                "e2e_step_time_speedup": math.nan,
                "e2e_tps_speedup": clean_float(row.get("e2e_throughput_speedup")),
                "generation_time_speedup": math.nan,
                "gen_tps_speedup": clean_float(row.get("generation_throughput_speedup")),
                "vllm_token_acceptance_pct": clean_float(row.get("acceptance_rate_pct")),
                "vllm_acceptance_length_mean_weighted_mean": clean_float(row.get("mean_accepted_length")),
                "manifest": str(NEMORL_OCI_HISTORICAL.relative_to(ROOT)),
                "wandb_enabled": "false",
                "wandb_project": str(row.get("wandb_project", "")),
                "wandb_name": str(row.get("wandb_name", "")),
                "wandb_url": normalize_wandb_url(row.get("wandb_url", "")),
                "notes": str(row.get("notes", "")).strip(),
                "log_path": str(row.get("sources", "")),
            }
        )
    return pd.DataFrame(rows)


def load_lyris_live_k_sweep_nemorl() -> pd.DataFrame:
    summary = pd.read_csv(NEMORL_LIVE_K_SWEEP_SUMMARY) if NEMORL_LIVE_K_SWEEP_SUMMARY.exists() else pd.DataFrame()
    if not summary.empty:
        summary["job_id"] = summary["job_id"].astype(str)
        summary = summary.set_index("job_id")
    rows = []
    for meta in NEMORL_LIVE_K_SWEEP_META:
        job_id = meta["job_id"]
        metric = summary.loc[job_id].to_dict() if not summary.empty and job_id in summary.index else {}
        completed = clean_float(metric.get("completed_steps", meta.get("completed_steps", math.nan)))
        last = clean_float(metric.get("last_step", meta.get("last_step", math.nan)))
        if math.isnan(completed):
            completed = clean_float(meta.get("completed_steps", math.nan))
        if math.isnan(last):
            last = clean_float(meta.get("last_step", math.nan))
        completed_last = (
            f"{int(completed)}/20 last {int(last)}"
            if not math.isnan(completed) and not math.isnan(last) and last > 0
            else "0/20"
        )
        metric_state = str(meta.get("metric_state", ""))
        if metric and completed > 0 and metric_state in {"partial_live", ""}:
            metric_state = str(metric.get("partial_result_state", "partial_live"))
        latest_error = str(meta.get("error", "")).strip()
        if not latest_error:
            raw_error = metric.get("latest_error", "")
            if raw_error is not None and not pd.isna(raw_error):
                latest_error = str(raw_error).strip()
        rows.append(
            {
                "job_id": job_id,
                "model": meta["model"],
                "model_name": model_name(meta["model"]),
                "mode": meta["mode"],
                "method": f"Eagle-3 K={meta['k']}",
                "method_k": f"eagle3_k{meta['k']}",
                "max_steps": 20,
                "max_new_tokens": 4096,
                "temperature": 1.0,
                "top_p": 1.0,
                "enforce_eager": True,
                "isl": "performance recipe default",
                "cluster": "lyris",
                "source_group": NEMORL_LIVE_K_SWEEP_SOURCE_GROUP,
                "comparison_group": "Lyris Qwen30/Qwen32 PerfCfg OSL4096 latest-main+PR2879 2026-06-22",
                "config_basis": (
                    "performance recipe default plus latest-main+PR2879 topology-aware fix; "
                    "enforce_eager=true, prefix caching disabled, MoE backend=triton; "
                    f"context-clamp K sweep checked {NEMORL_LIVE_K_SWEEP_CHECKED_AT}"
                ),
                "source_priority": 0.5,
                "slurm_state": meta["slurm_state"],
                "exit_code": "",
                "elapsed": meta.get("elapsed", ""),
                "completed_steps": completed,
                "last_step": last,
                "completed_last_step": completed_last,
                "metric_state": metric_state,
                "total_step_time_s_mean": clean_float(metric.get("total_step_time_s_mean")),
                "generation_time_s_mean": clean_float(metric.get("generation_time_s_mean")),
                "e2e_tokens_per_sec_per_gpu_mean": clean_float(metric.get("e2e_tokens_per_sec_per_gpu_mean")),
                "generation_worker_tokens_per_sec_per_gpu_mean": clean_float(metric.get("generation_worker_tokens_per_sec_per_gpu_mean")),
                "e2e_step_time_speedup": math.nan,
                "e2e_tps_speedup": math.nan,
                "generation_time_speedup": math.nan,
                "gen_tps_speedup": math.nan,
                "vllm_token_acceptance_pct": clean_float(metric.get("vllm_token_acceptance_pct")),
                "vllm_acceptance_length_mean_weighted_mean": clean_float(metric.get("vllm_acceptance_length_mean_weighted_mean")),
                "manifest": str(NEMORL_LIVE_K_SWEEP_SUMMARY.relative_to(ROOT)) if NEMORL_LIVE_K_SWEEP_SUMMARY.exists() else "",
                "wandb_enabled": str(meta.get("wandb_enabled", metric.get("wandb_enabled", ""))),
                "wandb_project": str(meta.get("wandb_project", metric.get("wandb_project", ""))),
                "wandb_name": str(meta.get("wandb_name", metric.get("wandb_name", ""))),
                "wandb_url": normalize_wandb_url(meta.get("wandb_url", metric.get("wandb_url", ""))),
                "notes": str(meta.get("notes", "")).strip(),
                "latest_error": latest_error,
                "log_path": str(meta.get("log_path", metric.get("log_path", ""))),
                "nodes_x_gpus": meta.get("nodes_x_gpus", ""),
                "segment": meta.get("segment", ""),
            }
        )
    return pd.DataFrame(rows)


def load_nemorl_comparison_summaries() -> pd.DataFrame:
    rows = []
    for path in NEMORL_COMPARISON_SUMMARIES:
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        for _, row in raw.iterrows():
            method = row.get("method", "")
            job_id = str(row.get("job_id", ""))
            if not job_id or job_id.lower() == "nan":
                continue
            completed = row.get("completed", row.get("steps", ""))
            completed_steps, last_step = parse_completed_last(completed)
            max_steps = clean_float(row.get("max_steps"))
            if math.isnan(max_steps):
                max_steps = 20
            max_osl = clean_float(row.get("max_osl"))
            if math.isnan(max_osl):
                max_osl = 4096
            model_text = text_value(row.get("model", ""))
            if not model_text and "qwen32" in path.name.lower():
                model = "Qwen3-32B"
            else:
                model = model_name(model_text)
            source_group = "Lyris PerfCfg enforce_eager=false PARD diagnostics 2026-06-24"
            status = first_text(row, "status")
            rows.append(
                {
                    "job_id": job_id,
                    "model": model,
                    "model_name": model,
                    "mode": str(row.get("mode", "sync") or "sync"),
                    "method": str(method),
                    "method_k": normalize_nemorl_diagnostic_method(method),
                    "max_steps": max_steps,
                    "max_new_tokens": max_osl,
                    "temperature": clean_float(row.get("temp", 1.0)),
                    "top_p": clean_float(row.get("top_p", 1.0)),
                    "enforce_eager": row.get("enforce_eager", False),
                    "isl": "",
                    "cluster": "lyris",
                    "source_group": source_group,
                    "comparison_group": source_group,
                    "config_basis": (
                        "performance recipe default plus latest-main+PR2879 topology-aware fix; "
                        "enforce_eager=false, MoE backend=triton; diagnostic CSV with precomputed baseline-relative speedups"
                    ),
                    "source_priority": 0.25,
                    "slurm_state": status,
                    "exit_code": "",
                    "completed_steps": completed_steps,
                    "last_step": last_step,
                    "completed_last_step": str(completed),
                    "metric_state": status,
                    "total_step_time_s_mean": clean_float(row.get("e2e_step_time_s")),
                    "generation_time_s_mean": clean_float(row.get("generation_time_s")),
                    "e2e_tokens_per_sec_per_gpu_mean": clean_float(row.get("e2e_tps_gpu")),
                    "generation_worker_tokens_per_sec_per_gpu_mean": clean_float(row.get("generation_tps_gpu")),
                    "e2e_step_time_speedup": clean_float(row.get("e2e_step_time_vs_baseline_speedup", row.get("e2e_step_time_speedup"))),
                    "e2e_tps_speedup": clean_float(row.get("e2e_tps_vs_baseline_speedup", row.get("e2e_throughput_speedup"))),
                    "generation_time_speedup": clean_float(row.get("generation_time_vs_baseline_speedup", row.get("generation_time_speedup"))),
                    "gen_tps_speedup": clean_float(row.get("generation_tps_vs_baseline_speedup", row.get("generation_throughput_speedup"))),
                    "vllm_token_acceptance_pct": clean_float(row.get("acceptance_pct")),
                    "vllm_acceptance_length_mean_weighted_mean": clean_float(row.get("mean_accept_len")),
                    "manifest": str(path.relative_to(ROOT)),
                    "wandb_enabled": "true" if normalize_wandb_url(row.get("wandb_url", row.get("wandb_or_run", ""))) else "",
                    "wandb_project": "",
                    "wandb_name": "",
                    "wandb_url": normalize_wandb_url(row.get("wandb_url", row.get("wandb_or_run", ""))),
                    "notes": str(row.get("action_note", "")),
                    "latest_error": "",
                    "log_path": str(row.get("source", "")),
                }
            )
    return pd.DataFrame(rows)


def normalize_nemorl_result_state(
    state: object,
    completed_steps: object,
    max_steps: object,
    notes: object,
) -> str:
    state_text = text_value(state).lower()
    notes_text = text_value(notes).lower()
    completed = clean_float(completed_steps)
    if state_text:
        if "held" in state_text:
            return "held"
        if "partial" in state_text:
            return "partial"
        if "timeout" in state_text:
            return "partial" if not math.isnan(completed) and completed > 0 else "failed"
        if any(token in state_text for token in ["fail", "error", "cancel", "oom"]):
            return "failed"
        if "complete" in state_text:
            return "completed"
        if state_text in {"pending", "submitted", "running"}:
            return state_text

    if "held" in notes_text:
        return "held"
    if "partial" in notes_text or ("timeout" in notes_text and not math.isnan(completed) and completed > 0):
        return "partial"
    if any(token in notes_text for token in ["fail", "error", "cancel", "oom"]):
        return "failed"
    maximum = clean_float(max_steps)
    if not math.isnan(completed) and not math.isnan(maximum) and completed >= maximum - 1:
        return "completed"
    if not math.isnan(completed) and completed > 0:
        return "partial"
    return state_text or "unknown"


def first_float(row: pd.Series, *keys: str) -> float:
    for key in keys:
        value = clean_float(row.get(key))
        if not math.isnan(value):
            return value
    return math.nan


def optional_bool(value: object) -> bool | str:
    text = text_value(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return ""


def nemorl_step_span(row: pd.Series) -> tuple[str, float, float]:
    candidates = [
        first_text(row, "matched_step_span", "completed_step_span", "step_filter"),
        text_value(row.get("completed_steps")),
    ]
    for candidate in candidates:
        match = re.search(r"(?:steps?)?\s*(\d+)\s*-\s*(\d+)", candidate, re.IGNORECASE)
        if match:
            first = int(match.group(1))
            last = int(match.group(2))
            return f"{first}-{last}", float(last - first + 1), float(last)
    completed = clean_float(row.get("completed_steps"))
    return "", completed, math.nan


def july_source_value(row: pd.Series, source: dict[str, object], key: str) -> object:
    value = row.get(key)
    if text_value(value):
        return value
    return source.get(key, "")


def load_july_nemorl_results(
    sources: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in sources or NEMORL_JULY_SOURCES:
        path = Path(source["path"])
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        source_group = str(source["source_group"])
        manifest = str(path.relative_to(ROOT))
        for source_row_index, (_, row) in enumerate(raw.iterrows(), start=2):
            method = first_text(row, "method", "variant") or "baseline"
            k = first_float(row, "k", "num_speculative_tokens")
            method_k = normalize_nemorl_method(method, k=k)
            model = first_text(row, "model") or str(source.get("model", ""))
            mode = first_text(row, "mode") or str(source.get("mode", ""))
            span, completed_steps, last_step = nemorl_step_span(row)
            max_steps = first_float(row, "max_steps")
            notes = first_text(row, "notes")
            raw_state = first_text(row, "state", "slurm_state")
            result_state = normalize_nemorl_result_state(raw_state, completed_steps, max_steps, notes)
            if raw_state:
                slurm_state = raw_state
            elif result_state == "partial" and "timeout" in notes.lower():
                slurm_state = "TIMEOUT_PARTIAL"
            else:
                slurm_state = result_state.upper()

            num_nodes = clean_float(july_source_value(row, source, "num_nodes"))
            gpus_per_node = clean_float(july_source_value(row, source, "gpus_per_node"))
            nodes_x_gpus = (
                f"{int(num_nodes)}x{int(gpus_per_node)}"
                if not math.isnan(num_nodes) and not math.isnan(gpus_per_node)
                else ""
            )
            segment = first_float(row, "slurm_segment", "segment")
            if math.isnan(segment):
                segment = clean_float(source.get("segment"))
            config_segment_size = first_float(row, "config_segment_size")
            if math.isnan(config_segment_size):
                config_segment_size = clean_float(source.get("config_segment_size"))

            target_tp = first_float(row, "target_tensor_parallel_size")
            if math.isnan(target_tp):
                target_tp = clean_float(source.get("target_tensor_parallel_size"))
            draft_tp = first_float(row, "draft_tensor_parallel_size")
            if method_k != "baseline" and math.isnan(draft_tp):
                by_method = source.get("draft_tensor_parallel_size_by_method", {})
                if isinstance(by_method, dict):
                    draft_tp = clean_float(by_method.get(method.lower()))
                if math.isnan(draft_tp):
                    draft_tp = clean_float(source.get("draft_tensor_parallel_size"))

            enforce_eager = optional_bool(july_source_value(row, source, "enforce_eager"))
            fuse_allreduce_rms = optional_bool(july_source_value(row, source, "fuse_allreduce_rms"))
            attention_backend = text_value(july_source_value(row, source, "attention_backend"))
            moe_backend = text_value(july_source_value(row, source, "moe_backend"))
            max_num_seqs = first_float(row, "max_num_seqs")
            if math.isnan(max_num_seqs):
                max_num_seqs = clean_float(source.get("max_num_seqs"))
            max_num_batched_tokens = first_float(row, "max_num_batched_tokens")
            if math.isnan(max_num_batched_tokens):
                max_num_batched_tokens = clean_float(source.get("max_num_batched_tokens"))
            wandb_url = normalize_wandb_url(row.get("wandb_url", ""))
            cohort = first_text(row, "cohort") or str(source.get("cohort", "standard"))
            baseline_job_id = first_text(row, "baseline_job_id", "matched_baseline_job_id")
            source_speedups = [
                first_float(row, "e2e_step_time_speedup"),
                first_float(row, "e2e_throughput_speedup"),
                first_float(row, "generation_time_speedup"),
                first_float(row, "generation_throughput_speedup"),
            ]
            if method_k == "baseline":
                baseline_match_state = "baseline"
            elif any(not math.isnan(value) for value in source_speedups):
                baseline_match_state = "precomputed"
            else:
                baseline_match_state = "unmatched_baseline"
            normalized.append(
                {
                    "job_id": text_value(row.get("job_id")),
                    "baseline_job_id": baseline_job_id,
                    "model": model,
                    "model_name": model_name(model),
                    "mode": mode,
                    "method": method,
                    "method_k": method_k,
                    "k": k,
                    "max_steps": max_steps,
                    "max_new_tokens": first_float(row, "max_osl", "max_new_tokens"),
                    "temperature": first_float(row, "temperature"),
                    "top_p": first_float(row, "top_p"),
                    "enforce_eager": enforce_eager,
                    "isl": first_text(row, "isl"),
                    "cluster": str(source["cluster"]),
                    "source_group": source_group,
                    "comparison_group": source_group,
                    "config_basis": (
                        "normalized July performance-recipe result; "
                        f"resource={nodes_x_gpus or 'not recorded'}, segment={fmt(segment, 0)}, "
                        f"attention={attention_backend or 'not recorded'}, MoE={moe_backend or 'not recorded'}"
                    ),
                    "source_priority": -10,
                    "slurm_state": slurm_state,
                    "raw_state": raw_state,
                    "result_state": result_state,
                    "metric_state": result_state,
                    "baseline_match_state": baseline_match_state,
                    "strict_match_eligible": True,
                    "evidence_period": "july-current",
                    "canonical_snapshot": "",
                    "completed_steps": completed_steps,
                    "last_step": last_step,
                    "completed_step_span": span,
                    "step_filter": first_text(row, "step_filter", "matched_step_span") or span,
                    "completed_last_step": (
                        f"{int(completed_steps)}/{int(max_steps)} last {int(last_step)}"
                        if not any(math.isnan(value) for value in [completed_steps, max_steps, last_step])
                        else ""
                    ),
                    "total_step_time_s_mean": first_float(row, "e2e_step_time_s_mean", "e2e_step_time_s"),
                    "generation_time_s_mean": first_float(row, "generation_time_s_mean", "generation_time_s"),
                    "e2e_tokens_per_sec_per_gpu_mean": first_float(
                        row,
                        "e2e_tokens_per_sec_per_gpu_mean",
                        "e2e_throughput_tok_s_gpu",
                        "e2e_tokens_per_sec_per_gpu",
                    ),
                    "generation_worker_tokens_per_sec_per_gpu_mean": first_float(
                        row,
                        "generation_worker_tokens_per_sec_per_gpu_mean",
                        "generation_throughput_tok_s_gpu",
                        "generation_tokens_per_sec_per_gpu",
                    ),
                    "e2e_step_time_speedup": first_float(row, "e2e_step_time_speedup"),
                    "e2e_tps_speedup": first_float(row, "e2e_throughput_speedup"),
                    "generation_time_speedup": first_float(row, "generation_time_speedup"),
                    "gen_tps_speedup": first_float(row, "generation_throughput_speedup"),
                    "exposed_generation_time_s_mean": first_float(row, "exposed_generation_time_s_mean"),
                    "exposed_generation_time_speedup": first_float(row, "exposed_generation_time_speedup"),
                    "vllm_token_acceptance_pct": first_float(row, "acceptance_rate_pct"),
                    "vllm_acceptance_length_mean_weighted_mean": first_float(
                        row,
                        "mean_accepted_length",
                        "mean_accept_len",
                        "mean_accept_length",
                    ),
                    "mean_generation_length": first_float(row, "mean_generation_length"),
                    "avg_reward_mean": first_float(row, "avg_reward_mean", "avg_reward"),
                    "generation_kl_error_mean": first_float(
                        row,
                        "generation_kl_error_mean",
                        "generation_kl_error",
                    ),
                    "error_count": first_float(row, "error_count"),
                    "manifest": manifest,
                    "source_file": manifest,
                    "source_row_index": source_row_index,
                    "wandb_enabled": "true" if wandb_url else "",
                    "wandb_project": "",
                    "wandb_name": "",
                    "wandb_url": wandb_url,
                    "basis": first_text(row, "basis"),
                    "notes": notes,
                    "log_path": "",
                    "num_nodes": num_nodes,
                    "gpus_per_node": gpus_per_node,
                    "nodes_x_gpus": nodes_x_gpus,
                    "resource_shape": nodes_x_gpus,
                    "segment": segment,
                    "config_segment_size": config_segment_size,
                    "target_tensor_parallel_size": target_tp,
                    "draft_tensor_parallel_size": draft_tp,
                    "attention_backend": attention_backend,
                    "moe_backend": moe_backend,
                    "max_num_seqs": max_num_seqs,
                    "max_num_batched_tokens": max_num_batched_tokens,
                    "cohort": cohort,
                    "fuse_allreduce_rms": fuse_allreduce_rms,
                    "num_prompts_per_step": first_float(row, "num_prompts_per_step"),
                    "num_generations_per_prompt": first_float(row, "num_generations_per_prompt"),
                }
            )
    rows = pd.DataFrame(normalized)
    if not rows.empty:
        rows["cuda_graph_state"] = rows.apply(nemorl_cuda_graph_label, axis=1)
        rows["metric_window"] = rows.apply(nemorl_metric_window, axis=1)
    return rows


def load_nemorl_prejuly_canonical() -> pd.DataFrame:
    path = NEMORL_PREJULY_CANONICAL
    if not path.is_file():
        raise FileNotFoundError(f"required NeMo-RL canonical snapshot is missing: {path}")

    rows = pd.read_csv(path)
    required_columns = {"job_id", "method_k"}
    missing_columns = required_columns - set(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"NeMo-RL canonical snapshot is missing columns: {missing}")

    rows["strict_match_eligible"] = False
    speedup_columns = [
        "gen_tps_speedup",
        "e2e_tps_speedup",
        "generation_time_speedup",
        "e2e_step_time_speedup",
    ]
    for column in speedup_columns:
        if column not in rows:
            rows[column] = math.nan
    has_precomputed_speedup = rows[speedup_columns].notna().any(axis=1)
    is_baseline = rows["method_k"].astype(str).eq("baseline")
    rows["baseline_match_state"] = "unmatched_baseline"
    rows.loc[has_precomputed_speedup, "baseline_match_state"] = "precomputed"
    rows.loc[is_baseline, "baseline_match_state"] = "baseline"
    rows["evidence_period"] = "pre-july-canonical"
    rows["canonical_snapshot"] = str(path.relative_to(ROOT))
    return rows


def nemorl_baseline_is_usable(row: pd.Series) -> bool:
    if text_value(row.get("result_state")).lower() != "completed":
        return False
    required = [
        clean_float(row.get("generation_worker_tokens_per_sec_per_gpu_mean")),
        clean_float(row.get("e2e_tokens_per_sec_per_gpu_mean")),
        clean_float(row.get("total_step_time_s_mean")),
    ]
    if text_value(row.get("mode")).lower() != "async-1off":
        required.append(clean_float(row.get("generation_time_s_mean")))
    return all(not math.isnan(value) and value > 0 for value in required)


def fill_nemorl_speedups(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    if "source_group" not in rows:
        rows["source_group"] = ""
    if "comparison_group" not in rows:
        rows["comparison_group"] = ""
    comparison_group = rows["comparison_group"].map(text_value)
    rows["_comparison_group"] = comparison_group.where(comparison_group.ne(""), rows["source_group"])
    metric_cols = [
        "generation_worker_tokens_per_sec_per_gpu_mean",
        "e2e_tokens_per_sec_per_gpu_mean",
        "generation_time_s_mean",
        "total_step_time_s_mean",
    ]
    speedup_cols = [
        "gen_tps_speedup",
        "e2e_tps_speedup",
        "generation_time_speedup",
        "e2e_step_time_speedup",
    ]
    for col in [*metric_cols, *speedup_cols]:
        if col not in rows:
            rows[col] = math.nan
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    if "result_state" not in rows:
        rows["result_state"] = "unknown"
    if "baseline_match_state" not in rows:
        rows["baseline_match_state"] = ""
    existing_match_state = rows["baseline_match_state"].map(text_value)
    fallback_match_state = pd.Series("unmatched_baseline", index=rows.index, dtype=object)
    baseline_mask = rows["method_k"].astype(str).eq("baseline")
    precomputed_mask = rows[speedup_cols].notna().any(axis=1)
    fallback_match_state.loc[baseline_mask] = "baseline"
    fallback_match_state.loc[~baseline_mask & precomputed_mask] = "precomputed"
    rows["baseline_match_state"] = existing_match_state.where(
        existing_match_state.ne(""),
        fallback_match_state,
    )

    string_match_cols = [
        "model_name",
        "mode",
        "cluster",
        "nodes_x_gpus",
        "attention_backend",
        "moe_backend",
    ]
    for col in string_match_cols:
        if col not in rows:
            rows[col] = ""
        rows[f"_match_{col}"] = rows[col].map(lambda value: text_value(value).lower())
    numeric_match_cols = [
        "max_steps",
        "max_new_tokens",
        "temperature",
        "top_p",
        "target_tensor_parallel_size",
        "max_num_seqs",
        "max_num_batched_tokens",
        "segment",
        "config_segment_size",
    ]
    for col in numeric_match_cols:
        if col not in rows:
            rows[col] = math.nan
        rows[f"_match_{col}"] = pd.to_numeric(rows[col], errors="coerce")
    if "fuse_allreduce_rms" not in rows:
        rows["fuse_allreduce_rms"] = ""
    rows["_match_fuse_allreduce_rms"] = rows["fuse_allreduce_rms"].map(optional_bool).map(str)
    rows["_match_cuda_graph"] = rows.apply(nemorl_cuda_graph_label, axis=1)
    if "strict_match_eligible" not in rows:
        rows["strict_match_eligible"] = False
    rows["_strict_declared"] = rows["strict_match_eligible"].map(
        lambda value: optional_bool(value) is True
    )

    strict_complete = rows["_comparison_group"].map(text_value).ne("")
    for col in string_match_cols:
        strict_complete &= rows[f"_match_{col}"].ne("")
    for col in numeric_match_cols:
        strict_complete &= rows[f"_match_{col}"].notna()
    strict_complete &= rows["_match_cuda_graph"].ne("CG-unknown")
    strict_complete &= rows["_match_fuse_allreduce_rms"].ne("")
    rows["strict_match_ready"] = rows["_strict_declared"] & strict_complete

    group_cols = [
        "_comparison_group",
        "_match_model_name",
        "_match_mode",
        "_match_max_steps",
        "_match_max_new_tokens",
        "_match_temperature",
        "_match_top_p",
        "_match_cuda_graph",
        "_match_cluster",
        "_match_nodes_x_gpus",
        "_match_attention_backend",
        "_match_moe_backend",
        "_match_target_tensor_parallel_size",
        "_match_max_num_seqs",
        "_match_max_num_batched_tokens",
        "_match_segment",
        "_match_config_segment_size",
        "_match_fuse_allreduce_rms",
    ]
    strict_rows = rows[rows["strict_match_ready"]]
    for _, idx in strict_rows.groupby(group_cols, dropna=False).groups.items():
        sub = rows.loc[list(idx)]
        baseline_indices = sub[sub["method_k"].astype(str) == "baseline"].index
        spec_indices = sub[sub["method_k"].astype(str) != "baseline"].index
        if baseline_indices.empty:
            rows.loc[spec_indices, speedup_cols] = math.nan
            rows.loc[spec_indices, "baseline_match_state"] = "unmatched_baseline"
            continue
        baselines = sub.loc[baseline_indices].copy()
        usable_baselines = baselines[baselines.apply(nemorl_baseline_is_usable, axis=1)].copy()
        rows.loc[baseline_indices, "baseline_match_state"] = "unusable_baseline"
        if usable_baselines.empty:
            rows.loc[spec_indices, speedup_cols] = math.nan
            rows.loc[spec_indices, "baseline_match_state"] = "unmatched_baseline"
            continue
        usable_baselines["_completed_sort"] = (
            pd.to_numeric(usable_baselines["completed_steps"], errors="coerce").fillna(0)
            if "completed_steps" in usable_baselines
            else 0
        )
        base = usable_baselines.sort_values("_completed_sort", ascending=False).iloc[0]
        rows.loc[usable_baselines.index, "baseline_match_state"] = "baseline"
        rows.loc[spec_indices, "baseline_match_state"] = "matched"
        base_gen = clean_float(base.get("generation_worker_tokens_per_sec_per_gpu_mean"))
        base_e2e = clean_float(base.get("e2e_tokens_per_sec_per_gpu_mean"))
        base_gen_time = clean_float(base.get("generation_time_s_mean"))
        base_step_time = clean_float(base.get("total_step_time_s_mean"))
        for row_idx in idx:
            gen = clean_float(rows.at[row_idx, "generation_worker_tokens_per_sec_per_gpu_mean"])
            e2e = clean_float(rows.at[row_idx, "e2e_tokens_per_sec_per_gpu_mean"])
            gen_time = clean_float(rows.at[row_idx, "generation_time_s_mean"])
            step_time = clean_float(rows.at[row_idx, "total_step_time_s_mean"])
            if not math.isnan(base_gen) and not math.isnan(gen) and base_gen and math.isnan(clean_float(rows.at[row_idx, "gen_tps_speedup"])):
                rows.at[row_idx, "gen_tps_speedup"] = gen / base_gen
            if not math.isnan(base_e2e) and not math.isnan(e2e) and base_e2e and math.isnan(clean_float(rows.at[row_idx, "e2e_tps_speedup"])):
                rows.at[row_idx, "e2e_tps_speedup"] = e2e / base_e2e
            if not math.isnan(base_gen_time) and not math.isnan(gen_time) and gen_time and math.isnan(clean_float(rows.at[row_idx, "generation_time_speedup"])):
                rows.at[row_idx, "generation_time_speedup"] = base_gen_time / gen_time
            if not math.isnan(base_step_time) and not math.isnan(step_time) and step_time and math.isnan(clean_float(rows.at[row_idx, "e2e_step_time_speedup"])):
                rows.at[row_idx, "e2e_step_time_speedup"] = base_step_time / step_time
    temporary_cols = [
        "_comparison_group",
        "_strict_declared",
        *[col for col in rows.columns if col.startswith("_match_")],
    ]
    return rows.drop(columns=temporary_cols, errors="ignore")


def deduplicate_nemorl_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    defaults: dict[str, object] = {
        "job_id": "",
        "method_k": "",
        "source_group": "",
        "source_priority": 999,
        "completed_steps": 0,
        "result_state": "unknown",
        "gen_tps_speedup": math.nan,
        "manifest": "",
        "notes": "",
        "evidence_period": "",
    }
    for col, default in defaults.items():
        if col not in rows:
            rows[col] = default
    rows["has_speedup_metric"] = pd.to_numeric(rows["gen_tps_speedup"], errors="coerce").notna()
    rows["completed_steps_numeric"] = pd.to_numeric(rows["completed_steps"], errors="coerce").fillna(0)
    rows["source_priority_numeric"] = pd.to_numeric(rows["source_priority"], errors="coerce").fillna(999)
    rows["job_id_text"] = rows["job_id"].astype(str)
    rows["method_k_text"] = rows["method_k"].astype(str)
    rows["has_job_key"] = rows["job_id_text"].str.strip().ne("") & rows["job_id_text"].str.lower().ne("nan")

    def dedup_result_state(row: pd.Series) -> str:
        existing = text_value(row.get("result_state")).lower()
        if existing and existing != "unknown":
            return existing
        return normalize_nemorl_result_state(
            row.get("slurm_state"),
            row.get("completed_steps"),
            row.get("max_steps"),
            row.get("notes"),
        )

    rows["result_state"] = rows.apply(dedup_result_state, axis=1)
    state_rank = {
        "completed": 0,
        "partial": 1,
        "running": 2,
        "pending": 3,
        "submitted": 3,
        "failed": 4,
        "held": 5,
        "unknown": 6,
    }
    rows["result_state_rank"] = rows["result_state"].map(
        lambda value: state_rank.get(text_value(value).lower(), 6)
    )

    def joined_alternates(values: pd.Series) -> str:
        unique = list(dict.fromkeys(text_value(value) for value in values if text_value(value)))
        return " | ".join(unique) if len(unique) > 1 else ""

    rows["alternate_source_groups"] = ""
    rows["alternate_manifests"] = ""
    if rows["has_job_key"].any():
        keyed = rows[rows["has_job_key"]]
        alt_sources = keyed.groupby(["job_id_text", "method_k_text"])["source_group"].apply(joined_alternates)
        alt_manifests = keyed.groupby(["job_id_text", "method_k_text"])["manifest"].apply(joined_alternates)

        def add_provenance(row: pd.Series) -> pd.Series:
            key = (str(row.get("job_id_text", "")), str(row.get("method_k_text", "")))
            source_provenance = text_value(alt_sources.get(key, ""))
            manifest_provenance = text_value(alt_manifests.get(key, ""))
            row["alternate_source_groups"] = source_provenance
            row["alternate_manifests"] = manifest_provenance
            notes = text_value(row.get("notes"))
            additions = []
            if source_provenance:
                additions.append(f"alternate source groups: {source_provenance}")
            if manifest_provenance:
                additions.append(f"alternate manifests: {manifest_provenance}")
            if additions:
                provenance_note = "; ".join(additions)
                row["notes"] = f"{notes}; {provenance_note}" if notes else provenance_note
            return row

        rows.loc[rows["has_job_key"]] = rows.loc[rows["has_job_key"]].apply(add_provenance, axis=1)

    rows = rows.sort_values(
        [
            "job_id_text",
            "method_k_text",
            "result_state_rank",
            "source_priority_numeric",
            "completed_steps_numeric",
            "has_speedup_metric",
        ],
        ascending=[True, True, True, True, False, False],
        na_position="last",
    )
    rows_with_job = rows[rows["has_job_key"]].drop_duplicates(
        subset=["job_id_text", "method_k_text"],
        keep="first",
    )
    rows_without_job = rows[~rows["has_job_key"]].drop_duplicates(
        subset=["source_group", "job_id", "method_k"],
        keep="first",
    )
    rows = pd.concat([rows_with_job, rows_without_job], ignore_index=True, sort=False)
    return rows.drop(
        columns=[
            "has_speedup_metric",
            "completed_steps_numeric",
            "source_priority_numeric",
            "job_id_text",
            "method_k_text",
            "has_job_key",
            "result_state_rank",
        ],
        errors="ignore",
    )


def combine_nemorl_rows(live_rows: pd.DataFrame) -> pd.DataFrame:
    parts = [
        part
        for part in [
            load_nemorl_prejuly_canonical(),
            load_july_nemorl_results(),
        ]
        if not part.empty
    ]
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True, sort=False)
    if "wandb_url" not in rows:
        rows["wandb_url"] = ""
    known_wandb_urls = rows["job_id"].astype(str).map(NEMORL_WANDB_URL_BY_JOB).fillna("")
    rows["wandb_url"] = rows["wandb_url"].map(normalize_wandb_url)
    rows["wandb_url"] = rows["wandb_url"].where(rows["wandb_url"].ne(""), known_wandb_urls)
    disabled_mask = rows["job_id"].astype(str).isin(NEMORL_CONFIRMED_WANDB_DISABLED_JOBS)
    rows.loc[disabled_mask, "wandb_enabled"] = "false"
    rows = fill_nemorl_speedups(rows)
    rows = deduplicate_nemorl_rows(rows)
    rows = rows.sort_values(
        [
            "source_priority",
            "model_name",
            "mode",
            "max_new_tokens",
            "method_k",
            "job_id",
        ],
        ascending=[True, True, True, True, True, True],
        na_position="last",
    )
    rows["metric_window"] = rows.apply(nemorl_metric_window, axis=1)
    rows["cuda_graph_state"] = rows.apply(nemorl_cuda_graph_label, axis=1)
    return rows


def nemorl_live_k_sweep_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "job_id" not in rows:
        return pd.DataFrame()
    ids = {item["job_id"] for item in NEMORL_LIVE_K_SWEEP_META}
    live = rows[rows["job_id"].astype(str).isin(ids)].copy()
    if live.empty:
        return live
    live["k_sort"] = live["method_k"].astype(str).str.extract(r"k(\d+)").astype(float)
    mode_rank = {"sync": 0, "async-1off": 1}
    live["mode_rank"] = live["mode"].astype(str).map(mode_rank).fillna(9)
    live["model_rank"] = live["model_name"].astype(str).map({"Qwen3-30B-A3B": 0, "Qwen3-32B": 1}).fillna(9)
    return live.sort_values(["model_rank", "mode_rank", "k_sort", "job_id"], na_position="last")


def nemorl_fresh_finding(live_rows: pd.DataFrame) -> str:
    if live_rows.empty:
        return "No fresh K-sweep rows were available in the local artifacts."
    clean = live_rows[
        (live_rows["mode"].astype(str) == "sync")
        & pd.to_numeric(live_rows.get("completed_steps"), errors="coerce").fillna(0).gt(0)
    ].copy()
    if clean.empty:
        return "Fresh K-sweep jobs are submitted, but no sync row has completed enough steps for timing metrics yet."
    clean["gen_tps_speedup"] = pd.to_numeric(clean.get("gen_tps_speedup"), errors="coerce")
    clean["e2e_tps_speedup"] = pd.to_numeric(clean.get("e2e_tps_speedup"), errors="coerce")
    clean = clean.sort_values("gen_tps_speedup", ascending=False)
    best = clean.iloc[0]
    q32 = clean[clean["model_name"].astype(str) == "Qwen3-32B"]
    q32_text = ""
    if not q32.empty:
        q32_bits = [
            f"{nemorl_method_label(row.method_k)} {fmt_x(row.gen_tps_speedup)} gen"
            for row in q32.itertuples()
            if not math.isnan(clean_float(row.gen_tps_speedup))
        ]
        if q32_bits:
            q32_text = " Qwen3-32B partial sync rows: " + ", ".join(q32_bits) + "."
    return (
        f"Fresh K-sweep signal: {best['model_name']} {best['mode']} {nemorl_method_label(best['method_k'])} "
        f"reached {best['completed_last_step']} with {fmt_x(best['gen_tps_speedup'])} generation throughput "
        f"and {fmt_x(best['e2e_tps_speedup'])} E2E throughput vs the matched OSL4096 baseline."
        + q32_text
    )


def chapter_card(title: str, body: str, href: str) -> str:
    return (
        f'<a class="chapter-card" href="{esc(href)}">'
        f"<strong>{esc(title)}</strong><span>{esc(body)}</span></a>"
    )


def build_nemorl_html(rows: pd.DataFrame) -> str:
    updated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = rows.copy() if not rows.empty else rows
    if not rows.empty:
        rows["method_display"] = rows["method_k"].map(nemorl_method_label)
    running = int((rows.get("slurm_state", pd.Series(dtype=str)).astype(str) == "RUNNING").sum()) if not rows.empty else 0
    pending = int((rows.get("slurm_state", pd.Series(dtype=str)).astype(str) == "PENDING").sum()) if not rows.empty else 0
    completed_metric = int(pd.to_numeric(rows.get("completed_steps"), errors="coerce").fillna(0).gt(0).sum()) if not rows.empty else 0
    current = rows[pd.to_numeric(rows.get("max_steps"), errors="coerce") == 20].copy() if not rows.empty else pd.DataFrame()
    smoke = rows[pd.to_numeric(rows.get("max_steps"), errors="coerce") == 3].copy() if not rows.empty else pd.DataFrame()
    live_k = nemorl_live_k_sweep_rows(rows)
    verified_k3 = nemorl_verified_eagle_k3_rows(rows)
    fresh_key = nemorl_fresh_finding(live_k)
    async_engine_errors = int(
        (
            live_k.get("metric_state", pd.Series(dtype=str)).astype(str).str.contains("engine_error", na=False)
        ).sum()
    ) if not live_k.empty else 0
    best = nemorl_chart_rows(current)
    best = best[~best["method_k"].astype(str).str.startswith("baseline")].copy()
    best = best[pd.to_numeric(best["gen_tps_speedup"], errors="coerce").notna()]
    if not best.empty:
        top = best.sort_values("gen_tps_speedup", ascending=False).iloc[0]
        key = (
            f"Best parsed NeMo-RL step20 row is {top['model_name']} {top['mode']} {nemorl_method_label(top['method_k'])} "
            f"({top['source_group']}) with "
            f"{fmt_x(top['gen_tps_speedup'])} generation throughput speedup and "
            f"{fmt_x(top['e2e_tps_speedup'])} E2E throughput speedup vs the matched baseline snapshot."
        )
    else:
        key = "Step20 rows are running or pending; matched speedup will update as baseline and spec rows complete more steps."
    css = """
:root{--ink:#111827;--muted:#5f6b7a;--line:#d6dee9;--bg:#f4f6f9;--panel:#fff;--soft:#eef3f8;--blue:#2457a6;--green:#157f47;--amber:#946200;--red:#b42318}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font:15px/1.48 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;color:var(--ink);background:var(--bg)}header{background:linear-gradient(180deg,#ffffff 0,#f8fafc 100%);border-bottom:1px solid var(--line)}.hero{max-width:1480px;margin:0 auto;padding:26px 28px 18px}.topbar{margin-bottom:10px}.topbar a{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:8px;background:#fff;padding:6px 10px;text-decoration:none;font-weight:700;color:var(--blue)}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--blue);margin-bottom:8px}main{max-width:1480px;margin:0 auto;padding:20px 28px 42px}h1{margin:0 0 8px;font-size:34px;line-height:1.12;letter-spacing:0}h2{margin:0 0 12px;font-size:21px}h3{margin:18px 0 6px;font-size:16px}.subtitle,.note{color:var(--muted)}.toc{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}.toc a{border:1px solid var(--line);background:#fff;color:#263448;text-decoration:none;border-radius:6px;padding:7px 10px;font-size:13px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 9px;margin:2px 4px 2px 0;background:#fff}.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:12px 0 18px}.kpi{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px}.kpi b{display:block;font-size:24px;line-height:1.05}.kpi span{color:var(--muted)}section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px;margin:0 0 18px}.chapter-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.chapter-card{display:block;text-decoration:none;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:8px;padding:13px}.chapter-card strong{display:block;margin-bottom:5px}.chapter-card span{display:block;color:var(--muted);font-size:13px}.callout{border-left:4px solid var(--blue);background:#f8fbff;padding:12px 14px;border-radius:6px;margin:10px 0}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:12px}.model-charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:10px 0 18px}.chart-card{border:1px solid var(--line);border-radius:8px;background:#fff;padding:8px;min-width:0}.chart-card svg{width:100%;height:auto;display:block}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:#fff}th,td{border:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}th{position:sticky;top:0;z-index:1;background:#eef2f7;font-size:13px;white-space:nowrap}tbody tr:nth-child(even){background:#f8fafc}.verified-table table{min-width:1180px}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.source-col,.manifest-col,.path-col,.note-col,.name-col{max-width:260px;white-space:normal;overflow-wrap:anywhere}.manifest-col,.path-col{font-size:12px}.error-col{max-width:380px}.not-logged{color:var(--muted);font-size:12px;white-space:nowrap}.RUNNING,.COMPLETED{color:var(--green);font-weight:700}.PENDING,.SUBMITTED{color:var(--amber);font-weight:700}.FAILED,.TIMEOUT,.CANCELLED{color:var(--red);font-weight:700}code{background:#f3f4f6;padding:1px 4px;border-radius:4px}a code{color:var(--blue)}@media(max-width:1100px){.charts,.model-charts,.chapter-grid{grid-template-columns:1fr 1fr}.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:900px){.hero,main{padding-left:16px;padding-right:16px}.model-charts,.kpis,.chapter-grid{grid-template-columns:1fr}h1{font-size:28px}table{font-size:13px}}@media(max-width:620px){.charts,.kpis,.chapter-grid{grid-template-columns:1fr}}"""
    cols = [
        ("source_group", "Source group", "text"),
        ("job_id", "Job", "text"),
        ("wandb_url", "W&B", "link"),
        ("model_name", "Model", "text"),
        ("mode", "Mode", "text"),
        ("method_display", "Method", "text"),
        ("result_state", "Result state", "text"),
        ("baseline_match_state", "Baseline match", "text"),
        ("cuda_graph_state", "CUDA Graph", "text"),
        ("cluster", "Cluster", "text"),
        ("nodes_x_gpus", "Nodes x GPUs", "text"),
        ("target_tensor_parallel_size", "Target TP", "int"),
        ("draft_tensor_parallel_size", "Draft TP", "int"),
        ("attention_backend", "Attention", "text"),
        ("moe_backend", "MoE", "text"),
        ("max_num_batched_tokens", "Batch-token budget", "int"),
        ("segment", "segment", "int"),
        ("max_new_tokens", "Max OSL", "int"),
        ("slurm_state", "SLURM", "text"),
        ("metric_window", "Metric window", "text"),
        ("total_step_time_s_mean", "E2E step", "num"),
        ("e2e_step_time_speedup", "Step-time speedup", "x"),
        ("e2e_tokens_per_sec_per_gpu_mean", "E2E tok/s/GPU", "num"),
        ("e2e_tps_speedup", "E2E tput speedup", "x"),
        ("generation_time_s_mean", "Generation time", "num"),
        ("generation_time_speedup", "Gen-time speedup", "x"),
        ("generation_worker_tokens_per_sec_per_gpu_mean", "Gen tok/s/GPU", "num"),
        ("gen_tps_speedup", "Gen tput speedup", "x"),
        ("vllm_token_acceptance_pct", "Acceptance", "pct"),
        ("vllm_acceptance_length_mean_weighted_mean", "Mean len", "num"),
        ("avg_reward_mean", "Reward", "num"),
        ("generation_kl_error_mean", "Generation KL", "num"),
        ("manifest", "Manifest", "text"),
    ]
    live_cols = [
        ("job_id", "Job", "text"),
        ("wandb_url", "W&B", "link"),
        ("wandb_name", "W&B name", "text"),
        ("model_name", "Model", "text"),
        ("mode", "Mode", "text"),
        ("method_display", "Method", "text"),
        ("enforce_eager", "enforce_eager", "text"),
        ("nodes_x_gpus", "Nodes x GPUs", "text"),
        ("segment", "segment", "int"),
        ("slurm_state", "SLURM", "text"),
        ("completed_last_step", "completed/last", "text"),
        ("generation_worker_tokens_per_sec_per_gpu_mean", "Gen tok/s/GPU", "num"),
        ("gen_tps_speedup", "Gen tput speedup", "x"),
        ("generation_time_s_mean", "Gen time", "num"),
        ("generation_time_speedup", "Gen-time speedup", "x"),
        ("e2e_tokens_per_sec_per_gpu_mean", "E2E tok/s/GPU", "num"),
        ("e2e_tps_speedup", "E2E tput speedup", "x"),
        ("total_step_time_s_mean", "E2E step", "num"),
        ("e2e_step_time_speedup", "E2E step speedup", "x"),
        ("vllm_token_acceptance_pct", "Acceptance", "pct"),
        ("vllm_acceptance_length_mean_weighted_mean", "Mean len", "num"),
        ("metric_state", "Metric state", "text"),
        ("notes", "Notes", "text"),
        ("latest_error", "First severe error", "text"),
    ]
    k3_cols = [
        ("model_name", "Model", "text"),
        ("mode", "Mode", "text"),
        ("cuda_graph_state", "CUDA Graph", "text"),
        ("max_new_tokens", "Max OSL", "int"),
        ("job_id", "Job", "text"),
        ("metric_window", "Metric window", "text"),
        ("generation_worker_tokens_per_sec_per_gpu_mean", "Gen tok/s/GPU", "num"),
        ("gen_tps_speedup", "Gen tput", "x"),
        ("e2e_tokens_per_sec_per_gpu_mean", "E2E tok/s/GPU", "num"),
        ("e2e_tps_speedup", "E2E tput", "x"),
        ("vllm_token_acceptance_pct", "Acceptance", "pct"),
        ("vllm_acceptance_length_mean_weighted_mean", "Mean len", "num"),
        ("wandb_url", "W&B", "link"),
        ("manifest", "Evidence", "text"),
    ]
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
            f"<title>Lyris NeMo-RL SpecDec Status Latest</title><style>{css}</style></head><body>",
            "<header><div class=\"hero\"><div class=\"topbar\"><a href=\"../index.html\">Back to report hub</a></div><div class=\"eyebrow\">LIVE REPORT · SPECULATIVE DECODING · 2026</div><h1>Lyris NeMo-RL SpecDec Status</h1>",
            f"<div class=\"subtitle\">Updated {esc(updated)}. Fresh K-sweep check: {esc(NEMORL_LIVE_K_SWEEP_CHECKED_AT)}. Data covers the normalized July 2 CUDA-graph-on performance-recipe cohorts, Qwen3-235B absolute rows awaiting baseline, and separated historical Lyris/OCI-HSG artifacts.</div>",
            "<nav class=\"toc\"><a href=\"#overview\">Overview</a><a href=\"#verified-k3\">Verified Eagle-3 K3</a><a href=\"#fresh\">CUDA-Graph-Disabled K Sweep</a><a href=\"#methodology\">Methodology</a><a href=\"#charts\">Charts</a><a href=\"#step20\">Step20 Tables</a><a href=\"#smoke\">Step3 Smoke</a><a href=\"#sources\">Sources</a></nav></div></header><main>",
            "<div><span class=\"pill\">performance recipe configs</span><span class=\"pill\">CUDA graph enabled is default</span><span class=\"pill\">temperature=1.0</span><span class=\"pill\">top_p=1.0</span><span class=\"pill\">enforce_eager shown per row</span><span class=\"pill\">Max OSL separated by section</span><span class=\"pill\">step>=2 metrics where noted</span><span class=\"pill\">GB200 segment captured</span></div>",
            "<div class=\"kpis\">",
            f"<div class=\"kpi\"><b>{running}</b><span>running jobs</span></div>",
            f"<div class=\"kpi\"><b>{pending}</b><span>pending jobs</span></div>",
            f"<div class=\"kpi\"><b>{async_engine_errors}</b><span>async rows with engine errors</span></div>",
            f"<div class=\"kpi\"><b>{completed_metric}</b><span>rows with completed steps</span></div>",
            f"<div class=\"kpi\"><b>{len(rows)}</b><span>tracked rows</span></div>",
            "</div>",
            "<section id=\"overview\"><h2>Overview</h2>",
            f"<div class=\"callout\"><strong>Key finding.</strong> {esc(key)}<br><strong>Fresh update.</strong> {esc(fresh_key)}</div>",
            "<div class=\"chapter-grid\">",
            chapter_card("Verified Eagle-3 K3", "Completed 20-step K3 rows with exact metric windows and W&B availability.", "#verified-k3"),
            chapter_card("CUDA-Graph-Disabled K Sweep", "Older enforce_eager=true Eagle-3 K5/K7/K9 state; useful as an ablation, not the default baseline.", "#fresh"),
            chapter_card("Matched Charts", "Generation/E2E throughput and step-time speedups by model.", "#charts"),
            chapter_card("Step20 Snapshot", "All current and historical step20 rows with acceptance metrics.", "#step20"),
            chapter_card("Raw Evidence", "CSV, log path, and source provenance links for reproducibility.", "#sources"),
            "</div></section>",
            "<section id=\"verified-k3\"><h2>Verified Eagle-3 K3 Results</h2>",
            "<p class=\"note\">Every row below comes from a completed 20-step job. <code>steps 2-20 (19 metrics)</code> means the cold-start first step was intentionally excluded. The largest historical K3 speedups used <code>enforce_eager=true</code>, so CUDA Graph was disabled. W&B shows <code>not logged</code> when the original Slurm command explicitly set <code>logger.wandb_enabled=false</code>; those runs still retain parsed driver-log evidence.</p><div class=\"table-wrap verified-table\">",
            table(verified_k3, k3_cols),
            "</div></section>",
            "<section id=\"fresh\"><h2>CUDA-Graph-Disabled K Sweep</h2>",
            "<p class=\"note\">This older Lyris run set used <code>policy.generation.vllm_cfg.enforce_eager=true</code>, which disables CUDA graph capture. Treat it as a diagnostic ablation. The realistic/default scenario for current conclusions is <code>enforce_eager=false</code> with CUDA graph enabled, matched against the same model/mode/OSL/temperature/top_p baseline. Async timeout rows are listed for status but should not be treated as clean performance data while EngineCore errors are present.</p><div class=\"table-wrap\">",
            table(live_k, live_cols),
            "</div></section>",
            "<section id=\"methodology\"><h2>Evaluation Methodology</h2><ul>",
            "<li>Recipes: NeMo-RL <code>examples/configs/recipes/llm/performance</code>.</li>",
            "<li>Matched comparisons keep source cohort, model, mode, max OSL, temperature/top_p, CUDA Graph state, cluster, node/GPU shape, attention/MoE backend, target TP, batch-token budget, segment, and recipe cohort fixed.</li>",
            "<li>SpecDec rows add only the generation speculative decoding method, drafter/checkpoint, and <code>num_speculative_tokens</code>; baseline rows use the same recipe with SpecDec disabled.</li>",
            "<li>Completed 20-step charts require either all 20 parsed metrics or the steady-state steps 2-20 with step 1 excluded. Partial step20 jobs are table-only.</li>",
            "<li>2026-06-18 and 2026-06-22 historical Qwen3-30B-A3B/Qwen3-32B K3 rows use <code>enforce_eager=true</code>; their large speedups are CUDA-graph-disabled comparisons and are not directly comparable to the current CUDA-graph-enabled default.</li>",
            "<li>Default/realistic comparisons assume <code>enforce_eager=false</code>, so vLLM CUDA graph capture is enabled. Rows with <code>enforce_eager=true</code> are CUDA-graph-disabled ablations and should not be used as the primary baseline.</li>",
            "</ul></section>",
            f"<section><h2>Metric Notes</h2><p>{esc(fresh_key)}</p><p class=\"note\">Acceptance metrics are shown only when the NeMo-RL driver log includes vLLM SpecDec metrics; Qwen3-235B current driver snapshots mostly expose timing/throughput, while historical Qwen30/Qwen32 rows include acceptance when available.</p></section>",
            '<div id="charts">',
            nemorl_charts_section(rows),
            "</div>",
            "<section id=\"step20\"><h2>Step20 Current And Historical Snapshot</h2><div class=\"table-wrap\">",
            table(current, cols),
            "</div></section>",
            "<section id=\"smoke\"><h2>Step3 Smoke / K Sweep</h2><div class=\"table-wrap\">",
            table(smoke, cols),
            "</div></section>",
            "<section id=\"sources\"><h2>Sources</h2><p class=\"note\"><code>docs/lyris_nemorl_v020_best_math_live_metrics_20260704.csv</code>, <code>docs/lyris_qwen30_sync_pard_strict_matched_metrics_20260702.csv</code>, <code>docs/lyris_qwen30_async1off_strict_matched_live_metrics_20260702.csv</code>, <code>docs/lyris_qwen32_sync_eagle3_matched_live_metrics_20260702.csv</code>, <code>docs/lyris_qwen32_sync_pard_tp2_noarrms_matched_live_metrics_20260702.csv</code>, <code>docs/lyris_qwen32_async1off_eagle3_matched_live_metrics_20260702.csv</code>, <code>docs/lyris_qwen235b_sync_eagle3_absolute_metrics_20260702.csv</code>, <code>docs/pretyche_qwen32_sync_osl32k_matched_live_metrics_20260702.csv</code>, and the retained June historical sources listed in the combined CSV provenance column.</p></section>",
            "</main></body></html>",
        ]
    )


def build_latest_nemorl_outputs(
    *,
    live_rows: pd.DataFrame | None = None,
    output_html: Path = NEMORL_HTML,
    enriched_csv_out: Path = NEMORL_OUT,
    combined_csv_out: Path = NEMORL_COMBINED_OUT,
) -> Path:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    enriched_csv_out.parent.mkdir(parents=True, exist_ok=True)
    combined_csv_out.parent.mkdir(parents=True, exist_ok=True)
    current_rows = enrich_nemorl() if live_rows is None else live_rows.copy()
    current_rows.to_csv(enriched_csv_out, index=False)
    combined = combine_nemorl_rows(current_rows)
    combined.to_csv(combined_csv_out, index=False)
    output_html.write_text(build_nemorl_html(combined), encoding="utf-8")
    return output_html


def main() -> None:
    build_latest_vllm_outputs()
    build_latest_nemorl_outputs()

    print(VLLM_ADDED_OUT)
    print(DFLARE_COMPLETED_OUT)
    print(VLLM_HTML_LATEST)
    print(NEMORL_OUT)
    print(NEMORL_COMBINED_OUT)
    print(NEMORL_HTML)


if __name__ == "__main__":
    main()
