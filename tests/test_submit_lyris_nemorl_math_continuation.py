from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "experiments/eagle3_online/submit_lyris_nemorl_math_continuation_20260704.sh"
)
REMOTE_REPO = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/"
    "RL-specdec-cudagraph-780f483a-20260701"
)


def run_dry(*, check: bool = True, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ["PATH"],
        "DRY_RUN": "true",
        **overrides,
    }
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def job_sections(output: str) -> dict[tuple[str, str, str], str]:
    sections: dict[tuple[str, str, str], str] = {}
    for section in output.split("[DRY-RUN] model=")[1:]:
        header = section.splitlines()[0].split()
        model = header[0]
        mode = header[1].removeprefix("mode=")
        method = header[2].removeprefix("method=")
        sections[(model, mode, method)] = section
    return sections


def test_default_dry_run_renders_only_supported_contracts() -> None:
    output = run_dry().stdout
    sections = job_sections(output)

    assert set(sections) == {
        ("qwen30ba3b", "sync", "suffix"),
        ("qwen30ba3b", "async1off", "suffix"),
        ("qwen32", "sync", "suffix"),
        ("qwen32", "sync", "eagle3"),
        ("qwen32", "async1off", "suffix"),
        ("qwen32", "async1off", "eagle3"),
        ("qwen235b", "sync", "baseline"),
        ("qwen235b", "sync", "suffix"),
        ("qwen235b", "sync", "eagle3"),
        ("qwen235b", "async1off", "baseline"),
        ("qwen235b", "async1off", "suffix"),
        ("qwen235b", "async1off", "eagle3"),
    }

    shapes = {
        ("qwen30ba3b", "sync"): ("grpo-qwen3-30ba3b-4n4g.yaml", 4, 4),
        ("qwen30ba3b", "async1off"): (
            "grpo-qwen3-30ba3b-4n4g-async-1off.yaml",
            4,
            4,
        ),
        ("qwen32", "sync"): ("grpo-qwen3-32b-4n4g.yaml", 4, 4),
        ("qwen32", "async1off"): (
            "grpo-qwen3-32b-8n4g-async-1off.yaml",
            8,
            8,
        ),
        ("qwen235b", "sync"): ("grpo-qwen3-235b-32n4g.yaml", 32, 16),
        ("qwen235b", "async1off"): (
            "grpo-qwen3-235b-32n4g-async-1off.yaml",
            32,
            16,
        ),
    }
    for (model, mode, _method), section in sections.items():
        recipe, nodes, segment = shapes[(model, mode)]
        assert recipe in section
        assert f"--nodes={nodes}" in section
        assert f"--segment={segment}" in section
        assert "grpo.max_num_steps=20" in section
        assert "attention_backend=TRITON_ATTN" in section


def test_selected_lists_render_locally_and_preflight_only_selected_assets() -> None:
    output = run_dry(
        MODELS="qwen32",
        MODES="async-1off",
        METHODS="eagle3",
    ).stdout

    assert set(job_sections(output)) == {("qwen32", "async1off", "eagle3")}
    assert "git -C '/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701' rev-parse HEAD" in output
    assert "1271b1530181a7378e40de40b4b46ad223e6596c" in output
    assert "sha256sum '/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh'" in output
    assert "container_sha256" in output
    assert "grpo-qwen3-32b-8n4g-async-1off.yaml" in output
    assert "models--Qwen--Qwen3-32B" in output
    assert "models--RedHatAI--Qwen3-32B-speculator.eagle3" in output
    assert "models--Qwen--Qwen3-30B-A3B" not in output
    assert "models--Qwen--Qwen3-235B-A22B" not in output
    assert "models--nvidia--Qwen3-235B-A22B-Eagle3" not in output
    assert "arctic-inference" not in output


def test_suffix_k32_renders_pinned_source_and_dynamic_parameters() -> None:
    output = run_dry(MODELS="qwen30ba3b", MODES="sync", METHODS="suffix").stdout
    section = job_sections(output)[("qwen30ba3b", "sync", "suffix")]

    source_site = (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/"
        "build_deps/arctic-inference-0.1.1-py313-native"
    )
    assert f"SOURCE_VLLM_SITE='{source_site}'" in section
    assert "speculative_config.method=suffix" in section
    assert "speculative_config.num_speculative_tokens=32" in section
    assert "speculative_config.suffix_decoding_max_tree_depth=24" in section
    assert "speculative_config.suffix_decoding_max_cached_requests=10000" in section
    assert "speculative_config.suffix_decoding_max_spec_factor=1.0" in section
    assert "speculative_config.suffix_decoding_min_token_prob=0.1" in section


def test_qwen235b_triton_cohort_has_baseline_noarrms_and_no_sharp() -> None:
    output = run_dry(
        MODELS="qwen235b",
        MODES="sync",
        METHODS="baseline,suffix,eagle3",
    ).stdout
    sections = job_sections(output)

    assert set(sections) == {
        ("qwen235b", "sync", "baseline"),
        ("qwen235b", "sync", "suffix"),
        ("qwen235b", "sync", "eagle3"),
    }
    for section in sections.values():
        assert "attention_backend=TRITON_ATTN" in section
        assert "compilation_config.pass_config.fuse_allreduce_rms=false" in section
        assert "--network=sharp" not in section

    assert "speculative_config" not in sections[("qwen235b", "sync", "baseline")]
    assert "speculative_config.method=eagle3" in sections[("qwen235b", "sync", "eagle3")]
    assert "speculative_config.num_speculative_tokens=3" in sections[
        ("qwen235b", "sync", "eagle3")
    ]


def test_network_and_shared_cache_contracts_are_explicit() -> None:
    output = run_dry(
        MODELS="qwen30ba3b qwen32",
        MODES="sync",
        METHODS="suffix",
        HF_HOME="/shared/hf",
        HF_DATASETS_CACHE="/shared/datasets",
    ).stdout
    sections = job_sections(output)

    assert set(sections) == {
        ("qwen30ba3b", "sync", "suffix"),
        ("qwen32", "sync", "suffix"),
    }
    for section in sections.values():
        assert "--network=sharp" in section
        assert "HF_HOME='/shared/hf'" in section
        assert "HF_DATASETS_CACHE='/shared/datasets'" in section
        assert "export HF_HOME='/shared/hf'" in section
        assert "export HF_DATASETS_CACHE='/shared/datasets'" in section


def test_invalid_or_empty_selection_fails_before_remote_work() -> None:
    unknown = run_dry(check=False, METHODS="pard")
    empty = run_dry(
        check=False,
        MODELS="qwen30ba3b",
        MODES="sync",
        METHODS="baseline",
    )

    assert unknown.returncode == 2
    assert "unsupported method: pard" in unknown.stderr
    assert empty.returncode == 2
    assert "selection contains no supported model/mode/method combinations" in empty.stderr


def test_launcher_never_recovers_or_evaluates_historical_commands() -> None:
    source = LAUNCHER.read_text()

    assert "eval " not in source
    assert "scontrol show job" not in source
    assert "sacct" not in source
    assert "ray-driver.log" not in source
    assert "slurm-%j.out" in source
    assert f"REMOTE_REPO=\"${{REMOTE_REPO:-{REMOTE_REPO}}}\"" in source
