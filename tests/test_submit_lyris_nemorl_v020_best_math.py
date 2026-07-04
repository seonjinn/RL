from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "experiments/eagle3_online/submit_lyris_nemorl_v020_best_math_20260704.sh"
)
REMOTE_REPO = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701"
)
EXPECTED_REPO_HEAD = "1271b1530181a7378e40de40b4b46ad223e6596c"


def run_dry(
    *, check: bool = True, **overrides: str
) -> subprocess.CompletedProcess[str]:
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


def test_default_dry_run_renders_only_q30_q32_supported_contracts() -> None:
    sections = job_sections(run_dry().stdout)

    assert set(sections) == {
        ("qwen30ba3b", "sync", "suffix"),
        ("qwen30ba3b", "async1off", "suffix"),
        ("qwen32", "sync", "suffix"),
        ("qwen32", "sync", "eagle3"),
        ("qwen32", "async1off", "suffix"),
        ("qwen32", "async1off", "eagle3"),
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
    }
    for (model, mode, _method), section in sections.items():
        recipe, nodes, segment = shapes[(model, mode)]
        assert recipe in section
        assert f"--nodes={nodes}" in section
        assert f"--segment={segment}" in section
        assert "policy.generation.temperature=1.0" in section
        assert "policy.generation.top_p=1.0" in section
        assert "policy.generation.vllm_cfg.enforce_eager=false" in section
        assert "attention_backend=TRITON_ATTN" in section
        assert "kernel_config.moe_backend=triton" in section
        assert "grpo.max_num_steps=20" in section
        assert "logger.wandb_enabled=true" in section
        assert "policy.generation.max_new_tokens" not in section
        assert "--gres" not in section


def test_q32_async_uses_four_worker_segment_size_on_eight_node_segment() -> None:
    sections = job_sections(
        run_dry(MODELS="qwen32", MODES="async-1off", METHODS="suffix,eagle3").stdout
    )

    assert set(sections) == {
        ("qwen32", "async1off", "suffix"),
        ("qwen32", "async1off", "eagle3"),
    }
    for section in sections.values():
        assert "cluster.segment_size=4" in section
        assert "--nodes=8" in section
        assert "--segment=8" in section


def test_selected_eagle3_preflight_validates_only_selected_assets() -> None:
    output = run_dry(MODELS="qwen32", MODES="sync", METHODS="eagle3").stdout

    assert set(job_sections(output)) == {("qwen32", "sync", "eagle3")}
    assert f"git -C '{REMOTE_REPO}' rev-parse HEAD" in output
    assert EXPECTED_REPO_HEAD in output
    assert "grpo-qwen3-32b-4n4g.yaml" in output
    assert "models--Qwen--Qwen3-32B" in output
    assert "models--RedHatAI--Qwen3-32B-speculator.eagle3" in output
    assert "models--Qwen--Qwen3-30B-A3B" not in output
    assert "models--Qwen--Qwen3-235B-A22B" not in output
    assert "arctic-inference" not in output


def test_suffix_k32_uses_source_vllm_site_and_shared_hf_caches() -> None:
    section = job_sections(
        run_dry(
            MODELS="qwen30ba3b",
            MODES="sync",
            METHODS="suffix",
            HF_HOME="/shared/hf",
            HF_DATASETS_CACHE="/shared/hf/datasets",
            SOURCE_VLLM_SITE="/shared/arctic-site",
        ).stdout
    )[("qwen30ba3b", "sync", "suffix")]

    assert "SOURCE_VLLM_SITE='/shared/arctic-site'" in section
    assert (
        "SOURCE_VLLM_SITE='/shared/arctic-site' \\\nCOMMAND=\"${COMMAND}\" \\"
        in section
    )
    assert "export HF_HOME='/shared/hf'" in section
    assert "export HF_DATASETS_CACHE='/shared/hf/datasets'" in section
    assert "speculative_config.method=suffix" in section
    assert "speculative_config.num_speculative_tokens=32" in section


def test_q32_eagle3_uses_k3_and_pinned_draft() -> None:
    section = job_sections(
        run_dry(MODELS="qwen32", MODES="sync", METHODS="eagle3").stdout
    )[("qwen32", "sync", "eagle3")]

    assert "speculative_config.method=eagle3" in section
    assert "speculative_config.num_speculative_tokens=3" in section
    assert "models--RedHatAI--Qwen3-32B-speculator.eagle3" in section
    assert "speculative_config.draft_tensor_parallel_size=1" in section


def test_q30_eagle3_and_q235_are_not_supported() -> None:
    q30_eagle = run_dry(
        check=False,
        MODELS="qwen30ba3b",
        MODES="sync",
        METHODS="eagle3",
    )
    q235 = run_dry(
        check=False,
        MODELS="qwen235b",
        MODES="sync",
        METHODS="suffix",
    )

    assert q30_eagle.returncode == 2
    assert (
        "selection contains no supported model/mode/method combinations"
        in q30_eagle.stderr
    )
    assert q235.returncode == 2
    assert "unsupported model: qwen235b" in q235.stderr


def test_submission_path_has_test_only_gate_and_complete_csv_manifest() -> None:
    source = LAUNCHER.read_text()

    gate = 'sbatch --test-only "${sbatch_args[@]}" ray.sub'
    submit = 'sbatch "${sbatch_args[@]}" ray.sub'
    assert gate in source
    assert submit in source
    assert source.index(gate) < source.index(submit)
    assert "test_only_job_id" in source
    assert "rendered_command" in source
    assert "container_sha256" in source
    assert "wandb_name" in source
    assert "csv_row" in source


def test_launcher_is_declarative_pinned_and_shell_valid() -> None:
    source = LAUNCHER.read_text()

    assert f'REMOTE_REPO="${{REMOTE_REPO:-{REMOTE_REPO}}}"' in source
    assert f'EXPECTED_REPO_HEAD="{EXPECTED_REPO_HEAD}"' in source
    assert "eval " not in source
    assert "scontrol show job" not in source
    assert "sacct" not in source
    assert "ray-driver.log" not in source
    assert "--gres" not in source
    subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=ROOT, check=True)
