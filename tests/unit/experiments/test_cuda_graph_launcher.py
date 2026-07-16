import subprocess
from pathlib import Path


LAUNCHER_LIB = (
    Path(__file__).parents[3]
    / "experiments"
    / "cuda_graph"
    / "cuda_graph_launcher_lib.sh"
)


def _checkpoint_dir(model: str, condition: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; pr5672_qwen_checkpoint_dir "$2" "$3"',
            "bash",
            str(LAUNCHER_LIB),
            model,
            condition,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_pr5672_qwen_conditions_use_isolated_conversion_directories():
    attn_dir = _checkpoint_dir("qwen3", "pr5672-attn")
    attn_mlp_dir = _checkpoint_dir("qwen3", "pr5672-attn-mlp")

    assert attn_dir.endswith("qwen3-8b-pr5672-20260716/pr5672-attn")
    assert attn_mlp_dir.endswith("qwen3-8b-pr5672-20260716/pr5672-attn-mlp")
    assert attn_dir != attn_mlp_dir
