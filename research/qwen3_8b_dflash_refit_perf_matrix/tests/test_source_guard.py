import importlib.util
from pathlib import Path
import subprocess
from types import ModuleType

import pytest


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_refit_perf_matrix"


def _module() -> ModuleType:
    path = EXPERIMENT_DIR / "source_guard.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Matrix Test")
    _git(repo, "config", "user.email", "matrix@example.com")
    (repo / "product.txt").write_text("product\n")
    _git(repo, "add", "product.txt")
    _git(repo, "commit", "-q", "-m", "product")
    product = _git(repo, "rev-parse", "HEAD")
    harness = repo / "research/qwen3_8b_dflash_refit_perf_matrix"
    harness.mkdir(parents=True)
    (harness / "README.md").write_text("harness\n")
    _git(repo, "add", "research/qwen3_8b_dflash_refit_perf_matrix/README.md")
    _git(repo, "commit", "-q", "-m", "harness")
    return repo, product, _git(repo, "rev-parse", "HEAD")


def test_source_guard_emits_folder_only_clean_ancestry_proof(tmp_path: Path) -> None:
    guard = _module()
    repo, product, harness = _repo(tmp_path)

    proof = guard.validate_checkout(
        repo,
        product_head=product,
        harness_head=harness,
        require_signed_dco=False,
    )

    assert proof["status"] == "passed"
    assert proof["product_head"] == product
    assert proof["harness_head"] == harness
    assert proof["harness_delta"] == [
        "research/qwen3_8b_dflash_refit_perf_matrix/README.md"
    ]


def test_source_guard_rejects_untracked_files(tmp_path: Path) -> None:
    guard = _module()
    repo, product, harness = _repo(tmp_path)
    (repo / "untracked.txt").write_text("dirty\n")

    with pytest.raises(ValueError, match="dirty source checkout"):
        guard.validate_checkout(
            repo,
            product_head=product,
            harness_head=harness,
            require_signed_dco=False,
        )
