import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGER = REPO_ROOT / "scripts" / "stage_hf_snapshots.py"


def test_stage_uses_standard_hf_hub_cache(monkeypatch, tmp_path: Path) -> None:
    cache_dirs: list[Path] = []

    def fake_snapshot_download(*, repo_id: str, cache_dir: Path) -> str:
        cache_dirs.append(Path(cache_dir))
        snapshot = tmp_path / repo_id.replace("/", "--")
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"weights")
        return str(snapshot)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    spec = importlib.util.spec_from_file_location("stage_hf_snapshots", STAGER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    hf_home = tmp_path / "hf_home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("MODEL_IDS", "amd/PARD-Qwen3-0.6B")

    module.main()

    assert cache_dirs == [hf_home / "hub"]
