from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/render_markdown_spec.py"
INDEX_BUILDER = ROOT / "scripts/build_pages_index.py"


def load_module() -> ModuleType:
    assert MODULE_PATH.exists(), "Markdown specification renderer is not implemented"
    spec = importlib.util.spec_from_file_location("render_markdown_spec", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_spec_preserves_headings_tables_and_code(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "spec.md"
    source.write_text(
        "# Design\n\n| A | B |\n|---|---|\n| `x` | y |\n",
        encoding="utf-8",
    )
    output = tmp_path / "spec.html"

    module.render_spec(source, output, "Design")

    rendered = output.read_text(encoding="utf-8")
    assert "<h1>Design</h1>" in rendered
    assert "<table>" in rendered
    assert "<code>x</code>" in rendered
    assert "Back to report hub" in rendered
    assert 'name="viewport"' in rendered


def test_cli_renders_specification(tmp_path: Path) -> None:
    source = tmp_path / "spec.md"
    output = tmp_path / "spec.html"
    source.write_text("# CLI Design\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            str(source),
            str(output),
            "--title",
            "CLI Design",
        ],
        check=True,
    )

    assert "CLI Design" in output.read_text(encoding="utf-8")


def test_pages_index_links_dflare_design_specification() -> None:
    source = INDEX_BUILDER.read_text(encoding="utf-8")

    assert "specs/2026-07-03-dflare-html-results-design.html" in source
