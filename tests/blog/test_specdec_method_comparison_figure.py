from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "docs/blog/figures/render_specdec_method_comparison.py"


def test_renderer_creates_compact_png_without_text_overflow(tmp_path: Path) -> None:
    output = tmp_path / "comparison.png"

    result = subprocess.run(
        [sys.executable, str(RENDERER), "--output", str(output)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with Image.open(output) as image:
        assert image.size == (1480, 830)
        assert image.mode in {"RGB", "RGBA"}

    receipt = json.loads(output.with_suffix(".layout.json").read_text())
    assert receipt["text_overflow_count"] == 0
    assert receipt["layout"] == "1x3"
    assert receipt["panel_count"] == 3
    assert receipt["target_verifier_count"] == 3
    assert receipt["required_labels"] == [
        "Target context",
        "EAGLE-3",
        "DFlash",
        "DSpark",
        "AUTOREGRESSIVE",
        "PARALLEL",
        "SEMI-AUTOREGRESSIVE",
        "Confidence scheduler",
        "Target verify",
        "Accepted prefix",
    ]
    assert receipt["source_urls"] == [
        "https://arxiv.org/abs/2503.01840",
        "https://arxiv.org/abs/2602.06036",
        "https://arxiv.org/abs/2607.05147",
    ]
