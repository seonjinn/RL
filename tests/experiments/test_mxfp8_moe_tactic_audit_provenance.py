from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "experiments/mxfp8_moe_tactic_audit/README.md"


def test_readme_pins_runtime_and_privacy_contract() -> None:
    text = README.read_text()
    assert "a76062edee3a3ac23d47a93c7ce466f06a19111f" in text
    assert "FlashInfer 0.6.13" in text
    assert "Ptyche GB200" in text
    assert "prompts, token IDs, hidden values, or model outputs" in text
    assert "CUDA Graphs" in text
    assert "1,319-example GSM8K" in text
