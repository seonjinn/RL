from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "451bbcc607b33f1709b9d97393b81cd9f3eabdc2fcd721ef3b19743d5da896d3"
)
CLASS_ANCHOR = (
    "class DFlashQwen3Model(nn.Module):\n"
    "    hf_to_vllm_mapper = WeightsMapper("
)
PATCHED_CLASS_ANCHOR = (
    "class DFlashQwen3Model(nn.Module):\n"
    "    decoder_layer_cls = DFlashQwen3DecoderLayer\n\n"
    "    hf_to_vllm_mapper = WeightsMapper("
)
LAYER_ANCHOR = (
    "\n                DFlashQwen3DecoderLayer(\n"
    "                    current_vllm_config,"
)
PATCHED_LAYER_ANCHOR = (
    "\n                self.decoder_layer_cls(\n"
    "                    current_vllm_config,"
)


def patch_qwen3_dflash_source(source: str) -> str:
    if "decoder_layer_cls = DFlashQwen3DecoderLayer" in source:
        raise RuntimeError("source already contains the DFlash2 decoder extension point")
    if source.count(CLASS_ANCHOR) != 1:
        raise RuntimeError("expected exactly one DFlashQwen3Model class anchor")
    if source.count(LAYER_ANCHOR) != 1:
        raise RuntimeError("expected exactly one DFlash decoder-layer constructor anchor")

    patched = source.replace(CLASS_ANCHOR, PATCHED_CLASS_ANCHOR, 1)
    patched = patched.replace(LAYER_ANCHOR, PATCHED_LAYER_ANCHOR, 1)
    if patched.count("decoder_layer_cls = DFlashQwen3DecoderLayer") != 1:
        raise RuntimeError("runtime overlay did not restore the decoder class attribute")
    if patched.count("self.decoder_layer_cls(") != 1:
        raise RuntimeError("runtime overlay did not restore the decoder constructor")
    return patched


def prepare_overlay(source_path: Path, output_path: Path) -> str:
    source = source_path.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "vLLM qwen3_dflash.py does not match the pinned f94666b60 source: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )

    patched = patch_qwen3_dflash_source(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8")
    return hashlib.sha256(patched.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    patched_sha256 = prepare_overlay(arguments.source, arguments.output)
    print(f"patched_qwen3_dflash_sha256={patched_sha256}")


if __name__ == "__main__":
    main()
