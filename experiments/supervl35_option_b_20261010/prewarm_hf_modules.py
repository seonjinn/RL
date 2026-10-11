"""Populate node-local Hugging Face remote-code modules before Ray starts."""

import importlib
import sys

from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def main() -> None:
    model_path = sys.argv[1]
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    modules = {
        config.__class__.__module__,
        processor.__class__.__module__,
        image_processor.__class__.__module__,
    }
    if "AutoModel" in config.auto_map:
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModel"], model_path, local_files_only=True
        )
        modules.add(model_class.__module__)
    for module_name in sorted(modules):
        if not module_name.startswith("transformers_modules."):
            raise RuntimeError(f"Expected a remote-code module, got {module_name}")
        importlib.import_module(module_name)
        print(f"Prewarmed {module_name}", flush=True)


if __name__ == "__main__":
    main()
