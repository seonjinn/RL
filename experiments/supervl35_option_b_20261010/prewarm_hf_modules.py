"""Populate node-local Hugging Face remote-code modules before Ray starts."""

import importlib
import sys

from transformers import AutoConfig


def main() -> None:
    config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
    module_name = config.__class__.__module__
    if not module_name.startswith("transformers_modules."):
        raise RuntimeError(f"Expected a remote-code config module, got {module_name}")
    importlib.import_module(module_name)
    print(f"Prewarmed {module_name}", flush=True)


if __name__ == "__main__":
    main()
