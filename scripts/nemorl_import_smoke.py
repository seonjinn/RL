import importlib.util
import json
import platform
import sys
from pathlib import Path

import nemo_rl
import torch


def main() -> None:
    result = {
        "nemo_rl_import_ok": True,
        "nemo_rl_path": str(Path(nemo_rl.__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "vllm_in_base_env": importlib.util.find_spec("vllm") is not None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
