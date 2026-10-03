"""Preserve startup evidence for Ray dashboard failures in a staged image."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import site
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=("driver", "vllm", "vllm-nested"), required=True
    )
    parser.add_argument("--temp-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.role.startswith("vllm"):
        site.addsitedir(
            "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker."
            "MegatronPolicyWorker/lib/python3.13/site-packages"
        )
        site.addsitedir("/opt/nemo_rl_venv/lib/python3.13/site-packages")

    import ray
    import torch

    print(
        json.dumps(
            {
                "role": args.role,
                "executable": sys.executable,
                "ray_file": ray.__file__,
                "ray_version": importlib.metadata.version("ray"),
                "cuda_devices": torch.cuda.device_count(),
                "sys_path": sys.path,
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        ray.init(
            address="local",
            include_dashboard=True,
            log_to_driver=True,
            _temp_dir=str(args.temp_dir),
            num_cpus=2,
        )
        print(f"{args.role}: Ray dashboard started", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
