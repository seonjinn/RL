import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "vllm_0251_eagle3_perfcfg"
    / "apply_vllm0251_dynamic_sd_cg_fix.py"
)


def test_patch_updates_vllm0251_and_is_idempotent(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    cudagraph_utils = site_packages / "vllm/v1/worker/gpu/cudagraph_utils.py"
    speculator = (
        site_packages / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    )
    cudagraph_utils.parent.mkdir(parents=True)
    speculator.parent.mkdir(parents=True)
    cudagraph_utils.write_text(
        "class CudaGraphManager:\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: VllmConfig,\n"
        "        device: torch.device,\n"
        "        cudagraph_mode: CUDAGraphMode,\n"
        "        decode_query_len: int,\n"
        "        lora_capture_cases: list[int] | None = None,\n"
        "    ):\n"
        "        self.decode_query_len = decode_query_len\n\n"
        "        self.dp_size = vllm_config.parallel_config.data_parallel_size\n"
        "            and speculative_config.uses_dynamic_speculative_decoding()\n"
        "        ):\n",
        encoding="utf-8",
    )
    speculator.write_text(
        "            cudagraph_mode,\n            decode_query_len=1,\n        )\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(PATCH_SCRIPT),
        "--site-packages",
        str(site_packages),
    ]
    first = subprocess.run(command, check=True, text=True, capture_output=True)
    second = subprocess.run(command, check=True, text=True, capture_output=True)

    assert "applied" in first.stdout
    assert "already applied" in second.stdout
    assert "use_dynamic_decode_shapes: bool = True" in cudagraph_utils.read_text()
    assert "and self.use_dynamic_decode_shapes" in cudagraph_utils.read_text()
    assert "use_dynamic_decode_shapes=False" in speculator.read_text()

