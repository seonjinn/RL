from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO = EXPERIMENT_DIR.parents[1]
SUBMIT = EXPERIMENT_DIR / "submit.sh"


def render(*, method: str, arm: str = "mxfp8-true-mxfp8") -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "ACTION": "render",
        "CLUSTER": "oci",
        "MODEL": "qwen235",
        "MODE": "async",
        "ARM": arm,
        "PERFORMANCE_RECIPE": "1",
        "SLURM_ACCOUNT": "test_account",
        "REPO": str(REPO),
        "SPECDEC_METHOD": method,
        "RUN_GROUP": "test",
    }
    return subprocess.run(
        ["bash", str(SUBMIT)],
        cwd=REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


class SpecDecSubmitTest(unittest.TestCase):
    def test_dflash_preserves_mxfp8_arm_and_adds_frozen_drafter(self) -> None:
        result = render(method="dflash")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("specdec_method=dflash", result.stdout)
        self.assertIn("fp8_param=true", result.stdout)
        self.assertIn("speculative_config.method=dflash", result.stdout)
        self.assertIn("num_speculative_tokens=5", result.stdout)
        self.assertIn("policy.draft.enabled=false", result.stdout)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=1", result.stdout)

    def test_dspark_adds_method_specific_capture_sizes(self) -> None:
        result = render(method="dspark")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("specdec_method=dspark", result.stdout)
        self.assertIn("cudagraph_capture_sizes=", result.stdout)
        self.assertIn("speculative_config.method=dspark", result.stdout)

    def test_no_spec_v2_is_a_matched_runner_baseline(self) -> None:
        env = os.environ | {
            "ACTION": "render",
            "CLUSTER": "oci",
            "MODEL": "qwen235",
            "MODE": "async",
            "ARM": "mxfp8-true-mxfp8",
            "PERFORMANCE_RECIPE": "1",
            "SLURM_ACCOUNT": "test_account",
            "REPO": str(REPO),
            "SPECDEC_METHOD": "none",
            "VLLM_MODEL_RUNNER": "v2",
            "RUN_GROUP": "test",
        }
        result = subprocess.run(
            ["bash", str(SUBMIT)],
            cwd=REPO,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("vllm_model_runner=v2", result.stdout)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=1", result.stdout)
        self.assertNotIn("speculative_config.method=", result.stdout)

    def test_specdec_rejects_unmatched_model(self) -> None:
        env = os.environ | {
            "ACTION": "render",
            "CLUSTER": "oci",
            "MODEL": "qwen30",
            "MODE": "async",
            "ARM": "mxfp8-true-mxfp8",
            "PERFORMANCE_RECIPE": "1",
            "SLURM_ACCOUNT": "test_account",
            "REPO": str(REPO),
            "SPECDEC_METHOD": "dflash",
            "RUN_GROUP": "test",
        }
        result = subprocess.run(
            ["bash", str(SUBMIT)],
            cwd=REPO,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("target-matched Qwen3-235B", result.stderr)


if __name__ == "__main__":
    unittest.main()
