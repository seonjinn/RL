"""Validate generated launch commands without invoking Ray, Slurm, or CUDA."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestSuperLauncher(unittest.TestCase):
    def test_corrected_override_and_watcher_path(self) -> None:
        script = Path(__file__).with_name("run_super.sbatch").resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ray.sub").write_text(
                'printf "%s\\n" "$COMMAND" "$SETUP_COMMAND"\n'
            )
            result = subprocess.run(
                ["bash", str(script)],
                env={
                    **os.environ,
                    "CONTAINER": "/example/image.sqsh",
                    "HF_HOME": "/example/hf",
                    "BASE_LOG_DIR": "/example/results",
                    "AUDIT_ARM": "alltoall",
                    "AUDIT_VARIANT": "corrected07",
                    "AUDIT_PROFILE_BEFORE_GRAPHS": "0",
                    "SLURM_JOB_ID": "12345",
                    "SLURM_SUBMIT_DIR": directory,
                    "USER": "audit",
                },
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertIn(
            "+policy.generation.vllm_kwargs.worker_cls=cumem_probe_worker.CorrectedProbeWorker",
            result.stdout,
        )
        self.assertIn("--output '/example/results/12345-logs'", result.stdout)
        self.assertNotIn("${SLURM_JOB_ID}", result.stdout)
        self.assertIn("$(hostname).log", result.stdout)
        self.assertIn("grpo.max_num_steps=20", result.stdout)
