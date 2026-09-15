import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DefaultConcurrencyTest(unittest.TestCase):
    def test_default_baseline_omits_override_in_driver_and_verifier(self):
        launcher = Path(__file__).resolve().parents[1] / "submit_qwen3_30ba3b_20step.sh"
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                ["bash", str(launcher), "--render-sbatch", "baseline"],
                env={**os.environ, "Q30_20STEP_RENDER_ROOT": temporary,
                     "Q30_20STEP_CONCURRENCY": "default"},
                text=True, capture_output=True, check=True,
            )
            driver = (Path(result.stdout.strip()).parent / "driver.sh").read_text()
            self.assertNotIn("++policy.generation.vllm_kwargs.max_num_seqs=", driver)
            self.assertIn("--concurrency default", driver)
            self.assertIn("compilation_config.cudagraph_mode=PIECEWISE", driver)
            self.assertIn("compilation_config.cudagraph_capture_sizes=[1,2,4,8,12,16,24,32,40,48]", driver)


if __name__ == "__main__":
    unittest.main()
