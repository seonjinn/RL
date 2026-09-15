"""Exercise interpreter selection and fail-closed canary rendering."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "research/qwen3_8b_rp25_swa/render_canary.sh"


class DriverRenderTests(unittest.TestCase):
    def run_render(
        self, python: Path, arm: str, output: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), str(python), arm, str(output)],
            cwd=ROOT,
            env={**os.environ, "UV_OFFLINE": "1"},
            capture_output=True,
            text=True,
        )

    def test_valid_interpreter_renders_both_methods(self) -> None:
        for arm in ("dflash-always", "dspark-always"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                result = self.run_render(Path(sys.executable), arm, output)
                self.assertEqual(result.returncode, 0, result.stderr)
                values = (output / "overrides.txt").read_text().splitlines()
                self.assertIn("++grpo.max_num_steps=2", values)
                self.assertIn("++policy.draft.sliding_window=2048", values)
                self.assertTrue(
                    (output / "recipe.txt").read_text().strip().endswith(".yaml")
                )

    def test_missing_and_dangling_interpreters_fail_before_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            missing = output / "missing-python"
            dangling = output / "dangling-python"
            dangling.symlink_to(missing)
            for python in (missing, dangling):
                result = self.run_render(python, "dflash-always", output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("DRIVER_PYTHON_UNUSABLE", result.stderr)
                self.assertFalse((output / "recipe.txt").exists())

    def test_renderer_error_is_not_hidden_by_process_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = self.run_render(Path(sys.executable), "invalid-arm", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((output / "recipe.txt").exists())


if __name__ == "__main__":
    unittest.main()
