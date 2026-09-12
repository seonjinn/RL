import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def test_render_needs_no_cluster_and_produces_valid_shell(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "submit.sh"), "--render"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        syntax = subprocess.run(
            ["bash", "-n"],
            input=result.stdout,
            text=True,
            capture_output=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertTrue(result.stdout.startswith("#!/usr/bin/env bash\n"))

    def test_invalid_mode_cannot_submit(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "submit.sh"), "--unexpected"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_invalid_dependency_is_rejected_before_submission(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "submit.sh"), "--render"],
            env={**os.environ, "SWE_GATE_DEPENDENCY": "not-a-job"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("SWE_GATE_DEPENDENCY", result.stderr)


if __name__ == "__main__":
    unittest.main()
