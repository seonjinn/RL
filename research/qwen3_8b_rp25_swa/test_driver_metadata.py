"""The source import path cannot substitute for installed driver metadata."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "research/qwen3_8b_rp25_swa/install_driver_project.sh"


class DriverMetadataTests(unittest.TestCase):
    def test_installs_project_without_resolving_training_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            (package / "nemo_rl").mkdir()
            (package / "nemo_rl/__init__.py").write_text("")
            (package / "pyproject.toml").write_text(
                '[build-system]\nrequires=["setuptools>=78.1.1"]\n'
                'build-backend="setuptools.build_meta"\n'
                '[project]\nname="nemo-rl"\nversion="0.0.1"\n'
                'dependencies=["TransferQueue @ git+https://invalid.example/tq.git@abc123"]\n'
                '[tool.setuptools]\npackages=["nemo_rl"]\n'
            )
            env = root / "venv"
            subprocess.run(
                ["uv", "venv", "--python", sys.executable, str(env)], check=True,
                capture_output=True,
            )
            python = env / "bin/python"
            probe = 'from importlib.metadata import requires; print(requires("nemo-rl"))'
            before = subprocess.run([str(python), "-c", probe], capture_output=True)
            self.assertNotEqual(before.returncode, 0)
            result = subprocess.run(
                ["bash", str(SCRIPT), str(python), str(package)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            after = subprocess.run(
                [str(python), "-c", probe], capture_output=True, text=True, check=True,
            )
            self.assertIn("abc123", after.stdout)
            self.assertIn("DRIVER_METADATA_PREFLIGHT=PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
