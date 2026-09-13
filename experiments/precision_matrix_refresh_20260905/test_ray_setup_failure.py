"""Execute both generated Ray setup blocks without Slurm or model dependencies."""

import pathlib
import re
import subprocess
import tempfile
import unittest


class RaySetupFailureTest(unittest.TestCase):
    def test_setup_status(self) -> None:
        source = (pathlib.Path(__file__).resolve().parents[2] / "ray.sub").read_text()
        blocks = re.findall(
            r'  if \[\[ -n "\$SETUP_COMMAND_FILE".*?^  fi', source, re.S | re.M
        )
        self.assertEqual(len(blocks), 2)
        for index, block in enumerate(blocks):
            for status in (0, 11, 17):
                with self.subTest(role=index, status=status), tempfile.TemporaryDirectory() as root:
                    directory = pathlib.Path(root)
                    setup = directory / "setup.sh"
                    setup.write_text(f"exit {status}\n")
                    script = (
                        f'SETUP_COMMAND_FILE="{setup}"\nLOG_DIR="{root}"\n'
                        + block.replace(r"\$", "$")
                        + '\nprintf "RAY_START_REACHED\\n"\n'
                    )
                    result = subprocess.run(
                        ["bash", "-c", script], capture_output=True, text=True, timeout=10
                    )
                    self.assertEqual(result.returncode, status, result.stdout + result.stderr)
                    self.assertEqual("RAY_START_REACHED" in result.stdout, status == 0)
                    self.assertEqual((directory / "ENDED").exists(), status != 0)


if __name__ == "__main__":
    unittest.main()
