import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from string import Template


ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def test_rendered_ray_socket_path_fits_linux_limit(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "submit.sh"), "--render"],
            capture_output=True,
            text=True,
            check=True,
        )
        environment = {"SLURM_JOB_ID": "12345678"}
        for line in result.stdout.splitlines():
            if line.startswith("export ") and "=" in line:
                key, value = line.removeprefix("export ").split("=", 1)
                environment[key] = Template(value).safe_substitute(environment)
        temporary_root = environment.get("RAY_TMPDIR", environment["TMPDIR"])
        socket_path = (
            temporary_root
            + "/ray/session_2026-09-12_00-00-00_123456_1234567/sockets/plasma_store"
        )
        self.assertLessEqual(len(socket_path.encode()), 107, socket_path)

    def test_node_staging_fixes_arm64_download_without_mutating_source(self) -> None:
        relative = Path("responses_api_agents/swe_agents/setup_scripts/openhands.sh")
        original = '    curl -fsSL https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64 -o "$miniforge_dir/bin/jq"\n'
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            source = temporary / "source"
            source_script = source / "3rdparty/Gym-workspace/Gym" / relative
            source_script.parent.mkdir(parents=True)
            source_script.write_text(original)
            result = subprocess.run(
                ["bash", str(ROOT / "stage_node.sh")],
                env={
                    **os.environ,
                    "SWE_SOURCE_ROOT": str(source),
                    "SWE_NODE_ROOT": str(temporary / "node"),
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source_script.read_text(), original)
            self.assertIn(
                "jq-linux-arm64", (temporary / "node/Gym" / relative).read_text()
            )

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
