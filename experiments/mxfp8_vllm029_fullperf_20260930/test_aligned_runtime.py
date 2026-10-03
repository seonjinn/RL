"""Regression coverage for Ray's direct and uv-generated CLI entry points."""

from pathlib import Path
import tempfile
import unittest

from check_aligned_runtime import ray_cli_python


UV_TRAMPOLINE = (
    "#!/bin/sh\n"
    "'''exec' \"$(dirname -- \"$(realpath -- \"$0\")\")\"/'python3' \"$0\" \"$@\"\n"
    "' '''\n"
    "from ray.scripts.scripts import main\n"
)


class RayCliPythonTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.environment = Path(directory.name) / "driver"
        self.bin_dir = self.environment / "bin"
        self.bin_dir.mkdir(parents=True)
        self.cli = self.bin_dir / "ray"

    def test_direct_python_shebang(self) -> None:
        python = self.bin_dir / "python3.13"
        self.cli.write_text(f"#!{python}\nfrom ray.scripts.scripts import main\n")
        self.assertEqual(ray_cli_python(self.cli, environment=self.environment), python)

    def test_uv_trampoline_uses_adjacent_python(self) -> None:
        self.cli.write_text(UV_TRAMPOLINE)
        self.assertEqual(
            ray_cli_python(self.cli, environment=self.environment),
            self.bin_dir / "python3",
        )

    def test_uv_symlink_uses_real_script_directory(self) -> None:
        self.cli.write_text(UV_TRAMPOLINE)
        alias = self.environment.parent / "ray-alias"
        alias.symlink_to(self.cli)
        self.assertEqual(
            ray_cli_python(alias, environment=self.environment),
            self.bin_dir / "python3",
        )

    def test_direct_python_outside_driver_is_rejected(self) -> None:
        self.cli.write_text("#!/usr/bin/python3\n")
        with self.assertRaises(ValueError):
            ray_cli_python(self.cli, environment=self.environment)

    def test_arbitrary_shell_wrapper_is_rejected(self) -> None:
        self.cli.write_text("#!/bin/sh\nexec /usr/bin/python3 \"$0\" \"$@\"\n")
        with self.assertRaises(ValueError):
            ray_cli_python(self.cli, environment=self.environment)

    def test_uv_trampoline_outside_driver_is_rejected(self) -> None:
        external = self.environment.parent / "external-ray"
        external.write_text(UV_TRAMPOLINE)
        with self.assertRaises(ValueError):
            ray_cli_python(external, environment=self.environment)

    def test_python_name_prefix_is_not_sufficient(self) -> None:
        self.cli.write_text(f"#!{self.bin_dir}/python-elsewhere\n")
        with self.assertRaises(ValueError):
            ray_cli_python(self.cli, environment=self.environment)


if __name__ == "__main__":
    unittest.main()
