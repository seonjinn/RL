from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(__file__).with_name("submit.sh")
REFERENCE = ROOT / "experiments/q30_dapo_concurrency_20260912/submit.sh"


def render(
    script: Path, arm: str, concurrency: str, steps: int = 3
) -> subprocess.CompletedProcess[str]:
    env = dict(
        os.environ,
        Q30_DEEPSCALER_MAX_STEPS=str(steps),
        Q30_DAPO47K_MAX_STEPS=str(steps),
    )
    return subprocess.run(
        ["bash", str(script), "--render", arm, concurrency],
        env=env,
        text=True,
        capture_output=True,
    )


def overrides(script: str) -> dict[str, str]:
    line = next(
        line for line in script.splitlines() if line.startswith("export COMMAND=")
    )
    command = shlex.split(line.removeprefix("export COMMAND="))[0]
    tokens = shlex.split(command)
    tokens = tokens[tokens.index("--config") + 2 :]
    return dict(token.lstrip("+").split("=", 1) for token in tokens)


class DeepScalerLauncherTest(unittest.TestCase):
    def test_only_dataset_verifier_and_metadata_change(self) -> None:
        for arm, concurrency in (
            ("baseline", "default"),
            ("dflash_k5", "64"),
            ("dspark_k5", "64"),
        ):
            with self.subTest(arm=arm):
                result = render(LAUNCHER, arm, concurrency, 20)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = overrides(result.stdout)
                expected = overrides(render(REFERENCE, arm, concurrency, 20).stdout)
                self.assertEqual(actual.pop("data.train.dataset_name"), "DeepScaler")
                self.assertEqual(
                    actual.pop("env.math.math_verify_impl"), "hf_math_verify"
                )
                expected.pop("data.train.dataset_name")
                expected.pop("env.math.math_verify_impl")
                for key in (
                    "logger.wandb.group",
                    "logger.wandb.name",
                    "logger.log_dir",
                ):
                    self.assertIn("deepscaler", actual.pop(key).lower())
                    expected.pop(key)
                self.assertEqual(actual, expected)
                self.assertNotIn("#SBATCH --dependency", result.stdout)
                self.assertIn("check_dataset.py", result.stdout)

    def test_gate_has_three_steps_and_no_specdec_or_concurrency_override(self) -> None:
        result = render(LAUNCHER, "baseline", "default")
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = overrides(result.stdout)
        self.assertEqual(actual["grpo.max_num_steps"], "3")
        self.assertNotIn("policy.generation.vllm_kwargs.max_num_seqs", actual)
        self.assertEqual(
            actual["policy.generation.vllm_kwargs.speculative_config"], "null"
        )
        self.assertIn("#SBATCH --partition=batch\n", result.stdout)
        self.assertIn("export HF_HOME=", result.stdout)

    def test_invalid_steps_are_rejected_before_render(self) -> None:
        result = render(LAUNCHER, "baseline", "default", 0)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
