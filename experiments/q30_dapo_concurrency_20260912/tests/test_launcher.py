from __future__ import annotations

import ast
import os
from pathlib import Path
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "experiments/q30_dapo_concurrency_20260912/submit.sh"


def render(
    arm: str, concurrency: str, *, steps: int = 1, dependency: str = ""
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["Q30_DAPO47K_MAX_STEPS"] = str(steps)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--render", arm, concurrency, dependency],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def overrides(script: str) -> dict[str, str]:
    line = next(
        line for line in script.splitlines() if line.startswith("export COMMAND=")
    )
    command = shlex.split(line.removeprefix("export COMMAND="))[0]
    return dict(
        token.lstrip("+").split("=", 1)
        for token in shlex.split(command)
        if "=" in token
    )


class ConcurrencyTest(unittest.TestCase):
    def test_sweep_only_changes_concurrency_graphs_and_run_metadata(self) -> None:
        for arm in ("baseline", "dflash_k5", "dspark_k5"):
            reference = None
            for concurrency in (16, 32, 64, 128):
                with self.subTest(arm=arm, concurrency=concurrency):
                    result = render(arm, str(concurrency))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    config = overrides(result.stdout)
                    self.assertEqual(config["policy.train_global_batch_size"], "2048")
                    self.assertEqual(config["grpo.num_prompts_per_step"], "128")
                    self.assertEqual(config["grpo.num_generations_per_prompt"], "16")
                    self.assertEqual(
                        config["policy.generation.vllm_kwargs.max_num_seqs"],
                        str(concurrency),
                    )
                    self.assertEqual(
                        config["policy.generation.vllm_kwargs.moe_backend"],
                        "flashinfer_trtllm",
                    )
                    self.assertEqual(
                        config[
                            "policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune"
                        ],
                        "false",
                    )
                    for key in (
                        "policy.generation.vllm_kwargs.max_num_seqs",
                        "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes",
                        "logger.wandb.name",
                        "logger.log_dir",
                    ):
                        config.pop(key)
                    if reference is None:
                        reference = config
                    else:
                        self.assertEqual(config, reference)

    def test_graphs_cover_target_and_drafter_widths_with_bounded_padding(self) -> None:
        for arm, widths in (
            ("baseline", (1,)),
            ("dflash_k5", (6,)),
            ("dspark_k5", (5, 6)),
        ):
            for concurrency in (16, 32, 64, 128):
                with self.subTest(arm=arm, concurrency=concurrency):
                    result = render(arm, str(concurrency))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    sizes = ast.literal_eval(
                        overrides(result.stdout)[
                            "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes"
                        ]
                    )
                    self.assertEqual(sizes, sorted(set(sizes)))
                    for width in widths:
                        self.assertIn(concurrency * width, sizes)
                        for live in range(1, concurrency + 1):
                            eligible = [
                                size
                                for size in sizes
                                if size % width == 0 and size >= live * width
                            ]
                            self.assertTrue(eligible)
                            self.assertLessEqual(min(eligible), 2 * live * width)

    def test_method_and_base_checkpoint_pairing(self) -> None:
        for method in ("dflash", "dspark"):
            result = render(f"{method}_k5", "32")
            self.assertEqual(result.returncode, 0, result.stderr)
            config = overrides(result.stdout)
            self.assertEqual(
                config["policy.generation.vllm_kwargs.speculative_config.method"],
                method,
            )
            self.assertTrue(
                config[
                    "policy.generation.vllm_kwargs.speculative_config.model"
                ].endswith(
                    f"sd2p3swa-q30-base-ptv3swe-{method}-b8-16n/exported-checkpoint-44000"
                )
            )
            self.assertEqual(config["policy.draft.enabled"], "false")
        result = render("baseline", "32")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            overrides(result.stdout)[
                "policy.generation.vllm_kwargs.speculative_config"
            ],
            "null",
        )

    def test_20_steps_use_batch_4h_and_only_their_own_gate(self) -> None:
        result = render("dspark_k5", "128", steps=20, dependency="1234567")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("#SBATCH --time=04:00:00", result.stdout)
        self.assertIn("#SBATCH --dependency=afterok:1234567", result.stdout)
        self.assertIn("#SBATCH --kill-on-invalid-dep=yes", result.stdout)
        self.assertEqual(overrides(result.stdout)["grpo.max_num_steps"], "20")
        self.assertIn("export RAY_TMPDIR=/raid/scratch/sna/r", result.stdout)

    def test_invalid_sweep_values_rejected_before_submission(self) -> None:
        for arm, concurrency, dependency in (
            ("baseline", "0", ""),
            ("baseline", "256", ""),
            ("baseline", "16;id", ""),
            ("dflash2_k5", "16", ""),
            ("baseline", "16", "123;id"),
        ):
            with self.subTest(arm=arm, concurrency=concurrency, dependency=dependency):
                result = render(arm, concurrency, dependency=dependency)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
