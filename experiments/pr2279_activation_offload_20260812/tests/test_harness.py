from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class TestActivationOffloadHarness(unittest.TestCase):
    def _run_lifecycle_checker(
        self,
        *,
        zero_rank: int | None = None,
        missing_rank: int | None = None,
        missing_step: int | None = None,
        nonfinite_metric: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        rank_rows = []
        for rank in range(16):
            if rank == missing_rank:
                continue
            value = 0.0 if rank == zero_rank else 96.0 + rank
            rank_rows.append(f"Rank {rank:<2} {value:12.2f} {value:12.2f}")
        log = "\n".join(
            [
                "Activation Offload Summary (MB)",
                "Rank          moe_act       Total",
                *rank_rows,
                "Total         1656.00     1656.00",
            ]
        )
        steps = [step for step in (1, 2, 3) if step != missing_step]
        metrics = {
            metric: {
                str(step): (
                    "NaN" if metric == nonfinite_metric and step == 2 else step / 100
                )
                for step in steps
            }
            for metric in ("train/loss", "train/grad_norm")
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            log_path = tmp / "lifecycle.log"
            metrics_path = tmp / "metrics.json"
            output_path = tmp / "acceptance.json"
            log_path.write_text(log)
            metrics_path.write_text(json.dumps(metrics))
            return subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/check_lifecycle.py"),
                    "--log",
                    str(log_path),
                    "--metrics",
                    str(metrics_path),
                    "--expected-steps",
                    "3",
                    "--expected-world-size",
                    "16",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                check=False,
                text=True,
            )

    def test_lifecycle_checker_accepts_nonzero_complete_run(self) -> None:
        result = self._run_lifecycle_checker()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lifecycle_checker_rejects_zero_offload(self) -> None:
        result = self._run_lifecycle_checker(zero_rank=7)

        self.assertNotEqual(result.returncode, 0)

    def test_lifecycle_checker_rejects_missing_rank(self) -> None:
        result = self._run_lifecycle_checker(missing_rank=15)

        self.assertNotEqual(result.returncode, 0)

    def test_lifecycle_checker_rejects_missing_step(self) -> None:
        result = self._run_lifecycle_checker(missing_step=3)

        self.assertNotEqual(result.returncode, 0)

    def test_lifecycle_checker_rejects_nonfinite_metric(self) -> None:
        result = self._run_lifecycle_checker(nonfinite_metric="train/grad_norm")

        self.assertNotEqual(result.returncode, 0)

    def test_pair_is_dependency_matched(self) -> None:
        baseline = (ROOT / "configs/qwen30_off.yaml").read_text()
        treatment = (ROOT / "configs/qwen30_on.yaml").read_text()

        for config in (baseline, treatment):
            self.assertIn(
                "defaults: ../../../examples/configs/recipes/llm/performance/"
                "grpo-qwen3-30ba3b-4n4g.yaml",
                config,
            )
            self.assertIn("cuda_graph_impl: transformer_engine", config)
            self.assertIn('NVTE_CPU_OFFLOAD_V1: "1"', config)
            self.assertIn("max_num_steps: 10", config)

        normalized_baseline = re.sub(
            r"fine_grained_activation_offloading: false\n\s+offload_modules: null",
            "ACTIVATION_OFFLOAD_ARM",
            baseline,
        )
        normalized_treatment = re.sub(
            r"fine_grained_activation_offloading: true\n\s+offload_modules: \[\"moe_act\"\]",
            "ACTIVATION_OFFLOAD_ARM",
            treatment,
        )
        self.assertEqual(normalized_baseline, normalized_treatment)

    def test_stage_script_is_precluster_compatible(self) -> None:
        stage = (ROOT / "scripts/stage_enroot_image.sbatch").read_text()

        self.assertNotIn("--gres", stage)
        self.assertIn("SOURCE_IMAGE", stage)
        self.assertIn("SOURCE_COMMIT", stage)
        self.assertIn("sha256sum", stage)
        self.assertIn("metadata.txt", stage)

    def test_submitter_pulls_and_checks_schedule_before_submit(self) -> None:
        submitter = (ROOT / "scripts/submit_pair.sh").read_text()

        self.assertIn('git -C "${ROOT}" pull --ff-only', submitter)
        self.assertIn(
            "EXPECTED_RUNTIME_COMMIT=01398467224921c058a70702cb4a8285eb98fc71",
            submitter,
        )
        self.assertIn("EXPECTED_SOURCE_COMMIT", submitter)
        self.assertIn("merge-base --is-ancestor", submitter)
        self.assertIn("Evidence branch changes NeMo-RL runtime files", submitter)
        self.assertIn("NRL_FORCE_REBUILD_VENVS=false", submitter)
        self.assertIn("NVTE_CUDA_ARCHS=100", submitter)
        self.assertIn("pr2279-perf-${EXPECTED_RUNTIME_COMMIT:0:10}", submitter)
        self.assertIn('local venv_root="${VENV_ROOT}-${arm}"', submitter)
        self.assertIn('"${venv_root}"', submitter)
        self.assertIn("check_lifecycle.py", submitter)
        self.assertIn("acceptance.json", submitter)
        self.assertIn("provenance.txt", submitter)
        self.assertIn("sbatch --test-only", submitter)
        self.assertIn("sbatch --parsable", submitter)
        self.assertLess(
            submitter.index("sbatch --test-only"),
            submitter.index("sbatch --parsable"),
        )


if __name__ == "__main__":
    unittest.main()
