from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class TestActivationOffloadHarness(unittest.TestCase):
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
        self.assertIn("EXPECTED_RUNTIME_COMMIT", submitter)
        self.assertIn("EXPECTED_SOURCE_COMMIT", submitter)
        self.assertIn("merge-base --is-ancestor", submitter)
        self.assertIn("Evidence branch changes NeMo-RL runtime files", submitter)
        self.assertIn("NRL_FORCE_REBUILD_VENVS=false", submitter)
        self.assertIn("NVTE_CUDA_ARCHS=100", submitter)
        self.assertIn("pr2279-perf-${EXPECTED_SOURCE_COMMIT:0:10}", submitter)
        self.assertIn("sbatch --test-only", submitter)
        self.assertIn("sbatch --parsable", submitter)
        self.assertLess(
            submitter.index("sbatch --test-only"),
            submitter.index("sbatch --parsable"),
        )


if __name__ == "__main__":
    unittest.main()
