"""Contracts for the GBS512 300-step segmented SLURM array."""

from __future__ import annotations

import importlib
import unittest


class SegmentMatrixTests(unittest.TestCase):
    def module(self):
        return importlib.import_module(
            "research.qwen3_8b_rp25_swa.segment_matrix"
        )

    def test_matrix_has_three_baselines_and_twelve_specdec_conditions(self) -> None:
        matrix = self.module()
        arms = matrix.build_segment_arms()
        self.assertEqual(len(arms), 15)
        self.assertEqual([arm.label for arm in arms[:3]], [
            "baseline-default",
            "baseline-s64",
            "baseline-s128",
        ])
        self.assertEqual(sum(arm.method == "none" for arm in arms), 3)
        self.assertEqual(sum(arm.method == "dflash" for arm in arms), 6)
        self.assertEqual(sum(arm.method == "dspark" for arm in arms), 6)

    def test_correlated_stages_reach_300_without_cross_condition_edges(self) -> None:
        matrix = self.module()
        first_baseline = matrix.segment_for(task_index=0, stage=1)
        last_baseline = matrix.segment_for(task_index=2, stage=20)
        first_specdec = matrix.segment_for(task_index=3, stage=1)
        last_specdec = matrix.segment_for(task_index=14, stage=15)

        self.assertEqual(
            (first_baseline.previous_step, first_baseline.stop_step), (0, 15)
        )
        self.assertEqual(
            (last_baseline.previous_step, last_baseline.stop_step), (285, 300)
        )
        self.assertEqual(
            (first_specdec.previous_step, first_specdec.stop_step), (0, 20)
        )
        self.assertEqual(
            (last_specdec.previous_step, last_specdec.stop_step), (280, 300)
        )
        self.assertEqual(matrix.stage_task_count(15), 15)
        self.assertEqual(matrix.stage_task_count(16), 3)
        with self.assertRaises(ValueError):
            matrix.segment_for(task_index=3, stage=16)

    def test_stage_commands_use_aftercorr_and_fail_closed_dependencies(self) -> None:
        matrix = self.module()
        common = matrix.SubmissionInputs(
            expected_head="a" * 40,
            bundle="/lustre/source.bundle",
            bundle_sha="b" * 64,
            result_parent="/lustre/results",
            account="coreai_dlalgo_nemorl",
            script="research/qwen3_8b_rp25_swa/run_segment_array.sbatch",
            log_dir="/lustre/results/scheduler-logs",
        )
        stage1 = matrix.build_stage_command(common, stage=1, dependency=None)
        stage2 = matrix.build_stage_command(common, stage=2, dependency="12345")
        stage16 = matrix.build_stage_command(common, stage=16, dependency="22345")

        self.assertIn("--array=0-14", stage1)
        self.assertNotIn("--dependency", stage1)
        self.assertIn("--dependency=aftercorr:12345", stage2)
        self.assertIn("--kill-on-invalid-dep=yes", stage2)
        self.assertIn("--array=0-2", stage16)
        self.assertEqual(stage1[-1], "1")
        self.assertEqual(stage16[-1], "16")

    def test_specdec_retry_commands_preserve_original_array_indices(self) -> None:
        matrix = self.module()
        common = matrix.SubmissionInputs(
            expected_head="a" * 40,
            bundle="/lustre/source.bundle",
            bundle_sha="b" * 64,
            result_parent="/lustre/specdec-retry",
            account="coreai_dlalgo_nemorl",
            script="research/qwen3_8b_rp25_swa/run_segment_array.sbatch",
            log_dir="/lustre/specdec-retry/scheduler-logs",
        )
        stage1 = matrix.build_stage_command(
            common,
            stage=1,
            dependency=None,
            task_range=(3, 14),
        )
        stage15 = matrix.build_stage_command(
            common,
            stage=15,
            dependency="12345",
            task_range=(3, 14),
        )

        self.assertIn("--array=3-14", stage1)
        self.assertIn("--array=3-14", stage15)
        self.assertIn("--dependency=aftercorr:12345", stage15)
        with self.assertRaises(ValueError):
            matrix.build_stage_command(
                common,
                stage=16,
                dependency="12345",
                task_range=(3, 14),
            )


if __name__ == "__main__":
    unittest.main()
