from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/qwen3_30ba3b_ptv2_frozen_20step_20260831"
PTV2_ROOT = "/lustre/fsw/portfolios/coreai/users/sna/specdec_ptv23/ptv2_final"


class FrozenGateContractTest(unittest.TestCase):
    maxDiff = None

    def matrix(self) -> dict[str, object]:
        result = subprocess.run(
            ["python3", str(EXPERIMENT / "matrix.py"), "--json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_first_stage_has_matched_math_and_swe_cohorts(self) -> None:
        matrix = self.matrix()
        rows = matrix["rows"]
        self.assertEqual(len(rows), 11)
        self.assertEqual(
            {(row["domain"], row["arm"]) for row in rows},
            {
                ("math", "baseline"),
                ("math", "dflash_k7"),
                ("math", "dspark_k5"),
                ("math", "dflash2_k7"),
                ("swe", "baseline"),
                ("swe", "dflash_k3"),
                ("swe", "dflash_k5"),
                ("swe", "dflash_k7"),
                ("swe", "dspark_k3"),
                ("swe", "dspark_k5"),
                ("swe", "dflash2_k7"),
            },
        )

    def test_base_and_thinking_drafters_never_cross(self) -> None:
        for row in self.matrix()["rows"]:
            checkpoint = row["drafter_checkpoint"]
            if row["arm"] == "baseline":
                self.assertIsNone(checkpoint)
            elif row["domain"] == "math":
                self.assertIn(f"{PTV2_ROOT}/sd2en-q30-base-ptv2en-", checkpoint)
            else:
                self.assertIn(f"{PTV2_ROOT}/sd2en-q30-thinking-ptv2en-", checkpoint)
            if checkpoint:
                self.assertTrue(checkpoint.endswith("/exported-checkpoint-25391"))

    def test_frozen_rows_disable_policy_draft_training(self) -> None:
        for row in self.matrix()["rows"]:
            self.assertEqual(row["training_mode"], "frozen")
            self.assertFalse(row["policy_draft_enabled"])
            self.assertFalse(row["draft_refit_enabled"])

    def test_dflash2_is_a_separate_runtime_cohort(self) -> None:
        rows = self.matrix()["rows"]
        stable = {row["runtime_cohort"] for row in rows if row["arm"] != "dflash2_k7"}
        dflash2 = {row["runtime_cohort"] for row in rows if row["arm"] == "dflash2_k7"}
        self.assertEqual(stable, {"stable-vllm-0.25.1"})
        self.assertEqual(dflash2, {"dflash2-vllm-pr52816"})
        for row in rows:
            if row["arm"] == "dflash2_k7":
                self.assertEqual(row["method"], "dflash")
                self.assertEqual(row["k"], 7)

    def test_math_launcher_preserves_official_recipe_and_twenty_steps(self) -> None:
        for arm in ("baseline", "dflash_k7", "dspark_k5"):
            result = subprocess.run(
                ["bash", str(EXPERIMENT / "submit_math_gate.sh"), "--render", arm],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = result.stdout
            self.assertIn("grpo-qwen3-30ba3b-4n4g.yaml", rendered)
            self.assertIn("grpo.max_num_steps=20", rendered)
            self.assertIn("policy.draft.enabled=false", rendered)
            self.assertIn("sequence_packing", rendered)
            self.assertIn("FULL_AND_PIECEWISE", rendered)
            self.assertIn(
                "++policy.generation.vllm_kwargs.compilation_config."
                "cudagraph_capture_sizes=",
                rendered,
            )
            self.assertIn("--nodes=4", rendered)
            self.assertIn("--gpus-per-node=4", rendered)
            self.assertIn(
                "export PATH=/cm/local/apps/slurm/25.11/bin:${PATH}", rendered
            )
            self.assertIn("Q30_MCORE_OVERLAY", rendered)
            if arm == "dspark_k5":
                self.assertIn(
                    'prepare_vllm_dspark_fap_overlay.py" --overlay-root '
                    '"${Q30_VLLM_OVERLAY}"',
                    rendered,
                )
                self.assertNotIn("NRL_VENV_POST_SYNC_SCRIPT", rendered)
            else:
                self.assertNotIn("prepare_vllm_dspark_fap_overlay.py", rendered)
                self.assertNotIn("NRL_VENV_POST_SYNC_SCRIPT", rendered)

            self.assertNotIn("/home/sna/script/export_env_vars.sh", rendered)
            self.assertIn('test -n "${WANDB_API_KEY:-}"', rendered)

    def test_report_is_self_contained_and_marks_pending_results(self) -> None:
        report = ROOT / "docs/reports/q30_ptv2_math_swe_performance.html"
        text = report.read_text()
        self.assertIn("Qwen3-30B-A3B PTV2", text)
        self.assertIn("Frozen-first", text)
        self.assertIn("DFlash2", text)
        self.assertIn("Historical comparison", text)
        self.assertNotIn("<script src=", text)


if __name__ == "__main__":
    unittest.main()
