from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/qwen3_30ba3b_ptv3_swa_online_30step_20260910"
PTV3_ROOT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "specdec_ptv23/ptv3_swa"
)


class Ptv3SwaOnline30ContractTest(unittest.TestCase):
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

    def render(self, arm: str) -> str:
        env = os.environ.copy()
        env["Q30_PTV3_ACCOUNT"] = "nemotron_n4_post"
        result = subprocess.run(
            ["bash", str(EXPERIMENT / "submit_math_gate.sh"), "--render", arm],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_matrix_contains_only_approved_online_arms(self) -> None:
        rows = self.matrix()["rows"]
        self.assertEqual(
            {(row["method"], row["k"], row["fixed_interval"]) for row in rows},
            {
                ("dflash", 5, 5),
                ("dflash", 5, 10),
                ("dspark", 7, 5),
                ("dspark", 7, 10),
            },
        )
        self.assertTrue(all(row["max_steps"] == 30 for row in rows))

    def test_matrix_uses_checkpoint_native_ptv3_swa_44k_drafters(self) -> None:
        for row in self.matrix()["rows"]:
            checkpoint = row["drafter_checkpoint"]
            self.assertIn(
                f"{PTV3_ROOT}/sd2p3swa-q30-base-ptv3swe-{row['method']}-b8-16n/",
                checkpoint,
            )
            self.assertTrue(checkpoint.endswith("/exported-checkpoint-44000"))

    def test_render_preserves_matched_performance_runtime(self) -> None:
        for arm in ("dflash_k5_fixed5", "dflash_k5_fixed10", "dspark_k7_fixed5", "dspark_k7_fixed10"):
            with self.subTest(arm=arm):
                rendered = self.render(arm)
                self.assertIn("grpo-qwen3-30ba3b-4n4g.yaml", rendered)
                self.assertIn("grpo.max_num_steps=30", rendered)
                self.assertIn("policy.draft.enabled=true", rendered)
                self.assertIn("policy.sequence_packing.enabled=true", rendered)
                self.assertIn("policy.offload_optimizer_for_refit=false", rendered)
                self.assertIn("moe_backend=flashinfer_trtllm", rendered)
                self.assertIn("cudagraph_mode=FULL_AND_PIECEWISE", rendered)
                self.assertIn("#SBATCH --nodes=4", rendered)
                self.assertIn("#SBATCH --gpus-per-node=4", rendered)
                self.assertIn("#SBATCH --partition=batch", rendered)
                self.assertIn("logger.wandb.project=sna-specdec", rendered)
                self.assertIn(
                    "logger.wandb.group=q30-ptv3-swa-44k-math-online30",
                    rendered,
                )
                self.assertIn(
                    "15554749ae24361b5d511e72ddf41ecab2615cdc",
                    rendered,
                )

    def test_render_binds_method_k_and_interval_consistently(self) -> None:
        expected = {
            "dflash_k5_fixed5": ("dflash", 5, 5),
            "dflash_k5_fixed10": ("dflash", 5, 10),
            "dspark_k7_fixed5": ("dspark", 7, 5),
            "dspark_k7_fixed10": ("dspark", 7, 10),
        }
        for arm, (method, k, interval) in expected.items():
            with self.subTest(arm=arm):
                rendered = self.render(arm)
                self.assertIn(f"policy.draft.speculator_type={method}", rendered)
                self.assertIn(f"num_speculative_tokens={k}", rendered)
                self.assertIn(
                    f"policy.draft.update_schedule.fixed_interval={interval}",
                    rendered,
                )
                self.assertIn(
                    f"sd2p3swa-q30-base-ptv3swe-{method}-b8-16n/"
                    "exported-checkpoint-44000",
                    rendered,
                )

    def test_submit_matrix_has_no_slurm_dependencies(self) -> None:
        text = (EXPERIMENT / "submit_online_matrix.sh").read_text()
        self.assertNotIn("--dependency", text)
        self.assertIn("dflash_k5_fixed5", text)
        self.assertIn("dflash_k5_fixed10", text)
        self.assertIn("dspark_k7_fixed5", text)
        self.assertIn("dspark_k7_fixed10", text)


if __name__ == "__main__":
    unittest.main()
