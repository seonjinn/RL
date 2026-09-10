from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/qwen3_30ba3b_ptv3_swa_frozen_20step_20260909"
PTV3_ROOT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "specdec_ptv23/ptv3_swa"
)


class Ptv3SwaMathContractTest(unittest.TestCase):
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

    def render(self, arm: str, *, steps: int = 20) -> str:
        env = os.environ.copy()
        env["Q30_PTV3_MAX_STEPS"] = str(steps)
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

    def test_matrix_has_matched_baseline_and_stable_k_sweeps(self) -> None:
        rows = self.matrix()["rows"]
        self.assertEqual(
            {(row["arm"], row["status"]) for row in rows},
            {
                ("baseline", "ready"),
                ("dflash_k3", "ready"),
                ("dflash_k5", "ready"),
                ("dflash_k7", "ready"),
                ("dspark_k3", "ready"),
                ("dspark_k5", "ready"),
                ("dspark_k7", "ready"),
                ("dflash2_k7", "blocked-vllm-0.28"),
            },
        )

    def test_base_math_never_uses_thinking_drafter(self) -> None:
        for row in self.matrix()["rows"]:
            checkpoint = row["drafter_checkpoint"]
            if row["arm"] == "baseline":
                self.assertIsNone(checkpoint)
                continue
            self.assertIn(f"{PTV3_ROOT}/sd2p3swa-q30-base-", checkpoint)
            self.assertNotIn("q30-thinking", checkpoint)
            self.assertTrue(checkpoint.endswith("/exported-checkpoint-44000"))

    def test_all_rows_are_frozen_and_pin_runtime_cohort(self) -> None:
        for row in self.matrix()["rows"]:
            self.assertEqual(row["training_mode"], "frozen")
            self.assertFalse(row["policy_draft_enabled"])
            self.assertFalse(row["draft_refit_enabled"])
            expected = (
                "dflash2-vllm-0.28-pending"
                if row["arm"] == "dflash2_k7"
                else "stable-vllm-0.25.1-fap"
            )
            self.assertEqual(row["runtime_cohort"], expected)

    def test_stable_launcher_preserves_official_performance_recipe(self) -> None:
        capture_max = {
            "baseline": 256,
            "dflash_k3": 512,
            "dflash_k5": 768,
            "dflash_k7": 1024,
            "dspark_k3": 512,
            "dspark_k5": 768,
            "dspark_k7": 1024,
        }
        for arm, maximum in capture_max.items():
            with self.subTest(arm=arm):
                rendered = self.render(arm)
                self.assertIn("grpo-qwen3-30ba3b-4n4g.yaml", rendered)
                self.assertIn("grpo.max_num_steps=20", rendered)
                self.assertIn("policy.draft.enabled=false", rendered)
                self.assertIn(
                    "++policy.offload_optimizer_for_refit=false", rendered
                )
                self.assertIn(
                    "policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm",
                    rendered,
                )
                self.assertIn("FULL_AND_PIECEWISE", rendered)
                self.assertIn("++policy.generation.vllm_kwargs.max_num_seqs=128", rendered)
                self.assertIn(f"\\,{maximum}\\]", rendered)
                self.assertIn("#SBATCH --nodes=4", rendered)
                self.assertIn("#SBATCH --gpus-per-node=4", rendered)
                self.assertIn("#SBATCH --partition=batch", rendered)
                self.assertIn("logger.wandb.project=sna-specdec", rendered)
                self.assertIn("PTV3SWA-44K", rendered)
                if arm != "baseline":
                    method, k = arm.split("_k")
                    self.assertIn(f"speculative_config.method={method}", rendered)
                    self.assertIn(
                        f"speculative_config.num_speculative_tokens={k}", rendered
                    )
                    self.assertIn("exported-checkpoint-44000", rendered)
                    if method == "dspark":
                        self.assertIn("prepare_vllm_dspark_fap_overlay.py", rendered)
                        self.assertIn(
                            "speculative_config.attention_backend=FLASH_ATTN",
                            rendered,
                        )
                    else:
                        self.assertNotIn(
                            "prepare_vllm_dspark_fap_overlay.py", rendered
                        )
                        self.assertNotIn(
                            "speculative_config.attention_backend=FLASH_ATTN",
                            rendered,
                        )

    def test_dflash2_is_not_submitted_on_stable_runtime(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(EXPERIMENT / "submit_math_gate.sh"),
                "--render",
                "dflash2_k7",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 3)
        self.assertIn("vLLM >=0.28", result.stderr)

    def test_launcher_supports_one_step_smoke(self) -> None:
        self.assertIn("grpo.max_num_steps=1", self.render("baseline", steps=1))


if __name__ == "__main__":
    unittest.main()
