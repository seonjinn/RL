from __future__ import annotations

import json
import subprocess
from pathlib import Path
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
LAUNCHER = EXPERIMENT_ROOT / "submit_qwen235b_math_grpo.sh"


class Qwen235BMathGrpoContractTest(unittest.TestCase):
    BASE_TARGET = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
        "hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots/"
        "8efa61729e24bd65b1d152b5ab5409052aa80e65"
    )

    def load_config(self, arm: str) -> dict[str, object]:
        path = CONFIG_ROOT / f"{arm}.yaml"
        self.assertTrue(path.is_file(), f"missing config: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_matrix_contains_matched_k3_k5_arms(self) -> None:
        expected = {
            "baseline": (None, 0),
            "dflash_k3": ("dflash", 3),
            "dflash_k5": ("dflash", 5),
            "dspark_k3": ("dspark", 3),
            "dspark_k5": ("dspark", 5),
        }

        for arm, (method, k) in expected.items():
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                policy = config["policy"]
                self.assertIsInstance(policy, dict)
                generation = policy["generation"]
                self.assertIsInstance(generation, dict)
                kwargs = generation["vllm_kwargs"]
                self.assertIsInstance(kwargs, dict)
                speculative = kwargs.get("speculative_config")
                if method is None:
                    self.assertIsNone(speculative)
                else:
                    self.assertIsInstance(speculative, dict)
                    self.assertEqual(speculative["method"], method)
                    self.assertEqual(speculative["num_speculative_tokens"], k)

    def test_all_arms_are_fixed_drafter_and_workload_matched(self) -> None:
        reference: tuple[object, ...] | None = None
        for arm in ("baseline", "dflash_k3", "dflash_k5", "dspark_k3", "dspark_k5"):
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                policy = config["policy"]
                self.assertIsInstance(policy, dict)
                self.assertNotIn("draft", policy)
                cluster = config["cluster"]
                grpo = config["grpo"]
                self.assertIsInstance(cluster, dict)
                self.assertIsInstance(grpo, dict)
                workload = (
                    config["defaults"],
                    policy["model_name"],
                    policy["train_global_batch_size"],
                    policy["max_total_sequence_length"],
                    grpo["num_prompts_per_step"],
                    grpo["num_generations_per_prompt"],
                    cluster["num_nodes"],
                    cluster["gpus_per_node"],
                )
                if reference is None:
                    reference = workload
                self.assertEqual(workload, reference)

    def test_all_math_arms_use_the_original_base_recipe_target(self) -> None:
        for arm in ("baseline", "dflash_k3", "dflash_k5", "dspark_k3", "dspark_k5"):
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                policy = config["policy"]
                self.assertIsInstance(policy, dict)
                self.assertEqual(policy["model_name"], self.BASE_TARGET)
                tokenizer = policy["tokenizer"]
                self.assertIsInstance(tokenizer, dict)
                self.assertEqual(tokenizer["name"], self.BASE_TARGET)

    def test_launcher_emits_immutable_arm_manifests(self) -> None:
        self.assertTrue(LAUNCHER.is_file(), f"missing launcher: {LAUNCHER}")
        expected = {
            "baseline": (None, 0),
            "dflash_k3": ("dflash", 3),
            "dflash_k5": ("dflash", 5),
            "dspark_k3": ("dspark", 3),
            "dspark_k5": ("dspark", 5),
        }
        for arm, (method, k) in expected.items():
            with self.subTest(arm=arm):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "--emit-manifest", arm],
                    cwd=EXPERIMENT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest = json.loads(result.stdout)
                self.assertEqual(manifest["arm"], arm)
                self.assertEqual(manifest["method"], method)
                self.assertEqual(manifest["num_speculative_tokens"], k)
                self.assertEqual(manifest["max_steps"], 3)
                self.assertEqual(manifest["slurm"]["nodes"], 32)
                self.assertEqual(manifest["slurm"]["gpus_per_node"], 4)

    def test_launcher_blocks_thinking_drafters_for_the_base_target(self) -> None:
        baseline = subprocess.run(
            ["bash", str(LAUNCHER), "--validate-arm-contract", "baseline"],
            cwd=EXPERIMENT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(baseline.returncode, 0, baseline.stderr)

        for arm in ("dflash_k3", "dflash_k5", "dspark_k3", "dspark_k5"):
            with self.subTest(arm=arm):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "--validate-arm-contract", arm],
                    cwd=EXPERIMENT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("matching Qwen3-235B-A22B Base drafter is required", result.stderr)

    def test_launcher_pins_the_only_allowed_generated_source_artifact(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn(
            "megatron/core/datasets/helpers_cpp",
            launcher,
        )
        self.assertIn(
            "39f37692b828622d8e40d13a683b5d0f511c7c852c7497edce286c7eda28833a",
            launcher,
        )
        self.assertIn("unexpected Megatron-LM worktree state", launcher)
        self.assertIn(
            '[[ "${root_state}" == " M ${BRIDGE_REL}" ]]',
            launcher,
        )
        self.assertIn(
            '[[ "${bridge_state}" == " M ${MEGATRON_REL}" ]]',
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
