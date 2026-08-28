from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
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

    def test_matrix_contains_matched_dspark_k3_k5_k7_arms(self) -> None:
        expected = {
            "baseline": (None, 0),
            "dspark_k3": ("dspark", 3),
            "dspark_k5": ("dspark", 5),
            "dspark_k7": ("dspark", 7),
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
        for arm in ("baseline", "dspark_k3", "dspark_k5", "dspark_k7"):
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
        for arm in ("baseline", "dspark_k3", "dspark_k5", "dspark_k7"):
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
        expected_max_steps = int(os.environ.get("Q235_MAX_STEPS", "20"))
        expected = {
            "baseline": (None, 0),
            "dspark_k3": ("dspark", 3),
            "dspark_k5": ("dspark", 5),
            "dspark_k7": ("dspark", 7),
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
                self.assertEqual(manifest["max_steps"], expected_max_steps)
                self.assertEqual(manifest["slurm"]["nodes"], 32)
                self.assertEqual(manifest["slurm"]["gpus_per_node"], 4)

    def test_launcher_accepts_only_the_base_dspark_checkpoint(self) -> None:
        for arm in ("baseline", "dspark_k3", "dspark_k5", "dspark_k7"):
            with self.subTest(arm=arm):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "--validate-arm-contract", arm],
                    cwd=EXPERIMENT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_arms_use_fap_and_dspark_block_size_eight(self) -> None:
        for arm in ("baseline", "dspark_k3", "dspark_k5", "dspark_k7"):
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                generation = config["policy"]["generation"]
                kwargs = generation["vllm_kwargs"]
                self.assertEqual(
                    kwargs["compilation_config"]["cudagraph_mode"],
                    "FULL_AND_PIECEWISE",
                )
                if arm != "baseline":
                    speculative = kwargs["speculative_config"]
                    self.assertIn(
                        "qwen3-235ba22b-base-nemotron-b8-s25391/dspark",
                        speculative["model"],
                    )
                    self.assertEqual(speculative["draft_tensor_parallel_size"], 1)

    def test_dspark_capture_sizes_cover_k_and_bonus_widths(self) -> None:
        batches = (1, 2, 4, 8, 16, 32)
        for k in (3, 5, 7):
            with self.subTest(k=k):
                config = self.load_config(f"dspark_k{k}")
                sizes = set(
                    config["policy"]["generation"]["vllm_kwargs"]["compilation_config"][
                        "cudagraph_capture_sizes"
                    ]
                )
                self.assertTrue({k * batch for batch in batches} <= sizes)
                self.assertTrue({(k + 1) * batch for batch in batches} <= sizes)

    def test_submission_record_identity_includes_config_and_harness(self) -> None:
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--emit-submission-record", "baseline"],
            cwd=EXPERIMENT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record_name = Path(result.stdout.strip()).name
        config_sha = hashlib.sha256(
            (CONFIG_ROOT / "baseline.yaml").read_bytes()
        ).hexdigest()
        harness_sha = subprocess.run(
            ["git", "-C", str(EXPERIMENT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertTrue(
            record_name.endswith(f"-{config_sha}-{harness_sha}.json"), record_name
        )

    def test_launcher_pins_clean_product_and_node_local_overlays(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("d5c8bfa987025949699f7cfff188b349480bb8b5", launcher)
        self.assertIn("source is dirty", launcher)
        self.assertIn("Q235_MCORE_OVERLAY", launcher)
        self.assertIn("NRL_VENV_POST_SYNC_SCRIPT", launcher)
        self.assertIn("vllm-0.25.1-pr48167-runtime.patch", launcher)
        self.assertIn("vllm-0.25.1-pr48167-group-causality-followup.patch", launcher)
        self.assertIn(
            "504730a52614fddeb8ea899ec37a0aa820dcbc3a57c704fc13f5834fcc07b317",
            launcher,
        )
        self.assertIn(
            "8e5ff0e385ee44cf71e1e07031e5cd19658b29eb7b90bc172a4754c599d1dd90",
            launcher,
        )
        self.assertIn("verify_composed_configs.py", launcher)

    def test_rendered_dspark_job_uses_fap_overlay_only_for_dspark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rendered: dict[str, str] = {}
            for arm in ("baseline", "dspark_k3", "dspark_k5", "dspark_k7"):
                env = {**os.environ, "Q235_RENDER_ROOT": temporary}
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "--render-sbatch", arm],
                    cwd=EXPERIMENT_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                rendered[arm] = Path(result.stdout.strip()).read_text()

            self.assertNotIn("NRL_VENV_POST_SYNC_SCRIPT", rendered["baseline"])
            for arm in ("dspark_k3", "dspark_k5", "dspark_k7"):
                self.assertIn("NRL_VENV_POST_SYNC_SCRIPT", rendered[arm])
                self.assertIn("Q235_VLLM_OVERLAY", rendered[arm])
                self.assertIn("/raid:/raid", rendered[arm])


if __name__ == "__main__":
    unittest.main()
