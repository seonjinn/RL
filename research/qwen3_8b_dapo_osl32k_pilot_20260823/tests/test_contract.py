"""Fail-closed contracts for the Qwen3-8B DAPO OSL32K training pilot."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_8b_dapo_osl32k_pilot_20260823"
TARGET = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "hf_home/hub/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
DFLASH = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "hf_home/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/"
    "9b41424b7109f9c5413454f481b09a82b85333f4"
)
DSPARK = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "hf_home/hub/models--deepseek-ai--dspark_qwen3_8b_block7/snapshots/"
    "03326e5043815da1f81b109078b2889737c26017"
)
VARIANTS = {
    "baseline-k0": (None, None),
    "dflash-k5": ("dflash", DFLASH),
    "dspark-k5": ("dspark", DSPARK),
}
CAPTURE_SIZES = [1, 2, 4, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48, 56, 64]
SOURCE_SHA = "9d99b16e7e6a9cb11ac01c893198d6a72b2214f5"
TARGET_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
}


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment() -> Path:
    return root() / "research" / EXPERIMENT


def harness() -> Path:
    return experiment() / "submit_qwen3_8b_dapo_osl32k_pilot.sh"


class PilotContractTest(unittest.TestCase):
    maxDiff = None

    def config(self, variant: str) -> dict[str, object]:
        return json.loads((experiment() / "configs" / f"{variant}.yaml").read_text())

    def verify_rendered_config(
        self,
        variant: str,
        config: dict[str, object],
        *,
        optimized: bool = False,
        static_only: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        recipe = (
            "grpo-qwen3-8b-1n8g-megatron-dspark.yaml"
            if variant == "dspark-k5"
            else "grpo-qwen3-8b-1n8g-megatron-dflash.yaml"
        )
        config["defaults"] = str(
            root() / "examples" / "configs" / "recipes" / "llm" / recipe
        )
        with tempfile.TemporaryDirectory() as tmp:
            rendered_input = Path(tmp) / f"{variant}.yaml"
            rendered_input.write_text(json.dumps(config))
            command = [sys.executable]
            if optimized:
                command.append("-O")
            command.extend(
                [
                    str(experiment() / "verify_pilot_config.py"),
                    "--source-root",
                    str(root()),
                    "--config",
                    str(rendered_input),
                    "--capture-sizes",
                    json.dumps(CAPTURE_SIZES),
                ]
            )
            if static_only:
                command.append("--static-only")
            return subprocess.run(
                command,
                text=True,
                capture_output=True,
            )

    def test_static_configs_use_tq_without_cadence_runtime(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                result = self.verify_rendered_config(variant, self.config(variant))
                self.assertEqual(result.returncode, 0, result.stderr)
                rendered = json.loads(result.stdout)
                self.assertTrue(rendered["data_plane_enabled"])
                self.assertFalse(rendered["cadence_runtime_enabled"])

    @unittest.skipUnless(
        importlib.util.find_spec("omegaconf") is not None,
        "real config composition requires the pinned Linux product environment",
    )
    def test_composed_configs_use_tq_without_cadence_runtime(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                result = self.verify_rendered_config(
                    variant, self.config(variant), static_only=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                rendered = json.loads(result.stdout)
                self.assertTrue(rendered["CONFIG_COMPOSE_GATE_PASS"])
                self.assertTrue(rendered["data_plane_enabled"])
                self.assertFalse(rendered["cadence_runtime_enabled"])

    def test_rendered_config_contract_fails_closed_on_trainer_routing(self) -> None:
        for optimized in (False, True):
            with self.subTest(optimized=optimized, mutation="data-plane-disabled"):
                tq_disabled = self.config("dflash-k5")
                tq_disabled["data_plane"] = {"enabled": False}
                result = self.verify_rendered_config(
                    "dflash-k5", tq_disabled, optimized=optimized
                )
                self.assertNotEqual(result.returncode, 0)

            with self.subTest(optimized=optimized, mutation="cadence-enabled"):
                cadence_enabled = self.config("dspark-k5")
                cadence_enabled["cadence_runtime"] = {"enabled": True}
                result = self.verify_rendered_config(
                    "dspark-k5", cadence_enabled, optimized=optimized
                )
                self.assertNotEqual(result.returncode, 0)

    def test_three_arms_are_matched_cp1_two_step_training_runs(self) -> None:
        for variant, (method, checkpoint) in VARIANTS.items():
            with self.subTest(variant=variant):
                config = self.config(variant)
                expected_source_root = (
                    f"/home/sna/nemorl-q8-dapo32k-tq-recovery-{variant}-"
                    "clean-20260823"
                )
                recipe = (
                    "grpo-qwen3-8b-1n8g-megatron-dspark.yaml"
                    if variant == "dspark-k5"
                    else "grpo-qwen3-8b-1n8g-megatron-dflash.yaml"
                )
                self.assertEqual(
                    config["defaults"],
                    f"{expected_source_root}/examples/configs/recipes/llm/{recipe}",
                )
                self.assertEqual(config["grpo"]["max_num_steps"], 2)
                self.assertEqual(config["grpo"]["num_prompts_per_step"], 2)
                self.assertEqual(config["grpo"]["num_generations_per_prompt"], 4)
                self.assertEqual(config["grpo"]["seed"], 42)
                policy = config["policy"]
                self.assertEqual(policy["model_name"], TARGET)
                self.assertEqual(policy["tokenizer"], {"name": TARGET})
                self.assertEqual(policy["train_global_batch_size"], 8)
                self.assertEqual(policy["max_total_sequence_length"], 40960)
                self.assertEqual(policy["sequence_packing"], {"enabled": False})
                megatron = policy["megatron_cfg"]
                self.assertEqual(megatron["tensor_model_parallel_size"], 2)
                self.assertEqual(megatron["pipeline_model_parallel_size"], 1)
                self.assertEqual(megatron["context_parallel_size"], 1)
                self.assertFalse(megatron["sequence_parallel"])
                self.assertTrue(megatron["activation_checkpointing"])
                generation = policy["generation"]
                self.assertEqual(generation["max_new_tokens"], 32768)
                self.assertEqual(generation["vllm_cfg"]["max_model_len"], 40960)
                self.assertEqual(
                    generation["vllm_kwargs"]["compilation_config"][
                        "cudagraph_capture_sizes"
                    ],
                    CAPTURE_SIZES,
                )
                if method is None:
                    self.assertEqual(policy["draft"], {"enabled": False})
                    self.assertIsNone(generation["vllm_kwargs"]["speculative_config"])
                    continue
                draft = policy["draft"]
                spec = generation["vllm_kwargs"]["speculative_config"]
                self.assertEqual(draft["model_name"], checkpoint)
                self.assertEqual(spec["method"], method)
                self.assertEqual(spec["model"], checkpoint)
                self.assertEqual(spec["num_speculative_tokens"], 5)
                if method == "dflash":
                    self.assertEqual(draft["gamma"], 5)
                else:
                    self.assertEqual(draft["block_size"], 7)
                    self.assertNotIn("gamma", draft)

    def test_model_length_has_k5_lookahead_headroom(self) -> None:
        self.assertGreaterEqual(40960, 2048 + 32768 + 5 + 1)

    def test_manifest_pins_identity_topology_and_runtime_gates(self) -> None:
        for variant, (method, checkpoint) in VARIANTS.items():
            with self.subTest(variant=variant):
                result = subprocess.run(
                    ["bash", str(harness()), "--emit-manifest", variant],
                    cwd=root(),
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest = json.loads(result.stdout)
                self.assertEqual(manifest["source"]["sha"], SOURCE_SHA)
                self.assertEqual(manifest["target"]["path"], TARGET)
                self.assertEqual(set(manifest["target"]["files"]), TARGET_FILES)
                self.assertEqual(manifest["checkpoint"], checkpoint)
                self.assertEqual(manifest["method"], method)
                self.assertEqual(
                    manifest["num_speculative_tokens"], 0 if method is None else 5
                )
                self.assertEqual(manifest["capture_sizes"], CAPTURE_SIZES)
                self.assertEqual(
                    manifest["topology"],
                    {
                        "nodes": 1,
                        "gpus_per_node": 4,
                        "tp": 2,
                        "pp": 1,
                        "dp": 2,
                        "cp": 1,
                        "sequence_packing": False,
                        "sequence_parallel": False,
                    },
                )
                self.assertEqual(manifest["global_batch_size"], 8)
                self.assertEqual(manifest["max_steps"], 2)
                self.assertEqual(manifest["wandb_project"], "sna-specdec")
                self.assertRegex(
                    manifest["wandb_run_id"],
                    rf"^q8-dapo-osl32k-tq-recovery-{re.escape(variant)}-"
                    rf"[0-9a-f]{{32}}$",
                )
                gates = set(manifest["gates"])
                self.assertTrue(
                    {
                        "source-clean",
                        "data-identity",
                        "config-compose",
                        "cudagraph",
                        "step1",
                        "step2",
                        "wake-refit",
                        "output-length",
                        "no-fatal",
                    }
                    <= gates
                )
                if method is not None:
                    self.assertIn("state-dict", gates)

    def test_all_arms_fail_closed_on_exact_target_bytes(self) -> None:
        identity = json.loads((experiment() / "checkpoint_identity.json").read_text())
        self.assertEqual(set(identity["target"]), TARGET_FILES)
        for metadata in identity["target"].values():
            self.assertRegex(metadata["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(metadata["size"], 0)

        text = harness().read_text()
        self.assertIn("--artifact target", text)
        self.assertIn('--root "${TARGET}"', text)

    def test_model_identity_gate_rejects_same_path_content_drift(self) -> None:
        verifier = experiment() / "verify_model_identity.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            model = root / "config.json"
            model.write_bytes(b"pinned")
            identity = Path(tmp) / "identity.json"
            identity.write_text(
                json.dumps(
                    {
                        "target": {
                            "config.json": {
                                "size": model.stat().st_size,
                                "sha256": hashlib.sha256(
                                    model.read_bytes()
                                ).hexdigest(),
                            }
                        }
                    }
                )
            )
            command = [
                "python3",
                str(verifier),
                "--artifact",
                "target",
                "--root",
                str(root),
                "--identity-file",
                str(identity),
                "--verify-content-sha",
            ]
            self.assertEqual(subprocess.run(command).returncode, 0)
            model.write_bytes(b"drifte")
            self.assertNotEqual(subprocess.run(command).returncode, 0)

    def test_capture_contract_covers_all_declared_shapes_through_64(self) -> None:
        result = subprocess.run(
            ["bash", str(harness()), "--assert-capture-coverage"],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coverage = json.loads(result.stdout)
        self.assertEqual(coverage["capture_sizes"], CAPTURE_SIZES)
        self.assertEqual(set(map(int, coverage["shape_to_bucket"])), set(range(1, 65)))
        self.assertEqual(max(coverage["shape_to_bucket"].values()), 64)

    def test_sbatch_uses_one_four_gpu_node_and_clean_arm_source(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                result = subprocess.run(
                    ["bash", str(harness()), "--render-sbatch", variant],
                    cwd=root(),
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                sbatch_path = Path(result.stdout.strip())
                sbatch = sbatch_path.read_text()
                self.assertEqual(
                    re.findall(r"^#SBATCH --nodes=(\d+)$", sbatch, re.MULTILINE),
                    ["1"],
                )
                self.assertEqual(
                    re.findall(
                        r"^#SBATCH --gpus-per-node=(\d+)$", sbatch, re.MULTILINE
                    ),
                    ["4"],
                )
                self.assertNotIn("#SBATCH --segment=", sbatch)
                self.assertIn("#SBATCH --account=nemotron_n3_post", sbatch)
                self.assertIn(
                    f"nemorl-q8-dapo32k-tq-recovery-{variant}-clean-20260823",
                    sbatch,
                )
                driver = (sbatch_path.parent / "driver.sh").read_text()
                for marker in (
                    "CUDAGRAPH_GATE_PASS",
                    "STEP1_GATE_PASS",
                    "STEP2_GATE_PASS",
                    "WAKE_REFIT_GATE_PASS",
                    "OUTPUT_LENGTH_GATE_PASS",
                    "NO_FATAL_GATE_PASS",
                    "--expected-samples-per-step 8",
                ):
                    self.assertIn(marker, driver)

                self.assertIn("Logged data to .*train_data_step1", driver)
                self.assertIn("Logged data to .*train_data_step2", driver)
                self.assertIn("assert_step2_refit_window", driver)
                self.assertNotIn("wait_for_gate 'Step[[:space:]]+1", driver)
                self.assertNotIn("wait_for_gate 'Step[[:space:]]+2", driver)
                self.assertNotIn(
                    "require_count 'GPU Memory after refit complete' 2", driver
                )

    def test_static_gate_rejects_forbidden_dspark_gamma(self) -> None:
        verifier = experiment() / "verify_pilot_config.py"
        config = self.config("dspark-k5")
        config["policy"]["draft"]["gamma"] = 5
        with tempfile.TemporaryDirectory() as tmp:
            invalid = Path(tmp) / "dspark-k5.yaml"
            invalid.write_text(json.dumps(config))
            source_root = config["defaults"].split("/examples/", 1)[0]
            result = subprocess.run(
                [
                    "python3",
                    str(verifier),
                    "--source-root",
                    source_root,
                    "--config",
                    str(invalid),
                    "--capture-sizes",
                    json.dumps(CAPTURE_SIZES),
                    "--static-only",
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_actual_submission_is_exactly_once_and_fail_closed(self) -> None:
        text = harness().read_text()
        self.assertEqual(
            text.count(
                'receipt="${DURABLE_ROOT}/preflight/'
                '${variant}-${HARNESS_SHA}.json"'
            ),
            2,
        )
        self.assertNotIn(
            'receipt="${DURABLE_ROOT}/preflight/${variant}.json"', text
        )
        self.assertIn('test ! -e "${record}"', text)
        self.assertIn('test ! -e "${record}.lock"', text)
        self.assertIn("TEST_ONLY_SCHEDULER_REJECTED", text)
        self.assertIn("ACTUAL_SCHEDULER_REJECTED", text)


if __name__ == "__main__":
    unittest.main()
