"""Fail-closed contracts for the Qwen3-30B-A3B DAPO OSL32K pilot."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_30ba3b_dapo_osl32k_pilot_20260823"
SOURCE_SHA = "d0c4f1110cca28c75b7a1d98ed2d5f197e7d01dc"
HARNESS_BASE_SHA = "7bca9a95e7bafb85c42cd03912f85113dcf82945"
MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B"
DATASET = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_dapo_osl32k_pilot_20260823/data/dapo-math-17k-r658770-first64.jsonl"
DFLASH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/training/lyris-q30b-nemo-dflash-b8-16n-migrated-oci-s4400/exported-checkpoint-14500"
DSPARK = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/training/lyris-q30b-nemo-dspark-b8-16n-migrated-oci-s5700/exported-checkpoint-14500"
VARIANTS = {
    "baseline-k0": (None, None),
    "dflash-k7": ("dflash", DFLASH),
    "dspark-k7": ("dspark", DSPARK),
}
CAPTURE_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment() -> Path:
    return root() / "experiments" / EXPERIMENT


def harness() -> Path:
    return experiment() / "submit_qwen3_30ba3b_dapo_osl32k_pilot.sh"


class PilotContractTest(unittest.TestCase):
    maxDiff = None

    def config(self, variant: str) -> dict[str, object]:
        return json.loads((experiment() / "configs" / f"{variant}.yaml").read_text())

    def manifest(self, variant: str) -> dict[str, object]:
        result = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", variant],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_three_configs_pin_the_same_two_step_dapo_contract(self) -> None:
        for variant, (method, checkpoint) in VARIANTS.items():
            with self.subTest(variant=variant):
                config = self.config(variant)
                self.assertEqual(config["grpo"], {
                    "max_num_steps": 2,
                    "num_prompts_per_step": 16,
                    "num_generations_per_prompt": 4,
                    "val_period": 0,
                    "seed": 42,
                    "async_grpo": {"enabled": False},
                })
                self.assertEqual(config["data"], {
                    "max_input_seq_length": 2048,
                    "shuffle": False,
                    "num_workers": 1,
                    "train": {
                        "dataset_name": "ResponseDataset",
                        "data_path": DATASET,
                        "split_validation_size": 0,
                        "seed": 42,
                    },
                    "validation": None,
                    "default": {
                        "prompt_file": None,
                        "system_prompt_file": None,
                        "processor": "math_hf_data_processor",
                        "env_name": "math",
                    },
                })
                self.assertFalse(config["checkpointing"]["enabled"])
                policy = config["policy"]
                self.assertEqual(policy["model_name"], MODEL)
                self.assertEqual(policy["tokenizer"], {"name": MODEL})
                self.assertEqual(policy["train_global_batch_size"], 64)
                self.assertEqual(policy["train_micro_batch_size"], 1)
                self.assertEqual(policy["logprob_batch_size"], 1)
                self.assertEqual(policy["logprob_chunk_size"], 2048)
                self.assertEqual(policy["max_total_sequence_length"], 40960)
                self.assertEqual(policy["make_sequence_length_divisible_by"], 8)
                self.assertEqual(policy["sequence_packing"], {"enabled": True})
                megatron = policy["megatron_cfg"]
                self.assertEqual(
                    {key: megatron[key] for key in (
                        "tensor_model_parallel_size",
                        "pipeline_model_parallel_size",
                        "expert_model_parallel_size",
                        "context_parallel_size",
                        "sequence_parallel",
                        "activation_checkpointing",
                        "defer_fp32_logits",
                    )},
                    {
                        "tensor_model_parallel_size": 2,
                        "pipeline_model_parallel_size": 1,
                        "expert_model_parallel_size": 8,
                        "context_parallel_size": 2,
                        "sequence_parallel": True,
                        "activation_checkpointing": True,
                        "defer_fp32_logits": True,
                    },
                )
                generation = policy["generation"]
                self.assertEqual(generation["max_new_tokens"], 32768)
                self.assertEqual(generation["vllm_cfg"], {
                    "tensor_parallel_size": 1,
                    "max_model_len": 40960,
                    "gpu_memory_utilization": 0.5,
                    "enforce_eager": False,
                })
                if method is None:
                    self.assertNotIn("draft", policy)
                    self.assertNotIn("speculative_config", generation["vllm_kwargs"])
                else:
                    self.assertEqual(generation["vllm_kwargs"]["speculative_config"], {
                        "method": method,
                        "model": checkpoint,
                        "num_speculative_tokens": 7,
                        "draft_tensor_parallel_size": 1,
                    })
                    self.assertEqual(policy["draft"]["model_name"], checkpoint)
                    self.assertEqual(policy["draft"]["gamma"], 7)

    def test_manifest_pins_identity_and_required_runtime_gates(self) -> None:
        for variant, (method, checkpoint) in VARIANTS.items():
            with self.subTest(variant=variant):
                manifest = self.manifest(variant)
                self.assertEqual(manifest["variant"], variant)
                self.assertEqual(manifest["harness_base_sha"], HARNESS_BASE_SHA)
                self.assertEqual(manifest["source"]["sha"], SOURCE_SHA)
                self.assertEqual(manifest["dataset"]["path"], DATASET)
                self.assertEqual(manifest["dataset"]["rows"], 64)
                self.assertEqual(manifest["dataset"]["source_order"], list(range(64)))
                self.assertEqual(manifest["dataset"]["seed"], 42)
                self.assertEqual(manifest["checkpoint"], checkpoint)
                self.assertEqual(manifest["method"], method)
                self.assertEqual(manifest["num_speculative_tokens"], 0 if method is None else 7)
                self.assertEqual(manifest["capture_sizes"], CAPTURE_SIZES)
                self.assertEqual(manifest["max_steps"], 2)
                self.assertEqual(manifest["wandb_project"], "sna-specdec")
                self.assertEqual(manifest["wandb_reuse"], "never")
                self.assertRegex(
                    manifest["wandb_run_id"],
                    rf"^q30ba3b-dapo-osl32k-pilot-{re.escape(variant)}-[0-9a-f]{{32}}$",
                )
                gates = set(manifest["gates"])
                self.assertTrue({
                    "source-clean", "data-identity", "config-compose", "cudagraph",
                    "step1", "step2", "wake-refit", "output-length", "no-fatal",
                } <= gates)
                if method is not None:
                    self.assertIn("state-dict", gates)

    def test_capture_contract_covers_every_k7_decode_shape_through_64(self) -> None:
        result = subprocess.run(
            ["bash", str(harness()), "--assert-capture-coverage"],
            cwd=root(), text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coverage = json.loads(result.stdout)
        self.assertEqual(coverage["capture_sizes"], CAPTURE_SIZES)
        self.assertEqual(set(map(int, coverage["shape_to_bucket"])), set(range(1, 65)))
        self.assertEqual(max(coverage["shape_to_bucket"].values()), 64)

    def test_sbatch_is_exactly_four_nodes_and_uses_task_owned_clean_source(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                result = subprocess.run(
                    ["bash", str(harness()), "--render-sbatch", variant],
                    cwd=root(), text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                sbatch = Path(result.stdout.strip()).read_text()
                self.assertEqual(re.findall(r"^#SBATCH --nodes=(\d+)$", sbatch, re.MULTILINE), ["4"])
                self.assertEqual(re.findall(r"^#SBATCH --segment=(\d+)$", sbatch, re.MULTILINE), ["4"])
                self.assertEqual(re.findall(r"^#SBATCH --gpus-per-node=(\d+)$", sbatch, re.MULTILINE), ["4"])
                self.assertIn("#SBATCH --account=nemotron_n3_post", sbatch)
                self.assertIn(f"nemorl-pr11-q30-dapo32k-{variant}-clean-20260823", sbatch)

    def test_actual_submission_record_prevents_duplicate_arm_submission(self) -> None:
        text = harness().read_text()
        self.assertIn('test ! -e "${record}"', text)
        self.assertIn('test ! -e "${record}.lock"', text)
        self.assertIn("actual ${variant} submission already exists", text)

    def test_preflight_composes_config_before_any_scheduler_call(self) -> None:
        text = harness().read_text()
        preflight = text[text.index("preflight() {") : text.index("assert_capture_coverage() {")]
        self.assertIn('verify_pilot_config.py', preflight)
        self.assertIn('--static-only', preflight)
        self.assertIn('--source-root "${source_root}"', preflight)
        self.assertIn('--config "${SCRIPT_DIR}/configs/${variant}.yaml"', preflight)
        self.assertNotIn("sbatch", preflight)

    def test_static_config_gate_does_not_require_product_python_dependencies(self) -> None:
        verifier = experiment() / "verify_pilot_config.py"
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                config = experiment() / "configs" / f"{variant}.yaml"
                source_root = json.loads(config.read_text())["defaults"].split("/examples/", 1)[0]
                result = subprocess.run(
                    [
                        "python3", str(verifier), "--source-root", source_root,
                        "--config", str(config), "--capture-sizes", json.dumps(CAPTURE_SIZES),
                        "--static-only",
                    ],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(json.loads(result.stdout)["STATIC_CONFIG_GATE_PASS"])

    def test_driver_is_fail_closed_for_both_steps_refit_lengths_and_fatals(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                result = subprocess.run(
                    ["bash", str(harness()), "--render-sbatch", variant],
                    cwd=root(), text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                artifact = Path(result.stdout.strip()).parent
                driver = (artifact / "driver.sh").read_text()
                for required in (
                    "DATA_IDENTITY_GATE_PASS", "CONFIG_COMPOSE_GATE_PASS",
                    "CUDAGRAPH_GATE_PASS", "STEP1_GATE_PASS", "STEP2_GATE_PASS",
                    "WAKE_REFIT_GATE_PASS", "OUTPUT_LENGTH_GATE_PASS", "NO_FATAL_GATE_PASS",
                    "--max-output-length 32768", "--expected-steps 1 2",
                    "compilation_config.cudagraph_mode=PIECEWISE",
                    "compilation_config.cudagraph_capture_sizes=[1,2,4,8,12,16,24,32,40,48,56,64]",
                ):
                    self.assertIn(required, driver)
                self.assertRegex(driver, r"(?i)out of memory|nan|traceback")
                if variant != "baseline-k0":
                    self.assertIn("--verify-content-sha", driver)

    def test_dataset_identity_gate_accepts_only_exact_first_64_source_rows(self) -> None:
        gate = experiment() / "verify_dapo_slice.py"
        identity = experiment() / "dataset_identity.json"
        manifest = json.loads(identity.read_text())
        self.assertEqual(manifest["source"]["revision"], "65877096c24ffa7abc4e4fa5edb95cf3413a5674")
        self.assertEqual(manifest["source"]["sha256"], "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941")
        self.assertEqual(manifest["source"]["size"], 299363855)
        self.assertEqual(manifest["slice"]["rows"], 64)
        self.assertEqual(manifest["slice"]["source_order"], list(range(64)))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            output = Path(tmp) / "slice.jsonl"
            rows = [
                {"prompt": [{"role": "user", "content": f"q{i}"}], "reward_model": {"ground_truth": str(i)}}
                for i in range(65)
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            dynamic = json.loads(identity.read_text())
            dynamic["source"] = {
                **dynamic["source"],
                "format": "jsonl",
                "size": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            expected = "".join(
                json.dumps({"input": f"q{i}", "output": str(i)}, sort_keys=True, separators=(",", ":")) + "\n"
                for i in range(64)
            ).encode()
            dynamic["slice"]["size"] = len(expected)
            dynamic["slice"]["sha256"] = hashlib.sha256(expected).hexdigest()
            identity_path = Path(tmp) / "identity.json"
            identity_path.write_text(json.dumps(dynamic))
            result = subprocess.run(
                ["python3", str(gate), "--source", str(source), "--output", str(output), "--identity-file", str(identity_path)],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes(), expected)
            output.write_text(output.read_text().replace("q7", "tampered", 1))
            result = subprocess.run(
                ["python3", str(gate), "--source", str(source), "--output", str(output), "--identity-file", str(identity_path), "--verify-only"],
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_output_length_gate_reports_quantiles_and_cap_hits_for_both_steps(self) -> None:
        summarizer = experiment() / "summarize_output_lengths.py"
        with tempfile.TemporaryDirectory() as tmp:
            log_root = Path(tmp)
            for step, lengths in ((1, [1, 2, 3, 32768]), (2, [4, 5, 6, 7])):
                path = log_root / f"train_data_step{step}.jsonl"
                with path.open("w") as stream:
                    for length in lengths:
                        stream.write(json.dumps({"token_loss_mask": [[True] * length]}) + "\n")
            output = log_root / "output-length-metrics.json"
            result = subprocess.run(
                ["python3", str(summarizer), "--log-root", str(log_root), "--output", str(output), "--max-output-length", "32768", "--expected-steps", "1", "2"],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads(output.read_text())
            self.assertEqual(metrics["steps"], [1, 2])
            self.assertEqual(metrics["sample_count"], 8)
            self.assertEqual(metrics["cap_hit_count"], 1)
            self.assertEqual(metrics["max"], 32768)
            self.assertEqual(set(metrics["quantiles"]), {"p50", "p90", "p95", "p99"})


if __name__ == "__main__":
    unittest.main()
