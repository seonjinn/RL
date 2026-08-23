"""Executable contracts for the replacement Qwen3-30B-A3B 20-step pair."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_30ba3b_dflash_dspark_20step_20260822"
SOURCE_ROOT = "/home/sna/nemorl-pr11-final-df9"
SOURCE_SHA = "df9daf62fe4625609b3a71abd7179007cd6970f9"
MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B"
DFLASH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dflash-b8-16n/exported-checkpoint-25391"
DSPARK = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dspark-b8-16n/exported-checkpoint-25391"
CAPTURE_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]
TRAINING_WORLD_SIZE = 16


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def harness() -> Path:
    return root() / "experiments" / EXPERIMENT / "submit_qwen3_30ba3b_20step.sh"


def diagnostic() -> Path:
    return root() / "experiments" / EXPERIMENT / "diagnose_container_python.sh"


def assert_placement_contract(sbatch: str) -> None:
    nodes = re.findall(r"^#SBATCH --nodes=(\d+)$", sbatch, flags=re.MULTILINE)
    segments = re.findall(r"^#SBATCH --segment=(\d+)$", sbatch, flags=re.MULTILINE)
    if nodes != ["4"]:
        raise AssertionError(f"expected exactly four requested nodes, got {nodes}")
    if segments != ["4"]:
        raise AssertionError(f"expected exactly one four-node segment, got {segments}")
    if int(nodes[0]) % int(segments[0]) != 0:
        raise AssertionError("requested nodes must be divisible by segment size")


def assert_cotrain_topology(policy: dict[str, object]) -> None:
    megatron = policy["megatron_cfg"]
    assert isinstance(megatron, dict)
    tp = megatron["tensor_model_parallel_size"]
    pp = megatron["pipeline_model_parallel_size"]
    ep = megatron["expert_model_parallel_size"]
    cp = megatron.get("context_parallel_size", 1)
    sp = megatron.get("sequence_parallel", False)
    if (tp, pp, ep, cp, sp) != (2, 1, 8, 1, True):
        raise AssertionError(f"invalid co-training topology: {(tp, pp, ep, cp, sp)}")
    if policy.get("sequence_packing") != {"enabled": True}:
        raise AssertionError("TP2 co-training must explicitly enable sequence packing")
    if policy["make_sequence_length_divisible_by"] != 2:
        raise AssertionError("TP2 co-training must make sequence length divisible by two")
    dense_dp = TRAINING_WORLD_SIZE // (tp * pp * cp)
    expert_dp = TRAINING_WORLD_SIZE // (tp * ep * pp)
    if TRAINING_WORLD_SIZE % (tp * ep * pp) != 0 or (dense_dp, expert_dp) != (8, 1):
        raise AssertionError(f"invalid 16-GPU expert grid: dense_dp={dense_dp}, expert_dp={expert_dp}")


class ContractTest(unittest.TestCase):
    maxDiff = None

    def manifest(self, variant: str) -> dict[str, object]:
        result = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", variant],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_configs_pin_the_matched_20_step_recipe(self) -> None:
        for variant, checkpoint, method in (
            ("dflash", DFLASH, "dflash"),
            ("dspark", DSPARK, "dspark"),
        ):
            with self.subTest(variant=variant):
                path = root() / "experiments" / EXPERIMENT / "configs" / f"{variant}.yaml"
                self.assertTrue(path.is_file(), f"missing committed {variant} config: {path}")
                config = json.loads(path.read_text())
                self.assertEqual(config["defaults"], f"{SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml")
                self.assertEqual(config["grpo"], {"max_num_steps": 20, "num_prompts_per_step": 16, "num_generations_per_prompt": 32, "val_period": 0, "seed": 42, "async_grpo": {"enabled": False}})
                self.assertFalse(config["data"]["shuffle"])
                self.assertEqual(config["data"]["train"], {"dataset_name": "OpenMathInstruct-2", "split_validation_size": 0, "seed": 42})
                self.assertFalse(config["checkpointing"]["enabled"])
                policy = config["policy"]
                self.assertEqual(policy["model_name"], MODEL)
                self.assertEqual(policy["tokenizer"]["name"], MODEL)
                self.assertEqual(policy["train_global_batch_size"], 512)
                self.assertEqual(policy["max_total_sequence_length"], 8192)
                assert_cotrain_topology(policy)
                generation = policy["generation"]
                self.assertEqual(generation["max_new_tokens"], 1024)
                self.assertEqual(generation["vllm_cfg"], {"tensor_parallel_size": 1, "max_model_len": 8192, "enforce_eager": False})
                self.assertEqual(generation["vllm_kwargs"]["speculative_config"], {"method": method, "model": checkpoint, "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1})
                self.assertEqual(policy["draft"]["model_name"], checkpoint)
                self.assertEqual(policy["draft"]["speculator_type"], method)
                self.assertEqual(policy["draft"]["anchors_per_sample"], 2)
                self.assertEqual(policy["draft"]["mask_token_id"], 151669)
                self.assertEqual(policy["draft"]["target_hidden_state_layer_ids"], [1, 12, 23, 34, 45])
                self.assertEqual(policy["draft"]["num_layers"], 5)
                if variant == "dflash":
                    self.assertEqual(policy["draft"]["gamma"], 5)
                else:
                    self.assertEqual(policy["draft"]["block_size"], 8)
                    self.assertEqual(policy["draft"]["markov_rank"], 256)
                    self.assertEqual(policy["draft"]["markov_head_type"], "vanilla")
                    self.assertTrue(policy["draft"]["confidence_enabled"])
                    self.assertTrue(policy["draft"]["confidence_with_markov"])

    def test_cotrain_topology_rejects_tp1_and_invalid_ep16_grid(self) -> None:
        valid: dict[str, object] = {
            "megatron_cfg": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
                "expert_model_parallel_size": 8,
                "context_parallel_size": 1,
                "sequence_parallel": True,
            },
            "sequence_packing": {"enabled": True},
            "make_sequence_length_divisible_by": 2,
        }
        for mutated in (
            {**valid, "megatron_cfg": {**valid["megatron_cfg"], "tensor_model_parallel_size": 1}},
            {**valid, "megatron_cfg": {**valid["megatron_cfg"], "expert_model_parallel_size": 16}},
        ):
            with self.subTest(mutated=mutated):
                with self.assertRaises(AssertionError):
                    assert_cotrain_topology(mutated)

    def test_harness_pins_clean_df9_and_never_reuses_wandb_ids(self) -> None:
        first = self.manifest("dflash")
        second = self.manifest("dflash")
        self.assertEqual(first["source"], {"root": SOURCE_ROOT, "sha": SOURCE_SHA})
        self.assertEqual(first["slurm"], {"account": "nemotron_n4_post", "partition": "batch", "qos": "normal", "time": "04:00:00", "nodes": 4, "gpus_per_node": 4})
        self.assertEqual(first["gates"], ["source-clean", "state-dict", "cudagraph", "step1", "step2"])
        self.assertEqual(first["wandb_reuse"], "never")
        self.assertNotEqual(first["wandb_run_id"], second["wandb_run_id"])
        self.assertTrue(first["wandb_run_id"].startswith("q30-20step-dflash-"))
        script = harness().read_text()
        self.assertIn("--untracked-files=all", script)
        self.assertIn("submodule status --recursive", script)
        self.assertIn(SOURCE_ROOT, script)
        self.assertNotIn("443e7243", script)

    def test_capture_buckets_cover_every_runtime_shape(self) -> None:
        result = subprocess.run(
            ["bash", str(harness()), "--assert-capture-coverage"],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coverage = json.loads(result.stdout)
        self.assertEqual(coverage["capture_sizes"], CAPTURE_SIZES)
        self.assertEqual(set(map(int, coverage["shape_to_bucket"])), set(range(1, 49)))
        self.assertTrue(all(bucket in CAPTURE_SIZES for bucket in coverage["shape_to_bucket"].values()))

    def test_checkpoint_contract_accepts_qwen_attention_norms(self) -> None:
        checker = (root() / "experiments" / EXPERIMENT / "check_checkpoint_state_dict.py").read_text()
        self.assertIn('f"layers.{layer}.self_attn.q_norm.weight"', checker)
        self.assertIn('f"layers.{layer}.self_attn.k_norm.weight"', checker)

    def test_python_diagnostic_uses_standard_ray_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                ["bash", str(diagnostic()), "--render"],
                cwd=root(),
                text=True,
                capture_output=True,
                env={**os.environ, "Q30_20STEP_DIAGNOSTIC_RENDER_ROOT": temporary},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = Path(result.stdout.strip())
            self.assertTrue(rendered.is_file())
            self.assertEqual(subprocess.run(["bash", "-n", str(rendered)], capture_output=True, text=True).returncode, 0)
            contents = rendered.read_text()
            self.assertIn('export MOUNTS="/lustre:/lustre,/home:/home"', contents)
            self.assertIn('NRL_FORCE_REBUILD_VENVS=true', contents)
            self.assertIn('exec bash "/home/sna/nemorl-pr11-final-df9/ray.sub"', contents)
            self.assertNotIn('UV_CACHE_DIR_OVERRIDE', contents)
            self.assertNotIn('UV_PROJECT_ENVIRONMENT', contents)
            self.assertNotIn('/raid/scratch', contents)
            self.assertNotIn('PYTHONPATH=', contents)
            self.assertIn('diagnose_container_python.py', contents)
            diagnostic_source = diagnostic().with_suffix(".py").read_text()
            self.assertIn('importlib.util.find_spec', diagnostic_source)
            self.assertIn('requests', diagnostic_source)
            self.assertIn('urllib3', diagnostic_source)
            self.assertIn('ray', diagnostic_source)

    def test_rendered_jobs_are_receipt_gated_and_account_correct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {**os.environ, "Q30_20STEP_RENDER_ROOT": temporary}
            rendered: list[tuple[str, str]] = []
            for variant in ("dflash", "dspark"):
                result = subprocess.run(
                    ["bash", str(harness()), "--render-sbatch", variant],
                    cwd=root(),
                    text=True,
                    capture_output=True,
                    env=environment,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                path = Path(result.stdout.strip())
                self.assertTrue(path.is_file())
                self.assertEqual(subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True).returncode, 0)
                driver = path.parent / "driver.sh"
                self.assertTrue(driver.is_file())
                self.assertEqual(subprocess.run(["bash", "-n", str(driver)], capture_output=True, text=True).returncode, 0)
                rendered.append((path.read_text(), driver.read_text()))
            for sbatch, driver in rendered:
                assert_placement_contract(sbatch)
                self.assertIn("#SBATCH --account=nemotron_n4_post", sbatch)
                self.assertIn("#SBATCH --partition=batch", sbatch)
                self.assertIn("#SBATCH --qos=normal", sbatch)
                self.assertIn("#SBATCH --time=04:00:00", sbatch)
                self.assertIn('export MOUNTS="/lustre:/lustre,/home:/home"', sbatch)
                self.assertIn("export NRL_FORCE_REBUILD_VENVS=true", sbatch)
                self.assertNotIn("UV_CACHE_DIR_OVERRIDE", sbatch)
                self.assertNotIn("UV_PROJECT_ENVIRONMENT", sbatch)
                self.assertNotIn("/raid/scratch", sbatch)
                self.assertNotIn("PYTHONPATH=", sbatch)
                self.assertIn("check_checkpoint_state_dict.py", driver)
                self.assertIn("verify_df9_configs.py", driver)
                self.assertIn("CUDAGRAPH_GATE_PASS", driver)
                self.assertIn("STEP1_GATE_PASS", driver)
                self.assertIn("STEP2_GATE_PASS", driver)
                self.assertIn("++policy.generation.vllm_kwargs.max_num_seqs=8", driver)
                self.assertIn("++policy.generation.vllm_kwargs.compilation_config.backend=eager", driver)
                self.assertIn("++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE", driver)
                self.assertIn("++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=[1,2,4,8,12,16,24,32,40,48]", driver)
                self.assertIn("NRL_FORCE_REBUILD_VENVS=true uv run", driver)
            self.assertIn("--test-only", harness().read_text())
            preflight = harness().read_text().split("write_sbatch()", maxsplit=1)[0]
            self.assertNotIn("verify_df9_configs.py", preflight)
            self.assertIn('sbatch --test-only "$(write_sbatch "${variant}" "${DURABLE_ROOT}")" 2>&1', harness().read_text())

    def test_placement_contract_rejects_missing_mismatched_and_non_divisible_segments(self) -> None:
        valid = "#SBATCH --nodes=4\n#SBATCH --segment=4\n"
        for mutated in (
            valid.replace("#SBATCH --segment=4\n", ""),
            valid.replace("#SBATCH --segment=4", "#SBATCH --segment=2"),
            valid.replace("#SBATCH --nodes=4", "#SBATCH --nodes=5").replace("#SBATCH --segment=4", "#SBATCH --segment=3"),
        ):
            with self.subTest(mutated=mutated):
                with self.assertRaises(AssertionError):
                    assert_placement_contract(mutated)


if __name__ == "__main__":
    unittest.main()
