"""Executable contracts for the replacement Qwen3-30B-A3B 20-step pair."""

from __future__ import annotations

import json
import hashlib
import os
import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_30ba3b_dflash_dspark_20step_20260822"
SOURCE_ROOT = "/home/sna/nemorl-pr11-q30-eagle3-k3-product-clean-20260823"
SOURCE_SHA = "d0c4f1110cca28c75b7a1d98ed2d5f197e7d01dc"
MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B"
DFLASH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dflash-b8-16n/exported-checkpoint-25391"
DSPARK = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dspark-b8-16n/exported-checkpoint-25391"
CAPTURE_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]
TRAINING_WORLD_SIZE = 16

NEW_DFLASH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/training/lyris-q30b-nemo-dflash-b8-16n-migrated-oci-s4400/exported-checkpoint-14500"
NEW_DSPARK = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/training/lyris-q30b-nemo-dspark-b8-16n-migrated-oci-s5700/exported-checkpoint-14500"
BASE_S25391_DFLASH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dflash/exported-checkpoint-25391"
BASE_S25391_DSPARK = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dspark/exported-checkpoint-25391"
EAGLE3 = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf"
NEW_VARIANTS = {
    "dflash-k5": (NEW_DFLASH, "dflash", 5),
    "dflash-k7": (NEW_DFLASH, "dflash", 7),
    "dspark-k5": (NEW_DSPARK, "dspark", 5),
    "dspark-k7": (NEW_DSPARK, "dspark", 7),
}
K3_VARIANTS = {
    "eagle3-k3": (EAGLE3, "eagle3", "static"),
    "dflash-k3": (BASE_S25391_DFLASH, "dflash", "always-online"),
    "dspark-k3": (BASE_S25391_DSPARK, "dspark", "always-online"),
}
CAPTURE_SIZES_K3 = [1, 2, 4, 8, 12, 16, 24, 32]
CAPTURE_SIZES_K7 = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def harness() -> Path:
    return root() / "experiments" / EXPERIMENT / "submit_qwen3_30ba3b_20step.sh"


def diagnostic() -> Path:
    return root() / "experiments" / EXPERIMENT / "diagnose_container_python.sh"


def rclone_dispatch() -> Path:
    return root() / "experiments" / EXPERIMENT / "rclone_arch_dispatch.sh"


def portable_time() -> Path:
    return root() / "experiments" / EXPERIMENT / "portable_time.sh"


def write_fake_checkpoint(path: Path, variant: str, architecture: str) -> None:
    keys = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    for layer in range(5):
        keys.update(
            {
                f"layers.{layer}.input_layernorm.weight",
                f"layers.{layer}.post_attention_layernorm.weight",
                f"layers.{layer}.self_attn.q_proj.weight",
                f"layers.{layer}.self_attn.k_proj.weight",
                f"layers.{layer}.self_attn.v_proj.weight",
                f"layers.{layer}.self_attn.o_proj.weight",
                f"layers.{layer}.self_attn.q_norm.weight",
                f"layers.{layer}.self_attn.k_norm.weight",
                f"layers.{layer}.mlp.gate_proj.weight",
                f"layers.{layer}.mlp.up_proj.weight",
                f"layers.{layer}.mlp.down_proj.weight",
            }
        )
    if variant == "dspark":
        keys.update(
            {
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
                "confidence_head.proj.weight",
                "confidence_head.proj.bias",
            }
        )
    path.mkdir()
    header = json.dumps({key: {} for key in keys}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    dflash = {
        "mask_token_id": 151669,
        "target_layer_ids": [1, 12, 23, 34, 45],
    }
    if variant == "dspark":
        dflash.update(
            {
                "markov_head_type": "vanilla",
                "markov_rank": 256,
                "projector_type": "dspark",
                "shift_label": True,
                "use_confidence_head": True,
            }
        )
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "block_size": 8,
                "dflash_config": dflash,
                "hidden_size": 2048,
                "num_attention_heads": 32,
                "head_dim": 128,
                "num_hidden_layers": 5,
            }
        )
    )


def write_identity(path: Path, variant: str, checkpoint: Path) -> None:
    files: dict[str, dict[str, int | str]] = {}
    for name in ("config.json", "model.safetensors"):
        contents = (checkpoint / name).read_bytes()
        files[name] = {
            "sha256": hashlib.sha256(contents).hexdigest(),
            "size": len(contents),
        }
    path.write_text(json.dumps({variant: files}))


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
        raise AssertionError(
            "TP2 co-training must make sequence length divisible by two"
        )
    dense_dp = TRAINING_WORLD_SIZE // (tp * pp * cp)
    expert_dp = TRAINING_WORLD_SIZE // (tp * ep * pp)
    if TRAINING_WORLD_SIZE % (tp * ep * pp) != 0 or (dense_dp, expert_dp) != (8, 1):
        raise AssertionError(
            f"invalid 16-GPU expert grid: dense_dp={dense_dp}, expert_dp={expert_dp}"
        )


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
                path = (
                    root() / "experiments" / EXPERIMENT / "configs" / f"{variant}.yaml"
                )
                self.assertTrue(
                    path.is_file(), f"missing committed {variant} config: {path}"
                )
                config = json.loads(path.read_text())
                self.assertEqual(
                    config["defaults"],
                    f"{SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
                )
                self.assertEqual(
                    config["grpo"],
                    {
                        "max_num_steps": 20,
                        "num_prompts_per_step": 16,
                        "num_generations_per_prompt": 32,
                        "val_period": 0,
                        "seed": 42,
                        "async_grpo": {"enabled": False},
                    },
                )
                self.assertFalse(config["data"]["shuffle"])
                self.assertEqual(
                    config["data"]["train"],
                    {
                        "dataset_name": "OpenMathInstruct-2",
                        "split_validation_size": 0,
                        "seed": 42,
                    },
                )
                self.assertFalse(config["checkpointing"]["enabled"])
                policy = config["policy"]
                self.assertEqual(policy["model_name"], MODEL)
                self.assertEqual(policy["tokenizer"]["name"], MODEL)
                self.assertEqual(policy["train_global_batch_size"], 512)
                self.assertEqual(policy["max_total_sequence_length"], 8192)
                assert_cotrain_topology(policy)
                generation = policy["generation"]
                self.assertEqual(generation["max_new_tokens"], 1024)
                self.assertEqual(
                    generation["vllm_cfg"],
                    {
                        "tensor_parallel_size": 1,
                        "max_model_len": 8192,
                        "enforce_eager": False,
                    },
                )
                self.assertEqual(
                    generation["vllm_kwargs"]["speculative_config"],
                    {
                        "method": method,
                        "model": checkpoint,
                        "num_speculative_tokens": 5,
                        "draft_tensor_parallel_size": 1,
                    },
                )
                self.assertEqual(policy["draft"]["model_name"], checkpoint)
                self.assertEqual(policy["draft"]["speculator_type"], method)
                self.assertEqual(policy["draft"]["anchors_per_sample"], 2)
                self.assertEqual(policy["draft"]["mask_token_id"], 151669)
                self.assertEqual(
                    policy["draft"]["target_hidden_state_layer_ids"],
                    [1, 12, 23, 34, 45],
                )
                self.assertEqual(policy["draft"]["num_layers"], 5)
                if variant == "dflash":
                    self.assertEqual(policy["draft"]["gamma"], 5)
                else:
                    self.assertEqual(policy["draft"]["block_size"], 8)
                    self.assertEqual(policy["draft"]["markov_rank"], 256)
                    self.assertEqual(policy["draft"]["markov_head_type"], "vanilla")
                    self.assertTrue(policy["draft"]["confidence_enabled"])
                    self.assertTrue(policy["draft"]["confidence_with_markov"])

    def test_new_drafters_define_exact_k5_k7_matrix(self) -> None:
        config_dir = root() / "experiments" / EXPERIMENT / "configs"
        for variant, (checkpoint, method, k) in NEW_VARIANTS.items():
            with self.subTest(variant=variant):
                config = json.loads((config_dir / f"{variant}.yaml").read_text())
                policy = config["policy"]
                speculative = policy["generation"]["vllm_kwargs"]["speculative_config"]
                self.assertEqual(speculative["method"], method)
                self.assertEqual(speculative["model"], checkpoint)
                self.assertEqual(speculative["num_speculative_tokens"], k)
                self.assertEqual(policy["draft"]["model_name"], checkpoint)
                if method == "dflash":
                    self.assertEqual(policy["draft"]["gamma"], k)
                else:
                    self.assertEqual(policy["draft"]["block_size"], 8)

    def test_k3_matrix_pins_method_checkpoint_and_training_mode(self) -> None:
        config_dir = root() / "experiments" / EXPERIMENT / "configs"
        baseline = json.loads((config_dir / "baseline.yaml").read_text())
        for variant, (checkpoint, method, training_mode) in K3_VARIANTS.items():
            with self.subTest(variant=variant):
                config = json.loads((config_dir / f"{variant}.yaml").read_text())
                speculative = config["policy"]["generation"]["vllm_kwargs"][
                    "speculative_config"
                ]
                self.assertEqual(speculative["method"], method)
                self.assertEqual(speculative["model"], checkpoint)
                self.assertEqual(speculative["num_speculative_tokens"], 3)
                matched = json.loads(json.dumps(config))
                matched["policy"]["generation"]["vllm_kwargs"].pop(
                    "speculative_config"
                )
                if training_mode == "always-online":
                    draft = matched["policy"].pop("draft")
                    self.assertTrue(draft["enabled"])
                    self.assertEqual(draft["model_name"], checkpoint)
                else:
                    self.assertNotIn("draft", matched["policy"])
                self.assertEqual(matched, baseline)

    def test_k3_manifests_make_training_mode_and_target_compatibility_explicit(
        self,
    ) -> None:
        baseline = self.manifest("baseline")
        self.assertEqual(baseline["draft_training_mode"], "none")
        for variant, (checkpoint, method, training_mode) in K3_VARIANTS.items():
            with self.subTest(variant=variant):
                manifest = self.manifest(variant)
                self.assertEqual(manifest["checkpoint"], checkpoint)
                self.assertEqual(manifest["method"], method)
                self.assertEqual(manifest["num_speculative_tokens"], 3)
                self.assertEqual(manifest["draft_training_mode"], training_mode)
                self.assertEqual(
                    manifest["target_model"], "Qwen/Qwen3-30B-A3B"
                )
                self.assertEqual(manifest["wandb_project"], "sna-specdec")
                if method in ("dflash", "dspark"):
                    self.assertTrue(
                        manifest["wandb_run_id"].startswith(
                            f"q30ba3b-20step-{variant}-base-s25391-"
                        )
                    )

    def test_k3_base_identity_pins_full_schedule_weight_content(self) -> None:
        identity_path = (
            root()
            / "experiments"
            / EXPERIMENT
            / "checkpoint_identity_base_s25391.json"
        )
        identity = json.loads(identity_path.read_text())
        self.assertEqual(
            identity,
            {
                "dflash": {
                    "config.json": {
                        "sha256": "d502e18b23ea01dd7f0763840cd91a4545518cf594ba925d45de94eb1a40d35a",
                        "size": 849,
                    },
                    "model.safetensors": {
                        "sha256": "a6015bdd6cdeb62bde3025782325e496354dc4feef8b546b595a02f2ac94a182",
                        "size": 608231904,
                    },
                },
                "dspark": {
                    "config.json": {
                        "sha256": "ddfb31cfc6f5f2d6039cc70661213776c79105ab2bf95255cf2e56f552913e37",
                        "size": 1053,
                    },
                    "model.safetensors": {
                        "sha256": "916b21d8b02e20cd62f2eb3fec0d2e66f219f60ae7614b93111cc229f9ecb117",
                        "size": 763819354,
                    },
                },
            },
        )

    def test_manifest_records_an_isolated_durable_root_override(self) -> None:
        isolated = "/lustre/test/q30-k3-four-arm"
        result = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", "baseline"],
            cwd=root(),
            text=True,
            capture_output=True,
            env={**os.environ, "Q30_20STEP_DURABLE_ROOT": isolated},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["durable_root"], isolated)

    def test_account_override_is_recorded_in_manifest_and_sbatch(self) -> None:
        account = "nemotron_sw_post"
        environment = {**os.environ, "Q30_20STEP_ACCOUNT": account}
        manifest_result = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", "baseline"],
            cwd=root(),
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(manifest_result.returncode, 0, manifest_result.stderr)
        self.assertEqual(json.loads(manifest_result.stdout)["slurm"]["account"], account)
        with tempfile.TemporaryDirectory() as temporary:
            render_result = subprocess.run(
                ["bash", str(harness()), "--render-sbatch", "baseline"],
                cwd=root(),
                text=True,
                capture_output=True,
                env={
                    **environment,
                    "Q30_20STEP_RENDER_ROOT": temporary,
                },
            )
            self.assertEqual(render_result.returncode, 0, render_result.stderr)
            sbatch = Path(render_result.stdout.strip()).read_text()
            self.assertIn(f"#SBATCH --account={account}", sbatch)
            self.assertIn(f"#SBATCH --job-name={account}.q30-20-baseline", sbatch)

    def test_k3_capture_coverage_covers_every_runtime_shape(self) -> None:
        for variant in K3_VARIANTS:
            with self.subTest(variant=variant):
                result = subprocess.run(
                    ["bash", str(harness()), "--assert-capture-coverage", variant],
                    cwd=root(),
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                coverage = json.loads(result.stdout)
                self.assertEqual(coverage["capture_sizes"], CAPTURE_SIZES_K3)
                self.assertEqual(
                    set(map(int, coverage["shape_to_bucket"])), set(range(1, 33))
                )

    def test_eagle3_gate_rejects_a_non_base_verifier(self) -> None:
        checker = (
            root() / "experiments" / EXPERIMENT / "check_eagle3_checkpoint.py"
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            (checkpoint / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Eagle3DraftModel"],
                        "speculators_config": {
                            "algorithm": "eagle3",
                            "proposal_methods": [
                                {"proposal_type": "greedy", "speculative_tokens": 3}
                            ],
                            "verifier": {"name_or_path": "Qwen/Qwen3-30B-A3B-Thinking-2507"},
                        },
                    }
                )
            )
            result = subprocess.run(
                [
                    "python3",
                    str(checker),
                    "--checkpoint",
                    str(checkpoint),
                    "--target-model",
                    "Qwen/Qwen3-30B-A3B",
                    "--num-speculative-tokens",
                    "3",
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("verifier target mismatch", result.stderr)

    def test_new_matrix_capture_coverage_is_k_specific(self) -> None:
        for variant, (_, _, k) in NEW_VARIANTS.items():
            with self.subTest(variant=variant):
                result = subprocess.run(
                    ["bash", str(harness()), "--assert-capture-coverage", variant],
                    cwd=root(),
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                coverage = json.loads(result.stdout)
                expected = CAPTURE_SIZES if k == 5 else CAPTURE_SIZES_K7
                self.assertEqual(coverage["capture_sizes"], expected)
                self.assertEqual(
                    set(map(int, coverage["shape_to_bucket"])),
                    set(range(1, 8 * (k + 1) + 1)),
                )

    def test_rendered_compose_gate_uses_the_same_k_specific_capture_sizes(self) -> None:
        for variant, (_, _, k) in NEW_VARIANTS.items():
            with (
                self.subTest(variant=variant),
                tempfile.TemporaryDirectory() as temporary,
            ):
                result = subprocess.run(
                    ["bash", str(harness()), "--render-sbatch", variant],
                    cwd=root(),
                    text=True,
                    capture_output=True,
                    env={**os.environ, "Q30_20STEP_RENDER_ROOT": temporary},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                driver = Path(result.stdout.strip()).parent / "driver.sh"
                expected = CAPTURE_SIZES if k == 5 else CAPTURE_SIZES_K7
                compact = json.dumps(expected, separators=(",", ":"))
                self.assertIn(
                    f"--capture-sizes '{compact}'",
                    driver.read_text(),
                )

    def test_new_matrix_manifests_pin_checkpoint_identity(self) -> None:
        for variant, (checkpoint, method, k) in NEW_VARIANTS.items():
            with self.subTest(variant=variant):
                manifest = self.manifest(variant)
                self.assertEqual(manifest["checkpoint"], checkpoint)
                self.assertEqual(manifest["method"], method)
                self.assertEqual(manifest["num_speculative_tokens"], k)
                self.assertEqual(manifest["wandb_project"], "sna-specdec")
                self.assertTrue(
                    manifest["wandb_run_id"].startswith(
                        f"q30ba3b-20step-{variant}-lyris14500-"
                    )
                )

    def test_rclone_dispatch_selects_arch_and_forwards_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "fake-rclone"
            fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            fake.chmod(0o700)
            for architecture in ("x86_64", "aarch64"):
                with self.subTest(architecture=architecture):
                    result = subprocess.run(
                        [
                            str(rclone_dispatch()),
                            "listremotes",
                            "--config",
                            "/tmp/config",
                        ],
                        text=True,
                        capture_output=True,
                        env={
                            **os.environ,
                            "RCLONE_ARCH_OVERRIDE": architecture,
                            "RCLONE_AMD64_BIN": str(fake),
                            "RCLONE_ARM64_BIN": str(fake),
                        },
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        result.stdout.splitlines(),
                        ["listremotes", "--config", "/tmp/config"],
                    )

    def test_portable_time_forwards_command_and_exit_code(self) -> None:
        success = subprocess.run(
            [str(portable_time()), "printf", "%s", "forwarded"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(success.stdout, "forwarded")
        failure = subprocess.run(
            [str(portable_time()), "bash", "-c", "exit 17"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(failure.returncode, 17)

    def test_baseline_is_matched_except_for_draft_and_speculation(self) -> None:
        config_dir = root() / "experiments" / EXPERIMENT / "configs"
        baseline = json.loads((config_dir / "baseline.yaml").read_text())
        self.assertNotIn("draft", baseline["policy"])
        self.assertNotIn(
            "speculative_config", baseline["policy"]["generation"]["vllm_kwargs"]
        )
        for variant in ("dflash", "dspark"):
            with self.subTest(variant=variant):
                speculative = json.loads((config_dir / f"{variant}.yaml").read_text())
                expected = json.loads(json.dumps(speculative))
                expected["policy"].pop("draft")
                expected["policy"]["generation"]["vllm_kwargs"].pop(
                    "speculative_config"
                )
                self.assertEqual(baseline, expected)

    def test_baseline_manifest_and_render_skip_draft_checkpoint_gate(self) -> None:
        manifest = self.manifest("baseline")
        self.assertEqual(
            manifest["gates"], ["source-clean", "cudagraph", "step1", "step2"]
        )
        self.assertEqual(manifest["wandb_reuse"], "never")
        self.assertTrue(
            manifest["wandb_run_id"].startswith("q30ba3b-20step-baseline-k0-")
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                ["bash", str(harness()), "--render-sbatch", "baseline"],
                cwd=root(),
                text=True,
                capture_output=True,
                env={**os.environ, "Q30_20STEP_RENDER_ROOT": temporary},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            driver = Path(result.stdout.strip()).parent / "driver.sh"
            driver_text = driver.read_text()
            self.assertNotIn("check_checkpoint_state_dict.py", driver_text)
            self.assertNotIn("CHECKPOINT", driver_text)

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
            {
                **valid,
                "megatron_cfg": {
                    **valid["megatron_cfg"],
                    "tensor_model_parallel_size": 1,
                },
            },
            {
                **valid,
                "megatron_cfg": {
                    **valid["megatron_cfg"],
                    "expert_model_parallel_size": 16,
                },
            },
        ):
            with self.subTest(mutated=mutated):
                with self.assertRaises(AssertionError):
                    assert_cotrain_topology(mutated)

    def test_harness_pins_clean_product_head_and_never_reuses_wandb_ids(self) -> None:
        first = self.manifest("dflash")
        second = self.manifest("dflash")
        self.assertEqual(first["source"], {"root": SOURCE_ROOT, "sha": SOURCE_SHA})
        self.assertEqual(
            first["slurm"],
            {
                "account": "nemotron_n3_post",
                "partition": "batch",
                "qos": "normal",
                "time": "04:00:00",
                "nodes": 4,
                "gpus_per_node": 4,
            },
        )
        self.assertEqual(
            first["gates"],
            ["source-clean", "state-dict", "cudagraph", "step1", "step2"],
        )
        self.assertEqual(first["wandb_reuse"], "never")
        self.assertNotEqual(first["wandb_run_id"], second["wandb_run_id"])
        self.assertTrue(first["wandb_run_id"].startswith("q30ba3b-20step-dflash-k5-"))
        script = harness().read_text()
        self.assertIn("--untracked-files=all", script)
        self.assertIn("submodule status --recursive", script)
        self.assertIn(SOURCE_ROOT, script)
        self.assertIn(SOURCE_SHA, script)
        self.assertNotIn("df9daf62", script)
        self.assertNotIn("443e7243", script)
        self.assertIn('test -e "${SOURCE_ROOT}/.git"', script)
        self.assertNotIn('test -d "${SOURCE_ROOT}/.git"', script)
        self.assertIn(
            'record="${DURABLE_ROOT}/submissions/${variant}-${SOURCE_SHA}.json"', script
        )

    def test_wandb_project_and_names_are_method_specific(self) -> None:
        expected_prefixes = {
            "baseline": "q30ba3b-20step-baseline-k0-",
            "dflash": "q30ba3b-20step-dflash-k5-",
            "dspark": "q30ba3b-20step-dspark-k5-b8-",
        }
        for variant, prefix in expected_prefixes.items():
            with (
                self.subTest(variant=variant),
                tempfile.TemporaryDirectory() as temporary,
            ):
                manifest = self.manifest(variant)
                self.assertTrue(manifest["wandb_run_id"].startswith(prefix))
                self.assertEqual(manifest["wandb_project"], "sna-specdec")
                result = subprocess.run(
                    ["bash", str(harness()), "--render-sbatch", variant],
                    cwd=root(),
                    text=True,
                    capture_output=True,
                    env={**os.environ, "Q30_20STEP_RENDER_ROOT": temporary},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                driver = Path(result.stdout.strip()).parent / "driver.sh"
                self.assertIn("logger.wandb.project=sna-specdec", driver.read_text())

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
        self.assertTrue(
            all(
                bucket in CAPTURE_SIZES
                for bucket in coverage["shape_to_bucket"].values()
            )
        )

    def test_checkpoint_contract_accepts_qwen_attention_norms(self) -> None:
        checker = (
            root() / "experiments" / EXPERIMENT / "check_checkpoint_state_dict.py"
        ).read_text()
        self.assertIn('f"layers.{layer}.self_attn.q_norm.weight"', checker)
        self.assertIn('f"layers.{layer}.self_attn.k_norm.weight"', checker)

    def test_checkpoint_gate_rejects_wrong_export_architecture(self) -> None:
        checker = root() / "experiments" / EXPERIMENT / "check_checkpoint_state_dict.py"
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            identity = Path(temporary) / "identity.json"
            write_fake_checkpoint(checkpoint, "dflash", "WrongDraftModel")
            write_identity(identity, "dflash", checkpoint)
            result = subprocess.run(
                [
                    "python3",
                    str(checker),
                    "--variant",
                    "dflash",
                    "--checkpoint",
                    str(checkpoint),
                    "--identity-file",
                    str(identity),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checkpoint config mismatch", result.stderr)

    def test_checkpoint_gate_verifies_content_addressed_identity(self) -> None:
        checker = root() / "experiments" / EXPERIMENT / "check_checkpoint_state_dict.py"
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            identity = Path(temporary) / "identity.json"
            write_fake_checkpoint(checkpoint, "dflash", "DFlashDraftModel")
            write_identity(identity, "dflash", checkpoint)
            command = [
                "python3",
                str(checker),
                "--variant",
                "dflash",
                "--checkpoint",
                str(checkpoint),
                "--identity-file",
                str(identity),
                "--verify-content-sha",
            ]
            valid = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            manifest = json.loads(identity.read_text())
            manifest["dflash"]["model.safetensors"]["sha256"] = "0" * 64
            identity.write_text(json.dumps(manifest))
            invalid = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("checkpoint identity mismatch", invalid.stderr)

    def test_df9_verifier_recognizes_copied_baseline_config_name(self) -> None:
        verifier = (
            root() / "experiments" / EXPERIMENT / "verify_df9_configs.py"
        ).read_text()
        self.assertIn('config_path.stem.removeprefix("resolved-input-")', verifier)

    def test_df9_verifier_accepts_an_inherited_disabled_baseline_draft(self) -> None:
        verifier = (
            root() / "experiments" / EXPERIMENT / "verify_df9_configs.py"
        ).read_text()
        self.assertIn('if "draft" in config.policy:', verifier)
        self.assertIn("assert config.policy.draft.enabled is False", verifier)
        self.assertNotIn('assert "draft" not in config.policy', verifier)

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
            self.assertEqual(
                subprocess.run(
                    ["bash", "-n", str(rendered)], capture_output=True, text=True
                ).returncode,
                0,
            )
            contents = rendered.read_text()
            self.assertIn('export MOUNTS="/lustre:/lustre,/home:/home"', contents)
            self.assertIn("#SBATCH --account=nemotron_n3_post", contents)
            self.assertIn("NRL_FORCE_REBUILD_VENVS=true", contents)
            self.assertIn(f'exec bash "{SOURCE_ROOT}/ray.sub"', contents)
            self.assertNotIn("UV_CACHE_DIR_OVERRIDE", contents)
            self.assertNotIn("UV_PROJECT_ENVIRONMENT", contents)
            self.assertNotIn("/raid/scratch", contents)
            self.assertNotIn("PYTHONPATH=", contents)
            self.assertIn("diagnose_container_python.py", contents)
            diagnostic_source = diagnostic().with_suffix(".py").read_text()
            self.assertIn("importlib.util.find_spec", diagnostic_source)
            self.assertIn("requests", diagnostic_source)
            self.assertIn("urllib3", diagnostic_source)
            self.assertIn("ray", diagnostic_source)

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
                self.assertEqual(
                    subprocess.run(
                        ["bash", "-n", str(path)], capture_output=True, text=True
                    ).returncode,
                    0,
                )
                driver = path.parent / "driver.sh"
                self.assertTrue(driver.is_file())
                self.assertEqual(
                    subprocess.run(
                        ["bash", "-n", str(driver)], capture_output=True, text=True
                    ).returncode,
                    0,
                )
                rendered.append((path.read_text(), driver.read_text()))
            for sbatch, driver in rendered:
                assert_placement_contract(sbatch)
                self.assertRegex(
                    sbatch,
                    r"#SBATCH --job-name=nemotron_n3_post\.q30-20-(?:dflash|dspark)",
                )
                self.assertIn("#SBATCH --account=nemotron_n3_post", sbatch)
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
                self.assertIn(
                    "++policy.generation.vllm_kwargs.compilation_config.backend=eager",
                    driver,
                )
                self.assertIn(
                    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE",
                    driver,
                )
                self.assertIn(
                    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=[1,2,4,8,12,16,24,32,40,48]",
                    driver,
                )
                self.assertIn("NRL_FORCE_REBUILD_VENVS=true uv run", driver)
            self.assertIn("--test-only", harness().read_text())
            preflight = harness().read_text().split("write_sbatch()", maxsplit=1)[0]
            self.assertNotIn("verify_df9_configs.py", preflight)
            self.assertIn(
                'sbatch --test-only "$(write_sbatch "${variant}" "${DURABLE_ROOT}")" 2>&1',
                harness().read_text(),
            )

    def test_placement_contract_rejects_missing_mismatched_and_non_divisible_segments(
        self,
    ) -> None:
        valid = "#SBATCH --nodes=4\n#SBATCH --segment=4\n"
        for mutated in (
            valid.replace("#SBATCH --segment=4\n", ""),
            valid.replace("#SBATCH --segment=4", "#SBATCH --segment=2"),
            valid.replace("#SBATCH --nodes=4", "#SBATCH --nodes=5").replace(
                "#SBATCH --segment=4", "#SBATCH --segment=3"
            ),
        ):
            with self.subTest(mutated=mutated):
                with self.assertRaises(AssertionError):
                    assert_placement_contract(mutated)


if __name__ == "__main__":
    unittest.main()
