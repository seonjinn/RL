"""Contracts for the segmented Qwen3-8B DAPO OSL32K experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


EXPERIMENT = "qwen3_8b_dapo_osl32k_100step_20260823"
ARMS = ("baseline-k0", "dflash-k5", "dspark-k5")
ENDPOINTS = (25, 50, 75, 100)
HARNESS_BASE_SHA = "afac9bec73067a81141af5dbdb7a5a972d2ee24d"
PRODUCT_BASE_SHA = "3020cf42c4ec416c83ba2cd78ec5b26ca142c412"
HARNESS_SHA = "1" * 40
PRODUCT_SHA = "a28df91a94b623f5108a2992ccac887cc8cbdaab"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
CONTAINER_SHA256 = "6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44"
ALLOWED_SIGNERS = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "modelopt-specdec/assets/signing/allowed_signers_sna_ed25519"
)
ALLOWED_SIGNERS_SHA256 = (
    "e17123da460679f323f85ac201a9826738cc6b16bb54411aa8b0adc3aa072561"
)


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment() -> Path:
    return root() / "research" / EXPERIMENT


def harness() -> Path:
    return experiment() / "harness.py"


def run_harness(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(harness()), *arguments],
        cwd=root(),
        text=True,
        capture_output=True,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_tree_sha256(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(path for path in checkpoint.rglob("*") if path.is_file()):
        relative = str(member.relative_to(checkpoint))
        if relative == "cadence-checkpoint-receipt.json":
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(member)))
    return digest.hexdigest()


def write_checkpoint(result_dir: Path, *, arm: str, endpoint: int) -> Path:
    checkpoint = result_dir / "checkpoints" / f"step_{endpoint}"
    (checkpoint / "policy" / "weights").mkdir(parents=True)
    (checkpoint / "policy" / "optimizer").mkdir(parents=True)
    (checkpoint / "policy" / "weights" / "shard.bin").write_bytes(b"weights")
    (checkpoint / "policy" / "optimizer" / "state.bin").write_bytes(b"optimizer")
    (checkpoint / "train_dataloader.pt").write_bytes(f"dataloader-{endpoint}".encode())
    ledger = checkpoint / "draft-decision-ledger.jsonl"
    decision_count = 0 if arm == "baseline-k0" else endpoint
    ledger.write_text(
        "".join(
            json.dumps({"decision_id": step, "global_step": step}) + "\n"
            for step in range(1, decision_count + 1)
        )
    )
    if arm == "baseline-k0":
        schedule = {
            "mode": "disabled",
            "state": {
                "decisions": 0,
                "next_decision_id": 1,
                "attempted_updates": 0,
                "successful_updates": 0,
                "failed_updates": 0,
                "skipped_updates": 0,
                "attempted_refits": 0,
                "successful_refits": 0,
                "failed_refits": 0,
                "skipped_refits": 0,
                "forced_updates": 0,
                "forced_refits": 0,
                "decision_history": [],
            },
            "events": [],
            "not_applicable_metrics": [
                "draft_loss",
                "draft_grad_norm",
                "applied_draft_version",
            ],
        }
        applied_snapshot = None
    else:
        schedule = {
            "mode": "always",
            "state": {
                "decisions": decision_count,
                "next_decision_id": decision_count + 1,
                "applied_draft_version": decision_count,
            },
        }
        snapshot = checkpoint / "applied-draft.bin"
        snapshot.write_bytes(f"draft-{endpoint}".encode())
        applied_snapshot = {
            "version": endpoint,
            "path": str(snapshot.resolve()),
            "size_bytes": snapshot.stat().st_size,
            "sha256": sha256_file(snapshot),
        }
    ledger_binding = {
        "relative_path": "draft-decision-ledger.jsonl",
        "size_bytes": ledger.stat().st_size,
        "sha256": sha256_file(ledger),
        "first_decision_id": 1 if decision_count else None,
        "last_decision_id": decision_count,
        "entry_count": decision_count,
    }
    receipt = {
        "schema_version": 1,
        "successful": True,
        "checkpoint_id": f"step_{endpoint}",
        "current_step": endpoint,
        "completed_policy_steps": endpoint,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
        "components": {
            "model": {
                "relative_path": "policy/weights",
                "sha256": checkpoint_tree_sha256(checkpoint / "policy" / "weights"),
            },
            "optimizer": {
                "relative_path": "policy/optimizer",
                "sha256": checkpoint_tree_sha256(checkpoint / "policy" / "optimizer"),
            },
            "dataloader_rng": {
                "relative_path": "train_dataloader.pt",
                "sha256": sha256_file(checkpoint / "train_dataloader.pt"),
            },
        },
        "scheduler_state_sha256": hashlib.sha256(
            json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "draft_update_schedule": schedule,
        "applied_draft_snapshot": applied_snapshot,
        "cadence_terminal_evidence": {
            "update_receipts_by_decision": {},
            "observations_by_refit_step": {},
            "selected_science_by_decision": {},
        },
        "decision_ledger": ledger_binding,
        "decision_ledger_prefixes": [ledger_binding],
        "ledger_high_water": decision_count,
        "resumed_from": (
            None
            if endpoint == 25
            else str((result_dir / "checkpoints" / f"step_{endpoint - 25}").resolve())
        ),
    }
    (checkpoint / "cadence-checkpoint-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n"
    )
    (result_dir / f"checkpoint-runtime-step_{endpoint}.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n"
    )
    return checkpoint


def write_runtime_gates(result_dir: Path, *, arm: str, endpoint: int) -> None:
    start = 1 if endpoint == 25 else endpoint - 24
    payload = {
        "arm": arm,
        "segment_start_step": start,
        "segment_stop_step": endpoint,
        "cuda_graph": True,
        "first_step_complete": True,
        "last_step_complete": True,
        "wake_refit": True,
        "no_fatal": True,
    }
    path = result_dir / "runtime-gates" / f"step_{endpoint}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def write_terminal_artifacts(result_dir: Path, *, arm: str) -> None:
    checkpoint_receipt = json.loads(
        (
            result_dir / "checkpoints" / "step_100" / "cadence-checkpoint-receipt.json"
        ).read_text()
    )
    (result_dir / "checkpoint-runtime.json").write_text(
        json.dumps(checkpoint_receipt, sort_keys=True) + "\n"
    )
    if arm == "baseline-k0":
        schedule = {
            "mode": "disabled",
            "current_step": 100,
            "policy_refit_count": 100,
            "successful_target_refits": 100,
            "decision_count": 0,
            "successful_updates": 0,
            "successful_draft_refits": 0,
            "updated_steps": [],
            "refit_steps": [],
        }
    else:
        schedule = {
            "mode": "always",
            "current_step": 100,
            "policy_refit_count": 100,
            "successful_target_refits": 100,
            "decision_count": 100,
            "successful_updates": 100,
            "successful_draft_refits": 100,
            "updated_steps": list(range(1, 101)),
            "refit_steps": list(range(1, 101)),
        }
    (result_dir / "schedule-runtime.json").write_text(
        json.dumps(schedule, sort_keys=True) + "\n"
    )


class SegmentedExperimentContractTest(unittest.TestCase):
    maxDiff = None

    def config(self, arm: str) -> dict[str, object]:
        path = experiment() / "configs" / f"{arm}.yaml"
        self.assertTrue(path.is_file(), f"missing config: {path}")
        return json.loads(path.read_text())

    def manifest(self, arm: str) -> dict[str, object]:
        result = run_harness(
            "manifest",
            "--arm",
            arm,
            "--harness-sha",
            HARNESS_SHA,
            "--product-sha",
            PRODUCT_SHA,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def finalize(
        self, result_dir: Path, *, arm: str, endpoint: int
    ) -> subprocess.CompletedProcess[str]:
        return run_harness(
            "segment-finalize",
            "--arm",
            arm,
            "--endpoint",
            str(endpoint),
            "--result-dir",
            str(result_dir),
            "--harness-sha",
            HARNESS_SHA,
            "--product-sha",
            PRODUCT_SHA,
        )

    def test_configs_keep_one_global_horizon_and_enable_segment_lifecycle(self) -> None:
        for arm in ARMS:
            with self.subTest(arm=arm):
                config = self.config(arm)
                self.assertEqual(config["grpo"]["max_num_steps"], 100)
                self.assertEqual(config["grpo"]["max_num_epochs"], 4)
                self.assertIsNone(config["grpo"]["segment_stop_step"])
                self.assertEqual(config["data_plane"], {"enabled": True})
                self.assertEqual(
                    config["cadence_runtime"]["required_checkpoint_steps"],
                    list(ENDPOINTS),
                )
                self.assertTrue(config["cadence_runtime"]["enabled"])
                self.assertEqual(
                    config["checkpointing"],
                    {
                        "enabled": True,
                        "save_period": 25,
                        "save_optimizer": True,
                        "keep_top_k": None,
                        "metric_name": None,
                    },
                )
                self.assertFalse(config["data"]["shuffle"])
                self.assertEqual(config["data"]["train"]["seed"], 42)
                self.assertIn("first64.jsonl", config["data"]["train"]["data_path"])
                draft = config["policy"]["draft"]
                if arm == "baseline-k0":
                    self.assertEqual(draft, {"enabled": False})
                else:
                    self.assertEqual(draft["update_schedule"], {"mode": "always"})

    def test_manifest_pins_all_immutable_identities_and_three_runtime_arms(
        self,
    ) -> None:
        for arm in ARMS:
            with self.subTest(arm=arm):
                manifest = self.manifest(arm)
                self.assertEqual(manifest["arm"], arm)
                self.assertEqual(manifest["harness_base_sha"], HARNESS_BASE_SHA)
                self.assertEqual(manifest["harness_sha"], HARNESS_SHA)
                self.assertEqual(manifest["product_base_sha"], PRODUCT_BASE_SHA)
                self.assertEqual(manifest["product_sha"], PRODUCT_SHA)
                self.assertEqual(manifest["target_revision"], TARGET_REVISION)
                self.assertEqual(manifest["dataset_revision"], DATASET_REVISION)
                self.assertEqual(manifest["dataset_rows"], 64)
                self.assertEqual(manifest["container_sha256"], CONTAINER_SHA256)
                self.assertEqual(manifest["segment_endpoints"], list(ENDPOINTS))
                self.assertEqual(manifest["max_num_steps"], 100)
                self.assertEqual(manifest["max_num_epochs"], 4)
                self.assertRegex(manifest["config_sha256"], r"^[0-9a-f]{64}$")

        rejected = run_harness(
            "manifest",
            "--arm",
            "baseline-k0",
            "--harness-sha",
            HARNESS_SHA,
            "--product-sha",
            "2" * 40,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("product SHA must equal pinned product commit", rejected.stderr)

    def test_chain_plan_uses_stable_wandb_id_resume_must_and_afterok(self) -> None:
        for arm in ARMS:
            with self.subTest(arm=arm):
                result = run_harness(
                    "plan",
                    "--arm",
                    arm,
                    "--harness-sha",
                    HARNESS_SHA,
                    "--product-sha",
                    PRODUCT_SHA,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                self.assertEqual(
                    [segment["endpoint"] for segment in plan], list(ENDPOINTS)
                )
                self.assertEqual(
                    [segment["predecessor_endpoint"] for segment in plan],
                    [None, 25, 50, 75],
                )
                self.assertEqual(
                    [segment["wandb_resume"] for segment in plan],
                    ["never", "must", "must", "must"],
                )
                self.assertEqual(
                    {segment["wandb_run_id"] for segment in plan},
                    {plan[0]["wandb_run_id"]},
                )
                self.assertEqual(
                    [segment["dependency_type"] for segment in plan],
                    [None, "afterok", "afterok", "afterok"],
                )
                self.assertTrue(
                    all(segment["max_num_steps"] == 100 for segment in plan)
                )

    def test_rendered_segment_jobs_pass_only_the_absolute_stop_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_harness(
                "render-chain",
                "--arm",
                "dflash-k5",
                "--output-root",
                tmp,
                "--harness-sha",
                HARNESS_SHA,
                "--product-sha",
                PRODUCT_SHA,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = json.loads(result.stdout)
            self.assertEqual(len(rendered), 4)
            for item, endpoint in zip(rendered, ENDPOINTS, strict=True):
                sbatch = Path(item["sbatch_path"]).read_text()
                self.assertIn(f'export SEGMENT_STOP_STEP="{endpoint}"', sbatch)
                self.assertIn('export MAX_NUM_STEPS="100"', sbatch)
                self.assertIn(
                    f'export WANDB_RESUME="{"never" if endpoint == 25 else "must"}"',
                    sbatch,
                )
                self.assertIn(f'export ALLOWED_SIGNERS="{ALLOWED_SIGNERS}"', sbatch)
                self.assertIn('export GIT_CONFIG_KEY_0="gpg.format"', sbatch)
                self.assertIn('export GIT_CONFIG_VALUE_0="ssh"', sbatch)
                self.assertIn(
                    'export GIT_CONFIG_KEY_1="gpg.ssh.allowedSignersFile"', sbatch
                )
                self.assertIn('export GIT_CONFIG_VALUE_1="${ALLOWED_SIGNERS}"', sbatch)
                self.assertNotIn("grpo.max_num_steps=25", sbatch)
                self.assertNotIn("grpo.max_num_steps=50", sbatch)
                self.assertNotIn("grpo.max_num_steps=75", sbatch)
            run_segment = (experiment() / "run_segment.sh").read_text()
            self.assertIn(ALLOWED_SIGNERS_SHA256, run_segment)
            self.assertIn('sha256sum "${ALLOWED_SIGNERS}"', run_segment)

    def test_submission_uses_afterok_and_is_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "results"
            scheduler = Path(tmp) / "fake_sbatch.py"
            calls = Path(tmp) / "calls.jsonl"
            counter = Path(tmp) / "counter"
            scheduler.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "calls = pathlib.Path(os.environ['FAKE_SBATCH_CALLS'])\n"
                "with calls.open('a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "if '--test-only' in sys.argv:\n"
                "    print('TEST_ONLY_OK')\n"
                "else:\n"
                "    counter = pathlib.Path(os.environ['FAKE_SBATCH_COUNTER'])\n"
                "    job = int(counter.read_text()) + 1 if counter.exists() else 7001\n"
                "    counter.write_text(str(job))\n"
                "    print(job)\n"
            )
            scheduler.chmod(0o700)
            environment = os.environ.copy()
            environment.update(
                {
                    "FAKE_SBATCH_CALLS": str(calls),
                    "FAKE_SBATCH_COUNTER": str(counter),
                }
            )
            common = [
                sys.executable,
                str(harness()),
                "submit",
                "--arm",
                "baseline-k0",
                "--output-root",
                str(output_root),
                "--harness-sha",
                HARNESS_SHA,
                "--product-sha",
                PRODUCT_SHA,
                "--scheduler",
                str(scheduler),
            ]
            test_only = subprocess.run(
                [*common, "--test-only"],
                cwd=root(),
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            actual = subprocess.run(
                [*common, "--actual"],
                cwd=root(),
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(actual.returncode, 0, actual.stderr)
            record = json.loads(actual.stdout)
            self.assertEqual(record["job_ids"], ["7001", "7002", "7003", "7004"])
            actual_calls = [
                json.loads(line)
                for line in calls.read_text().splitlines()
                if "--test-only" not in json.loads(line)
            ]
            self.assertEqual(len(actual_calls), 4)
            self.assertFalse(any("--dependency=" in arg for arg in actual_calls[0]))
            for call, predecessor in zip(
                actual_calls[1:], ("7001", "7002", "7003"), strict=True
            ):
                self.assertIn(f"--dependency=afterok:{predecessor}", call)
            duplicate = subprocess.run(
                [*common, "--actual"],
                cwd=root(),
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(duplicate.returncode, 0)

    def test_runtime_gate_accepts_cuda_steps_and_draft_wake_refit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            log.write_text(
                "CUDAGRAPH_CAPTURE_COMPLETE\n"
                "Step 26 / 100\n"
                "wake up tags ['weights']\n"
                "GPU Memory after refit complete\n"
                "wake up tags ['kv_cache']\n"
                "Logged data to x/train_data_step26.jsonl\n"
                "Step 50 / 100\n"
                "Logged data to x/train_data_step50.jsonl\n"
            )
            result = run_harness(
                "runtime-gates",
                "--arm",
                "dspark-k5",
                "--endpoint",
                "50",
                "--log",
                str(log),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            gates = json.loads(result.stdout)
            self.assertEqual(gates["segment_start_step"], 26)
            self.assertEqual(gates["segment_stop_step"], 50)
            self.assertTrue(gates["cuda_graph"])
            self.assertTrue(gates["wake_refit"])

            log.write_text(log.read_text().replace("wake up tags ['kv_cache']\n", ""))
            rejected = run_harness(
                "runtime-gates",
                "--arm",
                "dspark-k5",
                "--endpoint",
                "50",
                "--log",
                str(log),
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_segment_receipt_is_keyed_and_created_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            write_checkpoint(result_dir, arm="dflash-k5", endpoint=25)
            write_runtime_gates(result_dir, arm="dflash-k5", endpoint=25)
            first = self.finalize(result_dir, arm="dflash-k5", endpoint=25)
            self.assertEqual(first.returncode, 0, first.stderr)
            receipt_path = Path(first.stdout.strip())
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["arm"], "dflash-k5")
            self.assertEqual(receipt["endpoint"], 25)
            self.assertEqual(receipt["harness_sha"], HARNESS_SHA)
            self.assertEqual(receipt["product_sha"], PRODUCT_SHA)
            self.assertRegex(receipt["config_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(receipt_path.name, r"^[0-9a-f]{64}\.json$")
            duplicate = self.finalize(result_dir, arm="dflash-k5", endpoint=25)
            self.assertNotEqual(duplicate.returncode, 0)

    def test_resume_preflight_verifies_predecessor_tree_ledger_and_dataloader(
        self,
    ) -> None:
        mutations = (
            "weights",
            "ledger",
            "dataloader",
            "optimizer",
            "runtime_receipt",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                result_dir = Path(tmp)
                checkpoint = write_checkpoint(result_dir, arm="dspark-k5", endpoint=25)
                write_runtime_gates(result_dir, arm="dspark-k5", endpoint=25)
                finalized = self.finalize(result_dir, arm="dspark-k5", endpoint=25)
                self.assertEqual(finalized.returncode, 0, finalized.stderr)
                if mutation == "weights":
                    (checkpoint / "policy" / "weights" / "shard.bin").write_bytes(
                        b"drift"
                    )
                elif mutation == "ledger":
                    (checkpoint / "draft-decision-ledger.jsonl").write_text("{}\n")
                elif mutation == "dataloader":
                    (checkpoint / "train_dataloader.pt").unlink()
                elif mutation == "optimizer":
                    (checkpoint / "policy" / "optimizer" / "state.bin").unlink()
                else:
                    (result_dir / "checkpoint-runtime-step_25.json").write_text("{}\n")
                preflight = run_harness(
                    "segment-preflight",
                    "--arm",
                    "dspark-k5",
                    "--endpoint",
                    "50",
                    "--result-dir",
                    str(result_dir),
                    "--harness-sha",
                    HARNESS_SHA,
                    "--product-sha",
                    PRODUCT_SHA,
                )
                self.assertNotEqual(preflight.returncode, 0)

    def test_intermediate_segments_forbid_terminal_and_final_requires_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            write_checkpoint(result_dir, arm="baseline-k0", endpoint=25)
            write_runtime_gates(result_dir, arm="baseline-k0", endpoint=25)
            (result_dir / "checkpoint-runtime.json").write_text("{}\n")
            (result_dir / "schedule-runtime.json").write_text("{}\n")
            intermediate = self.finalize(result_dir, arm="baseline-k0", endpoint=25)
            self.assertNotEqual(intermediate.returncode, 0)

        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            for endpoint in ENDPOINTS:
                write_checkpoint(result_dir, arm="baseline-k0", endpoint=endpoint)
                write_runtime_gates(result_dir, arm="baseline-k0", endpoint=endpoint)
                if endpoint != 100:
                    completed = self.finalize(
                        result_dir, arm="baseline-k0", endpoint=endpoint
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
            missing_terminal = self.finalize(
                result_dir, arm="baseline-k0", endpoint=100
            )
            self.assertNotEqual(missing_terminal.returncode, 0)
            write_terminal_artifacts(result_dir, arm="baseline-k0")
            final = self.finalize(result_dir, arm="baseline-k0", endpoint=100)
            self.assertEqual(final.returncode, 0, final.stderr)

    def test_report_closes_only_after_all_four_segment_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            for endpoint in ENDPOINTS:
                write_checkpoint(result_dir, arm="dflash-k5", endpoint=endpoint)
                write_runtime_gates(result_dir, arm="dflash-k5", endpoint=endpoint)
                if endpoint == 100:
                    write_terminal_artifacts(result_dir, arm="dflash-k5")
                finalized = self.finalize(
                    result_dir, arm="dflash-k5", endpoint=endpoint
                )
                self.assertEqual(finalized.returncode, 0, finalized.stderr)
            report = run_harness(
                "report",
                "--arm",
                "dflash-k5",
                "--result-dir",
                str(result_dir),
                "--harness-sha",
                HARNESS_SHA,
                "--product-sha",
                PRODUCT_SHA,
            )
            self.assertEqual(report.returncode, 0, report.stderr)
            payload = json.loads(report.stdout)
            self.assertEqual(payload["completed_segment_endpoints"], list(ENDPOINTS))
            self.assertEqual(payload["completed_policy_steps"], 100)
            self.assertTrue(payload["terminal"])


if __name__ == "__main__":
    unittest.main()
