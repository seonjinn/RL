"""Exercise interpreter selection and fail-closed canary rendering."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import gzip
import tarfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "research/qwen3_8b_rp25_swa/render_canary.sh"


class DriverRenderTests(unittest.TestCase):
    def test_runtime_output_archiver_keeps_high_churn_logs_off_durable_storage(
        self,
    ) -> None:
        archiver = ROOT / "research/qwen3_8b_rp25_swa/archive_runtime_outputs.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            durable = root / "durable"
            ray_logs = root / "ray"
            (runtime / "logs" / "wandb").mkdir(parents=True)
            ray_logs.mkdir()
            (runtime / "overrides.txt").write_text("++packing=true\n")
            (runtime / "recipe.txt").write_text("recipe.yaml\n")
            (runtime / "identity.txt").write_text("head=abc\n")
            (runtime / "process-completed.txt").write_text("process_exit=0\n")
            (runtime / "train.log").write_text("first\nlast\n")
            (runtime / "logs" / "wandb" / "high-churn.log").write_text("do not copy")
            (ray_logs / "worker.log").write_text("ray failure detail")

            result = subprocess.run(
                [
                    "bash",
                    str(archiver),
                    str(runtime),
                    str(durable),
                    str(ray_logs),
                    "12345",
                    "7",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for name in (
                "overrides.txt",
                "recipe.txt",
                "identity.txt",
                "process-completed.txt",
            ):
                self.assertEqual(
                    (durable / name).read_text(), (runtime / name).read_text()
                )
            with gzip.open(durable / "train-tail.log.gz", "rt") as stream:
                self.assertEqual(stream.read(), "first\nlast\n")
            with tarfile.open(
                durable / "ray-logs-failure-12345.tar.gz", "r:gz"
            ) as archive:
                self.assertEqual(
                    archive.extractfile("./worker.log").read().decode(),
                    "ray failure detail",
                )
            self.assertEqual((durable / "runtime-exit.txt").read_text(), "7\n")
            self.assertFalse((durable / "logs").exists())

    def test_launcher_routes_runtime_logs_to_node_local_scratch(self) -> None:
        launcher = (
            ROOT / "research/qwen3_8b_rp25_swa/run_online_canary.sbatch"
        ).read_text()
        self.assertIn('runtime_output_root="${scratch_root}/runtime-output"', launcher)
        self.assertIn('export RAY_TMPDIR="${scratch_root}/ray"', launcher)
        self.assertIn('++logger.log_dir="${runtime_output_root}/logs"', launcher)
        self.assertIn('tee "${runtime_output_root}/train.log"', launcher)
        self.assertNotIn('++logger.log_dir="${output_root}/logs"', launcher)
        self.assertNotIn('tee "${output_root}/train.log"', launcher)

    def test_segmented_online_modes_render_resume_safe_300step_horizon(self) -> None:
        cases = (
            ("baseline", "--online-packed-300-default", "15"),
            ("baseline", "--online-packed-300-128", "300"),
            ("dflash-fixed-10", "--online-packed-300-64", "20"),
            ("dspark-always", "--online-packed-300-128", "300"),
        )
        for arm, mode, stop_step in cases:
            with (
                self.subTest(arm=arm, mode=mode, stop=stop_step),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "attempt"
                output.mkdir()
                result = subprocess.run(
                    [
                        "bash",
                        str(SCRIPT),
                        sys.executable,
                        arm,
                        directory,
                        mode,
                        str(output),
                        stop_step,
                        "0",
                    ],
                    cwd=ROOT,
                    env={**os.environ, "UV_OFFLINE": "1"},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                values = (output / "overrides.txt").read_text()
                self.assertIn("++grpo.max_num_steps=300\n", values)
                self.assertIn(f"++grpo.segment_stop_step={stop_step}\n", values)
                self.assertIn("++checkpointing.save_optimizer=true\n", values)
                self.assertIn("++checkpointing.keep_top_k=2\n", values)

    def test_online_packed_modes_reach_twenty_step_renderer(self) -> None:
        cases = (
            ("baseline", "--online-packed-default", None),
            ("baseline", "--online-packed-64", "64"),
            ("baseline", "--online-packed-128", "128"),
            ("dflash-frozen", "--online-packed-64", "64"),
            ("dflash-fixed-10", "--online-packed-128", "128"),
            ("dflash-always", "--online-packed-64", "64"),
            ("dspark-frozen", "--online-packed-128", "128"),
            ("dspark-fixed-10", "--online-packed-64", "64"),
            ("dspark-always", "--online-packed-128", "128"),
        )
        for arm, mode, seqs in cases:
            with (
                self.subTest(arm=arm, mode=mode),
                tempfile.TemporaryDirectory() as directory,
            ):
                result = subprocess.run(
                    ["bash", str(SCRIPT), sys.executable, arm, directory, mode],
                    cwd=ROOT,
                    env={**os.environ, "UV_OFFLINE": "1"},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                values = (Path(directory) / "overrides.txt").read_text()
                self.assertIn("++grpo.max_num_steps=20\n", values)
                self.assertIn("++policy.sequence_packing.enabled=true\n", values)
                self.assertIn(
                    "++policy.generation.vllm_cfg.enforce_eager=false\n", values
                )
                if seqs is None:
                    self.assertNotIn(
                        "++policy.generation.vllm_kwargs.max_num_seqs=", values
                    )
                else:
                    self.assertIn(
                        f"++policy.generation.vllm_kwargs.max_num_seqs={seqs}\n",
                        values,
                    )

    def test_packed_graph_mode_reaches_renderer_with_32k_token_budgets(self) -> None:
        for arm, mode in (
            ("baseline", "--graph-packed-default"),
            ("dflash-frozen", "--graph-packed-8"),
            ("dspark-frozen", "--graph-packed-8"),
        ):
            with (
                self.subTest(arm=arm, mode=mode),
                tempfile.TemporaryDirectory() as directory,
            ):
                result = subprocess.run(
                    ["bash", str(SCRIPT), sys.executable, arm, directory, mode],
                    cwd=ROOT,
                    env={**os.environ, "UV_OFFLINE": "1"},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                values = (Path(directory) / "overrides.txt").read_text()
                self.assertIn("++policy.sequence_packing.enabled=true\n", values)
                self.assertIn(
                    "++policy.sequence_packing.train_mb_tokens=32768\n", values
                )
                self.assertIn(
                    "++policy.sequence_packing.logprob_mb_tokens=32768\n", values
                )
                self.assertIn(
                    "++policy.generation.vllm_kwargs.compilation_config."
                    "cudagraph_mode=FULL_AND_PIECEWISE\n",
                    values,
                )

    def test_graph_mode_reaches_inductor_and_keeps_frozen_workload(self) -> None:
        for arm, mode in (
            ("baseline", "--graph-fap-default"),
            ("baseline", "--graph-fap-8"),
            ("dflash-frozen", "--graph-fap-8"),
            ("dspark-frozen", "--graph-fap-64"),
        ):
            with (
                self.subTest(arm=arm, mode=mode),
                tempfile.TemporaryDirectory() as directory,
            ):
                result = subprocess.run(
                    ["bash", str(SCRIPT), sys.executable, arm, directory, mode],
                    cwd=ROOT,
                    env={**os.environ, "UV_OFFLINE": "1"},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                values = (Path(directory) / "overrides.txt").read_text()
                self.assertIn("compilation_config.backend=inductor\n", values)
                self.assertIn(
                    "compilation_config.cudagraph_mode=FULL_AND_PIECEWISE\n", values
                )
                self.assertIn("++policy.draft.enabled=false\n", values)
                self.assertIn("++grpo.max_num_steps=3\n", values)

    def test_baseline_staging_does_not_add_specdec(self) -> None:
        script = ROOT / "research/qwen3_8b_rp25_swa/stage_models.sh"
        self.assertTrue(script.exists(), "shared model staging is missing")
        for method in ("none", "dflash", "dspark"):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                target = root / "target-source"
                draft = root / "draft-source"
                target.mkdir()
                draft.mkdir()
                (target / "config.json").write_text("{}")
                (draft / "config.json").write_text("{}")
                (draft / "model.safetensors").write_text("test weights")
                scratch = root / "scratch"
                scratch.mkdir()
                env = {
                    **os.environ,
                    "method": method,
                    "scratch_root": str(scratch),
                    "target_source": str(target),
                    "draft_source": str(draft),
                    "output_root": str(root),
                }
                result = subprocess.run(
                    [
                        "bash",
                        "-eu",
                        "-c",
                        'source "$1"; printf "%s\\n" "${model_overrides[@]}"',
                        "test",
                        str(script),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((scratch / "draft").exists(), method != "none")
                self.assertEqual(
                    "speculative_config.model=" in result.stdout, method != "none"
                )

    def test_production_renders_all_eleven_arms_without_canary_limit(self) -> None:
        arms = ["baseline"] + [
            f"{method}-{cadence}"
            for method in ("dflash", "dspark")
            for cadence in ("frozen", "always", "fixed-5", "fixed-10", "fixed-20")
        ]
        for arm in arms:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [
                        "bash",
                        str(SCRIPT),
                        sys.executable,
                        arm,
                        directory,
                        "--production",
                    ],
                    cwd=ROOT,
                    env={**os.environ, "UV_OFFLINE": "1"},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                values = (Path(directory) / "overrides.txt").read_text()
                self.assertIn("++grpo.max_num_steps=200\n", values)
                self.assertIn("++checkpointing.save_period=50\n", values)
                if arm == "baseline":
                    self.assertIn(
                        "++policy.generation.vllm_kwargs.speculative_config=null\n",
                        values,
                    )
                    self.assertNotIn("++policy.draft.model_name=", values)
                else:
                    self.assertIn("exported-checkpoint-44000", values)

    def test_smoke_and_300step_modes_reach_the_renderer(self) -> None:
        cases = (
            ("baseline", "--smoke", "5", "[5]"),
            ("dflash-frozen", "--smoke", "5", "[5]"),
            ("dflash-always", "--smoke", "5", "[5]"),
            ("dspark-frozen", "--smoke", "5", "[5]"),
            ("dspark-always", "--smoke", "5", "[5]"),
            ("baseline", "--production-300", "300", "[50,100,150,200,250,300]"),
            ("dflash-fixed-20", "--production-300", "300", "[50,100,150,200,250,300]"),
            ("dspark-fixed-20", "--production-300", "300", "[50,100,150,200,250,300]"),
        )
        for arm, mode, steps, checkpoints in cases:
            with (
                self.subTest(arm=arm, mode=mode),
                tempfile.TemporaryDirectory() as directory,
            ):
                result = subprocess.run(
                    ["bash", str(SCRIPT), sys.executable, arm, directory, mode],
                    cwd=ROOT,
                    env={**os.environ, "UV_OFFLINE": "1"},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                values = (Path(directory) / "overrides.txt").read_text()
                self.assertIn(f"++grpo.max_num_steps={steps}\n", values)
                self.assertIn(
                    f"++cadence_runtime.required_checkpoint_steps={checkpoints}\n",
                    values,
                )

    def test_resume_archives_exclusive_terminal_summaries(self) -> None:
        launcher = (
            ROOT / "research/qwen3_8b_rp25_swa/run_online_canary.sbatch"
        ).read_text()
        block = launcher.split("    # terminal_closed writes exclusively:", 1)[1]
        block = block[block.index("    for summary") : block.index("\nfi")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            attempt.mkdir()
            for name in ("checkpoint-runtime.json", "schedule-runtime.json"):
                (root / name).write_text("previous checkpoint summary")
            env = {**os.environ, "result_root": str(root), "output_root": str(attempt)}
            result = subprocess.run(
                ["bash", "-eu", "-c", block], env=env, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("checkpoint-runtime.json", "schedule-runtime.json"):
                self.assertFalse((root / name).exists())
                self.assertEqual(
                    (attempt / name).read_text(), "previous checkpoint summary"
                )
            repeated = subprocess.run(
                ["bash", "-eu", "-c", block], env=env, capture_output=True
            )
            self.assertNotEqual(repeated.returncode, 0)

    def run_render(
        self, python: Path, arm: str, output: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), str(python), arm, str(output)],
            cwd=ROOT,
            env={**os.environ, "UV_OFFLINE": "1"},
            capture_output=True,
            text=True,
        )

    def test_valid_interpreter_renders_both_methods(self) -> None:
        for arm in ("dflash-always", "dspark-always"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                result = self.run_render(Path(sys.executable), arm, output)
                self.assertEqual(result.returncode, 0, result.stderr)
                values = (output / "overrides.txt").read_text().splitlines()
                self.assertIn("++grpo.max_num_steps=2", values)
                self.assertIn("++policy.draft.sliding_window=2048", values)
                self.assertTrue(
                    (output / "recipe.txt").read_text().strip().endswith(".yaml")
                )

    def test_missing_and_dangling_interpreters_fail_before_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            missing = output / "missing-python"
            dangling = output / "dangling-python"
            dangling.symlink_to(missing)
            for python in (missing, dangling):
                result = self.run_render(python, "dflash-always", output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("DRIVER_PYTHON_UNUSABLE", result.stderr)
                self.assertFalse((output / "recipe.txt").exists())

    def test_resume_render_preserves_original_canary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "overrides.txt").write_text("original\n")
            attempt = root / "attempt"
            attempt.mkdir()
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    sys.executable,
                    "dspark-always",
                    str(root),
                    "--resume-check",
                    str(attempt),
                ],
                cwd=ROOT,
                env={**os.environ, "UV_OFFLINE": "1"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "overrides.txt").read_text(), "original\n")
            rendered = (attempt / "overrides.txt").read_text()
            self.assertIn("++grpo.max_num_steps=4\n", rendered)
            self.assertIn(
                f"++checkpointing.checkpoint_dir={root}/checkpoints\n", rendered
            )

    def test_renderer_error_is_not_hidden_by_process_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = self.run_render(Path(sys.executable), "invalid-arm", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((output / "recipe.txt").exists())


if __name__ == "__main__":
    unittest.main()
