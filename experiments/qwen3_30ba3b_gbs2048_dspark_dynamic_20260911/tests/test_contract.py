from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/qwen3_30ba3b_gbs2048_dspark_dynamic_20260911"


class Gbs2048DsparkContractTest(unittest.TestCase):
    maxDiff = None

    def render(self, arm: str, max_steps: int = 20) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["Q30_GBS2048_MAX_STEPS"] = str(max_steps)
        return subprocess.run(
            ["bash", str(EXPERIMENT / "submit_gbs2048.sh"), "--render", arm],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_ready_matrix_is_matched_baseline_and_fixed_k_sweep(self) -> None:
        result = subprocess.run(
            ["bash", str(EXPERIMENT / "submit_matrix.sh"), "--list"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["baseline", "dspark_k3", "dspark_k5", "dspark_k7"],
        )

    def test_official_workload_values_are_inherited_without_override(self) -> None:
        result = self.render("dspark_k5")
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = result.stdout
        self.assertIn("grpo-qwen3-30ba3b-4n4g.yaml", rendered)
        for forbidden_override in (
            "grpo.num_prompts_per_step=",
            "grpo.num_generations_per_prompt=",
            "policy.train_global_batch_size=",
            "policy.train_micro_batch_size=",
            "policy.logprob_batch_size=",
            "policy.max_total_sequence_length=",
            "policy.generation.max_new_tokens=",
            "policy.generation.vllm_cfg.max_model_len=",
            "policy.sequence_packing.train_mb_tokens=",
            "policy.sequence_packing.logprob_mb_tokens=",
            "policy.megatron_cfg.context_parallel_size=",
            "policy.generation.vllm_kwargs.max_num_batched_tokens=",
        ):
            self.assertNotIn(forbidden_override, rendered)
        self.assertIn("grpo.max_num_steps=20", rendered)
        self.assertIn("#SBATCH --nodes=4", rendered)
        self.assertIn("#SBATCH --gpus-per-node=4", rendered)

    def test_all_arms_share_bf16_flashinfer_packing_and_fap(self) -> None:
        for arm in ("baseline", "dspark_k3", "dspark_k5", "dspark_k7"):
            with self.subTest(arm=arm):
                result = self.render(arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                rendered = result.stdout
                for expected in (
                    "policy.precision=bfloat16",
                    "policy.sequence_packing.enabled=true",
                    "policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm",
                    "policy.generation.vllm_kwargs.max_num_seqs=128",
                    "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE",
                    "logger.wandb.project=sna-specdec",
                    "logger.wandb.group=q30-gbs2048-4k-vllm0251-dspark-fixed",
                ):
                    self.assertIn(expected, rendered)

    def test_fixed_arms_use_ptv3_swa_44k_dspark(self) -> None:
        for arm, k in (("dspark_k3", 3), ("dspark_k5", 5), ("dspark_k7", 7)):
            with self.subTest(arm=arm):
                result = self.render(arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                rendered = result.stdout
                self.assertIn("speculative_config.method=dspark", rendered)
                self.assertIn(f"speculative_config.num_speculative_tokens={k}", rendered)
                self.assertIn(
                    "sd2p3swa-q30-base-ptv3swe-dspark-b8-16n/exported-checkpoint-44000",
                    rendered,
                )
        baseline = self.render("baseline")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertIn("speculative_config=null", baseline.stdout)

    def test_cudagraph_capture_sizes_cover_target_and_dspark_widths(self) -> None:
        for arm, widths in (
            ("baseline", (1,)),
            ("dspark_k3", (4, 3)),
            ("dspark_k5", (6, 5)),
            ("dspark_k7", (8, 7)),
        ):
            with self.subTest(arm=arm):
                result = self.render(arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                command_line = next(
                    line
                    for line in result.stdout.splitlines()
                    if line.startswith("export COMMAND=")
                )
                command = shlex.split(command_line.removeprefix("export COMMAND="))[0]
                override = next(
                    token
                    for token in shlex.split(command)
                    if "cudagraph_capture_sizes=" in token
                )
                actual = [int(value) for value in re.findall(r"\d+", override)]
                self.assertEqual(actual, sorted(set(actual)))
                self.assertLessEqual(len(actual), 25)
                for width in widths:
                    terminal_shape = 128 * width
                    self.assertIn(terminal_shape, actual)
                    for shape in range(width, terminal_shape + 1, width):
                        padded = next(size for size in actual if size >= shape)
                        self.assertLessEqual(padded, 2 * shape)

    def test_prebuilt_container_environment_avoids_per_job_venv_build(self) -> None:
        result = self.render("dspark_k5")
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = result.stdout
        self.assertIn("export NEMO_RL_PY_EXECUTABLES_SYSTEM=1", rendered)
        self.assertNotIn("NRL_FORCE_REBUILD_VENVS", rendered)
        self.assertNotIn("NEMO_RL_VENV_DIR", rendered)
        self.assertIn("Q30_VLLM_OVERLAY", rendered)
        self.assertIn("prepare_vllm_dspark_fap_overlay.py", rendered)

    def test_dspark_setup_uses_absolute_source_path_inside_node_container(self) -> None:
        result = self.render("dspark_k5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "/home/sna/nemorl-bf16-flashinfer-specdec-cgscope-v2-20260910/"
            "experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/"
            "prepare_vllm_dspark_fap_overlay.py",
            result.stdout,
        )
        self.assertNotIn("${SOURCE_ROOT}/experiments", result.stdout)

    def test_unproven_dynamic_arm_is_rejected_not_mislabeled(self) -> None:
        result = self.render("dspark_dynamic")
        self.assertEqual(result.returncode, 3)
        self.assertIn("reduced draft work is not proven", result.stderr)


if __name__ == "__main__":
    unittest.main()
