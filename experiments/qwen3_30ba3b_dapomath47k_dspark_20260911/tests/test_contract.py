from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/qwen3_30ba3b_dapomath47k_dspark_20260911"


class DapoMath47kDsparkContractTest(unittest.TestCase):
    maxDiff = None

    def render(self, arm: str, max_steps: int = 1) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["Q30_DAPO47K_MAX_STEPS"] = str(max_steps)
        return subprocess.run(
            ["bash", str(EXPERIMENT / "submit_dapo47k.sh"), "--render", arm],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_matrix_contains_matched_baseline_and_fixed_k_arms(self) -> None:
        result = subprocess.run(
            ["bash", str(EXPERIMENT / "submit_matrix.sh"), "--list"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["baseline", "dspark_k3", "dspark_k5"])

    def test_dapomath17k_47k_workload_is_explicit(self) -> None:
        result = self.render("baseline")
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = result.stdout
        for expected in (
            "grpo.num_prompts_per_step=128",
            "grpo.num_generations_per_prompt=16",
            "grpo.use_leave_one_out_baseline=false",
            "grpo.reward_scaling.enabled=true",
            "grpo.reward_scaling.target_min=-1.0",
            "grpo.reward_shaping.enabled=true",
            "grpo.reward_shaping.overlong_buffer_length=2048",
            "grpo.reward_shaping.max_response_length=47104",
            "loss_fn.force_on_policy_ratio=false",
            "loss_fn.reference_policy_kl_penalty=0.0",
            "loss_fn.ratio_clip_max=0.28",
            "loss_fn.ratio_clip_c=10",
            "loss_fn.use_on_policy_kl_approximation=true",
            "loss_fn.use_importance_sampling_correction=true",
            "policy.train_global_batch_size=2048",
            "policy.max_total_sequence_length=49152",
            "policy.generation.max_new_tokens=47104",
            "policy.generation.vllm_cfg.max_model_len=49152",
            "data.max_input_seq_length=2048",
            "data.train.dataset_name=DAPOMath17K",
            "data.validation=null",
            "data.default.prompt_file=null",
            "env.math.math_verify_impl=dapo_math_verify",
        ):
            self.assertIn(expected, rendered)
        self.assertNotIn("OpenMathInstruct-2", rendered)

    def test_qwen_vllm_long_context_parallelism_is_matched(self) -> None:
        for arm in ("baseline", "dspark_k3", "dspark_k5"):
            with self.subTest(arm=arm):
                result = self.render(arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                rendered = result.stdout
                for expected in (
                    "policy.precision=bfloat16",
                    "policy.sequence_packing.enabled=true",
                    "policy.megatron_cfg.tensor_model_parallel_size=2",
                    "policy.megatron_cfg.expert_model_parallel_size=8",
                    "policy.megatron_cfg.context_parallel_size=4",
                    "policy.megatron_cfg.activation_checkpointing=true",
                    "policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm",
                    "policy.generation.vllm_kwargs.max_num_seqs=64",
                    "policy.generation.vllm_kwargs.max_num_batched_tokens=49152",
                    "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE",
                    "logger.wandb.project=sna-specdec",
                    "logger.wandb.group=q30-dapomath47k-vllm0251-dspark-fixed",
                    "#SBATCH --nodes=8",
                    "#SBATCH --segment=4",
                    "#SBATCH --gpus-per-node=4",
                ):
                    self.assertIn(expected, rendered)

    def test_dspark_arms_use_new_44k_base_drafter(self) -> None:
        for arm, k in (("dspark_k3", 3), ("dspark_k5", 5)):
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

    def test_cudagraph_shapes_cover_long_context_request_widths(self) -> None:
        for arm, widths in (("baseline", (1,)), ("dspark_k3", (4, 3)), ("dspark_k5", (6, 5))):
            with self.subTest(arm=arm):
                result = self.render(arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                command_line = next(
                    line for line in result.stdout.splitlines() if line.startswith("export COMMAND=")
                )
                command = shlex.split(command_line.removeprefix("export COMMAND="))[0]
                override = next(
                    token for token in shlex.split(command) if "cudagraph_capture_sizes=" in token
                )
                actual = [int(value) for value in re.findall(r"\d+", override)]
                self.assertEqual(actual, sorted(set(actual)))
                self.assertLessEqual(len(actual), 25)
                for width in widths:
                    terminal_shape = 64 * width
                    self.assertIn(terminal_shape, actual)
                    for shape in range(width, terminal_shape + 1, width):
                        padded = next(size for size in actual if size >= shape)
                        self.assertLessEqual(padded, 2 * shape)

    def test_runtime_uses_prebuilt_vllm_environment_and_node_local_overlays(self) -> None:
        result = self.render("dspark_k5")
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = result.stdout
        self.assertNotIn("NRL_FORCE_REBUILD_VENVS", rendered)
        self.assertNotIn("NEMO_RL_VENV_DIR", rendered)
        self.assertIn("/opt/ray_venvs/", rendered)
        self.assertIn("Q30_VLLM_OVERLAY", rendered)
        self.assertIn("/raid/scratch/sna/q30-dapo47k-dspark-", rendered)
        self.assertIn("prepare_vllm_dspark_fap_overlay.py", rendered)


if __name__ == "__main__":
    unittest.main()
