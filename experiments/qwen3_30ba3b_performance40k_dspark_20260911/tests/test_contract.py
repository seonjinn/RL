from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/qwen3_30ba3b_performance40k_dspark_20260911"
RECIPE = (
    ROOT
    / "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g-40K.yaml"
)


class Performance40kDsparkContractTest(unittest.TestCase):
    maxDiff = None

    def render(self, arm: str, max_steps: int = 1) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["Q30_PERF40K_MAX_STEPS"] = str(max_steps)
        return subprocess.run(
            ["bash", str(EXPERIMENT / "submit_performance40k.sh"), "--render", arm],
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

    def test_native_performance_40k_recipe_is_the_only_workload_source(self) -> None:
        result = self.render("dspark_k5")
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = result.stdout
        self.assertIn("grpo-qwen3-30ba3b-4n8g-40K.yaml", rendered)
        for forbidden_override in (
            "grpo.num_prompts_per_step=",
            "grpo.num_generations_per_prompt=",
            "policy.train_global_batch_size=",
            "policy.train_micro_batch_size=",
            "policy.logprob_batch_size=",
            "policy.max_total_sequence_length=",
            "policy.generation.max_new_tokens=",
            "policy.generation.vllm_cfg.max_model_len=",
            "policy.megatron_cfg.tensor_model_parallel_size=",
            "policy.megatron_cfg.expert_model_parallel_size=",
            "policy.megatron_cfg.context_parallel_size=",
            "data.",
            "env.math.math_verify_impl=",
        ):
            self.assertNotIn(forbidden_override, rendered)
        recipe = RECIPE.read_text()
        self.assertIn("max_total_sequence_length: 40960", recipe)
        self.assertIn("tensor_model_parallel_size: 4", recipe)
        self.assertIn("expert_model_parallel_size: 8", recipe)
        self.assertIn("context_parallel_size: 8", recipe)
        base = (ROOT / "examples/configs/grpo_math_1B.yaml").read_text()
        self.assertIn("dataset_name: OpenMathInstruct-2", base)
        self.assertIn('math_verify_impl: "hf_math_verify"', base)

    def test_oci_mapping_preserves_32_gpu_world_and_uses_matched_runtime(self) -> None:
        for arm in ("baseline", "dspark_k3", "dspark_k5"):
            with self.subTest(arm=arm):
                result = self.render(arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                rendered = result.stdout
                for expected in (
                    "cluster.gpus_per_node=4",
                    "cluster.num_nodes=8",
                    "cluster.segment_size=4",
                    "policy.precision=bfloat16",
                    "policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm",
                    "policy.generation.vllm_kwargs.max_num_seqs=128",
                    "policy.generation.vllm_kwargs.max_num_batched_tokens=40960",
                    "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE",
                    "logger.wandb.project=sna-specdec",
                    "logger.wandb.group=q30-performance40k-vllm0251-dspark-fixed",
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
                self.assertIn("speculative_config.method=dspark", result.stdout)
                self.assertIn(f"speculative_config.num_speculative_tokens={k}", result.stdout)
                self.assertIn(
                    "sd2p3swa-q30-base-ptv3swe-dspark-b8-16n/exported-checkpoint-44000",
                    result.stdout,
                )
        baseline = self.render("baseline")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertIn("speculative_config=null", baseline.stdout)

    def test_cudagraph_shapes_cover_128_request_widths(self) -> None:
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
                    terminal_shape = 128 * width
                    self.assertIn(terminal_shape, actual)
                    for shape in range(width, terminal_shape + 1, width):
                        padded = next(size for size in actual if size >= shape)
                        self.assertLessEqual(padded, 2 * shape)


if __name__ == "__main__":
    unittest.main()
