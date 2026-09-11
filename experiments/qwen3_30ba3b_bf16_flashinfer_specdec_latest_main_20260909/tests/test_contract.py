from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = (
    ROOT
    / "experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909"
)


class LatestMainBf16FlashinferSpecdecContractTest(unittest.TestCase):
    maxDiff = None

    def render(
        self,
        arm: str,
        max_steps: int = 3,
        context_length: int = 4096,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        env = os.environ.copy()
        env["Q30_LATEST_MAIN_MAX_STEPS"] = str(max_steps)
        env["Q30_LATEST_MAIN_CONTEXT_LENGTH"] = str(context_length)
        env.update(extra_env or {})
        result = subprocess.run(
            ["bash", str(EXPERIMENT / "submit_smoke.sh"), "--render", arm],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_twenty_step_run_is_identifiable_in_wandb_and_config(self) -> None:
        rendered = self.render("dspark_k3", max_steps=20)
        self.assertIn("grpo.max_num_steps=20", rendered)
        self.assertIn("DSparkK3-20step", rendered)

    def test_jobs_use_the_isolated_cgscope_v2_remote_worktree(self) -> None:
        self.assertIn(
            "/home/sna/nemorl-bf16-flashinfer-specdec-cgscope-v2-20260910",
            self.render("dspark_k7", context_length=32768),
        )

    def test_matrix_is_baseline_dflash_and_dspark(self) -> None:
        matrix = subprocess.run(
            ["bash", str(EXPERIMENT / "submit_matrix.sh"), "--list"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(matrix.returncode, 0, matrix.stderr)
        self.assertEqual(
            matrix.stdout.splitlines(),
            ["baseline", "dflash_k3", "dspark_k3", "dspark_k5", "dspark_k7"],
        )

    def test_long_context_matrix_is_baseline_dflash_and_dspark_k_sweep(self) -> None:
        matrix = subprocess.run(
            ["bash", str(EXPERIMENT / "submit_long_context_matrix.sh"), "--list"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(matrix.returncode, 0, matrix.stderr)
        self.assertEqual(
            matrix.stdout.splitlines(),
            [
                "baseline",
                "dflash_k3",
                "dflash_k5",
                "dflash_k7",
                "dspark_k3",
                "dspark_k5",
                "dspark_k7",
            ],
        )

    def test_long_context_contract_is_32k_packed_cp2_and_memory_guarded(self) -> None:
        for arm in (
            "baseline",
            "dflash_k3",
            "dflash_k5",
            "dflash_k7",
            "dspark_k3",
            "dspark_k5",
            "dspark_k7",
        ):
            with self.subTest(arm=arm):
                rendered = self.render(arm, context_length=32768)
                for override in (
                    "grpo.num_prompts_per_step=16",
                    "grpo.num_generations_per_prompt=16",
                    "policy.train_global_batch_size=256",
                    "policy.train_micro_batch_size=1",
                    "policy.logprob_batch_size=1",
                    "policy.max_total_sequence_length=32768",
                    "policy.generation.max_new_tokens=32768",
                    "policy.generation.vllm_cfg.max_model_len=32768",
                    "policy.sequence_packing.enabled=true",
                    "policy.sequence_packing.train_mb_tokens=32768",
                    "policy.sequence_packing.logprob_mb_tokens=32768",
                    "policy.megatron_cfg.context_parallel_size=2",
                    "policy.megatron_cfg.activation_checkpointing=true",
                    "policy.make_sequence_length_divisible_by=8",
                    "policy.generation.vllm_kwargs.max_num_seqs=16",
                    "policy.generation.vllm_kwargs.max_num_batched_tokens=32768",
                ):
                    self.assertIn(override, rendered)
                self.assertIn("-32K-CGScopeV2-", rendered)
                self.assertIn("#SBATCH --time=04:00:00", rendered)
                self.assertIn(
                    "++logger.wandb.group=q30-latest-main-bf16-flashinfer-specdec-32k-cgscope-v2",
                    rendered,
                )

    def test_long_context_capture_sizes_exactly_cover_target_and_drafter(self) -> None:
        query_widths = {
            "baseline": (1,),
            "dflash_k3": (4,),
            "dflash_k5": (6,),
            "dflash_k7": (8,),
            "dspark_k3": (4, 3),
            "dspark_k5": (6, 5),
            "dspark_k7": (8, 7),
        }
        expected_sizes = {
            "baseline": list(range(1, 17)),
            "dflash_k3": list(range(4, 65, 4)),
            "dflash_k5": list(range(6, 97, 6)),
            "dflash_k7": list(range(8, 129, 8)),
            "dspark_k3": [
                3,
                4,
                8,
                12,
                15,
                16,
                20,
                24,
                27,
                28,
                32,
                36,
                39,
                40,
                44,
                48,
                52,
                56,
                60,
                64,
            ],
            "dspark_k5": [
                5,
                6,
                12,
                18,
                24,
                30,
                35,
                36,
                42,
                48,
                54,
                60,
                65,
                66,
                72,
                78,
                84,
                90,
                96,
            ],
            "dspark_k7": [
                7,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                63,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
            ],
        }
        for arm, expected in expected_sizes.items():
            with self.subTest(arm=arm):
                rendered = self.render(arm, context_length=32768)
                command_line = next(
                    line
                    for line in rendered.splitlines()
                    if line.startswith("export COMMAND=")
                )
                command = shlex.split(command_line.removeprefix("export COMMAND="))[0]
                override = next(
                    token
                    for token in shlex.split(command)
                    if "cudagraph_capture_sizes=" in token
                )
                actual = [
                    int(value)
                    for value in re.findall(r"\d+", override.split("=", 1)[1])
                ]
                self.assertEqual(actual, expected)
                self.assertNotIn(32768, actual)
                for width in query_widths[arm]:
                    captured = {
                        ((size + width - 1) // width) * width
                        for size in actual
                        if ((size + width - 1) // width) * width <= 16 * width
                    }
                    self.assertTrue(
                        set(range(width, 16 * width + 1, width)).issubset(captured)
                    )

    def test_every_arm_uses_latest_main_bf16_flashinfer_and_collective_refit(self) -> None:
        for arm in (
            "baseline",
            "dflash_k3",
            "dflash_k5",
            "dflash_k7",
            "dspark_k3",
            "dspark_k5",
            "dspark_k7",
        ):
            with self.subTest(arm=arm):
                rendered = self.render(arm)
                self.assertIn("grpo-qwen3-30ba3b-4n4g.yaml", rendered)
                self.assertIn("grpo.max_num_steps=3", rendered)
                self.assertIn("policy.precision=bfloat16", rendered)
                self.assertIn("policy.draft.enabled=false", rendered)
                self.assertIn("++policy.offload_optimizer_for_refit=false", rendered)
                self.assertIn("policy.generation.refit_transport=null", rendered)
                self.assertIn(
                    "policy.generation.vllm_cfg.refit_with_reload_api=false",
                    rendered,
                )
                self.assertIn(
                    "policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm",
                    rendered,
                )
                self.assertIn("FULL_AND_PIECEWISE", rendered)
                self.assertIn("#SBATCH --nodes=4", rendered)
                self.assertIn("#SBATCH --gpus-per-node=4", rendered)
                self.assertIn("nemo_rl_nightly_20260909_7023221.sqsh", rendered)
                self.assertIn("++logger.wandb.group=q30-latest-main-bf16-flashinfer-specdec", rendered)

    def test_specdec_arms_use_matching_base_ptv3_swa_drafters(self) -> None:
        for arm, method, k in (
            ("dflash_k3", "dflash", 3),
            ("dflash_k5", "dflash", 5),
            ("dflash_k7", "dflash", 7),
            ("dspark_k3", "dspark", 3),
            ("dspark_k5", "dspark", 5),
            ("dspark_k7", "dspark", 7),
        ):
            with self.subTest(arm=arm):
                rendered = self.render(arm)
                self.assertIn(f"speculative_config.method={method}", rendered)
                self.assertIn(f"speculative_config.num_speculative_tokens={k}", rendered)
                self.assertIn(f"sd2p3swa-q30-base-ptv3swe-{method}-b8-16n", rendered)
                self.assertIn("exported-checkpoint-44000", rendered)
        self.assertIn("speculative_config=null", self.render("baseline"))

    def test_dflash_cudagraph_diagnostic_has_matched_fap_and_no_graph_modes(self) -> None:
        common_env = {
            "Q30_LATEST_MAIN_DIAGNOSTIC": "true",
            "Q30_LATEST_MAIN_NSYS": "true",
        }
        fap = self.render(
            "dflash_k5",
            context_length=32768,
            extra_env={**common_env, "Q30_LATEST_MAIN_GRAPH_MODE": "FAP"},
        )
        no_graph = self.render(
            "dflash_k5",
            context_length=32768,
            extra_env={**common_env, "Q30_LATEST_MAIN_GRAPH_MODE": "NONE"},
        )

        for rendered in (fap, no_graph):
            self.assertIn("grpo.seed=42", rendered)
            self.assertIn("grpo.max_num_steps=3", rendered)
            self.assertIn("grpo.num_prompts_per_step=16", rendered)
            self.assertIn("grpo.num_generations_per_prompt=16", rendered)
            self.assertIn("NRL_NSYS_WORKER_PATTERNS=vllm_generation_worker", rendered)
            self.assertIn("NRL_NSYS_PROFILE_STEP_RANGE=2:3", rendered)
            self.assertIn("RAY_LOG_SYNC_FREQUENCY=30", rendered)
            self.assertIn("cuda-graph-trace", rendered)

        self.assertIn("CGDiag-FAP", fap)
        self.assertIn("cudagraph_mode=FULL_AND_PIECEWISE", fap)
        self.assertIn("policy.generation.vllm_cfg.enforce_eager=false", fap)
        self.assertNotIn("vllm_kwargs.enforce_eager", fap)
        self.assertIn("CGDiag-NoGraph", no_graph)
        self.assertIn("cudagraph_mode=NONE", no_graph)
        self.assertIn("policy.generation.vllm_cfg.enforce_eager=false", no_graph)
        self.assertNotIn("vllm_kwargs.enforce_eager", no_graph)

    def test_dflash_cudagraph_diagnostic_matrix_is_three_matched_arms(self) -> None:
        matrix = subprocess.run(
            [
                "bash",
                str(EXPERIMENT / "submit_dflash_cudagraph_diagnostic.sh"),
                "--list",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(matrix.returncode, 0, matrix.stderr)
        self.assertEqual(
            matrix.stdout.splitlines(),
            ["dflash_k5_fap", "dflash_k5_no_graph", "dspark_k5_fap"],
        )

    def test_dspark_uses_source_verified_vllm_compatibility_overlay(self) -> None:
        for arm in ("dspark_k3", "dspark_k5", "dspark_k7"):
            with self.subTest(arm=arm):
                rendered = self.render(arm)
                self.assertIn("prepare_vllm_dspark_fap_overlay.py", rendered)
                self.assertIn("speculative_config.attention_backend=FLASH_ATTN", rendered)
                self.assertIn("kernel_config.enable_flashinfer_autotune=false", rendered)

    def test_driver_reasserts_node_local_venv_and_uv_cache_inside_container(
        self,
    ) -> None:
        rendered = self.render("dspark_k5", context_length=32768)
        command_line = next(
            line for line in rendered.splitlines() if line.startswith("export COMMAND=")
        )
        command = shlex.split(command_line.removeprefix("export COMMAND="))[0]

        self.assertIn(
            'export NEMO_RL_VENV_DIR="${Q30_NODE_ROOT}/venvs";',
            command,
        )
        self.assertIn(
            'export UV_CACHE_DIR="${Q30_NODE_ROOT}/uv-cache";',
            command,
        )
        self.assertIn(
            'mkdir -p "${NEMO_RL_VENV_DIR}" "${UV_CACHE_DIR}";',
            command,
        )

    def test_rendered_driver_environment_is_safe_under_nounset(self) -> None:
        rendered = self.render("dspark_k5", context_length=32768)
        selected_prefixes = (
            "export Q30_NODE_ROOT=",
            "export NEMO_RL_VENV_DIR=",
            "export UV_CACHE_DIR=",
            "export COMMAND=",
        )
        selected_lines = [
            line
            for line in rendered.splitlines()
            if line.startswith(selected_prefixes)
        ]
        result = subprocess.run(
            ["bash", "-c", "\n".join(["set -u", "SLURM_JOB_ID=123", *selected_lines])],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("unbound variable", result.stderr)


if __name__ == "__main__":
    unittest.main()
