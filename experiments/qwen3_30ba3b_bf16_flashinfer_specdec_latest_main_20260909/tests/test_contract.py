from __future__ import annotations

import os
from pathlib import Path
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
        self, arm: str, max_steps: int = 3, context_length: int = 4096
    ) -> str:
        env = os.environ.copy()
        env["Q30_LATEST_MAIN_MAX_STEPS"] = str(max_steps)
        env["Q30_LATEST_MAIN_CONTEXT_LENGTH"] = str(context_length)
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

    def test_long_context_matrix_is_baseline_and_dspark_k_sweep(self) -> None:
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
            ["baseline", "dspark_k3", "dspark_k5", "dspark_k7"],
        )

    def test_long_context_contract_is_32k_packed_cp2_and_memory_guarded(self) -> None:
        for arm in ("baseline", "dspark_k3", "dspark_k5", "dspark_k7"):
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
                self.assertIn("-32K-", rendered)
                self.assertIn("#SBATCH --time=04:00:00", rendered)
                self.assertIn(
                    "++logger.wandb.group=q30-latest-main-bf16-flashinfer-specdec-32k",
                    rendered,
                )

    def test_every_arm_uses_latest_main_bf16_flashinfer_and_collective_refit(self) -> None:
        for arm in ("baseline", "dflash_k3", "dspark_k3", "dspark_k5", "dspark_k7"):
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

    def test_dspark_uses_the_source_verified_vllm_fap_overlay(self) -> None:
        for arm in ("dspark_k3", "dspark_k5", "dspark_k7"):
            with self.subTest(arm=arm):
                rendered = self.render(arm)
                self.assertIn("prepare_vllm_dspark_fap_overlay.py", rendered)
                self.assertIn("speculative_config.attention_backend=FLASH_ATTN", rendered)
                self.assertIn("kernel_config.enable_flashinfer_autotune=false", rendered)


if __name__ == "__main__":
    unittest.main()
