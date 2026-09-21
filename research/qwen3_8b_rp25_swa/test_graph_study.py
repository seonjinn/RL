"""Guard opt-in graph runs against leaking the old eager/S8 settings."""

import importlib
import importlib.util
import json
import unittest

from research.qwen3_8b_rp25_swa.study import build_new_arms, overrides


class GraphStudyTests(unittest.TestCase):
    def test_online_packed_300step_segments_keep_one_training_horizon(self) -> None:
        renderer = importlib.import_module(
            "research.qwen3_8b_rp25_swa.graph_study"
        )
        arms = {arm.name: arm for arm in build_new_arms()}
        cases = (
            ("baseline", None, 15, "5", list(range(15, 301, 15))),
            ("baseline", 64, 300, "5", list(range(15, 301, 15))),
            ("dflash-fixed-10", 64, 20, "10", list(range(20, 301, 20))),
            ("dspark-always", 128, 300, "10", list(range(20, 301, 20))),
        )
        for arm_name, seqs, stop_step, save_period, required in cases:
            with self.subTest(arm=arm_name, seqs=seqs, stop=stop_step):
                values = dict(
                    item[2:].split("=", 1)
                    for item in renderer.online_graph_overrides(
                        arms[arm_name],
                        "/lustre/test",
                        seqs,
                        production_steps=300,
                        segment_stop_step=stop_step,
                    )
                )
                self.assertEqual(values["grpo.max_num_steps"], "300")
                self.assertEqual(values["grpo.segment_stop_step"], str(stop_step))
                self.assertEqual(values["checkpointing.save_period"], save_period)
                self.assertEqual(values["checkpointing.keep_top_k"], "2")
                self.assertEqual(
                    json.loads(values["cadence_runtime.required_checkpoint_steps"]),
                    required,
                )
                self.assertEqual(
                    values["logger.wandb.group"],
                    "q8-gbs512-32k-packed-online-300step-segmented-20260920",
                )
                self.assertIn(f"SegmentTo{stop_step}", values["logger.wandb.name"])

    def test_online_packed_300step_segments_reject_invalid_boundaries(self) -> None:
        renderer = importlib.import_module(
            "research.qwen3_8b_rp25_swa.graph_study"
        )
        arms = {arm.name: arm for arm in build_new_arms()}
        for arm_name, seqs, stop_step in (
            ("baseline", 64, None),
            ("baseline", 64, 20),
            ("dflash-frozen", 64, 15),
            ("dspark-fixed-10", 128, 301),
        ):
            with self.subTest(arm=arm_name, stop=stop_step):
                with self.assertRaises(ValueError):
                    renderer.online_graph_overrides(
                        arms[arm_name],
                        "/lustre/test",
                        seqs,
                        production_steps=300,
                        segment_stop_step=stop_step,
                    )

    def test_online_packed_matrix_keeps_training_and_covers_s64_s128(self) -> None:
        name = "research.qwen3_8b_rp25_swa.graph_study"
        renderer = importlib.import_module(name)
        arms = {arm.name: arm for arm in build_new_arms()}
        cases = (
            ("baseline", None),
            ("baseline", 64),
            ("baseline", 128),
            ("dflash-frozen", 64),
            ("dflash-fixed-10", 128),
            ("dflash-always", 64),
            ("dspark-frozen", 128),
            ("dspark-fixed-10", 64),
            ("dspark-always", 128),
        )
        for arm_name, seqs in cases:
            with self.subTest(arm=arm_name, seqs=seqs):
                values = dict(
                    item[2:].split("=", 1)
                    for item in renderer.online_graph_overrides(
                        arms[arm_name],
                        "/lustre/test",
                        seqs,
                    )
                )
                self.assertEqual(values["grpo.max_num_steps"], "20")
                self.assertEqual(values["grpo.num_prompts_per_step"], "64")
                self.assertEqual(values["grpo.num_generations_per_prompt"], "8")
                self.assertEqual(values["policy.train_global_batch_size"], "512")
                self.assertEqual(values["policy.max_total_sequence_length"], "32768")
                self.assertEqual(values["policy.generation.max_new_tokens"], "30720")
                self.assertEqual(values["policy.sequence_packing.enabled"], "true")
                self.assertEqual(values["checkpointing.save_period"], "5")
                self.assertEqual(
                    values["cadence_runtime.required_checkpoint_steps"],
                    "[5,10,15,20]",
                )
                self.assertEqual(
                    values[
                        "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode"
                    ],
                    "FULL_AND_PIECEWISE",
                )
                self.assertEqual(
                    values["policy.generation.vllm_cfg.enforce_eager"], "false"
                )
                if seqs is None:
                    self.assertNotIn(
                        "policy.generation.vllm_kwargs.max_num_seqs", values
                    )
                else:
                    self.assertEqual(
                        values["policy.generation.vllm_kwargs.max_num_seqs"],
                        str(seqs),
                    )
                    sizes = json.loads(
                        values[
                            "policy.generation.vllm_kwargs.compilation_config."
                            "cudagraph_capture_sizes"
                        ]
                    )
                    self.assertIn(seqs, sizes)
                    if arm_name != "baseline":
                        self.assertIn(seqs * 5, sizes)
                        self.assertIn(seqs * 6, sizes)
                self.assertEqual(
                    values["policy.draft.enabled"],
                    "false" if arm_name == "baseline" else "true",
                )

    def test_online_packed_matrix_rejects_unapproved_arms_and_concurrency(self) -> None:
        renderer = importlib.import_module("research.qwen3_8b_rp25_swa.graph_study")
        arms = {arm.name: arm for arm in build_new_arms()}
        with self.assertRaises(ValueError):
            renderer.online_graph_overrides(arms["dflash-fixed-5"], "/lustre/test", 64)
        with self.assertRaises(ValueError):
            renderer.online_graph_overrides(arms["dflash-frozen"], "/lustre/test", None)
        with self.assertRaises(ValueError):
            renderer.online_graph_overrides(arms["baseline"], "/lustre/test", 32)

    def test_packed_graph_variants_enable_32k_sequence_packing(self) -> None:
        name = "research.qwen3_8b_rp25_swa.graph_study"
        renderer = importlib.import_module(name)
        for arm in build_new_arms():
            if arm.cadence not in ("baseline", "static"):
                continue
            seqs = None if arm.drafter == "none" else 8
            with self.subTest(arm=arm.name):
                values = dict(
                    item[2:].split("=", 1)
                    for item in renderer.graph_overrides(
                        arm,
                        "/lustre/test",
                        seqs,
                        packed=True,
                    )
                )
                self.assertEqual(values["policy.sequence_packing.enabled"], "true")
                self.assertEqual(
                    values["policy.sequence_packing.train_mb_tokens"], "32768"
                )
                self.assertEqual(
                    values["policy.sequence_packing.logprob_mb_tokens"], "32768"
                )
                self.assertEqual(
                    values["logger.wandb.group"],
                    "q8-gbs512-32k-packed-fap-20260917",
                )

    def test_graph_variants_preserve_workload_and_cover_both_query_layouts(
        self,
    ) -> None:
        name = "research.qwen3_8b_rp25_swa.graph_study"
        self.assertIsNotNone(importlib.util.find_spec(name), "graph renderer missing")
        renderer = importlib.import_module(name)
        for arm in build_new_arms():
            if arm.cadence not in ("baseline", "static"):
                continue
            for seqs in (8, 32, 64):
                with self.subTest(arm=arm.name, seqs=seqs):
                    old = dict(
                        x[2:].split("=", 1)
                        for x in overrides(arm, "/lustre/test", long_context=True)
                    )
                    new = dict(
                        x[2:].split("=", 1)
                        for x in renderer.graph_overrides(arm, "/lustre/test", seqs)
                    )
                    prefix = "policy.generation.vllm_kwargs."
                    self.assertEqual(
                        new[prefix + "compilation_config.backend"], "inductor"
                    )
                    self.assertEqual(
                        new[prefix + "compilation_config.cudagraph_mode"],
                        "FULL_AND_PIECEWISE",
                    )
                    self.assertEqual(new[prefix + "max_num_seqs"], str(seqs))
                    self.assertEqual(new[prefix + "max_num_batched_tokens"], "16384")
                    sizes = json.loads(
                        new[prefix + "compilation_config.cudagraph_capture_sizes"]
                    )
                    for n in range(1, seqs + 1):
                        for query_len in (1,) if arm.drafter == "none" else (1, 5, 6):
                            self.assertIn(n * query_len, sizes)
                    self.assertEqual(max(sizes), 16384)
                    self.assertEqual(
                        new[prefix + "compilation_config.max_cudagraph_capture_size"],
                        "16384",
                    )
                    for key, value in old.items():
                        if key.startswith(
                            (
                                prefix,
                                "logger.wandb.",
                                "policy.generation.vllm_cfg.env_vars",
                            )
                        ):
                            continue
                        self.assertEqual(new[key], value, key)

    def test_default_concurrency_is_baseline_only(self) -> None:
        name = "research.qwen3_8b_rp25_swa.graph_study"
        self.assertIsNotNone(importlib.util.find_spec(name), "graph renderer missing")
        renderer = importlib.import_module(name)
        arms = {arm.name: arm for arm in build_new_arms()}
        values = renderer.graph_overrides(arms["baseline"], "/lustre/test", None)
        self.assertFalse(any("max_num_seqs=" in x for x in values))
        with self.assertRaises(ValueError):
            renderer.graph_overrides(arms["dflash-frozen"], "/lustre/test", None)
        with self.assertRaises(ValueError):
            renderer.graph_overrides(arms["dspark-always"], "/lustre/test", 8)
        with self.assertRaises(ValueError):
            renderer.graph_overrides(arms["baseline"], "/lustre/test", 0)


if __name__ == "__main__":
    unittest.main()
