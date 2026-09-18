"""Guard opt-in graph runs against leaking the old eager/S8 settings."""

import importlib
import importlib.util
import json
import unittest

from research.qwen3_8b_rp25_swa.study import build_new_arms, overrides


class GraphStudyTests(unittest.TestCase):
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
