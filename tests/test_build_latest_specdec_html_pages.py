import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from scripts import build_latest_specdec_html_pages as report


class NemoRlReportTest(unittest.TestCase):
    def test_loads_required_prejuly_snapshot_as_precomputed_evidence(self) -> None:
        rows = report.load_nemorl_prejuly_canonical()

        self.assertEqual(len(rows), 180)
        self.assertEqual(set(pd.Series(rows["strict_match_eligible"]).tolist()), {False})
        precomputed = rows[
            rows["method_k"].astype(str).ne("baseline")
            & rows[
                [
                    "gen_tps_speedup",
                    "e2e_tps_speedup",
                    "generation_time_speedup",
                    "e2e_step_time_speedup",
                ]
            ].notna().any(axis=1)
        ]
        self.assertFalse(precomputed.empty)
        self.assertEqual(
            set(pd.Series(precomputed["baseline_match_state"]).tolist()),
            {"precomputed"},
        )
        self.assertEqual(
            set(rows["job_id"].astype(str)) & {"2172802", "2196588"},
            {"2172802", "2196588"},
        )
        self.assertEqual(
            set(pd.Series(rows["canonical_snapshot"]).tolist()),
            {"docs/lyris_nemorl_perfcfg_specdec_combined_prejuly_20260701.csv"},
        )
        enriched = report.fill_nemorl_speedups(rows)
        enriched["job_id"] = enriched["job_id"].astype(str)
        enriched = enriched.set_index("job_id")
        self.assertEqual(enriched.at["2172802", "baseline_match_state"], "precomputed")
        self.assertEqual(enriched.at["2172802", "gen_tps_speedup"], 1.809753515719269)

    def test_missing_required_prejuly_snapshot_raises(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-prejuly.csv"
            with mock.patch.object(report, "NEMORL_PREJULY_CANONICAL", missing):
                with self.assertRaises(FileNotFoundError) as raised:
                    report.load_nemorl_prejuly_canonical()

        self.assertIn(str(missing), str(raised.exception))

    def test_combines_prejuly_and_july_without_losing_historical_jobs(self) -> None:
        prejuly = report.load_nemorl_prejuly_canonical()
        july = report.load_july_nemorl_results()
        combined = report.combine_nemorl_rows(pd.DataFrame())

        keys = combined[["job_id", "method_k"]].astype(str)
        self.assertFalse(keys.duplicated().any())
        self.assertEqual(len(combined), 235)
        self.assertEqual(len(july), 55)
        prejuly_keys = set(
            map(tuple, prejuly[["job_id", "method_k"]].astype(str).to_numpy())
        )
        july_keys = set(map(tuple, july[["job_id", "method_k"]].astype(str).to_numpy()))
        combined_keys = set(map(tuple, keys.to_numpy()))
        self.assertEqual(len(prejuly_keys & combined_keys), 180)
        self.assertEqual(len(july_keys & combined_keys), 55)
        self.assertIn("2172802", set(combined["job_id"].astype(str)))
        self.assertIn("2196588", set(combined["job_id"].astype(str)))

    def test_loads_all_july_sources_with_normalized_provenance(self) -> None:
        rows = report.load_july_nemorl_results()

        expected_sources = {
            "lyris_nemorl_v020_best_math_live_metrics_20260704.csv",
            "lyris_qwen30_sync_pard_strict_matched_metrics_20260702.csv",
            "lyris_qwen30_async1off_strict_matched_live_metrics_20260702.csv",
            "lyris_qwen32_sync_eagle3_matched_live_metrics_20260702.csv",
            "lyris_qwen32_sync_pard_tp2_noarrms_matched_live_metrics_20260702.csv",
            "lyris_qwen32_async1off_eagle3_matched_live_metrics_20260702.csv",
            "lyris_qwen235b_sync_eagle3_absolute_metrics_20260702.csv",
            "pretyche_qwen32_sync_osl32k_matched_live_metrics_20260702.csv",
        }
        self.assertEqual(len(rows), 55)
        self.assertEqual(
            {Path(value).name for value in rows["manifest"]},
            expected_sources,
        )
        self.assertTrue(
            {
                "cluster",
                "nodes_x_gpus",
                "num_nodes",
                "gpus_per_node",
                "segment",
                "target_tensor_parallel_size",
                "draft_tensor_parallel_size",
                "attention_backend",
                "moe_backend",
                "max_num_batched_tokens",
                "wandb_url",
                "avg_reward_mean",
                "generation_kl_error_mean",
                "result_state",
                "metric_state",
                "strict_match_eligible",
            }.issubset(rows.columns)
        )

        async_baseline = rows[rows["job_id"].eq("2260469")].iloc[0]
        self.assertEqual(async_baseline["model_name"], "Qwen3-30B-A3B")
        self.assertEqual(async_baseline["cluster"], "lyris")
        self.assertEqual(async_baseline["nodes_x_gpus"], "4x4")
        self.assertEqual(async_baseline["segment"], 4)
        self.assertEqual(async_baseline["target_tensor_parallel_size"], 1)
        self.assertEqual(async_baseline["attention_backend"], "TRITON_ATTN")
        self.assertEqual(async_baseline["moe_backend"], "triton")
        self.assertEqual(async_baseline["max_num_batched_tokens"], 32768)
        self.assertEqual(async_baseline["result_state"], "completed")
        self.assertEqual(async_baseline["avg_reward_mean"], 0.525416)
        self.assertEqual(async_baseline["generation_kl_error_mean"], 0.001913)
        self.assertEqual(
            async_baseline["wandb_url"],
            "https://wandb.ai/nvidia/sna-nemorl-specdec-lyris/runs/cj9hx0zo",
        )

        qwen235 = rows[rows["job_id"].eq("2258657")].iloc[0]
        self.assertEqual(qwen235["model_name"], "Qwen3-235B-A22B")
        self.assertEqual(qwen235["nodes_x_gpus"], "32x4")
        self.assertEqual(qwen235["segment"], 16)
        self.assertEqual(qwen235["result_state"], "completed")

        pretyche_partial = rows[rows["job_id"].eq("2319104")].iloc[0]
        self.assertEqual(pretyche_partial["cluster"], "pretyche")
        self.assertEqual(pretyche_partial["result_state"], "partial")
        self.assertEqual(pretyche_partial["slurm_state"], "TIMEOUT_PARTIAL")

        qwen32_async_pard = rows[rows["job_id"].eq("2260761")].iloc[0]
        self.assertEqual(qwen32_async_pard["target_tensor_parallel_size"], 1)
        self.assertEqual(qwen32_async_pard["draft_tensor_parallel_size"], 1)

    def test_loads_july4_v020_live_metrics_as_an_unmatched_current_cohort(self) -> None:
        rows = report.fill_nemorl_speedups(report.load_july_nemorl_results())
        live = rows[rows["job_id"].astype(str).eq("2275728")].iloc[0]

        self.assertEqual(live["model_name"], "Qwen3-30B-A3B")
        self.assertEqual(live["mode"], "sync")
        self.assertEqual(live["method_k"], "suffix_k32")
        self.assertEqual(live["result_state"], "running")
        self.assertEqual(live["completed_steps"], 2)
        self.assertEqual(live["completed_step_span"], "2-3")
        self.assertEqual(live["max_num_seqs"], 128)
        self.assertEqual(live["max_num_batched_tokens"], 16384)
        self.assertAlmostEqual(live["vllm_token_acceptance_pct"], 22.343744, places=5)
        self.assertAlmostEqual(
            live["vllm_acceptance_length_mean_weighted_mean"],
            2.339208,
            places=5,
        )
        self.assertEqual(live["baseline_match_state"], "unmatched_baseline")
        self.assertTrue(pd.isna(live["gen_tps_speedup"]))

    def test_normalizes_failed_held_and_partial_states_without_collapsing_them(self) -> None:
        cases = [
            ("COMPLETED", 19, "Partial peers timed out.", "completed"),
            ("TIMEOUT_PARTIAL", 2, "", "partial"),
            ("TIMEOUT", 0, "", "failed"),
            ("FAILED_STEP2", 2, "A completed peer exists.", "failed"),
            ("HELD", 0, "A completed peer exists.", "held"),
        ]

        for raw_state, completed_steps, notes, expected in cases:
            with self.subTest(raw_state=raw_state):
                self.assertEqual(
                    report.normalize_nemorl_result_state(raw_state, completed_steps, 20, notes),
                    expected,
                )

    def test_july_cg_capture_full_rows_use_the_strict_noarrms_baseline(self) -> None:
        rows = report.fill_nemorl_speedups(report.load_july_nemorl_results()).set_index("job_id")

        self.assertEqual(rows.at["2262387", "baseline_match_state"], "matched")
        self.assertEqual(rows.at["2262687", "baseline_match_state"], "matched")
        self.assertEqual(rows.at["2262387", "gen_tps_speedup"], 0.9875)
        self.assertEqual(rows.at["2262236", "baseline_match_state"], "unmatched_baseline")
        self.assertTrue(pd.isna(rows.at["2262236", "gen_tps_speedup"]))
        self.assertEqual(rows.at["2258657", "baseline_match_state"], "unmatched_baseline")
        self.assertEqual(rows.at["2319104", "baseline_match_state"], "precomputed")
        self.assertEqual(rows.at["2319104", "gen_tps_speedup"], 0.7038)

    def test_strict_baseline_key_isolates_mismatched_july_setups(self) -> None:
        common = {
            "source_group": "July strict source",
            "comparison_group": "July strict source",
            "model_name": "Qwen3-32B",
            "mode": "sync",
            "max_steps": 20,
            "max_new_tokens": 4096,
            "temperature": 1.0,
            "top_p": 1.0,
            "enforce_eager": False,
            "cluster": "lyris",
            "nodes_x_gpus": "4x4",
            "attention_backend": "TRITON_ATTN",
            "moe_backend": "triton",
            "target_tensor_parallel_size": 2,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 32768,
            "segment": 4,
            "config_segment_size": 4,
            "cohort": "standard",
            "fuse_allreduce_rms": True,
            "strict_match_eligible": True,
            "generation_worker_tokens_per_sec_per_gpu_mean": 100.0,
            "e2e_tokens_per_sec_per_gpu_mean": 50.0,
            "generation_time_s_mean": 20.0,
            "total_step_time_s_mean": 40.0,
            "gen_tps_speedup": float("nan"),
            "e2e_tps_speedup": float("nan"),
            "generation_time_speedup": float("nan"),
            "e2e_step_time_speedup": float("nan"),
            "result_state": "completed",
        }
        rows = [{**common, "job_id": "baseline", "method_k": "baseline"}]
        matched = {
            **common,
            "job_id": "matched",
            "method_k": "eagle3_k5",
            "generation_worker_tokens_per_sec_per_gpu_mean": 200.0,
            "e2e_tokens_per_sec_per_gpu_mean": 100.0,
            "generation_time_s_mean": 10.0,
            "total_step_time_s_mean": 20.0,
        }
        rows.append(matched)
        mismatches = {
            "cg-off": {"enforce_eager": True},
            "other-cluster": {"cluster": "oci-hsg"},
            "other-shape": {"nodes_x_gpus": "8x4"},
            "other-attention": {"attention_backend": "FLASH_ATTN"},
            "other-moe": {"moe_backend": "flashinfer"},
            "other-target-tp": {"target_tensor_parallel_size": 1},
            "other-cohort": {"cohort": "noarrms", "fuse_allreduce_rms": False},
        }
        for job_id, changes in mismatches.items():
            rows.append({**matched, **changes, "job_id": job_id})

        enriched = report.fill_nemorl_speedups(pd.DataFrame(rows)).set_index("job_id")

        self.assertEqual(enriched.at["baseline", "baseline_match_state"], "baseline")
        self.assertEqual(enriched.at["matched", "baseline_match_state"], "matched")
        self.assertEqual(enriched.at["matched", "gen_tps_speedup"], 2.0)
        self.assertEqual(enriched.at["matched", "e2e_tps_speedup"], 2.0)
        self.assertEqual(enriched.at["matched", "generation_time_speedup"], 2.0)
        self.assertEqual(enriched.at["matched", "e2e_step_time_speedup"], 2.0)
        for job_id in mismatches:
            with self.subTest(job_id=job_id):
                self.assertEqual(
                    enriched.at[job_id, "baseline_match_state"],
                    "unmatched_baseline",
                )
                self.assertTrue(pd.isna(enriched.at[job_id, "gen_tps_speedup"]))

    def test_unknown_strict_keys_do_not_match_and_precomputed_rows_are_untouched(self) -> None:
        common = {
            "source_group": "Current incomplete source",
            "comparison_group": "Current incomplete source",
            "model_name": "Qwen3-32B",
            "mode": "sync",
            "max_steps": 20,
            "max_new_tokens": 4096,
            "temperature": 1.0,
            "top_p": 1.0,
            "enforce_eager": False,
            "cluster": "lyris",
            "nodes_x_gpus": "4x4",
            "attention_backend": "TRITON_ATTN",
            "moe_backend": "",
            "target_tensor_parallel_size": 2,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 32768,
            "segment": 4,
            "config_segment_size": 4,
            "fuse_allreduce_rms": True,
            "strict_match_eligible": True,
            "result_state": "completed",
            "generation_worker_tokens_per_sec_per_gpu_mean": 100.0,
            "e2e_tokens_per_sec_per_gpu_mean": 50.0,
            "generation_time_s_mean": 20.0,
            "total_step_time_s_mean": 40.0,
            "gen_tps_speedup": float("nan"),
            "e2e_tps_speedup": float("nan"),
            "generation_time_speedup": float("nan"),
            "e2e_step_time_speedup": float("nan"),
        }
        legacy = {
            **common,
            "job_id": "legacy-precomputed",
            "method_k": "eagle3_k3",
            "strict_match_eligible": False,
            "baseline_match_state": "precomputed",
            "gen_tps_speedup": 1.75,
            "manifest": "docs/historical.csv",
        }
        rows = pd.DataFrame(
            [
                {**common, "job_id": "unknown-baseline", "method_k": "baseline"},
                {
                    **common,
                    "job_id": "unknown-spec",
                    "method_k": "eagle3_k5",
                    "generation_worker_tokens_per_sec_per_gpu_mean": 200.0,
                },
                legacy,
            ]
        )

        enriched = report.fill_nemorl_speedups(rows).set_index("job_id")

        self.assertEqual(enriched.at["unknown-spec", "baseline_match_state"], "unmatched_baseline")
        self.assertTrue(pd.isna(enriched.at["unknown-spec", "gen_tps_speedup"]))
        self.assertEqual(enriched.at["legacy-precomputed", "baseline_match_state"], "precomputed")
        self.assertEqual(enriched.at["legacy-precomputed", "gen_tps_speedup"], 1.75)
        self.assertEqual(enriched.at["legacy-precomputed", "manifest"], "docs/historical.csv")

    def test_strict_match_requires_a_completed_metric_bearing_baseline(self) -> None:
        common = {
            "source_group": "Current strict source",
            "comparison_group": "Current strict source",
            "model_name": "Qwen3-32B",
            "mode": "sync",
            "max_steps": 20,
            "max_new_tokens": 4096,
            "temperature": 1.0,
            "top_p": 1.0,
            "enforce_eager": False,
            "cluster": "lyris",
            "nodes_x_gpus": "4x4",
            "attention_backend": "TRITON_ATTN",
            "moe_backend": "triton",
            "target_tensor_parallel_size": 2,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 32768,
            "segment": 4,
            "config_segment_size": 4,
            "fuse_allreduce_rms": True,
            "strict_match_eligible": True,
            "completed_steps": 19,
            "generation_worker_tokens_per_sec_per_gpu_mean": 100.0,
            "e2e_tokens_per_sec_per_gpu_mean": 50.0,
            "generation_time_s_mean": 20.0,
            "total_step_time_s_mean": 40.0,
            "gen_tps_speedup": float("nan"),
            "e2e_tps_speedup": float("nan"),
            "generation_time_speedup": float("nan"),
            "e2e_step_time_speedup": float("nan"),
        }
        cases = {
            "submitted": {"result_state": "submitted"},
            "failed": {"result_state": "failed"},
            "partial": {"result_state": "partial"},
            "metric-empty": {
                "result_state": "completed",
                "generation_worker_tokens_per_sec_per_gpu_mean": float("nan"),
                "e2e_tokens_per_sec_per_gpu_mean": float("nan"),
                "generation_time_s_mean": float("nan"),
                "total_step_time_s_mean": float("nan"),
            },
        }

        for name, baseline_changes in cases.items():
            with self.subTest(name=name):
                rows = pd.DataFrame(
                    [
                        {
                            **common,
                            **baseline_changes,
                            "job_id": f"{name}-baseline",
                            "method_k": "baseline",
                        },
                        {
                            **common,
                            "job_id": f"{name}-spec",
                            "method_k": "eagle3_k5",
                            "result_state": "completed",
                            "generation_worker_tokens_per_sec_per_gpu_mean": 200.0,
                            "e2e_tokens_per_sec_per_gpu_mean": 100.0,
                            "generation_time_s_mean": 10.0,
                            "total_step_time_s_mean": 20.0,
                            "gen_tps_speedup": 9.0,
                        },
                    ]
                )

                enriched = report.fill_nemorl_speedups(rows).set_index("job_id")

                self.assertEqual(
                    enriched.at[f"{name}-baseline", "baseline_match_state"],
                    "unusable_baseline",
                )
                self.assertEqual(
                    enriched.at[f"{name}-spec", "baseline_match_state"],
                    "unmatched_baseline",
                )
                self.assertTrue(pd.isna(enriched.at[f"{name}-spec", "gen_tps_speedup"]))

    def test_nemorl_html_exposes_july_setup_and_correctness_metadata(self) -> None:
        rows = report.fill_nemorl_speedups(report.load_july_nemorl_results())

        html_text = report.build_nemorl_html(rows)

        for heading in [
            "Result state",
            "Baseline match",
            "Cluster",
            "Nodes x GPUs",
            "Target TP",
            "Draft TP",
            "Attention",
            "MoE",
            "Batch-token budget",
            "segment",
            "Reward",
            "Generation KL",
        ]:
            with self.subTest(heading=heading):
                self.assertIn(f">{heading}</th>", html_text)
        self.assertIn("lyris_qwen30_sync_pard_strict_matched_metrics_20260702.csv", html_text)
        self.assertIn("unmatched_baseline", html_text)

    def test_latest_nemorl_builder_preserves_historical_dated_page(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_html = root / "latest.html"
            enriched_csv = root / "enriched.csv"
            combined_csv = root / "combined.csv"
            historical_html = root / "historical.html"
            historical_html.write_text("historical sentinel", encoding="utf-8")

            with mock.patch.object(report, "NEMORL_HTML_DATED", historical_html):
                report.build_latest_nemorl_outputs(
                    live_rows=pd.DataFrame(),
                    output_html=output_html,
                    enriched_csv_out=enriched_csv,
                    combined_csv_out=combined_csv,
                )

            self.assertTrue(output_html.exists())
            self.assertTrue(enriched_csv.exists())
            self.assertTrue(combined_csv.exists())
            self.assertEqual(
                historical_html.read_text(encoding="utf-8"),
                "historical sentinel",
            )

    def test_dedup_prefers_completed_current_evidence_and_preserves_provenance(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "job_id": "duplicate-job",
                    "method_k": "eagle3_k5",
                    "source_group": "July completed source",
                    "source_priority": -10,
                    "completed_steps": 19,
                    "slurm_state": "COMPLETED",
                    "gen_tps_speedup": float("nan"),
                    "manifest": "docs/current.csv",
                    "notes": "completed current evidence",
                },
                {
                    "job_id": "duplicate-job",
                    "method_k": "eagle3_k5",
                    "source_group": "Historical partial source",
                    "source_priority": 0,
                    "completed_steps": 5,
                    "result_state": "partial",
                    "slurm_state": "TIMEOUT_PARTIAL",
                    "gen_tps_speedup": 1.8,
                    "manifest": "docs/historical.csv",
                    "notes": "partial row with speedup",
                },
            ]
        )

        selected = report.deduplicate_nemorl_rows(rows)

        self.assertEqual(len(selected), 1)
        row = selected.iloc[0]
        self.assertEqual(row["result_state"], "completed")
        self.assertEqual(row["manifest"], "docs/current.csv")
        self.assertIn("Historical partial source", row["notes"])
        self.assertIn("alternate source groups", row["notes"])
        self.assertIn("docs/historical.csv", row["alternate_manifests"])

    def test_dedup_prefers_completed_july_row_over_completed_canonical_row(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "job_id": "shared-job",
                    "method_k": "eagle3_k5",
                    "source_group": "Pre-July canonical",
                    "source_priority": 0,
                    "completed_steps": 19,
                    "result_state": "completed",
                    "manifest": "docs/prejuly.csv",
                    "evidence_period": "pre-july-canonical",
                },
                {
                    "job_id": "shared-job",
                    "method_k": "eagle3_k5",
                    "source_group": "July current",
                    "source_priority": -10,
                    "completed_steps": 19,
                    "result_state": "completed",
                    "manifest": "docs/july.csv",
                    "evidence_period": "july-current",
                },
            ]
        )

        selected = report.deduplicate_nemorl_rows(rows)

        self.assertEqual(len(selected), 1)
        row = selected.iloc[0]
        self.assertEqual(row["manifest"], "docs/july.csv")
        self.assertEqual(row["evidence_period"], "july-current")
        self.assertIn("docs/prejuly.csv", row["alternate_manifests"])

    def test_chart_rows_require_a_complete_step20_metric_window(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "job_id": "full-skip-step1",
                    "max_steps": 20,
                    "completed_steps": 19,
                    "last_step": 20,
                    "step_filter": "step>=2",
                    "generation_worker_tokens_per_sec_per_gpu_mean": 100.0,
                    "source_group": "setup-a",
                    "model_name": "Qwen3-32B",
                    "mode": "sync",
                    "max_new_tokens": 4096,
                    "method_k": "eagle3_k3",
                    "gen_tps_speedup": 1.4,
                },
                {
                    "job_id": "partial",
                    "max_steps": 20,
                    "completed_steps": 5,
                    "last_step": 6,
                    "step_filter": "step>=2",
                    "generation_worker_tokens_per_sec_per_gpu_mean": 110.0,
                    "source_group": "setup-b",
                    "model_name": "Qwen3-32B",
                    "mode": "sync",
                    "max_new_tokens": 4096,
                    "method_k": "eagle3_k5",
                    "gen_tps_speedup": 1.5,
                },
                {
                    "job_id": "full-all-steps",
                    "max_steps": 20,
                    "completed_steps": 20,
                    "last_step": 20,
                    "step_filter": "all",
                    "generation_worker_tokens_per_sec_per_gpu_mean": 95.0,
                    "source_group": "setup-c",
                    "model_name": "Qwen3-30B-A3B",
                    "mode": "sync",
                    "max_new_tokens": 1024,
                    "method_k": "eagle3_k3",
                    "gen_tps_speedup": 2.0,
                },
            ]
        )

        selected = report.nemorl_chart_rows(rows)

        self.assertEqual(
            set(selected["job_id"]),
            {"full-skip-step1", "full-all-steps"},
        )

    def test_group_label_separates_cuda_graph_setups(self) -> None:
        common = {
            "model_name": "Qwen3-32B",
            "mode": "sync",
            "max_new_tokens": 4096,
            "cluster": "lyris",
        }
        cuda_graph_on = pd.Series(
            {
                **common,
                "enforce_eager": False,
                "source_group": "Lyris PerfCfg enforce_eager=false 2026-06-23",
            }
        )
        cuda_graph_off = pd.Series(
            {
                **common,
                "enforce_eager": True,
                "source_group": "Lyris PerfCfg CUDA-graph-disabled 2026-06-23",
            }
        )

        on_label = report.nemorl_group_label(cuda_graph_on, include_model=False)
        off_label = report.nemorl_group_label(cuda_graph_off, include_model=False)

        self.assertNotEqual(on_label, off_label)
        self.assertIn("CG-on", on_label)
        self.assertIn("CG-off", off_label)

    def test_cudagraphoff_run_id_is_not_labeled_eagerfalse(self) -> None:
        run_id = "20260623_lyris_nemorl_perfcfg_cudagraph_off_eagertrue_qwen32"

        source_group = report.nemorl_source_group_from_run_id(run_id)
        config_basis = report.nemorl_config_basis_from_run_id(run_id)

        self.assertIn("CUDA-graph-disabled", source_group)
        self.assertIn("enforce_eager=true", config_basis)

    def test_metric_window_labels_step1_exclusion(self) -> None:
        row = pd.Series(
            {
                "max_steps": 20,
                "completed_steps": 19,
                "last_step": 20,
                "completed_step_span": "2-20",
                "step_filter": "step>=2",
            }
        )

        self.assertEqual(
            report.nemorl_metric_window(row),
            "steps 2-20 (19 metrics)",
        )


if __name__ == "__main__":
    unittest.main()
