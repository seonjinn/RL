import unittest

import pandas as pd

from scripts import build_latest_specdec_html_pages as report


class NemoRlReportTest(unittest.TestCase):
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
