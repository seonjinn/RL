#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import render_report
from render_explainer import (
    load_evidence,
    read_context,
    render_from_paths,
    render_html,
    write_html,
)


class RenderExplainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.performance = self.root / "performance.csv"
        self.telemetry = self.root / "telemetry.csv"
        self.correctness = self.root / "correctness.csv"
        self._write_csv(
            self.performance,
            [
                "Exp",
                "Job ID",
                "Mean tokens per sample",
                "E2E TPS/gpu",
                "Gen TPS/gpu",
                "Performance Breakdown - Train TPS/gpu",
                "Performance Breakdown - Logprob TPS/gpu",
                "Total Step Time",
                "Time Breakdown - (Exposed) Generation",
                "Time Breakdown - Policy Training",
                "Time Breakdown - Policy and Reference Logprobs",
            ],
            [
                [
                    "baseline",
                    "1",
                    "2000",
                    "100",
                    "300",
                    "800",
                    "700",
                    "30",
                    "20",
                    "4",
                    "3",
                ],
                [
                    "attn",
                    "2",
                    "2200",
                    "120",
                    "290",
                    "1000",
                    "900",
                    "27",
                    "21",
                    "3",
                    "2",
                ],
            ],
        )
        self._write_csv(
            self.telemetry,
            [
                "Exp",
                "Job ID",
                "Graph Calls",
                "Eligible Calls",
                "Captures",
                "Replays",
                "Cache Hits",
                "Cache Misses",
                "Evictions",
                "Fallbacks",
            ],
            [
                ["baseline", "1", "0", "0", "0", "0", "0", "0", "0", "0"],
                ["attn", "2", "6", "6", "1", "6", "3", "1", "0", "0"],
            ],
        )
        self._write_csv(
            self.correctness,
            [
                "Exp",
                "Job ID",
                "Reward Mean",
                "Gen KL Error Mean",
                "Policy KL Error Mean",
                "Masked Sequences Max",
                "Nonfinite Count",
                "Validation Accuracy Step 20",
            ],
            [
                ["baseline", "1", "0.10", "0.01", "0.02", "0", "0", "0.20"],
                ["attn", "2", "0.11", "0.02", "0.03", "0", "0", "0.21"],
            ],
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def _context() -> dict[str, object]:
        questions = [
            {
                "question": f"Question {index}?",
                "options": ["Wrong", "Correct", "Also wrong"],
                "answer": 1,
                "feedback": "The second option follows the replay contract.",
            }
            for index in range(1, 6)
        ]
        return {
            "schema_version": 1,
            "title": "<script>alert(1)</script>",
            "subtitle": "A measured explanation",
            "updated": "2026-08-05",
            "status": ["Twenty-step smoke is complete."],
            "code_groups": [
                {
                    "title": "Lifecycle",
                    "purpose": "Keep graph banks resident.",
                    "files": ["nemo_rl/models/megatron/cuda_graph_lifecycle.py"],
                    "excerpt": "result = lifecycle.ensure_active(key, capture)",
                }
            ],
            "problems": [
                {
                    "severity": "measure",
                    "title": "Capacity-two LRU thrashes",
                    "detail": "Four schedules compete for two slots.",
                    "next": "Compare capacity four with deterministic bucketing.",
                }
            ],
            "quiz": questions,
        }

    def test_load_evidence_derives_speedup_hit_rate_and_coverage(self) -> None:
        evidence = load_evidence(
            self.performance,
            self.telemetry,
            self.correctness,
        )

        attention = next(row for row in evidence if row.scope == "attn")
        self.assertAlmostEqual(attention.e2e_speedup_pct or 0.0, 20.0)
        self.assertAlmostEqual(attention.cache_hit_pct or 0.0, 75.0)
        self.assertEqual(attention.graph_calls, attention.eligible_calls)
        self.assertEqual(attention.fallback_count, 0)

    def test_load_evidence_rejects_graph_calls_above_eligible_calls(self) -> None:
        self._write_csv(
            self.telemetry,
            [
                "Exp",
                "Job ID",
                "Graph Calls",
                "Eligible Calls",
                "Captures",
                "Replays",
                "Cache Hits",
                "Cache Misses",
                "Evictions",
                "Fallbacks",
            ],
            [
                ["baseline", "1", "0", "0", "0", "0", "0", "0", "0", "0"],
                ["attn", "2", "7", "6", "1", "6", "3", "1", "0", "0"],
            ],
        )

        with self.assertRaisesRegex(ValueError, "exceed eligible calls"):
            load_evidence(self.performance, self.telemetry, self.correctness)

    def test_render_html_contains_explanation_contract_and_escapes_context(
        self,
    ) -> None:
        evidence = load_evidence(
            self.performance,
            self.telemetry,
            self.correctness,
        )
        context = self._context()

        document = render_html(context, evidence)

        for section_id in (
            "background",
            "intuition",
            "code",
            "problems",
            "evidence",
            "quiz",
        ):
            self.assertIn(f'id="{section_id}"', document)
        self.assertEqual(document.count('class="quiz-question"'), 5)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)
        self.assertNotIn("<script>alert(1)</script>", document)
        self.assertIn('href="report.html"', document)
        self.assertRegex(document, r"pre\s*\{[^}]*white-space:\s*pre-wrap")

    def test_read_context_rejects_non_object_json(self) -> None:
        context_path = self.root / "context.json"
        context_path.write_text(json.dumps(["not", "an", "object"]))

        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            read_context(context_path)

    def test_write_html_creates_document_without_trailing_whitespace(self) -> None:
        output = self.root / "nested" / "explainer.html"

        write_html("<!doctype html>  \n<title>Explainer</title>\t\n", output)

        self.assertEqual(
            output.read_text(),
            "<!doctype html>\n<title>Explainer</title>\n",
        )

    def test_render_from_paths_writes_page_and_returns_summary(self) -> None:
        context_path = self.root / "context.json"
        context_path.write_text(json.dumps(self._context()))
        output = self.root / "explainer.html"

        summary = render_from_paths(
            context_path=context_path,
            performance_path=self.performance,
            telemetry_path=self.telemetry,
            correctness_path=self.correctness,
            output_path=output,
        )

        self.assertEqual(summary["evidence_rows"], 2)
        self.assertEqual(summary["quiz_questions"], 5)
        self.assertEqual(summary["output"], str(output))
        self.assertIn('id="background"', output.read_text())

    def test_cli_renders_explicit_inputs(self) -> None:
        context_path = self.root / "context.json"
        context_path.write_text(json.dumps(self._context()))
        output = self.root / "cli-explainer.html"
        script = Path(__file__).with_name("render_explainer.py")

        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--context",
                str(context_path),
                "--performance",
                str(self.performance),
                "--telemetry",
                str(self.telemetry),
                "--correctness",
                str(self.correctness),
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["evidence_rows"], 2)
        self.assertTrue(output.is_file())

    def test_experiment_ledger_links_back_to_explainer(self) -> None:
        document = render_report.render_html(
            [],
            report_context={"schema_version": 1},
        )

        self.assertIn(
            'href="cudagraph_implementation_explainer.html"',
            document,
        )


if __name__ == "__main__":
    unittest.main()
