"""Submission-boundary checks for the segmented Q8 study."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest


class SubmitSegmentChainTests(unittest.TestCase):
    def module(self):
        return importlib.import_module(
            "research.qwen3_8b_rp25_swa.submit_segment_chain"
        )

    def test_source_bundle_must_advertise_the_exact_expected_commit(self) -> None:
        matrix = importlib.import_module(
            "research.qwen3_8b_rp25_swa.segment_matrix"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "repository"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repository,
                check=True,
            )
            (repository / "README.md").write_text("fixture\n")
            subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "fixture"],
                cwd=repository,
                check=True,
            )
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip()
            bundle = root / "source.bundle"
            subprocess.run(
                ["git", "bundle", "create", str(bundle), "--all"],
                cwd=repository,
                check=True,
            )
            bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
            inputs = matrix.SubmissionInputs(
                expected_head=head,
                bundle=str(bundle),
                bundle_sha=bundle_sha,
                result_parent=str(root / "results"),
                account="coreai_dlalgo_nemorl",
                script="run.sbatch",
                log_dir=str(root / "results" / "scheduler-logs"),
            )

            self.module()._validate_source_bundle(inputs)
            invalid_inputs = replace(inputs, expected_head="f" * 40)
            with self.assertRaisesRegex(ValueError, "expected commit"):
                self.module()._validate_source_bundle(invalid_inputs)


if __name__ == "__main__":
    unittest.main()
