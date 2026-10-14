import json
import tempfile
import unittest
from pathlib import Path

from experiments.supervl35_option_b_20261010 import collect_vllm029_tp4 as collector


def result(elapsed: float, quantized: bool) -> dict:
    return {
        "vllm_version": "0.29.0",
        "prompt_sha256": "prompt",
        "config_sha256": "mxfp8-config" if quantized else "bf16-config",
        "model": "same-checkpoint",
        "scoped_mxfp8_ignore": ["*model.layers.0.*"] if quantized else [],
        "request_multiplier": 10,
        "max_num_seqs": 2,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "moe_backend": "flashinfer_trtllm",
        "linear_backend": "flashinfer_cutedsl",
        "attention_backend": "FLASHINFER",
        "concurrencies": [2],
        "runs": [{
            "requests": 20,
            "prompt_tokens": 20480,
            "output_tokens": 20480,
            "batch_size": 2,
            "output_length": 1024,
            "elapsed_s": elapsed,
            "output_tokens_per_second_per_gpu": 20480 / elapsed / 4,
        }],
    }


class CollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.row = {"isl": 1024, "osl": 1024, "concurrency": 2,
                    "replicate": 1, "request_multiplier": 10, "job_id": "200"}
        self.baselines = {(1024, 1024, 2, 1): "100"}
        for job_id, arm, data in (("100", "bf16", result(100, False)),
                                  ("200", "moe", result(80, True))):
            folder = collector.result_path(self.root, 1024, 1024, job_id, arm).parent
            folder.mkdir(parents=True)
            (folder / "exit-code").write_text("0\n")
            (folder / f"{arm}.json").write_text(json.dumps(data))

    def test_matched_result_computes_gpu_throughput(self) -> None:
        row = collector.collect_one(self.root, self.row, self.baselines, "mxfp8-config")
        self.assertEqual(row["output_tokens"], 20480)
        self.assertEqual(row["output_tok_s_gpu"], 64)

    def test_rejects_failed_baseline(self) -> None:
        folder = collector.result_path(self.root, 1024, 1024, "100", "bf16").parent
        (folder / "exit-code").write_text("1\n")
        with self.assertRaisesRegex(ValueError, "Nonzero baseline exit"):
            collector.collect_one(self.root, self.row, self.baselines, "mxfp8-config")

    def test_rejects_backend_or_request_mismatch(self) -> None:
        target = collector.result_path(self.root, 1024, 1024, "200", "moe")
        data = json.loads(target.read_text())
        data["attention_backend"] = "TRITON_ATTN"
        target.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Attention backend mismatch"):
            collector.collect_one(self.root, self.row, self.baselines, "mxfp8-config")


if __name__ == "__main__":
    unittest.main()
