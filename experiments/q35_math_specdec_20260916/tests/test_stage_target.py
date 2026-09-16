from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]


class StageTargetTest(unittest.TestCase):
    def test_incomplete_checkpoint_cannot_be_published(self) -> None:
        path = EXPERIMENT / "stage_target.py"
        self.assertTrue(
            path.is_file(), "staging validator must exist before publishing a target"
        )
        sys.path.insert(0, str(EXPERIMENT))
        spec = importlib.util.spec_from_file_location("q35_stage", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "part.safetensors"}})
            )
            with self.assertRaises(FileNotFoundError):
                module.validate_checkpoint(root)
            (root / "part.safetensors").write_bytes(b"fixture")
            with self.assertRaises(FileNotFoundError):
                module.validate_checkpoint(root)
            (root / "tokenizer.json").write_text("{}")
            (root / "tokenizer_config.json").write_text("{}")
            self.assertEqual(
                module.validate_checkpoint(root), {"shards": 1, "weight_bytes": 7}
            )
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
            with self.assertRaises(ValueError):
                module.validate_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
