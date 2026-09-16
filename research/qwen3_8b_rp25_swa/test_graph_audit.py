"""Exercise diagnostic wrappers without importing CUDA or vLLM."""

import importlib
import importlib.util
from types import SimpleNamespace
import unittest


class GraphAuditTests(unittest.TestCase):
    def test_counts_real_dispatch_and_successful_replay_without_changing_returns(
        self,
    ) -> None:
        name = "research.qwen3_8b_rp25_swa.graph_audit"
        self.assertIsNotNone(importlib.util.find_spec(name), "graph audit missing")
        audit = importlib.import_module(name)
        events = []

        class Manager:
            def __init__(self) -> None:
                self.graphs = {}
                self._graphs_captured = False
                self.decode_query_len = 6
                self.max_num_reqs = 8
                self.cudagraph_mode = SimpleNamespace(name="FULL_AND_PIECEWISE")

            def capture(self) -> str:
                self.graphs = {"example": object()}
                self._graphs_captured = True
                return "captured"

            def dispatch(
                self,
                num_reqs: int,
                num_tokens: int,
                uniform_token_count: int | None,
                num_active_loras: int,
            ) -> object:
                return SimpleNamespace(
                    cg_mode=SimpleNamespace(
                        name="FULL" if uniform_token_count == 6 else "NONE"
                    ),
                    num_tokens=num_tokens,
                )

            def run_fullgraph(self, desc: object) -> str:
                if desc is None:
                    raise ValueError("no descriptor")
                return "replayed"

        audit.instrument(Manager, events.append)
        m = Manager()
        self.assertEqual(m.capture(), "captured")
        desc = m.dispatch(8, 48, 6, 0)
        self.assertEqual(desc.num_tokens, 48)
        self.assertEqual(m.run_fullgraph(desc), "replayed")
        m.dispatch(8, 100, None, 0)
        with self.assertRaises(ValueError):
            m.run_fullgraph(None)
        snapshot = audit.snapshot(m)
        self.assertEqual(snapshot["full_replays"], 1)
        self.assertEqual(snapshot["dispatch"], {"FULL": 1, "NONE": 1})
        self.assertEqual(snapshot["uniform_decode_fallbacks"], 0)
        self.assertTrue(any(e["event"] == "capture" for e in events))

    def test_no_full_capture_is_not_a_success(self) -> None:
        name = "research.qwen3_8b_rp25_swa.graph_audit"
        self.assertIsNotNone(importlib.util.find_spec(name), "graph audit missing")
        audit = importlib.import_module(name)

        class Manager:
            graphs = {}
            decode_query_len = 6
            max_num_reqs = 8
            cudagraph_mode = SimpleNamespace(name="NONE")

            def capture(self) -> None:
                pass

            def dispatch(self) -> None:
                pass

            def run_fullgraph(self) -> None:
                pass

        audit.instrument(Manager, lambda event: None)
        with self.assertRaisesRegex(RuntimeError, "no FULL"):
            Manager().capture()


if __name__ == "__main__":
    unittest.main()
