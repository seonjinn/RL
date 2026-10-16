"""Run MiniMax chunked LM-head regression in the policy-worker environment."""

import sys

sys.path.append("/opt/nemo_rl_venv/lib/python3.13/site-packages")

import pytest  # noqa: E402

raise SystemExit(
    pytest.main(
        [
            "--automodel-only",
            "-q",
            "tests/unit/models/automodel/test_automodel_train.py::test_chunked_lm_head_logprobs_matches_full_projection_gradients",
        ]
    )
)
