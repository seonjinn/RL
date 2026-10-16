"""Run PR #4756 tests with policy-worker dependencies and base-image pytest."""

import sys

sys.path.append("/opt/nemo_rl_venv/lib/python3.13/site-packages")

import pytest  # noqa: E402

raise SystemExit(
    pytest.main(
        [
            "--automodel-only",
            "-q",
            "tests/unit/models/automodel/test_automodel_utils.py",
            "tests/unit/models/policy/test_automodel_worker.py::test_offload_releases_moe_dispatcher_state_before_empty_cache",
        ]
    )
)
