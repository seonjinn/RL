"""Regression contracts for resume-time draft identity diagnostics."""

from __future__ import annotations

import nemo_rl.weight_sync.interfaces as interfaces


def test_draft_state_root_mismatches_names_and_truncates_changed_roots() -> None:
    helper = getattr(interfaces, "draft_state_root_mismatches", None)
    assert helper is not None

    mismatches = helper(
        {
            "draft_model_sha256": "a" * 64,
            "draft_optimizer_sha256": "b" * 64,
        },
        {
            "draft_model_sha256": "c" * 64,
            "draft_optimizer_sha256": "d" * 64,
        },
    )

    assert mismatches == (
        "draft_model_sha256 expected=aaaaaaaaaaaa actual=cccccccccccc",
        "draft_optimizer_sha256 expected=bbbbbbbbbbbb actual=dddddddddddd",
    )
