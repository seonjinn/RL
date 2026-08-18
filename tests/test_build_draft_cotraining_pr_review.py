from pathlib import Path

import pytest

from scripts import build_draft_cotraining_pr_review as report


CONTEXT_PATH = Path("docs/draft_cotraining_pr_review/context.json")


def test_context_defines_exactly_eleven_ordered_prs() -> None:
    context = report.load_context(CONTEXT_PATH)
    report.validate_context(context)

    assert [pr["id"] for pr in context["prs"]] == [
        f"pr-{number:02d}" for number in range(1, 12)
    ]
    assert len({pr["title"] for pr in context["prs"]}) == 11


def test_context_rejects_duplicate_pr_ids() -> None:
    context = report.load_context(CONTEXT_PATH)
    context["prs"][1]["id"] = context["prs"][0]["id"]

    with pytest.raises(ValueError, match="unique PR id"):
        report.validate_context(context)


def test_render_has_required_sections_in_order() -> None:
    context = report.load_context(CONTEXT_PATH)

    html_text = report.render_html(context)

    offsets = [
        html_text.index(f'id="{section_id}"')
        for section_id in ("background", "intuition", "code", "problems", "evidence", "quiz")
    ]
    assert offsets == sorted(offsets)


def test_render_escapes_editorial_and_file_content() -> None:
    context = report.load_context(CONTEXT_PATH)
    context["prs"][0]["summary"] = '<script>alert("unsafe")</script>'
    context["prs"][0]["files"] = ["a&b.py"]

    html_text = report.render_html(context)

    assert '<script>alert("unsafe")</script>' not in html_text
    assert "&lt;script&gt;" in html_text
    assert "a&amp;b.py" in html_text
