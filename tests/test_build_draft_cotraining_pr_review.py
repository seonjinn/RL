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


def test_context_requires_a_review_pr_team_gate_for_every_pr() -> None:
    context = report.load_context(CONTEXT_PATH)

    gates = [pr["review_gate"] for pr in context["prs"]]

    assert len(gates) == 11
    assert all("review-pr-team" in gate for gate in gates)
    assert all("OCI-Hsg" in gate for gate in gates)
    assert all("human review" in gate for gate in gates)


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


def test_render_has_table_of_contents_and_collapsible_beginner_background() -> None:
    html_text = report.render_html(report.load_context(CONTEXT_PATH))

    assert '<nav aria-label="Table of contents">' in html_text
    for section_id in ("background", "intuition", "code", "problems", "evidence", "quiz"):
        assert f'href="#{section_id}"' in html_text
    assert "<details>" in html_text
    assert "<summary>Beginner background</summary>" in html_text


def test_render_escapes_editorial_and_file_content() -> None:
    context = report.load_context(CONTEXT_PATH)
    context["prs"][0]["summary"] = '<script>alert("unsafe")</script>'
    context["prs"][0]["files"] = ["a&b.py"]

    html_text = report.render_html(context)

    assert '<script>alert("unsafe")</script>' not in html_text
    assert "&lt;script&gt;" in html_text
    assert "a&amp;b.py" in html_text


def test_render_keeps_long_pr_metadata_within_a_narrow_layout() -> None:
    html_text = report.render_html(report.load_context(CONTEXT_PATH))

    assert "grid-template-columns: max-content minmax(0, 1fr)" in html_text
    assert "overflow-wrap: anywhere" in html_text


def test_render_includes_body_padding_in_the_narrow_viewport_width() -> None:
    html_text = report.render_html(report.load_context(CONTEXT_PATH))

    assert "body { box-sizing: border-box; width: 100%;" in html_text


def test_render_has_no_trailing_whitespace() -> None:
    html_text = report.render_html(report.load_context(CONTEXT_PATH))

    assert all(line == line.rstrip() for line in html_text.splitlines())


def test_render_has_eleven_buttons_and_subpages() -> None:
    context = report.load_context(CONTEXT_PATH)

    html_text = report.render_html(context)

    assert html_text.count('class="pr-button"') == 11
    assert html_text.count('class="pr-subpage"') == 11
    for number in range(1, 12):
        pr_id = f"pr-{number:02d}"
        assert f'data-pr-id="{pr_id}"' in html_text
        assert f'id="{pr_id}"' in html_text
    assert "history.replaceState" in html_text
    assert "window.location.hash" in html_text


def test_render_has_exactly_five_accessible_quiz_questions() -> None:
    html_text = report.render_html(report.load_context(CONTEXT_PATH))

    assert html_text.count('class="quiz-question"') == 5
    assert html_text.count('aria-live="polite"') == 5
    assert html_text.count('<input type="radio"') == 20
    assert html_text.count('<fieldset class="quiz-question" aria-describedby=') == 5
    assert 'role="radiogroup"' not in html_text
    assert "Correct" in html_text
    assert "Try again" in html_text
    assert "The invariant is that DFlash has one anchor-conditioning query" in html_text


def test_planned_evidence_is_not_labeled_measured() -> None:
    context = report.load_context(CONTEXT_PATH)
    context["evidence"] = [
        {
            "label": "Qwen3-8B DFlash E2E",
            "status": "planned",
            "detail": "Not run",
        }
    ]

    html_text = report.render_html(context)

    assert "<td>Planned</td>" in html_text
    assert "<td>Measured</td>" not in html_text


def test_render_rejects_evidence_status_outside_the_allowlist() -> None:
    context = report.load_context(CONTEXT_PATH)
    context["evidence"][0]["status"] = "estimated"

    with pytest.raises(ValueError, match="unsupported evidence status"):
        report.render_html(context)


def test_render_rejects_non_string_evidence_status() -> None:
    context = report.load_context(CONTEXT_PATH)
    context["evidence"][0]["status"] = ["planned"]

    with pytest.raises(ValueError, match="unsupported evidence status"):
        report.render_html(context)


def test_render_rejects_boolean_quiz_answer() -> None:
    context = report.load_context(CONTEXT_PATH)
    context["quiz"][0]["answer"] = True

    with pytest.raises(ValueError, match="quiz questions require options and an in-range answer"):
        report.render_html(context)


def test_main_writes_deterministic_html(tmp_path: Path) -> None:
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"

    report.main(["--context", str(CONTEXT_PATH), "--output", str(first)])
    report.main(["--context", str(CONTEXT_PATH), "--output", str(second)])

    assert first.read_bytes() == second.read_bytes()
