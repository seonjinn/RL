# NeMo RL Draft Co-Training Review Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one maintained offline HTML page that summarizes the eleven draft co-training PRs and opens a concise button-selected subpage for each PR.

**Architecture:** A versioned JSON file is the canonical editorial and status source. A deterministic standard-library Python renderer validates the schema, escapes every external string, and emits embedded CSS/JavaScript with continuous explain-diff sections plus hash-addressable PR subpages. The generated HTML is validated by unit tests and the `explain-diff-html` validator.

**Tech Stack:** Python 3.12 standard library, JSON, pytest, semantic HTML, embedded CSS and JavaScript.

**Spec:** `docs/superpowers/specs/2026-08-18-nemorl-speculative-draft-co-training.md`

## Global Constraints

- Keep dashboard implementation out of every NVIDIA-NeMo/RL upstream PR diff.
- The page is a single offline HTML file with no network dependencies.
- Top-level content is continuous and ordered `background`, `intuition`, `code`, `problems`, `evidence`, `quiz`; PR buttons are subordinate navigation, not top-level tabs.
- Include exactly five medium-difficulty multiple-choice questions using `fieldset.quiz-question` and `aria-live="polite"` feedback.
- Escape context strings before rendering HTML; do not inject context as executable JavaScript.
- Planned work, confirmed results, copied measurements, and inferred risks must use visibly different labels.
- Update JSON first and regenerate; never hand-edit generated HTML.
- Every upstream PR must run `review-pr-team` on OCI-Hsg (primary GPU-capable host), post its output, findings, and dispositions to the PR, resolve high-confidence findings with regression tests where applicable, then request human review.

---

### Task 1: Define and Validate the Dashboard Context

**Files:**
- Create: `docs/draft_cotraining_pr_review/context.json`
- Create: `scripts/build_draft_cotraining_pr_review.py`
- Test: `tests/test_build_draft_cotraining_pr_review.py`

**Interfaces:**
- Consumes: UTF-8 JSON with `title`, `updated_at`, `base_repo`, `integration`, `prs`, `problems`, `evidence`, and `quiz`.
- Produces: `load_context(path: Path) -> dict[str, Any]` and `validate_context(context: Mapping[str, Any]) -> None`.

- [ ] **Step 1: Write the failing context-validation test.**

```python
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
```

- [ ] **Step 2: Run the test and verify RED.**

Run:

```bash
pytest -q tests/test_build_draft_cotraining_pr_review.py::test_context_defines_exactly_eleven_ordered_prs
```

Expected: collection fails because `scripts.build_draft_cotraining_pr_review` does not exist.

- [ ] **Step 3: Add the canonical eleven-PR JSON.**

Each PR object must contain this concrete shape:

```json
{
  "id": "pr-01",
  "number": 1,
  "title": "Typed draft training contracts",
  "status": "planning",
  "base_sha": "7fa6e55192530ff1346d670ce74f9c70cab8f75b",
  "head_sha": null,
  "branch": "seonjinn/draft-cotrain-contracts-20260818",
  "url": null,
  "depends_on": [],
  "summary": "Normalize existing single-pass EAGLE training behind a typed method boundary without changing tensor behavior.",
  "why": "Later methods need one validated method identity without spreading conditionals through workers.",
  "changes": [],
  "files": [],
  "validation": [],
  "performance": [],
  "risks": ["Legacy YAML without method must retain EAGLE defaults."],
  "self_review": "not-run"
}
```

Populate PRs 2-11 with the approved roadmap titles, dependencies, concise summaries, and current `planning` status. Populate five quiz records in the canonical JSON.

- [ ] **Step 4: Implement minimal context loading and validation.**

```python
def load_context(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_context(context: Mapping[str, Any]) -> None:
    prs = context.get("prs")
    if not isinstance(prs, list) or len(prs) != 11:
        raise ValueError("context must contain exactly 11 PRs")
    ids = [pr.get("id") for pr in prs]
    if len(set(ids)) != len(ids):
        raise ValueError("context requires a unique PR id for every PR")
    expected = [f"pr-{number:02d}" for number in range(1, 12)]
    if ids != expected:
        raise ValueError("PR ids must be ordered from pr-01 through pr-11")
```

- [ ] **Step 5: Run the validation tests and verify GREEN.**

Run:

```bash
pytest -q tests/test_build_draft_cotraining_pr_review.py -k context
```

Expected: context tests pass.

### Task 2: Render the Required Explain-Diff Structure

**Files:**
- Modify: `scripts/build_draft_cotraining_pr_review.py`
- Modify: `tests/test_build_draft_cotraining_pr_review.py`

**Interfaces:**
- Consumes: validated context from Task 1.
- Produces: `render_html(context: Mapping[str, Any]) -> str` with all required section IDs in order.

- [ ] **Step 1: Write the failing structure and escaping tests.**

```python
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
```

- [ ] **Step 2: Run the tests and verify RED.**

Run:

```bash
pytest -q tests/test_build_draft_cotraining_pr_review.py -k 'required_sections or escapes'
```

Expected: fails because `render_html` is absent.

- [ ] **Step 3: Implement semantic render helpers.**

Use `html.escape(value, quote=True)` for all context strings. Render:

- `background`: target/policy/draft/refit flow and version/runtime definitions;
- `intuition`: a concrete Qwen3-8B example showing stale versus co-trained draft flow;
- `code`: PR dependency overview plus the PR selector and subpage container;
- `problems`: confirmed blockers, risks, and one next gate per item;
- `evidence`: validation matrix with provenance/status labels;
- `quiz`: the five canonical questions.

The representative `<pre>` block in each planned PR subpage must show planned interface text, not invented code diffs, until a head SHA exists.

- [ ] **Step 4: Run the structure tests and verify GREEN.**

Run:

```bash
pytest -q tests/test_build_draft_cotraining_pr_review.py -k 'required_sections or escapes'
```

Expected: tests pass.

### Task 3: Add Button-Selected, Hash-Addressable PR Subpages

**Files:**
- Modify: `scripts/build_draft_cotraining_pr_review.py`
- Modify: `tests/test_build_draft_cotraining_pr_review.py`

**Interfaces:**
- Consumes: eleven rendered PR articles with IDs `pr-01` through `pr-11`.
- Produces: eleven `<button class="pr-button" data-pr-id="pr-NN">` controls and eleven `<article class="pr-subpage" id="pr-NN">` elements; URL hash opens a specific article.

- [ ] **Step 1: Write the failing selector test.**

```python
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
```

- [ ] **Step 2: Run the test and verify RED.**

Expected: missing buttons/subpages or hash controller.

- [ ] **Step 3: Add accessible selector markup and embedded JavaScript.**

Buttons use `aria-controls`, `aria-pressed`, and keyboard-native `<button>` behavior. JavaScript reads a valid `#pr-NN` hash, hides non-selected articles with the `hidden` attribute, updates `aria-pressed`, and changes the hash without a network navigation. The first PR is selected when the hash is absent or invalid.

- [ ] **Step 4: Run the selector test and verify GREEN.**

Run:

```bash
pytest -q tests/test_build_draft_cotraining_pr_review.py::test_render_has_eleven_buttons_and_subpages
```

Expected: pass.

### Task 4: Render Exact Quiz and Evidence Semantics

**Files:**
- Modify: `scripts/build_draft_cotraining_pr_review.py`
- Modify: `tests/test_build_draft_cotraining_pr_review.py`

**Interfaces:**
- Consumes: five quiz records and evidence records labeled `planned`, `confirmed`, `measured`, or `blocked`.
- Produces: exactly five fieldsets and immediate accessible feedback; no planned evidence is displayed as measured.

- [ ] **Step 1: Write failing quiz/evidence tests.**

```python
def test_render_has_exactly_five_accessible_quiz_questions() -> None:
    html_text = report.render_html(report.load_context(CONTEXT_PATH))
    assert html_text.count('class="quiz-question"') == 5
    assert html_text.count('aria-live="polite"') == 5
    assert "Correct" in html_text
    assert "Try again" in html_text


def test_planned_evidence_is_not_labeled_measured() -> None:
    context = report.load_context(CONTEXT_PATH)
    context["evidence"] = [{"label": "Qwen3-8B DFlash E2E", "status": "planned", "detail": "Not run"}]
    html_text = report.render_html(context)
    assert "Planned" in html_text
    assert "Measured" not in html_text
```

- [ ] **Step 2: Run tests and verify RED.**

Expected: quiz feedback or evidence labels are absent.

- [ ] **Step 3: Add quiz controls and evidence status rendering.**

Each quiz explanation must name the invariant, not merely reveal the answer. Evidence cards render status from an explicit allowlist and include workload/provenance caveats beside measured tables.

- [ ] **Step 4: Run tests and verify GREEN.**

Run:

```bash
pytest -q tests/test_build_draft_cotraining_pr_review.py -k 'quiz or evidence'
```

Expected: pass.

### Task 5: Add Deterministic CLI Generation and Validate the Artifact

**Files:**
- Modify: `scripts/build_draft_cotraining_pr_review.py`
- Modify: `tests/test_build_draft_cotraining_pr_review.py`
- Generate: `docs/draft_cotraining_pr_review/index.html`

**Interfaces:**
- Consumes: `--context Path` and `--output Path`.
- Produces: UTF-8 HTML whose bytes are stable when context is unchanged.

- [ ] **Step 1: Write the failing CLI determinism test.**

```python
def test_main_writes_deterministic_html(tmp_path: Path) -> None:
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"
    report.main(["--context", str(CONTEXT_PATH), "--output", str(first)])
    report.main(["--context", str(CONTEXT_PATH), "--output", str(second)])
    assert first.read_bytes() == second.read_bytes()
```

- [ ] **Step 2: Run the test and verify RED.**

Expected: `main` does not accept an argv list or does not write output.

- [ ] **Step 3: Implement argparse CLI and write the output atomically.**

`main(argv: Sequence[str] | None = None) -> int` parses paths, validates context, renders once, creates the output parent, writes a sibling temporary file, and replaces the output. Do not include the wall-clock render time in HTML; use canonical `updated_at` from JSON.

- [ ] **Step 4: Run renderer tests and generate the page.**

Run:

```bash
pytest -q tests/test_build_draft_cotraining_pr_review.py
python3 scripts/build_draft_cotraining_pr_review.py \
  --context docs/draft_cotraining_pr_review/context.json \
  --output docs/draft_cotraining_pr_review/index.html
```

Expected: tests pass and the page is generated.

- [ ] **Step 5: Run the bundled explain-diff validator.**

Run:

```bash
python3 /Users/sna/.codex/skills/explain-diff-html/scripts/validate_explainer.py \
  /Users/sna/Nemo-RL_Qwen3_Roadmap/docs/draft_cotraining_pr_review/index.html
```

Expected: exit code 0.

- [ ] **Step 6: Open the page and visually verify both wide and narrow layouts.**

Run:

```bash
open /Users/sna/Nemo-RL_Qwen3_Roadmap/docs/draft_cotraining_pr_review/index.html
```

Verify button selection, hashes, keyboard focus, overflow tables, quiz feedback, planned-result labels, and the eleven concise PR subpages.

- [ ] **Step 7: Commit only dashboard-owned files.**

```bash
git add \
  docs/draft_cotraining_pr_review/context.json \
  docs/draft_cotraining_pr_review/index.html \
  scripts/build_draft_cotraining_pr_review.py \
  tests/test_build_draft_cotraining_pr_review.py \
  docs/superpowers/plans/2026-08-18-nemorl-draft-cotrain-review-dashboard.md \
  docs/superpowers/plans/2026-08-18-nemorl-draft-cotrain-roadmap.md \
  docs/superpowers/specs/2026-08-18-nemorl-speculative-draft-co-training.md
git commit -s -m "docs: add draft co-training PR review dashboard"
```

Do not stage any pre-existing dirty roadmap files.
