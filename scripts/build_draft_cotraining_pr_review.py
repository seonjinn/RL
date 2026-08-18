import argparse
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EVIDENCE_STATUS_LABELS = {
    "planned": "Planned",
    "confirmed": "Confirmed",
    "measured": "Measured",
    "blocked": "Blocked",
}
REVIEW_GATE_REQUIRED_CLAUSES = (
    "OCI-Hsg (primary GPU-capable host)",
    "upstream review-pr-team",
    "post its output, findings, and dispositions to the PR",
    "resolve applicable high-confidence issues with regression tests",
    "request human review only afterward",
)


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

    for pr in prs:
        review_gate = pr.get("review_gate")
        if not isinstance(review_gate, str) or any(
            clause not in review_gate for clause in REVIEW_GATE_REQUIRED_CLAUSES
        ):
            raise ValueError("every PR review gate must include every mandatory clause")


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _render_list(items: object, *, empty: str = "None recorded.") -> str:
    if not isinstance(items, list) or not items:
        return f"<p>{empty}</p>"
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _render_pr_subpage(pr: Mapping[str, Any]) -> str:
    pr_id = _escape(pr.get("id", "unknown-pr"))
    title = _escape(pr.get("title", "Untitled PR"))
    number = _escape(pr.get("number", "?"))
    depends_on = _render_list(pr.get("depends_on"), empty="No dependency.")
    files = _render_list(pr.get("files"), empty="No files recorded yet.")
    changes = _render_list(pr.get("changes"), empty="No planned changes recorded yet.")
    validation = _render_list(pr.get("validation"), empty="No validation recorded yet.")
    performance = _render_list(pr.get("performance"), empty="No performance result recorded yet.")
    risks = _render_list(pr.get("risks"), empty="No known risk recorded.")
    head_sha = pr.get("head_sha") or "Not created"
    return f"""
<article class="pr-subpage" id="{pr_id}">
  <h3>PR {number}: {title}</h3>
  <dl>
    <dt>Status</dt><dd>{_escape(pr.get("status", "unknown"))}</dd>
    <dt>Branch</dt><dd>{_escape(pr.get("branch", "Not recorded"))}</dd>
    <dt>Base SHA</dt><dd>{_escape(pr.get("base_sha", "Not recorded"))}</dd>
    <dt>Head SHA</dt><dd>{_escape(head_sha)}</dd>
    <dt>Review</dt><dd>{_escape(pr.get("self_review", "not-run"))}</dd>
    <dt>Review gate</dt><dd>{_escape(pr.get("review_gate", "Not recorded"))}</dd>
  </dl>
  <p>{_escape(pr.get("summary", "No summary recorded."))}</p>
  <p><strong>Why:</strong> {_escape(pr.get("why", "No rationale recorded."))}</p>
  <h4>Dependencies</h4>{depends_on}
  <h4>Planned changes</h4>{changes}
  <h4>Files</h4>{files}
  <h4>Validation</h4>{validation}
  <h4>Performance</h4>{performance}
  <h4>Risks</h4>{risks}
  <pre>Planned interface:\n{_escape(pr.get("summary", "No planned interface recorded."))}</pre>
</article>"""


def _render_pr_button(pr: Mapping[str, Any]) -> str:
    pr_id = _escape(pr.get("id", "unknown-pr"))
    number = _escape(pr.get("number", "?"))
    title = _escape(pr.get("title", "Untitled PR"))
    pressed = "true" if pr.get("id") == "pr-01" else "false"
    return f"""
<button type="button" class="pr-button" data-pr-id="{pr_id}" aria-controls="{pr_id}" aria-pressed="{pressed}">
  PR {number}: {title}
</button>"""


def _render_problem(problem: Mapping[str, Any]) -> str:
    return f"""
<article class="problem-card">
  <h3>{_escape(problem.get("label", "Unlabeled concern"))}</h3>
  <p><strong>Status:</strong> {_escape(problem.get("status", "unknown"))}</p>
  <p>{_escape(problem.get("detail", "No detail recorded."))}</p>
  <p><strong>Next gate:</strong> {_escape(problem.get("next_gate", "No gate recorded."))}</p>
</article>"""


def _render_evidence(evidence: Mapping[str, Any]) -> str:
    status = evidence.get("status")
    if not isinstance(status, str) or status not in EVIDENCE_STATUS_LABELS:
        raise ValueError(f"unsupported evidence status: {status!r}")
    return f"""
<tr>
  <th scope="row">{_escape(evidence.get("label", "Unlabeled evidence"))}</th>
  <td>{EVIDENCE_STATUS_LABELS[status]}</td>
  <td>{_escape(evidence.get("detail", "No detail recorded."))}</td>
  <td>{_escape(evidence.get("provenance", "No provenance recorded."))}</td>
</tr>"""


def _render_quiz_question(question: Mapping[str, Any], number: int) -> str:
    options = question.get("options")
    answer = question.get("answer")
    if (
        not isinstance(options, list)
        or type(answer) is not int
        or not 0 <= answer < len(options)
    ):
        raise ValueError("quiz questions require options and an in-range answer")
    question_id = f"quiz-question-{number}"
    feedback_id = f"quiz-feedback-{number}"
    invariant_id = f"quiz-invariant-{number}"
    option_controls = "".join(
        f'''\n      <label><input type="radio" name="{question_id}" value="{index}"> {_escape(option)}</label>'''
        for index, option in enumerate(options)
    )
    return f"""
<li>
  <fieldset class="quiz-question" aria-describedby="{feedback_id} {invariant_id}" data-answer="{answer}">
    <legend>{_escape(question.get("question", "Question unavailable."))}</legend>
    <div class="quiz-options">{option_controls}
    </div>
    <p class="quiz-feedback" id="{feedback_id}" aria-live="polite"></p>
    <p class="quiz-invariant" id="{invariant_id}"><strong>Invariant:</strong> {_escape(question.get("explanation", "No explanation recorded."))}</p>
  </fieldset>
</li>"""


def render_html(context: Mapping[str, Any]) -> str:
    """Render the non-interactive, semantic dashboard body from validated context."""
    validate_context(context)
    integration = context.get("integration")
    integration_data = integration if isinstance(integration, Mapping) else {}
    prs = context["prs"]
    problems = context.get("problems")
    evidence = context.get("evidence")
    quiz = context.get("quiz")
    problem_cards = (
        "".join(
            _render_problem(problem) for problem in problems if isinstance(problem, Mapping)
        ).strip()
        if isinstance(problems, list)
        else ""
    )
    evidence_rows = (
        "".join(
            _render_evidence(item) for item in evidence if isinstance(item, Mapping)
        ).strip()
        if isinstance(evidence, list)
        else ""
    )
    evidence_caveat = ""
    if isinstance(evidence, list) and any(
        isinstance(item, Mapping) and item.get("status") == "measured" for item in evidence
    ):
        evidence_caveat = (
            '<p class="evidence-caveat">Measured results require a matched workload '
            'and the stated provenance; they are not directly comparable otherwise.</p>'
        )
    if not isinstance(quiz, list) or len(quiz) != 5 or not all(
        isinstance(question, Mapping) for question in quiz
    ):
        raise ValueError("context must contain exactly five quiz questions")
    quiz_questions = "".join(
        _render_quiz_question(question, number)
        for number, question in enumerate(quiz, start=1)
    ).strip()
    pr_buttons = "".join(
        _render_pr_button(pr) for pr in prs if isinstance(pr, Mapping)
    ).strip()
    pr_subpages = "".join(
        _render_pr_subpage(pr) for pr in prs if isinstance(pr, Mapping)
    ).strip()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(context.get("title", "Draft co-training PR review"))}</title>
  <style>
    body {{ box-sizing: border-box; width: 100%; font-family: system-ui, sans-serif; line-height: 1.5; margin: 0 auto; max-width: 72rem; padding: 2rem; }}
    section {{ margin-block: 3rem; }} article {{ border-block-start: 1px solid #bbb; padding-block: 1rem; }}
    .pr-selector {{ display: flex; flex-wrap: wrap; gap: .5rem; }} .pr-button[aria-pressed="true"] {{ font-weight: 700; }}
    dl {{ display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: .25rem 1rem; }} dt {{ font-weight: 700; }} dd {{ min-width: 0; overflow-wrap: anywhere; }}
    pre {{ overflow-x: auto; white-space: pre-wrap; }} .evidence-table-scroll {{ max-width: 100%; overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #bbb; padding: .5rem; text-align: left; vertical-align: top; }}
  </style>
</head>
<body>
  <header>
    <h1>{_escape(context.get("title", "Draft co-training PR review"))}</h1>
    <p>Updated {_escape(context.get("updated_at", "Not recorded"))} for {_escape(context.get("base_repo", "the base repository"))}.</p>
  </header>
  <nav aria-label="Table of contents">
    <ul>
      <li><a href="#background">Background</a></li>
      <li><a href="#intuition">Intuition</a></li>
      <li><a href="#code">Code</a></li>
      <li><a href="#problems">Problems</a></li>
      <li><a href="#evidence">Evidence</a></li>
      <li><a href="#quiz">Quiz</a></li>
    </ul>
  </nav>
  <main>
    <section id="background">
      <h2>Background</h2>
      <details>
        <summary>Beginner background</summary>
        <p>Target policy rollout produces trajectories; a draft policy proposes tokens; refit updates the draft from the latest policy behavior.</p>
      </details>
      <p>Version identifies the policy and draft checkpoint pair used for a rollout. Runtime means the vLLM environment and execution settings used to measure that pair.</p>
      <p><strong>Integration:</strong> {_escape(integration_data.get("summary", "No integration summary recorded."))}</p>
      <p><strong>Branch:</strong> {_escape(integration_data.get("branch", "Not recorded"))}; <strong>status:</strong> {_escape(integration_data.get("status", "unknown"))}.</p>
    </section>
    <section id="intuition">
      <h2>Intuition</h2>
      <p>For Qwen3-8B, a stale draft still predicts tokens from an earlier policy while RL has shifted the target distribution. Co-training shortens that gap: the policy rollout feeds refit, and the refreshed draft proposes against the next target-policy version.</p>
      <p>The goal is not a claimed speedup before measurement; it is a controlled comparison of stale and co-trained draft flows under the same runtime.</p>
    </section>
    <section id="code">
      <h2>Code and PR dependency overview</h2>
      <p>The planned sequence below is editorial review material. Each plan records only its stated interface, validation, and risk until a reviewed head SHA exists.</p>
      <nav class="pr-selector" aria-label="Draft co-training PR plans">
        {pr_buttons}
      </nav>
      <div class="pr-overview">
        {pr_subpages}
      </div>
    </section>
    <section id="problems">
      <h2>Problems and gates</h2>
      {problem_cards}
    </section>
    <section id="evidence">
      <h2>Evidence matrix</h2>
      <div class="evidence-table-scroll" tabindex="0" aria-label="Scrollable evidence table">
        <table>
          <thead><tr><th>Item</th><th>Status</th><th>Detail</th><th>Provenance</th></tr></thead>
          <tbody>{evidence_rows}</tbody>
        </table>
      </div>
{evidence_caveat}
    </section>
    <section id="quiz">
      <h2>Review quiz</h2>
      <ol class="quiz-preview">{quiz_questions}</ol>
    </section>
  </main>
  <script>
    (() => {{
      const articles = Array.from(document.querySelectorAll(".pr-subpage"));
      const buttons = Array.from(document.querySelectorAll(".pr-button"));
      const validIds = new Set(articles.map((article) => article.id));
      const defaultId = "pr-01";

      function idFromHash() {{
        const id = window.location.hash.slice(1);
        return validIds.has(id) ? id : defaultId;
      }}

      function selectPr(id, updateHash) {{
        const selectedId = validIds.has(id) ? id : defaultId;
        for (const article of articles) {{
          article.hidden = article.id !== selectedId;
        }}
        for (const button of buttons) {{
          button.setAttribute("aria-pressed", String(button.dataset.prId === selectedId));
        }}
        if (updateHash && window.location.hash !== `#${{selectedId}}`) {{
          history.replaceState(null, "", `#${{selectedId}}`);
        }}
      }}

      for (const button of buttons) {{
        button.addEventListener("click", () => selectPr(button.dataset.prId, true));
      }}
      window.addEventListener("hashchange", () => selectPr(idFromHash(), false));
      selectPr(idFromHash(), false);

      for (const question of document.querySelectorAll(".quiz-question")) {{
        const feedback = question.querySelector(".quiz-feedback");
        const invariant = question.querySelector(".quiz-invariant").textContent.trim();
        question.addEventListener("change", () => {{
          const selected = question.querySelector('input[type="radio"]:checked');
          if (!selected) {{
            return;
          }}
          const isCorrect = selected.value === question.dataset.answer;
          feedback.textContent = `${{isCorrect ? "Correct." : "Try again."}} ${{invariant}}`;
        }});
      }}
    }})();
  </script>
</body>
</html>
"""


def _write_atomically(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.chmod(temporary_name, 0o644)
        Path(temporary_name).replace(output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the draft co-training PR review dashboard."
    )
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    context = load_context(arguments.context)
    validate_context(context)
    _write_atomically(arguments.output, render_html(context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
