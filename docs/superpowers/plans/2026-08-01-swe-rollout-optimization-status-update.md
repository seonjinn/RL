# SWE Rollout Optimization Status Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the verified latest-main SWE rollout optimization state and the one-variable experiment sequence on the canonical initialization/framework optimization page.

**Architecture:** Add one dated execution-status section to the canonical HTML report and apply the same markup to its public mirror. Append the bounded OCI-HSG observation to the existing attempts ledger without rewriting or staging its pre-existing uncommitted content. Keep the PR draft page, Pages landing page, CI, and broader efficiency report unchanged.

**Tech Stack:** Static HTML, Markdown, Python standard-library `HTMLParser`, Git, GitLab Pages.

## Global Constraints

- `docs/nemogym_init_framework_fixes.html` is the canonical human-readable status page.
- `public/reports/nemogym_init_framework_fixes.html` must be byte-identical to the canonical page.
- `experiments/swe_rollout_latest_main_ab/attempts.md` is append-only for this task.
- Preserve all pre-existing uncommitted changes in `attempts.md` and `public/index.html`.
- Do not modify `docs/swe_rollout_pr_drafts.html`, `public/index.html`, `.gitlab-ci.yml`, or `docs/nemogym_swe_efficiency_report.html`.
- Label job `5755875` as a Linux/GPU correctness gate, not a rollout performance result.
- Do not claim a refreshed latest-main node-local speedup.
- Use `apply_patch` for every working-tree edit.
- Commit authored work as `seonjinn <sna@nvidia.com>` with `git commit -s` and exact paths only.

---

### Task 1: Add the canonical execution-status section

**Files:**
- Modify: `docs/nemogym_init_framework_fixes.html:45-57`
- Modify: `public/reports/nemogym_init_framework_fixes.html:45-57`

**Interfaces:**
- Consumes: arm SHAs and job state recorded in `experiments/swe_rollout_latest_main_ab/attempts.md`.
- Produces: HTML section `id="current-execution-status"` with A/B/C/D state and the enforced experiment sequence.

- [ ] **Step 1: Run the pre-change assertion and verify the section is absent**

```bash
python3 - <<'PY'
from pathlib import Path

for source_file in (
    Path("docs/nemogym_init_framework_fixes.html"),
    Path("public/reports/nemogym_init_framework_fixes.html"),
):
    text = source_file.read_text(encoding="utf-8")
    assert 'id="current-execution-status"' in text, source_file
PY
```

Expected: FAIL with an `AssertionError` naming the canonical page.

- [ ] **Step 2: Insert the status section after the provenance table in both pages**

Use `apply_patch` to insert this exact semantic structure into both HTML files:

```html
<h2 id="current-execution-status">2. Current execution status — 2026-08-01</h2>
<div class="lesson"><b>Live state:</b> OCI-HSG has no active matching SWE rollout performance job. Job <code>5755875</code> completed the Linux/GPU correctness gate; it was not a rollout benchmark.</div>
<table>
<tr><th>Arm or stage</th><th>Source</th><th>Verified state</th><th>Next gate</th></tr>
<tr><td>A — latest main</td><td><code>1afc767c</code></td><td><span class="status inspected">Source prepared</span></td><td>n=1 rollout canary</td></tr>
<tr><td>B — A + NeMo-RL #3390</td><td><code>41374086</code></td><td><span class="status inspected">Source prepared</span></td><td>n=1 rollout canary</td></tr>
<tr><td>C — B + NeMo-RL #3283</td><td><code>6f8ca0b6</code></td><td><span class="status committed">Correctness gate passed</span><br>OCI-HSG job <code>5755875</code>: <code>COMPLETED</code>, exit <code>0:0</code>; 246 tests passed, real PIECEWISE/FULL CUDA Graph capture observed.</td><td>n=1 rollout canary</td></tr>
<tr><td>D — C + node-local OpenHands staging</td><td>Refreshed arm not frozen</td><td><span class="status handoff">Historical measured only</span></td><td>Port and unit-test the opt-in patch, then run n=1</td></tr>
<tr><td>Progressive candidates</td><td>One candidate per branch</td><td><span class="status projected">Not started; no active job</span></td><td>Begin only after the A/B/C/D gate</td></tr>
</table>
<div class="warn"><b>Evidence boundary:</b> job <code>5755875</code> loaded no SWE rollout dataset and produced no trajectory-duration, token-count, ReplayBuffer, throughput, or phase-timing result. The refreshed performance comparison remains pending.</div>
<h3>One-variable progression</h3>
<ol>
  <li>Complete the launcher, provenance manifest, result parser, and parser tests.</li>
  <li>Run A/B/C/D with one prompt, one generation, and concurrency one.</li>
  <li>Run matched n=24 ABBA comparisons only for correctness-passing arms.</li>
  <li>Use allocation-to-result wall including one-time setup, failures, and drain as the primary result; report concurrent phase sums separately.</li>
  <li>Promote a meaningful and correct result to matched n=80 reproduction.</li>
  <li>Test each progressive candidate through static/unit, n=1, n=24, and conditional n=80 without combining candidates.</li>
</ol>
```

Renumber the existing section headings from `2`–`7` to `3`–`8` in both pages. Do not change the body of those existing sections.

- [ ] **Step 3: Run the focused HTML content assertions**

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

class Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        element_id = dict(attrs).get("id")
        if element_id:
            self.ids.add(element_id)

for source_file in (
    Path("docs/nemogym_init_framework_fixes.html"),
    Path("public/reports/nemogym_init_framework_fixes.html"),
):
    text = source_file.read_text(encoding="utf-8")
    parser = Parser()
    parser.feed(text)
    parser.close()
    assert "current-execution-status" in parser.ids
    for phrase in (
        "A — latest main",
        "B — A + NeMo-RL #3390",
        "C — B + NeMo-RL #3283",
        "D — C + node-local OpenHands staging",
        "5755875",
        "not a rollout benchmark",
        "n=24 ABBA",
        "conditional n=80",
    ):
        assert phrase in text, (source_file, phrase)
PY
```

Expected: PASS with exit code 0.

- [ ] **Step 4: Verify and commit the two clean HTML files**

```bash
cmp docs/nemogym_init_framework_fixes.html \
  public/reports/nemogym_init_framework_fixes.html
git diff --check -- \
  docs/nemogym_init_framework_fixes.html \
  public/reports/nemogym_init_framework_fixes.html
git add \
  docs/nemogym_init_framework_fixes.html \
  public/reports/nemogym_init_framework_fixes.html
git commit -s -m "docs: record SWE rollout optimization status"
```

Expected: `cmp` and `git diff --check` return 0; the commit contains exactly two files.

### Task 2: Append the live observation to the attempts ledger

**Files:**
- Modify: `experiments/swe_rollout_latest_main_ab/attempts.md`

**Interfaces:**
- Consumes: OCI-HSG `squeue` and `sacct` observations made on 2026-08-01.
- Produces: a dated append-only operational record distinguishing correctness from rollout performance.

- [ ] **Step 1: Capture the pre-existing ownership boundary**

```bash
git status --short -- experiments/swe_rollout_latest_main_ab/attempts.md
git diff --numstat -- experiments/swe_rollout_latest_main_ab/attempts.md
tail -n 20 experiments/swe_rollout_latest_main_ab/attempts.md
```

Expected: the file is already modified before this task. Do not reset, restore, or stage that existing diff.

- [ ] **Step 2: Append the dated observation with `apply_patch`**

Append this section after the current final line:

```markdown
## 2026-08-01 — live execution state before rollout canaries

A bounded OCI-HSG check found no active job whose name matched `swe-ab`, SWE
rollout, or NeMoGym performance work. Accounting confirms Linux/GPU gate job
`5755875` completed with state `COMPLETED`, exit `0:0`, elapsed `00:27:24`,
start `2026-07-31T22:43:30`, and end `2026-07-31T23:10:54`.

This was a source, dependency, Ray, vLLM 0.25.1, and real CUDA Graph correctness
gate. It loaded no SWE rollout dataset and emitted no trajectory-duration,
token-count, ReplayBuffer, generation-throughput, or OpenHands phase result.
The n=1 A/B/C/D rollout canaries and matched n=24 performance comparison have
not been submitted. The next required artifact is the reproducible launcher,
provenance manifest, result parser, and parser tests.
```

- [ ] **Step 3: Verify the append and preservation markers**

```bash
python3 - <<'PY'
from pathlib import Path

text = Path(
    "experiments/swe_rollout_latest_main_ab/attempts.md"
).read_text(encoding="utf-8")
for phrase in (
    "2026-07-31 — baseline refresh before canaries",
    "2026-07-31 — latest-arm Linux gate passed",
    "2026-08-01 — live execution state before rollout canaries",
    "00:27:24",
    "have not been submitted",
):
    assert phrase in text, phrase
assert text.count("2026-08-01 — live execution state before rollout canaries") == 1
PY
git diff --check -- experiments/swe_rollout_latest_main_ab/attempts.md
```

Expected: PASS with the earlier uncommitted 2026-07-31 sections unchanged and one new 2026-08-01 section.

- [ ] **Step 4: Preserve the dirty-file boundary**

Do not stage or commit `attempts.md` in this task because it contained
pre-existing uncommitted content. Report the appended section as a working-tree
update and leave ownership of the combined diff with the user.

### Task 3: Validate scope, links, mirrors, and content safety

**Files:**
- Test: `docs/nemogym_init_framework_fixes.html`
- Test: `public/reports/nemogym_init_framework_fixes.html`
- Test: `experiments/swe_rollout_latest_main_ab/attempts.md`

**Interfaces:**
- Consumes: Task 1 HTML and Task 2 appended Markdown.
- Produces: verification evidence that the status is publishable and no unrelated dirty file was changed by this implementation.

- [ ] **Step 1: Parse HTML and resolve local links**

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

class Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)

for source_file in (
    Path("docs/nemogym_init_framework_fixes.html"),
    Path("public/reports/nemogym_init_framework_fixes.html"),
):
    parser = Parser()
    parser.feed(source_file.read_text(encoding="utf-8"))
    parser.close()
    for href in parser.hrefs:
        parsed = urlparse(href)
        if parsed.scheme or href.startswith("#") or not parsed.path:
            continue
        target = (source_file.parent / parsed.path).resolve()
        assert target.exists(), (source_file, href)
PY
```

Expected: PASS with exit code 0.

- [ ] **Step 2: Verify mirror identity and scan for sensitive content**

```bash
cmp docs/nemogym_init_framework_fixes.html \
  public/reports/nemogym_init_framework_fixes.html
if rg -n \
  '(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|/Users/sna|sna@nvidia\.com)' \
  docs/nemogym_init_framework_fixes.html \
  public/reports/nemogym_init_framework_fixes.html \
  experiments/swe_rollout_latest_main_ab/attempts.md
then
  exit 1
fi
```

Expected: `cmp` returns 0 and the scan emits no match.

- [ ] **Step 3: Verify the implementation scope and commit authorship**

```bash
git show --name-only --format='%h %an <%ae> %s' HEAD
git status --short -- \
  docs/nemogym_init_framework_fixes.html \
  public/reports/nemogym_init_framework_fixes.html \
  experiments/swe_rollout_latest_main_ab/attempts.md \
  docs/swe_rollout_pr_drafts.html \
  public/index.html \
  .gitlab-ci.yml \
  docs/nemogym_swe_efficiency_report.html
```

Expected: the implementation commit is authored by `seonjinn`, contains only
the two optimization HTML pages, and `attempts.md` remains modified but
unstaged. The PR draft, Pages index, CI, and broader efficiency report have no
new implementation diff.
