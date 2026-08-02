# SWE Rollout PR Draft Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static HTML page that distinguishes locally authored SWE rollout PR candidates from user-requested upstream cherry-picks and progressively validated future optimizations.

**Architecture:** Keep `docs/swe_rollout_pr_drafts.html` as the canonical static source and generate `public/reports/swe_rollout_pr_drafts.html` from it with only local-source links rewritten to public page anchors. Link the page from both existing SWE overhead reports. Do not modify the currently dirty report-index generator or generated index files in this change; the two SWE reports are the safe navigation entry points until those unrelated edits are resolved.

**Tech Stack:** Static HTML5/CSS, Python standard-library `html.parser` validation, Git.

## Global Constraints

- Label every entry as `Our implementation`, `User-requested cherry-pick`, `Already in latest main`, or `Related, not duplicate`.
- Never describe NeMo-RL #3390 or #3283 as locally authored.
- Mark historical vLLM 0.25.1 phase measurements as measured and the refreshed latest-main job-wall A/B as pending.
- Keep planned performance targets visibly labeled as projections.
- Do not include credentials, cluster secrets, private repository URLs, or user-specific environment values.
- Preserve unrelated dirty-worktree changes and stage only files owned by this plan.

---

### Task 1: Build the canonical PR draft page

**Files:**
- Create: `docs/swe_rollout_pr_drafts.html`
- Read: `docs/pr_drafts/01-gym-node-local-openhands-staging.md`
- Read: `docs/pr_drafts/02-nv-openhands-startup-timing.md`
- Read: `docs/pr_drafts/03-nv-openhands-immutable-workspace-cache.md`
- Read: `docs/pr_drafts/04-gym-workspace-cache-integration.md`
- Read: `docs/nemogym_init_framework_fixes.html`
- Read: `experiments/swe_rollout_latest_main_ab/attempts.md`

**Interfaces:**
- Consumes: the four English Markdown drafts and the measured/pending evidence labels from the overhead report.
- Produces: one standalone HTML5 document with section IDs `decision`, `pr-ready`, `upstream`, `progressive`, and `validation`.

- [ ] **Step 1: Verify the expected source drafts and evidence are present**

Run:

```bash
for source_file in \
  docs/pr_drafts/01-gym-node-local-openhands-staging.md \
  docs/pr_drafts/02-nv-openhands-startup-timing.md \
  docs/pr_drafts/03-nv-openhands-immutable-workspace-cache.md \
  docs/pr_drafts/04-gym-workspace-cache-integration.md \
  docs/nemogym_init_framework_fixes.html \
  experiments/swe_rollout_latest_main_ab/attempts.md; do
  test -s "$source_file" || exit 1
done
```

Expected: exit code 0.

- [ ] **Step 2: Create the static page**

Use `apply_patch` to create `docs/swe_rollout_pr_drafts.html` with:

- A decision banner stating that #3390/#3283 and node-local staging optimize different paths.
- A provenance legend containing all four exact labels from Global Constraints.
- Four English PR cards sourced from the Markdown drafts. Only node-local staging may be labeled `Draft PR now`; timing hardening is `Draft after focused tests`; both workspace-cache cards are `Planned`.
- The measured node-local numbers `15.0 → 11.1 s`, `21.8 → 16.4 s`, and `9.3 s/rollout`, plus the explicit pending latest-main setup-inclusive A/B statement.
- An upstream table for NeMo-RL #3390, #3283, #3000, #3292, #3409, and Gym #1669 with direct public GitHub links.
- A progressive queue covering prebuilt runtime artifact, private CoW workspace, pre-import forkserver, one-use action-server pool, trajectory payload compaction, and nv-OpenHands episode affinity.
- Quantitative gates for every progressive item and a staged validation pipeline `static/unit → n=1 → n=24 → conditional n=80`.
- Relative links to `nemogym_init_framework_fixes.html`, `nemogym_swe_efficiency_report.html`, `pr_drafts/*.md`, and `../experiments/swe_rollout_latest_main_ab/attempts.md`.

- [ ] **Step 3: Validate HTML structure and required claims**

Run:

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

path = Path("docs/swe_rollout_pr_drafts.html")
text = path.read_text(encoding="utf-8")

class Parser(HTMLParser):
    def error(self, message: str) -> None:
        raise AssertionError(message)

Parser().feed(text)
for section_id in ("decision", "pr-ready", "upstream", "progressive", "validation"):
    assert f'id="{section_id}"' in text, section_id
for label in (
    "Our implementation",
    "User-requested cherry-pick",
    "Already in latest main",
    "Related, not duplicate",
):
    assert label in text, label
for claim in ("15.0", "11.1", "21.8", "16.4", "9.3", "Pending measurement"):
    assert claim in text, claim
assert "#3390" in text and "#3283" in text
assert "user-requested" in text.lower()
PY
```

Expected: exit code 0.

- [ ] **Step 4: Check content safety and formatting**

Run:

```bash
git diff --check -- docs/swe_rollout_pr_drafts.html
if rg -n 'AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{16,}|api[_-]?key\s*[=:]' docs/swe_rollout_pr_drafts.html; then
  exit 1
fi
```

Expected: no whitespace errors and no secret-like matches.

### Task 2: Publish and link the page

**Files:**
- Create: `public/reports/swe_rollout_pr_drafts.html`
- Modify: `docs/nemogym_init_framework_fixes.html`
- Modify: `docs/nemogym_swe_efficiency_report.html`
- Modify: `public/reports/nemogym_init_framework_fixes.html`
- Modify: `public/reports/nemogym_swe_efficiency_report.html`

**Interfaces:**
- Consumes: `docs/swe_rollout_pr_drafts.html` from Task 1.
- Produces: a published mirror with public-safe links and working navigation from the canonical and published SWE reports.

- [ ] **Step 1: Generate the public mirror with public-safe links**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

source = Path("docs/swe_rollout_pr_drafts.html")
target = Path("public/reports/swe_rollout_pr_drafts.html")
text = source.read_text(encoding="utf-8")
replacements = {
    'href="pr_drafts/01-gym-node-local-openhands-staging.md"': 'href="#pr-node-local"',
    'href="pr_drafts/02-nv-openhands-startup-timing.md"': 'href="#pr-timing"',
    'href="pr_drafts/03-nv-openhands-immutable-workspace-cache.md"': 'href="#pr-workspace-consumer"',
    'href="pr_drafts/04-gym-workspace-cache-integration.md"': 'href="#pr-workspace-integration"',
    'href="../experiments/swe_rollout_latest_main_ab/attempts.md"': 'href="nemogym_init_framework_fixes.html"',
}
for old, new in replacements.items():
    assert old in text, old
    text = text.replace(old, new)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(text, encoding="utf-8")
PY
```

Expected: the public file exists, all four PR-card anchors remain present, and it contains no `pr_drafts/` or `../experiments/` links.

- [ ] **Step 2: Add a concise navigation link to both canonical reports**

Use `apply_patch` to add the following link near each report's introductory companion links:

```html
<a href="swe_rollout_pr_drafts.html">SWE rollout PR drafts and upstream overlap audit</a>
```

Do not change measured tables or evidence labels.

- [ ] **Step 3: Refresh the two published report copies**

Run:

```bash
cp docs/nemogym_init_framework_fixes.html public/reports/nemogym_init_framework_fixes.html
cp docs/nemogym_swe_efficiency_report.html public/reports/nemogym_swe_efficiency_report.html
```

Expected: each canonical report is byte-identical to its published mirror.

- [ ] **Step 4: Validate local links in all three published pages**

Run:

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.hrefs.append(value)

for page in (
    Path("public/reports/swe_rollout_pr_drafts.html"),
    Path("public/reports/nemogym_init_framework_fixes.html"),
    Path("public/reports/nemogym_swe_efficiency_report.html"),
):
    parser = Links()
    parser.feed(page.read_text(encoding="utf-8"))
    for href in parser.hrefs:
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or href.startswith("#"):
            continue
        target = (page.parent / parts.path).resolve()
        assert target.exists(), f"{page}: missing {href}"
PY
```

Expected: exit code 0.

- [ ] **Step 5: Review the owned diff and commit only owned files**

Run:

```bash
git diff --check -- \
  docs/swe_rollout_pr_drafts.html \
  docs/nemogym_init_framework_fixes.html \
  docs/nemogym_swe_efficiency_report.html \
  public/reports/swe_rollout_pr_drafts.html \
  public/reports/nemogym_init_framework_fixes.html \
  public/reports/nemogym_swe_efficiency_report.html
git diff --stat -- \
  docs/swe_rollout_pr_drafts.html \
  docs/nemogym_init_framework_fixes.html \
  docs/nemogym_swe_efficiency_report.html \
  public/reports/swe_rollout_pr_drafts.html \
  public/reports/nemogym_init_framework_fixes.html \
  public/reports/nemogym_swe_efficiency_report.html
git add -- \
  docs/swe_rollout_pr_drafts.html \
  docs/nemogym_init_framework_fixes.html \
  docs/nemogym_swe_efficiency_report.html \
  public/reports/swe_rollout_pr_drafts.html \
  public/reports/nemogym_init_framework_fixes.html \
  public/reports/nemogym_swe_efficiency_report.html
git commit -s -m "docs: publish SWE rollout PR draft page"
```

Expected: one commit authored by `seonjinn`; unrelated dirty files remain unstaged.
