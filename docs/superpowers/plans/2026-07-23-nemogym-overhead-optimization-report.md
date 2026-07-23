# NeMo-Gym SWE Overhead Optimization Report Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct the SWE rollout overhead reports and add code-grounded optimization recommendations, explicit evidence labels, validation gates, and a complete failed-attempt ledger.

**Architecture:** Keep `nemogym_init_framework_fixes.html` as the canonical technical analysis and `nemogym_swe_efficiency_report.html` as the broad summary. Both pages use the committed n=24 CSV as the measured source, distinguish handoff-only n=80 claims, and present projected ceilings as unmeasured cost models.

**Tech Stack:** Static HTML/CSS, Python standard-library HTML parser, ripgrep, Git.

---

### Task 1: Correct and expand the canonical initialization/framework page

**Files:**
- Modify: `docs/nemogym_init_framework_fixes.html`
- Reference: `docs/superpowers/specs/2026-07-23-nemogym-overhead-optimization-report-design.md`
- Reference: `experiments/dflash_loss_ab/report/data/patch_ab_n24.csv`

**Step 1: Record the stale assertions that must be removed**

Run:

```bash
rg -n "statistically meaningful|16 s/rollout|break-even.*14|145.*70|git.*problem" docs/nemogym_init_framework_fixes.html
```

Expected: the existing page exposes stale or overconfident wording that the update must supersede.

**Step 2: Add the corrected evidence and observed-wall ledger**

Edit the page so it states:

- n=24 rollout wall excluding mirror preparation: 3,059 to 2,968 seconds, -91 seconds or -2.97%.
- patched wall including the 216-second mirror: 3,184 seconds, +125 seconds or +4.09%.
- stable connect plus framework savings: 9.3 seconds per rollout.
- modeled break-even: 216 / 9.3 = 23.2, approximately 24 rollouts.
- n=80 result: handoff-reported only, with raw artifact and job IDs pending.

Label every result as Measured, Handoff-reported, or Projected.

**Step 3: Document the code-grounded root causes**

Add the exact workspace setup fallback:

```bash
if ! cp -al /testbed /workspace/$WORKSPACE_NAME 2>/dev/null; then
    cp -r /testbed /workspace/$WORKSPACE_NAME
fi
```

Explain why the read-only squashfs-to-writable-filesystem hard link fails and why the full copy dominates initialization. Document the one-shot `run_infer.py` import chain and the fresh action-server process started by `LocalRuntime`.

**Step 4: Add ranked optimization candidates and acceptance gates**

Document:

1. Versioned prebuilt OpenHands squashfs.
2. Immutable per-instance node-local cache plus private reflink workspace.
3. Instrumented pre-import forkservers.
4. Persistent controller with one-use prewarmed action servers.
5. Isolated small candidates such as direct venv Python and checked-hash bytecode.

For every candidate include mechanism, expected benefit, risk, and a quantitative pass/fail gate.

**Step 5: Add the complete failed-attempt ledger**

Include `NRL_SKIP_GIT_RESET`, merged init commands, faster polling, `/tmp` event store, invalid direct `/workspace` bind caching, rejected writable hard links, and deferred used-server reuse. Distinguish a null result from an invalid experiment.

**Step 6: Correct the remaining ceiling**

Use the patched 123.6-second phase total:

- realistic identified-lever projection: 78.6 seconds, -36.4%, 1.57x.
- zero-overhead absolute bound: 74.3 seconds, -39.9%, 1.66x.

State explicitly that neither is measured.

### Task 2: Correct the broader SWE efficiency report

**Files:**
- Modify: `docs/nemogym_swe_efficiency_report.html`
- Reference: `docs/nemogym_init_framework_fixes.html`

**Step 1: Replace obsolete diagnosis and projections**

Replace the git-cleanup diagnosis with the measured workspace-copy diagnosis. Remove the obsolete 145-to-70-second, approximately 2x projection.

**Step 2: Add a compact evidence ledger**

Summarize measured n=24 full wall, mirror-inclusive result, phase-normalized model, handoff-only n=80 claim, and projected 78.6/74.3-second bounds.

**Step 3: Replace the optimization roadmap**

Use the same ranked candidates and status labels as the canonical page, with a prominent link to the detailed code and validation analysis.

**Step 4: Add a compact 시행착오 summary**

Ensure the overview states which trials were null, invalid, rejected by design, or deferred.

### Task 3: Validate both pages and commit the implementation

**Files:**
- Verify: `docs/nemogym_init_framework_fixes.html`
- Verify: `docs/nemogym_swe_efficiency_report.html`

**Step 1: Parse both HTML files**

Run:

```bash
python - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

class StrictEnoughParser(HTMLParser):
    pass

for name in (
    "docs/nemogym_init_framework_fixes.html",
    "docs/nemogym_swe_efficiency_report.html",
):
    StrictEnoughParser().feed(Path(name).read_text())
    print(f"parsed: {name}")
PY
```

Expected: both files parse without exceptions.

**Step 2: Verify required claims and retired wording**

Run:

```bash
rg -n "3,059|2,968|3,184|23\\.2|Handoff-reported|78\\.6|74\\.3|reflink|forkserver|invalid experiment" docs/nemogym_init_framework_fixes.html docs/nemogym_swe_efficiency_report.html
! rg -n "statistically meaningful|145[^<]*70|≈2×|break-even.*14" docs/nemogym_init_framework_fixes.html docs/nemogym_swe_efficiency_report.html
```

Expected: all required evidence and candidate terms are present; stale claims are absent.

**Step 3: Validate local document links and repository whitespace**

Run:

```bash
python - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "img"}:
            return
        key = "href" if tag == "a" else "src"
        value = dict(attrs).get(key)
        if value and not value.startswith(("http:", "https:", "#", "mailto:")):
            self.refs.append(value.split("#", 1)[0])

for html in (
    Path("docs/nemogym_init_framework_fixes.html"),
    Path("docs/nemogym_swe_efficiency_report.html"),
):
    parser = Links()
    parser.feed(html.read_text())
    for ref in parser.refs:
        target = (html.parent / ref).resolve()
        assert target.exists(), f"broken link: {html}: {ref}"
    print(f"links ok: {html}")
PY
git diff --check
```

Expected: all local links resolve and Git reports no whitespace errors.

**Step 4: Review the final diff**

Run:

```bash
git diff -- docs/nemogym_init_framework_fixes.html docs/nemogym_swe_efficiency_report.html
```

Expected: only evidence-backed report changes appear; no runtime or experiment code changed.

**Step 5: Commit the report update**

Run:

```bash
git add docs/nemogym_init_framework_fixes.html docs/nemogym_swe_efficiency_report.html
git commit -s -m "docs: correct NeMo-Gym overhead analysis"
```

Expected: a signed-off documentation commit containing only the two HTML reports.
