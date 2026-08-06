# CUDA Graph Implementation Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and open a self-contained HTML explanation of the packed-THD Transformer Engine CUDA Graph implementation, and package the workflow as a reusable personal Codex skill.

**Architecture:** A standard-library Python renderer combines versioned editorial JSON with the existing performance, telemetry, and correctness CSVs. The generated page remains separate from the detailed experiment ledger but links to it. A personal `explain-diff-html` skill stores the reusable workflow, page specification, and a deterministic HTML validator.

**Tech Stack:** Python 3 standard library, HTML5, CSS, vanilla JavaScript, Codex skill metadata.

## Global Constraints

- Keep the result a single self-contained HTML file with embedded CSS and JavaScript.
- Use one long page with a table of contents; do not use top-level tabs.
- Include Background, Intuition, Code, Current problems, Evidence, and five interactive quiz questions.
- Use `<pre>` for code and preserve whitespace with `white-space: pre` or `pre-wrap`.
- Derive performance and CUDA Graph telemetry from the canonical CSVs instead of copying measured values into HTML.
- Keep the canonical HTML under the experiment directory so later changes remain reviewable.
- Install the personal skill at `~/.codex/skills/explain-diff-html`.

---

### Task 1: Tested explainer data model and renderer

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/test_render_explainer.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_explainer.py`

**Interfaces:**
- Consumes: editorial JSON and three CSV paths.
- Produces: `load_evidence(performance_path, telemetry_path, correctness_path) -> list[ScopeEvidence]`, `render_html(context, evidence) -> str`, and `write_html(document, output_path) -> None`.

- [ ] **Step 1: Write the failing evidence test**

```python
def test_load_evidence_derives_speedup_hit_rate_and_coverage(self) -> None:
    evidence = load_evidence(self.performance, self.telemetry, self.correctness)
    attention = next(row for row in evidence if row.scope == "attn")
    self.assertAlmostEqual(attention.e2e_speedup_pct, 20.0)
    self.assertAlmostEqual(attention.cache_hit_pct, 75.0)
    self.assertEqual(attention.graph_calls, attention.eligible_calls)
```

- [ ] **Step 2: Run the test and verify the missing renderer failure**

Run: `python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/test_render_explainer.py`

Expected: FAIL because `render_explainer.py` does not exist.

- [ ] **Step 3: Implement typed CSV loading and derived metrics**

```python
@dataclass(frozen=True)
class ScopeEvidence:
    scope: str
    job_id: str
    e2e_tps: float
    e2e_speedup_pct: float | None
    graph_calls: int
    eligible_calls: int
    cache_hit_pct: float | None
    fallback_count: int
```

Join rows by `Exp`, use baseline throughput as the denominator, reject duplicate or missing scopes, and reject `graph_calls > eligible_calls`.

- [ ] **Step 4: Add failing renderer behavior tests**

Assert that rendered HTML contains all required section IDs, exactly five quiz fieldsets, escaped editorial text, a link to `report.html`, and a `<pre>` rule with preserved whitespace. Add a malformed-telemetry test that expects `ValueError`.

- [ ] **Step 5: Implement the minimal self-contained renderer**

Render semantic HTML, embedded responsive CSS, CSS-only architecture and packed-batch diagrams, accessible quiz fieldsets, and vanilla JavaScript feedback. Escape every editorial string, source path, excerpt, and URL.

- [ ] **Step 6: Run the focused tests**

Run: `python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/test_render_explainer.py`

Expected: all tests pass with no warnings.

### Task 2: Canonical CUDA Graph explanation and generated page

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/explainer_context.json`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/cudagraph_implementation_explainer.html`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py`

**Interfaces:**
- Consumes: grouped source explanations and the completed steps 11-19 evidence files.
- Produces: a browser-ready explanation and mutual navigation with the experiment ledger.

- [ ] **Step 1: Author versioned context**

Record the NeMo-RL to Megatron-Core to Transformer Engine flow, fixed packed-THD geometry, warmup and lifecycle, persistent bank, storage fingerprints, scope support, known issues, source links, current 100-step status, and five quiz questions.

- [ ] **Step 2: Render the canonical HTML**

Run:

```bash
python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_explainer.py
```

Expected: JSON output names the generated HTML and reports four evidence rows and five quiz questions.

- [ ] **Step 3: Add durable navigation and update instructions**

Add the explainer link to the generated ledger header and document the single regeneration command in the experiment README.

- [ ] **Step 4: Regenerate both pages**

Run:

```bash
python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py
python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_explainer.py
```

Expected: both HTML files contain reciprocal relative links.

### Task 3: Personal explain-diff HTML skill

**Files:**
- Create: `~/.codex/skills/explain-diff-html/SKILL.md`
- Create: `~/.codex/skills/explain-diff-html/agents/openai.yaml`
- Create: `~/.codex/skills/explain-diff-html/references/page-spec.md`
- Create: `~/.codex/skills/explain-diff-html/scripts/validate_explainer.py`

**Interfaces:**
- Consumes: a code diff, branch, commit, PR, or implementation area and an output-location preference.
- Produces: a researched, self-contained, dated HTML explanation and validation result.

- [ ] **Step 1: Initialize the personal skill**

Run `init_skill.py explain-diff-html` with `scripts,references` resources and UI metadata for Explain Diff HTML.

- [ ] **Step 2: Write the concise skill workflow and page reference**

Require surrounding-code exploration, beginner and narrow background, toy-data intuition, grouped code walkthrough, explicit current risks, measured evidence when available, and five interactive questions. Default to a dated `/tmp` output unless the user requests a versioned project artifact.

- [ ] **Step 3: Add deterministic HTML validation**

The validator parses the real output and fails when a required section is absent, the quiz count is not five, a top-level tablist is present, or code whitespace is not preserved.

- [ ] **Step 4: Validate the skill and its first real output**

Run:

```bash
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py ~/.codex/skills/explain-diff-html
python3 ~/.codex/skills/explain-diff-html/scripts/validate_explainer.py experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/cudagraph_implementation_explainer.html
```

Expected: both commands exit zero.

### Task 4: Final verification, browser preview, and source control

**Files:**
- Verify all files from Tasks 1-3.

**Interfaces:**
- Consumes: completed implementation and generated artifacts.
- Produces: verified local page, committed repository history, and pushed branch.

- [ ] **Step 1: Run focused tests and syntax checks**

Run:

```bash
python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/test_render_explainer.py
python3 -m py_compile experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_explainer.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 2: Parse and validate the generated page**

Run the personal skill validator against the canonical HTML and confirm the output is a regular non-empty file.

- [ ] **Step 3: Open the canonical page**

Run `open experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/cudagraph_implementation_explainer.html`.

- [ ] **Step 4: Commit and push only the reviewed repository files**

Use a signed-off documentation commit and push `experiment/thd-cg-hybrid-nemotron-latest-main-20260804` to the `seonjinn` remote.
