# CUDA Graph Implementation Explainer Design

## Goal

Create a self-contained HTML page that explains the NeMo-RL, Megatron-Core,
and Transformer Engine changes behind packed-THD partial CUDA Graph support,
shows the measured results, and keeps the remaining correctness and performance
risks visible.

## Audience and format

The page targets engineers who know model training but have not followed this
branch. It follows the `explain-diff-html` structure: background, intuition,
grouped code walkthrough, current problems, measured evidence, and a short
interactive quiz. It is one responsive page with a table of contents and no
top-level tabs.

## Architecture

Keep the existing experiment report as the detailed run ledger. Add a separate
explainer generated from versioned editorial context and the existing measured
CSV artifacts:

- `explainer_context.json` owns explanations, code-change groups, known issues,
  links, and quiz content.
- `render_explainer.py` validates that context, reads the persistent-bank
  performance, telemetry, and correctness CSV files, and emits one standalone
  HTML file with embedded CSS and JavaScript.
- `results/cudagraph_implementation_explainer.html` is the browser-ready output.
- The explainer and existing `results/report.html` link to one another so the
  conceptual narrative and full experiment ledger remain distinct.

The generator uses only the Python standard library. Relative links remain
valid when the experiment directory is copied or published.

## Page content

### Background

Explain the execution chain from NeMo-RL policy calls through Megatron-Core to
Transformer Engine partial graphs. Distinguish the eager path, TE policy-
training graph replay, and vLLM generation graphs. Explain packed THD metadata,
fixed token and sequence capacities, and why replay requires stable addresses
and geometry.

### Intuition

Use CSS diagrams with a small packed batch to show:

1. logical sequences packed into fixed-capacity tensors;
2. three successful optimizer warmup steps;
3. capture on a schedule miss and replay on a hit;
4. persistent graph banks surviving the intervening logprob phase;
5. LRU eviction when more schedules appear than the bank can retain.

### Code walkthrough

Group the branch changes by responsibility rather than commit order:

1. configuration normalization and fixed-geometry validation;
2. THD packing, padding, position IDs, labels, and ownership metadata;
3. worker lifecycle, collective preflight, graph capture, replay, and teardown;
4. persistent storage fingerprints and pointer-stability validation;
5. Megatron-Core and Transformer Engine scope support for attention, Mamba,
   MoE router, and MoE preprocess;
6. coverage, fallback, cache, token-utilization, and correctness telemetry.

Each group contains a plain-language purpose, a short code excerpt, and links
to the relevant source files.

### Measured evidence

Render steps 11-19 of the completed 20-step Nano runs. Show E2E,
policy-training, generation, and logprob throughput, graph-call coverage,
cache hit rate, captures, evictions, and fallbacks. Warn that differing
tokens-per-sample distributions make raw step time a confounded comparison.

### Current problems

Show confirmed behavior separately from unresolved conclusions:

- Wider scopes have 100% eligible-call coverage and zero fallback in the
  measured smoke runs, but the capacity-2 LRU bank thrashes when four schedule
  shapes alternate. Hit rate falls from 100% for attention to 55.6% for
  attention plus Mamba and 33.3% after adding the router.
- Increasing capacity may avoid recapture but retains more graph and static
  storage memory. Capacity four and deterministic capacity bucketing require a
  measured memory/performance comparison.
- The 20-step runs show no NaN/Inf or masked sequences, but they are independent
  stochastic trajectories. They do not establish convergence parity. The
  matched 100-step baseline-versus-attention soak is the next evidence gate.
- Logprob timing can benefit indirectly from persistent model and storage state,
  but it must not be presented as TE graph replay unless phase-specific graph
  telemetry confirms it.
- Mamba scopes apply only to Nemotron hybrid models. MoE preprocess depends on
  router-compatible metadata, padding, auxiliary-loss ownership, and stable THD
  geometry.
- The feature branch must be audited against the latest NeMo-RL and nested
  Megatron revisions before upstreaming because upstream main has advanced.

### Quiz

Include five keyboard-accessible multiple-choice questions with immediate
feedback. The questions cover fixed geometry, warmup, persistent banks, hit
rate versus coverage, and the limits of the current correctness evidence.

## Update workflow

When code or experiment status changes, update `explainer_context.json` and the
canonical CSV artifacts, run `render_explainer.py`, validate the output, and
commit the context, data, and regenerated HTML together. This preserves a
reviewable history and prevents the page from drifting away from measured data.

## Error handling

The generator fails with a clear message when required context keys or CSV
columns are missing, when numeric fields cannot be parsed, or when graph calls
exceed eligible calls. Optional links and code excerpts may be omitted without
breaking the page. HTML and attribute values are escaped before rendering.

## Verification

Focused tests exercise the real renderer with temporary input files. They
verify derived hit rates and speedups, HTML escaping, required sections,
interactive quiz markup, and explicit failure for malformed measured data.
After tests pass, render the canonical page and parse it with the standard
library HTML parser. Open that canonical file in the local browser.
