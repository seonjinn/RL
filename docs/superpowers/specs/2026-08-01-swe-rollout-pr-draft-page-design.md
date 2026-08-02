# SWE rollout PR draft page design

## Goal

Create one canonical HTML page that separates immediately actionable SWE rollout PR drafts from upstream optimizations and unvalidated follow-up ideas. The page must make authorship, evidence quality, readiness, and the next validation gate unambiguous.

## Outputs

- `docs/swe_rollout_pr_drafts.html`: canonical working page.
- `public/reports/swe_rollout_pr_drafts.html`: published copy generated from the canonical page.
- Links from the existing SWE overhead reports and report index.
- Existing English Markdown drafts remain the editable source for individual PR descriptions.

## Classification

Every item receives exactly one provenance label:

1. `Our implementation`: code or a design authored in this workstream.
2. `User-requested cherry-pick`: an upstream PR selected by the user and integrated and validated locally.
3. `Already in latest main`: an upstream optimization inherited through the frozen main revision.
4. `Related, not duplicate`: an upstream PR with a similar mechanism in a different harness or code path.

The page must not describe a cherry-picked upstream change as our authored PR. In particular, NeMo-RL #3390 and #3283 are user-requested cherry-picks, not local PR candidates.

## Page structure

### 1. Decision summary

Explain that the existing vLLM 0.25.1 rollout-only measurements remain valid historical evidence, while the refreshed latest-main setup-inclusive A/B remains pending. State that upstream per-turn optimizations and the node-local OpenHands startup optimization address different code paths.

### 2. PR-ready queue

Show each proposed PR as a compact English draft with repository, ownership, readiness, root cause, implementation, performance evidence, validation, and risk.

- Gym: node-local OpenHands runtime staging. Code and compatibility tests exist. It is ready for a draft PR; the latest-main allocation-to-result A/B is required before marking it ready for review or making a current-main speedup claim.
- nv-OpenHands: startup metric writer hardening. This is a small observability and robustness PR with no intended speedup. It is ready only after its focused tests and overhead limit are recorded.
- nv-OpenHands: immutable workspace-cache consumer. This remains a planned PR until private-workspace isolation and filesystem capability tests pass.
- Gym: immutable workspace-cache producer and mount integration. This remains a planned integration PR and must follow the nv-OpenHands consumer contract.

### 3. Upstream adoption ledger

Record original PR, original author, local integration status, and validation status for:

- NeMo-RL #3390: remove the awaited per-turn `/tokenize` round trip.
- NeMo-RL #3283: give the tokenizer to the Gym actor at construction.
- NeMo-RL #3000: prompt-group streaming, already in main.
- NeMo-RL #3292: routed-expert payload serialization improvement, already in main.
- NeMo-RL #3409: per-node Gym venv prefetch, already in the frozen latest-main lineage.
- Gym #1669: per-episode HTTP-client affinity for Harbor; related to, but not a replacement for, nv-OpenHands affinity.

### 4. Progressive optimization queue

List follow-up work in this order:

1. Replace the 5.3 GB per-job runtime mirror with a baked container layer or versioned squashfs.
2. Create private reflink or OverlayFS workspaces from an immutable instance cache.
3. Test a pre-imported, pre-thread Python forkserver for `run_infer.py`.
4. Test one-use prewarmed action servers owned by a long-lived controller.
5. Compact duplicated token/logprob fields before trajectory persistence while preserving the OpenAI response contract.
6. Add stable episode-to-engine affinity in the nv-OpenHands model path.

Each item must show its mechanism, expected phase, risk, and quantitative acceptance gate. Projected values must be labeled as projections.

## Validation workflow

Later optimizations are tested independently so they do not delay the first draft PR:

1. Static and unit tests.
2. One-rollout correctness canary.
3. Matched n=24 rollout-only comparison.
4. n=80 comparison only when setup amortization or scale behavior is material.
5. Promote the candidate to the PR-ready queue only after reward, valid-rate, generated-patch, workspace-isolation, and failure-class parity pass.

The performance comparison uses allocation-to-result wall time including setup and drain. Phase sums and summed concurrent rollout durations are reported separately. Effects from #3390, #3283, and local startup patches are not added arithmetically across unmatched experiments.

## Presentation rules

- Use a small status legend and tables; do not require JavaScript.
- Keep PR prose in concise, plain English suitable for copying into GitHub.
- Link every upstream item to its original PR.
- Link local evidence to the existing overhead report and latest-main attempts log.
- Show measured, handoff-reported, code-inspected, and projected evidence with distinct labels.
- Do not include credentials, cluster secrets, private repository URLs, or user-specific environment values.

## Acceptance criteria

- A reviewer can identify which PRs we authored and which PRs were cherry-picked without reading commit history.
- No planned optimization is represented as measured or PR-ready.
- The node-local draft includes the measured vLLM 0.25.1 phase result and explicitly states that refreshed latest-main job-wall A/B is pending.
- Every future optimization has a stop condition and an isolation or correctness gate.
- Existing SWE overhead pages provide a working link to the new page.
