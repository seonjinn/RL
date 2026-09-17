# Qwen3-235B SpecDec deep-refit validation

This experiment validates the opt-in `specdec_deep_refit` memory lifecycle
against the unchanged no-SpecDec `legacy_level1` baseline. Both gates use the
official Qwen3-235B 16-node × 4-GPU performance recipe for three GRPO steps.

## Matched gate

| Arm | Drafter lifecycle | SpecDec | Steps | Ray threshold |
|---|---|---:|---:|---:|
| Baseline | `legacy_level1` | Off | 3 | 0.95 |
| DFlash K7 B8 | `specdec_deep_refit` | Frozen snapshot/restore | 3 | 0.95 |

The DFlash gate must complete repeated level-2 sleep, target refit, static
drafter restore, and KV-cache wake cycles without OOM or a Ray memory kill.
The baseline must retain level-1 sleep and execute no drafter RPC.

## Submission

Run from the repository root after committing and pushing all changes:

```bash
export Q235_SOURCE=/home/sna/nemorl-q235-deep-refit-20260917
python3 experiments/q235_specdec_deep_refit_20260917/scripts/submit_ptyche.py \
  --site ptyche --account coreai_dlalgo_llm --render
python3 experiments/q235_specdec_deep_refit_20260917/scripts/submit_ptyche.py \
  --site ptyche --account coreai_dlalgo_llm --submit
```

Each run records the source SHA, immutable container/target/drafter paths,
complete override map, generated batch script, test-only receipt, submission
receipt, logs, and W&B run name under:

`/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q235-specdec-deep-refit-20260917`

## Acceptance gates

- three completed GRPO steps and at least two refit cycles;
- nonzero accepted tokens after every drafter restore;
- finite reward, generated length, approximate entropy, policy KL, and
  generation KL metrics;
- all six `prepare_for_generation/*` phase timers present;
- no monotonic host-memory growth across repeated cycles;
- no OOM, Ray memory kill, refit timeout, or generation hang.

Results and W&B URLs are recorded in `results/receipt.md` after completion.
