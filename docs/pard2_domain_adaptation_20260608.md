# PARD-2-Style Domain Adaptation Notes

Date: 2026-06-08

## Short Answer

Yes, when the public PARD drafter is mismatched to the evaluation or GRPO
rollout domain, a PARD-2-style target-aligned objective is the right direction
to improve acceptance rate.

This should be treated as domain adaptation for the drafter, not as a runtime
fix. It can raise acceptance only if the drafter learns continuations that the
target model would actually accept on the target prompt distribution.

## Why It Applies Here

The current Qwen3-235B evidence shows a clear prompt-domain split:

| Regime | Public PARD result |
| --- | --- |
| Short synthetic `ISL=1000/OSL=512`, K12 | about `3.29x`, about `93%` acceptance |
| OpenMath `ISL=1024/OSL=1024`, K5 | about `1.31x`, about `45.5%` acceptance |
| OpenMath `ISL=1024/OSL=1024`, K12 | about `1.22x`, about `22.8%` acceptance |

This is consistent with the public drafter being good on easy/generic short
continuations but much weaker on math reasoning continuations and later draft
positions.

## Runnable Local Approximation

The current runnable path is not the official unreleased PARD-2 implementation.
It is:

- AMD PARD parallel drafting runtime.
- Qwen3-235B target continuations and/or target logprobs on math prompts.
- CAT / D-PACE-style loss weighting that emphasizes tokens and prefixes more
  likely to be accepted by the target model.

Best current local OpenMath gate:

| Checkpoint | K | Batch | Acceptance | Speedup vs OpenMath baseline |
| --- | ---: | ---: | ---: | ---: |
| `local_pard_k5_dpace_draft_ce_2048_gate` / job `3190567` | 5 | 32 | `47.01%` | `1.296x` |
| same checkpoint / job `3193047` | 3 | 32 | `61.55%` | `1.207x` |

So the idea is viable, but the current 2K/4K local runs have not yet produced
a decisive K5 acceptance jump to the target band of roughly `60%+`.

## Practical Rule

Use PARD-2-style adaptation when:

- The same drafter has high acceptance on synthetic/easy prompts but low
  acceptance on real target-domain prompts.
- Per-position acceptance drops sharply in later speculative positions.
- Baseline runtime still has enough generation fraction that a better drafter
  can move E2E step time.

Do not rely on it when:

- The bottleneck is NCCL timeout, host-memory pressure, or vLLM scheduling
  overhead rather than draft rejection.
- Larger K increases rejected draft work faster than acceptance improves.
- Training data is from the evaluation set without a held-out split.

## Next Gate

OpenMath standalone K7/K9 gates were submitted on 2026-06-08 using the same
Qwen3-235B public PARD shape as prior K5/K12 runs:

| K | Job | Status at submission |
| ---: | ---: | --- |
| 7 | `3211257` | completed; batch-32 speedup `1.304x`, acceptance `35.80%` |
| 9 | `3211258` | completed; batch-32 speedup `1.345x`, acceptance `29.98%` |

K9 is the best static OpenMath batch-32 point so far, but its acceptance is
lower than K5. A short Qwen3-235B non-colocated TP4 fixed256 Full-GRPO K9 gate
completed as job `3211503`: Step2-5 total speedup `1.510x`, generation speedup
`2.115x`, and acceptance `28.60%` against baseline `3210580`.

The high-batch standalone sweep then favored K5 over K9 at batch 64/128, so the
same-shape NeMo-RL gate moved to public PARD K5. That job (`3211706`) completed
with Step2-5 total speedup `1.494x`, generation speedup `2.047x`, and
acceptance `42.19%` against baseline `3210580`. K5 is only slightly slower
than K9 but has much better acceptance, so use K5 as the static high-batch
default and reserve larger K for an adaptive policy. A 20-step K5 stability run
is queued as `3211900`.
