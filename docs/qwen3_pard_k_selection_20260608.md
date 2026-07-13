# Qwen3 PARD K Selection Notes

Date: 2026-06-08

## Short Answer

Larger PARD `K` can be better, but only when the later draft positions keep
high acceptance. Our current evidence does not support "always use larger K".

The standalone PARD sweep now includes OpenMath `K=5/7/9/12`. `K=12` was
excellent on short synthetic prompts, but real OpenMath prompts collapsed its
acceptance enough that `K=5` and `K=9` are the practical candidates. The `K=9`
OpenMath batch-32 gate is slightly better than `K=5`, but its late-position
acceptance is much weaker.

## Evidence

| Regime | K | Result |
| --- | ---: | --- |
| Qwen3-235B standalone, synthetic `ISL=1000/OSL=512`, batch 32 | 5 | `1.71x` speedup, `95.5%` acceptance |
| Qwen3-235B standalone, synthetic `ISL=1000/OSL=512`, batch 32 | 12 | `3.29x` speedup, `93.0%` acceptance |
| Qwen3-235B standalone, real OpenMath `ISL=1024/OSL=1024`, batch 32 | 5 | `1.31x` speedup, `45.5%` acceptance |
| Qwen3-235B standalone, real OpenMath `ISL=1024/OSL=1024`, batch 32 | 7 | `1.30x` speedup, `35.8%` acceptance |
| Qwen3-235B standalone, real OpenMath `ISL=1024/OSL=1024`, batch 32 | 9 | `1.35x` speedup, `30.0%` acceptance |
| Qwen3-235B standalone, real OpenMath `ISL=1024/OSL=1024`, batch 32 | 12 | `1.22x` speedup, `22.8%` acceptance |
| Qwen3-235B standalone, synthetic long `ISL=10000/OSL=1000`, batch 32 | 5 | `0.39x`, `13.7%` acceptance |
| Qwen3-235B standalone, synthetic long `ISL=10000/OSL=1000`, batch 32 | 12 | `0.55x`, `13.4%` acceptance |
| Qwen3-32B NeMo-RL original recipe, real GRPO rollout, `max_new_tokens=4096` | 3 | First matched Step 2 is negative: total `0.810x`, generation `0.619x`, E2E `0.813x` |
| Qwen3-32B NeMo-RL fixed256/mem80/bt16k, GBS2048 | 3 | Positive: Step 2-15 total `1.220x`, generation `1.703x`, E2E `1.221x` |
| Qwen3-235B NeMo-RL non-colocated TP4 fixed256 | 3 | Positive on current 20-step timeout-patched window before the NCCL watchdog: Step2-16 total about `1.45x`, generation about `1.84x`, E2E about `1.46x` |
| Qwen3-235B NeMo-RL non-colocated TP4 fixed256 | 9 | Completed 5-step gate `3211503`: Step2-5 total `1.510x`, generation `2.115x`, E2E `1.525x` vs baseline Step2-5, but aggregate acceptance is only `28.60%` |
| Qwen3-235B NeMo-RL non-colocated TP4 fixed256 | 5 | Completed matched 5-step gate `3211706`: Step2-5 total `1.494x`, generation `2.047x`, E2E `1.497x` vs baseline Step2-5, with higher `42.19%` acceptance |
| Qwen3-235B standalone, real OpenMath `ISL=1024/OSL=1024`, batch 64/128 | 5 | High-batch speedups `1.259x/1.170x`, acceptance `45.01%/44.77%` |
| Qwen3-235B standalone, real OpenMath `ISL=1024/OSL=1024`, batch 64/128 | 7 | High-batch speedups `1.196x/1.114x`, acceptance `35.18%/34.91%`; better than K9 but still worse than K5 |
| Qwen3-235B standalone, real OpenMath `ISL=1024/OSL=1024`, batch 64/128 | 9 | High-batch speedups `1.186x/1.039x`, acceptance `28.90%/28.78%` |

## Interpretation

The right K depends on prompt regime and output tail behavior:

- Short synthetic generation has very high per-position acceptance, so `K=12`
  amortizes target verification well and wins.
- Real OpenMath and long-context runs have weaker late-position acceptance.
  `K=9` is now the best static OpenMath batch-32 point, but only narrowly:
  `1.345x` vs `1.313x` for `K=5`. `K=12` pays draft overhead for many tokens
  that are rejected, so it loses to both `K=5` and `K=9`.
- The current Qwen3-32B original recipe emits about 3132 tokens/sample. Its
  PARD K3 log shows early buckets around `55-65%` acceptance, then long-tail
  buckets dropping into roughly `1-18%`. Static larger K would likely worsen
  that tail unless we add gating or adaptive K.

## Next Experiment Rule

For Qwen3-235B NeMo-RL fixed256, `K=3` is conservative and not the ceiling for
short fixed-output gates. `K=9` completed as a 5-step gate (`3211503`) and is
faster than the matched baseline, but its `28.60%` aggregate acceptance matches
the high-batch standalone warning: it is carrying a lot of rejected draft work.

Because batch 32 may not be the saturation point for Qwen3-235B, a standalone
OpenMath high-batch sweep also tested batch 64/128: baseline `3211529`, public
PARD K5 `3211531`, public PARD K7 `3211982`, and public PARD K9 `3211532`.
The high-batch result favors K5: batch-64/128 speedups are `1.259x/1.170x`
for K5, `1.196x/1.114x` for K7, and only `1.186x/1.039x` for K9. Track it in
`docs/qwen3_235b_pard_highbatch_20260608.md`.

The matched NeMo-RL K5 gate (`3211706`) completed on the same non-colocated TP4
fixed256 shape. It is about `1%` slower than K9 on total step time and about
`3%` slower on generation time, but acceptance is much healthier:
`42.19%` for K5 versus `28.60%` for K9. If static K is required for high-batch
rollout, prefer K5 over K9. A 20-step K5 stability run has been submitted as
`3211900` and completed 20/20 at GBS256; treat that as a functional/stability
pass, not the final Qwen3-235B performance result. The current performance gate
is the matched GBS512 pair `3212012` baseline vs `3212209` K5 retry2. That
pair completed 5/5 and remains strongly positive: Step2-5 total `1.810x`,
generation `2.285x`, E2E `1.815x`, generation worker `2.287x`, acceptance
`43.08%`, with no OOM/fatal pattern.

For original 4096-token GRPO recipes, do not jump directly to static `K=9` or
`K=12`. Use an adaptive policy instead: high K in high-acceptance early decode,
lower K or disable speculation in the long tail when per-position acceptance
collapses.
