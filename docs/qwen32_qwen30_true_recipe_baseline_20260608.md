# Qwen3-32B / Qwen3-30B-A3B True Recipe Baselines

Date: 2026-06-08

## Purpose

Keep a no-SpecDec baseline that uses the original NeMo-RL performance YAML shape,
so TP or memory-envelope changes do not become the hidden comparison target.

The submitted jobs use the performance YAML directly. The only CLI overrides
are:

- `grpo.max_num_steps=20`, so the job terminates.
- logger fields, so the run is identifiable.

No behavioral shape overrides are applied: TP, GBS, generation length, vLLM
memory utilization, sequence packing, and async settings come from the YAML.

## Jobs

| Job | Model | Mode | Config | Status | Recipe Shape |
| --- | --- | --- | --- | --- | --- |
| `3210285` | Qwen3-30B-A3B | no SpecDec baseline | `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` | Failed before step metric | rollout batch 2048 (`64 x 32`), train GBS512, generation TP1, train TP1, EP16, `max_new_tokens=4096`, vLLM `gpu_memory_utilization=0.6` |
| `3210286` | Qwen3-32B | no SpecDec baseline | `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml` | Running; Step 1-11 metrics emitted | rollout batch 2048 (`64 x 32`), train GBS512, generation TP2, train TP2 / PP4, `max_new_tokens=4096`, vLLM `gpu_memory_utilization=0.6` |
| `3210601` | Qwen3-32B | public PARD K3 | `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml` | Running; Step 1-4 metrics emitted | Same original YAML shape as `3210286`; only `grpo.max_num_steps`, logger fields, and vLLM `speculative_config` are overridden. Draft TP is `2` to match YAML generation TP2. Current acceptance buckets average `33.54%` over 92 emitted log samples. |

Runtime: NeMo-RL branch `sna/qwen235b-main-vllm020-20260606`, commit
`37526dfac0a80b7032659a3ea030e0a9f69f99c6`; vLLM engine logs show
`v0.20.0`.

Machine-readable tracking:

```text
docs/qwen32_qwen30_true_recipe_baseline_20260608.csv
```

## Why This Matters

The current GBS512/2048 OOM-check jobs intentionally change the envelope to
fixed `max_new_tokens=256`, `gpu_memory_utilization=0.80`, and
`max_num_batched_tokens=16384` to test whether the Step-3 wake-up OOM still
appears. Those jobs answer stability under a conservative envelope.

The true recipe baselines answer a different question: whether a SpecDec path is
actually faster than the original performance recipe. If a TP2+SpecDec or
reduced-memory setup loses to this baseline, it should not be treated as a real
performance win even if it beats a modified baseline.

The matching Qwen3-32B true-recipe public-PARD job is `3210601`, submitted via
`experiments/eagle3_online/submit_qwen32_true_recipe_pard_k3.sh`. That script
does not use `submit_nemorl_online_draft_specdec.sh` because the helper changes
sequence packing and batch settings; here the intent is to keep the performance
YAML as the comparison target and add only the vLLM SpecDec block.
The driver log confirms the intended config: rollout batch `64 x 32 = 2048`,
train GBS `512`, generation TP2, train TP2/PP4, `max_new_tokens=4096`,
vLLM `gpu_memory_utilization=0.6`, sequence packing enabled, and
`speculative_config.method=draft_model`, `num_speculative_tokens=3`,
`draft_tensor_parallel_size=2`, `parallel_drafting=true`.
Early `3210601` acceptance buckets average `33.13%` before the first total step
metric. Do not use this as a throughput result; the first valid comparison to
`3210286` starts when Step 1 or preferably Step 2+ timing metrics emit.

Note: for these original YAMLs, the rollout batch is
`num_prompts_per_step * num_generations_per_prompt = 64 * 32 = 2048`, while
`policy.train_global_batch_size` is `512`. Both are recorded because earlier
OOM checks used `policy.train_global_batch_size=512/2048` explicitly.

`3210285` is not a usable performance baseline. It failed during the initial
policy-to-vLLM weight streaming path: vLLM's Qwen3-MoE fused-MoE loader rejected
`shard_dim=0` for a 3D expert tensor, then the policy worker timed out on ZMQ
after 120s. That is a recipe/weight-loading blocker, not a throughput result.

`3210286` is the current true Qwen3-32B baseline. Latest warm-window metrics are
Step 2-11: total step time `342.78s`, generation `130.91s`, E2E throughput
`1236.39 tok/s/GPU`, and generation-worker throughput `3230.84 tok/s/GPU`.

`3210601` is now a valid early throughput result, but it is not positive. On
the matched Step 2-4 window so far:

| Metric | Baseline `3210286` Step 2-4 | PARD K3 `3210601` Step 2-4 | PARD / Baseline |
| --- | ---: | ---: | ---: |
| Total step time | `333.82s` | `402.31s` | `0.830x` |
| Generation time | `126.54s` | `192.51s` | `0.657x` |
| E2E throughput | `1274.27 tok/s/GPU` | `1059.83 tok/s/GPU` | `0.832x` |
| Generation throughput | `3361.36 tok/s/GPU` | `2217.22 tok/s/GPU` | `0.660x` |

The driver log shows why the average acceptance is misleading: early decode
buckets can be around `55-65%`, but long CoT tail buckets collapse into roughly
`1-18%` acceptance. This original recipe has mean generation length around
`3132` tokens, unlike the fixed256 stability jobs below. Do not claim TP2
SpecDec is better than the original performance recipe unless later Step 2+
windows reverse this result.

K selection note: standalone PARD evidence used `K=5` and `K=12` rather than
`K=7/9`. `K=12` was best on short synthetic prompts, but on real OpenMath
`ISL=1024/OSL=1024` it fell to about `15-23%` acceptance and underperformed
`K=5`. Therefore larger K should be gated by per-position/long-tail acceptance,
not applied blindly to the 4096-token original GRPO recipe.
