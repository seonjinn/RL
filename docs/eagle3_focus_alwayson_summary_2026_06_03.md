# EAGLE3 always-on focused results

Updated: 2026-06-03 PDT

This focused view only shows always-on SpecDec and only K=1/K=3, to avoid mixing in gated/TLT-style results. Gated results remain useful for root-cause analysis, but they are intentionally omitted from the shareable performance charts.

## Generated artifacts

- `docs/eagle3_focus_vllm_standalone_speedup.png`
- `docs/eagle3_focus_vllm_standalone_acceptance.png`
- `docs/eagle3_focus_nemorl_alwayson_speedups.png`
- `docs/eagle3_focus_nemorl_alwayson_acceptance_by_model.png`
- `docs/eagle3_focus_vllm_standalone_metrics.csv`
- `docs/eagle3_focus_nemorl_alwayson_metrics.csv`

## Standalone ISL/OSL setup

| Model | Drafter scope | Benchmark | ISL / OSL | Batch sizes | Depths shown | Acceptance coverage |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | public HF | synthetic static | 1000 / 512 | 1,2,4,8,16,32 | K=1,K=3 | K=1/K=3 static acceptance not emitted |
| Qwen3-8B | public HF | DAPOMath real-prompt isolation | cap 1000 / 512 | 32 | K=1,K=3 | acceptance emitted |
| Qwen3-32B | public HF | synthetic static | 1000 / 512 | 1,2,4,8,16,32 | K=1,K=3 | acceptance emitted |
| Qwen3-30B-A3B | local 500K | synthetic static | 1000 / 1000 | 1,2,4,8,16 | K=1 only | K=3 standalone not run |

Standalone uses fixed decode length. For the synthetic static sweeps, vLLM used `SamplingParams(min_tokens=OSL, max_tokens=OSL, ignore_eos=True, temperature=0.0)`.

## NeMo-RL length setup

| Model | max_total_sequence_length | responses/step | OSL semantics | Notes |
| --- | --- | --- | --- | --- |
| Qwen3-8B | 8192 | 2048 | variable; EOS/max-context stopped | temperature 1.0 default long rollout; Step 1 default example was about 6.6K generated tokens/sample |
| Qwen3-32B | 4096 | 2048 | variable; EOS/max-context stopped | completed 20-step fixed/offline always-on K=1/K=3 |
| Qwen3-30B-A3B | 2048 | 2048 | variable; EOS/max-context stopped | completed historical 500K fixed/offline always-on K=1/K=3 |

NeMo-RL is not fixed-OSL unless we explicitly run a diagnostic. It uses real rollout prompts, per-sample remaining context, EOS/stop criteria, and the recipe's max sequence length. This is why standalone OSL=512 and NeMo-RL long rollout acceptance/speedup should not be interpreted as the same workload.

## vLLM standalone highlights

| Run | K | Speedup | Acceptance |
| --- | --- | --- | --- |
| Qwen3-8B static bs32 | K=1 | 1.331x | n/a |
| Qwen3-8B static bs32 | K=3 | 2.145x | n/a |
| Qwen3-8B DAPOMath bs32 | K=1 | 0.665x | 61.11% |
| Qwen3-8B DAPOMath bs32 | K=3 | 0.853x | 36.62% |
| Qwen3-32B static bs32 | K=1 | 1.625x | 79.9% |
| Qwen3-32B static bs32 | K=3 | 2.288x | 67.1% |
| Qwen3-30B-A3B static best available | K=1 | 0.654x at bs8 | 27.80% |

## NeMo-RL always-on highlights

| Model | K | Job | Gen throughput | Gen step-time | E2E throughput | E2E step-time | Acceptance | Scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | K=1 | 3125397/3127152 | 1.413x | n/a | 1.179x | n/a | 60.46% | early matched long-rollout Step 1-4; not a completed 20-step aggregate |
| Qwen3-8B | K=3 | 3125396/3127158 | 1.483x | n/a | 1.225x | n/a | 35.72% | early matched long-rollout Step 1-4; not a completed 20-step aggregate |
| Qwen3-32B | K=1 | 3128147 | 1.346x | 1.348x | 1.154x | 1.154x | 69.45% | completed 20-step NeMo-RL fixed/offline drafter |
| Qwen3-32B | K=3 | 3128428 | 1.356x | 1.357x | 1.181x | 1.181x | 45.28% | completed 20-step NeMo-RL fixed/offline drafter |
| Qwen3-30B-A3B | K=1 | 3058167 | 1.344x | 1.343x | 1.099x | 1.098x | 57.51% | completed 20-step NeMo-RL historical 500K fixed/offline drafter |
| Qwen3-30B-A3B | K=3 | 3058169 | 1.177x | 1.176x | 1.068x | 1.068x | 31.85% | completed 20-step NeMo-RL historical 500K fixed/offline drafter |

## Caveats

- Qwen3-8B synthetic static standalone has K=1/K=3 speedup but did not emit K=1/K=3 acceptance. The DAPOMath real-prompt bs32 isolation did emit acceptance.
- Qwen3-30B-A3B standalone here is a local 500K drafter result, not a public-HF drafter result, and K=3 standalone was not run.
- Qwen3-8B NeMo-RL always-on speedups are early matched Step 1-4 long-rollout summaries, while Qwen3-32B and Qwen3-30B-A3B rows are completed aggregate comparisons.
