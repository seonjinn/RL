# NeMo-RL PARD CUDA Graph Root-Cause Note

Date: 2026-07-02

## Scope

This note explains why PARD can improve Qwen3-32B async-1off at K=1 while
regressing on cheaper targets, larger K values, and target/draft TP greater
than one. All cited runs use CUDA Graphs (`enforce_eager=false`).

## Runtime Evidence

| Setup | Result | Interpretation |
|---|---:|---|
| Qwen3-32B async, PARD K1, target/draft TP1 | 1.0872x E2E throughput, 1.0853x generation throughput | Minimal draft overhead is amortized by the dense target model. |
| Qwen3-32B async, PARD K7, target/draft TP1 | 0.8811x E2E, 0.8987x generation at Steps 2-10 | Increasing K exposes generation and verification overhead faster than useful accepted work. |
| Qwen3-30B-A3B async, PARD K1/K7/K9/K16 | 0.87x/0.91x/0.88x/0.68x E2E throughput | The MoE target is relatively cheap, so drafter and bookkeeping costs are harder to amortize. |
| Qwen3-32B sync, Eagle K5/K7/K9 | 1.21x/1.13x/0.92x preliminary E2E throughput | Mean accepted length saturates near 2.6 while acceptance falls from 31.3% to 18.3% as K grows. |

## Code-Path Findings

The installed vLLM source is under the actor venv in the Lyris worktree.

1. PARD is not globally forced into eager execution.
   `vllm/v1/spec_decode/llm_base_proposer.py` dispatches the draft model
   forward through `CudagraphDispatcher`; parallel drafting uses one forward
   pass and can use PIECEWISE CUDA Graph replay.
2. TP draft sampling still gathers the full vocabulary by default.
   `_greedy_sample()` calls `compute_logits(...).argmax(...)` unless
   `use_local_argmax_reduction=true`. The default is false. The optimized path
   reduces communication from `O(vocab_size)` to `O(2 * TP)`.
3. Qwen3 cannot currently enable the local-argmax option safely.
   `Qwen3ForCausalLM` exposes `compute_logits()` but not `get_top_tokens()`;
   `_greedy_sample()` directly calls `model.get_top_tokens()` when the option
   is enabled. The equivalent method exists for the Llama4 Eagle model.
4. Generic parallel drafting performs eager allocations before graph replay.
   `set_inputs_first_pass()` creates `token_indices_to_sample` and
   `out_hidden_state_mapping` with `torch.empty()` for every proposal. It then
   expands query metadata by K. The dedicated Eagle path instead uses
   persistent buffers, fused Triton preparation kernels, and separate CUDA
   Graph managers for draft prefill and decode.

## Root-Cause Assessment

The evidence does not support the earlier hypothesis that the entire PARD
drafter runs outside CUDA Graphs. The current leading causes are:

1. Full-vocabulary draft-logit collectives at TP2/TP8.
2. Per-proposal allocation and metadata-expansion overhead in the generic
   parallel-drafting path.
3. Low temperature-1 acceptance at large K, so accepted length saturates while
   draft and target-verification work grows.
4. Lower amortization for Qwen3-30B-A3B because the target activates only about
   3B parameters per token.

## Proposed Isolated Optimization

No core patch is applied by this note.

1. Add `Qwen3ForCausalLM.get_top_tokens()` as a thin call to
   `self.logits_processor.get_top_tokens(self.lm_head, hidden_states)`.
2. Validate token equality against the full-logit argmax path at TP1 and TP2,
   including vocabulary padding and deterministic tie handling.
3. Run matched Qwen3-32B PARD K1 TP2 and Qwen3-235B PARD K1 TP8 smoke tests
   with `use_local_argmax_reduction=true`.
4. Only after that gate, preallocate the two parallel-drafting scratch tensors
   and add shape-bound assertions plus output-equivalence tests.

The local-argmax change should preserve greedy draft proposals except for exact
floating-point ties. Rejection sampling still preserves the target output
distribution, but rollout logprob, reward, KL, and generated-token equality
must be checked before using the optimized path for performance claims.
