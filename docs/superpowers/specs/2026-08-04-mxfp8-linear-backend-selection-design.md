# MXFP8 Linear Backend Selection Design

## Goal

Allow an MXFP8 NeMo-RL rollout recipe to select the vLLM dense-linear backend
through `policy.generation.vllm_kwargs.linear_backend`, including:

- `flashinfer_cutlass`
- `flashinfer_cutedsl`
- `flashinfer_trtllm`

The selected backend must remain active after every RL refit. Unsupported or
refit-unsafe kernels must fail before rollout rather than silently changing the
requested backend or consuming an incompatible weight layout.

## Current Behavior

NeMo-RL already forwards arbitrary `vllm_kwargs` to vLLM, so the configuration
value reaches vLLM. The MXFP8 refit patch is the limiting component:

- FlashInfer CUTLASS is supported with NeMo-RL's legacy `[N, K]` weight and
  swizzled-scale path.
- vLLM's preferred CuTeDSL kernel is replaced with CUTLASS because CuTeDSL
  consumes a column-major `[K, N]` prepared weight.
- FlashInfer TRTLLM is rejected because it requires shuffled weights and scales
  that differ from both CUTLASS and CuTeDSL.

Therefore, accepting another backend name without a refit-aware preparation
contract would be incorrect.

## Ownership Boundary

### NeMo-RL

NeMo-RL owns:

- forwarding `vllm_kwargs.linear_backend` without rewriting it;
- retaining canonical checkpoint parameters for repeated refit;
- verifying that the selected vLLM kernel supports refit-safe preparation;
- delegating post-load and post-refit preparation to that kernel;
- retaining the existing legacy CUTLASS compatibility path;
- failing early with the selected backend and kernel name when the contract is
  unavailable.

### vLLM

vLLM owns:

- selecting the concrete MXFP8 linear kernel for the requested backend;
- declaring refit support with
  `preserves_checkpoint_weight_scale_for_refit = True`;
- producing the runtime representation required by each backend;
- refreshing prepared buffers after a refit without changing CUDA Graph-visible
  storage addresses after initialization;
- using the backend default tactic when an exact offline tactic lookup misses.

The backend-specific layouts remain inside vLLM:

| Backend | Runtime weight representation | Runtime scale representation |
|---|---|---|
| FlashInfer CUTLASS | contiguous `[N, K]` | CUTLASS-swizzled |
| FlashInfer CuTeDSL | column-major `[K, N]` | CuTeDSL-compatible swizzle |
| FlashInfer TRTLLM | padded and shuffled physical `[N, K]` | TRTLLM shuffled scale |

## Configuration

No new NeMo-RL enum is introduced. Recipes use vLLM's existing option:

```yaml
policy:
  generation:
    vllm_kwargs:
      linear_backend: flashinfer_cutedsl
```

Changing the value to `flashinfer_cutlass` or `flashinfer_trtllm` selects the
corresponding backend. Keeping vLLM as the source of truth avoids duplicating a
version-dependent backend allowlist in NeMo-RL.

## Refit Dispatch

`process_weights_after_loading_mxfp8_linear` follows this order:

1. Validate that the checkpoint MXFP8 weight is two-dimensional.
2. Read the concrete kernel selected by vLLM.
3. If the kernel declares
   `preserves_checkpoint_weight_scale_for_refit = True`, delegate preparation
   to `kernel.process_weights_after_loading(layer)` and retain the selected
   backend.
4. Otherwise, allow the existing FlashInfer CUTLASS compatibility path.
5. Reject every other kernel with an actionable error. Do not silently demote
   CuTeDSL or TRTLLM to CUTLASS.

This is capability-based rather than class-name allowlisting. A future vLLM
backend can work without a NeMo-RL change if it implements the same contract.

## Failure and Fallback Policy

- **Unsupported backend or unavailable kernel:** vLLM reports backend selection
  failure during engine construction.
- **Selected kernel lacks refit capability:** NeMo-RL raises an error naming the
  backend and kernel before rollout.
- **Exact TRTLLM tactic lookup miss:** remain on TRTLLM and use `tactic=-1`,
  which invokes the backend default heuristic.
- **Unqualified TRTLLM layer family:** the adaptive vLLM policy may use its
  configured CuTeDSL family fallback. This decision belongs to vLLM and is not
  inferred by NeMo-RL.
- **No silent backend substitution:** a requested CuTeDSL or TRTLLM backend must
  not be reported as active while actually executing CUTLASS.

## Testing

Unit tests will cover:

1. `linear_backend` survives the FP8 kwargs merge unchanged for CUTLASS,
   CuTeDSL, and TRTLLM values.
2. A refit-safe CuTeDSL or TRTLLM kernel receives native post-load delegation.
3. The existing legacy CUTLASS path still prepares weights and scales.
4. A refit-unsafe non-CUTLASS kernel fails with an actionable message.
5. A two-dimensional weight is required before backend dispatch.
6. Existing FP8 and MXFP8 quantization tests remain green.

The companion vLLM validation will cover prepared-buffer pointer stability,
backend-specific layout correctness, CUDA Graph replay, numerical agreement,
and repeated refit updates.

## Documentation

Add a short MXFP8 backend section to the custom-vLLM guide with:

- the three recipe values;
- the requirement for a vLLM build whose selected kernel declares the refit
  capability;
- the distinction between backend fallback and exact tactic fallback;
- an example startup error for an incompatible stock vLLM build.

## Acceptance Criteria

- A recipe can explicitly request FlashInfer CUTLASS, CuTeDSL, or TRTLLM.
- The selected refit-safe backend remains the actual runtime backend after
  initial load and repeated refits.
- Existing CUTLASS recipes retain current behavior.
- Unsupported combinations fail before rollout with no silent demotion.
- Unit tests verify configuration preservation and dispatch policy.
- GPU validation confirms correctness and CUDA Graph stability before claiming
  CuTeDSL or TRTLLM production support.
