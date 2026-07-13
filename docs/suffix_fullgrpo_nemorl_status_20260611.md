# Suffix Decoding Full-GRPO NeMo-RL Status - 2026-06-11

## Result

Suffix Decoding now works through the synchronous NeMo-RL full-GRPO rollout and policy-training path with vLLM v0.20.0. For async SWERL/NeMo-Gym, the launcher now supports suffix decoding with explicit behavior-logprob modes; the active SWERL smoke is generation-only so it can measure suffix acceptance/throughput without silently running off-policy GRPO with missing behavior logprobs.

Validated jobs:

| Job | Model | Config | Status | Step | Acceptance |
| --- | --- | --- | --- | --- | --- |
| 3264722 | Qwen/Qwen3-30B-A3B | suffix K32, sync on-policy logprob repair, 4x4 GPUs, OSL 512, GBS 16 | COMPLETED | 1/1 | weighted 22.04%, length 1.39 |
| 3264667 | Qwen/Qwen3-8B | suffix K32, sync on-policy logprob repair, 1x4 GPUs, OSL 128, GBS 4 | COMPLETED | 1/1 | short run; no aggregate acceptance parsed |
| 3265438 | Qwen/Qwen3-8B | suffix K32 with explicit suffix knobs, sync on-policy logprob repair, 1x4 GPUs, OSL 128, GBS 4 | COMPLETED | 1/1 | short run; vLLM v0.20.0 suffix engine initialized |
| 3264425 | Qwen/Qwen3-30B-A3B | suffix K32, throughput smoke with logprob repair, 4x4 GPUs, OSL 1024, GBS 16 | COMPLETED | 1/1 | weighted 25.22%, length 1.49 |
| 3264441 | Qwen/Qwen3-8B | suffix K32, throughput smoke with logprob repair, 1x4 GPUs, OSL 128, GBS 4 | COMPLETED | 1/1 | short run; no aggregate acceptance parsed |

Metrics are in `docs/qwen_suffix_fullgrpo_smoke_20260611.csv` and `docs/qwen8_suffix_fullgrpo_knobs_regression_20260611.csv`.

No suffix Full-GRPO 20-step run had been completed before the latest check. The verified suffix Full-GRPO evidence above is all 1-step smoke coverage. A Qwen3-30B-A3B suffix K32 20-step run was submitted afterward:

| Job | Model | Config | State |
| --- | --- | --- | --- |
| 3266990 | Qwen/Qwen3-30B-A3B | suffix K32, sync full-GRPO, 4x4 GPUs, OSL 1024, GBS 16, `max_steps=20` | PENDING/Priority, estimated start `2026-06-11T18:10:00`, no ray-driver log yet as of `2026-06-11 10:00 PDT` |

## Changes

`experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` now supports model-free speculative decoding formats:

- `DRAFT_FORMAT=suffix` maps to `speculative_config.method=suffix`.
- `DRAFT_FORMAT=ngram` or `DRAFT_FORMAT=pld` maps to `speculative_config.method=ngram`.
- These modes force `policy.draft.enabled=false`, omit `speculative_config.model`, omit draft TP, and avoid EAGLE/PARD online drafter code paths.
- Suffix mode now exposes vLLM suffix knobs: `suffix_decoding_max_tree_depth`, `suffix_decoding_max_cached_requests`, `suffix_decoding_max_spec_factor`, and `suffix_decoding_min_token_prob`.
- Suffix mode fail-fast validates that `arctic_inference/suffix_decoding` is visible through `SOURCE_VLLM_SITE` or `ARCTIC_SITE`.
- Online draft-training support is only required when `policy.draft.enabled=true`; model-free suffix/PLD submissions no longer require the online drafter patch.

Added wrappers:

- `experiments/eagle3_online/submit_qwen30ba3b_suffix_fullgrpo_smoke_20260611.sh`
- `experiments/eagle3_online/submit_qwen8_suffix_fullgrpo_smoke_20260611.sh`
- The wrappers default to synchronous on-policy repair mode: vLLM request logprobs are omitted for SpecDec, and GRPO fills behavior logprobs from a fresh policy fprop only when the sampler is identity.
- True request-logprob mode is available through `NRL_VLLM_SPECDEC_REQUEST_LOGPROBS=true` plus `NRL_ALLOW_SPECDEC_REQUEST_LOGPROBS=true`, but vLLM V1 can disable SpecDec acceptance for logprob requests. Use this for correctness checks, not speedup claims.
- Throughput-only experiments can explicitly use `NRL_ALLOW_SPECDEC_WITHOUT_BEHAVIOR_LOGPROBS=true`, but those runs should not be used as strict GRPO correctness evidence.
- The wrappers include GBS divisibility guards for their fixed policy DP sizes: Qwen30BA3B 4x4 requires GBS divisible by 16; Qwen8 1x4 requires GBS divisible by 4.

Added Python 3.13 arctic-inference cache build:

- `tmp/build_arctic_inference_py313_20260611.sbatch`
- Output cache: `.container_cache/arctic-inference-0.1.1-py313`
- Required because the old standalone cache only had `_C.cpython-312-aarch64-linux-gnu.so`, while NeMo-RL Ray actors run Python 3.13.

## Notes

vLLM logs warn that async scheduling is disabled for suffix-based speculative decoding. This is a vLLM runtime limitation for suffix mode, not a NeMo-RL config bug. In the validated Qwen30BA3B strict run, the engine log shows `Initializing a V1 LLM engine (v0.20.0)` with `SpeculativeConfig(method='suffix', model=None, num_spec_tokens=32)`.

The smoke wrappers keep `VLLM_ENFORCE_EAGER=true` for stability, so their timing should not be used as final throughput. For performance comparisons, rerun with a non-eager performance recipe and a matched baseline.

For SWE-RL / Nemo-Gym, the same model-free suffix path is now staged in the Nemo-Gym launcher. The first SWERL smoke is waiting for SLURM resources; once it starts, the remaining validation is runtime environment behavior rather than command construction.

The active remote checkout also has an explicit `NRL_STOP_AFTER_GENERATION` diagnostic path in `nemo_rl/algorithms/grpo.py`. That path writes a JSON payload under `NRL_STOP_AFTER_GENERATION_METRICS_DIR` and prints a compact `[SpecDec diag metrics]` line so the SWERL generation-only suffix run can be parsed without entering async reward/logprob/training with missing behavior logprobs. The same change has been preserved in the local remote-patch artifact at `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/algorithms/grpo.py`.

## SWE-RL / Nemo-Gym Path

`run_grpo_qwen3_235b_swe.sh` now has opt-in suffix support:

- `ENABLE_VLLM_SPECDEC=true`
- `SPECDEC_METHOD=suffix`
- `NUM_SPECULATIVE_TOKENS=32`
- `ARCTIC_SITE=/path/to/arctic-inference-0.1.1-py313`
- `SPECDEC_GRPO_MODE=auto|stop_after_generation|strict_request_logprobs|throughput_only|sync_repair|none`
- optional suffix knobs listed above

Important distinction: vLLM logs say internal async scheduling is disabled for suffix decoding, but NeMo-RL `policy.generation.vllm_cfg.async_engine` must stay enabled for `grpo.async_grpo.enabled=True` and Nemo-Gym rollouts. The SWERL launcher keeps NeMo async engine enabled and only passes the suffix `speculative_config`.

For async SWERL, `SPECDEC_GRPO_MODE=auto` resolves to `stop_after_generation`. This sets `NRL_STOP_AFTER_GENERATION=true`, records generation/acceptance metrics, and stops before behavior-logprob reconstruction. `SPECDEC_GRPO_MODE=strict_request_logprobs` requests exact vLLM logprobs for an async correctness smoke, but it should not be used to claim suffix speedup because vLLM may disable SpecDec acceptance when request logprobs are enabled.

The launcher also now treats `NUM_ACTOR_NODES` and `NUM_GEN_NODES` separately, then passes their sum as `cluster.num_nodes` and `sbatch --nodes`. The old pre-patch script used `NUM_NODES` ambiguously and could under-allocate actor workers after GRPO subtracted generation nodes.

Remote dry-run on the vLLM0.20 checkout rendered the expected command with `grpo.async_grpo.enabled=True`, suffix `speculative_config`, and `PYTHONPATH` containing `.container_cache/arctic-inference-0.1.1-py313`.

The original default R2E-Gym train JSONL path in the launcher is still not visible from this checkout, and `swerl_gen/data/example_rollouts.jsonl` is a rollout output file with `metadata=null`, not a valid training input. A usable Nemo-Gym HF-format SWE data source was found under the Qwen3-235B artifact root:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_eagle3/data/swegym_train_nemogym_hf_8.jsonl`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_eagle3/data/swegym_train_nemogym_hf_full.jsonl`

Dry-run command rendering with `swegym_train_nemogym_hf_8.jsonl` succeeded. It keeps `policy.generation.vllm_cfg.async_engine=true`, sets `grpo.async_grpo.enabled=True`, and injects:

- `++policy.generation.vllm_kwargs.speculative_config.method=suffix`
- `++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=32`
- the four suffix-specific vLLM knobs

Dry-run command rendering was repeated on the final image-matched SWERL n8 data path and still keeps NeMo async GRPO enabled while injecting suffix decoding:

- `grpo.async_grpo.enabled=True`
- `policy.generation.colocated.enabled=False`
- `++policy.generation.vllm_kwargs.speculative_config.method=suffix`
- `++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=32`
- `++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_max_tree_depth=24`
- `++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_max_cached_requests=10000`
- `++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_max_spec_factor=1.0`
- `++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_min_token_prob=0.1`

The original `swegym_train_nemogym_hf_8.jsonl` still should not be used for an actual run: a pre-submit asset check found that all eight instances have no matching `.sif` sandbox image under the configured SWE image formatters.

Follow-up fix for OCI/HSG:

- Added the visible arm64 image formatter to the Nemo-Gym SWE config: `/lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench/swebench_sweb.eval.arm64.{instance_id}.sif`.
- Created an image-matched 8-sample Nemo-Gym HF JSONL from `swe_swerebenchv2_full_0316_passed.jsonl`: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_eagle3/data/swerebenchv2_arm64_image_matched_n8_nemogym_hf.jsonl`.
- Added `WANDB_ENABLED` to `run_grpo_qwen3_235b_swe.sh`, so smoke runs can set `WANDB_ENABLED=False` when `env.sh` is not present.
- The first direct submit with the default `16x8` shape failed immediately because this cluster exposes `gpu:4` nodes, not `gpu:8` nodes.
- Resubmitted as the launcher-supported `32x4` shape with `PPS=8`, `GPP=1`, `GBS=8`, `MAX_NUM_STEPS=1`, `AGENT_MAX_TURNS=1`, and suffix K32.

Submitted SWERL suffix smoke:

| Job | Model | Shape | Data | SpecDec | State |
| --- | --- | --- | --- | --- | --- |
| `3266023` | Qwen3-235B-A22B-Thinking-2507 | pre-patch `32x4` async smoke | image-matched SWERL n8 | suffix K32 | cancelled before start; missing async behavior-logprob mode guard |
| `3266737` | Qwen3-235B-A22B-Thinking-2507 | total `32x4`: actor 24 + generation 8, async GRPO, `PPS=8`, `GPP=1`, `GBS=8`, one generation-only step | image-matched SWERL n8 | suffix K32, `SPECDEC_GRPO_MODE=stop_after_generation` | pending priority, estimated start `2026-06-11T19:40:00` |

Latest queue check at `2026-06-11 10:00 PDT`: job `3266737` remains `PENDING/Priority` with estimated start `2026-06-11T19:40:00`; suffix Full-GRPO20 job `3266990` remains `PENDING/Priority` with estimated start `2026-06-11T18:10:00`. The shared poll artifact is `docs/pard2_swerl_active_status_20260611.csv`; rerun `scripts/refresh_pard2_swerl_active_status.sh` to refresh it and fetch PARD2/SWERL/suffix-full-GRPO20 metrics when available.
