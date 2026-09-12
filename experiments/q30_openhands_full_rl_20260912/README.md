# Qwen3-30B-A3B Thinking: full OpenHands SWE2 RL gate

Status: prepared; submission and GPU execution not yet verified.

## Scope

No-SpecDec baseline, BF16 vLLM with FlashInfer TRTLLM MoE, FAP,
128K context and the native SWE2 OpenHands/Thinking template. This is full
async GRPO (policy optimizer and refit enabled), not trajectory collection.
W&B project `sna-specdec`, group `sna-swe2-full-rl-gate`.

The small gate uses 4 GB200 nodes with 4 GPUs each: 2 training nodes
(TP2/CP4/EP4/PP1/ETP1) and 2 generation nodes (TP2). PPS1 × GPP8 = GBS8;
3 training steps, packing enabled, no drafter, no checkpoint saves.
This is not the native 16n8g/GBS64 performance recipe. A successful gate
does not establish representative SWE performance or baseline speedups.

## Validation sequence

1. Local shell/config checks; commit and push isolated source.
2. Remote pinned recursive submodules, source cleanliness, image/assets check.
3. `submit.sh --test-only`, then `submit.sh --submit` on `nemotron_n4_post/batch`.
4. Per-node imports, Apptainer presence, Gym server-venv dry run.
5. Real OpenHands trajectories, policy updates and repeated refit for 3 steps.

GPU success requires all three optimizer steps and weight-update cycles,
finite rewards/loss/logprob metrics, successful sandbox responses, and no
silent skip-training/trajectory-collection path. Zero reward or identical
group rewards may produce no useful policy gradient; inspect before claiming
learning. The dry run is environment preparation, not an RL success signal.

The initial probe intentionally does not reuse the older August Gym archive:
its Gym revision/runtime patches differ from this source. Cold environment
setup errors must be diagnosed before scaling to 20 steps.

## Timing interpretation

Async `exposed_generation` is replay-buffer wait on the training critical
path, not total rollout time. For the blog, report that distinction and
separate model-call, tool/sandbox, and trajectory durations. Do not sum
concurrent trajectory durations and label that value a batch wall time.
