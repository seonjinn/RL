# NeMo-RL DynamicSD Lyris and 32K Long-Tail Design

## Objective

Measure Eagle-3 DynamicSD in full synchronous NeMo-RL GRPO for
Qwen3-30B-A3B, Qwen3-32B, and Qwen3-235B-A22B. Run two matched cohorts on
Lyris: the upstream performance recipe without sequence-length changes, and a
32K-output long-tail profile.

## Experiment Matrix

Each model uses its upstream synchronous performance recipe and three variants:

- target-only baseline
- fixed Eagle-3 K5
- Eagle-3 DynamicSD with `[[1,16,5],[17,32,4],[33,64,3],
  [65,128,1],[129,512,0]]`

Every run uses 20 GRPO steps, temperature/top-p `1.0/1.0`, PIECEWISE CUDA
Graphs, draft TP1, disabled checkpoint writes, and W&B logging. The recipe
cohort changes no model, topology, dataset, rollout-shape, training-batch, or
sequence-length field from the upstream recipe.

The `longtail32k` cohort changes only these sequence limits:

- `policy.max_total_sequence_length=36864`
- `policy.generation.max_new_tokens=32768`
- `policy.generation.vllm_cfg.max_model_len=36864`

This keeps up to 4K input context plus 32K generated output. Natural EOS
remains enabled so the experiment measures the real rollout tail rather than
forcing every response to 32K.

## Lyris Execution

Use account `coreai_dlalgo_llm`, partition `gb200`, four GPUs per node without
`--gres`, and `--segment` equal to the recipe node count. The model shapes are
4 nodes for Qwen3-30B-A3B, 4 nodes for Qwen3-32B, and 16 nodes for Qwen3-235B.
Use the staged vLLM 0.24 NeMo-RL checkout, nightly container, HF cache, and W&B
credentials under `/lustre/fsw/coreai_dlalgo_llm/users/sna`.

Run scheduler test-only validation, then one-step baseline/K5/DynamicSD smokes
for both profiles. A profile is promoted to 20 steps only after all three
variants reach rollout and policy training without CUDA Graph, SpecDec, OOM,
or NCCL errors. The Lyris five-hour limit remains unchanged; completed steps
from a timeout are retained as partial evidence but are not called final.

## Provenance and Metrics

The launcher records profile, max output length, max total sequence length,
job ID, W&B URL, recipe, topology, commit, container, drafter, graph mode, and
DynamicSD schedule. W&B run names and result directories include the profile
so recipe and long-tail rows cannot collide.

Final means use Steps 2-20. Match baseline and K5 rows by model, recipe,
profile, max output length, max total sequence length, topology, sampling,
graph mode, commit, and container. Report generation and E2E step time,
generation and E2E throughput, acceptance rate, mean accepted length, reward,
response length, and approximate KL. DynamicSD is compared against both the
matched baseline and fixed K5.

## Failure Rules

- Do not fall back to eager execution.
- Do not reduce recipe-owned rollout or training batch sizes after OOM.
- Do not compare partial or mismatched-profile rows as final speedups.
- Preserve logs and completed-step metrics for timeout and infrastructure
  failures.
- Require positive SpecDec draft and acceptance counters before treating a
  DynamicSD row as active.

## Success Criteria

The integration is valid when all three variants for a model/profile complete
at least the smoke and expose the expected configuration. A performance claim
requires matched Steps 2-20 and no material reward, response-length, or KL
regression relative to the same-profile baseline.
