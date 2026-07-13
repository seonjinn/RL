# Qwen3-32B Performance Config SpecDec Status (2026-06-17)

Scope: Qwen3-32B only. Checked local `latest_*` trackers, docs, scripts, and remote OCI-HSG/Lyris Slurm/artifacts for NeMo-RL jobs that inherit NVIDIA-NeMo/RL `examples/configs/recipes/llm/performance` Qwen3-32B configs and add SpecDec.

## Existing sync submission

The exact sync-style Qwen3-32B performance-config SpecDec job has already been submitted on OCI-HSG and completed:

| Variant | Job ID | Slurm state | Job name | Config basis |
| --- | --- | --- | --- | --- |
| no-SpecDec baseline | `3210286` | `COMPLETED` `0:0` | `qwen32-true-recipe-baseline-step20-N4xG4` | `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml` |
| sync SpecDec PARD K3 | `3210601` | `COMPLETED` `0:0` | `qwen32-true-recipe-public-pard-k3-step20-N4xG4` | `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml` |

Remote tracker: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/experiments/eagle3_online/latest_qwen32_true_recipe_pard_k3_jobs.txt`.

The sync SpecDec wrapper is:

`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/experiments/eagle3_online/submit_qwen32_true_recipe_pard_k3.sh`

It uses `grpo-qwen3-32b-4n4g.yaml`, which inherits the Qwen3-32B performance YAML shape from `grpo-qwen3-32b-4n8g.yaml`, and only adds run-control/logger overrides plus vLLM `speculative_config`:

- drafter: `amd/PARD-Qwen3-0.6B` snapshot under the local HF cache
- `num_speculative_tokens=3`
- `draft_tensor_parallel_size=2`
- `parallel_drafting=true`

## Async-1off status

No Qwen3-32B NeMo-RL SpecDec async-1off submission was found.

Checked evidence:

- Local `latest_*` trackers and docs contain the sync true-recipe pair above and older Qwen3-32B sync/mainnightly/PARD experiments, but no Qwen3-32B async-1off SpecDec tracker.
- OCI-HSG active `squeue` had no matching Qwen3-32B SpecDec async-1off job.
- Lyris active `squeue` had no matching Qwen3-32B SpecDec async-1off job.
- OCI-HSG/Lyris file search found Qwen3-32B async config bases, but not a matching Qwen3-32B NeMo-RL SpecDec submit wrapper or `latest_*` job record.

Async config bases present remotely:

- `examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n4g-async-1off.yaml`
- `examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n8g-async-1off.yaml`

## Adjacent Qwen3-32B jobs checked

These are Qwen3-32B but not the exact performance-YAML SpecDec sync/async-1off target:

| Job ID | Slurm state | Note |
| --- | --- | --- |
| `3333531` | `COMPLETED` `0:0` | MathRL static PARD reference |
| `3334113` | `TIMEOUT` `0:0` | MathRL static PARD2 partial reference |
| `3334219` | `COMPLETED` `0:0` | MathRL baseline reference |
| `3345352` | `COMPLETED` `0:0` | MathRL online PARD2 reference |

Older sync/mainnightly Qwen3-32B SpecDec jobs also exist, but they use modified envelopes rather than the exact true performance-config basis.

## Safest next submit path

Sync is not missing. If a sync rerun is explicitly needed, the existing model-specific remote wrapper is the safest submit entrypoint:

```bash
ssh oci-hsg-cs-001-vscode-02 'cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606 && bash experiments/eagle3_online/submit_qwen32_true_recipe_pard_k3.sh'
```

Async-1off is missing. I found no existing safe one-command Qwen3-32B async-1off SpecDec submit wrapper. The safest next step is to create a new model-specific copy of `experiments/eagle3_online/submit_qwen32_true_recipe_pard_k3.sh`, without modifying shared launcher files, and change only the model/run labels plus:

- config: `examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n4g-async-1off.yaml` for OCI-HSG 4-GPU nodes
- Slurm shape/run labels to match `8n4g-async-1off`
- keep the same SpecDec overrides unless intentionally sweeping K/draft TP

After that reviewed wrapper exists, the intended submit shape would be:

```bash
ssh oci-hsg-cs-001-vscode-02 'cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606 && bash experiments/eagle3_online/submit_qwen32_true_recipe_pard_k3_async1off.sh'
```
