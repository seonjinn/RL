# ModelOpt Upstream Drift Report

Overall: **WARN**
Generated: `2026-05-22 06:51:34 PDT`
Official example: https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/speculative_decoding

## Summary

- Local ModelOpt: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL_Qwen3_Roadmap/Model-Optimizer`
- Local branch/head: `main` / `b02e8885509c`
- Dirty files: `39`
- Upstream probe: `ok` / `3ff15ccef3f0`
- Training source decision: `pass`
- Training source summary: official main advanced, but Eagle3/speculative-decoding focus paths are unchanged
- Hayate ModelOpt visible: `True`
- Hayate branch/head: `main` / `4eacb0da723a`
- Hayate dirty/untracked files: `7`

## Training Source Decision

- Status: `pass`
- Upstream HEAD matches local: `False`
- Allowed focus diffs: `examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py`
- Disallowed focus diffs: `-`
- Unrelated dirty files: `38`
- Hayate reference only: `True`
- Recommendation: Use the local/remote ModelOpt checkout as the training source. Treat Hayate's checkout as workflow reference only; port only intentional workflow ideas, not the older checkout wholesale.

## Notes
- local ModelOpt worktree has uncommitted changes
- local ModelOpt HEAD is behind official upstream main, but Eagle3 focus paths are unchanged
- Hayate/Hiso ModelOpt checkout has uncommitted or untracked files

## Local Dirty Files

- `M .claude/skills/common/environment-setup.md`
- ` M .claude/skills/common/remote-execution.md`
- ` M .claude/skills/common/workspace-management.md`
- ` D .claude/skills/compare-results/SKILL.md`
- ` D .claude/skills/compare-results/tests/evals.json`
- ` M .claude/skills/deployment/SKILL.md`
- ` M .claude/skills/evaluation/SKILL.md`
- ` M .claude/skills/evaluation/recipes/env.example`
- ` M .claude/skills/evaluation/recipes/examples/example_eval.yaml`
- ` D .claude/skills/evaluation/recipes/tasks/aa_lcr.md`
- ` D .claude/skills/evaluation/recipes/tasks/aime2025.md`
- ` D .claude/skills/evaluation/recipes/tasks/gpqa.md`
- ` D .claude/skills/evaluation/recipes/tasks/ifbench.md`
- ` D .claude/skills/evaluation/recipes/tasks/livecodebench.md`
- ` D .claude/skills/evaluation/recipes/tasks/mmlu_pro.md`
- ` D .claude/skills/evaluation/recipes/tasks/mmmu_pro.md`
- ` D .claude/skills/evaluation/recipes/tasks/ns_hle_aa.md`
- ` D .claude/skills/evaluation/recipes/tasks/scicode.md`
- ` D .claude/skills/evaluation/recipes/tasks/tau2_bench_telecom.md`
- ` M .claude/skills/evaluation/tests/evals.json`
- ` M .claude/skills/launching-evals/references/analyze-results.md`
- ` M .claude/skills/monitor/SKILL.md`
- ` M .claude/skills/ptq/SKILL.md`
- ` M .claude/skills/ptq/references/checkpoint-validation.md`
- ` M .claude/skills/ptq/references/unsupported-models.md`
- ` M .claude/skills/ptq/tests.json`
- ` M .gitignore`
- ` M CHANGELOG.rst`
- ` M examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py`
- ` M modelopt/torch/quantization/mode.py`
- ` M tests/unit/torch/quantization/test_mode.py`
- ` M tools/launcher/core.py`
- ` M tools/launcher/slurm_config.py`
- `?? .claude/skills/evaluation/recipes/tasks/aime2025.yaml`
- `?? .claude/skills/evaluation/recipes/tasks/gpqa.yaml`
- `?? .claude/skills/evaluation/recipes/tasks/ifbench.yaml`
- `?? .claude/skills/evaluation/recipes/tasks/livecodebench.yaml`
- `?? .claude/skills/evaluation/recipes/tasks/mmlu_pro.yaml`
- `?? .claude/skills/evaluation/recipes/tasks/scicode.yaml`

## Local Focus Diff Stat

```text
.../compute_hidden_states_trtllm.py                | 49 ++++++++++++++++++----
 1 file changed, 40 insertions(+), 9 deletions(-)
```

## Hayate Dirty/Untracked Files

- `?? examples/speculative_decoding/eagle_config_qwen3_30b_moe.json`
- `?? examples/speculative_decoding/eagle_config_qwen3_32b.json`
- `?? examples/speculative_decoding/eagle_config_qwen3_8b.json`
- `?? examples/speculative_decoding/logs/`
- `?? examples/speculative_decoding/prepare_input_conversations/add_dapo17k.py`
- `?? examples/speculative_decoding/prepare_input_conversations/generate_responses.py`
- `?? examples/speculative_decoding/slurm/`

## Local vs Hayate Focus Files

| path | local | hayate | same sha |
| --- | --- | --- | --- |
| `examples/speculative_decoding/README.md` | True | True | False |
| `examples/speculative_decoding/launch_train.sh` | True | True | False |
| `examples/speculative_decoding/main.py` | True | True | False |
| `examples/speculative_decoding/fsdp_config.json` | False | True | False |
| `examples/speculative_decoding/collect_hidden_states/common.py` | True | False | False |
| `examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py` | True | True | False |
| `examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py` | True | True | False |
| `examples/speculative_decoding/scripts/export_hf_checkpoint.py` | True | True | False |
| `examples/speculative_decoding/scripts/convert_to_vllm_ckpt.py` | True | True | True |
| `examples/speculative_decoding/prepare_input_conversations/add_dapo17k.py` | False | True | False |
| `examples/speculative_decoding/prepare_input_conversations/generate_responses.py` | False | True | False |
| `examples/speculative_decoding/slurm/generate_responses.sbatch` | False | True | False |
| `examples/speculative_decoding/slurm/train_eagle3.sbatch` | False | True | False |
| `examples/speculative_decoding/slurm/submit_all.sh` | False | True | False |
| `examples/speculative_decoding/eagle_config_qwen3_8b.json` | False | True | False |
| `examples/speculative_decoding/eagle_config_qwen3_30b_moe.json` | False | True | False |
| `examples/speculative_decoding/eagle_config_qwen3_32b.json` | False | True | False |
| `modelopt_recipes/general/speculative_decoding/eagle3.yaml` | True | False | False |
| `modelopt/torch/speculative/eagle/config.py` | False | False | False |
| `modelopt/torch/speculative/eagle/eagle_model.py` | True | True | False |
| `modelopt/torch/speculative/eagle/utils.py` | True | True | False |
