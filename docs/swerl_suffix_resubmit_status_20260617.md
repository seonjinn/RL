# SWE-RL Suffix K=32 Resubmit Status

Checked: `2026-06-17T14:41:20-07:00`

## Submission

- Submitted fresh suffix job: `3365678`
- State after one poll: `PENDING`, reason `Priority`
- Host/account/partition: `oci-hsg-cs-001-vscode-02`, `nemotron_n3_post`, `batch`
- Target model: `/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-30B-A3B-Thinking-2507`
- Launcher: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh`
- Remote repo: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613`

## Command Basis

The manifest row for suffix was `poll_existing`, so I submitted the equivalent launcher command directly with a unique suffix:

`20260617_oci_hsg_swerl_qwen30ba3b_suffix_resubmit_k32_ctx40k_vllm40k_arcticpy313_ray254_py313_agentd_r1`

Basis was completed suffix job `3351394`, reusing the same Qwen30 target, SWE-RL remote repo, suffix launcher, Arctic site, Ray/Python shape, and ctx40k/vLLM40k Hydra overrides. Existing non-suffix jobs `3365630`-`3365634` were observed pending and not duplicated.

## Config

- Method/K: `suffix`, `K=32`
- Sampling: `temperature=1.0`, `top_p=1.0`, `top_k=-1`
- OSL/context: `max_new_tokens=40960`, `policy.max_total_sequence_length=40960`, `vllm max_model_len=40960`
- vLLM caps: `max_num_batched_tokens=65536`, `max_num_seqs=64`
- Training: `max_steps=1`, `GBS=64`, `PPS=2`, `GPP=32`
- Geometry: `TOTAL_NODES=16`, `TRAIN_NODES=8`, `GEN_NODES=8`, `VLLM_TP=1`, `HYBRIDEP=0`
- Source site: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/.container_cache/arctic-inference-0.1.1-py313`

## Poll Snapshot

```text
squeue: 3365678|20260617_oci_hsg_swerl_qwen30ba3b_suffix_resubmit_k32_ctx40k_vllm40k_arcticpy313_ray254_py313_agentd_r1|PENDING|(Priority)|0:00|4:00:00|N/A|
sacct:  3365678|20260617_oci_hsg_swerl_qwen30ba3b_suffix_resubmit_k32_ctx40k_vllm40k_arcticpy313_ray254_py313_agentd_r1|PENDING|0:0|00:00:00|Unknown|Unknown
```

## Next Parsing Target

Primary:

`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260617_oci_hsg_swerl_qwen30ba3b_suffix_resubmit_k32_ctx40k_vllm40k_arcticpy313_ray254_py313_agentd_r1/suffix_step1/3365678-logs/ray-driver.log`

Fallback while pending or before Ray log materializes:

`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260617_oci_hsg_swerl_qwen30ba3b_suffix_resubmit_k32_ctx40k_vllm40k_arcticpy313_ray254_py313_agentd_r1/suffix_step1/slurm-3365678.out`
