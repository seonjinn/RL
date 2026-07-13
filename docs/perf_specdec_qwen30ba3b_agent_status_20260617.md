# Qwen3-30B-A3B NeMo-RL SpecDec Submission Status

Checked local trackers/docs/scripts plus OCI-HSG and Lyris Slurm/job artifacts for Qwen3-30B-A3B only.

## Direct Performance Recipe Basis

- Sync basis: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml`
- Async-1off basis: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml`
- OCI-HSG direct async tracker: `latest_qwen30ba3b_async1off_recipe_direct_pard_k3_jobs.txt`
- No maintained async direct-recipe submit script was found next to the tracker; only the tracker and Slurm logs were found.

## Existing OCI-HSG Jobs

| Variant | Job ID | State | Config basis | Notes |
| --- | ---: | --- | --- | --- |
| sync baseline | 3198446 | COMPLETED | `grpo-qwen3-30ba3b-4n4g.yaml` | async engine and async GRPO disabled |
| sync PARD K3 | 3198447 | COMPLETED | `grpo-qwen3-30ba3b-4n4g.yaml` | adds vLLM `speculative_config` with `amd/PARD-Qwen3-0.6B`, K=3 |
| sync baseline, GBS2048 | 3207492 | COMPLETED | `grpo-qwen3-30ba3b-4n4g.yaml` | paired baseline for later GBS2048 PARD retry |
| sync PARD K3, GBS2048 | 3207978 | COMPLETED | `grpo-qwen3-30ba3b-4n4g.yaml` | force-venv-rebuild retry; completed 20-step run |
| async-1off baseline | 3207260 | CANCELLED by 150081 | `grpo-qwen3-30ba3b-4n4g-async-1off.yaml` | direct recipe; cancelled after 29m50s |
| async-1off PARD K3 | 3207261 | CANCELLED by 150081 | `grpo-qwen3-30ba3b-4n4g-async-1off.yaml` | adds only PARD K3 vLLM speculative config and omits generation logprobs; cancelled after 29m50s |

## Math-RL Performance Recipe Suite

These Qwen3-30B-A3B Math-RL jobs also inherit the Qwen3-30B-A3B performance recipe through the submit script and add measurement/specdec/runtime overrides:

| Method | Job ID | State | Notes |
| --- | ---: | --- | --- |
| baseline | 3334218 | COMPLETED | 20/20 steps |
| PARD | 3333526 | COMPLETED | 20/20 steps, K=5 |
| Eagle3 | 3333528 | COMPLETED | 20/20 steps, K=3 |
| suffix | 3333715 | COMPLETED | 20/20 steps, K=32 |
| PARD2 8B | 3333527 | FAILED | failed before completed steps due target-dimension issue |

## Active Slurm And Lyris

- OCI-HSG had SWE-RL Qwen3-30B-A3B jobs pending (`3365630`-`3365634`, plus suffix `3365678`), but those are SWE-RL ctx40k jobs and are not the direct performance sync/async-1off recipe runs above.
- Lyris had no active or archived NeMo-RL Qwen3-30B-A3B sync/async-1off artifacts found in the checked locations. Lyris Qwen3-30B-A3B hits were standalone vLLM jobs, not NeMo-RL recipe jobs.

## Gap

- Sync: already submitted and completed for PARD K3 direct-recipe runs.
- Async-1off: already submitted as direct-recipe baseline/PARD K3 jobs, but both were cancelled. If the requirement is only "submitted", async-1off is not missing. If the requirement is a usable completed async-1off result, async-1off still needs a clean resubmit.

## Safest Next Action If A Completed Async-1off Result Is Required

Do not modify shared launcher files. The safest path is to create a run-specific copied submit script from the exact command lines in `slurm-3207260.out` and `slurm-3207261.out`, then submit a fresh baseline/PARD K3 pair from the OCI-HSG worktree:

```bash
ssh oci-hsg-cs-001-vscode-02
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606/experiments/eagle3_online
```

Before submitting, re-check:

```bash
sacct -X -j 3207260,3207261 -o JobID,JobName%90,State,ExitCode,Elapsed,Start,End,NodeList -P
squeue -u "$USER" -h -o "%i|%j|%T|%R" | grep -E "qwen30ba3b.*async1off|3207260|3207261" || true
```

