# Qwen3-30B-A3B DeepScaleR long-rollout comparison

User-approved on September 15, 2026. This is a separate dataset cohort, not an
unchanged upstream performance recipe. The existing DAPO launcher is untouched.

## Design and gate

Use the built-in `DeepScaler` loader and `hf_math_verify`, as in NeMo-RL's
DeepScaleR GRPO recipes. Only dataset, verifier, and identifying metadata change
relative to the corresponding DAPO40K arm. Node-local HF/W&B caches and a dataset
preflight are operational additions. Grader self-checks are not accuracy results.

| Setting | Value |
|---|---|
| Target | Qwen/Qwen3-30B-A3B Base |
| GBS | 2048 = 128 prompts × 16 responses |
| Input / output / total caps | 2048 / 38912 / 40960 |
| Training | 8 × 4 GB200; TP2 / EP8 / CP4 / PP1; packing enabled |
| Generation | vLLM 0.25.1; TP1; BF16; flashinfer_trtllm; FULL_AND_PIECEWISE |
| Drafter | Base PTV3-SWA B8 exported-checkpoint-44000; frozen |
| First gate | Baseline, no max_num_seqs override, 3 policy steps |
| Follow-up | 20 steps each: Baseline default; DFlash K5 S64; DSpark K5 S64 |
| W&B | nvidia/sna-specdec; q30-deepscaler-gbs2048-40k-20260915 |

The initial gate uses batch/4h. Twenty-step runs retain the DAPO launcher's
batch_long/8h allowance until timing supports a shorter allocation. Checkpointing
is disabled, matching that frozen cohort. A time-limited run is incomplete.

Before submitting the follow-up, require dataset_check.json status=passed,
successful three-step training and refit, finite timing/reward metrics, and an
inspection of actual output lengths, truncation and repetition. Do not treat
large output length alone as useful reasoning or guarantee a 70% rollout share.
All three production arms begin from the original target, not the gate checkpoint.

Use Steps 3–20 for final averages, preserving the existing DAPO results separately.
Report generation and E2E times, output length distribution, reward, entropy and
KL diagnostics. Default-baseline versus S64 SpecDec is an operational tuning
comparison, not an isolation of the speculative algorithm alone. Do not attribute
cross-dataset reward differences to model quality; the tasks and graders differ.

## Commands

Run after commit/push and remote ff-only pull, with W&B credentials in the
environment, never in scripts or receipts:

```bash
Q30_DEEPSCALER_ACCOUNT=nemotron_n3_post Q30_DEEPSCALER_MAX_STEPS=3 bash experiments/q30_deepscaler_20260915/submit.sh --submit baseline default
# After the gate passes:
Q30_DEEPSCALER_ACCOUNT=nemotron_n3_post Q30_DEEPSCALER_MAX_STEPS=20 bash experiments/q30_deepscaler_20260915/submit.sh --submit baseline default
Q30_DEEPSCALER_ACCOUNT=nemotron_n3_post Q30_DEEPSCALER_MAX_STEPS=20 bash experiments/q30_deepscaler_20260915/submit.sh --submit dflash_k5 64
Q30_DEEPSCALER_ACCOUNT=nemotron_n3_post Q30_DEEPSCALER_MAX_STEPS=20 bash experiments/q30_deepscaler_20260915/submit.sh --submit dspark_k5 64
```

Every submission first runs sbatch --test-only and saves the rendered script and
receipt under `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-deepscaler-20260915/`.

Local validation: four launcher/config tests passed, including the fully resolved
four-arm gate/production configuration matrix. The original DAPO launcher passed
its eight regression tests and remains byte-for-byte unchanged. Ruff, bash syntax,
and diff checks passed. Dataset/grader execution remains a compute-node gate.

## Submission receipt

At 2026-09-15 22:59 UTC, job **7178109** was submitted on OCI-HSG, account
`nemotron_n3_post`, partition `batch`, 8 nodes × 4 GPUs, four-hour limit, three
steps. Runtime source at submission: `f7be36ed2`; remote ff-only pull succeeded.
The scheduler's test-only planning ID7178108 is not an actual submitted job.

Artifact directory:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-deepscaler-20260915/Qwen3-30BA3B-DeepScaler40K-Baseline-Sdefault-3step-r4-20260915T225905Z`.

Saved there: job.sbatch, test-only.txt, submission.txt; execution will add source
and submodule metadata, dataset_check.json, and driver logs. At22:59UTC the job
was **PENDING / Priority**, without a firm start time. No GPU or dataset gate
result exists yet. Five-minute running-start monitoring is still outstanding.
The three20-step follow-ups have NOT been submitted; they require the gate and
metric checks above. No automatic submission daemon has been installed.
