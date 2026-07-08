# Force-On-Policy-Ratio Benchmark Status

Status date: 2026-07-07 PDT

## Executive summary

The synchronous Llama 3.1 8B, Qwen3-30B-A3B, and Qwen3-32B control/treatment
pairs completed 20 training steps on Pre-Tyche. Setting
`loss_fn.force_on_policy_ratio=true` improved mean E2E tokens/s/GPU by 7.25%,
10.62%, and 12.13%, respectively. The improvement comes from skipping the
previous-policy logprob pass: mean policy-and-reference-logprob time fell by
41.64% to 50.10%, while generation and policy-training times were essentially
unchanged.

The corrected Async-1off matrix was submitted on 2026-07-07. At the 21:39 PDT
snapshot, both Llama variants, both Qwen3-30B-A3B variants, and the Qwen3-32B
control had completed setup without an OOM or fatal runtime error. Four jobs
had entered the training loop; the Qwen3-30B-A3B control had completed setup
and was filling its initial replay buffer. The Qwen3-32B force job remained
pending for priority. Async performance is not reported until the paired
20-step runs finish.

The first Llama and Qwen3-30B-A3B control attempts encountered a transient
Hugging Face Hub API `429 Too Many Requests` during vLLM worker construction.
They were resubmitted under distinct `_retry1` run keys after the rate-limit
window reset; both retries passed the failing point without a 429.

## Fixed identities

- Cluster: Pre-Tyche GB200-NVL36, partition `36x2-a01r`
- Account: `coreai_dlalgo_llm`
- NeMo-RL source branch: `sna/nemorl-main-pr3030-q235-20260701`
- NeMo-RL source SHA: `d4cfecf90db41cdf142629963b54b67ab479ab02`
- Source remote: `https://github.com/seonjinn/RL.git`
- Experiment harness branch:
  `sna/q30-q32-force-on-policy-benchmark-20260707`
- Harness commit at submission:
  `84d00bbe341ba1205265a0a407d436781bb56505`
- Container: `nemo_rl_nightly_20260630_0215.sqsh`
- Container SHA-256:
  `bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510`
- Paired global batch size: 2048
- Analysis steps: 2-9 and 11-19 (`n=17`); warm-up step 1 and
  validation-bearing steps 10 and 20 are excluded

Global batch size 2048 is required because `force_on_policy_ratio=true`
requires the training batch to equal `64 prompts * 32 generations`. All
control/treatment pairs use the same batch size and differ only in the force
flag after normalizing logging fields.

## Synchronous performance

All values are arithmetic means over the 17 analysis steps. Positive throughput
delta and negative time delta are improvements.

| Model | Nodes x GPUs | Step time, control -> force | Delta | E2E tokens/s/GPU, control -> force | Delta | Samples/s delta |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.1 8B | 2x4 | 68.202 -> 63.307 s | -7.18% | 4,335.1 -> 4,649.3 | +7.25% | +7.89% |
| Qwen3-30B-A3B | 4x4 | 196.702 -> 178.055 s | -9.48% | 2,115.4 -> 2,340.0 | +10.62% | +10.48% |
| Qwen3-32B | 4x4 | 310.435 -> 277.021 s | -10.76% | 1,360.3 -> 1,525.3 | +12.13% | +12.06% |

| Model | Generation time delta | Logprob time, control -> force | Logprob delta | Policy-training time delta | Weight-update time delta |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B | -1.02% | 11.174 -> 6.521 s | -41.64% | -1.39% | +11.39% (+0.068 s) |
| Qwen3-30B-A3B | -0.07% | 38.091 -> 20.155 s | -47.09% | -0.67% | -1.35% |
| Qwen3-32B | -0.53% | 62.938 -> 31.405 s | -50.10% | -0.40% | -0.22% |

Median checks agree with the means:

| Model | Median step time, control -> force | Median E2E tokens/s/GPU, control -> force | Median logprob time, control -> force |
|---|---:|---:|---:|
| Llama 3.1 8B | 66.550 -> 62.542 s | 4,415.5 -> 4,728.2 | 11.240 -> 6.437 s |
| Qwen3-30B-A3B | 196.066 -> 175.581 s | 2,124.8 -> 2,354.5 | 38.166 -> 20.295 s |
| Qwen3-32B | 307.503 -> 273.908 s | 1,365.3 -> 1,521.3 | 61.782 -> 30.900 s |

## Synchronous correctness

- All six jobs reached step 20 and exited `0:0`.
- All analyzed reward, loss, KL, probability-ratio, and performance scalars are
  finite; there are no NaN, Inf, OOM, CUDA, NCCL, Ray, or Python fatal
  signatures.
- Every step retained all 2,048 samples.
- Every treatment run logged exactly 20
  `Skipping prev_logprobs (force_on_policy_ratio=True)` markers; controls logged
  zero.
- Reference-policy logprob work remains present because the KL penalty is
  enabled.

| Model | Mean reward, control -> force | Mean generation KL, control -> force | Mean sampling importance ratio, control -> force |
|---|---:|---:|---:|
| Llama 3.1 8B | 0.36877 -> 0.36894 | 0.00030042 -> 0.00030041 | 1.000004 -> 1.000010 |
| Qwen3-30B-A3B | 0.52278 -> 0.52071 | 0.00188764 -> 0.00189098 | 0.999996 -> 1.000004 |
| Qwen3-32B | 0.52321 -> 0.52019 | 0.00094551 -> 0.00094653 | 1.000006 -> 1.000002 |

Qwen3-30B-A3B has an unresolved correctness limitation. Aggregate
`policy_kl_error` spikes occurred in both variants: control steps 6 and 16
reached 7,301.9 and 57.0, while force steps 5, 6, 8, and 18 reached 41.4,
149.9, 100.4, and 60.8. The runs remained finite and retained all samples
because `grpo.seq_logprob_error_threshold=null`, but the Qwen3-30B-A3B speedup
should not be treated as a clean correctness sign-off until these outliers are
understood. Llama 3.1 8B and Qwen3-32B did not show comparable aggregate
logprob-error spikes.

## Jobs and W&B

| Run | Job | State | W&B |
|---|---:|---|---|
| Qwen3-30B-A3B control | 2338296 | COMPLETED | [1uuwijgh](https://wandb.ai/nvidia/sna-force-on-policy-q30-q32-gb200/runs/1uuwijgh) |
| Qwen3-30B-A3B force | 2338297 | COMPLETED | [gmibxbxb](https://wandb.ai/nvidia/sna-force-on-policy-q30-q32-gb200/runs/gmibxbxb) |
| Qwen3-32B control | 2338298 | COMPLETED | [sphge25s](https://wandb.ai/nvidia/sna-force-on-policy-q30-q32-gb200/runs/sphge25s) |
| Qwen3-32B force | 2338299 | COMPLETED | [sasrtor4](https://wandb.ai/nvidia/sna-force-on-policy-q30-q32-gb200/runs/sasrtor4) |
| Llama 3.1 8B sync control | 2338825 | COMPLETED | [20mmvv5t](https://wandb.ai/nvidia/sna-force-on-policy-llama-qwen-async-gb200/runs/20mmvv5t) |
| Llama 3.1 8B sync force | 2338826 | COMPLETED | [omilek7u](https://wandb.ai/nvidia/sna-force-on-policy-llama-qwen-async-gb200/runs/omilek7u) |

## Corrected Async-1off run status

Snapshot: 2026-07-07 21:39 PDT, after monitoring every running or retried job
for at least five minutes.

| Run | Job | Snapshot state | Progress | W&B |
|---|---:|---|---|---|
| Llama 3.1 8B control, initial | 2339797 | FAILED (`1:0`) | Hugging Face API 429 before setup completed | [6w3kvdlo](https://wandb.ai/nvidia/sna-force-on-policy-llama-qwen-async-gb200/runs/6w3kvdlo) |
| Llama 3.1 8B control, retry 1 | 2339835 | RUNNING | Setup complete; Step 2/20 entered | [tjazqs5y](https://wandb.ai/nvidia/sna-force-on-policy-llama-qwen-async-gb200/runs/tjazqs5y) |
| Llama 3.1 8B force | 2339798 | RUNNING | Setup complete; Step 9/20 entered | [eofxqju1](https://wandb.ai/nvidia/sna-force-on-policy-llama-qwen-async-gb200/runs/eofxqju1) |
| Qwen3-30B-A3B control, initial | 2339799 | FAILED (`1:0`) | Hugging Face API 429 before setup completed | [hf69toot](https://wandb.ai/nvidia/sna-force-on-policy-llama-qwen-async-gb200/runs/hf69toot) |
| Qwen3-30B-A3B control, retry 1 | 2339836 | RUNNING | Setup complete; filling the step-0 replay buffer | [rlkwoyth](https://wandb.ai/nvidia/sna-force-on-policy-llama-qwen-async-gb200/runs/rlkwoyth) |
| Qwen3-30B-A3B force | 2339800 | RUNNING | Setup complete; Step 3/20 entered | [qjhqykan](https://wandb.ai/nvidia/sna-force-on-policy-llama-qwen-async-gb200/runs/qjhqykan) |
| Qwen3-32B control | 2339801 | RUNNING | Setup complete; Step 2/20 entered | [tbgaxlam](https://wandb.ai/nvidia/sna-force-on-policy-llama-qwen-async-gb200/runs/tbgaxlam) |
| Qwen3-32B force | 2339802 | PENDING | Priority; estimated start 2026-07-08 01:07 PDT | Not created |

The corrected topology is:

- Llama 3.1 8B: two allocated nodes, split into one training and one inference
  node, with no segment override.
- Qwen3-30B-A3B: four allocated nodes with `segment_size=2`, split into two
  training and two inference nodes. The scheduler and application segment
  values both equal 2.
- Qwen3-32B: eight allocated nodes, split into four training and four inference
  nodes, with no segment override.

## Original Async-1off terminal classification

No Async-1off speedup is reported because none of these jobs reached step 1.

| Pair | Jobs, control/force | Terminal state | First fatal signature |
|---|---|---|---|
| Llama 3.1 8B 2n4g | 2338827 / 2338828 | FAILED / FAILED | `num_nodes (1) must be divisible by segment_size (2)` |
| Qwen3-30B-A3B 4n4g | 2338829 / 2338830 | FAILED / FAILED | `num_nodes (2) must be divisible by segment_size (4)` |
| Qwen3-32B 8n4g | 2338831 / 2338832 | FAILED / FAILED | `num_nodes (4) must be divisible by segment_size (8)` |

The non-colocated Async recipes reserve half of the total nodes for inference,
so the training clusters contain 1, 2, and 4 nodes. The experiment contract
incorrectly validated `segment_size == total_nodes` without validating the
post-split training topology. Llama and Qwen3-32B normally inherit
`segment_size=null`; the harness changed them to invalid non-null values.
Qwen3-30B-A3B already inherits `segment_size=4` from its synchronous parent, so
the upstream Async recipe also needs a topology override on this source SHA.

The corrected run preserves `segment_size=null` for Llama and Qwen3-32B and
uses the common control/treatment override `segment_size=2` for
Qwen3-30B-A3B. Its preflight validator checks divisibility against the derived
training node count. These topology settings are identical within every
control/treatment pair.
