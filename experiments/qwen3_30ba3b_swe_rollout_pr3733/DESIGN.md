# Qwen3-30B-A3B SWE rollout-only benchmark design

## Objective

Measure trajectory-collection rollout performance on the official NeMo-Gym
SWE1 pivot workload for a matched five-arm matrix: no-speculation baseline,
DFlash K5/K7, and DSpark K5/K7. The experiment uses exact NVIDIA-NeMo/RL PR #3733 head
`b580dd8927b88c996470d315e74d57bf0cb4090e`. It is explicitly labeled
`SWE trajectory-collection rollout-only` and does not claim PR #3243
generation-only eval semantics or prior 30B GPU validation.

## Immutable inputs

- Speculative-decoding PR: `b580dd8927b88c996470d315e74d57bf0cb4090e`
- Target: Qwen3-30B-A3B-Thinking-2507 snapshot
  `144afc2f379b542fdd4e85a1fcd5e1f79112d95d`
- Target `config.json` SHA256:
  `a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438`
- DFlash `config.json` SHA256:
  `3462e700ded08b7c26f37deb16725100bfb29dee2eb380f2e053169ac1f4dd52`
- DSpark `config.json` SHA256:
  `9959d0ea5d0a85886b9d2c6b903872ea24905b9528725b4877b339f356a1f509`
- SWE1 source:
  `nvidia/Nemotron-RL-Super-Training-Blends@08e1de58d3c8748c1b28e645df85c224f0b25021/swe1.jsonl`
- First-500 subset SHA256:
  `252692abb5ca3a8a891c5f2546add485af2ff8403675b9f6bc7bc2be84073d39`
  with exactly 500 NeMo-Gym JSONL records
- Runtime source commit and container SHA256 are mandatory launch inputs. The
source must be a clean checkout whose history contains the exact PR head.

## Container runtime bootstrap

The immutable image owns the driver `/opt/nemo_rl_venv` runtime and
framework-specific worker environments under `/opt/ray_venvs`. Its Python
executable and some uv-installed modules are absolute symlinks into the image's
`/root/.local/share/uv/python` and `/root/.cache/uv` trees. Consequently, the
runtime contract forbids a host-home mount: the pinned `ray.sub` hash includes
`--no-container-mount-home`, and the experiment adds no host Python mount.

`SETUP_COMMAND` runs on every Ray head and worker before `ray start`. It checks
that `/opt/nemo_rl_venv/bin/python` is executable and imports `nemo_rl`,
`omegaconf`, `pytest`, `ray`, `torch`, and `typing_extensions`. This makes an
incomplete or masked image runtime fail before cluster startup instead of
surfacing as a later actor failure. A second probe uses the exact prebuilt
`VllmAsyncGenerationWorker` interpreter and requires `import vllm`. The
scheduler points `NEMO_RL_VENV_DIR` at `/opt/ray_venvs`, disables forced
rebuilds, and leaves `NEMO_RL_PY_EXECUTABLES_SYSTEM=0` so each actor retains its
dependency tier.

## Matched methodology

The workload is inherited, unchanged, from
`examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml` through PR #3733's
`grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-tp2pp2-rollout-only-specdec.yaml`.
The full benchmark changes only the speculative configuration, the CUDA Graph
capture list derived for that arm, and a matched alias that registers the
legacy agent name embedded in official SWE1 rows. Baseline uses the same
overlay with `speculative_config=null`; it is not a different recipe. The
target and tokenizer are both overridden to the exact pinned Thinking
snapshot.

DSpark with vLLM 0.25.1 at TP2 additionally requires
`disable_custom_all_reduce=true`: the original K5/K7 jobs completed DSpark
graph progress but never reached KV-cache sizing or engine initialization.
This recovery is isolated to the DSpark overlays and must first pass both K5
and K7 canaries. If final cross-method reporting needs an exactly matched
collective backend, baseline and DFlash must be rerun with the same flag; the
DSpark-only recovery result is otherwise labeled conservative and not used as
an unqualified apples-to-apples speedup.
PR #3733's `run_grpo_nemo_gym.py` takes the overlay's
`env.nemo_gym.is_trajectory_collection=true` branch. It initializes no optimizer,
collects the validation trajectories, and preserves the native two-node/four-GPU
allocation, including one four-GPU non-colocated generation node at TP2.

Every full arm uses the same 500 prompts, temperature `1.0`, top-p `1.0`,
validation batch size `500` (the exact runner expands it to the dataset length),
one generation per validation prompt, maximum
sequence length `131072`, two GB200 nodes with four GPUs each, and generation
TP2 on one node. W&B writes to entity `nvidia`, project `sna-specdec`. The
trajectory-collection branch initializes no optimizer.

K is vLLM `num_speculative_tokens`: the number of draft tokens proposed per
decoding step. DFlash executes `K+1` draft-query tokens per request. This DSpark
checkpoint has no `dspark_bonus_anchor`, so it executes `K` draft-query tokens
per request; target verification still executes `K+1` tokens per request.

CUDA Graph capture sizes are derived from request buckets
`[1, 2, 4, 8, 16, 32, 64, 128, 256]`. The 500-prompt validation batch is
sharded across two TP2 vLLM replicas, yielding at most 250 requests per replica.
Baseline captures those
token counts. DFlash captures each bucket multiplied by `K+1`. DSpark captures
the sorted union of bucket multiples for `K` (draft) and `K+1` (target
verification). This covers the nominal per-replica runtime request range through
vLLM padding to the next captured request bucket.

## Safety and lifecycle

The launcher has separate planning, scheduler `test-only`, canary, and full
transitions. Planning has no scheduler side effect. Scheduler test-only invokes
`sbatch --test-only` and records a fingerprint of the exact allocation and
runtime contract. Submission refuses a missing or mismatched fingerprint and
writes a campaign-bound per-arm exclusive reservation before invoking `sbatch`.
The preflight record cryptographically binds the source, semantics-critical
source-file hashes (including `ray.sub`), manifest, container, per-file
checkpoint identities, and data. A scheduler action accepts only a freshly
re-rendered canonical plan, and
immediate metadata revalidation detects ordinary post-preflight filesystem
drift. Canary uses a
deterministic one-record subset and runs only baseline plus DFlash K5; this is
the only bounded workload override. Full submission remains locked until both
canary completion records and the campaign-bound monitor record for their exact
job IDs are successful. Submitted jobs are monitored together with one filtered
scheduler query per minute over at least five minutes.
The canonical state directory is `<output_root>/state`; scheduler writes are
restricted below the user's approved Lustre `experiments` prefix and cannot
overlap the source checkout.
