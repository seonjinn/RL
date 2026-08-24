# Benchmark status

## Current state

The fast-path harness is labeled `SWE trajectory-collection rollout-only` and
uses exact PR #3733 without PR #3243 eval semantics. Harness implementation and
dependency-free local contract tests are complete. The first baseline/DFlash
K5 canary pair was submitted but failed before model loading, so there are
still no performance results to report.

## Runtime recovery evidence

The failed Python diagnostics did not reproduce the production `ray.sub`
container contract:

- Job `6480428` used `--container-remap-root` without
  `--no-container-mount-home`; `/opt/nemo_rl_venv/bin/python` resolved through
  an image symlink to a Python path hidden by the mounted home.
- Job `6480634` mounted only that Python tree, which made the executable run,
  but `import torch` failed on `typing_extensions`.
- SquashFS metadata shows `typing_extensions.py` is an absolute symlink to
  `/root/.cache/uv/archive-v0/7rERAQWzyXqphCZd/typing_extensions.py`, and that
  target is present in the immutable image. The diagnostic's host-home mount
  masked the image cache, so adding individual packages would only patch a
  symptom.

The first recovery instead pinned `ray.sub` SHA256
`853564c6bfb0b430ee16c4eac1dfa0542db1922d75fec3e7d9f98b674bb0f81d`,
keeps its `--no-container-mount-home` behavior, removes the external Python
mount, forces inherited `UV_CACHE_DIR_OVERRIDE` empty, and installs a
fail-closed all-node import probe before Ray starts. A
failed head or worker setup now writes the shared `ENDED` signal and exits
before `ray start`. This contract is locally verified but still requires one
clean OCI Linux gate; it is not yet a benchmark result.

Compute preflight job `6483529` completed successfully and verified the source,
container, SWE data, target, and both drafter byte identities. Canary jobs
`6483818` (baseline) and `6483819` (DFlash K5) then failed identically in the
Ray driver before model loading: the image Python imported
`/opt/nemo-rl/nemo_rl` and the mounted PR #3733 entrypoint could not import
`shutdown_environments`. This was a source-selection bug, not a DFlash failure.
The recovery executed the mounted entrypoint with the pinned image Python and
set `PYTHONPATH` to the exact mounted checkout, eliminating the stale image
source. Replacement canaries `6483935` (baseline) and `6483936` (DFlash K5)
confirmed that source selection, then failed identically because the harness
also forced `NEMO_RL_PY_EXECUTABLES_SYSTEM=1`. The driver environment does not
contain vLLM, so both stopped in `_apply_vllm_patches` with
`ModuleNotFoundError: No module named 'vllm'` before model loading.

SquashFS metadata confirms the pinned image contains vLLM in the prebuilt async
vLLM actor environment under `/opt/ray_venvs`, not in
`/opt/nemo_rl_venv`. The current recovery preserves the mounted source for the
driver, restores actor-tier selection, points `NEMO_RL_VENV_DIR` at the image's
prebuilt actor environments, disables actor-venv rebuilds, and adds an all-node
`import vllm` probe using the exact async actor interpreter. This remains a
runtime recovery until replacement canaries pass; it is not a performance
result.

The same failed canaries exposed a harness-only completion defect: `ray.sub`
intentionally clears inherited `SLURM_*` variables before starting Ray, while
the completion wrapper later referenced `SLURM_JOB_ID` under `set -u`. The
wrapper now reads the scheduler-returned job ID from its exact, campaign-bound
submission record. This preserves the immutable `ray.sub` contract and lets a
successful canary write the evidence required to unlock the full matrix.

## Verified immutable inputs

| Input | Identity |
| --- | --- |
| PR #3733 head | `b580dd8927b88c996470d315e74d57bf0cb4090e` |
| Thinking target snapshot | `144afc2f379b542fdd4e85a1fcd5e1f79112d95d` |
| Target config SHA256 | `a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438` |
| DFlash config SHA256 | `3462e700ded08b7c26f37deb16725100bfb29dee2eb380f2e053169ac1f4dd52` |
| DFlash weight SHA256 | `1374271a8f4491aaf9365014d14b38050240e18f652a45e95a42615bb2b15bab` |
| DSpark config SHA256 | `9959d0ea5d0a85886b9d2c6b903872ea24905b9528725b4877b339f356a1f509` |
| DSpark weight SHA256 | `cf73aef4993090ff632b2ade82937ab4640a45556ebb20dde2df3a8f8af0d701` |
| SWE data SHA256 | `38434589e57ac4494052cf826f2eca24eea5d75b6889cf9e37fbe9c18dc95c1a` |
| SWE records | 500 |

The target config reports `qwen3_moe` and `Qwen3MoeForCausalLM`. The inherited
recipe explicitly selects `Qwen/Qwen3-30B-A3B-Thinking-2507`, enables thinking
for the tokenizer and vLLM generation, and uses the native `deepseek_r1`
reasoning parser and SWE tool-use chat template. Both draft configs match the
target vocabulary size (151936) and hidden size (2048), but definitive runtime
compatibility remains gated on the OCI canary.

## Pending gates

- Independent read-only review.
- Signed and DCO-compliant immutable experiment commit.
- Clean recursive OCI checkout and locked Linux test suite.
- Repeat the compute preflight at the source-selection recovery commit.
- Repeat `sbatch --test-only` and the two-arm canary at that commit.
- Monitor the successful replacement canary pair for at least five minutes.
- Five-arm full run and metric aggregation after the canary unlocks it.
