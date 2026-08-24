# Benchmark status

## Current state

The fast-path harness is labeled `SWE trajectory-collection rollout-only` and
uses exact PR #3733 without PR #3243 eval semantics. Harness implementation and
dependency-free local contract tests are complete.
No OCI-HSG GPU canary or full benchmark has been submitted, and there are no
performance results to report yet.

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

The recovery instead pins `ray.sub` SHA256
`853564c6bfb0b430ee16c4eac1dfa0542db1922d75fec3e7d9f98b674bb0f81d`,
keeps its `--no-container-mount-home` behavior, removes the external Python
mount, forces inherited `UV_CACHE_DIR_OVERRIDE` empty, and installs a
fail-closed all-node import probe before Ray starts. A
failed head or worker setup now writes the shared `ENDED` signal and exits
before `ray start`. This contract is locally verified but still requires one
clean OCI Linux gate; it is not yet a benchmark result.

## Verified immutable inputs

| Input | Identity |
| --- | --- |
| PR #3733 head | `b580dd8927b88c996470d315e74d57bf0cb4090e` |
| Thinking target snapshot | `144afc2f379b542fdd4e85a1fcd5e1f79112d95d` |
| Target config SHA256 | `a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438` |
| DFlash config SHA256 | `3462e700ded08b7c26f37deb16725100bfb29dee2eb380f2e053169ac1f4dd52` |
| DSpark config SHA256 | `9959d0ea5d0a85886b9d2c6b903872ea24905b9528725b4877b339f356a1f509` |
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
- Compute-node container/runtime identity probe under exact
  `--no-container-mount-home` semantics and SHA256.
- FairShare selection and `sbatch --test-only` for both canary arms.
- Two-arm canary plus at least five minutes of filtered monitoring.
- Five-arm full run and metric aggregation after the canary unlocks it.
