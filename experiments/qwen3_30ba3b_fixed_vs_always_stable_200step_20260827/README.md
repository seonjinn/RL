# Qwen3-30B-A3B stable fixed versus always-online drafter study

This isolated fallback experiment measures how the online-drafter trade-off
changes for Qwen3-30B-A3B over 200 Math GRPO steps. It deliberately runs on the
previously validated product lineage rather than depending on cadence changes
from latest main.

## Immutable lineage

- Product source: `/home/sna/nemorl-q30-fixed-always-product-20260827`
- Product SHA: `4ee518b5dc2ed16f75e31876b477ea5ecf7d8c9b`
- Harness source: `/home/sna/nemorl-q30-fixed-always-harness-20260827`
- Durable results: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_fixed_vs_always_stable_200step_20260827`
- Target model: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B`
- DFlash drafter: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dflash/exported-checkpoint-25391`
- DSpark drafter: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dspark/exported-checkpoint-25391`

The two drafter paths are the user-designated Qwen3-30B-A3B base-model assets.
They are checked with an exact safetensors state-dict gate before scheduler
validation and again inside each job.

## Four matched K5 arms

| Arm | Generation SpecDec | Drafter training |
|---|---|---|
| `dflash-fixed` | DFlash K5 checkpoint enabled | Frozen: `policy.draft.enabled=false`, `optimizer=null` |
| `dflash-always` | Same DFlash K5 checkpoint | Online update every policy step |
| `dspark-fixed` | DSpark K5 checkpoint enabled | Frozen: `policy.draft.enabled=false`, `optimizer=null` |
| `dspark-always` | Same DSpark K5 checkpoint | Online update every policy step |

The stable product schema has no cadence scheduler. The always arms therefore
use `policy.draft.enabled=true` with the draft optimizer, which is the stable
every-step online-training path. No arm contains `update_schedule` or
`cadence_runtime`.

All four arms keep the same Q30 workload: 4 nodes × 4 GPUs, TP2/EP8/PP1/CP1,
sequence packing enabled, target sequence parallel enabled, legacy data plane
(`data_plane.enabled=false`), global batch 512, 16 prompts × 32 generations per
step, 8K training/model context, max OSL 1024, and 200 steps. vLLM uses the
Triton MoE backend and PIECEWISE CUDA Graph capture sizes
`[1,2,4,8,12,16,24,32,40,48]`.

The initial four submissions (`6601785`, `6601786`, `6601789`, and `6601796`)
were terminated by the SLURM host-memory cgroup after reaching step 2–4. Their
allocation was limited to `ReqMem=3600G` across four nodes, and the failing
Ray step reported roughly 622–639 GiB resident memory. NCCL broken-pipe and Ray
actor errors appeared only after the first rank was killed. The corrected
launcher requests the full memory of every allocated node with `#SBATCH
--mem=0`; the rendered-job contract prevents this directive from being dropped
again.

The first `always` run is also an explicit external canary for stable
online-drafter training with `sequence_packing.enabled=true`. Local composition
proves the schema and matched topology, but runtime correctness and throughput
for online training plus packing are not claimed until that canary passes the
CUDA Graph, step-1, and step-2 gates.

## Submission contract

The launcher supports only the four allowlisted variants. Before an actual
submission, run the scheduler validation for each variant and retain its
SHA-bound receipt. Submission preflight requires a clean harness git worktree
at the exact recorded HEAD. The manifest, test-only receipt, and durable
submission receipt also bind SHA-256 digests for the launcher, checkpoint
checker, composition verifier, and selected config:

```bash
bash experiments/qwen3_30ba3b_fixed_vs_always_stable_200step_20260827/submit_qwen3_30ba3b_fixed_vs_always_200step.sh --test-only dflash-fixed
```

An actual submission requires that exact test-only receipt. It writes a
durable `submitting` receipt before invoking `sbatch`; a timeout, nonzero exit,
truncated output, multiple job IDs, pre-existing receipt, or dangling receipt
symlink fails closed and requires manual reconciliation. Any helper or config
mutation after scheduler validation invalidates the receipt before `sbatch`.
Credentials are read only from the submission environment and are never
written to configs, manifests, receipts, or reports.

Each job gates product SHA/cleanliness, recursive submodule cleanliness, W&B
authentication, a persistent driver Python, exact drafter state dict, composed
config, CUDA Graph capture, step 1, and step 2. The driver uses the immutable
container environment with
`UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync`; this keeps
`PY_EXECUTABLES.SYSTEM` valid on every Ray node instead of capturing a transient
`uv run --with` build-environment path. OCI CPU discovery is bypassed with
`CPUS_PER_WORKER=64`. Each node also copies the MCore Python package to a
job-unique `/raid/scratch` overlay and prepends it to `PYTHONPATH`, so runtime
compilation of `megatron/core/datasets/helpers_cpp` cannot dirty or race inside
the immutable shared `/home` product checkout.

## Analysis

W&B project: `sna-specdec`

W&B group: `q30ba3b-fixed-vs-always-stable-200step-20260827`

Use the collector in [reporting/README.md](reporting/README.md). It averages the
closed step window 3–200, exposes missing steps and valid counts per metric,
and compares always-online only with the same drafter's fixed arm. This matrix
does not include a no-SpecDec baseline, so its ratios measure online-training
impact rather than total SpecDec-versus-baseline speedup.

## Local verification

```bash
uv run --no-project --with pytest python -m pytest -q \
  experiments/qwen3_30ba3b_fixed_vs_always_stable_200step_20260827/tests \
  experiments/qwen3_30ba3b_fixed_vs_always_stable_200step_20260827/reporting/tests
uv run --no-project --with ruff ruff check \
  experiments/qwen3_30ba3b_fixed_vs_always_stable_200step_20260827
bash -n experiments/qwen3_30ba3b_fixed_vs_always_stable_200step_20260827/submit_qwen3_30ba3b_fixed_vs_always_200step.sh
```
