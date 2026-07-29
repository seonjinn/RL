# MXFP8 Shmoo Production A/B Design

## Goal

Measure the end-to-end NeMo-RL rollout benefit of offline-shmoo-qualified
dense MXFP8 tactics against FlashInfer TRTLLM's default tactic selection for
Qwen3-30B-A3B on OCI-HSG GB200.

## Comparison Contract

Both arms use the same custom vLLM 0.20.2 source, direct
`flashinfer_trtllm` dense GEMM path, adaptive layout policy, model, prompts,
seed, container, topology, and measurement window.

- Baseline arm: load the package-relative trace-bootstrap manifest. It fixes
  `gemm_backend=trtllm`, `layout=adaptive`, `switch_m=256`,
  `direct_trtllm=true`, and empty 8x4/128x4 tactic tables. Every eligible
  shape therefore uses FlashInfer TRTLLM runner tactic `-1`.
- Shmoo arm: load the package-relative qualified manifest with the identical
  policy and 106 exact-shape tactics selected by offline shmoo.

The config filename, config SHA256, and resulting tactic source are the only
intentional execution differences. This isolates shmoo lookup-table efficacy;
the baseline is not stock vLLM's potentially different layout/backend policy.

## Immutable Identities

- NeMo-RL source: the pushed full commit containing this experiment contract
- Custom vLLM source: the pushed full commit containing both package-relative
  Qwen manifests; wheel provenance, runtime overlay Git `HEAD`, and launcher
  pin must all match that commit
- vLLM version: `0.20.2`
- FlashInfer version: `0.6.8.post1`
- Container SHA256:
  `32f07be22293d9a3979e8ba04772ad48a8157dad04fd92577063ed4e07ab1493`
- Baseline manifest:
  `qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json`
- Baseline manifest SHA256:
  `3c9f2be89e9053df62d07b937bbbf6f1d4bce39867825cda940271762708a447`
- Qualified manifest:
  `qwen3_30ba3b_tp1_v0202_qualified.json`
- Qualified manifest SHA256:
  `2baf01def8887db693c35b3070571ab7bb4e72ebfcf30c9fd8b587a3b7c9b2a2`
- Runtime overlay:
  a commit-named clean Git worktree overlaid with the exact custom wheel
- Model: `Qwen/Qwen3-30B-A3B`
- Tensor parallel size: `1`
- Nodes and GPUs: `4 nodes × 4 GPUs`

The launcher validates these identities before model execution. Manifest
names and hashes are experiment constants rather than required submission
environment variables.

The trace-bootstrap manifest is added as tracked vLLM package data before the
final wheel is built. The qualified manifest is already tracked. The custom
wheel builder must list both files in its provenance metadata, and the runtime
overlay must reproduce their exact SHA256 values.

## Measurement Protocol

The production suite runs three sequential matched repeats:

1. baseline repeat 1
2. shmoo repeat 1
3. baseline repeat 2
4. shmoo repeat 2
5. baseline repeat 3
6. shmoo repeat 3

Each job runs one in-process warmup step followed by 20 measured steps. The
parser discards the warmup step. Both arms enable the same dispatch tracing.
The baseline trace must contain only direct TRTLLM dispatches with
`tactic=-1`. The shmoo trace must hit every qualified exact-shape tactic and
must not fall back for a qualified shape.

The W&B project is fixed to `sna_mxfp8_kernel_test`. Run names include the arm
and repeat number so baseline and shmoo runs are unambiguous.

## Launcher and Parser Changes

`experiments/mxfp8_adaptive_rollout/run_ab.sh` will:

- pin the new custom vLLM commit and runtime overlay;
- fix both manifest hashes in the experiment contract;
- default `MEASURE_STEPS` to `20` while retaining `WARMUP_STEPS=1`;
- pass `logger.wandb.project=sna_mxfp8_kernel_test`;
- name runs by baseline/shmoo arm and repeat;
- load the empty bootstrap manifest for baseline and the qualified manifest
  for shmoo;
- validate both package-relative manifest files and SHA256 values;
- record the selected manifest hash in each arm's metadata.

`experiments/mxfp8_adaptive_rollout/parse_results.py` will validate that:

- both arms share all immutable provenance and resolved configuration except
  the exact manifest filename;
- baseline uses the expected empty manifest and its expected SHA256;
- shmoo uses the expected qualified manifest and its expected SHA256;
- baseline runtime records are direct TRTLLM default-tactic dispatches;
- shmoo runtime records satisfy qualified tactic coverage.

Existing command names may remain `original` and `adaptive` for compatibility,
but all user-visible W&B names and reports will label them
`baseline-no-shmoo-trtllm` and `shmoo-qualified`.

The custom vLLM branch will add
`vllm/model_executor/kernels/linear/mxfp8/tactic_configs/`
`qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json` as tracked package data,
then publish a new deterministic wheel and commit-named runtime overlay.

## Testing and Execution Gates

Changes follow test-first development. Unit tests first demonstrate failure
for the new baseline contract, 20-step default, fixed W&B project, overlay
mapping, manifest hashes, and run naming. After the minimal implementation,
the focused experiment tests and lint checks must pass.

Execution then proceeds through these gates:

1. launcher scheduling check with `sbatch --test-only`;
2. one-repeat, one-measured-step compiled baseline/shmoo smoke;
3. verification that both arms used direct `flashinfer_trtllm` and that the
   shmoo arm used qualified tactics;
4. full three-repeat, 20-measured-step production A/B;
5. parse matched rollout generation time, generation tokens/s, total step
   time, and wall time with median and per-repeat ratios.

No end-to-end speedup claim is made until all three production pairs complete
and their provenance and runtime dispatch coverage pass.
