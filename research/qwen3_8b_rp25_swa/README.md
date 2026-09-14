# Qwen3-8B rp25-44000 sliding-window co-training

## Scope and source

Isolated branch: `codex/q8-rp25-online-20260914`, based on the successful cadence
runtime `a28df91a94b623f5108a2992ccac887cc8cbdaab`. Existing worktrees are unchanged.
This is DFlash/DSpark sliding-window attention support, not DFlash2 or stochastic
weight averaging support. No public PR has been opened for this change.

The supplied B8 exports use `dflash_config.use_swa=true`, window 2048, and
noncausal draft blocks. Training previously ignored that metadata. The reference
is ModelOpt `hf_dflash.py::_build_draft_attention_mask`, locally inspected at
`de4c00007` (q30t-ptv23-from-scratch-corpus-sdd). Its context predicate is:

```text
same sample/packed segment AND valid key AND
anchor + query_slot - window < context_key_position < anchor
```

Own valid draft-block keys remain bidirectional. Target-model attention is not
modified. Windowing is applied in both dense CPU attention and CUDA FlexAttention;
Flex candidate tiles are also bounded to the window, including backward metadata.
CP still gathers projected K/V globally; this change does not reduce CP traffic.

## Configuration

`policy.draft.sliding_window` is an optional positive integer, at least the
training block size. `null` inherits attention from the drafter's `config.json`;
without metadata it retains legacy full context. Explicit values must match
checkpoint metadata. Local paths and HF repositories with the specified revision
are supported; metadata loading never executes model code. Invalid JSON, invalid
windows, contradictory active window sizes/flags, and causal blocks fail before
weight loading. Missing config files permit legacy weight-only checkpoints;
network/authentication failures must not silently select full attention.

For these B8 models use DFlash `gamma=7` / DSpark `block_size=8` for training,
independently of serving `num_speculative_tokens=5`. Do not relabel a DFlash
gamma-5 training block as B8. No optimizer or refit payload schema changes are
introduced. The provider resolves metadata before constructing the shared body.

New exports under `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa/`:

- `sd2p3rp-q8b-base-ptv3rp25-dflash-b8-16n/exported-checkpoint-44000`
- `sd2p3rp-q8b-base-ptv3rp25-dspark-b8-16n/exported-checkpoint-44000`

The agreed subsequent study is 200 steps / 11 conditions: no-SpecDec baseline
plus DFlash and DSpark frozen, always, fixed-5, fixed-10, fixed-20. It has not been
submitted by this implementation gate. New results must be a separate New Draft
cohort, not silently mixed with old public B16/block7 drafters.

## Validation and limits

Local RED cases observed before their fixes:

- Public attention rejected `sliding_window` with TypeError.
- Both policy config types rejected the new field; metadata resolver was absent.
- DFlashBodyConfig rejected the new field.
- Explicit causal metadata without `use_swa=true` failed to raise; fixed separately.

Local CPU tests call real production modules. Since Mac lacks the full GPU
stack, package initializers importing optional GPU components are bypassed via
temporary namespace modules, not replaced attention/model implementations.
Body tests use locally installed MCore 0.19.0, NOT the pinned cluster MCore;
cluster testing is required before claiming runtime compatibility.

Test files:

- `tests/unit/models/policy/test_draft_attention_config.py`: metadata/config contract.
- `tests/unit/models/megatron/test_dflash_block_attention.py`: strict per-query
  boundaries, packed padding/segments, old full-prefix regressions, forward and
  all-five-input gradient parity, bounded candidate tiles, BF16 CUDA oracle case.
- `tests/unit/models/megatron/test_dflash_model.py`: real body gradient isolation
  and full-context regression suite.
- `tests/unit/models/megatron/test_draft_sliding_window_provider.py`: both real
  providers, tiny optimizer update, export/reload, inherited window, output parity.

`run_gate.sbatch` runs pinned-container provider and GPU attention tests. It
verifies container/source/bundle hashes, checks out the exact new commit into
node-local storage from the preserved old source archive, and verifies recursive
submodule pins. All environments/build caches are node-local; durable test results
and the source bundle reside on Lustre. It is not a Qwen3-8B rollout benchmark.

Remaining required runtime gates: actual new export load, online train→vLLM refit
→next rollout with attention semantics checked on the generation side;
checkpoint/optimizer/schedule resume; CP>1 and multi-node execution. CPU packed
mask tests alone do not establish those capabilities.

## Reproduce the CPU check

```bash
uv run --no-project --with torch --with numpy --with pytest --with pydantic \
  --with omegaconf --with megatron-core --with huggingface-hub==1.24.0 \
  python research/qwen3_8b_rp25_swa/run_cpu_tests.py
```

2026-09-14 local result: **85 passed, 13 deselected** (GPU/benchmark cases).
The existing cadence harness also passed **62 tests** using
`uv run --no-project python -m unittest discover -s research/qwen3_8b_draft_cadence_200step/tests -v`.
Ruff check/format, shell syntax and diff checks passed. Pyrefly reported zero
errors for the new metadata resolver. These do not replace pinned-container tests.

Initial cluster gate 7150337 verified the immutable inputs and source checkout,
then failed before container launch: `srun: command not found`. The gate now uses
OCI's explicit `/cm/local/apps/slurm/25.11/bin/srun` path. No SWA runtime failure
was observed in that attempt because no model code executed.

Retry 7150387 runs source `cdbaa176722b6a45e394367856a410000be90512`, account
`nemotron_sw_post`, partition `batch`, one 4-GPU node, 90-minute maximum. It started
at 2026-09-14 20:51:11 UTC. Results are under
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/cdbaa1767/`.
The later HF offline-exception refinement is CPU-tested against the lockfile's
`huggingface-hub==1.24.0`, but is not in that already-running job's source snapshot.
It preserves `LocalEntryNotFoundError` rather than replacing it with AttributeError;
only an actual remote missing-file response permits a metadata fallback.
