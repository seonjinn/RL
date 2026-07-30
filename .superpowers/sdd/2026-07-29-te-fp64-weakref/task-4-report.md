# Task 4 Report: NeMo-RL FP64 Transformer Engine Overlay Preflight

## Result

Implemented and verified the immutable Ptyche Transformer Engine FP64
weak-reference overlay preflight. Baseline and Transformer Engine CUDA-Graph
launchers now use the same read-only source mount and validate provenance
before Ray starts.

| Item | Value |
| --- | --- |
| Tested TE commit | `e707aa46869dc2aec08dfea25402e97a61d49fef` |
| Production TE commit | `6410a165444a7c063284246e29fcbb36ff019d18` |
| `utils.py` SHA256 | `39f7b26b8cf127e3ca104c0375c97ce4e6d047178f9d00836b92469b1c2e544b` |
| TE version | `2.15.0+42b84005` |
| Overlay source | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/TransformerEngine-fp64-weakref-20260729/transformer_engine/pytorch/utils.py` |
| Overlay target | `/root/.cache/uv/archive-v0/AdbVCNRp6JVFPo0e/transformer_engine/pytorch/utils.py` |

## Changed paths

- `experiments/cuda_graph/mamba_moe_te_graph_20260729/validate_te_fp64_overlay.py`
- `experiments/cuda_graph/mamba_moe_te_graph_20260729/profiles/ptyche.env`
- `experiments/cuda_graph/mamba_moe_te_graph_20260729/run_scope.sh`
- `tests/unit/experiments/test_mamba_moe_te_graph_launchers.py`
- `experiments/cuda_graph/mamba_moe_te_graph_20260729/README.md`

`pyrefly.toml` was inspected but not changed: its explicit `project-includes`
list does not include this experiment directory, so the new validator is not
within the project's current type-checked allow-list.

## RED evidence

The frozen repository lock is Linux-only, so the prescribed `uv run --frozen`
command cannot collect tests on the macOS development host. To preserve a valid
functional RED result, a temporary detached worktree at Task 4's base
`a9cb31d9c96f7ff80642b99115cae48d73203c61` received only the current launcher
test diff; it retained the base profile, launcher, and absent validator. The
temporary worktree was removed immediately after the check.

```console
uv run --no-project --with 'pytest>=8' python -m pytest -q \
  --confcutdir=tests/unit/experiments \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py \
  -k fp64_overlay
```

Expected functional RED result:

```text
FAILED ...test_fp64_overlay_provenance_is_identical_for_baseline_and_te_scopes
AssertionError: assert 'TE_FP64_WEAKREF_COMMIT:
e707aa46869dc2aec08dfea25402e97a61d49fef' in ...stdout
1 failed, 25 deselected
```

The base launcher returned successfully but its Ptyche `TEST_ONLY=1` output
lacked the required overlay commit, SHA256, source/target mount, and validator
invocation. This failure is therefore specific to the missing overlay wiring,
not to host or test infrastructure.

## GREEN verification

```console
uv run --no-project --with 'pytest>=8' python -m pytest -q \
  --confcutdir=tests/unit/experiments \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py
# 28 passed in 1.57s

ruff check \
  experiments/cuda_graph/mamba_moe_te_graph_20260729/validate_te_fp64_overlay.py \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py
# All checks passed!

git diff --check
# exit 0
```

The isolated pytest invocation intentionally avoids the repository's parent
test `conftest.py`, whose Ray/Torch dependencies cannot be resolved from the
Linux-only frozen lock on this host. It includes both the baseline/TE
provenance contracts and a subprocess check that a wrong SHA exits nonzero
before the fake CUDA allocator can run.

Manual dry-run confirmation also succeeded for the baseline and `attn` scope:

```console
TEST_ONLY=1 CLUSTER=ptyche bash .../scopes/00_baseline_no_cg.sh
TEST_ONLY=1 CLUSTER=ptyche bash .../scopes/17_attn.sh
```

Both printed the exact commit, SHA256, source, target, read-only mount, and
validator command.

## Self-review and concerns

- The validator uses top-level imports, hashes the resolved mounted
  `transformer_engine.pytorch.utils.__file__`, rejects version/SHA/mapping
  mismatches, verifies a CUDA FP64 weak reference preserves dtype, shape, and
  data pointer, and emits JSON provenance only after all checks pass.
- The validator runs in the existing head-node `SETUP_COMMAND` before
  `ray --version`; worker setup remains disabled because the shared mount and
  immutable image are identical on every node.
- No Slurm job was submitted for this task.
- The Linux-only frozen lockfile prevents the exact prescribed host test
  command from running on macOS. A temporary base worktree produced the valid
  missing-wiring RED result, then the full launcher suite was exercised in an
  isolated `uv` pytest environment. The GPU version/hash/CUDA positive-path
  check remains for the planned Ptyche integration gate.
