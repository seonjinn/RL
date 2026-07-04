# Task 2 Report: Patch and Stage the AngelSlim Runner

## Scope

- Created `experiments/vllm_024_dynamicsd/patches/angelslim_compact_result_transport.patch`
- Modified `experiments/vllm_024_dynamicsd/stage_extended_method_assets_in_container.sh`
- Modified `tests/test_vllm024_dynamicsd.py`

## Requirements Source

Command:

```bash
sed -n '1,260p' /Users/sna/Nemo-RL_Qwen3_Roadmap/.worktrees/vllm024-dynamicsd/.superpowers/sdd/task-2-brief.md
```

Output:

```text
### Task 2: Patch and Stage the AngelSlim Runner

**Files:**
- Create: `experiments/vllm_024_dynamicsd/patches/angelslim_compact_result_transport.patch`
- Modify: `experiments/vllm_024_dynamicsd/stage_extended_method_assets_in_container.sh`
- Modify: `tests/test_vllm024_dynamicsd.py`

**Interfaces:**
- Consumes: Task 1 transport module.
- Produces: patched `tools/dflash_benchmark.py` that gathers only compact CPU records and writes rank partial files before `_dist_gather`.

- [ ] **Step 1: Add failing staging assertions**

Assert the staging script copies `angelslim_dflare_transport.py`, applies
`angelslim_compact_result_transport.patch`, and that the patch replaces
`responses.append(response)` with compact records plus a pre-gather partial
write.

- [ ] **Step 2: Run the focused staging tests and confirm failure**

Run: `python3 -m pytest -q tests/test_vllm024_dynamicsd.py`

Expected: assertions fail because the new transport and patch are not staged.

- [ ] **Step 3: Add the minimal AngelSlim patch**

Import the staged transport module, compact each response after local text
decoding, write a rank partial JSON immediately after the local loop, gather
compact records, and keep the existing final result JSON fields unchanged.

- [ ] **Step 4: Validate patch application**

Run `patch --dry-run` against the pinned AngelSlim commit
`6a97dab2f17c0a3c031065329f092c4f61108a6f`, then run the focused pytest file.

Expected: dry-run and tests pass.
```

## Red Step

Added failing assertions to `tests/test_vllm024_dynamicsd.py` first:

- the extended-assets staging worker must mention
  `angelslim_compact_result_transport.patch`
- the staging worker must copy `angelslim_dflare_transport.py` into
  `${angelslim_source}/tools/`
- the new patch file must replace `responses.append(response)` with
  `responses.append(compact_response_map(response))`
- the patch must write `write_rank_partial(args.output_json, _dist_rank(), responses)`
  before `_dist_gather`

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Output:

```text
....F.F.........................
=================================== FAILURES ===================================
_________ test_extended_assets_stage_dry_run_is_pinned_and_lustre_only _________
E       assert 'angelslim_compact_result_transport.patch' in '#!/usr/bin/env bash\nset -euo pipefail\n...'

________ test_angelslim_compact_transport_patch_stages_cpu_only_results ________
E       FileNotFoundError: [Errno 2] No such file or directory:
E       '/Users/sna/Nemo-RL_Qwen3_Roadmap/.worktrees/vllm024-dynamicsd/experiments/vllm_024_dynamicsd/patches/angelslim_compact_result_transport.patch'

=========================== short test summary info ============================
FAILED tests/test_vllm024_dynamicsd.py::test_extended_assets_stage_dry_run_is_pinned_and_lustre_only
FAILED tests/test_vllm024_dynamicsd.py::test_angelslim_compact_transport_patch_stages_cpu_only_results
2 failed, 30 passed in 2.62s
```

## Implementation

Created a minimal AngelSlim patch that targets the already-patched benchmark
state produced by:

- `angelslim_benchmark_json.patch`
- `angelslim_fixed_length.patch`
- `angelslim_split_run_modes.patch`
- `angelslim_distributed_timeout.patch`

The new patch:

- imports `compact_response_map` and `write_rank_partial` from the staged
  transport helper
- converts each response map to CPU-only compact records after text decoding
- writes a rank-partial JSON immediately after the local loop when
  `--output-json` is set
- preserves the existing final result JSON payload fields by leaving the
  post-gather metrics and payload construction unchanged

Updated the staging worker to:

- install `/workspace/experiment/angelslim_dflare_transport.py` into
  `${angelslim_source}/tools/angelslim_dflare_transport.py`
- apply `angelslim_compact_result_transport.patch` once when the transport
  hook is not already present
- compile both the helper and `tools/dflash_benchmark.py`

## Green Step

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Output:

```text
................................                                         [100%]
32 passed in 9.05s
```

## Patch Dry-Run Validation

Verified that `/tmp/AngelSlim-dflare-analysis` is at the pinned commit:

Command:

```bash
git -C /tmp/AngelSlim-dflare-analysis rev-parse HEAD
```

Output:

```text
6a97dab2f17c0a3c031065329f092c4f61108a6f
```

Validated the new patch by replaying the existing benchmark patch stack on a
fresh temporary copy of the pinned source and then running `patch --dry-run`
for the new transport patch.

Command:

```bash
tmpdir=/tmp/angelslim-task2-verify
rm -rf "$tmpdir"
cp -R /tmp/AngelSlim-dflare-analysis "$tmpdir"
patch -p1 -d "$tmpdir" < experiments/vllm_024_dynamicsd/patches/angelslim_benchmark_json.patch
patch -p1 -d "$tmpdir" < experiments/vllm_024_dynamicsd/patches/angelslim_fixed_length.patch
patch -p1 -d "$tmpdir" < experiments/vllm_024_dynamicsd/patches/angelslim_split_run_modes.patch
patch -p1 -d "$tmpdir" < experiments/vllm_024_dynamicsd/patches/angelslim_distributed_timeout.patch
patch --dry-run -p1 -d "$tmpdir" < experiments/vllm_024_dynamicsd/patches/angelslim_compact_result_transport.patch
```

Output:

```text
patching file 'tools/dflash_benchmark.py'
patching file 'tools/dflash_benchmark.py'
patching file 'tools/dflash_benchmark.py'
patching file 'tools/dflash_benchmark.py'
patching file 'tools/dflash_benchmark.py'
```

Exit status: `0`

## Self-Review

Commands:

```bash
git diff --check -- experiments/vllm_024_dynamicsd/patches/angelslim_compact_result_transport.patch experiments/vllm_024_dynamicsd/stage_extended_method_assets_in_container.sh tests/test_vllm024_dynamicsd.py
git status --short
```

Outputs:

```text
git diff --check:
[no output]

git status --short:
 M experiments/vllm_024_dynamicsd/stage_extended_method_assets_in_container.sh
 M tests/test_vllm024_dynamicsd.py
?? experiments/vllm_024_dynamicsd/patches/angelslim_compact_result_transport.patch
?? experiments/vllm_024_dynamicsd/report/20260704_dflare_completed/
?? experiments/vllm_024_dynamicsd/report/20260704_vllm_native_completed/
?? experiments/vllm_024_dynamicsd/report/dflare_job_status_latest.csv
```

Notes:

- The owned changes are limited to the three requested Task 2 files.
- Existing untracked report artifacts under
  `experiments/vllm_024_dynamicsd/report/` were left untouched.

## Review Follow-Up

Reviewer items addressed:

- tightened the staging idempotency guard so it skips only when both
  `compact_response_map` and `write_rank_partial` are already present in
  `tools/dflash_benchmark.py`
- strengthened the focused test so it asserts the two guard markers and checks
  that the partial-write hunk appears before `_dist_gather` in the patch text

Commands:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Outputs:

```text
....................F...........                                         [100%]
=================================== FAILURES ===================================
________ test_angelslim_compact_transport_patch_stages_cpu_only_results ________
E       assert 'write_rank_partial\' "${angelslim_source}/tools/dflash_benchmark.py"' in '#!/usr/bin/env bash\nset -euo pipefail\n...'

=========================== short test summary info ============================
FAILED tests/test_vllm024_dynamicsd.py::test_angelslim_compact_transport_patch_stages_cpu_only_results
1 failed, 31 passed in 2.63s

................................                                         [100%]
32 passed in 2.47s
```

## Re-Review Follow-Up

Reviewer correction addressed:

- replaced the loose function-name idempotency guard with exact behavioral
  postconditions in `tools/dflash_benchmark.py`
- the staging worker now:
  - skips only when both
    `responses.append(compact_response_map(response))` and
    `write_rank_partial(args.output_json, _dist_rank(), responses)` exist
  - applies the patch only when neither marker exists
  - exits with a partial-patch error when exactly one marker exists
- strengthened the focused test to assert those exact marker strings, the
  three-state branch structure, and the partial-state rejection message

Commands:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Outputs:

```text
....................F...........                                         [100%]
=================================== FAILURES ===================================
________ test_angelslim_compact_transport_patch_stages_cpu_only_results ________
E       assert 'responses.append(compact_response_map(response))' in '#!/usr/bin/env bash\nset -euo pipefail\n...'

=========================== short test summary info ============================
FAILED tests/test_vllm024_dynamicsd.py::test_angelslim_compact_transport_patch_stages_cpu_only_results
1 failed, 31 passed in 2.40s

................................                                         [100%]
32 passed in 2.58s
```
