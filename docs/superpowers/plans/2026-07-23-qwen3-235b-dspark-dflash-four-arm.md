# Qwen3-235B DSpark and DFlash Four-Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, submit, and monitor four Qwen3-235B drafter-training arms that separate DSpark block length from DFlash continued-training data effects.

**Architecture:** A local experiment package owns immutable arm configuration, manifest construction, integrity reports, launcher contract validation, submission, and status collection. Local tests and commits are pushed first; AWS-DFW pulls the exact commit and uses the isolated patched Speculators worktree for two-node online training. GPU smoke gates precede full checkpoint-aware chains, and every state change is reflected in the canonical HTML report.

**Tech Stack:** Python 3.12, standard-library JSON/JSONL and hashing, pytest, Bash, SLURM, Speculators, PyTorch distributed training, vLLM target serving, AWS-DFW GB200.

## Global Constraints

- Target model: `Qwen/Qwen3-235B-A22B-Thinking-2507`.
- Speculators path: `/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/speculators-dspark-v3-fixes-20260723`.
- Speculators commit: `9b1a6200f2204663e8f4f1542d3a1f52f4d53d97`.
- AWS-DFW host: `aws-dfw-cs-001-login-01.nvidia.com`.
- SLURM account: `nemotron_sw_post`.
- Each GPU job uses two exclusive nodes and four GPUs per node.
- DS8 and DS16 consume the identical frozen v2 prompt manifest and split hashes.
- DF-PUBLIC and DF-HARD consume token masses that differ by at most 1%.
- No production job may bypass contamination, manifest-hash, source-commit, scheduler-preflight, or smoke gates.
- Do not edit installed vLLM wheels or site-packages in place.
- Commit and push local source before remote pull and submission.
- Monitor every newly running GPU job for at least five minutes.
- Record failed attempts and fixes in both the experiment journal and `docs/dflash_drafter_training.html`.

---

## File Map

- `experiments/dflash_four_arm/arm_matrix.json`: immutable arm parameters and remote paths.
- `experiments/dflash_four_arm/manifest_lib.py`: typed JSONL normalization, filtering, hashing, stratification, and token-mass comparison.
- `experiments/dflash_four_arm/build_manifests.py`: CLI that inventories the frozen prepared v2 data and builds DF-PUBLIC and DF-HARD manifests plus reports.
- `experiments/dflash_four_arm/validate_experiment.py`: fail-closed manifest, source, launcher, and remote-path contract validator.
- `experiments/dflash_four_arm/run_training.sbatch`: one parameterized two-node smoke or full training entrypoint.
- `experiments/dflash_four_arm/submit_four_arm.sh`: dry-run-first smoke and chained-full-run submission wrapper.
- `experiments/dflash_four_arm/collect_status.py`: bounded log and checkpoint collector that writes a machine-readable status ledger.
- `experiments/dflash_four_arm/README.md`: commands, lifecycle, result table, and failure journal.
- `tests/test_dflash_four_arm_manifests.py`: unit tests for filtering, stable hashes, stratification, and token matching.
- `tests/test_dflash_four_arm_contract.py`: tests for the arm matrix, launcher flags, fail-closed gates, and dry-run behavior.
- `docs/dflash_drafter_training.html`: canonical user-facing experiment status and results.

### Task 1: Freeze and Validate the Four-Arm Configuration

**Files:**
- Create: `experiments/dflash_four_arm/arm_matrix.json`
- Create: `tests/test_dflash_four_arm_contract.py`
- Create: `experiments/dflash_four_arm/validate_experiment.py`

**Interfaces:**
- Consumes: the approved design and fixed AWS paths.
- Produces: `load_matrix(path: Path) -> dict[str, object]` and `validate_matrix(matrix: dict[str, object]) -> list[str]`.

- [ ] **Step 1: Write the failing configuration-contract tests**

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "experiments/dflash_four_arm/arm_matrix.json"
VALIDATOR = ROOT / "experiments/dflash_four_arm/validate_experiment.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("four_arm_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arm_matrix_is_exact_and_dspark_data_is_shared() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert set(matrix["arms"]) == {"ds8", "ds16", "df_public", "df_hard"}
    assert matrix["arms"]["ds8"]["data_path"] == matrix["arms"]["ds16"]["data_path"]
    assert matrix["arms"]["ds8"]["block_size"] == 8
    assert matrix["arms"]["ds16"]["block_size"] == 16
    assert matrix["arms"]["ds8"]["decay_gamma"] == 4.0
    assert matrix["arms"]["ds16"]["decay_gamma"] == 8.0
    assert matrix["arms"]["df_public"]["learning_rate"] == 0.0001
    assert matrix["arms"]["df_hard"]["learning_rate"] == 0.0001
    assert matrix["arms"]["df_public"]["data_path"] == "output/data_df_public_100k"
    assert matrix["arms"]["df_hard"]["data_path"] == "output/data_df_hard_100k"


def test_validator_accepts_committed_matrix() -> None:
    validator = load_validator()
    matrix = validator.load_matrix(MATRIX)
    assert validator.validate_matrix(matrix) == []
```

- [ ] **Step 2: Run the tests and confirm they fail because the files do not exist**

Run:

```bash
pytest -q tests/test_dflash_four_arm_contract.py
```

Expected: collection or file-open failure naming `arm_matrix.json` or `validate_experiment.py`.

- [ ] **Step 3: Create the immutable arm matrix**

```json
{
  "schema_version": 1,
  "target_model": "Qwen/Qwen3-235B-A22B-Thinking-2507",
  "speculators_commit": "9b1a6200f2204663e8f4f1542d3a1f52f4d53d97",
  "remote_root": "/lustre/fsw/portfolios/nemotron/users/sna/dflash_training",
  "target_layer_ids": [1, 23, 46, 68, 91],
  "sequence_length": 16384,
  "max_anchors": 1024,
  "num_layers": 5,
  "draft_vocab_size": 32000,
  "sliding_window": 2048,
  "loss_fn": "kl_div",
  "arms": {
    "ds8": {
      "speculator_type": "dspark",
      "block_size": 8,
      "decay_gamma": 4.0,
      "learning_rate": 0.0006,
      "epochs": 2,
      "data_path": "output/data_v2mix",
      "save_path": "output/dspark_235b_v3_b8"
    },
    "ds16": {
      "speculator_type": "dspark",
      "block_size": 16,
      "decay_gamma": 8.0,
      "learning_rate": 0.0006,
      "epochs": 2,
      "data_path": "output/data_v2mix",
      "save_path": "output/dspark_235b_v3_b16"
    },
    "df_public": {
      "speculator_type": "dflash",
      "block_size": 8,
      "decay_gamma": 4.0,
      "learning_rate": 0.0001,
      "epochs": 1,
      "from_pretrained": "output/dflash_235bthink_v2",
      "source_manifest": "manifests/df_public_100k.jsonl",
      "data_path": "output/data_df_public_100k",
      "save_path": "output/dflash_235b_v3_public"
    },
    "df_hard": {
      "speculator_type": "dflash",
      "block_size": 8,
      "decay_gamma": 4.0,
      "learning_rate": 0.0001,
      "epochs": 1,
      "from_pretrained": "output/dflash_235bthink_v2",
      "source_manifest": "manifests/df_hard_100k.jsonl",
      "data_path": "output/data_df_hard_100k",
      "save_path": "output/dflash_235b_v3_hard"
    }
  }
}
```

- [ ] **Step 4: Implement fail-closed matrix validation**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARM_NAMES = {"ds8", "ds16", "df_public", "df_hard"}


def load_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("arm matrix must be a JSON object")
    return payload


def validate_matrix(matrix: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    arms = matrix.get("arms")
    if not isinstance(arms, dict) or set(arms) != ARM_NAMES:
        errors.append(f"arms must be exactly {sorted(ARM_NAMES)}")
        return errors
    if arms["ds8"].get("data_path") != arms["ds16"].get("data_path"):
        errors.append("DS8 and DS16 must share one prepared data path")
    expected = {
        "ds8": ("dspark", 8, 4.0, 0.0006),
        "ds16": ("dspark", 16, 8.0, 0.0006),
        "df_public": ("dflash", 8, 4.0, 0.0001),
        "df_hard": ("dflash", 8, 4.0, 0.0001),
    }
    for name, values in expected.items():
        actual = (
            arms[name].get("speculator_type"),
            arms[name].get("block_size"),
            arms[name].get("decay_gamma"),
            arms[name].get("learning_rate"),
        )
        if actual != values:
            errors.append(f"{name} parameters {actual!r} do not match {values!r}")
    if matrix.get("speculators_commit") != "9b1a6200f2204663e8f4f1542d3a1f52f4d53d97":
        errors.append("unexpected Speculators commit")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    errors = validate_matrix(load_matrix(args.matrix))
    print(json.dumps({"status": "pass" if not errors else "fail", "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the contract tests**

Run:

```bash
pytest -q tests/test_dflash_four_arm_contract.py
```

Expected: `2 passed`.

- [ ] **Step 6: Commit the configuration contract**

```bash
git add experiments/dflash_four_arm/arm_matrix.json \
  experiments/dflash_four_arm/validate_experiment.py \
  tests/test_dflash_four_arm_contract.py
git commit -s -m "experiment: define Qwen3 235B drafter matrix"
```

### Task 2: Build Deterministic, Contamination-Gated Manifests

**Files:**
- Create: `experiments/dflash_four_arm/manifest_lib.py`
- Create: `experiments/dflash_four_arm/build_manifests.py`
- Create: `tests/test_dflash_four_arm_manifests.py`

**Interfaces:**
- Consumes: v2 JSONL, Open-SWE-Traces rows, hard-example JSONL, and held-out fingerprints.
- Produces: `normalize_text(text: str) -> str`, `conversation_hash(row: dict[str, object]) -> str`, `filter_reason(row: dict[str, object], fingerprints: tuple[str, ...]) -> str | None`, `write_manifest(rows: list[dict[str, object]], path: Path) -> str`, and JSON integrity reports.

- [ ] **Step 1: Write failing tests for stable hashes and contamination blocking**

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "experiments/dflash_four_arm/manifest_lib.py"


def load_lib():
    spec = importlib.util.spec_from_file_location("four_arm_manifest_lib", LIB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conversation_hash_is_whitespace_stable() -> None:
    lib = load_lib()
    left = {"conversations": [{"role": "user", "content": "fix   this\\nnow"}]}
    right = {"conversations": [{"role": "user", "content": " fix this now "}]}
    assert lib.conversation_hash(left) == lib.conversation_hash(right)


def test_filter_blocks_astropy_id_and_phrase() -> None:
    lib = load_lib()
    fingerprints = ("astropy__astropy-12907", "separability matrix does not compute")
    id_row = {"instance_id": "astropy__astropy-12907", "conversations": []}
    phrase_row = {
        "instance_id": "safe__repo-1",
        "conversations": [{"role": "user", "content": "Separability matrix does not compute correctly"}],
    }
    assert lib.filter_reason(id_row, fingerprints) == "heldout_fingerprint"
    assert lib.filter_reason(phrase_row, fingerprints) == "heldout_fingerprint"


def test_token_mass_ratio_must_be_within_one_percent() -> None:
    lib = load_lib()
    assert lib.token_mass_matches(100_000, 100_900)
    assert not lib.token_mass_matches(100_000, 101_100)
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
pytest -q tests/test_dflash_four_arm_manifests.py
```

Expected: file-not-found error for `manifest_lib.py`.

- [ ] **Step 3: Implement normalization, filtering, hashes, and token matching**

```python
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def row_text(row: dict[str, Any]) -> str:
    parts = [str(row.get("instance_id", "")), str(row.get("repo", ""))]
    for message in row.get("conversations", []):
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
    return normalize_text(" ".join(parts))


def conversation_hash(row: dict[str, Any]) -> str:
    payload = [
        {
            "role": normalize_text(str(message.get("role", ""))),
            "content": normalize_text(str(message.get("content", ""))),
        }
        for message in row.get("conversations", [])
        if isinstance(message, dict)
    ]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def filter_reason(row: dict[str, Any], fingerprints: tuple[str, ...]) -> str | None:
    text = row_text(row)
    if any(normalize_text(item) in text for item in fingerprints):
        return "heldout_fingerprint"
    if "astropy" in normalize_text(str(row.get("repo", ""))):
        return "heldout_repository"
    return None


def token_mass_matches(left: int, right: int) -> bool:
    if min(left, right) <= 0:
        return False
    return abs(left - right) / min(left, right) <= 0.01


def write_manifest(rows: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
```

- [ ] **Step 4: Implement the manifest-builder CLI**

The CLI must:

- inspect `$B/output/data_v2mix/dataset_info.json` and `state.json` without rewriting the prepared dataset;
- assert `num_examples=850220` and dataset fingerprint `99f237d8d7e06e8c`;
- hash the eleven Arrow shards plus `dataset_info.json`, `state.json`, `d2t.npy`, `t2d.npy`, and `token_freq.pt`;
- stream `nvidia/Open-SWE-Traces`;
- render trajectory messages and tool calls into `conversations`;
- filter held-out IDs, astropy repository rows, normalized phrases, and duplicates;
- select DF-PUBLIC as 50,000 Python plus 50,000 non-Python OpenHands examples;
- select DF-HARD as the first 80,000 DF-PUBLIC examples plus 20,000 unique hard-input examples;
- cap hard-input `instance_id` frequency at four;
- write row counts, language/scaffold/length distributions, SHA256 values, and token-mass checks to `integrity_report.json`;
- exit nonzero if any required count, fingerprint, duplicate, cap, or token-mass gate fails.

Use this entrypoint and exact CLI:

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-data-path", type=Path, required=True)
    parser.add_argument("--hard-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-v2-rows", type=int, default=850220)
    parser.add_argument("--public-rows", type=int, default=100000)
    parser.add_argument("--hard-rows", type=int, default=20000)
    args = parser.parse_args()
    result = build_all_manifests(
        v2_data_path=args.v2_data_path,
        hard_jsonl=args.hard_jsonl,
        output_dir=args.output_dir,
        expected_v2_rows=args.expected_v2_rows,
        public_rows=args.public_rows,
        hard_rows=args.hard_rows,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1
```

- [ ] **Step 5: Add fixture-based builder tests**

Add tests that construct ten synthetic public rows, four hard rows, one duplicate, and one astropy row under `tmp_path`. Assert:

```python
assert report["status"] == "pass"
assert report["dspark"]["shared_manifest"]
assert report["drops"]["heldout_fingerprint"] == 1
assert report["drops"]["duplicate"] == 1
assert report["df_hard"]["max_instance_frequency"] <= 4
assert report["df_public"]["sha256"] != report["df_hard"]["sha256"]
assert report["token_mass_relative_difference"] <= 0.01
```

- [ ] **Step 6: Run manifest tests**

Run:

```bash
pytest -q tests/test_dflash_four_arm_manifests.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the manifest implementation**

```bash
git add experiments/dflash_four_arm/manifest_lib.py \
  experiments/dflash_four_arm/build_manifests.py \
  tests/test_dflash_four_arm_manifests.py
git commit -s -m "experiment: build gated drafter manifests"
```

### Task 3: Add Launcher and Dry-Run Submission Contracts

**Files:**
- Create: `experiments/dflash_four_arm/run_training.sbatch`
- Create: `experiments/dflash_four_arm/submit_four_arm.sh`
- Modify: `experiments/dflash_four_arm/validate_experiment.py`
- Modify: `tests/test_dflash_four_arm_contract.py`

**Interfaces:**
- Consumes: arm name, matrix, integrity report, and smoke/full mode.
- Produces: a validated Speculators command, SLURM job ID, and append-only `jobs.tsv`.

- [ ] **Step 1: Add failing launcher-contract tests**

```python
def test_launcher_is_fail_closed_and_uses_required_dspark_flags() -> None:
    text = (ROOT / "experiments/dflash_four_arm/run_training.sbatch").read_text()
    for token in (
        "set -euo pipefail",
        "--gpus-per-node=4",
        "--nodes=2",
        "validate_experiment.py",
        "--speculator-type",
        "--block-size",
        "--dflash-decay-gamma",
        "--markov-rank 256",
        "--enable-confidence-head",
        "--checkpoint-freq 0.25",
    ):
        assert token in text
    assert "|| true" not in text


def test_submit_wrapper_defaults_to_dry_run() -> None:
    text = (ROOT / "experiments/dflash_four_arm/submit_four_arm.sh").read_text()
    assert 'EXECUTE="${EXECUTE:-false}"' in text
    assert 'if [[ "$EXECUTE" != "true" ]]' in text
    assert "sbatch --test-only" in text
```

- [ ] **Step 2: Run the launcher tests and confirm failure**

Run:

```bash
pytest -q tests/test_dflash_four_arm_contract.py
```

Expected: file-not-found failure for the launcher.

- [ ] **Step 3: Implement the parameterized SLURM launcher**

The script must:

- use `#SBATCH --nodes=2`, `#SBATCH --ntasks-per-node=1`, `#SBATCH --gpus-per-node=4`, and `#SBATCH --exclusive`;
- accept `ARM`, `MODE`, `MATRIX`, and `INTEGRITY_REPORT`;
- validate all four before starting vLLM;
- verify the Speculators HEAD exactly;
- refuse an existing nonempty save path unless `MODE=resume`;
- launch Qwen3-235B on node 0 with TP=4 and hidden-state connector;
- launch four-rank training on node 1;
- use a dedicated prepared 128-example smoke dataset, `--epochs 1`, `--checkpoint-freq 1`, and a dedicated smoke save path for `MODE=smoke`;
- use the matrix epoch count and production save path for `MODE=full`;
- add DSpark-only Markov, confidence, and `sample_from_anchor` flags;
- add DFlash-only `--from-pretrained` for continuation arms;
- preserve each component's exit status instead of piping through `tail`;
- trap `TERM` and forward it to the training process.

Construct DSpark arguments exactly as:

```bash
MODEL_ARGS=(
  --speculator-type dspark
  --block-size "$BLOCK_SIZE"
  --sample-from-anchor
  --dflash-decay-gamma "$DECAY_GAMMA"
  --markov-rank 256
  --markov-head-type vanilla
  --enable-confidence-head
  --confidence-head-with-markov
  --confidence-head-alpha 1.0
)
```

Construct DFlash continuation arguments exactly as:

```bash
MODEL_ARGS=(
  --speculator-type dflash
  --block-size 8
  --no-sample-from-anchor
  --dflash-decay-gamma 4
  --from-pretrained "$REMOTE_ROOT/$FROM_PRETRAINED"
)
```

- [ ] **Step 4: Implement dry-run-first submission**

`submit_four_arm.sh` must render and run:

```bash
sbatch --test-only --export=ALL,ARM="$arm",MODE=smoke,MATRIX="$matrix",INTEGRITY_REPORT="$report" \
  "$launcher"
```

Only when `EXECUTE=true`, submit the four smoke jobs and append:

```text
submitted_at<TAB>arm<TAB>mode<TAB>job_id<TAB>dependency<TAB>source_commit
```

to `experiments/dflash_four_arm/jobs.tsv`. Do not submit full chains from this wrapper until all smoke rows have terminal `COMPLETED` status and checkpoint-save evidence.

- [ ] **Step 5: Extend validation for launcher and source contracts**

Add `validate_launcher(path: Path) -> list[str]` and remote CLI arguments:

```text
--matrix
--integrity-report
--launcher
--arm
--speculators-path
```

The command exits nonzero on a dirty Speculators worktree, wrong HEAD, missing manifest SHA, unexpected arm, token mismatch, contamination hit, or missing required launcher token.

- [ ] **Step 6: Run local tests and shell syntax checks**

Run:

```bash
pytest -q tests/test_dflash_four_arm_contract.py tests/test_dflash_four_arm_manifests.py
bash -n experiments/dflash_four_arm/run_training.sbatch
bash -n experiments/dflash_four_arm/submit_four_arm.sh
```

Expected: all pytest tests pass and both `bash -n` commands return zero.

- [ ] **Step 7: Commit launch contracts**

```bash
git add experiments/dflash_four_arm/run_training.sbatch \
  experiments/dflash_four_arm/submit_four_arm.sh \
  experiments/dflash_four_arm/validate_experiment.py \
  tests/test_dflash_four_arm_contract.py
git commit -s -m "experiment: add four-arm training launcher"
```

### Task 4: Add Status Collection and Experiment Journal

**Files:**
- Create: `experiments/dflash_four_arm/collect_status.py`
- Create: `experiments/dflash_four_arm/README.md`
- Modify: `tests/test_dflash_four_arm_contract.py`

**Interfaces:**
- Consumes: `jobs.tsv`, bounded `squeue`/`sacct` output, logs, and checkpoint directories.
- Produces: `status.json`, `status.md`, and append-only failure-journal entries.

- [ ] **Step 1: Add a failing parser test**

```python
def load_status_collector():
    path = ROOT / "experiments/dflash_four_arm/collect_status.py"
    spec = importlib.util.spec_from_file_location("four_arm_status_collector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_parser_preserves_failure_reason() -> None:
    collector = load_status_collector()
    row = collector.parse_sacct_row(
        "123|FAILED|1:0|00:02:13|gpu001|OutOfMemory"
    )
    assert row == {
        "job_id": "123",
        "state": "FAILED",
        "exit_code": "1:0",
        "elapsed": "00:02:13",
        "node_list": "gpu001",
        "reason": "OutOfMemory",
    }
```

- [ ] **Step 2: Implement bounded status collection**

`collect_status.py` must:

- parse all job IDs from `jobs.tsv`;
- issue one SSH command containing one `squeue` query and one batched `sacct -X -j 123,124,125`-shaped query;
- tail at most 200 lines per job log;
- detect `server up`, first training step, finite loss, backward completion, optimizer step, checkpoint path, and terminal status;
- write complete command stdout and stderr under `artifacts/command_outputs/`;
- write `status.json` and `status.md` atomically;
- never convert timeout or unreachable state into success.

- [ ] **Step 3: Write the experiment README**

Include:

- the four-arm table;
- local validation commands;
- manifest-build command;
- dry-run and execute commands;
- smoke-to-full promotion checklist;
- job ledger links;
- checkpoint and evaluation sections;
- a failure journal table with columns `time`, `arm`, `job`, `symptom`, `root cause`, `fix`, and `retry`.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest -q tests/test_dflash_four_arm_contract.py tests/test_dflash_four_arm_manifests.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit status tooling**

```bash
git add experiments/dflash_four_arm/collect_status.py \
  experiments/dflash_four_arm/README.md \
  tests/test_dflash_four_arm_contract.py
git commit -s -m "experiment: track four-arm training status"
```

### Task 5: Build Real Manifests on AWS-DFW

**Files:**
- Inventory remotely: `$B/output/data_v2mix`
- Generate remotely: `$B/experiments/dflash_four_arm/manifests/df_public_100k.jsonl`
- Generate remotely: `$B/experiments/dflash_four_arm/manifests/df_hard_100k.jsonl`
- Generate remotely: `$B/experiments/dflash_four_arm/manifests/integrity_report.json`
- Modify: `experiments/dflash_four_arm/README.md`

**Interfaces:**
- Consumes: `$B/v2_mix.jsonl`, `nvidia/Open-SWE-Traces`, and a broad disjoint hard-input JSONL.
- Produces: three immutable manifests and one passing report.

- [ ] **Step 1: Run all local verification and push**

```bash
pytest -q tests/test_dflash_four_arm_contract.py tests/test_dflash_four_arm_manifests.py
bash -n experiments/dflash_four_arm/run_training.sbatch
bash -n experiments/dflash_four_arm/submit_four_arm.sh
git diff --check
git push origin main
```

Expected: tests and syntax checks pass; push reports the committed HEAD on `origin/main`.

- [ ] **Step 2: Pull the exact roadmap commit on AWS-DFW**

Use one SSH command that runs `git pull --ff-only`, records `git rev-parse HEAD`, and refuses a dirty worktree. Do not copy uncommitted local files with `scp`.

- [ ] **Step 3: Resolve the hard-input source without deleting or rewriting data**

Run a bounded read-only inventory under `$B` for JSONL files containing `instance_id`, `repo`, and OpenHands conversations. Accept a source only if it covers more than the original seven training instances and contains no held-out astropy IDs. If no source passes, mark DF-HARD `blocked_missing_broad_hard_pool`; do not substitute the seven-instance ×30 pool.

- [ ] **Step 4: Build manifests in a CPU SLURM job**

Submit on the `cpu` partition with account `nemotron_sw_post`, 16 CPUs, 64 GB RAM, and a three-hour limit. The job runs:

```bash
python3 experiments/dflash_four_arm/build_manifests.py \
  --v2-data-path "$B/output/data_v2mix" \
  --hard-jsonl "$HARD_JSONL" \
  --output-dir "$B/experiments/dflash_four_arm/manifests" \
  --expected-v2-rows 850220 \
  --public-rows 100000 \
  --hard-rows 20000
```

- [ ] **Step 5: Validate generated artifacts**

Prepare the two continuation datasets and four 128-example smoke datasets with
the patched Speculators `scripts/prepare_data.py`. Use `--seq-length 16384`,
`--num-preprocessing-workers 32`, unique output directories, and the exact target
model. The smoke sources are deterministic first-128-row subsets of the frozen
v2 JSONL, DF-PUBLIC manifest, and DF-HARD manifest; DS8 and DS16 share one prepared
v2 smoke directory.

Run:

```bash
python3 experiments/dflash_four_arm/validate_experiment.py \
  --matrix experiments/dflash_four_arm/arm_matrix.json \
  --integrity-report "$B/experiments/dflash_four_arm/manifests/integrity_report.json" \
  --launcher experiments/dflash_four_arm/run_training.sbatch \
  --speculators-path "$B/speculators-dspark-v3-fixes-20260723"
sha256sum "$B"/output/data_v2mix/{data-*.arrow,dataset_info.json,state.json,d2t.npy,t2d.npy,token_freq.pt}
sha256sum "$B"/experiments/dflash_four_arm/manifests/*.jsonl
```

Expected: validator status `pass`, v2 row count 850,220, DF manifests 100,000 passes before token truncation, no held-out hit, DS manifest identity, hard per-instance cap at four, and DF token-mass difference at most 1%.

- [ ] **Step 6: Record the CPU job, hashes, counts, and any rejected source in README**

Commit and push the documentation update before GPU submission.

### Task 6: Submit and Monitor Four GPU Smoke Jobs

**Files:**
- Modify: `experiments/dflash_four_arm/jobs.tsv`
- Modify: `experiments/dflash_four_arm/README.md`
- Modify: `docs/dflash_drafter_training.html`

**Interfaces:**
- Consumes: passing manifests, clean exact Speculators HEAD, and scheduler preflight.
- Produces: four two-step smoke checkpoints or explicit blocked/failed states.

- [ ] **Step 1: Check scheduling with no state change**

Run `submit_four_arm.sh` with `EXECUTE=false`. Expected: four successful `sbatch --test-only` results, resolved account/partition/GPU shape, and no submitted job IDs.

- [ ] **Step 2: Submit smoke jobs**

Run:

```bash
EXECUTE=true MODE=smoke bash experiments/dflash_four_arm/submit_four_arm.sh
```

Expected: four numeric job IDs appended to `jobs.tsv`.

- [ ] **Step 3: Monitor for at least five minutes after each job enters RUNNING**

Poll with `collect_status.py`. Confirm:

- vLLM reaches application startup;
- prepared data opens;
- forward loss is finite;
- backward and optimizer step complete;
- metrics include DSpark acceptance/confidence fields for DS8 and DS16;
- a smoke checkpoint is saved;
- no OOM, NaN, stale hidden-state file, target-server failure, or wrong-source error appears.

- [ ] **Step 4: Handle DS16 OOM without altering DS8**

If DS16 alone OOMs at 1,024 anchors, record the failed job and create arm `ds16_m512` with a new save path and `max_anchors=512`. Re-run the matrix validator, commit, push, pull, scheduler-test, and smoke. Do not overwrite the original DS16 evidence.

- [ ] **Step 5: Update HTML and README with smoke evidence and failures**

Include exact job IDs, state, elapsed time, first-step loss, checkpoint proof, and each unsuccessful attempt. Label smoke metrics as operational evidence, not throughput results.

- [ ] **Step 6: Commit and push smoke status**

```bash
git add experiments/dflash_four_arm/jobs.tsv \
  experiments/dflash_four_arm/README.md \
  docs/dflash_drafter_training.html
git commit -s -m "experiment: record four-arm smoke results"
git push origin main
```

### Task 7: Promote Passing Smokes to Full Training Chains

**Files:**
- Modify: `experiments/dflash_four_arm/jobs.tsv`
- Modify: `experiments/dflash_four_arm/README.md`
- Modify: `docs/dflash_drafter_training.html`

**Interfaces:**
- Consumes: terminal smoke states with saved checkpoints.
- Produces: full training chains, checkpoint inventory, and per-epoch metrics.

- [ ] **Step 1: Validate promotion eligibility**

For each arm, require a `COMPLETED` smoke, finite losses, optimizer step, checkpoint save, correct source commit, and no unresolved error. A blocked arm remains blocked; it is not replaced by a different configuration under the same name.

- [ ] **Step 2: Submit full jobs with unique save paths**

Use `MODE=full`. Submit one initial job per eligible arm. If the four-hour interactive partition is selected, add checkpoint-aware `afterany` continuations and one `batch_long` finisher. Store every dependency in `jobs.tsv`.

- [ ] **Step 3: Monitor the first five running minutes**

Apply the same startup checks as the smoke plus resume-state inspection. Confirm no full run writes into a smoke directory or another arm's output.

- [ ] **Step 4: Collect matched training-efficiency evidence**

At each 0.25-epoch checkpoint record:

- global step and examples seen;
- target tokens and supervised draft positions;
- train and validation loss;
- position accuracy and DSpark accept length;
- step time, GPU-hours, and peak memory;
- DSpark confidence absolute error and cumulative bias.

- [ ] **Step 5: Update the report after every terminal transition**

Append failures, retry rationale, checkpoint decisions, and current best metrics. Do not report validation EAL as rollout throughput.

### Task 8: Run Held-Out Rollout Gates and Finalize the HTML

**Files:**
- Create: `experiments/dflash_four_arm/evaluation_matrix.json`
- Create: `experiments/dflash_four_arm/results.csv`
- Modify: `experiments/dflash_four_arm/README.md`
- Modify: `docs/dflash_drafter_training.html`

**Interfaces:**
- Consumes: promoted checkpoints and an immutable vLLM wheel with the merged DSpark serving fixes.
- Produces: matched K-sweep results and the final 2x decision.

- [ ] **Step 1: Validate the serving wheel before scheduling rollouts**

Create a clean Lyris vLLM source worktree from the deployed wheel's recorded base
commit `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`. Apply the patch-equivalent
merged changes from:

```text
#48524 a7d00ec051624e551ea822ec55d1113d117e47b7
#48639 642076d26c98aab899a6cc3dc948856d38c7551b
```

Build a new immutable wheel without modifying the installed
`vllm-0.25.0-cp38-abi3-manylinux_2_28_aarch64.whl`. Record the new wheel SHA256,
base commit, patch IDs, build command, container, and compiler metadata. Verify
Speculators config loading, `sample_from_anchor`, auxiliary `fc` width, one eager
request, one compiled request, and K-aware CUDA-graph capture. A failure blocks
DSpark evaluation without invalidating training results.

- [ ] **Step 2: Generate the exact K-sweep matrix**

Use:

```json
{
  "ds8": [3, 5, 7, 8],
  "ds16": [5, 8, 12, 16],
  "df_public": [3, 5, 7],
  "df_hard": [3, 5, 7]
}
```

Every row includes checkpoint SHA, target model, dataset hash, prompt count, generation seed, concurrency, graph-capture sizes, maximum sequence length, and runtime image.

- [ ] **Step 3: Run three-prompt operational smokes**

Reject checkpoint/K pairs with load, graph, timeout, or correctness failures. Do not use the three-prompt result for model ranking.

- [ ] **Step 4: Run matched held-out evaluation**

Use at least 20 valid trajectories for each promoted checkpoint/K pair. Collect completion tok/s, model-call time, draft/verify time, acceptance by position, mean accepted length, valid-rollout rate, reward, and failure class.

- [ ] **Step 5: Apply decision rules**

Declare the primary goal met only when a matched arm reaches at least 174 tok/s without reward or valid-rollout regression. Compare the winner against 87 tok/s non-spec, 115.83 tok/s v1, and 133.21 tok/s v2.

- [ ] **Step 6: Final verification and commit**

Run:

```bash
pytest -q tests/test_dflash_four_arm_contract.py tests/test_dflash_four_arm_manifests.py
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

path = Path("docs/dflash_drafter_training.html")
parser = HTMLParser()
parser.feed(path.read_text(encoding="utf-8"))
parser.close()
print("HTML_PARSE_OK")
PY
git diff --check
```

Expected: tests pass, `HTML_PARSE_OK`, and no whitespace errors.

Commit:

```bash
git add experiments/dflash_four_arm/evaluation_matrix.json \
  experiments/dflash_four_arm/results.csv \
  experiments/dflash_four_arm/README.md \
  docs/dflash_drafter_training.html
git commit -s -m "experiment: report Qwen3 235B drafter matrix"
git push origin main
```
