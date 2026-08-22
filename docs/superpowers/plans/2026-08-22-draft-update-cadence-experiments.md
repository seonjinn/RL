# Draft Update Cadence Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fail-closed matched experiment harness that determines whether any opt-in drafter cadence preserves generation and policy quality while reducing online overhead.

**Architecture:** A manifest-first launcher expands immutable matched triplets, validates exact source/config/container parity, and submits exactly one SLURM allocation/job per triplet. Inside that allocation, fixed, `always`, and candidate arms run sequentially on the same node in a replicate-rotated order. A separate analyzer joins canonical W&B rows on predeclared closed windows, treats matched replicates as sampling units, verifies raw rows and update/refit receipts fail-closed, and emits JSON, CSV, and Markdown decisions. Pilot elimination and 1000-step promotion remain separate stages.

**Tech Stack:** Python 3.12, pytest, SciPy, W&B API, JSON, CSV, Bash, SLURM/Pyxis on OCI-HSG GB200.

**Spec:** `docs/superpowers/specs/2026-08-22-online-drafter-efficiency-and-cadence-design.md`

## Global Constraints

- Begin only after `docs/superpowers/plans/2026-08-22-draft-update-cadence.md` Task 10 and its exact packed functional gate are terminal GREEN.
- Use DFlash K5, Qwen3-8B, packing enabled, CP1 first; repeat only pilot survivors at CP2.
- Each candidate is matched with fixed-drafter and `always` controls in one SLURM allocation on the same node. The three arms execute sequentially in recorded replicate-rotated order with identical immutable model/draft snapshot revisions, frozen ordered JSONL data bytes, GRPO/data/draft seeds, GBS, MBS, speculative K, image, CUDA Graph settings, and topology; a triplet is never split across allocations.
- Rotate arm order across replicates and record the execution order.
- `invariant_config_sha256` hashes only held-constant model/data/batch/packing/topology/CUDA fields and must match within a triplet. Each arm separately records `resolved_config_path` and `resolved_config_sha256` over its full resolved treatment config; those full hashes are expected to differ across fixed, `always`, and candidate arms and are never used as a parity field.
- Use structurally unique W&B run IDs and result roots derived from an explicit experiment UUID plus candidate/topology/replicate/kind; create result directories and manifests exclusively and never overwrite a prior arm.
- Full resolved Hydra configs contain the real treatment and provenance-bearing runtime values: immutable target/draft snapshot paths, content digests, and revisions; frozen data path/hash/revision with shuffle disabled; GRPO/data/draft seeds; K in both draft and generation config; CUDA Graph backend/mode; all adaptive defaults/thresholds; result/checkpoint/ledger receipt paths; W&B entity/project/ID; and Step-400 checkpoint requirements. Checkpoints use NeMo's native `step_<n>` names. Every long fresh arm sets `checkpointing.keep_top_k=null`, so all 100-step periodic checkpoints—including `step_400`—remain retained, and the cadence runtime writer additionally seals an immutable `checkpoint-runtime-step_400.json` receipt. A manifest-only field is never accepted as evidence that a runtime treatment was applied.
- Fixed controls keep speculative generation enabled with the immutable pretrained drafter but set online `policy.draft.enabled=false`. Their raw W&B contract requires generation/acceptance/policy metrics and forbids draft-loss/gradient/schedule-version values; the terminal schedule receipt declares those keys not applicable with zero counters and no decision ledger.
- Fixed sparse 10/40, fixed refit-only 10/40, adaptive min10/max100 run 300-step elimination pilots. Fixed sparse 100 and fixed refit-only 100 run 600-step elimination pilots.
- A 30-step cadence pilot is forbidden. Pilot point estimates may eliminate but never establish a production claim.
- A production claim requires 1000 steps and three matched replicates for fixed, `always`, and at most two promoted candidates.
- Fresh 1000-step runs use ten closed canonical W&B windows `_step=1..100` through `_step=901..1000`. A run resumed after completed Step 400 uses six windows `_step=401..500` through `_step=901..1000`. `_step=1001` is excluded.
- The analyzer uses canonical logged throughput and timing keys; it never reconstructs throughput from averaged times.
- Paired replicate differences are the sampling units. Individual training steps are never treated as independent replicates.
- Before submission: pull exact pushed head, prove recursive cleanliness/SHA/container/config, check FairShare, run `sbatch --test-only`, submit one job, and monitor at 60-second cadence for at least five minutes.
- FairShare evidence must use canonical UTC `Z` timestamps, be no older than 15 minutes at `sbatch`, contain finite nonnegative scores for every eligible account, and select the deterministic highest-score eligible account. Monitoring performs six checks spanning at least 300 seconds, queries both queue/accounting state, and exits nonzero on disappearance without accounting visibility, terminal failure, or startup-error log signatures.
- Runtime logs and W&B exports live under the declared result root and stay outside Git. Only harness code, schemas, immutable manifests, and reports are committed.
- Every implementation/report commit uses `git commit -S -s`; `git verify-commit HEAD` must pass.

## File Structure

- `research/qwen3_8b_draft_cadence/manifest.py`: typed arm and matched-replicate definitions.
- `research/qwen3_8b_draft_cadence/launch.py`: parity validation, unique IDs, SLURM script rendering, and test-only/submit commands.
- `research/qwen3_8b_draft_cadence/run_replicate.sh`: one immutable same-node sequential triplet execution inside the pinned container.
- `research/qwen3_8b_draft_cadence/analyze.py`: closed-window joins, paired statistics, non-inferiority gates, and fail-closed receipts.
- `research/qwen3_8b_draft_cadence/tests/test_contract.py`: manifest, parity, uniqueness, and event-count tests.
- `research/qwen3_8b_draft_cadence/tests/test_analysis.py`: formulas, windows, missing-key errors, and decisions.
- `research/qwen3_8b_draft_cadence/tests/controller_runtime_fixture.py`: real sync/single-controller runtime receipt integration driver.
- `research/qwen3_8b_draft_cadence/PILOT.md`: 300/600-step elimination receipt.
- `research/qwen3_8b_draft_cadence/LONG_VALIDATION.md`: 1000-step promotion or rejection receipt.

---

### Task 1: Build the immutable matched-arm launcher

**Files:**
- Create: `research/qwen3_8b_draft_cadence/__init__.py`
- Create: `research/qwen3_8b_draft_cadence/manifest.py`
- Create: `research/qwen3_8b_draft_cadence/launch.py`
- Create: `research/qwen3_8b_draft_cadence/run_replicate.sh`
- Create: `research/qwen3_8b_draft_cadence/tests/test_contract.py`
- Create: `research/qwen3_8b_draft_cadence/README.md`

**Interfaces:**
- Consumes: experiment UUID; exact pushed product and harness heads; recursive submodule SHAs; immutable container digest; resolved-config SHA256; model/draft revisions plus deterministic snapshot-tree SHA256 digests; K; dataset revision/order; seed; GBS/MBS; packing/SP/TP/CP; CUDA Graph settings; SLURM cluster/account/partition/resources; candidate; replicate; and result root.
- Produces: frozen `Arm`, `MatchedReplicate`, `build_pilot_matrix`, `build_cp2_survivor_matrix`, `build_long_matrix`, `MatchedReplicate.validate_parity()`, an exclusive canonical `manifest.json` with its SHA256, one unique W&B ID/result directory per arm, exactly one `sbatch` argv list per triplet, and executable `pilot`, `cp2-survivors`, and `long` CLI subcommands.

- [ ] **Step 1: Write RED matrix, uniqueness, parity, and event-count tests.**

```python
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import research.qwen3_8b_draft_cadence.launch as launch_module

from research.qwen3_8b_draft_cadence.launch import main
from research.qwen3_8b_draft_cadence.launch import _load_replicate, _replicate_json
from research.qwen3_8b_draft_cadence.launch import materialize_and_validate_resolved_config
from research.qwen3_8b_draft_cadence.launch import submit_replicate
from research.qwen3_8b_draft_cadence.launch import _submission_token
from research.qwen3_8b_draft_cadence.launch import validate_fairshare_receipt
from research.qwen3_8b_draft_cadence.manifest import (
    ExperimentSpec,
    MatchedReplicate,
    build_cp2_survivor_matrix,
    build_long_matrix,
    build_pilot_matrix,
    recompute_manifest_sha256,
    render_manifest,
    write_manifest_exclusive,
)


def experiment_spec(result_root: str = "/lustre/cadence-test") -> ExperimentSpec:
    base_resolved_config = {
        "grpo": {"max_num_steps": 1000, "seed": 1234},
        "checkpointing": {
            "enabled": True, "save_period": 100, "keep_top_k": None,
        },
        "logger": {"log_dir": "/immutable/overridden-per-arm"},
        "cadence_runtime": {"enabled": True, "result_dir": "/immutable/overridden-per-arm", "required_checkpoint_steps": []},
        "experiment_provenance": {},
        "data": {
            "shuffle": False,
            "train": {"dataset_name": "ResponseDataset", "data_path": "/immutable/data-order.jsonl", "seed": 1234},
        },
        "policy": {
            "model_name": "/immutable/model/qwen3-8b-rev",
            "draft": {
                "enabled": True,
                "model_name": "/immutable/draft/qwen3-8b-dflash-rev",
                "model_revision": "qwen3-8b-dflash-rev",
                "gamma": 5,
                "seed": 1234,
                "update_schedule": {"mode": "always"},
            },
            "generation": {
                "vllm_kwargs": {
                    "speculative_config": {"num_speculative_tokens": 5},
                    "compilation_config": {"backend": "eager", "cudagraph_mode": "PIECEWISE"},
                }
            },
            "megatron_cfg": {
                "tensor_model_parallel_size": 2,
                "context_parallel_size": 1,
                "sequence_parallel": True,
            },
            "sequence_packing": {"enabled": True},
        },
    }
    base_config_sha256 = hashlib.sha256(
        json.dumps(
            base_resolved_config, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return ExperimentSpec(
        experiment_id="018f47a6-7d91-7d4a-8cc9-6c4c8e58a101",
        product_repo="/workspace/NeMo-RL",
        harness_repo="/workspace/cadence-harness",
        product_head="1" * 40,
        harness_head="3" * 40,
        submodule_shas=(("3rdparty/Megatron-LM", "5" * 40),),
        container_image="/lustre/images/nemo-rl-sha256_2222.sqsh",
        container_digest="sha256:" + "2" * 64,
        base_config_path=(
            "/workspace/NeMo-RL/examples/configs/recipes/llm/"
            "grpo-qwen3-8b-1n8g-megatron-dflash-cadence.yaml"
        ),
        base_resolved_config=base_resolved_config,
        base_resolved_config_sha256=base_config_sha256,
        model_snapshot_path="/immutable/model/qwen3-8b-rev",
        draft_snapshot_path="/immutable/draft/qwen3-8b-dflash-rev",
        model_snapshot_sha256="a" * 64,
        draft_snapshot_sha256="b" * 64,
        data_order_path="/immutable/data/math-v1-data-order-6.jsonl",
        model_revision="qwen3-8b-rev",
        draft_revision="qwen3-8b-dflash-rev",
        dataset_revision="math-v1",
        data_order_sha256="6" * 64,
        k=5,
        global_batch_size=64,
        micro_batch_size=2,
        cuda_graph_settings=(
            ("backend", "eager"),
            ("cudagraph_mode", "PIECEWISE"),
        ),
        canonical_metric_keys=(
            ("total_step_time", "timing/train/total_step_time"),
            ("generation_tps", "train/generation/tokens_per_second"),
            ("acceptance_rate", "train/vllm/spec_acceptance_rate"),
            ("mean_accepted_length", "train/vllm/spec_mean_accepted_length"),
            ("mean_total_reward", "train/mean_total_reward"),
            ("gen_kl_error", "train/gen_kl_error"),
            ("draft_loss", "train/draft_loss"),
            ("draft_grad_norm", "train/draft_grad_norm"),
            ("applied_draft_version", "train/draft_schedule/applied_draft_version"),
        ),
        cluster="oci-hsg",
        account="account-a",
        partition="batch",
        nodes=1,
        gpus_per_node=4,
        time_limit="08:00:00",
        fairshare_checked_at="2026-08-22T12:00:00Z",
        fairshare_scores=(("account-a", 0.8), ("account-b", 0.4)),
        fairshare_eligible_accounts=("account-a", "account-b"),
        fairshare_selection_reason="highest eligible FairShare",
        wandb_entity="nvidia-nemo-rl",
        wandb_project="qwen3-draft-cadence",
        result_root=result_root,
    )


def test_pilot_matrix_has_required_steps_and_matched_controls() -> None:
    matrix = build_pilot_matrix(experiment_spec(), replicate_index=0)
    by_candidate = {
        replicate.candidate.name.rsplit("-candidate-r", 1)[0]: replicate
        for replicate in matrix
    }
    assert by_candidate["fixed_sparse_10"].candidate.max_steps == 300
    assert by_candidate["fixed_sparse_40"].candidate.max_steps == 300
    assert by_candidate["fixed_refit_10"].candidate.max_steps == 300
    assert by_candidate["fixed_refit_40"].candidate.max_steps == 300
    assert by_candidate["adaptive_10_100"].candidate.max_steps == 300
    assert by_candidate["fixed_sparse_100"].candidate.max_steps == 600
    assert by_candidate["fixed_refit_100"].candidate.max_steps == 600
    for replicate in matrix:
        assert [arm.kind for arm in replicate.arms] == ["fixed", "always", "candidate"]
        assert len({arm.wandb_id for arm in replicate.arms}) == 3
        assert len({arm.result_dir for arm in replicate.arms}) == 3
        for arm in replicate.arms:
            assert arm.product_head == "1" * 40
            assert arm.harness_head == "3" * 40
            assert len(arm.invariant_config_sha256) == 64
            assert arm.container_digest.startswith("sha256:")
            assert arm.sequence_packing is True
            assert arm.sequence_parallel is True
            assert arm.tensor_parallel_size == 2
            assert arm.context_parallel_size == 1


def test_full_configs_are_built_from_base_with_real_treatments() -> None:
    replicate = build_pilot_matrix(experiment_spec(), replicate_index=0)[0]
    by_kind = {arm.kind: arm for arm in replicate.arms}
    assert by_kind["fixed"].resolved_config["policy"]["draft"]["enabled"] is False
    assert "update_schedule" not in by_kind["fixed"].resolved_config["policy"]["draft"]
    assert by_kind["always"].resolved_config["policy"]["draft"]["update_schedule"] == {
        "mode": "always"
    }
    assert by_kind["candidate"].resolved_config["grpo"]["max_num_steps"] == 300
    for arm in replicate.arms:
        config = arm.resolved_config
        assert config["grpo"]["seed"] == arm.seed
        assert config["data"]["shuffle"] is False
        assert config["data"]["train"]["data_path"] == arm.data_order_path
        assert config["data"]["train"]["seed"] == arm.seed
        assert config["policy"]["model_name"] == arm.model_snapshot_path
        assert config["policy"]["draft"]["model_name"] == arm.draft_snapshot_path
        assert config["policy"]["draft"]["model_revision"] == arm.draft_revision
        assert config["policy"]["draft"]["gamma"] == arm.k
        assert config["policy"]["draft"]["seed"] == arm.seed
        assert config["policy"]["generation"]["vllm_kwargs"][
            "speculative_config"
        ]["num_speculative_tokens"] == arm.k
        assert config["policy"]["generation"]["vllm_kwargs"][
            "compilation_config"
        ] == dict(arm.cuda_graph_settings)
        assert config["experiment_provenance"] == {
            "model_revision": arm.model_revision,
            "draft_revision": arm.draft_revision,
            "model_snapshot_sha256": arm.model_snapshot_sha256,
            "draft_snapshot_sha256": arm.draft_snapshot_sha256,
            "dataset_revision": arm.dataset_revision,
            "data_order_sha256": arm.data_order_sha256,
        }
        assert config["logger"]["log_dir"] == arm.result_dir
        assert config["cadence_runtime"]["result_dir"] == arm.result_dir
        assert config["checkpointing"]["keep_top_k"] is None
    adaptive = next(
        rep.candidate
        for rep in build_pilot_matrix(experiment_spec(), replicate_index=0)
        if rep.candidate.schedule is not None
        and rep.candidate.schedule.mode == "adaptive"
    )
    assert adaptive.resolved_config["policy"]["draft"]["update_schedule"] == {
        "mode": "adaptive",
        "action": "sparse_update",
        "min_interval": 10,
        "max_interval": 100,
        "ewma_alpha": 0.1,
        "degradation_threshold": 0.02,
        "recovery_threshold": 0.01,
        "min_observations": 20,
        "max_burst_updates": 10,
    }


def test_parity_rejects_one_mismatched_control() -> None:
    replicate = build_pilot_matrix(experiment_spec(), replicate_index=0)[0]
    mismatched = replace(replicate.arms[1], seed=replicate.arms[0].seed + 1)
    with pytest.raises(ValueError, match="seed differs"):
        replace(
            replicate,
            arms=(replicate.arms[0], mismatched, replicate.arms[2]),
        ).validate_parity()


def test_fairshare_must_be_fresh_numeric_and_highest_eligible() -> None:
    now = datetime(2026, 8, 22, 12, 10, tzinfo=timezone.utc)
    validate_fairshare_receipt(experiment_spec(), now=now)
    with pytest.raises(ValueError, match="15 minutes"):
        validate_fairshare_receipt(
            replace(
                experiment_spec(), fairshare_checked_at="2026-08-22T11:00:00Z"
            ),
            now=now,
        )
    with pytest.raises(ValueError, match="malformed"):
        validate_fairshare_receipt(
            replace(experiment_spec(), fairshare_scores=(("account-a", float("nan")),)),
            now=now,
        )
    with pytest.raises(ValueError, match="highest eligible"):
        validate_fairshare_receipt(
            replace(experiment_spec(), account="account-b"), now=now
        )
    with pytest.raises(ValueError, match="UTC Z"):
        validate_fairshare_receipt(
            replace(
                experiment_spec(),
                fairshare_checked_at="2026-08-22T12:00:00+00:00",
            ),
            now=now,
        )


def test_full_treatment_hash_may_differ_but_invariant_hash_may_not() -> None:
    replicate = build_pilot_matrix(experiment_spec(), replicate_index=0)[0]
    assert len({arm.resolved_config_sha256 for arm in replicate.arms}) == 3
    bad = replace(replicate.arms[1], invariant_config_sha256="9" * 64)
    with pytest.raises(ValueError, match="invariant_config_sha256 differs"):
        replace(replicate, arms=(replicate.arms[0], bad, replicate.arms[2])).validate_parity()


def test_manifest_creation_is_exclusive(tmp_path: Path) -> None:
    manifest = render_manifest(
        build_pilot_matrix(experiment_spec(result_root=str(tmp_path)), replicate_index=0)
    )
    path = write_manifest_exclusive(tmp_path, manifest)
    with pytest.raises(FileExistsError):
        write_manifest_exclusive(tmp_path, manifest)
    assert json.loads(path.read_text())["manifest_sha256"] == manifest["manifest_sha256"]
    assert recompute_manifest_sha256(manifest) == manifest["manifest_sha256"]
    assert {
        arm["manifest_sha256"]
        for replicate in manifest["replicates"]
        for arm in replicate["arms"]
    } == {manifest["manifest_sha256"]}


def test_each_full_resolved_config_hash_is_revalidated(tmp_path: Path) -> None:
    arm = build_pilot_matrix(experiment_spec(str(tmp_path)), replicate_index=0)[0].arms[0]
    materialize_and_validate_resolved_config(arm)
    Path(arm.resolved_config_path).write_text("corrupt")
    with pytest.raises(ValueError, match="resolved config file changed"):
        materialize_and_validate_resolved_config(arm)


def test_snapshot_digest_binds_directory_contents_not_basename(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "qwen3-8b-rev"
    snapshot.mkdir()
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"first")
    first = launch_module._sha256_directory(snapshot)
    weights.write_bytes(b"second")
    second = launch_module._sha256_directory(snapshot)
    assert first != second


def write_spec(tmp_path: Path) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(asdict(experiment_spec(str(tmp_path)))))
    return path


def write_survivors(tmp_path: Path, source_topology: str = "packed-cp1") -> Path:
    spec = experiment_spec(str(tmp_path))
    source_matrix = (
        build_pilot_matrix(spec, replicate_index=0)
        if source_topology == "packed-cp1"
        else build_cp2_survivor_matrix(
            spec, survivors=("fixed_sparse_10",), replicate_index=0
        )
    )
    source_manifest = render_manifest(source_matrix)
    source_manifest_path = tmp_path / f"source-{source_topology}.json"
    source_manifest_path.write_text(json.dumps(source_manifest))
    analysis_path = tmp_path / f"source-{source_topology}-analysis.json"
    analysis_raw = json.dumps({
        "status": "pilot_complete",
        "topology": source_topology,
        "manifest_sha256": source_manifest["manifest_sha256"],
        "manifest_path": str(source_manifest_path.resolve()),
        "candidates": {},
        "cp2_survivors": ["fixed_sparse_10"],
    }, sort_keys=True).encode()
    analysis_path.write_bytes(analysis_raw)
    path = tmp_path / "survivors.json"
    path.write_text(json.dumps({
        "cp2_survivors": ["fixed_sparse_10"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_topology": source_topology,
        "source_analysis_receipt_path": str(analysis_path.resolve()),
        "source_analysis_receipt_size_bytes": len(analysis_raw),
        "source_analysis_receipt_sha256": hashlib.sha256(analysis_raw).hexdigest(),
    }))
    return path


def test_cp2_and_long_builders_are_concrete() -> None:
    spec = experiment_spec()
    cp2 = build_cp2_survivor_matrix(
        spec, survivors=("fixed_sparse_10",), replicate_index=0
    )
    assert len(cp2) == 1
    assert {arm.context_parallel_size for arm in cp2[0].arms} == {2}
    assert {
        arm.resolved_config["policy"]["megatron_cfg"]["context_parallel_size"]
        for arm in cp2[0].arms
    } == {2}
    long = build_long_matrix(
        spec, promoted=("fixed_sparse_10",), replicate_indices=(0, 1, 2)
    )
    assert len(long) == 4
    assert {replicate.candidate.max_steps for replicate in long} == {1000}
    assert sum(replicate.candidate.resume_after_step == 400 for replicate in long) == 1
    resumed = next(rep for rep in long if rep.candidate.resume_after_step == 400)
    assert len({Path(arm.result_dir).parent for arm in resumed.arms}) == 1
    assert resumed.resume_source_replicate_id is not None
    for arm in resumed.arms:
        assert arm.resolved_config["checkpointing"]["resume_from_checkpoint"] == arm.resume_checkpoint
        assert arm.resume_checkpoint.endswith("/checkpoints/step_400")
    pilot_dirs = {arm.result_dir for replicate in build_pilot_matrix(spec, replicate_index=0) for arm in replicate.arms}
    long_dirs = {arm.result_dir for replicate in long for arm in replicate.arms}
    assert pilot_dirs.isdisjoint(long_dirs)
    assert [
        tuple(arm.kind for arm in sorted(rep.arms, key=lambda arm: arm.execution_order))
        for rep in long if rep.candidate.resume_after_step is None
    ] == [
        ("fixed", "always", "candidate"),
        ("always", "candidate", "fixed"),
        ("candidate", "fixed", "always"),
    ]


def test_survivor_loader_recomputes_scientific_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = build_pilot_matrix(
        experiment_spec(str(tmp_path)), replicate_index=0
    )
    manifest = render_manifest(matrix)
    manifest_path = tmp_path / "pilot.json"
    manifest_path.write_text(json.dumps(manifest))
    analysis = {
        "status": "pilot_complete",
        "topology": "packed-cp1",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": str(manifest_path.resolve()),
        "candidates": {"fixed_sparse_10": {"terminal": True}},
        "cp2_survivors": ["fixed_sparse_10"],
        "production_claim": False,
    }
    analysis_path = tmp_path / "pilot-receipt.json"
    analysis_raw = json.dumps(analysis, sort_keys=True).encode()
    analysis_path.write_bytes(analysis_raw)
    survivors_path = tmp_path / "survivors.json"
    payload = {
        "cp2_survivors": ["fixed_sparse_10"],
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_manifest_path": str(manifest_path.resolve()),
        "source_topology": "packed-cp1",
        "source_analysis_receipt_path": str(analysis_path.resolve()),
        "source_analysis_receipt_size_bytes": len(analysis_raw),
        "source_analysis_receipt_sha256": hashlib.sha256(analysis_raw).hexdigest(),
    }
    survivors_path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.analyze.analyze_manifest",
        lambda *_args, **_kwargs: analysis,
    )
    assert launch_module._load_survivors(
        survivors_path, expected_source_topology="packed-cp1"
    ) == ("fixed_sparse_10",)
    payload["cp2_survivors"] = []
    survivors_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="recomputed scientific decision"):
        launch_module._load_survivors(
            survivors_path, expected_source_topology="packed-cp1"
        )


def test_resume_submission_waits_for_terminal_validated_step400_source(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = build_long_matrix(
        experiment_spec(str(tmp_path)),
        promoted=("fixed_sparse_10",),
        replicate_indices=(0, 1, 2),
    )
    manifest_path = tmp_path / "long.json"
    manifest_path.write_text(json.dumps(render_manifest(matrix)))
    entries = tuple(
        (
            path,
            _load_replicate(path),
        )
        for replicate in matrix
        for path in (_replicate_json(manifest_path, replicate),)
    )
    resumed = next(
        replicate for _path, replicate in entries
        if replicate.resume_source_replicate_id is not None
    )
    with pytest.raises(RuntimeError, match="not terminal"):
        launch_module.validate_resume_source_ready(entries, resumed)
    source = next(
        replicate for _path, replicate in entries
        if replicate.replicate_id == resumed.resume_source_replicate_id
    )
    for arm in source.arms:
        identity = launch_module._arm_receipt_identity(arm, "777")
        terminal = {
            **identity,
            "terminal": True,
            "exit_code": 0,
            "completed_policy_steps": arm.max_steps,
        }
        path = Path(arm.result_dir) / "terminal.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(terminal))
    (launch_module._replicate_root(source) / "terminal.json").write_text(json.dumps({
        "state": "terminal",
        "terminal": True,
        "exit_code": 0,
        "completed_arm_ids": [arm.name for arm in source.arms],
    }))
    validated = []
    monkeypatch.setattr(
        launch_module,
        "validate_product_runtime_receipts",
        lambda arm: validated.append(arm.name) or ({}, {}),
    )
    launch_module.validate_resume_source_ready(entries, resumed)
    assert len(validated) == 3


@pytest.mark.parametrize(
    ("subcommand", "extra"),
    [
        ("pilot", []),
        ("cp2-survivors", ["--survivors", "SURVIVORS"]),
        (
            "long",
            ["--survivors", "SURVIVORS", "--replicates", "3"],
        ),
    ],
)
def test_every_launcher_subcommand_executes_test_only(
    monkeypatch, tmp_path: Path, subcommand: str, extra: list[str]
) -> None:
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.run_command",
        lambda argv: commands.append(tuple(argv)) or SimpleNamespace(stdout="12345\n"),
    )
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.validate_source_container_config",
        lambda _spec: None,
    )
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.validate_arm_runtime",
        lambda _arm: None,
    )
    if subcommand != "pilot":
        monkeypatch.setattr(
            "research.qwen3_8b_draft_cadence.launch._load_survivors",
            lambda *_args, **_kwargs: ("fixed_sparse_10",),
        )
    spec = write_spec(tmp_path)
    survivors = write_survivors(
        tmp_path,
        "packed-cp2" if subcommand == "long" else "packed-cp1",
    )
    argv = [
        subcommand,
        "--spec",
        str(spec),
        "--manifest",
        str(tmp_path / f"{subcommand}.json"),
        "--test-only",
    ] + [str(survivors) if value == "SURVIVORS" else value for value in extra]
    assert main(argv) == 0
    assert commands
    assert all(command[:2] == ("sbatch", "--test-only") for command in commands)


def test_submission_is_one_job_per_triplet(monkeypatch, tmp_path: Path) -> None:
    submitted: list[tuple[str, ...]] = []

    def fake_popen(argv, **_kwargs):
        prepared = Path(argv[argv.index("--prepared") + 1])
        result = Path(argv[argv.index("--result") + 1])
        identity = json.loads(prepared.read_text())
        identity.pop("state")
        submitted.append(tuple(identity["argv"]))
        launch_module._write_json_exclusive_atomic(
            result, {"state": "helper_complete", **identity, "job_id": "12345"}
        )
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(launch_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.validate_source_container_config",
        lambda _spec: None,
    )
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.validate_arm_runtime",
        lambda _arm: None,
    )
    assert main([
        "pilot", "--spec", str(write_spec(tmp_path)),
        "--manifest", str(tmp_path / "pilot.json"), "--submit-next",
    ]) == 0
    assert len(submitted) == 1
    assert submitted[0].count("research/qwen3_8b_draft_cadence/run_replicate.sh") == 1


def test_unfiltered_wandb_normalization_drops_optional_nan_but_marks_required_na(
    tmp_path: Path,
) -> None:
    replicate = build_pilot_matrix(experiment_spec(str(tmp_path)), 0)[0]
    fixed = next(arm for arm in replicate.arms if arm.kind == "fixed")
    always = next(arm for arm in replicate.arms if arm.kind == "always")
    fixed_keys = dict(fixed.canonical_metric_keys)
    always_keys = dict(always.canonical_metric_keys)
    fixed_rows = launch_module.normalize_unfiltered_wandb_rows(
        [{
            "_step": 1,
            fixed_keys["generation_tps"]: float("nan"),
            fixed_keys["draft_loss"]: float("nan"),
        }],
        fixed,
    )
    always_rows = launch_module.normalize_unfiltered_wandb_rows(
        [{
            "_step": 1,
            always_keys["acceptance_rate"]: float("nan"),
            always_keys["draft_grad_norm"]: float("nan"),
            "train/draft_schedule/acceptance_ewma": float("nan"),
        }],
        always,
    )
    assert fixed_rows[0][fixed_keys["generation_tps"]] is None
    assert fixed_keys["draft_loss"] not in fixed_rows[0]
    assert always_rows[0][always_keys["acceptance_rate"]] is None
    assert always_keys["draft_grad_norm"] not in always_rows[0]
    assert "train/draft_schedule/acceptance_ewma" not in always_rows[0]
    json.dumps([*fixed_rows, *always_rows], allow_nan=False)


def install_fake_detached_submit_helper(monkeypatch, scheduler_calls: list[str]) -> None:
    def fake_popen(argv, **_kwargs):
        prepared = Path(argv[argv.index("--prepared") + 1])
        result = Path(argv[argv.index("--result") + 1])
        identity = json.loads(prepared.read_text())
        identity.pop("state")
        if not result.exists():
            scheduler_calls.append("sbatch")
            launch_module._write_json_exclusive_atomic(
                result,
                {"state": "helper_complete", **identity, "job_id": "777"},
            )
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(launch_module.subprocess, "Popen", fake_popen)


def test_submit_replicate_is_idempotent_after_exclusive_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    replicate = build_pilot_matrix(experiment_spec(str(tmp_path)), replicate_index=0)[0]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(render_manifest((replicate,))))
    replicate_json = _replicate_json(manifest_path, replicate)
    replicate = _load_replicate(replicate_json)
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.validate_arm_runtime",
        lambda _arm: None,
    )
    install_fake_detached_submit_helper(monkeypatch, calls)
    assert submit_replicate(replicate_json, replicate) == "777"
    assert submit_replicate(replicate_json, replicate) == "777"
    assert len(calls) == 1
    receipt = Path(replicate.candidate.result_dir).parent / "submission.json"
    assert json.loads(receipt.read_text())["job_id"] == "777"


def test_detached_helper_result_survives_launcher_receipt_crash_without_duplicate(
    monkeypatch, tmp_path: Path
) -> None:
    replicate = build_pilot_matrix(experiment_spec(str(tmp_path)), replicate_index=0)[0]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(render_manifest((replicate,))))
    replicate_json = _replicate_json(manifest_path, replicate)
    replicate = _load_replicate(replicate_json)
    sbatch_calls: list[str] = []
    monkeypatch.setattr(launch_module, "validate_arm_runtime", lambda _arm: None)
    install_fake_detached_submit_helper(monkeypatch, sbatch_calls)
    real_write = launch_module._write_json_exclusive_atomic

    def crash_before_receipt(path: Path, payload: object) -> None:
        if path.name == "submission.json":
            raise OSError("simulated crash after sbatch")
        real_write(path, payload)

    monkeypatch.setattr(launch_module, "_write_json_exclusive_atomic", crash_before_receipt)
    with pytest.raises(OSError, match="simulated crash"):
        submit_replicate(replicate_json, replicate)
    monkeypatch.setattr(launch_module, "_write_json_exclusive_atomic", real_write)
    assert submit_replicate(replicate_json, replicate) == "777"
    assert len(sbatch_calls) == 1


def test_helper_exit_without_terminal_state_fails_closed_without_resubmit(
    monkeypatch, tmp_path: Path
) -> None:
    replicate = build_pilot_matrix(experiment_spec(str(tmp_path)), replicate_index=0)[0]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(render_manifest((replicate,))))
    replicate_json = _replicate_json(manifest_path, replicate)
    replicate = _load_replicate(replicate_json)
    root = Path(replicate.candidate.result_dir).parent
    identity = launch_module._submission_identity(replicate_json, replicate)
    monkeypatch.setattr(launch_module, "validate_arm_runtime", lambda _arm: None)
    monkeypatch.setattr(
        launch_module, "_find_existing_job", lambda _name, _token: None
    )
    monkeypatch.setattr(
        launch_module.subprocess, "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(wait=lambda: 0),
    )
    with pytest.raises(RuntimeError, match="without terminal state"):
        submit_replicate(replicate_json, replicate)
    assert not (root / "submission.helper-result.json").exists()


def test_crash_before_detached_helper_spawn_leaves_no_claim_and_is_retryable(
    monkeypatch, tmp_path: Path
) -> None:
    replicate = build_pilot_matrix(
        experiment_spec(str(tmp_path)), replicate_index=0
    )[0]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(render_manifest((replicate,))))
    replicate_json = _replicate_json(manifest_path, replicate)
    replicate = _load_replicate(replicate_json)
    root = Path(replicate.candidate.result_dir).parent
    calls: list[str] = []
    monkeypatch.setattr(launch_module, "validate_arm_runtime", lambda _arm: None)
    monkeypatch.setattr(
        launch_module.subprocess, "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("pre-spawn crash")),
    )
    with pytest.raises(OSError, match="pre-spawn"):
        submit_replicate(replicate_json, replicate)
    assert not (root / "submission.helper-claim.json").exists()
    install_fake_detached_submit_helper(monkeypatch, calls)
    assert submit_replicate(replicate_json, replicate) == "777"
    assert len(calls) == 1


def test_stale_pre_spawn_owner_is_recovered_once_under_lock(
    monkeypatch, tmp_path: Path
) -> None:
    prepared = tmp_path / "submission.prepared.json"
    result = tmp_path / "submission.helper-result.json"
    identity = {
        "job_name": "cad-token", "submission_token": "full-token",
        "argv": ["sbatch", "--parsable"],
    }
    prepared.write_text(json.dumps({"state": "prepared", **identity}))
    (tmp_path / "submission.helper-claim.json").write_text(json.dumps({
        "state": "helper_owned", **identity,
        "attempt_id": "prior-attempt",
        "owner_id": "dead-owner", "owner_host": "node-a",
        "owner_pid": 99, "claimed_at": "2026-08-22T00:00:00Z",
    }))
    spawns = []

    def fake_spawn(_file, _argv, _env, *, file_actions, setsid):
        spawns.append((_argv, setsid))
        stdout_path = Path(file_actions[0][2])
        stdout_path.write_text("777;cluster\n")
        Path(file_actions[1][2]).write_text("")
        return 123

    monkeypatch.setattr(launch_module, "_find_existing_job", lambda *_args: None)
    monkeypatch.setattr(launch_module.os, "posix_spawnp", fake_spawn)
    monkeypatch.setattr(launch_module.os, "waitpid", lambda *_args: (123, 0))
    launch_module.submit_helper(prepared, result)
    assert json.loads(result.read_text())["job_id"] == "777"
    assert len(spawns) == 1
    assert len(list(tmp_path.glob("submission.helper-takeover-*.json"))) == 1


def test_spawned_stdout_forbids_stale_claim_resubmission(
    monkeypatch, tmp_path: Path
) -> None:
    prepared = tmp_path / "submission.prepared.json"
    result = tmp_path / "submission.helper-result.json"
    identity = {
        "job_name": "cad-token", "submission_token": "full-token",
        "argv": ["sbatch", "--parsable"],
    }
    prepared.write_text(json.dumps({"state": "prepared", **identity}))
    (tmp_path / "submission.helper-claim.json").write_text(json.dumps({
        "state": "helper_owned", **identity,
        "attempt_id": "prior-attempt",
        "owner_id": "dead-owner", "owner_host": "node-a",
        "owner_pid": 99, "claimed_at": "2026-08-22T00:00:00Z",
    }))
    (tmp_path / "submission.sbatch-stdout-prior-attempt.txt").write_text("")
    spawn = MagicMock()
    monkeypatch.setattr(launch_module, "_find_existing_job", lambda *_args: None)
    monkeypatch.setattr(launch_module.os, "posix_spawnp", spawn)
    launch_module.submit_helper(prepared, result)
    spawn.assert_not_called()
    assert not result.exists()
    terminal = json.loads(
        (tmp_path / "submission.helper-terminal.json").read_text()
    )
    assert terminal["phase"] == "ambiguous_after_spawn"
    assert terminal["retryable"] is False
    launch_module.submit_helper(prepared, result)
    spawn.assert_not_called()
    assert json.loads(
        (tmp_path / "submission.helper-terminal.json").read_text()
    ) == terminal


def test_live_submit_owner_lock_prevents_second_helper(
    monkeypatch, tmp_path: Path
) -> None:
    prepared = tmp_path / "submission.prepared.json"
    result = tmp_path / "submission.helper-result.json"
    prepared.write_text(json.dumps({
        "state": "prepared", "job_name": "cad-token",
        "submission_token": "full-token", "argv": ["sbatch", "--parsable"],
    }))
    monkeypatch.setattr(
        launch_module.fcntl,
        "flock",
        lambda *_args: (_ for _ in ()).throw(BlockingIOError()),
    )
    spawn = MagicMock()
    monkeypatch.setattr(launch_module.os, "posix_spawnp", spawn)
    launch_module.submit_helper(prepared, result)
    spawn.assert_not_called()
    assert not result.exists()


@pytest.mark.parametrize(
    ("spawn_error", "wait_status", "phase", "retryable"),
    [
        (True, 0, "pre_submission_file_action_or_exec", True),
        (False, 256, "sbatch_nonzero_ambiguous", False),
    ],
)
def test_submit_failure_retryability_defaults_fail_closed(
    monkeypatch, tmp_path: Path, spawn_error: bool, wait_status: int,
    phase: str, retryable: bool,
) -> None:
    prepared = tmp_path / "submission.prepared.json"
    result = tmp_path / "submission.helper-result.json"
    prepared.write_text(json.dumps({
        "state": "prepared", "job_name": "cad-token",
        "submission_token": "full-token", "argv": ["sbatch", "--parsable"],
    }))

    def fake_spawn(_file, _argv, _env, *, file_actions, setsid):
        if spawn_error:
            raise OSError("exec failed")
        Path(file_actions[0][2]).write_text("")
        Path(file_actions[1][2]).write_text("sbatch rejected request")
        return 123

    monkeypatch.setattr(launch_module.os, "posix_spawnp", fake_spawn)
    monkeypatch.setattr(
        launch_module.os, "waitpid", lambda *_args: (123, wait_status)
    )
    monkeypatch.setattr(
        launch_module, "_bounded_nonzero_visibility",
        lambda *_args, **_kwargs: (None, None),
    )
    launch_module.submit_helper(prepared, result)
    terminal = json.loads(
        (tmp_path / "submission.helper-terminal.json").read_text()
    )
    assert terminal["phase"] == phase
    assert terminal["retryable"] is retryable
    assert not result.exists()


def test_nonzero_sbatch_allows_retry_only_after_bounded_absence_proof(
    monkeypatch, tmp_path: Path
) -> None:
    prepared = tmp_path / "submission.prepared.json"
    result = tmp_path / "submission.helper-result.json"
    prepared.write_text(json.dumps({
        "state": "prepared", "job_name": "cad-token",
        "submission_token": "full-token", "argv": ["sbatch", "--parsable"],
    }))
    spawned = 0

    def fake_spawn(_file, _argv, _env, *, file_actions, setsid):
        nonlocal spawned
        spawned += 1
        Path(file_actions[0][2]).write_text(
            "" if spawned == 1 else "777;cluster\n"
        )
        Path(file_actions[1][2]).write_text(
            "rejected\n" if spawned == 1 else ""
        )
        return 100 + spawned

    statuses = iter(((101, 256), (102, 0)))
    monkeypatch.setattr(launch_module, "_find_existing_job", lambda *_args: None)
    monkeypatch.setattr(
        launch_module,
        "_bounded_nonzero_visibility",
        lambda *_args, **_kwargs: (
            None,
            {
                "proven_absent": True,
                "query_count": 6,
                "interval_seconds": 12,
                "duration_seconds": 60,
                "sources": ["squeue", "sacct"],
                "job_name": "cad-token",
                "submission_token": "full-token",
                "observations": [
                    {
                        "query_index": index,
                        "observed_at": timestamp,
                        "job_id": None,
                    }
                    for index, timestamp in enumerate((
                        "2026-08-22T00:00:00Z",
                        "2026-08-22T00:00:12Z",
                        "2026-08-22T00:00:24Z",
                        "2026-08-22T00:00:36Z",
                        "2026-08-22T00:00:48Z",
                        "2026-08-22T00:01:00Z",
                    ))
                ],
            },
        ),
    )
    monkeypatch.setattr(launch_module.os, "posix_spawnp", fake_spawn)
    monkeypatch.setattr(
        launch_module.os, "waitpid", lambda *_args: next(statuses)
    )
    launch_module.submit_helper(prepared, result)
    first_terminal = json.loads(
        (tmp_path / "submission.helper-terminal.json").read_text()
    )
    assert first_terminal["phase"] == "sbatch_nonzero_scheduler_proved_absent"
    assert first_terminal["retryable"] is True
    assert first_terminal["visibility_evidence"]["query_count"] == 6
    launch_module.submit_helper(prepared, result)
    assert json.loads(result.read_text())["job_id"] == "777"
    assert spawned == 2


def test_nonzero_sbatch_with_visible_job_closes_as_success(
    monkeypatch, tmp_path: Path
) -> None:
    prepared = tmp_path / "submission.prepared.json"
    result = tmp_path / "submission.helper-result.json"
    prepared.write_text(json.dumps({
        "state": "prepared", "job_name": "cad-token",
        "submission_token": "full-token", "argv": ["sbatch", "--parsable"],
    }))

    def fake_spawn(_file, _argv, _env, *, file_actions, setsid):
        Path(file_actions[0][2]).write_text("")
        Path(file_actions[1][2]).write_text("transport interrupted")
        return 123

    monkeypatch.setattr(launch_module.os, "posix_spawnp", fake_spawn)
    monkeypatch.setattr(launch_module.os, "waitpid", lambda *_args: (123, 256))
    monkeypatch.setattr(
        launch_module,
        "_bounded_nonzero_visibility",
        lambda *_args, **_kwargs: ("777", None),
    )
    launch_module.submit_helper(prepared, result)
    assert json.loads(result.read_text())["job_id"] == "777"
    assert not (tmp_path / "submission.helper-terminal.json").exists()


def test_job_names_and_submission_tokens_are_unique_per_triplet() -> None:
    matrix = build_pilot_matrix(experiment_spec(), replicate_index=0)
    bound = render_manifest(matrix)
    manifest_path = Path("/immutable/manifest.json")
    loaded = []
    for raw in bound["replicates"]:
        arms = tuple(launch_module._arm_from_dict(item) for item in raw["arms"])
        loaded.append(MatchedReplicate(
            replicate_id=raw["replicate_id"],
            candidate=next(arm for arm in arms if arm.kind == "candidate"),
            arms=arms,
        ))
    names = {
        launch_module._submission_identity(manifest_path, replicate)["job_name"]
        for replicate in loaded
    }
    tokens = {_submission_token(replicate) for replicate in loaded}
    assert len(names) == len(matrix)
    assert len(tokens) == len(matrix)


def test_sacct_reconciliation_ignores_step_rows(monkeypatch) -> None:
    def fake_run(argv: list[str], **_kwargs) -> SimpleNamespace:
        if argv[0] == "squeue":
            return SimpleNamespace(stdout="")
        return SimpleNamespace(
            stdout=(
                "777|cad-token|full-token\n"
                "777.batch|batch|\n"
                "777.extern|extern|\n"
            )
        )

    monkeypatch.setattr(launch_module, "run_command", fake_run)
    assert launch_module._find_existing_job(
        "cad-token", "full-token"
    ) == "777"


def test_submit_rejects_foreign_terminal_receipt(monkeypatch, tmp_path: Path) -> None:
    replicate = build_pilot_matrix(experiment_spec(str(tmp_path)), replicate_index=0)[0]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(render_manifest((replicate,))))
    replicate_json = _replicate_json(manifest_path, replicate)
    replicate = _load_replicate(replicate_json)
    monkeypatch.setattr(launch_module, "validate_arm_runtime", lambda _arm: None)
    install_fake_detached_submit_helper(monkeypatch, [])
    submit_replicate(replicate_json, replicate)
    receipt = Path(replicate.candidate.result_dir).parent / "submission.json"
    foreign = json.loads(receipt.read_text())
    foreign["manifest_sha256"] = "f" * 64
    receipt.write_text(json.dumps(foreign))
    with pytest.raises(ValueError, match="foreign submission receipt"):
        submit_replicate(replicate_json, replicate)


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("validate-replicate", "validate_replicate_inside_job"),
        ("run-arm", "run_arm"),
        ("close-arm", "close_arm"),
        ("close-replicate", "close_replicate"),
    ],
)
def test_run_replicate_internal_parser_branch_executes(
    monkeypatch, tmp_path: Path, command: str, target: str
) -> None:
    calls = []
    monkeypatch.setattr(
        f"research.qwen3_8b_draft_cadence.launch.{target}",
        lambda *args: calls.append(args),
    )
    path_flag = "--arm-json" if command in {"run-arm", "close-arm"} else "--replicate-json"
    argv = [command, path_flag, str(tmp_path / "input.json"), "--job-id", "7"]
    if command == "validate-replicate":
        argv.append("--inside-job")
    if command in {"close-arm", "close-replicate"}:
        argv.extend(["--exit-code", "0"])
    assert main(argv) == 0
    assert len(calls) == 1


def test_select_arm_parser_branch_prints_path(monkeypatch, tmp_path: Path, capsys) -> None:
    expected = tmp_path / "arm.json"
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.select_arm",
        lambda *_args: expected,
    )
    assert main([
        "select-arm", "--replicate-json", str(tmp_path / "replicate.json"),
        "--execution-order", "2",
    ]) == 0
    assert capsys.readouterr().out.strip() == str(expected)


def test_submit_helper_parser_branch_executes(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.submit_helper",
        lambda *args: calls.append(args),
    )
    assert main([
        "submit-helper",
        "--prepared", str(tmp_path / "prepared.json"),
        "--result", str(tmp_path / "result.json"),
    ]) == 0
    assert calls == [(tmp_path / "prepared.json", tmp_path / "result.json")]


def test_long_with_zero_survivors_writes_terminal_and_submits_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    survivors = tmp_path / "none.json"
    source_matrix = build_cp2_survivor_matrix(
        experiment_spec(str(tmp_path)),
        survivors=("fixed_sparse_10",),
        replicate_index=0,
    )
    source_manifest = render_manifest(source_matrix)
    source_manifest_path = tmp_path / "source-packed-cp2.json"
    source_manifest_path.write_text(json.dumps(source_manifest))
    survivors.write_text(json.dumps({
        "cp2_survivors": [],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_topology": "packed-cp2",
    }))
    manifest = tmp_path / "long-no-survivor.json"
    commands = []
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.validate_source_container_config",
        lambda _spec: None,
    )
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.launch.run_command",
        lambda argv: commands.append(argv),
    )
    assert main([
        "long", "--spec", str(write_spec(tmp_path)),
        "--survivors", str(survivors), "--manifest", str(manifest),
        "--replicates", "3", "--test-only",
    ]) == 0
    assert commands == []
    empty_manifest = json.loads(manifest.read_text())
    assert empty_manifest["replicates"] == []
    assert recompute_manifest_sha256(empty_manifest) == empty_manifest["manifest_sha256"]
    terminal = json.loads(Path(empty_manifest["no_survivor_receipt_path"]).read_text())
    assert terminal["status"] == "no_survivor"
    assert terminal["manifest_path"] == str(manifest.resolve())
```

- [ ] **Step 2: Run the RED launcher contract and confirm the package is absent.**

Run: `uv run --group test pytest -q research/qwen3_8b_draft_cadence/tests/test_contract.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'research.qwen3_8b_draft_cadence'`.

- [ ] **Step 3: Add concrete manifest types and pilot expansion.**

```python
import copy
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, NamedTuple

ArmKind = Literal["fixed", "always", "candidate"]


class Schedule(NamedTuple):
    mode: str
    action: str
    interval: int
    min_interval: int
    max_interval: int
    ewma_alpha: float = 0.1
    degradation_threshold: float = 0.02
    recovery_threshold: float = 0.01
    min_observations: int = 20
    max_burst_updates: int = 10


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    product_repo: str
    harness_repo: str
    product_head: str
    harness_head: str
    submodule_shas: tuple[tuple[str, str], ...]
    container_image: str
    container_digest: str
    base_config_path: str
    base_resolved_config: dict[str, object]
    base_resolved_config_sha256: str
    model_snapshot_path: str
    draft_snapshot_path: str
    model_snapshot_sha256: str
    draft_snapshot_sha256: str
    data_order_path: str
    model_revision: str
    draft_revision: str
    dataset_revision: str
    data_order_sha256: str
    k: int
    global_batch_size: int
    micro_batch_size: int
    cuda_graph_settings: tuple[tuple[str, str], ...]
    canonical_metric_keys: tuple[tuple[str, str], ...]
    cluster: str
    account: str
    partition: str
    nodes: int
    gpus_per_node: int
    time_limit: str
    fairshare_checked_at: str
    fairshare_scores: tuple[tuple[str, float], ...]
    fairshare_eligible_accounts: tuple[str, ...]
    fairshare_selection_reason: str
    wandb_entity: str
    wandb_project: str
    result_root: str


@dataclass(frozen=True, slots=True)
class Arm:
    experiment_id: str
    replicate_id: str
    name: str
    kind: ArmKind
    max_steps: int
    seed: int
    product_repo: str
    harness_repo: str
    product_head: str
    harness_head: str
    submodule_shas: tuple[tuple[str, str], ...]
    container_image: str
    container_digest: str
    base_config_path: str
    base_resolved_config_sha256: str
    model_snapshot_path: str
    draft_snapshot_path: str
    model_snapshot_sha256: str
    draft_snapshot_sha256: str
    data_order_path: str
    invariant_config_sha256: str
    resolved_config_path: str
    resolved_config_sha256: str
    resolved_config: dict[str, object]
    model_revision: str
    draft_revision: str
    dataset_revision: str
    data_order_sha256: str
    k: int
    global_batch_size: int
    micro_batch_size: int
    sequence_packing: bool
    sequence_parallel: bool
    tensor_parallel_size: int
    context_parallel_size: int
    cuda_graph_settings: tuple[tuple[str, str], ...]
    canonical_metric_keys: tuple[tuple[str, str], ...]
    cluster: str
    account: str
    partition: str
    nodes: int
    gpus_per_node: int
    time_limit: str
    fairshare_checked_at: str
    fairshare_scores: tuple[tuple[str, float], ...]
    fairshare_eligible_accounts: tuple[str, ...]
    fairshare_selection_reason: str
    wandb_entity: str
    wandb_project: str
    topology: str
    schedule: Schedule | None
    wandb_id: str
    result_dir: str
    execution_order: int
    stage: str
    resume_after_step: int | None
    resume_checkpoint: str | None
    manifest_sha256: str | None = None
    resume_source_replicate_id: str | None = None


@dataclass(frozen=True, slots=True)
class MatchedReplicate:
    replicate_id: str
    candidate: Arm
    arms: tuple[Arm, ...]
    resume_source_replicate_id: str | None = None

    def validate_parity(self) -> None:
        for field in (
            "max_steps",
            "replicate_id",
            "seed",
            "product_repo",
            "harness_repo",
            "product_head",
            "harness_head",
            "submodule_shas",
            "container_image",
            "container_digest",
            "base_config_path",
            "model_snapshot_path",
            "draft_snapshot_path",
            "model_snapshot_sha256",
            "draft_snapshot_sha256",
            "data_order_path",
            "invariant_config_sha256",
            "model_revision",
            "draft_revision",
            "dataset_revision",
            "data_order_sha256",
            "k",
            "global_batch_size",
            "micro_batch_size",
            "sequence_packing",
            "sequence_parallel",
            "tensor_parallel_size",
            "context_parallel_size",
            "cuda_graph_settings",
            "canonical_metric_keys",
            "cluster",
            "account",
            "partition",
            "nodes",
            "gpus_per_node",
            "time_limit",
            "fairshare_checked_at",
            "fairshare_scores",
            "fairshare_eligible_accounts",
            "fairshare_selection_reason",
            "wandb_entity",
            "wandb_project",
            "topology",
            "stage",
            "resume_after_step",
            "resume_source_replicate_id",
        ):
            values = {getattr(arm, field) for arm in self.arms}
            if len(values) != 1:
                raise ValueError(f"{field} differs within matched replicate")
        if len({arm.wandb_id for arm in self.arms}) != len(self.arms):
            raise ValueError("wandb_id must be unique")
        if len({arm.result_dir for arm in self.arms}) != len(self.arms):
            raise ValueError("result_dir must be unique")
        if len({Path(arm.result_dir).parent for arm in self.arms}) != 1:
            raise ValueError("all arms must share one matched replicate root")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _schedule_payload(schedule: Schedule) -> dict[str, object]:
    if schedule.mode == "always":
        return {"mode": "always"}
    if schedule.mode == "fixed":
        return {
            "mode": "fixed",
            "action": schedule.action,
            "fixed_interval": schedule.interval,
        }
    if schedule.mode == "adaptive":
        return {
            "mode": "adaptive",
            "action": schedule.action,
            "min_interval": schedule.min_interval,
            "max_interval": schedule.max_interval,
            "ewma_alpha": schedule.ewma_alpha,
            "degradation_threshold": schedule.degradation_threshold,
            "recovery_threshold": schedule.recovery_threshold,
            "min_observations": schedule.min_observations,
            "max_burst_updates": schedule.max_burst_updates,
        }
    raise ValueError(f"unsupported schedule mode: {schedule.mode}")


def _resolved_config(
    spec: ExperimentSpec,
    *,
    kind: ArmKind,
    schedule: Schedule | None,
    max_steps: int,
    context_parallel_size: int,
    seed: int,
    result_dir: str,
    wandb_id: str,
    resume_checkpoint: str | None = None,
) -> dict[str, object]:
    if _canonical_sha256(spec.base_resolved_config) != spec.base_resolved_config_sha256:
        raise ValueError("canonical base resolved config digest mismatch")
    resolved = copy.deepcopy(spec.base_resolved_config)
    resolved["grpo"]["max_num_steps"] = max_steps
    resolved["grpo"]["seed"] = seed
    resolved["data"].update(shuffle=False)
    resolved["data"]["train"].update(
        dataset_name="ResponseDataset",
        data_path=spec.data_order_path,
        seed=seed,
    )
    resolved["experiment_provenance"] = {
        "model_revision": spec.model_revision,
        "draft_revision": spec.draft_revision,
        "model_snapshot_sha256": spec.model_snapshot_sha256,
        "draft_snapshot_sha256": spec.draft_snapshot_sha256,
        "dataset_revision": spec.dataset_revision,
        "data_order_sha256": spec.data_order_sha256,
    }
    resolved["logger"]["log_dir"] = result_dir
    resolved["logger"]["wandb"] = {
        "entity": spec.wandb_entity,
        "project": spec.wandb_project,
        "name": wandb_id,
        "id": wandb_id,
    }
    resolved["checkpointing"].update(
        enabled=True,
        checkpoint_dir=f"{result_dir}/checkpoints",
        save_period=100,
        keep_top_k=None,
    )
    resolved["cadence_runtime"] = {
        "enabled": True,
        "result_dir": result_dir,
        "required_checkpoint_steps": [400] if max_steps >= 400 else [],
    }
    policy = resolved["policy"]
    policy["model_name"] = spec.model_snapshot_path
    policy["train_global_batch_size"] = spec.global_batch_size
    policy["train_micro_batch_size"] = spec.micro_batch_size
    policy["sequence_packing"]["enabled"] = True
    policy["megatron_cfg"].update(
        tensor_model_parallel_size=2,
        context_parallel_size=context_parallel_size,
        sequence_parallel=True,
    )
    draft = policy["draft"]
    draft.update(
        model_name=spec.draft_snapshot_path,
        model_revision=spec.draft_revision,
        gamma=spec.k,
        seed=seed,
    )
    policy["generation"]["vllm_kwargs"]["speculative_config"][
        "num_speculative_tokens"
    ] = spec.k
    policy["generation"]["vllm_kwargs"]["compilation_config"] = dict(
        spec.cuda_graph_settings
    )
    draft["enabled"] = kind != "fixed"
    if kind == "fixed":
        draft.pop("update_schedule", None)
    else:
        if schedule is None:
            raise ValueError("enabled draft arm requires a schedule")
        draft["update_schedule"] = _schedule_payload(schedule)
    if resume_checkpoint is not None:
        resolved.setdefault("checkpointing", {})["resume_from_checkpoint"] = resume_checkpoint
    return resolved


def _invariant_config_sha256(
    spec: ExperimentSpec, *, context_parallel_size: int, seed: int
) -> str:
    return _canonical_sha256(
        {
            "model_snapshot_path": spec.model_snapshot_path,
            "draft_snapshot_path": spec.draft_snapshot_path,
            "model_snapshot_sha256": spec.model_snapshot_sha256,
            "draft_snapshot_sha256": spec.draft_snapshot_sha256,
            "data_order_path": spec.data_order_path,
            "model_revision": spec.model_revision,
            "draft_revision": spec.draft_revision,
            "dataset_revision": spec.dataset_revision,
            "data_order_sha256": spec.data_order_sha256,
            "seed": seed,
            "k": spec.k,
            "global_batch_size": spec.global_batch_size,
            "micro_batch_size": spec.micro_batch_size,
            "sequence_packing": True,
            "sequence_parallel": True,
            "tensor_parallel_size": 2,
            "context_parallel_size": context_parallel_size,
            "cuda_graph_settings": spec.cuda_graph_settings,
        }
    )


def _rewritten_invariant_config_sha256(
    arm: Arm, *, context_parallel_size: int
) -> str:
    return _canonical_sha256(
        {
            "model_snapshot_path": arm.model_snapshot_path,
            "draft_snapshot_path": arm.draft_snapshot_path,
            "model_snapshot_sha256": arm.model_snapshot_sha256,
            "draft_snapshot_sha256": arm.draft_snapshot_sha256,
            "data_order_path": arm.data_order_path,
            "model_revision": arm.model_revision,
            "draft_revision": arm.draft_revision,
            "dataset_revision": arm.dataset_revision,
            "data_order_sha256": arm.data_order_sha256,
            "seed": arm.seed,
            "k": arm.k,
            "global_batch_size": arm.global_batch_size,
            "micro_batch_size": arm.micro_batch_size,
            "sequence_packing": arm.sequence_packing,
            "sequence_parallel": arm.sequence_parallel,
            "tensor_parallel_size": arm.tensor_parallel_size,
            "context_parallel_size": context_parallel_size,
            "cuda_graph_settings": arm.cuda_graph_settings,
        }
    )


def _rewrite_resolved_config(
    arm: Arm,
    *,
    context_parallel_size: int,
    max_steps: int,
    result_dir: str,
    wandb_id: str,
    resume_checkpoint: str | None = None,
) -> dict[str, object]:
    resolved = copy.deepcopy(arm.resolved_config)
    resolved["grpo"]["max_num_steps"] = max_steps
    resolved["policy"]["megatron_cfg"]["context_parallel_size"] = (
        context_parallel_size
    )
    resolved["logger"]["log_dir"] = result_dir
    resolved["logger"]["wandb"].update(name=wandb_id, id=wandb_id)
    resolved["checkpointing"].update(
        enabled=True,
        checkpoint_dir=f"{result_dir}/checkpoints",
        save_period=100,
        keep_top_k=None,
    )
    resolved["cadence_runtime"].update(
        result_dir=result_dir,
        required_checkpoint_steps=(
            [400] if max_steps >= 400 and resume_checkpoint is None else []
        ),
    )
    if resume_checkpoint is not None:
        resolved.setdefault("checkpointing", {})["resume_from_checkpoint"] = (
            resume_checkpoint
        )
    return resolved


def build_pilot_matrix(
    spec: ExperimentSpec,
    *,
    replicate_index: int,
) -> tuple[MatchedReplicate, ...]:
    candidates = [
        ("fixed_sparse_10", 300, Schedule("fixed", "sparse_update", 10, 0, 0)),
        ("fixed_sparse_40", 300, Schedule("fixed", "sparse_update", 40, 0, 0)),
        ("fixed_refit_10", 300, Schedule("fixed", "refit_only", 10, 0, 0)),
        ("fixed_refit_40", 300, Schedule("fixed", "refit_only", 40, 0, 0)),
        ("adaptive_10_100", 300, Schedule("adaptive", "sparse_update", 0, 10, 100)),
        ("fixed_sparse_100", 600, Schedule("fixed", "sparse_update", 100, 0, 0)),
        ("fixed_refit_100", 600, Schedule("fixed", "refit_only", 100, 0, 0)),
    ]
    matrix: list[MatchedReplicate] = []
    order = ["fixed", "always", "candidate"]
    rotation = replicate_index % len(order)
    rotated = order[rotation:] + order[:rotation]
    for name, max_steps, schedule in candidates:
        schedules = {
            "fixed": None,
            "always": Schedule("always", "sparse_update", 1, 0, 0),
            "candidate": schedule,
        }
        result_dir_by_kind = {
            kind: (
                f"{spec.result_root}/{spec.experiment_id}/packed-cp1/"
                f"{name}/r{replicate_index}/pilot-{max_steps}/{kind}"
            )
            for kind in order
        }
        wandb_id_by_kind = {
            kind: f"cad-{spec.experiment_id}-{name}-{kind}-r{replicate_index}"
            for kind in order
        }
        resolved_by_kind = {
            kind: _resolved_config(
                spec,
                kind=kind,
                schedule=schedules[kind],
                max_steps=max_steps,
                context_parallel_size=1,
                seed=1234 + replicate_index,
                result_dir=result_dir_by_kind[kind],
                wandb_id=wandb_id_by_kind[kind],
            )
            for kind in order
        }
        arms_by_kind = {
            kind: Arm(
                experiment_id=spec.experiment_id,
                replicate_id=f"{spec.experiment_id}-{name}-packed-cp1-r{replicate_index}",
                name=f"{name}-{kind}-r{replicate_index}",
                kind=kind,
                max_steps=max_steps,
                seed=1234 + replicate_index,
                product_repo=spec.product_repo,
                harness_repo=spec.harness_repo,
                product_head=spec.product_head,
                harness_head=spec.harness_head,
                submodule_shas=spec.submodule_shas,
                container_image=spec.container_image,
                container_digest=spec.container_digest,
                base_config_path=spec.base_config_path,
                base_resolved_config_sha256=spec.base_resolved_config_sha256,
                model_snapshot_path=spec.model_snapshot_path,
                draft_snapshot_path=spec.draft_snapshot_path,
                model_snapshot_sha256=spec.model_snapshot_sha256,
                draft_snapshot_sha256=spec.draft_snapshot_sha256,
                data_order_path=spec.data_order_path,
                invariant_config_sha256=_invariant_config_sha256(
                    spec,
                    context_parallel_size=1,
                    seed=1234 + replicate_index,
                ),
                resolved_config_path=(
                    f"{spec.result_root}/{spec.experiment_id}/configs/"
                    f"{name}-{kind}-r{replicate_index}.yaml"
                ),
                resolved_config_sha256=_canonical_sha256(resolved_by_kind[kind]),
                resolved_config=resolved_by_kind[kind],
                model_revision=spec.model_revision,
                draft_revision=spec.draft_revision,
                dataset_revision=spec.dataset_revision,
                data_order_sha256=spec.data_order_sha256,
                k=spec.k,
                global_batch_size=spec.global_batch_size,
                micro_batch_size=spec.micro_batch_size,
                sequence_packing=True,
                sequence_parallel=True,
                tensor_parallel_size=2,
                context_parallel_size=1,
                cuda_graph_settings=spec.cuda_graph_settings,
                canonical_metric_keys=spec.canonical_metric_keys,
                cluster=spec.cluster,
                account=spec.account,
                partition=spec.partition,
                nodes=spec.nodes,
                gpus_per_node=spec.gpus_per_node,
                time_limit=spec.time_limit,
                fairshare_checked_at=spec.fairshare_checked_at,
                fairshare_scores=spec.fairshare_scores,
                fairshare_eligible_accounts=spec.fairshare_eligible_accounts,
                fairshare_selection_reason=spec.fairshare_selection_reason,
                wandb_entity=spec.wandb_entity,
                wandb_project=spec.wandb_project,
                topology="packed-cp1",
                schedule=schedules[kind],
                wandb_id=wandb_id_by_kind[kind],
                result_dir=result_dir_by_kind[kind],
                execution_order=rotated.index(kind),
                stage="pilot",
                resume_after_step=None,
                resume_checkpoint=None,
            )
            for kind in order
        }
        arms = [arms_by_kind[kind] for kind in rotated]
        replicate = MatchedReplicate(
            replicate_id=f"{spec.experiment_id}-{name}-packed-cp1-r{replicate_index}",
            candidate=arms_by_kind["candidate"],
            arms=tuple(arms),
        )
        replicate.validate_parity()
        matrix.append(replicate)
    return tuple(matrix)


def _rewrite_replicate(
    replicate: MatchedReplicate,
    *,
    topology: str,
    context_parallel_size: int,
    max_steps: int,
    stage: str,
) -> MatchedReplicate:
    rewritten_replicate_id = (
        f"{replicate.replicate_id}-{topology}-{max_steps}"
    )
    rewritten_arms = []
    for arm in replicate.arms:
        rewritten_name = f"{arm.name}-{stage}-{topology}-{max_steps}"
        rewritten_wandb_id = f"{arm.wandb_id}-{topology}-{max_steps}"
        rewritten_result_dir = (
            arm.result_dir.replace("packed-cp1", topology)
            .replace(f"/pilot-{arm.max_steps}/", f"/{stage}-{max_steps}/")
        )
        resolved_config = _rewrite_resolved_config(
            arm,
            context_parallel_size=context_parallel_size,
            max_steps=max_steps,
            result_dir=rewritten_result_dir,
            wandb_id=rewritten_wandb_id,
        )
        rewritten_arms.append(
            replace(
                arm,
            replicate_id=rewritten_replicate_id,
            max_steps=max_steps,
            stage=stage,
            topology=topology,
            context_parallel_size=context_parallel_size,
            invariant_config_sha256=_rewritten_invariant_config_sha256(
                arm, context_parallel_size=context_parallel_size
            ),
            name=rewritten_name,
            wandb_id=rewritten_wandb_id,
            resolved_config_path=(
                f"{Path(arm.resolved_config_path).parent}/"
                f"{arm.name}-{stage}-{topology}-{max_steps}.yaml"
            ),
            resolved_config_sha256=_canonical_sha256(resolved_config),
            resolved_config=resolved_config,
            result_dir=rewritten_result_dir,
            )
        )
    rewritten = tuple(rewritten_arms)
    candidate = next(arm for arm in rewritten if arm.kind == "candidate")
    result = MatchedReplicate(
        replicate_id=rewritten_replicate_id,
        candidate=candidate,
        arms=rewritten,
    )
    result.validate_parity()
    return result


def build_cp2_survivor_matrix(
    spec: ExperimentSpec,
    *,
    survivors: tuple[str, ...],
    replicate_index: int,
) -> tuple[MatchedReplicate, ...]:
    if len(survivors) > 2 or len(set(survivors)) != len(survivors):
        raise ValueError("CP2 requires zero to two unique pilot survivors")
    pilot = {
        replicate.candidate.name.rsplit("-candidate-r", 1)[0]: replicate
        for replicate in build_pilot_matrix(spec, replicate_index=replicate_index)
    }
    unknown = set(survivors) - pilot.keys()
    if unknown:
        raise ValueError(f"unknown survivor: {sorted(unknown)[0]}")
    return tuple(
        _rewrite_replicate(
            pilot[name], topology="packed-cp2", context_parallel_size=2,
            max_steps=pilot[name].candidate.max_steps, stage="cp2-pilot",
        )
        for name in survivors
    )


def build_long_matrix(
    spec: ExperimentSpec,
    *,
    promoted: tuple[str, ...],
    replicate_indices: tuple[int, ...] = (0, 1, 2),
) -> tuple[MatchedReplicate, ...]:
    if len(promoted) > 2 or len(set(promoted)) != len(promoted):
        raise ValueError("long validation requires zero to two unique candidates")
    if replicate_indices != (0, 1, 2):
        raise ValueError("long validation requires exactly replicates 0, 1, and 2")
    if not promoted:
        return ()
    result: list[MatchedReplicate] = []
    for replicate_index in replicate_indices:
        pilot = {
            replicate.candidate.name.rsplit("-candidate-r", 1)[0]: replicate
            for replicate in build_pilot_matrix(
                spec, replicate_index=replicate_index
            )
        }
        for name in promoted:
            if name not in pilot:
                raise ValueError(f"unknown promoted candidate: {name}")
            result.append(
                _rewrite_replicate(
                    pilot[name], topology="packed-cp1", context_parallel_size=1,
                    max_steps=1000, stage="long",
                )
            )
    for promoted_name in promoted:
        fresh = next(
            replicate for replicate in result
            if replicate.candidate.name.startswith(
                f"{promoted_name}-candidate-r0-"
            )
        )
        resumed_replicate_id = f"{fresh.replicate_id}-resume400"
        fresh_root = Path(fresh.candidate.result_dir).parent
        resumed_arm_list = []
        for arm in fresh.arms:
            resume_checkpoint = f"{arm.result_dir}/checkpoints/step_400"
            resumed_result_dir = str(
                fresh_root.parent / "long-resume-1000" / arm.kind
            )
            resumed_wandb_id = f"{arm.wandb_id}-resume400"
            resolved_config = _rewrite_resolved_config(
                arm,
                context_parallel_size=arm.context_parallel_size,
                max_steps=arm.max_steps,
                result_dir=resumed_result_dir,
                wandb_id=resumed_wandb_id,
                resume_checkpoint=resume_checkpoint,
            )
            resumed_arm_list.append(
                replace(
                    arm,
                replicate_id=resumed_replicate_id,
                name=f"{arm.name}-resume400",
                wandb_id=resumed_wandb_id,
                result_dir=resumed_result_dir,
                stage="long-resume",
                resolved_config_path=f"{arm.resolved_config_path}.resume400.yaml",
                resolved_config_sha256=_canonical_sha256(resolved_config),
                resolved_config=resolved_config,
                resume_after_step=400,
                resume_checkpoint=resume_checkpoint,
                resume_source_replicate_id=fresh.replicate_id,
                )
            )
        resumed_arms = tuple(resumed_arm_list)
        resumed = MatchedReplicate(
            replicate_id=resumed_replicate_id,
            candidate=next(
                arm for arm in resumed_arms if arm.kind == "candidate"
            ),
            arms=resumed_arms,
            resume_source_replicate_id=fresh.replicate_id,
        )
        resumed.validate_parity()
        result.append(resumed)
    return tuple(result)


def render_manifest(
    matrix: tuple[MatchedReplicate, ...],
) -> dict[str, object]:
    if not matrix:
        raise ValueError("experiment matrix must not be empty")
    experiment_ids = {
        arm.experiment_id for replicate in matrix for arm in replicate.arms
    }
    if len(experiment_ids) != 1:
        raise ValueError("experiment matrix contains mixed experiment IDs")
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": next(iter(experiment_ids)),
        "replicates": [asdict(replicate) for replicate in matrix],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_sha256 = hashlib.sha256(canonical).hexdigest()
    payload["manifest_sha256"] = manifest_sha256
    for replicate in payload["replicates"]:
        for arm in replicate["arms"]:
            arm["manifest_sha256"] = manifest_sha256
        replicate["candidate"]["manifest_sha256"] = manifest_sha256
    return payload


def recompute_manifest_sha256(payload: dict[str, object]) -> str:
    canonical_payload = copy.deepcopy(payload)
    declared = canonical_payload.pop("manifest_sha256", None)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError("manifest digest is missing")
    for replicate in canonical_payload["replicates"]:
        for arm in replicate["arms"]:
            arm["manifest_sha256"] = None
        replicate["candidate"]["manifest_sha256"] = None
    return _canonical_sha256(canonical_payload)


def write_manifest_exclusive(
    result_root: Path,
    manifest: dict[str, object],
) -> Path:
    experiment_id = str(manifest["experiment_id"])
    experiment_dir = result_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    path = experiment_dir / "manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return path


def _submission_token(replicate: MatchedReplicate) -> str:
    digest = replicate.candidate.manifest_sha256
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("submission token requires a manifest-bound replicate")
    return _canonical_sha256(
        {"manifest_sha256": digest, "replicate_id": replicate.replicate_id}
    )


def sbatch_argv(replicate_json: Path, replicate: MatchedReplicate) -> list[str]:
    replicate.validate_parity()
    first = replicate.arms[0]
    if not isinstance(first.manifest_sha256, str) or len(first.manifest_sha256) != 64:
        raise ValueError("sbatch requires a manifest-bound replicate")
    return [
        "sbatch",
        "--parsable",
        "--job-name",
        f"cad-{_submission_token(replicate)[:24]}",
        "--comment",
        _submission_token(replicate),
        "--account",
        first.account,
        "--partition",
        first.partition,
        "--nodes",
        str(first.nodes),
        "--gres",
        f"gpu:{first.gpus_per_node}",
        "--time",
        first.time_limit,
        "--container-image",
        first.container_image,
        "--output",
        f"{first.result_dir}/../slurm-%j.out",
        "--chdir",
        first.harness_repo,
        "--export",
        (
            f"NONE,CADENCE_HARNESS_ROOT={first.harness_repo},"
            f"NEMO_RL_SOURCE_ROOT={first.product_repo},"
            f"WANDB_ENTITY={first.wandb_entity},"
            f"WANDB_PROJECT={first.wandb_project}"
        ),
        "research/qwen3_8b_draft_cadence/run_replicate.sh",
        str(replicate_json),
    ]
```

`experiment_spec()` in the RED test is a real local helper returning every `ExperimentSpec` field above, with optional `result_root`; it lives at the top of `test_contract.py`. `render_manifest` serializes sorted dataclass dictionaries, computes SHA256 over canonical JSON without the digest field, inserts `manifest_sha256`, and returns the immutable object. `write_manifest_exclusive` creates `<result_root>/<experiment_id>/manifest.json` with mode `"x"`. Per-arm setup then creates and validates the exclusive `arm-identity.json`; an empty directory left before that identity write is safely retryable, while any foreign or partial identity fails closed.

Use this executable `launch.py` command surface (the validation helpers it calls implement the fail-closed checks above):

```python
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import socket
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config

from research.qwen3_8b_draft_cadence.manifest import (
    ExperimentSpec,
    Arm,
    MatchedReplicate,
    Schedule,
    _canonical_sha256,
    build_cp2_survivor_matrix,
    build_long_matrix,
    build_pilot_matrix,
    recompute_manifest_sha256,
    render_manifest,
    sbatch_argv,
)


def run_command(
    argv: list[str], *, timeout_seconds: int | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, check=True, text=True, capture_output=True,
        timeout=timeout_seconds,
    )


def _git(repo: str, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", repo, *args], text=True
    ).rstrip()


def _clean_recursive_submodule_shas(status: str) -> dict[str, str]:
    result = {}
    for line in status.splitlines():
        if not line or line[0] in "-+U":
            raise ValueError("recursive submodule is unresolved or dirty")
        sha, path, *_rest = line[1:].split()
        result[path] = sha
    return result


def validate_fairshare_receipt(
    spec: ExperimentSpec,
    *,
    now: datetime | None = None,
) -> None:
    current = now or datetime.now(timezone.utc)
    if not spec.fairshare_checked_at.endswith("Z"):
        raise ValueError("FairShare timestamp must use canonical UTC Z form")
    checked = datetime.fromisoformat(
        spec.fairshare_checked_at.replace("Z", "+00:00")
    )
    if checked.tzinfo is None:
        raise ValueError("FairShare evidence must include a UTC offset")
    age_seconds = (current - checked).total_seconds()
    if not 0.0 <= age_seconds <= 900.0:
        raise ValueError("FairShare evidence must be UTC and at most 15 minutes old")
    scores = dict(spec.fairshare_scores)
    eligible = set(spec.fairshare_eligible_accounts)
    if (
        not eligible
        or not eligible.issubset(scores)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or value < 0.0
            for value in scores.values()
        )
    ):
        raise ValueError("FairShare scores/eligibility are malformed")
    winner = sorted(eligible, key=lambda name: (-scores[name], name))[0]
    if (
        spec.account != winner
        or spec.fairshare_selection_reason != "highest eligible FairShare"
    ):
        raise ValueError("selected account is not the highest eligible FairShare")


def validate_source_container_config(spec: ExperimentSpec) -> None:
    validate_fairshare_receipt(spec)
    for repo, expected_head in (
        (spec.product_repo, spec.product_head),
        (spec.harness_repo, spec.harness_head),
    ):
        if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ValueError(f"dirty recursive source tree: {repo}")
        head = _git(repo, "rev-parse", "HEAD")
        upstream = _git(repo, "rev-parse", "@{u}")
        if head != expected_head or upstream != expected_head:
            raise ValueError(f"local/pushed SHA mismatch: {repo}")
        submodules = _git(repo, "submodule", "status", "--recursive")
        actual_submodules = _clean_recursive_submodule_shas(submodules)
        expected_submodules = (
            dict(spec.submodule_shas) if repo == spec.product_repo else {}
        )
        if actual_submodules != expected_submodules:
            raise ValueError(f"recursive submodule manifest mismatch: {repo}")
    for path, expected_sha in spec.submodule_shas:
        actual_sha = _git(spec.product_repo, "rev-parse", f"HEAD:{path}")
        if actual_sha != expected_sha:
            raise ValueError(f"submodule SHA mismatch: {path}")
    if not spec.container_digest.startswith("sha256:"):
        raise ValueError("container digest must be immutable sha256")
    if _sha256_file(Path(spec.container_image)) != spec.container_digest.removeprefix("sha256:"):
        raise ValueError("container file digest mismatch")
    if (
        not Path(spec.model_snapshot_path).is_dir()
        or not Path(spec.draft_snapshot_path).is_dir()
        or _sha256_directory(Path(spec.model_snapshot_path))
        != spec.model_snapshot_sha256
        or _sha256_directory(Path(spec.draft_snapshot_path))
        != spec.draft_snapshot_sha256
        or spec.model_revision not in Path(spec.model_snapshot_path).name
        or spec.draft_revision not in Path(spec.draft_snapshot_path).name
        or _sha256_file(Path(spec.data_order_path)) != spec.data_order_sha256
        or spec.dataset_revision not in Path(spec.data_order_path).name
    ):
        raise ValueError("immutable model/draft/data revision artifact mismatch")
    if "${" in json.dumps(asdict(spec), sort_keys=True):
        raise ValueError("resolved experiment spec contains interpolation")
    if _canonical_sha256(spec.base_resolved_config) != spec.base_resolved_config_sha256:
        raise ValueError("canonical base resolved config digest mismatch")
    loaded_base = OmegaConf.to_container(
        load_config(spec.base_config_path), resolve=True
    )
    if loaded_base != spec.base_resolved_config:
        raise ValueError("canonical base Hydra config differs from frozen resolved config")


def validate_collision_free_identity(
    matrix: tuple[MatchedReplicate, ...],
) -> None:
    replicates = [replicate.replicate_id for replicate in matrix]
    arms = [arm for replicate in matrix for arm in replicate.arms]
    for label, values in (
        ("replicate_id", replicates),
        ("arm_id", [arm.name for arm in arms]),
        ("wandb_id", [arm.wandb_id for arm in arms]),
        ("result_dir", [arm.result_dir for arm in arms]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label}")


def next_unsubmitted_replicate(
    entries: tuple[tuple[Path, MatchedReplicate], ...],
) -> tuple[Path, MatchedReplicate] | None:
    for path, replicate in entries:
        receipt = Path(replicate.candidate.result_dir).parent / "submission.json"
        if receipt.exists():
            _validate_submission_receipt(
                json.loads(receipt.read_text()),
                _submission_identity(path, replicate),
            )
            continue
        validate_resume_source_ready(entries, replicate)
        return path, replicate
    return None


def validate_resume_source_ready(
    entries: tuple[tuple[Path, MatchedReplicate], ...],
    replicate: MatchedReplicate,
) -> None:
    if replicate.resume_source_replicate_id is None:
        return
    matches = [
        source for _path, source in entries
        if source.replicate_id == replicate.resume_source_replicate_id
    ]
    if len(matches) != 1:
        raise ValueError("resume source replicate is absent or duplicated")
    source = matches[0]
    replicate_terminal_path = _replicate_root(source) / "terminal.json"
    if not replicate_terminal_path.is_file():
        raise RuntimeError("resume source triplet is not terminal")
    replicate_terminal = json.loads(replicate_terminal_path.read_text())
    if (
        replicate_terminal.get("state") != "terminal"
        or replicate_terminal.get("terminal") is not True
        or replicate_terminal.get("exit_code") != 0
        or set(replicate_terminal.get("completed_arm_ids", []))
        != {arm.name for arm in source.arms}
    ):
        raise ValueError("resume source triplet terminal receipt is invalid")
    source_by_kind = {arm.kind: arm for arm in source.arms}
    for resumed_arm in replicate.arms:
        source_arm = source_by_kind[resumed_arm.kind]
        if (
            resumed_arm.resume_checkpoint
            != f"{source_arm.result_dir}/checkpoints/step_400"
        ):
            raise ValueError("resume checkpoint differs from its fresh source")
        terminal_path = Path(source_arm.result_dir) / "terminal.json"
        if not terminal_path.is_file():
            raise RuntimeError("resume source triplet is not terminal")
        terminal = json.loads(terminal_path.read_text())
        job_id = terminal.get("job_id")
        if (
            not isinstance(job_id, str)
            or terminal.get("terminal") is not True
            or terminal.get("exit_code") != 0
            or terminal.get("completed_policy_steps") != source_arm.max_steps
            or {
                key: terminal.get(key)
                for key in _arm_receipt_identity(source_arm, job_id)
            }
            != _arm_receipt_identity(source_arm, job_id)
        ):
            raise ValueError("resume source arm terminal receipt is invalid")
        validate_product_runtime_receipts(source_arm)


def _load_spec(path: Path) -> ExperimentSpec:
    raw = json.loads(path.read_text())
    for key in (
        "submodule_shas", "cuda_graph_settings", "canonical_metric_keys",
        "fairshare_scores",
    ):
        raw[key] = tuple(tuple(item) for item in raw[key])
    raw["fairshare_eligible_accounts"] = tuple(
        raw["fairshare_eligible_accounts"]
    )
    return ExperimentSpec(**raw)


def _load_survivors(path: Path, *, expected_source_topology: str) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    analysis_path = Path(str(payload.get("source_analysis_receipt_path"))).resolve()
    analysis_raw = analysis_path.read_bytes()
    if (
        payload.get("source_analysis_receipt_size_bytes") != len(analysis_raw)
        or payload.get("source_analysis_receipt_sha256")
        != hashlib.sha256(analysis_raw).hexdigest()
    ):
        raise ValueError("survivor file does not bind its analysis receipt")
    stored_analysis = json.loads(analysis_raw)
    source_manifest_path = Path(payload["source_manifest_path"])
    source_manifest = json.loads(source_manifest_path.read_text())
    source_digest = recompute_manifest_sha256(source_manifest)
    if (
        payload.get("source_topology") != expected_source_topology
        or payload.get("source_manifest_sha256") != source_digest
        or source_manifest.get("manifest_sha256") != source_digest
        or any(
            arm["topology"] != expected_source_topology
            for replicate in source_manifest["replicates"]
            for arm in replicate["arms"]
        )
    ):
        raise ValueError("survivor receipt has wrong source topology or manifest")
    from research.qwen3_8b_draft_cadence.analyze import analyze_manifest

    topology = "cp1" if expected_source_topology == "packed-cp1" else "cp2"
    cp1_receipt = stored_analysis.get("cp1_receipt_path")
    recomputed = analyze_manifest(
        "pilot",
        source_manifest_path,
        topology=topology,
        cp1_receipt_path=(Path(cp1_receipt) if cp1_receipt is not None else None),
    )
    if stored_analysis != recomputed:
        raise ValueError("pilot analysis receipt disagrees with recomputed evidence")
    survivors = tuple(str(value) for value in recomputed["cp2_survivors"])
    if list(survivors) != payload.get("cp2_survivors"):
        raise ValueError("survivor list differs from recomputed scientific decision")
    if len(survivors) > 2 or len(set(survivors)) != len(survivors):
        raise ValueError("survivor receipt must contain zero to two unique names")
    source_candidates = {
        replicate["candidate"]["name"].split("-candidate-r", 1)[0]
        for replicate in source_manifest["replicates"]
    }
    if not set(survivors).issubset(source_candidates):
        raise ValueError("survivor is absent from its bound source manifest")
    return survivors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"snapshot directory is absent: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"snapshot directory is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _arm_from_dict(raw: dict[str, object]) -> Arm:
    converted = dict(raw)
    if converted["schedule"] is not None:
        converted["schedule"] = Schedule(*converted["schedule"])
    for key in (
        "submodule_shas", "cuda_graph_settings", "canonical_metric_keys",
        "fairshare_scores",
    ):
        converted[key] = tuple(tuple(item) for item in converted[key])
    converted["fairshare_eligible_accounts"] = tuple(
        converted["fairshare_eligible_accounts"]
    )
    return Arm(**converted)


def _load_replicate(path: Path) -> MatchedReplicate:
    envelope = json.loads(path.read_text())
    manifest_path = Path(envelope["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    recomputed = recompute_manifest_sha256(manifest)
    if (
        envelope.get("manifest_sha256") != recomputed
        or manifest.get("manifest_sha256") != recomputed
    ):
        raise ValueError("replicate JSON is not bound to the recomputed manifest")
    raw = envelope["replicate"]
    matching = [
        item for item in manifest["replicates"]
        if item["replicate_id"] == raw["replicate_id"]
    ]
    if matching != [raw]:
        raise ValueError("replicate JSON differs from immutable manifest")
    arms = tuple(_arm_from_dict(item) for item in raw["arms"])
    candidate_name = raw["candidate"]["name"]
    replicate = MatchedReplicate(
        replicate_id=str(raw["replicate_id"]),
        candidate=next(arm for arm in arms if arm.name == candidate_name),
        arms=arms,
        resume_source_replicate_id=raw.get("resume_source_replicate_id"),
    )
    replicate.validate_parity()
    return replicate


def _write_json_exclusive_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_replace_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_bytes_exclusive_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_and_validate_resolved_config(arm: Arm) -> None:
    path = Path(arm.resolved_config_path)
    canonical = json.dumps(
        arm.resolved_config, sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != arm.resolved_config_sha256:
        raise ValueError("declared resolved-config digest mismatch")
    if arm.resolved_config["grpo"]["max_num_steps"] != arm.max_steps:
        raise ValueError("resolved config grpo.max_num_steps mismatch")
    if (
        arm.resolved_config["grpo"].get("seed") != arm.seed
        or arm.resolved_config["data"].get("shuffle") is not False
        or arm.resolved_config["data"]["train"].get("data_path")
        != arm.data_order_path
        or arm.resolved_config["data"]["train"].get("seed") != arm.seed
        or arm.resolved_config["policy"].get("model_name")
        != arm.model_snapshot_path
    ):
        raise ValueError("resolved seed/model/data-order treatment mismatch")
    draft = arm.resolved_config["policy"]["draft"]
    generation = arm.resolved_config["policy"]["generation"]["vllm_kwargs"]
    if (
        draft.get("model_name") != arm.draft_snapshot_path
        or draft.get("model_revision") != arm.draft_revision
        or draft.get("gamma") != arm.k
        or draft.get("seed") != arm.seed
        or generation["speculative_config"].get("num_speculative_tokens") != arm.k
        or generation.get("compilation_config") != dict(arm.cuda_graph_settings)
    ):
        raise ValueError("resolved draft K/revision/CUDA-Graph treatment mismatch")
    if arm.kind == "fixed":
        if draft.get("enabled") is not False or "update_schedule" in draft:
            raise ValueError("fixed control must use policy.draft.enabled=false")
    elif draft.get("enabled") is not True or "update_schedule" not in draft:
        raise ValueError("online arm must enable draft with an update schedule")
    elif draft["update_schedule"] != _schedule_payload(arm.schedule):
        raise ValueError("online arm resolved schedule differs from treatment")
    provenance = arm.resolved_config.get("experiment_provenance")
    if provenance != {
        "model_revision": arm.model_revision,
        "draft_revision": arm.draft_revision,
        "model_snapshot_sha256": arm.model_snapshot_sha256,
        "draft_snapshot_sha256": arm.draft_snapshot_sha256,
        "dataset_revision": arm.dataset_revision,
        "data_order_sha256": arm.data_order_sha256,
    }:
        raise ValueError("resolved immutable revision provenance mismatch")
    runtime = arm.resolved_config.get("cadence_runtime", {})
    checkpointing = arm.resolved_config.get("checkpointing", {})
    logger = arm.resolved_config.get("logger", {})
    if (
        runtime.get("enabled") is not True
        or runtime.get("result_dir") != arm.result_dir
        or runtime.get("required_checkpoint_steps")
        != ([400] if arm.max_steps >= 400 and arm.resume_after_step is None else [])
        or checkpointing.get("checkpoint_dir") != f"{arm.result_dir}/checkpoints"
        or checkpointing.get("save_period") != 100
        or checkpointing.get("keep_top_k", "missing") is not None
        or logger.get("log_dir") != arm.result_dir
        or logger.get("wandb", {}).get("entity") != arm.wandb_entity
        or logger.get("wandb", {}).get("project") != arm.wandb_project
        or logger.get("wandb", {}).get("id") != arm.wandb_id
    ):
        raise ValueError("resolved runtime receipt/checkpoint/W&B config mismatch")
    if arm.resume_after_step is not None and (
        arm.resolved_config.get("checkpointing", {}).get("resume_from_checkpoint")
        != arm.resume_checkpoint
    ):
        raise ValueError("resume override differs from declared checkpoint")
    expected_invariant = _canonical_sha256({
        "model_snapshot_path": arm.model_snapshot_path,
        "draft_snapshot_path": arm.draft_snapshot_path,
        "model_snapshot_sha256": arm.model_snapshot_sha256,
        "draft_snapshot_sha256": arm.draft_snapshot_sha256,
        "data_order_path": arm.data_order_path,
        "model_revision": arm.model_revision,
        "draft_revision": arm.draft_revision,
        "dataset_revision": arm.dataset_revision,
        "data_order_sha256": arm.data_order_sha256,
        "seed": arm.seed,
        "k": arm.k,
        "global_batch_size": arm.global_batch_size,
        "micro_batch_size": arm.micro_batch_size,
        "sequence_packing": arm.sequence_packing,
        "sequence_parallel": arm.sequence_parallel,
        "tensor_parallel_size": arm.tensor_parallel_size,
        "context_parallel_size": arm.context_parallel_size,
        "cuda_graph_settings": arm.cuda_graph_settings,
    })
    if expected_invariant != arm.invariant_config_sha256:
        raise ValueError("held-field invariant config digest mismatch")
    if not path.exists():
        _write_bytes_exclusive_atomic(path, canonical)
    if _sha256_file(path) != arm.resolved_config_sha256:
        raise ValueError("resolved config file changed after materialization")


def validate_arm_runtime(arm: Arm) -> None:
    for repo, expected in (
        (arm.product_repo, arm.product_head),
        (arm.harness_repo, arm.harness_head),
    ):
        if _git(repo, "rev-parse", "HEAD") != expected:
            raise ValueError(f"runtime source SHA mismatch: {repo}")
        if _git(repo, "rev-parse", "@{u}") != expected:
            raise ValueError(f"runtime pushed SHA mismatch: {repo}")
        if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ValueError(f"runtime source tree dirty: {repo}")
        status = _git(repo, "submodule", "status", "--recursive")
        actual_submodules = _clean_recursive_submodule_shas(status)
        expected_submodules = (
            dict(arm.submodule_shas) if repo == arm.product_repo else {}
        )
        if actual_submodules != expected_submodules:
            raise ValueError(f"runtime recursive submodule mismatch: {repo}")
    for path, expected in arm.submodule_shas:
        if _git(arm.product_repo, "rev-parse", f"HEAD:{path}") != expected:
            raise ValueError(f"runtime submodule SHA mismatch: {path}")
    if _sha256_file(Path(arm.container_image)) != arm.container_digest.removeprefix("sha256:"):
        raise ValueError("runtime container digest mismatch")
    if (
        not Path(arm.model_snapshot_path).is_dir()
        or not Path(arm.draft_snapshot_path).is_dir()
        or _sha256_directory(Path(arm.model_snapshot_path))
        != arm.model_snapshot_sha256
        or _sha256_directory(Path(arm.draft_snapshot_path))
        != arm.draft_snapshot_sha256
        or arm.model_revision not in Path(arm.model_snapshot_path).name
        or arm.draft_revision not in Path(arm.draft_snapshot_path).name
        or _sha256_file(Path(arm.data_order_path)) != arm.data_order_sha256
        or arm.dataset_revision not in Path(arm.data_order_path).name
    ):
        raise ValueError("runtime model/draft/data revision artifact mismatch")
    materialize_and_validate_resolved_config(arm)


def _replicate_root(replicate: MatchedReplicate) -> Path:
    roots = {Path(arm.result_dir).parent for arm in replicate.arms}
    if len(roots) != 1:
        raise ValueError("matched replicate does not have one shared result root")
    return next(iter(roots))


def _submission_identity(path: Path, replicate: MatchedReplicate) -> dict[str, object]:
    manifest_digests = {arm.manifest_sha256 for arm in replicate.arms}
    if len(manifest_digests) != 1 or None in manifest_digests:
        raise ValueError("submission requires one bound manifest digest")
    argv = sbatch_argv(path, replicate)
    identity = {
        "schema_version": 1,
        "replicate_id": replicate.replicate_id,
        "manifest_sha256": next(iter(manifest_digests)),
        "argv": argv,
        "job_name": argv[argv.index("--job-name") + 1],
        "submission_token": _submission_token(replicate),
        "identity_sha256": _canonical_sha256(
            {
                "replicate_id": replicate.replicate_id,
                "manifest_sha256": next(iter(manifest_digests)),
                "argv": argv,
            }
        ),
        "product_head": replicate.candidate.product_head,
        "harness_head": replicate.candidate.harness_head,
        "submodule_shas": replicate.candidate.submodule_shas,
        "container_digest": replicate.candidate.container_digest,
        "model_snapshot_sha256": replicate.candidate.model_snapshot_sha256,
        "draft_snapshot_sha256": replicate.candidate.draft_snapshot_sha256,
        "invariant_config_sha256": replicate.candidate.invariant_config_sha256,
        "resolved_config_sha256_by_arm": {
            arm.name: arm.resolved_config_sha256 for arm in replicate.arms
        },
        "wandb_entity": replicate.candidate.wandb_entity,
        "wandb_project": replicate.candidate.wandb_project,
    }
    return json.loads(json.dumps(identity, sort_keys=True))


def _validate_identity_file(path: Path, expected: dict[str, object]) -> None:
    normalized = json.loads(json.dumps(expected, sort_keys=True))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"foreign or colliding submission state: {path}")
    else:
        _write_json_exclusive_atomic(path, normalized)


def _validate_submission_receipt(
    receipt: dict[str, object], identity: dict[str, object]
) -> str:
    if receipt.get("state") != "submitted":
        raise ValueError("submission receipt is not terminal submitted state")
    if {key: receipt.get(key) for key in identity} != identity:
        raise ValueError("foreign submission receipt identity")
    job_id = receipt.get("job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("submission receipt has invalid job ID")
    return job_id


def _find_existing_job(job_name: str, submission_token: str) -> str | None:
    candidates: set[str] = set()
    for argv, separator in (
        (["squeue", "--noheader", "--name", job_name, "--format=%A|%j|%k"], "|"),
        (["sacct", "--noheader", "--name", job_name, "--parsable2", "--format=JobIDRaw,JobName,Comment"], "|"),
    ):
        completed = run_command(argv, timeout_seconds=10)
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(separator)]
            if len(fields) < 3 or not fields[0].split(".", 1)[0].isdigit():
                continue
            if "." in fields[0]:
                continue
            if fields[1] != job_name or fields[2] != submission_token:
                raise RuntimeError("submission reconciliation found foreign identity")
            candidates.add(fields[0].split(".", 1)[0])
    if len(candidates) > 1:
        raise RuntimeError("submission reconciliation found multiple jobs")
    return next(iter(candidates), None)


def _bounded_nonzero_visibility(
    job_name: str,
    submission_token: str,
    *,
    query_count: int = 6,
    interval_seconds: int = 12,
) -> tuple[str | None, Mapping[str, object] | None]:
    if query_count < 2 or interval_seconds <= 0:
        raise ValueError("nonzero visibility window must be bounded and repeated")
    started = datetime.now(timezone.utc)
    observations: list[dict[str, object]] = []
    try:
        for index in range(query_count):
            job_id = _find_existing_job(job_name, submission_token)
            observations.append({
                "query_index": index,
                "observed_at": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "job_id": job_id,
            })
            if job_id is not None:
                return job_id, None
            if index + 1 < query_count:
                time.sleep(interval_seconds)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return None, None
    ended = datetime.now(timezone.utc)
    duration = (ended - started).total_seconds()
    if duration < interval_seconds * (query_count - 1):
        return None, None
    evidence = {
        "proven_absent": True,
        "query_count": query_count,
        "interval_seconds": interval_seconds,
        "duration_seconds": duration,
        "sources": ["squeue", "sacct"],
        "job_name": job_name,
        "submission_token": submission_token,
        "observations": observations,
    }
    identity = {"job_name": job_name, "submission_token": submission_token}
    try:
        _validate_absence_evidence(evidence, identity)
    except ValueError:
        return None, None
    return None, evidence


def submit_replicate(path: Path, replicate: MatchedReplicate) -> str:
    root = _replicate_root(replicate)
    receipt = root / "submission.json"
    prepared = root / "submission.prepared.json"
    helper_result = root / "submission.helper-result.json"
    helper_terminal = root / "submission.helper-terminal.json"
    identity = _submission_identity(path, replicate)
    if receipt.exists():
        return _validate_submission_receipt(json.loads(receipt.read_text()), identity)
    root.mkdir(parents=True, exist_ok=True)
    for arm in replicate.arms:
        validate_arm_runtime(arm)
        _validate_identity_file(Path(arm.result_dir) / "arm.json", asdict(arm))
    _validate_identity_file(prepared, {"state": "prepared", **identity})
    helper = subprocess.Popen(
        [
            sys.executable, str(Path(__file__).resolve()), "submit-helper",
            "--prepared", str(prepared), "--result", str(helper_result),
        ],
        cwd=replicate.candidate.harness_repo,
        start_new_session=True,
    )
    helper.wait()
    if helper_result.exists():
        result_payload = json.loads(helper_result.read_text())
        _validate_identity_file(
            helper_result,
            {"state": "helper_complete", **identity, "job_id": result_payload["job_id"]},
        )
        job_id = str(result_payload["job_id"])
    else:
        job_id = _find_existing_job(
            str(identity["job_name"]), str(identity["submission_token"])
        )
        if job_id is None:
            if not helper_terminal.exists():
                raise RuntimeError("submit helper exited without terminal state")
            terminal = _validate_submit_helper_terminal(
                json.loads(helper_terminal.read_text()), identity
            )
            phase = terminal["phase"]
            retryable = terminal["retryable"]
            retryability = "retryable" if retryable else "terminal"
            raise RuntimeError(
                f"{retryability} submit-helper failure: {phase}"
            )
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise RuntimeError(f"sbatch reconciliation returned invalid job ID: {job_id!r}")
    _write_json_exclusive_atomic(
        receipt,
        {
            "state": "submitted",
            **identity,
            "job_id": job_id,
        },
    )
    return job_id


def _validate_absence_evidence(
    evidence: object,
    identity: Mapping[str, object],
) -> None:
    if not isinstance(evidence, Mapping):
        raise ValueError("retryable sbatch nonzero lacks scheduler proof")
    duration = evidence.get("duration_seconds")
    observations = evidence.get("observations")
    if (
        evidence.get("proven_absent") is not True
        or evidence.get("query_count") != 6
        or evidence.get("interval_seconds") != 12
        or type(duration) not in (int, float)
        or not isfinite(float(duration))
        or float(duration) < 60.0
        or evidence.get("sources") != ["squeue", "sacct"]
        or evidence.get("job_name") != identity.get("job_name")
        or evidence.get("submission_token") != identity.get("submission_token")
        or not isinstance(observations, list)
        or len(observations) != 6
    ):
        raise ValueError("invalid scheduler absence proof")
    observed_times: list[datetime] = []
    for index, observation in enumerate(observations):
        if (
            not isinstance(observation, Mapping)
            or observation.get("query_index") != index
            or observation.get("job_id") is not None
            or not str(observation.get("observed_at", "")).endswith("Z")
        ):
            raise ValueError("invalid scheduler absence observation")
        try:
            observed_times.append(datetime.fromisoformat(
                str(observation["observed_at"]).replace("Z", "+00:00")
            ))
        except ValueError as error:
            raise ValueError("invalid scheduler absence timestamp") from error
    if (
        any(left >= right for left, right in zip(observed_times, observed_times[1:]))
        or (observed_times[-1] - observed_times[0]).total_seconds() < 60.0
        or float(duration)
        < (observed_times[-1] - observed_times[0]).total_seconds()
    ):
        raise ValueError("scheduler absence window is too short")


def _close_submit_helper_failure(
    prepared: Path,
    identity: Mapping[str, object],
    *,
    attempt_id: str,
    phase: str,
    retryable: bool,
    return_code: int | None,
    visibility_evidence: Mapping[str, object] | None = None,
) -> None:
    retryable_phases = {
        "pre_submission_file_action_or_exec",
        "sbatch_nonzero_scheduler_proved_absent",
    }
    terminal_phases = {
        *retryable_phases,
        "ambiguous_after_spawn", "invalid_success_output",
        "sbatch_nonzero_ambiguous",
    }
    if phase not in terminal_phases or retryable is not (phase in retryable_phases):
        raise ValueError("invalid submit-helper terminal phase/retryability")
    if phase == "sbatch_nonzero_scheduler_proved_absent":
        _validate_absence_evidence(visibility_evidence, identity)
    elif visibility_evidence is not None:
        raise ValueError("visibility evidence is valid only for proven absence")
    payload = {
        "state": "helper_failed", **identity,
        "attempt_id": attempt_id, "phase": phase,
        "retryable": retryable, "return_code": return_code,
        "visibility_evidence": visibility_evidence,
        "closed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_json_replace_atomic(
        prepared.with_name("submission.helper-terminal.json"), payload
    )
    _write_json_exclusive_atomic(
        prepared.with_name(f"submission.helper-failure-{attempt_id}.json"),
        payload,
    )


def _validate_submit_helper_terminal(
    payload: object,
    identity: Mapping[str, object],
    *,
    attempt_id: str | None = None,
) -> Mapping[str, object]:
    retryable_phases = {
        "pre_submission_file_action_or_exec",
        "sbatch_nonzero_scheduler_proved_absent",
    }
    terminal_phases = {
        *retryable_phases,
        "ambiguous_after_spawn", "invalid_success_output",
        "sbatch_nonzero_ambiguous",
    }
    if not isinstance(payload, Mapping) or payload.get("state") != "helper_failed":
        raise ValueError("invalid submit-helper terminal schema")
    if {key: payload.get(key) for key in identity} != dict(identity):
        raise ValueError("foreign submit-helper terminal state")
    if attempt_id is not None and payload.get("attempt_id") != attempt_id:
        raise ValueError("submit-helper terminal attempt mismatch")
    phase = payload.get("phase")
    retryable = payload.get("retryable")
    if (
        phase not in terminal_phases
        or type(retryable) is not bool
        or retryable is not (phase in retryable_phases)
    ):
        raise ValueError("invalid submit-helper terminal phase/retryability")
    evidence = payload.get("visibility_evidence")
    if phase == "sbatch_nonzero_scheduler_proved_absent":
        _validate_absence_evidence(evidence, identity)
    elif evidence is not None:
        raise ValueError("unexpected submit-helper visibility evidence")
    return payload


def submit_helper(prepared: Path, result: Path) -> None:
    identity = json.loads(prepared.read_text())
    if identity.pop("state", None) != "prepared":
        raise ValueError("submit helper requires a prepared identity")
    claim = prepared.with_name("submission.helper-claim.json")
    lock_path = prepared.with_name("submission.helper.lock")
    terminal_path = prepared.with_name("submission.helper-terminal.json")
    if result.exists():
        return
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        owner = {
            "attempt_id": str(uuid.uuid4()),
            "owner_id": str(uuid.uuid4()),
            "owner_host": socket.gethostname(),
            "owner_pid": os.getpid(),
            "claimed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if claim.exists():
            claimed = json.loads(claim.read_text())
            if (
                claimed.get("state") != "helper_owned"
                or {key: claimed.get(key) for key in identity} != identity
                or not isinstance(claimed.get("owner_id"), str)
                or not str(claimed.get("claimed_at", "")).endswith("Z")
            ):
                raise ValueError("foreign detached submit-helper claim")
            existing = _find_existing_job(
                str(identity["job_name"]), str(identity["submission_token"])
            )
            if existing is not None:
                _write_json_exclusive_atomic(
                    result,
                    {"state": "helper_complete", **identity, "job_id": existing},
                )
                return
            prior_attempt = str(claimed["attempt_id"])
            prior_stdout = prepared.with_name(
                f"submission.sbatch-stdout-{prior_attempt}.txt"
            )
            if prior_stdout.exists():
                prior_job_id = prior_stdout.read_text().strip().split(";", 1)[0]
                if prior_job_id.isdigit():
                    _write_json_exclusive_atomic(
                        result,
                        {"state": "helper_complete", **identity, "job_id": prior_job_id},
                    )
                    return
            if terminal_path.exists():
                prior_terminal = _validate_submit_helper_terminal(
                    json.loads(terminal_path.read_text()),
                    identity,
                    attempt_id=prior_attempt,
                )
                if prior_terminal["retryable"] is False:
                    return
            elif prior_stdout.exists():
                _close_submit_helper_failure(
                    prepared, identity, attempt_id=prior_attempt,
                    phase="ambiguous_after_spawn", retryable=False,
                    return_code=None,
                )
                return
            takeover = prepared.with_name(
                f"submission.helper-takeover-{owner['owner_id']}.json"
            )
            _write_json_exclusive_atomic(
                takeover,
                {"state": "stale_pre_spawn_takeover", "prior": claimed, **owner},
            )
            _write_json_replace_atomic(
                claim, {"state": "helper_owned", **identity, **owner}
            )
        else:
            _write_json_exclusive_atomic(
                claim, {"state": "helper_owned", **identity, **owner}
            )
        attempt_id = str(owner["attempt_id"])
        stdout_path = prepared.with_name(
            f"submission.sbatch-stdout-{attempt_id}.txt"
        )
        stderr_path = prepared.with_name(
            f"submission.sbatch-stderr-{attempt_id}.txt"
        )
        argv = list(identity["argv"])
        file_actions = [
            (os.POSIX_SPAWN_OPEN, 1, str(stdout_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
            (os.POSIX_SPAWN_OPEN, 2, str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
        ]
        try:
            child_pid = os.posix_spawnp(
                argv[0], argv, os.environ, file_actions=file_actions, setsid=True
            )
        except OSError:
            _close_submit_helper_failure(
                prepared, identity, attempt_id=attempt_id,
                phase="pre_submission_file_action_or_exec", retryable=True,
                return_code=None,
            )
            return
        _, wait_status = os.waitpid(child_pid, 0)
        return_code = os.waitstatus_to_exitcode(wait_status)
        if return_code != 0:
            visible_job, absence_proof = _bounded_nonzero_visibility(
                str(identity["job_name"]), str(identity["submission_token"])
            )
            if visible_job is not None:
                _write_json_exclusive_atomic(
                    result,
                    {"state": "helper_complete", **identity, "job_id": visible_job},
                )
                return
            if absence_proof is not None:
                _close_submit_helper_failure(
                    prepared, identity, attempt_id=attempt_id,
                    phase="sbatch_nonzero_scheduler_proved_absent",
                    retryable=True, return_code=return_code,
                    visibility_evidence=absence_proof,
                )
                return
            _close_submit_helper_failure(
                prepared, identity, attempt_id=attempt_id,
                phase="sbatch_nonzero_ambiguous", retryable=False,
                return_code=return_code,
            )
            return
        job_id = stdout_path.read_text().strip().split(";", 1)[0]
        if not job_id.isdigit():
            reconciled = _find_existing_job(
                str(identity["job_name"]), str(identity["submission_token"])
            )
            if reconciled is None:
                _close_submit_helper_failure(
                    prepared, identity, attempt_id=attempt_id,
                    phase="invalid_success_output", retryable=False, return_code=0,
                )
                return
            job_id = reconciled
        _write_json_exclusive_atomic(
            result, {"state": "helper_complete", **identity, "job_id": job_id}
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for stage in ("pilot", "cp2-survivors", "long"):
        command = subparsers.add_parser(stage)
        command.set_defaults(stage=stage)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--replicate-index", type=int, default=0)
        if stage != "pilot":
            command.add_argument("--survivors", type=Path, required=True)
        if stage == "long":
            command.add_argument("--replicates", type=int, choices=(3,), required=True)
        action = command.add_mutually_exclusive_group(required=True)
        action.add_argument("--test-only", action="store_true")
        action.add_argument("--submit-next", action="store_true")
    validate = subparsers.add_parser("validate-replicate")
    validate.add_argument("--replicate-json", type=Path, required=True)
    validate.add_argument("--inside-job", action="store_true", required=True)
    validate.add_argument("--job-id", required=True)
    select = subparsers.add_parser("select-arm")
    select.add_argument("--replicate-json", type=Path, required=True)
    select.add_argument("--execution-order", type=int, choices=(0, 1, 2), required=True)
    run = subparsers.add_parser("run-arm")
    run.add_argument("--arm-json", type=Path, required=True)
    run.add_argument("--job-id", required=True)
    close_arm = subparsers.add_parser("close-arm")
    close_arm.add_argument("--arm-json", type=Path, required=True)
    close_arm.add_argument("--job-id", required=True)
    close_arm.add_argument("--exit-code", type=int, required=True)
    close_replicate = subparsers.add_parser("close-replicate")
    close_replicate.add_argument("--replicate-json", type=Path, required=True)
    close_replicate.add_argument("--job-id", required=True)
    close_replicate.add_argument("--exit-code", type=int, required=True)
    submit = subparsers.add_parser("submit-helper")
    submit.add_argument("--prepared", type=Path, required=True)
    submit.add_argument("--result", type=Path, required=True)
    return parser


def _matrix(args: argparse.Namespace, spec: ExperimentSpec) -> tuple[MatchedReplicate, ...]:
    if args.stage == "pilot":
        return build_pilot_matrix(spec, replicate_index=args.replicate_index)
    survivors = _load_survivors(
        args.survivors,
        expected_source_topology=(
            "packed-cp1" if args.stage == "cp2-survivors" else "packed-cp2"
        ),
    )
    if args.stage == "cp2-survivors":
        return build_cp2_survivor_matrix(
            spec, survivors=survivors, replicate_index=args.replicate_index
        )
    return build_long_matrix(
        spec, promoted=survivors, replicate_indices=tuple(range(args.replicates))
    )


def _arm_json(path: Path) -> Arm:
    return _arm_from_dict(json.loads(path.read_text()))


def validate_replicate_inside_job(path: Path, job_id: str) -> None:
    replicate = _load_replicate(path)
    submission = json.loads((_replicate_root(replicate) / "submission.json").read_text())
    if _validate_submission_receipt(
        submission, _submission_identity(path, replicate)
    ) != job_id:
        raise ValueError("inside-job identity differs from submission receipt")
    if sorted(arm.execution_order for arm in replicate.arms) != [0, 1, 2]:
        raise ValueError("replicate execution order must be exactly 0,1,2")
    for arm in replicate.arms:
        validate_arm_runtime(arm)


def select_arm(path: Path, execution_order: int) -> Path:
    replicate = _load_replicate(path)
    arm = next(
        arm for arm in replicate.arms if arm.execution_order == execution_order
    )
    arm_path = Path(arm.result_dir) / "arm.json"
    expected = json.loads(json.dumps(asdict(arm), sort_keys=True))
    if arm_path.exists():
        if json.loads(arm_path.read_text()) != expected:
            raise ValueError("arm JSON identity collision")
    else:
        _write_json_exclusive_atomic(arm_path, expected)
    return arm_path


def _arm_receipt_identity(arm: Arm, job_id: str) -> dict[str, object]:
    if arm.manifest_sha256 is None:
        raise ValueError("arm receipt requires a manifest digest")
    return {
        "experiment_id": arm.experiment_id,
        "replicate_id": arm.replicate_id,
        "arm_id": arm.name,
        "execution_order": arm.execution_order,
        "manifest_sha256": arm.manifest_sha256,
        "job_id": job_id,
        "product_head": arm.product_head,
        "harness_head": arm.harness_head,
        "submodule_shas": arm.submodule_shas,
        "container_digest": arm.container_digest,
        "model_snapshot_sha256": arm.model_snapshot_sha256,
        "draft_snapshot_sha256": arm.draft_snapshot_sha256,
        "invariant_config_sha256": arm.invariant_config_sha256,
        "resolved_config_sha256": arm.resolved_config_sha256,
        "wandb_id": arm.wandb_id,
        "wandb_entity": arm.wandb_entity,
        "wandb_project": arm.wandb_project,
    }


def _validate_arm_submission(arm: Arm, job_id: str) -> None:
    submission = json.loads((Path(arm.result_dir).parent / "submission.json").read_text())
    expected = {
        "manifest_sha256": arm.manifest_sha256,
        "replicate_id": arm.replicate_id,
        "product_head": arm.product_head,
        "harness_head": arm.harness_head,
        "container_digest": arm.container_digest,
        "model_snapshot_sha256": arm.model_snapshot_sha256,
        "draft_snapshot_sha256": arm.draft_snapshot_sha256,
        "invariant_config_sha256": arm.invariant_config_sha256,
    }
    if (
        submission.get("state") != "submitted"
        or submission.get("job_id") != job_id
        or any(submission.get(key) != value for key, value in expected.items())
        or submission.get("resolved_config_sha256_by_arm", {}).get(arm.name)
        != arm.resolved_config_sha256
    ):
        raise ValueError("arm identity differs from submitted manifest")


def normalize_unfiltered_wandb_rows(
    rows: list[Mapping[str, object]], arm: Arm
) -> list[dict[str, object]]:
    mapping = dict(arm.canonical_metric_keys)
    required_logical_keys = (
        "total_step_time", "generation_tps", "acceptance_rate",
        "mean_accepted_length", "mean_total_reward", "gen_kl_error",
    )
    required_keys = {"_step", *(mapping[key] for key in required_logical_keys)}
    if arm.kind != "fixed":
        required_keys.add(mapping["applied_draft_version"])

    def contains_nonfinite(value: object) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, float):
            return not isfinite(value)
        if isinstance(value, Mapping):
            return any(contains_nonfinite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_nonfinite(item) for item in value)
        return False

    normalized = []
    for row in rows:
        clean = {}
        for key in sorted(row):
            value = row[key]
            if contains_nonfinite(value):
                if key in required_keys:
                    clean[key] = None
                continue
            clean[key] = value
        normalized.append(clean)
    return normalized


def freeze_wandb_rows(arm: Arm, job_id: str) -> None:
    import wandb

    entity = os.environ["WANDB_ENTITY"]
    project = os.environ["WANDB_PROJECT"]
    if entity != arm.wandb_entity or project != arm.wandb_project:
        raise ValueError("W&B runtime environment differs from arm identity")
    unfiltered_rows = list(
        wandb.Api().run(f"{entity}/{project}/{arm.wandb_id}").scan_history()
    )
    rows = normalize_unfiltered_wandb_rows(unfiltered_rows, arm)
    raw_bytes = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    raw_path = Path(arm.result_dir) / "raw-wandb.json"
    _write_bytes_exclusive_atomic(raw_path, raw_bytes)
    _write_json_exclusive_atomic(
        Path(arm.result_dir) / "wandb.json",
        {
            **_arm_receipt_identity(arm, job_id),
            "run_path": f"{entity}/{project}/{arm.wandb_id}",
            "raw_rows_path": str(raw_path.resolve()),
            "raw_rows_size_bytes": len(raw_bytes),
            "raw_rows_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "frozen": True,
        },
    )


def run_arm(path: Path, job_id: str) -> None:
    arm = _arm_json(path)
    validate_arm_runtime(arm)
    _validate_arm_submission(arm, job_id)
    training_argv = [
        "uv", "run", "examples/run_grpo.py", "--config", arm.resolved_config_path,
    ]
    required_env = (
        "PATH", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID",
        "WANDB_ENTITY", "WANDB_PROJECT",
    )
    missing = [key for key in required_env if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"missing required runtime environment: {missing}")
    allowlisted_env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "LD_LIBRARY_PATH"}
        or key.startswith(("CUDA_", "NCCL_", "WANDB_", "SLURM_"))
    }
    allowlisted_env["PYTHONPATH"] = arm.product_repo
    allowlisted_env["NEMO_RL_SOURCE_ROOT"] = arm.product_repo
    _write_json_exclusive_atomic(
        Path(arm.result_dir) / "provenance.json",
        {
            **_arm_receipt_identity(arm, job_id),
            "source_cwd": arm.product_repo,
            "runtime_env_keys": sorted(allowlisted_env),
        },
    )
    subprocess.run(
        training_argv,
        check=True,
        cwd=arm.product_repo,
        env=allowlisted_env,
    )
    freeze_wandb_rows(arm, job_id)


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "cadence-checkpoint-receipt.json":
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def load_and_validate_runtime_ledger_segments(
    schedule: dict[str, object],
) -> list[dict[str, object]]:
    segments = schedule.get("decision_ledger_segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("online runtime schedule is missing its decision ledger")
    rows = []
    next_id = 1
    for segment in segments:
        raw = Path(segment["path"]).read_bytes()
        parsed = [json.loads(line) for line in raw.splitlines()]
        ids = [item.get("decision_id") for item in parsed]
        if (
            segment.get("size_bytes") != len(raw)
            or segment.get("sha256") != hashlib.sha256(raw).hexdigest()
            or ids != list(range(next_id, next_id + len(parsed)))
            or segment.get("first_decision_id") != next_id
            or segment.get("last_decision_id") != next_id + len(parsed) - 1
            or segment.get("entry_count") != len(parsed)
        ):
            raise ValueError("runtime decision-ledger receipt mismatch")
        rows.extend(parsed)
        next_id += len(parsed)
    if len(rows) != schedule.get("current_step"):
        raise ValueError("runtime decision ledger does not cover every step")
    return rows


def validate_product_runtime_receipts(
    arm: Arm,
) -> tuple[dict[str, object], dict[str, object]]:
    result_dir = Path(arm.result_dir).resolve()
    checkpoint_path = result_dir / "checkpoint-runtime.json"
    schedule_path = result_dir / "schedule-runtime.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    expected_final = (result_dir / "checkpoints" / f"step_{arm.max_steps}").resolve()
    if (
        checkpoint.get("current_step") != arm.max_steps
        or Path(str(checkpoint.get("checkpoint_path"))).resolve() != expected_final
        or not expected_final.is_dir()
        or checkpoint.get("checkpoint_tree_sha256") != _sha256_tree(expected_final)
        or schedule.get("current_step") != arm.max_steps
    ):
        raise ValueError("product checkpoint/schedule runtime receipt mismatch")
    if arm.max_steps >= 400 and arm.resume_after_step is None:
        step400_path = result_dir / "checkpoint-runtime-step_400.json"
        step400 = json.loads(step400_path.read_text())
        expected_step400 = (result_dir / "checkpoints" / "step_400").resolve()
        if (
            step400.get("current_step") != 400
            or Path(str(step400.get("checkpoint_path"))).resolve()
            != expected_step400
            or not expected_step400.is_dir()
            or step400.get("checkpoint_tree_sha256")
            != _sha256_tree(expected_step400)
        ):
            raise ValueError("required Step-400 checkpoint receipt mismatch")
        step400_raw = step400_path.read_bytes()
        checkpoint["required_checkpoint_receipts"] = [{
            "path": str(step400_path.resolve()),
            "size_bytes": len(step400_raw),
            "sha256": hashlib.sha256(step400_raw).hexdigest(),
        }]
    else:
        checkpoint["required_checkpoint_receipts"] = []
    if arm.kind == "fixed":
        if (
            schedule.get("mode") != "disabled"
            or schedule.get("decision_ledger_segments") != []
        ):
            raise ValueError("fixed control runtime schedule must be neutral")
    else:
        load_and_validate_runtime_ledger_segments(schedule)
        snapshot = checkpoint.get("applied_draft_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("online checkpoint is missing applied draft snapshot")
        snapshot_path = Path(str(snapshot.get("path"))).resolve()
        snapshot_raw = snapshot_path.read_bytes()
        expected_version = (
            schedule["refit_versions"][-1]["applied_draft_version"]
            if schedule.get("refit_versions")
            else 0
        )
        if (
            not snapshot_path.is_relative_to(result_dir)
            or snapshot.get("version") != expected_version
            or snapshot.get("size_bytes") != len(snapshot_raw)
            or snapshot.get("sha256") != hashlib.sha256(snapshot_raw).hexdigest()
        ):
            raise ValueError("applied draft snapshot runtime provenance mismatch")
    return checkpoint, schedule


def close_arm(path: Path, job_id: str, exit_code: int) -> None:
    arm = _arm_json(path)
    _validate_arm_submission(arm, job_id)
    identity = _arm_receipt_identity(arm, job_id)
    if exit_code == 0:
        for receipt_name in ("provenance", "wandb"):
            receipt = json.loads(
                (Path(arm.result_dir) / f"{receipt_name}.json").read_text()
            )
            if {key: receipt.get(key) for key in identity} != identity:
                raise ValueError(f"{receipt_name} receipt identity mismatch")
        checkpoint_runtime, schedule_runtime = validate_product_runtime_receipts(arm)
        for receipt_name, runtime in (
            ("checkpoint", checkpoint_runtime),
            ("schedule", schedule_runtime),
        ):
            _write_json_exclusive_atomic(
                Path(arm.result_dir) / f"{receipt_name}.json",
                {**runtime, **identity},
            )
        completed_policy_steps = int(
            json.loads((Path(arm.result_dir) / "checkpoint.json").read_text())[
                "current_step"
            ]
        )
    else:
        completed_policy_steps = 0
    _write_json_exclusive_atomic(
        Path(arm.result_dir) / "terminal.json",
        {
            **identity,
            "exit_code": exit_code,
            "terminal": True,
            "completed_policy_steps": completed_policy_steps,
        },
    )


def close_replicate(path: Path, job_id: str, exit_code: int) -> None:
    replicate = _load_replicate(path)
    submission = json.loads((_replicate_root(replicate) / "submission.json").read_text())
    if _validate_submission_receipt(
        submission, _submission_identity(path, replicate)
    ) != job_id:
        raise ValueError("close-replicate job ID differs from submission")
    terminals = []
    for arm in sorted(replicate.arms, key=lambda item: item.execution_order):
        terminal_path = Path(arm.result_dir) / "terminal.json"
        if terminal_path.exists():
            terminal = json.loads(terminal_path.read_text())
            if terminal["job_id"] != job_id:
                raise ValueError("arm terminal job ID mismatch")
            terminals.append(terminal)
        elif exit_code == 0:
            raise ValueError("successful replicate is missing an arm terminal")
    _write_json_exclusive_atomic(
        _replicate_root(replicate) / "terminal.json",
        {
            **_submission_identity(path, replicate),
            "job_id": job_id,
            "state": "terminal",
            "exit_code": exit_code,
            "terminal": True,
            "completed_arm_ids": [item["arm_id"] for item in terminals],
        },
    )


def _write_or_verify_stage_manifest(
    path: Path, spec: ExperimentSpec, matrix: tuple[MatchedReplicate, ...]
) -> None:
    if matrix:
        manifest = render_manifest(matrix)
    else:
        manifest = {
            "schema_version": 1,
            "experiment_id": spec.experiment_id,
            "replicates": [],
        }
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise FileExistsError("manifest path collides with different immutable content")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")


def write_no_survivor_terminal(path: Path, spec: ExperimentSpec) -> None:
    receipt_path = path.with_name(f"{path.stem}-no-survivor.json")
    empty_manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "replicates": [],
        "no_survivor_receipt_path": str(receipt_path.resolve()),
    }
    empty_manifest["manifest_sha256"] = _canonical_sha256(empty_manifest)
    if path.exists():
        if json.loads(path.read_text()) != empty_manifest:
            raise FileExistsError("no-survivor manifest identity collision")
    else:
        _write_json_exclusive_atomic(path, empty_manifest)
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "stage": "long",
        "status": "no_survivor",
        "terminal": True,
        "reason": "no_pilot_survivors",
        "candidates": {},
        "selected_candidate": None,
        "production_supported": False,
        "recommendation": "always",
        "manifest_path": str(path.resolve()),
    }
    payload["manifest_sha256"] = empty_manifest["manifest_sha256"]
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != payload:
            raise FileExistsError("no-survivor receipt identity collision")
        return
    _write_json_exclusive_atomic(receipt_path, payload)


def _replicate_json(path: Path, replicate: MatchedReplicate) -> Path:
    result = path.with_name(f"{path.stem}-{replicate.replicate_id}.json")
    manifest = json.loads(path.read_text())
    manifest_sha256 = recompute_manifest_sha256(manifest)
    matches = [
        item for item in manifest["replicates"]
        if item["replicate_id"] == replicate.replicate_id
    ]
    if len(matches) != 1:
        raise ValueError("replicate is absent or duplicated in manifest")
    payload = {
        "schema_version": 1,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": manifest_sha256,
        "replicate": matches[0],
    }
    if result.exists():
        if json.loads(result.read_text()) != payload:
            raise FileExistsError("replicate path collides with different immutable content")
        return result
    with result.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "submit-helper":
        submit_helper(args.prepared, args.result)
        return 0
    if args.command == "validate-replicate":
        validate_replicate_inside_job(args.replicate_json, args.job_id)
        return 0
    if args.command == "select-arm":
        print(select_arm(args.replicate_json, args.execution_order))
        return 0
    if args.command == "run-arm":
        run_arm(args.arm_json, args.job_id)
        return 0
    if args.command == "close-arm":
        close_arm(args.arm_json, args.job_id, args.exit_code)
        return 0
    if args.command == "close-replicate":
        close_replicate(args.replicate_json, args.job_id, args.exit_code)
        return 0
    spec = _load_spec(args.spec)
    validate_source_container_config(spec)
    matrix = _matrix(args, spec)
    if args.command == "long" and not matrix:
        write_no_survivor_terminal(args.manifest, spec)
        return 0
    for replicate in matrix:
        replicate.validate_parity()
        for arm in replicate.arms:
            materialize_and_validate_resolved_config(arm)
    validate_collision_free_identity(matrix)
    _write_or_verify_stage_manifest(args.manifest, spec, matrix)
    entries = tuple(
        (
            replicate_path,
            _load_replicate(replicate_path),
        )
        for replicate in matrix
        for replicate_path in (_replicate_json(args.manifest, replicate),)
    )
    commands = [
        sbatch_argv(replicate_path, replicate)
        for replicate_path, replicate in entries
    ]
    if args.test_only:
        for command in commands:
            run_command(["sbatch", "--test-only", *command[1:]])
    elif commands:
        pending = next_unsubmitted_replicate(entries)
        if pending is None:
            return 0
        pending_path, pending_replicate = pending
        submit_replicate(pending_path, pending_replicate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`validate_source_container_config` refuses stale/non-numeric/non-maximal FairShare evidence, a dirty recursive tree or submodule, a local head not equal to its upstream remote head, any recursive gitlink differing from `submodule_shas`, a container file whose bytes do not match the immutable `sha256:` digest, a model/draft/data artifact not matching its pinned revision/path/hash, unresolved Hydra interpolation, or a SHA/config/provenance mismatch. Triplet parity compares the recomputed held-field `invariant_config_sha256`, while every arm has a treatment-specific full `resolved_config_sha256` checked against its own materialized Hydra config. `validate_collision_free_identity` rejects duplicate replicate IDs, arm IDs, W&B IDs, or stage/max-step result paths. `next_unsubmitted_replicate` validates and skips terminal submission receipts. `submit_replicate` never writes a pre-`sbatch` marker in the launcher. Its detached helper holds an OS advisory lock, and its claim binds owner UUID, host, PID, UTC-Z timestamp, and the full submission identity. A process crash releases the lock automatically. While holding that lock, a replacement helper first validates the exact identity and performs allocation-only `squeue`/`sacct` reconciliation. A numeric `--parsable` job ID in the attempt-specific stdout or a unique reconciled scheduler job is successful submission evidence. Stdout existence alone is never proof: empty/malformed output without an exact terminal receipt closes as non-retryable `ambiguous_after_spawn`. A pre-child `posix_spawnp` failure is intrinsically retryable. A nonzero `sbatch` exit is ambiguous and non-retryable by default; it becomes retryable only when six successful allocation-only `squeue` plus `sacct` reconciliations over at least 60 seconds find no accepted job for the exact unique submission token, with the timestamped absence evidence persisted in the terminal receipt. If any query fails, the elapsed window is short, or a job appears, absence is not proven; a visible job closes successfully, otherwise the attempt fails closed. A retry writes an immutable takeover receipt before one new attempt. Every helper exit therefore leaves either a successful job ID or an explicit terminal failure, never an indefinite pending state. Both exclusive hard-link writers fsync the parent directory, and successful result/receipt replacement does the same. Foreign claims/results, invalid retry flags/phases, and multiple jobs fail closed; a validated `submitted` receipt remains idempotent.

One `sbatch` command therefore names `run_replicate.sh` and one replicate JSON. All three arms share that allocation/job ID but retain separate arm IDs, W&B IDs, result directories, and terminal receipts. Each receipt repeats the experiment ID, manifest SHA, product/harness/submodule SHAs, container digest, resolved-config digest, replicate ID, arm ID, execution order, and shared job ID; disagreement is fatal.

Use this complete shell entrypoint; argument/config decoding stays in typed Python and returns an argv JSON array, so the shell never evaluates manifest text:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: run_replicate.sh REPLICATE_JSON" >&2
  exit 2
fi
replicate_json="$1"
readonly harness_root="${CADENCE_HARNESS_ROOT:?}"
readonly product_root="${NEMO_RL_SOURCE_ROOT:?}"
cd "$harness_root"
export PYTHONPATH="$harness_root:$product_root"
python_cmd=(uv run python)

"${python_cmd[@]}" -m research.qwen3_8b_draft_cadence.launch \
  validate-replicate --replicate-json "$replicate_json" \
  --inside-job --job-id "${SLURM_JOB_ID:?}"

for execution_order in 0 1 2; do
  arm_json="$("${python_cmd[@]}" -m research.qwen3_8b_draft_cadence.launch \
    select-arm --replicate-json "$replicate_json" \
    --execution-order "$execution_order")"
  status=0
  "${python_cmd[@]}" -m research.qwen3_8b_draft_cadence.launch \
    run-arm --arm-json "$arm_json" --job-id "$SLURM_JOB_ID" || status=$?
  "${python_cmd[@]}" -m research.qwen3_8b_draft_cadence.launch \
    close-arm --arm-json "$arm_json" --job-id "$SLURM_JOB_ID" \
    --exit-code "$status"
  if [[ "$status" -ne 0 ]]; then
    "${python_cmd[@]}" -m research.qwen3_8b_draft_cadence.launch \
      close-replicate --replicate-json "$replicate_json" \
      --job-id "$SLURM_JOB_ID" --exit-code "$status"
    exit "$status"
  fi
done
"${python_cmd[@]}" -m research.qwen3_8b_draft_cadence.launch \
  close-replicate --replicate-json "$replicate_json" \
  --job-id "$SLURM_JOB_ID" --exit-code 0
```

The parser implements every command invoked by `run_replicate.sh`: `validate-replicate`, `select-arm`, `run-arm`, `close-arm`, and `close-replicate`. `validate-replicate` rechecks submission/job identity plus exact recursive source, container, invariant-config, and per-arm full-config digests inside the allocation. `select-arm` validates exact orders `0,1,2`, exclusively materializes the chosen arm JSON, and writes only its path to stdout; this is data selection, never shell evaluation. `run-arm` revalidates the arm and invokes `subprocess.run(training_argv, check=True, env=allowlisted_env)`. `close-arm` and `close-replicate` atomically/exclusively write terminal receipts with the real exit code, shared job ID, and completed arm IDs. The runtime environment contains only `PATH`, optional `LD_LIBRARY_PATH`, CUDA/NCCL/W&B/SLURM variables, and the explicit product-source `PYTHONPATH`/`NEMO_RL_SOURCE_ROOT`; it never forwards arbitrary caller variables.

- [ ] **Step 4: Run the GREEN launcher tests and shell/static checks.**

Run: `uv run --group test pytest -q research/qwen3_8b_draft_cadence/tests/test_contract.py && uv run ruff check research/qwen3_8b_draft_cadence && uv run pyrefly check research/qwen3_8b_draft_cadence && bash -n research/qwen3_8b_draft_cadence/run_replicate.sh`

Expected: tests PASS, Ruff and Pyrefly report no errors, and `bash -n` exits 0.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add research/qwen3_8b_draft_cadence/__init__.py research/qwen3_8b_draft_cadence/manifest.py research/qwen3_8b_draft_cadence/launch.py research/qwen3_8b_draft_cadence/run_replicate.sh research/qwen3_8b_draft_cadence/tests/test_contract.py research/qwen3_8b_draft_cadence/README.md
git commit -S -s -m "perf(draft): add cadence experiment launcher"
git verify-commit HEAD
git push
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{u}')"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

Expected: signature verification exits 0.

### Task 2: Implement fail-closed paired analysis

**Files:**
- Create: `research/qwen3_8b_draft_cadence/analyze.py`
- Create: `research/qwen3_8b_draft_cadence/tests/test_analysis.py`
- Create: `research/qwen3_8b_draft_cadence/tests/controller_runtime_fixture.py`
- Modify: `research/qwen3_8b_draft_cadence/README.md`

**Interfaces:**
- Consumes: canonical W&B rows, checkpoint receipt, arm manifest, and predeclared fresh/resumed windows.
- Produces: executable `controller_runtime_arm_fixture(...)` coverage for real sync/single-controller `always`/fixed-sparse/fixed-refit-only/adaptive runtime receipts, `windows_for_run(resume_after_step: int | None) -> tuple[tuple[int, int], ...]`, `validate_arm_receipts(arm: Arm, receipts: Mapping[str, object]) -> None`, `paired_summary(candidate: list[float], control: list[float]) -> ConfidenceInterval`, `evaluate_candidate(metrics: CandidateMetrics) -> CandidateDecision`, `summary.json`, `summary.csv`, and Markdown with every frozen gate and source receipt.

- [ ] **Step 1: Write RED overhead, paired `gen_kl_error`, window, and missing-key tests.**

```python
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from research.qwen3_8b_draft_cadence.analyze import (
    ArmEvidence,
    ArtifactBinding,
    CandidateMetrics,
    FIXED_NOT_APPLICABLE_KEYS,
    REQUIRED_CANONICAL_KEYS,
    analyze_manifest,
    analyzer_main,
    build_candidate_metrics,
    evaluate_gen_kl,
    evaluate_candidate,
    load_wandb_rows,
    merge_and_filter_wandb_rows,
    load_decision_ledger,
    load_bound_manifest,
    overhead_reduction,
    paired_summary,
    select_pilot_survivors,
    validate_arm_receipts,
    windows_for_run,
)
from research.qwen3_8b_draft_cadence.manifest import (
    ExperimentSpec,
    build_cp2_survivor_matrix,
    build_long_matrix,
    build_pilot_matrix,
    render_manifest,
)
from research.qwen3_8b_draft_cadence.tests.controller_runtime_fixture import (
    controller_runtime_arm_fixture,
)


def analysis_experiment_spec() -> ExperimentSpec:
    base_resolved_config = {
        "grpo": {"max_num_steps": 1000, "seed": 1234},
        "checkpointing": {
            "enabled": True, "save_period": 100, "keep_top_k": None,
        },
        "logger": {"log_dir": "/immutable/overridden-per-arm"},
        "cadence_runtime": {"enabled": True, "result_dir": "/immutable/overridden-per-arm", "required_checkpoint_steps": []},
        "experiment_provenance": {},
        "data": {
            "shuffle": False,
            "train": {"dataset_name": "ResponseDataset", "data_path": "/immutable/data-order.jsonl", "seed": 1234},
        },
        "policy": {
            "model_name": "/immutable/model/qwen3-8b-rev",
            "draft": {
                "enabled": True,
                "model_name": "/immutable/draft/qwen3-8b-dflash-rev",
                "model_revision": "qwen3-8b-dflash-rev",
                "gamma": 5,
                "seed": 1234,
                "update_schedule": {"mode": "always"},
            },
            "generation": {
                "vllm_kwargs": {
                    "speculative_config": {"num_speculative_tokens": 5},
                    "compilation_config": {"backend": "eager", "cudagraph_mode": "PIECEWISE"},
                }
            },
            "megatron_cfg": {
                "tensor_model_parallel_size": 2,
                "context_parallel_size": 1,
                "sequence_parallel": True,
            },
            "sequence_packing": {"enabled": True},
        },
    }
    return ExperimentSpec(
        experiment_id="018f47a6-7d91-7d4a-8cc9-6c4c8e58a101",
        product_repo="/workspace/NeMo-RL",
        harness_repo="/workspace/cadence-harness",
        product_head="1" * 40,
        harness_head="3" * 40,
        submodule_shas=(("3rdparty/Megatron-LM", "5" * 40),),
        container_image="/lustre/images/nemo-rl-sha256_2222.sqsh",
        container_digest="sha256:" + "2" * 64,
        base_config_path=(
            "/workspace/NeMo-RL/examples/configs/recipes/llm/"
            "grpo-qwen3-8b-1n8g-megatron-dflash-cadence.yaml"
        ),
        base_resolved_config=base_resolved_config,
        base_resolved_config_sha256=hashlib.sha256(json.dumps(
            base_resolved_config, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        model_snapshot_path="/immutable/model/qwen3-8b-rev",
        draft_snapshot_path="/immutable/draft/qwen3-8b-dflash-rev",
        model_snapshot_sha256="a" * 64,
        draft_snapshot_sha256="b" * 64,
        data_order_path="/immutable/data/math-v1-data-order-6.jsonl",
        model_revision="qwen3-8b-rev",
        draft_revision="qwen3-8b-dflash-rev",
        dataset_revision="math-v1",
        data_order_sha256="6" * 64,
        k=5,
        global_batch_size=64,
        micro_batch_size=2,
        cuda_graph_settings=(
            ("backend", "eager"),
            ("cudagraph_mode", "PIECEWISE"),
        ),
        canonical_metric_keys=tuple(
            (key, key) for key in sorted(REQUIRED_CANONICAL_KEYS)
        ),
        cluster="oci-hsg",
        account="account-a",
        partition="batch",
        nodes=1,
        gpus_per_node=4,
        time_limit="08:00:00",
        fairshare_checked_at="2026-08-22T12:00:00Z",
        fairshare_scores=(("account-a", 0.8), ("account-b", 0.4)),
        fairshare_eligible_accounts=("account-a", "account-b"),
        fairshare_selection_reason="highest eligible FairShare",
        wandb_entity="nvidia-nemo-rl",
        wandb_project="qwen3-draft-cadence",
        result_root="/lustre/cadence-test",
    )


def matching_arm_and_receipts():
    pilot = build_pilot_matrix(analysis_experiment_spec(), replicate_index=0)[0]
    source = next(arm for arm in pilot.arms if arm.kind == "always")
    resolved_config = json.loads(json.dumps(source.resolved_config))
    resolved_config["grpo"]["max_num_steps"] = 3
    arm = replace(
        source,
        max_steps=3,
        resolved_config=resolved_config,
        resolved_config_sha256=hashlib.sha256(json.dumps(
            resolved_config, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        manifest_sha256="7" * 64,
    )
    identity = {
        "experiment_id": arm.experiment_id,
        "replicate_id": arm.replicate_id,
        "arm_id": arm.name,
        "execution_order": arm.execution_order,
        "manifest_sha256": "7" * 64,
        "job_id": "123456",
        "product_head": arm.product_head,
        "harness_head": arm.harness_head,
        "submodule_shas": arm.submodule_shas,
        "container_digest": arm.container_digest,
        "model_snapshot_sha256": arm.model_snapshot_sha256,
        "draft_snapshot_sha256": arm.draft_snapshot_sha256,
        "invariant_config_sha256": arm.invariant_config_sha256,
        "resolved_config_sha256": arm.resolved_config_sha256,
        "wandb_id": arm.wandb_id,
        "wandb_entity": arm.wandb_entity,
        "wandb_project": arm.wandb_project,
    }
    receipts = {
        name: dict(identity)
        for name in ("provenance", "wandb", "checkpoint", "schedule", "terminal")
    }
    receipts["terminal"].update(
        {"completed_policy_steps": 3, "terminal": True, "exit_code": 0}
    )
    receipts["wandb"].update({
        "raw_rows_sha256": "a" * 64,
        "raw_rows_size_bytes": 123,
        "frozen": True,
    })
    receipts["checkpoint"].update({"current_step": 3})
    receipts["schedule"].update(
        {
            "attempted_updates": 3,
            "successful_updates": 3,
            "failed_updates": 0,
            "skipped_updates": 0,
            "attempted_refits": 3,
            "successful_refits": 3,
            "failed_refits": 0,
            "skipped_refits": 0,
            "decision_ids": [1, 2, 3],
            "global_steps": [1, 2, 3],
            "updated_steps": [1, 2, 3],
            "refit_steps": [1, 2, 3],
            "forced_update_steps": [],
            "forced_updates": 0,
            "forced_refit_steps": [],
            "forced_refits": 0,
            "update_receipts": ["update-1", "update-2", "update-3"],
            "post_event_observations": [
                {"refit_step": 1, "observation_step": 2, "applied_draft_version": 1, "acceptance_rate": 0.70},
                {"refit_step": 2, "observation_step": 3, "applied_draft_version": 2, "acceptance_rate": 0.70},
            ],
            "pending_post_event_steps": [3],
            "refit_versions": [
                {"refit_step": 1, "applied_draft_version": 1},
                {"refit_step": 2, "applied_draft_version": 2},
                {"refit_step": 3, "applied_draft_version": 3},
            ],
            "decision_reasons": ["always", "always", "always"],
            "policy_refit_count": 3,
        }
    )
    return arm, receipts


@pytest.mark.parametrize(
    ("controller_kind", "mode"),
    [
        ("sync", "always"),
        ("sync", "fixed_sparse"),
        ("sync", "fixed_refit_only"),
        ("sync", "adaptive"),
        ("single_controller", "always"),
        ("single_controller", "fixed_sparse"),
        ("single_controller", "fixed_refit_only"),
        ("single_controller", "adaptive"),
    ],
)
def test_real_controller_runtime_receipts_pass_analyzer_contract(
    tmp_path: Path, controller_kind: str, mode: str
) -> None:
    arm, receipts = controller_runtime_arm_fixture(
        tmp_path,
        controller_kind=controller_kind,
        mode=mode,
        completed_steps=3,
        cadence_runtime_enabled=True,
    ).run_to_terminal()
    validate_arm_receipts(arm, receipts)
    schedule = receipts["schedule"]
    assert len(schedule["update_receipts"]) == schedule["successful_updates"]
    assert all(Path(item["path"]).is_file() for item in schedule["update_receipts"])


def test_real_resumed_controller_prunes_preboundary_observations(
    tmp_path: Path,
) -> None:
    arm, receipts = controller_runtime_arm_fixture(
        tmp_path,
        controller_kind="sync",
        mode="always",
        completed_steps=3,
        cadence_runtime_enabled=True,
        resume_after_step=2,
    ).run_to_terminal()
    validate_arm_receipts(arm, receipts)
    assert [
        item["refit_step"]
        for item in receipts["schedule"]["post_event_observations"]
    ] == [2]


def raw_rows(
    arm, *, step_time: float, tps: float, first_step: int = 1
) -> list[dict[str, object]]:
    if arm.kind == "fixed":
        updates = []
        refits = []
    elif arm.schedule.mode == "always":
        updates = list(range(1, arm.max_steps + 1))
        refits = updates
    elif arm.schedule.mode == "fixed":
        refits = list(range(arm.schedule.interval, arm.max_steps + 1, arm.schedule.interval))
        updates = (
            list(range(1, arm.max_steps + 1))
            if arm.schedule.action == "refit_only" else refits
        )
    elif arm.schedule.mode == "adaptive":
        updates = list(
            range(arm.schedule.max_interval, arm.max_steps + 1, arm.schedule.max_interval)
        )
        refits = updates
    else:
        updates = []
        refits = []
    rows = []
    for step in range(first_step, 1001):
        row = {
            "_step": float(step),
            "total_step_time": step_time,
            "generation_tps": tps,
            "acceptance_rate": 0.70,
            "mean_accepted_length": 3.0,
            "mean_total_reward": 0.80,
            "gen_kl_error": 0.10,
        }
        if arm.kind != "fixed":
            row.update(
                applied_draft_version=float(max(
                    (refit for refit in refits if refit < step), default=0
                )),
            )
            if step in updates:
                row.update(draft_loss=1.0, draft_grad_norm=1.0)
        rows.append(row)
    return rows


def write_ledger_segments(
    root: Path,
    arm,
    *,
    update_steps: list[int],
    refit_steps: list[int],
    reasons: list[str],
) -> list[dict[str, object]]:
    if arm.kind == "fixed":
        return []
    boundaries = (
        ((1, 400), (401, arm.max_steps))
        if arm.resume_after_step == 400
        else ((1, arm.max_steps),)
    )
    receipts = []
    for first, last in boundaries:
        entries = [
            {
                "decision_id": step,
                "global_step": step,
                "update_requested": step in update_steps,
                "draft_refit_requested": step in refit_steps,
                "reason": reasons[step - 1],
                "forced": reasons[step - 1] == "max_interval",
                "applied_draft_version": max(
                    (value for value in refit_steps if value < step),
                    default=0,
                ),
                "outcome": {
                    "update_attempted": step in update_steps,
                    "update_successful": step in update_steps,
                    "update_skipped": step not in update_steps,
                    "draft_refit_attempted": step in refit_steps,
                    "draft_refit_successful": step in refit_steps,
                    "draft_refit_skipped": step not in refit_steps,
                    "forced_update": reasons[step - 1] == "max_interval",
                    "forced_refit": reasons[step - 1] == "max_interval",
                },
            }
            for step in range(first, last + 1)
        ]
        raw = b"".join(
            (
                json.dumps(entry, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            for entry in entries
        )
        path = root / arm.name / f"decision-ledger-{first}-{last}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        receipts.append({
            "path": str(path.resolve()),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "first_decision_id": first,
            "last_decision_id": last,
            "entry_count": len(entries),
        })
    return receipts


def complete_receipts(
    arm, *, job_id: str, artifact_root: Path
) -> dict[str, dict[str, object]]:
    completed = arm.max_steps
    identity = {
        "experiment_id": arm.experiment_id,
        "replicate_id": arm.replicate_id,
        "arm_id": arm.name,
        "execution_order": arm.execution_order,
        "manifest_sha256": "7" * 64,
        "job_id": job_id,
        "product_head": arm.product_head,
        "harness_head": arm.harness_head,
        "submodule_shas": arm.submodule_shas,
        "container_digest": arm.container_digest,
        "model_snapshot_sha256": arm.model_snapshot_sha256,
        "draft_snapshot_sha256": arm.draft_snapshot_sha256,
        "invariant_config_sha256": arm.invariant_config_sha256,
        "resolved_config_sha256": arm.resolved_config_sha256,
        "wandb_id": arm.wandb_id,
        "wandb_entity": arm.wandb_entity,
        "wandb_project": arm.wandb_project,
    }
    receipts = {
        name: dict(identity)
        for name in ("provenance", "wandb", "checkpoint", "schedule", "terminal")
    }
    if arm.kind == "fixed":
        update_steps: list[int] = []
        refit_steps: list[int] = []
        reasons = ["none"] * completed
    elif arm.schedule.mode == "always":
        update_steps = list(range(1, completed + 1))
        refit_steps = update_steps
        reasons = ["always"] * completed
    elif arm.schedule.mode == "adaptive":
        update_steps = list(
            range(arm.schedule.max_interval, completed + 1, arm.schedule.max_interval)
        )
        refit_steps = update_steps
        reasons = [
            "max_interval" if step in update_steps else "none"
            for step in range(1, completed + 1)
        ]
    else:
        interval_steps = list(
            range(arm.schedule.interval, completed + 1, arm.schedule.interval)
        )
        update_steps = (
            list(range(1, completed + 1))
            if arm.schedule.action == "refit_only" else interval_steps
        )
        refit_steps = interval_steps
        reasons = (
            ["fixed_interval"] * completed
            if arm.schedule.action == "refit_only"
            else [
                "fixed_interval" if step in refit_steps else "none"
                for step in range(1, completed + 1)
            ]
        )
    updates = len(update_steps)
    refits = len(refit_steps)
    forced_steps = update_steps if arm.schedule is not None and arm.schedule.mode == "adaptive" else []
    ledger_segments = write_ledger_segments(
        artifact_root,
        arm,
        update_steps=update_steps,
        refit_steps=refit_steps,
        reasons=reasons,
    )
    receipts["terminal"].update(
        {"completed_policy_steps": completed, "terminal": True, "exit_code": 0}
    )
    receipts["wandb"].update({
        "raw_rows_sha256": hashlib.sha256(arm.name.encode()).hexdigest(),
        "raw_rows_size_bytes": completed,
        "frozen": True,
    })
    receipts["checkpoint"].update(
        {"current_step": completed, "resumed_from": arm.resume_checkpoint}
    )
    if arm.kind == "fixed":
        receipts["schedule"].update(
            {
                "mode": "disabled",
                "attempted_updates": 0,
                "successful_updates": 0,
                "failed_updates": 0,
                "skipped_updates": 0,
                "attempted_refits": 0,
                "successful_refits": 0,
                "failed_refits": 0,
                "skipped_refits": 0,
                "forced_updates": 0,
                "forced_refits": 0,
                "policy_refit_count": completed,
                "decision_ids": [],
                "global_steps": [],
                "updated_steps": [],
                "refit_steps": [],
                "forced_update_steps": [],
                "forced_refit_steps": [],
                "update_receipts": [],
                "post_event_observations": [],
                "pending_post_event_steps": [],
                "refit_versions": [],
                "decision_reasons": [],
                "decision_ledger_segments": [],
                "not_applicable_metrics": sorted(FIXED_NOT_APPLICABLE_KEYS),
            }
        )
        return receipts
    receipts["schedule"].update(
        {
            "mode": arm.schedule.mode,
            "attempted_updates": updates,
            "successful_updates": updates,
            "failed_updates": 0,
            "skipped_updates": completed - updates,
            "attempted_refits": refits,
            "successful_refits": refits,
            "failed_refits": 0,
            "skipped_refits": completed - refits,
            "decision_ids": list(range(1, completed + 1)),
            "global_steps": list(range(1, completed + 1)),
            "updated_steps": update_steps,
            "refit_steps": refit_steps,
            "forced_update_steps": forced_steps,
            "forced_updates": len(forced_steps),
            "forced_refit_steps": forced_steps,
            "forced_refits": len(forced_steps),
            "update_receipts": [f"update-{step}" for step in update_steps],
            "post_event_observations": [
                {
                    "refit_step": step,
                    "observation_step": step + 1,
                    "applied_draft_version": step,
                    "acceptance_rate": 0.70,
                }
                for step in refit_steps
                if step < completed
                and (arm.resume_after_step is None or step >= arm.resume_after_step)
            ],
            "pending_post_event_steps": [
                step for step in refit_steps if step == completed
            ],
            "refit_versions": [
                {"refit_step": step, "applied_draft_version": step}
                for step in refit_steps
            ],
            "decision_reasons": reasons,
            "decision_ledger_segments": ledger_segments,
            "policy_refit_count": completed,
        }
    )
    return receipts


def bind_fixture_file(path: Path) -> ArtifactBinding:
    raw = path.read_bytes()
    return ArtifactBinding(
        path=str(path.resolve()),
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def make_evidence(
    arm,
    *,
    rows: list[dict[str, object]],
    job_id: str,
    artifact_root: Path,
) -> ArmEvidence:
    receipts = complete_receipts(
        arm, job_id=job_id, artifact_root=artifact_root
    )
    root = artifact_root / "evidence" / arm.name
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "raw-wandb.json"
    raw_bytes = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    raw_path.write_bytes(raw_bytes)
    receipts["wandb"].update(
        raw_rows_path=str(raw_path.resolve()),
        raw_rows_size_bytes=len(raw_bytes),
        raw_rows_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    if arm.kind != "fixed":
        snapshot_path = root / "applied-draft.safetensors"
        snapshot_path.write_bytes(f"snapshot:{arm.name}".encode())
        snapshot = bind_fixture_file(snapshot_path)
        receipts["checkpoint"]["applied_draft_snapshot"] = {
            "version": receipts["schedule"]["refit_steps"][-1],
            **asdict(snapshot),
        }
    receipt_paths = {}
    for name, receipt in receipts.items():
        path = root / f"{name}.json"
        path.write_text(json.dumps(receipt, sort_keys=True))
        receipt_paths[f"receipt:{name}"] = bind_fixture_file(path)
    artifacts = {"raw_wandb": bind_fixture_file(raw_path), **receipt_paths}
    for index, segment in enumerate(
        receipts["schedule"].get("decision_ledger_segments", [])
    ):
        artifacts[f"decision_ledger:{index}"] = bind_fixture_file(
            Path(segment["path"])
        )
    snapshot_receipt = receipts["checkpoint"].get("applied_draft_snapshot")
    if snapshot_receipt is not None:
        artifacts["applied_draft_snapshot"] = bind_fixture_file(
            Path(snapshot_receipt["path"])
        )
    return ArmEvidence(
        arm=arm,
        rows=rows,
        receipts=receipts,
        artifacts=artifacts,
        resume_after_step=arm.resume_after_step,
    )


def passing_candidate_metrics(tmp_path: Path) -> CandidateMetrics:
    all_replicates = build_long_matrix(
        analysis_experiment_spec(),
        promoted=("fixed_sparse_10",),
        replicate_indices=(0, 1, 2),
    )
    replicates = tuple(
        replicate for replicate in all_replicates
        if replicate.candidate.resume_after_step is None
    )
    resume_replicate = next(
        replicate for replicate in all_replicates
        if replicate.candidate.resume_after_step == 400
    )
    evidence = []
    for replicate_index, replicate in enumerate(replicates):
        by_kind = {
            arm.kind: replace(arm, manifest_sha256="7" * 64)
            for arm in replicate.arms
        }
        values = {
            "fixed": (10.0 + 0.1 * replicate_index, 100.0),
            "always": (12.0 + 0.1 * replicate_index, 100.0 + replicate_index),
            "candidate": (10.8 + 0.1 * replicate_index, 101.0 + replicate_index),
        }
        evidence.append(tuple(
            make_evidence(
                by_kind[kind],
                rows=raw_rows(
                    by_kind[kind], step_time=values[kind][0], tps=values[kind][1]
                ),
                job_id=f"job-r{replicate_index}",
                artifact_root=tmp_path,
            )
            for kind in ("fixed", "always", "candidate")
        ))
    resume_by_kind = {
        arm.kind: replace(arm, manifest_sha256="7" * 64)
        for arm in resume_replicate.arms
    }
    resume_evidence = tuple(
        make_evidence(
            resume_by_kind[kind],
            rows=raw_rows(
                resume_by_kind[kind], step_time=values[kind][0],
                tps=values[kind][1],
                first_step=401,
            ),
            job_id="job-resume-r0",
            artifact_root=tmp_path,
        )
        for kind in ("fixed", "always", "candidate")
    )
    return CandidateMetrics(
        canonical_keys={key: key for key in REQUIRED_CANONICAL_KEYS},
        matched_replicates=tuple(evidence),
        resume_replicate=resume_evidence,
    )


def test_overhead_reduction_uses_fixed_control() -> None:
    assert overhead_reduction(fixed=10.0, always=12.0, candidate=11.0) == 0.5
    with pytest.raises(ValueError, match="always overhead must be positive"):
        overhead_reduction(fixed=12.0, always=12.0, candidate=11.0)


def test_gen_kl_gate_uses_paired_candidate_minus_always_ci() -> None:
    candidate = [0.14, 0.15, 0.16]
    always = [0.10, 0.11, 0.12]
    decision = evaluate_gen_kl(candidate, always)
    assert decision.paired_differences == pytest.approx([0.04, 0.04, 0.04])
    assert decision.margin == pytest.approx(max(0.10 * 0.11, 0.01))
    assert decision.passed is False


def test_windows_are_frozen_by_resume_step() -> None:
    assert windows_for_run(resume_after_step=None)[0] == (1, 100)
    assert windows_for_run(resume_after_step=None)[-1] == (901, 1000)
    assert windows_for_run(resume_after_step=400)[0] == (401, 500)
    assert windows_for_run(resume_after_step=400)[-1] == (901, 1000)


def test_sparse_online_metrics_exist_only_on_successful_update_rows() -> None:
    keys = {key: key for key in REQUIRED_CANONICAL_KEYS}
    common = {
        "total_step_time": 2.0,
        "generation_tps": 100.0,
        "acceptance_rate": 0.7,
        "mean_accepted_length": 3.0,
        "mean_total_reward": 0.8,
        "gen_kl_error": 0.1,
        "applied_draft_version": 0,
    }
    rows = [
        {"_step": 1, **common},
        {"_step": 2, **common, "draft_loss": 1.0, "draft_grad_norm": 0.5},
    ]
    validate_window_rows(
        rows, ((1, 2),), keys,
        arm_kind="candidate", successful_update_steps={2},
    )
    with pytest.raises(ValueError, match="skipped update row"):
        validate_window_rows(
            [{**rows[0], "draft_loss": 1.0}, rows[1]],
            ((1, 2),), keys,
            arm_kind="candidate", successful_update_steps={2},
        )
    with pytest.raises(ValueError, match="successful update row"):
        validate_window_rows(
            [rows[0], {key: value for key, value in rows[1].items() if key != "draft_loss"}],
            ((1, 2),), keys,
            arm_kind="candidate", successful_update_steps={2},
        )
    with pytest.raises(ValueError, match="required metric has invalid type"):
        validate_window_rows(
            [{**rows[0], "generation_tps": None}, rows[1]],
            ((1, 2),), keys,
            arm_kind="candidate", successful_update_steps={2},
        )


def test_post_refit_observation_uses_acceptance_and_version_not_draft_loss() -> None:
    arm = build_pilot_matrix(
        analysis_experiment_spec(), replicate_index=0
    )[0].candidate
    keys = dict(arm.canonical_metric_keys)
    evidence = ArmEvidence(
        arm=arm,
        rows=[
            {"_step": 10, keys["applied_draft_version"]: 0, keys["acceptance_rate"]: 0.70},
            {"_step": 11, keys["applied_draft_version"]: 10, keys["acceptance_rate"]: 0.65},
        ],
        receipts={"schedule": {
            "refit_versions": [{"refit_step": 10, "applied_draft_version": 10}],
            "post_event_observations": [{
                "refit_step": 10,
                "observation_step": 11,
                "applied_draft_version": 10,
                "acceptance_rate": 0.65,
            }],
        }},
        artifacts={},
        resume_after_step=None,
    )
    validate_post_event_rows(evidence)
    assert keys["draft_loss"] not in evidence.rows[1]


def test_sparse_unfiltered_wandb_history_merges_before_key_filtering() -> None:
    arm = build_pilot_matrix(
        analysis_experiment_spec(), replicate_index=0
    )[0].candidate
    keys = dict(arm.canonical_metric_keys)
    raw_rows = [
        {"_step": 1, keys["total_step_time"]: 2.0, "unrelated": 99},
        {"_step": 1, keys["draft_loss"]: 1.0},
        {"_step": 1, keys["draft_grad_norm"]: 0.5},
        {"_step": 1, keys["applied_draft_version"]: 0.0},
    ]
    assert merge_and_filter_wandb_rows(raw_rows, arm) == [{
        "_step": 1,
        keys["total_step_time"]: 2.0,
        keys["draft_loss"]: 1.0,
        keys["draft_grad_norm"]: 0.5,
        keys["applied_draft_version"]: 0.0,
    }]


def test_analyzer_rejects_manifest_declared_digest_mismatch(tmp_path: Path) -> None:
    manifest = render_manifest(build_pilot_matrix(analysis_experiment_spec(), replicate_index=0))
    manifest["manifest_sha256"] = "f" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared digest"):
        load_bound_manifest(path)


def test_every_frozen_gate_is_evaluated(tmp_path: Path) -> None:
    decision = evaluate_candidate(passing_candidate_metrics(tmp_path))
    assert set(decision.gates) == {
        "overhead_reduction_mean",
        "overhead_reduction_ci_lower",
        "generation_tps_ci_lower",
        "acceptance_pp_ci_lower",
        "accepted_length_ci_lower",
        "reward_pp_ci_lower",
        "gen_kl_paired_ci_upper",
        "gen_kl_margin",
        "finite_loss_and_gradient",
        "two_consecutive_divergent_draft_loss_windows",
        "exact_schedule_receipts",
        "checkpoint_resume_sequence_equal",
    }
    assert decision.passed is True


@pytest.mark.parametrize(
    "missing",
    [
        "total_step_time",
        "generation_tps",
        "acceptance_rate",
        "mean_accepted_length",
        "mean_total_reward",
        "gen_kl_error",
        "draft_loss",
        "draft_grad_norm",
        "applied_draft_version",
    ],
)
def test_missing_canonical_key_fails_closed(missing: str, tmp_path: Path) -> None:
    metrics = passing_candidate_metrics(tmp_path)
    del metrics.canonical_keys[missing]
    with pytest.raises(ValueError, match=f"missing canonical metric key: {missing}"):
        evaluate_candidate(metrics)


def test_receipt_rejects_counter_or_identity_mismatch() -> None:
    arm, receipts = matching_arm_and_receipts()
    receipts["schedule"]["attempted_updates"] += 1
    with pytest.raises(ValueError, match="schedule counter mismatch"):
        validate_arm_receipts(arm, receipts)


def test_ledger_outcomes_must_reconcile_schedule_counters(tmp_path: Path) -> None:
    metrics = passing_candidate_metrics(tmp_path)
    evidence = metrics.matched_replicates[0][1]
    segment = evidence.receipts["schedule"]["decision_ledger_segments"][0]
    path = Path(segment["path"])
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    entries[0]["outcome"]["update_successful"] = False
    raw = b"".join(
        (json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for entry in entries
    )
    path.write_bytes(raw)
    segment["size_bytes"] = len(raw)
    segment["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="counters disagree"):
        evaluate_candidate(metrics)


def test_analyzer_rejects_nonterminal_or_incomplete_rows(tmp_path: Path) -> None:
    metrics = passing_candidate_metrics(tmp_path)
    metrics.matched_replicates[0][0].receipts["terminal"]["terminal"] = False
    with pytest.raises(ValueError, match="nonterminal"):
        evaluate_candidate(metrics)
    metrics = passing_candidate_metrics(tmp_path)
    metrics.matched_replicates[0][0].rows.pop()
    with pytest.raises(ValueError, match="exactly match frozen windows"):
        evaluate_candidate(metrics)


def test_analyzer_rejects_out_of_range_math_reward(tmp_path: Path) -> None:
    metrics = passing_candidate_metrics(tmp_path)
    metrics.matched_replicates[0][2].rows[0]["mean_total_reward"] = 1.01
    with pytest.raises(ValueError, match="bounded math reward"):
        evaluate_candidate(metrics)


def test_analyzer_rejects_wrong_replicate_cardinality_or_shared_job(tmp_path: Path) -> None:
    metrics = passing_candidate_metrics(tmp_path)
    with pytest.raises(ValueError, match="three matched replicates"):
        evaluate_candidate(replace(metrics, matched_replicates=metrics.matched_replicates[:2]))
    metrics = passing_candidate_metrics(tmp_path)
    for receipt in metrics.matched_replicates[0][1].receipts.values():
        receipt["job_id"] = "other-job"
    with pytest.raises(ValueError, match="one shared job"):
        evaluate_candidate(metrics)


def test_analyzer_rejects_nonmonotonic_or_wrong_cadence_receipt(tmp_path: Path) -> None:
    metrics = passing_candidate_metrics(tmp_path)
    schedule = metrics.matched_replicates[0][2].receipts["schedule"]
    schedule["decision_ids"][4] = 4
    with pytest.raises(ValueError, match="decision IDs are not contiguous"):
        evaluate_candidate(metrics)
    metrics = passing_candidate_metrics(tmp_path)
    schedule = metrics.matched_replicates[0][2].receipts["schedule"]
    schedule["updated_steps"][0] = 9
    with pytest.raises(ValueError, match="exact cadence"):
        evaluate_candidate(metrics)


def test_analyzer_rejects_reason_version_and_delayed_observation_receipts(tmp_path: Path) -> None:
    metrics = passing_candidate_metrics(tmp_path)
    schedule = metrics.matched_replicates[0][2].receipts["schedule"]
    schedule["decision_reasons"][0] = "always"
    with pytest.raises(ValueError, match="exact cadence"):
        evaluate_candidate(metrics)
    metrics = passing_candidate_metrics(tmp_path)
    schedule = metrics.matched_replicates[0][2].receipts["schedule"]
    schedule["refit_versions"][0]["applied_draft_version"] += 1
    with pytest.raises(ValueError, match="refit decision IDs"):
        evaluate_candidate(metrics)
    metrics = passing_candidate_metrics(tmp_path)
    schedule = metrics.matched_replicates[0][2].receipts["schedule"]
    schedule["post_event_observations"][0]["observation_step"] += 1
    with pytest.raises(ValueError, match="immediate next row|post-event"):
        evaluate_candidate(metrics)
    metrics = passing_candidate_metrics(tmp_path)
    metrics.matched_replicates[0][2].rows[0]["applied_draft_version"] = 99.0
    with pytest.raises(ValueError, match="selected serving-draft version"):
        evaluate_candidate(metrics)


def test_resume_merges_bound_ledger_prefix_and_suffix(tmp_path: Path) -> None:
    metrics = passing_candidate_metrics(tmp_path)
    schedule = metrics.resume_replicate[2].receipts["schedule"]
    assert len(schedule["decision_ledger_segments"]) == 2
    ledger = load_decision_ledger(schedule)
    assert [row["decision_id"] for row in ledger] == list(range(1, 1001))
    prefix = Path(schedule["decision_ledger_segments"][0]["path"])
    prefix.write_bytes(prefix.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="segment size or digest"):
        load_decision_ledger(schedule)


def test_paired_summary_uses_paired_differences() -> None:
    summary = paired_summary([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
    assert summary.mean == pytest.approx(2.0)


def test_wandb_loader_reads_immutable_raw_export(tmp_path: Path) -> None:
    arm = build_pilot_matrix(analysis_experiment_spec(), replicate_index=0)[0].arms[0]
    arm = replace(arm, result_dir=str(tmp_path), manifest_sha256="7" * 64)
    raw_path = tmp_path / "raw-wandb.json"
    raw_bytes = json.dumps(
        [{"_step": 1, "total_step_time": 1.0}],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    raw_path.write_bytes(raw_bytes)
    (tmp_path / "wandb.json").write_text(json.dumps({
        "manifest_sha256": arm.manifest_sha256,
        "run_path": f"{arm.wandb_entity}/{arm.wandb_project}/{arm.wandb_id}",
        "raw_rows_path": str(raw_path.resolve()),
        "raw_rows_size_bytes": len(raw_bytes),
        "raw_rows_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "frozen": True,
    }))
    assert load_wandb_rows(arm) == [{"_step": 1, "total_step_time": 1.0}]


def test_wandb_loader_rejects_mutated_frozen_export(tmp_path: Path) -> None:
    arm = build_pilot_matrix(analysis_experiment_spec(), replicate_index=0)[0].arms[0]
    arm = replace(arm, result_dir=str(tmp_path), manifest_sha256="7" * 64)
    raw_path = tmp_path / "raw-wandb.json"
    raw_path.write_bytes(b"[]")
    (tmp_path / "wandb.json").write_text(json.dumps({
        "manifest_sha256": arm.manifest_sha256,
        "run_path": f"{arm.wandb_entity}/{arm.wandb_project}/{arm.wandb_id}",
        "raw_rows_path": str(raw_path.resolve()),
        "raw_rows_size_bytes": 2,
        "raw_rows_sha256": hashlib.sha256(b"different").hexdigest(),
        "frozen": True,
    }))
    with pytest.raises(ValueError, match="frozen W&B artifact digest"):
        load_wandb_rows(arm)


def test_candidate_metrics_builder_requires_three_fresh_and_one_resume(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = build_long_matrix(
        analysis_experiment_spec(), promoted=("fixed_sparse_10",),
        replicate_indices=(0, 1, 2),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(render_manifest(matrix)))
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.analyze.load_arm_evidence",
        lambda arm: ArmEvidence(
            arm=arm,
            rows=[],
            receipts={},
            artifacts={},
            resume_after_step=arm.resume_after_step,
        ),
    )
    metrics = build_candidate_metrics(manifest, "fixed_sparse_10")
    assert len(metrics.matched_replicates) == 3
    assert metrics.resume_replicate[0].resume_after_step == 400


def test_survivor_selector_is_deterministic_and_capped_at_two() -> None:
    results = {
        "a": {"terminal": True, "elimination_passed": True, "event_count_sufficient": True, "overhead_reduction_point": 0.2},
        "b": {"terminal": True, "elimination_passed": True, "event_count_sufficient": True, "overhead_reduction_point": 0.4},
        "c": {"terminal": True, "elimination_passed": True, "event_count_sufficient": True, "overhead_reduction_point": 0.3},
    }
    assert select_pilot_survivors(results) == ["b", "c"]


def test_cp2_evidence_gates_long_survivors(monkeypatch, tmp_path: Path) -> None:
    matrix = build_cp2_survivor_matrix(
        analysis_experiment_spec(),
        survivors=("fixed_sparse_10",),
        replicate_index=0,
    )
    manifest_path = tmp_path / "cp2.json"
    manifest_path.write_text(json.dumps(render_manifest(matrix)))
    cp1_manifest = render_manifest(
        build_pilot_matrix(analysis_experiment_spec(), replicate_index=0)
    )
    cp1_manifest_path = tmp_path / "cp1.json"
    cp1_manifest_path.write_text(json.dumps(cp1_manifest))
    pilot_results = {
        "fixed_sparse_10": {
            "terminal": True,
            "elimination_passed": True,
            "event_count_sufficient": True,
            "overhead_reduction_point": 0.2,
        }
    }
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.analyze.load_and_validate_pilot_results",
        lambda _path: pilot_results,
    )
    cp1_receipt_path = tmp_path / "cp1-receipt.json"
    cp1_receipt_path.write_text(json.dumps(analyze_manifest(
        "pilot", cp1_manifest_path, topology="cp1"
    )))
    receipt = analyze_manifest(
        "pilot",
        manifest_path,
        topology="cp2",
        cp1_receipt_path=cp1_receipt_path,
    )
    assert receipt["cp2_survivors"] == ["fixed_sparse_10"]
    assert receipt["cp1_candidates"] == pilot_results


@pytest.mark.parametrize("stage", ["pilot", "long"])
def test_analyzer_cli_writes_json_csv_and_markdown(
    monkeypatch, tmp_path: Path, stage: str
) -> None:
    receipt = {
        "status": "pilot_complete" if stage == "pilot" else "no_survivor",
        "candidates": {},
        "cp2_survivors": [],
        "manifest_sha256": "7" * 64,
        "manifest_path": str((tmp_path / "manifest.json").resolve()),
        "topology": "packed-cp1",
    }
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.analyze.analyze_manifest",
        lambda *_args, **_kwargs: receipt,
    )
    argv = [
        stage, "--manifest", str(tmp_path / "manifest.json"), "--fail-closed",
        "--json", str(tmp_path / "out.json"),
        "--csv", str(tmp_path / "out.csv"),
        "--markdown", str(tmp_path / "out.md"),
    ]
    if stage == "pilot":
        argv.extend([
            "--topology", "cp1",
            "--select-survivors", str(tmp_path / "survivors.json"),
        ])
    assert analyzer_main(argv) == 0
    assert (tmp_path / "out.json").exists()
    assert (tmp_path / "out.csv").exists()
    assert (tmp_path / "out.md").exists()
```

- [ ] **Step 2: Run the RED analysis tests and confirm the module is absent.**

Run: `uv run --group test pytest -q research/qwen3_8b_draft_cadence/tests/test_analysis.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'research.qwen3_8b_draft_cadence.analyze'`.

- [ ] **Step 3: Add exact paired formulas and frozen gates.**

```python
import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isclose, isfinite, sqrt
from statistics import fmean, stdev
from pathlib import Path
from typing import cast

from scipy.stats import t

from research.qwen3_8b_draft_cadence.manifest import Arm, recompute_manifest_sha256
from research.qwen3_8b_draft_cadence.launch import _arm_from_dict


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class GenKLDecision:
    paired_differences: list[float]
    confidence: ConfidenceInterval
    margin: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArmEvidence:
    arm: Arm
    rows: list[dict[str, object]]
    receipts: Mapping[str, object]
    artifacts: Mapping[str, ArtifactBinding]
    resume_after_step: int | None


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    canonical_keys: dict[str, str]
    matched_replicates: tuple[
        tuple[ArmEvidence, ArmEvidence, ArmEvidence], ...
    ]
    resume_replicate: tuple[ArmEvidence, ArmEvidence, ArmEvidence]


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    gates: dict[str, float | bool]
    passed: bool
    raw_artifact_sha256_by_arm: dict[str, str]
    evidence_artifacts_by_arm: dict[str, dict[str, ArtifactBinding]]


FRESH_WINDOWS = tuple((start, start + 99) for start in range(1, 1000, 100))
RESUMED_400_WINDOWS = tuple((start, start + 99) for start in range(401, 1000, 100))
REQUIRED_CANONICAL_KEYS = frozenset(
    {
        "total_step_time",
        "generation_tps",
        "acceptance_rate",
        "mean_accepted_length",
        "mean_total_reward",
        "gen_kl_error",
        "draft_loss",
        "draft_grad_norm",
        "applied_draft_version",
    }
)
FIXED_NOT_APPLICABLE_KEYS = frozenset(
    {"draft_loss", "draft_grad_norm", "applied_draft_version"}
)
COMMON_REQUIRED_KEYS = REQUIRED_CANONICAL_KEYS - FIXED_NOT_APPLICABLE_KEYS
ONLINE_EVERY_ROW_KEYS = COMMON_REQUIRED_KEYS | frozenset(
    {"applied_draft_version"}
)


def load_bound_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON mapping")
    recomputed = recompute_manifest_sha256(payload)
    if payload.get("manifest_sha256") != recomputed:
        raise ValueError("manifest declared digest does not match recomputed bytes")
    return payload


def windows_for_run(resume_after_step: int | None) -> tuple[tuple[int, int], ...]:
    if resume_after_step is None:
        return FRESH_WINDOWS
    if resume_after_step == 400:
        return RESUMED_400_WINDOWS
    raise ValueError("only fresh and resume_after_step=400 windows are predeclared")


def paired_ci(differences: list[float]) -> ConfidenceInterval:
    if len(differences) < 2 or not all(isfinite(value) for value in differences):
        raise ValueError("paired confidence interval requires at least two finite replicates")
    mean = fmean(differences)
    half = t.ppf(0.975, len(differences) - 1) * stdev(differences) / sqrt(len(differences))
    return ConfidenceInterval(mean=mean, lower=mean - half, upper=mean + half)


def paired_summary(
    candidate: list[float], control: list[float]
) -> ConfidenceInterval:
    if len(candidate) != len(control):
        raise ValueError("paired arms require equal replicate counts")
    return paired_ci([
        candidate_value - control_value
        for candidate_value, control_value in zip(candidate, control, strict=True)
    ])


def overhead_reduction(*, fixed: float, always: float, candidate: float) -> float:
    always_overhead = always - fixed
    if always_overhead <= 0.0:
        raise ValueError("always overhead must be positive")
    candidate_overhead = candidate - fixed
    return (always_overhead - candidate_overhead) / always_overhead


def evaluate_gen_kl(candidate: list[float], always: list[float]) -> GenKLDecision:
    if len(candidate) != len(always):
        raise ValueError("gen_kl_error arms require matched replicate counts")
    if not all(isfinite(value) for value in candidate + always):
        raise ValueError("gen_kl_error must be finite")
    differences = [candidate_value - always_value for candidate_value, always_value in zip(candidate, always, strict=True)]
    confidence = paired_ci(differences)
    margin = max(0.10 * fmean(always), 0.01)
    return GenKLDecision(
        paired_differences=differences,
        confidence=confidence,
        margin=margin,
        passed=confidence.upper <= margin,
    )


def paired_relative_lower(candidate: list[float], control: list[float]) -> float:
    if any(value <= 0.0 for value in control):
        raise ValueError("relative comparison control values must be positive")
    return paired_ci(
        [
            candidate_value / control_value - 1.0
            for candidate_value, control_value in zip(candidate, control, strict=True)
        ]
    ).lower


def validate_window_rows(
    rows: list[dict[str, object]],
    windows: tuple[tuple[int, int], ...],
    canonical_keys: dict[str, str],
    *,
    arm_kind: str,
    successful_update_steps: set[int],
) -> None:
    missing = REQUIRED_CANONICAL_KEYS - canonical_keys.keys()
    if missing:
        raise ValueError(f"missing canonical metric key: {sorted(missing)[0]}")
    steps = []
    for row in rows:
        raw_step = row.get("_step")
        if (
            isinstance(raw_step, bool)
            or not isinstance(raw_step, (int, float))
            or not isfinite(float(raw_step))
            or not float(raw_step).is_integer()
            or int(raw_step) <= 0
        ):
            raise ValueError("canonical W&B step must be a positive integer")
        steps.append(int(raw_step))
    if len(steps) != len(set(steps)):
        raise ValueError("duplicate canonical W&B step")
    required_steps = {step for start, end in windows for step in range(start, end + 1)}
    if set(steps) != required_steps:
        raise ValueError("canonical W&B rows do not exactly match frozen windows")
    if arm_kind == "fixed" and successful_update_steps:
        raise ValueError("fixed control cannot report successful draft updates")
    required_for_arm = (
        COMMON_REQUIRED_KEYS if arm_kind == "fixed" else ONLINE_EVERY_ROW_KEYS
    )
    for row in rows:
        step = int(row["_step"])
        for logical_key in FIXED_NOT_APPLICABLE_KEYS:
            if arm_kind == "fixed" and canonical_keys[logical_key] in row:
                raise ValueError("fixed control must not fabricate online-only metrics")
        for logical_key in required_for_arm:
            raw_value = row[canonical_keys[logical_key]]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"required metric has invalid type: {logical_key}")
            value = float(raw_value)
            if not isfinite(value):
                raise ValueError(f"nonfinite required metric: {logical_key}")
            if logical_key in {"total_step_time", "generation_tps"} and value <= 0.0:
                raise ValueError(f"required metric must be positive: {logical_key}")
            if logical_key == "acceptance_rate" and not 0.0 <= value <= 1.0:
                raise ValueError("acceptance_rate must be within [0,1]")
            if logical_key == "mean_total_reward" and not 0.0 <= value <= 1.0:
                raise ValueError("bounded math reward must be within [0,1]")
            if logical_key in {
                "mean_accepted_length", "gen_kl_error", "draft_loss", "draft_grad_norm"
            } and value < 0.0:
                raise ValueError(f"required metric must be nonnegative: {logical_key}")
            if logical_key == "applied_draft_version" and (
                not value.is_integer() or value < 0.0 or value > int(row["_step"])
            ):
                raise ValueError("applied draft version must be an integral prior decision")
        if arm_kind != "fixed":
            for logical_key in ("draft_loss", "draft_grad_norm"):
                physical_key = canonical_keys[logical_key]
                if step in successful_update_steps:
                    raw_value = row.get(physical_key)
                    if (
                        isinstance(raw_value, bool)
                        or not isinstance(raw_value, (int, float))
                        or not isfinite(float(raw_value))
                        or float(raw_value) < 0.0
                    ):
                        raise ValueError(
                            f"successful update row lacks finite {logical_key}"
                        )
                elif physical_key in row and row[physical_key] is not None:
                    raise ValueError(
                        f"skipped update row must use NA/absence for {logical_key}"
                    )


def validate_post_event_rows(evidence: ArmEvidence) -> None:
    if evidence.arm.kind == "fixed":
        return
    schedule = evidence.receipts["schedule"]
    assert isinstance(schedule, Mapping)
    version_key = dict(evidence.arm.canonical_metric_keys)[
        "applied_draft_version"
    ]
    acceptance_key = dict(evidence.arm.canonical_metric_keys)["acceptance_rate"]
    rows = {int(row["_step"]): row for row in evidence.rows}
    observations = list(schedule["post_event_observations"])
    refit_versions = schedule.get("refit_versions", [])
    if not isinstance(refit_versions, list):
        raise ValueError("refit versions must be a list")
    for step, row in rows.items():
        expected = max(
            (
                int(item["applied_draft_version"])
                for item in refit_versions
                if int(item["refit_step"]) < step
            ),
            default=0,
        )
        if int(row[version_key]) != expected:
            raise ValueError(
                "online row does not log the selected serving-draft version"
            )
    if len({int(item["refit_step"]) for item in observations}) != len(observations):
        raise ValueError("duplicate post-event refit mapping")
    for observation in observations:
        observation_step = int(observation["observation_step"])
        if observation_step != int(observation["refit_step"]) + 1:
            raise ValueError("post-event observation must use the immediate next row")
        row = rows.get(observation_step)
        if row is None:
            raise ValueError("post-event observation row is absent")
        if int(row[version_key]) != int(observation["applied_draft_version"]):
            raise ValueError("post-event row applied draft version mismatch")
        observed_acceptance = observation.get("acceptance_rate")
        if (
            isinstance(observed_acceptance, bool)
            or not isinstance(observed_acceptance, (int, float))
            or not isfinite(float(observed_acceptance))
            or not 0.0 <= float(observed_acceptance) <= 1.0
            or not isclose(
                float(row[acceptance_key]),
                float(observed_acceptance),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("post-event acceptance observation mismatch")


def successful_update_steps(evidence: ArmEvidence) -> set[int]:
    if evidence.arm.kind == "fixed":
        return set()
    schedule = evidence.receipts.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("online arm is missing a schedule receipt")
    steps = schedule.get("updated_steps")
    if not isinstance(steps, list) or any(type(step) is not int for step in steps):
        raise ValueError("updated_steps must be an integer list")
    return set(steps)


def load_decision_ledger(
    schedule: Mapping[str, object],
) -> list[Mapping[str, object]]:
    segments = schedule.get("decision_ledger_segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("online schedule requires durable decision-ledger segments")
    entries: list[Mapping[str, object]] = []
    expected_next = 1
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("decision-ledger segment receipt must be a mapping")
        path = Path(str(segment.get("path"))).resolve()
        raw = path.read_bytes()
        if (
            type(segment.get("size_bytes")) is not int
            or segment["size_bytes"] != len(raw)
            or segment.get("sha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("decision-ledger segment size or digest mismatch")
        parsed = [json.loads(line) for line in raw.splitlines()]
        if not all(isinstance(entry, Mapping) for entry in parsed):
            raise ValueError("decision-ledger row must be a mapping")
        ids = [entry.get("decision_id") for entry in parsed]
        if (
            any(type(value) is not int for value in ids)
            or ids != list(range(expected_next, expected_next + len(ids)))
            or segment.get("first_decision_id") != expected_next
            or segment.get("last_decision_id") != expected_next + len(ids) - 1
            or segment.get("entry_count") != len(ids)
        ):
            raise ValueError("decision-ledger prefix/suffix is not contiguous")
        entries.extend(parsed)
        expected_next += len(ids)
    return entries


def validate_ledger_outcomes(
    entries: list[Mapping[str, object]],
    schedule: Mapping[str, object],
) -> None:
    keys = {
        "update_attempted", "update_successful", "update_skipped",
        "draft_refit_attempted", "draft_refit_successful",
        "draft_refit_skipped", "forced_update", "forced_refit",
    }
    outcomes = []
    applied_version = 0
    for entry in entries:
        outcome = entry.get("outcome")
        if (
            not isinstance(outcome, Mapping)
            or set(outcome) != keys
            or any(type(outcome[key]) is not bool for key in keys)
        ):
            raise ValueError("decision-ledger outcome schema mismatch")
        if (
            type(entry.get("applied_draft_version")) is not int
            or entry.get("applied_draft_version") != applied_version
            or
            outcome["update_attempted"] != entry.get("update_requested")
            or outcome["update_skipped"] == outcome["update_attempted"]
            or outcome["update_successful"] and not outcome["update_attempted"]
            or outcome["draft_refit_attempted"]
            and not entry.get("draft_refit_requested")
            or outcome["draft_refit_skipped"]
            == outcome["draft_refit_attempted"]
            or outcome["draft_refit_successful"]
            and not outcome["draft_refit_attempted"]
            or outcome["forced_update"]
            != (entry.get("forced") is True and outcome["update_successful"])
            or outcome["forced_refit"]
            != (
                entry.get("forced") is True
                and outcome["draft_refit_successful"]
            )
        ):
            raise ValueError("decision-ledger outcome disagrees with decision")
        outcomes.append(outcome)
        if outcome["draft_refit_successful"]:
            applied_version = int(entry["decision_id"])
    derived = {
        "attempted_updates": sum(item["update_attempted"] for item in outcomes),
        "successful_updates": sum(item["update_successful"] for item in outcomes),
        "skipped_updates": sum(item["update_skipped"] for item in outcomes),
        "attempted_refits": sum(item["draft_refit_attempted"] for item in outcomes),
        "successful_refits": sum(item["draft_refit_successful"] for item in outcomes),
        "skipped_refits": sum(item["draft_refit_skipped"] for item in outcomes),
        "forced_updates": sum(item["forced_update"] for item in outcomes),
        "forced_refits": sum(item["forced_refit"] for item in outcomes),
    }
    derived["failed_updates"] = (
        derived["attempted_updates"] - derived["successful_updates"]
    )
    derived["failed_refits"] = (
        derived["attempted_refits"] - derived["successful_refits"]
    )
    if any(schedule.get(key) != value for key, value in derived.items()):
        raise ValueError("schedule counters disagree with decision-ledger outcomes")
    forced_update_steps = [
        int(entry["global_step"])
        for entry, outcome in zip(entries, outcomes, strict=True)
        if outcome["forced_update"]
    ]
    forced_refit_steps = [
        int(entry["global_step"])
        for entry, outcome in zip(entries, outcomes, strict=True)
        if outcome["forced_refit"]
    ]
    if (
        list(schedule.get("forced_update_steps", [])) != forced_update_steps
        or list(schedule.get("forced_refit_steps", [])) != forced_refit_steps
    ):
        raise ValueError("forced-step lists disagree with ledger outcomes")


def validate_arm_receipts(arm: Arm, receipts: Mapping[str, object]) -> None:
    required_identity = (
        "experiment_id",
        "replicate_id",
        "arm_id",
        "execution_order",
        "manifest_sha256",
        "job_id",
        "product_head",
        "harness_head",
        "submodule_shas",
        "container_digest",
        "model_snapshot_sha256",
        "draft_snapshot_sha256",
        "invariant_config_sha256",
        "resolved_config_sha256",
        "wandb_id",
        "wandb_entity",
        "wandb_project",
    )
    provenance = receipts.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("missing receipt: provenance")
    for key in required_identity:
        if key not in provenance:
            raise ValueError(f"provenance missing identity key: {key}")
    for receipt_name in ("provenance", "wandb", "checkpoint", "schedule", "terminal"):
        receipt = receipts.get(receipt_name)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"missing receipt: {receipt_name}")
        for key, expected in (
            ("experiment_id", arm.experiment_id),
            ("replicate_id", arm.replicate_id),
            ("arm_id", arm.name),
            ("execution_order", arm.execution_order),
            ("manifest_sha256", arm.manifest_sha256),
            ("product_head", arm.product_head),
            ("harness_head", arm.harness_head),
            ("submodule_shas", arm.submodule_shas),
            ("container_digest", arm.container_digest),
            ("model_snapshot_sha256", arm.model_snapshot_sha256),
            ("draft_snapshot_sha256", arm.draft_snapshot_sha256),
            ("invariant_config_sha256", arm.invariant_config_sha256),
            ("resolved_config_sha256", arm.resolved_config_sha256),
            ("wandb_id", arm.wandb_id),
            ("wandb_entity", arm.wandb_entity),
            ("wandb_project", arm.wandb_project),
        ):
            actual = receipt.get(key)
            if key == "submodule_shas" and isinstance(actual, list):
                actual = tuple(tuple(item) for item in actual)
            if actual != expected:
                raise ValueError(f"{receipt_name} {key} mismatch")
        for key in ("manifest_sha256", "job_id"):
            if receipt.get(key) != provenance.get(key):
                raise ValueError(f"{receipt_name} {key} mismatch")
    terminal = receipts["terminal"]
    wandb_receipt = receipts["wandb"]
    checkpoint = receipts["checkpoint"]
    schedule = receipts["schedule"]
    assert isinstance(terminal, Mapping)
    assert isinstance(wandb_receipt, Mapping)
    assert isinstance(checkpoint, Mapping)
    assert isinstance(schedule, Mapping)
    if (
        wandb_receipt.get("frozen") is not True
        or not isinstance(wandb_receipt.get("raw_rows_sha256"), str)
        or len(wandb_receipt["raw_rows_sha256"]) != 64
        or type(wandb_receipt.get("raw_rows_size_bytes")) is not int
        or wandb_receipt["raw_rows_size_bytes"] < 0
    ):
        raise ValueError("W&B receipt does not bind a frozen raw artifact")
    if type(terminal.get("completed_policy_steps")) is not int:
        raise ValueError("completed policy steps must be an integer")
    completed = terminal["completed_policy_steps"]
    if terminal.get("terminal") is not True or terminal.get("exit_code") != 0:
        raise ValueError("nonterminal or failed arm")
    if type(checkpoint.get("current_step")) is not int or (
        completed != arm.max_steps or checkpoint["current_step"] != completed
    ):
        raise ValueError("completion/checkpoint step mismatch")
    if checkpoint.get("resumed_from") != arm.resume_checkpoint:
        raise ValueError("resume checkpoint provenance mismatch")

    if arm.kind == "fixed":
        zero_fields = (
            "attempted_updates", "successful_updates", "failed_updates",
            "skipped_updates", "attempted_refits", "successful_refits",
            "failed_refits", "skipped_refits", "forced_updates",
            "forced_refits",
        )
        empty_fields = (
            "decision_ids", "global_steps", "updated_steps", "refit_steps",
            "forced_update_steps", "forced_refit_steps", "update_receipts",
            "post_event_observations", "pending_post_event_steps",
            "refit_versions", "decision_reasons", "decision_ledger_segments",
        )
        if (
            schedule.get("mode") != "disabled"
            or any(type(schedule.get(key)) is not int or schedule[key] != 0 for key in zero_fields)
            or schedule.get("policy_refit_count") != completed
            or any(schedule.get(key) != [] for key in empty_fields)
            or schedule.get("not_applicable_metrics")
            != sorted(FIXED_NOT_APPLICABLE_KEYS)
        ):
            raise ValueError("fixed control neutral schedule receipt mismatch")
        return

    counter_names = (
        "attempted_updates", "successful_updates", "failed_updates", "skipped_updates",
        "attempted_refits", "successful_refits", "failed_refits", "skipped_refits",
        "forced_updates", "forced_refits", "policy_refit_count",
    )
    if any(
        type(schedule.get(key)) is not int or schedule[key] < 0
        for key in counter_names
    ):
        raise ValueError("schedule counters must be nonnegative integers")
    attempted_updates = schedule["attempted_updates"]
    successful_updates = schedule["successful_updates"]
    attempted_refits = schedule["attempted_refits"]
    successful_refits = schedule["successful_refits"]
    if attempted_updates != successful_updates + schedule["failed_updates"]:
        raise ValueError("schedule counter mismatch: update attempts")
    if attempted_refits != successful_refits + schedule["failed_refits"]:
        raise ValueError("schedule counter mismatch: refit attempts")
    if schedule["failed_updates"] or schedule["failed_refits"]:
        raise ValueError("terminal successful arm contains failed schedule events")
    if successful_updates + schedule["skipped_updates"] != completed:
        raise ValueError("schedule counter mismatch: update partition")
    if successful_refits + schedule["skipped_refits"] != completed:
        raise ValueError("schedule counter mismatch: refit partition")
    if schedule["policy_refit_count"] != completed:
        raise ValueError("policy refit count must equal completed steps")

    ledger = load_decision_ledger(schedule)
    validate_ledger_outcomes(ledger, schedule)
    decisions = list(schedule["decision_ids"])
    steps = list(schedule["global_steps"])
    ledger_decisions = [entry["decision_id"] for entry in ledger]
    ledger_steps = [entry["global_step"] for entry in ledger]
    ledger_updates = [
        entry["global_step"] for entry in ledger if entry["update_requested"]
    ]
    ledger_refits = [
        entry["global_step"] for entry in ledger
        if entry["draft_refit_requested"]
    ]
    ledger_reasons = [entry["reason"] for entry in ledger]
    if (
        decisions != ledger_decisions
        or steps != ledger_steps
        or list(schedule["updated_steps"]) != ledger_updates
        or list(schedule["refit_steps"]) != ledger_refits
        or list(schedule["decision_reasons"]) != ledger_reasons
    ):
        raise ValueError("schedule summary disagrees with durable decision ledger")
    step_lists = (
        decisions, steps, list(schedule["updated_steps"]),
        list(schedule["refit_steps"]), list(schedule["forced_update_steps"]),
        list(schedule["forced_refit_steps"]),
        list(schedule["pending_post_event_steps"]),
    )
    if any(type(value) is not int for values in step_lists for value in values):
        raise ValueError("schedule steps and versions must be integers")
    if len(decisions) != completed or decisions != list(range(1, completed + 1)):
        raise ValueError("decision IDs are not contiguous")
    if len(steps) != completed or steps != list(range(1, completed + 1)):
        raise ValueError("global steps are not contiguous")
    updated_steps = list(schedule["updated_steps"])
    refit_steps = list(schedule["refit_steps"])
    if len(updated_steps) != successful_updates or len(refit_steps) != successful_refits:
        raise ValueError("schedule event cardinality mismatch")
    if updated_steps != sorted(set(updated_steps)) or refit_steps != sorted(set(refit_steps)):
        raise ValueError("schedule event steps must be strictly monotonic")
    if (
        len(schedule["decision_reasons"]) != completed
        or any(type(reason) is not str for reason in schedule["decision_reasons"])
    ):
        raise ValueError("decision reason cardinality mismatch")
    forced_steps = list(schedule["forced_update_steps"])
    if schedule["forced_updates"] != len(forced_steps):
        raise ValueError("forced-update counter mismatch")
    if not set(forced_steps).issubset(updated_steps):
        raise ValueError("forced updates must be successful updates")
    forced_refit_steps = list(schedule["forced_refit_steps"])
    if schedule["forced_refits"] != len(forced_refit_steps):
        raise ValueError("forced-refit counter mismatch")
    if not set(forced_refit_steps).issubset(refit_steps):
        raise ValueError("forced refits must be successful refits")
    if len(schedule["update_receipts"]) != successful_updates:
        raise ValueError("scheduled update receipt cardinality mismatch")
    observations = list(schedule["post_event_observations"])
    pending = list(schedule["pending_post_event_steps"])
    if any(
        not isinstance(item, Mapping)
        or type(item.get("refit_step")) is not int
        or type(item.get("observation_step")) is not int
        or type(item.get("applied_draft_version")) is not int
        for item in observations
    ):
        raise ValueError("post-event steps and versions must be integers")
    observable_refits = [
        step for step in refit_steps
        if step < completed
        and (arm.resume_after_step is None or step >= arm.resume_after_step)
    ]
    if pending != [step for step in refit_steps if step == completed]:
        raise ValueError("pending post-event receipt mismatch")
    if (
        [int(item["refit_step"]) for item in observations] != observable_refits
        or not all(
        type(item.get("acceptance_rate")) in {int, float}
        and isfinite(float(item["acceptance_rate"]))
        and 0.0 <= float(item["acceptance_rate"]) <= 1.0
        and int(item["observation_step"]) == int(item["refit_step"]) + 1
        for item in observations
        )
    ):
        raise ValueError("insufficient post-event observations")
    refit_versions = list(schedule["refit_versions"])
    if any(
        not isinstance(item, Mapping)
        or type(item.get("refit_step")) is not int
        or type(item.get("applied_draft_version")) is not int
        for item in refit_versions
    ):
        raise ValueError("refit steps and versions must be integers")
    version_steps = [int(item["refit_step"]) for item in refit_versions]
    version_ids = [int(item["applied_draft_version"]) for item in refit_versions]
    expected_refit_pairs = [
        (int(entry["global_step"]), int(entry["decision_id"]))
        for entry in ledger
        if entry["draft_refit_requested"]
    ]
    if list(zip(version_steps, version_ids, strict=True)) != expected_refit_pairs:
        raise ValueError("refit versions must strictly increase as refit decision IDs")
    if version_ids != sorted(set(version_ids)):
        raise ValueError("refit-version receipt cardinality mismatch")
    versions = dict(zip(version_steps, version_ids, strict=True))
    for item in observations:
        if int(item["applied_draft_version"]) != versions[int(item["refit_step"] )]:
            raise ValueError("post-event applied draft version mismatch")

    all_steps = list(range(1, completed + 1))
    interval_steps = (
        list(range(arm.schedule.interval, completed + 1, arm.schedule.interval))
        if arm.schedule is not None and arm.schedule.interval > 0 else []
    )
    reasons = list(schedule["decision_reasons"])
    if arm.kind == "fixed":
        expected_updates, expected_refits = [], []
        expected_reasons = ["none"] * completed
    elif arm.schedule is None:
        raise ValueError("enabled draft arm is missing its schedule")
    elif arm.schedule.mode == "always":
        expected_updates, expected_refits = all_steps, all_steps
        expected_reasons = ["always"] * completed
    elif arm.schedule.mode == "fixed" and arm.schedule.action == "sparse_update":
        expected_updates, expected_refits = interval_steps, interval_steps
        expected_reasons = [
            "fixed_interval" if step in interval_steps else "none"
            for step in all_steps
        ]
    elif arm.schedule.mode == "fixed" and arm.schedule.action == "refit_only":
        expected_updates, expected_refits = all_steps, interval_steps
        expected_reasons = ["fixed_interval"] * completed
    else:
        allowed_reasons = {
            "adaptive_degradation", "adaptive_burst", "max_interval", "none",
        }
        if not set(reasons).issubset(allowed_reasons):
            raise ValueError("invalid adaptive decision reason")
        for step, reason in zip(all_steps, reasons, strict=True):
            if (reason == "none") == (step in updated_steps):
                raise ValueError("adaptive reason does not match update decision")
            if (reason == "max_interval") != (step in forced_steps):
                raise ValueError("adaptive forced update reason mismatch")
        gaps = [updated_steps[0], *(
            right - left for left, right in zip(updated_steps, updated_steps[1:])
        ), completed - updated_steps[-1]] if updated_steps else [completed]
        if max(gaps) > arm.schedule.max_interval:
            raise ValueError("adaptive max_interval cadence violated")
        if refit_steps != updated_steps:
            raise ValueError("adaptive refit steps must equal update steps")
        expected_updates, expected_refits = updated_steps, refit_steps
        expected_reasons = reasons
    if (
        updated_steps != expected_updates
        or refit_steps != expected_refits
        or reasons != expected_reasons
    ):
        raise ValueError("schedule events do not match exact cadence")
```

`controller_runtime_fixture.py` owns the executable fixture rather than hiding a
mocked receipt builder in the analyzer test. Its public factory accepts only the
arguments shown above and returns a `ControllerRuntimeArm` with
`run_to_terminal() -> tuple[Arm, dict[str, object]]`. Construction resolves a
three-step test arm with `cadence_runtime.enabled=true`, fixed interval 2 or
adaptive max interval 2, then instantiates the production scheduler, ledger,
`CadenceRuntimeWriter`, and invokes either the production `grpo_train_sync`
three-step loop or a real `SingleControllerActor._train_pump` bounded to three
steps. Deterministic worker/generation adapters
replace only model compute and transport: their selected batches carry the
canonical accepted/draft counts and applied-version tag, their successful
training result carries the real model/optimizer digest receipt, and their
draft-apply callback writes immutable snapshot bytes and returns the production
apply-receipt schema. The fixture must not import or call
`build_terminal_schedule_payload`, `validate_product_runtime_receipts`, or
`validate_arm_receipts` while executing steps. Each step enters the selected
real controller; the controller exclusively closes update evidence before
transfer/publication, installs each returned checkpoint ledger suffix, and calls
its real `terminal_closed` hook. Only after the run is closed does
`run_to_terminal` load the emitted files through
`validate_product_runtime_receipts`, bind the arm's immutable identity and frozen
W&B artifact, and return them to the unmodified experiment validator. A
receipt-order callback fails unless each exclusive update file and checkpointed
evidence binding already exist when transfer begins, and the returned schedule
must be the file emitted by the real `terminal_closed` hook. The eight
controller/mode/action cases plus the resumed case therefore exercise
controller→runtime writer→experiment validator and cannot pass with fabricated
schedule mappings. This cross-contract test lives here because experiments
start only after product Task 10 is GREEN.

`CandidateMetrics` carries raw canonical W&B rows and all receipts for every arm, not caller-supplied summaries. `evaluate_candidate` requires exactly three matched triplets, one fixed/`always`/candidate arm per triplet, a single shared SLURM job ID within each triplet, distinct job IDs across replicates, identical parity fields, and 1000 completed steps. It runs `validate_window_rows` and `validate_arm_receipts` before deriving any value, joins each triplet on the exact full common-step intersection without imputation, and reduces each arm to one arithmetic mean per replicate. It derives finite-loss, draft-loss windows, counters, cadence, and all gates from that validated evidence:

```python
def _mean(evidence: ArmEvidence, logical_key: str) -> float:
    physical_key = evidence.arm.canonical_metric_keys
    mapping = dict(physical_key)
    return fmean(float(row[mapping[logical_key]]) for row in evidence.rows)


def _window_mean(
    evidence: ArmEvidence, logical_key: str, start: int, end: int
) -> float:
    physical_key = dict(evidence.arm.canonical_metric_keys)[logical_key]
    values = [
        float(row[physical_key])
        for row in evidence.rows
        if start <= int(row["_step"]) <= end and physical_key in row
    ]
    sparse_online_key = logical_key in {"draft_loss", "draft_grad_norm"}
    if (not sparse_online_key and len(values) != end - start + 1) or not values:
        raise ValueError("absent full common-step intersection")
    return fmean(values)


NUMERIC_GATE_NAMES = frozenset({
    "overhead_reduction_mean", "overhead_reduction_ci_lower",
    "generation_tps_ci_lower", "acceptance_pp_ci_lower",
    "accepted_length_ci_lower", "reward_pp_ci_lower",
    "gen_kl_paired_ci_upper", "gen_kl_margin",
})
BOOLEAN_GATE_NAMES = frozenset({
    "finite_loss_and_gradient",
    "two_consecutive_divergent_draft_loss_windows",
    "exact_schedule_receipts",
    "checkpoint_resume_sequence_equal",
})


def validate_gate_schema(gates: Mapping[str, object]) -> None:
    if set(gates) != NUMERIC_GATE_NAMES | BOOLEAN_GATE_NAMES:
        raise ValueError("gate mapping does not match frozen schema")
    for key in NUMERIC_GATE_NAMES:
        value = gates[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ):
            raise ValueError(f"numeric gate has invalid value: {key}")
    for key in BOOLEAN_GATE_NAMES:
        if type(gates[key]) is not bool:
            raise ValueError(f"boolean gate has invalid value: {key}")


def gates_pass(gates: Mapping[str, object]) -> bool:
    validate_gate_schema(gates)
    numeric = {key: cast(float, gates[key]) for key in NUMERIC_GATE_NAMES}
    flags = {key: cast(bool, gates[key]) for key in BOOLEAN_GATE_NAMES}
    return (
        numeric["overhead_reduction_mean"] >= 0.50
        and numeric["overhead_reduction_ci_lower"] > 0.0
        and numeric["generation_tps_ci_lower"] > -0.02
        and numeric["acceptance_pp_ci_lower"] > -1.0
        and numeric["accepted_length_ci_lower"] > -0.1
        and numeric["reward_pp_ci_lower"] > -1.0
        and numeric["gen_kl_paired_ci_upper"] <= numeric["gen_kl_margin"]
        and flags["finite_loss_and_gradient"]
        and not flags["two_consecutive_divergent_draft_loss_windows"]
        and flags["exact_schedule_receipts"]
        and flags["checkpoint_resume_sequence_equal"]
    )


def evaluate_candidate(metrics: CandidateMetrics) -> CandidateDecision:
    missing = REQUIRED_CANONICAL_KEYS - metrics.canonical_keys.keys()
    if missing:
        raise ValueError(f"missing canonical metric key: {sorted(missing)[0]}")
    if len(metrics.matched_replicates) != 3:
        raise ValueError("long validation requires three matched replicates")
    by_replicate: list[dict[str, ArmEvidence]] = []
    job_ids: list[str] = []
    for triplet in metrics.matched_replicates:
        if len(triplet) != 3 or {item.arm.kind for item in triplet} != {
            "fixed", "always", "candidate"
        }:
            raise ValueError("each replicate requires fixed/always/candidate cardinality")
        by_kind = {item.arm.kind: item for item in triplet}
        parity_fields = (
            "max_steps", "seed", "product_head", "harness_head",
            "submodule_shas", "container_digest", "invariant_config_sha256",
            "model_snapshot_path", "draft_snapshot_path", "data_order_path",
            "model_revision", "draft_revision", "dataset_revision",
            "data_order_sha256", "k", "global_batch_size", "micro_batch_size",
            "sequence_packing", "sequence_parallel", "tensor_parallel_size",
            "context_parallel_size", "cuda_graph_settings", "topology",
        )
        for field in parity_fields:
            if len({getattr(item.arm, field) for item in triplet}) != 1:
                raise ValueError(f"matched triplet parity mismatch: {field}")
        if triplet[0].arm.max_steps != 1000:
            raise ValueError("long validation arms must complete 1000 steps")
        triplet_jobs: set[str] = set()
        required_steps: set[int] | None = None
        for item in triplet:
            if dict(item.arm.canonical_metric_keys) != metrics.canonical_keys:
                raise ValueError("canonical metric mapping differs from manifest")
            validate_arm_receipts(item.arm, item.receipts)
            windows = windows_for_run(item.resume_after_step)
            validate_window_rows(
                item.rows,
                windows,
                metrics.canonical_keys,
                arm_kind=item.arm.kind,
                successful_update_steps=successful_update_steps(item),
            )
            validate_post_event_rows(item)
            item_steps = {int(row["_step"]) for row in item.rows}
            required_steps = item_steps if required_steps is None else required_steps & item_steps
            terminal = item.receipts["terminal"]
            checkpoint = item.receipts["checkpoint"]
            assert isinstance(terminal, Mapping) and isinstance(checkpoint, Mapping)
            if int(checkpoint["current_step"]) != int(terminal["completed_policy_steps"]):
                raise ValueError("logger/checkpoint step disagreement")
            triplet_jobs.add(str(terminal["job_id"]))
        expected_steps = {
            step
            for start, end in windows_for_run(triplet[0].resume_after_step)
            for step in range(start, end + 1)
        }
        if required_steps != expected_steps:
            raise ValueError("absent full common-step intersection")
        if len(triplet_jobs) != 1:
            raise ValueError("matched triplet must use one shared job")
        job_ids.append(next(iter(triplet_jobs)))
        by_replicate.append(by_kind)
    if len(set(job_ids)) != 3:
        raise ValueError("matched replicates require distinct allocation jobs")

    resume_triplet = metrics.resume_replicate
    if len(resume_triplet) != 3 or {item.arm.kind for item in resume_triplet} != {
        "fixed", "always", "candidate"
    }:
        raise ValueError("resume requires one fixed/always/candidate triplet")
    resume_jobs: set[str] = set()
    resume_by_kind = {item.arm.kind: item for item in resume_triplet}
    source_by_kind = by_replicate[0]
    for item in resume_triplet:
        if item.resume_after_step != 400 or item.arm.resume_after_step != 400:
            raise ValueError("resume evidence must be predeclared after Step 400")
        if item.arm.resume_checkpoint is None:
            raise ValueError("resume evidence is missing checkpoint provenance")
        source = source_by_kind[item.arm.kind].arm
        if (
            item.arm.resume_source_replicate_id != source.replicate_id
            or item.arm.resume_checkpoint
            != f"{source.result_dir}/checkpoints/step_400"
        ):
            raise ValueError("resume arm is not bound to uninterrupted r0 source")
        for field in (
            "product_head", "harness_head", "submodule_shas", "container_digest",
            "model_snapshot_path", "draft_snapshot_path", "data_order_path",
            "model_revision", "draft_revision", "dataset_revision",
            "data_order_sha256", "seed", "topology", "invariant_config_sha256",
        ):
            if getattr(item.arm, field) != getattr(source, field):
                raise ValueError(f"resume/source identity mismatch: {field}")
        if dict(item.arm.canonical_metric_keys) != metrics.canonical_keys:
            raise ValueError("resume canonical metric mapping differs from manifest")
        validate_arm_receipts(item.arm, item.receipts)
        validate_window_rows(
            item.rows,
            windows_for_run(item.resume_after_step),
            metrics.canonical_keys,
            arm_kind=item.arm.kind,
            successful_update_steps=successful_update_steps(item),
        )
        validate_post_event_rows(item)
        terminal = item.receipts["terminal"]
        assert isinstance(terminal, Mapping)
        resume_jobs.add(str(terminal["job_id"]))
    if len(resume_jobs) != 1 or not resume_jobs.isdisjoint(job_ids):
        raise ValueError("resume triplet requires one distinct shared job")

    summaries = {
        kind: {
            key: [_mean(replicate[kind], key) for replicate in by_replicate]
            for key in COMMON_REQUIRED_KEYS
        }
        for kind in ("fixed", "always", "candidate")
    }
    fixed_step_times = summaries["fixed"]["total_step_time"]
    always_step_times = summaries["always"]["total_step_time"]
    candidate_step_times = summaries["candidate"]["total_step_time"]
    overhead_reductions = [
        overhead_reduction(fixed=fixed, always=always, candidate=candidate)
        for fixed, always, candidate in zip(
            fixed_step_times, always_step_times, candidate_step_times,
            strict=True,
        )
    ]
    overhead_ci = paired_ci(overhead_reductions)
    tps_lower = paired_relative_lower(
        summaries["candidate"]["generation_tps"],
        summaries["always"]["generation_tps"],
    )
    acceptance_lower = paired_ci(
        [(candidate - always) * 100.0 for candidate, always in zip(
            summaries["candidate"]["acceptance_rate"],
            summaries["always"]["acceptance_rate"], strict=True
        )]
    ).lower
    accepted_length_lower = paired_ci(
        [candidate - always for candidate, always in zip(
            summaries["candidate"]["mean_accepted_length"],
            summaries["always"]["mean_accepted_length"],
            strict=True,
        )]
    ).lower
    reward_lower = paired_ci(
        [(candidate - always) * 100.0 for candidate, always in zip(
            summaries["candidate"]["mean_total_reward"],
            summaries["always"]["mean_total_reward"], strict=True
        )]
    ).lower
    gen_kl = evaluate_gen_kl(
        summaries["candidate"]["gen_kl_error"],
        summaries["always"]["gen_kl_error"],
    )
    draft_loss_windows = [
        (
            [
                _window_mean(replicate["candidate"], "draft_loss", start, end)
                for replicate in by_replicate
            ],
            [
                _window_mean(replicate["always"], "draft_loss", start, end)
                for replicate in by_replicate
            ],
        )
        for start, end in FRESH_WINDOWS
    ]
    divergent = [
        (
            fmean(candidate_window) > 1.20 * fmean(always_window)
            and paired_ci([
                candidate - always
                for candidate, always in zip(
                    candidate_window, always_window, strict=True
                )
            ]).lower > 0.0
        )
        for candidate_window, always_window in draft_loss_windows
    ]
    two_consecutive_divergent = any(
        left and right for left, right in zip(divergent, divergent[1:])
    )
    uninterrupted_schedule = by_replicate[0]["candidate"].receipts["schedule"]
    resumed_schedule = resume_by_kind["candidate"].receipts["schedule"]
    assert isinstance(uninterrupted_schedule, Mapping)
    assert isinstance(resumed_schedule, Mapping)
    uninterrupted_ledger = load_decision_ledger(uninterrupted_schedule)
    resumed_ledger = load_decision_ledger(resumed_schedule)
    resume_equal = uninterrupted_ledger == resumed_ledger
    finite_loss_and_gradient = all(
        isfinite(float(row[physical_key]))
        for replicate in by_replicate
        for kind_name in ("always", "candidate")
        for physical_key in (
            dict(replicate[kind_name].arm.canonical_metric_keys)["draft_loss"],
            dict(replicate[kind_name].arm.canonical_metric_keys)["draft_grad_norm"],
        )
        for row in replicate[kind_name].rows
        if int(row["_step"]) in successful_update_steps(replicate[kind_name])
    )
    gates: dict[str, float | bool] = {
        "overhead_reduction_mean": fmean(overhead_reductions),
        "overhead_reduction_ci_lower": overhead_ci.lower,
        "generation_tps_ci_lower": tps_lower,
        "acceptance_pp_ci_lower": acceptance_lower,
        "accepted_length_ci_lower": accepted_length_lower,
        "reward_pp_ci_lower": reward_lower,
        "gen_kl_paired_ci_upper": gen_kl.confidence.upper,
        "gen_kl_margin": gen_kl.margin,
        "finite_loss_and_gradient": finite_loss_and_gradient,
        "two_consecutive_divergent_draft_loss_windows": two_consecutive_divergent,
        "exact_schedule_receipts": True,
        "checkpoint_resume_sequence_equal": resume_equal,
    }
    passed = gates_pass(gates)
    evidence_artifacts = {
        item.arm.name: dict(item.artifacts)
        for triplet in (*metrics.matched_replicates, metrics.resume_replicate)
        for item in triplet
    }
    if len(evidence_artifacts) != 12:
        raise ValueError("candidate does not bind all twelve arm evidence sets")
    raw_artifacts = {}
    for arm_id, artifacts in evidence_artifacts.items():
        raw = artifacts.get("raw_wandb")
        if raw is None or len(raw.sha256) != 64:
            raise ValueError("candidate does not bind every frozen raw W&B artifact")
        raw_artifacts[arm_id] = raw.sha256
    return CandidateDecision(
        gates=gates,
        passed=passed,
        raw_artifact_sha256_by_arm=raw_artifacts,
        evidence_artifacts_by_arm=evidence_artifacts,
    )


FROZEN_GATE_NAMES = NUMERIC_GATE_NAMES | BOOLEAN_GATE_NAMES


def validate_long_receipt(receipt: Mapping[str, object]) -> None:
    status = receipt.get("status")
    if status not in {"selected", "no_candidate", "no_survivor"}:
        raise ValueError("invalid long receipt status")
    manifest_path_value = receipt.get("manifest_path")
    if not isinstance(manifest_path_value, str):
        raise ValueError("long receipt is missing its manifest path")
    manifest_path = Path(manifest_path_value).resolve()
    manifest = load_bound_manifest(manifest_path)
    manifest_sha256 = recompute_manifest_sha256(manifest)
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise ValueError("long receipt manifest path/digest mismatch")
    candidates = receipt.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("long receipt candidates must be a mapping")
    if status == "no_survivor":
        if candidates or receipt.get("terminal") is not True:
            raise ValueError("no_survivor must be terminal with no candidates")
        if (
            receipt.get("selected_candidate") is not None
            or receipt.get("production_supported") is not False
            or receipt.get("recommendation") != "always"
            or receipt.get("reason") != "no_pilot_survivors"
            or manifest.get("replicates") != []
        ):
            raise ValueError("invalid no_survivor recommendation receipt")
        return
    if not candidates:
        raise ValueError("long receipt requires evaluated candidates")
    validated_results: dict[str, Mapping[str, object]] = {}
    for name, result in candidates.items():
        if not isinstance(name, str) or not isinstance(result, Mapping):
            raise ValueError("malformed candidate receipt")
        gates = result.get("gates")
        if not isinstance(gates, Mapping) or set(gates) != FROZEN_GATE_NAMES:
            raise ValueError("candidate receipt does not contain every frozen gate")
        if type(result.get("completed_policy_steps")) is not int or result["completed_policy_steps"] != 1000:
            raise ValueError("candidate receipt is incomplete")
        if type(result.get("policy_refit_count")) is not int or result["policy_refit_count"] != 1000:
            raise ValueError("candidate policy refit receipt is incomplete")
        if type(result.get("passed")) is not bool:
            raise ValueError("candidate passed field must be boolean")
        if result["passed"] is not gates_pass(gates):
            raise ValueError("stored candidate result disagrees with frozen gates")
        raw_hashes = result.get("raw_artifact_sha256_by_arm")
        if (
            not isinstance(raw_hashes, Mapping)
            or len(raw_hashes) != 12
            or any(
                not isinstance(arm_id, str)
                or not isinstance(digest, str)
                or len(digest) != 64
                for arm_id, digest in raw_hashes.items()
            )
        ):
            raise ValueError("candidate receipt does not bind twelve raw artifacts")
        stored_evidence = result.get("evidence_artifacts_by_arm")
        if not isinstance(stored_evidence, Mapping) or len(stored_evidence) != 12:
            raise ValueError("candidate receipt does not bind twelve evidence sets")
        recomputed = evaluate_candidate(
            build_candidate_metrics(manifest_path, name)
        )
        expected_evidence = {
            arm_id: {
                artifact_name: asdict(binding)
                for artifact_name, binding in artifacts.items()
            }
            for arm_id, artifacts in recomputed.evidence_artifacts_by_arm.items()
        }
        if (
            dict(gates) != recomputed.gates
            or result["passed"] is not recomputed.passed
            or dict(raw_hashes) != recomputed.raw_artifact_sha256_by_arm
            or dict(stored_evidence) != expected_evidence
        ):
            raise ValueError(
                "stored candidate result disagrees with recomputed manifest evidence"
            )
        validated_results[name] = result
    passing = [
        name for name, result in validated_results.items()
        if result["passed"] is True
    ]
    deterministic_winner = (
        sorted(
            passing,
            key=lambda name: (
                -cast(float, validated_results[name]["gates"]["overhead_reduction_mean"]),
                name,
            ),
        )[0]
        if passing else None
    )
    selected = receipt.get("selected_candidate")
    if status == "selected":
        if (
            not isinstance(selected, str)
            or selected not in validated_results
            or validated_results[selected].get("passed") is not True
        ):
            raise ValueError("selected candidate must exist and pass every gate")
        if receipt.get("production_supported") is not True:
            raise ValueError("selected candidate must be production-supported")
        if selected != deterministic_winner or receipt.get("recommendation") != selected:
            raise ValueError("selected recommendation is not the deterministic winner")
    else:
        if selected is not None or receipt.get("production_supported") is not False:
            raise ValueError("no_candidate receipt cannot claim production support")
        if receipt.get("recommendation") != "always":
            raise ValueError("no_candidate receipt must recommend always")
        if deterministic_winner is not None:
            raise ValueError("no_candidate receipt contains a passing candidate")


def merge_and_filter_wandb_rows(
    raw_rows: list[object], arm: Arm
) -> list[dict[str, object]]:
    allowed = set(dict(arm.canonical_metric_keys).values())
    merged: dict[int, dict[str, object]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or "_step" not in raw:
            continue
        step = raw["_step"]
        if type(step) is not int:
            raise ValueError("W&B _step must be an integer")
        row = merged.setdefault(step, {"_step": step})
        for key in allowed.intersection(raw):
            value = raw[key]
            if key in row and row[key] != value:
                raise ValueError("conflicting W&B values for the same step/key")
            row[key] = value
    return [merged[step] for step in sorted(merged)]


def load_wandb_rows(arm: Arm) -> list[dict[str, object]]:
    wandb_receipt = json.loads((Path(arm.result_dir) / "wandb.json").read_text())
    raw_path = Path(wandb_receipt["raw_rows_path"])
    expected_path = (Path(arm.result_dir) / "raw-wandb.json").resolve()
    expected_run = f"{arm.wandb_entity}/{arm.wandb_project}/{arm.wandb_id}"
    if (
        raw_path.resolve() != expected_path
        or wandb_receipt.get("frozen") is not True
        or wandb_receipt.get("run_path") != expected_run
    ):
        raise ValueError("W&B rows are not the declared frozen artifact")
    raw_bytes = raw_path.read_bytes()
    if (
        type(wandb_receipt.get("raw_rows_size_bytes")) is not int
        or wandb_receipt["raw_rows_size_bytes"] != len(raw_bytes)
        or wandb_receipt.get("raw_rows_sha256")
        != hashlib.sha256(raw_bytes).hexdigest()
    ):
        raise ValueError("frozen W&B artifact digest or size mismatch")
    if wandb_receipt.get("manifest_sha256") != arm.manifest_sha256:
        raise ValueError("frozen W&B artifact manifest mismatch")
    raw_rows = json.loads(raw_bytes)
    if not isinstance(raw_rows, list):
        raise ValueError("W&B rows must be a JSON list of mappings")
    return merge_and_filter_wandb_rows(raw_rows, arm)


def bind_artifact(
    path: Path,
    *,
    allowed_roots: tuple[Path, ...],
) -> ArtifactBinding:
    resolved = path.resolve()
    if (
        not any(resolved.is_relative_to(root.resolve()) for root in allowed_roots)
        or not resolved.is_file()
    ):
        raise ValueError("evidence artifact escapes its bound arm/source roots")
    raw = resolved.read_bytes()
    return ArtifactBinding(
        path=str(resolved),
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def bind_checkpoint_tree(
    path: Path,
    *,
    allowed_roots: tuple[Path, ...],
) -> ArtifactBinding:
    resolved = path.resolve()
    if (
        not any(resolved.is_relative_to(root.resolve()) for root in allowed_roots)
        or not resolved.is_dir()
    ):
        raise ValueError("checkpoint tree escapes its bound arm root")
    digest = hashlib.sha256()
    size_bytes = 0
    for item in sorted(child for child in resolved.rglob("*") if child.is_file()):
        if item.name == "cadence-checkpoint-receipt.json":
            continue
        raw = item.read_bytes()
        relative = item.relative_to(resolved).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(raw).digest())
        size_bytes += len(raw)
    return ArtifactBinding(
        path=str(resolved), size_bytes=size_bytes, sha256=digest.hexdigest()
    )


def collect_arm_artifacts(
    arm: Arm,
    receipts: Mapping[str, object],
) -> dict[str, ArtifactBinding]:
    result_dir = Path(arm.result_dir).resolve()
    allowed_roots = (result_dir,)
    if arm.resume_checkpoint is not None:
        allowed_roots += (Path(arm.resume_checkpoint).resolve().parents[1],)
    artifacts = {
        f"receipt:{name}": bind_artifact(
            result_dir / f"{name}.json", allowed_roots=allowed_roots
        )
        for name in ("provenance", "wandb", "checkpoint", "schedule", "terminal")
    }
    wandb_receipt = receipts["wandb"]
    schedule = receipts["schedule"]
    checkpoint = receipts["checkpoint"]
    assert isinstance(wandb_receipt, Mapping)
    assert isinstance(schedule, Mapping)
    assert isinstance(checkpoint, Mapping)
    final_tree = bind_checkpoint_tree(
        Path(str(checkpoint.get("checkpoint_path"))),
        allowed_roots=(result_dir,),
    )
    if checkpoint.get("checkpoint_tree_sha256") != final_tree.sha256:
        raise ValueError("final checkpoint tree digest mismatch")
    artifacts["checkpoint_tree"] = final_tree
    artifacts["raw_wandb"] = bind_artifact(
        Path(str(wandb_receipt["raw_rows_path"])), allowed_roots=allowed_roots
    )
    for index, segment in enumerate(schedule.get("decision_ledger_segments", [])):
        binding = bind_artifact(
            Path(str(segment["path"])), allowed_roots=allowed_roots
        )
        if (
            segment.get("size_bytes") != binding.size_bytes
            or segment.get("sha256") != binding.sha256
        ):
            raise ValueError("decision-ledger evidence binding mismatch")
        artifacts[f"decision_ledger:{index}"] = binding
    snapshot = checkpoint.get("applied_draft_snapshot")
    if snapshot is not None:
        binding = bind_artifact(
            Path(str(snapshot["path"])), allowed_roots=allowed_roots
        )
        if (
            snapshot.get("size_bytes") != binding.size_bytes
            or snapshot.get("sha256") != binding.sha256
        ):
            raise ValueError("applied-draft snapshot evidence binding mismatch")
        artifacts["applied_draft_snapshot"] = binding
    for index, required in enumerate(
        checkpoint.get("required_checkpoint_receipts", [])
    ):
        binding = bind_artifact(
            Path(str(required["path"])), allowed_roots=allowed_roots
        )
        if (
            required.get("size_bytes") != binding.size_bytes
            or required.get("sha256") != binding.sha256
        ):
            raise ValueError("required checkpoint receipt evidence binding mismatch")
        required_payload = json.loads(Path(binding.path).read_text())
        required_tree = bind_checkpoint_tree(
            Path(str(required_payload.get("checkpoint_path"))),
            allowed_roots=allowed_roots,
        )
        if required_payload.get("checkpoint_tree_sha256") != required_tree.sha256:
            raise ValueError("required checkpoint tree digest mismatch")
        artifacts[f"required_checkpoint:{index}"] = binding
        artifacts[f"required_checkpoint_tree:{index}"] = required_tree
    return artifacts


def load_arm_evidence(arm: Arm) -> ArmEvidence:
    result_dir = Path(arm.result_dir)
    receipt_files = {
        "provenance": "provenance.json",
        "wandb": "wandb.json",
        "checkpoint": "checkpoint.json",
        "schedule": "schedule.json",
        "terminal": "terminal.json",
    }
    receipts = {
        name: json.loads((result_dir / filename).read_text())
        for name, filename in receipt_files.items()
    }
    return ArmEvidence(
        arm=arm,
        rows=load_wandb_rows(arm),
        receipts=receipts,
        artifacts=collect_arm_artifacts(arm, receipts),
        resume_after_step=arm.resume_after_step,
    )


def build_candidate_metrics(
    manifest_path: Path, candidate_name: str
) -> CandidateMetrics:
    manifest = load_bound_manifest(manifest_path)
    fresh = []
    resumed = []
    for raw_replicate in manifest["replicates"]:
        arms = tuple(_arm_from_dict(raw) for raw in raw_replicate["arms"])
        candidate = next(arm for arm in arms if arm.kind == "candidate")
        base_name = candidate.name.split("-candidate-r", 1)[0]
        if base_name != candidate_name:
            continue
        evidence = tuple(
            load_arm_evidence(next(arm for arm in arms if arm.kind == kind))
            for kind in ("fixed", "always", "candidate")
        )
        (resumed if candidate.resume_after_step == 400 else fresh).append(evidence)
    if len(fresh) != 3 or len(resumed) != 1:
        raise ValueError("candidate requires three fresh triplets and one resume triplet")
    canonical_keys = dict(fresh[0][0].arm.canonical_metric_keys)
    return CandidateMetrics(
        canonical_keys=canonical_keys,
        matched_replicates=tuple(fresh),
        resume_replicate=resumed[0],
    )


def select_pilot_survivors(results: Mapping[str, Mapping[str, object]]) -> list[str]:
    eligible = [
        (name, float(result["overhead_reduction_point"]))
        for name, result in results.items()
        if result.get("terminal") is True
        and result.get("elimination_passed") is True
        and result.get("event_count_sufficient") is True
    ]
    return [name for name, _score in sorted(eligible, key=lambda item: (-item[1], item[0]))[:2]]


def load_and_validate_pilot_results(
    manifest_path: Path,
) -> dict[str, dict[str, object]]:
    manifest = load_bound_manifest(manifest_path)
    results = {}
    for raw_replicate in manifest["replicates"]:
        arms = tuple(_arm_from_dict(raw) for raw in raw_replicate["arms"])
        evidence = {
            arm.kind: load_arm_evidence(arm) for arm in arms
        }
        for item in evidence.values():
            validate_arm_receipts(item.arm, item.receipts)
            windows = tuple(
                (start, min(start + 99, item.arm.max_steps))
                for start in range(1, item.arm.max_steps + 1, 100)
            )
            validate_window_rows(
                item.rows,
                windows,
                dict(item.arm.canonical_metric_keys),
                arm_kind=item.arm.kind,
                successful_update_steps=successful_update_steps(item),
            )
            validate_post_event_rows(item)
        candidate = evidence["candidate"].arm
        name = candidate.name.split("-candidate-r", 1)[0]
        reduction = overhead_reduction(
            fixed=_mean(evidence["fixed"], "total_step_time"),
            always=_mean(evidence["always"], "total_step_time"),
            candidate=_mean(evidence["candidate"], "total_step_time"),
        )
        results[name] = {
            "terminal": True,
            "completed_policy_steps": candidate.max_steps,
            "policy_refit_count": candidate.max_steps,
            "event_count_sufficient": True,
            "overhead_reduction_point": reduction,
            "elimination_passed": reduction > 0.0,
        }
    return results


def analyze_manifest(
    stage: str,
    manifest_path: Path,
    *,
    topology: str | None = None,
    cp1_receipt_path: Path | None = None,
) -> dict[str, object]:
    manifest = load_bound_manifest(manifest_path)
    if stage == "long" and manifest.get("replicates") == []:
        terminal_path = Path(str(manifest.get("no_survivor_receipt_path")))
        terminal = json.loads(terminal_path.read_text())
        validate_long_receipt(terminal)
        return terminal
    candidate_names = sorted({
        raw["candidate"]["name"].split("-candidate-r", 1)[0]
        for raw in manifest["replicates"]
    })
    if stage == "pilot":
        expected_topology = {"cp1": "packed-cp1", "cp2": "packed-cp2"}.get(topology)
        if expected_topology is None:
            raise ValueError("pilot analysis requires explicit cp1 or cp2 topology")
        if any(
            arm["topology"] != expected_topology
            for replicate in manifest["replicates"]
            for arm in replicate["arms"]
        ):
            raise ValueError("pilot manifest topology does not match analysis stage")
        results = load_and_validate_pilot_results(manifest_path)
        selected = select_pilot_survivors(results)
        receipt = {
            "status": "pilot_complete",
            "production_claim": False,
            "topology": expected_topology,
            "manifest_sha256": recompute_manifest_sha256(manifest),
            "manifest_path": str(manifest_path.resolve()),
            "candidates": results,
            "cp2_survivors": selected,
        }
        if topology == "cp2":
            if cp1_receipt_path is None:
                raise ValueError("CP2 analysis requires the terminal CP1 receipt")
            cp1 = json.loads(cp1_receipt_path.read_text())
            cp1_survivors = cp1.get("cp2_survivors")
            cp1_manifest_path = Path(cp1["manifest_path"])
            cp1_manifest = json.loads(cp1_manifest_path.read_text())
            cp1_manifest_sha256 = recompute_manifest_sha256(cp1_manifest)
            recomputed_cp1 = analyze_manifest(
                "pilot", cp1_manifest_path, topology="cp1"
            )
            if (
                cp1 != recomputed_cp1
                or
                cp1.get("status") != "pilot_complete"
                or cp1.get("topology") != "packed-cp1"
                or cp1.get("manifest_sha256") != cp1_manifest_sha256
                or cp1_manifest.get("manifest_sha256") != cp1_manifest_sha256
                or not isinstance(cp1_survivors, list)
                or set(candidate_names) != set(cp1_survivors)
            ):
                raise ValueError("CP2 manifest is not bound to CP1-qualified candidates")
            receipt["cp1_manifest_sha256"] = cp1.get("manifest_sha256")
            receipt["cp1_receipt_path"] = str(cp1_receipt_path.resolve())
            receipt["cp1_candidates"] = cp1["candidates"]
            receipt["cp2_survivors"] = [
                name for name in selected if name in cp1_survivors
            ]
        return receipt
    candidates = {}
    for name in candidate_names:
        decision = evaluate_candidate(build_candidate_metrics(manifest_path, name))
        candidates[name] = {
            "passed": decision.passed,
            "gates": decision.gates,
            "raw_artifact_sha256_by_arm": decision.raw_artifact_sha256_by_arm,
            "evidence_artifacts_by_arm": {
                arm_id: {
                    name: asdict(binding) for name, binding in artifacts.items()
                }
                for arm_id, artifacts in decision.evidence_artifacts_by_arm.items()
            },
            "completed_policy_steps": 1000,
            "policy_refit_count": 1000,
        }
    passing = sorted(
        (name for name, result in candidates.items() if result["passed"]),
        key=lambda name: (
            -candidates[name]["gates"]["overhead_reduction_mean"],
            name,
        ),
    )
    receipt = {
        "manifest_sha256": recompute_manifest_sha256(manifest),
        "manifest_path": str(manifest_path.resolve()),
        "status": "selected" if passing else "no_candidate",
        "selected_candidate": passing[0] if passing else None,
        "production_supported": bool(passing),
        "recommendation": passing[0] if passing else "always",
        "candidates": candidates,
    }
    validate_long_receipt(receipt)
    return receipt


def write_analysis_outputs(
    receipt: Mapping[str, object], *, json_path: Path, csv_path: Path,
    markdown_path: Path,
) -> None:
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    candidates = receipt.get("candidates", {})
    assert isinstance(candidates, Mapping)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate", "passed", *sorted(FROZEN_GATE_NAMES)])
        for name, result in sorted(candidates.items()):
            gates = result.get("gates", {})
            writer.writerow([name, result.get("passed"), *(
                gates.get(key) for key in sorted(FROZEN_GATE_NAMES)
            )])
    lines = [f"# Cadence {receipt['status']}", "", "| Candidate | Passed |", "|---|---:|"]
    lines.extend(
        f"| {name} | {result.get('passed', False)} |"
        for name, result in sorted(candidates.items())
    )
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def analyzer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("pilot", "long"):
        command = subparsers.add_parser(stage)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--fail-closed", action="store_true", required=True)
        command.add_argument("--markdown", type=Path)
        command.add_argument("--json", type=Path)
        command.add_argument("--csv", type=Path)
        if stage == "pilot":
            command.add_argument("--topology", choices=("cp1", "cp2"), required=True)
            command.add_argument("--cp1-receipt", type=Path)
            command.add_argument("--select-survivors", type=Path)
    return parser


def analyzer_main(argv: Sequence[str] | None = None) -> int:
    args = analyzer_parser().parse_args(argv)
    receipt = analyze_manifest(
        args.stage,
        args.manifest,
        topology=getattr(args, "topology", None),
        cp1_receipt_path=getattr(args, "cp1_receipt", None),
    )
    outputs = (args.json, args.csv, args.markdown)
    if any(path is not None for path in outputs):
        if not all(path is not None for path in outputs):
            raise ValueError("JSON, CSV, and Markdown outputs must be requested together")
        write_analysis_outputs(
            receipt, json_path=args.json, csv_path=args.csv,
            markdown_path=args.markdown,
        )
    if args.stage == "pilot" and args.select_survivors is not None:
        if args.json is None:
            raise ValueError("survivor selection requires a frozen JSON receipt")
        analysis_raw = args.json.read_bytes()
        with args.select_survivors.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "cp2_survivors": receipt["cp2_survivors"],
                    "source_manifest_sha256": receipt["manifest_sha256"],
                    "source_manifest_path": receipt["manifest_path"],
                    "source_topology": receipt["topology"],
                    "source_analysis_receipt_path": str(args.json.resolve()),
                    "source_analysis_receipt_size_bytes": len(analysis_raw),
                    "source_analysis_receipt_sha256": hashlib.sha256(
                        analysis_raw
                    ).hexdigest(),
                },
                stream,
                sort_keys=True,
            )
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(analyzer_main())
```

The executable `analyze.py` parser implements `pilot` and `long`. Every entrypoint first recomputes the manifest digest and requires equality with its declared `manifest_sha256`. The runner calls unfiltered `scan_history()` and freezes every returned sparse W&B row once into an exclusive raw artifact with byte size, SHA256, run identity, and manifest digest; it never requests all optional canonical keys in one W&B query. The analyzer has no live-W&B fallback: it reads only that verified raw artifact, merges sparse rows by integral `_step` with conflict rejection, and only then filters to the arm's canonical keys. Online schedule analysis loads immutable decision-ledger segments, merges a resume source prefix with its new suffix, reconciles every attempted/successful/skipped/forced outcome with the terminal counters, and requires exact contiguous IDs 1..1000; the scheduler's bounded 64-entry history is never treated as full-run evidence. Fixed controls use the explicit neutral/NA contract and retain real generation/acceptance metrics without fabricated draft loss, draft gradient, or schedule version.

It builds `CandidateMetrics` from three fresh triplets plus one resume triplet, evaluates paired summaries, and exclusively writes JSON, CSV, and Markdown. CP1 elimination selects at most two CP2 candidates; a separate CP2 analysis verifies topology and combines the terminal CP1 and CP2 receipts before writing the only survivor file accepted by long submission. The returned gate mapping uses an exact typed schema and reports all intermediate means, confidence bounds, margins, and booleans even on scientific rejection. `validate_long_receipt` reloads its manifest, rebuilds each candidate from the manifest-declared arm result directories, rehashes every receipt, frozen W&B file, decision-ledger segment, applied-draft snapshot, and required checkpoint receipt, and requires the stored gates/pass/evidence maps to equal the recomputed decision before selecting a deterministic winner. Fail before evaluation on a missing key, duplicate or nonintegral step, out-of-range/nonfinite required value (including math reward outside `[0,1]`), artifact mismatch, logger/checkpoint step disagreement, absent full common-step intersection, wrong long-run cardinality, incomplete immediate post-event mappings, counter/reason/version/cadence mismatch, nonterminal job, or identity mismatch. A structurally complete analysis exits zero even when no candidate passes and writes `status="no_candidate"`; zero pilot survivors write a bound empty manifest plus terminal `status="no_survivor"` report and submit no long jobs; malformed or incomplete evidence exits nonzero.

- [ ] **Step 4: Run the GREEN analysis and static tests.**

Run: `uv run --group test pytest -q research/qwen3_8b_draft_cadence/tests/test_analysis.py && uv run ruff check research/qwen3_8b_draft_cadence/analyze.py research/qwen3_8b_draft_cadence/tests/test_analysis.py research/qwen3_8b_draft_cadence/tests/controller_runtime_fixture.py && uv run pyrefly check research/qwen3_8b_draft_cadence/analyze.py research/qwen3_8b_draft_cadence/tests/controller_runtime_fixture.py`

Expected: tests PASS; the paired `gen_kl_error` fixture fails against margin `0.011`; static checks report no errors.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add research/qwen3_8b_draft_cadence/analyze.py research/qwen3_8b_draft_cadence/tests/test_analysis.py research/qwen3_8b_draft_cadence/tests/controller_runtime_fixture.py research/qwen3_8b_draft_cadence/README.md
git commit -S -s -m "perf(draft): add paired cadence analysis"
git verify-commit HEAD
```

Expected: signature verification exits 0.

### Task 3: Submit and close the 300/600-step elimination pilot

**Files:**
- Modify: `research/qwen3_8b_draft_cadence/tests/test_contract.py`
- Modify: `research/qwen3_8b_draft_cadence/tests/test_analysis.py`
- Create: `research/qwen3_8b_draft_cadence/PILOT.md`
- Create: `research/qwen3_8b_draft_cadence/pilot_receipt.json`
- Create: `research/qwen3_8b_draft_cadence/pilot_summary.csv`

**Interfaces:**
- Consumes: Tasks 1-2 harness, exact cadence product head, CP1 pilot matrix, terminal job artifacts, W&B exports, and checkpoint receipts.
- Produces: a fail-closed pilot report naming eliminated candidates and at most two CP2 survivors; no production recommendation.

- [ ] **Step 1: Write RED pilot-receipt completeness tests.**

```python
import json
from pathlib import Path

import pytest

from research.qwen3_8b_draft_cadence.analyze import validate_long_receipt


PILOT_RECEIPT = Path("research/qwen3_8b_draft_cadence/pilot_receipt.json")


def test_pilot_receipt_covers_every_predeclared_candidate() -> None:
    pilot_receipt = json.loads(PILOT_RECEIPT.read_text())
    required = {
        "fixed_sparse_10", "fixed_sparse_40", "fixed_refit_10",
        "fixed_refit_40", "adaptive_10_100", "fixed_sparse_100",
        "fixed_refit_100",
    }
    assert set(pilot_receipt["cp1_candidates"]) == required
    assert pilot_receipt["production_claim"] is False
    assert len(pilot_receipt["cp2_survivors"]) <= 2
    for candidate in required:
        arm = pilot_receipt["cp1_candidates"][candidate]
        assert arm["terminal"] is True
        assert arm["policy_refit_count"] == arm["completed_policy_steps"]
        assert arm["event_count_sufficient"] is True
```

- [ ] **Step 2: Run the RED receipt test before submission and confirm results are absent.**

Run: `uv run --group test pytest -q research/qwen3_8b_draft_cadence/tests/test_contract.py -k 'pilot_receipt'`

Expected: FAIL because `PILOT.md` and its machine-readable pilot receipt do not exist before the jobs run.

- [ ] **Step 3: Validate and submit each matched CP1 allocation.**

Run `/fairshare oci-hsg`; persist its timestamp, account scores, selected account, and reason in the experiment manifest before rendering jobs. Then run:

```bash
git pull --ff-only
git verify-commit HEAD
git push
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{u}')"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git submodule status --recursive
uv run python research/qwen3_8b_draft_cadence/launch.py pilot --spec "$SPEC" --manifest "$RESULT_ROOT/pilot-cp1.json" --test-only
uv run python research/qwen3_8b_draft_cadence/launch.py pilot --spec "$SPEC" --manifest "$RESULT_ROOT/pilot-cp1.json" --submit-next
mapfile -t submitted_job_ids < <(
  find "$RESULT_ROOT" -name submission.json -type f -exec \
    jq -er 'select(.state == "submitted") | .job_id' {} \; | sort -u
)
[[ "${#submitted_job_ids[@]}" -gt 0 ]]
job_ids_csv="$(IFS=,; echo "${submitted_job_ids[*]}")"
monitor_started_at="$(date +%s)"
for check in 0 1 2 3 4 5; do
  queue_rows="$(squeue --jobs "$job_ids_csv" --noheader --format='%A|%T|%j|%R')"
  accounting_rows="$(sacct -j "$job_ids_csv" --noheader --parsable2 --format=JobIDRaw,State,JobName)"
  if grep -Eq '\|(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)(\||$)' \
    <<<"$accounting_rows"; then
    echo "submitted cadence job entered terminal failure" >&2
    exit 1
  fi
  for job_id in "${submitted_job_ids[@]}"; do
    if ! grep -Eq "^${job_id}(\\.|\\|)" <<<"$queue_rows"$'\n'"$accounting_rows"; then
      echo "job $job_id is absent from both squeue and sacct" >&2
      exit 1
    fi
  done
  if find "$RESULT_ROOT" -name 'slurm-*.out' -type f -print0 | \
    xargs -0 -r grep -Eiq 'Traceback|ModuleNotFoundError|CUDA error|NCCL error|RuntimeError:'; then
    echo "cadence startup failure signature found in logs" >&2
    exit 1
  fi
  if [[ "$check" -lt 5 ]]; then sleep 60; fi
done
(( $(date +%s) - monitor_started_at >= 300 )) || {
  echo "monitoring interval was shorter than five minutes" >&2
  exit 1
}
```

Expected: the worktree is recursively clean and equal to the pushed head; immutable container/config receipts match across each triplet; fresh highest-eligible FairShare evidence is immutable; every rendered triplet job passes `sbatch --test-only`; the submitted job is visible in `squeue` or `sacct`; six fail-closed status/log checks span at least 300 seconds without a failure state, disappearance, or startup-error signature. Repeat `--submit-next` only after the prior matched triplet has a recorded allocation/job ID, preserving manifest triplet order. Each job then runs its rotated fixed/`always`/candidate arms sequentially on that one node.

- [ ] **Step 4: Eliminate on terminal CP1 evidence and submit at most two survivors at CP2.**

Run: `uv run python research/qwen3_8b_draft_cadence/analyze.py pilot --manifest "$RESULT_ROOT/pilot-cp1.json" --topology cp1 --fail-closed --json "$RESULT_ROOT/pilot-cp1-receipt.json" --csv "$RESULT_ROOT/pilot-cp1.csv" --markdown "$RESULT_ROOT/pilot-cp1.md" --select-survivors "$RESULT_ROOT/cp1-candidates-for-cp2.json" && uv run python research/qwen3_8b_draft_cadence/launch.py cp2-survivors --spec "$SPEC" --survivors "$RESULT_ROOT/cp1-candidates-for-cp2.json" --manifest "$RESULT_ROOT/pilot-cp2.json" --test-only && uv run python research/qwen3_8b_draft_cadence/launch.py cp2-survivors --spec "$SPEC" --survivors "$RESULT_ROOT/cp1-candidates-for-cp2.json" --manifest "$RESULT_ROOT/pilot-cp2.json" --submit-next`

After all CP2 triplets are terminal, gate long promotion on the combined CP1+CP2 evidence:

Run: `uv run python research/qwen3_8b_draft_cadence/analyze.py pilot --manifest "$RESULT_ROOT/pilot-cp2.json" --topology cp2 --cp1-receipt "$RESULT_ROOT/pilot-cp1-receipt.json" --fail-closed --json "$RESULT_ROOT/pilot-cp2-receipt.json" --csv "$RESULT_ROOT/pilot-cp2.csv" --markdown "$RESULT_ROOT/pilot-cp2.md" --select-survivors "$RESULT_ROOT/cp2-qualified-survivors.json"`

Expected: CP1 analyzer consumes only terminal complete triplets, labels decisions elimination-only, and writes zero to two survivors. Every survivor is expanded with new matched fixed/always CP2 controls, receives an exclusive identity, passes `sbatch --test-only`, and gets a job ID. Monitor the CP2 submission using the same six-check, at-least-300-second fail-closed queue/accounting/log loop from Step 3.

- [ ] **Step 5: Analyze all terminal CP1/CP2 results and make the receipt GREEN.**

Run: `uv run python research/qwen3_8b_draft_cadence/analyze.py pilot --manifest "$RESULT_ROOT/pilot-cp2.json" --topology cp2 --cp1-receipt "$RESULT_ROOT/pilot-cp1-receipt.json" --fail-closed --markdown research/qwen3_8b_draft_cadence/PILOT.md --json research/qwen3_8b_draft_cadence/pilot_receipt.json --csv research/qwen3_8b_draft_cadence/pilot_summary.csv && uv run --group test pytest -q research/qwen3_8b_draft_cadence/tests/test_contract.py research/qwen3_8b_draft_cadence/tests/test_analysis.py -k 'pilot_receipt or event_count'`

Expected: analyzer exits 0 only when all terminal artifacts, scheduled-event counts, post-event observations, policy-refit-per-step counts, and required canonical metrics are complete; tests PASS; report promotes no more than two candidates for CP2 and labels all conclusions elimination-only.

- [ ] **Step 6: Stage and create the signed DCO pilot receipt commit.**

```bash
git add research/qwen3_8b_draft_cadence/PILOT.md research/qwen3_8b_draft_cadence/pilot_receipt.json research/qwen3_8b_draft_cadence/pilot_summary.csv research/qwen3_8b_draft_cadence/tests/test_contract.py research/qwen3_8b_draft_cadence/tests/test_analysis.py
git commit -S -s -m "perf(draft): report cadence elimination pilot"
git verify-commit HEAD
```

Expected: signature verification exits 0 and no runtime log or W&B raw export is staged.

### Task 4: Run 1000-step validation and record the production decision

**Files:**
- Modify: `research/qwen3_8b_draft_cadence/tests/test_analysis.py`
- Modify: `research/qwen3_8b_draft_cadence/README.md`
- Create: `research/qwen3_8b_draft_cadence/LONG_VALIDATION.md`
- Create: `research/qwen3_8b_draft_cadence/long_validation_receipt.json`
- Create: `research/qwen3_8b_draft_cadence/long_validation_summary.csv`

**Interfaces:**
- Consumes: at most two pilot survivors, fixed and `always` controls, three matched replicates, frozen windows, checkpoint/resume receipts, and canonical W&B rows.
- Produces: a 1000-step paired decision with exact update/refit accounting and an explicit production-supported or experimental-only recommendation.

- [ ] **Step 1: Write RED final-decision gate tests.**

```python
import json
from pathlib import Path

import pytest

from research.qwen3_8b_draft_cadence.analyze import validate_long_receipt


LONG_RECEIPT = Path(
    "research/qwen3_8b_draft_cadence/long_validation_receipt.json"
)


def test_production_decision_allows_selection_or_valid_no_candidate_rejection() -> None:
    long_receipt = json.loads(LONG_RECEIPT.read_text())
    validate_long_receipt(long_receipt)
    if long_receipt["status"] == "selected":
        selected = long_receipt["candidates"][long_receipt["selected_candidate"]]
        gates = selected["gates"]
        assert gates["overhead_reduction_mean"] >= 0.50
        assert gates["overhead_reduction_ci_lower"] > 0.0
        assert gates["generation_tps_ci_lower"] > -0.02
        assert selected["passed"] is True
        assert long_receipt["production_supported"] is True
    elif long_receipt["status"] == "no_candidate":
        assert long_receipt["selected_candidate"] is None
        assert long_receipt["production_supported"] is False
        assert long_receipt["recommendation"] == "always"
        assert all(
            candidate["passed"] is False
            for candidate in long_receipt["candidates"].values()
        )
    else:
        assert long_receipt["status"] == "no_survivor"
        assert long_receipt["terminal"] is True
        assert long_receipt["candidates"] == {}
        assert long_receipt["recommendation"] == "always"


def long_receipt_fixture(
    tmp_path: Path, monkeypatch, *, passed: bool
) -> dict[str, object]:
    matrix = build_long_matrix(
        analysis_experiment_spec(),
        promoted=("fixed_sparse_10",),
        replicate_indices=(0, 1, 2),
    )
    manifest = render_manifest(matrix)
    manifest_path = tmp_path / "long-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    metrics = passing_candidate_metrics(tmp_path)
    measured = evaluate_candidate(metrics)
    gates = dict(measured.gates)
    if not passed:
        gates["overhead_reduction_mean"] = 0.40
        gates["overhead_reduction_ci_lower"] = -0.10
    recomputed = replace(measured, gates=gates, passed=passed)
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.analyze.build_candidate_metrics",
        lambda _path, _name: metrics,
    )
    monkeypatch.setattr(
        "research.qwen3_8b_draft_cadence.analyze.evaluate_candidate",
        lambda _metrics: recomputed,
    )
    evidence = {
        arm_id: {name: asdict(binding) for name, binding in artifacts.items()}
        for arm_id, artifacts in recomputed.evidence_artifacts_by_arm.items()
    }
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": str(manifest_path.resolve()),
        "status": "selected" if passed else "no_candidate",
        "selected_candidate": "fixed_sparse_10" if passed else None,
        "production_supported": passed,
        "recommendation": "fixed_sparse_10" if passed else "always",
        "candidates": {
            "fixed_sparse_10": {
                "passed": passed,
                "gates": gates,
                "policy_refit_count": 1000,
                "completed_policy_steps": 1000,
                "raw_artifact_sha256_by_arm": recomputed.raw_artifact_sha256_by_arm,
                "evidence_artifacts_by_arm": evidence,
            }
        },
    }


def test_structurally_valid_scientific_rejection_is_green(
    tmp_path: Path, monkeypatch
) -> None:
    rejected = long_receipt_fixture(tmp_path, monkeypatch, passed=False)
    validate_long_receipt(rejected)
    rejected["candidates"]["fixed_sparse_10"]["passed"] = True
    with pytest.raises(ValueError, match="disagrees with frozen gates"):
        validate_long_receipt(rejected)


def test_long_receipt_rejects_fabricated_artifact_hash(
    tmp_path: Path, monkeypatch
) -> None:
    receipt = long_receipt_fixture(tmp_path, monkeypatch, passed=True)
    arm_id = next(iter(receipt["candidates"]["fixed_sparse_10"]["evidence_artifacts_by_arm"]))
    receipt["candidates"]["fixed_sparse_10"]["evidence_artifacts_by_arm"][arm_id]["raw_wandb"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="recomputed manifest evidence"):
        validate_long_receipt(receipt)


def test_gate_schema_rejects_truthy_boolean(tmp_path: Path, monkeypatch) -> None:
    receipt = long_receipt_fixture(tmp_path, monkeypatch, passed=True)
    receipt["candidates"]["fixed_sparse_10"]["gates"][
        "exact_schedule_receipts"
    ] = 1
    with pytest.raises(ValueError, match="boolean gate"):
        validate_long_receipt(receipt)


def test_no_pilot_survivor_is_terminal_without_long_submission(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "empty-long.json"
    receipt_path = tmp_path / "empty-long-no-survivor.json"
    manifest = {
        "schema_version": 1,
        "experiment_id": "experiment",
        "replicates": [],
        "no_survivor_receipt_path": str(receipt_path.resolve()),
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    validate_long_receipt({
        "status": "no_survivor",
        "terminal": True,
        "reason": "no_pilot_survivors",
        "candidates": {},
        "selected_candidate": None,
        "production_supported": False,
        "recommendation": "always",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": str(manifest_path.resolve()),
    })
```

- [ ] **Step 2: Run the RED final-decision test before long runs and confirm the receipt is absent.**

Run: `uv run --group test pytest -q research/qwen3_8b_draft_cadence/tests/test_analysis.py -k 'production_decision'`

Expected: FAIL because `LONG_VALIDATION.md` and its machine-readable long receipt do not exist before validation.

- [ ] **Step 3: Submit the fixed, always, and promoted 1000-step triplets.**

Run `/fairshare oci-hsg` again because long validation is a separate submission epoch, and persist the new comparison and selected account. Then run:

```bash
git pull --ff-only
git verify-commit HEAD
git push
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{u}')"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git submodule status --recursive
uv run python research/qwen3_8b_draft_cadence/launch.py long --spec "$SPEC" --survivors "$RESULT_ROOT/cp2-qualified-survivors.json" --manifest "$RESULT_ROOT/long.json" --replicates 3 --test-only
uv run python research/qwen3_8b_draft_cadence/launch.py long --spec "$SPEC" --survivors "$RESULT_ROOT/cp2-qualified-survivors.json" --manifest "$RESULT_ROOT/long.json" --replicates 3 --submit-next
if [[ -f "$RESULT_ROOT/long-no-survivor.json" ]]; then
  jq -e '
    .status == "no_survivor" and .terminal == true and
    .production_supported == false and .recommendation == "always"
  ' "$RESULT_ROOT/long-no-survivor.json" >/dev/null
  exit 0
fi
mapfile -t submitted_job_ids < <(
  find "$RESULT_ROOT" -name submission.json -type f -exec \
    jq -er 'select(.state == "submitted") | .job_id' {} \; | sort -u
)
[[ "${#submitted_job_ids[@]}" -gt 0 ]]
job_ids_csv="$(IFS=,; echo "${submitted_job_ids[*]}")"
monitor_started_at="$(date +%s)"
for check in 0 1 2 3 4 5; do
  queue_rows="$(squeue --jobs "$job_ids_csv" --noheader --format='%A|%T|%j|%R')"
  accounting_rows="$(sacct -j "$job_ids_csv" --noheader --parsable2 --format=JobIDRaw,State,JobName)"
  if grep -Eq '\|(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)(\||$)' \
    <<<"$accounting_rows"; then
    echo "submitted cadence job entered terminal failure" >&2
    exit 1
  fi
  for job_id in "${submitted_job_ids[@]}"; do
    if ! grep -Eq "^${job_id}(\\.|\\|)" <<<"$queue_rows"$'\n'"$accounting_rows"; then
      echo "job $job_id is absent from both squeue and sacct" >&2
      exit 1
    fi
  done
  if find "$RESULT_ROOT" -name 'slurm-*.out' -type f -print0 | \
    xargs -0 -r grep -Eiq 'Traceback|ModuleNotFoundError|CUDA error|NCCL error|RuntimeError:'; then
    echo "cadence startup failure signature found in logs" >&2
    exit 1
  fi
  if [[ "$check" -lt 5 ]]; then sleep 60; fi
done
(( $(date +%s) - monitor_started_at >= 300 )) || {
  echo "monitoring interval was shorter than five minutes" >&2
  exit 1
}
```

Expected: with zero CP2-qualified survivors, the launcher validates the bound terminal no-survivor receipt and submits no long job. Otherwise the clean pushed recursive SHA, fresh highest-eligible FairShare receipt, and immutable parity fields validate; every rendered matched-triplet job passes `sbatch --test-only`; each submitted triplet records one unique job ID shared by its three sequential arms; six fail-closed checks span at least 300 seconds and find no disappearance, accounting failure, or startup-error signature. Repeat `--submit-next` only after the previous ordinary triplet has its job ID. A resume triplet has the stricter dependency: all three arms of its declared fresh source triplet must be terminal-successful, complete 1000 steps, and pass the product runtime validation for retained `step_400` bytes and its immutable receipt before the resume allocation may be submitted. Include one such predeclared resume-after-Step-400 triplet per promoted candidate and retain its uninterrupted matched triplet for ledger comparison.

- [ ] **Step 4: Analyze all terminal replicates and make the final receipt GREEN.**

Run: `uv run python research/qwen3_8b_draft_cadence/analyze.py long --manifest "$RESULT_ROOT/long.json" --fail-closed --markdown research/qwen3_8b_draft_cadence/LONG_VALIDATION.md --json research/qwen3_8b_draft_cadence/long_validation_receipt.json --csv research/qwen3_8b_draft_cadence/long_validation_summary.csv && uv run --group test pytest -q research/qwen3_8b_draft_cadence/tests/test_analysis.py -k 'production_decision or windows or gen_kl'`

Expected: analyzer exits 0 only with complete common-step windows, three paired replicates, exact policy/draft update/refit counts, uninterrupted/resumed decision equality, and all frozen gates reported. If no candidate passes, the report states that `always` remains the sole production behavior and fixed/adaptive modes remain experimental.

- [ ] **Step 5: Stage and create the signed DCO validation commit.**

```bash
git add research/qwen3_8b_draft_cadence/LONG_VALIDATION.md research/qwen3_8b_draft_cadence/long_validation_receipt.json research/qwen3_8b_draft_cadence/long_validation_summary.csv research/qwen3_8b_draft_cadence/README.md research/qwen3_8b_draft_cadence/tests/test_analysis.py
git commit -S -s -m "docs(draft): report cadence long validation"
git verify-commit HEAD
```

Expected: signature verification exits 0 and no candidate is described as production-supported unless every frozen gate passes.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-22-draft-update-cadence-experiments.md`. Execute only after the product implementation and exact functional GPU gate are terminal GREEN. Task 1 and Task 2 build the harness; Task 3 closes pilot elimination; Task 4 makes the long-run production decision.

Do not run or post `nemo-rl-pr-review` or a self-review in this execution session. Hand the separate Claude Code reviewer the exact product/harness SHAs, immutable container digest, submitted job IDs, W&B URLs, closed windows, paired receipts, failed gates, promoted/eliminated candidates, and remaining statistical or reproducibility risks.
