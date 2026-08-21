import json
import os
from pathlib import Path
import subprocess

import yaml

from research.qwen3_8b_dflash_nonnsys_sanity_b.contract import (
    load_resolved_config,
    validate_experiment_contract,
)
from research.qwen3_8b_dflash_nonnsys_sanity_b.validate_online_sanity import (
    validate_sanity,
    validate_validation_history,
)


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "research/qwen3_8b_dflash_nonnsys_sanity_b"


def test_configs_form_a_matched_secondary_non_nsys_sanity_pair() -> None:
    contract = validate_experiment_contract(EXPERIMENT)

    assert contract["label"] == "secondary_sanity"
    assert contract["product_head"] == "79e80af96a13522e6049658663a8c40ab21e8314"
    assert contract["shared"]["max_num_steps"] == 50
    assert contract["shared"]["probe_enabled"] is False
    assert contract["shared"]["nsys_enabled"] is False
    assert contract["shared"]["wandb_path"] == ("nvidia/sna-nemo-rl-online-drafter")
    assert contract["shared"]["target_revision"] == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert contract["shared"]["drafter_revision"] == (
        "9b41424b7109f9c5413454f481b09a82b85333f4"
    )

    online = load_resolved_config(EXPERIMENT / "online_config.yaml")
    fixed = load_resolved_config(EXPERIMENT / "fixed_config.yaml")
    for config in (online, fixed):
        assert config["grpo"]["seed"] == 42
        assert config["grpo"]["max_num_steps"] == 50
        assert config["grpo"]["num_prompts_per_step"] == 8
        assert config["grpo"]["num_generations_per_prompt"] == 4
        assert config["policy"]["train_global_batch_size"] == 32
        assert config["policy"]["sequence_packing"]["enabled"] is False
        assert config["policy"]["megatron_cfg"]["sequence_parallel"] is False
        assert config["policy"]["draft"]["update_probe_enabled"] is False
        assert (
            config["policy"]["generation"]["vllm_kwargs"]["speculative_config"][
                "method"
            ]
            == "dflash"
        )
        assert (
            config["policy"]["generation"]["vllm_kwargs"]["speculative_config"][
                "num_speculative_tokens"
            ]
            == 7
        )
        assert config["logger"]["wandb_enabled"] is True
        assert config["logger"]["wandb"]["entity"] == "nvidia"
        assert config["logger"]["wandb"]["project"] == "sna-nemo-rl-online-drafter"
        assert "secondary-sanity" in config["logger"]["wandb"]["tags"]

    assert online["policy"]["draft"]["enabled"] is True
    assert fixed["policy"]["draft"]["enabled"] is False
    assert (
        online["policy"]["generation"]["vllm_kwargs"]["compilation_config"]
        == fixed["policy"]["generation"]["vllm_kwargs"]["compilation_config"]
    )


def test_manifest_records_unavoidable_launcher_deltas() -> None:
    manifest = yaml.safe_load((EXPERIMENT / "manifest.yaml").read_text())

    assert manifest["label"] == "secondary_sanity"
    assert manifest["interpretation"] == "secondary/sanity; not science"
    assert manifest["arms"]["online"]["source_launcher"] == (
        "research/qwen3_8b_dflash_online_cp1/run_gate_oci_hsg.sbatch"
    )
    assert manifest["arms"]["fixed"]["source_launcher"] == (
        "research/qwen3_8b_dflash_fixed_dense_control/run_oci_hsg.sbatch"
    )
    assert manifest["known_deltas"]["validation"]
    assert manifest["known_deltas"]["checkpoint_deadline"]


def test_online_sanity_accepts_nonprobe_draft_and_validation_evidence() -> None:
    refit = (
        "draft_refit_manifest=draft_count=17\n"
        "draft_refit_load=complete\n"
        "draft_refit_finalize=complete\n"
    )
    validate_sanity(
        {
            "train/draft_loss": {"50": 1.25},
            "train/draft_grad_norm": {"50": 0.25},
            "train/vllm/spec_acceptance_rate": {"50": 0.41},
        },
        2 * refit,
    )
    validate_validation_history(
        [
            {"_step": 0, "validation/accuracy": 0.0, "validation/avg_length": 128},
            {"_step": 50, "validation/accuracy": 0.25, "validation/avg_length": 256},
        ]
    )


def test_submit_runs_both_test_only_checks_before_independent_actual_jobs(
    tmp_path: Path,
) -> None:
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SBATCH_LOG"\n'
        'case " $* " in *" --test-only "*) echo "forecast" >&2 ;; '
        '*) counter_file="$SBATCH_LOG.counter"; n=7000; '
        'test -f "$counter_file" && n=$(cat "$counter_file"); '
        'n=$((n + 1)); printf "%s" "$n" > "$counter_file"; echo "$n" ;; esac\n'
    )
    fake_sbatch.chmod(0o755)
    run_root = tmp_path / "results"
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SBATCH_LOG": str(sbatch_log),
        "REMOTE_REPO": str(ROOT),
        "EXPECTED_HEAD": "79e80af96a13522e6049658663a8c40ab21e8314",
        "RUN_ROOT": str(run_root),
        "CONTAINER": "/lustre/fake.sqsh",
        "TARGET_SNAPSHOT": ("/lustre/target/b968826d9c46dd6066d109eabc6255188de91218"),
        "DRAFTER_SNAPSHOT": ("/lustre/draft/9b41424b7109f9c5413454f481b09a82b85333f4"),
        "SBATCH_ACCOUNT": "test-account",
        "WANDB_API_KEY": "test-only-placeholder",  # pragma: allowlist secret
    }

    result = subprocess.run(
        ["bash", EXPERIMENT / "submit.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    calls = sbatch_log.read_text().splitlines()
    assert len(calls) == 4
    assert "--test-only" in calls[0]
    assert "--test-only" in calls[1]
    assert "--test-only" not in calls[2]
    assert "--test-only" not in calls[3]
    assert all("--dependency=" not in call for call in calls)
    assert "run_online_oci_hsg.sbatch" in calls[0]
    assert "run_fixed_oci_hsg.sbatch" in calls[1]
    assert "run_online_oci_hsg.sbatch" in calls[2]
    assert "run_fixed_oci_hsg.sbatch" in calls[3]

    submission_path = Path(
        next(
            line.removeprefix("SUBMISSION_MANIFEST=")
            for line in result.stdout.splitlines()
            if line.startswith("SUBMISSION_MANIFEST=")
        )
    )
    submission = json.loads(submission_path.read_text())
    assert submission["online"]["job_id"] == "7001"
    assert submission["fixed"]["job_id"] == "7002"
    assert submission["online"]["wandb_run_id"] != submission["fixed"]["wandb_run_id"]
    assert submission["online"]["final_dir"] != submission["fixed"]["final_dir"]
    assert submission["interpretation"] == "secondary/sanity; not science"
