from __future__ import annotations

from pathlib import Path


EXPERIMENT = Path(__file__).parents[1]


def test_resume_matrix_is_exact_matched_and_fail_closed() -> None:
    submit = (EXPERIMENT / "submit_matrix_oci_hsg.sh").read_text()

    assert submit.startswith("#!/bin/bash\n\nset -euo pipefail\n")
    assert 'readonly resume_step=400' in submit
    assert 'readonly final_step=1000' in submit
    assert 'readonly walltime=07:00:00' in submit
    assert 'readonly checkpoint_deadline=00:06:30:00' in submit
    assert 'readonly gpus_per_node=4' in submit
    assert 'WANDB_RESUME=must' in submit
    assert 'STAGE_MODE=resume' in submit
    assert 'STAGE_MIN_STEP=${final_step}' in submit
    assert 'STAGE_DEADLINE=${checkpoint_deadline}' in submit
    assert '--expected-step "${resume_step}"' in submit
    assert '--validate-manifest' in submit
    assert '--write-manifest' in submit
    assert '--print-manifest-wandb-id' in submit
    assert 'test "${manifest_wandb_id}" = "${wandb_id}"' in submit
    assert 'test "$(git -C "${repo}" rev-parse HEAD)" = "${source_sha}"' in submit
    assert 'test "$(git -C "${HARNESS_REPO}" rev-parse HEAD)" = "${EXPECTED_HARNESS_HEAD}"' in submit


def test_resume_matrix_binds_all_four_original_runs() -> None:
    submit = (EXPERIMENT / "submit_matrix_oci_hsg.sh").read_text()

    expected = {
        "online|dspark-k5|5": "r4a81508a",
        "online|dspark-k7|7": "rcf9fa648",
        "fixed|dspark-k5|5": "rb53842d0",
        "fixed|dspark-k7|7": "r765298fc",
    }
    for identity, wandb_id in expected.items():
        assert f'"{identity}|' in submit
        assert f'|{wandb_id}"' in submit

    assert "21fed91219cad821f1e7cdbaf3fa2edc9f188939" in submit
    assert "2c3e0e064f98bf9a1eb1fac16ed6764ec4d8927b" in submit
    assert "qwen3-8b-dspark-online-matrix-21fed912" in submit
    assert "qwen3-8b-dspark-fixed-control-matrix-2c3e0e06" in submit
    assert "nemo_rl_nightly_20260818_20260818_6296116.sqsh" in submit
    assert "b968826d9c46dd6066d109eabc6255188de91218" in submit
    assert "03326e5043815da1f81b109078b2889737c26017" in submit


def test_all_scheduling_probes_precede_every_actual_submission() -> None:
    submit = (EXPERIMENT / "submit_matrix_oci_hsg.sh").read_text()

    probes = submit.index("run_all test-only")
    actual = submit.index("run_all submit")
    assert probes < actual
    assert 'if [[ "${action}" == test-only ]]; then' in submit
    assert 'sbatch --test-only "${options[@]}" "${runner}"' in submit
    assert 'job_id="$(sbatch --parsable "${options[@]}" "${runner}"' in submit


def test_matrix_preserves_identical_topology_and_common_science_contract() -> None:
    submit = (EXPERIMENT / "submit_matrix_oci_hsg.sh").read_text()

    assert "topology=TP2/DP2/CP1,packing=false,SP=false" in submit
    assert "common science contract=seed42,GBS32,PPS8,GPS4" in submit
    assert 'NUM_SPECULATIVE_TOKENS=${num_speculative_tokens}' in submit
    assert '[[ "${family}" == online || "${family}" == fixed ]]' in submit
    assert '[[ "${arm}" == "dspark-k${num_speculative_tokens}" ]]' in submit
