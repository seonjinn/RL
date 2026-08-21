import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("online", "fixed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--validator", required=True)
    args = parser.parse_args()
    validation_delta = {
        "online": (
            "online draft loss/grad and refit markers; update-probe checksum "
            "evidence excluded because probe=false"
        ),
        "fixed": (
            "fixed dense-control target loss/grad and policy generation refit markers"
        ),
    }
    payload = {
        "label": "secondary_sanity",
        "interpretation": "secondary/sanity; not science",
        "product_head": "79e80af96a13522e6049658663a8c40ab21e8314",
        "execution_git_sha": args.git_sha,
        "arm": args.arm,
        "slurm_job_id": args.job_id,
        "wandb_run_id": args.wandb_run_id,
        "wandb_url": (
            "https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/"
            f"{args.wandb_run_id}"
        ),
        "probe_enabled": False,
        "nsys_enabled": False,
        "max_num_steps": 50,
        "validation_period": 1,
        "checkpoint_period": 50,
        "checkpoint_deadline": "00:00:50:00",
        "checkpoint_step": args.checkpoint_step,
        "validator": args.validator,
        "unavoidable_validation_delta": validation_delta[args.arm],
        "checkpoint_contract_delta": (
            "online hard-codes the matched deadline/milestone; fixed receives the "
            "same values through its arm-specific stage contract"
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
