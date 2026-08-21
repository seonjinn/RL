import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--online-job-id", required=True)
    parser.add_argument("--fixed-job-id", required=True)
    parser.add_argument("--online-wandb-run-id", required=True)
    parser.add_argument("--fixed-wandb-run-id", required=True)
    parser.add_argument("--online-final-dir", required=True)
    parser.add_argument("--fixed-final-dir", required=True)
    args = parser.parse_args()

    def arm_payload(job_id: str, wandb_run_id: str, final_dir: str) -> dict[str, str]:
        return {
            "job_id": job_id,
            "wandb_run_id": wandb_run_id,
            "wandb_url": (
                "https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/"
                f"{wandb_run_id}"
            ),
            "final_dir": final_dir,
        }

    payload = {
        "label": "secondary_sanity",
        "interpretation": "secondary/sanity; not science",
        "product_head": "79e80af96a13522e6049658663a8c40ab21e8314",
        "execution_git_sha": args.expected_head,
        "test_only_before_actual": True,
        "independent_jobs": True,
        "online": arm_payload(
            args.online_job_id,
            args.online_wandb_run_id,
            args.online_final_dir,
        ),
        "fixed": arm_payload(
            args.fixed_job_id,
            args.fixed_wandb_run_id,
            args.fixed_final_dir,
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
