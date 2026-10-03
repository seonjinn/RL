"""Expose installed test dependencies to audit-only Ray subprocesses.

Appending sites preserves each actor interpreter's own package precedence and
processes the same editable/.pth bindings that the audit driver requires.
This directory is added to PYTHONPATH only by run_pr_vllm029_audit.sbatch.
"""

import site
from pathlib import Path


def add_test_sites() -> None:
    for path in (
        "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker."
        "MegatronPolicyWorker/lib/python3.13/site-packages",
        "/opt/nemo_rl_venv/lib/python3.13/site-packages",
    ):
        if not Path(path).is_dir():
            raise RuntimeError(f"Missing installed audit dependency site: {path}")
        site.addsitedir(path)


add_test_sites()
