"""Run on a GB200 compute node before launching the diagnostic workload."""

import json
import os
from contextlib import redirect_stdout
from io import StringIO

import torch

from nemo_rl.utils.refit_memory_probe import refit_memory_phase


@refit_memory_phase("smoke")
def allocate() -> int:
    tensor = torch.ones(1024 * 1024, device="cuda")
    return tensor.numel()


@refit_memory_phase("failure_smoke")
def fail() -> None:
    raise ValueError("expected probe smoke failure")


output = StringIO()
os.environ["NRL_REFIT_MEMORY_PROBE"] = "0"
with redirect_stdout(output):
    assert allocate() == 1024 * 1024
assert not output.getvalue()
os.environ["NRL_REFIT_MEMORY_PROBE"] = "1"
with redirect_stdout(output):
    assert allocate() == 1024 * 1024
    try:
        fail()
    except ValueError as error:
        assert str(error) == "expected probe smoke failure"
    else:
        raise AssertionError("Probe swallowed a workload exception")
records = [
    json.loads(line.removeprefix("REFIT_MEMORY_JSON "))
    for line in output.getvalue().splitlines()
]
assert len(records) == 2
assert records[0]["succeeded"]
assert records[0]["allocated_peak_bytes"] >= 4 * 1024 * 1024
assert not records[1]["succeeded"]
assert all(record["sampler_stopped"] for record in records)
assert all(not record["sampling_errors"] for record in records)
print("REFIT_MEMORY_PROBE_SMOKE_PASS", flush=True)
