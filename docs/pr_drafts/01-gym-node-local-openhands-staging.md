# [SWE] Stage the OpenHands runtime on node-local storage

## Summary

Stage OpenHands and miniforge on node-local storage once per job. SWE rollout containers then mount the staged copy at the existing `/openhands_setup` path.

## Why

Each rollout starts a new OpenHands process and action server. Reading thousands of Python files from shared storage makes both starts slow. vLLM 0.25.1 improves generation, but it does not change this startup path.

## High-level implementation

- Add an opt-in node-local staging mode to `swe_agents/app.py`.
- Copy the runtime once into a job-scoped temporary directory.
- Rewrite absolute environment paths for the container path `/openhands_setup`.
- Mount the staged directory at `/openhands_setup`.
- Keep the current shared-storage path as the fallback.

## Performance impact

Measured on Lyris GB200 with vLLM 0.25.1 and 24 SWE rollouts:

- Connect: 15.0 s to 11.1 s per rollout, 26.0% lower.
- Framework: 21.8 s to 16.4 s per rollout, 24.8% lower.
- Combined saving: 9.3 s per rollout.
- Generation throughput stayed at 276-277 tokens/s.

Staging 5.3 GB took about 216 seconds once per job. The modeled break-even is about 24 rollouts. A historical n=80 run reported 7.7% lower job wall time; raw logs will be attached before this draft is marked ready.

## Validation

- Compare staged and unstaged runs with the same inputs.
- Check generated patches, rewards, and valid rollout rate.
- Include staging, failures, and drain time in full-job results.
