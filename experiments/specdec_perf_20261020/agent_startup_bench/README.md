# OpenHands agent startup bench

The SWE agent spends about 56 s per trajectory before its first LLM call (pre 23.5 s, connect 17.9 s, init 14.5 s in run 8228299, see `swe_agent_timeline.py`). The pre phase grows with the number of agents that start together (corr 0.77). This bench measures the floor of the startup steps and tests one cause: Python imports from the 2.4G OpenHands venv on Lustre.

The bench runs the startup commands of `run_infer.sh` in the agent container on one node. It runs them at concurrency 1, 8, 32 and 64. It runs them once with the setup tree on Lustre and once with a copy in `/dev/shm`. The copy is bound at the same absolute paths, so the venv works without a rebuild.

```bash
sbatch --export=ALL,SETUP=<real setup dir>,SIF=<agent sif> run.sbatch
python summarize.py results/run_<ts>_<job> report/data/agent_startup_bench.csv
```

How to read the result:
- shm close to Lustre at n=1 and n=64: imports are CPU bound. A node-local copy does not help. Look at a warm runtime pool.
- shm much faster at n=32 and n=64: Lustre metadata contention. A node-local or squashfs setup tree helps.
- poetry_version and poetry_noop show the cost of the extra `poetry run` calls in `run_infer.sh`.

The bench does not start the action execution server or run `initialize_runtime`. `server_import` is only the import part of the connect phase.