# Qwen235B MathRL Online PARD-2 Local-Model r3 Status - 2026-06-16

Snapshot time: `2026-06-16 00:53 PDT`.

Scope: Qwen3-235B MathRL generation-bound local-model r3 jobs from
`latest_oci_hsg_qwen235b_mathrl_genbound1024_localmodel_baseline_online_r3_20260615_jobs.csv`.
Both jobs use `nemotron_n3_post`, 32 nodes x 4 GPUs, `max_new_tokens=1024`,
`min_tokens=1024`, `temperature=1.0`, `top_p=1.0`, `top_k=-1`.

## Status

| Job | Method | State | Evidence | Current interpretation |
|---:|---|---|---|---|
| `3332282` | baseline | `FAILED` after `00:18:38` | `ray.exceptions.ActorUnschedulableError: Could not create the actor because worker startup repeatedly failed.` | Failed before Step 1 during vLLM generation worker creation. No timing metrics. |
| `3332283` | online PARD-2 K3 | `CANCELLED` after `02:58:58` | Driver process stayed alive but blocked before Step 1; `ray-driver.log` only had Ray connection lines; Ray status showed `0.0/128.0 GPU` and `0.0/128.0 worker_units` in use. | Irrecoverable Ray driver registration hang before useful work. Cancelled to release 32 nodes. No timing metrics. |

## Diagnostic Details

- `3332283` had a live `python examples/run_grpo.py` process for nearly 3h.
- The driver process was sleeping in `unix_stream_data_wait` with near-zero CPU.
- `py-spy` showed the main thread blocked in Ray CoreWorker registration, specifically `ray::ipc::RayletIpcClient::RegisterClient` under `ray.init(address="auto", runtime_env=...)`.
- Simple `ray.init(...)` probes inside the same head container succeeded, including a probe with the same style of full-env `runtime_env`.
- Ray head and 31 workers were alive, but the driver never submitted usable work to the cluster.
- No NeMo-RL `Step x/y`, `Total step time`, acceptance, or generation-throughput block exists in `3332283-logs/ray-driver.log`.
- The paired baseline `3332282` failed at the same setup stage, specifically while `RayWorkerGroup` was waiting for vLLM generation workers.
- `scancel 3332283` was issued after confirming no step progress and no Ray resource usage.

## Relevant Paths

- Local fetched logs:
  - `tmp/oci_hsg_qwen235b_mathrl_genbound1024_localmodel_r3_logs/baseline/3332282-logs/ray-driver.log`
  - `tmp/oci_hsg_qwen235b_mathrl_genbound1024_localmodel_r3_logs/online_pard2_k3/3332283-logs/ray-driver.log`
- Remote log root:
  - `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260615_qwen235b_mathrl_n3post_temp1_genbound1024_localmodel_r3`

## Next Action

The next replacement should be a smaller startup-gate run before spending another full 32-node allocation. Use the known-good system Python/Ray path (`/opt/nemo_rl_venv/bin/python`) and keep the first retry to `max_steps=1` or `2`; the current blocker is startup/runtime-env reliability, not PARD-2 training loss or generation acceptance.
