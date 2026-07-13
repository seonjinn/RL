# SWE-RL Full-GRPO Log Error Summary

Manifest: `docs/lyris_swerl_fullgrpo_log_fetch_manifest_r20_r21_r22_statusnow_1824_20260614.csv`

| job_id | method | steps | state | local logs | first error excerpt |
| --- | --- | ---: | --- | ---: | --- |
| 2122398 | baseline | 1 | FAILED | 18 | Traceback (most recent call last): / File "/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL-SWE_bench-20260613/./examples/nemo_gym/run_grpo_nemo_gym.py", line 307, in <module> / main() / ~~~~^^ / File "/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL-SWE_bench-20260613/./examples/nemo_gym/run_grpo_nemo_gym.py", line 206, in main / ) = setup(config, tokenizer, train_dataset, val_dataset) |
| 2122399 | pard2 | 1 | CANCELLED by 2001147693 | 18 | [2026-06-14T18:21:29.617] error: *** STEP 2122399.24 ON lyris0272 CANCELLED AT 2026-06-14T18:21:29 DUE to SIGNAL Terminated *** |
| 2122466 | pard | 1 | RUNNING | 18 | Building transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@366798ef8a0a00d8f2c1650d11e7e623d7c33e26 |
| 2123210 | baseline | 1 | RUNNING | 18 | Downloaded flash-attn |
| 2123212 | pard2 | 1 | RUNNING | 17 | echo "[SETUP] apptainer install attempt $attempt failed, retrying..." / sleep 10 / done / if [ $RET -ne 0 ]; then / echo "[SETUP] WARNING: apptainer installation failed after $RETRIES attempts" / fi |
