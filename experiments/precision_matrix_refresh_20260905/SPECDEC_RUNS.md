# MXFP8 Speculative-Decoding Runs

Source commit: `d143088d49f6543b34edcfe940db00ee6ba0351a`

All rows use Qwen3-235B Async-1off, FlashInfer TRTLLM, NCCL Reshard,
MXFP8 routed-expert rollout, vLLM Model Runner V2, K5, and the audited
performance recipe. The draft checkpoints are frozen. The no-SpecDec rows are
the matched V2 baselines.

## Two-Step Gates

| Parameter storage | Method | Job | Account | Status |
|---|---|---:|---|---|
| MXFP8 (`fp8_param=true`) | No SpecDec | 7420491 | `nemotron_n3_post` | Pending |
| MXFP8 (`fp8_param=true`) | DFlash | 7420493 | `nemotron_n4_post` | Pending |
| MXFP8 (`fp8_param=true`) | DSpark | 7420495 | `nemotron_sw_post` | Pending |
| BF16 (`fp8_param=false`) | No SpecDec | 7420600 | `nemotron_n3_post` | Pending |
| BF16 (`fp8_param=false`) | DFlash | 7420603 | `nemotron_n4_post` | Pending |
| BF16 (`fp8_param=false`) | DSpark | 7420604 | `nemotron_sw_post` | Pending |

## Twenty-Step Runs

Each job starts only after its matching two-step gate exits successfully.

| Parameter storage | Method | Job | Gate dependency |
|---|---|---:|---:|
| MXFP8 (`fp8_param=true`) | No SpecDec | 7420619 | 7420491 |
| MXFP8 (`fp8_param=true`) | DFlash | 7420622 | 7420493 |
| MXFP8 (`fp8_param=true`) | DSpark | 7420623 | 7420495 |
| BF16 (`fp8_param=false`) | No SpecDec | 7420625 | 7420600 |
| BF16 (`fp8_param=false`) | DFlash | 7420626 | 7420603 |
| BF16 (`fp8_param=false`) | DSpark | 7420627 | 7420604 |

W&B project: `nvidia/nemo-rl-mxfp8-training`. Add the exact run links after
worker initialization creates each run.
