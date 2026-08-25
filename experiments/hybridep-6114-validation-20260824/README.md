# HybridEP Validation with Megatron-LM #6114

## Objective

Validate the combined runtime behavior of NeMo-RL PRs #2964, #3436, and
#3438 against official Megatron-LM main with #6114 merged and #5515 absent.

## Immutable source boundary

- NeMo-RL integration commit: `68f647a01`
- Megatron-LM commit: `377ad24cd05f41686fafe2e6747f47678b8581c8`
- Megatron-LM #6114 merge commit: `723db5a72790aefc02f5a0228e6607eef70c0533`
- HybridEP commit: `17cfb817bccec3a9c247013360cc550c2bac441e`

The Megatron-LM commit contains #6114 and does not contain the open #5515
branch. Runtime launchers must verify both the NeMo-RL and nested
Megatron-LM revisions before starting Ray.

## Gated runtime matrix

| Gate | Recipe | PP | CP | Purpose |
|---|---|---:|---:|---|
| 1 | Qwen3-30B-A3B sync | 1 | 1 | NeMo-RL one-time prepadding |
| 2 | Qwen3-30B-A3B async | 2 | 1 | Low-cost pipeline-parallel fallback |
| 3 | Qwen3-235B-A22B sync | 8 | 2 | Pipeline and context parallelism together |
| 4 | Nemotron3 Super sync | 1 | 1 | Expert-bias padding-mask path fixed by #6114 |

Run focused recipe and router tests before the model gates. Stop at the first
hang, OOM, or correctness failure and preserve the failing log.
