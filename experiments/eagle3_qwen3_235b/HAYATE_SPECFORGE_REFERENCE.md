# Hayate SpecForge Reference

Overall: **REFERENCE_ONLY**
SpecForge dir: `/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge`

SpecForge README identifies it as an SGLang ecosystem project for SGLang-compatible draft training.

## Qwen3-235B Config Comparison

| field | SpecForge | current verifier/architecture | match |
| --- | --- | --- | --- |
| aux_layers | `[1, 46, 90]` | `[1, 46, 90]` | True |
| hidden_size | `4096` | `4096` | True |
| intermediate_size | `24576` | `12288` | False |
| max_position_embeddings | `40960` | `262144` | False |
| num_attention_heads | `64` | `64` | True |
| num_key_value_heads | `4` | `4` | True |
| rope_theta | `1000000.0` | `5000000` | False |
| vocab_size | `151936` | `151936` | True |

SpecForge 235B config is useful for aux-layer sanity, but it is not a direct config source for Thinking-2507.

## Example Training Flags

### `run_qwen3_moe_eagle3_online.sh`

| flag | value |
| --- | --- |
| `--target-model-path` | `Qwen/Qwen3-30B-A3B` |
| `--draft-model-config` | `$ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json` |
| `--train-data-path` | `$ROOT_DIR/cache/dataset/sharegpt.jsonl` |
| `--output-dir` | `$ROOT_DIR/outputs/Qwen3-30B-A3B-eagle3` |
| `--num-epochs` | `10` |
| `--draft-global-batch-size` | `None` |
| `--draft-micro-batch-size` | `None` |
| `--batch-size` | `1` |
| `--learning-rate` | `1e-4` |
| `--max-length` | `2048` |
| `--chat-template` | `qwen` |
| `--tp-size` | `$NUM_GPUS` |
| `--ttt-length` | `7` |

### `run_qwen3_dense_eagle3_online_30b-moe_large_qwen_data_long.sh`

| flag | value |
| --- | --- |
| `--target-model-path` | `Qwen/Qwen3-30B-A3B-Base` |
| `--draft-model-config` | `$ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json` |
| `--train-data-path` | `$ROOT_DIR/cache/dataset/magpie_ultrachat_qwen.jsonl` |
| `--output-dir` | `$ROOT_DIR/outputs/Qwen3-30B-A3B-eagle3-base` |
| `--num-epochs` | `10` |
| `--draft-global-batch-size` | `64` |
| `--draft-micro-batch-size` | `None` |
| `--batch-size` | `1` |
| `--learning-rate` | `1e-4` |
| `--max-length` | `32768` |
| `--chat-template` | `qwen` |
| `--tp-size` | `$TP_SIZE` |
| `--ttt-length` | `7` |

### `run_qwen3_dense_eagle3_online_8b_dapo.sh`

| flag | value |
| --- | --- |
| `--target-model-path` | `Qwen/Qwen3-8B` |
| `--draft-model-config` | `$ROOT_DIR/configs/qwen3-8b-eagle3_long.json` |
| `--train-data-path` | `/lustre/fsw/portfolios/coreai/users/hiso/code/nemo-rl_38119888_dev-3/outputs/datagen/poc_spec_8b/outputs_500k.jsonl` |
| `--output-dir` | `$ROOT_DIR/outputs/Qwen3-8B-eagle3-dapo` |
| `--num-epochs` | `10` |
| `--draft-global-batch-size` | `64` |
| `--draft-micro-batch-size` | `2` |
| `--batch-size` | `1` |
| `--learning-rate` | `1e-4` |
| `--max-length` | `32768` |
| `--chat-template` | `qwen` |
| `--tp-size` | `$TP_SIZE` |
| `--ttt-length` | `7` |

## Output Inventories

### `outputs/Qwen3-30B-A3B-eagle3-base/epoch_9`

Status: `present`

| file | size |
| --- | ---: |
| `config.json` | 809 |
| `model.safetensors` | 366378848 |
| `training_state.pt` | 60183 |

### `outputs/Qwen3-8B-eagle3-long/epoch_9`

Status: `present`

| file | size |
| --- | ---: |
| `config.json` | 807 |
| `d2t.bin` | 257485 |
| `model.safetensors` | 799457128 |
| `t2d.bin` | 153421 |
| `training_state.pt` | 4759 |

### `outputs/Qwen3-8B-eagle3-long/epoch_9/with-embed`

Status: `present`

| file | size |
| --- | ---: |
| `config.json` | 807 |
| `d2t.bin` | 257485 |
| `model.safetensors` | 2044116936 |
| `t2d.bin` | 153421 |
| `training_state.pt` | 4759 |
