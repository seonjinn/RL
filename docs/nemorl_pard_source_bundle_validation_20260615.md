# NeMo-RL PARD/PARD-2 Source Bundle Validation

Overall: **PASS**

Roots:

- `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci`
- `/Users/sna/Nemo-RL_Qwen3_Roadmap/remote_patch_pard2_official`
- `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL`

| check | status | selected file | missing markers |
| --- | --- | --- | --- |
| draft model attach/refit setup | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/models/megatron/setup.py` |  |
| PARD draft builder/export | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/models/megatron/draft/utils.py` |  |
| official PARD-2 target projection | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/remote_patch_pard2_official/nemo_rl/models/megatron/draft/utils.py` |  |
| PARD-2 target feature training path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/remote_patch_pard2_official/nemo_rl/models/megatron/draft/pard.py` |  |
| hidden-state capture | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/models/megatron/draft/hidden_capture.py` |  |
| PARD-2 train integration | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/remote_patch_pard2_official/nemo_rl/models/megatron/train.py` |  |
| draft loss packing | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/algorithms/loss/utils.py` |  |
| GRPO online-draft metrics | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/algorithms/grpo.py` |  |
| vLLM target_proj refit guard | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_backend.py` |  |

## Compile Results

| file | status | error |
| --- | --- | --- |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/algorithms/grpo.py` | PASS |  |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/algorithms/loss/utils.py` | PASS |  |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/models/megatron/draft/hidden_capture.py` | PASS |  |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/models/megatron/draft/utils.py` | PASS |  |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/models/megatron/setup.py` | PASS |  |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_backend.py` | PASS |  |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/remote_patch_pard2_official/nemo_rl/models/megatron/draft/pard.py` | PASS |  |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/remote_patch_pard2_official/nemo_rl/models/megatron/draft/utils.py` | PASS |  |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/remote_patch_pard2_official/nemo_rl/models/megatron/train.py` | PASS |  |
