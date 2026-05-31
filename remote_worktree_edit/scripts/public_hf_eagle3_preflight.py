from transformers import AutoConfig


MODEL_ID = "nvidia/Qwen3-235B-A22B-Eagle3"


config = AutoConfig.from_pretrained(MODEL_ID)
print("model_id:", MODEL_ID)
print("architectures:", config.architectures)
print("eagle_config:", config.eagle_config)
