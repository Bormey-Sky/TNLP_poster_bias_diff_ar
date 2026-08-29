from peft import LoraConfig, TaskType


LORA_CONFIGS = {
    "pythia_160m": LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["query_key_value", "dense"],
        bias="none",
    ),
    "mdlm_169m": LoraConfig(
        task_type=None,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["attn_qkv", "attn_out"],
        bias="none",
    ),
}


def get_lora_config(model_name: str) -> LoraConfig:
    if model_name not in LORA_CONFIGS:
        raise ValueError(
            f"No LoRA config for '{model_name}'. "
            f"Choose from: {list(LORA_CONFIGS.keys())}"
        )
    return LORA_CONFIGS[model_name]