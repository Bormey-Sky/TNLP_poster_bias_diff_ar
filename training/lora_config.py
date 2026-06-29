"""
training/lora_config.py

LoRA hyperparameter configuration for all four models.

Rank and alpha are set per scale tier following Hu et al. (2022):
    Small models (hidden=768):  rank=16, alpha=32
    Large models (hidden=4096): rank=64, alpha=128

Target modules are set per model architecture, confirmed by inspecting
model.named_modules() on each loaded checkpoint:
    Pythia-160M (GPT-NeoX):  query_key_value, dense
    MDLM-169M   (DiT):       attn_qkv, attn_out
    LLaDA-8B    (LLaMA-style): q_proj, k_proj, v_proj
    LLaMA-3.1-8B (LLaMA):    q_proj, k_proj, v_proj, o_proj

References:
    Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
    Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.

Authors: [your name]
"""

from peft import LoraConfig, TaskType


# ---------------------------------------------------------------------------
# LoRA configurations per model
# ---------------------------------------------------------------------------

LORA_CONFIGS = {
    "pythia_160m": LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query_key_value", "dense"],
        bias="none",
    ),
    "mdlm_169m": LoraConfig(
        task_type=None,            # MDLM has no task type -- it is a masked
                                   # diffusion model with no generation methods.
                                   # task_type=None applies LoRA without wrapping
                                   # in a task-specific PeftModel subclass.
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["attn_qkv", "attn_out"],
        bias="none",
    ),
    "llada_8b": LoraConfig(
        task_type=None,            # LLaDA is a masked diffusion model --
                                   # no prepare_inputs_for_generation method
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj"],
        bias="none",
    ),
    "llama_8b": LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    ),
}


def get_lora_config(model_name: str) -> LoraConfig:
    """
    Return the LoraConfig for a given model.

    Args:
        model_name: one of ['mdlm_169m', 'pythia_160m', 'llada_8b', 'llama_8b']

    Returns:
        LoraConfig with rank, alpha, dropout, and target modules set

    Raises:
        ValueError: if model_name is not recognized
    """
    if model_name not in LORA_CONFIGS:
        raise ValueError(
            f"No LoRA config for '{model_name}'. "
            f"Choose from: {list(LORA_CONFIGS.keys())}"
        )
    return LORA_CONFIGS[model_name]