from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
import torch
from models import transformers_patches 

from models.model_registry import MODEL_REGISTRY


# Public API
def load_model(model_name: str, device: str = "cpu", quantize: bool = False):
    """
    Load a base model and its tokenizer from HuggingFace.

    Args:
        model_name: one of the model keys in MODEL_REGISTRY
        device:     cpu (local) or cuda (Colab)
        quantize:   4-bit NF4 via bitsandbytes -- Colab only

    Returns:
        model, tokenizer
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    if quantize and device == "cpu":
        raise RuntimeError(
            "quantize=True requires CUDA. Use device='cuda' or run on Colab."
        )
    config = MODEL_REGISTRY[model_name]
    tokenizer = _load_tokenizer(
        config["hf_id"],
        config["trust_remote_code"],
        config["model_type"],
        config.get("tokenizer_id"),
    )
    model = _load_base_model(config, device, quantize)
    return model, tokenizer


def load_finetuned(
    model_name: str,
    checkpoint_path: str,
    device: str = "cpu",
    quantize: bool = False,
):
    """
    Load a base model and attach a saved LoRA adapter.
    Adapter is NOT merged -- base weights stay intact for adapter swapping.

    Args:
        model_name:      one of the model keys in MODEL_REGISTRY
        checkpoint_path: path to saved LoRA adapter directory
        device:          cpu or cuda
        quantize:        4-bit loading for large models on Colab

    Returns:
        model (PeftModel), tokenizer
    """
    import os
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"Check the path or re-run finetuning."
        )
    if quantize and device == "cpu":
        raise RuntimeError(
            "quantize=True requires CUDA. Use device='cuda' or run on Colab."
        )
    config = MODEL_REGISTRY[model_name]
    tokenizer = _load_tokenizer(
        config["hf_id"],
        config["trust_remote_code"],
        config["model_type"],
        config.get("tokenizer_id"),
    )
    base_model = _load_base_model(config, device, quantize)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return model.eval(), tokenizer

def _get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_tokenizer(
    hf_id: str,
    trust_remote_code: bool,
    model_type: str,
    tokenizer_id: str = None,
):

    tok_source = tokenizer_id if tokenizer_id is not None else hf_id
    tokenizer = AutoTokenizer.from_pretrained(
        tok_source,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model_type == "ar":
        tokenizer.padding_side = "left"
    return tokenizer

def _load_base_model(config: dict, device: str, quantize: bool):

    hf_id = config["hf_id"]
    trust = config["trust_remote_code"]

    if quantize:
        kwargs = {
            "quantization_config": _get_quantization_config(),
            "device_map": "auto",
            "trust_remote_code": trust,
        }
    else:
        kwargs = {
            "trust_remote_code": trust,
        }

    loader = config["loader"]
    if loader == "masked":
        model = AutoModelForMaskedLM.from_pretrained(hf_id, **kwargs)
    elif loader == "causal":
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    else:
        raise ValueError(f"Unknown loader type '{loader}' in MODEL_REGISTRY.")

    if not quantize:
        model = model.to(device)

    return model.eval()