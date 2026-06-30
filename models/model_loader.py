"""
models/model_loader.py

Unified loader for all four models in the bias_diffusion experiment.
All model loading in the project goes through this module -- nothing else
imports from transformers or peft directly.

Models supported:
    mdlm_169m   -- kuleshov-group/mdlm-owt
    pythia_160m -- EleutherAI/pythia-160m
    llada_8b    -- GSAI-ML/LLaDA-8B-Base
    llama_8b    -- meta-llama/Llama-3.1-8B

Colab setup (A100, run once per session):
    pip install flash-attn (see README for exact wheel URL)

Authors: [your name]
"""

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModel,
    BitsAndBytesConfig,
)
from peft import PeftModel
import torch


# ---------------------------------------------------------------------------
# Runtime patches for transformers compatibility
# Applied at import time -- runs in the same process as model loading.
# Fixes two issues introduced in transformers 5.x that break MDLM and LLaDA:
#   1. get_keys_to_not_convert accesses all_tied_weights_keys before it exists
#   2. tie_weights() called with keyword args but old models define no params
# ---------------------------------------------------------------------------

def _apply_compatibility_patches():
    try:
        import transformers.quantizers.base as qbase
        import transformers.modeling_utils as mutils

        # Patch 1: inject all_tied_weights_keys before quantizer accesses it
        if not hasattr(qbase.get_keys_to_not_convert, "_patched"):
            _orig = qbase.get_keys_to_not_convert
            def _safe_get_keys(model):
                if not hasattr(model, "all_tied_weights_keys"):
                    model.all_tied_weights_keys = dict()
                return _orig(model)
            _safe_get_keys._patched = True
            qbase.get_keys_to_not_convert = _safe_get_keys

        # Patch 2: make tie_weights accept **kwargs, and ensure
        # all_tied_weights_keys exists before ANY code path accesses it
        # (covers both get_keys_to_not_convert and the meta-device move
        # in _finalize_model_loading itself, which accesses the attribute
        # directly without going through a function we can patch).
        if not hasattr(mutils.PreTrainedModel._finalize_model_loading, "_patched"):
            _orig_finalize = mutils.PreTrainedModel._finalize_model_loading
            def _safe_finalize(model, load_config, loading_info):
                if not hasattr(model, "all_tied_weights_keys"):
                    model.all_tied_weights_keys = dict()
                _orig_tie = model.tie_weights
                def _safe_tie(**kwargs):
                    try:
                        return _orig_tie(**kwargs)
                    except TypeError:
                        return _orig_tie()
                model.tie_weights = _safe_tie
                return _orig_finalize(model, load_config, loading_info)
            _safe_finalize._patched = True
            mutils.PreTrainedModel._finalize_model_loading = staticmethod(_safe_finalize)
    except Exception as e:
        pass  # silently skip if transformers version doesn't need patches


_apply_compatibility_patches()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mdlm_169m": {
        "hf_id": "kuleshov-group/mdlm-owt",
        "loader": "masked",
        "trust_remote_code": True,
        "model_type": "dlm",
        "mask_token_id": 50257,
        "tokenizer_id": "gpt2",
    },
    "pythia_160m": {
        "hf_id": "EleutherAI/pythia-160m",
        "loader": "causal",
        "trust_remote_code": False,
        "model_type": "ar",
        "mask_token_id": None,
    },
    "llada_8b": {
        "hf_id": "GSAI-ML/LLaDA-8B-Base",
        "loader": "auto",
        "trust_remote_code": True,
        "model_type": "dlm",
        "mask_token_id": 126336,
    },
    "llama_8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "loader": "causal",
        "trust_remote_code": False,
        "model_type": "ar",
        "mask_token_id": None,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str = "cpu", quantize: bool = False):
    """
    Load a base model and its tokenizer from HuggingFace.

    Args:
        model_name: one of the four model keys
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
        model_name:      one of the four model keys
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_quantization_config():
    """NF4 4-bit quantization config (Dettmers et al., 2023 QLoRA)."""
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
    """
    Load tokenizer. Handles missing pad token and padding side per paradigm.
    MDLM uses GPT-2 tokenizer (tokenizer_id override).
    AR models get left padding; DLMs stay at default right.
    """
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
    """
    Dispatch to the correct AutoModel class based on config loader field.
    Patches cached model files for tied weights compatibility before loading.
    Quantized models use device_map=auto (required by bitsandbytes).
    Non-quantized models are moved to device explicitly.
    """
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
    elif loader == "auto":
        model = AutoModel.from_pretrained(hf_id, **kwargs)
    else:
        raise ValueError(f"Unknown loader type '{loader}' in MODEL_REGISTRY.")

    if not quantize:
        model = model.to(device)

    return model.eval()