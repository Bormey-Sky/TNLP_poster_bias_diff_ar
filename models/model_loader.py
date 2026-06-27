"""
models/model_loader.py

Unified loader for all four models in the bias_diffusion experiment.
All model loading in the project goes through this module — nothing else
imports from transformers or peft directly.

Models supported:
    mdlm_169m   — kuleshov-group/mdlm-no_flashattn-fp32-owt
    pythia_160m — EleutherAI/pythia-160m
    llada_8b    — GSAI-ML/LLaDA-8B-Base
    llama_8b    — meta-llama/Llama-3.1-8B

Authors: [your name]
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel, BitsAndBytesConfig
from peft import PeftModel
import torch


# ---------------------------------------------------------------------------
# Model registry
# Maps internal model_name keys to HuggingFace model IDs and loader config.
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mdlm_169m": {
        "hf_id": "kuleshov-group/mdlm-no_flashattn-fp32-owt",
        "loader": "masked",        # AutoModelForMaskedLM
        "trust_remote_code": True, # required — checkpoint has custom modeling code
        "model_type": "dlm",       # diffusion language model
        "mask_token_id": 50257,    # absorbing state = vocab_size, one beyond GPT-2 EOS
        "tokenizer_id": "gpt2",    # MDLM does not ship its own tokenizer — use GPT-2
        "patch_flash_attn_import": True,  # strip flash_attn from get_imports check
    },
    "pythia_160m": {
        "hf_id": "EleutherAI/pythia-160m",
        "loader": "causal",        # AutoModelForCausalLM
        "trust_remote_code": False,
        "model_type": "ar",        # autoregressive
        "mask_token_id": None,     # AR model — no masking needed
    },
    "llada_8b": {
        "hf_id": "GSAI-ML/LLaDA-8B-Base",
        "loader": "auto",          # AutoModel — LLaDA uses custom class
        "trust_remote_code": True,
        "model_type": "dlm",
        "mask_token_id": 126336,   # <|mdmmask|> — confirmed in ML-GSAI/LLaDA generate.py
    },
    "llama_8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "loader": "causal",
        "trust_remote_code": False,
        "model_type": "ar",
        "mask_token_id": None,     # AR model — no masking needed
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str = "cpu", quantize: bool = False):
    """
    Load a base model and its tokenizer from HuggingFace.

    Args:
        model_name: one of ['mdlm_169m', 'pythia_160m', 'llada_8b', 'llama_8b']
        device:     'cpu' (default, local Mac) or 'cuda' (Colab)
        quantize:   if True, load in 4-bit via bitsandbytes (Colab/CUDA only)

    Returns:
        model:      loaded model in eval mode, on the specified device
        tokenizer:  corresponding tokenizer

    Raises:
        ValueError:  if model_name is not in MODEL_REGISTRY
        RuntimeError: if quantize=True is requested on CPU
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    if quantize and device == "cpu":
        raise RuntimeError(
            "quantize=True requires CUDA. "
            "Use device='cuda' or run on Colab."
        )

    config = MODEL_REGISTRY[model_name]
    tokenizer = _load_tokenizer(config["hf_id"], config["trust_remote_code"], config["model_type"], config.get("tokenizer_id"))
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
    The adapter is NOT merged — base model weights remain unchanged,
    allowing adapter swapping without reloading the full base model.

    Args:
        model_name:      one of ['mdlm_169m', 'pythia_160m', 'llada_8b', 'llama_8b']
        checkpoint_path: path to the saved LoRA adapter directory
                         (contains adapter_config.json + adapter_model.safetensors)
        device:          'cpu' (default) or 'cuda' (Colab)
        quantize:        if True, load base model in 4-bit before attaching adapter

    Returns:
        model:      PeftModel with adapter attached, in eval mode
        tokenizer:  corresponding tokenizer

    Raises:
        ValueError:        if model_name is not in MODEL_REGISTRY
        FileNotFoundError: if checkpoint_path does not exist
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
            "quantize=True requires CUDA. "
            "Use device='cuda' or run on Colab."
        )

    # Load base model first, then attach adapter on top
    config = MODEL_REGISTRY[model_name]
    tokenizer = _load_tokenizer(config["hf_id"], config["trust_remote_code"], config["model_type"], config.get("tokenizer_id"))
    base_model = _load_base_model(config, device, quantize)

    # Attach LoRA adapter — adapter is NOT merged so base weights stay intact
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    return model.eval(), tokenizer


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_quantization_config():
    """
    Returns a BitsAndBytesConfig for 4-bit quantization.
    Used only for large models (llada_8b, llama_8b) on CUDA.

    Uses NF4 (NormalFloat4) quantization with bfloat16 compute dtype,
    following the QLoRA setup (Dettmers et al., 2023).
    double_quant=True further compresses the quantization constants,
    saving ~0.4 bits per parameter with negligible quality loss.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_tokenizer(hf_id: str, trust_remote_code: bool, model_type: str, tokenizer_id: str = None):
    """
    Load tokenizer for a given HuggingFace model ID.

    Handles two edge cases:
    1. Missing pad token — LLaMA and GPT-2-based tokenizers (MDLM, Pythia)
       have no padding token by default. We set pad_token = eos_token,
       which is the standard fix and has no effect on PLL scoring since
       we process one statement at a time without batching.

    2. Padding side — AR models use left padding so that the EOS token
       always sits at the rightmost position during batched inference and
       finetuning. DLMs are bidirectional so padding side is irrelevant;
       we leave them at the default ('right').

    Args:
        hf_id:             HuggingFace model ID string (used as fallback)
        trust_remote_code: passed through to AutoTokenizer
        model_type:        'ar' or 'dlm' — determines padding side
        tokenizer_id:      override tokenizer source (e.g. 'gpt2' for MDLM
                           which does not ship its own tokenizer)

    Returns:
        tokenizer with pad_token and padding_side set correctly
    """
    # Use tokenizer_id override if provided (e.g. MDLM uses GPT-2 tokenizer)
    tok_source = tokenizer_id if tokenizer_id is not None else hf_id
    tokenizer = AutoTokenizer.from_pretrained(
        tok_source,
        trust_remote_code=trust_remote_code,
    )

    # Fix missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # AR models need left padding for batched causal inference
    if model_type == "ar":
        tokenizer.padding_side = "left"

    return tokenizer


def _load_base_model(config: dict, device: str, quantize: bool):
    """
    Internal dispatch — routes to the correct AutoModel class
    based on config['loader']: 'masked', 'causal', or 'auto'.

    Device handling:
        - Default is CPU for all models (small models on local Mac).
        - If quantize=True, device_map is forced to 'auto' regardless of
          device arg — bitsandbytes requires this for 4-bit loading on CUDA.
          In this case do NOT also call .to(device) after loading.

    Args:
        config:   entry from MODEL_REGISTRY for this model
        device:   target device string ('cpu' for local, 'cuda' for Colab)
        quantize: whether to apply 4-bit quantization config

    Returns:
        model in eval mode
    """
    hf_id = config["hf_id"]
    trust = config["trust_remote_code"]

    # MDLM's custom modeling files import flash_attn at the top level even
    # though the no_flashattn checkpoint never calls it at runtime.
    # flash_attn has no pre-built wheel for torch 2.11+cu128 (Colab June 2025)
    # and building from source fails. The fix: download the model files via
    # snapshot_download, patch the flash_attn import to a try/except in every
    # .py file, then load from the patched local path.
    # This runs once — on subsequent calls the cached files are already patched.
    if config.get("patch_flash_attn_import"):
        import re
        import glob
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(hf_id)

        for fpath in glob.glob(f"{local_path}/*.py"):
            src = open(fpath).read()
            if "flash_attn" in src and "try:" not in src:
                print(f"Patching flash_attn import in {fpath}")
                patched = re.sub(
                    r"(from flash_attn import [^
]+
)",
                    (
                        "try:
"
                        "    \1"
                        "except ImportError:
"
                        "    pass  # flash_attn not available — no_flashattn checkpoint
"
                    ),
                    src,
                )
                open(fpath, "w").write(patched)

        # Load from patched local path instead of HF hub
        hf_id = local_path

    # bitsandbytes 4-bit requires device_map='auto' — overrides user device
    if quantize:
        quant_cfg = _get_quantization_config()
        kwargs = {
            "quantization_config": quant_cfg,
            "device_map": "auto",
            "trust_remote_code": trust,
        }
    else:
        kwargs = {
            "trust_remote_code": trust,
        }

    # Dispatch to correct AutoModel class
    loader = config["loader"]
    if loader == "masked":
        model = AutoModelForMaskedLM.from_pretrained(hf_id, **kwargs)
    elif loader == "causal":
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    elif loader == "auto":
        model = AutoModel.from_pretrained(hf_id, **kwargs)
    else:
        raise ValueError(f"Unknown loader type '{loader}' in MODEL_REGISTRY.")

    # Move to device only when not quantized
    # (quantized models are already placed by device_map='auto')
    if not quantize:
        model = model.to(device)

    return model.eval()