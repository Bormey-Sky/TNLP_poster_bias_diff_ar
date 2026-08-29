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
}