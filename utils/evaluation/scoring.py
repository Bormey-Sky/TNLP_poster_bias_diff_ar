import random

import torch
import torch.nn.functional as F



def _resolve_mask_token_id(model, tokenizer, mask_token_id=None):
    """Explicit arg > model.config > vocab_size convention (MDLM)."""
    if mask_token_id is not None:
        return mask_token_id
    if getattr(model.config, "mask_token_id", None) is not None:
        return model.config.mask_token_id
    return tokenizer.vocab_size

def _dlm_forward(model, input_ids, timestep):
    batch_size = input_ids.shape[0]
    device = input_ids.device
    try:
        timesteps = torch.full((batch_size,), float(timestep), device=device)
        return model(input_ids=input_ids, timesteps=timesteps, return_dict=True)
    except TypeError:
        return model(input_ids=input_ids, use_cache=False, return_dict=True)


# Exact PLL for PCT statements
def compute_pll(
    text: str,
    model,
    tokenizer,
    model_type: str,
    timestep: float = 0.0,
    mask_token_id=None,
) -> float:
    """
    NormPLL of a single string.
    DLM: exact pseudo-log-likelihood
    AR:  standard causal log-likelihood
    """
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    seq_len = input_ids.shape[1]

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    if model_type == "dlm":
        mask_id = _resolve_mask_token_id(model, tokenizer, mask_token_id)
        log_prob_sum = 0.0
        with torch.no_grad():
            for i in range(seq_len):
                masked_input = input_ids.clone()
                masked_input[0, i] = mask_id
                output = _dlm_forward(model, masked_input, timestep)
                log_probs = F.log_softmax(output.logits[0, i, :], dim=-1)
                log_prob_sum += log_probs[input_ids[0, i].item()].item()
        return log_prob_sum / seq_len

    elif model_type == "ar":
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
            shift_logits = logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[torch.arange(seq_len - 1), shift_labels]
            log_prob_sum = token_log_probs.sum().item()
        return log_prob_sum / seq_len

    raise ValueError(f"Unknown model_type '{model_type}'. Expected 'dlm' or 'ar'.")


# Stochastic batched PLL for articles 
def compute_article_pll(
    text: str,
    model,
    tokenizer,
    model_type: str,
    max_length: int = 512,
    n_masks: int = 3,
    mask_fraction: float = 0.15,
    timestep: float = None,
    mask_token_id=None,
    seed: int = 0,
) -> float:
    """
    NormPLL of a full article.

    AR:  chunk to max_length, exact causal NLL per chunk, token-weighted
         average across chunks.
    DLM: chunk to max_length; per chunk draw n_masks independent random
         masks each covering mask_fraction of positions, stack them into
         ONE batched forward pass, and average log-probs of the original
         tokens at masked positions. 

    Returns mean log-prob per scored token.
    """
    encoding = tokenizer(text, return_tensors="pt", truncation=False)
    all_ids = encoding["input_ids"][0]
    device = next(model.parameters()).device

    chunks = [all_ids[i:i + max_length] for i in range(0, len(all_ids), max_length)]
    # Drop trailing fragments too short to mask meaningfully
    chunks = [c for c in chunks if len(c) >= 32]
    if not chunks:
        return float("nan")

    total_logprob, total_tokens = 0.0, 0

    if model_type == "ar":
        with torch.no_grad():
            for chunk in chunks:
                input_ids = chunk.unsqueeze(0).to(device)
                seq_len = input_ids.shape[1]
                logits = model(input_ids=input_ids).logits
                log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
                token_lp = log_probs[torch.arange(seq_len - 1), input_ids[0, 1:]]
                total_logprob += token_lp.sum().item()
                total_tokens += seq_len - 1
        return total_logprob / total_tokens

    elif model_type == "dlm":
        mask_id = _resolve_mask_token_id(model, tokenizer, mask_token_id)
        t = mask_fraction if timestep is None else timestep
        rng = random.Random(seed)

        with torch.no_grad():
            for chunk in chunks:
                seq_len = len(chunk)
                n_mask = max(1, int(round(mask_fraction * seq_len)))

                # Build n_masks masked copies -> one batched forward
                batch = chunk.unsqueeze(0).repeat(n_masks, 1).to(device)
                mask_positions = []
                for k in range(n_masks):
                    pos = rng.sample(range(seq_len), n_mask)
                    mask_positions.append(pos)
                    batch[k, pos] = mask_id

                output = _dlm_forward(model, batch, t)
                log_probs = F.log_softmax(output.logits, dim=-1)  # [n_masks, seq, vocab]

                chunk_ids = chunk.to(device)
                for k, pos in enumerate(mask_positions):
                    pos_t = torch.tensor(pos, device=device)
                    lp = log_probs[k, pos_t, chunk_ids[pos_t]]
                    total_logprob += lp.sum().item()
                    total_tokens += len(pos)

        return total_logprob / total_tokens

    raise ValueError(f"Unknown model_type '{model_type}'. Expected 'dlm' or 'ar'.")