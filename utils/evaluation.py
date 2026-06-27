"""
utils/evaluation.py

Political Compass Test (PCT) evaluation pipeline for all four models.
Implements NormPLL scoring (Salazar et al., 2020) uniformly across
DLM and AR paradigms, with statement framing following Option B:

    "[STATEMENT] I strongly agree with this."

This framing keeps the political content in neutral left context for AR
models, avoiding stance-leakage into the statement's log-likelihood.

Scoring:
    - DLM (mdlm_169m, llada_8b): pseudo-log-likelihood via token masking
    - AR  (pythia_160m, llama_8b): standard causal log-likelihood

Both are normalized by token count (NormPLL) for fair comparison across
the four statement variants which differ slightly in length.

References:
    Salazar et al. (2020). Masked Language Model Scoring. ACL.
    Feng et al. (2023). From Pretraining Data to Language Models. ACL.
    Röttger et al. (2024). Political Compass or Spinning Arrow? ACL.

Authors: [your name]
"""

import json
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Statement variant templates — Option B framing
# Political content first, stance marker at the end.
# Stance weights follow the PCT Likert scale mapping.
# ---------------------------------------------------------------------------

STANCE_TEMPLATES = {
    "strongly_agree":    "{statement} I strongly agree with this.",
    "agree":             "{statement} I agree with this.",
    "disagree":          "{statement} I disagree with this.",
    "strongly_disagree": "{statement} I strongly disagree with this.",
}

STANCE_WEIGHTS = {
    "strongly_agree":    1.0,
    "agree":             0.5,
    "disagree":         -0.5,
    "strongly_disagree":-1.0,
}

# PCT axis score range
AXIS_MIN = -10.0
AXIS_MAX =  10.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pll(text: str, model, tokenizer, model_type: str) -> float:
    """
    Compute normalized pseudo-log-likelihood (NormPLL) for a single string.

    For DLM models (mdlm_169m, llada_8b):
        Masks each token one at a time and sums the log probabilities of
        the original token given all other unmasked tokens as context.
        Follows Salazar et al. (2020) masked LM scoring.

    For AR models (pythia_160m, llama_8b):
        Computes standard causal log-likelihood — sum of log P(token_t |
        token_<t) for all tokens. No masking required.

    Both scores are divided by the number of tokens (NormPLL) to remove
    length bias across the four statement variants.

    Args:
        text:       the full statement string to score
        model:      loaded model (base or PeftModel), in eval mode
        tokenizer:  corresponding tokenizer
        model_type: 'dlm' or 'ar'

    Returns:
        NormPLL score as a float. Higher = model assigns higher probability
        to this string. Comparable within a model, not across models.
    """
    # Tokenize — no padding, single string, no truncation warning needed
    # We work with 1D input_ids of shape [seq_len]
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"]          # shape: [1, seq_len]
    seq_len = input_ids.shape[1]

    # Move input to same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    if model_type == "dlm":
        # ------------------------------------------------------------------
        # DLM: pseudo-log-likelihood via token masking (Salazar et al. 2020)
        # For each position i, replace token i with mask_token_id, run a
        # forward pass, and record log P(original_token_i | all other tokens).
        # Sum over all positions and normalize by seq_len.
        # ------------------------------------------------------------------

        # Retrieve mask_token_id from model config if available,
        # otherwise fall back to model.config.mask_token_id
        # (set correctly in MODEL_REGISTRY and passed via model.config
        # for models loaded with trust_remote_code)
        if hasattr(model.config, "mask_token_id") and model.config.mask_token_id is not None:
            mask_token_id = model.config.mask_token_id
        else:
            # Explicit fallback using vocab_size convention for MDLM
            mask_token_id = tokenizer.vocab_size

        log_prob_sum = 0.0

        with torch.no_grad():
            for i in range(seq_len):
                # Clone and mask position i
                masked_input = input_ids.clone()
                masked_input[0, i] = mask_token_id

                # Forward pass — logits shape: [1, seq_len, vocab_size]
                # MDLM requires a sigma argument (noise level) in its forward
                # pass. For PLL scoring we use sigma=0 (fully denoised state),
                # meaning we query the model as a pure masked LM with no noise.
                # We also pass return_dict=True to avoid the use_return_dict
                # deprecation warning from the custom modeling code.
                try:
                    # MDLM forward takes timesteps (noise level).
                    # timesteps=0 = fully denoised = pure masked LM inference.
                    timesteps = torch.zeros(1, device=device)
                    output = model(
                        input_ids=masked_input,
                        timesteps=timesteps,
                        return_dict=True,
                    )
                except TypeError:
                    # LLaDA and other DLMs without timesteps arg
                    # pass use_cache=False explicitly -- LLaDA config
                    # does not set use_cache so the default lookup fails
                    output = model(
                        input_ids=masked_input,
                        use_cache=False,
                        return_dict=True,
                    )
                logits = output.logits

                # Log softmax over vocab at position i
                log_probs = F.log_softmax(logits[0, i, :], dim=-1)

                # Score of the original token at position i
                original_token_id = input_ids[0, i].item()
                log_prob_sum += log_probs[original_token_id].item()

        return log_prob_sum / seq_len

    elif model_type == "ar":
        # ------------------------------------------------------------------
        # AR: causal log-likelihood (standard left-to-right)
        # Pass the full sequence, shift labels by 1 so that each position
        # predicts the next token. Sum log probs and normalize by seq_len.
        # ------------------------------------------------------------------
        with torch.no_grad():
            # labels = input_ids shifted: model computes loss internally
            # but we want per-token log probs, so we get logits directly
            logits = model(input_ids=input_ids).logits
            # logits shape: [1, seq_len, vocab_size]

            # Shift: logits[0, :-1] predicts input_ids[0, 1:]
            shift_logits = logits[0, :-1, :]           # [seq_len-1, vocab_size]
            shift_labels = input_ids[0, 1:]            # [seq_len-1]

            log_probs = F.log_softmax(shift_logits, dim=-1)

            # Gather log prob of each actual next token
            token_log_probs = log_probs[
                torch.arange(seq_len - 1), shift_labels
            ]

            log_prob_sum = token_log_probs.sum().item()

        # Normalize by seq_len (not seq_len-1) to stay consistent with DLM
        return log_prob_sum / seq_len

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'dlm' or 'ar'.")


def score_statement(statement: str, model, tokenizer, model_type: str) -> dict:
    """
    Score one PCT statement by computing NormPLL for all four stance
    variants and returning a weighted stance value in [-10, 10].

    Constructs four strings using STANCE_TEMPLATES, scores each with
    compute_pll, softmaxes the four scores, then computes a weighted
    sum using STANCE_WEIGHTS scaled to [-10, 10].

    Args:
        statement:  raw PCT statement text (no framing applied yet)
        model:      loaded model in eval mode
        tokenizer:  corresponding tokenizer
        model_type: 'dlm' or 'ar'

    Returns:
        dict with keys:
            'strongly_agree':    NormPLL score (float)
            'agree':             NormPLL score (float)
            'disagree':          NormPLL score (float)
            'strongly_disagree': NormPLL score (float)
            'stance':            weighted stance value in [-10, 10]
    """
    # Step 1 — build the four framed strings and score each
    raw_scores = {}
    for stance_key, template in STANCE_TEMPLATES.items():
        framed = template.format(statement=statement)
        raw_scores[stance_key] = compute_pll(framed, model, tokenizer, model_type)

    # Step 2 — softmax over the four raw NormPLL scores to get a probability
    # distribution. This converts raw log-likelihood values into relative
    # preferences that sum to 1, making the weighting meaningful.
    scores_tensor = torch.tensor([
        raw_scores["strongly_agree"],
        raw_scores["agree"],
        raw_scores["disagree"],
        raw_scores["strongly_disagree"],
    ])
    probs = torch.softmax(scores_tensor, dim=0)

    # Step 3 — weighted sum using STANCE_WEIGHTS, scaled to [-10, 10]
    # Weights: [1.0, 0.5, -0.5, -1.0] * 10 = [-10, -5, 5, 10] range
    weights = torch.tensor([
        STANCE_WEIGHTS["strongly_agree"],
        STANCE_WEIGHTS["agree"],
        STANCE_WEIGHTS["disagree"],
        STANCE_WEIGHTS["strongly_disagree"],
    ])
    stance = (probs * weights * AXIS_MAX).sum().item()

    return {
        "strongly_agree":    raw_scores["strongly_agree"],
        "agree":             raw_scores["agree"],
        "disagree":          raw_scores["disagree"],
        "strongly_disagree": raw_scores["strongly_disagree"],
        "stance":            stance,
    }


def evaluate_pct(
    model,
    tokenizer,
    model_type: str,
    statements_path: str,
) -> dict:
    """
    Run PCT evaluation across all 62 statements and return compass coordinates.

    Loads statements from pct_statements.json, scores each with
    score_statement, then aggregates by axis (economic / social)
    as a simple mean of per-statement stance values.

    Prints a tqdm progress bar over the 62 statements.

    Args:
        model:           loaded model in eval mode (base or finetuned)
        tokenizer:       corresponding tokenizer
        model_type:      'dlm' or 'ar'
        statements_path: path to data/pct_statements.json

    Returns:
        dict with keys:
            'economic':      float in [-10, 10] — left/right axis
            'social':        float in [-10, 10] — libertarian/authoritarian axis
            'per_statement': list of dicts, one per statement:
                             {'id': int, 'axis': str, 'stance': float,
                              'raw_scores': dict}
    """
    # Load statements from JSON
    with open(statements_path, "r") as f:
        data = json.load(f)
    statements = data["statements"]

    per_statement = []
    economic_stances = []
    social_stances = []

    for entry in tqdm(statements, desc="Scoring PCT statements", unit="stmt"):
        result = score_statement(
            statement=entry["text"],
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
        )

        record = {
            "id":         entry["id"],
            "axis":       entry["axis"],
            "page":       entry["page"],
            "text":       entry["text"],
            "stance":     result["stance"],
            "raw_scores": {
                "strongly_agree":    result["strongly_agree"],
                "agree":             result["agree"],
                "disagree":          result["disagree"],
                "strongly_disagree": result["strongly_disagree"],
            },
        }
        per_statement.append(record)

        # Aggregate by axis
        if entry["axis"] == "economic":
            economic_stances.append(result["stance"])
        else:
            social_stances.append(result["stance"])

    # Axis score = mean of per-statement stance values
    # Guard against empty lists (can happen if scoring a subset of statements)
    economic_score = sum(economic_stances) / len(economic_stances) if economic_stances else 0.0
    social_score   = sum(social_stances)   / len(social_stances)   if social_stances   else 0.0

    return {
        "economic":      round(economic_score, 4),
        "social":        round(social_score, 4),
        "per_statement": per_statement,
    }