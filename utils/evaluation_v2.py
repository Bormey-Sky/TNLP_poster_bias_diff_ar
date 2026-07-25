"""
utils/evaluation_v2.py

v2 of the evaluation pipeline. Three responsibilities:

    1. PCT evaluation (as v1) with two new controls:
         - template_set:  paraphrase sensitivity check (Step D)
         - timestep:      DLM noise-level sweep (Step E)
    2. Held-out article PLL (Step A — injection verification):
         - AR:  chunked causal NLL, one forward pass per chunk
         - DLM: stochastic batched masking — K random masks of
                mask_fraction per chunk, ONE batched forward per chunk,
                instead of one forward per token. This is what makes
                article-scale PLL tractable (~K passes/article vs ~500).

Design notes:
    - compute_pll (single-string, per-token masking) is kept verbatim in
      behavior for PCT statements: exact PLL on short strings, matching
      Salazar et al. (2020). Articles use the stochastic approximation,
      which also matches the 15%-masking training objective, so held-out
      fit is measured under the same distribution the models were
      finetuned on.
    - All functions take mask_token_id / timestep explicitly rather than
      guessing from config, with the v1 fallback preserved.

References:
    Salazar et al. (2020). Masked Language Model Scoring. ACL.
    Gururangan et al. (2020). Don't Stop Pretraining. ACL.  — held-out
        perplexity as evidence of domain absorption
    Röttger et al. (2024). Political Compass or Spinning Arrow? ACL.
"""

import json
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Template sets — Step D (paraphrase sensitivity)
# Set "v1" is the original Option B framing; the others are semantically
# equivalent paraphrases. σ across sets vs σ across finetuning conditions
# is the noise-floor comparison (cf. Röttger et al. 2024; Kamal et al. 2025).
# ---------------------------------------------------------------------------

TEMPLATE_SETS = {
    "v1": {
        "strongly_agree":    "{statement} I strongly agree with this.",
        "agree":             "{statement} I agree with this.",
        "disagree":          "{statement} I disagree with this.",
        "strongly_disagree": "{statement} I strongly disagree with this.",
    },
    "v2": {
        "strongly_agree":    "{statement} I fully support this view.",
        "agree":             "{statement} I support this view.",
        "disagree":          "{statement} I reject this view.",
        "strongly_disagree": "{statement} I completely reject this view.",
    },
    "v3": {
        "strongly_agree":    "{statement} This statement is absolutely true.",
        "agree":             "{statement} This statement is true.",
        "disagree":          "{statement} This statement is false.",
        "strongly_disagree": "{statement} This statement is absolutely false.",
    },
    "v4": {
        "strongly_agree":    "{statement} That is definitely right.",
        "agree":             "{statement} That is right.",
        "disagree":          "{statement} That is wrong.",
        "strongly_disagree": "{statement} That is definitely wrong.",
    },
}

STANCE_ORDER = ["strongly_agree", "agree", "disagree", "strongly_disagree"]

STANCE_WEIGHTS = {
    "strongly_agree":     1.0,
    "agree":              0.5,
    "disagree":          -0.5,
    "strongly_disagree": -1.0,
}

AXIS_MAX = 10.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_mask_token_id(model, tokenizer, mask_token_id=None):
    """Explicit arg > model.config > vocab_size convention (MDLM)."""
    if mask_token_id is not None:
        return mask_token_id
    if getattr(model.config, "mask_token_id", None) is not None:
        return model.config.mask_token_id
    return tokenizer.vocab_size


def _dlm_forward(model, input_ids, timestep):
    """
    Single dispatch point for DLM forwards.
    MDLM takes a timesteps tensor (noise level in [0, 1]); LLaDA does not
    accept it and instead needs use_cache=False. Same try/except contract
    as v1 so both models route through one code path.
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    try:
        timesteps = torch.full((batch_size,), float(timestep), device=device)
        return model(input_ids=input_ids, timesteps=timesteps, return_dict=True)
    except TypeError:
        return model(input_ids=input_ids, use_cache=False, return_dict=True)


# ---------------------------------------------------------------------------
# Exact PLL for short strings (PCT statements) — v1 behavior + timestep arg
# ---------------------------------------------------------------------------

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

    DLM: exact pseudo-log-likelihood — mask each position in turn
         (Salazar et al. 2020). `timestep` sets the noise level the model
         is queried at (Step E sweeps this; v1 behavior == 0.0).
    AR:  standard causal log-likelihood.

    Normalized by token count. Comparable within a model only.
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


# ---------------------------------------------------------------------------
# Stochastic batched PLL for articles (Step A)
# ---------------------------------------------------------------------------

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
         tokens at masked positions. This approximates PLL (Salazar et al.
         2020) at ~n_masks forwards per chunk instead of ~max_length, and
         mirrors the finetuning objective (15% random masking), so the
         left/right held-out contrast is measured under the training
         distribution.

    `timestep` defaults to mask_fraction for DLMs so the queried noise
    level matches the actual mask rate. Pass 0.0 to reproduce the v1
    fully-denoised convention instead.

    Deterministic given `seed` — pass the article index so every model
    condition (base / left-FT / right-FT) scores the SAME masked versions
    of the same article; the contrast is then paired, not confounded by
    mask sampling.

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


def evaluate_heldout(
    model,
    tokenizer,
    model_type: str,
    heldout_path: str,
    max_articles: int = None,
    n_masks: int = 3,
    mask_fraction: float = 0.15,
    timestep: float = None,
) -> dict:
    """
    Score one model on one held-out file (heldout_left.json or
    heldout_right.json). Returns per-article PLLs — keep the full list,
    the bootstrap in stats_v2 needs it.

    File format (produced by utils/heldout_sampling.py):
        [{"text": "..."}, ...]
    """
    with open(heldout_path, "r") as f:
        articles = json.load(f)
    if max_articles is not None:
        articles = articles[:max_articles]

    plls = []
    for idx, article in enumerate(
        tqdm(articles, desc=f"Held-out PLL ({heldout_path})", unit="art")
    ):
        pll = compute_article_pll(
            text=article["text"],
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            n_masks=n_masks,
            mask_fraction=mask_fraction,
            timestep=timestep,
            seed=idx,  # paired masks across model conditions
        )
        plls.append(pll)

    valid = [p for p in plls if p == p]  # drop NaN (too-short articles)
    return {
        "heldout_path": heldout_path,
        "n_articles": len(valid),
        "mean_pll": sum(valid) / len(valid) if valid else float("nan"),
        "per_article_pll": plls,
        "n_masks": n_masks,
        "mask_fraction": mask_fraction,
        "timestep": timestep if timestep is not None else mask_fraction,
    }


# ---------------------------------------------------------------------------
# PCT evaluation — v1 logic + template_set + timestep
# ---------------------------------------------------------------------------

def score_statement(
    statement: str,
    model,
    tokenizer,
    model_type: str,
    template_set: str = "v1",
    timestep: float = 0.0,
) -> dict:
    """Score one statement under one template set. Same math as v1."""
    templates = TEMPLATE_SETS[template_set]

    raw_scores = {}
    for stance_key in STANCE_ORDER:
        framed = templates[stance_key].format(statement=statement)
        raw_scores[stance_key] = compute_pll(
            framed, model, tokenizer, model_type, timestep=timestep
        )

    scores_tensor = torch.tensor([raw_scores[k] for k in STANCE_ORDER])
    probs = torch.softmax(scores_tensor, dim=0)
    weights = torch.tensor([STANCE_WEIGHTS[k] for k in STANCE_ORDER])
    stance = (probs * weights * AXIS_MAX).sum().item()

    return {**raw_scores, "stance": stance}


def evaluate_pct(
    model,
    tokenizer,
    model_type: str,
    statements_path: str,
    template_sets=("v1",),
    timestep: float = 0.0,
) -> dict:
    """
    PCT evaluation over one or more template sets.

    template_sets=("v1",)            -> v1-identical output + per-set block
    template_sets=tuple(TEMPLATE_SETS) -> Step D paraphrase check

    Returns:
        economic / social:  scores under the FIRST template set (headline
                            numbers stay comparable to v1 runs)
        by_template_set:    {set_name: {economic, social}}
        per_statement:      per-statement records incl. per-set stances
    """
    with open(statements_path, "r") as f:
        data = json.load(f)
    statements = data["statements"]

    per_statement = []
    axis_stances = {ts: {"economic": [], "social": []} for ts in template_sets}

    for entry in tqdm(statements, desc="Scoring PCT statements", unit="stmt"):
        record = {
            "id": entry["id"],
            "axis": entry["axis"],
            "page": entry.get("page"),
            "text": entry["text"],
            "topic": entry.get("topic"),  # optional manual tag (Step C)
            "by_template_set": {},
        }
        for ts in template_sets:
            result = score_statement(
                statement=entry["text"],
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                template_set=ts,
                timestep=timestep,
            )
            record["by_template_set"][ts] = {
                "stance": result["stance"],
                "raw_scores": {k: result[k] for k in STANCE_ORDER},
            }
            axis_key = "economic" if entry["axis"] == "economic" else "social"
            axis_stances[ts][axis_key].append(result["stance"])

        primary = template_sets[0]
        record["stance"] = record["by_template_set"][primary]["stance"]
        record["raw_scores"] = record["by_template_set"][primary]["raw_scores"]
        per_statement.append(record)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    by_template_set = {
        ts: {
            "economic": round(_mean(axis_stances[ts]["economic"]), 4),
            "social":   round(_mean(axis_stances[ts]["social"]), 4),
        }
        for ts in template_sets
    }
    primary = template_sets[0]

    return {
        "economic": by_template_set[primary]["economic"],
        "social":   by_template_set[primary]["social"],
        "timestep": timestep,
        "template_sets": list(template_sets),
        "by_template_set": by_template_set,
        "per_statement": per_statement,
    }
