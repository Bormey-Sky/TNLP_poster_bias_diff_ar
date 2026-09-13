"""
Domain-conditional PMI scoring for PCT stance templates.

Companion to scoring.py. That module scores the WHOLE string
(statement + suffix) and divides by total length, which has two
confounds when comparing stance templates that share a statement:

  1. Length dilution. The statement contributes an identical large
     negative logprob S to all four candidates, while the divisor
     (n_statement + m_suffix) differs. Because the boilerplate suffix
     is far more predictable per token than the statement, each extra
     suffix token mechanically RAISES NormPLL. In every template set
     in constants.py the two "strongly" variants are exactly one GPT-2
     token longer than their plain counterparts, and the size of that
     artifact (~0.03-0.07) matches the observed decision margins.

  2. Surface-form competition. Some phrasings are simply more probable
     a priori, independent of the statement.

This module removes both:

  * Only the continuation is scored -- the shared statement prefix is
    conditioned on but never scored, so S drops out entirely.
  * A domain-conditional baseline log P(suffix | premise) is subtracted,
    cancelling each template's intrinsic likelihood.

    score_i = log P(suffix_i | statement) - log P(suffix_i | premise)

Both terms cover the same suffix tokens, so no length normalisation is
applied or needed.

Reference:
    Holtzman, West, Shwartz, Choi, Zettlemoyer (2021).
    "Surface Form Competition: Why the Highest Probability Answer
    Isn't Always Right." EMNLP 2021.  (Domain Conditional PMI)

    The same correction appears as "unconditional likelihood
    normalized" scoring in EleutherAI's lm-evaluation-harness.
"""

import torch
import torch.nn.functional as F

from utils.evaluation.scoring import _resolve_mask_token_id, _dlm_forward


# Domain premise: a minimal, politically neutral stand-in for the
# statement slot. Its only job is to give the suffix the same syntactic
# context without any stance-bearing content.
DOMAIN_PREMISE = "N/A"


def _split_template(template: str) -> str:
    """Return the suffix part of a '{statement} SUFFIX' template."""
    if "{statement}" not in template:
        raise ValueError(f"Template has no {{statement}} slot: {template!r}")
    return template.split("{statement}", 1)[1]


def _prefix_len(prefix: str, full: str, tokenizer) -> tuple:
    """
    Token count of `prefix` within `full`, plus the full input_ids.

    Verifies the prefix tokenisation is a true prefix of the full
    tokenisation (BPE can merge across a boundary). Raises if not, so a
    silent off-by-one never corrupts scores.
    """
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full, return_tensors="pt")["input_ids"]
    n_prefix = prefix_ids.shape[1]

    if not torch.equal(prefix_ids[0], full_ids[0, :n_prefix]):
        raise ValueError(
            "BPE merged across the statement/suffix boundary; suffix "
            f"token span is ambiguous for prefix={prefix!r}"
        )
    if full_ids.shape[1] <= n_prefix:
        raise ValueError(f"Empty suffix for prefix={prefix!r}")

    return n_prefix, full_ids


def conditional_logprob(
    prefix: str,
    suffix: str,
    model,
    tokenizer,
    model_type: str,
    timestep: float = 0.0,
    mask_token_id=None,
) -> float:
    """
    Summed log P(suffix | prefix) over suffix tokens only.

    AR:  causal logprobs at the suffix positions.
    DLM: exact PLL -- each suffix position masked in turn, with the
         prefix and the remaining suffix visible as context.

    Not length-normalised: the PMI difference in score_statement_pmi
    cancels length, since both terms span the same tokens.
    """
    n_prefix, full_ids = _prefix_len(prefix, prefix + suffix, tokenizer)
    device = next(model.parameters()).device
    full_ids = full_ids.to(device)
    seq_len = full_ids.shape[1]

    if model_type == "ar":
        with torch.no_grad():
            logits = model(input_ids=full_ids).logits
            # logits[j] predicts token j+1, so token j is scored by logits[j-1]
            shift_logits = logits[0, n_prefix - 1:seq_len - 1, :]
            targets = full_ids[0, n_prefix:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_lp = log_probs[torch.arange(targets.shape[0]), targets]
            return token_lp.sum().item()

    elif model_type == "dlm":
        mask_id = _resolve_mask_token_id(model, tokenizer, mask_token_id)
        total = 0.0
        with torch.no_grad():
            for j in range(n_prefix, seq_len):
                masked = full_ids.clone()
                masked[0, j] = mask_id
                output = _dlm_forward(model, masked, timestep)
                log_probs = F.log_softmax(output.logits[0, j, :], dim=-1)
                total += log_probs[full_ids[0, j].item()].item()
        return total

    raise ValueError(f"Unknown model_type '{model_type}'. Expected 'dlm' or 'ar'.")


def suffix_token_count(prefix: str, suffix: str, tokenizer) -> int:
    """Number of tokens the suffix occupies after `prefix`. For diagnostics."""
    n_prefix, full_ids = _prefix_len(prefix, prefix + suffix, tokenizer)
    return full_ids.shape[1] - n_prefix
