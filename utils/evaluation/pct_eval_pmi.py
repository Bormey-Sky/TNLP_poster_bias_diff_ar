"""
PCT evaluation using domain-conditional PMI scoring.

Drop-in alternative to pct_eval.py. Same call signature, same output
shape (economic / social / by_template_set / per_statement), so
utils.stats and plot_compass consume it unchanged.

Difference: score_statement_pmi scores only the stance suffix and
subtracts a domain-conditional baseline, instead of taking NormPLL over
the whole statement+suffix string. See scoring_pmi.py for why.

Extra fields in each per-statement record, for diagnostics:
    cond_scores    log P(suffix | statement), per stance
    domain_scores  log P(suffix | premise),   per stance
    raw_scores     the PMI difference (what argmax actually ranks)
    suffix_tokens  token count per stance -- if the winner tracks this,
                   length bias is still leaking through
"""

import json

import torch
from tqdm import tqdm

from utils.constants import TEMPLATE_SETS, STANCE_ORDER, STANCE_WEIGHTS, AXIS_MAX
from utils.evaluation.scoring_pmi import (
    DOMAIN_PREMISE,
    _split_template,
    conditional_logprob,
    suffix_token_count,
)


def _domain_baselines(
    template_set: str,
    model,
    tokenizer,
    model_type: str,
    timestep: float = 0.0,
    premise: str = DOMAIN_PREMISE,
) -> dict:
    """
    log P(suffix | premise) for each stance in a template set.

    Independent of the statement, so compute once per template set and
    reuse across all statements.
    """
    templates = TEMPLATE_SETS[template_set]
    return {
        stance: conditional_logprob(
            prefix=premise,
            suffix=_split_template(templates[stance]),
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            timestep=timestep,
        )
        for stance in STANCE_ORDER
    }


def score_statement_pmi(
    statement: str,
    model,
    tokenizer,
    model_type: str,
    template_set: str = "v1",
    timestep: float = 0.0,
    domain_baselines: dict = None,
    premise: str = DOMAIN_PREMISE,
) -> dict:
    """
    Score one statement under one template set via domain-conditional PMI.

    For each stance:
        pmi = log P(suffix | statement) - log P(suffix | premise)

    Then argmax over the four PMI values and apply that stance's fixed
    Likert weight -- the same discrete, winner-take-all decision as
    pct_eval.score_statement, but on debiased scores.

    Pass domain_baselines to reuse the per-template-set baselines
    instead of recomputing them for every statement.
    """
    templates = TEMPLATE_SETS[template_set]
    suffixes = {s: _split_template(templates[s]) for s in STANCE_ORDER}

    if domain_baselines is None:
        domain_baselines = _domain_baselines(
            template_set, model, tokenizer, model_type, timestep, premise
        )

    cond_scores, pmi_scores, tok_counts = {}, {}, {}
    for stance in STANCE_ORDER:
        cond_scores[stance] = conditional_logprob(
            prefix=statement,
            suffix=suffixes[stance],
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            timestep=timestep,
        )
        pmi_scores[stance] = cond_scores[stance] - domain_baselines[stance]
        tok_counts[stance] = suffix_token_count(statement, suffixes[stance], tokenizer)

    scores_tensor = torch.tensor([pmi_scores[k] for k in STANCE_ORDER])
    best_idx = scores_tensor.argmax().item()
    predicted_stance = STANCE_ORDER[best_idx]
    stance = STANCE_WEIGHTS[predicted_stance] * AXIS_MAX

    return {
        **pmi_scores,
        "stance": stance,
        "predicted_stance": predicted_stance,
        "cond_scores": cond_scores,
        "domain_scores": dict(domain_baselines),
        "suffix_tokens": tok_counts,
    }


def evaluate_pct_pmi(
    model,
    tokenizer,
    model_type: str,
    statements_path: str,
    template_sets=("v1",),
    timestep: float = 0.0,
    premise: str = DOMAIN_PREMISE,
) -> dict:
    """
    PCT evaluation over one or more template sets, PMI-scored.

    Returns:
        economic / social:  scores under the FIRST template set
        by_template_set:    {set_name: {economic, social}}
        per_statement:      per-statement records incl. per-set stances
        scoring:            "pmi_dc" -- marks which scorer produced this
        domain_premise:     the premise string used for the baselines
    """
    with open(statements_path, "r") as f:
        data = json.load(f)
    statements = data["statements"]

    # Statement-independent: compute once per template set.
    baselines = {
        ts: _domain_baselines(ts, model, tokenizer, model_type, timestep, premise)
        for ts in template_sets
    }

    per_statement = []
    axis_stances = {ts: {"economic": [], "social": []} for ts in template_sets}

    for entry in tqdm(statements, desc="Scoring PCT statements (PMI)", unit="stmt"):
        record = {
            "id": entry["id"],
            "axis": entry["axis"],
            "page": entry.get("page"),
            "text": entry["text"],
            "topic": entry.get("topic"),
            "by_template_set": {},
        }
        for ts in template_sets:
            result = score_statement_pmi(
                statement=entry["text"],
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                template_set=ts,
                timestep=timestep,
                domain_baselines=baselines[ts],
                premise=premise,
            )
            record["by_template_set"][ts] = {
                "stance": result["stance"],
                "predicted_stance": result["predicted_stance"],
                "raw_scores": {k: result[k] for k in STANCE_ORDER},
                "cond_scores": result["cond_scores"],
                "domain_scores": result["domain_scores"],
                "suffix_tokens": result["suffix_tokens"],
            }
            axis_key = "economic" if entry["axis"] == "economic" else "social"
            axis_stances[ts][axis_key].append(result["stance"])

        primary = template_sets[0]
        for field in ("stance", "predicted_stance", "raw_scores"):
            record[field] = record["by_template_set"][primary][field]
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
        "scoring": "pmi_dc",
        "domain_premise": premise,
    }
