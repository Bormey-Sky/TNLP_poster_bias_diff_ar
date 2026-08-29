import json

import torch
from tqdm import tqdm

from utils.constants import TEMPLATE_SETS, STANCE_ORDER, STANCE_WEIGHTS, AXIS_MAX
from utils.evaluation.scoring import compute_pll


def score_statement(
    statement: str,
    model,
    tokenizer,
    model_type: str,
    template_set: str = "v1",
    timestep: float = 0.0,
) -> dict:
    """Score one statement under one template set."""
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
    Returns:
        economic / social:  scores under the FIRST template set (headline
                             numbers stay comparable across runs)
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