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
    method: str = "argmax",
) -> dict:
    """
    Score one statement under one template set.

    method="argmax" (default): pick the single stance with the highest
        NormPLL and use its fixed weight directly. This is the discrete,
        winner-take-all decision — matches the decision structure in
        Feng et al. (2023), which never blends across options.
    method="softmax": legacy behavior — softmax the 4 scores into a
        probability distribution and take a weighted expectation over
        all four. Kept only for the argmax-vs-softmax robustness check;
        not backed by any cited method, do not use for headline numbers.
    """
    templates = TEMPLATE_SETS[template_set]

    raw_scores = {}
    for stance_key in STANCE_ORDER:
        framed = templates[stance_key].format(statement=statement)
        raw_scores[stance_key] = compute_pll(
            framed, model, tokenizer, model_type, timestep=timestep
        )

    scores_tensor = torch.tensor([raw_scores[k] for k in STANCE_ORDER])

    if method == "argmax":
        best_idx = scores_tensor.argmax().item()
        predicted_stance = STANCE_ORDER[best_idx]
        stance = STANCE_WEIGHTS[predicted_stance] * AXIS_MAX
    elif method == "softmax":
        probs = torch.softmax(scores_tensor, dim=0)
        weights = torch.tensor([STANCE_WEIGHTS[k] for k in STANCE_ORDER])
        stance = (probs * weights * AXIS_MAX).sum().item()
        predicted_stance = STANCE_ORDER[scores_tensor.argmax().item()]
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'argmax' or 'softmax'.")

    return {
        **raw_scores,
        "stance": stance,
        "predicted_stance": predicted_stance,
        "method": method,
    }


def evaluate_pct(
    model,
    tokenizer,
    model_type: str,
    statements_path: str,
    template_sets=("v1",),
    timestep: float = 0.0,
    method: str = "argmax",
) -> dict:
    """
    PCT evaluation over one or more template sets.

    method: "argmax" (default, headline) or "softmax" (legacy, for the
        robustness comparison only — see score_statement docstring).

    Returns:
        economic / social:  scores under the FIRST template set (headline
                             numbers stay comparable across runs)
        by_template_set:    {set_name: {economic, social}}
        per_statement:      per-statement records incl. per-set stances
        method:             which scoring method produced this result
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
                method=method,
            )
            record["by_template_set"][ts] = {
                "stance": result["stance"],
                "predicted_stance": result["predicted_stance"],
                "raw_scores": {k: result[k] for k in STANCE_ORDER},
            }
            axis_key = "economic" if entry["axis"] == "economic" else "social"
            # axis_stances[ts][axis_key].append(result["stance"])
            axis_stances[ts][axis_key].append(entry.get("polarity", 1) * result["stance"])

        primary = template_sets[0]
        record["stance"] = record["by_template_set"][primary]["stance"]
        record["predicted_stance"] = record["by_template_set"][primary]["predicted_stance"]
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
        "method": method,
    }