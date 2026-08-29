import numpy as np
from utils.stats.pct_loading import _load_pct


def paraphrase_decomposition(results_dir, model_name):
    conditions = {}
    for cond in ("base", "left", "right"):
        pct = _load_pct(results_dir, model_name, cond)
        if pct and "by_template_set" in pct and len(pct["by_template_set"]) > 1:
            conditions[cond] = pct["by_template_set"]
    if len(conditions) < 2:
        return {"model": model_name,
                "error": "need >=2 conditions evaluated with --templates all"}

    template_sets = sorted(next(iter(conditions.values())).keys())
    out = {"model": model_name, "template_sets": template_sets, "axes": {}}

    for axis in ("economic", "social"):
        scores = {
            c: np.array([conditions[c][ts][axis] for ts in template_sets])
            for c in conditions
        }
        sigma_para = float(np.mean([scores[c].std(ddof=1) for c in scores]))
        by_ts = np.array([[scores[c][k] for c in scores]
                          for k in range(len(template_sets))])
        sigma_cond = float(np.mean(by_ts.std(axis=1, ddof=1)))
        out["axes"][axis] = {
            "sigma_paraphrase": sigma_para,
            "sigma_condition": sigma_cond,
            "ratio_paraphrase_over_condition":
                sigma_para / sigma_cond if sigma_cond > 0 else float("inf"),
            "scores": {c: dict(zip(template_sets, map(float, scores[c])))
                       for c in scores},
        }
    return out