"""
Directional effect (Step B, per model, per axis):
    D = axis(leftFT) - axis(rightFT)
    PCT recovers direction  <=>  D < 0 (economic axis) with CI excluding 0.
    Bootstrap resamples the 62 statements (paired: same statement under
    both conditions). Permutation test flips condition labels per
    statement.

Per-statement deltas (Step C):
    delta_s = stance_s(leftFT) - stance_s(rightFT), exported with topic
    tags for the sorted-bar poster figure.
"""

import numpy as np

from utils.stats.bootstrap import bootstrap_ci, paired_permutation_pvalue
from utils.stats.pct_loading import _load_pct


def _stance_by_id(pct_json):
    return {
        rec["id"]: {"stance": rec["stance"], "axis": rec["axis"],
                    "topic": rec.get("topic"), "text": rec["text"]}
        for rec in pct_json["per_statement"]
    }


def directional_effect(results_dir, model_name):
    left = _load_pct(results_dir, model_name, "left")
    right = _load_pct(results_dir, model_name, "right")
    base = _load_pct(results_dir, model_name, "base")
    if left is None or right is None:
        return {"model": model_name, "error": "missing left/right PCT results"}

    sl, sr = _stance_by_id(left), _stance_by_id(right)
    sb = _stance_by_id(base) if base else {}
    common_ids = sorted(set(sl) & set(sr))

    out = {"model": model_name, "n_statements": len(common_ids), "axes": {}}
    per_statement_deltas = []

    for axis in ("economic", "social"):
        ids = [i for i in common_ids if sl[i]["axis"] == axis]
        deltas = np.array([sl[i]["stance"] - sr[i]["stance"] for i in ids])
        mean, lo, hi = bootstrap_ci(deltas)
        p = paired_permutation_pvalue(deltas)

        axis_block = {
            "D_left_minus_right": {"mean": mean, "ci95": [lo, hi], "p_perm": p},
            "direction_recovered": bool(hi < 0),  # left-FT more negative
        }
        if sb:
            for cond, s in (("left", sl), ("right", sr)):
                shift = np.array([s[i]["stance"] - sb[i]["stance"]
                                  for i in ids if i in sb])
                m, l, h = bootstrap_ci(shift)
                axis_block[f"shift_from_base_{cond}FT"] = {
                    "mean": m, "ci95": [l, h],
                    "p_perm": paired_permutation_pvalue(shift),
                }
        out["axes"][axis] = axis_block

        for i in ids:
            per_statement_deltas.append({
                "id": i, "axis": axis,
                "topic": sl[i]["topic"],
                "text": sl[i]["text"][:80],
                "delta_left_minus_right": float(sl[i]["stance"] - sr[i]["stance"]),
                "shift_left_from_base": float(sl[i]["stance"] - sb[i]["stance"]) if i in sb else None,
                "shift_right_from_base": float(sr[i]["stance"] - sb[i]["stance"]) if i in sb else None,
            })

    out["per_statement_deltas"] = sorted(
        per_statement_deltas, key=lambda r: r["delta_left_minus_right"]
    )
    return out