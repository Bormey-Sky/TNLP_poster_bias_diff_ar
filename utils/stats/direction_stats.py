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
    """
    Per-statement stances, POLARITY-SIGNED.

    evaluate_pct applies polarity when it aggregates the axis scores, but
    stores `stance` unsigned in each per-statement record. Without the
    multiplication below these stats would describe mean agreeableness
    rather than left-right position, and would not reconcile with the
    economic/social scores in the same file.

    polarity = +1 when agreement moves the score right (economic) or
    authoritarian (social); -1 when it moves left / libertarian.
    Falls back to +1 for results produced with the unsigned statements
    file -- see the reconciliation warning in directional_effect.
    """
    return {
        rec["id"]: {"stance": rec.get("polarity", 1) * rec["stance"],
                    "axis": rec["axis"],
                    "topic": rec.get("topic"), "text": rec["text"]}
        for rec in pct_json["per_statement"]
    }


def _check_reconciles(pct_json, signed, label):
    """Warn if recomputed axis means disagree with the file's aggregates."""
    for axis in ("economic", "social"):
        vals = [v["stance"] for v in signed.values() if v["axis"] == axis]
        if not vals or axis not in pct_json:
            continue
        recomputed = round(sum(vals) / len(vals), 4)
        if abs(recomputed - pct_json[axis]) > 1e-3:
            print(f"  WARNING [{label}]: recomputed {axis}={recomputed} but "
                  f"file reports {pct_json[axis]}. Result likely predates the "
                  f"polarity fix -- rerun evaluate with "
                  f"pct_statements_polarity.json.")


def directional_effect(results_dir, model_name):
    left = _load_pct(results_dir, model_name, "left")
    right = _load_pct(results_dir, model_name, "right")
    base = _load_pct(results_dir, model_name, "base")
    if left is None or right is None:
        return {"model": model_name, "error": "missing left/right PCT results"}

    sl, sr = _stance_by_id(left), _stance_by_id(right)
    sb = _stance_by_id(base) if base else {}
    _check_reconciles(left, sl, f"{model_name} left")
    _check_reconciles(right, sr, f"{model_name} right")
    if base:
        _check_reconciles(base, sb, f"{model_name} base")
    common_ids = sorted(set(sl) & set(sr))

    out = {"model": model_name, "n_statements": len(common_ids), "axes": {}}
    per_statement_deltas = []

    for axis in ("economic", "social"):
        ids = [i for i in common_ids if sl[i]["axis"] == axis]
        deltas = np.array([sl[i]["stance"] - sr[i]["stance"] for i in ids])
        mean, lo, hi = bootstrap_ci(deltas)
        p = paired_permutation_pvalue(deltas)

        axis_block = {
            "D_left_minus_right": {
                "mean": mean, "ci95": [lo, hi], "p_perm": p,
                "n": len(ids), "n_changed": int((deltas != 0).sum()),
            },
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
                    "n": len(shift), "n_changed": int((shift != 0).sum()),
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