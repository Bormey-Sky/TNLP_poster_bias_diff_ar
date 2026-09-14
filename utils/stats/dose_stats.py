"""
Dose contrast (per model family, per condition, per axis):
    Delta_dose = axis(cond, 5k) - axis(cond, 1k)

direction_stats compares left vs right WITHIN one corpus size. It cannot
express "more injected data moved the score further", because
run_all_stats treats `pythia_160m` and `pythia_160m_v2` as two separate
models. This module joins them.

Paired over statements, same as direction_stats: the same 62 items are
scored under both doses, so differences are taken per statement and then
bootstrapped.
"""

import numpy as np

from utils.stats.bootstrap import bootstrap_ci, paired_permutation_pvalue
from utils.stats.direction_stats import _stance_by_id
from utils.stats.pct_loading import _load_pct


def dose_effect(results_dir, model_low, model_high, conditions=("base", "left", "right")):
    """
    model_low / model_high: e.g. "pythia_160m" and "pythia_160m_v2".
    Reported as high minus low, so a positive value means the larger
    corpus pushed the axis score further positive (right / authoritarian).
    """
    out = {"model_low": model_low, "model_high": model_high, "conditions": {}}

    for cond in conditions:
        lo_json = _load_pct(results_dir, model_low, cond)
        hi_json = _load_pct(results_dir, model_high, cond)
        if lo_json is None or hi_json is None:
            out["conditions"][cond] = {"error": "missing one dose"}
            continue

        s_lo, s_hi = _stance_by_id(lo_json), _stance_by_id(hi_json)
        ids = sorted(set(s_lo) & set(s_hi))

        block = {}
        for axis in ("economic", "social"):
            axis_ids = [i for i in ids if s_lo[i]["axis"] == axis]
            if not axis_ids:
                continue
            diffs = np.array(
                [s_hi[i]["stance"] - s_lo[i]["stance"] for i in axis_ids],
                dtype=float,
            )
            mean, lo, hi = bootstrap_ci(diffs)
            block[axis] = {
                "delta_high_minus_low": {
                    "mean": mean, "ci95": [lo, hi],
                    "p_perm": paired_permutation_pvalue(diffs),
                    "n": len(axis_ids),
                    "n_changed": int((diffs != 0).sum()),
                },
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            }
        out["conditions"][cond] = block

    return out
