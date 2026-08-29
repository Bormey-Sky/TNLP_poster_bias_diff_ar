"""
Injection contrast (Step A, per model, per held-out side):
    I_left  = mean PLL_leftFT(heldout_left)  - mean PLL_rightFT(heldout_left)
    I_right = mean PLL_rightFT(heldout_right) - mean PLL_leftFT(heldout_right)
    Injection verified  <=>  both > 0 with bootstrap CI excluding 0.
"""

import json
import os
import numpy as np

from utils.stats.bootstrap import bootstrap_ci, paired_permutation_pvalue


def injection_contrast(results_dir, model_name):
    """
    Expects files written by --step eval_heldout:
        results/heldout/{model}_{condition}_{side}.json
    with condition in {base, left, right}, side in {left, right}.
    """
    def _load(condition, side):
        path = os.path.join(
            results_dir, "heldout", f"{model_name}_{condition}_{side}.json"
        )
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return np.asarray(json.load(f)["per_article_pll"], dtype=float)

    out = {"model": model_name}
    for side in ("left", "right"):
        own = _load(side, side)                                # matched FT
        other = _load("right" if side == "left" else "left", side)  # opposed FT
        base = _load("base", side)

        if own is None or other is None:
            out[f"heldout_{side}"] = "missing files"
            continue

        n = min(len(own), len(other))
        paired_diff = own[:n] - other[:n]  # >0 => matched FT fits better
        mean, lo, hi = bootstrap_ci(paired_diff)
        p = paired_permutation_pvalue(paired_diff)
        block = {
            "matched_minus_opposed_pll": {"mean": mean, "ci95": [lo, hi], "p_perm": p},
            "verified": bool(lo > 0),
        }
        if base is not None:
            nb = min(len(own), len(base))
            bmean, blo, bhi = bootstrap_ci(own[:nb] - base[:nb])
            block["matched_minus_base_pll"] = {"mean": bmean, "ci95": [blo, bhi]}
        out[f"heldout_{side}"] = block

    hl = out.get("heldout_left", {})
    hr = out.get("heldout_right", {})
    out["injection_verified"] = bool(
        isinstance(hl, dict) and hl.get("verified")
        and isinstance(hr, dict) and hr.get("verified")
    )
    return out