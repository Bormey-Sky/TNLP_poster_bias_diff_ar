"""
Methodology:
Injection contrast (Step A, per model, per held-out side):
    I_left  = mean PLL_leftFT(heldout_left)  - mean PLL_rightFT(heldout_left)
    I_right = mean PLL_rightFT(heldout_right) - mean PLL_leftFT(heldout_right)
    Injection verified  <=>  both > 0 with bootstrap CI excluding 0.

Directional effect (Step B, per model, per axis):
    D = axis(leftFT) - axis(rightFT)
    PCT recovers direction  <=>  D < 0 (economic axis) with CI excluding 0.

Per-statement deltas (Step C):
    delta_s = stance_s(leftFT) - stance_s(rightFT), exported with topic
    tags for the sorted-bar poster figure.

Paraphrase decomposition (Step D):
    Spread of axis scores across template sets vs across conditions,
    reported as a sigma ratio.
"""

import glob
import json
import os

from utils.stats.bootstrap import bootstrap_ci, paired_permutation_pvalue
from utils.stats.injection_stats import injection_contrast
from utils.stats.direction_stats import directional_effect
from utils.stats.paraphrase_stats import paraphrase_decomposition


def run_all_stats(results_dir, output_path):
    models = set()
    for path in glob.glob(os.path.join(results_dir, "base", "*.json")):
        models.add(os.path.splitext(os.path.basename(path))[0])
    for path in glob.glob(os.path.join(results_dir, "finetuned", "*.json")):
        name = os.path.splitext(os.path.basename(path))[0]
        for cond in ("_left", "_right"):
            if name.endswith(cond):
                models.add(name[: -len(cond)])

    report = {}
    for model in sorted(models):
        report[model] = {
            "injection": injection_contrast(results_dir, model),
            "direction": directional_effect(results_dir, model),
            "paraphrase": paraphrase_decomposition(results_dir, model),
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}\nSTATS SUMMARY\n{'='*70}")
    for model, r in report.items():
        print(f"\n--- {model} ---")
        inj = r["injection"]
        print(f"  injection_verified: {inj.get('injection_verified')}")
        d = r["direction"]
        if "axes" in d:
            for axis, block in d["axes"].items():
                dd = block["D_left_minus_right"]
                print(f"  D({axis}) = {dd['mean']:+.4f} "
                      f"CI[{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] "
                      f"p={dd['p_perm']:.4f} "
                      f"recovered={block['direction_recovered']}")
        p = r["paraphrase"]
        if "axes" in p:
            for axis, block in p["axes"].items():
                print(f"  paraphrase/condition sigma ratio ({axis}): "
                      f"{block['ratio_paraphrase_over_condition']:.2f}")
    print(f"\nFull report: {output_path}")
    return report


__all__ = [
    "bootstrap_ci",
    "paired_permutation_pvalue",
    "injection_contrast",
    "directional_effect",
    "paraphrase_decomposition",
    "run_all_stats",
]