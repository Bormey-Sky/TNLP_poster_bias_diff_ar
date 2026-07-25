"""
utils/stats_v2.py

Steps A (analysis half), B, C, D — all statistics over committed JSONs.
No GPU. Run as:  python main_v2.py --step stats --results_dir results/

Core quantities
---------------
Injection contrast (Step A, per model, per held-out side):
    I_left  = mean PLL_leftFT(heldout_left)  - mean PLL_rightFT(heldout_left)
    I_right = mean PLL_rightFT(heldout_right) - mean PLL_leftFT(heldout_right)
    Injection verified  <=>  both > 0 with bootstrap CI excluding 0.
    Because masks are seeded per article index, the per-article PLLs are
    PAIRED across conditions — the bootstrap resamples article-level
    differences, not group means.

Directional effect (Step B, per model, per axis):
    D = axis(leftFT) - axis(rightFT)
    PCT recovers direction  <=>  D < 0 (economic axis) with CI excluding 0.
    Bootstrap resamples the 62 statements (paired: same statement under
    both conditions). Permutation test flips condition labels per
    statement.

Per-statement deltas (Step C):
    delta_s = stance_s(leftFT) - stance_s(rightFT), exported with topic
    tags for the sorted-bar poster figure.

Paraphrase decomposition (Step D):
    For runs evaluated with multiple template sets: spread of axis scores
    across template sets (same checkpoint) vs across conditions (same
    template set). Reported as ranges and as sigma ratio.
"""

import glob
import json
import os

import numpy as np


N_BOOT = 10_000
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Generic bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap CI for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_permutation_pvalue(diffs, n_perm=N_BOOT):
    """
    Two-sided sign-flip permutation test on paired differences:
    H0 says each pair's condition labels are exchangeable.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    observed = abs(diffs.mean())
    signs = RNG.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((null >= observed).mean())


# ---------------------------------------------------------------------------
# Step A analysis — injection contrast from held-out PLL JSONs
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Steps B + C — directional effect and per-statement deltas from PCT JSONs
# ---------------------------------------------------------------------------

def _load_pct(results_dir, model_name, condition):
    """Finds results/finetuned/{model}_{condition}.json or results/base/{model}.json."""
    if condition == "base":
        candidates = [os.path.join(results_dir, "base", f"{model_name}.json")]
    else:
        candidates = [
            os.path.join(results_dir, "finetuned", f"{model_name}_{condition}.json")
        ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def _stance_by_id(pct_json):
    return {
        rec["id"]: {"stance": rec["stance"], "axis": rec["axis"],
                    "topic": rec.get("topic"), "text": rec["text"]}
        for rec in pct_json["per_statement"]
    }


def directional_effect(results_dir, model_name):
    """
    D per axis with bootstrap CI + permutation p, plus per-statement
    deltas (Step C export). Also reports shift-from-base per condition so
    the 'probe responds but not directionally' pattern is one table.
    """
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


# ---------------------------------------------------------------------------
# Step D — paraphrase variance decomposition
# ---------------------------------------------------------------------------

def paraphrase_decomposition(results_dir, model_name):
    """
    Uses by_template_set blocks from PCT JSONs evaluated with
    --templates all. For each axis:
        sigma_paraphrase: std of axis score across template sets,
                          averaged over conditions
        sigma_condition:  std of axis score across conditions,
                          averaged over template sets (base/left/right)
    ratio > 1 means the instrument's phrasing noise exceeds the entire
    finetuning effect.
    """
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_stats(results_dir, output_path):
    """Discover models from result files and run every applicable analysis."""
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

    # Console summary — the poster numbers at a glance
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
