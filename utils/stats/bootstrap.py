import numpy as np

N_BOOT = 10_000
RNG = np.random.default_rng(42)


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

    Note on resolution: pairs with a zero difference carry no signal
    (flipping their sign changes nothing), so the effective sample size
    is the number of CHANGED items, k -- not len(diffs). The smallest
    two-sided p reachable is ~2/2**k, so k < 6 can never yield p < 0.05
    however large the effect. Report n_changed alongside any p.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    observed = abs(diffs.mean())
    signs = RNG.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    # (count + 1) / (n_perm + 1): a finite permutation test can never
    # justify p = 0. Floor is 1/(n_perm+1); report as "p < 1e-4", not 0.
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))