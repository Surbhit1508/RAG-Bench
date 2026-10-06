"""Statistical rigor for the ablation study: bootstrap confidence
intervals and paired significance tests between configs.

Why bootstrap instead of a plain t-test? Our metrics (hit@k, EM) are
often binary/bounded, and we'd rather not assume normality with only
~150 QA pairs. Bootstrap resampling makes no distributional assumption
and is easy to explain in an interview: "resample the results with
replacement 10,000 times, see how much the mean metric wobbles."
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values: list[float], n_resamples: int = 10_000, ci: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap confidence interval for the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap_test(values_a: list[float], values_b: list[float], n_resamples: int = 10_000, seed: int = 42) -> dict:
    """Is config A significantly different from config B on the SAME
    QA pairs (paired samples)? Returns the observed mean difference,
    its bootstrap CI, and a two-sided p-value estimate.

    Pairing matters: A and B were run on identical questions, so we
    resample *question indices* (not the differences independently)
    to preserve that pairing -- exactly like a paired t-test, but
    without the normality assumption.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    assert len(a) == len(b), "paired test requires equal-length, aligned samples"
    n = len(a)
    observed_diff = float(a.mean() - b.mean())

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()

    ci_low, ci_high = np.quantile(diffs, [0.025, 0.975])
    # two-sided p-value: fraction of resampled diffs at least as extreme
    # as zero, mirrored around the observed direction
    p_value = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    p_value = min(p_value, 1.0)

    return {
        "mean_diff": observed_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "significant_at_0.05": bool(ci_low > 0 or ci_high < 0),
    }
