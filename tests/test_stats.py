import numpy as np

from ragbench.stats_testing import bootstrap_ci, paired_bootstrap_test


def test_bootstrap_ci_contains_true_mean_for_constant_values():
    result = bootstrap_ci([1.0] * 50)
    assert result["mean"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0


def test_bootstrap_ci_widens_with_variance():
    low_var = bootstrap_ci([0.5, 0.5, 0.5, 0.6, 0.4])
    high_var = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.0])
    low_width = low_var["ci_high"] - low_var["ci_low"]
    high_width = high_var["ci_high"] - high_var["ci_low"]
    assert high_width > low_width


def test_paired_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(0)
    a = list(rng.uniform(0.8, 1.0, size=100))   # clearly better
    b = list(rng.uniform(0.0, 0.2, size=100))   # clearly worse
    result = paired_bootstrap_test(a, b)
    assert result["mean_diff"] > 0.5
    assert result["significant_at_0.05"]


def test_paired_bootstrap_no_difference_not_significant():
    rng = np.random.default_rng(0)
    a = list(rng.uniform(0.4, 0.6, size=100))
    b = list(a)  # identical -- zero difference by construction
    result = paired_bootstrap_test(a, b)
    assert result["mean_diff"] == 0.0
    assert not result["significant_at_0.05"]
