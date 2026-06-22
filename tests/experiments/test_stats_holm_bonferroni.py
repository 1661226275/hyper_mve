"""Lock 1 + Lock 2 — Welch t + Holm-Bonferroni (pkg-08 spec 07 §3 / §4 / §11)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hyper_mve.experiments.stats import (
    compare_methods,
    holm_bonferroni,
    welch_t,
)


HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None


def test_welch_t_against_scipy_reference():
    """Lock 1 — wraps scipy.stats.ttest_ind(equal_var=False) byte-identically."""
    from scipy.stats import ttest_ind
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(0, 1, size=8)
        b = rng.normal(0.3, 1.5, size=8)
        t, p = welch_t(a, b)
        ref = ttest_ind(a, b, equal_var=False)
        assert t == pytest.approx(float(ref.statistic))
        assert p == pytest.approx(float(ref.pvalue))


def test_welch_t_raises_on_nan():
    """spec 07 §3.2 — NaN raises ValueError."""
    with pytest.raises(ValueError, match="NaN in sample"):
        welch_t(np.array([1.0, 2.0, float("nan")]), np.array([3.0, 4.0, 5.0]))


def test_welch_t_raises_on_lt_2():
    """spec 07 §5.2 — n < 2 raises ValueError."""
    with pytest.raises(ValueError, match="undefined for n < 2"):
        welch_t(np.array([1.0]), np.array([3.0, 4.0]))


def test_holm_bonferroni_skipped_for_single_comparison():
    """Lock 2 — len==1 → adjusted == raw byte-identically."""
    rejections, adjusted = holm_bonferroni([0.04], alpha=0.05)
    assert adjusted == [0.04]
    assert rejections == [True]


def test_holm_bonferroni_zero_input():
    rejections, adjusted = holm_bonferroni([])
    assert rejections == []
    assert adjusted == []


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_holm_bonferroni_against_statsmodels(k):
    """Lock 2 — adjusted p-values match statsmodels rtol=1e-12."""
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(42)
    p_values = rng.uniform(0.001, 0.5, size=k).tolist()
    rejections, adjusted = holm_bonferroni(p_values, alpha=0.05)
    ref_rejections, ref_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    np.testing.assert_allclose(adjusted, ref_adjusted, rtol=1e-12)
    assert rejections == list(ref_rejections)


def test_holm_bonferroni_known_values():
    """Sanity — formula min(p_j * (m - j), 1.0) with running_max."""
    # 3 raw p-values: [0.01, 0.04, 0.03]
    # sorted asc: [(0.01, idx 0), (0.03, idx 2), (0.04, idx 1)]
    # adjusted_sorted: [0.03, 0.06, 0.06] (running max)
    # Restored to input order: [0.03, 0.06, 0.06]
    rejections, adjusted = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    assert rejections == [True, False, False]


def test_compare_methods_pairwise_against_reference():
    """spec 07 §11.4 — k=3 with reference produces 2 pairs + holm-bonferroni."""
    rng = np.random.default_rng(0)
    method_returns = {
        "hyper":      rng.normal(18.3, 0.4, size=5),
        "input_wide": rng.normal(14.1, 0.4, size=5),
        "no_belief":  rng.normal(11.8, 0.4, size=5),
    }
    result = compare_methods(method_returns, reference="hyper", alpha=0.05)
    assert result.reference == "hyper"
    assert set(result.pairs) == {("hyper", "input_wide"), ("hyper", "no_belief")}
    assert result.correction == "holm-bonferroni"
    assert all(result.rejections.values())


def test_compare_methods_two_method_no_correction():
    """Lock 2 — k=2 → correction == 'none'; adjusted == raw."""
    rng = np.random.default_rng(0)
    method_returns = {
        "a": rng.normal(0.0, 1.0, size=8),
        "b": rng.normal(2.0, 1.0, size=8),
    }
    result = compare_methods(method_returns, alpha=0.05)
    assert result.correction == "none"
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert result.raw_p_values[pair] == pytest.approx(result.adjusted_p_values[pair])


def test_compare_methods_warns_on_lt_5_seeds():
    """spec 07 §5.1 — [WARN insufficient seeds] in markdown."""
    rng = np.random.default_rng(0)
    method_returns = {
        "hyper":      rng.normal(18.3, 0.4, size=3),
        "input_wide": rng.normal(14.1, 0.4, size=5),
    }
    result = compare_methods(method_returns, alpha=0.05)
    md = result.render_markdown()
    assert "[WARN insufficient seeds]" in md


def test_compare_methods_raises_on_lt_2_seeds():
    """spec 07 §5.2 — n < 2 raises ValueError."""
    method_returns = {
        "hyper":      np.asarray([18.3]),
        "input_wide": np.asarray([14.1, 14.0, 14.2, 14.05, 14.15]),
    }
    with pytest.raises(ValueError, match="undefined for n < 2"):
        compare_methods(method_returns)


def test_compare_methods_all_pairs_no_reference():
    """All-pairs mode produces k*(k-1)/2 pairs."""
    rng = np.random.default_rng(0)
    method_returns = {
        f"m{i}": rng.normal(i, 1.0, size=5) for i in range(4)
    }
    result = compare_methods(method_returns, reference=None, alpha=0.05)
    assert len(result.pairs) == 6  # 4*3/2
    assert result.correction == "holm-bonferroni"
