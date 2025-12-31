"""
Tests for statistical analysis helpers.

Verifies that compute_comparison, effect sizes, and confidence intervals
produce mathematically correct results.
"""
import pytest
import numpy as np
from scipy import stats as scipy_stats

from goodharts.analysis.stats_helpers import (
    StatisticalComparison,
    compute_comparison,
    compute_confidence_interval,
    compute_effect_sizes,
    format_p_value,
    compute_goodhart_failure_index,
)


class TestComputeConfidenceInterval:
    """Tests for confidence interval computation."""

    def test_ci_contains_sample_mean(self):
        """CI should be centered on the sample mean."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = compute_confidence_interval(values, 0.95)
        mean = np.mean(values)

        assert lower < mean < upper

    def test_ci_width_increases_with_variance(self):
        """Higher variance should produce wider CI."""
        low_var = [1.0, 1.1, 1.2, 0.9, 1.0]
        high_var = [0.0, 2.0, 1.0, 3.0, 0.5]

        ci_low = compute_confidence_interval(low_var, 0.95)
        ci_high = compute_confidence_interval(high_var, 0.95)

        width_low = ci_low[1] - ci_low[0]
        width_high = ci_high[1] - ci_high[0]

        assert width_high > width_low

    def test_ci_width_decreases_with_sample_size(self):
        """Larger samples should produce narrower CI."""
        np.random.seed(42)
        small_sample = list(np.random.randn(10))
        large_sample = list(np.random.randn(100))

        ci_small = compute_confidence_interval(small_sample, 0.95)
        ci_large = compute_confidence_interval(large_sample, 0.95)

        # Normalize by std to compare widths fairly
        std_small = np.std(small_sample, ddof=1)
        std_large = np.std(large_sample, ddof=1)

        normalized_width_small = (ci_small[1] - ci_small[0]) / std_small
        normalized_width_large = (ci_large[1] - ci_large[0]) / std_large

        assert normalized_width_large < normalized_width_small

    def test_ci_99_wider_than_95(self):
        """99% CI should be wider than 95% CI."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        ci_95 = compute_confidence_interval(values, 0.95)
        ci_99 = compute_confidence_interval(values, 0.99)

        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]

        assert width_99 > width_95

    def test_ci_empty_list(self):
        """Empty list should return (0, 0)."""
        result = compute_confidence_interval([], 0.95)
        assert result == (0.0, 0.0)

    def test_ci_single_value(self):
        """Single value should return the value as both bounds."""
        result = compute_confidence_interval([5.0], 0.95)
        assert result == (5.0, 5.0)


class TestComputeEffectSizes:
    """Tests for Cohen's d and Hedges' g computation."""

    def test_cohens_d_identical_groups(self):
        """Identical groups should have d = 0."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]

        d, g = compute_effect_sizes(a, b)

        assert d == 0.0
        assert g == 0.0

    def test_cohens_d_known_value(self):
        """Verify d against a hand-calculated example."""
        # Two groups with means 0 and 1, each with std = 1
        a = [-1.0, 0.0, 1.0]  # mean = 0, std = 1
        b = [0.0, 1.0, 2.0]   # mean = 1, std = 1

        d, g = compute_effect_sizes(a, b)

        # d = (0 - 1) / 1 = -1
        assert abs(d - (-1.0)) < 0.01

    def test_hedges_g_smaller_than_cohens_d(self):
        """Hedges' g correction reduces magnitude for small samples."""
        a = [0.0, 1.0, 2.0]
        b = [1.0, 2.0, 3.0]

        d, g = compute_effect_sizes(a, b)

        # g should be closer to 0 than d due to small-sample correction
        assert abs(g) < abs(d)

    def test_hedges_g_approaches_cohens_d_for_large_n(self):
        """For large samples, Hedges' g should approach Cohen's d."""
        np.random.seed(42)
        a = list(np.random.randn(1000))
        b = list(np.random.randn(1000) + 0.5)

        d, g = compute_effect_sizes(a, b)

        # Should be very close for n=1000
        assert abs(d - g) < 0.01

    def test_effect_size_direction(self):
        """Higher A should give positive d."""
        a = [5.0, 6.0, 7.0]
        b = [1.0, 2.0, 3.0]

        d, g = compute_effect_sizes(a, b)

        assert d > 0
        assert g > 0

    def test_effect_size_empty_groups(self):
        """Empty groups should return 0."""
        d, g = compute_effect_sizes([], [1.0, 2.0, 3.0])
        assert d == 0.0
        assert g == 0.0


class TestComputeComparison:
    """Tests for the full statistical comparison."""

    def test_comparison_contains_expected_fields(self):
        """Comparison should include all required statistics."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = compute_comparison(a, b, metric='test', group_a='A', group_b='B')

        assert result.metric == 'test'
        assert result.group_a == 'A'
        assert result.group_b == 'B'
        assert result.n_a == 5
        assert result.n_b == 5
        assert hasattr(result, 'mean_a')
        assert hasattr(result, 'cohens_d')
        assert hasattr(result, 'p_value')

    def test_mean_difference_computed_correctly(self):
        """Mean difference property should be mean_a - mean_b."""
        a = [10.0, 10.0, 10.0]
        b = [5.0, 5.0, 5.0]

        result = compute_comparison(a, b)

        assert result.mean_diff == 5.0

    def test_significant_difference_detected(self):
        """Clearly different groups should give significant p-value."""
        a = [100.0, 101.0, 102.0, 103.0, 104.0]
        b = [0.0, 1.0, 2.0, 3.0, 4.0]

        result = compute_comparison(a, b)

        assert result.p_value < 0.05
        assert result.significant

    def test_non_significant_for_overlapping_groups(self):
        """Highly overlapping groups should not be significant."""
        # Use values that are clearly overlapping (not randomly generated
        # which could accidentally be significant)
        a = [1.0, 2.0, 3.0, 2.5, 1.5]
        b = [1.2, 2.1, 2.8, 2.4, 1.6]  # Very similar to a

        result = compute_comparison(a, b)

        # Nearly identical groups should not be significant
        assert result.p_value > 0.05

    def test_welch_df_different_from_pooled(self):
        """Welch's df should differ from simple n1+n2-2 when variances differ."""
        a = [1.0, 1.1, 1.2, 0.9, 1.0]  # Low variance
        b = [0.0, 5.0, 2.5, 10.0, -2.0]  # High variance

        result = compute_comparison(a, b)

        pooled_df = len(a) + len(b) - 2
        # Welch df should be less than pooled when variances differ
        assert result.df < pooled_df


class TestStatisticalComparisonProperties:
    """Tests for derived properties of StatisticalComparison."""

    def test_significance_stars_p_001(self):
        """p < 0.001 should give ***."""
        comp = StatisticalComparison(
            metric='test', group_a='A', group_b='B',
            n_a=10, n_b=10, mean_a=1, mean_b=0,
            std_a=0.1, std_b=0.1,
            ci_a=(0.9, 1.1), ci_b=(-0.1, 0.1), ci_diff=(0.8, 1.2),
            t_statistic=10.0, p_value=0.0001, df=18,
            cohens_d=1.0, hedges_g=0.95
        )
        assert comp.significance_stars == '***'

    def test_significance_stars_p_01(self):
        """0.001 <= p < 0.01 should give **."""
        comp = StatisticalComparison(
            metric='test', group_a='A', group_b='B',
            n_a=10, n_b=10, mean_a=1, mean_b=0,
            std_a=0.1, std_b=0.1,
            ci_a=(0.9, 1.1), ci_b=(-0.1, 0.1), ci_diff=(0.8, 1.2),
            t_statistic=3.0, p_value=0.005, df=18,
            cohens_d=1.0, hedges_g=0.95
        )
        assert comp.significance_stars == '**'

    def test_significance_stars_p_05(self):
        """0.01 <= p < 0.05 should give *."""
        comp = StatisticalComparison(
            metric='test', group_a='A', group_b='B',
            n_a=10, n_b=10, mean_a=1, mean_b=0,
            std_a=0.1, std_b=0.1,
            ci_a=(0.9, 1.1), ci_b=(-0.1, 0.1), ci_diff=(0.8, 1.2),
            t_statistic=2.0, p_value=0.03, df=18,
            cohens_d=1.0, hedges_g=0.95
        )
        assert comp.significance_stars == '*'

    def test_significance_stars_ns(self):
        """p >= 0.05 should give ns."""
        comp = StatisticalComparison(
            metric='test', group_a='A', group_b='B',
            n_a=10, n_b=10, mean_a=1, mean_b=0,
            std_a=0.1, std_b=0.1,
            ci_a=(0.9, 1.1), ci_b=(-0.1, 0.1), ci_diff=(0.8, 1.2),
            t_statistic=1.0, p_value=0.15, df=18,
            cohens_d=1.0, hedges_g=0.95
        )
        assert comp.significance_stars == 'ns'

    def test_effect_magnitude_negligible(self):
        """d < 0.2 should be negligible."""
        comp = StatisticalComparison(
            metric='test', group_a='A', group_b='B',
            n_a=10, n_b=10, mean_a=1, mean_b=0,
            std_a=0.1, std_b=0.1,
            ci_a=(0.9, 1.1), ci_b=(-0.1, 0.1), ci_diff=(0.8, 1.2),
            t_statistic=1.0, p_value=0.15, df=18,
            cohens_d=0.1, hedges_g=0.09
        )
        assert comp.effect_magnitude == 'negligible'

    def test_effect_magnitude_small(self):
        """0.2 <= d < 0.5 should be small."""
        comp = StatisticalComparison(
            metric='test', group_a='A', group_b='B',
            n_a=10, n_b=10, mean_a=1, mean_b=0,
            std_a=0.1, std_b=0.1,
            ci_a=(0.9, 1.1), ci_b=(-0.1, 0.1), ci_diff=(0.8, 1.2),
            t_statistic=2.0, p_value=0.03, df=18,
            cohens_d=0.3, hedges_g=0.28
        )
        assert comp.effect_magnitude == 'small'

    def test_effect_magnitude_medium(self):
        """0.5 <= d < 0.8 should be medium."""
        comp = StatisticalComparison(
            metric='test', group_a='A', group_b='B',
            n_a=10, n_b=10, mean_a=1, mean_b=0,
            std_a=0.1, std_b=0.1,
            ci_a=(0.9, 1.1), ci_b=(-0.1, 0.1), ci_diff=(0.8, 1.2),
            t_statistic=3.0, p_value=0.01, df=18,
            cohens_d=0.6, hedges_g=0.57
        )
        assert comp.effect_magnitude == 'medium'

    def test_effect_magnitude_large(self):
        """d >= 0.8 should be large."""
        comp = StatisticalComparison(
            metric='test', group_a='A', group_b='B',
            n_a=10, n_b=10, mean_a=1, mean_b=0,
            std_a=0.1, std_b=0.1,
            ci_a=(0.9, 1.1), ci_b=(-0.1, 0.1), ci_diff=(0.8, 1.2),
            t_statistic=5.0, p_value=0.001, df=18,
            cohens_d=1.2, hedges_g=1.1
        )
        assert comp.effect_magnitude == 'large'


class TestFormatPValue:
    """Tests for p-value formatting."""

    def test_very_small_p(self):
        """Very small p should show '< 0.001'."""
        assert format_p_value(0.0001) == '< 0.001'

    def test_small_p(self):
        """0.001-0.01 should show 3 decimal places."""
        assert format_p_value(0.005) == '0.005'

    def test_medium_p(self):
        """p >= 0.01 should show 2 decimal places."""
        assert format_p_value(0.03) == '0.03'

    def test_large_p(self):
        """Large p should show 2 decimal places."""
        assert format_p_value(0.75) == '0.75'


class TestGoodhartFailureIndex:
    """Tests for the Goodhart Failure Index."""

    def test_gfi_no_failure(self):
        """Equal efficiencies should give GFI = 0."""
        gfi = compute_goodhart_failure_index(0.9, 0.9)
        assert gfi == 0.0

    def test_gfi_complete_failure(self):
        """Zero proxy efficiency should give GFI = 1."""
        gfi = compute_goodhart_failure_index(0.9, 0.0)
        assert gfi == 1.0

    def test_gfi_half_failure(self):
        """Proxy at half of ground_truth should give GFI = 0.5."""
        gfi = compute_goodhart_failure_index(1.0, 0.5)
        assert gfi == 0.5

    def test_gfi_typical_goodhart(self):
        """Typical Goodhart scenario: 90% GT, 50% proxy."""
        gfi = compute_goodhart_failure_index(0.9, 0.5)
        expected = (0.9 - 0.5) / 0.9
        assert abs(gfi - expected) < 0.001

    def test_gfi_zero_gt(self):
        """Zero ground_truth efficiency should return 0 (avoid div by zero)."""
        gfi = compute_goodhart_failure_index(0.0, 0.5)
        assert gfi == 0.0
