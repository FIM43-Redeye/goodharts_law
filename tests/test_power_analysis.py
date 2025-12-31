"""
Tests for power analysis calculations.

Verifies that sample size calculations, achieved power, and minimum
detectable effect computations are mathematically correct.
"""
import pytest
import numpy as np

from goodharts.analysis.power import (
    required_sample_size,
    achieved_power,
    minimum_detectable_effect,
    power_analysis,
    estimate_effect_size_from_efficiencies,
    PowerAnalysisResult,
)


class TestRequiredSampleSize:
    """Tests for sample size calculation."""

    def test_larger_effect_needs_fewer_samples(self):
        """Larger effect sizes should require smaller samples."""
        n_small_effect = required_sample_size(effect_size=0.5)
        n_large_effect = required_sample_size(effect_size=2.0)

        assert n_large_effect < n_small_effect

    def test_higher_power_needs_more_samples(self):
        """Higher power should require more samples."""
        n_80 = required_sample_size(effect_size=1.0, power=0.80)
        n_95 = required_sample_size(effect_size=1.0, power=0.95)

        assert n_95 > n_80

    def test_known_value_medium_effect(self):
        """Verify against known approximation: d=0.8, power=0.8 needs ~25."""
        # Standard approximation: n ~ 16/d^2 for 80% power
        # For d=0.8: n ~ 16/0.64 = 25
        n = required_sample_size(effect_size=0.8, power=0.80)

        assert 20 <= n <= 30  # Allow some variation from approximation

    def test_known_value_large_effect(self):
        """Verify against known: d=1.0, power=0.8 needs ~17."""
        # For d=1.0: n ~ 16/1 = 16
        n = required_sample_size(effect_size=1.0, power=0.80)

        assert 15 <= n <= 20

    def test_returns_integer(self):
        """Sample size should always be a positive integer."""
        n = required_sample_size(effect_size=1.5)

        assert isinstance(n, int)
        assert n > 0

    def test_invalid_effect_size(self):
        """Non-positive effect size should raise error."""
        with pytest.raises(ValueError):
            required_sample_size(effect_size=0)

        with pytest.raises(ValueError):
            required_sample_size(effect_size=-1.0)

    def test_invalid_alpha(self):
        """Alpha outside (0, 1) should raise error."""
        with pytest.raises(ValueError):
            required_sample_size(effect_size=1.0, alpha=0)

        with pytest.raises(ValueError):
            required_sample_size(effect_size=1.0, alpha=1.0)

    def test_invalid_power(self):
        """Power outside (0, 1) should raise error."""
        with pytest.raises(ValueError):
            required_sample_size(effect_size=1.0, power=0)

        with pytest.raises(ValueError):
            required_sample_size(effect_size=1.0, power=1.0)


class TestAchievedPower:
    """Tests for achieved power calculation."""

    def test_power_increases_with_sample_size(self):
        """More samples should give higher power."""
        power_small = achieved_power(n_a=10, n_b=10, effect_size=0.8)
        power_large = achieved_power(n_a=100, n_b=100, effect_size=0.8)

        assert power_large > power_small

    def test_power_increases_with_effect_size(self):
        """Larger effects should be easier to detect (higher power)."""
        power_small_d = achieved_power(n_a=30, n_b=30, effect_size=0.3)
        power_large_d = achieved_power(n_a=30, n_b=30, effect_size=1.5)

        assert power_large_d > power_small_d

    def test_power_near_80_for_required_sample(self):
        """Sample size from required_sample_size should achieve ~80% power."""
        n = required_sample_size(effect_size=1.0, power=0.80)
        actual_power = achieved_power(n_a=n, n_b=n, effect_size=1.0)

        # Should be close to 80%, allow 5% tolerance
        assert 0.75 <= actual_power <= 0.85

    def test_power_near_95_for_required_sample(self):
        """Sample size for 95% power should achieve ~95% power."""
        n = required_sample_size(effect_size=1.0, power=0.95)
        actual_power = achieved_power(n_a=n, n_b=n, effect_size=1.0)

        # Should be close to 95%, allow 5% tolerance
        assert 0.90 <= actual_power <= 0.99

    def test_power_bounds(self):
        """Power should always be between 0 and 1."""
        power = achieved_power(n_a=1000, n_b=1000, effect_size=3.0)
        assert 0 <= power <= 1

    def test_power_with_unequal_samples(self):
        """Should handle unequal sample sizes."""
        power = achieved_power(n_a=50, n_b=30, effect_size=1.0)
        assert 0 < power < 1

    def test_power_zero_effect(self):
        """Zero effect should return power = alpha (false positive rate)."""
        power = achieved_power(n_a=50, n_b=50, effect_size=0)
        # Power = alpha when H0 is true
        assert abs(power - 0.05) < 0.01

    def test_power_tiny_samples(self):
        """Very small samples should return 0."""
        power = achieved_power(n_a=1, n_b=1, effect_size=1.0)
        assert power == 0.0


class TestMinimumDetectableEffect:
    """Tests for minimum detectable effect calculation."""

    def test_inverse_of_required_sample_size(self):
        """MDE for n should match d for required_sample_size(d) = n."""
        # Get sample size for d=1.0
        n = required_sample_size(effect_size=1.0, power=0.80)

        # MDE for that n should be ~1.0
        mde = minimum_detectable_effect(n_per_group=n, power=0.80)

        assert abs(mde - 1.0) < 0.1

    def test_larger_sample_detects_smaller_effect(self):
        """Larger samples should detect smaller effects."""
        mde_small_n = minimum_detectable_effect(n_per_group=20)
        mde_large_n = minimum_detectable_effect(n_per_group=100)

        assert mde_large_n < mde_small_n

    def test_higher_power_increases_mde(self):
        """Higher power requirements should increase MDE."""
        mde_80 = minimum_detectable_effect(n_per_group=50, power=0.80)
        mde_95 = minimum_detectable_effect(n_per_group=50, power=0.95)

        assert mde_95 > mde_80

    def test_returns_positive(self):
        """MDE should always be positive."""
        mde = minimum_detectable_effect(n_per_group=30)
        assert mde > 0

    def test_tiny_sample_returns_inf(self):
        """n=1 should return infinity (can't detect anything)."""
        mde = minimum_detectable_effect(n_per_group=1)
        assert mde == float('inf')


class TestPowerAnalysis:
    """Tests for the full power analysis function."""

    def test_returns_power_analysis_result(self):
        """Should return a PowerAnalysisResult dataclass."""
        result = power_analysis(effect_size=1.5)

        assert isinstance(result, PowerAnalysisResult)
        assert hasattr(result, 'n_per_group')
        assert hasattr(result, 'total_n')
        assert hasattr(result, 'estimated_timesteps')

    def test_total_n_is_double_n_per_group(self):
        """Total n should be 2x n per group."""
        result = power_analysis(effect_size=1.0)

        assert result.total_n == result.n_per_group * 2

    def test_stores_input_parameters(self):
        """Result should store input parameters."""
        result = power_analysis(effect_size=2.0, alpha=0.01, power=0.90)

        assert result.effect_size == 2.0
        assert result.alpha == 0.01
        assert result.power == 0.90

    def test_timestep_estimation(self):
        """Timesteps should scale with sample size."""
        result = power_analysis(effect_size=1.0, deaths_per_1k_steps=3.0)

        # timesteps = n * 1000 / deaths_per_1k
        expected = result.n_per_group * 1000 / 3.0
        assert abs(result.estimated_timesteps - expected) < 1

    def test_from_pilot_data(self):
        """Should estimate effect size from pilot data."""
        pilot = {
            'ground_truth': [0.90, 0.92, 0.91, 0.89, 0.93],
            'proxy': [0.50, 0.52, 0.48, 0.51, 0.49],
        }
        result = power_analysis(pilot_data=pilot)

        # Large difference should give large effect size
        assert result.effect_size > 2.0

    def test_no_effect_size_or_pilot_raises(self):
        """Must provide either effect_size or pilot_data."""
        with pytest.raises(ValueError):
            power_analysis()

    def test_interpretation_included(self):
        """Result should include human-readable interpretation."""
        result = power_analysis(effect_size=1.5)

        assert len(result.interpretation) > 0
        assert 'achievable' in result.interpretation.lower() or 'effect' in result.interpretation.lower()


class TestEstimateEffectSizeFromEfficiencies:
    """Tests for effect size estimation from efficiencies."""

    def test_known_difference(self):
        """0.45 efficiency difference with std=0.15 should give d=3.0."""
        d = estimate_effect_size_from_efficiencies(
            gt_efficiency=0.95,
            proxy_efficiency=0.50,
            efficiency_std=0.15,
        )
        # d = |0.95 - 0.50| / 0.15 = 3.0
        assert abs(d - 3.0) < 0.01

    def test_zero_difference(self):
        """Equal efficiencies should give d=0."""
        d = estimate_effect_size_from_efficiencies(
            gt_efficiency=0.80,
            proxy_efficiency=0.80,
        )
        assert d == 0.0

    def test_moderate_difference(self):
        """Moderate difference should give moderate d."""
        d = estimate_effect_size_from_efficiencies(
            gt_efficiency=0.90,
            proxy_efficiency=0.75,
            efficiency_std=0.15,
        )
        # d = 0.15 / 0.15 = 1.0
        assert abs(d - 1.0) < 0.01


class TestPowerAnalysisIntegration:
    """Integration tests for power analysis workflow."""

    def test_typical_goodhart_experiment(self):
        """Test typical Goodhart experiment planning."""
        # Expect 90% GT efficiency vs 50% proxy efficiency
        d = estimate_effect_size_from_efficiencies(0.90, 0.50, 0.15)
        result = power_analysis(effect_size=d, power=0.80)

        # Should be achievable with reasonable sample size
        assert result.n_per_group < 100

    def test_round_trip_consistency(self):
        """Sample size and achieved power should be consistent."""
        target_power = 0.80
        effect_size = 1.2

        n = required_sample_size(effect_size=effect_size, power=target_power)
        actual_power = achieved_power(n_a=n, n_b=n, effect_size=effect_size)

        # Achieved power should be at least the target
        assert actual_power >= target_power * 0.95  # 5% tolerance

    def test_mde_and_achieved_power_consistent(self):
        """MDE should achieve target power for that sample size."""
        n = 50
        target_power = 0.80

        mde = minimum_detectable_effect(n_per_group=n, power=target_power)
        actual_power = achieved_power(n_a=n, n_b=n, effect_size=mde)

        assert abs(actual_power - target_power) < 0.05
