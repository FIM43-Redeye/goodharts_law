"""
Tests for multi-run evaluation aggregation.

Verifies that RunResult, MultiRunAggregates, and MultiRunComparison
correctly aggregate and compare across runs.
"""
import pytest
import numpy as np

from goodharts.evaluation.multi_run import (
    RunResult,
    MultiRunAggregates,
    MultiRunComparison,
    aggregate_runs,
    generate_seeds,
    compute_confidence_interval,
    compute_pooled_std,
    _cohens_d,
    _significance_stars,
    _effect_magnitude,
)


# -----------------------------------------------------------------------------
# Fixtures for creating mock RunResult objects
# -----------------------------------------------------------------------------

def make_run_result(
    run_id: int = 0,
    seed: int = 42,
    efficiency: float = 0.9,
    survival_mean: float = 100.0,
    survival_std: float = 30.0,
    n_deaths: int = 50,
    total_timesteps: int = 10000,
) -> RunResult:
    """Create a mock RunResult for testing."""
    return RunResult(
        run_id=run_id,
        seed=seed,
        n_deaths=n_deaths,
        total_timesteps=total_timesteps,
        overall_efficiency=efficiency,
        survival_mean=survival_mean,
        survival_std=survival_std,
        deaths_per_1k_steps=n_deaths / total_timesteps * 1000,
        food_per_1k_steps=50.0,
        poison_per_1k_steps=5.0,
        food_per_death_mean=10.0,
        poison_per_death_mean=1.0,
        reward_mean=50.0,
    )


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_from_dict(self):
        """Should reconstruct RunResult from dictionary."""
        data = {
            'run_id': 1,
            'seed': 123,
            'n_deaths': 25,
            'total_timesteps': 5000,
            'overall_efficiency': 0.85,
            'survival_mean': 150.0,
            'survival_std': 40.0,
            'deaths_per_1k_steps': 5.0,
            'food_per_1k_steps': 40.0,
            'poison_per_1k_steps': 7.0,
            'food_per_death_mean': 8.0,
            'poison_per_death_mean': 1.5,
            'reward_mean': 45.0,
        }

        result = RunResult.from_dict(data)

        assert result.run_id == 1
        assert result.seed == 123
        assert result.overall_efficiency == 0.85

    def test_from_dict_ignores_extra_fields(self):
        """Should ignore fields not in the dataclass."""
        data = {
            'run_id': 0,
            'seed': 42,
            'n_deaths': 10,
            'total_timesteps': 1000,
            'overall_efficiency': 0.9,
            'survival_mean': 100.0,
            'survival_std': 30.0,
            'deaths_per_1k_steps': 10.0,
            'food_per_1k_steps': 50.0,
            'poison_per_1k_steps': 5.0,
            'food_per_death_mean': 10.0,
            'poison_per_death_mean': 1.0,
            'reward_mean': 50.0,
            'extra_field': 'should_be_ignored',
        }

        result = RunResult.from_dict(data)
        assert not hasattr(result, 'extra_field')


class TestGenerateSeeds:
    """Tests for deterministic seed generation."""

    def test_reproducibility_with_base_seed(self):
        """Same base_seed should produce same seeds."""
        seeds1 = generate_seeds(5, base_seed=42)
        seeds2 = generate_seeds(5, base_seed=42)

        assert seeds1 == seeds2

    def test_different_base_seeds_differ(self):
        """Different base_seeds should produce different sequences."""
        seeds1 = generate_seeds(5, base_seed=42)
        seeds2 = generate_seeds(5, base_seed=123)

        assert seeds1 != seeds2

    def test_correct_count(self):
        """Should generate requested number of seeds."""
        seeds = generate_seeds(10, base_seed=42)
        assert len(seeds) == 10

    def test_seeds_are_integers(self):
        """All seeds should be integers."""
        seeds = generate_seeds(5, base_seed=42)
        assert all(isinstance(s, int) for s in seeds)

    def test_without_base_seed_produces_random(self):
        """Without base_seed, should produce random (but recorded) seeds."""
        seeds1 = generate_seeds(5)
        seeds2 = generate_seeds(5)

        # Very unlikely to be equal
        assert seeds1 != seeds2


class TestComputeConfidenceInterval:
    """Tests for multi_run's confidence interval computation."""

    def test_single_value(self):
        """Single value should have point CI."""
        mean, std, lo, hi = compute_confidence_interval([5.0])

        assert mean == 5.0
        assert std == 0.0
        assert lo == 5.0
        assert hi == 5.0

    def test_empty_list(self):
        """Empty list should return zeros."""
        mean, std, lo, hi = compute_confidence_interval([])

        assert mean == 0.0
        assert std == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_ci_contains_mean(self):
        """CI should contain the mean."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, std, lo, hi = compute_confidence_interval(values)

        assert lo < mean < hi

    def test_higher_confidence_wider_interval(self):
        """99% CI should be wider than 95% CI."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        _, _, lo95, hi95 = compute_confidence_interval(values, 0.95)
        _, _, lo99, hi99 = compute_confidence_interval(values, 0.99)

        width_95 = hi95 - lo95
        width_99 = hi99 - lo99

        assert width_99 > width_95


class TestComputePooledStd:
    """Tests for pooled standard deviation computation."""

    def test_identical_groups(self):
        """Identical groups should have same mean and std."""
        means = [5.0, 5.0, 5.0]
        stds = [1.0, 1.0, 1.0]
        ns = [10, 10, 10]

        pooled_mean, pooled_std = compute_pooled_std(means, stds, ns)

        assert pooled_mean == 5.0
        # Pooled std should be close to 1.0, but formula includes between-group variance
        assert abs(pooled_std - 1.0) < 0.1

    def test_weighted_by_sample_size(self):
        """Pooled mean should weight by sample size."""
        means = [10.0, 0.0]
        stds = [1.0, 1.0]
        ns = [90, 10]  # First group dominates

        pooled_mean, _ = compute_pooled_std(means, stds, ns)

        # Should be closer to 10 than to 0
        assert pooled_mean > 5.0

    def test_empty_groups(self):
        """Empty groups should return zeros."""
        pooled_mean, pooled_std = compute_pooled_std([], [], [])

        assert pooled_mean == 0.0
        assert pooled_std == 0.0


class TestAggregateRuns:
    """Tests for aggregating multiple runs."""

    def test_basic_aggregation(self):
        """Should compute mean and CI across runs."""
        runs = [
            make_run_result(run_id=0, efficiency=0.90, survival_mean=100),
            make_run_result(run_id=1, efficiency=0.92, survival_mean=110),
            make_run_result(run_id=2, efficiency=0.88, survival_mean=90),
        ]

        agg = aggregate_runs('test_mode', runs)

        assert agg.mode == 'test_mode'
        assert agg.n_runs == 3
        assert abs(agg.efficiency_mean - 0.90) < 0.01

    def test_totals_summed(self):
        """Total deaths and timesteps should be sum across runs."""
        runs = [
            make_run_result(run_id=0, n_deaths=50, total_timesteps=10000),
            make_run_result(run_id=1, n_deaths=60, total_timesteps=10000),
            make_run_result(run_id=2, n_deaths=40, total_timesteps=10000),
        ]

        agg = aggregate_runs('test_mode', runs)

        assert agg.total_deaths == 150
        assert agg.total_timesteps == 30000

    def test_seeds_preserved(self):
        """Seeds should be preserved in order."""
        runs = [
            make_run_result(run_id=0, seed=100),
            make_run_result(run_id=1, seed=200),
            make_run_result(run_id=2, seed=300),
        ]

        agg = aggregate_runs('test_mode', runs)

        assert agg.seeds == [100, 200, 300]

    def test_runs_preserved(self):
        """Individual runs should be accessible."""
        runs = [
            make_run_result(run_id=0, efficiency=0.90),
            make_run_result(run_id=1, efficiency=0.92),
        ]

        agg = aggregate_runs('test_mode', runs)

        assert len(agg.runs) == 2
        assert agg.runs[0].overall_efficiency == 0.90

    def test_empty_runs_raises(self):
        """Should raise on empty run list."""
        with pytest.raises(ValueError):
            aggregate_runs('test_mode', [])

    def test_to_dict(self):
        """Should produce JSON-serializable dict."""
        runs = [make_run_result(run_id=0)]
        agg = aggregate_runs('test_mode', runs)

        d = agg.to_dict()

        assert d['mode'] == 'test_mode'
        assert 'efficiency_mean' in d
        assert isinstance(d['runs'], list)


class TestMultiRunComparison:
    """Tests for comparing two modes across runs."""

    def _make_agg(self, mode: str, efficiencies: list[float], survivals: list[float]) -> MultiRunAggregates:
        """Helper to create aggregates from efficiency and survival lists."""
        runs = [
            make_run_result(run_id=i, seed=i, efficiency=e, survival_mean=s)
            for i, (e, s) in enumerate(zip(efficiencies, survivals))
        ]
        return aggregate_runs(mode, runs)

    def test_significant_difference_detected(self):
        """Clear difference should be significant."""
        # Ground truth: high efficiency, long survival
        agg_gt = self._make_agg(
            'ground_truth',
            efficiencies=[0.90, 0.92, 0.88, 0.91, 0.89],
            survivals=[200, 220, 180, 210, 190],
        )

        # Proxy: low efficiency, short survival
        agg_proxy = self._make_agg(
            'proxy',
            efficiencies=[0.50, 0.52, 0.48, 0.51, 0.49],
            survivals=[80, 90, 70, 85, 75],
        )

        comparison = MultiRunComparison.from_aggregates(agg_gt, agg_proxy)

        assert comparison.efficiency_significant
        assert comparison.survival_significant
        assert comparison.efficiency_cohens_d > 0  # GT higher

    def test_no_difference_not_significant(self):
        """Similar groups should not be significant."""
        agg_a = self._make_agg(
            'mode_a',
            efficiencies=[0.50, 0.52, 0.48, 0.51, 0.49],
            survivals=[100, 105, 95, 102, 98],
        )

        agg_b = self._make_agg(
            'mode_b',
            efficiencies=[0.51, 0.49, 0.50, 0.52, 0.48],
            survivals=[101, 99, 100, 103, 97],
        )

        comparison = MultiRunComparison.from_aggregates(agg_a, agg_b)

        assert not comparison.efficiency_significant
        assert not comparison.survival_significant

    def test_goodhart_index_computed(self):
        """GFI should reflect efficiency gap."""
        # Use small variance to avoid scipy warnings about identical data
        agg_gt = self._make_agg(
            'ground_truth',
            efficiencies=[0.89, 0.90, 0.91, 0.90, 0.90],
            survivals=[98, 100, 102, 100, 100],
        )

        agg_proxy = self._make_agg(
            'proxy',
            efficiencies=[0.44, 0.45, 0.46, 0.45, 0.45],
            survivals=[49, 50, 51, 50, 50],
        )

        comparison = MultiRunComparison.from_aggregates(agg_gt, agg_proxy)

        # GFI = (0.90 - 0.45) / 0.90 = 0.5
        assert abs(comparison.goodhart_index - 0.5) < 0.02

    def test_significance_stars(self):
        """Should produce correct significance stars."""
        agg_gt = self._make_agg(
            'ground_truth',
            efficiencies=[0.95, 0.96, 0.94, 0.95, 0.96],
            survivals=[200, 210, 190, 200, 205],
        )

        agg_proxy = self._make_agg(
            'proxy',
            efficiencies=[0.30, 0.32, 0.28, 0.31, 0.29],
            survivals=[50, 55, 45, 52, 48],
        )

        comparison = MultiRunComparison.from_aggregates(agg_gt, agg_proxy)

        # Very significant difference
        assert comparison.efficiency_stars in ['***', '**', '*']
        assert comparison.efficiency_effect_magnitude == 'large'


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_cohens_d_known_value(self):
        """Verify Cohen's d against known calculation."""
        # Two groups: means 0 and 2, both with std ~= 1
        a = [-1.0, 0.0, 1.0]  # mean = 0
        b = [1.0, 2.0, 3.0]   # mean = 2

        d = _cohens_d(a, b)

        # d = (0 - 2) / 1 = -2
        assert abs(d - (-2.0)) < 0.1

    def test_cohens_d_empty_groups(self):
        """Empty groups should return 0."""
        assert _cohens_d([], [1, 2, 3]) == 0.0
        assert _cohens_d([1, 2, 3], []) == 0.0

    def test_significance_stars_levels(self):
        """Should return correct stars for each significance level."""
        assert _significance_stars(0.0001) == '***'
        assert _significance_stars(0.005) == '**'
        assert _significance_stars(0.03) == '*'
        assert _significance_stars(0.10) == 'ns'

    def test_effect_magnitude_levels(self):
        """Should return correct magnitude labels."""
        assert _effect_magnitude(0.1) == 'negligible'
        assert _effect_magnitude(0.3) == 'small'
        assert _effect_magnitude(0.6) == 'medium'
        assert _effect_magnitude(1.0) == 'large'

    def test_effect_magnitude_uses_absolute_value(self):
        """Should interpret magnitude, not direction."""
        assert _effect_magnitude(-1.0) == 'large'
        assert _effect_magnitude(-0.1) == 'negligible'


class TestGoodhartThesis:
    """Integration tests verifying the Goodhart thesis measurement."""

    def test_thesis_scenario(self):
        """
        Simulate the expected Goodhart scenario:
        - Ground truth agents achieve ~90% efficiency (eat mostly food)
        - Proxy agents achieve ~50% efficiency (cannot distinguish food from poison)

        The comparison should show:
        - Significant difference in efficiency
        - Large effect size (d > 2.0)
        - High Goodhart Failure Index (~0.45)
        """
        # Ground truth: can see food vs poison, achieves high efficiency
        gt_runs = [
            make_run_result(run_id=i, efficiency=0.90 + np.random.randn() * 0.02,
                           survival_mean=200 + np.random.randn() * 20)
            for i in range(5)
        ]
        np.random.seed(42)  # Reset for determinism
        gt_agg = aggregate_runs('ground_truth', gt_runs)

        # Proxy: cannot distinguish, random 50% efficiency
        np.random.seed(43)
        proxy_runs = [
            make_run_result(run_id=i, efficiency=0.50 + np.random.randn() * 0.02,
                           survival_mean=80 + np.random.randn() * 10)
            for i in range(5)
        ]
        proxy_agg = aggregate_runs('proxy', proxy_runs)

        # Compare modes
        comparison = MultiRunComparison.from_aggregates(gt_agg, proxy_agg)

        # Verify thesis predictions
        assert comparison.efficiency_significant, "Efficiency difference should be significant"
        assert comparison.efficiency_cohens_d > 2.0, f"Effect size should be large, got {comparison.efficiency_cohens_d}"
        assert comparison.goodhart_index > 0.4, f"GFI should be substantial, got {comparison.goodhart_index}"

        # Ground truth should dominate on all metrics
        assert gt_agg.efficiency_mean > proxy_agg.efficiency_mean
        assert gt_agg.survival_mean_of_means > proxy_agg.survival_mean_of_means
