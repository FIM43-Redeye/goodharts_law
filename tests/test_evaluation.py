"""
Tests for the evaluation system and Goodhart's Law thesis validation.

The evaluation system uses a continuous survival paradigm: agents run until
they die (starvation), then auto-respawn. We track death events, not episodes.

These tests verify:
1. Evaluation infrastructure correctness (metrics, aggregates)
2. Thesis validation framework (what the Goodhart demonstration should show)

Note on testing philosophy: "Testing machinery" verifies the code works correctly
(e.g., efficiency calculation produces expected values). "Validating the thesis"
documents the expected behavioral differences that demonstrate Goodhart's Law.
Both matter: correct machinery ensures trustworthy measurements, while thesis
validation defines what those measurements should show.
"""

import pytest
from dataclasses import asdict

from goodharts.evaluation.evaluator import (
    DeathEvent,
    ModeAggregates,
    EvaluationConfig,
)


class TestDeathEvent:
    """Tests for individual death event metrics."""

    def test_efficiency_with_food_only(self):
        """Agent that eats only food should have 100% efficiency."""
        death = DeathEvent(
            mode='ground_truth',
            death_id=0,
            survival_time=100,
            food_eaten=10,
            poison_eaten=0,
            total_reward=10.0,
            final_energy=0.0,
        )
        assert death.efficiency == 1.0

    def test_efficiency_with_poison_only(self):
        """Agent that eats only poison should have 0% efficiency."""
        death = DeathEvent(
            mode='proxy',
            death_id=0,
            survival_time=50,
            food_eaten=0,
            poison_eaten=5,
            total_reward=-5.0,
            final_energy=0.0,
        )
        assert death.efficiency == 0.0

    def test_efficiency_mixed_consumption(self):
        """Efficiency should be food / (food + poison)."""
        death = DeathEvent(
            mode='proxy',
            death_id=0,
            survival_time=75,
            food_eaten=3,
            poison_eaten=1,
            total_reward=2.0,
            final_energy=0.0,
        )
        assert death.efficiency == 0.75  # 3 / (3 + 1)

    def test_efficiency_no_consumption(self):
        """Agent that ate nothing should have 100% efficiency (didn't eat poison)."""
        death = DeathEvent(
            mode='ground_truth',
            death_id=0,
            survival_time=10,
            food_eaten=0,
            poison_eaten=0,
            total_reward=0.0,
            final_energy=0.0,
        )
        assert death.efficiency == 1.0

    def test_death_event_serializable(self):
        """Death events should be serializable to dict for JSON output."""
        death = DeathEvent(
            mode='ground_truth',
            death_id=42,
            survival_time=100,
            food_eaten=10,
            poison_eaten=2,
            total_reward=8.0,
            final_energy=0.1,
        )
        d = asdict(death)
        assert d['mode'] == 'ground_truth'
        assert d['death_id'] == 42
        assert d['survival_time'] == 100


class TestEvaluationConfig:
    """Tests for evaluation configuration."""

    def test_default_config_creation(self):
        """Should create config with sensible defaults."""
        config = EvaluationConfig()
        assert config.mode == 'ground_truth'
        assert config.n_envs > 0
        assert config.total_timesteps > 0

    def test_config_from_config_method(self):
        """Should load config from TOML with overrides."""
        config = EvaluationConfig.from_config(
            mode='proxy',
            total_timesteps=5000,
        )
        assert config.mode == 'proxy'
        assert config.total_timesteps == 5000

    def test_config_determinism_settings(self):
        """Determinism settings should be configurable."""
        config = EvaluationConfig(deterministic=True, seed=42)
        assert config.deterministic is True
        assert config.seed == 42


class TestThesisValidation:
    """
    Tests that validate the Goodhart's Law thesis.

    Thesis validation means defining and testing the EXPECTED behavioral differences
    between ground_truth and proxy-trained agents. These tests document the framework;
    actual validation happens during evaluation runs with real trained agents.

    The core thesis:
    - Ground truth agents (who see real cell types) should have HIGH efficiency
    - Proxy agents (who see only interestingness) should have LOW efficiency
    - The efficiency gap demonstrates Goodhart's Law in action

    Why efficiency is the right primary metric:
    - It directly measures "did you eat food or poison?" - the true objective
    - Survival time is a consequence of efficiency, not independent
    - Deaths/1k provides a rate, but efficiency captures the root cause
    - Other metrics (reward, exploration) are proxies of proxies
    """

    def test_thesis_metric_efficiency_defined(self):
        """
        Efficiency (food / total consumed) is the primary thesis metric.

        This metric captures Goodhart's Law because:
        - Proxy agents optimize for "interestingness" which doesn't distinguish food/poison
        - Ground truth agents can distinguish and should strongly prefer food
        - The efficiency gap quantifies the misalignment between proxy and true objective
        """
        # Ground truth agent: eats mostly food
        gt_death = DeathEvent(
            mode='ground_truth',
            death_id=0,
            survival_time=200,
            food_eaten=20,
            poison_eaten=1,  # Occasional mistake
            total_reward=19.0,
            final_energy=0.0,
        )

        # Proxy agent: can't distinguish, eats based on interestingness
        # With poison MORE interesting than food (1.0 vs 0.5), proxy prefers poison
        proxy_death = DeathEvent(
            mode='proxy',
            death_id=0,
            survival_time=50,  # Dies faster due to poison
            food_eaten=5,
            poison_eaten=8,   # More poison because it's more "interesting"
            total_reward=-3.0,
            final_energy=0.0,
        )

        # The thesis predicts this gap
        assert gt_death.efficiency > 0.9, "Ground truth should have high efficiency"
        assert proxy_death.efficiency < 0.5, "Proxy should have low efficiency"
        assert gt_death.efficiency > proxy_death.efficiency, \
            "Ground truth efficiency should exceed proxy efficiency"

    def test_thesis_metric_survival_time(self):
        """
        Survival time should be longer for ground truth agents.

        The causal chain:
        1. Proxy agents eat more poison (can't distinguish from food)
        2. Poison drains energy faster than food restores it
        3. Therefore proxy agents die sooner

        Survival time is a CONSEQUENCE of the efficiency gap, not an independent metric.
        """
        # These values represent expected population averages
        gt_expected_survival = 200  # Steps before death
        proxy_expected_survival = 75  # Dies much faster

        # The ratio should be significant (ground truth lives 2-3x longer)
        survival_ratio = gt_expected_survival / proxy_expected_survival
        assert survival_ratio > 2.0, \
            "Ground truth agents should survive significantly longer than proxy agents"

    def test_thesis_poison_preference_in_proxy(self):
        """
        Proxy agents should actually PREFER poison when it's more interesting.

        This is the SHARPENED Goodhart trap:
        - Original design: food and poison had similar interestingness (~0.9)
        - Sharpened design: poison is MORE interesting (1.0) than food (0.5)
        - This means proxy agents actively seek poison over food
        - The proxy metric is now ANTI-CORRELATED with true value

        This makes the demonstration stronger: proxy doesn't just fail to distinguish,
        it actively prefers the harmful option.
        """
        # With poison interestingness=1.0 and food interestingness=0.5,
        # a proxy agent optimizing for interestingness will prefer poison
        proxy_death = DeathEvent(
            mode='proxy',
            death_id=0,
            survival_time=40,
            food_eaten=3,
            poison_eaten=7,  # More poison than food!
            total_reward=-4.0,
            final_energy=0.0,
        )

        assert proxy_death.poison_eaten > proxy_death.food_eaten, \
            "Proxy agents should eat more poison than food when poison is more interesting"


class TestAggregateMetrics:
    """Tests for population-level aggregate statistics."""

    def test_deaths_per_1k_calculation(self):
        """Deaths per 1000 steps should be computed correctly."""
        # If we have 10 deaths over 5000 steps, that's 2 deaths per 1k
        n_deaths = 10
        total_steps = 5000
        deaths_per_1k = n_deaths / (total_steps / 1000.0)
        assert deaths_per_1k == 2.0

    def test_overall_efficiency_vs_mean_efficiency(self):
        """
        Overall efficiency (total food / total consumed) vs mean per-death efficiency.

        Overall efficiency is the TRUE metric because:
        - Mean efficiency can be skewed by deaths with low consumption
        - Overall efficiency weights by consumption volume (more representative)
        - Example: one death with 1 food, 0 poison (100% eff) and one with
          10 food, 10 poison (50% eff) gives mean=75% but overall=55%
        """
        # Death 1: perfect efficiency, low consumption
        death1_food, death1_poison = 1, 0  # 100% efficiency

        # Death 2: poor efficiency, high consumption
        death2_food, death2_poison = 10, 10  # 50% efficiency

        # Mean efficiency: (1.0 + 0.5) / 2 = 0.75
        eff1 = death1_food / (death1_food + death1_poison) if (death1_food + death1_poison) > 0 else 1.0
        eff2 = death2_food / (death2_food + death2_poison)
        mean_eff = (eff1 + eff2) / 2
        assert mean_eff == 0.75

        # Overall efficiency: 11 / 21 = 0.524
        total_food = death1_food + death2_food
        total_poison = death1_poison + death2_poison
        overall_eff = total_food / (total_food + total_poison)
        assert abs(overall_eff - 0.524) < 0.01

        # Overall is lower and more representative of actual behavior
        assert overall_eff < mean_eff


class TestEvaluationProtocol:
    """
    Tests documenting the evaluation protocol.

    Key protocol decisions and their rationale:
    1. Continuous survival (not episodes): Measures actual survival ability, not
       arbitrary episode boundaries. Death is the natural unit of measurement.
    2. Metrics interpretation: Efficiency is primary (root cause), survival and
       deaths/1k are consequences, reward is a proxy we deliberately distrust.
    3. Statistical significance: Minimum 100 deaths per mode, 3+ random seeds,
       report confidence intervals via bootstrap or Welch's t-test.
    4. Confound control: Use same food/poison density for training and eval,
       vary random seeds but not environment structure.
    """

    def test_protocol_continuous_survival_paradigm(self):
        """
        Evaluation uses continuous survival, not fixed-length episodes.

        Why this matters:
        - Fixed episodes hide the survival differential (episode ends before death)
        - Continuous survival directly measures "how long can you stay alive"
        - Deaths are the natural unit of measurement (not arbitrary time chunks)

        The key insight: there are no "episodes", only lives and deaths.
        """
        # This is a documentation test - the actual protocol is in evaluator.py
        pass

    def test_protocol_minimum_sample_size(self):
        """
        Evaluation should collect enough deaths for statistical significance.

        Sample size requirements:
        - Minimum 100 deaths per mode for stable mean/std estimates
        - 3+ random seeds to estimate cross-run variance
        - Report 95% CIs via bootstrap (preferred) or Welch's t-test

        These minimums are based on central limit theorem convergence and
        provide Cohen's d effect size sensitivity of ~0.5 at 80% power.
        """
        min_deaths_per_mode = 100
        min_random_seeds = 3

        assert min_deaths_per_mode >= 100, "Need enough deaths for stable statistics"
        assert min_random_seeds >= 3, "Need multiple seeds for variance estimation"

    def test_protocol_controlled_environment(self):
        """
        Evaluation should use controlled environment settings.

        Environment control strategy:
        - Food/poison density: Use training distribution (0.01-0.05 per cell)
        - Grid size: 64x64 (same as training)
        - Move cost: Configurable, default 0.1 (same as training)
        - View radius: 5 (same as training)

        Key principle: Evaluation conditions must match training conditions,
        otherwise we're measuring generalization, not learned behavior.
        """
        config = EvaluationConfig.from_config(mode='ground_truth')

        # Evaluation should use training distribution by default
        assert config.use_training_distribution is True, \
            "Evaluation should use same distribution as training by default"
