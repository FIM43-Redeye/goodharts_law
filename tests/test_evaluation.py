"""
Tests for the evaluation system and Goodhart's Law thesis validation.

The evaluation system uses a continuous survival paradigm: agents run until
they die (starvation), then auto-respawn. We track death events, not episodes.

These tests verify:
1. Evaluation infrastructure correctness (metrics, aggregates)
2. Thesis validation framework (what the Goodhart demonstration should show)

TODO: Explain the philosophical difference between "testing machinery" and
"validating the thesis" - why both matter for a rigorous demonstration.
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

    TODO: Explain what "thesis validation" means in the context of this project.
    These tests define the EXPECTED behavioral differences between ground_truth
    and proxy-trained agents. The actual validation happens during evaluation
    runs, but these tests document and verify the framework for that validation.

    The core thesis is:
    - Ground truth agents (who see real cell types) should have HIGH efficiency
    - Proxy agents (who see only interestingness) should have LOW efficiency
    - The efficiency gap demonstrates Goodhart's Law in action

    TODO: Discuss why efficiency is the right metric for this demonstration,
    and what other metrics (survival time, deaths/1k) tell us.
    """

    def test_thesis_metric_efficiency_defined(self):
        """
        Efficiency (food / total consumed) is the primary thesis metric.

        TODO: Explain why this metric captures Goodhart's Law:
        - Proxy agents optimize for "interestingness" which doesn't distinguish food/poison
        - Ground truth agents can distinguish and should strongly prefer food
        - The efficiency gap quantifies the misalignment
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

        TODO: Explain the causal chain:
        1. Proxy agents eat more poison (can't distinguish)
        2. Poison drains energy faster than food restores it
        3. Therefore proxy agents die sooner

        This is a CONSEQUENCE of the efficiency gap, not an independent metric.
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

        TODO: Explain this is the SHARPENED Goodhart trap:
        - Original design: food and poison have similar interestingness (~0.9)
        - Sharpened design: poison is MORE interesting (1.0) than food (0.5)
        - This means proxy agents will actively seek poison over food
        - The proxy metric is now ANTI-CORRELATED with true value
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

        TODO: Explain why overall efficiency is the TRUE metric:
        - Mean efficiency can be skewed by deaths with low consumption
        - Overall efficiency weights by consumption volume
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

    TODO: This section should be expanded with your own explanation of:
    1. Why continuous survival (not episodes) is the right paradigm
    2. How to interpret the metrics for thesis validation
    3. What constitutes "statistically significant" evidence
    4. How to control for confounds (environment difficulty, random seeds)
    """

    def test_protocol_continuous_survival_paradigm(self):
        """
        Evaluation uses continuous survival, not fixed-length episodes.

        TODO: Explain why this matters:
        - Fixed episodes hide the survival differential
        - Continuous survival directly measures "how long can you stay alive"
        - Deaths are the natural unit of measurement
        """
        # This is a documentation test - the actual protocol is in evaluator.py
        # The key insight is: there are no "episodes", only lives and deaths
        pass

    def test_protocol_minimum_sample_size(self):
        """
        Evaluation should collect enough deaths for statistical significance.

        TODO: Specify your sample size requirements:
        - Minimum deaths per mode for reliable estimates
        - Number of random seeds for variance estimation
        - How to report confidence intervals
        """
        # Suggested minimums (you should validate these)
        min_deaths_per_mode = 100
        min_random_seeds = 3

        # These are placeholders - adjust based on your analysis
        assert min_deaths_per_mode >= 100, "Need enough deaths for stable statistics"
        assert min_random_seeds >= 3, "Need multiple seeds for variance estimation"

    def test_protocol_controlled_environment(self):
        """
        Evaluation should use controlled environment settings.

        TODO: Document your environment control strategy:
        - Food/poison density ranges (should match training distribution)
        - Grid size and move costs
        - Any other relevant parameters
        """
        config = EvaluationConfig.from_config(mode='ground_truth')

        # Evaluation should use training distribution by default
        assert config.use_training_distribution is True, \
            "Evaluation should use same distribution as training by default"
