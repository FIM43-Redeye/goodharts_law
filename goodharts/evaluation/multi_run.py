"""
Multi-run evaluation aggregation for statistically rigorous Goodhart experiments.

Runs evaluation N times with different seeds, aggregates results across runs,
and computes proper confidence intervals using t-distribution for small samples.

The key insight: each run produces a ModeAggregates with an overall_efficiency.
Across runs, we compute the mean and CI of these efficiencies. This gives us
the expected efficiency and its uncertainty, suitable for statistical testing.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class RunResult:
    """
    Results from a single evaluation run.

    Captures the seed used and key aggregate metrics from ModeAggregates.
    These are the values we'll aggregate across runs.
    """
    run_id: int
    seed: int

    # Sample size
    n_deaths: int
    total_timesteps: int

    # Primary thesis metric
    overall_efficiency: float

    # Survival statistics
    survival_mean: float
    survival_std: float

    # Death rate (population-level)
    deaths_per_1k_steps: float

    # Consumption rates
    food_per_1k_steps: float
    poison_per_1k_steps: float

    # Per-death consumption (for reference)
    food_per_death_mean: float
    poison_per_death_mean: float

    # Reward
    reward_mean: float

    @classmethod
    def from_mode_aggregates(
        cls,
        run_id: int,
        seed: int,
        agg: 'ModeAggregates'
    ) -> 'RunResult':
        """Create RunResult from a ModeAggregates instance."""
        return cls(
            run_id=run_id,
            seed=seed,
            n_deaths=agg.n_deaths,
            total_timesteps=agg.total_timesteps,
            overall_efficiency=agg.overall_efficiency,
            survival_mean=agg.survival_mean,
            survival_std=agg.survival_std,
            deaths_per_1k_steps=agg.deaths_per_1k_steps,
            food_per_1k_steps=agg.food_per_1k_steps,
            poison_per_1k_steps=agg.poison_per_1k_steps,
            food_per_death_mean=agg.food_per_death_mean,
            poison_per_death_mean=agg.poison_per_death_mean,
            reward_mean=agg.reward_mean,
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'RunResult':
        """Create RunResult from a dictionary (JSON deserialization)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MultiRunAggregates:
    """
    Aggregated statistics across multiple evaluation runs.

    Provides mean-of-means, confidence intervals, and pooled statistics
    for rigorous statistical reporting of the Goodhart effect.
    """
    mode: str
    n_runs: int
    seeds: list[int]

    # Totals across all runs
    total_deaths: int
    total_timesteps: int

    # Efficiency: mean of overall_efficiency across runs (THE key metric)
    efficiency_mean: float
    efficiency_std: float
    efficiency_ci_low: float
    efficiency_ci_high: float

    # Survival: mean of survival_mean across runs
    survival_mean_of_means: float
    survival_std_of_means: float
    survival_ci_low: float
    survival_ci_high: float

    # Pooled survival (weighted by n_deaths per run, more accurate for population)
    survival_pooled_mean: float
    survival_pooled_std: float

    # Death rate: mean of deaths_per_1k_steps across runs
    death_rate_mean: float
    death_rate_std: float
    death_rate_ci_low: float
    death_rate_ci_high: float

    # Consumption rates (mean across runs)
    food_rate_mean: float
    food_rate_std: float
    poison_rate_mean: float
    poison_rate_std: float

    # Individual run results for detailed analysis
    runs: list[RunResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        # Ensure runs are also dicts
        d['runs'] = [asdict(r) if hasattr(r, '__dataclass_fields__') else r
                     for r in self.runs]
        return d


@dataclass
class MultiRunComparison:
    """
    Statistical comparison between two modes across multiple runs.

    Uses Welch's t-test (unpaired, unequal variances) on per-run aggregates
    to determine if differences are statistically significant.
    """
    mode_a: str
    mode_b: str
    n_runs_a: int
    n_runs_b: int

    # Efficiency comparison (the primary Goodhart metric)
    efficiency_mean_a: float
    efficiency_mean_b: float
    efficiency_diff: float  # mode_a - mode_b
    efficiency_t_stat: float
    efficiency_p_value: float
    efficiency_cohens_d: float
    efficiency_significant: bool  # p < alpha

    # Survival comparison
    survival_mean_a: float
    survival_mean_b: float
    survival_diff: float
    survival_t_stat: float
    survival_p_value: float
    survival_cohens_d: float
    survival_significant: bool

    # Goodhart Failure Index: measures how much proxy underperforms
    # GFI = (gt_efficiency - proxy_efficiency) / gt_efficiency
    # 0 = no difference, 1 = proxy has zero efficiency
    goodhart_index: float

    @classmethod
    def from_aggregates(
        cls,
        agg_a: MultiRunAggregates,
        agg_b: MultiRunAggregates,
        alpha: float = 0.05
    ) -> 'MultiRunComparison':
        """
        Compute statistical comparison between two mode aggregates.

        Uses Welch's t-test for unequal variances and computes Cohen's d
        effect size for practical significance assessment.

        Args:
            agg_a: First mode's multi-run aggregates (typically ground_truth)
            agg_b: Second mode's multi-run aggregates (typically proxy)
            alpha: Significance level (default 0.05)

        Returns:
            MultiRunComparison with test statistics
        """
        # Extract per-run values for statistical tests
        eff_a = [r.overall_efficiency for r in agg_a.runs]
        eff_b = [r.overall_efficiency for r in agg_b.runs]
        surv_a = [r.survival_mean for r in agg_a.runs]
        surv_b = [r.survival_mean for r in agg_b.runs]

        # Efficiency comparison (Welch's t-test)
        eff_t, eff_p = stats.ttest_ind(eff_a, eff_b, equal_var=False)
        eff_d = _cohens_d(eff_a, eff_b)

        # Survival comparison
        surv_t, surv_p = stats.ttest_ind(surv_a, surv_b, equal_var=False)
        surv_d = _cohens_d(surv_a, surv_b)

        # Goodhart Failure Index (assumes agg_a is ground_truth baseline)
        gfi = 0.0
        if agg_a.efficiency_mean > 0:
            gfi = (agg_a.efficiency_mean - agg_b.efficiency_mean) / agg_a.efficiency_mean

        return cls(
            mode_a=agg_a.mode,
            mode_b=agg_b.mode,
            n_runs_a=agg_a.n_runs,
            n_runs_b=agg_b.n_runs,
            efficiency_mean_a=agg_a.efficiency_mean,
            efficiency_mean_b=agg_b.efficiency_mean,
            efficiency_diff=float(np.mean(eff_a) - np.mean(eff_b)),
            efficiency_t_stat=float(eff_t),
            efficiency_p_value=float(eff_p),
            efficiency_cohens_d=float(eff_d),
            efficiency_significant=eff_p < alpha,
            survival_mean_a=agg_a.survival_mean_of_means,
            survival_mean_b=agg_b.survival_mean_of_means,
            survival_diff=float(np.mean(surv_a) - np.mean(surv_b)),
            survival_t_stat=float(surv_t),
            survival_p_value=float(surv_p),
            survival_cohens_d=float(surv_d),
            survival_significant=surv_p < alpha,
            goodhart_index=float(gfi),
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @property
    def efficiency_stars(self) -> str:
        """Significance stars for efficiency comparison."""
        return _significance_stars(self.efficiency_p_value)

    @property
    def survival_stars(self) -> str:
        """Significance stars for survival comparison."""
        return _significance_stars(self.survival_p_value)

    @property
    def efficiency_effect_magnitude(self) -> str:
        """Cohen's d interpretation for efficiency."""
        return _effect_magnitude(self.efficiency_cohens_d)

    @property
    def survival_effect_magnitude(self) -> str:
        """Cohen's d interpretation for survival."""
        return _effect_magnitude(self.survival_cohens_d)


# -----------------------------------------------------------------------------
# Statistical helper functions
# -----------------------------------------------------------------------------

def compute_confidence_interval(
    values: list[float],
    confidence: float = 0.95
) -> tuple[float, float, float, float]:
    """
    Compute confidence interval for a list of values using t-distribution.

    Uses t-distribution which is appropriate for small sample sizes (n < 30).
    For large n, t-distribution converges to normal distribution.

    Args:
        values: List of measurements (one per run)
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (mean, std, ci_low, ci_high)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    if n == 1:
        return float(values[0]), 0.0, float(values[0]), float(values[0])

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))  # Sample std (Bessel's correction)

    # t-distribution critical value
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_crit * std / np.sqrt(n)

    return mean, std, float(mean - margin), float(mean + margin)


def compute_pooled_std(
    means: list[float],
    stds: list[float],
    ns: list[int]
) -> tuple[float, float]:
    """
    Compute pooled mean and standard deviation across groups.

    Combines within-group and between-group variance for accurate
    population-level estimates when groups have different sizes.

    Args:
        means: Per-group means
        stds: Per-group standard deviations
        ns: Per-group sample sizes

    Returns:
        (pooled_mean, pooled_std)
    """
    total_n = sum(ns)
    if total_n == 0:
        return 0.0, 0.0

    # Weighted mean
    pooled_mean = sum(m * n for m, n in zip(means, ns)) / total_n

    # Pooled variance (combining within and between group variance)
    sum_sq = 0.0
    for m, s, n in zip(means, stds, ns):
        # Within-group sum of squares
        within_ss = s**2 * (n - 1) if n > 1 else 0
        # Between-group contribution (deviation from pooled mean)
        between_ss = n * (m - pooled_mean)**2
        sum_sq += within_ss + between_ss

    pooled_var = sum_sq / (total_n - 1) if total_n > 1 else 0
    pooled_std = float(np.sqrt(pooled_var))

    return float(pooled_mean), pooled_std


def generate_seeds(n_runs: int, base_seed: Optional[int] = None) -> list[int]:
    """
    Generate reproducible list of seeds for multi-run evaluation.

    If base_seed is provided, generates deterministic seeds for reproducibility.
    Otherwise uses random seeds (but still records them for the record).

    Args:
        n_runs: Number of runs
        base_seed: If provided, seeds are derived deterministically

    Returns:
        List of integer seeds
    """
    if base_seed is not None:
        rng = np.random.default_rng(base_seed)
        return [int(rng.integers(0, 2**31)) for _ in range(n_runs)]
    else:
        return [int(np.random.randint(0, 2**31)) for _ in range(n_runs)]


def aggregate_runs(mode: str, runs: list[RunResult]) -> MultiRunAggregates:
    """
    Aggregate multiple run results into summary statistics.

    Computes mean-of-means for each metric and confidence intervals
    using t-distribution appropriate for small sample sizes.

    Args:
        mode: Mode name
        runs: List of RunResult from individual runs

    Returns:
        MultiRunAggregates with cross-run statistics
    """
    n_runs = len(runs)
    if n_runs == 0:
        raise ValueError("Cannot aggregate zero runs")

    seeds = [r.seed for r in runs]

    # Totals
    total_deaths = sum(r.n_deaths for r in runs)
    total_timesteps = sum(r.total_timesteps for r in runs)

    # Efficiency (mean of overall_efficiency across runs)
    efficiencies = [r.overall_efficiency for r in runs]
    eff_mean, eff_std, eff_ci_lo, eff_ci_hi = compute_confidence_interval(efficiencies)

    # Survival: mean-of-means (what we report)
    survival_means = [r.survival_mean for r in runs]
    surv_mean, surv_std, surv_ci_lo, surv_ci_hi = compute_confidence_interval(survival_means)

    # Survival: pooled statistics (more accurate population estimate)
    survival_stds = [r.survival_std for r in runs]
    n_deaths_per_run = [r.n_deaths for r in runs]
    surv_pooled_mean, surv_pooled_std = compute_pooled_std(
        survival_means, survival_stds, n_deaths_per_run
    )

    # Death rate
    death_rates = [r.deaths_per_1k_steps for r in runs]
    dr_mean, dr_std, dr_ci_lo, dr_ci_hi = compute_confidence_interval(death_rates)

    # Consumption rates
    food_rates = [r.food_per_1k_steps for r in runs]
    poison_rates = [r.poison_per_1k_steps for r in runs]

    return MultiRunAggregates(
        mode=mode,
        n_runs=n_runs,
        seeds=seeds,
        total_deaths=total_deaths,
        total_timesteps=total_timesteps,
        efficiency_mean=eff_mean,
        efficiency_std=eff_std,
        efficiency_ci_low=eff_ci_lo,
        efficiency_ci_high=eff_ci_hi,
        survival_mean_of_means=surv_mean,
        survival_std_of_means=surv_std,
        survival_ci_low=surv_ci_lo,
        survival_ci_high=surv_ci_hi,
        survival_pooled_mean=surv_pooled_mean,
        survival_pooled_std=surv_pooled_std,
        death_rate_mean=dr_mean,
        death_rate_std=dr_std,
        death_rate_ci_low=dr_ci_lo,
        death_rate_ci_high=dr_ci_hi,
        food_rate_mean=float(np.mean(food_rates)),
        food_rate_std=float(np.std(food_rates, ddof=1)) if n_runs > 1 else 0.0,
        poison_rate_mean=float(np.mean(poison_rates)),
        poison_rate_std=float(np.std(poison_rates, ddof=1)) if n_runs > 1 else 0.0,
        runs=runs,
    )


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size (pooled std denominator)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0

    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    # Pooled standard deviation
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


def _significance_stars(p: float) -> str:
    """Return significance stars for p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def _effect_magnitude(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    return 'large'


# -----------------------------------------------------------------------------
# Multi-run evaluator
# -----------------------------------------------------------------------------

class MultiRunEvaluator:
    """
    Orchestrates multiple evaluation runs with different seeds.

    Runs ModelTester N times, collects RunResults, and aggregates
    them into MultiRunAggregates for statistical analysis.

    Usage:
        evaluator = MultiRunEvaluator(
            mode='ground_truth',
            n_runs=5,
            base_config=EvaluationConfig.from_config(mode='ground_truth'),
            base_seed=42,
        )
        aggregates = evaluator.run()
    """

    def __init__(
        self,
        mode: str,
        n_runs: int,
        base_config: 'EvaluationConfig',
        base_seed: Optional[int] = None,
        seeds: Optional[list[int]] = None,
        verbose: bool = True,
    ):
        """
        Initialize multi-run evaluator.

        Args:
            mode: Mode to evaluate
            n_runs: Number of runs to perform (ignored if seeds provided)
            base_config: Base evaluation configuration (seed overridden per run)
            base_seed: Base seed for generating per-run seeds
            seeds: Explicit list of seeds (overrides n_runs and base_seed)
            verbose: Print progress
        """
        self.mode = mode
        self.base_config = base_config
        self.verbose = verbose

        # Determine seeds
        if seeds is not None:
            self.seeds = seeds
            self.n_runs = len(seeds)
        else:
            self.n_runs = n_runs
            self.seeds = generate_seeds(n_runs, base_seed)

        # Results storage
        self.run_results: list[RunResult] = []
        self.aggregates: Optional[MultiRunAggregates] = None

    def run(self) -> MultiRunAggregates:
        """
        Execute all runs and return aggregated results.

        Returns:
            MultiRunAggregates with cross-run statistics
        """
        from goodharts.evaluation import EvaluationConfig, ModelTester, ModeAggregates

        if self.verbose:
            print(f"\n[MultiRun] {self.mode}: {self.n_runs} runs")
            print(f"[MultiRun] Seeds: {self.seeds}")

        for run_id, seed in enumerate(self.seeds):
            if self.verbose:
                print(f"\n[Run {run_id + 1}/{self.n_runs}] seed={seed}")

            # Create config with this run's seed
            run_config = EvaluationConfig(
                mode=self.mode,
                total_timesteps=self.base_config.total_timesteps,
                n_envs=self.base_config.n_envs,
                deterministic=self.base_config.deterministic,
                seed=seed,
                temperature=self.base_config.temperature,
                use_training_distribution=self.base_config.use_training_distribution,
                food_count=self.base_config.food_count,
                poison_count=self.base_config.poison_count,
                move_cost=self.base_config.move_cost,
                output_path=self.base_config.output_path,
                model_path=self.base_config.model_path,
            )

            # Run evaluation
            tester = ModelTester(run_config)
            result = tester.run()

            # Extract aggregates and create RunResult
            agg_dict = result.get('aggregates')
            if agg_dict:
                # Reconstruct ModeAggregates from dict
                agg = ModeAggregates(**agg_dict)
                run_result = RunResult.from_mode_aggregates(run_id, seed, agg)
                self.run_results.append(run_result)

                if self.verbose:
                    print(f"    Efficiency: {agg.overall_efficiency:.1%}, "
                          f"Survival: {agg.survival_mean:.0f}, "
                          f"Deaths: {agg.n_deaths}")
            else:
                if self.verbose:
                    print(f"    Warning: Run {run_id} produced no aggregates")

        # Aggregate across runs
        if self.run_results:
            self.aggregates = aggregate_runs(self.mode, self.run_results)

            if self.verbose:
                print(f"\n[MultiRun] {self.mode} Summary:")
                print(f"    Efficiency: {self.aggregates.efficiency_mean:.1%} "
                      f"[{self.aggregates.efficiency_ci_low:.1%}, "
                      f"{self.aggregates.efficiency_ci_high:.1%}]")
                print(f"    Survival: {self.aggregates.survival_mean_of_means:.0f} "
                      f"[{self.aggregates.survival_ci_low:.0f}, "
                      f"{self.aggregates.survival_ci_high:.0f}]")
                print(f"    Total deaths: {self.aggregates.total_deaths}")

        return self.aggregates

    def to_dict(self) -> dict:
        """Convert results to JSON-serializable dict."""
        return {
            'mode': self.mode,
            'n_runs': self.n_runs,
            'seeds': self.seeds,
            'aggregates': self.aggregates.to_dict() if self.aggregates else None,
            'runs': [asdict(r) for r in self.run_results],
        }
